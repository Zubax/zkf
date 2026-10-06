#!/usr/bin/env python3
"""zkf_cordic against Zkf.sincos / Zkf.atan2. At a fixed MODE only that mode's cases run and `vectoring` is noise."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

import zkf.oracle
from zkf import CordicModel, Timing, Zkf, ZkfFormat
from zkf._reference import atan2_bypass_shift, trig_spec
from zkf_bits import hex_bits, mask
from zkf_operands import (
    directed_numbers,
    normal,
    random_bits,
    random_inf,
    random_normal_near,
    random_operand,
    random_zero,
    saturating_y,
)
from zkf_params import check_width, float_context
from zkf_stream import drive_unsigned, expect_reaccept, reset_boundaries, start_clock


@dataclass(frozen=True)
class CordicCase:
    label: str
    vectoring: bool
    a: int
    b: int
    r0: int
    r1: int
    quadrant: int

    def describe(self, fmt: ZkfFormat) -> str:
        mode = "vectoring" if self.vectoring else "rotation"
        return f"{mode} {self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    sgn = 1 << fmt.sign_shift
    out: list[tuple[str, int]] = [
        ("raw_zero", 0),
        ("raw_neg_zero", sgn),  # canonicalized +0 outputs
        ("raw_pos_inf", fmt.exp_inf << fmt.wfrac),  # sin=cos=+inf
        ("raw_neg_inf", sgn | (fmt.exp_inf << fmt.wfrac)),  # sin=cos=-inf
        ("raw_all_ones", mask(fmt.wfull)),  # negative non-canonical
    ]
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            out.append((f"num_{label}", value))

        def turns(sign: int, k: int) -> int:
            # x = k/4 turns as a normalized float (k=1..7): k/4 = m * 2**exp, m in [1,2), exp = floor(log2(k/4)).
            exp_unb = (k.bit_length() - 1) - 2
            frac = (k << (fmt.wfrac - (k.bit_length() - 1))) & fmt.frac_mask
            be = fmt.bias + exp_unb
            return normal(fmt, sign, be, frac) if 1 <= be <= fmt.exp_max_finite else (sgn * sign)

        for sign in (0, 1):
            for k in range(1, 8):  # 1/4, 1/2, 3/4, 1, 5/4, 3/2, 7/4 turns
                out.append((f"quarter_{'-' if sign else '+'}{k}_4", turns(sign, k)))
            # Just before/after a quarter-turn boundary (x = 1/4 +- 1 ULP) -- exercises the boundary quadrant rule.
            q = fmt.bias - 2  # biased exp for 0.25
            if 2 <= q <= fmt.exp_max_finite:  # 0.25 - 1 ULP sits one binade lower (needs exp q-1 >= 1)
                out.append((f"just_below_quarter_{sign}", normal(fmt, sign, q - 1, fmt.frac_mask)))
            if 1 <= q <= fmt.exp_max_finite:
                out.append((f"just_above_quarter_{sign}", normal(fmt, sign, q, 1)))
            # Integer turns (frac == 0 -> quadrant 0, sin=+0, cos=+1) and a large finite with zero fraction.
            for k in (0, 1, 2):
                be = fmt.bias + k
                if 1 <= be <= fmt.exp_max_finite:
                    out.append((f"int_turn_{sign}_{k}", normal(fmt, sign, be, 0)))
            out.append((f"large_int_{sign}", normal(fmt, sign, fmt.exp_max_finite, 0)))
            # The small-angle handoff: the octant-local coordinate crossing tsa, straddled on both sides of the octant
            # fold in every quadrant (a mis-sized slice compare there silently disables or widens the bypass).
            spec = trig_spec(fmt.wman)
            edge = Fraction(spec["tsa"], 1 << (spec["wt"] + 2))
            for quarter in range(4):
                for side, center in (("lo", Fraction(quarter, 4) + edge), ("hi", Fraction(quarter + 1, 4) - edge)):
                    bits = fmt.encode(center).bits
                    for step in (-2, -1, 0, 1, 2) if (bits >> fmt.wfrac) >= 2 else ():  # skip if it underflows
                        out.append((f"tsa_{side}_{sign}_{quarter}_{step}", (bits + step) | (sign << fmt.sign_shift)))
            # Tiny phases below the reducer resolution (the bypass): smallest few exponents with assorted fractions.
            for be in (1, 2, 3):
                if be <= fmt.exp_max_finite:
                    out.append((f"tiny_{sign}_{be}_a", normal(fmt, sign, be, 1)))
                    out.append((f"tiny_{sign}_{be}_b", normal(fmt, sign, be, fmt.frac_mask)))
    return out


def rotation_inputs(fmt: ZkfFormat, kind: str, seed: int, count: int, shard_index: int, shard_count: int) -> dict:
    """{x: label}, in issue order. An exhaustive sweep is strided: the union over the shard indices is the full one."""
    if kind == "exhaustive":
        return {x: "exhaustive" for x in range(shard_index, 1 << fmt.wfull, shard_count)}
    inputs: dict[int, str] = {}
    for label, value in directed_values(fmt):
        inputs.setdefault(value & mask(fmt.wfull), label)
    rng = np.random.default_rng(seed)
    while kind != "directed" and len(inputs) < count:
        x = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        inputs.setdefault(x & mask(fmt.wfull), "random")
    # On top of `count`, near k/2 turn, where |sin| is smallest: its underflow decision (reachable when BIAS < WFRAC)
    # and the union packer's re-based exponent both sit here.
    for exp in (fmt.bias - 1, fmt.bias) if fmt.wexp >= 3 else ():
        for ulps in (1, 2, 3, 5, 8):
            for step in (-ulps, ulps):
                inputs.setdefault((exp << fmt.wfrac) + step, "half_turn")
    return inputs


# A full per-exponent sweep is O(2**WEXP) and at wide WEXP dominates CI wall time. Above the cap, sample only the
# coverage-bearing exponents: the extremes, the bypass-decision boundary (bias +- the bypass shift), and an even
# spread. Line coverage is unaffected.
_FULL_SWEEP_CAP = 4096


def bypass_sweep_exponents(fmt: ZkfFormat) -> list[int]:
    top = fmt.exp_inf
    if top - 1 <= _FULL_SWEEP_CAP:
        return list(range(1, top))
    shift = atan2_bypass_shift(fmt)
    keep: set[int] = set()
    keep.update(range(1, 9))
    keep.update(range(top - 8, top))
    for center in (fmt.bias, fmt.bias + shift, fmt.bias - shift):
        keep.update(range(center - 4, center + 5))
    keep.update(range(1, top, max(1, (top - 1) // _FULL_SWEEP_CAP)))
    return sorted(e for e in keep if 1 <= e < top)


def directed_pairs(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    sgn = 1 << fmt.sign_shift
    inf = fmt.exp_inf << fmt.wfrac
    raw: list[tuple[str, int]] = [
        ("z+", 0),
        ("z-", sgn),
        ("inf+", inf),
        ("inf-", sgn | inf),
        ("ones", mask(fmt.wfull)),
    ]
    out: list[tuple[str, int, int]] = []
    # Full cross of the raw specials (both zero, inf combos, signed-zero corners, non-canonical).
    for ly, vy in raw:
        for lx, vx in raw:
            out.append((f"raw_{ly}_{lx}", vy, vx))
    nums = directed_numbers(fmt)
    one, mone, two = nums["one"], nums["minus_one"], nums["two"]
    big = normal(fmt, 0, fmt.exp_max_finite, 0)
    tiny = normal(fmt, 0, 1, 0)
    neginf = sgn | inf
    # Four sign quadrants and the octant diagonals (|y| == |x| -> +-1/8, +-3/8).
    for sy in (0, 1):
        for sx in (0, 1):
            yv = one | (sy << fmt.sign_shift)
            xv = one | (sx << fmt.sign_shift)
            out.append((f"diag_{sy}{sx}", yv, xv))
            out.append((f"q_{sy}{sx}", (two | (sy << fmt.sign_shift)), xv))
    # Axes (y or x exactly zero), and the just-around boundaries.
    for s in (0, 1):
        sb = s << fmt.sign_shift
        out.append((f"yzero_x+_{s}", 0, one))
        out.append((f"yzero_x-_{s}", 0, mone))
        out.append((f"xzero_y_{s}", one | sb, 0))
        # |y| << |x| (theta -> 0 or near 1/4 after swap) and |x| << |y|.
        out.append((f"ysmall_{s}", tiny | sb, big))
        out.append((f"xsmall_{s}", big | sb, tiny))
        # Finite x<0 with |y| -> 0: theta rounds to the 1/2-turn endpoint and must canonicalize to the in-range
        # +1/2, never the out-of-range -1/2.
        out.append((f"xnegbig_ytiny_{s}", tiny | sb, big | sgn))
        out.append((f"xnegone_ytiny_{s}", tiny | sb, mone))
        out.append((f"ybig_xone_{s}", big | sb, one))
        out.append((f"yone_xbig_{s}", one | sb, big | sb))
    out.append(("xneginf_ypos_finite", one, neginf))
    out.append(("xneginf_yneg_finite", mone, neginf))
    # |y/x| straddling the small-ratio bypass boundary across the exponent range (x = +1).
    for e in bypass_sweep_exponents(fmt):
        out.append((f"sweep_y_{e}", normal(fmt, 0, e, fmt.frac_mask), one))
        out.append((f"sweep_x_{e}", one, normal(fmt, 0, e, 1)))
    # Magnitude overflow ladder, x = max_finite, over the exponent range derived at tools/zkf_trig.py's _atan2_pairs.
    # The ladder alone does not prove SATURATE_ROUND_CARRY fires -- that property is anchored by
    # proof/sby/zkf_pack_sat.sby -- so saturating_y() adds one input that provably reaches the carry, making a lost
    # SATURATE_ROUND_CARRY(1) visible here too.
    top = normal(fmt, 0, fmt.exp_max_finite, fmt.frac_mask)
    for e in range(max(1, fmt.exp_max_finite - (fmt.wman // 2) - 3), fmt.exp_max_finite + 1):
        for fr in (0, fmt.frac_mask):
            out.append((f"ovf_band_e{e}_f{fr}", normal(fmt, 0, e, fr), top))
    y_sat = saturating_y(fmt)
    # Only meaningful while the magnitude really does reach the carry, so pin it rather than trust the error bound
    # it rests on. Comparing against the CORRECTLY-ROUNDED answer is what distinguishes "saturated" from "rounded
    # down and never carried": the truth must round to +inf while the operator returns max-finite.
    assert zkf.oracle.atan2(Zkf(fmt, y_sat), Zkf(fmt, top)).magnitude.is_inf, "ovf_saturating: truth not +inf"
    assert Zkf(fmt, y_sat).atan2(Zkf(fmt, top)).magnitude.bits == top, "ovf_saturating no longer saturates"
    out.append(("ovf_saturating", y_sat, top))
    return out


def random_pair(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(0, 11))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 2:
        return random_zero(fmt, rng), random_zero(fmt, rng)
    if mode == 3:
        return random_inf(fmt, rng), random_operand(fmt, rng)
    if mode == 4:
        return random_operand(fmt, rng), random_inf(fmt, rng)
    if mode == 5:
        return random_inf(fmt, rng), random_inf(fmt, rng)
    fr = [0, fmt.frac_mask, fmt.frac_mask >> 1]
    near = [fmt.bias]
    lo = [1, 2, 3]
    hi = [fmt.exp_max_finite, fmt.exp_max_finite - 1]
    if mode == 6:  # near-equal magnitudes (octant edge)
        return random_normal_near(fmt, rng, near, fr), random_normal_near(fmt, rng, near, fr)
    if mode == 7:  # |y| << |x| (bypass region)
        return random_normal_near(fmt, rng, lo, fr), random_normal_near(fmt, rng, hi, fr)
    if mode == 8:  # |x| << |y|
        return random_normal_near(fmt, rng, hi, fr), random_normal_near(fmt, rng, lo, fr)
    return random_operand(fmt, rng), random_operand(fmt, rng)


def vectoring_inputs(fmt: ZkfFormat, kind: str, seed: int, count: int) -> dict:
    """{(y, x): label}, in issue order."""
    assert kind != "exhaustive", "the joint sweep is infeasible at every format vectoring accepts"
    inputs: dict[tuple[int, int], str] = {}
    for label, yv, xv in directed_pairs(fmt):
        inputs.setdefault((yv & mask(fmt.wfull), xv & mask(fmt.wfull)), label)
    if kind == "directed":
        return inputs
    rng = np.random.default_rng(seed)
    while len(inputs) < count:
        yv, xv = random_pair(fmt, rng)
        inputs.setdefault((yv & mask(fmt.wfull), xv & mask(fmt.wfull)), "random")
    return inputs


def cases_for(context) -> list[CordicCase]:
    fmt = ZkfFormat(context.wexp, context.wman)
    kind, seed, count, mode = context.kind, context.seed, context.count, context.mode
    rng = np.random.default_rng(seed)
    cases = []
    if mode != 1:
        swept = rotation_inputs(fmt, kind, seed, count, context.shard_index, context.shard_count)
        for x, label in swept.items():
            r = fmt.wrap(x).sincos()
            cases.append(CordicCase(label, False, x, random_bits(fmt.wfull, rng), r.sin.bits, r.cos.bits, r.quadrant))
    if mode != 0:
        for (y, x), label in vectoring_inputs(fmt, kind, seed, count).items():
            r = fmt.wrap(y).atan2(fmt.wrap(x))
            assert label != "xneginf_yneg_finite" or not r.theta.negative, "theta must be the in-range +1/2"
            cases.append(CordicCase(label, True, y, x, r.theta.bits, r.magnitude.bits, 0))
    return [cases[i] for i in rng.permutation(len(cases))]


def expected_timing(context) -> Mapping[int, Timing]:
    """Keyed by the case's `vectoring` bool, which hashes as 0/1."""
    timing = CordicModel(
        ZkfFormat(context.wexp, context.wman),
        mode=context.mode,
        unroll100=context.unroll100,
        stage_input=context.stage_input,
        stage_product=context.stage_product,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).timing
    timings = {context.mode: timing} if isinstance(timing, Timing) else dict(timing)
    if context.parallel != int(context.mode != 1 and context.unroll100 < 100):
        assert context.unroll100 == 50 and context.parallel == 0
        delay = 1 + context.stage_product  # the testing-only PARALLEL override delays rotation and its retirement
        timings[0] = Timing(timings[0].latency + delay, timings[0].initiation_interval + delay)
    return timings


def max_latency(context) -> int:
    return max(t.latency for t in expected_timing(context).values())


def _outputs(dut) -> dict[str, int]:
    return {"r0": int(dut.r0.value), "r1": int(dut.r1.value), "quadrant": int(dut.quadrant.value)}


def _expected(case: CordicCase) -> dict[str, int]:
    return {"r0": case.r0, "r1": case.r1, "quadrant": case.quadrant}


async def _reset(dut, out_ready: int) -> None:
    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = out_ready
    dut.vectoring.value = 0
    drive_unsigned(dut.a, 0)
    drive_unsigned(dut.b, 0)
    for _ in range(4):
        await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, "out_valid asserted during reset"
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def _issue(dut, case: CordicCase, rng: np.random.Generator, mode: int) -> None:
    dut.in_valid.value = 1
    dut.vectoring.value = int(case.vectoring if mode == 2 else rng.integers(0, 2))
    drive_unsigned(dut.a, case.a)
    drive_unsigned(dut.b, case.b)


def _scramble(dut, rng: np.random.Generator, wfull: int) -> None:
    dut.in_valid.value = 0
    dut.vectoring.value = int(rng.integers(0, 2))
    drive_unsigned(dut.a, random_bits(wfull, rng))
    drive_unsigned(dut.b, random_bits(wfull, rng))


@cocotb.test()
async def cordic_runtime_cases(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    timing = expected_timing(context)
    for name in ("a", "b", "r0", "r1"):
        check_width(name, getattr(dut, name), fmt.wfull, context)
    cases = cases_for(context)
    transitions = {(a.vectoring, b.vectoring) for a, b in zip(cases, cases[1:])}
    assert len(transitions) == (4 if context.mode == 2 else 1), f"{context.prefix()}: mode transitions {transitions}"
    rng = np.random.default_rng(context.seed + 1)
    await _reset(dut, out_ready=1)

    timeout = max_latency(context) + 8
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: in_ready stuck low (case {index})"
        _issue(dut, case, rng, context.mode)
        await RisingEdge(dut.clk)
        _scramble(dut, rng, fmt.wfull)
        guard = 0
        while int(dut.out_valid.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: out_valid timeout (case {index})"
            assert int(dut.in_ready.value) == 0, f"{context.prefix()}: in_ready high while busy (case {index})"
        want = timing[case.vectoring]
        assert guard == want.latency, f"{context.prefix()} {case.describe(fmt)}: latency {guard} != model {want}"
        got, exp = _outputs(dut), _expected(case)
        assert got == exp, f"{context.prefix()} case={index} {case.describe(fmt)}: got {got} expected {exp}"
        await expect_reaccept(dut, f"{context.prefix()} case={index}", want.initiation_interval - want.latency)


@cocotb.test()
async def cordic_random_handshake(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    timing = expected_timing(context)
    cases = cases_for(context)[:128]
    rng = np.random.default_rng(context.seed + 2)
    await _reset(dut, out_ready=0)

    issued, taken, cycle = 0, 0, 0
    pending: tuple[CordicCase, int] | None = None
    held: CordicCase | None = None
    budget = 8 * len(cases) * (max_latency(context) + 2)
    while taken < len(cases):
        assert cycle < budget, f"{context.prefix()}: {taken}/{len(cases)} results in {budget} cycles"
        offer = issued < len(cases) and rng.random() < 0.5
        if offer:
            _issue(dut, cases[issued], rng, context.mode)
        else:
            _scramble(dut, rng, fmt.wfull)
        dut.out_ready.value = out_ready = int(rng.random() < 0.5)
        await ReadOnly()
        if pending is not None or held is not None:
            assert int(dut.in_ready.value) == 0, f"{context.prefix()} cycle {cycle}: in_ready high while busy"
        assert held is None or int(dut.out_valid.value), f"{context.prefix()} cycle {cycle}: unread result withdrawn"
        if int(dut.out_valid.value):
            if held is None:
                assert pending is not None, f"{context.prefix()} cycle {cycle}: out_valid with nothing in flight"
                held, accepted_at = pending
                pending = None
                assert (
                    cycle - accepted_at == timing[held.vectoring].latency
                ), f"{context.prefix()} {held.describe(fmt)}: latency"
            got = _outputs(dut)
            assert got == _expected(held), f"{context.prefix()} {held.describe(fmt)}: got {got}"
            if out_ready:
                held, taken = None, taken + 1
        if pending is not None:
            case, accepted_at = pending
            assert (
                cycle - accepted_at < timing[case.vectoring].latency
            ), f"{context.prefix()} {case.describe(fmt)}: no result"
        if offer and int(dut.in_ready.value):
            pending, issued = (cases[issued], cycle), issued + 1
        await RisingEdge(dut.clk)
        cycle += 1


@cocotb.test()
async def cordic_reset_boundaries(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    directed = [c for c in cases_for(dataclasses.replace(context, kind="directed", count=0)) if c.r0 and c.r1]
    one = directed[0]
    other = next(c for c in directed if c.vectoring != one.vectoring or (context.mode != 2 and c.r0 != one.r0))
    rng = np.random.default_rng(context.seed + 3)
    await reset_boundaries(
        dut,
        context.prefix(),
        [(one, other), (other, one)],
        lambda case: _issue(dut, case, rng, context.mode),
        lambda: _scramble(dut, rng, fmt.wfull),
        lambda: _outputs(dut),
        _expected,
        lambda case: case.describe(fmt),
        max_latency(context) + 8,
    )

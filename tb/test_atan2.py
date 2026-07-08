#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge

from zkf import ZkfFormat
from zkf._reference import atan2_bypass_shift
from zkf_bits import hex_bits, mask
from zkf_operands import normal
from zkf_operands import directed_numbers, random_inf, random_normal_near, random_operand, random_zero
from zkf_params import check_width, float_context
from zkf_stream import drive_unsigned, start_clock


@dataclass(frozen=True)
class Atan2Case:
    label: str
    y: int
    x: int
    theta: int
    mag: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} y={hex_bits(self.y, fmt.wfull)} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[Atan2Case], seen: set[tuple[int, int]], label: str, fmt: ZkfFormat, y: int, x: int) -> None:
    key = (y & mask(fmt.wfull), x & mask(fmt.wfull))
    if key in seen:
        return
    seen.add(key)
    r = fmt.wrap(key[0]).atan2(fmt.wrap(key[1]))
    cases.append(Atan2Case(label, key[0], key[1], r.theta.bits, r.magnitude.bits))


# The bypass sweep in directed_pairs() straddles the small-ratio bypass boundary at every exponent scale. The exponent
# range is 2**WEXP, so a full per-exponent sweep is O(2**WEXP) simulated transactions -- at wide WEXP (e.g. w20_m16,
# ~1e6 exponents -> ~2e6 cases) it dominates the whole CI wall time. Keep the full sweep where it is already cheap;
# where it is not, sample the coverage-bearing exponents only: the extremes, the neighborhood of the bypass-decision
# boundary (bias +- the bypass shift), and an even spread across the range. Line coverage is unaffected.
_FULL_SWEEP_CAP = 4096


def bypass_sweep_exponents(fmt: ZkfFormat) -> list[int]:
    top = fmt.exp_inf  # exponents run [1, exp_inf); exp_inf encodes the non-finite class
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
    if fmt.wexp >= 3:
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
        # |y/x| straddling the small-ratio bypass boundary across the exponent range (x = +1); bounded at wide WEXP.
        for e in bypass_sweep_exponents(fmt):
            out.append((f"sweep_y_{e}", normal(fmt, 0, e, fmt.frac_mask), one))
            out.append((f"sweep_x_{e}", one, normal(fmt, 0, e, 1)))
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


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[Atan2Case]:
    cases: list[Atan2Case] = []
    seen: set[tuple[int, int]] = set()

    if kind == "exhaustive":
        for y in range(1 << fmt.wfull):
            for x in range(1 << fmt.wfull):
                add_unique(cases, seen, "exhaustive", fmt, y, x)
        return cases

    for label, yv, xv in directed_pairs(fmt):
        add_unique(cases, seen, label, fmt, yv, xv)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        yv, xv = random_pair(fmt, rng)
        add_unique(cases, seen, "random", fmt, yv, xv)
    return cases


@cocotb.test()
async def atan2_runtime_cases(dut) -> None:
    # Latency is data-independent and published: measure accept->out_valid so the model cannot drift from the RTL.
    context = float_context("atan2")
    fmt = ZkfFormat(context.wexp, context.wman)
    expected_latency = fmt.model_of("atan2")(
        unroll100=context.unroll100,
        stage_input=context.stage_input,
        stage_product=context.stage_product,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    check_width("y", dut.y, fmt.wfull, context)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("theta", dut.theta, fmt.wfull, context)
    check_width("mag", dut.mag, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = 1  # latency measurement assumes always-ready
    drive_unsigned(dut.y, 0)
    drive_unsigned(dut.x, 0)
    for _ in range(4):
        await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, f"{context.prefix()}: out_valid asserted during reset"
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    timeout = 8 * (context.wman + 64)  # generous upper bound on the iterative latency
    checked = 0
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: in_ready stuck low (case {index})"
        dut.in_valid.value = 1
        drive_unsigned(dut.y, case.y)
        drive_unsigned(dut.x, case.x)
        await RisingEdge(dut.clk)  # this edge accepts the transaction
        dut.in_valid.value = 0
        drive_unsigned(dut.y, mask(fmt.wfull))  # garbage between transactions
        drive_unsigned(dut.x, mask(fmt.wfull))
        guard = 0
        while int(dut.out_valid.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: out_valid timeout (case {index})"
        assert guard == expected_latency, (
            f"{context.prefix()} case={index}: measured latency {guard} != model {expected_latency} "
            f"(unroll100={context.unroll100} SI={context.stage_input} SP={context.stage_product} "
            f"SN={context.stage_normalize} PA={context.stage_pack} "
            f"SO={context.stage_output})"
        )
        got = {"theta": int(dut.theta.value), "mag": int(dut.mag.value)}
        if case.label == "xneginf_yneg_finite":
            assert ((got["theta"] >> fmt.sign_shift) & 1) == 0, (
                f"{context.prefix()} case={index} {case.describe(fmt)}: "
                f"negative-y x=-inf endpoint returned a negative theta"
            )
        exp = {"theta": case.theta, "mag": case.mag}
        assert got == exp, f"{context.prefix()} case={index} {case.describe(fmt)}: got {got} expected {exp}"
        checked += 1
    assert checked == len(cases), f"{context.prefix()} checked {checked}, expected {len(cases)}"


@cocotb.test()
async def atan2_backpressure(dut) -> None:
    # Back-pressure: while out_ready is low, out_valid and (theta, mag) persist and in_ready stays low; the result is
    # taken only on a cycle with out_valid & out_ready.
    context = float_context("atan2")
    fmt = ZkfFormat(context.wexp, context.wman)
    cases = cases_for(fmt, "directed", context.seed, 0)[:2]
    assert len(cases) == 2, f"{context.prefix()}: need two directed cases for the back-pressure test"

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    drive_unsigned(dut.y, 0)
    drive_unsigned(dut.x, 0)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    timeout = 8 * (context.wman + 64)
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()} bp: in_ready stuck low (case {index})"
        dut.in_valid.value = 1
        drive_unsigned(dut.y, case.y)
        drive_unsigned(dut.x, case.x)
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        drive_unsigned(dut.y, mask(fmt.wfull))
        drive_unsigned(dut.x, mask(fmt.wfull))
        guard = 0
        while int(dut.out_valid.value) == 0:  # out_ready is low; out_valid still arrives on time
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()} bp: out_valid timeout (case {index})"
        exp = {"theta": case.theta, "mag": case.mag}
        for hold in range(5):
            got = {"theta": int(dut.theta.value), "mag": int(dut.mag.value)}
            assert int(dut.out_valid.value) == 1, f"{context.prefix()} bp case={index}: out_valid dropped while stalled"
            assert (
                int(dut.in_ready.value) == 0
            ), f"{context.prefix()} bp case={index}: in_ready high while result unread"
            assert got == exp, f"{context.prefix()} bp case={index} hold={hold}: result changed: {got} != {exp}"
            await RisingEdge(dut.clk)
        dut.out_ready.value = 1
        await RisingEdge(dut.clk)
        dut.out_ready.value = 0
    guard = 0
    while int(dut.in_ready.value) == 0:
        await RisingEdge(dut.clk)
        guard += 1
        assert guard < timeout, f"{context.prefix()} bp: in_ready did not recover after consume"

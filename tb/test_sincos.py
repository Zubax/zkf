#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge

from zkf_latency import sincos_latency
from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import normal
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_params import check_width, float_context
from zkf_stream import drive_unsigned, start_clock


@dataclass(frozen=True)
class SincosCase:
    label: str
    x: int
    sin: int
    cos: int
    quadrant: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[SincosCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    key = x & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    r = fmt.wrap(x).sincos()
    cases.append(SincosCase(label, x, r.sin.bits, r.cos.bits, r.quadrant))


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    sgn = 1 << fmt.sign_shift
    out: list[tuple[str, int]] = [
        ("raw_zero", 0),
        ("raw_neg_zero", sgn),                                       # canonicalized +0 outputs
        ("raw_pos_inf", fmt.exp_inf << fmt.wfrac),                   # sin=cos=+inf
        ("raw_neg_inf", sgn | (fmt.exp_inf << fmt.wfrac)),          # sin=cos=-inf
        ("raw_all_ones", mask(fmt.wfull)),                          # negative non-canonical
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
            for k in range(1, 8):                                   # 1/4, 1/2, 3/4, 1, 5/4, 3/2, 7/4 turns
                out.append((f"quarter_{'-' if sign else '+'}{k}_4", turns(sign, k)))
            # Just before/after a quarter-turn boundary (x = 1/4 +- 1 ULP) -- exercises the boundary quadrant rule.
            q = fmt.bias - 2                                        # biased exp for 0.25
            if 2 <= q <= fmt.exp_max_finite:                       # 0.25 - 1 ULP sits one binade lower (needs exp q-1 >= 1)
                out.append((f"just_below_quarter_{sign}", normal(fmt, sign, q - 1, fmt.frac_mask)))
            if 1 <= q <= fmt.exp_max_finite:
                out.append((f"just_above_quarter_{sign}", normal(fmt, sign, q, 1)))
            # Integer turns (frac == 0 -> quadrant 0, sin=+0, cos=+1) and a large finite with zero fraction.
            for k in (0, 1, 2):
                be = fmt.bias + k
                if 1 <= be <= fmt.exp_max_finite:
                    out.append((f"int_turn_{sign}_{k}", normal(fmt, sign, be, 0)))
            out.append((f"large_int_{sign}", normal(fmt, sign, fmt.exp_max_finite, 0)))
            # Tiny phases below the reducer resolution (the bypass): smallest few exponents with assorted fractions.
            for be in (1, 2, 3):
                if be <= fmt.exp_max_finite:
                    out.append((f"tiny_{sign}_{be}_a", normal(fmt, sign, be, 1)))
                    out.append((f"tiny_{sign}_{be}_b", normal(fmt, sign, be, fmt.frac_mask)))
    return out


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[SincosCase]:
    cases: list[SincosCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for x in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, x)
        return cases

    for label, value in directed_values(fmt):
        add_unique(cases, seen, label, fmt, value)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        x = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        add_unique(cases, seen, "random", fmt, x)
    return cases


@cocotb.test()
async def sincos_runtime_cases(dut) -> None:
    # Latency is data-independent and published: measure accept->out_valid and assert it equals sincos_latency() (the
    # model shared by the RTL LATENCY parameter and the synthesis reports) so the model cannot drift from the RTL.
    context = float_context("sincos")
    fmt = ZkfFormat(context.wexp, context.wman)
    expected_latency = sincos_latency(
        fmt,
        unroll100=context.unroll100,
        parallel=context.parallel,
        stage_input=context.stage_input,
        stage_output=context.stage_output,
        stage_product=context.stage_product,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
    )
    check_width("x", dut.x, fmt.wfull, context)
    check_width("sin", dut.sin, fmt.wfull, context)
    check_width("cos", dut.cos, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = 1                                      # latency measurement assumes always-ready
    drive_unsigned(dut.x, 0)
    for _ in range(4):
        await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, f"{context.prefix()}: out_valid asserted during reset"
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    timeout = 4 * (context.wman + 64)                            # generous upper bound on the iterative latency
    checked = 0
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: in_ready stuck low (case {index})"
        dut.in_valid.value = 1
        drive_unsigned(dut.x, case.x)
        await RisingEdge(dut.clk)                                # this edge accepts the transaction
        dut.in_valid.value = 0
        drive_unsigned(dut.x, mask(fmt.wfull))                  # garbage between transactions
        guard = 0
        while int(dut.out_valid.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: out_valid timeout (case {index})"
        assert guard == expected_latency, (                     # II is data-independent; verify the published model
            f"{context.prefix()} case={index}: measured latency {guard} != model {expected_latency} "
            f"(unroll100={context.unroll100} parallel={context.parallel} SPROD={context.stage_product} "
            f"SN={context.stage_normalize} SPACK={context.stage_pack})"
        )
        got = {"sin": int(dut.sin.value), "cos": int(dut.cos.value), "quadrant": int(dut.quadrant.value)}
        exp = {"sin": case.sin, "cos": case.cos, "quadrant": case.quadrant}
        assert got == exp, (
            f"{context.prefix()} case={index} {case.describe(fmt)}: got {got} expected {exp}"
        )
        checked += 1
    assert checked == len(cases), f"{context.prefix()} checked {checked}, expected {len(cases)}"


@cocotb.test()
async def sincos_backpressure(dut) -> None:
    # Back-pressure: while out_ready is low, out_valid and (sin, cos, quadrant) persist and in_ready stays low; the
    # result is taken only on a cycle with out_valid & out_ready.
    context = float_context("sincos")
    fmt = ZkfFormat(context.wexp, context.wman)
    cases = cases_for(fmt, "directed", context.seed, 0)[:2]
    assert len(cases) == 2, f"{context.prefix()}: need two directed cases for the back-pressure test"

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    drive_unsigned(dut.x, 0)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    timeout = 4 * (context.wman + 64)
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()} bp: in_ready stuck low (case {index})"
        dut.in_valid.value = 1
        drive_unsigned(dut.x, case.x)
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        drive_unsigned(dut.x, mask(fmt.wfull))
        guard = 0
        while int(dut.out_valid.value) == 0:                    # out_ready is low; out_valid still arrives on time
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()} bp: out_valid timeout (case {index})"
        exp = {"sin": case.sin, "cos": case.cos, "quadrant": case.quadrant}
        for hold in range(5):
            got = {"sin": int(dut.sin.value), "cos": int(dut.cos.value), "quadrant": int(dut.quadrant.value)}
            assert int(dut.out_valid.value) == 1, f"{context.prefix()} bp case={index}: out_valid dropped while stalled"
            assert int(dut.in_ready.value) == 0, f"{context.prefix()} bp case={index}: in_ready high while result unread"
            assert got == exp, f"{context.prefix()} bp case={index} hold={hold}: result changed while stalled: {got} != {exp}"
            await RisingEdge(dut.clk)
        dut.out_ready.value = 1
        await RisingEdge(dut.clk)
        dut.out_ready.value = 0
    # After the last consume, in_ready must be able to go high again.
    guard = 0
    while int(dut.in_ready.value) == 0:
        await RisingEdge(dut.clk)
        guard += 1
        assert guard < timeout, f"{context.prefix()} bp: in_ready did not recover after consume"

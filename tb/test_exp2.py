#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import normal
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_latency import exp2_latency
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class UnaryCase:
    label: str
    x: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[UnaryCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    key = x & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    cases.append(UnaryCase(label, x, fmt.wrap(x).exp2().bits))


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = [
        ("raw_zero", 0),
        ("raw_one_frac", 1),
        ("raw_frac_mask", fmt.frac_mask),
        ("raw_neg_zero", 1 << fmt.sign_shift),
        ("raw_pos_inf", fmt.exp_inf << fmt.wfrac),
        ("raw_neg_inf", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac)),
        ("raw_all_ones", mask(fmt.wfull)),
    ]
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            out.append((f"num_{label}", value))
        # Integer exponents: 2**x is an exact power of two -> exercises the f==0 path and pack overflow/underflow.
        for k in (1, 2, 3, -1, -2, -3):
            exp = fmt.bias + k
            if 1 <= exp <= fmt.exp_max_finite:
                out.append((f"int_{k}", normal(fmt, 0, exp, 0)))
                out.append((f"int_neg_{k}", normal(fmt, 1, exp, 0)))
        # Half-integer arguments (e.g. 2**0.5) and near-1 small magnitudes.
        out.append(("half_pos", normal(fmt, 0, fmt.bias - 1, 1 << (fmt.wfrac - 1))))
        out.append(("tiny_pos", normal(fmt, 0, 1, 0)))
        out.append(("tiny_neg", normal(fmt, 1, 1, 0)))
    return out


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[UnaryCase]:
    cases: list[UnaryCase] = []
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
async def exp2_runtime_cases(dut) -> None:
    context = float_context("exp2")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.x.value = 0

    register_stages = exp2_latency(
        fmt,
        stage_input=context.stage_input,
        stage_reduce=context.stage_reduce,
        stage_product=context.stage_product,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    )
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: UnaryCase) -> dict[str, int]:
        drive_unsigned(dut.x, case.x)
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.x, mask(fmt.wfull))

    def describe(index: int, case: UnaryCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(cases), (
        f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"
    )

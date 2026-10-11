#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand, raw_directed_values
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, check_combinational, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class FiniteCase:
    label: str
    x: int
    finite: int
    saturated: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[FiniteCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    x &= mask(fmt.wfull)
    if x in seen:
        return
    seen.add(x)
    value = fmt.wrap(x)
    cases.append(FiniteCase(label, x, int(value.is_finite), value.saturate().bits))


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[FiniteCase]:
    cases: list[FiniteCase] = []
    seen: set[int] = set()
    if kind == "exhaustive":
        for x in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, x)
        return cases
    for index, value in enumerate(raw_directed_values(fmt)):
        add_unique(cases, seen, f"raw_{index}", fmt, value)
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            add_unique(cases, seen, label, fmt, value)
    if kind == "directed":
        return cases
    rng = np.random.default_rng(seed)
    while len(cases) < count:
        x = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        add_unique(cases, seen, "random", fmt, x)
    return cases


@cocotb.test()
async def finite_runtime_cases(dut) -> None:
    context = float_context("finite")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("finite", dut.finite, 1, context)
    check_width("saturated", dut.saturated, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)
    register_stages = fmt.model_of("finite")(
        stage_input=context.stage_input, stage_output=context.stage_output
    ).timing.latency

    def drive_case(case: FiniteCase) -> dict[str, int]:
        drive_unsigned(dut.x, case.x)
        return {"finite": case.finite, "saturated": case.saturated}

    outputs = {"finite": dut.finite, "saturated": dut.saturated}
    if register_stages == 0:
        await check_combinational(dut, context.prefix(), outputs, cases[:256], drive_case)
    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.x.value = 0
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"finite": (dut.finite, 1), "saturated": (dut.saturated, fmt.wfull)},
        reset_passthrough=register_stages == 0,
    )

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.x, (1 << fmt.wfull) - 1)

    def describe(index: int, case: FiniteCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> dict[str, int]:
        dut.in_valid.value = 1
        return drive_case(cases[0])

    await scoreboard.reset(register_stages + 2, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

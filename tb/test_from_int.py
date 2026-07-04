#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_bits import signed_int_max, signed_int_min, signed_to_bits
from zkf_operands import directed_integers, random_integer
from zkf_latency import from_int_latency
from zkf_params import cast_context, check_width
from zkf_stream import RegisterStageScoreboard, drive_signed, run_stream_cases, start_clock


@dataclass(frozen=True)
class FromIntCase:
    label: str
    value: int
    expected: int

    def describe(self, fmt: ZkfFormat, wint: int) -> str:
        return f"{self.label} value={self.value} bits={hex_bits(signed_to_bits(self.value, wint), wint)}"


def add_unique(
    cases: list[FromIntCase],
    seen: set[int],
    label: str,
    fmt: ZkfFormat,
    wint: int,
    value: int,
) -> None:
    key = signed_to_bits(value, wint)
    if key in seen:
        return
    seen.add(key)
    cases.append(FromIntCase(label, value, fmt.from_int(wint, value).bits))


def directed_case_values(wint: int) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for label, value in directed_integers(wint).items():
        cases.append((f"directed_{label}", value))
    # Round-boundary cases (powers of two and neighbours) for any WINT.
    int_max = signed_int_max(wint)
    int_min = signed_int_min(wint)
    extra_positive = [3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 127, 129]
    extra_negative = [-3, -5, -7, -8, -9, -15, -16, -17, -31, -33, -63, -65, -127, -129]
    for v in extra_positive + extra_negative:
        if int_min <= v <= int_max:
            cases.append((f"manual_{v}", v))
    return cases


def cases_for(fmt: ZkfFormat, wint: int, kind: str, seed: int, count: int) -> list[FromIntCase]:
    cases: list[FromIntCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        int_min = signed_int_min(wint)
        int_max = signed_int_max(wint)
        for value in range(int_min, int_max + 1):
            add_unique(cases, seen, "exhaustive", fmt, wint, value)
        return cases

    for label, value in directed_case_values(wint):
        add_unique(cases, seen, label, fmt, wint, value)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        value = random_integer(wint, rng)
        add_unique(cases, seen, "random", fmt, wint, value)
    return cases


@cocotb.test()
async def from_int_runtime_cases(dut) -> None:
    context = cast_context("from_int")
    fmt = ZkfFormat(context.wexp, context.wman)
    wint = context.wint
    assert wint is not None

    check_width("a", dut.a, wint, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, wint, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_signed(dut.a, 0)

    register_stages = from_int_latency(
        stage_input=context.stage_input,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    )
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: FromIntCase) -> dict[str, int]:
        drive_signed(dut.a, case.value)
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_signed(dut.a, signed_int_max(wint))

    def describe(index: int, case: FromIntCase) -> str:
        return f"case={index} {case.describe(fmt, wint)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(cases), (
        f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"
    )

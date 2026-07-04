#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_latency import add_latency
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class AddSubCase:
    label: str
    a: int
    b: int
    op_sub: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        op = "-" if self.op_sub else "+"
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} {op} b={hex_bits(self.b, fmt.wfull)}"


def addsub_reference(fmt: ZkfFormat, a: int, b: int, op_sub: int) -> int:
    b_effective = b ^ ((op_sub & 1) << fmt.sign_shift)
    return (fmt.wrap(a) + fmt.wrap(b_effective)).bits


def raw_directed_values(fmt: ZkfFormat) -> list[int]:
    return [
        0,
        1,
        fmt.frac_mask,
        1 << fmt.sign_shift,
        (1 << fmt.sign_shift) | min(fmt.frac_mask, 1),
        fmt.exp_inf << fmt.wfrac,
        (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac),
        mask(fmt.wfull),
    ]


def add_unique(
    cases: list[AddSubCase],
    seen: set[tuple[int, int, int]],
    label: str,
    fmt: ZkfFormat,
    a: int,
    b: int,
    op_sub: int,
) -> None:
    key = (a & mask(fmt.wfull), b & mask(fmt.wfull), op_sub & 1)
    if key in seen:
        return
    seen.add(key)
    cases.append(AddSubCase(label, a, b, op_sub & 1, addsub_reference(fmt, a, b, op_sub)))


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    values = [(f"raw_{index}", value) for index, value in enumerate(raw_directed_values(fmt))]
    if fmt.wexp >= 3:
        values.extend(directed_numbers(fmt).items())
    return values


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int, int]]:
    values = directed_values(fmt)
    cases = []
    for left_label, a in values:
        for right_label, b in values:
            cases.append((f"add_{left_label}_{right_label}", a, b, 0))
            cases.append((f"sub_{left_label}_{right_label}", a, b, 1))
    return cases


def binary32_manual_cases() -> list[tuple[str, int, int, int, int]]:
    return [
        ("manual_sub_just_below_half_min", 0x01000000, 0x00C00001, 1, 0x00000000),
        ("manual_sub_exact_half_min", 0x01000000, 0x00C00000, 1, 0x00800000),
        ("manual_sub_three_quarters_min", 0x01000000, 0x00A00000, 1, 0x00800000),
        ("manual_sub_negative_exact_half_min", 0x81000000, 0x80C00000, 1, 0x80800000),
        ("manual_sub_negative_three_quarters_min", 0x81000000, 0x80A00000, 1, 0x80800000),
    ]


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int, int]:
    mode = int(rng.integers(0, 8))
    if mode <= 4:
        a = random_operand(fmt, rng)
        b = random_operand(fmt, rng)
    elif mode == 5:
        a = random_bits(fmt.wfull, rng)
        b = random_operand(fmt, rng)
    elif mode == 6:
        a = random_operand(fmt, rng)
        b = random_bits(fmt.wfull, rng)
    else:
        a = random_bits(fmt.wfull, rng)
        b = random_bits(fmt.wfull, rng)
    return a, b, int(rng.integers(0, 2))


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[AddSubCase]:
    cases: list[AddSubCase] = []
    seen: set[tuple[int, int, int]] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            for b in range(1 << fmt.wfull):
                add_unique(cases, seen, "exhaustive_add", fmt, a, b, 0)
                add_unique(cases, seen, "exhaustive_sub", fmt, a, b, 1)
        return cases

    for label, a, b, op_sub in directed_case_operands(fmt):
        add_unique(cases, seen, label, fmt, a, b, op_sub)

    if (fmt.wexp, fmt.wman) == (8, 24):
        for label, a, b, op_sub, expected in binary32_manual_cases():
            actual = addsub_reference(fmt, a, b, op_sub)
            if actual != expected:
                raise AssertionError(f"{label}: expected {expected:08x}, model returned {actual:08x}")
            add_unique(cases, seen, label, fmt, a, b, op_sub)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        a, b, op_sub = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, a, b, op_sub)
    return cases


@cocotb.test()
async def addsub_runtime_cases(dut) -> None:
    context = float_context("addsub")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("op_sub", dut.op_sub, 1, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.op_sub.value = 0

    register_stages = add_latency(
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
        stage_align=context.stage_align,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    )
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: AddSubCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        dut.op_sub.value = case.op_sub
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_unsigned(dut.b, 0)
        dut.op_sub.value = 1

    def describe(index: int, case: AddSubCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(4, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(cases), (
        f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"
    )

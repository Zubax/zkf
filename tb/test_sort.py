#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_latency import cmp_latency
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class SortCase:
    label: str
    a: int
    b: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


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


def special_class_representatives(fmt: ZkfFormat) -> list[tuple[str, int]]:
    """Non-canonical zero/inf patterns at representative fractions; mirrors test_cmp."""
    frac_bits = [0, 1, fmt.frac_mask >> 1, fmt.frac_mask] if fmt.wfrac >= 2 else [0, fmt.frac_mask]
    frac_bits = sorted({f & fmt.frac_mask for f in frac_bits})
    cases: list[tuple[str, int]] = []
    for sign in (0, 1):
        for frac in frac_bits:
            cases.append((f"zero_s{sign}_f{frac:x}", (sign << fmt.sign_shift) | frac))
            inf_bits = (sign << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac) | frac
            cases.append((f"inf_s{sign}_f{frac:x}", inf_bits))
    return cases


def corner_pairs(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    """Special-class transitions (mirrors test_cmp) plus equal-operand reflexivity."""
    pairs: list[tuple[str, int, int]] = []
    specials = special_class_representatives(fmt)
    for left_label, left in specials:
        for right_label, right in specials:
            pairs.append((f"{left_label}_vs_{right_label}", left, right))
    if fmt.wexp >= 3:
        named = directed_numbers(fmt)
        for label, value in named.items():
            pairs.append((f"self_{label}", value, value))
            neighbour = value ^ 0x1
            pairs.append((f"{label}_vs_lsbflip", value, neighbour & mask(fmt.wfull)))
    return pairs


def add_unique(
    cases: list[SortCase],
    seen: set[tuple[int, int]],
    label: str,
    fmt: ZkfFormat,
    a: int,
    b: int,
) -> None:
    key = (a & mask(fmt.wfull), b & mask(fmt.wfull))
    if key in seen:
        return
    seen.add(key)
    cases.append(SortCase(label, a, b))


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    values = [(f"raw_{index}", value) for index, value in enumerate(raw_directed_values(fmt))]
    if fmt.wexp >= 3:
        values.extend(directed_numbers(fmt).items())

    cases = []
    for left_label, a in values:
        for right_label, b in values:
            cases.append((f"{left_label}_vs_{right_label}", a, b))
    return cases


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(0, 8))
    if mode <= 4:
        return random_operand(fmt, rng), random_operand(fmt, rng)
    if mode == 5:
        return random_bits(fmt.wfull, rng), random_operand(fmt, rng)
    if mode == 6:
        return random_operand(fmt, rng), random_bits(fmt.wfull, rng)
    return random_bits(fmt.wfull, rng), random_bits(fmt.wfull, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[SortCase]:
    cases: list[SortCase] = []
    seen: set[tuple[int, int]] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            for b in range(1 << fmt.wfull):
                add_unique(cases, seen, "exhaustive", fmt, a, b)
        return cases

    for label, a, b in directed_case_operands(fmt):
        add_unique(cases, seen, label, fmt, a, b)

    for label, a, b in corner_pairs(fmt):
        add_unique(cases, seen, label, fmt, a, b)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        a, b = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, a, b)
    return cases


@cocotb.test()
async def sort_runtime_cases(dut) -> None:
    context = float_context("sort")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("min", dut.min, fmt.wfull, context)
    check_width("max", dut.max, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0

    register_stages = cmp_latency(stage_input=context.stage_input)
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"min": (dut.min, fmt.wfull), "max": (dut.max, fmt.wfull)},
    )

    def drive_case(case: SortCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        lo, hi = fmt.wrap(case.a).sort(fmt.wrap(case.b))
        return {"min": lo.bits, "max": hi.bits}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_unsigned(dut.b, 0)

    def describe(index: int, case: SortCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(3, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

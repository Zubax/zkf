#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf.oracle import mul
from zkf_bits import hex_bits, mask
from zkf_operands import (
    directed_numbers,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class BinaryCase:
    label: str
    a: int
    b: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


def add_unique(
    cases: list[BinaryCase],
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
    expected = (fmt.wrap(a) * fmt.wrap(b)).bits
    np_ref = mul(fmt.wrap(a), fmt.wrap(b))
    if np_ref is not None and np_ref.bits != expected:
        raise AssertionError(
            f"NumPy cross-check failed for mul {fmt}: a={hex_bits(a, fmt.wfull)} "
            f"b={hex_bits(b, fmt.wfull)} exact={hex_bits(expected, fmt.wfull)} "
            f"numpy={hex_bits(np_ref.bits, fmt.wfull)}"
        )
    cases.append(BinaryCase(label, a, b, expected))


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    v = directed_numbers(fmt)
    return [
        ("zero_times_zero", v["zero"], v["zero"]),
        ("zero_times_neg_zero_payload", v["zero"], v["neg_zero"]),
        ("zero_times_one", v["zero"], v["one"]),
        ("zero_payload_beats_inf", v["neg_zero"], v["noncanonical_pos_inf"]),
        ("zero_exp_payload_ignored", v["minus_one"], v["neg_zero"]),
        ("one_times_one", v["one"], v["one"]),
        ("minus_one_times_one", v["minus_one"], v["one"]),
        ("minus_one_times_minus_one", v["minus_one"], v["minus_one"]),
        ("one_and_half_times_two", v["one_and_half"], v["two"]),
        ("one_and_quarter_times_one_and_half", v["one_and_quarter"], v["one_and_half"]),
        ("normalization_carry", v["one_and_half"], v["one_and_half"]),
        ("pos_inf_times_one", v["pos_inf"], v["one"]),
        ("noncanonical_pos_inf_times_one", v["noncanonical_pos_inf"], v["one"]),
        ("noncanonical_neg_inf_times_one", v["noncanonical_neg_inf"], v["one"]),
        ("one_times_pos_inf", v["one"], v["pos_inf"]),
        ("minus_one_times_pos_inf", v["minus_one"], v["noncanonical_pos_inf"]),
        ("two_times_neg_inf", v["two"], v["noncanonical_neg_inf"]),
        ("zero_times_inf", v["zero"], v["pos_inf"]),
        ("zero_payload_times_inf", v["neg_zero"], v["noncanonical_neg_inf"]),
        ("inf_times_inf", v["pos_inf"], v["noncanonical_pos_inf"]),
        ("neg_inf_times_pos_inf", v["neg_inf"], v["noncanonical_pos_inf"]),
        ("neg_inf_times_neg_inf", v["noncanonical_neg_inf"], v["noncanonical_neg_inf"]),
        ("min_normal_times_half_to_min", v["min_normal"], v["half"]),
        ("min_normal_times_one", v["min_normal"], v["one"]),
        ("neg_min_normal_times_one", v["neg_min_normal"], v["one"]),
        ("max_finite_times_one", v["max_finite"], v["one"]),
        ("neg_max_finite_times_one", v["neg_max_finite"], v["one"]),
        ("max_finite_overflow", v["max_finite"], v["two"]),
        ("neg_max_finite_overflow", v["neg_max_finite"], v["two"]),
    ]


def binary32_manual_cases() -> list[tuple[str, int, int, int]]:
    return [
        ("manual_zero", 0x00000000, 0x3F800000, 0x00000000),
        ("manual_zero_payload_beats_inf", 0x805A5A5A, 0x7FFFFFFF, 0x00000000),
        ("manual_zero_exp_ignored", 0xBF800000, 0x007FFFFF, 0x00000000),
        ("manual_one", 0x3F800000, 0x3F800000, 0x3F800000),
        ("manual_neg_one", 0xBF800000, 0x3F800000, 0xBF800000),
        ("manual_neg_neg", 0xBF800000, 0xBF800000, 0x3F800000),
        ("manual_1p5_times_2", 0x3FC00000, 0x40000000, 0x40400000),
        ("manual_1p25_times_1p5", 0x3FA00000, 0x3FC00000, 0x3FF00000),
        ("manual_product_carry", 0x3FC00000, 0x3FC00000, 0x40100000),
        ("manual_inf", 0x7F800000, 0x3F800000, 0x7F800000),
        ("manual_noncanonical_inf", 0x7FFFFFFF, 0x3F800000, 0x7F800000),
        ("manual_noncanonical_neg_inf", 0xFFABCDEF, 0x3F800000, 0xFF800000),
        ("manual_finite_times_inf", 0x3F800000, 0x7F800000, 0x7F800000),
        ("manual_negative_finite_times_inf", 0xBF800000, 0x7F800001, 0xFF800000),
        ("manual_finite_times_neg_inf", 0x40000000, 0xFF800001, 0xFF800000),
        ("manual_zero_times_inf", 0x00000000, 0x7F800000, 0x00000000),
        ("manual_zero_payload_times_inf", 0x805A5A5A, 0xFFABCDEF, 0x00000000),
        ("manual_inf_times_inf", 0x7F800000, 0x7FFFFFFF, 0x7F800000),
        ("manual_neg_inf_times_inf", 0xFF800000, 0x7F800001, 0xFF800000),
        ("manual_neg_inf_times_neg_inf", 0xFFFFFFFF, 0xFFABCDEF, 0x7F800000),
        ("manual_just_below_half_min_flush", 0x00800000, 0x3EFFFFFD, 0x00000000),
        ("manual_exact_half_min_to_min", 0x00800000, 0x3F000000, 0x00800000),
        ("manual_three_quarters_min_to_min", 0x00800000, 0x3F400000, 0x00800000),
        ("manual_negative_three_quarters_min_to_min", 0x80800000, 0x3F400000, 0x80800000),
        ("manual_min_normal", 0x00800000, 0x3F800000, 0x00800000),
        ("manual_negative_min_normal", 0x80800000, 0x3F800000, 0x80800000),
        ("manual_max_finite", 0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF),
        ("manual_negative_max_finite", 0xFF7FFFFF, 0x3F800000, 0xFF7FFFFF),
        ("manual_positive_overflow", 0x7F7FFFFF, 0x40000000, 0x7F800000),
        ("manual_negative_overflow", 0xFF7FFFFF, 0x40000000, 0xFF800000),
        ("manual_tie_retained_even", 0x3F800002, 0x3FA00000, 0x3FA00002),
        ("manual_tie_retained_odd", 0x3F800001, 0x3FC00000, 0x3FC00002),
        ("manual_round_down", 0x3F800001, 0x3FA00000, 0x3FA00001),
        ("manual_round_up", 0x3F800001, 0x3FE00000, 0x3FE00002),
    ]


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    v = directed_numbers(fmt)
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 2:
        return random_zero(fmt, rng), random_inf(fmt, rng)
    if mode == 3:
        return random_inf(fmt, rng), random_zero(fmt, rng)
    if mode == 4:
        return random_normal(fmt, rng), random_inf(fmt, rng)
    if mode == 5:
        return random_inf(fmt, rng), random_normal(fmt, rng)
    if mode == 6:
        return random_inf(fmt, rng), random_inf(fmt, rng)
    if mode == 7:
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [fmt.bias], [0])
    if mode == 8:
        return random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]), v["two"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[BinaryCase]:
    cases: list[BinaryCase] = []
    seen: set[tuple[int, int]] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            for b in range(1 << fmt.wfull):
                add_unique(cases, seen, "exhaustive", fmt, a, b)
        return cases

    if fmt.wexp >= 3:
        for label, a, b in directed_case_operands(fmt):
            add_unique(cases, seen, label, fmt, a, b)

    if (fmt.wexp, fmt.wman) == (8, 24):
        for label, a, b, expected in binary32_manual_cases():
            actual = (fmt.wrap(a) * fmt.wrap(b)).bits
            if actual != expected:
                raise AssertionError(f"{label}: expected {expected:08x}, model returned {actual:08x}")
            add_unique(cases, seen, label, fmt, a, b)

    if (fmt.wexp, fmt.wman) == (6, 18):
        add_unique(cases, seen, "default_parameter_smoke", fmt, 0x3E0000, 0x3E0000)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        a, b = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, a, b)
    return cases


@cocotb.test()
async def mul_runtime_cases(dut) -> None:
    context = float_context("mul")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0

    register_stages = fmt.model_of("mul")(
        stage_input=context.stage_input,
        stage_product=context.stage_product,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: BinaryCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_unsigned(dut.b, 0)

    def describe(index: int, case: BinaryCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

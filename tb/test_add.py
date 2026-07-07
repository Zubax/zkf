#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf.oracle import add
from zkf_bits import hex_bits, mask
from zkf_operands import normal, zero
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
class AddCase:
    label: str
    a: int
    b: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} + b={hex_bits(self.b, fmt.wfull)}"


def add_unique(
    cases: list[AddCase],
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
    expected = (fmt.wrap(a) + fmt.wrap(b)).bits
    np_ref = add(fmt.wrap(a), fmt.wrap(b))
    if np_ref is not None and np_ref.bits != expected:
        raise AssertionError(
            f"NumPy cross-check failed for add {fmt}: a={hex_bits(a, fmt.wfull)} + "
            f"b={hex_bits(b, fmt.wfull)} exact={hex_bits(expected, fmt.wfull)} "
            f"numpy={hex_bits(np_ref.bits, fmt.wfull)}"
        )
    cases.append(AddCase(label, a, b, expected))


def focused_directed(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    cases: list[tuple[str, int, int]] = []
    wman = fmt.wman
    bias = fmt.bias
    exp_max = fmt.exp_max_finite
    frac_mask = fmt.frac_mask
    one_sig = 1 << fmt.wfrac

    cases.append(("equal_exp_swap_subtract", normal(fmt, 0, bias, 0), normal(fmt, 1, bias, 1)))
    cases.append(("equal_exp_swap_subtract_neg", normal(fmt, 1, bias, 0), normal(fmt, 0, bias, 1)))
    cases.append(("zero_plus_zero_same_sign", zero(fmt), zero(fmt)))

    for exp in (1, bias, exp_max):
        for frac in (0, frac_mask):
            cases.append((f"full_cancel_exp{exp}_frac{frac}", normal(fmt, 0, exp, frac), normal(fmt, 1, exp, frac)))

    if (one_sig + (one_sig >> 1)) <= (frac_mask + one_sig):
        cases.append(
            (
                "sub_shift_zero_high_top_two_bits",
                normal(fmt, 0, bias, (one_sig >> 1) - 1 if one_sig > 1 else 0),
                normal(fmt, 1, bias - 1, 0),
            )
        )
    cases.append(("sub_shift_full_normalize", normal(fmt, 0, bias, 1), normal(fmt, 1, bias, 0)))

    diff = 1
    while diff <= max(wman + 4, 1) and diff <= exp_max - 1:
        small_exp = max(1, exp_max - diff)
        cases.append(
            (
                f"exp_diff_pow2_{diff}_same_sign",
                normal(fmt, 0, exp_max, 0),
                normal(fmt, 0, small_exp, frac_mask),
            )
        )
        cases.append(
            (
                f"exp_diff_pow2_{diff}_opposite_sign",
                normal(fmt, 0, exp_max, 0),
                normal(fmt, 1, small_exp, frac_mask),
            )
        )
        diff <<= 1

    sat_diff = max(wman + 4, exp_max - 1)
    if exp_max > 2 and sat_diff >= 1:
        small_exp = max(1, exp_max - sat_diff)
        if small_exp < exp_max:
            cases.append(
                ("far_saturating_align_same", normal(fmt, 0, exp_max, 0), normal(fmt, 0, small_exp, frac_mask))
            )
            cases.append(
                ("far_saturating_align_opposite", normal(fmt, 0, exp_max, 0), normal(fmt, 1, small_exp, frac_mask))
            )

    if exp_max > wman:
        small_exp_for_guard = exp_max - wman
        if small_exp_for_guard >= 1:
            cases.append(
                (
                    "round_carry_at_max_exp",
                    normal(fmt, 0, exp_max, frac_mask),
                    normal(fmt, 0, small_exp_for_guard, 0),
                )
            )

    if exp_max >= wman + 2:
        cases.append(
            (
                "underflow_one_below_min_round_carry_candidate",
                normal(fmt, 0, 2, 0),
                normal(fmt, 1, 1, frac_mask),
            )
        )

    return cases


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    v = directed_numbers(fmt)
    cases = [
        ("zero_plus_one", v["zero"], v["one"]),
        ("zero_payload_plus_one", v["neg_zero"], v["one"]),
        ("one_plus_zero_payload", v["one"], v["neg_zero"]),
        ("one_plus_one", v["one"], v["one"]),
        ("one_plus_minus_one", v["one"], v["minus_one"]),
        ("minus_one_plus_one", v["minus_one"], v["one"]),
        ("minus_one_plus_minus_one", v["minus_one"], v["minus_one"]),
        ("one_plus_half", v["one"], v["half"]),
        ("half_plus_minus_one", v["half"], v["minus_one"]),
        ("one_and_half_plus_one_and_quarter", v["one_and_half"], v["one_and_quarter"]),
        ("normalization_carry", v["one_and_three_quarters"], v["one_and_three_quarters"]),
        ("min_normal_plus_min_normal", v["min_normal"], v["min_normal"]),
        ("min_normal_plus_neg_min_normal", v["min_normal"], v["neg_min_normal"]),
        ("max_finite_plus_one", v["max_finite"], v["one"]),
        ("max_finite_plus_max_finite", v["max_finite"], v["max_finite"]),
        ("neg_max_finite_plus_neg_max_finite", v["neg_max_finite"], v["neg_max_finite"]),
        ("max_finite_plus_neg_max_finite", v["max_finite"], v["neg_max_finite"]),
        ("pos_inf_plus_one", v["pos_inf"], v["one"]),
        ("one_plus_pos_inf", v["one"], v["pos_inf"]),
        ("one_plus_neg_inf", v["one"], v["noncanonical_neg_inf"]),
        ("pos_inf_plus_pos_inf", v["pos_inf"], v["noncanonical_pos_inf"]),
        ("neg_inf_plus_neg_inf", v["neg_inf"], v["noncanonical_neg_inf"]),
        ("pos_inf_plus_neg_inf", v["pos_inf"], v["neg_inf"]),
        ("neg_inf_plus_pos_inf", v["noncanonical_neg_inf"], v["pos_inf"]),
        ("near_cancellation_positive", normal(fmt, 0, fmt.bias, 2), normal(fmt, 1, fmt.bias, 1)),
        ("near_cancellation_negative", normal(fmt, 1, fmt.bias, 2), normal(fmt, 0, fmt.bias, 1)),
        ("underflow_after_cancellation", normal(fmt, 0, 1, 1), normal(fmt, 1, 1, 0)),
        ("noncanonical_zero_plus_noncanonical_inf", v["neg_zero"], v["noncanonical_pos_inf"]),
        ("noncanonical_opposite_infinities_zero", v["noncanonical_pos_inf"], v["noncanonical_neg_inf"]),
        ("noncanonical_opposite_infinities_zero_reversed", v["noncanonical_neg_inf"], v["noncanonical_pos_inf"]),
    ]

    high_exp = min(fmt.exp_max_finite, fmt.bias + 4)
    for exp_diff in range(5):
        cases.append(
            (
                f"exp_diff_{exp_diff}_same_sign",
                normal(fmt, 0, high_exp, 0),
                normal(fmt, 0, high_exp - exp_diff, fmt.frac_mask),
            )
        )
        cases.append(
            (
                f"exp_diff_{exp_diff}_opposite_sign",
                normal(fmt, 0, high_exp, 0),
                normal(fmt, 1, high_exp - exp_diff, fmt.frac_mask),
            )
        )

    round_exp = fmt.exp_max_finite
    half_exp = round_exp - fmt.wman
    below_half_exp = half_exp - 1
    if half_exp >= 1:
        half_ulp = normal(fmt, 0, half_exp, 0)
        cases.extend(
            [
                ("tie_retains_even_add", normal(fmt, 0, round_exp, 0), half_ulp),
                ("tie_rounds_odd_up_add", normal(fmt, 0, round_exp, 1), half_ulp),
                ("round_up_above_half_add", normal(fmt, 0, round_exp, 0), normal(fmt, 0, half_exp, 1)),
                ("post_round_overflow_add", v["max_finite"], half_ulp),
            ]
        )
    if below_half_exp >= 1:
        cases.append(
            (
                "round_down_below_half_add",
                normal(fmt, 0, round_exp, 0),
                normal(fmt, 0, below_half_exp, fmt.frac_mask),
            )
        )

    sticky_diff = fmt.wman + 4
    if fmt.exp_max_finite - sticky_diff >= 1:
        cases.append(
            (
                "far_operand_sticky_only",
                normal(fmt, 0, fmt.exp_max_finite, 0),
                normal(fmt, 0, fmt.exp_max_finite - sticky_diff, fmt.frac_mask),
            )
        )

    cases.extend(focused_directed(fmt))
    if (fmt.wexp, fmt.wman) == (6, 100):
        cases.append(
            (
                "w6_m100_wide_cancel_lsb",
                normal(fmt, 0, fmt.bias, (1 << (fmt.wfrac - 1)) - 1),
                normal(fmt, 1, fmt.bias, (1 << (fmt.wfrac - 1)) - 2),
            )
        )
    return cases


def binary32_manual_cases() -> list[tuple[str, int, int, int]]:
    return [
        ("manual_zero_plus_zero", 0x00000000, 0x00000000, 0x00000000),
        ("manual_neg_zero_plus_zero", 0x80000000, 0x00000000, 0x00000000),
        ("manual_noncanonical_zero_payloads", 0x80123456, 0x007FEDCB, 0x00000000),
        ("manual_one_plus_one", 0x3F800000, 0x3F800000, 0x40000000),
        ("manual_one_minus_one", 0x3F800000, 0xBF800000, 0x00000000),
        ("manual_neg_one_plus_neg_one", 0xBF800000, 0xBF800000, 0xC0000000),
        ("manual_one_plus_half", 0x3F800000, 0x3F000000, 0x3FC00000),
        ("manual_half_plus_neg_one", 0x3F000000, 0xBF800000, 0xBF000000),
        ("manual_swap_equal_exp_sub", 0x3F800000, 0xBF800001, 0xB4000000),
        ("manual_swap_equal_exp_sub_neg", 0xBF800000, 0x3F800001, 0x34000000),
        ("manual_min_normal_plus_neg_min", 0x00800000, 0x80800000, 0x00000000),
        ("manual_min_normal_plus_min_normal", 0x00800000, 0x00800000, 0x01000000),
        ("manual_cancel_just_below_half_min", 0x01000000, 0x80C00001, 0x00000000),
        ("manual_cancel_exact_half_min", 0x01000000, 0x80C00000, 0x00800000),
        ("manual_cancel_three_quarters_min", 0x01000000, 0x80A00000, 0x00800000),
        ("manual_cancel_negative_exact_half_min", 0x81000000, 0x00C00000, 0x80800000),
        ("manual_cancel_negative_three_quarters_min", 0x81000000, 0x00A00000, 0x80800000),
        ("manual_max_finite_plus_one", 0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF),
        ("manual_max_minus_max", 0x7F7FFFFF, 0xFF7FFFFF, 0x00000000),
        ("manual_max_plus_max_overflow", 0x7F7FFFFF, 0x7F7FFFFF, 0x7F800000),
        ("manual_neg_max_plus_neg_max_overflow", 0xFF7FFFFF, 0xFF7FFFFF, 0xFF800000),
        ("manual_round_carry_tie_to_inf", 0x7F7FFFFF, 0x73000000, 0x7F800000),
        ("manual_inf_plus_one", 0x7F800000, 0x3F800000, 0x7F800000),
        ("manual_one_plus_inf", 0x3F800000, 0x7F800000, 0x7F800000),
        ("manual_neg_inf_plus_one", 0xFF800000, 0x3F800000, 0xFF800000),
        ("manual_one_plus_noncanonical_neg_inf", 0x3F800000, 0xFFABCDEF, 0xFF800000),
        ("manual_pos_inf_plus_pos_inf", 0x7F800000, 0x7FABCDEF, 0x7F800000),
        ("manual_neg_inf_plus_neg_inf", 0xFF800000, 0xFFFFFFFF, 0xFF800000),
        ("manual_pos_inf_plus_neg_inf_to_zero", 0x7F800000, 0xFF800000, 0x00000000),
        ("manual_noncanonical_inf_opposite", 0x7FABCDEF, 0xFFABCDEF, 0x00000000),
        ("manual_tie_to_even_lo", 0x3F800000, 0x33800000, 0x3F800000),
        ("manual_tie_to_odd_round_up", 0x3F800001, 0x33800000, 0x3F800002),
        ("manual_round_up_above_half", 0x3F800000, 0x33C00000, 0x3F800001),
        ("manual_round_down_below_half", 0x3F800001, 0x33000000, 0x3F800001),
        ("manual_far_operand_only_sticky", 0x3F800000, 0x20000000, 0x3F800000),
        ("manual_far_cancellation_at_max", 0x7F7FFFFF, 0xFF7FFFFE, 0x73800000),
    ]


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    v = directed_numbers(fmt)
    mode = int(rng.integers(0, 14))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 2:
        return random_inf(fmt, rng), random_operand(fmt, rng)
    if mode == 3:
        return random_operand(fmt, rng), random_inf(fmt, rng)
    if mode == 4:
        return random_inf(fmt, rng), random_inf(fmt, rng)
    if mode == 5:
        exp = int(rng.integers(1, fmt.exp_max_finite + 1))
        frac = int(rng.integers(0, fmt.frac_mask))
        sign = int(rng.integers(0, 2))
        return normal(fmt, sign, exp, frac + 1), normal(fmt, sign ^ 1, exp, frac)
    if mode == 6:
        sign = int(rng.integers(0, 2))
        large_exp = int(rng.integers(max(1, fmt.exp_max_finite - 2), fmt.exp_max_finite + 1))
        small_exp = int(rng.integers(1, max(2, min(fmt.exp_max_finite, fmt.wman + 3))))
        large_frac = int(rng.integers(0, fmt.frac_mask + 1))
        small_frac = int(rng.integers(0, fmt.frac_mask + 1))
        return normal(fmt, sign, large_exp, large_frac), normal(fmt, sign ^ 1, small_exp, small_frac)
    if mode == 7:
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [1, 2], [0, 1])
    if mode == 8:
        return random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]), random_normal_near(
            fmt,
            rng,
            [fmt.exp_max_finite],
            [fmt.frac_mask],
        )
    if mode == 9:
        return v["max_finite"], v["max_finite"]
    if mode == 10:
        return v["neg_max_finite"], v["neg_max_finite"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[AddCase]:
    cases: list[AddCase] = []
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
            actual = (fmt.wrap(a) + fmt.wrap(b)).bits
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
async def add_runtime_cases(dut) -> None:
    context = float_context("add")
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

    register_stages = fmt.model_of("add")(
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
        stage_align=context.stage_align,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: AddCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_unsigned(dut.b, 0)

    def describe(index: int, case: AddCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(4, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

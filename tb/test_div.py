#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cocotb
import numpy as np

from zkf import Zkf, ZkfFormat
from zkf.oracle import div
from zkf_bits import hex_bits, mask
from zkf_operands import normal
from zkf_operands import (
    directed_numbers,
    normal_from_significands,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf_latency import div_latency
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class BinaryCase:
    label: str
    a: int
    b: int
    expected: int
    div0: int = 0

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


@dataclass(frozen=True)
class DivObservation:
    high: bool
    significand_lsb: int
    guard: int
    round_bit: int
    produced_tail: bool
    final_rem_sticky: bool

    @property
    def sticky(self) -> bool:
        return self.produced_tail or self.final_rem_sticky

    @property
    def round_increment(self) -> bool:
        return bool(self.guard and (self.round_bit or self.sticky or self.significand_lsb))


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
    result = fmt.wrap(a).div(fmt.wrap(b))
    expected, div0 = result.quotient.bits, int(result.div_by_zero)
    np_ref = div(fmt.wrap(a), fmt.wrap(b))
    if np_ref is not None and (np_ref.quotient.bits, int(np_ref.div_by_zero)) != (expected, div0):
        raise AssertionError(
            f"NumPy cross-check failed for div {fmt}: a={hex_bits(a, fmt.wfull)} "
            f"b={hex_bits(b, fmt.wfull)} exact=({hex_bits(expected, fmt.wfull)}, {div0}) "
            f"numpy=({hex_bits(np_ref.quotient.bits, fmt.wfull)}, {int(np_ref.div_by_zero)})"
        )
    cases.append(BinaryCase(label, a, b, expected, div0))


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    v = directed_numbers(fmt)
    return [
        ("zero_div_one", v["zero"], v["one"]),
        ("negative_zero_encoding_div_inf", v["neg_zero"], v["pos_inf"]),
        ("zero_div_zero", v["zero"], v["zero"]),
        ("one_div_zero", v["one"], v["zero"]),
        ("minus_one_div_zero", v["minus_one"], v["zero"]),
        ("pos_inf_div_zero", v["pos_inf"], v["zero"]),
        ("neg_inf_div_zero", v["neg_inf"], v["zero"]),
        ("one_div_zero_payload", v["one"], v["neg_zero"]),
        ("one_div_one", v["one"], v["one"]),
        ("minus_one_div_one", v["minus_one"], v["one"]),
        ("one_div_minus_one", v["one"], v["minus_one"]),
        ("minus_one_div_minus_one", v["minus_one"], v["minus_one"]),
        ("one_and_half_div_two", v["one_and_half"], v["two"]),
        ("one_and_quarter_div_one_and_half", v["one_and_quarter"], v["one_and_half"]),
        ("one_and_half_div_one_and_half", v["one_and_half"], v["one_and_half"]),
        ("one_div_pos_inf", v["one"], v["pos_inf"]),
        ("minus_one_div_pos_inf", v["minus_one"], v["pos_inf"]),
        ("two_div_neg_inf", v["two"], v["neg_inf"]),
        ("pos_inf_div_one", v["pos_inf"], v["one"]),
        ("neg_inf_div_one", v["neg_inf"], v["one"]),
        ("noncanonical_pos_inf_div_minus_one", v["noncanonical_pos_inf"], v["minus_one"]),
        ("pos_inf_div_pos_inf", v["pos_inf"], v["pos_inf"]),
        ("neg_inf_div_pos_inf", v["neg_inf"], v["pos_inf"]),
        ("noncanonical_inf_div_noncanonical_inf", v["noncanonical_neg_inf"], v["noncanonical_pos_inf"]),
        ("min_normal_div_two_to_min", v["min_normal"], v["two"]),
        ("neg_min_normal_div_two_to_min", v["neg_min_normal"], v["two"]),
        ("min_normal_div_one", v["min_normal"], v["one"]),
        ("neg_min_normal_div_one", v["neg_min_normal"], v["one"]),
        ("max_finite_div_one", v["max_finite"], v["one"]),
        ("neg_max_finite_div_one", v["neg_max_finite"], v["one"]),
        ("max_finite_div_half_overflow", v["max_finite"], v["half"]),
        ("neg_max_finite_div_half_overflow", v["neg_max_finite"], v["half"]),
    ]


def qfrac(fmt: ZkfFormat) -> int:
    qfrac_base = fmt.wman + 2
    return qfrac_base + (qfrac_base % 2)


def div_observation(fmt: ZkfFormat, a: int, b: int) -> DivObservation | None:
    da = Zkf(fmt, a)
    db = Zkf(fmt, b)
    if not da.is_normal or not db.is_normal:
        return None

    qf = qfrac(fmt)
    sig_a = (1 << fmt.wfrac) | da.frac
    sig_b = (1 << fmt.wfrac) | db.frac
    raw = (sig_a << qf) // sig_b
    rem = (sig_a << qf) % sig_b
    high = ((raw >> qf) & 1) != 0

    if high:
        sig_shift = qf - fmt.wman + 1
        guard_shift = qf - fmt.wman
        round_shift = qf - fmt.wman - 1
        tail_width = qf - fmt.wman - 1
    else:
        sig_shift = qf - fmt.wman
        guard_shift = qf - fmt.wman - 1
        round_shift = qf - fmt.wman - 2
        tail_width = qf - fmt.wman - 2

    sig = (raw >> sig_shift) & mask(fmt.wman)
    tail_mask = mask(tail_width) if tail_width > 0 else 0
    return DivObservation(
        high=high,
        significand_lsb=sig & 1,
        guard=(raw >> guard_shift) & 1,
        round_bit=(raw >> round_shift) & 1,
        produced_tail=(raw & tail_mask) != 0,
        final_rem_sticky=rem != 0,
    )


def find_rounding_case(
    fmt: ZkfFormat,
    rng: np.random.Generator,
    predicate: Callable[[DivObservation], bool],
    max_random: int = 200_000,
) -> tuple[int, int] | None:
    lo = 1 << fmt.wfrac
    hi = 1 << fmt.wman

    if fmt.wman <= 11:
        for ma in range(lo, hi):
            for mb in range(lo, hi):
                a, b = normal_from_significands(fmt, ma, mb)
                obs = div_observation(fmt, a, b)
                if obs is not None and predicate(obs):
                    return a, b

    for _ in range(max_random):
        a, b = normal_from_significands(fmt, int(rng.integers(lo, hi)), int(rng.integers(lo, hi)))
        obs = div_observation(fmt, a, b)
        if obs is not None and predicate(obs):
            return a, b

    return None


def rounding_directed(fmt: ZkfFormat, rng: np.random.Generator) -> list[tuple[str, int, int]]:
    predicates: list[tuple[str, Callable[[DivObservation], bool]]] = [
        ("high_quotient_normalization", lambda obs: obs.high),
        ("low_quotient_normalization", lambda obs: not obs.high),
        ("guard_round_increment", lambda obs: bool(obs.guard and obs.round_bit and obs.round_increment)),
        (
            "sticky_from_produced_tail",
            lambda obs: bool(obs.guard and not obs.round_bit and obs.produced_tail and obs.round_increment),
        ),
        (
            "sticky_from_final_remainder",
            lambda obs: bool(obs.guard and not obs.round_bit and not obs.produced_tail and obs.final_rem_sticky),
        ),
    ]
    cases: list[tuple[str, int, int]] = []
    for label, predicate in predicates:
        case = find_rounding_case(fmt, rng, predicate)
        if case is not None:
            cases.append((label, *case))
    return cases


def radix_digit_cases(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    one_sig = 1 << fmt.wfrac
    b = normal(fmt, 0, fmt.bias, 0)
    quartile = one_sig // 4
    if quartile < 1:
        return []
    cases: list[tuple[str, int, int]] = []
    targets = {
        "radix_digit_0": 1,
        "radix_digit_1": quartile,
        "radix_digit_2": 2 * quartile,
        "radix_digit_3": 3 * quartile,
    }
    for label, rem_target in targets.items():
        if 0 <= rem_target <= fmt.frac_mask:
            cases.append((label, normal(fmt, 0, fmt.bias, rem_target), b))
    return cases


def binary32_manual_cases() -> list[tuple[str, int, int, int, int]]:
    return [
        ("manual_zero_div_one", 0x00000000, 0x3F800000, 0x00000000, 0),
        ("manual_zero_payload_div_inf", 0x805A5A5A, 0x7F800000, 0x00000000, 0),
        ("manual_zero_div_zero", 0x00000000, 0x00000000, 0x00000000, 1),
        ("manual_one_div_zero", 0x3F800000, 0x00000000, 0x7F800000, 1),
        ("manual_minus_one_div_zero", 0xBF800000, 0x00000000, 0xFF800000, 1),
        ("manual_one_div_zero_payload", 0x3F800000, 0x805A5A5A, 0x7F800000, 1),
        ("manual_one_div_one", 0x3F800000, 0x3F800000, 0x3F800000, 0),
        ("manual_minus_one_div_one", 0xBF800000, 0x3F800000, 0xBF800000, 0),
        ("manual_one_div_minus_one", 0x3F800000, 0xBF800000, 0xBF800000, 0),
        ("manual_minus_one_div_minus_one", 0xBF800000, 0xBF800000, 0x3F800000, 0),
        ("manual_1p5_div_2", 0x3FC00000, 0x40000000, 0x3F400000, 0),
        ("manual_1p25_div_1p5", 0x3FA00000, 0x3FC00000, 0x3F555555, 0),
        ("manual_1p5_div_1p5", 0x3FC00000, 0x3FC00000, 0x3F800000, 0),
        ("manual_one_div_inf", 0x3F800000, 0x7F800000, 0x00000000, 0),
        ("manual_minus_one_div_inf", 0xBF800000, 0x7F800000, 0x00000000, 0),
        ("manual_two_div_neg_inf", 0x40000000, 0xFF800000, 0x00000000, 0),
        ("manual_inf_div_one", 0x7F800000, 0x3F800000, 0x7F800000, 0),
        ("manual_neg_inf_div_one", 0xFF800000, 0x3F800000, 0xFF800000, 0),
        ("manual_noncanonical_inf_div_minus_one", 0x7F812345, 0xBF800000, 0xFF800000, 0),
        ("manual_inf_div_inf", 0x7F800000, 0x7F800000, 0x00000000, 0),
        ("manual_neg_inf_div_inf", 0xFF800000, 0x7F800000, 0x00000000, 0),
        ("manual_noncanonical_inf_div_noncanonical_inf", 0xFFFFFFFF, 0x7F812345, 0x00000000, 0),
        ("manual_min_normal_div_two_to_min", 0x00800000, 0x40000000, 0x00800000, 0),
        ("manual_min_normal_div_four_flush", 0x00800000, 0x40800000, 0x00000000, 0),
        ("manual_min_normal_div_one_and_half_to_min", 0x00800000, 0x3FC00000, 0x00800000, 0),
        ("manual_min_normal", 0x00800000, 0x3F800000, 0x00800000, 0),
        ("manual_negative_min_normal", 0x80800000, 0x3F800000, 0x80800000, 0),
        ("manual_max_finite", 0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF, 0),
        ("manual_negative_max_finite", 0xFF7FFFFF, 0x3F800000, 0xFF7FFFFF, 0),
        ("manual_positive_overflow", 0x7F7FFFFF, 0x3F000000, 0x7F800000, 0),
        ("manual_negative_overflow", 0xFF7FFFFF, 0x3F000000, 0xFF800000, 0),
        ("manual_three_div_two", 0x40400000, 0x40000000, 0x3FC00000, 0),
        ("manual_round_case_0", 0x3F800002, 0x3FA00000, 0x3F4CCCD0, 0),
        ("manual_round_case_1", 0x3F800001, 0x3FC00000, 0x3F2AAAAC, 0),
        ("manual_round_case_2", 0x3F800001, 0x3FA00000, 0x3F4CCCCE, 0),
        ("manual_round_case_3", 0x3F800001, 0x3FE00000, 0x3F124926, 0),
    ]


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    v = directed_numbers(fmt)
    mode = int(rng.integers(0, 13))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 2:
        return random_normal(fmt, rng), random_inf(fmt, rng)
    if mode == 3:
        return random_inf(fmt, rng), random_normal(fmt, rng)
    if mode == 4:
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [fmt.bias], [0])
    if mode == 5:
        return random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]), v["half"]
    if mode == 6:
        return random_normal_near(fmt, rng, [fmt.bias], [0, 1]), random_normal_near(
            fmt,
            rng,
            [fmt.bias],
            [0, 1, 1 << (fmt.wfrac - 1)],
        )
    if mode == 7:
        return normal(fmt, int(rng.integers(0, 2)), 1, int(rng.integers(0, min(4, fmt.frac_mask + 1)))), v["two"]
    if mode == 8:
        return normal(fmt, int(rng.integers(0, 2)), fmt.exp_max_finite, fmt.frac_mask), v["half"]
    if mode == 9:
        return random_normal(fmt, rng), v["one"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[BinaryCase]:
    cases: list[BinaryCase] = []
    seen: set[tuple[int, int]] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            for b in range(1 << fmt.wfull):
                add_unique(cases, seen, "exhaustive", fmt, a, b)
        return cases

    rng = np.random.default_rng(seed)
    if fmt.wexp >= 3:
        for label, a, b in directed_case_operands(fmt):
            add_unique(cases, seen, label, fmt, a, b)
        for label, a, b in rounding_directed(fmt, rng):
            add_unique(cases, seen, label, fmt, a, b)
        for label, a, b in radix_digit_cases(fmt):
            add_unique(cases, seen, label, fmt, a, b)

    if (fmt.wexp, fmt.wman) == (8, 24):
        for label, a, b, expected, expected_div0 in binary32_manual_cases():
            result = fmt.wrap(a).div(fmt.wrap(b))
            actual, actual_div0 = result.quotient.bits, int(result.div_by_zero)
            if (actual, actual_div0) != (expected, expected_div0):
                raise AssertionError(
                    f"{label}: expected ({expected:08x}, {expected_div0}), "
                    f"model returned ({actual:08x}, {actual_div0})"
                )
            add_unique(cases, seen, label, fmt, a, b)

    if (fmt.wexp, fmt.wman) == (6, 18):
        add_unique(cases, seen, "default_parameter_smoke", fmt, 0x3E0000, 0x3E0000)

    if kind == "directed":
        return cases

    while len(cases) < count:
        a, b = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, a, b)
    return cases


@cocotb.test()
async def div_runtime_cases(dut) -> None:
    context = float_context("div")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("q", dut.q, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0

    register_stages = div_latency(
        fmt.wman,
        stage_input=context.stage_input,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    )
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"q": (dut.q, fmt.wfull), "div0": (dut.div0, 1)},
    )

    def drive_case(case: BinaryCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        return {"q": case.expected, "div0": case.div0}

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

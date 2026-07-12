#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_bits import signed_int_max, signed_int_min, signed_to_bits
from zkf_operands import (
    directed_numbers,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf_params import cast_context, check_width
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class ToIntCase:
    label: str
    a: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} expected={self.expected}"


def add_unique(
    cases: list[ToIntCase],
    seen: set[int],
    label: str,
    fmt: ZkfFormat,
    wint: int,
    a: int,
) -> None:
    key = a & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    cases.append(ToIntCase(label, key, fmt.wrap(key).to_int(wint)))


def directed_case_inputs(fmt: ZkfFormat) -> list[tuple[str, int]]:
    v = directed_numbers(fmt)
    return [
        ("zero", v["zero"]),
        ("neg_zero_payload", v["neg_zero"]),
        ("one", v["one"]),
        ("minus_one", v["minus_one"]),
        ("half_tie_to_zero_even", v["half"]),
        ("one_and_half_tie_to_two", v["one_and_half"]),
        ("one_and_quarter", v["one_and_quarter"]),
        ("one_and_three_quarters", v["one_and_three_quarters"]),
        ("two", v["two"]),
        ("min_normal", v["min_normal"]),
        ("neg_min_normal", v["neg_min_normal"]),
        ("max_finite_saturates", v["max_finite"]),
        ("neg_max_finite_saturates", v["neg_max_finite"]),
        ("pos_inf_saturates_to_int_max", v["pos_inf"]),
        ("neg_inf_saturates_to_int_min", v["neg_inf"]),
        ("noncanonical_pos_inf", v["noncanonical_pos_inf"]),
        ("noncanonical_neg_inf", v["noncanonical_neg_inf"]),
    ]


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    mode = int(rng.integers(0, 8))
    if mode == 0:
        return random_zero(fmt, rng)
    if mode == 1:
        return random_inf(fmt, rng)
    if mode == 2:
        return random_normal_near(fmt, rng, [fmt.bias - 1, fmt.bias, fmt.bias + 1], [0, 1, fmt.frac_mask])
    if mode == 3:
        return random_normal_near(
            fmt,
            rng,
            [fmt.exp_max_finite - 2, fmt.exp_max_finite - 1, fmt.exp_max_finite],
            [0, 1 << (fmt.wfrac - 1), fmt.frac_mask],
        )
    if mode == 4:
        return random_normal_near(fmt, rng, [1, 2, 3], [0, 1])
    return random_operand(fmt, rng)


def rcarry_overflow_inputs(fmt: ZkfFormat, wint: int) -> list[tuple[str, int]]:
    """
    Floats that force the rounding carry to flip bit WINT of the rounded magnitude: mag_pre[WINT-1:0]
    all-1s with guard=1, so mag_pre + 1 sets the rcarry bit in zkf_to_int. Only reachable when WMAN > WINT
    (else mag_pre cannot fill the low WINT bits) and exp_unbiased fits the format's normal range.
    """
    cases: list[tuple[str, int]] = []
    if fmt.wman <= wint:
        return cases
    shamt = fmt.wman - wint  # right-shift amount that aligns the all-1s pattern at bit 0
    if shamt < 1:
        return cases
    exp_unbiased = fmt.wfrac - shamt
    if not (fmt.min_exp_unbiased <= exp_unbiased <= fmt.max_exp_unbiased):
        return cases
    # sig = (2^WINT - 1) << shamt + 2^(shamt-1) = 2^WMAN - 2^(shamt-1)
    # => mag_pre = 2^WINT - 1, guard = sig[shamt-1] = 1, sticky = 0.
    sig = (1 << fmt.wman) - (1 << (shamt - 1))
    frac = sig - (1 << fmt.wfrac)
    exp_biased = exp_unbiased + fmt.bias
    bits_pos = (exp_biased << fmt.wfrac) | frac
    bits_neg = bits_pos | (1 << fmt.sign_shift)
    cases.append(("rcarry_overflow_pos", bits_pos))
    cases.append(("rcarry_overflow_neg", bits_neg))
    return cases


def signed_boundary_inputs(fmt: ZkfFormat, wint: int) -> list[tuple[str, int]]:
    exp = wint - 1
    if not (fmt.min_exp_unbiased <= exp <= fmt.max_exp_unbiased):
        return []

    def pow2(e: int) -> Fraction:
        return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)

    boundary = pow2(exp)
    below_step = max(Fraction(1), pow2(exp - 1 - fmt.wfrac))
    above_step = max(Fraction(1), pow2(exp - fmt.wfrac))
    return [
        ("int_max_float_neighbor", fmt.encode(boundary - below_step).bits),
        ("int_max_plus_one", fmt.encode(boundary).bits),
        ("int_min_exact", fmt.encode(-boundary).bits),
        ("int_min_float_neighbor_overflow", fmt.encode(-(boundary + above_step)).bits),
    ]


def cases_for(fmt: ZkfFormat, wint: int, kind: str, seed: int, count: int) -> list[ToIntCase]:
    cases: list[ToIntCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, wint, a)
        return cases

    if fmt.wexp >= 3:
        for label, a in directed_case_inputs(fmt):
            add_unique(cases, seen, label, fmt, wint, a)
    else:
        for a in range(1 << fmt.wfull):
            add_unique(cases, seen, "small_format", fmt, wint, a)

    for label, a in rcarry_overflow_inputs(fmt, wint):
        add_unique(cases, seen, label, fmt, wint, a)

    for label, a in signed_boundary_inputs(fmt, wint):
        add_unique(cases, seen, label, fmt, wint, a)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        a = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, wint, a)
    return cases


@cocotb.test()
async def to_int_runtime_cases(dut) -> None:
    context = cast_context("to_int")
    fmt = ZkfFormat(context.wexp, context.wman)
    wint = context.wint
    assert wint is not None

    check_width("a", dut.a, fmt.wfull, context)
    check_width("y", dut.y, wint, context)
    cases = cases_for(fmt, wint, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_unsigned(dut.a, 0)

    register_stages = fmt.model_of("to_int")(wint=wint, stage_input=context.stage_input).latency
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, wint)})

    def drive_case(case: ToIntCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        return {"y": signed_to_bits(case.expected, wint)}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)

    def describe(index: int, case: ToIntCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

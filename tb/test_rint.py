#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import cocotb
import numpy as np

from zkf import RoundMode, ZkfFormat
from zkf_bits import hex_bits, mask, pow2_fraction, signed_to_bits
from zkf_operands import directed_numbers, random_inf, random_normal, random_normal_near, random_operand, random_zero
from zkf_params import cast_context, check_width
from zkf_stream import RegisterStageScoreboard, check_combinational, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class RintCase:
    label: str
    a: int
    mode: RoundMode
    y_float: int
    y_int: int

    def describe(self, fmt: ZkfFormat) -> str:
        return (
            f"{self.label} mode={self.mode} a={hex_bits(self.a, fmt.wfull)} "
            f"y_float={hex_bits(self.y_float, fmt.wfull)} y_int={self.y_int}"
        )


def add_unique(cases: list[RintCase], seen: set[int], label: str, fmt: ZkfFormat, wint: int, a: int) -> None:
    """Register one operand under every rounding mode."""
    key = a & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    value = fmt.wrap(key)
    for mode in RoundMode:
        cases.append(RintCase(label, key, mode, value.rint(mode).bits, value.rint_int(wint, mode)))


def directed_case_inputs(fmt: ZkfFormat, wint: int) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = [
        ("raw_zero_clean", 0),
        ("raw_zero_neg_payload", 1 << fmt.sign_shift),
        ("raw_zero_payload", min(fmt.frac_mask, 1)),
        ("raw_inf_pos", fmt.exp_inf << fmt.wfrac),
        ("raw_inf_neg", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac)),
        ("raw_inf_noncanonical_pos", (fmt.exp_inf << fmt.wfrac) | min(fmt.frac_mask, 1)),
        ("raw_inf_noncanonical_neg", mask(fmt.wfull)),
    ]
    values: list[tuple[str, Fraction]] = []
    if fmt.wexp >= 3:  # the named values and the fine neighbors of the ties need an exponent on both sides of zero
        named = directed_numbers(fmt)
        cases.extend(named.items())
        for label in ("half", "one_and_quarter", "one_and_half", "one_and_three_quarters", "two"):
            cases.append((f"negative_{label}", named[label] | (1 << fmt.sign_shift)))
        for label, tie in (
            ("half", Fraction(1, 2)),
            ("one_and_half", Fraction(3, 2)),
            ("two_and_half", Fraction(5, 2)),
        ):
            values += [(f"fine_below_{label}", tie - fmt.epsilon), (f"fine_above_{label}", tie + fmt.epsilon)]
    # Around the integer rails: the rounding carry into the sign bit and the neighboring representable values.
    rail_exp = wint - 1
    if fmt.min_exp_unbiased <= rail_exp <= fmt.max_exp_unbiased:
        rail = pow2_fraction(rail_exp)
        below = max(Fraction(1), pow2_fraction(rail_exp - 1 - fmt.wfrac))
        above = max(Fraction(1), pow2_fraction(rail_exp - fmt.wfrac))
        values += [
            ("rail", rail),
            ("rail_half_below", rail - Fraction(1, 2)),
            ("rail_half_above", rail + Fraction(1, 2)),
            ("rail_neighbor_below", rail - below),
            ("rail_neighbor_above", rail + above),
        ]
    for label, value in values:
        cases.append((label, fmt.encode(value).bits))
        cases.append((f"negative_{label}", fmt.encode(-value).bits))
    return cases


def random_case(fmt: ZkfFormat, wint: int, rng: np.random.Generator) -> int:
    choice = int(rng.integers(0, 10))
    if choice == 0:
        return random_zero(fmt, rng)
    if choice == 1:
        return random_inf(fmt, rng)
    if choice == 2:  # the |value| < 1 branch and the smallest in-fraction rounds
        return random_normal_near(fmt, rng, [fmt.bias - 2, fmt.bias - 1, fmt.bias, fmt.bias + 1], [0, 1, fmt.frac_mask])
    if choice == 3:  # half-way fractions around small integers
        return random_normal_near(
            fmt,
            rng,
            [fmt.bias, fmt.bias + 1, fmt.bias + 2],
            [0, 1 << (fmt.wfrac - 1), (1 << (fmt.wfrac - 1)) | 1, fmt.frac_mask],
        )
    if choice == 4:  # the top finite exponents, where a float round-up overflows to inf in tiny formats
        return random_normal_near(
            fmt,
            rng,
            [fmt.exp_max_finite - 2, fmt.exp_max_finite - 1, fmt.exp_max_finite],
            [0, 1 << (fmt.wfrac - 1), fmt.frac_mask],
        )
    if choice == 5:  # around the integer rails
        return random_normal_near(fmt, rng, [fmt.bias + wint - 2, fmt.bias + wint - 1], [0, 1, fmt.frac_mask])
    if choice == 6:  # around the exponent at which the fraction disappears
        return random_normal_near(fmt, rng, [fmt.bias + fmt.wfrac - 1, fmt.bias + fmt.wfrac], [0, 1, fmt.frac_mask])
    if choice == 7:
        return random_normal(fmt, rng)
    return random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, wint: int, kind: str, seed: int, count: int) -> list[RintCase]:
    cases: list[RintCase] = []
    seen: set[int] = set()
    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, wint, a)
        return cases
    for label, a in directed_case_inputs(fmt, wint):
        add_unique(cases, seen, label, fmt, wint, a)
    if kind == "directed":
        return cases
    rng = np.random.default_rng(seed)
    target_operands = len(seen) + count  # count counts random operands; each yields one case per mode
    while len(seen) < target_operands:
        add_unique(cases, seen, "random", fmt, wint, random_case(fmt, wint, rng))
    return cases


@cocotb.test()
async def rint_runtime_cases(dut) -> None:
    context = cast_context("rint")
    fmt = ZkfFormat(context.wexp, context.wman)
    wint = context.wint
    assert wint is not None

    check_width("a", dut.a, fmt.wfull, context)
    check_width("round_mode", dut.round_mode, 2, context)
    check_width("y_float", dut.y_float, fmt.wfull, context)
    check_width("y_int", dut.y_int, wint, context)
    cases = cases_for(fmt, wint, context.kind, context.seed, context.count)

    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_unsigned(dut.a, 0)
    dut.round_mode.value = 0

    register_stages = fmt.model_of("rint")(
        wint=wint,
        stage_input=context.stage_input,
        stage_shift=context.stage_shift,
        stage_round=context.stage_round,
        stage_output=context.stage_output,
    ).timing.latency
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"y_float": (dut.y_float, fmt.wfull), "y_int": (dut.y_int, wint)},
        reset_passthrough=register_stages == 0,
    )

    def drive_case(case: RintCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        dut.round_mode.value = case.mode
        return {"y_float": case.y_float, "y_int": signed_to_bits(case.y_int, wint)}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        dut.round_mode.value = 3

    def describe(index: int, case: RintCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> dict[str, int]:
        dut.in_valid.value = 1
        return drive_case(cases[0])

    if register_stages == 0:
        await check_combinational(
            dut, context.prefix(), {"y_float": dut.y_float, "y_int": dut.y_int}, cases[:256], drive_case
        )
    start_clock(dut)
    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

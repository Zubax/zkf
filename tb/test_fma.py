#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf.oracle import fma
from zkf_bits import hex_bits, mask
from zkf_operands import normal
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
class FmaCase:
    label: str
    a: int
    b: int
    c: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return (
            f"{self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)} "
            f"c={hex_bits(self.c, fmt.wfull)}"
        )


def add_unique(
    cases: list[FmaCase],
    seen: set[tuple[int, int, int]],
    label: str,
    fmt: ZkfFormat,
    a: int,
    b: int,
    c: int,
) -> None:
    key = (a & mask(fmt.wfull), b & mask(fmt.wfull), c & mask(fmt.wfull))
    if key in seen:
        return
    seen.add(key)
    expected = fmt.wrap(a).fma(fmt.wrap(b), fmt.wrap(c)).bits
    np_ref = fma(fmt.wrap(a), fmt.wrap(b), fmt.wrap(c))
    if np_ref is not None and np_ref.bits != expected:
        raise AssertionError(
            f"math.fma cross-check failed for fma {fmt}: a={hex_bits(a, fmt.wfull)} "
            f"b={hex_bits(b, fmt.wfull)} c={hex_bits(c, fmt.wfull)} "
            f"exact={hex_bits(expected, fmt.wfull)} fma={hex_bits(np_ref.bits, fmt.wfull)}"
        )
    cases.append(FmaCase(label, a, b, c, expected))


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int, int]]:
    v = directed_numbers(fmt)
    one = v["one"]
    cases: list[tuple[str, int, int, int]] = [
        # c=0 reduces to a rounded product.
        ("one_times_one_plus_zero", one, one, v["zero"]),
        ("two_times_one_plus_one", v["two"], one, one),
        ("one_times_one_plus_one", one, one, one),
        ("one_times_one_minus_one", one, one, v["minus_one"]),
        ("half_times_half_plus_zero", v["half"], v["half"], v["zero"]),
        ("max_times_one_plus_one", v["max_finite"], one, one),
        ("max_times_max_overflow", v["max_finite"], v["max_finite"], v["zero"]),
        ("max_times_max_plus_neg_max", v["max_finite"], v["max_finite"], v["neg_max_finite"]),
        ("neg_max_times_max_overflow", v["neg_max_finite"], v["max_finite"], v["zero"]),
        ("min_normal_times_half_underflow", v["min_normal"], v["half"], v["zero"]),
        ("min_times_min_underflow_plus_min", v["min_normal"], v["min_normal"], v["min_normal"]),
        # Zero product (incl. 0*inf=+0); result is c canonicalized.
        ("zero_times_one_plus_one", v["zero"], one, one),
        ("zero_times_inf_plus_two", v["zero"], v["pos_inf"], v["two"]),
        ("neg_zero_times_inf_plus_one", v["neg_zero"], v["noncanonical_pos_inf"], one),
        ("one_times_zero_plus_neg_one", one, v["zero"], v["minus_one"]),
        ("zero_times_zero_plus_zero", v["zero"], v["zero"], v["zero"]),
        ("inf_times_one_plus_one", v["pos_inf"], one, one),
        ("inf_times_one_minus_inf", v["pos_inf"], one, v["neg_inf"]),
        ("neg_inf_times_one_plus_inf", v["neg_inf"], one, v["pos_inf"]),
        ("inf_times_neg_one_plus_finite", v["pos_inf"], v["minus_one"], v["max_finite"]),
        ("finite_product_plus_inf", v["two"], v["two"], v["noncanonical_neg_inf"]),
        ("inf_times_inf_plus_neg_inf", v["pos_inf"], v["pos_inf"], v["neg_inf"]),
        ("inf_times_zero_plus_inf", v["pos_inf"], v["zero"], v["pos_inf"]),
        ("neg_times_neg_plus_zero", v["minus_one"], v["minus_one"], v["zero"]),
        ("pos_times_neg_plus_zero", one, v["minus_one"], v["zero"]),
        # Product exact; the addend forces a tie/round on the sum.
        ("one_and_half_times_one_plus_quarter", v["one_and_half"], one, v["one_and_quarter"]),
        ("three_quarters_carry", v["one_and_three_quarters"], v["one_and_three_quarters"], v["zero"]),
    ]

    # Catastrophic cancellation: c = -(rounded product), so a*b + c is the tiny exact-minus-rounded residual that
    # drives the leading one deep into the product's low half and forces the full-width normalize a single-rounded FMA
    # must keep but a chained mul->add would discard.
    for label, a, b in (
        ("cancel_one_half", v["one_and_half"], v["one_and_quarter"]),
        ("cancel_three_q", v["one_and_three_quarters"], v["one_and_three_quarters"]),
        ("cancel_max", v["max_finite"], v["one_and_half"]),
        ("cancel_min", v["min_normal"], v["one_and_three_quarters"]),
    ):
        cases.append((label, a, b, (-(fmt.wrap(a) * fmt.wrap(b))).bits))

    # Exponent-difference sweep between the product (~bias) and the addend, both signs.
    high_exp = min(fmt.exp_max_finite, fmt.bias + 4)
    for exp_diff in range(min(fmt.wman + 3, fmt.exp_max_finite)):
        c_exp = max(1, high_exp - exp_diff)
        cases.append((f"prod_vs_c_exp_diff_{exp_diff}_same", one, one, normal(fmt, 0, c_exp, fmt.frac_mask)))
        cases.append((f"prod_vs_c_exp_diff_{exp_diff}_opp", one, one, normal(fmt, 1, c_exp, fmt.frac_mask)))

    # (6,18) witnesses where single-rounding provably differs from add(mul(a,b),c).
    if (fmt.wexp, fmt.wman) == (6, 18):
        for label, a, b, c in (
            ("w6m18_witness0", 0x128B2F, 0xD23F08, 0x892F90),
            ("w6m18_witness1", 0x922766, 0x4EF8AA, 0x8F6D05),
            ("w6m18_witness2", 0x1A61DB, 0x94E3BF, 0x923A73),
        ):
            cases.append((label, a, b, c))

    # (4,30) deep cancellation whose corrected exponent underflows far below the product range: a sub-path exponent
    # field sized only WEXP+2 would wrap the tiny residual to a spurious large finite; these must round to +0.
    if (fmt.wexp, fmt.wman) == (4, 30):
        for label, a, b, c in (
            ("w4m30_cancel_underflow0", 0x08EA99F89, 0x07AEC4FCE, 0x22AF60440),
            ("w4m30_cancel_underflow1", 0x28DD00C89, 0x08A4ABE3C, 0x03C8C11EA),
        ):
            cases.append((label, a, b, c))

    return cases


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int, int]:
    mode = int(rng.integers(0, 12))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 2:
        return random_operand(fmt, rng), random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 3:
        return random_inf(fmt, rng), random_operand(fmt, rng), random_operand(fmt, rng)
    if mode == 4:
        return random_operand(fmt, rng), random_operand(fmt, rng), random_inf(fmt, rng)
    if mode == 5:
        return random_inf(fmt, rng), random_operand(fmt, rng), random_inf(fmt, rng)
    if mode == 6:
        # Addend close to the product magnitude (drives close cancellation / small alignment).
        a = random_normal(fmt, rng)
        b = random_normal(fmt, rng)
        product = fmt.wrap(a) * fmt.wrap(b)
        if int(rng.integers(0, 2)):
            return a, b, (-product).bits
        return a, b, product.bits
    if mode == 7:
        # Addend near the product exponent with an independent fraction/sign.
        a = random_normal(fmt, rng)
        b = random_normal(fmt, rng)
        pe = int(
            np.clip(
                ((a >> fmt.wfrac) & fmt.exp_inf) + ((b >> fmt.wfrac) & fmt.exp_inf) - fmt.bias, 1, fmt.exp_max_finite
            )
        )
        c = normal(fmt, int(rng.integers(0, 2)), pe, int(rng.integers(0, fmt.frac_mask + 1)))
        return a, b, c
    if mode == 8:
        return (
            random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]),
            random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]),
            random_operand(fmt, rng),
        )
    if mode == 9:
        return (
            random_normal_near(fmt, rng, [1, 2], [0, 1]),
            random_normal_near(fmt, rng, [1, 2], [0, 1]),
            random_normal_near(fmt, rng, [1, 2], [0, 1]),
        )
    return random_operand(fmt, rng), random_operand(fmt, rng), random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[FmaCase]:
    cases: list[FmaCase] = []
    seen: set[tuple[int, int, int]] = set()

    if kind == "exhaustive":
        space = 1 << fmt.wfull
        for a in range(space):
            for b in range(space):
                for c in range(space):
                    add_unique(cases, seen, "exhaustive", fmt, a, b, c)
        return cases

    if fmt.wexp >= 3:
        for label, a, b, c in directed_case_operands(fmt):
            add_unique(cases, seen, label, fmt, a, b, c)

    if (fmt.wexp, fmt.wman) == (6, 18):
        add_unique(cases, seen, "default_parameter_smoke", fmt, 0x3E0000, 0x3E0000, 0x3E0000)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        a, b, c = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, a, b, c)
    return cases


@cocotb.test()
async def fma_runtime_cases(dut) -> None:
    context = float_context("fma")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("c", dut.c, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.c.value = 0

    register_stages = fmt.model_of("fma")(
        stage_input=context.stage_input,
        stage_product=context.stage_product,
        stage_decode=context.stage_decode,
        stage_align=context.stage_align,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: FmaCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        drive_unsigned(dut.c, case.c)
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_unsigned(dut.b, 0)
        drive_unsigned(dut.c, (1 << fmt.wfull) - 1)

    def describe(index: int, case: FmaCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

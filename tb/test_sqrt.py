#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf.oracle import sqrt as np_sqrt
from zkf_bits import hex_bits, mask
from zkf_operands import (
    directed_numbers,
    normal,
    random_bits,
    random_inf,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class UnaryCase:
    label: str
    x: int
    y: int
    domain_error: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[UnaryCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    key = x & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    r = fmt.wrap(x).sqrt()
    expected, de = r.root.bits, int(r.domain_error)
    np_ref = np_sqrt(fmt.wrap(x))
    if np_ref is not None and (np_ref.root.bits, int(np_ref.domain_error)) != (expected, de):
        raise AssertionError(
            f"NumPy cross-check failed for sqrt {fmt}: x={hex_bits(x, fmt.wfull)} "
            f"exact=({hex_bits(expected, fmt.wfull)}, {de}) "
            f"numpy=({hex_bits(np_ref.root.bits, fmt.wfull)}, {int(np_ref.domain_error)})"
        )
    cases.append(UnaryCase(label, x, expected, de))


def exact_square_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    """
    Inputs that are exact squares of representable values: root significands with <= WMAN//2 significant bits,
    so y*y is exactly representable and the recurrence terminates with rem == 0 (the guard=0/sticky=0 path);
    squaring arbitrary in-format values rounds and would break the exactness. The root-exponent sweep lands
    the squares on both input exponent parities.
    """
    h = fmt.wman // 2
    out: list[tuple[str, int]] = []
    top_patterns = sorted({0, 1, mask(h - 1), 1 << (h - 2)})
    for i, pattern in enumerate(top_patterns):
        for er in (0, 1, -1):
            root_frac = (pattern << (fmt.wfrac - (h - 1))) & fmt.frac_mask
            root = Fraction((1 << fmt.wfrac) | root_frac, 1 << fmt.wfrac) * Fraction(2) ** er
            square = root * root
            encoded = fmt.encode(square)
            if encoded.is_normal and encoded.to_fraction() == square:
                out.append((f"exact_square_{i}_e{er}", encoded.bits))
    return out


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = [
        ("raw_zero", 0),
        ("raw_zero_payload", 1),
        ("raw_neg_zero", 1 << fmt.sign_shift),  # sign of zero ignored -> +0, no domain error
        ("raw_neg_zero_payload", (1 << fmt.sign_shift) | 1),
        ("raw_pos_inf", fmt.exp_inf << fmt.wfrac),  # +inf
        ("raw_neg_inf", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac)),  # -inf -> domain error
        ("raw_all_ones", mask(fmt.wfull)),  # non-canonical -inf -> domain error
    ]
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            out.append((f"num_{label}", value))  # covers +-min_normal/max_finite/inf and negative classes
        # Powers of two straddling odd/even input exponents (both parities of e = exp - BIAS).
        for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            exp = fmt.bias + k
            if 1 <= exp <= fmt.exp_max_finite:
                out.append((f"pow2_{k}", normal(fmt, 0, exp, 0)))
                out.append((f"neg_pow2_{k}", normal(fmt, 1, exp, 0)))  # negative -> domain error
        out.append(("just_above_one", normal(fmt, 0, fmt.bias, 1)))
        out.append(("just_below_one", normal(fmt, 0, fmt.bias - 1, fmt.frac_mask)))
        # All-ones fraction at both exponent parities: the max-remainder / near-2.0 round-down boundary, and at
        # odd WMAN the r=1 case drives raw to all-ones with rem == raw (the strict-guard corner).
        for k in (0, 1):
            if fmt.bias + k <= fmt.exp_max_finite:
                out.append((f"all_ones_frac_parity{k}", normal(fmt, 0, fmt.bias + k, fmt.frac_mask)))
        out.append(("min_normal_plus_one", normal(fmt, 0, 1, 1)))
        out.append(("second_exp", normal(fmt, 0, 2, 0)))
        for label, value in exact_square_values(fmt):
            out.append((label, value))
    return out


def binary32_manual_cases() -> list[tuple[str, int, int, int]]:
    return [
        ("manual_zero", 0x00000000, 0x00000000, 0),
        ("manual_neg_zero", 0x80000000, 0x00000000, 0),
        ("manual_neg_zero_payload", 0x805A5A5A, 0x00000000, 0),
        ("manual_one", 0x3F800000, 0x3F800000, 0),
        ("manual_two", 0x40000000, 0x3FB504F3, 0),
        ("manual_four", 0x40800000, 0x40000000, 0),
        ("manual_nine", 0x41100000, 0x40400000, 0),
        ("manual_half", 0x3F000000, 0x3F3504F3, 0),
        ("manual_quarter", 0x3E800000, 0x3F000000, 0),
        ("manual_one_and_half", 0x3FC00000, 0x3F9CC471, 0),
        ("manual_min_normal", 0x00800000, 0x20000000, 0),
        ("manual_two_min_normal", 0x01000000, 0x203504F3, 0),
        ("manual_max_finite", 0x7F7FFFFF, 0x5F7FFFFF, 0),
        ("manual_just_above_one", 0x3F800001, 0x3F800000, 0),
        ("manual_just_below_one", 0x3F7FFFFF, 0x3F7FFFFF, 0),
        ("manual_pos_inf", 0x7F800000, 0x7F800000, 0),
        ("manual_neg_inf", 0xFF800000, 0xFF800000, 1),
        ("manual_noncanon_pos_inf", 0x7F812345, 0x7F800000, 0),
        ("manual_noncanon_neg_inf", 0xFFFFFFFF, 0xFF800000, 1),
        ("manual_minus_one", 0xBF800000, 0xFF800000, 1),
        ("manual_neg_min_normal", 0x80800000, 0xFF800000, 1),
        ("manual_neg_max_finite", 0xFF7FFFFF, 0xFF800000, 1),
        ("manual_pi", 0x40490FDB, 0x3FE2DFC5, 0),
        ("manual_e_hundredth", 0x3CDE838A, 0x3E28C3F3, 0),
    ]


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return random_zero(fmt, rng)
    if mode == 1:
        return random_inf(fmt, rng)
    if mode == 2:
        return random_normal_near(fmt, rng, [fmt.bias, fmt.bias + 1], [0, 1, fmt.frac_mask])
    if mode == 3:
        return random_normal_near(fmt, rng, [1, 2], [0, 1, fmt.frac_mask])
    if mode == 4:
        return random_normal_near(fmt, rng, [fmt.exp_max_finite - 1, fmt.exp_max_finite], [0, fmt.frac_mask])
    if mode == 5:
        return random_bits(fmt.wfull, rng)
    return random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[UnaryCase]:
    cases: list[UnaryCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for x in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, x)
        return cases

    for label, value in directed_values(fmt):
        add_unique(cases, seen, label, fmt, value)

    if (fmt.wexp, fmt.wman) == (8, 24):
        for label, x, expected, expected_de in binary32_manual_cases():
            r = fmt.wrap(x).sqrt()
            actual, actual_de = r.root.bits, int(r.domain_error)
            if (actual, actual_de) != (expected, expected_de):
                raise AssertionError(
                    f"{label}: expected ({expected:08x}, {expected_de}), model returned ({actual:08x}, {actual_de})"
                )
            add_unique(cases, seen, label, fmt, x)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        add_unique(cases, seen, "random", fmt, random_case(fmt, rng))
    return cases


@cocotb.test()
async def sqrt_runtime_cases(dut) -> None:
    context = float_context("sqrt")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.x.value = 0

    register_stages = fmt.model_of("sqrt")(
        stage_input=context.stage_input,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"y": (dut.y, fmt.wfull), "domain_error": (dut.domain_error, 1)},
    )

    def drive_case(case: UnaryCase) -> dict[str, int]:
        drive_unsigned(dut.x, case.x)
        return {"y": case.y, "domain_error": case.domain_error}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.x, mask(fmt.wfull))

    def describe(index: int, case: UnaryCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

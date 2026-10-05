#!/usr/bin/env python3
"""
zkf_divsqrt against the references. At a fixed MODE only that operation runs and op_sqrt is noise; a root's b is noise
in every mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf.oracle import div as np_div
from zkf.oracle import sqrt as np_sqrt
from zkf_bits import hex_bits, mask
from zkf_operands import (
    directed_numbers,
    normal,
    random_bits,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class Case:
    label: str
    sqrt: bool
    a: int
    b: int
    y: int
    error: int

    def describe(self, fmt: ZkfFormat) -> str:
        if self.sqrt:
            return f"sqrt {self.label} a={hex_bits(self.a, fmt.wfull)}"
        return f"div {self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


class Cases:
    """Unique cases with the reference's results, cross-checked against NumPy where its formats allow."""

    def __init__(self, fmt: ZkfFormat) -> None:
        self.fmt = fmt
        self.cases: list[Case] = []
        self.seen: set[tuple[int, ...]] = set()  # the operands; one Cases holds one operation

    def division(self, label: str, a: int, b: int) -> None:
        fmt = self.fmt
        a, b = a & mask(fmt.wfull), b & mask(fmt.wfull)
        if (a, b) in self.seen:
            return
        self.seen.add((a, b))
        r = fmt.wrap(a).div(fmt.wrap(b))
        np_ref = np_div(fmt.wrap(a), fmt.wrap(b))
        if np_ref is not None and (np_ref.quotient.bits, np_ref.div_by_zero) != (r.quotient.bits, r.div_by_zero):
            raise AssertionError(f"NumPy disagrees with the reference on {fmt}: {a:#x} / {b:#x}")
        self.cases.append(Case(label, False, a, b, r.quotient.bits, int(r.div_by_zero)))

    def root(self, label: str, x: int) -> None:
        fmt = self.fmt
        x = x & mask(fmt.wfull)
        if (x,) in self.seen:
            return
        self.seen.add((x,))
        r = fmt.wrap(x).sqrt()
        np_ref = np_sqrt(fmt.wrap(x))
        if np_ref is not None and (np_ref.root.bits, np_ref.domain_error) != (r.root.bits, r.domain_error):
            raise AssertionError(f"NumPy disagrees with the reference on {fmt}: sqrt({x:#x})")
        self.cases.append(Case(label, True, x, 0, r.root.bits, int(r.domain_error)))

    def __len__(self) -> int:
        return len(self.cases)


def division_directed(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    """Every pair of special and boundary values, then operands at the prenormalization, the folded first digit, the
    compare-only last decision and the result's exponent limits."""
    out: list[tuple[str, int, int]] = []
    if fmt.wexp >= 3:
        numbers = directed_numbers(fmt)
        out += [(f"{la}_div_{lb}", a, b) for (la, a), (lb, b) in product(numbers.items(), repeat=2)]
    one = fmt.bias << fmt.wfrac
    for f in (0, 1, 2, fmt.frac_mask >> 1, fmt.frac_mask - 1, fmt.frac_mask):
        for delta in (-2, -1, 0, 1, 2):
            if ((one | f) + delta) >> fmt.wfrac == fmt.bias:
                out.append((f"near_equal_{f}_{delta}", (one | f) + delta, one | f))
    out.append(("largest_quotient", one | fmt.frac_mask, one))
    # 4n and 8n exactly on 5b, 6b and 7b, the folded digit's thresholds, where representable.
    sig = 1 << fmt.wfrac
    for k in (5, 6, 7):
        for shift in (2, 3):
            for f in range(min(sig, 64)):
                num = k * (sig | f)
                if num % (1 << shift) == 0 and sig <= num >> shift < 2 * sig:
                    out.append((f"threshold_{k}_{shift}_{f}", one | ((num >> shift) & fmt.frac_mask), one | f))
    for ea in (1, 2, fmt.bias, fmt.exp_max_finite - 1, fmt.exp_max_finite):
        for eb in (1, fmt.bias, fmt.exp_max_finite):
            for fa, fb in ((0, 0), (0, fmt.frac_mask), (fmt.frac_mask, 0)):
                out.append((f"exp_{ea}_{eb}_{fa}_{fb}", normal(fmt, 0, ea, fa), normal(fmt, 0, eb, fb)))
    return out


def root_directed(fmt: ZkfFormat) -> list[tuple[str, int]]:
    """Specials, powers of two at both exponent parities, exact squares and their neighbours, and radicands at the
    compare-only last digit and the guard: a single tail bit at every position, an all-ones fraction."""
    neg, inf = 1 << fmt.sign_shift, fmt.exp_inf << fmt.wfrac
    out: list[tuple[str, int]] = [
        ("zero", 0),
        ("zero_payload", 1),
        ("neg_zero", neg),
        ("neg_zero_payload", neg | 1),
        ("pos_inf", inf),
        ("neg_inf", neg | inf),
        ("all_ones", mask(fmt.wfull)),  # non-canonical -inf
    ]
    if fmt.wexp >= 3:
        out += [(f"num_{label}", value) for label, value in directed_numbers(fmt).items()]
        for k in range(-4, 5):
            if 1 <= fmt.bias + k <= fmt.exp_max_finite:
                out += [
                    (f"pow2_{k}", normal(fmt, 0, fmt.bias + k, 0)),
                    (f"neg_pow2_{k}", normal(fmt, 1, fmt.bias + k, 0)),
                ]
        out += [("min_normal_plus_one", normal(fmt, 0, 1, 1)), ("second_exp", normal(fmt, 0, 2, 0))]
    for e in (fmt.bias, fmt.bias + 1):
        if e > fmt.exp_max_finite:
            continue
        out.append((f"all_ones_frac_{e}", normal(fmt, 0, e, fmt.frac_mask)))
        out += [(f"tail_bit_{e}_{bit}", normal(fmt, 0, e, 1 << bit)) for bit in range(fmt.wfrac)]
    # Squares of roots with at most WMAN/2 significant bits are exact; the root exponents land them on both parities.
    h = fmt.wman // 2
    for q in range(1 << min(h - 1, 6)):
        for er in (0, 1, -1):
            root = Fraction((1 << (h - 1)) | (q << max(0, h - 7)), 1 << (h - 1)) * Fraction(2) ** er
            encoded = fmt.encode(root * root)
            if encoded.is_normal and encoded.to_fraction() == root * root:
                out += [(f"square_{q}_{er}_{d}", encoded.bits + d) for d in (-1, 0, 1)]
    return out


def random_division(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    v = directed_numbers(fmt)
    pick = int(rng.integers(0, 13))
    if pick == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if pick == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if pick == 2:
        return random_normal(fmt, rng), random_inf(fmt, rng)
    if pick == 3:
        return random_inf(fmt, rng), random_normal(fmt, rng)
    if pick == 4:  # quotients near the smallest normal
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [fmt.bias], [0])
    if pick == 5:  # and near the overflow
        return random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]), v["half"]
    if pick == 6:
        divisor = random_normal_near(fmt, rng, [fmt.bias], [0, 1, 1 << (fmt.wfrac - 1)])
        return random_normal_near(fmt, rng, [fmt.bias], [0, 1]), divisor
    if pick == 7:
        return normal(fmt, int(rng.integers(0, 2)), 1, int(rng.integers(0, min(4, fmt.frac_mask + 1)))), v["two"]
    if pick == 8:
        return random_normal(fmt, rng), v["one"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def random_radicand(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    pick = int(rng.integers(0, 10))
    if pick == 0:
        return random_zero(fmt, rng)
    if pick == 1:
        return random_inf(fmt, rng)
    if pick == 2:
        return random_normal_near(fmt, rng, [fmt.bias, fmt.bias + 1], [0, 1, fmt.frac_mask])
    if pick == 3:
        return random_normal_near(fmt, rng, [1, 2], [0, 1, fmt.frac_mask])
    if pick == 4:
        return random_normal_near(fmt, rng, [fmt.exp_max_finite - 1, fmt.exp_max_finite], [0, fmt.frac_mask])
    if pick == 5:
        return random_bits(fmt.wfull, rng)
    return random_operand(fmt, rng)


def binary32_division() -> list[tuple[str, int, int, int, int]]:
    """IEEE binary32 results, and ZKF's own where its zero and infinity handling differs."""
    return [
        ("zero_payload_div_inf", 0x805A5A5A, 0x7F800000, 0x00000000, 0),
        ("zero_div_zero", 0x00000000, 0x00000000, 0x00000000, 1),
        ("one_div_zero_payload", 0x3F800000, 0x805A5A5A, 0x7F800000, 1),
        ("1p25_div_1p5", 0x3FA00000, 0x3FC00000, 0x3F555555, 0),
        ("noncanonical_inf_div_minus_one", 0x7F812345, 0xBF800000, 0xFF800000, 0),
        ("inf_div_inf", 0x7F800000, 0x7F800000, 0x00000000, 0),
        ("noncanonical_inf_div_noncanonical_inf", 0xFFFFFFFF, 0x7F812345, 0x00000000, 0),
        ("min_normal_div_two", 0x00800000, 0x40000000, 0x00800000, 0),
        ("min_normal_div_four_flush", 0x00800000, 0x40800000, 0x00000000, 0),
        ("min_normal_div_1p5", 0x00800000, 0x3FC00000, 0x00800000, 0),
        ("positive_overflow", 0x7F7FFFFF, 0x3F000000, 0x7F800000, 0),
        ("negative_overflow", 0xFF7FFFFF, 0x3F000000, 0xFF800000, 0),
        ("three_div_two", 0x40400000, 0x40000000, 0x3FC00000, 0),
        ("round_case_0", 0x3F800002, 0x3FA00000, 0x3F4CCCD0, 0),
        ("round_case_1", 0x3F800001, 0x3FC00000, 0x3F2AAAAC, 0),
        ("round_case_2", 0x3F800001, 0x3FA00000, 0x3F4CCCCE, 0),
        ("round_case_3", 0x3F800001, 0x3FE00000, 0x3F124926, 0),
    ]


def binary32_root() -> list[tuple[str, int, int, int]]:
    return [
        ("neg_zero_payload", 0x805A5A5A, 0x00000000, 0),
        ("two", 0x40000000, 0x3FB504F3, 0),
        ("nine", 0x41100000, 0x40400000, 0),
        ("half", 0x3F000000, 0x3F3504F3, 0),
        ("one_and_half", 0x3FC00000, 0x3F9CC471, 0),
        ("min_normal", 0x00800000, 0x20000000, 0),
        ("two_min_normal", 0x01000000, 0x203504F3, 0),
        ("max_finite", 0x7F7FFFFF, 0x5F7FFFFF, 0),
        ("just_above_one", 0x3F800001, 0x3F800000, 0),
        ("just_below_one", 0x3F7FFFFF, 0x3F7FFFFF, 0),
        ("noncanon_pos_inf", 0x7F812345, 0x7F800000, 0),
        ("noncanon_neg_inf", 0xFFFFFFFF, 0xFF800000, 1),
        ("neg_min_normal", 0x80800000, 0xFF800000, 1),
        ("pi", 0x40490FDB, 0x3FE2DFC5, 0),
        ("e_hundredth", 0x3CDE838A, 0x3E28C3F3, 0),
    ]


def collect(fmt: ZkfFormat, context, rng: np.random.Generator, sqrt: bool) -> list[Case]:
    cases = Cases(fmt)
    add = cases.root if sqrt else cases.division
    kind, index, shards = context.kind, context.shard_index, context.shard_count
    if kind == "exhaustive":
        if sqrt:
            for x in range(index, 1 << fmt.wfull, shards):
                add("exhaustive", x)
        else:
            for i in range(index, 1 << (2 * fmt.wfull), shards):
                add("exhaustive", i >> fmt.wfull, i & mask(fmt.wfull))
        return cases.cases
    if kind == "significands":
        n = 1 << fmt.wfrac
        if sqrt:  # both exponent parities
            for e in (fmt.bias, fmt.bias + 1) if fmt.bias + 1 <= fmt.exp_max_finite else (fmt.bias,):
                for f in range(index, n, shards):
                    add("significands", normal(fmt, 0, e, f))
        else:  # the exponents only offset the result's
            for i in range(index, n * n, shards):
                add("significands", normal(fmt, 0, fmt.bias, i // n), normal(fmt, 0, fmt.bias, i % n))
        return cases.cases
    if sqrt:
        for label, x in root_directed(fmt):
            add(label, x)
    else:
        for label, a, b in division_directed(fmt):
            add(label, a, b)
    if (fmt.wexp, fmt.wman) == (8, 24):
        for label, *operands, y, error in binary32_root() if sqrt else binary32_division():
            r = fmt.wrap(operands[0]).sqrt() if sqrt else fmt.wrap(operands[0]).div(fmt.wrap(operands[1]))
            got = (r.root.bits, int(r.domain_error)) if sqrt else (r.quotient.bits, int(r.div_by_zero))
            if got != (y, error):
                raise AssertionError(f"binary32 {label}: expected ({y:#010x}, {error}), the reference gives {got}")
            add(f"binary32_{label}", *operands)
    # The random cases come on top of the directed ones, as many as the operand space has left.
    space = 1 << (fmt.wfull if sqrt else 2 * fmt.wfull)
    target = len(cases) + min(context.count, space - len(cases)) if kind == "random" else 0
    while len(cases) < target:
        if sqrt:
            add("random", random_radicand(fmt, rng))
        else:
            add("random", *random_division(fmt, rng))
    return cases.cases


def cases_for(fmt: ZkfFormat, context, rng: np.random.Generator) -> list[Case]:
    cases: list[Case] = []
    if context.mode != 1:
        cases += collect(fmt, context, rng, sqrt=False)
    if context.mode != 0:
        for case in collect(fmt, context, rng, sqrt=True):
            # b is noise, a zero exponent among it (a divisor zero must not raise the error).
            noise = random_bits(fmt.wfull, rng) if rng.integers(0, 3) else int(rng.integers(0, 1 << fmt.wfrac))
            cases.append(replace(case, b=noise))
    return [cases[i] for i in rng.permutation(len(cases))]


@cocotb.test()
async def divsqrt_runtime_cases(dut) -> None:
    context = float_context("divsqrt")
    fmt = ZkfFormat(context.wexp, context.wman)
    for name in ("a", "b", "y"):
        check_width(name, getattr(dut, name), fmt.wfull, context)
    rng = np.random.default_rng(context.seed)
    cases = cases_for(fmt, context, rng)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.op_sqrt.value = 0
    dut.a.value = 0
    dut.b.value = 0

    register_stages = fmt.model_of("divsqrt")(
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
        mode=context.mode,
    ).timing.latency
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"y": (dut.y, fmt.wfull), "error": (dut.error, 1)},
    )

    def drive_case(case: Case) -> dict[str, int]:
        dut.op_sqrt.value = int(case.sqrt) if context.mode == 2 else int(rng.integers(0, 2))
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        return {"y": case.y, "error": case.error}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        dut.op_sqrt.value = int(rng.integers(0, 2))
        drive_unsigned(dut.a, random_bits(fmt.wfull, rng))
        drive_unsigned(dut.b, random_bits(fmt.wfull, rng))

    def describe(index: int, case: Case) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(cases), f"{context.prefix()}: checked {scoreboard.checked} of {len(cases)}"

#!/usr/bin/env python3
"""
zkf_divsqrt against the references, on the zkf_div and zkf_sqrt benches' cases plus its own directed ones. At a fixed
MODE only that operation runs and op_sqrt is noise; a root's b is noise in every mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

import test_div
import test_sqrt
from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import normal, random_bits
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


def division_directed(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    """Operands at the prenormalization, the folded first digit and the compare-only last decision."""
    out: list[tuple[str, int, int]] = []
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
    # The result's exponent thresholds at both normalization outcomes.
    for ea in (1, 2, fmt.bias, fmt.exp_max_finite - 1, fmt.exp_max_finite):
        for eb in (1, fmt.bias, fmt.exp_max_finite):
            for fa, fb in ((0, 0), (0, fmt.frac_mask), (fmt.frac_mask, 0)):
                out.append((f"exp_{ea}_{eb}_{fa}_{fb}", normal(fmt, 0, ea, fa), normal(fmt, 0, eb, fb)))
    return out


def root_directed(fmt: ZkfFormat) -> list[tuple[str, int]]:
    """Radicands at the compare-only last digit, the guard, and a single tail bit at every position."""
    out: list[tuple[str, int]] = []
    for e in (fmt.bias, fmt.bias + 1):
        if e > fmt.exp_max_finite:
            continue
        for bit in range(fmt.wfrac):
            out.append((f"tail_bit_{e}_{bit}", normal(fmt, 0, e, 1 << bit)))
        # Squares of roots spread over [1, 2), and their neighbours.
        for q in range(1 << min(fmt.wfrac, 6)):
            r = (1 << fmt.wfrac) | (q << max(0, fmt.wfrac - 6))
            for delta in (-1, 0, 1):
                frac = ((r * r + delta * (1 << fmt.wfrac)) >> fmt.wfrac) - (1 << fmt.wfrac)
                if 0 <= frac <= fmt.frac_mask:
                    out.append((f"square_{e}_{q}_{delta}", normal(fmt, 0, e, frac)))
    return out


def division_cases(fmt: ZkfFormat, context) -> list[test_div.BinaryCase]:
    cases: list[test_div.BinaryCase] = []
    seen: set[tuple[int, int]] = set()
    stride = range(context.shard_index, 1 << (2 * fmt.wfull), context.shard_count)
    if context.kind == "exhaustive":
        for i in stride:
            test_div.add_unique(cases, seen, "exhaustive", fmt, i >> fmt.wfull, i & mask(fmt.wfull))
        return cases
    if context.kind == "significands":
        # Every significand pair; the exponents only offset the result's.
        n = 1 << fmt.wfrac
        for i in range(context.shard_index, n * n, context.shard_count):
            a, b = normal(fmt, 0, fmt.bias, i // n), normal(fmt, 0, fmt.bias, i % n)
            test_div.add_unique(cases, seen, "significands", fmt, a, b)
        return cases
    cases = test_div.cases_for(fmt, context.kind, context.seed, context.count)
    seen = {(c.a, c.b) for c in cases}
    for label, a, b in division_directed(fmt):
        test_div.add_unique(cases, seen, label, fmt, a, b)
    return cases


def root_cases(fmt: ZkfFormat, context) -> list[test_sqrt.UnaryCase]:
    cases: list[test_sqrt.UnaryCase] = []
    seen: set[int] = set()
    if context.kind == "exhaustive":
        for x in range(context.shard_index, 1 << fmt.wfull, context.shard_count):
            test_sqrt.add_unique(cases, seen, "exhaustive", fmt, x)
        return cases
    if context.kind == "significands":
        for e in (fmt.bias, fmt.bias + 1) if fmt.bias + 1 <= fmt.exp_max_finite else (fmt.bias,):  # both parities
            for f in range(context.shard_index, 1 << fmt.wfrac, context.shard_count):
                test_sqrt.add_unique(cases, seen, "significands", fmt, normal(fmt, 0, e, f))
        return cases
    cases = test_sqrt.cases_for(fmt, context.kind, context.seed, context.count)
    seen = {c.x for c in cases}
    for label, x in root_directed(fmt):
        test_sqrt.add_unique(cases, seen, label, fmt, x)
    return cases


def cases_for(fmt: ZkfFormat, context, rng: np.random.Generator) -> list[Case]:
    cases: list[Case] = []
    if context.mode != 1:
        cases += [Case(c.label, False, c.a, c.b, c.expected, c.div0) for c in division_cases(fmt, context)]
    if context.mode != 0:
        for c in root_cases(fmt, context):
            # b is noise, a zero exponent among it (a divisor zero must not raise the error).
            noise = random_bits(fmt.wfull, rng) if rng.integers(0, 3) else int(rng.integers(0, 1 << fmt.wfrac))
            cases.append(Case(c.label, True, c.x, noise, c.y, c.domain_error))
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

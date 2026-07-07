#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import normal
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class UnaryCase:
    label: str
    x: int
    y: int
    domain_error: int
    pole: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[UnaryCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    key = x & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    r = fmt.wrap(x).log2()
    cases.append(UnaryCase(label, x, r.value.bits, int(r.domain_error), int(r.pole)))


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = [
        ("raw_zero", 0),  # pole
        ("raw_one_frac", 1),
        ("raw_neg_zero", 1 << fmt.sign_shift),  # pole (canonicalized +0)
        ("raw_pos_inf", fmt.exp_inf << fmt.wfrac),  # +inf
        ("raw_neg_inf", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac)),  # domain error
        ("raw_all_ones", mask(fmt.wfull)),  # negative non-canonical -> domain error
    ]
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            out.append((f"num_{label}", value))
        # Exact powers of two: log2 is an exact integer -> exercises the frac == 0 / l == 0 paths.
        for k in (-2, -1, 0, 1, 2):
            exp = fmt.bias + k
            if 1 <= exp <= fmt.exp_max_finite:
                out.append((f"pow2_{k}", normal(fmt, 0, exp, 0)))
                out.append((f"neg_pow2_{k}", normal(fmt, 1, exp, 0)))  # negative -> domain error
        # Top finite exp/frac exercise the reduced WNORM upper range: the re-center branch may carry e to 2^(WEXP-1),
        # but log2(m') is then negative so the finite result stays just below that bound.
        for frac in sorted({0, 1, fmt.frac_mask >> 1, fmt.frac_mask - 1, fmt.frac_mask}):
            out.append((f"max_exp_frac_{frac}", normal(fmt, 0, fmt.exp_max_finite, frac)))
        # x=2^k yields exact integer y=k; for |k| a power of two, y's significand is a power of two, and the neighbors
        # exercise the rounder/normalizer around those output exponent boundaries.
        for p in range(fmt.wexp):
            k = 1 << p
            for delta in (-1, 0, 1):
                kp = k + delta
                if kp >= 0:
                    exp = fmt.bias + kp
                    if 1 <= exp <= fmt.exp_max_finite:
                        out.append((f"result_pos_pow2_{k}_{delta}", normal(fmt, 0, exp, 0)))
                kn = k + delta
                if kn >= 0:
                    exp = fmt.bias - kn
                    if 1 <= exp <= fmt.exp_max_finite:
                        out.append((f"result_neg_pow2_{k}_{delta}", normal(fmt, 0, exp, 0)))
        # Near 1.0 (small results, the log2 cancellation regime).
        out.append(("just_above_one", normal(fmt, 0, fmt.bias, 1)))
        out.append(("just_below_one", normal(fmt, 0, fmt.bias - 1, fmt.frac_mask)))
    return out


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[UnaryCase]:
    cases: list[UnaryCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for x in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, x)
        return cases

    for label, value in directed_values(fmt):
        add_unique(cases, seen, label, fmt, value)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        x = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        add_unique(cases, seen, "random", fmt, x)
    return cases


@cocotb.test()
async def log2_runtime_cases(dut) -> None:
    context = float_context("log2")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.x.value = 0

    register_stages = fmt.model_of("log2")(
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
        stage_product=context.stage_product,
        stage_product_final=context.stage_product_final,
        stage_normalize=context.stage_normalize,
        stage_normalize_output=context.stage_normalize_output,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"y": (dut.y, fmt.wfull), "domain_error": (dut.domain_error, 1), "pole": (dut.pole, 1)},
    )

    def drive_case(case: UnaryCase) -> dict[str, int]:
        drive_unsigned(dut.x, case.x)
        return {"y": case.y, "domain_error": case.domain_error, "pole": case.pole}

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

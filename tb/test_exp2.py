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
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


def add_unique(cases: list[UnaryCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    key = x & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    cases.append(UnaryCase(label, x, fmt.wrap(x).exp2().bits))


def directed_values(fmt: ZkfFormat) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = [
        ("raw_zero", 0),
        ("raw_one_frac", 1),
        ("raw_frac_mask", fmt.frac_mask),
        ("raw_neg_zero", 1 << fmt.sign_shift),
        ("raw_pos_inf", fmt.exp_inf << fmt.wfrac),
        ("raw_neg_inf", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac)),
        ("raw_all_ones", mask(fmt.wfull)),
    ]
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            out.append((f"num_{label}", value))
        # x = +-2**k: integers (an exact power of two, with the result assembly's overflow/underflow) and fractions.
        for k in (1, 2, 3, -1, -2, -3):
            exp = fmt.bias + k
            if 1 <= exp <= fmt.exp_max_finite:
                out.append((f"pow2_{k}", normal(fmt, 0, exp, 0)))
                out.append((f"pow2_neg_{k}", normal(fmt, 1, exp, 0)))
        # |x| = q / 2**shift around the bias: the ends of the result's range, where the zero, MIN_NORMAL and normal
        # classes meet below and the largest finite result sits above.
        shift = fmt.wfrac - (fmt.wexp - 2)
        if shift >= 1:
            ends = {
                "neg_zero_low": (1, ((fmt.bias + 1) << shift) - 1),
                "neg_zero_mid": (1, (fmt.bias << shift) + (1 << (shift - 1))),
                "neg_zero_high": (1, (fmt.bias << shift) + 1),
                "neg_boundary": (1, fmt.bias << shift),
                "neg_min_mid": (1, (fmt.bias << shift) - (1 << (shift - 1))),
                "neg_min_high": (1, ((fmt.bias - 1) << shift) + 1),
                "neg_normal": (1, (fmt.bias - 1) << shift),
                "pos_top": (0, ((fmt.bias + 1) << shift) - 1),
            }
            for label, (sign, q) in ends.items():
                out.append((f"end_{label}", normal(fmt, sign, fmt.bias + fmt.wexp - 2, q - (1 << fmt.wfrac))))
        out.append(("three_quarters", normal(fmt, 0, fmt.bias - 1, 1 << (fmt.wfrac - 1))))
        out.append(("tiny_pos", normal(fmt, 0, 1, 0)))
        out.append(("tiny_neg", normal(fmt, 1, 1, 0)))
    return out


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[UnaryCase]:
    cases: list[UnaryCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for x in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, x)
        return cases

    if kind == "strata":
        from tools.zkf_transcendental import directed_strata

        for label, values in directed_strata("exp2", fmt).items():
            for x in values:
                add_unique(cases, seen, label, fmt, x)
        return cases

    for label, value in directed_values(fmt):
        add_unique(cases, seen, label, fmt, value)

    if kind == "directed":
        return cases

    from zkf._reference import trans_spec

    rng = np.random.default_rng(seed)
    low = max(fmt.bias - trans_spec("exp2", fmt.wman)["cf"], 1)  # from here up the reduced argument is nonzero
    exps = (low, fmt.bias + fmt.wexp - 1)
    while len(cases) < count:
        mode = int(rng.integers(0, 4))
        if mode >= 2:  # most other draws saturate at a wide exponent
            x = fmt.pack(int(rng.integers(0, 2)), int(rng.integers(*exps)), random_bits(fmt.wfrac, rng)).bits
        else:
            x = random_operand(fmt, rng) if mode else random_bits(fmt.wfull, rng)
        add_unique(cases, seen, "random", fmt, x)
    return cases


@cocotb.test()
async def exp2_runtime_cases(dut) -> None:
    context = float_context("exp2")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.x.value = 0

    register_stages = fmt.model_of("exp2")(
        stage_input=context.stage_input,
        stage_reduce=context.stage_reduce,
        stage_product=context.stage_product,
        stage_product_square=context.stage_product_square,
        stage_product_odd=context.stage_product_odd,
        stage_product_chain=context.stage_product_chain,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).timing.latency
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: UnaryCase) -> dict[str, int]:
        drive_unsigned(dut.x, case.x)
        return {"y": case.expected}

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

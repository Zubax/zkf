#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask

# zkf_round round_mode port codes (zkf/rtl/zkf_round.v) -> the Zkf integral-rounding method modelling each.
_ROUND_BY_MODE = ("round", "floor", "ceil", "trunc")  # 0=RNTE, 1=floor, 2=ceil, 3=trunc
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
class RoundCase:
    label: str
    a: int
    mode: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return (
            f"{self.label} mode={self.mode} a={hex_bits(self.a, fmt.wfull)} "
            f"expected={hex_bits(self.expected, fmt.wfull)}"
        )


def add_unique(cases: list[RoundCase], seen: set[int], label: str, fmt: ZkfFormat, a: int) -> None:
    """Register one operand under every rounding mode (each (operand, mode) pair becomes a case)."""
    key = a & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    for mode, method in enumerate(_ROUND_BY_MODE):
        cases.append(RoundCase(label, key, mode, getattr(fmt.wrap(key), method)().bits))


def directed_case_inputs(fmt: ZkfFormat) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    # Named values (tie-to-even one_and_half, quarters, min/max_finite, +-inf, ...) need WEXP>=3.
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            cases.append((label, value))
    # Raw boundary patterns that exercise canonicalization regardless of WEXP (also reach the WEXP=2 corner).
    cases.extend(
        [
            ("raw_zero_clean", 0),
            ("raw_zero_neg_payload", 1 << fmt.sign_shift),
            ("raw_zero_payload", min(fmt.frac_mask, 1)),
            ("raw_inf_pos", fmt.exp_inf << fmt.wfrac),
            ("raw_inf_neg", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac)),
            ("raw_inf_noncanonical_pos", (fmt.exp_inf << fmt.wfrac) | min(fmt.frac_mask, 1)),
            ("raw_inf_noncanonical_neg", mask(fmt.wfull)),
        ]
    )
    return cases


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    mode = int(rng.integers(0, 8))
    if mode == 0:
        return random_zero(fmt, rng)
    if mode == 1:
        return random_inf(fmt, rng)
    if mode == 2:
        # Sub-one and the 1.x region: the |value| < 1 branch (0/+-1 results) and the smallest in-fraction rounds.
        return random_normal_near(fmt, rng, [fmt.bias - 2, fmt.bias - 1, fmt.bias, fmt.bias + 1], [0, 1, fmt.frac_mask])
    if mode == 3:
        # Half-way fractions to stress ties-to-even / floor / ceil sign behaviour around small integers.
        return random_normal_near(
            fmt,
            rng,
            [fmt.bias, fmt.bias + 1, fmt.bias + 2],
            [0, 1 << (fmt.wfrac - 1), (1 << (fmt.wfrac - 1)) | 1, fmt.frac_mask],
        )
    if mode == 4:
        # Near the top finite exponents: in tiny formats this is where a round-up overflows to inf.
        return random_normal_near(
            fmt,
            rng,
            [fmt.exp_max_finite - 2, fmt.exp_max_finite - 1, fmt.exp_max_finite],
            [0, 1 << (fmt.wfrac - 1), fmt.frac_mask],
        )
    if mode == 5:
        return random_normal(fmt, rng)
    return random_operand(fmt, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[RoundCase]:
    cases: list[RoundCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, a)
        return cases

    for label, a in directed_case_inputs(fmt):
        add_unique(cases, seen, label, fmt, a)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    target_operands = len(seen) + count  # count counts random operands; each yields one case per mode
    while len(seen) < target_operands:
        add_unique(cases, seen, "random", fmt, random_case(fmt, rng))
    return cases


@cocotb.test()
async def round_runtime_cases(dut) -> None:
    context = float_context("round")
    fmt = ZkfFormat(context.wexp, context.wman)

    check_width("a", dut.a, fmt.wfull, context)
    check_width("round_mode", dut.round_mode, 2, context)
    check_width("y", dut.y, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_unsigned(dut.a, 0)
    dut.round_mode.value = 0

    register_stages = fmt.model_of("round")(
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).latency
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {"y": (dut.y, fmt.wfull)},
        reset_passthrough=register_stages == 0,
    )

    def drive_case(case: RoundCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        dut.round_mode.value = case.mode
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        dut.round_mode.value = 0

    def describe(index: int, case: RoundCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> dict[str, int]:
        dut.in_valid.value = 1
        return drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_latency import mul_ilog2_const_latency
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


# Port-to-K map must mirror the K values instantiated in zkf_mul_ilog2_const_wrap.v.
def k_for_port(fmt: ZkfFormat) -> dict[str, int]:
    emax_minus_one = fmt.exp_max_finite - 1
    return {
        "y_k0": 0,
        "y_kp1": 1,
        "y_kn1": -1,
        "y_kp_mid": emax_minus_one // 2,
        "y_kn_mid": -(emax_minus_one // 2),
        "y_kp_max": emax_minus_one,
        "y_kn_max": -fmt.exp_max_finite,
    }


@dataclass(frozen=True)
class UnaryCase:
    label: str
    a: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)}"


def raw_directed_values(fmt: ZkfFormat) -> list[int]:
    return [
        0,
        1,
        fmt.frac_mask,
        1 << fmt.sign_shift,
        (1 << fmt.sign_shift) | min(fmt.frac_mask, 1),
        fmt.exp_inf << fmt.wfrac,
        (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac),
        mask(fmt.wfull),
    ]


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = [(f"raw_{i}", v) for i, v in enumerate(raw_directed_values(fmt))]
    if fmt.wexp >= 3:
        cases.extend(directed_numbers(fmt).items())
    return cases


def binary32_manual_cases() -> list[tuple[str, int]]:
    """
    Binary32 inputs covering each (input_class, sign, magnitude-edge). The wrap fans each across seven
    K instances, so every K-induced transition (identity, ±1, midrange, ±(EXP_MAX_FINITE-1)) reaches every
    input class too.
    """
    return [
        ("manual_zero", 0x00000000),
        ("manual_neg_zero", 0x80000000),
        ("manual_neg_zero_payload", 0x80000001),
        ("manual_zero_max_payload", 0x007FFFFF),
        ("manual_one", 0x3F800000),
        ("manual_neg_one", 0xBF800000),
        ("manual_two", 0x40000000),
        ("manual_half", 0x3F000000),
        ("manual_min_normal", 0x00800000),
        ("manual_neg_min_normal", 0x80800000),
        ("manual_one_and_half_min_normal", 0x00C00000),
        ("manual_neg_one_and_half_min_normal", 0x80C00000),
        ("manual_just_above_min_normal", 0x00800001),
        ("manual_max_finite", 0x7F7FFFFF),
        ("manual_neg_max_finite", 0xFF7FFFFF),
        ("manual_just_below_max_finite", 0x7F7FFFFE),
        ("manual_pos_inf", 0x7F800000),
        ("manual_neg_inf", 0xFF800000),
        ("manual_noncanonical_pos_inf", 0x7F800001),
        ("manual_noncanonical_neg_inf", 0xFFC00001),
        ("manual_inf_max_payload", 0x7FFFFFFF),
        ("manual_neg_inf_max_payload", 0xFFFFFFFF),
    ]


def add_unique(cases: list[UnaryCase], seen: set[int], label: str, fmt: ZkfFormat, a: int) -> None:
    key = a & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    cases.append(UnaryCase(label, a))


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[UnaryCase]:
    cases: list[UnaryCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, a)
        return cases

    for label, value in directed_case_operands(fmt):
        add_unique(cases, seen, label, fmt, value)

    if (fmt.wexp, fmt.wman) == (8, 24):
        for label, value in binary32_manual_cases():
            add_unique(cases, seen, label, fmt, value)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    # Cap at the input-universe size so the loop terminates when count exceeds the representable inputs.
    target = min(count, 1 << fmt.wfull)
    while len(cases) < target:
        a = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        add_unique(cases, seen, "random", fmt, a)
    return cases


@cocotb.test()
async def mul_ilog2_const_runtime_cases(dut) -> None:
    context = float_context("mul_ilog2_const")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    ports = k_for_port(fmt)
    for port in ports:
        check_width(port, getattr(dut, port), fmt.wfull, context)

    cases = cases_for(fmt, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0

    register_stages = mul_ilog2_const_latency(
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
    )
    outputs = {port: (getattr(dut, port), fmt.wfull) for port in ports}
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, outputs)

    def expected_for(a_bits: int) -> dict[str, int]:
        return {port: fmt.wrap(a_bits).mul_ilog2(k).bits for port, k in ports.items()}

    def drive_case(case: UnaryCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        return expected_for(case.a)

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)

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

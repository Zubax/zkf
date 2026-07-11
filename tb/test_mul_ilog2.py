#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, drive_signed, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class Case:
    label: str
    a: int
    k: int
    expected: int

    def describe(self, fmt: ZkfFormat, wk: int) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} k={self.k} ({hex_bits(self.k & mask(wk), wk)})"


def k_range(wk: int) -> tuple[int, int]:
    return -(1 << (wk - 1)), (1 << (wk - 1)) - 1


def directed_shifts(fmt: ZkfFormat, wk: int) -> list[int]:
    """Shifts crossing every input-class boundary, plus the saturating WK extremes."""
    kmin, kmax = k_range(wk)
    candidates = {
        0,
        1,
        -1,
        2,
        -2,
        fmt.bias,
        -fmt.bias,  # one -> {min-normal boundary, overflow-ish}
        -(fmt.bias - 1),
        -(fmt.bias + 1),  # one -> min-normal value / underflow-to-zero
        fmt.exp_max_finite,
        -fmt.exp_max_finite,
        -(1 << fmt.wexp) - 1,
        -(1 << fmt.wexp),
        (1 << fmt.wexp) - 1,
        1 << fmt.wexp,
        kmin,
        kmax,
        kmin + 1,
        kmax - 1,
    }
    return sorted(k for k in candidates if kmin <= k <= kmax)


def add_unique(
    cases: list[Case], seen: set[tuple[int, int]], label: str, fmt: ZkfFormat, a: int, k: int, wk: int
) -> None:
    key = (a & mask(fmt.wfull), k & mask(wk))
    if key in seen:
        return
    seen.add(key)
    cases.append(Case(label, a, k, fmt.wrap(a).mul_ilog2(k).bits))


def directed_operands(fmt: ZkfFormat) -> list[tuple[str, int]]:
    raw = [
        ("raw_zero", 0),
        ("raw_one_ulp", 1),
        ("raw_zero_max_payload", fmt.frac_mask),
        ("raw_neg_zero", 1 << fmt.sign_shift),
        ("raw_inf", fmt.exp_inf << fmt.wfrac),
        ("raw_neg_inf_payload", (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac) | min(fmt.frac_mask, 1)),
        ("raw_all_ones", mask(fmt.wfull)),
    ]
    if fmt.wexp >= 3:
        raw += list(directed_numbers(fmt).items())
    return raw


def cases_for(fmt: ZkfFormat, wk: int, kind: str, seed: int, count: int) -> list[Case]:
    cases: list[Case] = []
    seen: set[tuple[int, int]] = set()
    kmin, kmax = k_range(wk)

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            for k in range(kmin, kmax + 1):
                add_unique(cases, seen, "exhaustive", fmt, a, k, wk)
        return cases

    for label, a in directed_operands(fmt):
        for k in directed_shifts(fmt, wk):
            add_unique(cases, seen, label, fmt, a, k, wk)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    target = min(count, (1 << fmt.wfull) * (1 << wk))
    while len(cases) < target:
        a = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        k = int(rng.integers(kmin, kmax + 1))
        add_unique(cases, seen, "random", fmt, a, k, wk)
    return cases


@cocotb.test()
async def mul_ilog2_runtime_cases(dut) -> None:
    context = float_context("mul_ilog2")
    fmt = ZkfFormat(context.wexp, context.wman)
    wk = context.wk if context.wk is not None else fmt.wexp + 1
    check_width("a", dut.a, fmt.wfull, context)
    check_width("k", dut.k, wk, context)
    check_width("y", dut.y, fmt.wfull, context)

    cases = cases_for(fmt, wk, context.kind, context.seed, context.count)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.k.value = 0

    register_stages = fmt.model_of("mul_ilog2")(
        wk=wk,
        stage_input=context.stage_input,
        stage_decode=context.stage_decode,
    ).latency
    scoreboard = RegisterStageScoreboard(dut, register_stages, context, {"y": (dut.y, fmt.wfull)})

    def drive_case(case: Case) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_signed(dut.k, case.k)
        return {"y": case.expected}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_signed(dut.k, -1)

    def describe(index: int, case: Case) -> str:
        return f"case={index} {case.describe(fmt, wk)}"

    def drive_reset_sample() -> None:
        dut.in_valid.value = 1
        drive_case(cases[0])

    await scoreboard.reset(register_stages + 1, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

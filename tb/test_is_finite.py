#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np
from cocotb.triggers import Timer

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand
from zkf_params import check_width, float_context
from zkf_stream import drive_unsigned, is_resolvable


@dataclass(frozen=True)
class UnaryCase:
    label: str
    x: int
    expected: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} x={hex_bits(self.x, fmt.wfull)}"


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


def add_unique(cases: list[UnaryCase], seen: set[int], label: str, fmt: ZkfFormat, x: int) -> None:
    key = x & mask(fmt.wfull)
    if key in seen:
        return
    seen.add(key)
    cases.append(UnaryCase(label, x, int(fmt.wrap(x).is_finite)))


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[UnaryCase]:
    cases: list[UnaryCase] = []
    seen: set[int] = set()

    if kind == "exhaustive":
        for x in range(1 << fmt.wfull):
            add_unique(cases, seen, "exhaustive", fmt, x)
        return cases

    for index, value in enumerate(raw_directed_values(fmt)):
        add_unique(cases, seen, f"raw_{index}", fmt, value)
    if fmt.wexp >= 3:
        for label, value in directed_numbers(fmt).items():
            add_unique(cases, seen, label, fmt, value)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        x = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        add_unique(cases, seen, "random", fmt, x)
    return cases


@cocotb.test()
async def is_finite_runtime_cases(dut) -> None:
    context = float_context("is_finite")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("x", dut.x, fmt.wfull, context)
    check_width("y", dut.y, 1, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    for index, case in enumerate(cases):
        drive_unsigned(dut.x, case.x)
        await Timer(1, unit="ns")
        assert is_resolvable(dut.y), f"{context.prefix()} y unresolved case={index} {case.describe(fmt)}"
        observed = int(dut.y.value)
        assert observed == case.expected, (
            f"{context.prefix()} y mismatch case={index} {case.describe(fmt)} "
            f"expected={case.expected} observed={observed}"
        )

#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np

from zkf import ZkfFormat
from zkf_bits import hex_bits, mask
from zkf_operands import directed_numbers, random_bits, random_operand, raw_directed_values
from zkf_params import check_width, float_context
from zkf_stream import RegisterStageScoreboard, check_combinational, drive_unsigned, run_stream_cases, start_clock


@dataclass(frozen=True)
class CompareCase:
    label: str
    a: int
    b: int

    def describe(self, fmt: ZkfFormat) -> str:
        return f"{self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


def special_class_representatives(fmt: ZkfFormat) -> list[tuple[str, int]]:
    """
    Zero and infinity bit patterns at representative fractions, targeting the class overrides of zkf_cmp:
    every zero pattern compares equal to every other, same-sign infinities compare equal regardless of fraction, and
    crossing a class boundary produces strict ordering.
    """
    frac_bits = [0, 1, fmt.frac_mask >> 1, fmt.frac_mask] if fmt.wfrac >= 2 else [0, fmt.frac_mask]
    frac_bits = sorted({f & fmt.frac_mask for f in frac_bits})
    cases: list[tuple[str, int]] = []
    for sign in (0, 1):
        for frac in frac_bits:
            cases.append((f"zero_s{sign}_f{frac:x}", (sign << fmt.sign_shift) | frac))
            inf_bits = (sign << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac) | frac
            cases.append((f"inf_s{sign}_f{frac:x}", inf_bits))
    return cases


def corner_pairs(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    """Hand-picked pairings that exercise every transition in the comparator's decision tree."""
    pairs: list[tuple[str, int, int]] = []
    specials = special_class_representatives(fmt)
    # Cross-product of all zero/infinity representations (equal zeros, equal same-sign infinities, ordered
    # different-sign infinities).
    for left_label, left in specials:
        for right_label, right in specials:
            pairs.append((f"{left_label}_vs_{right_label}", left, right))
    # Reflexivity probes outside the special classes: covered by exhaustive at small widths, listed explicitly so they
    # also fire at the larger random configurations.
    if fmt.wexp >= 3:
        named = directed_numbers(fmt)
        for label, value in named.items():
            pairs.append((f"self_{label}", value, value))
            # Adjacent-value probe: smallest-LSB perturbation should always change the ordering.
            neighbour = value ^ 0x1
            pairs.append((f"{label}_vs_lsbflip", value, neighbour & mask(fmt.wfull)))
    return pairs


def add_unique(
    cases: list[CompareCase],
    seen: set[tuple[int, int]],
    label: str,
    fmt: ZkfFormat,
    a: int,
    b: int,
) -> None:
    key = (a & mask(fmt.wfull), b & mask(fmt.wfull))
    if key in seen:
        return
    seen.add(key)
    cases.append(CompareCase(label, a, b))


def directed_case_operands(fmt: ZkfFormat) -> list[tuple[str, int, int]]:
    values = [(f"raw_{index}", value) for index, value in enumerate(raw_directed_values(fmt))]
    if fmt.wexp >= 3:
        values.extend(directed_numbers(fmt).items())

    cases = []
    for left_label, a in values:
        for right_label, b in values:
            cases.append((f"{left_label}_vs_{right_label}", a, b))
    return cases


def random_case(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(0, 8))
    if mode <= 4:
        return random_operand(fmt, rng), random_operand(fmt, rng)
    if mode == 5:
        return random_bits(fmt.wfull, rng), random_operand(fmt, rng)
    if mode == 6:
        return random_operand(fmt, rng), random_bits(fmt.wfull, rng)
    return random_bits(fmt.wfull, rng), random_bits(fmt.wfull, rng)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[CompareCase]:
    cases: list[CompareCase] = []
    seen: set[tuple[int, int]] = set()

    if kind == "exhaustive":
        for a in range(1 << fmt.wfull):
            for b in range(1 << fmt.wfull):
                add_unique(cases, seen, "exhaustive", fmt, a, b)
        return cases

    for label, a, b in directed_case_operands(fmt):
        add_unique(cases, seen, label, fmt, a, b)

    for label, a, b in corner_pairs(fmt):
        add_unique(cases, seen, label, fmt, a, b)

    if kind == "directed":
        return cases

    rng = np.random.default_rng(seed)
    while len(cases) < count:
        a, b = random_case(fmt, rng)
        add_unique(cases, seen, "random", fmt, a, b)
    return cases


@cocotb.test()
async def cmp_runtime_cases(dut) -> None:
    context = float_context("cmp")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("a_gt_b", dut.a_gt_b, 1, context)
    check_width("a_eq_b", dut.a_eq_b, 1, context)
    check_width("a_lt_b", dut.a_lt_b, 1, context)
    check_width("min", dut.min, fmt.wfull, context)
    check_width("max", dut.max, fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)

    register_stages = fmt.model_of("cmp")(
        stage_input=context.stage_input, stage_output=context.stage_output
    ).timing.latency

    def drive_case(case: CompareCase) -> dict[str, int]:
        drive_unsigned(dut.a, case.a)
        drive_unsigned(dut.b, case.b)
        c = fmt.wrap(case.a).cmp(fmt.wrap(case.b))
        return {"a_gt_b": int(c.gt), "a_eq_b": int(c.eq), "a_lt_b": int(c.lt), "min": c.min.bits, "max": c.max.bits}

    outputs = {"a_gt_b": dut.a_gt_b, "a_eq_b": dut.a_eq_b, "a_lt_b": dut.a_lt_b, "min": dut.min, "max": dut.max}
    if register_stages == 0:
        await check_combinational(dut, context.prefix(), outputs, cases[:256], drive_case)
    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    scoreboard = RegisterStageScoreboard(
        dut,
        register_stages,
        context,
        {
            "a_gt_b": (dut.a_gt_b, 1),
            "a_eq_b": (dut.a_eq_b, 1),
            "a_lt_b": (dut.a_lt_b, 1),
            "min": (dut.min, fmt.wfull),
            "max": (dut.max, fmt.wfull),
        },
        reset_passthrough=register_stages == 0,
    )

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)
        drive_unsigned(dut.b, 0)

    def describe(index: int, case: CompareCase) -> str:
        return f"case={index} {case.describe(fmt)}"

    def drive_reset_sample() -> dict[str, int]:
        dut.in_valid.value = 1
        return drive_case(cases[0])

    await scoreboard.reset(register_stages + 2, drive_during_reset=drive_reset_sample)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, describe)
    assert scoreboard.checked == len(
        cases
    ), f"{context.prefix()} checked {scoreboard.checked} outputs, expected {len(cases)}"

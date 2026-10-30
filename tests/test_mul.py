import re
from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import (
    SEED,
    directed_numbers,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf import Mul, Zkf, ZkfFormat
from zkf._pmul import counts

# Operands and the product, worked by hand at 8/24.
BINARY32 = [
    (0x00000000, 0x3F800000, 0x00000000),
    (0x805A5A5A, 0x7FFFFFFF, 0x00000000),
    (0xBF800000, 0x007FFFFF, 0x00000000),
    (0x3F800000, 0x3F800000, 0x3F800000),
    (0xBF800000, 0x3F800000, 0xBF800000),
    (0xBF800000, 0xBF800000, 0x3F800000),
    (0x3FC00000, 0x40000000, 0x40400000),
    (0x3FA00000, 0x3FC00000, 0x3FF00000),
    (0x3FC00000, 0x3FC00000, 0x40100000),
    (0x7F800000, 0x3F800000, 0x7F800000),
    (0x7FFFFFFF, 0x3F800000, 0x7F800000),
    (0xFFABCDEF, 0x3F800000, 0xFF800000),
    (0x3F800000, 0x7F800000, 0x7F800000),
    (0xBF800000, 0x7F800001, 0xFF800000),
    (0x40000000, 0xFF800001, 0xFF800000),
    (0x00000000, 0x7F800000, 0x00000000),
    (0x805A5A5A, 0xFFABCDEF, 0x00000000),
    (0x7F800000, 0x7FFFFFFF, 0x7F800000),
    (0xFF800000, 0x7F800001, 0xFF800000),
    (0xFFFFFFFF, 0xFFABCDEF, 0x7F800000),
    (0x00800000, 0x3EFFFFFD, 0x00000000),
    (0x00800000, 0x3F000000, 0x00800000),
    (0x00800000, 0x3F400000, 0x00800000),
    (0x80800000, 0x3F400000, 0x80800000),
    (0x00800000, 0x3F800000, 0x00800000),
    (0x80800000, 0x3F800000, 0x80800000),
    (0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF),
    (0xFF7FFFFF, 0x3F800000, 0xFF7FFFFF),
    (0x7F7FFFFF, 0x40000000, 0x7F800000),
    (0xFF7FFFFF, 0x40000000, 0xFF800000),
    (0x3F800002, 0x3FA00000, 0x3FA00002),
    (0x3F800001, 0x3FC00000, 0x3FC00002),
    (0x3F800001, 0x3FA00000, 0x3FA00001),
    (0x3F800001, 0x3FE00000, 0x3FE00002),
]


def test_reference_by_hand() -> None:
    fmt = ZkfFormat(8, 24)
    model = Mul(fmt, stage_core_mul=2, wmultiplier=18)
    for a, b, want in BINARY32:
        assert model.evaluate(Zkf(fmt, a), Zkf(fmt, b)).bits == want


def directed(fmt: ZkfFormat) -> list[tuple[int, int]]:
    v = directed_numbers(fmt)
    names = [
        ("zero", "zero"),
        ("zero", "neg_zero"),
        ("zero", "one"),
        ("neg_zero", "noncanonical_pos_inf"),
        ("minus_one", "neg_zero"),
        ("one", "one"),
        ("minus_one", "one"),
        ("minus_one", "minus_one"),
        ("one_and_half", "two"),
        ("one_and_quarter", "one_and_half"),
        ("one_and_half", "one_and_half"),
        ("pos_inf", "one"),
        ("noncanonical_pos_inf", "one"),
        ("noncanonical_neg_inf", "one"),
        ("one", "pos_inf"),
        ("minus_one", "noncanonical_pos_inf"),
        ("two", "noncanonical_neg_inf"),
        ("zero", "pos_inf"),
        ("neg_zero", "noncanonical_neg_inf"),
        ("pos_inf", "noncanonical_pos_inf"),
        ("neg_inf", "noncanonical_pos_inf"),
        ("noncanonical_neg_inf", "noncanonical_neg_inf"),
        ("min_normal", "half"),
        ("min_normal", "one"),
        ("neg_min_normal", "one"),
        ("max_finite", "one"),
        ("neg_max_finite", "one"),
        ("max_finite", "two"),
        ("neg_max_finite", "two"),
    ]
    return [(v[a], v[b]) for a, b in names]


def drawn(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 2:
        return random_zero(fmt, rng), random_inf(fmt, rng)
    if mode == 3:
        return random_inf(fmt, rng), random_zero(fmt, rng)
    if mode == 4:
        return random_normal(fmt, rng), random_inf(fmt, rng)
    if mode == 5:
        return random_inf(fmt, rng), random_normal(fmt, rng)
    if mode == 6:
        return random_inf(fmt, rng), random_inf(fmt, rng)
    if mode == 7:
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [fmt.bias], [0])
    if mode == 8:
        return random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]), directed_numbers(fmt)["two"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def pairs(fmt: ZkfFormat, count: int) -> list[tuple[int, int]]:
    """`count` 0 is every pair; otherwise the directed ones made up to `count` with random ones."""
    if not count:
        return [(a, b) for a in range(1 << fmt.wfull) for b in range(1 << fmt.wfull)]
    out = directed(fmt) if fmt.wexp >= 3 else []
    if (fmt.wexp, fmt.wman) == (8, 24):
        out += [(a, b) for a, b, _ in BINARY32]
    rng = np.random.default_rng(SEED)
    seen = set(out)
    while len(out) < count:
        pair = drawn(fmt, rng)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def run(wexp: int, wman: int, count: int, wmultiplier: int = 18, **knobs: Any) -> None:
    simulate(Mul(ZkfFormat(wexp, wman), wmultiplier=wmultiplier, **knobs), count)


def simulate(model: Mul, count: int) -> None:
    fmt = model.fmt
    cases = [Case({"a": a, "b": b}, {"y": model.evaluate(Zkf(fmt, a), Zkf(fmt, b))}) for a, b in pairs(fmt, count)]
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp, wman, count
FORMATS = [
    (2, 4, 0),
    (3, 4, 0),
    (3, 5, 512),
    (4, 6, 512),
    (5, 11, 768),
    (6, 18, 768),
    (7, 17, 768),
    (8, 24, 1024),
    (11, 53, 384),
]


def taken(wman: int, wmultiplier: int) -> range:
    """The counts that the product takes."""
    return counts(wman, wman, signed=False, wmultiplier=wmultiplier)


@pytest.mark.parametrize("stage_input", [0, 1])
@pytest.mark.parametrize(
    "row, stage_core_mul", [(row, core) for row in FORMATS for core in range(4) if core in taken(row[1], 18)], ids=str
)
def test_formats(row: tuple[int, int, int], stage_core_mul: int, stage_input: int) -> None:
    run(*row, stage_core_mul=stage_core_mul, stage_input=stage_input)


@pytest.mark.parametrize(
    "knobs",
    [
        {"stage_core_mul": 3, "stage_input": 1, "stage_output": 2},
        {"stage_input": 2},
        {"stage_input": 3},
        {"stage_output": 3},
    ],
    ids=str,
)
def test_stages(knobs: dict[str, int]) -> None:
    run(3, 4, 0, **knobs)


# The narrowest significands that 8-bit multipliers split in two and in three, at every count that they take.
@pytest.mark.parametrize("row", [(12, 2), (12, 3), (17, 3), (17, 4)], ids=str)
def test_split(row: tuple[int, int]) -> None:
    run(3, row[0], 512, 8, stage_core_mul=row[1])


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Mul)])
def test_catalog(model: Mul) -> None:
    simulate(model, 512)


def test_refusals() -> None:
    with pytest.raises(ValueError, match=re.escape("stage_core_mul=1: set 2..3 on 18-bit multipliers")):
        Mul(ZkfFormat(8, 36), wmultiplier=18, stage_core_mul=1)


def test_every_stage_is_a_cycle() -> None:
    """The input stages are a cycle each ahead of the product's capture."""

    def latency(fmt: ZkfFormat, **knobs: Any) -> int:
        return Mul(fmt, wmultiplier=18, **knobs).timing.latency

    f36 = ZkfFormat(8, 36)
    assert [latency(f36, stage_input=1, stage_core_mul=k) for k in (2, 3)] == [3, 4]
    assert latency(f36, stage_input=1, stage_core_mul=3, stage_output=1) == 5


# The split product at 8/25 with its output registered.
def test_split_registered() -> None:
    run(8, 25, 512, stage_core_mul=3, stage_output=1)


DEEP = [
    (2, 5, 0),
    (4, 5, 0),
    (2, 7, 0),
    (3, 6, 0),
    (5, 4, 0),
    (4, 6, 0),
    (3, 7, 512),
    (6, 17, 768),
    (8, 23, 768),
    (7, 12, 512),
    (9, 24, 512),
    (6, 19, 512),
]


# No split at a width that has none, and one run on every pair once.
@pytest.mark.deep
@pytest.mark.parametrize(
    "row, stage_core_mul, stage_input, stage_output",
    [
        (row, core, si, so)
        for row in DEEP
        for core in taken(row[1], 8)
        for si in (0, 1)
        for so in (0, 1)
        if core < 3 or row[1] > 8 and (row[2] or not si + so)
    ],
    ids=str,
)
def test_deep(row: tuple[int, int, int], stage_core_mul: int, stage_input: int, stage_output: int) -> None:
    run(*row, 8, stage_core_mul=stage_core_mul, stage_input=stage_input, stage_output=stage_output)


@pytest.mark.deep
@pytest.mark.parametrize("stage_core_mul", range(4))
@pytest.mark.parametrize("row", [(3, 5), (2, 6)], ids=str)
def test_deep_every_pair(row: tuple[int, int], stage_core_mul: int) -> None:
    run(*row, 0, stage_core_mul=stage_core_mul)


# The count of a 2x2 and of a 3x3 that the default tier does not run: the grid unregistered, and every level summed.
@pytest.mark.deep
@pytest.mark.parametrize("row", [(8, 36, 512, 2), (11, 53, 384, 4)], ids=str)
def test_deep_wide_split(row: tuple[int, int, int, int]) -> None:
    run(*row[:3], stage_core_mul=row[3])

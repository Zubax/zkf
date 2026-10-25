from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import (
    SEED,
    directed_numbers,
    random_inf,
    random_normal_near,
    random_operand,
    random_zero,
    raw_corners,
)
from zkf import Add, Zkf, ZkfFormat

# Operands and the sum, worked by hand at 8/24.
BINARY32 = [
    (0x00000000, 0x00000000, 0x00000000),
    (0x80000000, 0x00000000, 0x00000000),
    (0x80123456, 0x007FEDCB, 0x00000000),
    (0x3F800000, 0x3F800000, 0x40000000),
    (0x3F800000, 0xBF800000, 0x00000000),
    (0xBF800000, 0xBF800000, 0xC0000000),
    (0x3F800000, 0x3F000000, 0x3FC00000),
    (0x3F000000, 0xBF800000, 0xBF000000),
    (0x3F800000, 0xBF800001, 0xB4000000),
    (0xBF800000, 0x3F800001, 0x34000000),
    (0x00800000, 0x80800000, 0x00000000),
    (0x00800000, 0x00800000, 0x01000000),
    (0x01000000, 0x80C00001, 0x00000000),
    (0x01000000, 0x80C00000, 0x00800000),
    (0x01000000, 0x80A00000, 0x00800000),
    (0x81000000, 0x00C00000, 0x80800000),
    (0x81000000, 0x00A00000, 0x80800000),
    (0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF),
    (0x7F7FFFFF, 0xFF7FFFFF, 0x00000000),
    (0x7F7FFFFF, 0x7F7FFFFF, 0x7F800000),
    (0xFF7FFFFF, 0xFF7FFFFF, 0xFF800000),
    (0x7F7FFFFF, 0x73000000, 0x7F800000),
    (0x7F800000, 0x3F800000, 0x7F800000),
    (0x3F800000, 0x7F800000, 0x7F800000),
    (0xFF800000, 0x3F800000, 0xFF800000),
    (0x3F800000, 0xFFABCDEF, 0xFF800000),
    (0x7F800000, 0x7FABCDEF, 0x7F800000),
    (0xFF800000, 0xFFFFFFFF, 0xFF800000),
    (0x7F800000, 0xFF800000, 0x00000000),
    (0x7FABCDEF, 0xFFABCDEF, 0x00000000),
    (0x3F800000, 0x33800000, 0x3F800000),
    (0x3F800001, 0x33800000, 0x3F800002),
    (0x3F800000, 0x33C00000, 0x3F800001),
    (0x3F800001, 0x33000000, 0x3F800001),
    (0x3F800000, 0x20000000, 0x3F800000),
    (0x7F7FFFFF, 0xFF7FFFFE, 0x73800000),
]


def test_reference_by_hand() -> None:
    fmt = ZkfFormat(8, 24)
    model = Add(fmt)
    for a, b, y in BINARY32:
        assert model.evaluate(Zkf(fmt, a), Zkf(fmt, b)).bits == y, (hex(a), hex(b))


def focused(fmt: ZkfFormat) -> list[tuple[int, int]]:
    """Each structure of the adder: the swap, full cancellation, every shift of the aligner, the rounding carry."""
    wman, bias, top, ones = fmt.wman, fmt.bias, fmt.exp_max_finite, fmt.frac_mask
    half = 1 << (fmt.wfrac - 1)
    out = [
        (fmt.pack(0, bias, 0).bits, fmt.pack(1, bias, 1).bits),
        (fmt.pack(1, bias, 0).bits, fmt.pack(0, bias, 1).bits),
        (0, 0),
        (fmt.pack(0, bias, half - 1).bits, fmt.pack(1, bias - 1, 0).bits),
        (fmt.pack(0, bias, 1).bits, fmt.pack(1, bias, 0).bits),
    ]
    out += [(fmt.pack(0, e, f).bits, fmt.pack(1, e, f).bits) for e in (1, bias, top) for f in (0, ones)]
    apart = [1 << k for k in range((wman + 4).bit_length()) if 1 << k <= min(wman + 4, top - 1)]
    if top > 2:
        apart.append(max(wman + 4, top - 1))
    for difference in apart:
        small = fmt.pack(0, max(1, top - difference), ones).bits
        out += [(fmt.pack(0, top, 0).bits, small ^ flip) for flip in (0, 1 << fmt.sign_shift)]
    if top > wman:
        out.append((fmt.pack(0, top, ones).bits, fmt.pack(0, top - wman, 0).bits))
    if top >= wman + 2:
        out.append((fmt.pack(0, 2, 0).bits, fmt.pack(1, 1, ones).bits))
    return out


def directed(fmt: ZkfFormat) -> list[tuple[int, int]]:
    v = directed_numbers(fmt)
    bias, top, ones = fmt.bias, fmt.exp_max_finite, fmt.frac_mask
    out = [
        (v["zero"], v["one"]),
        (v["neg_zero"], v["one"]),
        (v["one"], v["neg_zero"]),
        (v["one"], v["one"]),
        (v["one"], v["minus_one"]),
        (v["minus_one"], v["one"]),
        (v["minus_one"], v["minus_one"]),
        (v["one"], v["half"]),
        (v["half"], v["minus_one"]),
        (v["one_and_half"], v["one_and_quarter"]),
        (v["one_and_three_quarters"], v["one_and_three_quarters"]),
        (v["min_normal"], v["min_normal"]),
        (v["min_normal"], v["neg_min_normal"]),
        (v["max_finite"], v["one"]),
        (v["max_finite"], v["max_finite"]),
        (v["neg_max_finite"], v["neg_max_finite"]),
        (v["max_finite"], v["neg_max_finite"]),
        (v["pos_inf"], v["one"]),
        (v["one"], v["pos_inf"]),
        (v["one"], v["noncanonical_neg_inf"]),
        (v["pos_inf"], v["noncanonical_pos_inf"]),
        (v["neg_inf"], v["noncanonical_neg_inf"]),
        (v["pos_inf"], v["neg_inf"]),
        (v["noncanonical_neg_inf"], v["pos_inf"]),
        (fmt.pack(0, bias, 2).bits, fmt.pack(1, bias, 1).bits),
        (fmt.pack(1, bias, 2).bits, fmt.pack(0, bias, 1).bits),
        (fmt.pack(0, 1, 1).bits, fmt.pack(1, 1, 0).bits),
        (v["neg_zero"], v["noncanonical_pos_inf"]),
        (v["noncanonical_pos_inf"], v["noncanonical_neg_inf"]),
        (v["noncanonical_neg_inf"], v["noncanonical_pos_inf"]),
    ]
    high = min(top, bias + 4)
    for apart in range(5):
        out += [(fmt.pack(0, high, 0).bits, fmt.pack(sign, high - apart, ones).bits) for sign in (0, 1)]
    # The rounding of a sum: a tie that stays even, one that goes up to even, above and below a half, the carry out.
    if top - fmt.wman >= 1:
        half_ulp = fmt.pack(0, top - fmt.wman, 0).bits
        out += [
            (fmt.pack(0, top, 0).bits, half_ulp),
            (fmt.pack(0, top, 1).bits, half_ulp),
            (fmt.pack(0, top, 0).bits, fmt.pack(0, top - fmt.wman, 1).bits),
            (v["max_finite"], half_ulp),
        ]
    if top - fmt.wman - 1 >= 1:
        out.append((fmt.pack(0, top, 0).bits, fmt.pack(0, top - fmt.wman - 1, ones).bits))
    if top - fmt.wman - 4 >= 1:
        out.append((fmt.pack(0, top, 0).bits, fmt.pack(0, top - fmt.wman - 4, ones).bits))
    return out + focused(fmt)


def crossed(fmt: ZkfFormat) -> list[tuple[int, int]]:
    """Every pair of the raw corners and the directed numbers, with `b` in both signs."""
    values = raw_corners(fmt) + (list(directed_numbers(fmt).values()) if fmt.wexp >= 3 else [])
    return [(a, b ^ flip) for a in values for b in values for flip in (0, 1 << fmt.sign_shift)]


def drawn(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    v = directed_numbers(fmt)
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    mode = int(rng.integers(0, 14))
    if mode == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if mode == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if mode == 2:
        return random_inf(fmt, rng), random_operand(fmt, rng)
    if mode == 3:
        return random_operand(fmt, rng), random_inf(fmt, rng)
    if mode == 4:
        return random_inf(fmt, rng), random_inf(fmt, rng)
    if mode == 5:  # one unit apart at one exponent, of opposite signs
        exp, frac, sign = int(rng.integers(1, top + 1)), int(rng.integers(0, ones)), int(rng.integers(0, 2))
        return fmt.pack(sign, exp, frac + 1).bits, fmt.pack(sign ^ 1, exp, frac).bits
    if mode == 6:  # far apart, of opposite signs
        sign = int(rng.integers(0, 2))
        large = int(rng.integers(max(1, top - 2), top + 1))
        small = int(rng.integers(1, max(2, min(top, fmt.wman + 3))))
        fractions = [int(rng.integers(0, ones + 1)) for _ in range(2)]
        return fmt.pack(sign, large, fractions[0]).bits, fmt.pack(sign ^ 1, small, fractions[1]).bits
    if mode == 7:
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [1, 2], [0, 1])
    if mode == 8:
        return random_normal_near(fmt, rng, [top], [ones]), random_normal_near(fmt, rng, [top], [ones])
    if mode == 9:
        return v["max_finite"], v["max_finite"]
    if mode == 10:
        return v["neg_max_finite"], v["neg_max_finite"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def pairs(fmt: ZkfFormat, count: int | None) -> list[tuple[int, int]]:
    """`count` 0 is every pair; otherwise the directed ones and `count` random ones, or none for None."""
    if count == 0:
        return [(a, b) for a in range(1 << fmt.wfull) for b in range(1 << fmt.wfull)]
    out = (directed(fmt) if fmt.wexp >= 3 else []) + crossed(fmt)
    if (fmt.wexp, fmt.wman) == (8, 24):
        out += [(a, b) for a, b, _ in BINARY32]
    out = list(dict.fromkeys(out))
    rng = np.random.default_rng(SEED)
    seen = set(out)
    target = len(out) + (count or 0)
    while len(out) < target:
        pair = drawn(fmt, rng)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def run(wexp: int, wman: int, count: int | None, **knobs: Any) -> None:
    simulate(Add(ZkfFormat(wexp, wman), **knobs), count)


def simulate(model: Add, count: int | None) -> None:
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


@pytest.mark.parametrize("every_register", [False, True])
@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int], every_register: bool) -> None:
    run(*row, stage_core=~Add.CoreStages(0) if every_register else Add.CoreStages(0))


@pytest.mark.parametrize(
    "knobs",
    [
        {"stage_input": 1},
        {"stage_input": 2, "stage_core": ~Add.CoreStages(0), "stage_output": 1},
        {"stage_input": 3},
        {"stage_output": 2},
    ],
    ids=str,
)
def test_stages(knobs: dict[str, Any]) -> None:
    run(3, 4, 0, **knobs)


# Each register alone and all of them, at a normalizer of two levels and of three.
@pytest.mark.parametrize(
    "wman, stage_core", [(wman, core) for wman in (6, 18) for core in (*Add.CoreStages, ~Add.CoreStages(0))], ids=str
)
def test_registers(wman: int, stage_core: Add.CoreStages) -> None:
    run(4, wman, 256, stage_core=stage_core)


# Every set of registers, at the narrowest width of three normalizer levels.
@pytest.mark.deep
@pytest.mark.parametrize("shard", range(16))
def test_every_set(shard: int) -> None:
    for bits in range(shard, 1 << len(Add.CoreStages), 16):
        run(4, 14, 48, stage_core=Add.CoreStages(bits))


# Wider, with every register; a significand of 100 bits.
@pytest.mark.parametrize(
    "row",
    [
        (8, 18, 256, {"stage_input": 2}),
        (8, 36, 768, {"stage_core": ~Add.CoreStages(0)}),
        (6, 100, None, {}),
    ],
    ids=str,
)
def test_rows(row: tuple[int, int, int | None, dict[str, Any]]) -> None:
    run(row[0], row[1], row[2], **row[3])


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Add)])
def test_catalog(model: Add) -> None:
    simulate(model, 768)


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


@pytest.mark.deep
@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize(
    "stage_core",
    [
        Add.CoreStages(0),
        Add.CoreStages.SUM | Add.CoreStages.NORMALIZED,
        Add.CoreStages.DECODE | Add.CoreStages.ALIGN | Add.CoreStages.NORMALIZE | Add.CoreStages.PACK,
        ~Add.CoreStages(0),
    ],
    ids=str,
)
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int], stage_core: Add.CoreStages, stage_output: int) -> None:
    run(*row, stage_core=stage_core, stage_output=stage_output)

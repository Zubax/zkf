import re
from fractions import Fraction
from itertools import product
from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import (
    SEED,
    directed_numbers,
    random_bits,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf import Zkf, ZkfFormat
from zkf._divsqrt import Divsqrt

# Operands, the result and the error, worked by hand at 8/24: IEEE's, and ZKF's own for its zeros and infinities.
DIVISIONS = [
    (0x805A5A5A, 0x7F800000, 0x00000000, 0),
    (0x00000000, 0x00000000, 0x00000000, 1),
    (0x3F800000, 0x805A5A5A, 0x7F800000, 1),
    (0x3FA00000, 0x3FC00000, 0x3F555555, 0),
    (0x7F812345, 0xBF800000, 0xFF800000, 0),
    (0x7F800000, 0x7F800000, 0x00000000, 0),
    (0xFFFFFFFF, 0x7F812345, 0x00000000, 0),
    (0x00800000, 0x40000000, 0x00800000, 0),
    (0x00800000, 0x40800000, 0x00000000, 0),
    (0x00800000, 0x3FC00000, 0x00800000, 0),
    (0x7F7FFFFF, 0x3F000000, 0x7F800000, 0),
    (0xFF7FFFFF, 0x3F000000, 0xFF800000, 0),
    (0x40400000, 0x40000000, 0x3FC00000, 0),
    (0x3F800002, 0x3FA00000, 0x3F4CCCD0, 0),
    (0x3F800001, 0x3FC00000, 0x3F2AAAAC, 0),
    (0x3F800001, 0x3FA00000, 0x3F4CCCCE, 0),
    (0x3F800001, 0x3FE00000, 0x3F124926, 0),
]
ROOTS = [
    (0x805A5A5A, 0x00000000, 0),
    (0x40000000, 0x3FB504F3, 0),
    (0x41100000, 0x40400000, 0),
    (0x3F000000, 0x3F3504F3, 0),
    (0x3FC00000, 0x3F9CC471, 0),
    (0x00800000, 0x20000000, 0),
    (0x01000000, 0x203504F3, 0),
    (0x7F7FFFFF, 0x5F7FFFFF, 0),
    (0x3F800001, 0x3F800000, 0),
    (0x3F7FFFFF, 0x3F7FFFFF, 0),
    (0x7F812345, 0x7F800000, 0),
    (0xFFFFFFFF, 0xFF800000, 1),
    (0x80800000, 0xFF800000, 1),
    (0x40490FDB, 0x3FE2DFC5, 0),
    (0x3CDE838A, 0x3E28C3F3, 0),
]


def test_reference_by_hand() -> None:
    fmt = ZkfFormat(8, 24)
    model = Divsqrt(fmt)
    for a, b, y, error in DIVISIONS:
        quotient = model.evaluate(0, Zkf(fmt, a), Zkf(fmt, b))
        assert (quotient.y.bits, quotient.error) == (y, error), (hex(a), hex(b))
    for a, y, error in ROOTS:
        root = model.evaluate(1, Zkf(fmt, a), Zkf(fmt, 0))
        assert (root.y.bits, root.error) == (y, error), hex(a)


def divisions(fmt: ZkfFormat) -> list[tuple[int, int]]:
    """
    Every pair of the directed numbers; operands at the prenormalization, on the thresholds of the folded first
    digit, at the last decision by compares, and at the limits of the result's exponent.
    """
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    numbers = list(directed_numbers(fmt).values()) if fmt.wexp >= 3 else []
    out = [(a, b) for a in numbers for b in numbers]
    one, unit = fmt.bias << fmt.wfrac, 1 << fmt.wfrac
    for f in (0, 1, 2, ones >> 1, ones - 1, ones):
        out += [
            ((one | f) + near, one | f) for near in (-2, -1, 0, 1, 2) if ((one | f) + near) >> fmt.wfrac == fmt.bias
        ]
    out.append((one | ones, one))
    # Four and eight times the dividend on 5, 6 and 7 times the divisor exactly, where there are such operands.
    for k, shift, f in product((5, 6, 7), (2, 3), range(min(unit, 64))):
        scaled = k * (unit | f)
        if scaled % (1 << shift) == 0 and unit <= scaled >> shift < 2 * unit:
            out.append((one | ((scaled >> shift) & ones), one | f))
    for ea, eb in product((1, 2, fmt.bias, top - 1, top), (1, fmt.bias, top)):
        out += [(fmt.pack(0, ea, fa).bits, fmt.pack(0, eb, fb).bits) for fa, fb in ((0, 0), (0, ones), (ones, 0))]
    return out


def radicands(fmt: ZkfFormat) -> list[int]:
    """
    Zeros and infinities, powers of two at both parities of the exponent, exact squares with their neighbors, and
    what the last digit and the guard turn on: one set bit at each place of the fraction, and a fraction of ones.
    """
    negative, inf = 1 << fmt.sign_shift, fmt.exp_inf << fmt.wfrac
    out = [0, 1, negative, negative | 1, inf, negative | inf, (1 << fmt.wfull) - 1]
    if fmt.wexp >= 3:
        out += list(directed_numbers(fmt).values())
        exponents = [fmt.bias + k for k in range(-4, 5) if 1 <= fmt.bias + k <= fmt.exp_max_finite]
        out += [fmt.pack(sign, exp, 0).bits for exp in exponents for sign in (0, 1)]
        out += [fmt.pack(0, 1, 1).bits, fmt.pack(0, 2, 0).bits]
    for exp in (fmt.bias, fmt.bias + 1):
        if exp <= fmt.exp_max_finite:
            out += [fmt.pack(0, exp, fmt.frac_mask).bits] + [
                fmt.pack(0, exp, 1 << bit).bits for bit in range(fmt.wfrac)
            ]
    # A root of half the significand's bits at most has an exact square.
    half = fmt.wman // 2
    for q, scale in product(range(1 << min(half - 1, 6)), (0, 1, -1)):
        root = Fraction((1 << (half - 1)) | (q << max(0, half - 7)), 1 << (half - 1)) * Fraction(2) ** scale
        square = fmt.encode(root * root)
        if square.is_normal and square.to_fraction() == root * root:
            out += [square.bits + near for near in (-1, 0, 1)]
    return out


def drawn_division(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    v = directed_numbers(fmt)
    pick = int(rng.integers(0, 13))
    if pick == 0:
        return random_zero(fmt, rng), random_operand(fmt, rng)
    if pick == 1:
        return random_operand(fmt, rng), random_zero(fmt, rng)
    if pick == 2:
        return random_normal(fmt, rng), random_inf(fmt, rng)
    if pick == 3:
        return random_inf(fmt, rng), random_normal(fmt, rng)
    if pick == 4:  # quotients near the smallest normal
        return random_normal_near(fmt, rng, [1, 2], [0, 1]), random_normal_near(fmt, rng, [fmt.bias], [0])
    if pick == 5:  # and near the overflow
        return random_normal_near(fmt, rng, [fmt.exp_max_finite], [fmt.frac_mask]), v["half"]
    if pick == 6:
        divisor = random_normal_near(fmt, rng, [fmt.bias], [0, 1, 1 << (fmt.wfrac - 1)])
        return random_normal_near(fmt, rng, [fmt.bias], [0, 1]), divisor
    if pick == 7:
        return fmt.pack(int(rng.integers(0, 2)), 1, int(rng.integers(0, min(4, fmt.frac_mask + 1)))).bits, v["two"]
    if pick == 8:
        return random_normal(fmt, rng), v["one"]
    return random_operand(fmt, rng), random_operand(fmt, rng)


def drawn_radicand(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    pick = int(rng.integers(0, 10))
    if pick == 0:
        return random_zero(fmt, rng)
    if pick == 1:
        return random_inf(fmt, rng)
    if pick == 2:
        return random_normal_near(fmt, rng, [fmt.bias, fmt.bias + 1], [0, 1, ones])
    if pick == 3:
        return random_normal_near(fmt, rng, [1, 2], [0, 1, ones])
    if pick == 4:
        return random_normal_near(fmt, rng, [top - 1, top], [0, ones])
    if pick == 5:
        return random_bits(fmt.wfull, rng)
    return random_operand(fmt, rng)


def operands(
    fmt: ZkfFormat, kind: str | int, root: bool, rng: np.random.Generator, shard: int, shards: int
) -> list[tuple[int, int]]:
    """
    Of one operation, a root's second being unused: "every" operand, the dividends being those of one `shard` of so
    many; every pair of "significands" (the exponents only move the result's); or the directed ones and as many
    random ones as `kind` says.
    """
    every, fractions = range(1 << fmt.wfull), range(1 << fmt.wfrac)
    if kind == "every":
        return [(a, 0) for a in every] if root else [(a, b) for a in every[shard::shards] for b in every]
    if kind == "significands":
        parities = [fmt.bias, fmt.bias + 1][: 2 if fmt.bias < fmt.exp_max_finite else 1]
        if root:
            return [(fmt.pack(0, exp, f).bits, 0) for exp in parities for f in fractions]
        return [
            (fmt.pack(0, fmt.bias, fa).bits, fmt.pack(0, fmt.bias, fb).bits) for fa in fractions for fb in fractions
        ]
    assert isinstance(kind, int)
    out = [(a, 0) for a in radicands(fmt)] if root else divisions(fmt)
    if (fmt.wexp, fmt.wman) == (8, 24):
        out += [(a, 0) for a, _, _ in ROOTS] if root else [(a, b) for a, b, _, _ in DIVISIONS]
    out = list(dict.fromkeys(out))
    seen = set(out)
    target = len(out) + min(kind, (1 << (fmt.wfull * (1 if root else 2))) - len(out))
    while len(out) < target:
        drawn = (drawn_radicand(fmt, rng), 0) if root else drawn_division(fmt, rng)
        if drawn not in seen:
            seen.add(drawn)
            out.append(drawn)
    return out


def run(wexp: int, wman: int, kind: str | int, mode: int, shard: int = 0, shards: int = 1, **knobs: Any) -> None:
    simulate(Divsqrt(ZkfFormat(wexp, wman), mode=mode, **knobs), kind, shard, shards)


def simulate(model: Divsqrt, kind: str | int, shard: int = 0, shards: int = 1) -> None:
    """In the runtime mode the two operations come mixed; a root's `b` is noise, a zero exponent among it."""
    fmt, mode = model.fmt, model.mode
    rng = np.random.default_rng(SEED)
    cases = []
    for root in (False, True):
        if mode == 2 or mode == root:
            for a, b in operands(fmt, kind, root, rng, shard, shards):
                op = int(root) if mode == 2 else int(rng.integers(0, 2))
                if root:
                    b = random_bits(fmt.wfull, rng) if rng.integers(0, 3) else int(rng.integers(0, 1 << fmt.wfrac))
                found = model.evaluate(op, Zkf(fmt, a), Zkf(fmt, b))
                cases.append(Case({"a": a, "b": b, "op_sqrt": op}, found._asdict()))
    cases = [cases[i] for i in rng.permutation(len(cases))]
    run_pipeline(model.component(), model.timing.latency, cases)


def decodes(wman: int) -> tuple[Divsqrt.Core, ...]:
    """The values of `stage_core`: an odd significand has no first digit to unfold."""
    return (Divsqrt.Core(0), Divsqrt.Core.FIRST_DIGIT) if wman % 2 == 0 else (Divsqrt.Core(0),)


# mode, wexp, wman, kind: division where two operands can be walked, roots further, the runtime mode on both.
FORMATS = (
    [(0, 2, 4, "every"), (0, 3, 4, "every"), (0, 3, 5, 512), (0, 4, 6, 512), (0, 5, 11, 768), (0, 6, 18, 768)]
    + [(0, 7, 17, 768), (0, 8, 24, 1024), (0, 11, 53, 384)]
    + [(1, 2, 4, "every"), (1, 3, 4, "every"), (1, 5, 11, 512), (1, 8, 24, 1024), (1, 11, 53, 384)]
    + [(1, 3, 5, "every"), (1, 4, 6, "every"), (1, 4, 7, "every")]
    + [(2, 2, 4, "every"), (2, 3, 4, "every"), (2, 3, 5, "every"), (2, 4, 6, 1024), (2, 5, 11, 768)]
    + [(2, 6, 18, 768), (2, 8, 27, 768), (2, 8, 36, 768), (2, 11, 53, 384)]
)


@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int, str | int]) -> None:
    for stage_core in decodes(row[2]):
        run(row[1], row[2], row[3], row[0], stage_core=stage_core)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("wman", [4, 5])
def test_stages(mode: int, wman: int) -> None:
    each = ({"stage_input": 1}, {"stage_output": 1}) if wman == 4 else ()
    for knobs in (*each, {"stage_input": 2, "stage_core": decodes(wman)[-1], "stage_output": 1}, {"stage_output": 3}):
        run(3, wman, "every", mode, **knobs)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Divsqrt)])
def test_catalog(model: Divsqrt) -> None:
    simulate(model, 768)


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    unfolded = Divsqrt(fmt, stage_core=Divsqrt.Core.FIRST_DIGIT, stage_output=3)
    assert Divsqrt(fmt).timing.latency == 8 and unfolded.timing.latency == 12
    with pytest.raises(ValueError):
        Divsqrt(fmt, mode=3)
    with pytest.raises(
        ValueError, match=re.escape("stage_core=Core.FIRST_DIGIT: WMAN=17 is odd; leave FIRST_DIGIT out")
    ):
        Divsqrt(ZkfFormat(6, 17), stage_core=Divsqrt.Core.FIRST_DIGIT)


# mode, wexp, wman, kind, stage_core, shard, shards: a shard holds 2**18 divisions at most.
NONE = Divsqrt.Core(0)
DEEP: list[tuple[int, int, int, str | int, Divsqrt.Core, int, int]] = (
    [(mode, 4, 6, "every", NONE, shard, shards) for mode, shards in ((0, 4), (2, 4)) for shard in range(shards)]
    + [(mode, 4, 7, "every", NONE, shard, 16) for mode in (0, 2) for shard in range(16)]
    + [(mode, 5, m, "significands", sd, 0, 1) for mode in (0, 1, 2) for m in (8, 9, 10) for sd in decodes(m)]
    + [(mode, w, m, "every", sd, 0, 1) for mode in (0, 1, 2) for w, m in ((3, 6), (4, 5)) for sd in decodes(m)]
    + [
        (0, w, m, kind, NONE, 0, 1)
        for w, m, kind in ((3, 5, "every"), (2, 5, "every"), (5, 4, "every"), (6, 17, 512), (8, 23, 512), (7, 12, 512))
    ]
    + [(mode, w, m, 512, NONE, 0, 1) for mode in (1, 2) for w, m in ((6, 17), (8, 23))]
    + [(1, 2, 5, "every", NONE, 0, 1)]
    + [
        (mode, w, m, count, NONE, 0, 1)
        for mode in (0, 1, 2)
        for w, m, count in ((8, 48, 384), (7, 53, 384), (8, 36, 1024))
    ]
)


@pytest.mark.deep
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int, str | int, Divsqrt.Core, int, int]) -> None:
    mode, wexp, wman, kind, stage_core, shard, shards = row
    run(wexp, wman, kind, mode, shard, shards, stage_core=stage_core)


# Two knobs or more: each alone is a row of `test_formats` or `test_stages`.
@pytest.mark.deep
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("stages", [stages for stages in range(8) if stages & (stages - 1)])
def test_deep_stages(mode: int, stages: int) -> None:
    values: dict[str, Any] = {"stage_input": 1, "stage_core": Divsqrt.Core.FIRST_DIGIT, "stage_output": 1}
    run(3, 4, "every", mode, **{name: value for bit, (name, value) in enumerate(values.items()) if (stages >> bit) & 1})

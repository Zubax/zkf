"""The tables of `exp2` and `log2`: the search gives the certified ones, and the certificate holds what it says."""

import random
from dataclasses import replace
from fractions import Fraction

import pytest

from zkf import Exp2, Log2, UnsupportedFormat, ZkfFormat
from _certificate import certify
from _trans import OTHERS, TABLES, build, every_row, simulate
from zkf._trans import Kernel, Spec, _arith_range, kernel_levels, spec as table


@pytest.mark.parametrize("key", TABLES + OTHERS, ids=str)
def test_certificate(key: tuple[str, int, int, bool]) -> None:
    spec = table(*key)
    certify(spec)


@pytest.mark.deep
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("wman", range(10, 54))
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_every_width(function: str, wman: int, fast: bool) -> None:
    """
    The tables at every width from 8 to 64 that generation accepts (`fast`'s, at the native widths) and at which
    neither operand of `c1*w` is wider than four multipliers: each distinct one certified, the kernel at each width on
    every row with its products cut as far as they go, and the operator whole at the narrowest width that it takes (16
    at least), at 18 and at 24, with every register of its core.
    """
    found: dict[int, Spec] = {}
    for width in range(8, 65) if not fast else (16, 18, 24, 27):
        try:
            spec = table(function, wman, width, fast)
        except UnsupportedFormat:
            continue
        if max(max(row[1] for row in spec.coeffs).bit_length(), spec.t[0]) <= 4 * width:
            found[width] = spec
    for spec in set(found.values()):
        certify(spec)
    for width, spec in found.items():
        every_row(spec, Kernel(spec, wmultiplier=width, level=kernel_levels(spec, width)[-1]))
    every = ~(Exp2 if function == "exp2" else Log2).CoreStages.NONE
    for width in sorted({min(width for width in found if width >= 16), 18, 24}):
        simulate(build(function, ZkfFormat(8, wman), width, stage_core=every, fast=fast), 1000)


def test_what_has_no_table_is_refused() -> None:
    """
    Out of range, as the model is made; and, below the widths that the operators take, where no segmentation fits the
    multipliers: e.g. at 24 and 26 a square, and at 36 the cubic coefficient, is wider than two of them at every one.
    """
    for operator, wman in ((Exp2, 9), (Log2, 54)):
        with pytest.raises(UnsupportedFormat, match="no .* table"):
            operator(ZkfFormat(8, wman), wmultiplier=18)
    for function, wman, wmultiplier in (
        ("exp2", 20, 8),
        ("log2", 20, 8),
        ("exp2", 24, 8),
        ("exp2", 26, 8),
        ("log2", 33, 8),
        ("log2", 36, 9),
    ):
        with pytest.raises(UnsupportedFormat, match="no .* table"):
            table(function, wman, wmultiplier, True)
    Exp2(ZkfFormat(8, 16), wmultiplier=30)  # a multiplier that takes every segment


def test_the_arithmetic_bound_holds_the_kernel() -> None:
    """Random narrow kernels of every structure, each against its exact polynomial at every argument."""
    rng = random.Random(1)
    kinds: dict[str, int] = {}
    while sum(kinds.values()) < 1500:
        function, d, rw = rng.choice(["exp2", "log2"]), rng.choice([1, 2, 2, 3, 3, 3, 4, 5]), rng.randrange(3, 10)
        t2 = rng.randrange(1, rw + 1) if d > 1 else 0
        q = [rng.randrange(1, 2 * t2 + 1) for _ in range(d // 2)]
        if q:
            q[0] = max(q)
        t = tuple(rng.randrange(1, rw + 1) for _ in range((d + 1) // 2))
        widths = sorted((rng.randrange(2, 22) for _ in range(d + 1)), reverse=True)
        row = [rng.getrandbits(width) | (1 << (width - 1)) for width in widths]
        if function == "log2" and d >= 3:
            row.sort(reverse=True)
        spec = Spec(function, 0, 0, rw, 0, d, t, t2, tuple(q))
        kind = f"degree {d}"
        if d == 2 and rng.random() < 0.5:
            spec, kind = replace(spec, tc=rw - t[0], sc=rng.randrange(0, rw + 2)), kind + ", uneven split"
        if d == 3 and t2 >= 2 and rng.random() < 0.7:
            th, sg = rng.randrange((t2 + 1) // 2, t2), rng.randrange(1, 4)
            spec, kind = replace(spec, th=th, sg=sg, q=(max(1, min(q[0], 2 * th - sg)),)), kind + ", two-tile square"
        if d >= 4 and rng.random() < 0.5:
            spec, kind = replace(spec, tn=rng.randrange((max(q[1:]) + 1) // 2, rw + 1)), kind + ", narrow square"
        try:
            lo, hi = _arith_range(spec, row)
        except ValueError:
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
        for w in range(1 << rw):
            wn = Fraction(w, 1 << rw)
            exact = sum((-c if function == "log2" and j % 2 else c) * wn**j for j, c in enumerate(row))
            assert lo <= exact - spec._polynomial(row, w) <= hi, (spec, row, w)
    assert len(kinds) == 9 and min(kinds.values()) > 40, kinds

"""The tables of `exp2` and `log2`: the search gives the certified ones, and the certificate holds what it says."""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction

import mpmath
import pytest

import zkf._trans as tables
from zkf import Exp2, Log2, UnsupportedFormat, ZkfFormat
from _certificate import certify
from _trans import build, simulate
from test_kernel import every_row
from zkf._trans import Kernel, Spec, _arith_range, _estimate, _generate, degree, kernel_counts, spec as table

# Those of 0.9.0, and others that it never had: other significands, degree 4 among them, and other multipliers.
TABLES = [(function, wman, 18) for function in ("exp2", "log2") for wman in (16, 18, 24, 27, 32, 36, 48, 53)]
OTHERS = [
    (function, wman, width)
    for function in ("exp2", "log2")
    for wman, width in ((20, 18), (25, 17), (30, 24), (40, 18), (44, 27), (17, 8), (37, 25), (47, 18), (26, 12))
]
# A structure that the formats reach only at widths no test can enumerate, generated at WMAN 16: degree 3 with a
# square of two multipliers, a shortened `c3` and a shortened wide product; degree 4, whose last pair is one
# coefficient; degree 5.
INSTRUMENTS: dict[str, tuple[str, int, dict[str, int]]] = {
    "exp2 of degree 3": ("exp2", 11, {"d": 3, "k_base": 4}),
    "log2 of degree 3": ("log2", 12, {"d": 3, "k_base": 4}),
    "exp2 of degree 4": ("exp2", 18, {"d": 4, "k_base": 3}),
    "log2 of degree 4": ("log2", 18, {"d": 4, "k_base": 4}),
    "exp2 of degree 5": ("exp2", 18, {"d": 5, "k_base": 2}),
    "log2 of degree 5": ("log2", 18, {"d": 5, "k_base": 3}),
}


def instrument(name: str) -> Spec:
    function, wmultiplier, shape = INSTRUMENTS[name]
    return _generate(function, 16, wmultiplier, **shape)


@pytest.mark.parametrize("key", TABLES + OTHERS, ids=str)
def test_certificate(key: tuple[str, int, int]) -> None:
    spec = table(*key)
    certify(spec)
    assert spec.d == degree(spec.wman)


@pytest.mark.deep
@pytest.mark.parametrize("wman", range(16, 54))
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_every_width(function: str, wman: int) -> None:
    """
    The tables at every width from 8 to 64 that generation accepts and at which neither operand of `c1*w` is wider
    than four multipliers: each distinct one certified, the kernel at each width on every row with its products cut as
    far as they go, and the operator whole at the narrowest width, at 18 and at 24, with every register of its core.
    """
    found: dict[int, Spec] = {}
    for width in range(8, 65):
        try:
            spec = table(function, wman, width)
        except UnsupportedFormat:
            continue
        if max(max(row[1] for row in spec.coeffs).bit_length(), spec.t[0]) <= 4 * width:
            found[width] = spec
    for spec in set(found.values()):
        certify(spec)
        assert spec.d == degree(spec.wman)
    for width, spec in found.items():
        every_row(spec, Kernel(spec, wmultiplier=width, stages=kernel_counts(spec, width).stop - 1))
    every = ~(Exp2 if function == "exp2" else Log2).CoreStages(0)
    for width in sorted({min(found), 18, 24}):
        simulate(build(function, ZkfFormat(8, wman), width, stage_core=every), 1000)


def test_what_has_no_table_is_refused() -> None:
    for operator, wman in ((Exp2, 15), (Log2, 54)):
        with pytest.raises(UnsupportedFormat, match="no .* table"):
            operator(ZkfFormat(8, wman), wmultiplier=18)
    for operator in (Exp2, Log2):  # generated as the model is made
        with pytest.raises(UnsupportedFormat, match="no .* table"):
            operator(ZkfFormat(8, 20), wmultiplier=8)


def test_the_precision_of_the_caller_is_neither_read_nor_changed() -> None:
    for precision in (20, 53, 1000):
        with mpmath.workprec(precision):
            found = _generate("log2", 16, 18)
            certify(found)
            assert found == table("log2", 16, 18)
            assert mpmath.mp.prec == precision and mpmath.iv.prec == 53


def test_no_table_fits() -> None:
    """A square that no multiplier of the width holds at any segmentation; a multiplier that takes every segment."""
    with pytest.raises(UnsupportedFormat, match="no exp2 table"):
        _generate("exp2", 24, 8)
    with pytest.raises(UnsupportedFormat, match="no log2 table"):
        _generate("log2", 36, 9)  # the cubic coefficient is wider than two multipliers at every segmentation
    assert _generate("exp2", 16, 30).tc == 0


def test_segmentations_that_do_not_fit_are_passed_over() -> None:
    assert _generate("exp2", 16, 18, d=2, k_base=2).k == 6  # the estimate is over the budget below
    assert _generate("exp2", 16, 8, d=3, k_base=3).k == 7  # and a square is wider than two multipliers
    shortened, whole = _generate("exp2", 16, 10, d=3, k_base=2), _generate("exp2", 16, 12, d=3, k_base=2)
    assert shortened.t[0] < shortened.rw and whole.t[0] == whole.rw  # the budget allows a shorter wide product, or not
    exp2 = table("exp2", 16, 18)
    assert _estimate(replace(exp2, t2=0), [list(row) for row in exp2.coeffs]) == (float("inf"), ())


def test_the_arithmetic_bound_holds_the_kernel() -> None:
    """Random narrow kernels of every structure, each against its exact polynomial at every argument."""
    rng = random.Random(1)
    kinds: dict[str, int] = {}
    while sum(kinds.values()) < 1500:
        function, d, rw = rng.choice(["exp2", "log2"]), rng.choice([2, 2, 3, 3, 3, 4, 5]), rng.randrange(3, 10)
        t2 = rng.randrange(1, rw + 1)
        q = [rng.randrange(1, 2 * t2 + 1) for _ in range(d // 2)]
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
        try:
            lo, hi = _arith_range(spec, row)
        except ValueError:
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
        for w in range(1 << rw):
            wn = Fraction(w, 1 << rw)
            exact = sum((-c if function == "log2" and j % 2 else c) * wn**j for j, c in enumerate(row))
            assert lo <= exact - spec._polynomial(row, w) <= hi, (spec, row, w)
    assert len(kinds) == 6 and min(kinds.values()) > 40, kinds


def test_tables_are_generated_from_threads() -> None:
    """A flow compiles its specs from threads, and the search has one context: a second search must not move it."""
    tables._generated.cache_clear()
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda key: table(*key), TABLES))
    tables._generated.cache_clear()
    assert found == [table(*key) for key in TABLES]
    assert tables._fitting.prec == 280

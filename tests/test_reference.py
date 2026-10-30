"""
The reference itself, the value type and `evaluate` of the operators: against NumPy on the two formats that coincide
with IEEE 754 (the FPU of the machine is the oracle; neither has NaNs or subnormals here), against independent
formulations elsewhere, and the contract that every `evaluate` keeps.
"""

import math
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
from math import isqrt

import mpmath
import numpy as np
import pytest

import _ieee
import test_add
import test_divsqrt
import test_fma
import test_mul
from _ieee import BINARY32, BINARY64, of_numpy, to_numpy
from _operands import SEED
from zkf import AbsNeg, Add, Cmp, Cordic, Divsqrt, Exp2, Finite, Fma, FromInt, Ilog2, Log2, Mul, MulIlog2, Pipe
from zkf import Resize, Rint, RoundMode, UnsupportedFormat, Zkf, ZkfFormat
from zkf.__main__ import main
from zkf._cordic import _atan2_canon_half

# Name, bits, sign, exponent field and fraction, at 8/24 and at 11/53.
LAYOUT32 = [
    ("+zero", 0x00000000, 0, 0x00, 0x000000),
    ("-zero", 0x80000000, 1, 0x00, 0x000000),
    ("+min_normal", 0x00800000, 0, 0x01, 0x000000),
    ("-min_normal", 0x80800000, 1, 0x01, 0x000000),
    ("+half", 0x3F000000, 0, 0x7E, 0x000000),
    ("+one", 0x3F800000, 0, 0x7F, 0x000000),
    ("-one", 0xBF800000, 1, 0x7F, 0x000000),
    ("+one_and_half", 0x3FC00000, 0, 0x7F, 0x400000),
    ("+two", 0x40000000, 0, 0x80, 0x000000),
    ("+max_finite", 0x7F7FFFFF, 0, 0xFE, 0x7FFFFF),
    ("-max_finite", 0xFF7FFFFF, 1, 0xFE, 0x7FFFFF),
    ("+inf", 0x7F800000, 0, 0xFF, 0x000000),
    ("-inf", 0xFF800000, 1, 0xFF, 0x000000),
]
LAYOUT64 = [
    ("+zero", 0x0000000000000000, 0, 0x000, 0x0000000000000),
    ("-zero", 0x8000000000000000, 1, 0x000, 0x0000000000000),
    ("+min_normal", 0x0010000000000000, 0, 0x001, 0x0000000000000),
    ("-min_normal", 0x8010000000000000, 1, 0x001, 0x0000000000000),
    ("+half", 0x3FE0000000000000, 0, 0x3FE, 0x0000000000000),
    ("+one", 0x3FF0000000000000, 0, 0x3FF, 0x0000000000000),
    ("-one", 0xBFF0000000000000, 1, 0x3FF, 0x0000000000000),
    ("+one_and_half", 0x3FF8000000000000, 0, 0x3FF, 0x8000000000000),
    ("+two", 0x4000000000000000, 0, 0x400, 0x0000000000000),
    ("+max_finite", 0x7FEFFFFFFFFFFFFF, 0, 0x7FE, 0xFFFFFFFFFFFFF),
    ("-max_finite", 0xFFEFFFFFFFFFFFFF, 1, 0x7FE, 0xFFFFFFFFFFFFF),
    ("+inf", 0x7FF0000000000000, 0, 0x7FF, 0x0000000000000),
    ("-inf", 0xFFF0000000000000, 1, 0x7FF, 0x0000000000000),
]


def magnitude(fmt: ZkfFormat, exp: int, frac: int) -> Fraction:
    return ((1 << fmt.wfrac) | frac) * Fraction(2) ** (exp - fmt.bias - fmt.wfrac)


def encoded(fmt: ZkfFormat, sign: int, value: Fraction) -> int:
    return fmt.encode(-value if sign else value).bits


@pytest.mark.parametrize("fmt, cases", [(BINARY32, LAYOUT32), (BINARY64, LAYOUT64)], ids=str)
def test_layout(fmt: ZkfFormat, cases: list[tuple[str, int, int, int, int]]) -> None:
    for name, bits, sign, exp, frac in cases:
        value = to_numpy(fmt, bits)
        assert not np.isnan(value) and of_numpy(fmt, value) == bits, name
        decoded = Zkf(fmt, bits)
        assert fmt.pack(sign, exp, frac).bits == bits and (decoded.negative, decoded.exp, decoded.frac) == (
            sign,
            exp,
            frac,
        ), name
        classes = (decoded.is_zero, decoded.is_normal, decoded.is_inf)
        assert classes == (exp == 0, 0 < exp < fmt.exp_inf, exp == fmt.exp_inf), name
        if exp == 0:
            assert fmt.pack(sign, 0, 0).bits == bits, name
        elif exp == fmt.exp_inf:
            assert fmt.inf(sign).bits == bits, name
        else:
            assert fmt.pack(sign, exp, frac).bits == bits == encoded(fmt, sign, magnitude(fmt, exp, frac)), name


@pytest.mark.parametrize("fmt", [BINARY32, BINARY64], ids=str)
def test_layout_of_random_normals(fmt: ZkfFormat) -> None:
    rng = random.Random(fmt.wman)
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    exponents = (1, 2, fmt.bias - 1, fmt.bias, fmt.bias + 1, top - 1, top)
    fields = [(s, e, f) for s in (0, 1) for e in exponents for f in (0, 1, 1 << (fmt.wfrac - 1), ones - 1, ones)]
    fields += [(rng.randrange(2), rng.randint(1, top), rng.getrandbits(fmt.wfrac)) for _ in range(5000)]
    for sign, exp, frac in fields:
        bits = fmt.pack(sign, exp, frac).bits
        value = to_numpy(fmt, bits)
        assert np.isfinite(value) and value != 0 and of_numpy(fmt, value) == bits, hex(bits)
        decoded = Zkf(fmt, bits)
        assert (decoded.negative, decoded.exp, decoded.frac, decoded.is_normal) == (sign, exp, frac, True)
        assert encoded(fmt, sign, magnitude(fmt, exp, frac)) == bits


@pytest.mark.parametrize("fmt", [BINARY32, BINARY64, ZkfFormat(3, 4)], ids=str)
def test_rounding_between_zero_and_the_least_normal(fmt: ZkfFormat) -> None:
    least = Fraction(2) ** fmt.min_exp_unbiased
    for sign in (0, 1):
        assert encoded(fmt, sign, least * Fraction(3, 8)) == 0
        assert encoded(fmt, sign, least / 2) == fmt.pack(sign, 1, 0).bits
        assert encoded(fmt, sign, least * Fraction(3, 4)) == fmt.pack(sign, 1, 0).bits


def test_the_constants_and_the_conversions() -> None:
    """Those of a format as `python -m zkf` prints them, and what a value converts to."""
    fmt = ZkfFormat(6, 18)
    assert (fmt.lowest, fmt.max, fmt.epsilon) == (Fraction(1, 1 << 30), Fraction(0xFFFF_C000), Fraction(1, 1 << 17))
    assert fmt.encode(fmt.max) == fmt.pack(0, fmt.exp_max_finite, fmt.frac_mask)
    assert fmt.encode(fmt.lowest) == fmt.pack(0, 1, 0)
    assert (fmt.encode(math.inf), fmt.encode(-math.inf)) == (fmt.inf(0), fmt.inf(1))
    x, plus_inf, minus_inf = fmt.encode(-2.75), fmt.inf(0), fmt.inf(1)
    assert (x.is_finite, plus_inf.is_finite, x.to_mpf()) == (True, False, -2.75)
    assert (Zkf(fmt, 0).to_mpf(), plus_inf.to_mpf(), minus_inf.to_mpf()) == (0, mpmath.inf, -mpmath.inf)
    assert (fmt.encode(mpmath.inf), fmt.encode(-mpmath.inf)) == (plus_inf, minus_inf)
    wide = ZkfFormat(8, 64).encode(-1 - Fraction(1, 1 << 63))
    with mpmath.workprec(53):  # below its significand
        assert wide.fmt.encode(wide.to_mpf()) == wide
    with mpmath.workprec(160):  # above it
        assert fmt.encode(1).to_mpf() + mpmath.ldexp(1, -90) - 1 == mpmath.ldexp(1, -90)
    assert (float(x), float(plus_inf), float(minus_inf), int(x)) == (-2.75, math.inf, -math.inf, -2)
    assert (bool(x), bool(Zkf(fmt, 1 << 23 | 5))) == (True, False)
    assert (str(x), f"{x:x}", f"{x:.1f}", repr(x)) == ("-2.75", "0xc0c000", "-2.8", f"Zkf({fmt!r}, 0xc0c000)")


def test_printed_beyond_a_float(capsys: pytest.CaptureFixture[str]) -> None:
    """Values, and the constants of a format as `python -m zkf` prints them, beyond what a float holds."""
    fmt = ZkfFormat(12, 24)
    assert (f"{fmt.encode(1 << 1500):.3e}", f"{fmt.encode(fmt.lowest):.3e}") == ("3.507e+451", "1.238e-616")
    wide = ZkfFormat(12, 54)
    assert f"{wide.pack(0, 1023 + wide.bias, wide.frac_mask):.3e}" == "1.798e+308"  # a float rounds it to 2**1024
    assert f"{ZkfFormat(11, 64).encode(2**53 + 1):.0f}" == "9007199254740993"
    main(["zkf", "12", "24", "1e400"])
    printed = capsys.readouterr().out
    assert "max        = (2 - 2**-23) * 2**2047 ≈ 3.232e+616\n" in printed
    assert f"twos-complement: {fmt.encode(10**400).bits:036b}\n" in printed
    main(["zkf", "15", "113"])
    assert "\n" + "1" * 28 + "0" * 100 + "\n" in capsys.readouterr().out  # the hundreds of the bit numbers of binary128
    for value in ("nan", "1.5x"):
        with pytest.raises(SystemExit, match="^usage: "):
            main(["zkf", "8", "24", value])


def test_the_conversions_in_threads() -> None:
    """`to_mpf` and `encode` neither read nor change the precision that another thread computes at."""
    rng = random.Random(SEED)
    values = [
        fmt.pack(rng.getrandbits(1), rng.randint(1, fmt.exp_max_finite), rng.getrandbits(fmt.wfrac))
        for fmt in (BINARY32, ZkfFormat(8, 36), BINARY64, ZkfFormat(8, 64))
        for _ in range(5000)
    ]

    def wrong(part: list[Zkf]) -> int:
        return sum(x.fmt.encode(x.to_mpf()) != x for x in part)

    with mpmath.workprec(160), ThreadPoolExecutor(4) as pool:
        found = sum(pool.map(wrong, [values[i::4] for i in range(4)] * 3))
        assert (found, mpmath.mp.prec) == (0, 160)


def test_what_the_value_type_refuses() -> None:
    """A NaN, fields or bits that the format does not hold, and what an infinity has no value of."""
    for wexp, wman in ((1, 4), (2, 3)):
        with pytest.raises(UnsupportedFormat):
            ZkfFormat(wexp, wman)
    fmt = ZkfFormat(6, 18)
    for nan in (math.nan, mpmath.nan):
        with pytest.raises(ValueError):
            fmt.encode(nan)
    for sign, exp, frac in ((2, 1, 0), (0, -1, 0), (0, fmt.exp_inf + 1, 0), (0, 1, -1), (0, 1, fmt.frac_mask + 1)):
        with pytest.raises(ValueError):
            fmt.pack(sign, exp, frac)
    assert (fmt.pack(1, 0, 5).is_zero, fmt.pack(0, fmt.exp_inf, 5).is_inf) == (True, True)
    for bits in (-1, 1 << fmt.wfull):
        with pytest.raises(ValueError):
            Zkf(fmt, bits)
    with pytest.raises(ValueError):
        fmt.inf(0).to_fraction()
    with pytest.raises(OverflowError):
        int(fmt.inf(1))


def canonical(fmt: ZkfFormat, count: int) -> list[int]:
    """Corners, random normals, and small ones whose products and quotients fall to where a result is flushed."""
    rng = random.Random(fmt.wman)
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    out = [
        0,
        fmt.inf(0).bits,
        fmt.inf(1).bits,
        fmt.pack(0, 2, 0).bits,
        fmt.pack(0, fmt.bias - 1, 0).bits,
    ]
    out += [fmt.pack(sign, exp, frac).bits for sign in (0, 1) for exp, frac in ((1, 0), (fmt.bias, 0), (top, ones))]
    out += [fmt.pack(0, fmt.bias + 1, ones).bits]
    for highest in (top, fmt.bias):
        out += [
            fmt.pack(rng.randrange(2), rng.randint(1, highest), rng.getrandbits(fmt.wfrac)).bits for _ in range(count)
        ]
    return out


@pytest.mark.parametrize("fmt", [BINARY32, BINARY64], ids=str)
def test_arithmetic_is_that_of_numpy(fmt: ZkfFormat) -> None:
    """On canonical operands, the only ones the oracle answers, and on those of the benches."""
    add, divsqrt = Add(fmt), Divsqrt(fmt)
    mul, fma = Mul(fmt, stage_core_mul=3, wmultiplier=18), Fma(fmt, stage_core_mul=3, wmultiplier=18)
    rng = np.random.default_rng(SEED)
    values = [Zkf(fmt, bits) for bits in canonical(fmt, 28)]
    pairs = [(a, b) for a in values for b in values]
    pairs += [(Zkf(fmt, a), Zkf(fmt, b)) for a, b in (*test_add.pairs(fmt, 1500), *test_mul.pairs(fmt, 1500))]
    pairs += [(Zkf(fmt, a), Zkf(fmt, b)) for a, b in test_divsqrt.operands(fmt, 1500, False, rng, 0, 1)]
    checked = 0
    for a, b in pairs:
        product, total, quotient = _ieee.mul(a, b), _ieee.add(a, b), _ieee.div(a, b)
        if product is not None:
            assert mul.evaluate(a, b) == product, (a, b)
        if total is not None:
            assert add.evaluate(a, b) == total, (a, b)
        if quotient is not None:
            assert divsqrt.evaluate(0, a, b) == quotient, (a, b)
        checked += sum(found is not None for found in (product, total, quotient))
    roots = values + [Zkf(fmt, a) for a, _ in test_divsqrt.operands(fmt, 1500, True, rng, 0, 1)]
    for a in roots:
        root = _ieee.sqrt(a)
        if root is not None:
            assert divsqrt.evaluate(1, a, Zkf(fmt, 0)) == root, a
            checked += 1
    triples = test_fma.directed(fma) + [test_fma.drawn(fma, rng) for _ in range(1500)]
    for a, b, c in ((Zkf(fmt, a), Zkf(fmt, b), Zkf(fmt, c)) for a, b, c in triples):
        fused = _ieee.fma(a, b, c)
        if fused is not None:
            assert fma.evaluate(a, b, c) == fused, (a, b, c)
            checked += 1
    assert checked > 15000


@pytest.mark.parametrize("fmt", [ZkfFormat(2, 4), ZkfFormat(3, 5), ZkfFormat(4, 6), ZkfFormat(4, 7)], ids=str)
def test_root_against_the_square_of_the_midpoint(fmt: ZkfFormat) -> None:
    """
    On every operand, against a formulation of its own: the root of `y = s << (wfrac + r)`
    cut to WMAN bits is rounded by comparing `4y` with `(2 * lo + 1)**2`, and not by the scaling and the guard of
    the reference, so that a mistake in those cannot pass.
    """
    divsqrt = Divsqrt(fmt)
    for bits in range(1 << fmt.wfull):
        z = Zkf(fmt, bits)
        found = divsqrt.evaluate(1, z, Zkf(fmt, 0))
        if z.is_zero:
            want = (0, False)
        elif z.negative:
            want = (fmt.inf(1).bits, True)
        elif z.is_inf:
            want = (fmt.inf(0).bits, False)
        else:
            e = z.exp - fmt.bias
            r = e & 1
            y = z.significand() << (fmt.wfrac + r)
            lo = isqrt(y)
            up = 4 * y > (2 * lo + 1) ** 2 or (4 * y == (2 * lo + 1) ** 2 and lo & 1)
            rounded = lo + bool(up)
            assert not rounded >> fmt.wman
            want = (fmt.pack(0, (e - r) // 2 + fmt.bias, rounded & fmt.frac_mask).bits, False)
        assert (found.y.bits, found.error) == want, hex(bits)


def test_rounding_to_an_integer() -> None:
    fmt = BINARY32
    assert [int(mode) for mode in RoundMode] == [0, 1, 2, 3]
    assert (RoundMode.NEAREST_EVEN, RoundMode.FLOOR, RoundMode.CEIL, RoundMode.TRUNC) == tuple(RoundMode)
    rint = Rint(fmt, wint=8)
    for value, expected in (
        (Fraction(1, 2), (0, 0, 1, 0)),
        (Fraction(-1, 2), (0, -1, 0, 0)),
        (Fraction(5, 4), (1, 1, 2, 1)),
        (Fraction(-5, 4), (-1, -2, -1, -1)),
        (Fraction(3, 2), (2, 1, 2, 1)),
        (Fraction(-3, 2), (-2, -2, -1, -1)),
        (Fraction(7, 4), (2, 1, 2, 1)),
        (Fraction(-7, 4), (-2, -2, -1, -1)),
    ):
        x = fmt.encode(value)
        for mode, result in zip(RoundMode, expected):
            assert rint.evaluate(x, mode) == (fmt.encode(result), result), (value, mode)
    nearest, floor, ceil, trunc = RoundMode
    rails = [(fmt.encode(128), nearest), (fmt.encode(-128), nearest), (fmt.encode(-129), ceil), (Zkf(fmt, 0), floor)]
    rails += [(Zkf(fmt, ((1 << 32) - 1)), ceil), (Zkf(fmt, 0x7F800001), trunc)]
    assert [rint.evaluate(x, mode).y_int for x, mode in rails] == [127, -128, -128, 0, -128, 127]
    with pytest.raises(UnsupportedFormat):
        Rint(fmt, wint=1)


@pytest.mark.parametrize("fmt", [ZkfFormat(2, 4), ZkfFormat(6, 18), BINARY32, BINARY64], ids=str)
def test_exponent(fmt: ZkfFormat) -> None:
    ilog2, mul_ilog2 = Ilog2(fmt), MulIlog2(fmt)
    for exp in range(1 << fmt.wexp):
        for sign in (0, 1):
            for frac in (0, fmt.frac_mask):
                a = Zkf(fmt, fmt.pack(sign, exp, frac).bits)
                e = ilog2.evaluate(a).y
                assert e == exp - fmt.bias
                if a.is_normal:
                    normalized = mul_ilog2.evaluate(a, -e)
                    assert (normalized.exp, normalized.negative, normalized.frac) == (fmt.bias, a.negative, frac)


def test_a_float_of_another_format_is_refused() -> None:
    """As any operand, in any mode, also where the mode does not read it; that of `Resize` is of its input format."""
    fmt, other = ZkfFormat(6, 18), ZkfFormat(6, 17)
    exp2, log2, cordic = (
        Exp2(fmt, stage_core_kernel=3, wmultiplier=18),
        Log2(fmt, stage_core_kernel=3, stage_core_mul=3, wmultiplier=18),
        Cordic(fmt, stage_core_mul=3, wmultiplier=18),
    )
    divsqrt, mul_ilog2, rint = Divsqrt(fmt), MulIlog2(fmt), Rint(fmt)
    root, rotation = Divsqrt(fmt, mode=1), Cordic(fmt, stage_core_mul=3, wmultiplier=18, mode=0)
    evaluations: list[tuple[Callable[..., object], int]] = [
        (AbsNeg(fmt).evaluate, 1),
        (Add(fmt).evaluate, 2),
        (Cmp(fmt).evaluate, 2),
        (exp2.evaluate, 1),
        (lambda x: exp2.evaluate(x, oracle=True), 1),
        (Finite(fmt).evaluate, 1),
        (Fma(fmt, wmultiplier=18).evaluate, 3),
        (Ilog2(fmt).evaluate, 1),
        (log2.evaluate, 1),
        (lambda x: log2.evaluate(x, oracle=True), 1),
        (Mul(fmt, wmultiplier=18).evaluate, 2),
        (lambda a: mul_ilog2.evaluate(a, 1), 1),
        (lambda a: rint.evaluate(a, RoundMode.FLOOR), 1),
        (lambda a, b: cordic.evaluate(0, a, b), 2),
        (lambda a, b: cordic.evaluate(1, a, b), 2),
        (lambda a, b: cordic.evaluate(0, a, b, oracle=True), 2),
        (lambda a, b: cordic.evaluate(1, a, b, oracle=True), 2),
        (lambda a, b: divsqrt.evaluate(0, a, b), 2),
        (lambda a, b: divsqrt.evaluate(1, a, b), 2),
        (lambda a, b: root.evaluate(0, a, b), 2),
        (lambda a, b: rotation.evaluate(1, a, b), 2),
    ]
    for evaluate, arity in evaluations:
        evaluate(*[fmt.encode(3)] * arity)
        for at in range(arity):
            with pytest.raises(ValueError):
                evaluate(*[other.encode(3) if k == at else fmt.encode(3) for k in range(arity)])
    resize = Resize(fmt, fmt_in=other)
    assert resize.evaluate(other.encode(3)) == fmt.encode(3)
    with pytest.raises(ValueError):
        resize.evaluate(fmt.encode(3))


def test_an_integer_that_its_port_does_not_hold_is_refused() -> None:
    """One at either end of the port is taken; one beyond either is refused."""
    fmt = ZkfFormat(6, 18)
    one = fmt.encode(1)
    from_int, mul_ilog2, pipe = FromInt(fmt, wint=9), MulIlog2(fmt, wk=5), Pipe(w=5)
    cordic, divsqrt, rint = Cordic(fmt, stage_core_mul=3, wmultiplier=18), Divsqrt(fmt), Rint(fmt)
    ports: list[tuple[Callable[[int], object], range]] = [
        (from_int.evaluate, range(-256, 256)),
        (lambda k: mul_ilog2.evaluate(one, k), range(-16, 16)),
        (pipe.evaluate, range(32)),
        (lambda vectoring: cordic.evaluate(vectoring, one, one), range(2)),
        (lambda op_sqrt: divsqrt.evaluate(op_sqrt, one, one), range(2)),
        (lambda round_mode: rint.evaluate(one, round_mode), range(4)),  # type: ignore[arg-type]
    ]
    for evaluate, held in ports:
        evaluate(held[0])
        evaluate(held[-1])
        for beyond in (held[0] - 1, held[-1] + 1):
            with pytest.raises(ValueError):
                evaluate(beyond)
    assert (pipe.evaluate(0), pipe.evaluate(31)) == (0, 31)


def test_a_port_that_a_fixed_mode_ignores() -> None:
    """
    `op_sqrt` and `vectoring` at a fixed mode, and `b` of a root and of a rotation: what the port cannot hold is
    refused, and what it holds changes no result.
    """
    fmt = ZkfFormat(6, 18)
    a, b, c = fmt.encode(Fraction(3, 8)), fmt.encode(5), fmt.encode(-7)
    for mode in (0, 1):
        for fixed, either in (
            (Divsqrt(fmt, mode=mode), Divsqrt(fmt)),
            (Cordic(fmt, stage_core_mul=3, wmultiplier=18, mode=mode), Cordic(fmt, stage_core_mul=3, wmultiplier=18)),
        ):
            assert fixed.evaluate(0, a, b) == fixed.evaluate(1, a, b) == either.evaluate(mode, a, b)
            with pytest.raises(ValueError):
                fixed.evaluate(2, a, b)
    divsqrt, cordic = Divsqrt(fmt), Cordic(fmt, stage_core_mul=3, wmultiplier=18)
    assert divsqrt.evaluate(1, a, b) == divsqrt.evaluate(1, a, c)
    assert cordic.evaluate(0, a, b) == cordic.evaluate(0, a, c)
    assert cordic.evaluate(0, a, b, oracle=True) == cordic.evaluate(0, a, c, oracle=True)


def test_the_oracles() -> None:
    """
    `oracle=True` of `exp2`, `log2` and both modes of `cordic` at 8/24, rounded: the special operands, and others whose
    results are exact or those of IEEE, the angle of half a turn folded onto the positive one among them. Unrounded,
    the values are of more bits than the format's, and beyond its range.
    """
    fmt = BINARY32
    w, n = partial(Zkf, fmt), fmt.encode
    one, half, zero, inf, minus_inf = n(1), n(Fraction(1, 2)), Zkf(fmt, 0), fmt.inf(0), fmt.inf(1)
    minus_zero, root2, root_half, minus_root_half = w(1 << 31 | 5), w(0x3FB504F3), w(0x3F3504F3), w(0xBF3504F3)
    exp2, log2, cordic = (
        Exp2(fmt, stage_core_kernel=3, wmultiplier=18),
        Log2(fmt, stage_core_kernel=3, stage_core_mul=3, wmultiplier=18),
        Cordic(fmt, stage_core_mul=3, wmultiplier=18),
    )
    for x, y in (
        (inf, inf),
        (minus_inf, zero),
        (minus_zero, one),
        (n(128), inf),
        (n(-128), zero),
        (n(3), n(8)),
        (n(-1), half),
        (half, root2),
        (n(Fraction(-1, 4)), w(0x3F5744FD)),
        (n(-127), w(0x00800000)),  # half the least normal, which rounds up
    ):
        assert fmt.encode(exp2.evaluate(x, oracle=True)) == y, x
    for x, flagged in (
        (inf, (inf, False, False)),
        (minus_zero, (minus_inf, False, True)),
        (minus_inf, (minus_inf, True, False)),
        (n(-3), (minus_inf, True, False)),
        (one, (zero, False, False)),
        (n(8), (n(3), False, False)),
        (n(Fraction(1, 4)), (n(-2), False, False)),
        (n(3), (w(0x3FCAE00D), False, False)),
        (n(10), (w(0x40549A78), False, False)),
    ):
        value, *flags = log2.evaluate(x, oracle=True)
        assert (fmt.encode(value), *flags) == flagged, x
    for x, turned in (
        (minus_zero, (zero, one, 0)),
        (inf, (inf, inf, 0)),
        (minus_inf, (minus_inf, minus_inf, 0)),
        (n(Fraction(1, 4)), (one, zero, 1)),
        (n(Fraction(1, 2)), (zero, n(-1), 2)),
        (n(Fraction(-1, 4)), (n(-1), zero, 3)),
        (n(Fraction(1, 8)), (root_half, root_half, 0)),
        (n(Fraction(1, 32)), (w(0x3E47C5C2), w(0x3F7B14BE), 0)),  # below an eighth of a turn
        (n(Fraction(-1, 32)), (w(0xBE47C5C2), w(0x3F7B14BE), 3)),
        (n(Fraction(-13, 16)), (w(0x3F6C835E), w(0x3EC3EF15), 0)),  # 3/16 of a turn, past the middle of its quadrant
        (n(Fraction(-3, 8)), (minus_root_half, minus_root_half, 2)),
        (n(1 << 30), (zero, one, 0)),
    ):
        sin, cos, quadrant = cordic.evaluate(0, x, one, oracle=True)
        assert (fmt.encode(sin), fmt.encode(cos), quadrant) == turned, x
    for y, x, angled in (
        (zero, zero, (zero, zero, 0)),
        (one, one, (n(Fraction(1, 8)), root2, 0)),
        (one, n(-1), (n(Fraction(3, 8)), root2, 0)),
        (n(-1), n(-1), (n(Fraction(-3, 8)), root2, 0)),
        (minus_zero, n(-1), (half, one, 0)),
        (w(0x80800000), n(-1), (half, one, 0)),  # -1/2 once rounded
        (inf, one, (n(Fraction(1, 4)), inf, 0)),
        (minus_inf, minus_inf, (n(Fraction(-3, 8)), inf, 0)),
        (n(3), n(4), (w(0x3DD1BFAE), n(5), 0)),
    ):
        theta, length, quadrant = cordic.evaluate(1, y, x, oracle=True)
        assert (Zkf(fmt, _atan2_canon_half(fmt, fmt.encode(theta).bits)), fmt.encode(length), quadrant) == angled, (
            y,
            x,
        )
    assert abs(exp2.evaluate(half, oracle=True) ** 2 - 2) < mpmath.mpf(2) ** -150
    assert exp2.evaluate(n(128), oracle=True) == mpmath.mpf(2) ** 128


def test_the_oracles_do_not_read_the_precision_of_the_caller() -> None:
    fmt = ZkfFormat(8, 36)
    exp2, log2, cordic = (
        Exp2(fmt, stage_core_kernel=3, wmultiplier=18),
        Log2(fmt, stage_core_kernel=3, stage_core_mul=3, wmultiplier=18),
        Cordic(fmt, stage_core_mul=3, wmultiplier=18),
    )
    x, one, near = fmt.encode(Fraction(1, 10)), fmt.encode(1), fmt.pack(0, fmt.bias, 1)

    def results() -> tuple[object, ...]:
        turned = cordic.evaluate(0, x, one, oracle=True), cordic.evaluate(1, near, one, oracle=True)
        return exp2.evaluate(x, oracle=True), log2.evaluate(near, oracle=True), *turned

    expected = results()
    with mpmath.workprec(24):
        assert results() == expected

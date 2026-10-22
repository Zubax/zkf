"""The certificate of a table of `exp2` or `log2`: user code does not run it, the tests do."""

from collections.abc import Sequence
from fractions import Fraction
from math import comb
from typing import Any

import mpmath
from mpmath.ctx_iv import MPIntervalContext

from zkf._log2 import _G2, _GB
from zkf._trans import _ERR_GUARD, _GUARD, Spec, _arith_range

PREC = 400
"""Bits: the Taylor coefficients of `log2`'s kernel lose about `k` of them per order to the division by `f`."""
PIECES = 8
"""Of the Bernstein subdivision of a segment."""

# Contexts of its own: the precision of the caller is neither read nor changed.
mp = mpmath.mp.clone()
iv = MPIntervalContext()
mp.prec = iv.prec = PREC


class CertificateError(Exception):
    pass


def certify_segment(spec: Spec, segment: int) -> dict[str, Any]:
    """
    The bounds of one segment, as upper ends of intervals: the error of the kernel relative to the helper, `rel`; that
    of the operator in ULP beyond the half of its rounding, `ulp`; the range of the kernel in units of `2**-cf`,
    `smin` to `smax`; and for `log2` the largest `|log2(m')|`, `lmax`.
    """
    cf, wman = spec.cf, spec.wman
    arith_lo, arith_hi = _arith_range(spec, spec.coeffs[segment])
    error = _fit_range(spec, segment) - iv.mpf([_lo(_iv(arith_lo)), _hi(_iv(arith_hi))])  # S - T
    a, h = _segment(spec, segment)
    ln2 = iv.log(iv.mpf(2))
    scale = iv.mpf(1 << cf)
    if spec.function == "exp2":
        helper = iv.mpf([_lo(iv.exp(_iv(a) * ln2)), _hi(iv.exp(_iv(a + h) * ln2))]) * scale
        # The reduced argument is low by 2**-cf at most: the target is up to ln 2 * 2**f of an LSB higher. The
        # argument may then lie just past the end of the segment, which the factor covers.
        error = error - iv.mpf([0, 1]) * ln2 * (_hi(helper) / scale) * (1 + iv.mpf(2) ** -(cf - 2))
    else:
        ends = [iv.log(1 + _iv(x)) / (_iv(x) * ln2) if x else 1 / ln2 for x in (a, a + h)]
        helper = iv.mpf([_lo(ends[1]), _hi(ends[0])]) * scale  # C decreases
        lmax = max(_hi(abs(iv.log(1 + _iv(x)) / ln2)) for x in (a, a + h))
    rel = abs(error) / _lo(helper)
    out = {"rel": _hi(rel), "smin": _lo(helper + error), "smax": _hi(helper + error)}
    if spec.function == "exp2":
        out["ulp"] = _hi(abs(error) / (1 << (cf - wman + 1)))  # an ULP of the significand is 2**(cf - wfrac)
    else:
        # e = 0: the result is l = |f| * C, whose relative error is the kernel's, rel * 2**wman ULP at most.
        # e != 0: |e| - l >= 1 - lmax; below 1/2 (lmax above it, next to the moved threshold) the ULP halves again.
        apart = rel * lmax * iv.mpf(2) ** (wman + 1 if lmax > mp.mpf(1) / 2 else wman)
        # The shifted C drops under 2**-(GUARD + _G2) ULP of the lower candidate, the last sum under 2**-(_GB + 1).
        cuts = mp.mpf(2) ** -(_GUARD + _G2) + mp.mpf(2) ** -(_GB + 1)
        out["ulp"] = max(_hi(rel * iv.mpf(2) ** wman), _hi(apart)) + cuts
        out["lmax"] = lmax
    return out


def gates(spec: Spec, worst: dict[str, Any]) -> None:
    """The budget, and every range the result assembly assumes, on the worst bounds over the segments."""
    cf, wman = spec.cf, spec.wman
    if not worst["rel"] < mp.mpf(2) ** -(wman + _ERR_GUARD):
        raise CertificateError(f"helper error 2^{float(mp.log(worst['rel'], 2)):.3f} exceeds 2^-{wman + _ERR_GUARD}")
    if not worst["ulp"] <= mp.mpf(2) ** -_ERR_GUARD:
        raise CertificateError(f"result error {float(worst['ulp']):.6f} ULP exceeds 2^-{_ERR_GUARD}")
    if spec.function == "exp2":
        # The significand is cut out of a sum with its hidden bit, 2**cf at least, and one carry bit above it.
        half = 1 << (cf - wman)
        if not (worst["smin"] + half >= 1 << cf and worst["smax"] + half < 1 << (cf + 2)):
            raise CertificateError("exp2 sum leaves [2**cf, 2**(cf+2))")
        # A sum of 2 or more is the round-up to the next binade, whose fraction bits are taken as they are.
        if not worst["smax"] + half < (1 << (cf + 1)) + 2 * half:
            raise CertificateError("exp2 round-up with a nonzero fraction")
    else:
        if not (worst["smin"] >= 1 << cf and worst["smax"] < 1 << (cf + 1)):
            raise CertificateError("log2 kernel leaves [2**cf, 2**(cf+1))")
        # The predicted leading one of the result (1 at a fixed bit; candidates in [1, 2) and in [1/2, 1)): e = 0
        # gives |f| normalized, in [1/2, 1), times C; |e| = 1 less l gives 2 - 2l; the other differences are 1 - l/2
        # at least. The half ULP added before the selection is far below the slack demanded here.
        low, high = worst["smin"] / (1 << cf), worst["smax"] / (1 << cf)
        if not (low >= mp.mpf("1.01") and high <= mp.mpf("1.99") and worst["lmax"] <= mp.mpf("0.74")):
            raise CertificateError("log2 leading-one prediction")


def certify(spec: Spec) -> dict[str, Any]:
    """
    Proves a table, from its integers alone: that the kernel is within the error budget of its helper for every
    argument, and that every range the result assembly relies on holds. Returns the worst bounds, or raises
    `CertificateError`.

    Per segment the result of the kernel, `S(w)`, is compared with its target `T` (`exp2`: `2**cf * 2**f` plus half
    an ULP of the significand; `log2`: `2**cf * C(f)`, `C(f) = log2(1 + f) / f`) in two rigorous steps:

      fit         the range of `P(t) - T(t)` over the closed segment, `P` the polynomial of the stored coefficients.
                  `T` is replaced by a Taylor polynomial about the center of the segment plus a bound of its
                  remainder, and the difference is enclosed by its Bernstein coefficients on a subdivision, in
                  interval arithmetic.
      arithmetic  the range of `P(w) - S(w)`, through every truncated operand and floored product.

    Their sum, with the truncation of the argument of `exp2` and the product bits that `log2` drops, bounds the
    error.

    The ranges proven are those the result assemblies assume: the sum of `exp2` in `[2**cf, 2**(cf + 2))` with a
    zero fraction from `2**(cf + 1)` up, which the budget alone takes the last argument to (that of a negative
    integer, an LSB or two short of 2, the half ULP added); the kernel of `log2` in `[2**cf, 2**(cf + 1))`, and
    `|log2(m')| <= 0.74`.
    """
    _check_shape(spec)
    segments = [certify_segment(spec, segment) for segment in range(len(spec.coeffs))]
    worst = {key: max(s[key] for s in segments) for key in segments[0] if key != "smin"}
    worst["smin"] = min(s["smin"] for s in segments)
    gates(spec, worst)
    return worst


def _lo(x: Any) -> Any:
    return mp.mpf(x.a)


def _hi(x: Any) -> Any:
    return mp.mpf(x.b)


def _iv(x: Fraction) -> Any:
    return iv.mpf(x.numerator) / x.denominator


def _symmetric(x: Any) -> Any:
    """`[-max|x|, +max|x|]` of a non-negative interval."""
    return iv.mpf([-_hi(x), _hi(x)])


def _taylor_exp2(a: Fraction, h: Fraction, order: int) -> tuple[list[Any], Any]:
    """`2**(a + h*t)`, `t` in [0, 1], as the sum of `g[n] * u**n` in `u = t - 1/2`, and an interval of the remainder."""
    ln2 = iv.log(iv.mpf(2))
    step = _iv(h) * ln2
    g, term = [], iv.exp(_iv(a + h / 2) * ln2)
    for n in range(order + 1):
        g.append(term)
        term = term * step / (n + 1)
    # Lagrange: the derivative of the order above peaks at the right end.
    top = iv.exp(_iv(a + h) * ln2) * step ** (order + 1)
    for n in range(2, order + 2):
        top = top / n
    return g, _symmetric(abs(top) / iv.mpf(2) ** (order + 1))


def _taylor_log2(a: Fraction, h: Fraction, order: int) -> tuple[list[Any], Any]:
    """
    `C(a + h*t)` likewise. The coefficients come from exact series arithmetic about the center `f0`, which is not
    zero: the series of `log(1 + f0 + s)` divided by `f0 + s`. The remainder is a Cauchy estimate on the disc
    `|z - f0| <= 1/4`: `C(z)` is the sum of `(-z)**k / ((k + 1) ln 2)` for `|z| < 1`, which defines it across the
    removable singularity at 0 too, so `|C(z)| <= 1 / (ln 2 * (1 - |z|))`.
    """
    ln2 = iv.log(iv.mpf(2))
    f0 = _iv(a + h / 2)
    if a <= 0 <= a + h and a + h / 2 == 0:
        raise CertificateError("segment centered on f = 0")
    numerator = [iv.log(1 + f0)]
    r, power = 1 / (1 + f0), iv.mpf(1)
    for n in range(1, order + 1):
        power = power * r
        numerator.append(power / n if n % 2 else -power / n)
    quotient: list[Any] = []
    for n in range(order + 1):
        quotient.append((numerator[n] - (quotient[n - 1] if n else 0)) / f0)
    span = _iv(h)
    g = [quotient[n] / ln2 * span**n for n in range(order + 1)]
    rho = iv.mpf(1) / 4
    furthest = abs(f0) + rho
    ratio = abs(span) / 2 / rho
    if not (_hi(furthest) < 1 and _hi(ratio) < 1):
        raise CertificateError("segment too far from 0 for the kernel's power series")
    return g, _symmetric(ratio ** (order + 1) / (1 - ratio) / ((1 - furthest) * ln2))


def _poly_range(p: Sequence[Any], pieces: int) -> Any:
    """An interval that holds `p(u)` over `u` in [-1/2, 1/2], from the Bernstein coefficients on equal pieces of it."""
    n = len(p) - 1
    bounds: list[Any] = []
    width = iv.mpf(1) / pieces
    for piece in range(pieces):
        q = list(p)
        left = iv.mpf(-1) / 2 + width * piece
        for i in range(n):  # a Taylor shift to the left end
            for j in range(n - 1, i - 1, -1):
                q[j] = q[j] + left * q[j + 1]
        scale = iv.mpf(1)
        for i in range(n + 1):  # the piece scaled to [0, 1]
            q[i] = q[i] * scale
            scale = scale * width
        for m in range(n + 1):
            b = iv.mpf(0)
            for i in range(m + 1):
                b = b + q[i] * comb(m, i) / comb(n, i)
            bounds += [_lo(b), _hi(b)]
    return iv.mpf([min(bounds), max(bounds)])


def _segment(spec: Spec, segment: int) -> tuple[Fraction, Fraction]:
    """The helper's argument at the left end of a segment, and the span of the segment in it."""
    if spec.function == "log2" and segment >= spec.thr:  # re-centered: f = (frac / 2**WFRAC - 1) / 2
        return (Fraction(segment, 1 << spec.k) - 1) / 2, Fraction(1, 1 << (spec.k + 1))
    return Fraction(segment, 1 << spec.k), Fraction(1, 1 << spec.k)


def _fit_range(spec: Spec, segment: int) -> Any:
    """An interval that holds `P(t) - T(t)` over the closed segment, in units of `2**-cf`."""
    log2 = spec.function == "log2"
    a, h = _segment(spec, segment)
    order = spec.d + 8
    g, remainder = (_taylor_log2 if log2 else _taylor_exp2)(a, h, order)
    scale = iv.mpf(1 << spec.cf)
    half = iv.mpf(1) / 2
    difference = [-(g[i] * scale) for i in range(order + 1)]
    for j, c in enumerate(spec.coeffs[segment]):
        cj = iv.mpf(-c if log2 and j % 2 else c)
        for i in range(j + 1):  # (u + 1/2)**j
            difference[i] = difference[i] + cj * comb(j, i) * half ** (j - i)
    if not log2:
        difference[0] = difference[0] - (1 << (spec.cf - spec.wman))  # the half ULP in c0 belongs to the target
    return _poly_range(difference, PIECES) + remainder * scale


def _check_shape(spec: Spec) -> None:
    d, k, rw, cf, rows = spec.d, spec.k, spec.rw, spec.cf, spec.coeffs
    if cf != spec.wman + _GUARD or len(rows) != 1 << k or any(len(row) != d + 1 for row in rows):
        raise CertificateError("table shape")
    if any(c <= 0 for row in rows for c in row):
        raise CertificateError("non-positive coefficient magnitude")
    if len(spec.t) != (d + 1) // 2 or len(spec.q) != d // 2:
        raise CertificateError("operand width lists")
    if spec.function == "exp2" and k + rw != cf:
        raise CertificateError("exp2 argument width")
    if spec.function == "log2" and (k + rw != spec.wman - 1 or not 0 < spec.thr < 1 << k):
        raise CertificateError("log2 argument width or threshold")

"""
Independent reference values for verifying the ZKF model. Two families:

- correctly-rounded (ties-to-even) transcendentals via mpmath: exp2, log2, sincos, atan2;
- IEEE-754 cross-checks via the hardware FPU: add, mul, div (binary32 and binary64) and fma (binary64 only).

The transcendentals run at ~4*WMAN mpmath working precision (correct last-bit rounding even at large WMAN) and prove the
model's faithful-rounding (<=1 ULP) contract; the IEEE checks catch model bugs for the two formats where ZKF coincides
with IEEE (WEXP/WMAN = 8/24 and 11/53). The IEEE functions return None where ZKF and IEEE diverge and no independent
cross-check exists (non-IEEE format, non-canonical inf/zero, a subnormal result, or an IEEE-NaN operation -- inf-inf,
0*inf, 0/0, inf/inf -- which ZKF instead defines as +0).

Multi-operand functions require both operands to share a format (a mismatch raises ValueError, as in the model).

Importing this submodule requires numpy and mpmath.
"""

import math as _math
from fractions import Fraction as _Fraction

import mpmath as _mpmath
import numpy as _np

from ._core import (
    Atan2Result,
    DivResult,
    Log2Result,
    SinCos,
    Zkf,
    ZkfFormat,
    atan2_canon_half,
    atan2_special,
)


def exp2(z: Zkf) -> Zkf:
    """Correctly-rounded 2**z."""
    fmt = z.fmt
    if z.is_inf:
        return fmt.inf(0) if not z.negative else fmt.zero()  # +inf -> +inf, -inf -> +0
    if z.is_zero:
        return fmt.normal(0, fmt.bias, 0)  # 2**0 = 1.0
    e = z.exp - fmt.bias
    if e >= fmt.wexp - 1:  # mirror the model's out-of-range classification
        return fmt.inf(0) if not z.negative else fmt.zero()
    with _mpmath.workprec(4 * fmt.wman + 80):
        return _round_mpf(fmt, _mpmath.power(2, _to_mpf(z)))


def log2(z: Zkf) -> Log2Result:
    """Correctly-rounded log2(z), plus the domain-error (z<0) and pole (z==0) flags."""
    fmt = z.fmt
    if z.is_inf and not z.negative:
        return Log2Result(fmt.inf(0), False, False)  # log2(+inf) = +inf
    if z.is_zero:
        return Log2Result(fmt.inf(1), False, True)  # log2(+0) = -inf, pole
    if z.negative:
        return Log2Result(fmt.inf(1), True, False)  # log2(x<0) = -inf, domain error
    with _mpmath.workprec(4 * fmt.wman + 80):
        return Log2Result(_round_mpf(fmt, _mpmath.log(_to_mpf(z), 2)), False, False)


def sincos(z: Zkf) -> SinCos:
    """
    Correctly-rounded (sin(2*pi*z), cos(2*pi*z), quadrant).

    The phase is reduced mod 1 exactly with integer arithmetic before the transcendental, so the periodic identity holds
    without the large-argument cancellation of a direct sin(2*pi*x); the local angle is folded into one octant so mpmath
    only ever evaluates an angle <= pi/4, keeping the quadrant-boundary zeros exact.
    """
    fmt = z.fmt
    if z.is_inf:
        s = fmt.inf(z.negative)
        return SinCos(s, s, 0)
    if z.is_zero:
        return SinCos(fmt.zero(), fmt.normal(0, fmt.bias, 0), 0)  # sin(0)=+0, cos(0)=+1

    e = z.exp - fmt.bias
    rsh = fmt.wfrac - e  # |x| = sig / 2**rsh
    frac_abs = _Fraction(0) if rsh <= 0 else _Fraction(z.significand() % (1 << rsh), 1 << rsh)
    frac = (1 - frac_abs) if (z.negative and frac_abs != 0) else frac_abs  # frac(x) in [0,1)
    q4 = frac * 4
    quadrant = int(q4)  # floor (q4 in [0,4)); already in {0,1,2,3}
    t_local = q4 - quadrant  # in [0,1)
    with _mpmath.workprec(4 * fmt.wman + 80):
        if t_local <= _Fraction(1, 2):
            theta = (_mpmath.pi / 2) * _mpmath.mpf(t_local.numerator) / _mpmath.mpf(t_local.denominator)
            s0, c0 = _mpmath.sin(theta), _mpmath.cos(theta)
        else:
            comp = 1 - t_local
            theta = (_mpmath.pi / 2) * _mpmath.mpf(comp.numerator) / _mpmath.mpf(comp.denominator)
            s0, c0 = _mpmath.cos(theta), _mpmath.sin(theta)  # sin(pi/2-theta)=cos, cos(pi/2-theta)=sin
        sin_mag, cos_mag = (c0, s0) if (quadrant & 1) else (s0, c0)
        sin_v = -sin_mag if (quadrant >> 1) & 1 else sin_mag
        cos_v = -cos_mag if ((quadrant >> 1) ^ quadrant) & 1 else cos_mag
        return SinCos(_round_mpf(fmt, sin_v), _round_mpf(fmt, cos_v), quadrant)


def atan2(y: Zkf, x: Zkf) -> Atan2Result:
    """
    Correctly-rounded (theta, magnitude) of atan2(y, x). Shares the exact special-case table with the model; the
    generic path evaluates mpmath atan2/hypot at high precision (bounded ratios, no cancellation) and rounds.
    """
    _require_same(y, x)
    fmt = y.fmt
    special = atan2_special(fmt, y.bits, x.bits)
    if special is not None:
        theta_bits, mag_bits = special
        return Atan2Result(fmt.wrap(theta_bits), fmt.wrap(mag_bits))
    with _mpmath.workprec(4 * fmt.wman + 80):
        yv = _to_mpf(y)
        xv = _to_mpf(x)
        theta = _mpmath.atan2(yv, xv) / (2 * _mpmath.pi)  # turns, (-0.5, 0.5]
        theta_bits = atan2_canon_half(fmt, _round_mpf(fmt, theta).bits)
        return Atan2Result(fmt.wrap(theta_bits), _round_mpf(fmt, _mpmath.hypot(yv, xv)))


def mul(a: Zkf, b: Zkf) -> Zkf | None:
    """IEEE a*b, or None if the format/operands are not IEEE-mappable."""
    _require_same(a, b)
    dtype = _dtype(a.fmt)
    if dtype is None or not _is_ieee_canonical(a) or not _is_ieee_canonical(b):
        return None
    with _np.errstate(all="ignore"):
        result = _to_np(a.bits, dtype) * _to_np(b.bits, dtype)
    if _np.isnan(result):
        return None  # 0 * inf is IEEE NaN -> not cross-checkable
    raw = a.fmt.wrap(_from_np(result, dtype))
    return None if _underflowed_to_subnormal(raw) else _canonicalize(raw)


def add(a: Zkf, b: Zkf) -> Zkf | None:
    """IEEE a+b, or None if the format/operands are not IEEE-mappable."""
    _require_same(a, b)
    fmt = a.fmt
    dtype = _dtype(fmt)
    if dtype is None or not _is_ieee_canonical(a) or not _is_ieee_canonical(b):
        return None
    if a.is_inf and b.is_inf:
        return (
            fmt.inf(a.negative) if a.negative == b.negative else None
        )  # inf + -inf is IEEE NaN -> not cross-checkable
    if a.is_inf:
        return fmt.inf(a.negative)
    if b.is_inf:
        return fmt.inf(b.negative)
    with _np.errstate(all="ignore"):
        result = _to_np(a.bits, dtype) + _to_np(b.bits, dtype)
    raw = fmt.wrap(_from_np(result, dtype))
    return None if _underflowed_to_subnormal(raw) else _canonicalize(raw)


def div(a: Zkf, b: Zkf) -> DivResult | None:
    """IEEE a/b with the div-by-zero flag, or None if the format/operands are not IEEE-mappable."""
    _require_same(a, b)
    fmt = a.fmt
    dtype = _dtype(fmt)
    if dtype is None or not _is_ieee_canonical(a) or not _is_ieee_canonical(b):
        return None
    div0 = b.is_zero
    if (a.is_zero and b.is_zero) or (a.is_inf and b.is_inf):
        return None  # 0/0 and inf/inf are IEEE NaN -> not cross-checkable
    if a.is_zero or b.is_inf:
        return DivResult(fmt.zero(), div0)
    result_sign = a.negative if b.is_zero else (a.negative ^ b.negative)
    if b.is_zero or a.is_inf:
        return DivResult(fmt.inf(result_sign), div0)
    with _np.errstate(all="ignore"):
        result = _to_np(a.bits, dtype) / _to_np(b.bits, dtype)
    raw = fmt.wrap(_from_np(result, dtype))
    return None if _underflowed_to_subnormal(raw) else DivResult(_canonicalize(raw), div0)


def fma(a: Zkf, b: Zkf, c: Zkf) -> Zkf | None:
    """
    IEEE fused a*b + c (single rounding), or None if not applicable.

    math.fma is a correctly-rounded single-rounding FMA on binary64, so it is an exact oracle only for the (11, 53)
    format and finite canonical operands (inf operands are skipped: IEEE 0*inf -> NaN differs from ZKF); a fused
    overflow cross-checks to signed infinity. Requires Python 3.13+.
    """
    _require_same(a, b, c)
    if not hasattr(_math, "fma"):  # math.fma is Python 3.13+
        return None
    fmt = a.fmt
    if _dtype(fmt) is not _np.float64:
        return None
    if not all(_is_ieee_canonical(z) for z in (a, b, c)):
        return None
    if a.is_inf or b.is_inf or c.is_inf:
        return None
    try:
        result = _math.fma(
            float(_to_np(a.bits, _np.float64)),
            float(_to_np(b.bits, _np.float64)),
            float(_to_np(c.bits, _np.float64)),
        )
    except ValueError:
        return None  # invalid operation (NaN domain) -> not cross-checkable
    except OverflowError:  # fused result overflows: signed IEEE infinity
        exact = a.to_fraction() * b.to_fraction() + c.to_fraction()  # exact operands -> independent of the fma model
        return fmt.inf(1 if exact < 0 else 0)
    if _math.isnan(result):
        return None
    raw = fmt.wrap(_from_np(result, _np.float64))
    return None if _underflowed_to_subnormal(raw) else _canonicalize(raw)


def _require_same(*operands: Zkf) -> None:
    """Reject operands that do not share a format, matching the model's arithmetic contract."""
    fmt = operands[0].fmt
    for other in operands[1:]:
        if other.fmt != fmt:
            raise ValueError(f"operands have different formats: {fmt} vs {other.fmt}")


def _mpf_to_fraction(x) -> _Fraction:
    """Exact dyadic Fraction of an mpmath mpf (value = man * 2^exp)."""
    sign, man, exp, _bc = _mpmath.mpf(x)._mpf_
    value = _Fraction(int(man)) * (_Fraction(2) ** int(exp))
    return -value if sign else value


def _round_mpf(fmt: ZkfFormat, v) -> Zkf:
    """Round an mpmath value to the nearest ZKF value (ties-to-even); exact zero -> +0."""
    return fmt.encode(_mpf_to_fraction(v))


def _to_mpf(z: Zkf):
    """Exact signed mpf of a finite Zkf (must be called inside an ample mpmath.workprec context)."""
    v = _mpmath.mpf(z.significand()) * _mpmath.power(2, (z.exp - z.fmt.bias) - z.fmt.wfrac)
    return -v if z.negative else v


def _dtype(fmt: ZkfFormat):
    if (fmt.wexp, fmt.wman) == (8, 24):
        return _np.float32
    if (fmt.wexp, fmt.wman) == (11, 53):
        return _np.float64
    return None


def _is_ieee_canonical(z: Zkf) -> bool:
    """
    A ZKF operand whose bit pattern is a valid canonical IEEE value: the single canonical +0 (ZKF has no signed
    zero, so -0 diverges from IEEE and is rejected), or a sign-carrying infinity, both with no fraction payload.
    """
    if z.exp == 0:
        return not z.negative and z.frac == 0
    if z.exp == z.fmt.exp_inf:
        return z.frac == 0
    return True


def _to_np(bits: int, dtype):
    if dtype is _np.float32:
        return _np.array([bits], dtype=_np.uint32).view(_np.float32)[0]
    return _np.array([bits], dtype=_np.uint64).view(_np.float64)[0]


def _from_np(value, dtype) -> int:
    if dtype is _np.float32:
        return int(_np.array([value], dtype=_np.float32).view(_np.uint32)[0])
    return int(_np.array([value], dtype=_np.float64).view(_np.uint64)[0])


def _underflowed_to_subnormal(raw: Zkf) -> bool:
    """
    True if the raw IEEE result is a (nonzero) subnormal.

    ZKF rounds the exact result against the 0.5*MIN_NORMAL flush boundary, whereas the FPU first rounds to the nearest
    subnormal (gradual underflow) which we then canonicalize -- two roundings that can disagree within ~1 ULP of the
    boundary. So the numpy result is not a valid oracle once it underflows to a subnormal, and callers skip it there
    (the exact model still defines the value; the boundary is covered by the exhaustive small-format sims and proofs).
    """
    return raw.exp == 0 and raw.frac != 0


def _canonicalize(raw: Zkf) -> Zkf:
    """
    Canonicalize a raw IEEE result: +0 for zero, sign-preserving infinity, finite passthrough. Callers filter out
    subnormals (_underflowed_to_subnormal -> None) and NaNs (isnan / early inf handling) before this, so raw here is
    never a subnormal or NaN.
    """
    fmt = raw.fmt
    if raw.exp == 0:
        return fmt.zero()
    if raw.exp == fmt.exp_inf:
        return fmt.inf(raw.negative)
    return raw

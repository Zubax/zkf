"""
IEEE 754 as the FPU of the machine computes it, through NumPy: `add`, `mul`, `div` and `sqrt` on the two formats that
coincide with it, 8/24 and 11/53, and `fma` at 11/53. Each gives None where ZKF and IEEE part and there is nothing to
hold a result to: another format, a zero or an infinity that is not canonical, a subnormal result, or an operation that
IEEE answers with a NaN (inf - inf, 0 * inf, 0 / 0, inf / inf), which ZKF defines as +0.
"""

import math
from typing import Any

import numpy as np

from zkf import DivsqrtResult, Zkf, ZkfFormat

BINARY32, BINARY64 = ZkfFormat(8, 24), ZkfFormat(11, 53)
IEEE: dict[ZkfFormat, tuple[Any, Any]] = {BINARY32: (np.float32, np.uint32), BINARY64: (np.float64, np.uint64)}


def to_numpy(fmt: ZkfFormat, bits: int) -> Any:
    floating, integer = IEEE[fmt]
    return np.array([bits], dtype=integer).view(floating)[0]


def of_numpy(fmt: ZkfFormat, value: Any) -> int:
    floating, integer = IEEE[fmt]
    return int(np.array([value], dtype=floating).view(integer)[0])


def mul(a: Zkf, b: Zkf) -> Zkf | None:
    fmt = a.fmt
    if fmt not in IEEE or not _is_ieee_canonical(a) or not _is_ieee_canonical(b):
        return None
    with np.errstate(all="ignore"):
        result = to_numpy(fmt, a.bits) * to_numpy(fmt, b.bits)
    if np.isnan(result):
        return None  # 0 * inf
    raw = Zkf(fmt, of_numpy(fmt, result))
    return None if _underflowed_to_subnormal(raw) else raw.canonicalize()


def add(a: Zkf, b: Zkf) -> Zkf | None:
    fmt = a.fmt
    if fmt not in IEEE or not _is_ieee_canonical(a) or not _is_ieee_canonical(b):
        return None
    if a.is_inf and b.is_inf:
        return fmt.inf(a.negative) if a.negative == b.negative else None  # inf - inf
    if a.is_inf:
        return fmt.inf(a.negative)
    if b.is_inf:
        return fmt.inf(b.negative)
    with np.errstate(all="ignore"):
        result = to_numpy(fmt, a.bits) + to_numpy(fmt, b.bits)
    raw = Zkf(fmt, of_numpy(fmt, result))
    return None if _underflowed_to_subnormal(raw) else raw.canonicalize()


def div(a: Zkf, b: Zkf) -> DivsqrtResult | None:
    """With the flag of a division by zero."""
    fmt = a.fmt
    if fmt not in IEEE or not _is_ieee_canonical(a) or not _is_ieee_canonical(b):
        return None
    div0 = b.is_zero
    if (a.is_zero and b.is_zero) or (a.is_inf and b.is_inf):
        return None  # 0 / 0 and inf / inf
    if a.is_zero or b.is_inf:
        return DivsqrtResult(Zkf(fmt, 0), div0)
    result_sign = a.negative if b.is_zero else (a.negative ^ b.negative)
    if b.is_zero or a.is_inf:
        return DivsqrtResult(fmt.inf(result_sign), div0)
    with np.errstate(all="ignore"):
        result = to_numpy(fmt, a.bits) / to_numpy(fmt, b.bits)
    raw = Zkf(fmt, of_numpy(fmt, result))
    return None if _underflowed_to_subnormal(raw) else DivsqrtResult(raw.canonicalize(), div0)


def sqrt(a: Zkf) -> DivsqrtResult | None:
    """With the flag of a negative radicand, whose root ZKF defines as the negative infinity."""
    fmt = a.fmt
    if fmt not in IEEE or not _is_ieee_canonical(a):
        return None
    if a.is_zero:
        return DivsqrtResult(Zkf(fmt, 0), False)
    if a.negative:
        return DivsqrtResult(fmt.inf(1), True)
    if a.is_inf:
        return DivsqrtResult(fmt.inf(0), False)
    with np.errstate(all="ignore"):
        result = np.sqrt(to_numpy(fmt, a.bits))
    # The root of a positive normal is a positive normal: never subnormal, never overflowing.
    return DivsqrtResult(Zkf(fmt, of_numpy(fmt, result)), False)


def fma(a: Zkf, b: Zkf, c: Zkf) -> Zkf | None:
    """
    `math.fma` rounds once on binary64, so it answers at 11/53 alone, and on finite operands: IEEE's 0 * inf is a NaN.
    A result that overflows is the infinity of its sign.
    """
    fmt = a.fmt
    if fmt != BINARY64 or not all(_is_ieee_canonical(z) and not z.is_inf for z in (a, b, c)):
        return None
    try:
        result = math.fma(*(float(to_numpy(fmt, z.bits)) for z in (a, b, c)))
    except OverflowError:
        exact = a.to_fraction() * b.to_fraction() + c.to_fraction()  # of the operands, not of the model of `fma`
        return fmt.inf(int(exact < 0))
    raw = Zkf(fmt, of_numpy(fmt, result))
    return None if _underflowed_to_subnormal(raw) else raw.canonicalize()


def _is_ieee_canonical(z: Zkf) -> bool:
    """
    Whether the bits are a value of IEEE too: the one +0, since ZKF has no signed zero and -0 parts from IEEE, or an
    infinity of either sign, both without a fraction.
    """
    if z.exp == 0:
        return not z.negative and z.frac == 0
    if z.exp == z.fmt.exp_inf:
        return z.frac == 0
    return True


def _underflowed_to_subnormal(raw: Zkf) -> bool:
    """
    ZKF rounds the exact result against half the least normal, where the FPU rounds first to the nearest subnormal:
    two roundings that can disagree within an ULP of that boundary, so a subnormal of the FPU is no oracle.
    """
    return raw.exp == 0 and raw.frac != 0

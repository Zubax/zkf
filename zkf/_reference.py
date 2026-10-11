from __future__ import annotations

import enum
import functools
import math
from fractions import Fraction


class UnsupportedFormat(ValueError):
    """
    The operation cannot serve this format: WEXP, WMAN, or an integer width such as WINT is out of range, or WMAN has
    no generated table. An invalid tuning knob raises a plain ValueError instead. Support is per operation: an operator
    model refuses what its RTL cannot build, a Zkf method only what the reference does not define (so exp2 at WEXP=31
    computes as a value but has no model).
    """


def bits_to_signed(value: int, width: int) -> int:
    value &= mask(width)
    sign_bit = 1 << (width - 1)
    return value - (1 << width) if value & sign_bit else value


def mask(width: int) -> int:
    return (1 << width) - 1


def pack_bits(fmt: ZkfFormat, sign: int, exp: int, frac: int) -> int:
    return ((sign & 1) << fmt.sign_shift) | ((exp & mask(fmt.wexp)) << fmt.wfrac) | (frac & fmt.frac_mask)


def zero(fmt: ZkfFormat) -> int:
    return pack_bits(fmt, 0, 0, 0)


def canonical_inf(fmt: ZkfFormat, sign: int) -> int:
    return pack_bits(fmt, sign, fmt.exp_inf, 0)


def normal(fmt: ZkfFormat, sign: int, exp: int, frac: int) -> int:
    if not 1 <= exp <= fmt.exp_max_finite:
        raise ValueError(f"normal exponent out of range: {exp}")
    if not 0 <= frac <= fmt.frac_mask:
        raise ValueError(f"fraction out of range: {frac}")
    return pack_bits(fmt, sign, exp, frac)


def significand(fmt: ZkfFormat, bits: int) -> int:
    return (1 << fmt.wfrac) | (bits & fmt.frac_mask)


def pow2_fraction(exp: int) -> Fraction:
    return Fraction(1 << exp, 1) if exp >= 0 else Fraction(1, 1 << -exp)


def floor_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 is defined for positive values only")
    exp = value.numerator.bit_length() - value.denominator.bit_length()
    while pow2_fraction(exp + 1) <= value:
        exp += 1
    while pow2_fraction(exp) > value:
        exp -= 1
    return exp


def round_fraction_to_zkf(fmt: ZkfFormat, sign: int, value: Fraction) -> int:
    if value <= 0:
        return zero(fmt)

    exp_unbiased = floor_log2_fraction(value)
    if exp_unbiased < fmt.min_exp_unbiased:
        return normal(fmt, sign, 1, 0) if value >= pow2_fraction(fmt.min_exp_unbiased - 1) else zero(fmt)

    scaled = value / pow2_fraction(exp_unbiased) * (1 << fmt.wfrac)
    quotient = scaled.numerator // scaled.denominator
    remainder = scaled.numerator % scaled.denominator

    increment = (2 * remainder) > scaled.denominator
    increment = increment or ((2 * remainder) == scaled.denominator and (quotient & 1) != 0)
    if increment:
        quotient += 1

    if quotient >= (1 << fmt.wman):
        quotient >>= 1
        exp_unbiased += 1

    if exp_unbiased > fmt.max_exp_unbiased:
        return canonical_inf(fmt, sign)

    return normal(fmt, sign, exp_unbiased + fmt.bias, quotient & fmt.frac_mask)


def pack_reference(
    fmt: ZkfFormat,
    sign: int,
    force_zero: int,
    force_inf: int,
    exp_unbiased: int,
    significand_value: int,
    guard: int,
    round_bit: int,
    sticky: int,
    *,
    saturate_round_carry: bool = False,
) -> int:
    exp_biased = exp_unbiased + fmt.bias
    exp_underflow_zero = exp_unbiased < (fmt.min_exp_unbiased - 1)
    exp_one_below_min = exp_unbiased == (fmt.min_exp_unbiased - 1)
    exp_overflow = exp_unbiased > fmt.max_exp_unbiased

    round_increment = bool(guard and (round_bit or sticky or (significand_value & 1)))
    rounded_ext = (significand_value & mask(fmt.wman)) + (1 if round_increment else 0)
    round_carry = (rounded_ext >> fmt.wman) & 1
    rounded_significand = (rounded_ext >> 1) if round_carry else (rounded_ext & mask(fmt.wman))
    exp_round_overflow = (exp_biased == fmt.exp_max_finite) and bool(round_carry)
    # saturate_round_carry mirrors _zkf_pack's SATURATE_ROUND_CARRY.
    saturated = saturate_round_carry and exp_round_overflow
    infinity = bool(force_inf or exp_overflow or (exp_round_overflow and not saturate_round_carry))

    result_zero = bool(force_zero or ((not force_inf) and exp_underflow_zero))
    result_infinity = (not result_zero) and infinity
    result_min_normal = (not result_zero) and (not result_infinity) and (not force_inf) and exp_one_below_min

    if result_zero:
        return zero(fmt)
    if result_infinity:
        return canonical_inf(fmt, sign)
    if result_min_normal:
        return normal(fmt, sign, 1, 0)
    if saturated:
        return pack_bits(fmt, sign, fmt.exp_max_finite, fmt.frac_mask)

    exp_rounded = (exp_biased + round_carry) & mask(fmt.wexp)
    return pack_bits(fmt, sign, exp_rounded, rounded_significand & fmt.frac_mask)


def sticky_below(value: int, high_bit: int) -> int:
    if high_bit < 0:
        return 0
    return 1 if (value & mask(high_bit + 1)) != 0 else 0


def canonicalize_special(fmt: ZkfFormat, bits: int) -> int:
    bits &= mask(fmt.wfull)
    exp = (bits >> fmt.wfrac) & fmt.exp_inf
    if exp == 0:
        return zero(fmt)
    if exp == fmt.exp_inf:
        return canonical_inf(fmt, (bits >> fmt.sign_shift) & 1)
    return bits


def ordered_key(fmt: ZkfFormat, bits: int) -> int:
    canonical = canonicalize_special(fmt, bits)
    sign = (canonical >> fmt.sign_shift) & 1
    return (~canonical & mask(fmt.wfull)) if sign else (canonical | (1 << fmt.sign_shift))


def check_format_width(name: str, value: int, min: int, max: int | None = None) -> None:
    """Refuse a format width (WEXP, WMAN, WINT, ...) outside min..max; a non-integer is a plain ValueError."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < min:
        raise UnsupportedFormat(f"{name}={value} is below the minimum {min}")
    if max is not None and value > max:
        raise UnsupportedFormat(f"{name}={value} is above the maximum {max}")


def signed_int_min(wint: int) -> int:
    check_format_width("WINT", wint, 2)
    return -(1 << (wint - 1))


def signed_int_max(wint: int) -> int:
    check_format_width("WINT", wint, 2)
    return (1 << (wint - 1)) - 1


def round_fraction_to_int_ties_even(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    frac_part = value - Fraction(floor, 1)
    half = Fraction(1, 2)
    if frac_part < half:
        return floor
    if frac_part > half:
        return floor + 1
    return floor if (floor % 2 == 0) else floor + 1


@enum.unique
class RoundMode(enum.IntEnum):
    """Round-to-integer modes; the integer values match zkf/rtl/zkf_rint.v and its 2-bit round_mode port."""

    NEAREST_EVEN = 0  # round to nearest integer, ties to even (the IEEE default)
    FLOOR = 1  # round toward -inf
    CEIL = 2  # round toward +inf
    TRUNC = 3  # round toward zero (truncate)


def round_signed_fraction_to_int(value: Fraction, mode: int) -> int:
    """Round an exact signed value to an integer according to the selected zkf_rint mode."""
    if mode == RoundMode.NEAREST_EVEN:
        return round_fraction_to_int_ties_even(value)  # floor-based helper is already symmetric for negatives
    if mode == RoundMode.FLOOR:
        return math.floor(value)
    if mode == RoundMode.CEIL:
        return math.ceil(value)
    if mode == RoundMode.TRUNC:
        return math.trunc(value)
    raise ValueError(f"invalid round mode: {mode}")


@functools.cache
def trans_specs() -> dict:
    from ._tables import trans

    return trans.SPECS


def trans_spec(func: str, wman: int) -> dict:
    try:
        return trans_specs()[(func, wman)]
    except KeyError:
        supported = sorted(w for f, w in trans_specs() if f == func)
        raise UnsupportedFormat(f"no {func} table for WMAN={wman}; supported: {supported}") from None


def trans_kernel(func: str, spec: dict, row: list[int], w: int) -> int:
    """
    The table+polynomial kernel of zkf_exp2 / zkf_log2, bit-identical to the generated cores: the segment polynomial
    c0 +- c1*wn + c2*wn**2 +- ... in wn = w / 2**rw at scale 2**-cf, as c0 +- c1*wn beside the pairs
    c[2i] +- c[2i+1]*wn chained by wn**2, which is formed once. Every product takes a fixed number of operand bits and
    is floored; the constants that center those floors, and exp2's half ULP, are part of c0. log2's odd coefficients
    subtract (the table stores magnitudes).

      t[i]    argument bits (from the top of w) multiplying c[2i+1]
      tc, sc  degree 2: the argument bits below t[0] multiply c1 >> sc, so the wide product is one tile wide
      t2, q   argument bits squared; bits of the square taken by each wn**2 product, outermost first
      th, sg  degree 3, t2 wider than a multiplier tile: the top th bits are squared and multiplied by the rest, the
              square of the rest is left out, and the sum carries sg bits below the bits taken
      z3      degree 3: low bits, all zero, that the stored c3 lacks, so that c3*wn is one tile wide
    """
    rw, d, t, q, t2 = spec["rw"], spec["d"], spec["t"], spec["q"], spec["t2"]
    sub = func == "log2"
    q2 = max(q)
    wt = w >> (rw - t2)
    if "th" in spec:
        tl, base = t2 - spec["th"], 2 * t2 - q2 - spec["sg"]
        wh, wl = wt >> tl, wt & mask(tl)
        w2 = (((wh * wh) >> (base - 2 * tl)) + ((wh * wl) >> (base - tl - 1))) >> spec["sg"]
    else:
        w2 = (wt * wt) >> (2 * t2 - q2)

    def odd(i: int) -> int:
        return (row[2 * i + 1] * (w >> (rw - t[i]))) >> t[i]

    def signed(x: int) -> int:
        return -x if sub else x

    lin = odd(0)
    if "tc" in spec:
        lin += ((row[1] >> spec["sc"]) * (w & mask(spec["tc"]))) >> (rw - spec["sc"])
    m = d // 2
    v = row[2 * m] + (signed(odd(m)) if 2 * m + 1 <= d else 0)
    for i in range(m - 1, 0, -1):
        v = row[2 * i] + signed(odd(i)) + ((v * (w2 >> (q2 - q[i]))) >> q[i])
    return row[0] + signed(lin) + ((v * w2) >> q2)


@functools.cache
def trig_specs() -> dict:
    from ._tables import trig

    return trig.SPECS


def trig_spec(wman: int) -> dict:
    try:
        return trig_specs()[wman]
    except KeyError:
        raise UnsupportedFormat(f"no CORDIC table for WMAN={wman}; supported: {sorted(trig_specs())}") from None


def atan2_bypass_shift(fmt: ZkfFormat) -> int:
    from ._tables import trig

    return trig_spec(fmt.wman)["zf"] - fmt.wman - trig.GUARD_DIV


def fixed_to_float_ref(
    fmt: ZkfFormat,
    sign: int,
    mag: int,
    exp_offset: int,
    wmag: int,
    *,
    force_inf: int = 0,
    saturate_round_carry: bool = False,
) -> int:
    """
    Mirror zkf/rtl/_zkf_fixed_to_float.v: normalize the unsigned magnitude, extract G/R/S, exp = exp_offset - count,
    and pack (RTNE). mag == 0 forces +0 unless force_inf is set.
    """
    zero_flag, count, aligned = normshift_reference(wmag, mag)
    significand_value = (aligned >> (wmag - fmt.wman)) & mask(fmt.wman)
    guard = (aligned >> (wmag - fmt.wman - 1)) & 1
    round_bit = (aligned >> (wmag - fmt.wman - 2)) & 1
    sticky = 1 if (aligned & mask(wmag - fmt.wman - 2)) else 0
    exp_unbiased = exp_offset - count
    force_zero = 0 if force_inf else (1 if zero_flag else 0)
    return pack_reference(
        fmt,
        sign,
        force_zero,
        force_inf,
        exp_unbiased,
        significand_value,
        guard,
        round_bit,
        sticky,
        saturate_round_carry=saturate_round_carry,
    )


def cordic_rotate(spec: dict, z0: int, n: int) -> tuple[int, int, int]:
    """
    Fixed-point CORDIC rotation (n ~= WMAN/2 iterations), mirroring zkf/rtl/_zkf_cordic_core.v. Returns (x_K, y_K,
    z_K): the partially rotated vector at scale 2**-xf and the residual angle z_K at scale 2**-zf (turns); the caller
    finishes with one linear step. Inverse gain is folded into the x seed. Shifts truncate toward -inf (Verilog >>>);
    the N-iteration truncation bias stays below the result ULP.
    """
    kinv, lut = spec["kinv"], spec["lut"]
    x, y, z = kinv, 0, z0
    for i in range(n):
        if z < 0:  # sigma = -1
            x, y, z = x + (y >> i), y - (x >> i), z + lut[i]
        else:  # sigma = +1
            x, y, z = x - (y >> i), y + (x >> i), z - lut[i]
    return x, y, z


def cordic_vector(spec: dict, x0: int, y0: int, n: int) -> tuple[int, int, int]:
    """
    Fixed-point CORDIC in VECTORING mode (n = N_atan2 iterations), mirroring zkf/rtl/_zkf_cordic_core.v MODE=1.
    Drives y -> 0 and returns (x_K, y_K, z_K): the residual vector at scale 2**-xf (x_K ~= gain*hypot) and the
    accumulated angle z_K at scale 2**-zf (turns, ~= atan2(y0, x0)). Same update as cordic_rotate but sigma follows
    sign(y), and there is no inverse-gain seed (the gain stays in x_K for the magnitude path). x/y/z wrap to the
    engine widths (xw/zw); the caller pre-scales the seed by 1/4 so that wrap never fires.
    """
    lut, xw, zw = spec["lut"], spec["xw"], spec["zw"]
    x = bits_to_signed(x0 & mask(xw), xw)
    y = bits_to_signed(y0 & mask(xw), xw)
    z = 0
    for i in range(n):
        if y >= 0:  # sigma = -1 (drive y down): mirrors neg = ~y[msb]
            nx, ny, nz = x + (y >> i), y - (x >> i), z + lut[i]
        else:  # sigma = +1
            nx, ny, nz = x - (y >> i), y + (x >> i), z - lut[i]
        x = bits_to_signed(nx & mask(xw), xw)
        y = bits_to_signed(ny & mask(xw), xw)
        z = bits_to_signed(nz & mask(zw), zw)
    return x, y, z


def atan2_turn(fmt: ZkfFormat, sign: int, frac: Fraction) -> int:
    """A signed exact-dyadic turn constant as a ZKF float; the half-turn endpoint canonicalizes to +1/2."""
    if frac == Fraction(1, 2):
        sign = 0
    return round_fraction_to_zkf(fmt, sign, frac)


def normshift_reference(width: int, value: int) -> tuple[int, int, int]:
    """
    Reference for _zkf_normshift: returns (zero, count, y). count = (width-1) - leading_one_position, i.e. the
    left-shift that brings the leading 1 to the MSB; y = value << count, the normalized vector. count and y are
    don't-care when zero is asserted.
    """
    value &= mask(width)
    if value == 0:
        return 1, 0, 0
    count = (width - 1) - (value.bit_length() - 1)
    return 0, count, (value << count) & mask(width)

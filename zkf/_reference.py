from __future__ import annotations

import enum
import functools
import math
from fractions import Fraction


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
    infinity = bool(force_inf or exp_overflow or exp_round_overflow)

    result_zero = bool(force_zero or ((not force_inf) and exp_underflow_zero))
    result_infinity = (not result_zero) and infinity
    result_min_normal = (not result_zero) and (not result_infinity) and (not force_inf) and exp_one_below_min

    if result_zero:
        return zero(fmt)
    if result_infinity:
        return canonical_inf(fmt, sign)
    if result_min_normal:
        return normal(fmt, sign, 1, 0)

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


def signed_int_min(wint: int) -> int:
    if wint < 2:
        raise ValueError(f"wint must be at least 2, got {wint}")
    return -(1 << (wint - 1))


def signed_int_max(wint: int) -> int:
    if wint < 2:
        raise ValueError(f"wint must be at least 2, got {wint}")
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
    """Private round-to-integer modes; the integer values match zkf/rtl/zkf_round.v and its 2-bit round_mode port."""

    NEAREST_EVEN = 0  # round to nearest integer, ties to even (the IEEE default)
    FLOOR = 1  # round toward -inf
    CEIL = 2  # round toward +inf
    TRUNC = 3  # round toward zero (truncate)


def round_signed_fraction_to_int(value: Fraction, mode: int) -> int:
    """Round an exact signed value to an integer according to the selected zkf_round mode."""
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
        raise KeyError(f"no {func} table for WMAN={wman}; run zkf_transcendental.py --emit")


def trans_sqrt2_threshold(wfrac: int) -> int:
    """
    Integer significand threshold for the log2 symmetric-reduction re-center test m >= sqrt(2) (m = sig/2^WFRAC,
    sig the WMAN-bit significand): re-center iff sig >= THR with THR = round(sqrt(2) * 2**WFRAC). Computed
    exactly with integer isqrt (round-to-nearest), and MUST equal the generator's log2_sqrt2_threshold and the phase-2
    RTL constant. round(sqrt(S)) for S = 2**(2*WFRAC+1) is (floor(sqrt(4*S)) + 1) // 2 = (isqrt(4*S) + 1) // 2.
    """
    return (math.isqrt(1 << (2 * wfrac + 3)) + 1) // 2  # 4*S = 2**(2*wfrac+3)


def horner_eval(coeffs_idx: list[int], w: int, rw: int) -> int:
    """Truncating fixed-point Horner, bit-identical to zkf/rtl/_zkf_horner.v and the generator."""
    acc = coeffs_idx[-1]
    for j in range(len(coeffs_idx) - 2, -1, -1):
        acc = coeffs_idx[j] + ((acc * w) >> rw)  # Python >> floors, matching arithmetic >>> on signed
    return acc


@functools.cache
def trig_specs() -> dict:
    from ._tables import trig

    return trig.SPECS


def trig_spec(wman: int) -> dict:
    try:
        return trig_specs()[wman]
    except KeyError:
        raise KeyError(f"no sincos table for WMAN={wman}; run zkf_trig.py --emit")


def atan2_bypass_shift(fmt: ZkfFormat) -> int:
    from ._tables import trig

    return trig_spec(fmt.wman)["zf"] - fmt.wman - trig.GUARD_DIV


def fixed_to_float_ref(fmt: ZkfFormat, sign: int, mag: int, exp_offset: int, wmag: int, *, force_inf: int = 0) -> int:
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
    return pack_reference(fmt, sign, force_zero, force_inf, exp_unbiased, significand_value, guard, round_bit, sticky)


def cordic_rotate(spec: dict, z0: int, n: int) -> tuple[int, int, int]:
    """
    Fixed-point CORDIC rotation (n ~= WMAN/2 iterations), mirroring zkf/rtl/_zkf_cordic.v. Returns (x_K, y_K, z_K): the
    partially rotated vector at scale 2**-xf and the residual angle z_K at scale 2**-zf (turns); the caller finishes
    with one linear step. Inverse gain is folded into the x seed. Shifts truncate toward -inf (Verilog >>>); the
    N-iteration truncation bias stays below the result ULP.
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
    Fixed-point CORDIC in VECTORING mode (n = N_atan2 iterations), mirroring zkf/rtl/_zkf_cordic.v MODE=1. Drives y -> 0
    and returns (x_K, y_K, z_K): the residual vector at scale 2**-xf (x_K ~= gain*hypot) and the accumulated angle z_K
    at scale 2**-zf (turns, ~= atan2(y0, x0)). Same update as cordic_rotate but sigma follows sign(y), and there is no
    inverse-gain seed (the gain stays in x_K for the magnitude path). x/y/z wrap to the engine widths (xw/zw); the
    caller pre-scales the seed by 1/4 so that wrap never fires.
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

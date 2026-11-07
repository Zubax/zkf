import functools
import math
from dataclasses import dataclass
from fractions import Fraction

import mpmath


class UnsupportedFormat(ValueError):
    """
    The operator does not support this format: WEXP, WMAN or an integer width such as WINT is out of range, or there is
    no table for WMAN. An invalid tuning knob raises a plain `ValueError` instead.
    """


@dataclass(frozen=True, slots=True)
class ZkfFormat:
    """A ZKF format: `wexp` exponent bits and `wman` significand bits, the hidden one included."""

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        if self.wexp < 2 or self.wman < 4:
            raise UnsupportedFormat(f"{self} is below e2m4")

    def __str__(self) -> str:
        return f"e{self.wexp}m{self.wman}"

    @property
    def wfrac(self) -> int:
        return self.wman - 1

    @property
    def wfull(self) -> int:
        return self.wexp + self.wman

    @property
    def sign_shift(self) -> int:
        return self.wexp + self.wfrac

    @property
    def bias(self) -> int:
        return (1 << (self.wexp - 1)) - 1

    @property
    def exp_inf(self) -> int:
        return (1 << self.wexp) - 1

    @property
    def exp_max_finite(self) -> int:
        return self.exp_inf - 1

    @property
    def frac_mask(self) -> int:
        return (1 << self.wfrac) - 1

    @property
    def min_exp_unbiased(self) -> int:
        return 1 - self.bias

    @property
    def max_exp_unbiased(self) -> int:
        return self.exp_max_finite - self.bias

    @property
    def lowest(self) -> Fraction:
        """The smallest positive magnitude, which is the smallest normal."""
        return pow2(self.min_exp_unbiased)

    @property
    def max(self) -> Fraction:
        """Largest finite magnitude."""
        return (2 - self.epsilon) * pow2(self.max_exp_unbiased)

    @property
    def epsilon(self) -> Fraction:
        """Gap between 1.0 and the next representable value above it."""
        return pow2(-self.wfrac)

    def pack(self, sign: int, exp: int, frac: int) -> Zkf:
        """The value of these fields; non-canonical zeros and infinities included."""
        if not (0 <= sign <= 1 and 0 <= exp <= self.exp_inf and 0 <= frac <= self.frac_mask):
            raise ValueError(f"no value of {self} has sign={sign}, exp={exp}, frac={frac}")
        return Zkf(self, sign << self.sign_shift | exp << self.wfrac | frac)

    def inf(self, sign: int = 0) -> Zkf:
        return self.pack(sign, self.exp_inf, 0)

    def encode(self, value: float | int | Fraction | mpmath.mpf) -> Zkf:
        """
        `value` rounded to nearest, ties to even; NaN raises `ValueError`. An mpf is taken exactly, whatever the
        precision of its context.
        """
        if mpmath.isnan(value):
            raise ValueError("NaN is not representable in ZKF")
        sign = int(value < 0)
        if mpmath.isinf(value):
            return self.inf(sign)
        if not isinstance(value, int | float | Fraction) and value:
            # An mpf far outside the format, whose integer ratio can be astronomically wide.
            man, exp = value.man_exp
            binade = exp + abs(man).bit_length() - 1
            if binade > self.max_exp_unbiased:
                return self.inf(sign)
            if binade < self.min_exp_unbiased - 1:
                return Zkf(self, 0)
        mag = abs(Fraction(value))
        if mag == 0:
            return Zkf(self, 0)
        num, den = mag.numerator, mag.denominator
        exp = num.bit_length() - den.bit_length()  # floor(log2(mag)) or one above it
        if (num < den << exp) if exp >= 0 else (num << -exp < den):
            exp -= 1
        if exp < self.min_exp_unbiased:
            return self.pack(sign, 1, 0) if exp == self.min_exp_unbiased - 1 else Zkf(self, 0)
        shift = self.wfrac - exp
        num, den = (num << shift, den) if shift >= 0 else (num, den << -shift)
        quotient, remainder = divmod(num, den)
        if 2 * remainder > den or (2 * remainder == den and quotient & 1):
            quotient += 1
        if quotient >> self.wman:
            quotient >>= 1
            exp += 1
        if exp > self.max_exp_unbiased:
            return self.inf(sign)
        return self.pack(sign, exp + self.bias, quotient & self.frac_mask)


@dataclass(frozen=True, slots=True, repr=False)
class Zkf:
    """
    An immutable ZKF value: a format and the packed bits. Equality is structural, so non-canonical encodings of one
    number differ.
    """

    fmt: ZkfFormat
    bits: int

    def __post_init__(self) -> None:
        if not 0 <= self.bits < 1 << self.fmt.wfull:
            raise ValueError(f"{self.bits:#x} is no value of {self.fmt}")

    @property
    def negative(self) -> bool:
        return bool((self.bits >> self.fmt.sign_shift) & 1)

    @property
    def exp(self) -> int:
        return (self.bits >> self.fmt.wfrac) & self.fmt.exp_inf

    @property
    def frac(self) -> int:
        return self.bits & self.fmt.frac_mask

    @property
    def is_zero(self) -> bool:
        return self.exp == 0

    @property
    def is_inf(self) -> bool:
        return self.exp == self.fmt.exp_inf

    @property
    def is_normal(self) -> bool:
        return 0 < self.exp < self.fmt.exp_inf

    @property
    def is_finite(self) -> bool:
        return not self.is_inf

    def significand(self) -> int:
        """WMAN-bit significand including the implicit leading one."""
        return 1 << self.fmt.wfrac | self.frac

    def to_fraction(self) -> Fraction:
        """The exact value; an infinity raises `ValueError`."""
        if self.is_inf:
            raise ValueError("infinity has no finite value; use is_inf/negative")
        if self.is_zero:
            return Fraction(0)
        significand, exp = scaled(self)
        return significand * pow2(exp)

    def to_mpf(self) -> mpmath.mpf:
        """The exact value as an mpf, infinities included."""
        if self.is_zero:
            return mpmath.mpf(0)
        if self.is_inf:
            return -mpmath.inf if self.negative else mpmath.inf
        return mpmath.ldexp(*scaled(self))

    def __float__(self) -> float:
        if self.is_inf:
            return -math.inf if self.negative else math.inf
        return float(self.to_fraction())

    def __int__(self) -> int:
        if self.is_inf:
            raise OverflowError("cannot convert ZKF infinity to integer")
        return math.trunc(self.to_fraction())

    def __bool__(self) -> bool:
        return not self.is_zero

    def __repr__(self) -> str:
        return f"Zkf({self.fmt!r}, {self:x})"

    def __str__(self) -> str:
        return format(self)

    def __format__(self, spec: str) -> str:
        if spec == "x":
            return f"0x{self.bits:0{(self.fmt.wfull + 3) // 4}x}"
        if self.is_normal and (self.fmt.wman > 53 or not -1022 <= self.exp - self.fmt.bias <= 1023):
            # Where a float would round the value: at the precision of the format instead.
            return format(context(self.fmt.wman).ldexp(*scaled(self)), spec)
        return format(float(self), spec)

    def canonicalize(self) -> Zkf:
        """Any zero as +0 and any infinity as the canonical one of its sign; other values unchanged."""
        if self.is_zero:
            return Zkf(self.fmt, 0)
        return self.fmt.inf(self.negative) if self.is_inf else self


def scaled(x: Zkf) -> tuple[int, int]:
    """A finite nonzero `x` as `significand * 2**exp`: the signed significand and the exponent of its last bit."""
    return -x.significand() if x.negative else x.significand(), x.exp - x.fmt.bias - x.fmt.wfrac


def pow2(exp: int) -> Fraction:
    """`2**exp` exactly, by a shift."""
    return Fraction(1 << exp) if exp >= 0 else Fraction(1, 1 << -exp)


@functools.cache
def context(prec: int) -> mpmath.MPContext:
    """An mpmath context of `prec` bits, independent of the global one."""
    ctx = mpmath.mp.clone()
    ctx.prec = prec
    return ctx


def bits_to_signed(value: int, width: int) -> int:
    """The `width` low bits of `value` as a two's complement number."""
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value

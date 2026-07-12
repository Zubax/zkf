from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

from ._reference import *


class DivResult(NamedTuple):
    quotient: Zkf
    div_by_zero: bool


class Log2Result(NamedTuple):
    value: Zkf
    domain_error: bool
    pole: bool


class SinCos(NamedTuple):
    sin: Zkf
    cos: Zkf
    quadrant: int


class Atan2Result(NamedTuple):
    theta: Zkf
    magnitude: Zkf


class CmpResult(NamedTuple):
    lt: bool
    eq: bool
    gt: bool


@dataclass(frozen=True, slots=True, repr=False)
class Zkf:
    """
    An immutable ZKF value: a ZkfFormat and the packed integer bits.

    Bit-exact model of the RTL operators, exposed as Python operators/methods. Equality and hashing are structural
    (same format and same bits) -- the correct semantics for a verification value type, and what catches a
    canonicalization mismatch that a numeric compare would hide. Numeric comparison is the explicit cmp method.
    Arithmetic requires both operands to share a format; a mismatch raises ValueError.
    """

    fmt: ZkfFormat
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", self.bits & mask(self.fmt.wfull))

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
        return significand(self.fmt, self.bits)

    def to_fraction(self) -> Fraction:
        """Exact value as a Fraction. Raises on infinity (Fraction has no infinity)."""
        if self.is_inf:
            raise ValueError("infinity has no finite value; use is_inf/negative")
        if self.is_zero:
            return Fraction(0)
        mag = Fraction(self.significand()) * pow2_fraction(self.exp - self.fmt.bias - self.fmt.wfrac)
        return -mag if self.negative else mag

    def round_int(self, wint: int) -> int:
        """Round to the nearest signed wint-bit integer, ties to even, with saturation."""
        return self._round_to_int(wint, RoundMode.NEAREST_EVEN)

    def floor_int(self, wint: int) -> int:
        """Round toward -inf to a signed wint-bit integer with saturation."""
        return self._round_to_int(wint, RoundMode.FLOOR)

    def ceil_int(self, wint: int) -> int:
        """Round toward +inf to a signed wint-bit integer with saturation."""
        return self._round_to_int(wint, RoundMode.CEIL)

    def trunc_int(self, wint: int) -> int:
        """Round toward zero to a signed wint-bit integer with saturation."""
        return self._round_to_int(wint, RoundMode.TRUNC)

    def _round_to_int(self, wint: int, mode: RoundMode) -> int:
        int_max = signed_int_max(wint)
        int_min = signed_int_min(wint)
        if self.is_inf:
            return int_min if self.negative else int_max
        if self.is_zero:
            return 0
        rounded = round_signed_fraction_to_int(self.to_fraction(), mode)
        if rounded > int_max:
            return int_max
        if rounded < int_min:
            return int_min
        return rounded

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
        return format(float(self), spec)

    def _require_same(self, other: Zkf) -> None:
        if other.fmt != self.fmt:
            raise ValueError(f"operands have different formats: {self.fmt} vs {other.fmt}")

    def __mul__(self, other: Zkf) -> Zkf:
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        fmt = self.fmt
        a = self
        b = other
        result_zero = a.is_zero or b.is_zero
        result_inf = (not result_zero) and (a.is_inf or b.is_inf)

        product = significand(fmt, a.bits) * significand(fmt, b.bits)
        product_high = (product >> ((2 * fmt.wman) - 1)) & 1
        exp_unbiased_base = a.exp + b.exp - (fmt.bias << 1)

        if product_high:
            exp_unbiased = exp_unbiased_base + 1
            significand_value = (product >> fmt.wman) & mask(fmt.wman)
            guard = (product >> (fmt.wman - 1)) & 1
            round_bit = (product >> (fmt.wman - 2)) & 1
            sticky = sticky_below(product, fmt.wman - 3)
        else:
            exp_unbiased = exp_unbiased_base
            significand_value = (product >> (fmt.wman - 1)) & mask(fmt.wman)
            guard = (product >> (fmt.wman - 2)) & 1
            round_bit = (product >> (fmt.wman - 3)) & 1
            sticky = sticky_below(product, fmt.wman - 4)

        return Zkf(
            fmt,
            pack_reference(
                fmt,
                a.negative ^ b.negative,
                int(result_zero),
                int(result_inf),
                exp_unbiased,
                significand_value,
                guard,
                round_bit,
                sticky,
            ),
        )

    def __add__(self, other: Zkf) -> Zkf:
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        fmt = self.fmt
        a = self
        b = other

        if a.is_inf and b.is_inf:
            return Zkf(fmt, canonical_inf(fmt, a.negative) if a.negative == b.negative else zero(fmt))
        if a.is_inf:
            return Zkf(fmt, canonical_inf(fmt, a.negative))
        if b.is_inf:
            return Zkf(fmt, canonical_inf(fmt, b.negative))

        result = a.to_fraction() + b.to_fraction()
        if result == 0:
            return Zkf(fmt, zero(fmt))
        return Zkf(fmt, round_fraction_to_zkf(fmt, int(result < 0), abs(result)))

    def __sub__(self, other: Zkf) -> Zkf:
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        return self + (-other)

    def __truediv__(self, other: Zkf) -> Zkf:
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        return self.div(other).quotient

    def __neg__(self) -> Zkf:
        return Zkf(self.fmt, self.bits ^ (1 << self.fmt.sign_shift))  # Zkf.__post_init__ masks to WFULL

    def __abs__(self) -> Zkf:
        return Zkf(self.fmt, self.bits & mask(self.fmt.sign_shift))

    def div(self, other: Zkf) -> DivResult:
        """Quotient plus the div-by-zero status flag. a / b is a.div(b).quotient."""
        self._require_same(other)
        fmt = self.fmt
        a = self
        b = other
        div0 = b.is_zero

        if a.is_zero or b.is_inf:
            return DivResult(Zkf(fmt, zero(fmt)), div0)

        result_sign = a.negative if b.is_zero else (a.negative ^ b.negative)
        if b.is_zero or a.is_inf:
            return DivResult(Zkf(fmt, canonical_inf(fmt, result_sign)), div0)

        value = abs(a.to_fraction() / b.to_fraction())
        return DivResult(Zkf(fmt, round_fraction_to_zkf(fmt, result_sign, value)), div0)

    def fma(self, b: Zkf, c: Zkf) -> Zkf:
        """Correctly-rounded fused multiply-add: round(self*b + c) with a single rounding."""
        self._require_same(b)
        self._require_same(c)
        fmt = self.fmt
        a = self

        p_zero = a.is_zero or b.is_zero
        p_inf = (not p_zero) and (a.is_inf or b.is_inf)
        p_sign = a.negative ^ b.negative

        if p_inf and c.is_inf:
            return Zkf(fmt, canonical_inf(fmt, p_sign) if p_sign == c.negative else zero(fmt))
        if p_inf:
            return Zkf(fmt, canonical_inf(fmt, p_sign))
        if c.is_inf:
            return Zkf(fmt, canonical_inf(fmt, c.negative))

        product = Fraction(0) if p_zero else a.to_fraction() * b.to_fraction()
        result = product + c.to_fraction()
        if result == 0:
            return Zkf(fmt, zero(fmt))
        return Zkf(fmt, round_fraction_to_zkf(fmt, int(result < 0), abs(result)))

    def mul_ilog2(self, k: int) -> Zkf:
        """Multiply by 2**k for signed integer k (exponent add); models zkf_mul_ilog2 and zkf_mul_ilog2_const."""
        fmt = self.fmt
        a = self
        if a.is_zero:
            return Zkf(fmt, zero(fmt))
        if a.is_inf:
            return Zkf(fmt, canonical_inf(fmt, a.negative))
        new_exp = a.exp + k
        if new_exp < 0:
            return Zkf(fmt, zero(fmt))
        if new_exp == 0:
            return Zkf(fmt, normal(fmt, a.negative, 1, 0))
        if new_exp > fmt.exp_max_finite:
            return Zkf(fmt, canonical_inf(fmt, a.negative))
        return Zkf(fmt, normal(fmt, a.negative, new_exp, a.frac))

    def saturate(self) -> Zkf:
        """Clamp infinities to the largest finite magnitude; finite values pass through."""
        fmt = self.fmt
        if not self.is_inf:
            return Zkf(fmt, self.bits)
        return Zkf(fmt, normal(fmt, self.negative, fmt.exp_max_finite, fmt.frac_mask))

    def canonicalize(self) -> Zkf:
        """
        Fold non-canonical encodings to their canonical form: any zero -> +0 (sign and stray fraction cleared),
        any infinity -> sign-preserving canonical infinity (fraction cleared); finite values pass through unchanged.
        """
        return Zkf(self.fmt, canonicalize_special(self.fmt, self.bits))

    def resize(self, fmt_out: ZkfFormat) -> Zkf:
        """Re-encode into another format (RNTE)."""
        if self.is_zero:
            return Zkf(fmt_out, zero(fmt_out))
        if self.is_inf:
            return Zkf(fmt_out, canonical_inf(fmt_out, self.negative))
        return Zkf(fmt_out, round_fraction_to_zkf(fmt_out, self.negative, abs(self.to_fraction())))

    def round(self) -> Zkf:
        """Round to the nearest integral value, ties to even (the zkf_round RNTE mode)."""
        return self._round_to_integral(RoundMode.NEAREST_EVEN)

    def floor(self) -> Zkf:
        """Round toward -inf to an integral value (the zkf_round floor mode)."""
        return self._round_to_integral(RoundMode.FLOOR)

    def ceil(self) -> Zkf:
        """Round toward +inf to an integral value (the zkf_round ceil mode)."""
        return self._round_to_integral(RoundMode.CEIL)

    def trunc(self) -> Zkf:
        """Round toward zero to an integral value (the zkf_round trunc mode)."""
        return self._round_to_integral(RoundMode.TRUNC)

    def _round_to_integral(self, mode: RoundMode) -> Zkf:
        """Round to an integral value in the same format per mode (the zkf_round operator)."""
        fmt = self.fmt
        if self.is_inf:
            return Zkf(fmt, canonical_inf(fmt, self.negative))
        if self.is_zero:
            return Zkf(fmt, zero(fmt))
        rounded = round_signed_fraction_to_int(self.to_fraction(), mode)
        if rounded == 0:
            return Zkf(fmt, zero(fmt))
        return Zkf(fmt, round_fraction_to_zkf(fmt, int(rounded < 0), Fraction(abs(rounded))))

    def cmp(self, other: Zkf) -> CmpResult:
        """Numeric comparison mirroring the RTL zkf_cmp (total order over the format)."""
        self._require_same(other)
        a_key = ordered_key(self.fmt, self.bits)
        b_key = ordered_key(self.fmt, other.bits)
        return CmpResult(a_key < b_key, a_key == b_key, a_key > b_key)

    def sort(self, other: Zkf) -> tuple[Zkf, Zkf]:
        """Return (min, max) of the pair by numeric order (the zkf_sort operator)."""
        self._require_same(other)
        lt = self.cmp(other).lt
        a_bits, b_bits = self.bits, other.bits
        lo, hi = (a_bits, b_bits) if lt else (b_bits, a_bits)
        return Zkf(self.fmt, lo), Zkf(self.fmt, hi)

    def exp2(self) -> Zkf:
        """2 ** self (the zkf_exp2 operator)."""
        fmt, bits = self.fmt, self.bits
        d = self
        if d.is_inf:
            return Zkf(fmt, canonical_inf(fmt, 0) if not d.negative else zero(fmt))  # +inf -> +inf, -inf -> +0
        if d.is_zero:
            return Zkf(fmt, normal(fmt, 0, fmt.bias, 0))  # 2**0 = 1.0
        e = d.exp - fmt.bias
        # |x| >= 2^(WEXP-1) is always out of range: overflow (x>0) or underflow (x<0).
        if e >= fmt.wexp - 1:
            return Zkf(fmt, canonical_inf(fmt, 0) if not d.negative else zero(fmt))

        spec = trans_spec("exp2", fmt.wman)
        cf, rw = spec["cf"], spec["rw"]
        ff = spec["k"] + rw  # full reduced-argument width FF = K + RW
        sig = significand(fmt, bits)
        shift = e - fmt.wfrac + ff
        if shift >= 0:
            mfix = sig << shift
            lost_sticky = 0
        else:
            rs = -shift
            mfix = sig >> rs
            lost_sticky = 1 if (sig & mask(rs)) else 0
        v = -mfix if d.negative else mfix
        i = v >> ff  # arithmetic floor -> integer part of x
        f = v & mask(ff)  # fractional part in [0, 2^FF)

        acc = horner_eval(spec["coeffs"][(f >> rw) - spec.get("seg_base", 0)], f & mask(rw), rw)
        significand_value = (acc >> (cf - fmt.wfrac)) & mask(fmt.wman)
        guard = (acc >> (cf - fmt.wman)) & 1
        round_bit = (acc >> (cf - fmt.wman - 1)) & 1
        sticky = (1 if (acc & mask(cf - fmt.wman - 1)) else 0) | lost_sticky
        return Zkf(fmt, pack_reference(fmt, 0, 0, 0, i, significand_value, guard, round_bit, sticky))

    def log2(self) -> Log2Result:
        """log2(self) plus the domain-error (self<0) and pole (self==0) status flags."""
        fmt, bits = self.fmt, self.bits
        d = self
        if d.is_inf and not d.negative:
            return Log2Result(Zkf(fmt, canonical_inf(fmt, 0)), False, False)  # log2(+inf) = +inf
        if d.is_zero:
            return Log2Result(Zkf(fmt, canonical_inf(fmt, 1)), False, True)  # log2(+0) = -inf, pole
        if d.negative:
            return Log2Result(Zkf(fmt, canonical_inf(fmt, 1)), True, False)  # log2(x<0) = -inf, domain error

        e = d.exp - fmt.bias
        spec = trans_spec("log2", fmt.wman)
        cf, rw = spec["cf"], spec["rw"]

        # Symmetric argument reduction (mirrors the phase-2 RTL re-center; defines the bit-exact contract). x = m*2^e,
        # m = sig/2^WFRAC in [1,2). If m >= sqrt(2), halve m and increment e so m' in [sqrt(1/2), sqrt(2)). At scale
        # 2^-(WFRAC+1): v = f + 1/2 is the unsigned segment index, f_signed = m'-1 (= f) the signed combine operand.
        sig = significand(fmt, bits)  # WMAN-bit significand, m = sig / 2^WFRAC in [1,2)
        if sig >= trans_sqrt2_threshold(fmt.wfrac):  # m >= sqrt(2): re-center into [sqrt(1/2), sqrt(2))
            e += 1
            v = d.frac  # = sig - 2^WFRAC
            f_signed = d.frac - (1 << fmt.wfrac)  # < 0
        else:
            v = (1 << fmt.wfrac) + (d.frac << 1)  # = 2*sig - 2^WFRAC
            f_signed = d.frac << 1  # = 2*frac, >= 0

        acc = horner_eval(spec["coeffs"][(v >> rw) - spec.get("seg_base", 0)], v & mask(rw), rw)
        f2 = fmt.wfrac + 1 + cf  # f is at scale 2^-(WFRAC+1)
        l_signed = f_signed * acc  # log2(m') = f * C(f), signed, at scale 2^-f2
        r = (e << f2) + l_signed  # signed fixed point e + log2(m')
        sign_out = 1 if r < 0 else 0
        magnitude = -r if r < 0 else r
        w_norm = fmt.wexp + f2 + 1
        zero_flag, count, aligned = normshift_reference(w_norm, magnitude)
        significand_value = (aligned >> (w_norm - fmt.wman)) & mask(fmt.wman)
        guard = (aligned >> (w_norm - fmt.wman - 1)) & 1
        round_bit = (aligned >> (w_norm - fmt.wman - 2)) & 1
        sticky = 1 if (aligned & mask(w_norm - fmt.wman - 2)) else 0
        exp_unbiased = (w_norm - 1 - count) - f2
        y = pack_reference(fmt, sign_out, zero_flag, 0, exp_unbiased, significand_value, guard, round_bit, sticky)
        return Log2Result(Zkf(fmt, y), False, False)

    def sincos(self) -> SinCos:
        """(sin(2*pi*self), cos(2*pi*self), quadrant) -- the zkf_sincos operator."""
        fmt, bits = self.fmt, self.bits
        d = self
        if d.is_inf:
            s = canonical_inf(fmt, d.negative)
            return SinCos(Zkf(fmt, s), Zkf(fmt, s), 0)  # +inf -> (+inf,+inf,0); -inf -> (-inf,-inf,0)
        if d.is_zero:
            return SinCos(Zkf(fmt, zero(fmt)), Zkf(fmt, normal(fmt, 0, fmt.bias, 0)), 0)  # sin(0)=+0, cos(0)=+1

        spec = trig_spec(fmt.wman)
        xf, zf = spec["xf"], spec["zf"]
        # const2pi arrives PRE-NARROWED from the table (top WMAN+5 bits == round(2*pi * 2**const2pi_s)); const2pi_s is
        # the single source of scale for every consuming shift / exp-offset. Mirrors zkf/rtl/zkf_sincos.v.
        const2pi, const2pi_s = spec["const2pi"], spec["const2pi_s"]
        n_sincos = spec["n_sincos"]  # sincos iterations (linear-rotation termination); == table n
        wt = spec["wt"]  # quadrant-local coordinate width (FF - 2)
        ff = wt + 2
        zg = zf - (wt + 2)  # extra angle-accumulator fractional bits (GUARD_ZF)
        # Uniform magnitude width wmag sized for the widest product (const2pi * t'). CORDIC magnitudes are at scale
        # 2**-xf (exp offset eone_xf); const2pi products at 2**-const2pi_s (eone_s). Mirrors RTL width+exp.
        cwb = const2pi.bit_length()  # narrowed 2*pi width == WMAN+5
        wmag = cwb + wt + 1
        eone_xf = wmag - 1 - xf  # exp_offset for a magnitude at scale 2**-xf (corr / +1 path)
        eone_s = wmag - 1 - const2pi_s  # exp_offset for a const2pi product at scale 2**-const2pi_s
        one = (1 << xf, eone_xf)  # value +1.0
        tsa = spec["tsa"]
        sig = significand(fmt, bits)
        e = d.exp - fmt.bias

        # -- Reduce |x| mod 1 to the FF-bit fraction; SH = e - WFRAC + FF places |sig| at scale 2**-FF. Using
        # sin(2*pi*x) = sign*sin(2*pi*|x|), cos(2*pi*x) = cos(2*pi*|x|) collapses the x<0 negate to a sin sign flip.
        sh = e - fmt.wfrac + ff
        tiny = sh < 0  # below the reducer's resolution -> small-angle path on |x|
        lshamt = 0 if tiny else min(sh, ff)
        frac_pos = (sig << lshamt) & mask(ff)
        quadrant_abs = 0 if tiny else (frac_pos >> wt) & 3  # |x| quadrant (0 for tiny: |x| < 1/4)
        t = sig if tiny else (frac_pos & mask(wt))  # quadrant-local coordinate, scale 2**-WT
        tzero = (not tiny) and t == 0  # frac(|x|)*4 integer: a quadrant boundary / exact magnitude

        # -- Octant fold: bring the local angle into [0, pi/4] (t' <= 1/2). theta' in turns at scale 2**-zf is just t'.
        half = 1 << (wt - 1)
        oct_flip = (not tiny) and t > half  # theta in (pi/4, pi/2): use the pi/2 - theta complement
        tp = ((1 << wt) - t) if oct_flip else t

        # -- Octant-local (sin theta', cos theta') as (magnitude, exp_offset) pairs at the uniform WMAG scale.
        if tiny:
            # Under-resolution: sin ~= 2*pi*|x| = 2*pi*|sig|*2**(e-wfrac); cos = +1. const2pi*|sig| sits at scale
            # 2**-const2pi_s, so its exp_offset is eone_s plus the data binade (e - wfrac). No correction token.
            sin_tp = (const2pi * sig, eone_s + e - fmt.wfrac)
            cos_tp = one
        elif tzero:
            sin_tp, cos_tp = (0, eone_xf), one  # exact quadrant boundary: sin theta' = 0, cos theta' = 1
        elif tp < tsa:
            # Small octant-local angle (cos=1): sin theta' ~= 2*pi*tp (tp at scale 2**-(WT+2)); const2pi*tp is at
            # 2**-const2pi_s, so exp_offset = eone_s - (wt + 2).
            sin_tp = (const2pi * tp, eone_s - (wt + 2))
            cos_tp = one
        else:
            # K CORDIC iterations then one linear rotation by residual z_K (phi = 2*pi*z_K): sin theta' = y_K + x_K*phi,
            # cos theta' = x_K - y_K*phi.
            xk, yk, zk = cordic_rotate(spec, tp << zg, n_sincos)  # seed z0 = t' shifted into the finer 2**-zf scale
            # phi and x_K/y_K are narrowed to ~18-bit operands (one 18x18 DSP each); dropping the low bits of this small
            # fix-up stays < 1 ULP (--check confirms).
            n = n_sincos  # phi's natural width XF-N+2 uses the sincos iteration count
            phiw = min(fmt.wman + 6, max(2, xf - n + 2))  # phi top bits (natural width XF-N+2, capped at WMAN+6)
            phi_trunc = max(0, (xf - n + 2) - phiw)
            phi_s = xf - phi_trunc  # scale of the narrowed phi (== 2*pi*z_K at 2**-phi_s)
            xcw = fmt.wman + 6  # x_K/y_K correction-operand top bits
            xk_trunc = (xf + 2) - xcw  # XW = XF+2; keep the top XCW bits
            # phi = const2pi*z_K (scale 2**-(const2pi_s + zf)) narrowed to PHIW bits at scale 2**-phi_s by a single
            # right-shift (const2pi_s + zf) - phi_s. const2pi is the pre-narrowed operand, so no correction token.
            phi = (const2pi * zk) >> ((const2pi_s + zf) - phi_s)  # signed, PHIW bits, scale 2**-phi_s
            corr_s = ((xk >> xk_trunc) * phi) >> (xf - xk_trunc - phi_trunc)  # x_K*phi at scale 2**-xf
            corr_c = ((yk >> xk_trunc) * phi) >> (xf - xk_trunc - phi_trunc)
            sin_tp = (yk + corr_s, eone_xf)
            cos_tp = (xk - corr_c, eone_xf)

        # -- Unmap the octant (sin theta = cos theta', cos theta = sin theta' when folded), then the |x| quadrant.
        sin_loc, cos_loc = (cos_tp, sin_tp) if oct_flip else (sin_tp, cos_tp)
        sin_m, cos_m = (cos_loc, sin_loc) if (quadrant_abs & 1) else (sin_loc, cos_loc)
        sin_sign = ((quadrant_abs >> 1) & 1) ^ d.negative  # sin is odd: negative x flips it
        cos_sign = ((quadrant_abs >> 1) ^ quadrant_abs) & 1  # cos is even: unchanged by the sign of x
        # Output quadrant = floor(frac(x)*4): for x >= 0 the |x| quadrant; for x < 0 it reflects about a turn.
        if not d.negative:
            quadrant = quadrant_abs
        else:
            quadrant = ((4 - quadrant_abs) & 3) if tzero else (3 - quadrant_abs)

        sin_bits = fixed_to_float_ref(fmt, sin_sign, sin_m[0], sin_m[1], wmag)
        cos_bits = fixed_to_float_ref(fmt, cos_sign, cos_m[0], cos_m[1], wmag)
        return SinCos(Zkf(fmt, sin_bits), Zkf(fmt, cos_bits), quadrant)

    def atan2(self, x: Zkf) -> Atan2Result:
        """
        (theta, magnitude) of atan2(self, x): theta in turns (-0.5, 0.5], magnitude = hypot(self, x).

        Reuses the sincos CORDIC engine in vectoring mode: it yields z_K ~= atan2 (turns) and x_K ~= gain*hypot; a
        residual divide finishes the small angle, a small-ratio bypass covers the near-+x-axis, and the magnitude
        descales x_K by 1/gain (== KINV).
        """
        self._require_same(x)
        fmt, y_bits, x_bits = self.fmt, self.bits, x.bits
        sp = atan2_special(fmt, y_bits, x_bits)
        if sp is not None:
            return Atan2Result(Zkf(fmt, sp[0]), Zkf(fmt, sp[1]))

        spec = trig_spec(fmt.wman)
        # xf is the SHARED engine width (table WX/KINV/INV_TAU and the returned x_K/y_K all at 2**-xf). xf_atan2 is
        # atan2's own x/y width, driving only the divider quotient budget F; equal today, read separately to decouple.
        xf, zf = spec["xf"], spec["zf"]
        n = spec["n_atan2"]  # atan2 iterations (residual-divide termination)
        xf_div = spec["xf_atan2"]  # divider x/y fractional width (== xf today)
        wfrac = fmt.wfrac

        # The shared _zkf_pmul multiplies x_K*kinv_mag (magnitude) and Q*inv_tau (residual + bypass theta). Both
        # constants are PRE-NARROWED to WMAN+5 bits at their native scales (kinv_s, invtau_s), so every dependent shift
        # is "product-scale minus target-scale" with no fold-back. Mirrors zkf/rtl/zkf_atan2.v.
        kinv_mag, kinv_s = spec["kinv_mag"], spec["kinv_s"]  # narrowed 1/gain (MAG product) + its native scale
        inv_tau, invtau_s = spec["inv_tau"], spec["invtau_s"]  # narrowed 1/(2*pi) (residual + bypass) + native scale

        dy = self
        dx = x
        sx, sy = dx.negative, dy.negative
        sig_x = significand(fmt, x_bits)
        sig_y = significand(fmt, y_bits)
        ex = dx.exp - fmt.bias
        ey = dy.exp - fmt.bias

        # Order by magnitude: den = max(|x|,|y|), num = min. The reduced octant angle a0 = atan(num/den) in [0, 1/8].
        swap = (ey > ex) or (ey == ex and sig_y > sig_x)  # |y| > |x|
        if swap:
            den_sig, e_den, num_sig, e_num = sig_y, ey, sig_x, ex
        else:
            den_sig, e_den, num_sig, e_num = sig_x, ex, sig_y, ey
        shift_dn = e_den - e_num  # >= 0 (den >= num)
        # Pre-scale the vector by 1/4 (den_fixed in [0.25, 0.5)*2**xf) so x_K = gain*hypot stays inside the engine's
        # signed width WX=xf+2; the 1/4 is folded into the magnitude exponent (+2) and the angle is invariant.
        den_fixed = den_sig << (xf - wfrac - 2)  # in [0.25, 0.5) * 2**xf
        num_fixed = (num_sig << (xf - wfrac - 2)) >> shift_dn  # num aligned to den's scale 2**-xf

        # Magnitude (always via the engine): x_K = gain*hypot(num_fixed, den_fixed); descale by 1/gain == KINV (scale
        # 2**-xf each), so M = x_K*KINV is hypot at scale 2**-2xf, then carry the den binade e_den plus the +2 of 1/4.
        x_k, y_k, z_k = cordic_vector(spec, den_fixed, num_fixed, n)
        mag_prod = x_k * kinv_mag  # x_K * kinv_mag -> M at 2**-(xf+kinv_s)
        wmag_m = 2 * xf + 4  # holds M for normshift (value-invariant to field width)
        exp_off_m = (wmag_m - 1) - (xf + kinv_s) + e_den + 2  # read M back; den binade + 1/4 pre-scale undone
        mag_bits = fixed_to_float_ref(fmt, 0, mag_prod, exp_off_m, wmag_m)

        # Quotient fractional budget F = 2*ceil(xf/2): q = floor(|y_K|*2**F/x_K) must carry >= wman significant bits.
        # The divide is truncating floor division (radix-independent), so Python //,% match the radix-4 divider exactly.
        f_bits = 2 * ((xf_div + 1) // 2)  # divider F from atan2's OWN xf (== xf today)

        # theta. Bypass only the near-+x-axis tiny-theta corner; everywhere else theta sits near a boundary
        # (+-1/4, +-1/2) and the fixed-turns path (scale 2**-zf, zf >> wman) has ample relative precision.
        tiny_shift = atan2_bypass_shift(fmt)
        if (not swap) and (sx == 0) and (shift_dn > tiny_shift):
            # theta ~= (|y|/|x|)*INV_TAU (atan(r) ~= r): one truncating divide (F frac bits + sticky), *INV_TAU, then a
            # single RTNE pack consuming the sticky. F >= wman+GUARD_DIV keeps the truncation below the round bit.
            r = (num_sig << f_bits) // den_sig  # floor((|y|/|x|)*2**F), F frac bits (+ possible int bit)
            sticky = 1 if ((num_sig << f_bits) % den_sig) else 0
            prod_j = (r * inv_tau) | sticky  # r * inv_tau; sticky in bit 0 (RTNE)
            wmag_b = 2 * xf + 4  # same renormalize field as residual / magnitude paths
            exp_off_b = (wmag_b - 1) + (e_num - e_den) - f_bits - invtau_s  # read r*inv_tau back at 2**-(F+invtau_s)
            return Atan2Result(Zkf(fmt, fixed_to_float_ref(fmt, sy, prod_j, exp_off_b, wmag_b)), Zkf(fmt, mag_bits))

        # Residual correction: a0 = z_K + (y_K/x_K)*INV_TAU at the angle scale 2**-zf. The divide is truncating (matches
        # the folded radix-4 fixed-point divider); x_K > 0 always, y_K is signed (vectoring drives y through 0).
        qf = f_bits  # quotient fractional bits (>= wman+GUARD_DIV)
        aq = -y_k if y_k < 0 else y_k
        q = (aq << qf) // x_k
        # q*inv_tau is at scale 2**-(qf + invtau_s); the right-shift to the angle scale 2**-zf is the difference
        # qf + invtau_s - zf (>= 0 for every supported WMAN; the (-sh_d) guard mirrors the RTL).
        sh_d = qf + invtau_s - zf
        qti = q * inv_tau
        delta = (qti >> sh_d) if sh_d >= 0 else (qti << (-sh_d))  # |delta| at scale 2**-zf
        a0 = z_k - delta if y_k < 0 else z_k + delta

        # Octant/quadrant unmap into theta (turns, scale 2**-zf): phi1 in [0, 1/4], theta_mag in [0, 1/2], sign = sy.
        quarter = 1 << (zf - 2)
        half = 1 << (zf - 1)
        phi1 = (quarter - a0) if swap else a0
        theta_mag = (half - phi1) if sx else phi1
        wmag_t = zf + 2
        exp_off_t = wmag_t - 1 - zf  # reads theta_mag at scale 2**-zf back as itself
        theta_bits = fixed_to_float_ref(fmt, sy, theta_mag, exp_off_t, wmag_t)
        # fold the generic negative-x-axis limit -1/2 -> +1/2
        return Atan2Result(Zkf(fmt, atan2_canon_half(fmt, theta_bits)), Zkf(fmt, mag_bits))


def atan2_canon_half(fmt: ZkfFormat, theta_bits: int) -> int:
    """
    Fold the out-of-range -1/2-turn to the canonical +1/2. The generic atan2 path packs -1/2 (sign set) as the
    negative-x-axis limit (finite x<0, |y|->0), but the documented output range is the half-open (-0.5, 0.5], which
    excludes -1/2. This is the inverse of atan2_turn's frac==1/2 -> sign 0 rule, mirrored in the RTL turn8 k==4 clamp.
    -1/2 is the only out-of-range value the generic path (|theta| <= 1/2) can produce, so nothing else is affected.
    """
    pos_half = atan2_turn(fmt, 0, Fraction(1, 2))
    return pos_half if theta_bits == (pos_half | (1 << fmt.sign_shift)) else theta_bits


def atan2_special(fmt: ZkfFormat, y_bits: int, x_bits: int) -> tuple[int, int] | None:
    """
    Shared special-case table for atan2 (no NaN; only +0; tiny negatives flush to +0). Returns (theta, mag) bits or
    None for the both-finite-nonzero generic path. Used by BOTH the reference and the mpmath oracle so they agree on
    the exact dyadic constants at the axes/diagonals.
    """
    dy = Zkf(fmt, y_bits)
    dx = Zkf(fmt, x_bits)
    if dx.is_inf or dy.is_inf:
        mag = canonical_inf(fmt, 0)  # hypot with any inf operand is +inf
        if dx.is_inf and dy.is_inf:  # diagonals: +-pi/4 (x>0) / +-3pi/4 (x<0) -> +-1/8 / +-3/8
            return atan2_turn(fmt, dy.negative, Fraction(3, 8) if dx.negative else Fraction(1, 8)), mag
        if dy.is_inf:  # |y|=inf, x finite -> +-1/4 (vertical)
            return atan2_turn(fmt, dy.negative, Fraction(1, 4)), mag
        if dx.negative:  # x=-inf -> half-turn endpoint, canonicalized to +1/2
            return atan2_turn(fmt, 0 if dy.is_zero else dy.negative, Fraction(1, 2)), mag
        return zero(fmt), mag  # x=+inf, y finite -> +-0 -> +0 (no -0)
    if dx.is_zero and dy.is_zero:
        return zero(fmt), zero(fmt)  # atan2(0,0)=+0, hypot=+0
    if dy.is_zero:  # y=0, x finite nonzero: +0 (x>0) / 1/2 (x<0); mag=|x|
        theta = atan2_turn(fmt, 0, Fraction(1, 2)) if dx.negative else zero(fmt)
        return theta, (x_bits & mask(fmt.sign_shift))
    if dx.is_zero:  # x=0, y finite nonzero -> +-1/4; mag=|y|
        return atan2_turn(fmt, dy.negative, Fraction(1, 4)), (y_bits & mask(fmt.sign_shift))
    return None


from ._format import ZkfFormat

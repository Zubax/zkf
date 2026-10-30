from amaranth.hdl import Cat, Const, Module, Mux, Signal, Value, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, bundle, pipe
from ._value import ZkfFormat


class Pack(wiring.Component):
    """
    `y = (-1)**sign * 1.fraction * 2**exp_unbiased` rounded to nearest even and packed; `significand` holds the hidden
    bit, and `guard`, `round` and `sticky` are the discarded tail. `force_zero` and `force_inf` override the result,
    `force_zero` first. A magnitude below `fmt.lowest / 2` gives +0, one from there to `fmt.lowest` gives
    `+-fmt.lowest`, an overflow `+-inf`.

    `exp_is_biased`: `exp_unbiased` holds the biased exponent, signed, and no bias is added. `assume_no_overflow`: the
    caller keeps the biased exponent within the finite range, so no overflow detector is built; `force_inf` and a
    rounding carry out of the largest finite exponent still give an infinity. `saturate_round_carry`: that carry gives
    `+-max` instead, for a caller with an inexact input that must not invent an infinity; the result is then faithful,
    not correctly rounded. `wexp_unbiased` is the width of `exp_unbiased`: at least WEXP, plus one if biased.
    `stage_input` registers the inputs and `stage_output` the result. `classify_ahead` classifies the result ahead of
    `stage_input`'s register, which then holds the class and the exponent: only the rounding adder is left behind it,
    with the significand masked ahead of it. It has no effect without that register.
    """

    in_valid: Signal
    sign: Signal
    force_zero: Signal
    force_inf: Signal
    exp_unbiased: Signal
    significand: Signal
    guard: Signal
    round: Signal
    sticky: Signal
    out_valid: Signal
    y: FloatView

    def __init__(
        self,
        fmt: ZkfFormat,
        *,
        wexp_unbiased: int | None = None,
        exp_is_biased: bool = False,
        assume_no_overflow: bool = False,
        saturate_round_carry: bool = False,
        stage_input: int = 0,
        stage_output: int = 0,
        classify_ahead: bool = False,
    ) -> None:
        self._fmt = fmt
        self._classify_ahead = classify_ahead
        self._wexp_unbiased = fmt.wexp + 2 if wexp_unbiased is None else wexp_unbiased
        self._exp_is_biased = exp_is_biased
        self._assume_no_overflow = assume_no_overflow
        self._saturate_round_carry = saturate_round_carry
        self._stage_input = stage_input
        self._stage_output = stage_output
        super().__init__(
            {
                "in_valid": In(1),
                "sign": In(1),
                "force_zero": In(1),
                "force_inf": In(1),
                "exp_unbiased": In(signed(self._wexp_unbiased)),
                "significand": In(fmt.wman),
                "guard": In(1),
                "round": In(1),
                "sticky": In(1),
                "out_valid": Out(1),
                "y": Out(FloatLayout(fmt)),
            }
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, wexp, wfrac = self._fmt, self._fmt.wexp, self._fmt.wfrac
        wext = self._wexp_unbiased + 1
        ahead = self._classify_ahead and bool(self._stage_input)

        def taken(value: Value, name: str) -> Value:
            if not self._stage_input:
                return value
            captured = Signal(len(value), name=f"{name}_r", reset_less=True)
            m.d.sync += captured.eq(value)
            return captured

        valid = self.in_valid
        if self._stage_input:
            valid = Signal(name="valid_r")
            m.d.sync += valid.eq(self.in_valid)
        sign, significand = self.sign, self.significand
        force_zero, force_inf, exp = self.force_zero, self.force_inf, self.exp_unbiased
        guard, round_bit, sticky = self.guard, self.round, self.sticky
        if not ahead:
            sign, significand = taken(sign, "sign"), taken(significand, "significand")
            force_zero, force_inf = taken(force_zero, "force_zero"), taken(force_inf, "force_inf")
            exp = taken(exp, "exp_unbiased").as_signed()

        # A biased exponent of 0 is in `[min_normal / 2, min_normal)`, which gives `min_normal`.
        biased = Cat(exp, exp[-1]) if self._exp_is_biased else (exp + Const(fmt.bias, wexp))[:wext]
        exp_ext = Signal(wext, name="exp_biased_ext")
        m.d.comb += exp_ext.eq(biased)
        exp_biased, underflow = exp_ext[:wexp], exp_ext[-1]
        high = exp_ext[wexp : self._wexp_unbiased].any() if self._wexp_unbiased > wexp else Const(0, 1)
        above = ~underflow & (high | exp_biased.all())
        overflow = Const(0, 1) if self._assume_no_overflow else above
        at_max = Signal(name="exp_at_max")
        m.d.comb += at_max.eq(exp_biased == fmt.exp_max_finite)

        zero = Signal(name="result_zero")
        m.d.comb += zero.eq(force_zero | (~force_inf & underflow))
        infinity = Signal(name="result_infinity")
        m.d.comb += infinity.eq(~zero & (force_inf | overflow))
        min_normal = Signal(name="result_min_normal")
        m.d.comb += min_normal.eq(~zero & ~force_inf & ~exp_ext.any())
        finite = Signal(name="result_finite")
        m.d.comb += finite.eq(~zero & ~infinity)

        def exponent(name: str, saturate: Value | None, rounded: Value) -> Signal:
            out = Signal(wexp, name=name)
            with m.If(zero):
                m.d.comb += out.eq(0)
            with m.Elif(infinity):
                m.d.comb += out.eq(fmt.exp_inf)
            with m.Elif(min_normal):
                m.d.comb += out.eq(1)
            if saturate is not None:
                with m.Elif(saturate):
                    m.d.comb += out.eq(fmt.exp_max_finite)
            with m.Else():
                m.d.comb += out.eq(rounded)
            return out

        if ahead:
            keep = Signal(name="result_keep")
            m.d.comb += keep.eq(finite & ~min_normal)
            exp_held = exponent("exp_held", None, exp_biased)
            sign, keep, exp_held = taken(sign & ~zero, "sign"), taken(keep, "keep"), taken(exp_held, "exp_held")
            significand = taken(significand, "significand")
            guard, round_bit, sticky = taken(guard, "guard"), taken(round_bit, "round"), taken(sticky, "sticky")
            increment = guard & (round_bit | sticky | significand[0]) & keep
            held = Signal(fmt.wfull, name="expsig_held")
            m.d.comb += held.eq(Cat(significand & keep.replicate(fmt.wman), exp_held))
            rounded = Signal(fmt.wfull, name="expsig_rounded")
            m.d.comb += rounded.eq((held + increment)[: fmt.wfull])
            if self._saturate_round_carry:
                # A saturated result is the held one, picked rather than masked so that no set or reset takes it.
                saturating = Signal(name="result_saturating")
                m.d.comb += saturating.eq(significand.all() & taken(at_max, "at_max"))
                rounded = Mux(saturating, held, rounded)
            out_frac, out_exp = rounded[:wfrac], rounded[fmt.wman :]
        else:
            guard, round_bit, sticky = taken(guard, "guard"), taken(round_bit, "round"), taken(sticky, "sticky")
            # The increment goes into exponent and significand at once, so a carry out of the significand reaches
            # the exponent, and from the largest finite one lands on the infinity with a zero fraction.
            increment = guard & (round_bit | sticky | significand[0])
            rounded = Signal(fmt.wfull, name="expsig_rounded")
            m.d.comb += rounded.eq((Cat(significand, exp_biased) + increment)[: fmt.wfull])
            saturate = Signal(name="result_saturate")
            m.d.comb += saturate.eq(finite & significand.all() & at_max if self._saturate_round_carry else 0)
            normal = Signal(name="result_normal")
            m.d.comb += normal.eq(finite & ~min_normal & ~saturate)
            # The fraction is masked, not selected, which keeps it off the tree that selects the exponent.
            out_exp = exponent("out_exp", saturate, rounded[fmt.wman :])
            out_frac = (rounded[:wfrac] & normal.replicate(wfrac)) | saturate.replicate(wfrac)
            sign = sign & ~zero
        layout = FloatLayout(fmt)
        packed = Signal(layout, name="packed")
        m.d.comb += packed.eq(bundle(layout, frac=out_frac, exp=out_exp, sign=sign))
        out_valid, y = pipe(m, "out", valid, layout, packed, self._stage_output)
        m.d.comb += self.out_valid.eq(out_valid)
        m.d.comb += self.y.eq(y)
        return m


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
    """The bits that `Pack` gives for an unbiased exponent."""
    exp_biased = exp_unbiased + fmt.bias
    exp_underflow_zero = exp_unbiased < (fmt.min_exp_unbiased - 1)
    exp_one_below_min = exp_unbiased == (fmt.min_exp_unbiased - 1)
    exp_overflow = exp_unbiased > fmt.max_exp_unbiased

    round_increment = bool(guard and (round_bit or sticky or (significand_value & 1)))
    rounded_ext = (significand_value & ((1 << fmt.wman) - 1)) + (1 if round_increment else 0)
    round_carry = (rounded_ext >> fmt.wman) & 1
    exp_round_overflow = (exp_biased == fmt.exp_max_finite) and bool(round_carry)
    saturated = saturate_round_carry and exp_round_overflow
    infinity = bool(force_inf or exp_overflow or (exp_round_overflow and not saturate_round_carry))

    result_zero = bool(force_zero or ((not force_inf) and exp_underflow_zero))
    result_infinity = (not result_zero) and infinity
    result_min_normal = (not result_zero) and (not result_infinity) and (not force_inf) and exp_one_below_min

    if result_zero:
        return 0
    if result_infinity:
        return fmt.inf(sign).bits
    if result_min_normal:
        return fmt.pack(sign, 1, 0).bits
    if saturated:
        return fmt.pack(sign, fmt.exp_max_finite, fmt.frac_mask).bits

    exp_rounded = (exp_biased + round_carry) & ((1 << fmt.wexp) - 1)
    return fmt.pack(sign, exp_rounded, rounded_ext & fmt.frac_mask).bits

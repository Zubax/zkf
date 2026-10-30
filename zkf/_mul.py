from dataclasses import dataclass, field

from amaranth.hdl import Module, Mux, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import Attrs, FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._pack import Pack, pack_reference
from ._pmul import Pmul, check
from ._value import Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class Mul(Operator):
    """`y = a * b`."""

    operator = "mul"
    stage_input: int = 0
    stage_core_mul: int = 0
    """
    Taken in this order: the output, the operands' capture, the grid of products (a lone multiplier's pipeline
    register), then the levels of the grid's sum.
    """
    stage_output: int = 0
    wmultiplier: int
    """The operand width of the target's multipliers."""
    mul_attrs: Attrs | None = field(default=None, compare=False)
    """
    On the last input stage at `stage_core_mul=1`, which the multiplier takes the operands from; from 2 on, on the
    product's capture, where a register follows the multipliers (a grid's from 3 on).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wmultiplier < 8:
            raise ValueError(f"wmultiplier={self.wmultiplier} is below 8")
        wman = self.fmt.wman
        check("stage_core_mul", self.stage_core_mul, wman, wman, signed=False, wmultiplier=self.wmultiplier)

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + self.stage_core_mul + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature({"in_valid": In(1), "a": In(f), "b": In(f), "out_valid": Out(1), "y": Out(f)})

    def component(self) -> _Mul:
        return _Mul(self)

    def evaluate(self, a: Zkf, b: Zkf) -> Zkf:
        if a.fmt != self.fmt or b.fmt != self.fmt:
            raise ValueError(f"an operand is not of {self.fmt}")
        fmt = self.fmt
        result_zero = a.is_zero or b.is_zero
        result_inf = (not result_zero) and (a.is_inf or b.is_inf)

        product = a.significand() * b.significand()
        high = product >> (2 * fmt.wman - 1)  # the product of two significands is below 4
        drop = fmt.wman - 1 + high
        tail = (product >> (drop - 1)) & 1, (product >> (drop - 2)) & 1, int(product & ((1 << (drop - 2)) - 1) != 0)
        exp = a.exp + b.exp - 2 * fmt.bias + high
        significand = (product >> drop) & ((1 << fmt.wman) - 1)
        sign = a.negative ^ b.negative
        return Zkf(fmt, pack_reference(fmt, sign, int(result_zero), int(result_inf), exp, significand, *tail))


class _Mul(wiring.Component):
    in_valid: Signal
    a: FloatView
    b: FloatView
    out_valid: Signal
    y: FloatView

    def __init__(self, model: Mul) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        wman, wexp_biased = fmt.wman, fmt.wexp + 2

        direct = model.stage_core_mul == 1
        operands = data.StructLayout({"a": FloatLayout(fmt), "b": FloatLayout(fmt)})
        given = bundle(operands, a=self.a, b=self.b)
        valid, q = pipe(m, "in", self.in_valid, operands, given, model.stage_input, model.mul_attrs if direct else None)
        a, b = q.a, q.b

        # The exponent stays biased: one bias is subtracted here, and the packer adds none.
        side = data.StructLayout({"sign": 1, "exp": signed(wexp_biased), "zero": 1, "inf": 1})
        zero = Signal(name="result_zero")
        m.d.comb += zero.eq(a.is_zero | b.is_zero)
        m.submodules.pmul = pmul = Pmul(
            wman,
            wman,
            signed=False,
            wmultiplier=model.wmultiplier,
            stages=model.stage_core_mul,
            sideband=side,
            attrs=model.mul_attrs,
        )
        m.d.comb += pmul.in_valid.eq(valid)
        m.d.comb += pmul.a.eq(a.significand)
        m.d.comb += pmul.b.eq(b.significand)
        m.d.comb += pmul.sb_in.sign.eq(a.sign ^ b.sign)
        m.d.comb += pmul.sb_in.exp.eq(a.exp + b.exp - fmt.bias)
        m.d.comb += pmul.sb_in.zero.eq(zero)
        m.d.comb += pmul.sb_in.inf.eq(~zero & (a.is_inf | b.is_inf))

        product, carried = pmul.p, pmul.p[-1]
        m.submodules.pack = pack = Pack(fmt, exp_is_biased=True, stage_output=model.stage_output)
        m.d.comb += pack.in_valid.eq(pmul.out_valid)
        m.d.comb += pack.sign.eq(pmul.sb_out.sign)
        m.d.comb += pack.force_zero.eq(pmul.sb_out.zero)
        m.d.comb += pack.force_inf.eq(pmul.sb_out.inf)
        m.d.comb += pack.exp_unbiased.eq((pmul.sb_out.exp + carried)[:wexp_biased].as_signed())
        m.d.comb += pack.significand.eq(Mux(carried, product[wman:], product[wman - 1 : -1]))
        m.d.comb += pack.guard.eq(Mux(carried, product[wman - 1], product[wman - 2]))
        m.d.comb += pack.round.eq(Mux(carried, product[wman - 2], product[wman - 3]))
        m.d.comb += pack.sticky.eq(Mux(carried, product[: wman - 2].any(), product[: wman - 3].any()))
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        return m

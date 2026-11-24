from dataclasses import dataclass, field

from amaranth.hdl import Module, Mux, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import Attrs, FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._pack import Pack
from ._fixed_to_float import fixed_to_float_reference
from ._pmul import Pmul, latency
from ._value import Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class Mul(Operator):
    """`y = a * b`."""

    operator = "mul"
    stage_input: int = 0
    stage_core_mul_level: int = 0
    """
    The registers of the product, by level:

        level         registers
        0             none
        1             O
        2             G
        3             S
        4             G+O
        5             C+S
        6             C+O
        7             G+G2
        8             G+S
        9             G+S+O
        10            C+G+O
        11            C+G+S
        12            C+G+S+O
        13            C+G+G2+S+O
        14, 15, ...   each further level of the grid's sum: first in place of G2, then with it

    C captures the operands ahead of the multipliers; G registers the product of each multiplier, G2 again; S cuts the
    grid's sum after its first level, and is O where the sum has one level or none; O registers the whole product. On a
    lone multiplier, G, G2 and O are one chain behind it. The latency is the count of registers; a level past the last
    is refused.
    Which level of a latency serves best depends on the logic around the product and on the flow.
    """
    stage_output: int = 0
    wmultiplier: int
    """The operand width of the target's multipliers."""
    mul_attrs: Attrs | None = field(default=None, compare=False)
    """On the product's operand capture, at the levels that have it."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        self.timing  # refuses a level outside the product's range

    @property
    def timing(self) -> Timing:
        wman = self.fmt.wman
        product = latency(
            self.stage_core_mul_level,
            wman,
            wman,
            signed=False,
            wmultiplier=self.wmultiplier,
            knob="stage_core_mul_level",
        )
        return Timing(self.stage_input + product + self.stage_output, 1)

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
        if a.is_zero or b.is_zero:
            return Zkf(fmt, 0)
        sign = a.negative ^ b.negative
        if a.is_inf or b.is_inf:
            return fmt.inf(sign)
        product = a.significand() * b.significand()
        exp = a.exp + b.exp - 2 * fmt.bias + 1
        return fixed_to_float_reference(fmt, sign, product, exp, 2 * fmt.wman)


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

        operands = data.StructLayout({"a": FloatLayout(fmt), "b": FloatLayout(fmt)})
        given = bundle(operands, a=self.a, b=self.b)
        valid, q = pipe(m, "in", self.in_valid, operands, given, model.stage_input)
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
            level=model.stage_core_mul_level,
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
        m.d.comb += pack.exp_unbiased.eq(pmul.sb_out.exp + carried)
        m.d.comb += pack.significand.eq(Mux(carried, product[wman:], product[wman - 1 : -1]))
        m.d.comb += pack.guard.eq(Mux(carried, product[wman - 1], product[wman - 2]))
        m.d.comb += pack.round.eq(Mux(carried, product[wman - 2], product[wman - 3]))
        m.d.comb += pack.sticky.eq(Mux(carried, product[: wman - 2].any(), product[: wman - 3].any()))
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        return m

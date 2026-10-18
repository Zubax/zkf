from dataclasses import dataclass

from amaranth.hdl import Cat, Const, Module, Mux, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._pack import Pack
from ._value import Zkf, ZkfFormat


@dataclass(frozen=True, kw_only=True, eq=False)
class Resize(Operator):
    """
    `y = a` converted from `fmt_in` to `fmt`: exact if `fmt` holds every value of `fmt_in`, else rounded to nearest
    even, with `+-inf` above the range and +0 or `+-fmt.lowest` below it. Zeros and infinities come out canonical.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    operator = "resize"
    fmt_in: ZkfFormat
    stage_input: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        a, y = FloatLayout(self.fmt_in), FloatLayout(self.fmt)
        return wiring.Signature({"in_valid": In(1), "a": In(a), "out_valid": Out(1), "y": Out(y)})

    def component(self) -> _Resize:
        return _Resize(self)

    def evaluate(self, a: Zkf) -> Zkf:
        if a.fmt != self.fmt_in:
            raise ValueError(f"an operand of {a.fmt} where {self.fmt_in} is due")
        fmt = self.fmt
        if a.is_zero:
            return Zkf(fmt, 0)
        if a.is_inf:
            return fmt.inf(a.negative)
        return fmt.encode(a.to_fraction())


class _Resize(wiring.Component):
    in_valid: Signal
    a: FloatView
    out_valid: Signal
    y: FloatView

    def __init__(self, model: Resize) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, out, source = self._model, self._model.fmt, self._model.fmt_in
        valid, a = pipe(m, "in", self.in_valid, FloatLayout(source), self.a, model.stage_input)

        if out.wman >= source.wman and out.wexp >= source.wexp:
            # Nothing is rounded and nothing leaves the range.
            layout = FloatLayout(out)
            exp = (Cat(a.exp, Const(0, out.wexp - source.wexp)) + (out.bias - source.bias))[: out.wexp]
            frac = Cat(Const(0, out.wfrac - source.wfrac), a.frac)
            infinity = bundle(layout, frac=Const(0, out.wfrac), exp=Const(out.exp_inf, out.wexp), sign=a.sign)
            normal = bundle(layout, frac=frac, exp=exp, sign=a.sign)
            finite = Mux(a.is_inf, infinity.as_value(), normal.as_value())
            valid, y = pipe(
                m, "out", valid, layout, layout(Mux(a.is_zero, Const(0, out.wfull), finite)), model.stage_output
            )
            m.d.comb += self.out_valid.eq(valid)
            m.d.comb += self.y.eq(y)
            return m

        # `weu` holds the unbiased exponent of either format.
        weu = max(out.wexp + 2, source.wexp + 1)
        drop = source.wman - out.wman
        significand = a.significand
        m.submodules.pack = pack = Pack(out, wexp_unbiased=weu, stage_output=model.stage_output)
        m.d.comb += pack.in_valid.eq(valid)
        m.d.comb += pack.sign.eq(a.sign)
        m.d.comb += pack.force_zero.eq(a.is_zero)
        m.d.comb += pack.force_inf.eq(a.is_inf)
        m.d.comb += pack.exp_unbiased.eq((Cat(a.exp, Const(0, weu - source.wexp)) - source.bias)[:weu].as_signed())
        m.d.comb += pack.significand.eq(Cat(Const(0, -drop), significand) if drop <= 0 else significand[drop:])
        m.d.comb += pack.guard.eq(significand[drop - 1] if drop >= 1 else Const(0, 1))
        m.d.comb += pack.round.eq(significand[drop - 2] if drop >= 2 else Const(0, 1))
        m.d.comb += pack.sticky.eq(significand[: drop - 2].any() if drop >= 3 else Const(0, 1))
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        return m

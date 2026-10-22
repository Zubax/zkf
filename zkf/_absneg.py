from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Const, Module, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import Zkf


class AbsNegResult(NamedTuple):
    absolute: Zkf
    negated: Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class AbsNeg(Operator):
    """
    `absolute = |x|` and `negated = -x`, by clearing and flipping the sign bit; not canonicalized, so `-(+0)` is a zero
    with the sign bit set.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    operator = "absneg"
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
        f = FloatLayout(self.fmt)
        return wiring.Signature(
            {"in_valid": In(1), "x": In(f), "out_valid": Out(1), "absolute": Out(f), "negated": Out(f)}
        )

    def component(self) -> _AbsNeg:
        return _AbsNeg(self)

    def evaluate(self, x: Zkf) -> AbsNegResult:
        if x.fmt != self.fmt:
            raise ValueError(f"an operand of {x.fmt} where {self.fmt} is due")
        fmt = self.fmt
        return AbsNegResult(fmt.pack(0, x.exp, x.frac), fmt.pack(not x.negative, x.exp, x.frac))


class _AbsNeg(wiring.Component):
    in_valid: Signal
    x: FloatView
    out_valid: Signal
    absolute: FloatView
    negated: FloatView

    def __init__(self, model: AbsNeg) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, layout = self._model, FloatLayout(self._model.fmt)
        valid, x = pipe(m, "in", self.in_valid, layout, self.x, model.stage_input)
        flipped = Signal(layout, name="flipped")
        m.d.comb += flipped.eq(bundle(layout, frac=x.frac, exp=x.exp, sign=~x.sign))
        valid, negated = pipe(m, "out", valid, layout, flipped, model.stage_output)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.negated.eq(negated)
        m.d.comb += self.absolute.eq(bundle(layout, frac=negated.frac, exp=negated.exp, sign=Const(0, 1)))
        return m

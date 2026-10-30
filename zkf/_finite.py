from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Const, Module, Mux, Signal
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import Zkf


class FiniteResult(NamedTuple):
    finite: bool
    saturated: Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class Finite(Operator):
    """`finite = not isinf(x)`; `saturated` is `x` bit for bit if finite, else `copysign(max, x)`."""

    operator = "finite"
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
            {"in_valid": In(1), "x": In(f), "out_valid": Out(1), "finite": Out(1), "saturated": Out(f)}
        )

    def component(self) -> _Finite:
        return _Finite(self)

    def evaluate(self, x: Zkf) -> FiniteResult:
        if x.fmt != self.fmt:
            raise ValueError(f"an operand of {x.fmt} where {self.fmt} is due")
        fmt = self.fmt
        if not x.is_inf:
            return FiniteResult(True, x)
        return FiniteResult(False, fmt.pack(x.negative, fmt.exp_max_finite, fmt.frac_mask))


class _Finite(wiring.Component):
    in_valid: Signal
    x: FloatView
    out_valid: Signal
    finite: Signal
    saturated: FloatView

    def __init__(self, model: Finite) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        layout = FloatLayout(fmt)
        valid, x = pipe(m, "in", self.in_valid, layout, self.x, model.stage_input)
        finite = Signal(name="x_finite")
        m.d.comb += finite.eq(~x.is_inf)
        largest = bundle(
            layout, frac=Const(fmt.frac_mask, fmt.wfrac), exp=Const(fmt.exp_max_finite, fmt.wexp), sign=x.sign
        )
        result = data.StructLayout({"finite": 1, "saturated": layout})
        chosen = layout(Mux(finite, x.as_value(), largest.as_value()))
        payload = Signal(result, name="result")
        m.d.comb += payload.eq(bundle(result, finite=finite, saturated=chosen))
        valid, out = pipe(m, "out", valid, result, payload, model.stage_output)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.finite.eq(out.finite)
        m.d.comb += self.saturated.eq(out.saturated)
        return m

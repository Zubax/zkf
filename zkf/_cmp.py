from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Cat, Module, Mux, Signal, Value
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import Zkf


class CmpResult(NamedTuple):
    a_gt_b: bool
    a_eq_b: bool
    a_lt_b: bool
    min: Zkf
    max: Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class Cmp(Operator):
    """
    `a_gt_b = a > b`, `a_eq_b = a == b`, `a_lt_b = a < b`, `min = min(a, b)`, `max = max(a, b)`; `min` and `max` are
    operands as given, not canonicalized. All zeros are equal, and so are infinities of one sign whatever their
    fractions.
    """

    operator = "cmp"
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
            {
                "in_valid": In(1),
                "a": In(f),
                "b": In(f),
                "out_valid": Out(1),
                "a_gt_b": Out(1),
                "a_eq_b": Out(1),
                "a_lt_b": Out(1),
                "min": Out(f),
                "max": Out(f),
            }
        )

    def component(self) -> _Cmp:
        return _Cmp(self)

    def evaluate(self, a: Zkf, b: Zkf) -> CmpResult:
        if a.fmt != self.fmt or b.fmt != self.fmt:
            raise ValueError(f"an operand is not of {self.fmt}")
        a_key, b_key = _ordered_key(a), _ordered_key(b)
        lt = a_key < b_key
        return CmpResult(a_key > b_key, a_key == b_key, lt, a if lt else b, b if lt else a)


def _ordered_key(x: Zkf) -> int:
    canonical, fmt = x.canonicalize(), x.fmt
    return ((1 << fmt.wfull) - 1 - canonical.bits) if canonical.negative else (canonical.bits | (1 << fmt.sign_shift))


def _key(x: FloatView) -> Value:
    """Sign and magnitude as an unsigned number of the same order."""
    magnitude = Cat(x.frac, x.exp)
    return Cat(magnitude ^ x.sign.replicate(len(magnitude)), ~x.sign)


class _Cmp(wiring.Component):
    in_valid: Signal
    a: FloatView
    b: FloatView
    out_valid: Signal
    a_gt_b: Signal
    a_eq_b: Signal
    a_lt_b: Signal
    min: FloatView
    max: FloatView

    def __init__(self, model: Cmp) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        layout = FloatLayout(fmt)
        operands = data.StructLayout({"a": layout, "b": layout})
        valid, q = pipe(m, "in", self.in_valid, operands, bundle(operands, a=self.a, b=self.b), model.stage_input)
        a, b = q.a, q.b

        # The keys skip the classification, which `same` applies after the wide comparison instead of before it. `lt` is
        # the borrow of a subtraction, which maps onto the carry chain.
        a_key = Signal(fmt.wfull, name="a_key")
        m.d.comb += a_key.eq(_key(a))
        b_key = Signal(fmt.wfull, name="b_key")
        m.d.comb += b_key.eq(_key(b))
        same = Signal(name="same")
        m.d.comb += same.eq((a.is_zero & b.is_zero) | (a.is_inf & b.is_inf & ~(a.sign ^ b.sign)))
        eq = Signal(name="eq")
        m.d.comb += eq.eq((a_key == b_key) | same)
        lt = Signal(name="lt")
        m.d.comb += lt.eq((a_key - b_key)[-1] & ~same)
        result = data.StructLayout({"a_gt_b": 1, "a_eq_b": 1, "a_lt_b": 1, "min": layout, "max": layout})
        found = bundle(
            result,
            a_gt_b=~(lt | eq),
            a_eq_b=eq,
            a_lt_b=lt,
            min=layout(Mux(lt, a.as_value(), b.as_value())),
            max=layout(Mux(lt, b.as_value(), a.as_value())),
        )
        payload = Signal(result, name="result")
        m.d.comb += payload.eq(found)
        valid, out = pipe(m, "out", valid, result, payload, model.stage_output)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += [self.a_gt_b.eq(out.a_gt_b), self.a_eq_b.eq(out.a_eq_b), self.a_lt_b.eq(out.a_lt_b)]
        m.d.comb += [self.min.eq(out.min), self.max.eq(out.max)]
        return m

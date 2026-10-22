from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Cat, Module, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import UnsupportedFormat, Zkf


class Ilog2Result(NamedTuple):
    y: int
    zero: bool
    infinity: bool
    negative: bool


@dataclass(frozen=True, kw_only=True, eq=False)
class Ilog2(Operator):
    """
    `y = exp - bias`, the exponent field unbiased, as a signed `wint`-bit integer: `-bias` for a zero and `bias + 1` for
    an infinity, which `zero` and `infinity` flag. `negative` is the sign bit, also of a zero; the fraction is not read.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    operator = "ilog2"
    wint: int = 32
    stage_input: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wint < self.fmt.wexp + 1:
            raise UnsupportedFormat(f"WINT={self.wint} is below {self.fmt.wexp + 1}")

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        return wiring.Signature(
            {
                "in_valid": In(1),
                "a": In(FloatLayout(self.fmt)),
                "out_valid": Out(1),
                "y": Out(signed(self.wint)),
                "zero": Out(1),
                "infinity": Out(1),
                "negative": Out(1),
            }
        )

    def component(self) -> _Ilog2:
        return _Ilog2(self)

    def evaluate(self, a: Zkf) -> Ilog2Result:
        if a.fmt != self.fmt:
            raise ValueError(f"an operand of {a.fmt} where {self.fmt} is due")
        return Ilog2Result(a.exp - self.fmt.bias, a.is_zero, a.is_inf, a.negative)


class _Ilog2(wiring.Component):
    in_valid: Signal
    a: FloatView
    out_valid: Signal
    y: Signal
    zero: Signal
    infinity: Signal
    negative: Signal

    def __init__(self, model: Ilog2) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        read = data.StructLayout({"exp": fmt.wexp, "sign": 1})
        taken = bundle(read, exp=self.a.exp, sign=self.a.sign)
        valid, q = pipe(m, "in", self.in_valid, read, taken, model.stage_input)
        result = data.StructLayout({"unbiased": signed(fmt.wexp + 1), "zero": 1, "infinity": 1, "negative": 1})
        found = bundle(
            result,
            unbiased=(q.exp - fmt.bias)[: fmt.wexp + 1],
            zero=q.exp == 0,
            infinity=q.exp.all(),
            negative=q.sign,
        )
        payload = Signal(result, name="result")
        m.d.comb += payload.eq(found)
        valid, out = pipe(m, "out", valid, result, payload, model.stage_output)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.y.eq(Cat(out.unbiased, out.unbiased[-1].replicate(model.wint - fmt.wexp - 1)).as_signed())
        m.d.comb += [self.zero.eq(out.zero), self.infinity.eq(out.infinity), self.negative.eq(out.negative)]
        return m

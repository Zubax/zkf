import enum
from dataclasses import dataclass

from amaranth.hdl import Const, Module, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._fixed_to_float import FixedToFloat
from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import UnsupportedFormat, Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class FromInt(Operator):
    """`y = float(a)` for a signed `wint`-bit `a`, rounded to nearest even."""

    class CoreStages(enum.Flag):
        """The registers of `stage_core`, in path order."""

        ABS = enum.auto()
        """`|a|`, ahead of the normalizer."""
        NORMALIZE = enum.auto()
        """Behind the widest level of the normalizer's shift."""
        PACK = enum.auto()
        """The packer's input: the normalized magnitude and its exponent."""

    operator = "from_int"
    wint: int = 32
    stage_input: int = 0
    stage_core: CoreStages = CoreStages(0)
    """See `FromInt.CoreStages`."""
    stage_output: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wint < 2:
            raise UnsupportedFormat(f"WINT={self.wint} is below 2")

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + len(self.stage_core) + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature({"in_valid": In(1), "a": In(signed(self.wint)), "out_valid": Out(1), "y": Out(f)})

    def component(self) -> _FromInt:
        return _FromInt(self)

    def evaluate(self, a: int) -> Zkf:
        if not -(1 << (self.wint - 1)) <= a < 1 << (self.wint - 1):
            raise ValueError(f"a={a} does not fit {self.wint} signed bits")
        return self.fmt.encode(a)


class _FromInt(wiring.Component):
    in_valid: Signal
    a: Signal
    out_valid: Signal
    y: FloatView

    def __init__(self, model: FromInt) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt, wint = self._model, self._model.fmt, self._model.wint
        # Wide enough for the integer and for guard, round and sticky below the significand.
        wx = max(wint, fmt.wman + 3)
        # `top` is the biased exponent of a magnitude with its top bit set; `weu` keeps it positive as a signed number,
        # or the packer would take the widest integers for underflows.
        top = wx - 1 + fmt.bias
        weu = max(ceil_log2(top + 1) + 1, fmt.wexp + 2)

        valid, a = pipe(m, "in", self.in_valid, signed(wint), self.a, model.stage_input)
        # `|a|` by inversion and increment, one carry chain; that of the most negative `a` fits `wint` unsigned bits.
        negative = a[-1]
        magnitude = ((a.as_unsigned() ^ negative.replicate(wint)) + negative)[:wint]
        m.submodules.fixed_to_float = fixed_to_float = FixedToFloat(
            fmt,
            wmag=wx,
            weu=weu,
            exp_is_biased=True,
            stage_wide=int(FromInt.CoreStages.NORMALIZE in model.stage_core),
            stage_pack=int(FromInt.CoreStages.PACK in model.stage_core),
            stage_output=model.stage_output,
        )
        held = data.StructLayout({"sign": 1, "mag": wint})
        taken = bundle(held, sign=negative, mag=magnitude)
        valid, h = pipe(m, "abs", valid, held, taken, int(FromInt.CoreStages.ABS in model.stage_core))
        m.d.comb += fixed_to_float.in_valid.eq(valid)
        m.d.comb += fixed_to_float.sign.eq(h.sign)
        m.d.comb += fixed_to_float.force_inf.eq(0)
        m.d.comb += fixed_to_float.exp_offset.eq(Const(top, signed(weu)))
        m.d.comb += fixed_to_float.mag.eq(h.mag)
        m.d.comb += self.out_valid.eq(fixed_to_float.out_valid)
        m.d.comb += self.y.eq(fixed_to_float.y)
        return m

import enum
from dataclasses import dataclass
from typing import Any

from amaranth.hdl import Cat, Const, Module, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import UnsupportedFormat, Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class MulIlog2(Operator):
    """
    `y = a * 2**k` (ldexp) for a signed `wk`-bit `k`, exact within the normal range. Any `k` is legal: a result above
    the range gives `+-inf`, one below it +0 or `+-min_normal`. The default `wk`, WEXP + 1, spans every shift that keeps
    some input in range.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core`: the registers of `MulIlog2.CoreStages` that are built.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    class CoreStages(enum.Flag):
        """The registers of `stage_core`."""

        DECODE = enum.auto()
        """The decoding, ahead of the selection of the result."""

    operator = "mul_ilog2"
    wk: int | None = None
    stage_input: int = 0
    stage_core: CoreStages = CoreStages(0)
    stage_output: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.config["wk"] < 1:
            raise UnsupportedFormat(f"WK={self.wk} is below 1")

    @property
    def config(self) -> dict[str, Any]:
        return super().config | {"wk": self.fmt.wexp + 1 if self.wk is None else self.wk}

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + len(self.stage_core) + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        k = signed(self.config["wk"])
        return wiring.Signature({"in_valid": In(1), "a": In(f), "k": In(k), "out_valid": Out(1), "y": Out(f)})

    def component(self) -> _MulIlog2:
        return _MulIlog2(self)

    def evaluate(self, a: Zkf, k: int) -> Zkf:
        if a.fmt != self.fmt:
            raise ValueError(f"an operand of {a.fmt} where {self.fmt} is due")
        wk = self.config["wk"]
        if not -(1 << (wk - 1)) <= k < 1 << (wk - 1):
            raise ValueError(f"k={k} does not fit {wk} signed bits")
        fmt = self.fmt
        if a.is_zero:
            return Zkf(fmt, 0)
        if a.is_inf:
            return fmt.inf(a.negative)
        new_exp = a.exp + k
        if new_exp < 0:
            return Zkf(fmt, 0)
        if new_exp == 0:
            return fmt.pack(a.negative, 1, 0)
        if new_exp > fmt.exp_max_finite:
            return fmt.inf(a.negative)
        return fmt.pack(a.negative, new_exp, a.frac)


class _MulIlog2(wiring.Component):
    in_valid: Signal
    a: FloatView
    k: Signal
    out_valid: Signal
    y: FloatView

    def __init__(self, model: MulIlog2) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        wexp, wfrac, wk = fmt.wexp, fmt.wfrac, model.config["wk"]
        narrow = wexp + 1

        operands = data.StructLayout({"a": FloatLayout(fmt), "k": signed(wk)})
        valid, q = pipe(m, "in", self.in_valid, operands, bundle(operands, a=self.a, k=self.k), model.stage_input)
        a, k = q.a, q.k

        a_zero = Signal(name="a_zero")
        m.d.comb += a_zero.eq(a.is_zero)
        a_inf = Signal(name="a_inf")
        m.d.comb += a_inf.eq(a.is_inf)
        special = Signal(name="a_special")
        m.d.comb += special.eq(a_zero | a_inf)
        far_up = far_down = Const(0, 1)
        if wk <= narrow:
            k_narrow = Cat(k, k[-1].replicate(narrow - wk)).as_signed()
        else:
            # The bits cut off a wide `k` already force every normal input to a zero or an infinity.
            upper = k[narrow - 1 :]
            in_range = Signal(name="k_in_range")
            m.d.comb += in_range.eq((~(upper ^ k[-1].replicate(len(upper)))).all())
            k_narrow = k[:narrow].as_signed()
            far_up = Signal(name="k_far_up")
            m.d.comb += far_up.eq(~in_range & ~k[-1])
            far_down = Signal(name="k_far_down")
            m.d.comb += far_down.eq(~in_range & k[-1])
        moved = Signal(signed(wexp + 2), name="new_exp_acc")
        m.d.comb += moved.eq(a.exp + k_narrow)
        # `moved >= exp_inf` as the borrow of a subtraction, on a carry chain: a wide comparator mapped poorly in some
        # tools. Every reachable difference fits `wexp + 2` bits.
        short = (moved - Const(fmt.exp_inf, wexp))[wexp + 1]
        overflow = ~special & (far_up | (~far_down & ~short))
        underflow = ~special & (far_down | (~far_up & moved[-1]))

        decoded = data.StructLayout({"frac": wfrac, "exp": wexp, "min_normal": 1, "inf": 1, "zero": 1, "sign": 1})
        found = Signal(decoded, name="found")
        m.d.comb += found.eq(
            bundle(
                decoded,
                frac=a.frac,
                exp=moved[:wexp],
                min_normal=~special & ~moved.any(),
                inf=a_inf | overflow,
                zero=a_zero | underflow,
                sign=a.sign,
            )
        )
        valid, d = pipe(m, "decode", valid, decoded, found, int(MulIlog2.CoreStages.DECODE in model.stage_core))

        # A zero and an infinity exclude each other and take priority over `min_normal`, which a `k` out of range can
        # raise beside them.
        layout = FloatLayout(fmt)
        result = Signal(layout, name="y_next")
        nothing = Const(0, wfrac)
        with m.Switch(Cat(d.min_normal, d.inf, d.zero)):
            with m.Case("1--"):
                m.d.comb += result.eq(layout.const(Zkf(fmt, 0)))
            with m.Case("01-"):
                m.d.comb += result.eq(bundle(layout, frac=nothing, exp=Const(fmt.exp_inf, wexp), sign=d.sign))
            with m.Case("001"):
                m.d.comb += result.eq(bundle(layout, frac=nothing, exp=Const(1, wexp), sign=d.sign))
            with m.Default():
                m.d.comb += result.eq(bundle(layout, frac=d.frac, exp=d.exp, sign=d.sign))
        valid, y = pipe(m, "out", valid, layout, result, model.stage_output)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.y.eq(y)
        return m

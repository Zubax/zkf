import enum
import math
from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Cat, Const, Module, Mux, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._value import UnsupportedFormat, Zkf


@enum.unique
class RoundMode(enum.IntEnum):
    """Round-to-integer modes; the integer values are those of the 2-bit `round_mode` port of `Rint`."""

    NEAREST_EVEN = 0
    FLOOR = 1
    CEIL = 2
    TRUNC = 3


class RintResult(NamedTuple):
    y_float: Zkf
    y_int: int


@dataclass(frozen=True, kw_only=True, eq=False)
class Rint(Operator):
    """
    `a` rounded to an integer by `round_mode`, as `y_float` of the format of `a` and as `y_int`, a signed `wint`-bit
    integer. An infinity and a zero result come out canonical, and a rounded value beyond the format gives `+-inf`;
    `y_int` saturates on an infinity and on overflow.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core`: the registers of `Rint.Core` that are built.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    class Core(enum.Flag):
        """The registers of `stage_core`."""

        MASK = enum.auto()
        """The discard mask and the truncated magnitude, ahead of the rounding decision."""

    operator = "rint"
    wint: int = 32
    stage_input: int = 0
    stage_core: Core = Core(0)
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
        return wiring.Signature(
            {
                "in_valid": In(1),
                "a": In(f),
                "round_mode": In(RoundMode),
                "out_valid": Out(1),
                "y_float": Out(f),
                "y_int": Out(signed(self.wint)),
            }
        )

    def component(self) -> _Rint:
        return _Rint(self)

    def evaluate(self, a: Zkf, round_mode: RoundMode) -> RintResult:
        if a.fmt != self.fmt:
            raise ValueError(f"an operand of {a.fmt} where {self.fmt} is due")
        if not 0 <= round_mode <= 3:
            raise ValueError(f"round_mode={round_mode} is outside 0..3")
        int_max = (1 << (self.wint - 1)) - 1
        int_min = -int_max - 1
        if a.is_inf:
            return RintResult(self.fmt.inf(a.negative), int_min if a.negative else int_max)
        rounded = (round, math.floor, math.ceil, math.trunc)[round_mode](a.to_fraction())
        return RintResult(self.fmt.encode(rounded), max(int_min, min(int_max, rounded)))


class _Rint(wiring.Component):
    in_valid: Signal
    a: FloatView
    round_mode: Signal
    out_valid: Signal
    y_float: FloatView
    y_int: Signal

    def __init__(self, model: Rint) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt, wint = self._model, self._model.fmt, self._model.wint
        wexp, wfrac, wfull, bias = fmt.wexp, fmt.wfrac, fmt.wfull, fmt.bias
        layout = FloatLayout(fmt)
        wt = wint - 1  # bits of the integer's magnitude
        # `unit` is the exponent at which the last bit of the significand weighs one; `top` the largest whose magnitude
        # fits `wt` bits. Either less an exponent fits `wa` signed bits.
        unit, top = wfrac + bias, bias + wint - 2
        wpp, wa = ceil_log2(wfrac + 1), ceil_log2(max(unit, top) + 1) + 1

        taken = data.StructLayout({"a": layout, "round_mode": RoundMode})
        operands = bundle(taken, a=self.a, round_mode=self.round_mode)
        valid, q = pipe(m, "in", self.in_valid, taken, operands, model.stage_input)
        sign, exp, frac = q.a.sign, q.a.exp, q.a.frac

        # The classification, the mask of the fraction bits to discard and the truncated magnitude.
        positive = Signal(name="positive")
        m.d.comb += positive.eq(~sign)
        is_inf = Signal(name="is_inf")
        m.d.comb += is_inf.eq(exp.all())
        ordinary = Signal(name="ordinary")
        m.d.comb += ordinary.eq(~is_inf & exp.any())  # neither a zero nor an infinity
        sub_one = Signal(name="sub_one")
        m.d.comb += sub_one.eq(exp < bias)  # a magnitude below one, a zero too: the result is 0 or +-1
        whole = Signal(name="whole")
        m.d.comb += whole.eq(~sub_one)
        rounds = Signal(name="rounds")
        m.d.comb += rounds.eq(ordinary & whole)
        exp_s = Cat(exp, Const(0, wa - wexp)).as_signed()
        fraction_bits = Signal(signed(wa), name="fraction_bits")
        m.d.comb += fraction_bits.eq((Const(unit, signed(wa)) - exp_s)[:wa].as_signed())
        # At `wint = WMAN + 1`, `unit == top` and one difference serves both.
        distance = (Const(top, signed(wa)) - exp_s)[:wa].as_signed()
        shift = fraction_bits
        if unit != top:
            shift = Signal(signed(wa), name="shift")
            m.d.comb += shift.eq(distance)
        beyond = Signal(name="beyond")
        m.d.comb += beyond.eq(is_inf | shift[-1])  # an infinity, or a magnitude of more than `wt` bits
        discarded = Signal(wpp, name="discarded")
        with m.Switch(Cat(fraction_bits <= 0, sub_one | is_inf)):
            with m.Case(0b00):
                m.d.comb += discarded.eq(fraction_bits[:wpp])
            with m.Case(0b01):
                m.d.comb += discarded.eq(0)
            with m.Default():  # an infinity discards its payload
                m.d.comb += discarded.eq(Const(wfrac, wpp))
        if wt == 1:
            aligned = Const(1, 1)
        elif fmt.wman >= wt:
            aligned = Cat(frac[wfrac - (wt - 1) :], Const(1, 1))
        else:
            aligned = Cat(Const(0, wt - fmt.wman), frac, Const(1, 1))
        # `near` rounds to nearest; `away` rounds away from zero whenever anything is discarded.
        near, away = Signal(1, name="near"), Signal(1, name="away")
        with m.Switch(q.round_mode):
            with m.Case(RoundMode.NEAREST_EVEN):
                m.d.comb += near.eq(1)
                m.d.comb += away.eq(0)
            with m.Case(RoundMode.FLOOR):
                m.d.comb += near.eq(0)
                m.d.comb += away.eq(sign)
            with m.Case(RoundMode.CEIL):
                m.d.comb += near.eq(0)
                m.d.comb += away.eq(positive)
            with m.Default():
                m.d.comb += near.eq(0)
                m.d.comb += away.eq(0)

        # `inc_sub`: a magnitude below one rounds to +-1, to nearest only above a half. `has_next`: a positive magnitude
        # of all ones has no successor within the integer. `exp`: that of a half below one, so that the incremented
        # candidate is +-1. `trunc`: all ones beyond the integer, which the XOR with the sign turns into the rail.
        shifted = data.StructLayout(
            {"inc_sub": 1, "near": 1, "away": 1, "has_next": 1, "sign": 1, "sub_one": 1}
            | {"exp": wexp, "frac": wfrac, "mask": wfrac, "trunc": wt}
        )
        ones = Const((1 << wfrac) - 1, wfrac)
        decoded = bundle(
            shifted,
            inc_sub=ordinary & sub_one & (away | (near & (exp == bias - 1) & frac.any())),
            near=rounds & near,
            away=rounds & away,
            has_next=~beyond & ~(positive & ~shift.any() & aligned.all()),
            sign=sign,
            sub_one=sub_one,
            exp=Mux(sub_one, Const(bias - 1, wexp), exp),
            frac=frac,
            mask=~((ones << discarded)[:wfrac]),
            trunc=((aligned >> shift[: ceil_log2(wt) if wt > 1 else 1]) & whole.replicate(wt)) | beyond.replicate(wt),
        )
        payload = Signal(shifted, name="decoded")
        m.d.comb += payload.eq(decoded)
        valid, sh = pipe(m, "mask", valid, shifted, payload, int(Rint.Core.MASK in model.stage_core))

        # The rounding decision; `unit_bit` marks the bit of the significand that weighs one.
        mask, below = sh.mask, Cat(sh.mask[1:], Const(0, 1))
        unit_bit = Cat(Const(1, 1), mask) & ~Cat(mask, Const(0, 1))
        dropped = Signal(wfrac, name="dropped")
        m.d.comb += dropped.eq(sh.frac & mask)
        guard = (dropped & ~below).any()
        odd_or_sticky = (Cat(sh.frac, Const(1, 1)) & (unit_bit | Cat(below, Const(0, 1)))).any()
        inc = Signal(name="inc")
        m.d.comb += inc.eq(sh.inc_sub | (sh.away & dropped.any()) | (sh.near & guard & odd_or_sticky))

        # The decision only selects between a truncated and an incremented candidate of each result. The float's
        # increment is taken on exponent and fraction together with the discarded bits set, so that its carry reaches
        # the exponent and lands on the infinity by itself. `int_base + 1` is both the rounding increment and the
        # negation.
        float_next = (Cat(sh.frac | mask, sh.exp) + 1)[: wfull - 1]
        float_kept = Cat(sh.frac & ~mask, sh.exp, sh.sign) & (~sh.sub_one).replicate(wfull)
        int_base = Signal(wint, name="int_base")
        m.d.comb += int_base.eq(Cat(sh.trunc, Const(0, 1)) ^ sh.sign.replicate(wint))
        result = data.StructLayout({"y_float": layout, "y_int": signed(wint)})
        selected = bundle(
            result,
            y_float=layout(Mux(inc, Cat(float_next, sh.sign), float_kept)),
            y_int=Mux((inc ^ sh.sign) & sh.has_next, (int_base + 1)[:wint], int_base).as_signed(),
        )
        payload = Signal(result, name="selected")
        m.d.comb += payload.eq(selected)
        valid, out = pipe(m, "out", valid, result, payload, model.stage_output)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.y_float.eq(out.y_float)
        m.d.comb += self.y_int.eq(out.y_int)
        return m

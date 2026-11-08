import enum
import math
from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Cat, Const, Module, Mux, Signal, signed, Value
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, delay, ge, pipe, zext
from ._pack import Pack, pack_reference
from ._value import UnsupportedFormat, Zkf


class DivsqrtResult(NamedTuple):
    y: Zkf
    error: bool


@dataclass(frozen=True, kw_only=True, eq=False)
class Divsqrt(Operator):
    """
    `y = a / b` at `mode=0`, `y = sqrt(a)` at `mode=1`, either by `op_sqrt` per transaction at `mode=2`, correctly
    rounded, at one latency; a fixed mode builds only its operation, and a root does not read `b`. `error` flags
    `b = 0`, where `0 / 0 = +0`, or `a < 0` for a root, which gives `-inf`.

    Both run the digit recurrence `w' = r * w - F_digit` of radix `r` 4, or 8 under `fast`, on the partial remainder
    `w`, in one fixed-point frame and a digit per register stage: each digit is the largest whose subtrahend fits, found
    by subtractions side by side, with no redundant digits. A division works on the significands, its quotient brought
    into [1, 2) by doubling a dividend below the divisor. A root keeps its partial root where the divider keeps the
    divisor, its subtrahends the divider's with bits set where those are zero, so that both share the subtractors.
    Stage 0 resolves the root's first bits from the top bits of the radicand and, where WMAN leaves room, the divider's
    first digit by three-operand subtractions. The bits left after the last whole digit are decided after the last
    register by compares alone, no significand being a tie or rounding up to 2, and the rounding selects the digits so
    far or one more.
    """

    class CoreStages(enum.Flag):
        """The registers of `stage_core`."""

        FIRST_DIGIT = enum.auto()
        """
        A stage more, the divider's first digit resolved in it rather than folded into stage 0 and the root's digits
        moved a stage on: without `fast` at even WMAN, under `fast` at a WMAN not one above a multiple of 3.
        """

    operator = "divsqrt"
    stage_input: int = 0
    stage_core: CoreStages = CoreStages(0)
    """See `Divsqrt.CoreStages`; the other digits have a stage each, fixed."""
    stage_output: int = 0
    mode: int = 2
    fast: bool = True
    """A digit of radix 8 a stage, a third fewer stages, where radix 4 builds less fabric and a shorter stage."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.fmt.wman < 5:
            raise UnsupportedFormat(f"WMAN={self.fmt.wman} is below 5")
        if self.fast and self.fmt.wman < 9:
            raise UnsupportedFormat(f"fast: WMAN={self.fmt.wman} is below 9; set fast=False")
        if not 0 <= self.mode <= 2:
            raise ValueError(f"mode={self.mode} is outside 0..2")

    @property
    def timing(self) -> Timing:
        stages = _Digits.of(self).stages(self)
        return Timing(stages + self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature(
            {
                "in_valid": In(1),
                "op_sqrt": In(1),
                "a": In(f),
                "b": In(f),
                "out_valid": Out(1),
                "y": Out(f),
                "error": Out(1),
            }
        )

    def component(self) -> _Divsqrt:
        return _Divsqrt(self)

    def evaluate(self, op_sqrt: int, a: Zkf, b: Zkf) -> DivsqrtResult:
        if a.fmt != self.fmt or b.fmt != self.fmt:
            raise ValueError(f"an operand is not of {self.fmt}")
        if not 0 <= op_sqrt <= 1:
            raise ValueError(f"op_sqrt={op_sqrt} is not 0 or 1")
        fmt = self.fmt
        root = bool(op_sqrt) if self.mode == 2 else self.mode == 1
        if root:
            if a.is_zero:
                return DivsqrtResult(Zkf(fmt, 0), False)
            if a.negative:
                return DivsqrtResult(fmt.inf(1), True)
            if a.is_inf:
                return DivsqrtResult(fmt.inf(0), False)

            # `qfrac` fraction bits of the root of the radicand at an even exponent; the guard by WMAN's parity.
            qfrac = fmt.wfrac + (fmt.wfrac % 2)
            r = (a.exp - fmt.bias) & 1
            radicand = a.significand() << (2 * qfrac + r - fmt.wfrac)
            raw = math.isqrt(radicand)
            rem = radicand - raw * raw
            if fmt.wman % 2 == 0:
                significand_value = raw >> 1
                guard = raw & 1
            else:
                significand_value = raw
                guard = 1 if rem > raw else 0
            sticky = 1 if rem else 0
            exp_unbiased = (a.exp - fmt.bias) // 2
            return DivsqrtResult(pack_reference(fmt, 0, 0, 0, exp_unbiased, significand_value, guard, 0, sticky), False)

        div0 = b.is_zero
        if a.is_zero or b.is_inf:
            return DivsqrtResult(Zkf(fmt, 0), div0)
        result_sign = a.negative if b.is_zero else (a.negative ^ b.negative)
        if b.is_zero or a.is_inf:
            return DivsqrtResult(fmt.inf(result_sign), div0)
        return DivsqrtResult(fmt.encode(a.to_fraction() / b.to_fraction()), div0)


class DivsqrtStep(wiring.Component):
    """
    One radix-4 digit of the recurrence `w' = 4w - F_digit` of division and square root in one frame: `digit` is the
    largest of 1 to 3 with `F_digit <= 4w`, or 0. A divider holds `d` and `d3 = 3d`, and `F = (d, 2d, d3)`; any divisor
    works as long as `w < d`. A root keeps `d = 2Q` and `d3 = 2(3Q + u)`, `u` being the weight of the last digit of `Q`,
    at bit `iq + 2`: its subtrahends are the divider's with bit `iq` (`u/4`) set in `F1` and `F3` and bit `iq + 2` (`u`)
    in `F2`, bits which are zero there; `d` takes the digit at bit `iq + 1`, and `d3` moves by
    `(3 * digit - 3) << (iq + 1)`.

    `mode` 0 divides, 1 takes roots, and 2 does either by `sqrt`; there `inj` gives the subtrahends' bits at the
    injection points (`F1` at `iq`, `F2` at `iq + 2`, `F3` at `iq`), so that the caller can register them. `ww` is the
    width of `w` and `d`.
    """

    w: Signal
    d: Signal
    d3: Signal
    sqrt: Signal
    inj: Signal
    digit: Signal
    w_next: Signal
    d_next: Signal
    d3_next: Signal

    def __init__(self, ww: int, *, mode: int, iq: int) -> None:
        self._mode, self._iq = mode, iq
        taken = {"w": In(ww), "d": In(ww), "d3": In(ww + 2), "sqrt": In(1), "inj": In(3)}
        super().__init__(taken | {"digit": Out(2), "w_next": Out(ww), "d_next": Out(ww), "d3_next": Out(ww + 2)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        mode, iq, w, d, d3, sqrt, inj = self._mode, self._iq, self.w, self.d, self.d3, self.sqrt, self.inj
        ww, wt = len(w), len(w) + 2
        low = iq if mode == 1 else 0  # a root alone has nothing to subtract below `iq`
        scaled = Cat(Const(0, 2), w)

        subtrahends = [Cat(d, Const(0, 2)), Cat(Const(0, 1), d, Const(0, 1)), d3]
        if mode:
            bits = Const(7, 3) if mode == 1 else inj
            at = (iq, iq + 2, iq)
            subtrahends = [Cat(f[: at[n]], bits[n], f[at[n] + 1 :]) for n, f in enumerate(subtrahends)]
        differences = [Signal(signed(wt - low + 1), name=f"t{n}") for n in (1, 2, 3)]
        m.d.comb += [t.eq(scaled[low:] - f[low:]) for t, f in zip(differences, subtrahends)]
        ge1, ge2, ge3 = (~t[-1] for t in differences)
        remainders = [Cat(scaled[:low], t[: wt - low])[:ww] for t in differences]
        m.d.comb += self.digit.eq(Cat(ge3 | (ge1 & ~ge2), ge2))
        digit = self.digit

        def placed(value: Value) -> Value:
            return Cat(Const(0, iq + 1), value)

        high = d3[iq + 1 :]
        wh = len(high)
        if mode == 0:
            # A priority select by the compares measures fastest for the divider alone.
            w_next = Mux(ge3, remainders[2], Mux(ge2, remainders[1], Mux(ge1, remainders[0], scaled[:ww])))
            d_next, d3_next = d, d3
        elif mode == 1:
            # Selected by the digit, which lets a tool fold the candidates for `d3` into one adder.
            lower, upper, upper2 = (high + step for step in (-3, 3, 6))
            w_sel, hi_sel = Signal(ww, name="w_sel"), Signal(wh, name="hi_sel")
            with m.Switch(digit):
                for value, (w_case, hi_case) in enumerate(
                    ((scaled[:ww], lower), (remainders[0], high), (remainders[1], upper), (remainders[2], upper2))
                ):
                    with m.Case(value) if value < 3 else m.Default():
                        m.d.comb += w_sel.eq(w_case)
                        m.d.comb += hi_sel.eq(hi_case)
            w_next, d_next, d3_next = w_sel, d | placed(digit), Cat(d3[: iq + 1], hi_sel)
        else:
            # Selected by the monotonic compares, a level shallower than by the digit, where the operations share the
            # path; the root's candidates fall back to `high` for the divider, so the select of `d3` stays 4:1.
            w_next = Mux(ge2, Mux(ge3, remainders[2], remainders[1]), Mux(ge1, remainders[0], scaled[:ww]))
            lower, upper, upper2 = (Mux(sqrt, value[:wh], high) for value in (high - 3, high + 3, high + 6))
            d3_next = Cat(d3[: iq + 1], Mux(ge2, Mux(ge3, upper2, upper), Mux(ge1, high, lower)))
            d_next = d | placed(digit & sqrt.replicate(2))
        m.d.comb += self.w_next.eq(w_next)
        m.d.comb += self.d_next.eq(d_next)
        m.d.comb += self.d3_next.eq(d3_next)
        return m


class DivideStep8(wiring.Component):
    """
    One radix-8 digit of a division, `w' = 8w - digit * d`: `digit` is the largest of 0 to 7 with `digit * d <= 8w`,
    for `w < d`. The caller holds `d3`, `d5` and `d7`, `d`'s multiples. `ww` is the width of `w` and `d`.
    """

    w: Signal
    d: Signal
    d3: Signal
    d5: Signal
    d7: Signal
    digit: Signal
    w_next: Signal

    def __init__(self, ww: int) -> None:
        super().__init__(
            {"w": In(ww), "d": In(ww), "d3": In(ww + 2), "d5": In(ww + 3), "d7": In(ww + 3)}
            | {"digit": Out(3), "w_next": Out(ww)}
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        ww, d, d3 = len(self.w), self.d, self.d3
        scaled = Cat(Const(0, 3), self.w)
        subtrahends = [d, d << 1, d3, d << 2, self.d5, d3 << 1, self.d7]
        differences = [Signal(signed(ww + 4), name=f"t{k}") for k in range(1, 8)]
        m.d.comb += [t.eq(scaled - f) for t, f in zip(differences, subtrahends)]
        w_next, digit = _select8([~t[-1] for t in differences], [scaled[:ww]] + [t[:ww] for t in differences])
        m.d.comb += self.w_next.eq(w_next)
        m.d.comb += self.digit.eq(digit)
        return m


class _DivsqrtStep8(wiring.Component):
    """
    One radix-8 digit of the root, `mode` 1, or of either operation by `sqrt`, `mode` 2: `w' = 8w - F_digit`, `digit`
    the largest of 0 to 7 with `F_digit <= 8w`. Only `d` is carried: `2Q` of a root, the divisor of a division. The
    subtrahends are the divider's multiples of `d` formed from shifts of it, `d`, `2d`, `2d + d`, `4d`, `4d + d`,
    `4d + 2d` and `8d - d`; a root's are those with `k^2 v` ORed onto zeros, `v` the new digit's weight at bit `iq`,
    `Q`'s last digit weighing `8v`: `F_k = 2kQ + k^2 v`. `d` takes the digit at bit `iq + 1`. At `mode` 2, `inj` gives
    the bits of `d` from `iq - 3` to `iq + 3` ORed with `sqrt`, those that the squares fall on, so that the caller can
    register them.
    """

    w: Signal
    d: Signal
    sqrt: Signal
    inj: Signal
    digit: Signal
    w_next: Signal
    d_next: Signal

    def __init__(self, ww: int, *, mode: int, iq: int) -> None:
        self._mode, self._iq = mode, iq
        taken = {"w": In(ww), "d": In(ww), "sqrt": In(1), "inj": In(7)}
        super().__init__(taken | {"digit": Out(3), "w_next": Out(ww), "d_next": Out(ww)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        mode, iq, w, d = self._mode, self._iq, self.w, self.d
        ww, wt = len(w), len(w) + 4
        root = Const(1, 1) if mode == 1 else self.sqrt

        def operand(shift: int, square: int) -> Value:
            """`d << shift`, wt wide, with the root's `square * v` ORed in."""
            shifted = Cat(Const(0, shift), d, Const(0, wt - ww - shift))
            if mode == 1:
                return shifted | Const(square << iq, wt)
            squared = range(iq, iq + square.bit_length())
            injected = (k for k in squared if square >> (k - iq) & 1)
            bits = {k: self.inj[k - shift - iq + 3] for k in injected}
            return Cat(*(bits.get(k, shifted[k]) for k in range(wt)))

        x = Cat(Const(0, 3), w, Const(0, 1))
        x_n = Signal(wt, name="x_n")
        m.d.comb += x_n.eq(~x)

        def two(a: Value, name: str) -> Signal:
            """`~(x - a)`."""
            n = Signal(wt, name=name)
            m.d.comb += n.eq(x_n + a)
            return n

        def three(a: Value, b: Value, carry_in: int, name: str) -> Signal:
            """`~(x - a - b)`, or `~(x - a + d)` with `b = ~d` and a carry in."""
            n = Signal(wt, name=name)
            m.d.comb += n.eq(_sum3(x_n ^ a, x_n & a, b, carry_in))
            return n

        d_n = Signal(wt, name="d_n")
        m.d.comb += d_n.eq(~Cat(d, Const(0, wt - ww)))
        n = [
            two(operand(0, 1), "n1"),
            two(operand(1, 4), "n2"),
            three(operand(1, 9), operand(0, 0), 0, "n3"),
            two(operand(2, 16), "n4"),
            three(operand(2, 25), operand(0, 0), 0, "n5"),
            three(operand(2, 36), operand(1, 0), 0, "n6"),
            three(operand(3, 49), d_n, 1, "n7"),
        ]
        w_next, digit = _select8([v[-1] for v in n], [x[:ww]] + [(~v)[:ww] for v in n])
        m.d.comb += self.w_next.eq(w_next)
        m.d.comb += self.digit.eq(digit)
        taken = self.digit & root.replicate(3)
        m.d.comb += self.d_next.eq(d | Cat(Const(0, iq + 1), taken)[:ww])
        return m


def _select8(g: list[Value], r: list[Value]) -> tuple[Value, Value]:
    """
    The remainder `r[digit]` and the digit of a radix-8 step from its compares `g`, of the digits 1 to 7, which are
    monotonic: a tree by them measures faster than a priority select.
    """
    g1, g2, g3, g4, g5, g6, g7 = g
    w_next = Mux(
        g4, Mux(g6, Mux(g7, r[7], r[6]), Mux(g5, r[5], r[4])), Mux(g2, Mux(g3, r[3], r[2]), Mux(g1, r[1], r[0]))
    )
    return w_next, Cat(Mux(g4, Mux(g6, g7, g5), Mux(g2, g3, g1)), Mux(g4, g6, g2), g4)


class _Specials(NamedTuple):
    """
    Of a root or a division: what its operands decide outside the digits, the exponent biased and, a quotient's, before
    its normalization.
    """

    zero: Value
    inf: Value
    sign: Value
    exp: Value
    err: Value


class _Divsqrt(wiring.Component):
    """
    The special cases, the exponent and the packer around the digits of one radix; the packer classifies the result a
    stage ahead, from the metadata, beside the last digit.
    """

    in_valid: Signal
    op_sqrt: Signal
    a: FloatView
    b: FloatView
    out_valid: Signal
    y: FloatView
    error: Signal

    def __init__(self, model: Divsqrt) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt, mode = self._model, self._model.fmt, self._model.mode
        wexp, weu = fmt.wexp, fmt.wexp + 2
        divides, roots = mode != 1, mode != 0
        layout = FloatLayout(fmt)

        operands = data.StructLayout({"a": layout, "b": layout, "op_sqrt": 1})
        taken = bundle(operands, a=self.a, b=self.b, op_sqrt=self.op_sqrt)
        valid, q = pipe(m, "in", self.in_valid, operands, taken, model.stage_input)
        a, b, op = q.a, q.b, q.op_sqrt

        m.submodules.digits = digits = _Digits.of(model)(model)
        m.d.comb += [digits.a.eq(a), digits.b.eq(b), digits.op.eq(op)]

        a_zero = Signal(name="a_zero")
        m.d.comb += a_zero.eq(a.is_zero)
        a_inf = Signal(name="a_inf")
        m.d.comb += a_inf.eq(a.is_inf)
        root = quotient = None
        if roots:
            negative = Signal(name="negative")
            m.d.comb += negative.eq(a.sign & ~a_zero)
            # The bias is odd, so the halved sum is the floor of half the unbiased exponent, biased again.
            halved = Cat((a.exp + fmt.bias)[1 : wexp + 1], Const(0, 2)).as_signed()
            root = _Specials(zero=a_zero, inf=a_inf | negative, sign=negative, exp=halved, err=negative)
        if divides:
            b_zero = Signal(name="b_zero")
            m.d.comb += b_zero.eq(b.is_zero)
            b_inf = Signal(name="b_inf")
            m.d.comb += b_inf.eq(b.is_inf)
            quotient = _Specials(
                zero=a_zero | b_inf,
                inf=~a_zero & ~b_inf & (b_zero | a_inf),
                sign=a.sign ^ (b.sign & ~b_zero),
                exp=(a.exp - b.exp + fmt.bias)[:weu].as_signed(),
                err=b_zero,
            )
        zero0, inf0, sign0, exp0, err0 = _either(op, root, quotient)

        # Ahead of the packer's register, beside the last digit; the exponent's normalization from stage 0's on.
        flags = data.StructLayout({"sign": 1, "zero": 1, "inf": 1, "err": 1})
        held = bundle(flags, sign=sign0, zero=zero0, inf=inf0, err=err0)
        stages = digits.stages(model)
        valid, q = pipe(m, "meta", valid, flags, held, stages - 1)
        exp = delay(m, "exp", signed(weu), exp0, 1)
        exp = delay(m, "exp_n", signed(weu), exp - digits.lowered if divides else exp, stages - 2)
        sign, zero, inf, err = q.sign, q.zero, q.inf, q.err

        m.submodules.pack = pack = Pack(
            fmt,
            exp_is_biased=True,
            assume_no_overflow=mode == 1,
            stage_input=1,
            stage_output=model.stage_output,
            classify_ahead=True,
            late_significand=True,
        )
        m.d.comb += pack.in_valid.eq(valid)
        m.d.comb += pack.sign.eq(sign)
        m.d.comb += pack.force_zero.eq(zero)
        m.d.comb += pack.force_inf.eq(inf)
        m.d.comb += pack.exp_unbiased.eq(exp)
        m.d.comb += pack.significand.eq(digits.significand)
        for tail in (pack.guard, pack.round, pack.sticky):
            m.d.comb += tail.eq(0)
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        m.d.comb += self.error.eq(delay(m, "error", 1, err, model.stage_output + 1))
        return m


class _Digits(wiring.Component):
    """
    The digits of `Divsqrt` in one radix, the significand after the last register; `lowered` is the divider's
    normalization from stage 0's register on. Each radix's `stages(model)` counts the register stages, stage 0
    included, ahead of the significand.
    """

    a: FloatView
    b: FloatView
    op: Signal
    lowered: Signal
    significand: Signal

    def __init__(self, model: Divsqrt) -> None:
        self._model = model
        f = FloatLayout(model.fmt)
        super().__init__({"a": In(f), "b": In(f), "op": In(1), "lowered": Out(1), "significand": Out(model.fmt.wman)})

    @staticmethod
    def of(model: Divsqrt) -> type[_Radix4 | _Radix8]:
        return _Radix8 if model.fast else _Radix4

    @staticmethod
    def unfolds(model: Divsqrt) -> bool:
        """Whether WMAN leaves stage 0 room for the divider's first digit, which `FIRST_DIGIT` takes out of it."""
        return model.fmt.wman % 3 != 1 if model.fast else model.fmt.wman % 2 == 0

    @staticmethod
    def folds(model: Divsqrt) -> bool:
        return _Digits.unfolds(model) and Divsqrt.CoreStages.FIRST_DIGIT not in model.stage_core

    def root0(self, m: Module, wframe: int, n: int) -> _Root0:
        """The root's stage 0, its first `n` bits."""
        m.submodules.root0 = root0 = _Root0(self._model.fmt.wman, wframe, n)
        m.d.comb += [root0.frac.eq(self.a.frac), root0.r.eq(~self.a.exp[0])]
        return root0

    def div0(self, m: Module, fold: bool) -> _Div0:
        """The divider's stage 0, and `lowered` off it."""
        m.submodules.div0 = div0 = _Div0(self._model.fmt.wman, fold)
        m.d.comb += [div0.sa.eq(self.a.significand), div0.sb.eq(self.b.significand)]
        lowered = Signal(name="lowered", reset_less=True)
        m.d.sync += lowered.eq(div0.h & ~self.op if self._model.mode == 2 else div0.h)
        m.d.comb += self.lowered.eq(lowered)
        return div0


class _Radix4(_Digits):
    """
    The radix-4 digits of `Divsqrt`: stage 0, a register stage per further digit, and the last bits with the rounding
    after the last register.
    """

    @staticmethod
    def stages(model: Divsqrt) -> int:
        return model.fmt.wman // 2 - int(_Digits.folds(model))

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt, mode = self._model, self._model.fmt, self._model.mode
        wman, wfrac = fmt.wman, fmt.wfrac
        even = wman % 2 == 0
        fold = self.folds(model)
        stages = self.stages(model)
        divides, roots = mode != 1, mode != 0
        wframe = wman if roots and even else wfrac  # fraction bits of the frame
        ww = (2 if roots else 1) + wframe
        b, op = self.b, self.op

        # After the first digit, which an unfolded divider lacks: digits, remainder, the divisor or `2Q`, and 3x that.
        root = quotient = None
        if roots:
            root0 = self.root0(m, wframe, 2)
            root = (Cat(root0.digit, Const(1, 1)), root0.w, root0.d, root0.d3)
        den = None
        if divides:
            div0 = self.div0(m, fold)
            below, above = Const(0, wframe - wfrac), Const(0, ww + 2 - wman - (wframe - wfrac))
            den = Cat(below, b.significand, above)
            den3 = Signal(ww + 2, name="den3")
            m.d.comb += den3.eq(den + Cat(Const(0, 1), den[:-1]))
            first = Cat(div0.digit, Const(1, 1)) if fold else Const(1, 3)
            quotient = (first, Cat(below, div0.rem, above)[:ww], den[:ww], den3)
        prefix0, w0, d0, d30 = _either(op, root, quotient)

        ops = None
        if mode == 2:
            ops = delay(m, "op0", 1, op, 1)
        w, d, d3 = (
            Signal(ww, name="w0", reset_less=True),
            Signal(ww, name="d0", reset_less=True),
            Signal(ww + 2, name="d30", reset_less=True),
        )
        m.d.sync += [w.eq(w0), d.eq(d0), d3.eq(d30)]
        prefix = _Prefix(m, prefix0 if divides else None)
        inj = None
        if mode == 2:  # what the root puts into the subtrahends of the next stage, registered off its compares
            assert den is not None
            held_bits = Cat(den[wframe - 4], den[wframe - 3], den3[wframe - 4])
            inj = Signal(3, name="inj0", reset_less=True)
            m.d.sync += inj.eq(op.replicate(3) | held_bits)

        d5 = d7 = Const(0, ww + 3)
        for s in range(1, stages):
            iq = wframe - 2 * s - 2 if roots else 0  # where the root's quarter weight is after `s` digits
            m.submodules[f"step{s}"] = step = DivsqrtStep(ww, mode=mode, iq=iq)
            m.d.comb += [step.w.eq(w), step.d.eq(d), step.d3.eq(d3)]
            if ops is not None and inj is not None:
                m.d.comb += [step.sqrt.eq(ops), step.inj.eq(inj)]
            prefix.step(s, step.digit, d[wframe + 1 - 2 * s : wframe + 2], s == stages - 1)  # `d` is twice the root
            # 5 and 7 times the divisor, for the step of radix 8 that ends an odd WMAN.
            if s == stages - 1 and divides and not even:
                d5 = Signal(ww + 3, name="d5", reset_less=True)
                m.d.sync += d5.eq(d + Cat(Const(0, 2), d))
                d7 = Signal(ww + 3, name="d7", reset_less=True)
                m.d.sync += d7.eq(Cat(Const(0, 3), d) - d)
            if mode == 2:
                assert ops is not None
                held_bits = Cat(d[iq - 2], d[iq - 1], d3[iq - 2]) if iq >= 2 else None
                inj = Signal(3, name=f"inj{s}", reset_less=True)
                if held_bits is not None:
                    m.d.sync += inj.eq(ops.replicate(3) | held_bits)
                else:
                    m.d.comb += inj.eq(0)
                ops = delay(m, f"op{s}", 1, ops, 1)
            w, d, d3 = (
                Signal(ww, name=f"w{s}", reset_less=True),
                Signal(ww, name=f"d{s}", reset_less=True),
                Signal(ww + 2, name=f"d3{s}", reset_less=True),
            )
            m.d.sync += [w.eq(step.w_next), d.eq(step.d_next), d3.eq(step.d3_next)]

        wp = 2 * stages + 1
        digits, digits1 = prefix.final(d[wframe + 2 - wp : wframe + 2])
        if even:
            m.submodules.last = last = _Last4(wman, mode, fold, ww, wp)
            m.d.comb += last.inj.eq(Cat(inj[0], inj[2]) if inj is not None else Const(0, 2))
        else:  # a radix-8 step and the root's guard
            m.submodules.last = last = _Last(wman, mode, 3, 1, ww, wp)
            m.d.comb += [last.d5.eq(d5), last.d7.eq(d7)]
        m.d.comb += last.sqrt.eq(ops if ops is not None else Const(mode, 1))
        m.d.comb += last.w.eq(w)
        m.d.comb += last.d.eq(d)
        m.d.comb += last.d3.eq(d3)
        m.d.comb += [last.prefix.eq(digits), last.prefix1.eq(digits1)]
        m.d.comb += self.significand.eq(last.significand)
        return m


class _Layout(NamedTuple):
    """
    The bits of each operation that `_Radix8` resolves in stage 0 and after its last register, three a stage between
    them: the divider's none or two (`_Div0`'s fold) in stage 0, the root's two more (`_Root0`), and two fewer after.
    """

    first_q: int
    last_q: int

    @property
    def first_r(self) -> int:
        return self.first_q + 2

    @property
    def last_r(self) -> int:
        return self.last_q - 2


class _Radix8(_Digits):
    """
    The radix-8 digits of `Divsqrt`: stage 0, a radix-8 digit a register stage, and each operation's last bits decided
    after the last register by `_Last`. Stage 0 folds the first bits of each operation as `layout` gives them for
    WMAN's residue by 3, so that the bits left come out in whole digits.
    """

    @staticmethod
    def layout(model: Divsqrt) -> _Layout:
        folds = _Digits.folds(model)
        if model.fmt.wman % 3 == 0:
            return _Layout(2, 4) if folds else _Layout(0, 3)
        if model.fmt.wman % 3 == 1:
            return _Layout(0, 4)
        return _Layout(2, 3) if folds else _Layout(0, 2)

    @staticmethod
    def stages(model: Divsqrt) -> int:
        layout = _Radix8.layout(model)
        return 1 + (model.fmt.wman - layout.first_q - layout.last_q) // 3

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt, mode = self._model, self._model.fmt, self._model.mode
        wman, wfrac = fmt.wman, fmt.wfrac
        layout, stages = self.layout(model), self.stages(model)
        divides, roots = mode != 1, mode != 0
        # The root's last digit weighs 1 with a guard after it, 4 as a digit of radix 4, 1 with its guard in the digit.
        wframe = wfrac if not roots or layout.last_r == 1 else wman
        ww = (2 if roots else 1) + wframe
        b, op = self.b, self.op

        def iq(s: int) -> int:
            """The weight of the root's digit at stage `s`."""
            return wframe - layout.first_r - 3 * s

        root = quotient = None
        if roots:
            root0 = self.root0(m, wframe, layout.first_r)
            root = (Cat(root0.digit, Const(1, 1)), root0.w, root0.d)

        def multiples(den: Value, s: int) -> tuple[Value, Value, Value]:
            """3, 5 and 7 times the divisor, registered at stage `s`."""
            out = (
                Signal(ww + 2, name=f"d3{s}", reset_less=True),
                Signal(ww + 3, name=f"d5{s}", reset_less=True),
                Signal(ww + 3, name=f"d7{s}", reset_less=True),
            )
            m.d.sync += [out[0].eq(den + (den << 1)), out[1].eq(den + (den << 2)), out[2].eq((den << 3) - den)]
            return out

        d3, d5, d7 = Const(0, ww + 2), Const(0, ww + 3), Const(0, ww + 3)
        if divides:
            div0 = self.div0(m, layout.first_q == 2)
            den = zext(b.significand, ww)
            quotient = (Cat(div0.digit, Const(1, 1)) if layout.first_q else Const(1, 1), zext(div0.rem, ww), den)
            if mode == 0:  # carried for every radix-8 step
                d3, d5, d7 = multiples(den, 0)
        prefix0, w0, d0 = _either(op, root, quotient)

        ops = None
        if mode == 2:
            ops = delay(m, "op0", 1, op, 1)
        w, d = Signal(ww, name="w0", reset_less=True), Signal(ww, name="d0", reset_less=True)
        m.d.sync += [w.eq(w0), d.eq(d0)]
        prefix = _Prefix(m, prefix0 if divides else None)
        inj = None
        if mode == 2:  # what the root puts into the subtrahends of the next stage, registered off its compares
            inj = Signal(7, name="inj0", reset_less=True)
            m.d.sync += inj.eq(op.replicate(7) | _bits(den, iq(1) - 3, 7))

        for s in range(1, stages):
            if mode == 0:
                m.submodules[f"step{s}"] = step = DivideStep8(ww)
                m.d.comb += [step.d3.eq(d3), step.d5.eq(d5), step.d7.eq(d7)]
            else:
                m.submodules[f"step{s}"] = step = _DivsqrtStep8(ww, mode=mode, iq=iq(s))
                if ops is not None and inj is not None:
                    m.d.comb += [step.sqrt.eq(ops), step.inj.eq(inj)]
            m.d.comb += [step.w.eq(w), step.d.eq(d)]
            prefix.step(s, step.digit, d[iq(s - 1) + 1 : wframe + 2], s == stages - 1)  # `d` is twice the root
            if s == stages - 1 and mode == 2 and layout.last_q >= 3:  # the divisor's multiples for the last bits
                d3, d5, d7 = multiples(step.d, s)
            if mode == 2:
                assert ops is not None
                if s < stages - 1:
                    inj = Signal(7, name=f"inj{s}", reset_less=True)
                    m.d.sync += inj.eq(ops.replicate(7) | _bits(d, iq(s + 1) - 3, 7))
                ops = delay(m, f"op{s}", 1, ops, 1)
            d_next = step.d_next if mode else d
            w, d = Signal(ww, name=f"w{s}", reset_less=True), Signal(ww, name=f"d{s}", reset_less=True)
            m.d.sync += [w.eq(step.w_next), d.eq(d_next)]
            if mode == 0:
                d3 = delay(m, f"d3{s}", ww + 2, d3, 1)
                d5 = delay(m, f"d5{s}", ww + 3, d5, 1)
                d7 = delay(m, f"d7{s}", ww + 3, d7, 1)

        digits, digits1 = prefix.final(d[iq(stages - 1) + 1 : wframe + 2])
        m.submodules.last = last = _Last(wman, mode, layout.last_q, layout.last_r, ww, len(digits))
        m.d.comb += last.sqrt.eq(ops if ops is not None else Const(mode, 1))
        m.d.comb += [last.w.eq(w), last.d.eq(d), last.d3.eq(d3), last.d5.eq(d5), last.d7.eq(d7)]
        m.d.comb += [last.prefix.eq(digits), last.prefix1.eq(digits1)]
        m.d.comb += self.significand.eq(last.significand)
        return m


def _either(op: Value, root: tuple[Value, ...] | None, quotient: tuple[Value, ...] | None) -> tuple[Value, ...]:
    """The values of the operation `op` selects, or of the one a fixed mode builds."""
    if root is None or quotient is None:
        found = root or quotient
        assert found is not None
        return found
    return tuple(Mux(op, r, q) for r, q in zip(root, quotient))


class _Prefix:
    """
    The digits of a division so far, `first` shifted a digit a stage, and `inc`, those before the last digit plus one,
    registered beside it (see `_plus1`); a root's are read off `d`, given as `of_root`.
    """

    def __init__(self, m: Module, first: Value | None) -> None:
        self._m, self._bits = m, 0
        self._inc: Signal | None = None
        self._digits: Signal | None = None
        if first is not None:
            self._digits = Signal(len(first), name="prefix0", reset_less=True)
            m.d.sync += self._digits.eq(first)

    def step(self, s: int, digit: Value, of_root: Value, last: bool) -> None:
        self._bits = len(digit)
        so_far = self._digits if self._digits is not None else of_root
        if last:
            self._inc = Signal(len(so_far), name="inc", reset_less=True)
            self._m.d.sync += self._inc.eq(so_far + 1)
        if self._digits is not None:
            shifted = Signal(len(self._digits) + len(digit), name=f"prefix{s}", reset_less=True)
            self._m.d.sync += shifted.eq(Cat(digit, self._digits))
            self._digits = shifted

    def final(self, of_root: Value) -> tuple[Value, Value]:
        """The prefix and `prefix + 1`, as `_Last` takes them."""
        assert self._inc is not None
        digits = self._digits if self._digits is not None else of_root
        return digits, _plus1(digits, self._inc, self._bits)


def _plus1(prefix: Value, inc: Value, bits: int) -> Value:
    """
    `prefix + 1`, its last digit of `bits` bits, from `inc`, the digits before it plus one, registered beside the last
    digit: a select from registers alone, with no constant arm, which a tool maps onto the register's reset, and no
    increment after a compare.
    """
    digit = prefix[:bits]
    return Cat((digit + 1)[:bits], Mux(digit.all(), inc, prefix[bits:]))


def _rounded(reached: list[Value], prefix: Value, prefix1: Value, width: int) -> Value:
    """
    `width` bits of the significand from the compares of the odd digits of its last bits, `reached`: the `prefix` and
    the count of them, or `prefix1` followed by zeros when all are (see `Divsqrt`).
    """
    low = _count(reached)
    return Cat(low, Mux(reached[-1], prefix1, prefix)[: width - len(low)])


def _bits(value: Value, low: int, count: int) -> Value:
    """`count` bits of `value` from bit `low` on, zero outside it."""
    return Cat(*(value[k] if 0 <= k < len(value) else Const(0, 1) for k in range(low, low + count)))


def _ge3(x: Value, p: Value, q: Value) -> Value:
    """`x >= p + q`, unsigned, on one carry chain: the top bit of `~(x - p - q) = ~x + p + q`."""
    wt = max(len(x), len(p), len(q)) + 2
    x_n, p, q = ~zext(x, wt), zext(p, wt), zext(q, wt)
    return _sum3(x_n ^ p, x_n & p, q, 0)[-1]


def _count(reached: list[Value]) -> Value:
    """How many of `reached` hold, each implying the ones before it, modulo their number, a power of 2."""
    exactly = [reached[k - 1] & ~reached[k] for k in range(1, len(reached))]  # `k` of them
    bits = range(len(reached).bit_length() - 1)
    return Cat(*(Cat(*(e for k, e in enumerate(exactly, 1) if k >> b & 1)).any() for b in bits))


def _sum3(either: Value, both: Value, r: Value, carry_in: int) -> Value:
    """`p + q + r + carry_in` on one carry chain, given `p ^ q` and `p & q`: a carry-save layer and one addition."""
    carry = both | (r & either)
    return ((either ^ r) + Cat(Const(carry_in, 1), carry[:-1]))[: len(r)]


class _Div0(wiring.Component):
    """
    The divider's stage 0. `h = sa < sb`, and the dividend is `n = sa << h`, so that `n / sb` is in [1, 2). Unfolded,
    `rem = n - sb` and `digit = 0`. With `fold`, `digit` is the first quotient digit, the largest `k` of 0 to 3 with
    `4n >= (4 + k) * sb`, and `rem = 4n - (4 + k) * sb`. Both outcomes of `h` are worked out side by side, so that `h`,
    a carry chain itself, only selects.
    """

    sa: Signal
    sb: Signal
    h: Signal
    digit: Signal
    rem: Signal

    def __init__(self, wman: int, fold: bool) -> None:
        self._wman, self._fold = wman, fold
        super().__init__({"sa": In(wman), "sb": In(wman), "h": Out(1), "digit": Out(2), "rem": Out(wman)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wman, sa, sb = self._wman, self.sa, self.sb
        plain = Signal(wman + 1, name="r0")
        m.d.comb += plain.eq(sa - sb)
        doubled = Signal(wman + 1, name="r1")
        m.d.comb += doubled.eq(Cat(Const(0, 1), sa) - sb)
        h = plain[-1]
        m.d.comb += self.h.eq(h)
        if not self._fold:
            m.d.comb += self.digit.eq(0)
            m.d.comb += self.rem.eq(Mux(h, doubled[:wman], plain[:wman]))
            return m

        wc = wman + 4
        y1, y2, y4, y8 = (Cat(Const(0, n), sb, Const(0, 4 - n)) for n in range(4))
        y8_n = Signal(wc, name="y8_n")
        m.d.comb += y8_n.eq(~y8)
        digits, rems = [], []
        for k, r in enumerate((plain, doubled)):
            x = Cat(Const(0, 2 + k), sa, Const(0, 2 - k))  # 4n
            x_n = Signal(wc, name=f"x{k}_n")
            m.d.comb += x_n.eq(~x)
            either = Signal(wc, name=f"x{k}_xor_y4")
            m.d.comb += either.eq(x_n ^ y4)
            both = Signal(wc, name=f"x{k}_and_y4")
            m.d.comb += both.eq(x_n & y4)
            n1 = Signal(wc, name=f"n1_{k}")
            m.d.comb += n1.eq(_sum3(either, both, y1, 0))  # ~(4n - 5 sb)
            n2 = Signal(wc, name=f"n2_{k}")
            m.d.comb += n2.eq(_sum3(either, both, y2, 0))  # ~(4n - 6 sb)
            seven = Signal(wc, name=f"x{k}_xor_y1")
            m.d.comb += seven.eq(x ^ y1)
            c3 = Signal(wc, name=f"c3_{k}")
            m.d.comb += c3.eq(_sum3(seven, x & y1, y8_n, 1))  # 4n - 7 sb
            ge1, ge2 = n1[-1], n2[-1]
            ge3 = Signal(name=f"ge3_{k}")
            m.d.comb += ge3.eq(~c3[-1])
            # Selected by the compares, which are monotonic, and not by the digit, as in `DivsqrtStep`.
            low = Mux(ge1, ~n1[:wman], Cat(Const(0, 2), r[: wman - 2]))
            high = Mux(ge3, c3[:wman], ~n2[:wman])
            rems.append(Mux(ge2, high, low))
            digits.append(Cat(ge3 | (ge1 & ~ge2), ge2))
        m.d.comb += self.digit.eq(Mux(h, digits[1], digits[0]))
        m.d.comb += self.rem.eq(Mux(h, rems[1], rems[0]))
        return m


class _Root0(wiring.Component):
    """
    The root's stage 0, its first `n` fraction bits. The radicand `m' = (1 + frac) * 2**r` is in [1, 4), `r` being set
    at an odd unbiased exponent; `digit` counts the thresholds `(1 + j / 2**n)**2` that it reaches, `j` from 1 to
    `2**n - 1`, by subtractions from its top bits, which also give `2**n` times the remainder
    `m' - (1 + digit / 2**n)**2`. `d` is twice the root so far and `d3 = 2(3Q + u)`, `u` its last bit's weight, in a
    frame of `wframe` fraction bits.
    """

    frac: Signal
    r: Signal
    digit: Signal
    w: Signal
    d: Signal
    d3: Signal

    def __init__(self, wman: int, wframe: int, n: int) -> None:
        self._wman, self._wframe, self._n = wman, wframe, n
        ports = {"frac": In(wman - 1), "r": In(1), "digit": Out(n)}
        super().__init__(ports | {"w": Out(wframe + 2), "d": Out(wframe + 2), "d3": Out(wframe + 4)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wfrac, wframe, n, frac, r = self._wman - 1, self._wframe, self._n, self.frac, self.r
        assert wfrac >= 2 * n, (wfrac, n)  # the top bits are whole
        radicand = Mux(r, Cat(Const(0, 1), frac, Const(1, 1)), Cat(frac, Const(1, 2)))  # m' * 2**wfrac
        top = radicand[wfrac - 2 * n :]  # m' * 4**n, its fraction dropped
        differences = [Signal(signed(len(top) + 1), name=f"t{j}") for j in range(1 << n)]
        m.d.comb += [t.eq(top - (j + (1 << n)) ** 2) for j, t in enumerate(differences)]
        reached = [~t[-1] for t in differences]

        def select(values: list[Value], low: int, high: int) -> Value:
            """The value of the count of thresholds reached, of `low` to `high - 1`, by a tree of the compares."""
            if high - low == 1:
                return values[low]
            mid = (low + high) // 2
            return Mux(reached[mid], select(values, mid, high), select(values, low, mid))

        digit = Signal(n, name="first")
        m.d.comb += digit.eq(select([Const(j, n) for j in range(1 << n)], 0, 1 << n))
        h_next = Signal(n + 2, name="h_next")  # the remainder above the frame's bit `wframe - n`
        m.d.comb += h_next.eq(select([t[: n + 2] for t in differences], 0, 1 << n))
        scaled = Cat(Const(0, n + wframe - wfrac), radicand)  # 2**n m' in the frame
        m.d.comb += self.digit.eq(digit)
        m.d.comb += self.w.eq(Cat(scaled[: wframe - n], h_next))
        m.d.comb += self.d.eq(Cat(Const(0, wframe + 1 - n), digit, Const(1, 1)))
        d3 = select([Const(3 * (j + (1 << n)) + 1, n + 3) for j in range(1 << n)], 0, 1 << n)
        m.d.comb += self.d3.eq(Cat(Const(0, wframe + 1 - n), d3))
        return m


class _Last4(wiring.Component):
    """
    Radix 4's last digit at even WMAN, decided by compares alone (`_rounded`): folded, that of either operation, from
    `4w >= F1` and `4w >= F3`, the root's quarter weight being at bit 0 of the frame (`inj` gives those bits at `mode`
    2); unfolded, that of the divider, the root having all its digits and rounding half up.
    """

    sqrt: Signal
    w: Signal
    d: Signal
    d3: Signal
    inj: Signal
    prefix: Signal
    prefix1: Signal
    significand: Signal

    def __init__(self, wman: int, mode: int, fold: bool, ww: int, wp: int) -> None:
        self._wman, self._mode, self._fold = wman, mode, fold
        state = {"sqrt": In(1), "w": In(ww), "d": In(ww), "d3": In(ww + 2), "inj": In(2)}
        super().__init__(state | {"prefix": In(wp), "prefix1": In(wp), "significand": Out(wman)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wman, mode, fold, w, d, d3 = self._wman, self._mode, self._fold, self.w, self.d, self.d3
        prefix, prefix1 = self.prefix, self.prefix1
        if mode == 1 and not fold:
            m.d.comb += self.significand.eq(prefix1[1 : wman + 1])
            return m
        m4 = Cat(Const(0, 2), w, Const(0, 1))
        bits = (Const(3, 2) if mode == 1 else self.inj) if fold and mode else Cat(d[0], d3[0])
        ge1 = ge(m4, Cat(bits[0], d[1:], Const(0, 3)))
        ge3 = ge(m4, Cat(bits[1], d3[1:], Const(0, 1)))
        digit = _rounded([ge1, ge3], prefix, prefix1, wman)
        if fold or mode == 0:
            m.d.comb += self.significand.eq(digit)
        else:
            m.d.comb += self.significand.eq(Mux(self.sqrt, prefix1[1 : wman + 1], digit))
        return m


class _Last(wiring.Component):
    """
    The last bits of each operation, `last_q` of the divider's and `last_r` of the root's, decided after the last
    register by compares alone: `n` bits, the guard the lowest, from the compares of `2**n * w` against the subtrahends
    of the odd digits (`_rounded`). The divider's are `k * d`, from the multiples `d3`, `d5`, `d7` where given (`mode`
    0, and `mode` 2 for more than 2 bits), else `3d` as `d + 2d`. A root with no bits left has its guard in its prefix
    and is rounded half up; its digit's weight is at bit 0 for one bit, where `F1 = 2d + 1` against `4w`, and at bit 2
    for two, where `F1 = d + 1` and `F3 = 3d + 9`.
    """

    sqrt: Signal
    w: Signal
    d: Signal
    d3: Signal
    d5: Signal
    d7: Signal
    prefix: Signal
    prefix1: Signal
    significand: Signal

    def __init__(self, wman: int, mode: int, last_q: int, last_r: int, ww: int, wp: int) -> None:
        self._wman, self._mode, self._last_q, self._last_r = wman, mode, last_q, last_r
        state = {"sqrt": In(1), "w": In(ww), "d": In(ww), "d3": In(ww + 2), "d5": In(ww + 3), "d7": In(ww + 3)}
        super().__init__(state | {"prefix": In(wp), "prefix1": In(wp), "significand": Out(wman)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wman, mode, last_q, last_r = self._wman, self._mode, self._last_q, self._last_r
        w, d, prefix, prefix1 = self.w, self.d, self.prefix, self.prefix1
        m4 = Cat(Const(0, 2), w)

        def rounded(reached: list[Value]) -> Value:
            return _rounded(reached, prefix, prefix1, wman)

        def radix4(root: int) -> list[Value]:
            """The compares of a digit of radix 4 against `d` and `3d`, a root's `u/4` and `2u` ORed in at `root`."""
            return [ge(m4, d | root), _ge3(m4, d, Cat(Const(0, 1), d) | Const(9 * root, 4))]

        quotient = root = Const(0, wman)
        if mode != 1:
            scaled = Cat(Const(0, last_q), w)
            multiples = {1: d, 3: self.d3, 5: self.d5, 7: self.d7}
            if mode == 2 and last_q < 3:  # no multiples are formed
                reached = radix4(0)
            else:
                eight = Cat(Const(0, 3), d)
                reached = [ge(scaled, multiples[k]) for k in range(1, min(1 << last_q, 8), 2)]
                reached += [_ge3(scaled, eight, multiples[k - 8]) for k in range(9, 1 << last_q, 2)]
            quotient = rounded(reached)
        if mode != 0:
            if last_r == 0:
                root = prefix1[1 : wman + 1]
            elif last_r == 1:
                root = rounded([ge(m4, Cat(Const(1, 1), d))])
            else:
                root = rounded(radix4(1))
        m.d.comb += self.significand.eq(Mux(self.sqrt, root, quotient) if mode == 2 else (root if mode else quotient))
        return m

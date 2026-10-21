import enum
import math
from dataclasses import dataclass
from typing import NamedTuple

from amaranth.hdl import Cat, Const, Module, Mux, Signal, signed, Value
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, delay, pipe
from ._pack import Pack, pack_reference
from ._value import Zkf


class DivsqrtResult(NamedTuple):
    y: Zkf
    error: bool


@dataclass(frozen=True, kw_only=True, eq=False)
class Divsqrt(Operator):
    """
    `y = a / b` at `mode=0`, `y = sqrt(a)` at `mode=1`, either by `op_sqrt` per transaction at `mode=2`, correctly
    rounded, at one latency; a fixed mode builds only its operation, and a root does not read `b`. `error` flags
    `b = 0`, where `0 / 0 = +0`, or `a < 0` for a root, which gives `-inf`.

    Both run the radix-4 digit recurrence `w' = 4w - F_digit` on the partial remainder `w`, in one fixed-point frame and
    a digit per register stage: each digit, 0 to 3, is the largest whose subtrahend fits, found by three subtractions
    side by side, with no redundant digits. A division works on the significands, its quotient brought into [1, 2) by
    doubling a dividend below the divisor. A root keeps its partial root where the divider keeps the divisor, its
    subtrahends the divider's with one bit set where those are zero, so that both share the subtractors. Stage 0
    resolves the root's first digit from the top bits of the radicand and, at even WMAN, the divider's by three-operand
    subtractions, unless `FIRST_DIGIT` gives it a stage of its own; at odd WMAN the divider ends in one radix-8 step
    instead. No significand is a tie or rounds up to 2, so the last digit is decided by compares alone, and the
    rounding selects the digits so far or one more.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core`: the registers of `Divsqrt.CoreStages` that are built; the other digits have a stage each, fixed.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    class CoreStages(enum.Flag):
        """The registers of `stage_core`."""

        FIRST_DIGIT = enum.auto()
        """At even WMAN, a stage of its own for the divider's first digit and for the root's last."""

    operator = "divsqrt"
    stage_input: int = 0
    stage_core: CoreStages = CoreStages(0)
    stage_output: int = 0
    mode: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if Divsqrt.CoreStages.FIRST_DIGIT in self.stage_core and self.fmt.wman % 2:
            raise ValueError(f"stage_core={self.stage_core}: WMAN={self.fmt.wman} is odd; leave FIRST_DIGIT out")
        if not 0 <= self.mode <= 2:
            raise ValueError(f"mode={self.mode} is outside 0..2")

    @property
    def timing(self) -> Timing:
        digits = (self.fmt.wman + 1) // 2 - 1 + len(self.stage_core)
        return Timing(digits + self.stage_input + self.stage_output, 1)

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

            # The recurrence in integers: `qfrac` fraction bits of the root of the radicand aligned to an even exponent,
            # the guard by the rule of the parity of WMAN, the sticky from the exact remainder.
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
            y = pack_reference(fmt, 0, 0, 0, exp_unbiased, significand_value, guard, 0, sticky)
            return DivsqrtResult(Zkf(fmt, y), False)

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
            return Cat(Const(0, iq + 1), value, Const(0, ww))[:ww]

        high = d3[iq + 1 :]
        wh = len(high)
        if mode == 0:
            # A priority select by the compares measures fastest for the divider alone.
            w_next = Mux(ge3, remainders[2], Mux(ge2, remainders[1], Mux(ge1, remainders[0], scaled[:ww])))
            d_next, d3_next = d, d3
        elif mode == 1:
            # Selected by the digit, which lets a tool fold the candidates for `d3` into one adder.
            lower, upper, upper2 = ((high + step)[:wh] for step in (-3, 3, 6))
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
            # Selected by the compares, which are monotonic: a level shallower than by the digit, where the divider and
            # the root share the path. The root's candidates fall back to `high` for the divider, so the select of `d3`
            # stays 4:1.
            w_next = Mux(ge2, Mux(ge3, remainders[2], remainders[1]), Mux(ge1, remainders[0], scaled[:ww]))
            lower, upper, upper2 = (Mux(sqrt, value[:wh], high) for value in (high - 3, high + 3, high + 6))
            d3_next = Cat(d3[: iq + 1], Mux(ge2, Mux(ge3, upper2, upper), Mux(ge1, high, lower)))
            d_next = d | placed(digit & sqrt.replicate(2))
        m.d.comb += self.w_next.eq(w_next)
        m.d.comb += self.d_next.eq(d_next)
        m.d.comb += self.d3_next.eq(d3_next)
        return m


class _First(NamedTuple):
    """What the first stage of a root or of a division gives the registers of stage 0."""

    zero: Value
    inf: Value
    sign: Value
    exp: Value
    err: Value
    prefix: Value
    w: Value
    d: Value
    d3: Value


class _Divsqrt(wiring.Component):
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
        wexp, wman, wfrac, weu = fmt.wexp, fmt.wman, fmt.wfrac, fmt.wexp + 2
        even = wman % 2 == 0
        fold = even and Divsqrt.CoreStages.FIRST_DIGIT not in model.stage_core
        stages = wman // 2 - int(fold)  # the registers of the digit pipeline
        divides, roots = mode != 1, mode != 0
        wframe = wman if roots and even else wfrac  # fraction bits of the frame
        ww = (2 if roots else 1) + wframe
        layout = FloatLayout(fmt)

        operands = data.StructLayout({"a": layout, "b": layout, "op_sqrt": 1})
        taken = bundle(operands, a=self.a, b=self.b, op_sqrt=self.op_sqrt)
        valid, q = pipe(m, "in", self.in_valid, operands, taken, model.stage_input)
        a, b, op = q.a, q.b, q.op_sqrt

        # The `_First` of each operation: the exponent is biased, a quotient's before its normalization, and the state
        # is that after the first digit, of which an unfolded divider has none.
        a_zero = Signal(name="a_zero")
        m.d.comb += a_zero.eq(a.is_zero)
        a_inf = Signal(name="a_inf")
        m.d.comb += a_inf.eq(a.is_inf)
        root = quotient = None
        lowered = None
        if roots:
            negative = Signal(name="negative")
            m.d.comb += negative.eq(a.sign & ~a_zero & op if mode == 2 else a.sign & ~a_zero)
            m.submodules.root0 = root0 = _Root0(wman, wframe)
            m.d.comb += root0.frac.eq(a.frac)
            m.d.comb += root0.r.eq(~a.exp[0])
            # The bias is odd, so the halved sum is the floor of half the unbiased exponent, biased again.
            halved = Cat((a.exp + fmt.bias)[1 : wexp + 1], Const(0, 2)).as_signed()
            root = _First(
                zero=a_zero,
                inf=a_inf | negative,
                sign=negative,
                exp=halved,
                err=negative,
                prefix=Cat(root0.digit, Const(1, 1)),
                w=root0.w,
                d=root0.d,
                d3=root0.d3,
            )
        if divides:
            b_zero = Signal(name="b_zero")
            m.d.comb += b_zero.eq(b.is_zero & ~op if mode == 2 else b.is_zero)
            b_inf = Signal(name="b_inf")
            m.d.comb += b_inf.eq(b.is_inf)
            m.submodules.div0 = div0 = _Div0(wman, fold)
            m.d.comb += div0.sa.eq(a.significand)
            m.d.comb += div0.sb.eq(b.significand)
            below, above = Const(0, wframe - wfrac), Const(0, ww + 2 - wman - (wframe - wfrac))
            den = Cat(below, b.significand, above)
            den3 = Signal(ww + 2, name="den3")
            m.d.comb += den3.eq((den + Cat(Const(0, 1), den[:-1]))[: ww + 2])
            lowered = Signal(name="lowered", reset_less=True)
            m.d.sync += lowered.eq(div0.h & ~op if mode == 2 else div0.h)
            quotient = _First(
                zero=a_zero | b_inf,
                inf=~a_zero & ~b_inf & (b_zero | a_inf),
                sign=a.sign ^ (b.sign & ~b_zero),
                exp=(a.exp - b.exp + fmt.bias)[:weu].as_signed(),
                err=b_zero,
                # The integer bit and the first digit, of which an unfolded divider has none yet.
                prefix=Cat(div0.digit, Const(1, 1)) if fold else Const(1, 3),
                w=Cat(below, div0.rem, above)[:ww],
                d=den[:ww],
                d3=den3,
            )
        found = root or quotient
        if root is not None and quotient is not None:
            found = _First(*(Mux(op, r, q) for r, q in zip(root, quotient)))._replace(err=quotient.err | root.err)
        assert found is not None
        zero0, inf0, sign0, exp0, err0, prefix0, w0, d0, d30 = found

        valid0 = Signal(name="valid0")
        m.d.sync += valid0.eq(valid)
        valid = valid0
        ops = None
        if mode == 2:
            ops = Signal(name="op0", reset_less=True)
            m.d.sync += ops.eq(op)
        w, d, d3 = (
            Signal(ww, name="w0", reset_less=True),
            Signal(ww, name="d0", reset_less=True),
            Signal(ww + 2, name="d30", reset_less=True),
        )
        m.d.sync += [w.eq(w0), d.eq(d0), d3.eq(d30)]
        exp = Signal(signed(weu), name="exp0", reset_less=True)
        m.d.sync += exp.eq(exp0)
        if lowered is not None:
            exp = (exp - lowered)[:weu].as_signed()
        sign, zero = Signal(name="sign0", reset_less=True), Signal(name="zero0", reset_less=True)
        m.d.sync += [sign.eq(sign0), zero.eq(zero0)]
        inf, err = Signal(name="inf0", reset_less=True), Signal(name="err0", reset_less=True)
        m.d.sync += [inf.eq(inf0), err.eq(err0)]
        first = prefix0
        if divides and stages == 1:
            first = Signal(3, name="first")
            m.d.comb += first.eq(prefix0)
        prefix = plus1 = None
        if divides:
            prefix = Signal(3, name="prefix0", reset_less=True)
            m.d.sync += prefix.eq(first)
        if stages == 1:
            plus1 = Signal(3, name="plus1", reset_less=True)
            m.d.sync += plus1.eq((first + 1)[:3])
        inj = None
        if mode == 2:  # what the root puts into the subtrahends of the next stage, registered off its compares
            assert quotient is not None
            held_bits = Cat(quotient.d[wframe - 4], quotient.d[wframe - 3], quotient.d3[wframe - 4])
            inj = Signal(3, name="inj0", reset_less=True)
            m.d.sync += inj.eq(op.replicate(3) | held_bits)

        d5 = d7 = Const(0, ww + 3)
        for s in range(1, stages):
            iq = wframe - 2 * s - 2 if roots else 0  # where the root's quarter weight is after `s` digits
            m.submodules[f"step{s}"] = step = DivsqrtStep(ww, mode=mode, iq=iq)
            m.d.comb += [step.w.eq(w), step.d.eq(d), step.d3.eq(d3)]
            if ops is not None and inj is not None:
                m.d.comb += [step.sqrt.eq(ops), step.inj.eq(inj)]
            taken_so_far = prefix if prefix is not None else d[wframe + 1 - 2 * s : wframe + 2]  # `d` is twice the root
            if s == stages - 1:
                # The prefix plus one is formed beside the last digit, so that rounding only selects.
                carried = Cat(Const(0, 2), (taken_so_far + 1)[: 2 * s + 1])
                plus1 = Signal(2 * s + 3, name="plus1", reset_less=True)
                m.d.sync += plus1.eq(Mux(step.digit == 3, carried, Cat((step.digit + 1)[:2], taken_so_far)))
                if divides and not even:  # 5 and 7 times the divisor, for the step of radix 8 that ends it
                    d5 = Signal(ww + 3, name="d5", reset_less=True)
                    m.d.sync += d5.eq((d + Cat(Const(0, 2), d))[: ww + 3])
                    d7 = Signal(ww + 3, name="d7", reset_less=True)
                    m.d.sync += d7.eq((Cat(Const(0, 3), d) - d)[: ww + 3])
            if mode == 2:
                assert ops is not None
                held_bits = Cat(d[iq - 2], d[iq - 1], d3[iq - 2]) if iq >= 2 else None
                inj = Signal(3, name=f"inj{s}", reset_less=True)
                if held_bits is not None:
                    m.d.sync += inj.eq(ops.replicate(3) | held_bits)
                else:
                    m.d.comb += inj.eq(0)
                op_q = Signal(name=f"op{s}", reset_less=True)
                m.d.sync += op_q.eq(ops)
                ops = op_q
            if prefix is not None:
                prefix_q = Signal(2 * s + 3, name=f"prefix{s}", reset_less=True)
                m.d.sync += prefix_q.eq(Cat(step.digit, prefix))
                prefix = prefix_q
            valid_q = Signal(name=f"valid{s}")
            m.d.sync += valid_q.eq(valid)
            valid = valid_q
            w, d, d3 = (
                Signal(ww, name=f"w{s}", reset_less=True),
                Signal(ww, name=f"d{s}", reset_less=True),
                Signal(ww + 2, name=f"d3{s}", reset_less=True),
            )
            m.d.sync += [w.eq(step.w_next), d.eq(step.d_next), d3.eq(step.d3_next)]
            exp_q = Signal(signed(weu), name=f"exp{s}", reset_less=True)
            m.d.sync += exp_q.eq(exp)
            sign_q, zero_q = Signal(name=f"sign{s}", reset_less=True), Signal(name=f"zero{s}", reset_less=True)
            inf_q, err_q = Signal(name=f"inf{s}", reset_less=True), Signal(name=f"err{s}", reset_less=True)
            m.d.sync += [sign_q.eq(sign), zero_q.eq(zero), inf_q.eq(inf), err_q.eq(err)]
            exp, sign, zero, inf, err = exp_q, sign_q, zero_q, inf_q, err_q

        assert plus1 is not None
        wp = 2 * stages + 1
        m.submodules.last = last = _Last(wman, mode, fold, ww, wp)
        m.d.comb += last.sqrt.eq(ops if ops is not None else Const(mode, 1))
        m.d.comb += last.w.eq(w)
        m.d.comb += last.d.eq(d)
        m.d.comb += last.d3.eq(d3)
        m.d.comb += last.d5.eq(d5)
        m.d.comb += last.d7.eq(d7)
        m.d.comb += last.inj.eq(Cat(inj[0], inj[2]) if inj is not None else Const(0, 2))
        m.d.comb += last.prefix.eq(prefix if prefix is not None else Const(0, wp))
        m.d.comb += last.prefix1.eq(plus1)

        m.submodules.pack = pack = Pack(
            fmt,
            exp_is_biased=True,
            assume_no_overflow=mode == 1,
            stage_output=model.stage_output,
        )
        m.d.comb += pack.in_valid.eq(valid)
        m.d.comb += pack.sign.eq(sign)
        m.d.comb += pack.force_zero.eq(zero)
        m.d.comb += pack.force_inf.eq(inf)
        m.d.comb += pack.exp_unbiased.eq(exp)
        m.d.comb += pack.significand.eq(last.significand)
        for tail in (pack.guard, pack.round, pack.sticky):
            m.d.comb += tail.eq(0)
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        m.d.comb += self.error.eq(delay(m, "error", 1, err, model.stage_output))
        return m


def _sum3(either: Value, both: Value, r: Value, carry_in: int) -> Value:
    """`p + q + r + carry_in` on one carry chain, given `p ^ q` and `p & q`: a carry-save layer and one addition."""
    carry = both | (r & either)
    return ((either ^ r) + Cat(Const(carry_in, 1), carry[:-1]))[: len(r)]


def _ge(left: Value, right: Value) -> Value:
    """`left >= right` as the borrow of a subtraction, which goes onto a carry chain."""
    return ~(left - right)[-1]


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
        m.d.comb += plain.eq((sa - sb)[: wman + 1])
        doubled = Signal(wman + 1, name="r1")
        m.d.comb += doubled.eq((Cat(Const(0, 1), sa) - sb)[: wman + 1])
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
    The root's stage 0. The radicand `m' = (1 + frac) * 2**r` is in [1, 4), `r` being set at an odd unbiased exponent;
    the root starts at 1, so the remainder is `m' - 1`, `d = 2`, `d3 = 8`, and the last digit weighs 1. The first digit
    counts the thresholds 1.5625, 2.25 and 3.0625 that `m'` reaches, read off the top five bits of the fraction and not
    off subtractions; four times the remainder less 2.25, 5 and 8.25 is formed beside it. The outputs are in a frame of
    `wframe` fraction bits.
    """

    frac: Signal
    r: Signal
    digit: Signal
    w: Signal
    d: Signal
    d3: Signal

    def __init__(self, wman: int, wframe: int) -> None:
        self._wman, self._wframe = wman, wframe
        ports = {"frac": In(wman - 1), "r": In(1), "digit": Out(2)}
        super().__init__(ports | {"w": Out(wframe + 2), "d": Out(wframe + 2), "d3": Out(wframe + 4)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wman, wfrac, wframe, frac, r = self._wman, self._wman - 1, self._wframe, self.frac, self.r
        t = Cat(Const(0, 5), frac)[wfrac : wfrac + 5]
        ge1 = Signal(name="ge1")
        m.d.comb += ge1.eq(r | (t[4] & t[1:4].any()))
        ge2 = Signal(name="ge2")
        m.d.comb += ge2.eq(r & t[2:5].any())
        digit = Signal(2, name="first")
        m.d.comb += digit.eq(Cat((r & t[4] & t[0:4].any()) | (ge1 & ~ge2), ge2))
        radicand = Mux(r, Cat(Const(0, 1), frac, Const(1, 1)), Cat(frac, Const(1, 2)))
        m1 = Signal(wman + 1, name="m1")
        m.d.comb += m1.eq((radicand - (1 << wfrac))[: wman + 1])
        scaled = Cat(Const(0, 2 + wframe - wfrac), m1)  # four times the remainder, in the frame
        h = scaled[wframe - 2 :]  # in units of a quarter of the last digit's weight
        h_next, d3_half = Signal(6, name="h_next"), Signal(5, name="d3_half")  # `d3` in halves of that weight
        with m.Switch(digit):
            for value, (less, halves) in enumerate(((0, 13), (9, 16), (20, 19), (33, 22))):
                with m.Case(value) if value < 3 else m.Default():
                    m.d.comb += h_next.eq((h - less)[:6] if less else h)
                    m.d.comb += d3_half.eq(Const(halves, 5))
        m.d.comb += self.digit.eq(digit)
        m.d.comb += self.w.eq(Cat(scaled[: wframe - 2], h_next[:4]))
        m.d.comb += self.d.eq(Cat(Const(0, wframe - 1), digit, Const(1, 1)))
        m.d.comb += self.d3.eq(Cat(Const(0, wframe - 1), d3_half))
        return m


class _Last(wiring.Component):
    """
    The last digit, decided by compares alone, and the rounding that follows from it as a selection between the `prefix`
    of digits and `prefix1 = prefix + 1`. A build with no divider takes the prefix from `d = 2Q`.

    At even WMAN, folded, it is the last digit of either operation, from `4w >= F1` and `4w >= F3`, the root's quarter
    weight being at bit 0 of the frame (`inj` gives those bits at `mode` 2). Unfolded, it is that digit of the divider,
    and the root has all its digits and rounds half up. At odd WMAN it is the divider's last two bits and its guard in
    one step of radix 8, from `8w >= k * d` for `k` of 1, 3, 5 and 7 (`d5` and `d7` are given), and the root's guard is
    the next digit's `4w >= 2d | u`.
    """

    sqrt: Signal
    w: Signal
    d: Signal
    d3: Signal
    d5: Signal
    d7: Signal
    inj: Signal
    prefix: Signal
    prefix1: Signal
    significand: Signal

    def __init__(self, wman: int, mode: int, fold: bool, ww: int, wp: int) -> None:
        self._wman, self._mode, self._fold, self._ww, self._wp = wman, mode, fold, ww, wp
        state = {"sqrt": In(1), "w": In(ww), "d": In(ww), "d3": In(ww + 2), "d5": In(ww + 3), "d7": In(ww + 3)}
        digits = {"inj": In(2), "prefix": In(wp), "prefix1": In(wp), "significand": Out(wman)}
        super().__init__(state | digits)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wman, mode, ww, w, d, d3, prefix1 = self._wman, self._mode, self._ww, self.w, self.d, self.d3, self.prefix1
        divides, roots = mode != 1, mode != 0
        wframe = ww - (2 if roots else 1)
        m4 = Cat(Const(0, 2), w, Const(0, 1))
        prefix = self.prefix if divides else d[wframe + 2 - self._wp : wframe + 2]

        def either(root: Value, quotient: Value) -> Value:
            """`root` or `quotient` by the operation, with no multiplexer in a fixed mode."""
            return Mux(self.sqrt, root, quotient) if mode == 2 else (root if mode == 1 else quotient)

        if wman % 2 == 0:
            if mode == 1 and not self._fold:  # the root has all its digits: it is rounded half up
                m.d.comb += self.significand.eq(prefix1[1 : wman + 1])
                return m
            bits = (Const(3, 2) if mode == 1 else self.inj) if self._fold and roots else Cat(d[0], d3[0])
            ge1 = _ge(m4, Cat(bits[0], d[1:], Const(0, 3)))
            ge3 = _ge(m4, Cat(bits[1], d3[1:], Const(0, 1)))
            digit = Mux(ge3, Cat(Const(0, 1), prefix1[: wman - 1]), Cat(ge1, prefix[: wman - 1]))
            m.d.comb += self.significand.eq(digit if self._fold else either(prefix1[1 : wman + 1], digit))
            return m

        quotient, root = Const(0, wman), Const(0, wman)
        if divides:
            m8 = Cat(Const(0, 3), w)
            ge = [_ge(m8, Cat(d, Const(0, 3))), _ge(m8, Cat(d3, Const(0, 1))), _ge(m8, self.d5)]
            tail = Signal(2, name="tail")
            with m.Switch(Cat(*ge)):
                for value, pattern in ((3, 0b111), (2, 0b011), (1, 0b001), (0, None)):
                    with m.Default() if pattern is None else m.Case(pattern):
                        m.d.comb += tail.eq(Const(value, 2))
            exact = Cat(tail, prefix[: wman - 2])
            quotient = Mux(_ge(m8, self.d7), Cat(Const(0, 2), prefix1[: wman - 2]), exact)
        if roots:
            root = Mux(_ge(m4, Cat(Const(1, 1), d, Const(0, 2))), prefix1[:wman], prefix[:wman])
        m.d.comb += self.significand.eq(either(root, quotient))
        return m

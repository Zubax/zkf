from dataclasses import dataclass, field

from amaranth.hdl import Cat, Const, Module, Mux, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._add import Add, AddCore, check_core
from ._block import Attrs, FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._pmul import Pmul, check
from ._value import Zkf


@dataclass(frozen=True, kw_only=True, eq=False)
class Fma(Operator):
    """
    `y = a * b + c`, rounded once; `a * b - c` with the sign bit of `c` flipped.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core_mul`: the product's registers, as Mul's.
    `stage_core_add`: the sum's registers, a set of `Add.CoreStages`; its decoded operands are the product and `c`.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    `wmultiplier`: the operand width of the target's multipliers.
    """

    operator = "fma"
    stage_input: int = 0
    stage_core_mul: int = 0
    stage_core_add: Add.CoreStages = Add.CoreStages(0)
    stage_output: int = 0
    wmultiplier: int
    mul_attrs: Attrs | None = field(default=None, compare=False)
    """
    On the last input stage at `stage_core_mul=1`, which the multiplier takes the factors from; from 2 on, on the
    product's capture behind the masks, where a register follows the multipliers (a grid's from 3 on).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wmultiplier < 8:
            raise ValueError(f"wmultiplier={self.wmultiplier} is below 8")
        wman = self.fmt.wman
        check("stage_core_mul", self.stage_core_mul, wman, wman, signed=False, wmultiplier=self.wmultiplier)
        check_core("stage_core_add", self.stage_core_add, 2 * wman)

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + self.stage_core_mul + len(self.stage_core_add) + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature(
            {"in_valid": In(1), "a": In(f), "b": In(f), "c": In(f), "out_valid": Out(1), "y": Out(f)}
        )

    def component(self) -> _Fma:
        return _Fma(self)

    def evaluate(self, a: Zkf, b: Zkf, c: Zkf) -> Zkf:
        if a.fmt != self.fmt or b.fmt != self.fmt or c.fmt != self.fmt:
            raise ValueError(f"an operand is not of {self.fmt}")
        fmt = self.fmt
        p_zero = a.is_zero or b.is_zero
        p_inf = (not p_zero) and (a.is_inf or b.is_inf)
        p_sign = a.negative ^ b.negative

        if p_inf and c.is_inf:
            return fmt.inf(p_sign) if p_sign == c.negative else Zkf(fmt, 0)
        if p_inf:
            return fmt.inf(p_sign)
        if c.is_inf:
            return fmt.inf(c.negative)

        if p_zero:
            return fmt.encode(c.to_fraction())
        return fmt.encode(a.to_fraction() * b.to_fraction() + c.to_fraction())


class _Fma(wiring.Component):
    in_valid: Signal
    a: FloatView
    b: FloatView
    c: FloatView
    out_valid: Signal
    y: FloatView

    def __init__(self, model: Fma) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        wexp, wman = fmt.wexp, fmt.wman
        wmag, weu = 2 * wman, wexp + 2  # the whole product; a biased exponent with a sign, of `a` and `b` together
        layout = FloatLayout(fmt)
        # Below every finite exponent: a zero or non-finite operand, whose magnitude enters as zero, is the smaller.
        lowest = Const(-(1 << (weu - 1)), signed(weu))

        operands = data.StructLayout({"a": layout, "b": layout, "c": layout})
        taken = bundle(operands, a=self.a, b=self.b, c=self.c)
        # At `stage_core_mul=1`, which only a product of one multiplier takes, it takes the last input stage's factors
        # unmasked.
        direct = model.stage_core_mul == 1
        valid, q = pipe(m, "in", self.in_valid, operands, taken, model.stage_input, model.mul_attrs if direct else None)
        a, b, c = q.a, q.b, q.c

        # A zero factor gives a zero product, which only an infinite `c` overrides; otherwise an infinite factor gives
        # an infinite one. `ep_base` lacks the 1 that a product leading at its top bit adds, which is known later.
        side = data.StructLayout(
            {"p_sign": 1, "p_zero": 1, "p_inf": 1, "ep_base": signed(weu)}
            | {"c_sig": wman, "c_exp": wexp, "c_sign": 1, "c_zero": 1, "c_inf": 1}
        )
        a_zero, a_inf = Signal(name="a_zero"), Signal(name="a_inf")
        m.d.comb += [a_zero.eq(a.is_zero), a_inf.eq(a.is_inf)]
        b_zero, b_inf = Signal(name="b_zero"), Signal(name="b_inf")
        m.d.comb += [b_zero.eq(b.is_zero), b_inf.eq(b.is_inf)]
        c_zero, c_inf = Signal(name="c_zero"), Signal(name="c_inf")
        m.d.comb += [c_zero.eq(c.is_zero), c_inf.eq(c.is_inf)]
        p_zero = Signal(name="p_zero")
        m.d.comb += p_zero.eq(a_zero | b_zero)
        m.submodules.pmul = pmul = Pmul(
            wman,
            wman,
            signed=False,
            wmultiplier=model.wmultiplier,
            stages=model.stage_core_mul,
            sideband=side,
            attrs=model.mul_attrs,
        )
        m.d.comb += pmul.in_valid.eq(valid)
        # A non-finite operand enters as a zero magnitude, so that no mask of the product or of `c` is needed behind the
        # multiplier, where its flag would fan out over the whole magnitude; the factors only where the capture register
        # takes the mask off the multiplier's path.
        masked = model.stage_core_mul >= 2
        m.d.comb += pmul.a.eq(Mux(a_zero | a_inf, Const(0, wman), a.significand) if masked else a.significand)
        m.d.comb += pmul.b.eq(Mux(b_zero | b_inf, Const(0, wman), b.significand) if masked else b.significand)
        waiting = bundle(
            side,
            p_sign=a.sign ^ b.sign,
            p_zero=p_zero,
            p_inf=~p_zero & (a_inf | b_inf),
            ep_base=a.exp + b.exp - fmt.bias,
            c_sig=Mux(c_zero | c_inf, Const(0, wman), c.significand),
            c_exp=c.exp,
            c_sign=c.sign,
            c_zero=c_zero,
            c_inf=c_inf,
        )
        m.d.comb += pmul.sb_in.eq(waiting)

        # The product leads at its top bit or one below.
        pr, product = pmul.sb_out, pmul.p
        high = product[-1]
        p_finite = Signal(name="p_finite")
        m.d.comb += p_finite.eq(~pr.p_zero & ~pr.p_inf)
        c_finite = Signal(name="c_finite")
        m.d.comb += c_finite.eq(~pr.c_zero & ~pr.c_inf)
        keys = data.StructLayout(
            {"p_key": wmag, "c_key": wman, "ep": signed(weu), "ec": signed(weu)}
            | {"p_sign": 1, "c_sign": 1, "p_inf": 1, "c_inf": 1, "p_larger": 1}
        )
        normalized = Mux(high, product, Cat(Const(0, 1), product[:-1]))
        p_key = normalized if masked else Mux(p_finite, normalized, Const(0, wmag))
        ep = Mux(p_finite, (pr.ep_base + high)[:weu].as_signed(), lowest)
        ec = Mux(c_finite, Cat(pr.c_exp, Const(0, 2)).as_signed(), lowest)
        # The magnitude order: by exponent, then by significand with `c` aligned to the product's top; the product wins
        # a tie. Decided ahead of the decode register, as the adder decides its own, so that its fan-out over the
        # operands' selection starts at a register. The equality of the exponents is found beside their subtraction.
        level = ~(ep ^ ec).any()
        p_larger = (~(ep - ec)[-1] & ~level) | (level & ~(p_key - Cat(Const(0, wman), pr.c_sig))[-1])
        decoded = bundle(
            keys,
            p_key=p_key,
            c_key=pr.c_sig,
            ep=ep,
            ec=ec,
            p_sign=pr.p_sign,
            c_sign=pr.c_sign,
            p_inf=pr.p_inf,
            c_inf=pr.c_inf,
            p_larger=p_larger,
        )
        payload = Signal(keys, name="decoded")
        m.d.comb += payload.eq(decoded)
        valid, d = pipe(m, "decode", pmul.out_valid, keys, payload, int(Add.CoreStages.DECODE in model.stage_core_add))

        c_wide = Cat(Const(0, wman), d.c_key)
        apart = Signal(signed(weu + 1), name="apart")
        m.d.comb += apart.eq(d.ep - d.ec)
        differ = Signal(name="differ")
        m.d.comb += differ.eq(d.p_sign ^ d.c_sign)
        m.submodules.core = core = AddCore(
            fmt,
            wsig=wmag,
            we=weu,
            wshift=weu + 1,
            stages=model.stage_core_add,
            stage_output=model.stage_output,
        )
        m.d.comb += core.in_valid.eq(valid)
        m.d.comb += core.finite_sign.eq(Mux(d.p_larger, d.p_sign, d.c_sign))
        m.d.comb += core.inf_sign.eq((d.p_inf & d.p_sign) | (d.c_inf & d.c_sign))
        m.d.comb += core.same_sign.eq(~differ)
        m.d.comb += core.force_zero.eq(d.p_inf & d.c_inf & differ)
        m.d.comb += core.force_inf.eq(d.p_inf | d.c_inf)
        m.d.comb += core.exp.eq(Mux(d.p_larger, d.ep, d.ec))
        m.d.comb += core.shamt.eq(Mux(apart[-1], (-apart)[: weu + 1], apart.as_unsigned()))
        m.d.comb += core.larger.eq(Mux(d.p_larger, d.p_key, c_wide))
        m.d.comb += core.smaller.eq(Mux(d.p_larger, c_wide, d.p_key))
        m.d.comb += self.out_valid.eq(core.out_valid)
        m.d.comb += self.y.eq(core.y)
        return m

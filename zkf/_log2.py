import enum
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, overload

import mpmath

from amaranth.hdl import Cat, Const, Module, Mux, ShapeLike, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import Attrs, FloatLayout, FloatView, Operator, Timing, bundle, delay, pipe
from ._trans import Kernel, Spec, kernel_check, kernel_depth, pack_fixed, spec as table
from ._pmul import Pmul, check
from ._shift import Normshift
from ._value import UnsupportedFormat, Zkf, ZkfFormat, context

_G2, _GB = 2, 12
"""Bits the shifted kernel keeps below the kernel's own, and the last sum below the lower rounding position."""


class Log2Result[T](NamedTuple):
    y: T
    domain_error: bool
    pole: bool


@dataclass(frozen=True, kw_only=True, eq=False)
class Log2(Operator):
    """
    `y = log2(x)`, faithfully rounded; `log2(+inf) = +inf`, `log2(0) = -inf` with `pole`, and `log2(x < 0) = -inf` with
    `domain_error`.

    A tie rounds away from zero, not to even. A true tie needs `log2(2**k)` with `|k|` wider than the significand, i.e.
    WEXP >= WMAN + 2, which is refused; any other tie of the computed value is made by the kernel's own error, with both
    neighbors faithful.

    With `x = m * 2**e`, `log2(x) = e + log2(m)`. The significand is re-centered around 1, halved with `e` incremented
    from the table segment nearest `sqrt(2)` up, so that `f`, the re-centered significand less one, is exact and an `x`
    near 1 loses nothing to cancellation. A kernel like that of `Exp2` gives `C(f) = log2(1 + f) / f`, and one more
    product `f * C(f)`, aligned so that the result leads at a fixed bit or one below it: `|f|` is normalized where `e`
    is 0, and elsewhere `C` is shifted down against the leading one predicted for the result, which keeps the multiplier
    as wide as the fraction. The sum of `e` and that product is then rounded at both of those positions, and one of the
    two is selected.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core_kernel`: the registers of every product of the kernel, as that of `Exp2`.
    `stage_core_mul`: the registers of the last product, `f * C(f)`, as Mul's, without its input-stage credit.
    `stage_core`: the registers of `Log2.Core` that are built.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    `wmultiplier`: the operand width of the target's multipliers, which the table is generated for.
    """

    class Core(enum.Flag):
        """The registers of `stage_core`, in path order."""

        ALIGN = enum.auto()
        """The kernel's result and the operand, ahead of the alignment shift."""
        SELECT = enum.auto()
        """The rounded candidates, ahead of their selection."""

    operator = "log2"
    stage_input: int = 0
    stage_core_kernel: int = 0
    stage_core_mul: int = 0
    stage_core: Core = Core(0)
    stage_output: int = 0
    wmultiplier: int
    rom_attrs: Attrs | None = field(default=None, compare=False)
    """As that of `Exp2`."""
    mul_attrs: Attrs | None = field(default=None, compare=False)
    """As that of `Exp2`, and on the last product's capture where a register follows its multipliers."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wmultiplier < 8:
            raise ValueError(f"wmultiplier={self.wmultiplier} is below 8")
        if self.fmt.wexp > self.fmt.wman + 1:
            raise UnsupportedFormat(f"WEXP={self.fmt.wexp} is above WMAN + 1")
        spec = table("log2", self.fmt.wman, self.wmultiplier)
        kernel_check(self.stage_core_kernel, spec, self.wmultiplier)
        check("stage_core_mul", self.stage_core_mul, *_last(spec, self.fmt), signed=False, wmultiplier=self.wmultiplier)

    @property
    def timing(self) -> Timing:
        kernel = kernel_depth(table("log2", self.fmt.wman, self.wmultiplier), self.wmultiplier, self.stage_core_kernel)
        inside = 1 + kernel + self.stage_core_mul + len(self.stage_core)
        return Timing(self.stage_input + inside + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature(
            {"in_valid": In(1), "x": In(f), "out_valid": Out(1), "y": Out(f), "domain_error": Out(1), "pole": Out(1)}
        )

    def component(self) -> _Log2:
        return _Log2(self)

    @overload
    def evaluate(self, x: Zkf, *, oracle: Literal[False] = False) -> Log2Result[Zkf]: ...
    @overload
    def evaluate(self, x: Zkf, *, oracle: Literal[True]) -> Log2Result[mpmath.mpf]: ...
    def evaluate(self, x: Zkf, *, oracle: bool = False) -> Log2Result[Zkf] | Log2Result[mpmath.mpf]:
        """`oracle=True` gives `log2(x)` itself, as `Exp2.evaluate` gives `2**x`, with the same flags."""
        if x.fmt != self.fmt:
            raise ValueError(f"an operand of {x.fmt} where {self.fmt} is due")
        fmt = self.fmt
        if oracle:
            if x.is_zero or x.negative:
                return Log2Result(-mpmath.inf, x.negative and not x.is_zero, x.is_zero)
            prec = 4 * fmt.wman + 80
            return Log2Result(mpmath.mpf(context(prec).log(x.to_mpf(), 2), prec=prec), False, False)
        if x.is_inf and not x.negative:
            return Log2Result(fmt.inf(0), False, False)
        if x.is_zero:
            return Log2Result(fmt.inf(1), False, True)
        if x.negative:
            return Log2Result(fmt.inf(1), True, False)

        spec = table("log2", fmt.wman, self.wmultiplier)
        # x = m' * 2**e with the significand re-centered around 1 (see `Log2`), f = m' - 1, log2(m') = f * C(f).
        cf, wfrac, wexp = spec.cf, fmt.wfrac, fmt.wexp
        rc = int(x.frac >> spec.rw >= spec.thr)
        e = x.exp - fmt.bias + rc
        fmag = ((1 << wfrac) - x.frac) if rc else x.frac  # |f| at scale 2**-(WFRAC + rc)
        c = spec.at(x.frac)  # C(f) at scale 2**-CF
        assert c >> cf == 1, "log2 kernel range"
        # The alignment of `Log2`: the result leads at bit FB of the final sum or one below it.
        fb = wfrac + cf + _G2
        if e == 0:
            if fmag == 0:
                return Log2Result(Zkf(fmt, 0), False, False)  # x = 1
            lz = wfrac - fmag.bit_length()
            operand, shift, e_al, at, e_hi, sub, sign = fmag << lz, 1, 0, fb, -lz - rc, False, rc
        else:
            sub, sign = (e > 0) == bool(rc), int(e < 0)
            if abs(e) == 1 << (wexp - 1):
                p = wexp - 2
            elif sub and abs(e) == 1 and fmag:
                p = -1
            else:
                p = abs(e).bit_length() - 1
            operand, shift, e_al, at, e_hi = fmag, rc + p + 1, abs(e), fb - p, p
        # The sum keeps _GB bits below the lower of the two rounding positions; every bit of |e| is above them.
        low = cf + _G2 - 2 - _GB
        prod = (operand * ((c << (_G2 + 1)) >> shift)) >> low
        e_al, fb = e_al << (at - low), fb - low
        v = (e_al - prod) if sub else (e_al + prod)
        assert 1 << (fb - 1) <= v <= 1 << (fb + 1), "log2 leading-one prediction"
        return Log2Result(Zkf(fmt, pack_fixed(fmt, sign, e_hi, v, fb)), False, False)


def _last(spec: Spec, fmt: ZkfFormat) -> tuple[int, int]:
    """The operand widths of the last product: the fraction, and the kernel's result shifted."""
    return fmt.wfrac, spec.cf + _G2 + 2


class _Log2(wiring.Component):
    in_valid: Signal
    x: FloatView
    out_valid: Signal
    y: FloatView
    domain_error: Signal
    pole: Signal

    def __init__(self, model: Log2) -> None:
        self._model, self._spec = model, table("log2", model.fmt.wman, model.wmultiplier)
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        wexp, wfrac = fmt.wexp, fmt.wfrac
        spec, core = self._spec, model.stage_core
        f = FloatLayout(fmt)
        valid, x = pipe(m, "in", self.in_valid, f, self.x, model.stage_input)
        m.submodules.kernel = kernel = Kernel(
            spec,
            wmultiplier=model.wmultiplier,
            stages=model.stage_core_kernel,
            rom_attrs=model.rom_attrs,
            mul_attrs=model.mul_attrs,
        )
        m.d.comb += kernel.in_valid.eq(valid)
        m.d.comb += kernel.frac.eq(x.frac)

        # The side path, as deep as the kernel: the raw fields, held beside the table's register so that only `rc` has
        # logic ahead of it; then the decoding, over the stages that the kernel leaves it; last the normalizer's.
        decode_stage, hold = int(kernel.depth >= 3), int(kernel.depth >= 2)
        raw = data.StructLayout({"x": f, "rc": 1})
        taken = bundle(raw, x=x, rc=kernel.rc)
        s_valid, a = pipe(m, "raw", valid, raw, taken, kernel.depth - decode_stage - hold)
        m.submodules.decode = decode = _Decode(fmt, decode_stage)
        m.d.comb += decode.in_valid.eq(s_valid)
        m.d.comb += decode.rc.eq(a.rc)
        m.d.comb += decode.sign.eq(a.x.sign)
        m.d.comb += decode.exp.eq(a.x.exp)
        m.d.comb += decode.frac.eq(a.x.frac)
        names = ("pole", "domain_error", "is_inf", "one", "neg", "sub", "rc_q", "ezero", "emag", "p1", "sh", "fmag")
        decoded = data.StructLayout({name: len(getattr(decode, name)) for name in names})
        fields = {name: getattr(decode, name) for name in names}
        m_valid, d = pipe(m, "decoded", decode.out_valid, decoded, bundle(decoded, **fields), hold)

        # `|f|` is normalized for `e = 0`; `|e|` is shifted so that the predicted leading one of the result lands on its
        # bit `wexp - 2`.
        m.submodules.normalize = normalize = Normshift(wfrac, stage_split=1, sideband=decoded)
        up = Signal(len(d.p1), name="up")
        m.d.comb += up.eq((Const(wexp - 1, len(d.p1)) - d.p1)[: len(d.p1)])
        placed = {name: d[name] for name in names} | {"emag": (d.emag << up)[:wexp]}
        m.d.comb += normalize.in_valid.eq(m_valid)
        m.d.comb += normalize.x.eq(d.fmag)
        m.d.comb += normalize.sb_in.eq(bundle(decoded, **placed))
        n = normalize.sb_out
        kept = data.StructLayout({name: shape for name, shape in decoded.members.items() if name != "fmag"})
        aligned = data.StructLayout({"side": kept, "lz": len(normalize.count), "operand": wfrac, "c": spec.cf + 1})
        side = bundle(kept, **{name: n[name] for name in kept.members})
        ready = bundle(aligned, side=side, lz=normalize.count, operand=Mux(n.ezero, normalize.y, n.fmag), c=kernel.c)
        payload = Signal(aligned, name="ready")
        m.d.comb += payload.eq(ready)
        a_valid, q = pipe(m, "align", kernel.out_valid, aligned, payload, int(Log2.Core.ALIGN in core))

        m.submodules.final = final = Pmul(
            *_last(spec, fmt),
            signed=False,
            wmultiplier=model.wmultiplier,
            stages=model.stage_core_mul,
            attrs=model.mul_attrs,
        )
        m.d.comb += final.in_valid.eq(a_valid)
        m.d.comb += final.a.eq(q.operand)
        m.d.comb += final.b.eq(Cat(Const(0, _G2 + 1), q.c) >> q.side.sh)
        flags = data.StructLayout({"pole": 1, "domain_error": 1})
        m.submodules.pack = pack = _Pack(
            fmt, wgb=_GB, stage=final.latency, stage_select=int(Log2.Core.SELECT in core), sideband=flags
        )
        m.d.comb += [pack.p1.eq(q.side.p1), pack.ezero.eq(q.side.ezero), pack.sub.eq(q.side.sub)]
        m.d.comb += [pack.neg.eq(q.side.neg), pack.one.eq(q.side.one), pack.is_inf.eq(q.side.is_inf)]
        m.d.comb += pack.lz.eq(q.lz)
        m.d.comb += pack.eal.eq(q.side.emag)
        m.d.comb += pack.rc.eq(q.side.rc_q)
        m.d.comb += pack.sb_in.eq(bundle(flags, pole=q.side.pole, domain_error=q.side.domain_error))
        m.d.comb += pack.in_valid.eq(final.out_valid)
        m.d.comb += pack.prod.eq(final.p[spec.cf + _G2 - 2 - _GB :])
        result = data.StructLayout({"y": f, "pole": 1, "domain_error": 1})
        given = bundle(result, y=pack.y, pole=pack.sb_out.pole, domain_error=pack.sb_out.domain_error)
        out_valid, r = pipe(m, "out", pack.out_valid, result, given, model.stage_output)
        m.d.comb += self.out_valid.eq(out_valid)
        m.d.comb += self.y.eq(r.y)
        m.d.comb += self.pole.eq(r.pole)
        m.d.comb += self.domain_error.eq(r.domain_error)
        return m


class _Decode(wiring.Component):
    """
    From the fields of `x` and `rc`, which says that the significand is re-centered (`m' = m / 2`, `e + 1`): `e`,
    `f = m' - 1`, and where the result `|e + log2(m')| = |e| +- l` will lead.

    `fmag = |f|`: the fraction, or its two's complement when re-centered, where it stands one bit lower; `rc` implies a
    nonzero fraction. `emag = |e|`. With `n` the index of its leading one, the result leads at `n` or `n - 1`; `|e| = 1`
    less a nonzero `l` leads at -1 or -2, and `|e| = 2**(wexp - 1)`, which is always less `l`, at `wexp - 2`. `p1` is
    one more than the upper of each pair. The kernel is aligned to it by a right shift of `sh = rc + p1`, or of 1 when
    `e = 0` (`ezero`), where `|f|` is normalized instead.

    `sub`: `e` and `f` differ in sign, so that the result is `|e| - l`. `neg` is the sign of the result. `one`: `x = 1`,
    which gives +0. `is_inf`: the result is an infinity, `x` being zero (`pole`), negative (`domain_error`) or +inf.
    `stage=1` puts a register between `e` and `|f|` and what is derived from them; `rc_q` is `rc` beside the outputs.
    """

    in_valid: Signal
    rc: Signal
    sign: Signal
    exp: Signal
    frac: Signal
    out_valid: Signal
    fmag: Signal
    sh: Signal
    p1: Signal
    emag: Signal
    ezero: Signal
    rc_q: Signal
    sub: Signal
    neg: Signal
    one: Signal
    is_inf: Signal
    domain_error: Signal
    pole: Signal

    def __init__(self, fmt: ZkfFormat, stage: int) -> None:
        self._fmt, self._stage = fmt, stage
        wp = ceil_log2(fmt.wexp + 1)
        taken = {"in_valid": In(1), "rc": In(1), "sign": In(1), "exp": In(fmt.wexp), "frac": In(fmt.wfrac)}
        given = {"out_valid": Out(1), "fmag": Out(fmt.wfrac), "sh": Out(wp), "p1": Out(wp), "emag": Out(fmt.wexp)}
        flags = ("ezero", "rc_q", "sub", "neg", "one", "is_inf", "domain_error", "pole")
        super().__init__(taken | given | {name: Out(1) for name in flags})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, wexp, wfrac = self._fmt, self._fmt.wexp, self._fmt.wfrac
        wp = len(self.p1)
        bias = Signal(wexp + 1, name="bias")
        m.d.comb += bias.eq(Mux(self.rc, Const(fmt.bias - 1, wexp + 1), Const(fmt.bias, wexp + 1)))
        above = Signal(wexp + 1, name="e_p")
        m.d.comb += above.eq((self.exp - bias)[: wexp + 1])
        below = (bias[:wexp] - self.exp)[:wexp]  # |e| of a negative e
        split = data.StructLayout(
            {"rc": 1, "sign": 1, "inf": 1, "pole": 1, "f_zero": 1, "ezero": 1, "e_neg": 1, "emag": wexp, "fmag": wfrac}
        )
        taken = bundle(
            split,
            rc=self.rc,
            sign=self.sign,
            inf=self.exp.all(),
            pole=~self.exp.any(),
            f_zero=~self.frac.any(),
            ezero=~above.any(),
            e_neg=above[-1],
            emag=Mux(above[-1], below, above[:wexp]),
            fmag=Mux(self.rc, (~self.frac + 1)[:wfrac], self.frac),
        )
        payload = Signal(split, name="taken")
        m.d.comb += payload.eq(taken)
        valid, q = pipe(m, "split", self.in_valid, split, payload, self._stage)

        leading = Signal(wp, name="msb1")  # one more than the index of the leading one of |e|
        m.d.comb += leading.eq(0)
        for i in range(wexp - 1):
            with m.If(q.emag[i]):
                m.d.comb += leading.eq(Const(i + 1, wp))
        ezero, e_neg, rc = Signal(name="ezero"), Signal(name="e_neg"), Signal(name="rc")
        m.d.comb += [ezero.eq(q.ezero), e_neg.eq(q.e_neg), rc.eq(q.rc)]
        pole = Signal(name="x_zero")
        m.d.comb += pole.eq(q.pole)
        sign = Signal(name="x_sign")
        m.d.comb += sign.eq(q.sign)
        sub = Signal(name="differ")
        m.d.comb += sub.eq(~ezero & (e_neg ^ rc))
        is_inf = Signal(name="infinite")
        m.d.comb += is_inf.eq(pole | q.inf | sign)
        f_zero = Signal(name="f_zero")
        m.d.comb += f_zero.eq(q.f_zero)
        p1 = Signal(wp, name="p1_c")
        with m.If(q.emag[-1]):
            m.d.comb += p1.eq(Const(wexp - 1, wp))
        with m.Elif(sub & (q.emag == 1) & ~f_zero):
            m.d.comb += p1.eq(0)
        with m.Else():
            m.d.comb += p1.eq(leading)
        with m.If(is_inf):
            m.d.comb += self.neg.eq(pole | sign)
        with m.Elif(ezero):
            m.d.comb += self.neg.eq(rc)
        with m.Else():
            m.d.comb += self.neg.eq(e_neg)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.fmag.eq(q.fmag)
        m.d.comb += self.sh.eq(Mux(ezero, Const(1, wp), (p1 + rc)[:wp]))
        m.d.comb += self.p1.eq(p1)
        m.d.comb += self.emag.eq(q.emag)
        m.d.comb += self.ezero.eq(ezero)
        m.d.comb += self.rc_q.eq(rc)
        m.d.comb += self.sub.eq(sub)
        m.d.comb += self.one.eq(ezero & f_zero)
        m.d.comb += self.is_inf.eq(is_inf)
        m.d.comb += self.domain_error.eq(sign & ~pole)
        m.d.comb += self.pole.eq(pole)
        return m


class _Pack(wiring.Component):
    """
    `y = (-1)**neg * (|e| +- l)` from `eal`, `|e|` aligned to where the result is predicted to lead, and `l = prod`:
    what `Pack` gives of that sum, a tie rounding away from zero.

    The sum has `wfrac + wgb + 4` bits and leads at its second bit from the top, "hi", or one below, "lo", or is exactly
    its top bit; the caller guarantees that frame. Two sums, each with a half ULP of its own, cover both positions: the
    top bits of the lower one select, and a carry out of the upper one gives the next exponent with a zero fraction. The
    record of each candidate (sign, whether the fraction is kept, exponent field), with the zero, `min_normal` and
    normal classes and the special results folded in, is formed from the side inputs, which arrive `stage` register
    stages ahead of `prod`. A sum whose upper candidate is `min_normal` takes no half ULP: there the sum itself decides
    the class.

    A candidate's exponent is `p1 - 1` where `e` is not zero, and `-rc - lz` where it is (`ezero`), `lz` being the count
    that normalized `|f|`. `sub`: the sum is `|e| - prod`. `one` gives +0 and `is_inf` an infinity, which overrides the
    rest. `stage_select` is the number of register stages between the rounded candidates and their selection; `sb_in`
    travels beside the side inputs to `sb_out`.
    """

    lz: Signal
    p1: Signal
    eal: Signal
    ezero: Signal
    rc: Signal
    sub: Signal
    neg: Signal
    one: Signal
    is_inf: Signal
    sb_in: Signal
    in_valid: Signal
    prod: Signal
    out_valid: Signal
    y: FloatView
    sb_out: Signal

    def __init__(self, fmt: ZkfFormat, *, wgb: int, stage: int, stage_select: int, sideband: ShapeLike) -> None:
        self._fmt, self._wgb, self._stage, self._stage_select, self._sideband = fmt, wgb, stage, stage_select, sideband
        side = {"lz": In(ceil_log2(fmt.wfrac)), "p1": In(ceil_log2(fmt.wexp + 1)), "eal": In(fmt.wexp)}
        flags = {name: In(1) for name in ("ezero", "rc", "sub", "neg", "one", "is_inf")}
        late = {"sb_in": In(sideband), "in_valid": In(1), "prod": In(fmt.wman + wgb + 3)}
        super().__init__(
            side | flags | late | {"out_valid": Out(1), "y": Out(FloatLayout(fmt)), "sb_out": Out(sideband)}
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, wgb, wexp, wfrac = self._fmt, self._wgb, self._fmt.wexp, self._fmt.wfrac
        ws = wfrac + wgb + 4
        web = max(len(self.lz), wexp) + 3  # of the signed biased exponent of a candidate
        record = data.StructLayout({"exp": wexp, "keep": 1, "sign": 1})
        is_inf, neg, one = Signal(name="is_inf_c"), Signal(name="neg_c"), Signal(name="one_c")
        m.d.comb += [is_inf.eq(self.is_inf), neg.eq(self.neg), one.eq(self.one)]
        frame = Signal(2, name="frame")
        m.d.comb += frame.eq(Cat(self.rc, self.ezero))
        lz, p1 = Cat(self.lz, Const(0, web - len(self.lz))), Cat(self.p1, Const(0, web - len(self.p1)))
        offset = Signal(signed(web), name="off")
        m.d.comb += offset.eq(Mux(frame[1], ~lz, p1).as_signed())
        records, least = [], []
        for k, name in enumerate(("lo", "hi", "cy")):
            base = Signal(signed(web), name=f"base_{name}")
            with m.Switch(frame):
                with m.Case(0b10):
                    m.d.comb += base.eq(Const(fmt.bias + k, signed(web)))
                with m.Case(0b11):
                    m.d.comb += base.eq(Const(fmt.bias + k - 1, signed(web)))
                with m.Default():
                    m.d.comb += base.eq(Const(fmt.bias + k - 2, signed(web)))
            biased = Signal(signed(web), name=f"be_{name}")
            m.d.comb += biased.eq((base + offset)[:web].as_signed())
            is_least = Signal(name=f"least_{name}")
            m.d.comb += is_least.eq(biased == 0)
            candidate = Signal(record, name=f"rec_{name}")
            with m.If(is_inf):
                m.d.comb += candidate.eq(bundle(record, exp=Const((1 << wexp) - 1, wexp), keep=Const(0, 1), sign=neg))
            with m.Elif(one | biased[-1]):
                m.d.comb += candidate.eq(bundle(record, exp=Const(0, wexp), keep=Const(0, 1), sign=Const(0, 1)))
            with m.Elif(is_least):
                m.d.comb += candidate.eq(bundle(record, exp=Const(1, wexp), keep=Const(0, 1), sign=neg))
            with m.Else():
                m.d.comb += candidate.eq(bundle(record, exp=biased[:wexp].as_unsigned(), keep=Const(1, 1), sign=neg))
            records.append(candidate)
            least.append(is_least)
        side = data.StructLayout(
            {"sb": self._sideband, "sub": 1, "eal": wexp, "exact": 2, "lo": record, "hi": record, "cy": record}
        )
        early = bundle(
            side,
            sb=self.sb_in,
            sub=self.sub,
            eal=self.eal,
            exact=Cat(*least[1:]),
            **dict(zip(("lo", "hi", "cy"), records)),
        )
        payload = Signal(side, name="early")
        m.d.comb += payload.eq(early)
        q = delay(m, "side", side, payload, self._stage)

        # |e| with a half ULP at each rounding position.
        placed = Cat(Const(0, ws - wexp), q.eal)
        aligned = [Signal(ws, name="al_lo"), Signal(ws, name="al_hi")]
        m.d.comb += [a.eq((placed + Cat(Const(0, wgb + k), ~q.exact[k]))[:ws]) for k, a in enumerate(aligned)]
        sub = Signal(name="sub_q")
        m.d.comb += sub.eq(q.sub)
        low, high = Signal(ws, name="s_l"), Signal(ws, name="s_h")
        m.d.comb += [s.eq(Mux(sub, (a - self.prod)[:ws], (a + self.prod)[:ws])) for s, a in zip((low, high), aligned)]
        chosen = data.StructLayout(
            {"sb": self._sideband, "lo": record, "hi": record, "cy": record, "f_lo": wfrac, "f_hi": wfrac}
            | {"is_hi": 1, "carry": 1}
        )
        rounded = bundle(
            chosen,
            sb=q.sb,
            lo=q.lo,
            hi=q.hi,
            cy=q.cy,
            f_lo=low[wgb + 1 : wgb + 1 + wfrac],
            f_hi=high[wgb + 2 : wgb + 2 + wfrac],
            is_hi=low[-1] | low[-2],
            carry=high[-1],
        )
        payload = Signal(chosen, name="rounded")
        m.d.comb += payload.eq(rounded)
        valid, r = pipe(m, "select", self.in_valid, chosen, payload, self._stage_select)
        is_hi = Signal(name="is_hi")
        m.d.comb += is_hi.eq(r.is_hi)
        final = Signal(record, name="final")
        with m.If(r.carry):
            m.d.comb += final.eq(r.cy)
        with m.Elif(is_hi):
            m.d.comb += final.eq(r.hi)
        with m.Else():
            m.d.comb += final.eq(r.lo)
        # After a carry these fraction bits are zero by themselves.
        keep_hi = Signal(name="keep_hi")
        m.d.comb += keep_hi.eq(is_hi & r.hi.keep)
        keep_lo = Signal(name="keep_lo")
        m.d.comb += keep_lo.eq(~is_hi & r.lo.keep)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.y.sign.eq(final.sign)
        m.d.comb += self.y.exp.eq(final.exp)
        m.d.comb += self.y.frac.eq((r.f_hi & keep_hi.replicate(wfrac)) | (r.f_lo & keep_lo.replicate(wfrac)))
        m.d.comb += self.sb_out.eq(r.sb)
        return m

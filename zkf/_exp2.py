import enum
from dataclasses import dataclass, field
from typing import Literal, overload

import mpmath

from amaranth.hdl import Cat, Const, Module, Mux, Signal
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import Attrs, FloatLayout, FloatView, Operator, Timing, bundle, delay, pipe
from ._trans import Kernel, kernel_check, kernel_depth, pack_fixed, spec as table
from ._value import Zkf, ZkfFormat, context


@dataclass(frozen=True, kw_only=True, eq=False)
class Exp2(Operator):
    """
    `y = 2**x`, faithfully rounded; `2**0 = 1`, `2**-inf = +0`, `2**+inf = +inf`. A result below `fmt.lowest` gives
    `fmt.lowest` or +0, whichever is nearer, and one beyond the format +inf. A computed tie rounds up, not to even; both
    of its neighbors are faithful.

    `x` is split into an integer `i` and `f` in [0, 1), to within an LSB. A piecewise-polynomial kernel gives `2**f` in
    [1, 2): the top bits of `f` select a segment's coefficients from a table, and a polynomial of degree 2 to 5, by
    WMAN, is evaluated in the rest. The kernel adds half an ULP, so that its truncation is the rounded significand; `i`
    is the exponent, or `i + 1` where the significand rounds up to 2.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core_kernel`: the registers of every product of the kernel, as `stage_core_mul` of `Mul`, each capped at its
    own positions, those of one multiplier at its output and operands.
    `stage_core`: the registers of `Exp2.Core` that are built.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    `wmultiplier`: the operand width of the target's multipliers, which the table is generated for.
    """

    class Core(enum.Flag):
        """The registers of `stage_core`, in path order."""

        REDUCE = enum.auto()
        """The reduced argument, ahead of the table."""
        SUM = enum.auto()
        """The kernel's sum, ahead of the assembly."""

    operator = "exp2"
    stage_input: int = 0
    stage_core_kernel: int = 0
    stage_core: Core = Core(0)
    stage_output: int = 0
    wmultiplier: int
    rom_attrs: Attrs | None = field(default=None, compare=False)
    """
    Synthesis attributes of the table's memory, such as `{"rom_style": "block"}`; those of several tools can be mixed.
    """
    mul_attrs: Attrs | None = field(default=None, compare=False)
    """On the captures of the kernel's products where a register follows their multipliers."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wmultiplier < 8:
            raise ValueError(f"wmultiplier={self.wmultiplier} is below 8")
        kernel_check(self.stage_core_kernel, table("exp2", self.fmt.wman, self.wmultiplier), self.wmultiplier)

    @property
    def timing(self) -> Timing:
        kernel = kernel_depth(table("exp2", self.fmt.wman, self.wmultiplier), self.wmultiplier, self.stage_core_kernel)
        return Timing(self.stage_input + 1 + kernel + len(self.stage_core) + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature({"in_valid": In(1), "x": In(f), "out_valid": Out(1), "y": Out(f)})

    def component(self) -> _Exp2:
        return _Exp2(self)

    @overload
    def evaluate(self, x: Zkf, *, oracle: Literal[False] = False) -> Zkf: ...
    @overload
    def evaluate(self, x: Zkf, *, oracle: Literal[True]) -> mpmath.mpf: ...
    def evaluate(self, x: Zkf, *, oracle: bool = False) -> Zkf | mpmath.mpf:
        """
        `oracle=True` gives `2**x` itself in place of the hardware's result: an mpf of 4 * WMAN + 80 bits, which
        `fmt.encode` rounds correctly.
        """
        if x.fmt != self.fmt:
            raise ValueError(f"an operand of {x.fmt} where {self.fmt} is due")
        fmt = self.fmt
        if oracle:
            prec = 4 * fmt.wman + 80
            return mpmath.mpf(context(prec).power(2, x.to_mpf()), prec=prec)
        if x.is_inf:
            return fmt.inf() if not x.negative else Zkf(fmt, 0)
        if x.is_zero:
            return fmt.pack(0, fmt.bias, 0)
        e = x.exp - fmt.bias
        # `|x| >= 2**(WEXP - 1)` is always out of range.
        if e >= fmt.wexp - 1:
            return fmt.inf() if not x.negative else Zkf(fmt, 0)

        spec = table("exp2", fmt.wman, self.wmultiplier)
        cf = ff = spec.cf  # fraction bits of the kernel and of the reduced argument
        sig = x.significand()
        shift = e - fmt.wfrac + ff
        mag = (sig << shift) if shift >= 0 else (sig >> -shift)  # floor(|x| * 2**FF)
        # x = i + f, f in [0, 1), as `_Reduce` forms them.
        i, f = mag >> ff, mag & ((1 << ff) - 1)
        if x.negative:
            i, f = -i - 1, f ^ ((1 << ff) - 1)
        s = spec.at(f)  # 2**f, half an ULP included
        assert 1 <= s >> cf < 4, "exp2 kernel range"
        assert not (s >> (cf + 1) and (s >> (cf - fmt.wfrac)) & fmt.frac_mask), "exp2 round-up with a nonzero fraction"
        # Without the half ULP; 2**f is 1 at least, and 2 where it rounds up to it, as 2**x of a negative integer.
        v = 1 << (cf + 1) if s >> (cf + 1) else max(s - (1 << (cf - fmt.wman)), 1 << cf)
        return Zkf(fmt, pack_fixed(fmt, 0, i, v, cf))


class _Exp2(wiring.Component):
    in_valid: Signal
    x: FloatView
    out_valid: Signal
    y: FloatView

    def __init__(self, model: Exp2) -> None:
        self._model, self._spec = model, table("exp2", model.fmt.wman, model.wmultiplier)
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        spec, core = self._spec, model.stage_core
        f = FloatLayout(fmt)
        valid, x = pipe(m, "in", self.in_valid, f, self.x, model.stage_input)
        m.submodules.reduce = reduce = _Reduce(fmt, spec.cf)
        m.d.comb += reduce.x.eq(x)
        side = data.StructLayout({"ipart": fmt.wexp - 1, "oor": 1, "zero": 1, "sign": 1})
        reduced = bundle(side, ipart=reduce.ipart, oor=reduce.oor, zero=reduce.zero, sign=reduce.sign)
        m.submodules.kernel = kernel = Kernel(
            spec,
            wmultiplier=model.wmultiplier,
            stages=model.stage_core_kernel,
            rom_attrs=model.rom_attrs,
            mul_attrs=model.mul_attrs,
        )
        argument = data.StructLayout({"f": spec.cf, "side": side})
        given = bundle(argument, f=reduce.f, side=reduced)
        f_valid, r = pipe(m, "reduce", valid, argument, given, int(Exp2.Core.REDUCE in core))
        m.d.comb += kernel.in_valid.eq(f_valid)
        m.d.comb += kernel.f.eq(r.f)
        total = data.StructLayout({"s": spec.cf + 2, "side": side})
        given = bundle(total, s=kernel.s, side=delay(m, "side", side, r.side, kernel.latency))
        s_valid, q = pipe(m, "sum", kernel.out_valid, total, given, int(Exp2.Core.SUM in core))
        m.submodules.pack = pack = _Pack(fmt, spec.cf)
        m.d.comb += pack.sign.eq(q.side.sign)
        m.d.comb += pack.zero.eq(q.side.zero)
        m.d.comb += pack.oor.eq(q.side.oor)
        m.d.comb += pack.ipart.eq(q.side.ipart)
        m.d.comb += pack.s.eq(q.s)
        out_valid, y = pipe(m, "out", s_valid, f, pack.y, model.stage_output)
        m.d.comb += self.out_valid.eq(out_valid)
        m.d.comb += self.y.eq(y)
        return m


class _Reduce(wiring.Component):
    """
    `x = i + f`, `f` in [0, 1) at `wff` fraction bits, from `floor(|x| * 2**wff)`, the significand
    shifted right, of which `ipart = floor(|x|)` is the integer part. A negative `x` takes the ones' complement of the
    fraction, `floor(x * 2**wff)` or one below it, and `i = -ipart - 1`: a negative integer becomes the integer below it
    plus the last argument of the kernel. `oor = |x| >= 2**(wexp - 1)`; `ipart` and `f` mean nothing with `oor` or
    `zero`.

    The shift amount is offset so that its low bits are the complement of the exponent's: the fine shift has no carry
    chain ahead of it, and only the coarse one waits for a short subtraction.
    """

    x: FloatView
    sign: Signal
    zero: Signal
    oor: Signal
    ipart: Signal
    f: Signal

    def __init__(self, fmt: ZkfFormat, wff: int) -> None:
        self._fmt, self._wff = fmt, wff
        flags = {"sign": Out(1), "zero": Out(1), "oor": Out(1)}
        super().__init__({"x": In(FloatLayout(fmt)), **flags, "ipart": Out(fmt.wexp - 1), "f": Out(wff)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, wff, x = self._fmt, self._wff, self.x
        wexp, wmag = fmt.wexp, fmt.wexp - 1 + wff  # |x| < 2**(wexp - 1) in range
        top = fmt.bias + wexp - 2  # the largest exponent in range: the significand is at the top of the magnitude
        wfine = min(wexp, 3)
        pad = -(top + 1) % (1 << wfine)
        coarse_top = (top + pad) >> wfine
        coarse_count = ((wmag + pad - 1) >> wfine) + 1  # coarse amounts that leave bits
        wcount = ceil_log2(coarse_count + 1)
        high = x.exp[wfine:] if wexp > wfine else Const(0, 1)
        wcoarse = max(len(high) + 1, wcount)
        coarse = Signal(wcoarse, name="coarse")
        m.d.comb += coarse.eq((Const(coarse_top, wcoarse) - high)[:wcoarse])  # not negative in range
        fine = Cat(Const(0, wmag - fmt.wman + pad), x.frac, Const(1, 1)) >> ~x.exp[:wfine]
        shifted = (fine >> Cat(Const(0, wfine), coarse[:wcount]))[:wmag]
        magnitude = Signal(wmag, name="magnitude")
        m.d.comb += magnitude.eq(Mux(coarse >= coarse_count, Const(0, wmag), shifted))
        m.d.comb += self.sign.eq(x.sign)
        m.d.comb += self.zero.eq(x.is_zero)
        m.d.comb += self.oor.eq(x.exp >= fmt.bias + wexp - 1)
        m.d.comb += self.ipart.eq(magnitude[wff:])
        m.d.comb += self.f.eq(magnitude[:wff] ^ x.sign.replicate(wff))
        return m


class _Pack(wiring.Component):
    """
    `y = 2**i * v / 2**wcf`, `i = ipart` for a positive `x` and `-ipart - 1` for a negative one, from the kernel's sum
    `s = v + ulp / 2` in `[2**wcf, 2**(wcf + 2))`: what `Pack` gives of `v` rounding half up, `v` taken as `2**wcf`
    where it falls short of that and as `2**(wcf + 1)` where `s` reaches that. The sum is truncated, and its top bit,
    which is late, only selects between two exponent fields.

    Where `s` reaches `2**(wcf + 1)`, its fraction bits are zero by themselves; a negative integer `x`, the last
    argument of the kernel, gets there, which makes its `2**x` exact. In the binade of otherwise zero results that is
    `x = -bias` alone, `min_normal / 2`, which rounds up.

    `z = i + bias + 1` for either sign, so the exponent field is `z - 1`, or `z` after the round-up: -1 gives +0, 0
    gives `min_normal`, all ones +inf. `zero` (`x = 0`) overrides the rest; `oor` gives +inf, or +0 for a negative `x`.
    """

    sign: Signal
    zero: Signal
    oor: Signal
    ipart: Signal
    s: Signal
    y: FloatView

    def __init__(self, fmt: ZkfFormat, wcf: int) -> None:
        self._fmt, self._wcf = fmt, wcf
        flags = {"sign": In(1), "zero": In(1), "oor": In(1)}
        super().__init__({**flags, "ipart": In(fmt.wexp - 1), "s": In(wcf + 2), "y": Out(FloatLayout(fmt))})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, wcf, wexp = self._fmt, self._wcf, self._fmt.wexp
        z = Signal(wexp, name="z")
        m.d.comb += z.eq(Cat(self.ipart ^ self.sign.replicate(wexp - 1), ~self.sign))
        low, high, fraction = Signal(wexp, name="e_lo"), Signal(wexp, name="e_hi"), Signal(1, name="use_frac")
        m.d.comb += fraction.eq(0)
        with m.If(self.zero):
            m.d.comb += low.eq(Const(fmt.bias, wexp))
            m.d.comb += high.eq(Const(fmt.bias, wexp))
        with m.Elif(self.oor):
            m.d.comb += low.eq((~self.sign).replicate(wexp))
            m.d.comb += high.eq((~self.sign).replicate(wexp))
        with m.Elif(~z[1:].any()):
            m.d.comb += low.eq(z)
            m.d.comb += high.eq(1)
        with m.Else():
            m.d.comb += low.eq((z - 1)[:wexp])
            m.d.comb += high.eq(z)
            m.d.comb += fraction.eq(1)
        m.d.comb += self.y.sign.eq(0)
        m.d.comb += self.y.exp.eq(Mux(self.s[wcf + 1], high, low))
        m.d.comb += self.y.frac.eq(self.s[wcf - fmt.wfrac : wcf] & fraction.replicate(fmt.wfrac))
        return m

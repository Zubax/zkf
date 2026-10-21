import enum
from dataclasses import dataclass

from amaranth.hdl import Cat, Const, Module, Mux, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import FloatLayout, FloatView, Operator, Timing, bundle, delay, pipe
from ._pack import Pack
from ._shift import Normshift, RshiftSticky, split_points
from ._value import Zkf, ZkfFormat


@dataclass(frozen=True, kw_only=True, eq=False)
class Add(Operator):
    """
    `y = a + b`; `a - b` with the sign bit of `b` flipped.

    `stage_input`: input register stages; where >0, no combinational path is exposed at the input.
    `stage_core`: the registers of `Add.CoreStages` that are built.
    `stage_output`: output register stages; where >0, no combinational path is exposed at the output.
    """

    class CoreStages(enum.Flag):
        """The registers of `stage_core`, in path order."""

        DECODE = enum.auto()
        """The decoded operands."""
        OPERANDS = enum.auto()
        """The ordered operands and the shift amount."""
        ALIGN = enum.auto()
        """The middle of the alignment shift."""
        ALIGNED = enum.auto()
        """The aligned operand, inverted for a difference."""
        SUM = enum.auto()
        """The raw sum or difference."""
        NORMALIZE_INPUT = enum.auto()
        """The raw sum or difference with the zero-detects of the normalizer's widest level, ahead of its shift."""
        NORMALIZE_WIDE = enum.auto()
        """Behind the normalizer's widest level, where it has three levels or more; it needs `NORMALIZE`."""
        NORMALIZE = enum.auto()
        """The middle of the normalizer."""
        NORMALIZED = enum.auto()
        """The sum or the normalized difference."""
        PACK = enum.auto()
        """The packer's input."""

    operator = "add"
    stage_input: int = 0
    stage_core: CoreStages = CoreStages(0)
    stage_output: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        check_core("stage_core", self.stage_core, self.fmt.wman)

    @property
    def timing(self) -> Timing:
        return Timing(self.stage_input + len(self.stage_core) + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature({"in_valid": In(1), "a": In(f), "b": In(f), "out_valid": Out(1), "y": Out(f)})

    def component(self) -> _Add:
        return _Add(self)

    def evaluate(self, a: Zkf, b: Zkf) -> Zkf:
        fmt = self.fmt
        if a.fmt != fmt or b.fmt != fmt:
            raise ValueError(f"operands of {a.fmt} and {b.fmt} where {fmt} is due")
        if a.is_inf and b.is_inf:
            return fmt.inf(a.negative) if a.negative == b.negative else Zkf(fmt, 0)
        if a.is_inf:
            return fmt.inf(a.negative)
        if b.is_inf:
            return fmt.inf(b.negative)
        return fmt.encode(a.to_fraction() + b.to_fraction())


def check_core(knob: str, stages: Add.CoreStages, wsig: int) -> None:
    """Refuses, as `knob`, what an adder of `wsig`-bit magnitudes cannot build, saying what to set."""
    if Add.CoreStages.NORMALIZE_WIDE in stages:
        if split_points(wsig + 3) < 2:
            raise ValueError(f"{knob}={stages}: a {wsig + 3}-bit normalizer has two levels; leave NORMALIZE_WIDE out")
        if Add.CoreStages.NORMALIZE not in stages:
            raise ValueError(f"{knob}={stages}: NORMALIZE_WIDE needs NORMALIZE; set both")


class AddCore(wiring.Component):
    """
    `y = larger + smaller * 2**-shamt` if `same_sign`, else `larger - smaller * 2**-shamt`, rounded to nearest even and
    packed. `larger` is the larger magnitude, with its leading one at the top bit and the biased exponent `exp`. A zero
    or infinite operand arrives as a zero magnitude, and `force_zero` and `force_inf` decide such a result.
    `finite_sign` and `inf_sign` are the signs of a finite and of an infinite result. `wsig`, `we` and `wshift` are the
    widths of a magnitude, `exp` and `shamt`.

    `stages` is the set of registers, of which `DECODE` is the caller's; `stage_output` is the number of output
    registers.
    """

    in_valid: Signal
    finite_sign: Signal
    inf_sign: Signal
    same_sign: Signal
    force_zero: Signal
    force_inf: Signal
    exp: Signal
    shamt: Signal
    larger: Signal
    smaller: Signal
    out_valid: Signal
    y: FloatView

    def __init__(
        self,
        fmt: ZkfFormat,
        *,
        wsig: int | None = None,
        we: int | None = None,
        wshift: int | None = None,
        stages: Add.CoreStages = Add.CoreStages(0),
        stage_output: int = 0,
    ) -> None:
        self._fmt = fmt
        self._wsig = fmt.wman if wsig is None else wsig
        self._we = fmt.wexp + 1 if we is None else we
        self._wshift = fmt.wexp if wshift is None else wshift
        self._stages = stages
        self._stage_output = stage_output
        super().__init__(
            {
                "in_valid": In(1),
                "finite_sign": In(1),
                "inf_sign": In(1),
                "same_sign": In(1),
                "force_zero": In(1),
                "force_inf": In(1),
                "exp": In(signed(self._we)),
                "shamt": In(self._wshift),
                "larger": In(self._wsig),
                "smaller": In(self._wsig),
                "out_valid": Out(1),
                "y": Out(FloatLayout(fmt)),
            }
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, wman, wsig, we = self._fmt, self._fmt.wman, self._wsig, self._we
        w = wsig + 3  # guard, round and sticky below the operands
        wen = max(we, ceil_log2(w) + 2)  # `exp` less a normalize count
        on = {member: int(member in self._stages) for member in Add.CoreStages}

        # The alignment of `smaller`.
        side = data.StructLayout(
            {"finite_sign": 1, "inf_sign": 1, "same_sign": 1, "force_zero": 1, "force_inf": 1, "exp": signed(we)}
            | {"larger": wsig}
        )
        waiting = bundle(
            side,
            finite_sign=self.finite_sign,
            inf_sign=self.inf_sign,
            same_sign=self.same_sign,
            force_zero=self.force_zero,
            force_inf=self.force_inf,
            exp=self.exp,
            larger=self.larger,
        )
        ordered_valid, ordered = pipe(m, "operands", self.in_valid, side, waiting, on[Add.CoreStages.OPERANDS])
        aligning_valid, aligning = pipe(m, "align", ordered_valid, side, ordered, on[Add.CoreStages.ALIGN])
        m.submodules.align = align = RshiftSticky(w, wshift=self._wshift, stage_split=on[Add.CoreStages.ALIGN])
        m.d.comb += align.x.eq(
            Cat(Const(0, 3), delay(m, "operands_small", wsig, self.smaller, on[Add.CoreStages.OPERANDS]))
        )
        m.d.comb += align.shamt.eq(delay(m, "operands_shamt", self._wshift, self.shamt, on[Add.CoreStages.OPERANDS]))
        # The aligned operand is inverted for a difference as it is registered, so that the select, which fans out over
        # the whole operand, is clear of the carry chain.
        s1_small = delay(m, "aligned_small", w, Mux(aligning.same_sign, align.y, ~align.y), on[Add.CoreStages.ALIGNED])
        s1_valid, s1 = pipe(m, "aligned", aligning_valid, side, aligning, on[Add.CoreStages.ALIGNED])

        # One carry chain adds or subtracts. `larger` is zero below its `wsig` bits, so the sticky in bit 0 of `smaller`
        # survives.
        differ = Signal(name="differ")
        m.d.comb += differ.eq(~s1.same_sign)
        raw = Cat(Const(0, 3), s1.larger, Const(0, 1)) + Cat(s1_small, differ) + differ

        # The raw result. A sum leads at bit `w - 1` or carries into bit `w`; a difference never carries, and is
        # normalized, with the sum on the normalizer's sideband.
        summed = data.StructLayout(
            {"sign": 1, "same_sign": 1, "force_zero": 1, "force_inf": 1, "exp": signed(wen), "raw": w + 1}
        )
        s2_valid, s2 = pipe(
            m,
            "sum",
            s1_valid,
            summed,
            bundle(
                summed,
                sign=Mux(s1.force_inf, s1.inf_sign, s1.finite_sign),
                same_sign=s1.same_sign,
                force_zero=s1.force_zero,
                force_inf=s1.force_inf,
                exp=Cat(s1.exp, s1.exp[-1].replicate(wen - we)).as_signed(),
                raw=raw[: w + 1],
            ),
            on[Add.CoreStages.SUM],
        )
        carry = s2.raw[-1]
        rounded = data.StructLayout(
            {"sign": 1, "same_sign": 1, "force_zero": 1, "force_inf": 1, "exp": signed(wen), "sig": wman}
            | {"guard": 1, "round": 1, "sticky": 1}
        )
        m.submodules.normalize = normalize = Normshift(
            w,
            stage_input=on[Add.CoreStages.NORMALIZE_INPUT],
            stage_split=on[Add.CoreStages.NORMALIZE] + on[Add.CoreStages.NORMALIZE_WIDE],
            sideband=rounded,
        )
        m.d.comb += normalize.in_valid.eq(s2_valid)
        m.d.comb += normalize.x.eq(s2.raw[:w])
        low = w - wman
        m.d.comb += normalize.sb_in.eq(
            bundle(
                rounded,
                sign=s2.sign,
                same_sign=s2.same_sign,
                force_zero=s2.force_zero,
                force_inf=s2.force_inf,
                exp=(s2.exp + carry)[:wen].as_signed(),
                sig=Mux(carry, s2.raw[low + 1 :], s2.raw[low:w]),
                guard=Mux(carry, s2.raw[low], s2.raw[low - 1]),
                round=Mux(carry, s2.raw[low - 1], s2.raw[low - 2]),
                sticky=Mux(carry, s2.raw[: low - 1].any(), s2.raw[: low - 2].any()),
            )
        )

        # The sum or the normalized difference.
        normalized = data.StructLayout({"r": rounded, "zero": 1, "count": len(normalize.count), "aligned": w})
        s3_valid, s3 = pipe(
            m,
            "normalized",
            normalize.out_valid,
            normalized,
            bundle(normalized, r=normalize.sb_out, zero=normalize.zero, count=normalize.count, aligned=normalize.y),
            on[Add.CoreStages.NORMALIZED],
        )
        same = s3.r.same_sign
        finite_zero = Mux(same, ~Cat(s3.r.sig, s3.r.guard, s3.r.round, s3.r.sticky).any(), s3.zero)
        m.submodules.pack = pack = Pack(
            fmt,
            wexp_unbiased=wen,
            exp_is_biased=True,
            stage_input=on[Add.CoreStages.PACK],
            stage_output=self._stage_output,
        )
        m.d.comb += pack.in_valid.eq(s3_valid)
        m.d.comb += pack.sign.eq(s3.r.sign)
        m.d.comb += pack.force_zero.eq(s3.r.force_zero | (~s3.r.force_inf & finite_zero))
        m.d.comb += pack.force_inf.eq(s3.r.force_inf)
        m.d.comb += pack.exp_unbiased.eq(Mux(same, s3.r.exp, (s3.r.exp - s3.count)[:wen].as_signed()))
        m.d.comb += pack.significand.eq(Mux(same, s3.r.sig, s3.aligned[low:]))
        m.d.comb += pack.guard.eq(Mux(same, s3.r.guard, s3.aligned[low - 1]))
        m.d.comb += pack.round.eq(Mux(same, s3.r.round, s3.aligned[low - 2]))
        m.d.comb += pack.sticky.eq(Mux(same, s3.r.sticky, s3.aligned[: low - 2].any()))
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        return m


class _Add(wiring.Component):
    in_valid: Signal
    a: FloatView
    b: FloatView
    out_valid: Signal
    y: FloatView

    def __init__(self, model: Add) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt = self._model, self._model.fmt
        wexp, wman = fmt.wexp, fmt.wman

        operands = data.StructLayout({"a": FloatLayout(fmt), "b": FloatLayout(fmt)})
        valid, q = pipe(m, "in", self.in_valid, operands, bundle(operands, a=self.a, b=self.b), model.stage_input)
        a, b = q.a, q.b

        # Zero and infinite operands enter as zero magnitudes; the result of an infinite one is forced further on.
        a_inf = Signal(name="a_inf")
        m.d.comb += a_inf.eq(a.is_inf)
        b_inf = Signal(name="b_inf")
        m.d.comb += b_inf.eq(b.is_inf)
        a_finite = Signal(name="a_finite")
        m.d.comb += a_finite.eq(~a.is_zero & ~a_inf)
        b_finite = Signal(name="b_finite")
        m.d.comb += b_finite.eq(~b.is_zero & ~b_inf)
        # The magnitude order of finite operands, which also orders their exponents; that of others does not matter. It
        # is the borrow of a subtraction, which maps onto a carry chain where a comparison operator may not.
        a_ge_b = ~(Cat(a.frac, a.exp) - Cat(b.frac, b.exp))[-1]
        decoded = data.StructLayout(
            {"a_sign": 1, "b_sign": 1, "a_inf": 1, "b_inf": 1, "a_ge_b": 1}
            | {"a_exp": wexp, "b_exp": wexp, "a_sig": wman, "b_sig": wman}
        )
        operands_decoded = bundle(
            decoded,
            a_sign=a.sign,
            b_sign=b.sign,
            a_inf=a_inf,
            b_inf=b_inf,
            a_ge_b=a_ge_b,
            a_exp=Mux(a_finite, a.exp, Const(0, wexp)),
            b_exp=Mux(b_finite, b.exp, Const(0, wexp)),
            a_sig=Mux(a_finite, a.significand, Const(0, wman)),
            b_sig=Mux(b_finite, b.significand, Const(0, wman)),
        )
        payload = Signal(decoded, name="decoded")
        m.d.comb += payload.eq(operands_decoded)
        valid, d = pipe(m, "decode", valid, decoded, payload, int(Add.CoreStages.DECODE in model.stage_core))

        same_sign = Signal(name="same_sign")
        m.d.comb += same_sign.eq(~(d.a_sign ^ d.b_sign))
        large_exp = Signal(wexp, name="large_exp")
        m.d.comb += large_exp.eq(Mux(d.a_ge_b, d.a_exp, d.b_exp))
        m.submodules.core = core = AddCore(fmt, stages=model.stage_core, stage_output=model.stage_output)
        m.d.comb += core.in_valid.eq(valid)
        m.d.comb += core.finite_sign.eq(Mux(d.a_ge_b, d.a_sign, d.b_sign))
        m.d.comb += core.inf_sign.eq((d.a_inf & d.a_sign) | (d.b_inf & d.b_sign))
        m.d.comb += core.same_sign.eq(same_sign)
        m.d.comb += core.force_zero.eq(d.a_inf & d.b_inf & ~same_sign)
        m.d.comb += core.force_inf.eq(d.a_inf | d.b_inf)
        m.d.comb += core.exp.eq(Cat(large_exp, Const(0, 1)).as_signed())
        m.d.comb += core.shamt.eq((large_exp - Mux(d.a_ge_b, d.b_exp, d.a_exp))[:wexp])
        m.d.comb += core.larger.eq(Mux(d.a_ge_b, d.a_sig, d.b_sig))
        m.d.comb += core.smaller.eq(Mux(d.a_ge_b, d.b_sig, d.a_sig))
        m.d.comb += self.out_valid.eq(core.out_valid)
        m.d.comb += self.y.eq(core.y)
        return m

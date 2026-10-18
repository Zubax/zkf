from amaranth.hdl import Module, ShapeLike, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from ._block import FloatLayout, FloatView, bundle, delay
from ._pack import Pack, pack_reference
from ._shift import Normshift, normshift_reference
from ._value import ZkfFormat


class FixedToFloat(wiring.Component):
    """
    `y = (-1)**sign * mag * 2**(exp_offset - wmag + 1)` rounded to nearest even and packed: `exp_offset` is the unbiased
    exponent of a `mag` with its top bit set (biased under `exp_is_biased`). A zero `mag` gives +0, and `force_inf`
    overrides the result. `sideband` is delayed with `y`.

    `wmag`, the width of `mag`, is at least WMAN + 3, and `weu` is the width of `exp_offset`. `exp_is_biased`,
    `assume_no_overflow` and `saturate_round_carry` are those of `Pack`, and `stage_input` and `stage_normalize` are
    `stage_input` and `stage_split` of `Normshift`. `stage_pack` registers the packer's inputs and `stage_output` the
    result; there is no other register.
    """

    in_valid: Signal
    sign: Signal
    force_inf: Signal
    exp_offset: Signal
    mag: Signal
    out_valid: Signal
    y: FloatView

    def __init__(
        self,
        fmt: ZkfFormat,
        *,
        wmag: int,
        weu: int,
        exp_is_biased: bool = False,
        assume_no_overflow: bool = False,
        saturate_round_carry: bool = False,
        sideband: ShapeLike | None = None,
        stage_input: int = 0,
        stage_normalize: int = 0,
        stage_pack: int = 0,
        stage_output: int = 0,
    ) -> None:
        self._fmt = fmt
        self._wmag = wmag
        self._weu = weu
        self._options = {
            "exp_is_biased": exp_is_biased,
            "assume_no_overflow": assume_no_overflow,
            "saturate_round_carry": saturate_round_carry,
        }
        self._sideband = sideband
        self._stage_input = stage_input
        self._stage_normalize = stage_normalize
        self._stage_pack = stage_pack
        self._stage_output = stage_output
        members = {
            "in_valid": In(1),
            "sign": In(1),
            "force_inf": In(1),
            "exp_offset": In(signed(weu)),
            "mag": In(wmag),
            "out_valid": Out(1),
            "y": Out(FloatLayout(fmt)),
        }
        if sideband is not None:
            members |= {"sb_in": In(sideband), "sb_out": Out(sideband)}
        super().__init__(members)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        fmt, weu = self._fmt, self._weu
        low = self._wmag - fmt.wman

        fields = {"sign": self.sign, "force_inf": self.force_inf}
        fields |= {"exp_offset": self.exp_offset} | ({} if self._sideband is None else {"sb": self.sb_in})
        carried = data.StructLayout(
            {"sign": 1, "force_inf": 1, "exp_offset": signed(weu)}
            | ({} if self._sideband is None else {"sb": self._sideband})
        )
        m.submodules.normalize = normalize = Normshift(
            self._wmag, stage_input=self._stage_input, stage_split=self._stage_normalize, sideband=carried
        )
        m.d.comb += normalize.in_valid.eq(self.in_valid)
        m.d.comb += normalize.x.eq(self.mag)
        m.d.comb += normalize.sb_in.eq(bundle(carried, **fields))
        q, aligned = normalize.sb_out, normalize.y

        m.submodules.pack = pack = Pack(
            fmt, wexp_unbiased=weu, stage_input=self._stage_pack, stage_output=self._stage_output, **self._options
        )
        m.d.comb += pack.in_valid.eq(normalize.out_valid)
        m.d.comb += pack.sign.eq(q.sign)
        m.d.comb += pack.force_zero.eq(~q.force_inf & normalize.zero)
        m.d.comb += pack.force_inf.eq(q.force_inf)
        m.d.comb += pack.exp_unbiased.eq((q.exp_offset - normalize.count)[:weu].as_signed())
        m.d.comb += pack.significand.eq(aligned[low:])
        m.d.comb += pack.guard.eq(aligned[low - 1])
        m.d.comb += pack.round.eq(aligned[low - 2])
        m.d.comb += pack.sticky.eq(aligned[: low - 2].any())
        m.d.comb += self.out_valid.eq(pack.out_valid)
        m.d.comb += self.y.eq(pack.y)
        if self._sideband is not None:
            m.d.comb += self.sb_out.eq(delay(m, "sb", self._sideband, q.sb, self._stage_pack + self._stage_output))
        return m


def fixed_to_float_reference(
    fmt: ZkfFormat,
    sign: int,
    mag: int,
    exp_offset: int,
    wmag: int,
    *,
    saturate_round_carry: bool = False,
) -> int:
    """The bits that `FixedToFloat` gives for an unbiased `exp_offset` and no `force_inf`."""
    zero_flag, count, aligned = normshift_reference(wmag, mag)
    significand_value = (aligned >> (wmag - fmt.wman)) & ((1 << fmt.wman) - 1)
    guard = (aligned >> (wmag - fmt.wman - 1)) & 1
    round_bit = (aligned >> (wmag - fmt.wman - 2)) & 1
    sticky = 1 if (aligned & ((1 << wmag - fmt.wman - 2) - 1)) else 0
    exp_unbiased = exp_offset - count
    return pack_reference(
        fmt,
        sign,
        zero_flag,
        0,
        exp_unbiased,
        significand_value,
        guard,
        round_bit,
        sticky,
        saturate_round_carry=saturate_round_carry,
    )

"""
What the synthesis suite synthesizes, and what the simulation tests take of it: configurations of the operators, each
chosen as the closure procedure of AGENTS.md arrives at it from every stage off, or as that with an input or output
stage pinned or a register more, as 0.9.0's suite had them. Every multiplying operator is given 18-bit multipliers,
which every chip of the figures has.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from amaranth.back import verilog
from amaranth.lib import wiring

from zkf import Add, Block, Cmp, Cordic, Divsqrt, Exp2, Fma, FromInt, Ilog2, Log2, Mul, MulIlog2, Resize, Rint
from zkf import Timing, ZkfFormat
from zkf._pack import Pack


@dataclass(frozen=True)
class CoreSpec:
    """A private core, which is no operator, as a spec: what the harness reads of an operator."""

    name: str
    component: Callable[[], wiring.Component]
    latency: int

    @property
    def timing(self) -> Timing:
        return Timing(self.latency, 1)

    @property
    def clocked(self) -> bool:
        return self.latency > 0

    @property
    def signature(self) -> wiring.Signature:
        return self.component().signature

    def compile(self) -> str:
        return verilog.convert(self.component(), name=self.name, emit_src=False, strip_internal_attrs=True)


F18, F24, F36 = ZkfFormat(6, 18), ZkfFormat(8, 24), ZkfFormat(8, 36)
_FMA36: dict[str, Any] = dict(stage_core_mul=3, wmultiplier=18)
_CORDIC_SHARED = Cordic.Core.NORMALIZE_INPUT | Cordic.Core.NORMALIZE | Cordic.Core.PACK
_CORDIC_ROTATION = _CORDIC_SHARED | Cordic.Core.REDUCED | Cordic.Core.CORRECTIONS
_CORDIC_VECTORING = _CORDIC_SHARED | Cordic.Core.TAKEN | Cordic.Core.ORDERED | Cordic.Core.ANGLE
CATALOG: dict[str, Block | CoreSpec] = {
    spec.name: spec
    for spec in (
        Ilog2(F18),
        Ilog2(F24, wint=9),
        Ilog2(F24),
        Ilog2(F24, stage_input=1),
        Ilog2(ZkfFormat(11, 53), wint=64),
        CoreSpec("_zkf_pack_e6m18_wexp_unbiased8", partial(Pack, F18, wexp_unbiased=8), 0),
        # At 0 Diamond makes 79.9 MHz: the product and its rounding in one cycle.
        Mul(F18, stage_core_mul=1, wmultiplier=18),
        Mul(F18, stage_core_mul=2, wmultiplier=18),
        Mul(F18, stage_core_mul=1, stage_output=1, wmultiplier=18),
        # At 2 Vivado sums the DSP tiles' products in fabric in their cycle: 122.1 to 148.3 MHz.
        Mul(F36, stage_core_mul=3, stage_output=1, wmultiplier=18),
        Mul(F36, stage_core_mul=3, wmultiplier=18),
        Mul(F36, stage_input=1, stage_core_mul=3, wmultiplier=18),
        Mul(ZkfFormat(8, 25), stage_core_mul=3, wmultiplier=18),
        # Without ALIGN Diamond makes 97.2 MHz, in the alignment; without OPERANDS Yosys makes 74.8.
        Add(F18, stage_core=Add.Core.OPERANDS | Add.Core.ALIGN | Add.Core.SUM | Add.Core.NORMALIZE),
        # With ALIGNED for ALIGN Yosys makes 99.7 MHz; each register less misses Diamond.
        Add(F36, stage_core=Add.Core.OPERANDS | Add.Core.ALIGN | Add.Core.SUM | Add.Core.NORMALIZE),
        # Without PACK Yosys makes 98.2 MHz, in the rounding, even with DECODE and ALIGN.
        Fma(
            F18,
            stage_core_mul=1,
            stage_core_add=Add.Core.OPERANDS | Add.Core.ALIGNED | Add.Core.SUM | Add.Core.NORMALIZE | Add.Core.PACK,
            wmultiplier=18,
        ),
        # With NORMALIZED for PACK Diamond makes 97.3 MHz, in the rounding.
        Fma(F36, stage_output=1, stage_core_add=~(Add.Core.NORMALIZE_INPUT | Add.Core.NORMALIZED), **_FMA36),
        # Each register less misses Diamond, at 74.9 to 95.8 MHz.
        Fma(F36, stage_input=1, stage_core_add=~(Add.Core.NORMALIZE_INPUT | Add.Core.PACK), **_FMA36),
        Divsqrt(F18),
        Divsqrt(F18, mode=0),
        Divsqrt(F18, mode=1),
        Divsqrt(ZkfFormat(8, 27)),
        # Folded into stage 0 beside the root, the divider's first digit makes 98.7 MHz on Yosys.
        Divsqrt(F36, stage_core=Divsqrt.Core.FIRST_DIGIT),
        Divsqrt(F36, mode=0),
        Divsqrt(F36, mode=1),
        Cmp(F18),
        Cmp(F36),
        MulIlog2(F18, wk=7),
        MulIlog2(F36, wk=9),
        # Unstaged, the decode of k into the selection makes 84.3 MHz on Diamond.
        MulIlog2(F36, wk=44, stage_core=MulIlog2.Core.DECODE),
        MulIlog2(F36, wk=9, stage_core=MulIlog2.Core.DECODE),
        # The normalizer's middle alone makes 96.3, 94.3 and 92.3 MHz on Diamond, |a| into it; |a| alone, 66.9 to 76.5.
        FromInt(F18, stage_core=FromInt.Core.ABS | FromInt.Core.NORMALIZE),
        FromInt(F18, stage_input=1, stage_core=FromInt.Core.ABS | FromInt.Core.NORMALIZE),
        FromInt(F36, stage_core=FromInt.Core.ABS | FromInt.Core.NORMALIZE),
        Resize(ZkfFormat(5, 11), fmt_in=ZkfFormat(6, 18)),
        Resize(ZkfFormat(5, 11), fmt_in=ZkfFormat(6, 18), stage_input=1),
        Resize(F18, fmt_in=ZkfFormat(5, 11)),
        Resize(F18, fmt_in=ZkfFormat(5, 11), stage_input=1),
        Resize(F18, fmt_in=ZkfFormat(8, 36)),
        Resize(F36, fmt_in=ZkfFormat(6, 18)),
        # Unstaged, the discard mask into the rounding decision makes 74.1 and 59.5 MHz on Diamond.
        Rint(F18, stage_core=Rint.Core.MASK),
        Rint(F36, wint=44, stage_core=Rint.Core.MASK),
        Exp2(F18, stage_core_kernel=2, wmultiplier=18),
        Log2(F18, stage_core_kernel=2, stage_core_mul=2, wmultiplier=18),
        Log2(F18, stage_core_kernel=2, stage_core_mul=2, stage_output=1, wmultiplier=18),
        Exp2(F24, stage_core_kernel=2, wmultiplier=18),
        Log2(F24, stage_core_kernel=2, stage_core_mul=3, wmultiplier=18),
        Exp2(F36, stage_core_kernel=3, wmultiplier=18),
        # Without ALIGN Diamond makes 85.2 MHz.
        Log2(F36, stage_core_kernel=3, stage_core_mul=3, stage_core=Log2.Core.ALIGN, wmultiplier=18),
        # Without PACK or CORRECTIONS Yosys makes 92.4 and 98.7 MHz.
        Cordic(F18, mode=0, stage_core_mul=3, stage_core=_CORDIC_ROTATION, wmultiplier=18),
        # Without ANGLE or TAKEN Yosys makes 98.1 and 95.8 MHz.
        Cordic(F18, mode=1, stage_core_mul=3, stage_core=_CORDIC_VECTORING, wmultiplier=18),
        # Without SHIFT Yosys makes 94.6 MHz; one iteration a cycle, 95.5 on Diamond.
        Cordic(
            F36,
            mode=0,
            unroll100=50,
            stage_core_mul=4,
            stage_core=_CORDIC_ROTATION | Cordic.Core.SHIFT | Cordic.Core.FOLDED,
            wmultiplier=18,
        ),
        # Without SHIFTED Diamond makes 98.9 MHz in the rounding, without ROUNDED 92.0; one iteration a cycle, 93.5.
        Cordic(
            F36,
            mode=1,
            unroll100=50,
            stage_core_mul=4,
            stage_core=_CORDIC_VECTORING | Cordic.Core.SHIFTED | Cordic.Core.ROUNDED,
            wmultiplier=18,
        ),
        # Without SHIFT Yosys makes 97.8 MHz, without ANGLE Diamond 93.6.
        Cordic(
            F36,
            unroll100=50,
            stage_core_mul=4,
            stage_core=~(Cordic.Core.CORRECTIONS | Cordic.Core.ROUNDED),
            wmultiplier=18,
        ),
    )
}
"""The specs by their module names."""

KEPT = frozenset(name for name, spec in CATALOG.items() if isinstance(spec, Exp2 | Log2) and spec.fmt.wman < 36)
"""
The specs to which a flow that takes `mul_attrs` (Vivado's, `keep`) gives them, keeping a capture out of the DSP tile it
feeds. Without, Vivado misses 150 MHz on log2 6/18 (125.7, 127.7) and 8/24 (147.9) and exp2 8/24 (149.2), and closes
exp2 6/18 by 0.05 MHz; at 8/36 the gain is minor: exp2 155.2 without, 160.7 with, log2 154.4 and 155.2, the texts
without being those that the suite synthesizes.
"""

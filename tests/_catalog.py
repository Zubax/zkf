"""
What the synthesis suite synthesizes, and what the simulation tests take of it: configurations of the operators, each
chosen as the closure procedure of AGENTS.md arrives at it from every stage off, or as that with an input or output
stage pinned or a register more, as 0.9.0's suite had them. Every multiplying operator is given 18-bit multipliers,
which every chip of the figures has.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
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
_CORDIC_SHARED = Cordic.CoreStages.NORMALIZE | Cordic.CoreStages.PACK
_CORDIC_ROTATION = _CORDIC_SHARED | Cordic.CoreStages.REDUCED | Cordic.CoreStages.CORRECTIONS
_CORDIC_VECTORING = _CORDIC_SHARED | Cordic.CoreStages.TAKEN | Cordic.CoreStages.ORDERED | Cordic.CoreStages.ANGLE
_LARGE: list[str] = []
_GRADES: dict[str, dict[str, int]] = {}


def _large(spec: Block) -> Block:
    _LARGE.append(spec.name)
    return spec


def _grade(spec: Block, **grades: int) -> Block:
    _GRADES[spec.name] = grades
    return spec


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
        Add(
            F18,
            stage_core=Add.CoreStages.OPERANDS | Add.CoreStages.ALIGN | Add.CoreStages.SUM | Add.CoreStages.NORMALIZE,
        ),
        # Without NORMALIZED nextpnr makes 93.5 to 97.6 MHz.
        Add(
            F36,
            stage_core=Add.CoreStages.OPERANDS
            | Add.CoreStages.ALIGN
            | Add.CoreStages.SUM
            | Add.CoreStages.NORMALIZE
            | Add.CoreStages.NORMALIZED,
        ),
        # Without PACK nextpnr makes 92.1 to 96.0 MHz and Diamond 95.0, from the normalizer's count through the rounding
        # into the result; without SUM nextpnr's seeds make 97.0 to 99.0, from the sum into the count.
        Fma(
            F18,
            stage_core_mul=1,
            stage_core_add=Add.CoreStages.DECODE
            | Add.CoreStages.OPERANDS
            | Add.CoreStages.ALIGNED
            | Add.CoreStages.SUM
            | Add.CoreStages.NORMALIZE
            | Add.CoreStages.PACK,
            wmultiplier=18,
        ),
        # Without PACK nextpnr's seeds make 96.5 to 100.8 MHz.
        Fma(F36, stage_output=1, stage_core_add=~Add.CoreStages.NORMALIZE_INPUT, **_FMA36),
        # Without PACK nextpnr makes 97.1 to 103.7 MHz and Diamond 97.8.
        Fma(F36, stage_input=1, stage_core_add=~Add.CoreStages.NORMALIZE_INPUT, **_FMA36),
        Divsqrt(F18),
        Divsqrt(F18, mode=0),
        Divsqrt(F18, mode=1),
        # On speed grade 6 one of Diamond's placement cost tables makes 97.1 MHz, nextpnr 117.7, with or without an
        # output stage.
        _grade(Divsqrt(ZkfFormat(8, 27)), ecp5_grade=7),
        # Folded into stage 0 beside the root, the divider's first digit makes 96.4 to 100.8 MHz on nextpnr's seeds.
        _large(Divsqrt(F36, stage_core=Divsqrt.CoreStages.FIRST_DIGIT)),
        _large(Divsqrt(F36, mode=0)),
        Divsqrt(F36, mode=1),
        Cmp(F18),
        Cmp(F36),
        MulIlog2(F18, wk=7),
        # Unstaged, the decode of k into the selection makes 84.3 MHz on Diamond.
        MulIlog2(F36, wk=44, stage_core=MulIlog2.CoreStages.DECODE),
        # Unstaged, LSE puts the forcing of the result on the set/reset of its register, a slow net: one of Diamond's
        # placement cost tables makes 98.0 MHz, 99.3 on speed grade 7.
        MulIlog2(F36, wk=9, stage_core=MulIlog2.CoreStages.DECODE),
        # Without ABS nextpnr makes 76.4 to 84.5 MHz at e6m18 and 95.0 to 99.8 at e8m36; without NORMALIZE 79.1 to
        # 85.2 and 71.6 to 76.3.
        FromInt(F18, stage_core=FromInt.CoreStages.ABS | FromInt.CoreStages.NORMALIZE),
        FromInt(F18, stage_input=1, stage_core=FromInt.CoreStages.ABS | FromInt.CoreStages.NORMALIZE),
        FromInt(F36, stage_core=FromInt.CoreStages.ABS | FromInt.CoreStages.NORMALIZE),
        Resize(ZkfFormat(5, 11), fmt_in=ZkfFormat(6, 18)),
        Resize(ZkfFormat(5, 11), fmt_in=ZkfFormat(6, 18), stage_input=1),
        Resize(F18, fmt_in=ZkfFormat(5, 11)),
        Resize(F18, fmt_in=ZkfFormat(5, 11), stage_input=1),
        Resize(F18, fmt_in=ZkfFormat(8, 36)),
        Resize(F36, fmt_in=ZkfFormat(6, 18)),
        # Unstaged, the discard mask into the rounding decision makes 74.1 and 59.5 MHz on Diamond.
        Rint(F18, stage_core=Rint.CoreStages.MASK),
        Rint(F36, wint=44, stage_core=Rint.CoreStages.MASK),
        # Without SUM Diamond makes 98.2 MHz, at stage_core_kernel=2 Vivado 150.04, under a 0.5% margin.
        Exp2(F18, stage_core_kernel=3, stage_core=Exp2.CoreStages.SUM, wmultiplier=18),
        # Without ALIGN Diamond makes 87.3 to 94.4 MHz.
        Log2(F18, stage_core_kernel=2, stage_core_mul=2, stage_core=Log2.CoreStages.ALIGN, wmultiplier=18),
        Log2(
            F18, stage_core_kernel=2, stage_core_mul=2, stage_core=Log2.CoreStages.ALIGN, stage_output=1, wmultiplier=18
        ),
        # Without SUM Diamond makes 78.2 MHz, at stage_core_kernel=2 Vivado 149.1.
        Exp2(F24, stage_core_kernel=3, stage_core=Exp2.CoreStages.SUM, wmultiplier=18),
        # Without stage_output=1 one of nextpnr's seeds makes 99.8 MHz, in the rounding into the result; without ALIGN
        # Diamond makes 97.0 MHz, at stage_core_kernel=2 Vivado 149.1.
        Log2(
            F24, stage_core_kernel=3, stage_core_mul=3, stage_core=Log2.CoreStages.ALIGN, stage_output=1, wmultiplier=18
        ),
        # Without SUM Diamond makes 81.0 MHz.
        Exp2(F36, stage_core_kernel=3, stage_core=Exp2.CoreStages.SUM, wmultiplier=18),
        # Without SELECT one of nextpnr's seeds makes 99.5 MHz, from the packer's side inputs into the result, and
        # stage_output=1 in its place 98.8, Diamond 97.7; without ALIGN Diamond makes 76.5 to 84.8.
        Log2(
            F36,
            stage_core_kernel=3,
            stage_core_mul=3,
            stage_core=Log2.CoreStages.ALIGN | Log2.CoreStages.SELECT,
            wmultiplier=18,
        ),
        # Without PACK nextpnr's seeds make 86.5 to 90.1 MHz, from the normalizer's count into the result; without
        # FOLDED nextpnr's seeds make 99.4 to 103.2 and Vivado misses by 0.031 ns, from the reduced coordinate into the
        # engine.
        Cordic(
            F18,
            mode=0,
            stage_core_mul=3,
            stage_core=_CORDIC_ROTATION | Cordic.CoreStages.FOLDED,
            rotation_seed=0,
            wmultiplier=18,
        ),
        # At stage_core_mul=2 nextpnr's seeds make 88.5 to 92.4 MHz, in the length's product.
        Cordic(F18, mode=1, stage_core_mul=3, stage_core=_CORDIC_VECTORING, wmultiplier=18),
        # On speed grade 7 it closes thinly: Diamond makes 102.0 to 108.7 MHz, from the engine's adds of an iteration,
        # and nextpnr's seeds 102.8 to 110.1, from the sine's sum into the normalizer; without the attribute Diamond
        # puts the tables in block RAM and makes 94.0 to 104.0, from their reads.
        _large(
            _grade(
                Cordic(
                    F36,
                    mode=0,
                    stage_core_mul=4,
                    stage_core=(_CORDIC_ROTATION | Cordic.CoreStages.FOLDED) & ~Cordic.CoreStages.CORRECTIONS,
                    wmultiplier=18,
                    rom_attrs={"syn_romstyle": "logic"},
                ),
                ecp5_grade=8,
            )
        ),
        # On speed grade 6 Diamond makes 87.0 to 95.3 MHz, from the division's loop and the engine's angle into `z`, and
        # nextpnr's seeds 95.1 to 103.7, from the engine's `x` into the divisor's multiples at the load.
        _large(
            _grade(
                Cordic(F36, mode=1, stage_core_mul=4, stage_core=_CORDIC_VECTORING, wmultiplier=18),
                ecp5_grade=7,
            )
        ),
        # On speed grade 7 Diamond makes 93.6 to 98.1 MHz, from the engine's adds of an iteration, and nextpnr's seeds
        # 92.1 to 96.5, from the sine's sum into the normalizer; on Artix-7 -1 Vivado misses by 0.178 ns, in the
        # division's loop, no register helping; on grade 8 without the attribute Diamond puts the tables in block RAM
        # and makes 91.1 to 100.0, from their reads.
        _large(
            _grade(
                Cordic(
                    F36,
                    stage_core_mul=4,
                    stage_core=~(
                        Cordic.CoreStages.SHIFT
                        | Cordic.CoreStages.CORRECTIONS
                        | Cordic.CoreStages.SHIFTED
                        | Cordic.CoreStages.MEET
                        | Cordic.CoreStages.NORMALIZE_INPUT
                        | Cordic.CoreStages.ROUNDED
                    ),
                    wmultiplier=18,
                    rom_attrs={"syn_romstyle": "logic"},
                ),
                ecp5_grade=8,
                artix7_grade=2,
            )
        ),
    )
}
"""The specs by their module names."""

ECP5_LARGE = frozenset(_LARGE)
"""The specs that the ECP5 flows place on their larger part: each fills a third or more of the LFE5U-12F's slices."""

GRADES = MappingProxyType(_GRADES)
"""The specs that flows place on a faster speed grade, with those grades as `Design` takes them: each misses on a slower
one in one of them, no register helping."""

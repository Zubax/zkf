"""
What the synthesis suite synthesizes, and what the simulation tests take of it: configurations of the operators, each
chosen as the closure procedure of AGENTS.md arrives at it from every stage off, or as that with an input or output
stage pinned or a register more, as 0.9.0's suite had them. Every multiplying operator is given 18-bit multipliers,
which every chip of the figures has.
"""

from types import MappingProxyType
from typing import Any

from zkf import (
    Add,
    Block,
    Cmp,
    Cordic,
    Divsqrt,
    Exp2,
    Fma,
    FromInt,
    Ilog2,
    Log2,
    Mul,
    MulIlog2,
    Resize,
    Rint,
    ZkfFormat,
)

F18, F24, F36 = ZkfFormat(6, 18), ZkfFormat(8, 24), ZkfFormat(8, 36)
_FMA36: dict[str, Any] = dict(stage_core_mul=3, wmultiplier=18)
_CORDIC_SHARED = Cordic.CoreStages.NORMALIZE | Cordic.CoreStages.PACK
_CORDIC_ROTATION = _CORDIC_SHARED | Cordic.CoreStages.REDUCED | Cordic.CoreStages.FOLDED
_CORDIC_VECTORING = _CORDIC_SHARED | Cordic.CoreStages.TAKEN | Cordic.CoreStages.ORDERED | Cordic.CoreStages.ANGLE
_LARGE: list[str] = []
_GRADES: dict[str, dict[str, int]] = {}


def _large(spec: Block) -> Block:
    _LARGE.append(spec.name)
    return spec


def _grade(spec: Block, **grades: int) -> Block:
    _GRADES[spec.name] = grades
    return spec


CATALOG: dict[str, Block] = {
    spec.name: spec
    for spec in (
        Ilog2(F18),
        Ilog2(F24, wint=9),
        Ilog2(F24),
        Ilog2(F24, stage_input=1),
        Ilog2(ZkfFormat(11, 53), wint=64),
        # At 0 Diamond makes 79.9 MHz: the product and its rounding in one cycle.
        Mul(F18, stage_core_mul=1, wmultiplier=18),
        Mul(F18, stage_core_mul=2, wmultiplier=18),
        Mul(F18, stage_core_mul=1, stage_output=1, wmultiplier=18),
        # At 2 Vivado on Artix-7 sums the DSP tiles' products in fabric in their cycle: 122.1 to 148.3 MHz.
        Mul(F36, stage_core_mul=3, stage_output=1, wmultiplier=18),
        Mul(F36, stage_core_mul=3, wmultiplier=18),
        Mul(F36, stage_input=1, stage_core_mul=3, wmultiplier=18),
        Mul(ZkfFormat(8, 25), stage_core_mul=3, wmultiplier=18),
        # Without ALIGN Diamond's placements make 98.2 to 105.6 MHz, from the operands through the alignment into the
        # sum; without OPERANDS nextpnr's seeds make 73.1 to 75.9.
        Add(
            F18,
            stage_core=Add.CoreStages.OPERANDS | Add.CoreStages.ALIGN | Add.CoreStages.SUM | Add.CoreStages.NORMALIZE,
        ),
        # Without NORMALIZED nextpnr's seeds make 94.7 to 97.7 MHz.
        Add(
            F36,
            stage_core=Add.CoreStages.OPERANDS
            | Add.CoreStages.ALIGN
            | Add.CoreStages.SUM
            | Add.CoreStages.NORMALIZE
            | Add.CoreStages.NORMALIZED,
        ),
        # Without PACK nextpnr's seeds make 89.9 to 94.1 MHz and Diamond's placements 90.5 to 98.6, from the
        # normalizer's count through the rounding into the result; without SUM nextpnr's seeds make 98.8 to 110.8, from
        # the sum into the count.
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
        # Without PACK nextpnr's seeds make 92.5 to 107.8 MHz.
        Fma(F36, stage_output=1, stage_core_add=~Add.CoreStages.NONE, **_FMA36),
        # Without PACK nextpnr's seeds make 99.3 to 106.4 MHz and Diamond's placements 99.2 to 112.9.
        Fma(F36, stage_input=1, stage_core_add=~Add.CoreStages.NONE, **_FMA36),
        # Folded, nextpnr's placements make 99.4 to 106.3 MHz.
        Divsqrt(F18, stage_core=Divsqrt.CoreStages.FIRST_DIGIT),
        Divsqrt(F18, mode=0),
        Divsqrt(F18, mode=1),
        Divsqrt(F18, fast=False),
        # On speed grade 6 nextpnr's placements make 93.3 to 98.0 MHz, 98.8 to 105.9 with FIRST_DIGIT.
        _large(_grade(Divsqrt(F24), ecp5_grade=7)),
        # On speed grade 6 nextpnr's placements make 95.8 to 105.4 MHz, 92.7 to 97.3 without FIRST_DIGIT; without it
        # Quartus makes 97.6 on Cyclone IV E C8, from a digit into the exponent's delay line, which it puts in block
        # RAM.
        _large(_grade(Divsqrt(ZkfFormat(8, 27), stage_core=Divsqrt.CoreStages.FIRST_DIGIT), ecp5_grade=7)),
        Divsqrt(ZkfFormat(8, 27), fast=False),
        # On speed grade 6 nextpnr's placements make 91.8 to 97.6 MHz with FIRST_DIGIT; on Cyclone IV E C8 Quartus makes
        # 94.5, in the digits' loop, 96.0 with FIRST_DIGIT.
        _large(_grade(Divsqrt(F36), ecp5_grade=7, cyclone4_grade=7)),
        # Folded, nextpnr's placements make 99.9 to 107.3 MHz.
        _large(Divsqrt(F36, stage_core=Divsqrt.CoreStages.FIRST_DIGIT, fast=False)),
        # Folded, nextpnr's placements make 99.3 to 101.1 MHz; on Cyclone IV E C8 Quartus makes 99.4, in the digits'
        # loop.
        _large(_grade(Divsqrt(F36, mode=0, stage_core=Divsqrt.CoreStages.FIRST_DIGIT), cyclone4_grade=7)),
        _large(Divsqrt(F36, mode=0, fast=False)),
        # On speed grade 6 nextpnr's placements make 92.4 to 96.8 MHz, 89.6 to 100.6 with FIRST_DIGIT.
        _large(_grade(Divsqrt(F36, mode=1), ecp5_grade=7)),
        Divsqrt(F36, mode=1, fast=False),
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
        Resize(F18, fmt_in=ZkfFormat(6, 24)),  # the packer alone
        Resize(ZkfFormat(5, 11), fmt_in=ZkfFormat(6, 18)),
        Resize(ZkfFormat(5, 11), fmt_in=ZkfFormat(6, 18), stage_input=1),
        Resize(F18, fmt_in=ZkfFormat(5, 11)),
        Resize(F18, fmt_in=ZkfFormat(5, 11), stage_input=1),
        Resize(F18, fmt_in=ZkfFormat(8, 36)),
        Resize(F36, fmt_in=ZkfFormat(6, 18)),
        # Unstaged, the discard mask into the rounding decision makes 74.1 and 59.5 MHz on Diamond.
        Rint(F18, stage_core=Rint.CoreStages.MASK),
        # On Cyclone IV E C8 Quartus makes 95.7 MHz, from the mask into the integer.
        _grade(Rint(F36, wint=44, stage_core=Rint.CoreStages.MASK), cyclone4_grade=7),
        Exp2(F18, stage_core_kernel_level=6, wmultiplier=18),
        Log2(F18, stage_core_kernel_level=6, stage_core_mul=2, wmultiplier=18),
        Log2(F18, stage_core_kernel_level=6, stage_core_mul=2, stage_output=1, wmultiplier=18),
        Exp2(F24, stage_core_kernel_level=6, wmultiplier=18),
        Log2(F24, stage_core_kernel_level=6, stage_core_mul=3, wmultiplier=18),
        Exp2(F36, stage_core_kernel_level=6, wmultiplier=18),
        # Without ALIGN Vivado on Artix-7 misses by 0.634 ns, in the shift into the last product.
        Log2(F36, stage_core_kernel_level=6, stage_core_mul=3, stage_core=Log2.CoreStages.ALIGN, wmultiplier=18),
        # Without PACK nextpnr's seeds make 81.6 to 87.5 MHz, from the normalizer's count into the result; without
        # FOLDED nextpnr's seeds make 99.7 to 103.5, from the reduced coordinate into the engine.
        Cordic(
            F18,
            mode=0,
            stage_core_mul=3,
            stage_core=_CORDIC_ROTATION,
            rotation_seed=0,
            wmultiplier=18,
        ),
        # At stage_core_mul=2 nextpnr's seeds make 89.7 to 93.0 MHz, in the length's product.
        Cordic(F18, mode=1, stage_core_mul=3, stage_core=_CORDIC_VECTORING, wmultiplier=18),
        # Without NORMALIZE_INPUT Vivado on Artix-7 -1 misses by 0.093 ns, from the sine's sum into the normalizer.
        # Grade 7 is closed too, but thinly: Diamond makes 100.8 to 110.4 MHz there, from the engine's adds of an
        # iteration, and nextpnr's seeds 104.3 to 114.5; without the attribute Diamond puts the tables in block RAM and
        # makes 97.4 to 105.5, from their reads; on Cyclone IV E C8 Quartus makes 95.5, from the engine's adds.
        _large(
            _grade(
                Cordic(
                    F36,
                    mode=0,
                    stage_core_mul=4,
                    stage_core=_CORDIC_ROTATION | Cordic.CoreStages.NORMALIZE_INPUT,
                    wmultiplier=18,
                    rom_attrs={"syn_romstyle": "logic"},
                ),
                ecp5_grade=8,
                cyclone4_grade=7,
            )
        ),
        # On speed grade 6 Diamond makes 94.3 to 97.3 MHz, from the engine's angles in block RAM into `z`, and nextpnr's
        # seeds 97.4 to 103.2, from the division's loop; on Artix-7 -1 Vivado misses by 0.221 ns, from the engine's `x`
        # into `u`, no register helping; on Cyclone IV E C8 Quartus makes 88.6, from the angles in block RAM into `z`.
        _large(
            _grade(
                Cordic(F36, mode=1, stage_core_mul=4, stage_core=_CORDIC_VECTORING, wmultiplier=18),
                ecp5_grade=7,
                artix7_grade=2,
                cyclone4_grade=7,
            )
        ),
        # On speed grade 7 Diamond makes 99.2 to 101.3 MHz, from the engine's adds of an iteration and from the reduced
        # coordinate into the fold, and nextpnr's seeds 89.3 to 98.1, from the sine's sum into the normalizer; on
        # Artix-7 -1 Vivado misses by 0.123 ns, in the division's loop, no register helping; on grade 8 without the
        # attribute Diamond puts the tables in block RAM and makes 94.8 to 100.2, from their reads. On Cyclone IV E
        # Quartus makes 87.2 on C8 and 94.1 on C7, from the engine's adds, and on Cyclone V C8 88.2, from the mode into
        # the normalizer's count, 94.6 with NORMALIZE_INPUT and 95.8 with ROUNDED.
        _large(
            _grade(
                Cordic(
                    F36,
                    stage_core_mul=4,
                    stage_core=~(
                        Cordic.CoreStages.SHIFT
                        | Cordic.CoreStages.CORRECTIONS
                        | Cordic.CoreStages.SHIFTED
                        | Cordic.CoreStages.NORMALIZE_INPUT
                        | Cordic.CoreStages.ROUNDED
                    ),
                    wmultiplier=18,
                    rom_attrs={"syn_romstyle": "logic"},
                ),
                ecp5_grade=8,
                artix7_grade=2,
                cyclone4_grade=6,
                cyclone5_grade=7,
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

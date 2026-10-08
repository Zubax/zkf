"""
Float synthesis module catalog: the device-independent set of cores to evaluate.

Defines what gets synthesized (ModuleSpec + MODULES), the RTL source list per kind, and the derived
metadata shown in the reports (parameters, pipeline depth, variant grouping). No flow/tool specifics
live here; both the Yosys and Diamond entry points import from this module. Not runnable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import sys

from common import REPO

sys.path.insert(0, str(REPO))
from zkf import OperatorModel, Timing, ZkfFormat  # noqa: E402  (path set up immediately above)


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    label: str
    top: str
    kind: str
    wexp: int
    wman: int
    wexp_unbiased: int
    wint: int = 0
    wk: int = 0  # zkf_mul_ilog2: width of the signed runtime shift k (0 -> RTL default WEXP+1)
    wexp_in: int = 0
    wman_in: int = 0
    wexp_out: int = 0
    wman_out: int = 0
    stage_input: int = 0
    stage_reduce: int = 0  # zkf_exp2: register reduced fixed-point i/f/flags before evaluator ROM input.
    stage_product: int = 0  # zkf_mul/fma/exp2/log2/cordic: _zkf_pmul pipeline depth / split 0..4.
    stage_product_final: int = -1  # zkf_log2 only: final f*C(f) split; -1 mirrors stage_product.
    stage_align: int = 0  # zkf_add, zkf_addsub, zkf_fma: 0 or 1 (alignment shifter split).
    stage_decode: int = 0  # zkf_add, zkf_addsub, zkf_mul_ilog2, zkf_fma, zkf_log2, zkf_divsqrt
    stage_normalize: int = 0  # zkf_add, zkf_addsub, zkf_fma, zkf_log2, zkf_from_int: 0/1/2 (normshift STAGE_SPLIT).
    stage_normalize_output: int = 0  # zkf_log2: 0/1 _zkf_normshift.STAGE_OUTPUT register.
    stage_pack: int = 0  # zkf_fma, zkf_log2, zkf_exp2, zkf_from_int: 0 or 1 (forwarded to _zkf_pack.STAGE_INPUT).
    stage_shift: int = 0  # zkf_rint
    stage_round: int = 0  # zkf_rint
    stage_output: int = 0
    unroll100: int = 100  # zkf_cordic: iterations per engine cycle x100 (50 = half-rate; 100/200/300/400).
    mode: int = 2  # zkf_cordic: 0 = rotation, 1 = vectoring; zkf_divsqrt: 0 = a/b, 1 = sqrt(a); 2 = per transaction.
    wmultiplier: int = 0  # zkf_mul/fma/exp2/log2/cordic: _zkf_pmul DSP tile-width hint (0 = symmetric;
    #   >=8 -> slice grid).
    emit_schematic: bool = True  # wide flattened generic schematics can dominate runtime; timing does not need them.


MODULES = [
    *[
        ModuleSpec(
            name=name,
            label=f"zkf_ilog2 (WEXP={wexp}, WMAN={wman}, WINT={wint}, STAGE_INPUT={si})",
            top=f"{name}_synth_top",
            kind="ilog2",
            wexp=wexp,
            wman=wman,
            wexp_unbiased=0,
            wint=wint,
            stage_input=si,
        )
        for name, wexp, wman, wint, si in (
            ("zkf_ilog2", 6, 18, 32, 0),
            ("zkf_ilog2_w8m24_i9", 8, 24, 9, 0),
            ("zkf_ilog2_w8m24", 8, 24, 32, 0),
            ("zkf_ilog2_w8m24_si1", 8, 24, 32, 1),
            ("zkf_ilog2_w11m53_i64", 11, 53, 64, 0),
        )
    ],
    ModuleSpec(
        name="_zkf_pack",
        label="_zkf_pack (normalized GRS)",
        top="_zkf_pack_synth_top",
        kind="pack",
        wexp=6,
        wman=18,
        wexp_unbiased=8,
    ),
    ModuleSpec(
        name="zkf_mul",
        label="zkf_mul",
        top="zkf_mul_synth_top",
        kind="mul",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_mul_sp1",
        label="zkf_mul (STAGE_PRODUCT=1)",
        top="zkf_mul_sp1_synth_top",
        kind="mul",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_product=1,
    ),
    ModuleSpec(
        name="zkf_mul_so1",
        label="zkf_mul (STAGE_OUTPUT=1, registered output)",
        top="zkf_mul_so1_synth_top",
        kind="mul",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_output=1,
    ),
    ModuleSpec(
        name="zkf_mul_w8m36_so1",
        label="zkf_mul (WEXP=8, WMAN=36, STAGE_PRODUCT=2 registered 2x2 18x18 split, STAGE_PACK=1, STAGE_OUTPUT=1)",
        top="zkf_mul_w8m36_so1_synth_top",
        kind="mul",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=2,
        wmultiplier=18,
        stage_pack=1,
        stage_output=1,
    ),
    ModuleSpec(
        name="zkf_mul_w8m36_sp2",
        label="zkf_mul (WEXP=8, WMAN=36, STAGE_PRODUCT=2 registered 2x2 18x18 split, WMULTIPLIER=18, STAGE_PACK=1 "
        "registers the pack inputs so the product->round->pack route closes on Diamond)",
        top="zkf_mul_w8m36_sp2_synth_top",
        kind="mul",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=2,
        wmultiplier=18,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_mul_w8m36_si1_sp2",
        label="zkf_mul (WEXP=8, WMAN=36, STAGE_INPUT=1 latched inputs + STAGE_PRODUCT=2 registered 2x2 18x18 "
        "split, WMULTIPLIER=18, STAGE_PACK=1 registers pack inputs for Diamond closure)",
        top="zkf_mul_w8m36_si1_sp2_synth_top",
        kind="mul",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_input=1,
        stage_product=2,
        wmultiplier=18,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_mul_w8m25_sp2",
        label="zkf_mul (WEXP=8, WMAN=25, STAGE_PRODUCT=2 registered symmetric 2x2 split 13/12, STAGE_PACK=1 "
        "registers pack inputs for Diamond closure)",
        top="zkf_mul_w8m25_sp2_synth_top",
        kind="mul",
        wexp=8,
        wman=25,
        wexp_unbiased=0,
        stage_product=2,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_add",
        label="zkf_add",
        top="zkf_add_synth_top",
        kind="add",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_add_w8m36_sd1_sa1_sn1",
        label="zkf_add (WEXP=8, WMAN=36, FPGA-optimal: STAGE_DECODE=1 register decoded operands, STAGE_ALIGN=1 "
        "split align shifter, STAGE_NORMALIZE=1 split close-cancellation normshift)",
        top="zkf_add_w8m36_sd1_sa1_sn1_synth_top",
        kind="add",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_decode=1,
        stage_align=1,
        stage_normalize=1,
    ),
    ModuleSpec(
        name="zkf_addsub",
        label="zkf_addsub",
        top="zkf_addsub_synth_top",
        kind="addsub",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_fma",
        label="zkf_fma (true single-rounding a*b+c; WEXP=6, WMAN=18, STAGE_INPUT=1 latched operands + "
        "STAGE_DECODE=1 splits the post-product normalize + magnitude-compare/select cone + STAGE_ALIGN=1 "
        "split aligner + STAGE_NORMALIZE=2 FMA-local 3-segment normalizer + STAGE_PACK=1 registered packer "
        "inputs: closes every datapath cone on Yosys and Diamond using a single "
        "MULT18X18D.)",
        top="zkf_fma_synth_top",
        kind="fma",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_input=1,
        stage_decode=1,
        stage_align=1,
        stage_normalize=2,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_fma_w8m36_sp2_sd1_sa1_sn2_pa1",
        label="zkf_fma (WEXP=8, WMAN=36, STAGE_PRODUCT=2 registered 2x2 quad 18x18, WMULTIPLIER=18, STAGE_DECODE=1, "
        "STAGE_ALIGN=1, STAGE_NORMALIZE=2, STAGE_PACK=1: register pack inputs + FMA-local 3-segment normalizer "
        "so both wide cones close)",
        top="zkf_fma_w8m36_sp2_sd1_sa1_sn2_pa1_synth_top",
        kind="fma",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=2,
        wmultiplier=18,
        stage_decode=1,
        stage_align=1,
        stage_normalize=2,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_fma_w8m36_si1_sp2_sd1_sa1_sn2_pa1",
        label="zkf_fma (WEXP=8, WMAN=36, STAGE_INPUT=1 latched inputs + STAGE_PRODUCT=2 registered 2x2 quad 18x18, "
        "WMULTIPLIER=18, STAGE_DECODE=1, STAGE_ALIGN=1, STAGE_NORMALIZE=2, STAGE_PACK=1: input register shields "
        "the wide operand bus while the rest closes both wide datapath cones)",
        top="zkf_fma_w8m36_si1_sp2_sd1_sa1_sn2_pa1_synth_top",
        kind="fma",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_input=1,
        stage_product=2,
        wmultiplier=18,
        stage_decode=1,
        stage_align=1,
        stage_normalize=2,
        stage_pack=1,
    ),
    # zkf_divsqrt: one digit pipeline for a/b and sqrt(a). At WMAN=36 the divider's first digit takes a stage of its
    # own (STAGE_DECODE=1): folded into stage 0 it misses 100 MHz.
    *[
        ModuleSpec(
            name=name,
            label=f"zkf_divsqrt (MODE={mode}, WEXP={wexp}, WMAN={wman}" + (", STAGE_DECODE=1)" if sd else ")"),
            top=f"{name}_synth_top",
            kind="divsqrt",
            mode=mode,
            wexp=wexp,
            wman=wman,
            wexp_unbiased=0,
            stage_decode=sd,
            emit_schematic=wman < 36,
        )
        for name, mode, wexp, wman, sd in (
            ("zkf_divsqrt", 2, 6, 18, 0),
            ("zkf_divsqrt_div", 0, 6, 18, 0),
            ("zkf_divsqrt_sqrt", 1, 6, 18, 0),
            ("zkf_divsqrt_w8m27", 2, 8, 27, 0),
            ("zkf_divsqrt_w8m36", 2, 8, 36, 1),
            ("zkf_divsqrt_div_w8m36", 0, 8, 36, 1),
            ("zkf_divsqrt_sqrt_w8m36", 1, 8, 36, 0),
        )
    ],
    ModuleSpec(
        name="zkf_cmp",
        label="zkf_cmp",
        top="zkf_cmp_synth_top",
        kind="cmp",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_cmp_w8m36",
        label="zkf_cmp (WEXP=8, WMAN=36)",
        top="zkf_cmp_w8m36_synth_top",
        kind="cmp",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_mul_ilog2",
        label="zkf_mul_ilog2 (runtime k; WEXP=6, WMAN=18, WK=7)",
        top="zkf_mul_ilog2_synth_top",
        kind="mul_ilog2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wk=7,
    ),
    ModuleSpec(
        name="zkf_mul_ilog2_w8m36",
        label="zkf_mul_ilog2 (runtime k; WEXP=8, WMAN=36, WK=9)",
        top="zkf_mul_ilog2_w8m36_synth_top",
        kind="mul_ilog2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wk=9,
    ),
    # WK=44 exercises the saturating wide-k decode; its single-cycle cone misses 100 MHz on Yosys without STAGE_DECODE.
    ModuleSpec(
        name="zkf_mul_ilog2_w8m36_wk44",
        label="zkf_mul_ilog2 (runtime k; WEXP=8, WMAN=36, WK=44, STAGE_DECODE=1)",
        top="zkf_mul_ilog2_w8m36_wk44_synth_top",
        kind="mul_ilog2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wk=44,
        stage_decode=1,
    ),
    ModuleSpec(
        name="zkf_mul_ilog2_w8m36_sd1",
        label="zkf_mul_ilog2 (runtime k; WEXP=8, WMAN=36, WK=9, STAGE_DECODE=1)",
        top="zkf_mul_ilog2_w8m36_sd1_synth_top",
        kind="mul_ilog2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wk=9,
        stage_decode=1,
    ),
    ModuleSpec(
        name="zkf_from_int_sn1",
        label="zkf_from_int (WINT=32, STAGE_NORMALIZE=1 split normshift)",
        top="zkf_from_int_sn1_synth_top",
        kind="from_int",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wint=32,
        stage_normalize=1,
    ),
    ModuleSpec(
        name="zkf_from_int_si1_sn1",
        label="zkf_from_int (WINT=32, STAGE_INPUT=1 + STAGE_NORMALIZE=1)",
        top="zkf_from_int_si1_sn1_synth_top",
        kind="from_int",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wint=32,
        stage_input=1,
        stage_normalize=1,
    ),
    ModuleSpec(
        name="zkf_from_int_w8m36_sn1",
        label="zkf_from_int (WEXP=8, WMAN=36, WINT=32, STAGE_NORMALIZE=1 split normshift)",
        top="zkf_from_int_w8m36_sn1_synth_top",
        kind="from_int",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wint=32,
        stage_normalize=1,
    ),
    ModuleSpec(
        name="zkf_resize_narrow",
        label="zkf_resize 6/18 -> 5/11 (narrowing)",
        top="zkf_resize_narrow_synth_top",
        kind="resize",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wexp_in=6,
        wman_in=18,
        wexp_out=5,
        wman_out=11,
    ),
    ModuleSpec(
        name="zkf_resize_narrow_si1",
        label="zkf_resize 6/18 -> 5/11 (narrowing, STAGE_INPUT=1)",
        top="zkf_resize_narrow_si1_synth_top",
        kind="resize",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wexp_in=6,
        wman_in=18,
        wexp_out=5,
        wman_out=11,
        stage_input=1,
    ),
    ModuleSpec(
        name="zkf_resize_widen",
        label="zkf_resize 5/11 -> 6/18 (widening)",
        top="zkf_resize_widen_synth_top",
        kind="resize",
        wexp=5,
        wman=11,
        wexp_unbiased=0,
        wexp_in=5,
        wman_in=11,
        wexp_out=6,
        wman_out=18,
    ),
    ModuleSpec(
        name="zkf_resize_widen_si1",
        label="zkf_resize 5/11 -> 6/18 (widening, STAGE_INPUT=1)",
        top="zkf_resize_widen_si1_synth_top",
        kind="resize",
        wexp=5,
        wman=11,
        wexp_unbiased=0,
        wexp_in=5,
        wman_in=11,
        wexp_out=6,
        wman_out=18,
        stage_input=1,
    ),
    ModuleSpec(
        name="zkf_resize_narrow_w8m36",
        label="zkf_resize 8/36 -> 6/18 (narrowing)",
        top="zkf_resize_narrow_w8m36_synth_top",
        kind="resize",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wexp_in=8,
        wman_in=36,
        wexp_out=6,
        wman_out=18,
    ),
    ModuleSpec(
        name="zkf_resize_widen_w8m36",
        label="zkf_resize 6/18 -> 8/36 (widening)",
        top="zkf_resize_widen_w8m36_synth_top",
        kind="resize",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wexp_in=6,
        wman_in=18,
        wexp_out=8,
        wman_out=36,
    ),
    # zkf_rint: one stage ahead of the rounding decision closes 100 MHz for both results.
    ModuleSpec(
        name="zkf_rint",
        label="zkf_rint (WEXP=6, WMAN=18, WINT=32, STAGE_SHIFT=1)",
        top="zkf_rint_synth_top",
        kind="rint",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wint=32,
        stage_shift=1,
    ),
    ModuleSpec(
        name="zkf_rint_w8m36_i44",
        label="zkf_rint (WEXP=8, WMAN=36, WINT=44, STAGE_SHIFT=1)",
        top="zkf_rint_w8m36_i44_synth_top",
        kind="rint",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wint=44,
        stage_shift=1,
    ),
    # zkf_exp2 / zkf_log2 (table + polynomial). Both close 100 MHz with margin on the LFE5U-12F at the 6/18
    # reference, but along opposite axes, so their headline entries differ (cf. how zkf_fma's plain entry carries
    # the knobs it needs to close while zkf_divsqrt's 6/18 entries do not):
    #   - exp2's Horner argument is the full reduced fraction, so acc*w is a wide x wide product (35x10 at WMAN=18).
    #     The unsplit/native forms (STAGE_PRODUCT 0/1) leave the multi-DSP cascade's output sum unregistered and top
    #     out ~85 MHz on Yosys, so the headline carries STAGE_PRODUCT=2: each Horner multiply maps to a registered
    #     2x2 DSP grid with an operand-capture stage (in the shared _zkf_pmul the registered split starts at 2, since
    #     1 is operand-capture + native multiply). It then reaches ~125 MHz Yosys.
    #   - log2's argument is the narrow segment-local fraction, so acc*w is wide x narrow and the Horner multiply can
    #     stay unsplit. The generated tables insert their own mandatory post-ROM hold register, so STAGE_PRODUCT is not
    #     used merely to isolate the first multiply; the knobs that close it are in the entry's label.
    ModuleSpec(
        name="zkf_exp2",
        label="zkf_exp2 (2**x, table+polynomial; STAGE_PRODUCT=2 splits each Horner multiply into a registered "
        "2x2 DSP grid with an operand-capture stage -- needed to close timing on ECP5, as the capture+native "
        "product (STAGE_PRODUCT=1) leaves the DSP-output sum unregistered)",
        top="zkf_exp2_synth_top",
        kind="exp2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_product=2,
    ),
    ModuleSpec(
        name="zkf_log2",
        label="zkf_log2 (log2(x), symmetric-reduction table+polynomial; STAGE_NORMALIZE=1 normalize-shift split + "
        "STAGE_PACK=1 rounder-input register + STAGE_PRODUCT_FINAL=1 operand-capture stage that shields the final "
        "unsigned |f|*C(f) multiply's DSP "
        "from the |f| magnitude-negate cone. The biased fixed-to-float back-end (EXP_IS_BIASED) and the "
        "direct-magnitude reconstruct freed enough slack to drop STAGE_NORMALIZE 2->1; closes 100 MHz on Yosys "
        "ECP5 and Diamond)",
        top="zkf_log2_synth_top",
        kind="log2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_normalize=1,
        stage_product_final=1,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_log2_so1",
        label="zkf_log2 (STAGE_NORMALIZE=2 + STAGE_PRODUCT_FINAL=1 final-multiply operand capture + STAGE_OUTPUT=1 "
        "registered-output boundary. The registered output adds back-end FFs, so this variant keeps "
        "STAGE_NORMALIZE=2 -- with STAGE_NORMALIZE=1 the added congestion drops it below 100 MHz; closes on "
        "Yosys ECP5 and Diamond)",
        top="zkf_log2_so1_synth_top",
        kind="log2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_normalize=2,
        stage_product_final=1,
        stage_output=1,
    ),
    # WEXP=8, WMAN=36 (degree-3 evaluator: three wide Horner multiplies). The shallow product modes are deep DSP
    # cascades that top out near 60 MHz; the split product modes cut the operands into chunks and add the
    # operand-capture stage. WMULTIPLIER=18 pins each slice to an 18-bit DSP tile: a symmetric STAGE_PRODUCT=3 split
    # (WMULTIPLIER=0) would cut the 53-bit accumulator into 18/18/17-bit slices, but the signed slice product then
    # needs a 19-bit operand (18 magnitude + sign), one bit past the MULT18X18 limit, so Lattice synthesis can drop the
    # whole Horner multiply into a fabric carry-chain soft multiplier (~76 MHz). The 18-bit tile hint derives DSP-fit
    # grids (3x3 for exp2's Horner, 3x2 for log2's signed Horner, 2x3 for log2's unsigned final f*C(f)), so every
    # multiply maps to DSP on both Yosys and Diamond. exp2's STAGE_PRODUCT=3 splits the flat 9-term column sum
    # (STAGE_PRODUCT=2: 64 MHz); its 27 of 28 DSPs then leave the capture-FF->DSP->partial-product hop as the limiter,
    # which no stage splits and only placement moves (Yosys ~85-118 MHz across seeds): STAGE_PRODUCT=4 and the
    # STAGE_INPUT/REDUCE/PACK/OUTPUT stages merely reshuffle it. Lean-first, log2 (24 DSPs) needs STAGE_NORMALIZE=2
    # (the x->1 normalize), STAGE_PACK=1 and STAGE_PRODUCT=3 on both multiplies, then meets the same DSP hop (Yosys
    # ~85-126 MHz across seeds).
    ModuleSpec(
        name="zkf_exp2_w8m36",
        label="zkf_exp2 (WEXP=8, WMAN=36, STAGE_PRODUCT=3 + WMULTIPLIER=18 18-bit DSP-tile grid)",
        top="zkf_exp2_w8m36_synth_top",
        kind="exp2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=3,
        wmultiplier=18,
        emit_schematic=False,
    ),
    ModuleSpec(
        name="zkf_log2_w8m36",
        label="zkf_log2 (WEXP=8, WMAN=36, STAGE_PRODUCT=3 + STAGE_PRODUCT_FINAL=3 + WMULTIPLIER=18 18-bit DSP-tile "
        "grid + STAGE_NORMALIZE=2 + STAGE_PACK=1)",
        top="zkf_log2_w8m36_synth_top",
        kind="log2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=3,
        stage_product_final=3,
        wmultiplier=18,
        stage_normalize=2,
        stage_pack=1,
        emit_schematic=False,
    ),
    # zkf_cordic fixed to rotation: a turns-reduction front end, the folded engine, the tiny-input bypass multiply, and
    # one shared _zkf_fixed_to_float back end. The rotation array is pure logic; the only DSPs are the shared 2*pi
    # linear-correction multiply. STAGE_NORMALIZE=2 + STAGE_PACK=1 keep the shared fixed-to-float pre-pack cone under
    # the 100 MHz gate across PNR seeds.
    ModuleSpec(
        name="zkf_cordic_rot",
        label="zkf_cordic (MODE=0: sin/cos of x turns; one datapath reused over ceil(K*100/UNROLL100) cycles + a "
        "shared linear-correction multiply; accept interval is latency+1. The only DSPs are the 2*pi correction; it "
        "fits the LFE5U-25F many times over. UNROLL100=100: one iteration per cycle, the shortest path)",
        top="zkf_cordic_rot_synth_top",
        kind="cordic",
        mode=0,
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        unroll100=100,  # one CORDIC iteration per engine cycle (shortest combinational path).
        stage_product=2,  # 2x2 + operand-capture split of the shared correction multiply -> 100 MHz.
        #   (Post-narrowing SP=1 native multiply was tried: Yosys 87 MHz -- the unregistered DSP
        #   cascade limits; reverted.)
        stage_normalize=2,  # both normshift barriers load-bearing (SN=1 reproducibly drops M18 to 99.5 MHz).
        stage_pack=1,  # rounder pack register; both it and the 2x2 product split are needed for 100 MHz.
    ),
    # WEXP=8, WMAN=36: same folded engine, more iterations on a wider datapath. Still the default LFE5U-25F (the
    # rotation array uses no DSPs; only the correction multiplies do).
    ModuleSpec(
        name="zkf_cordic_rot_w8m36",
        label="zkf_cordic (MODE=0, WEXP=8, WMAN=36; UNROLL100=50 with the decoupled z-path so the PHI correction "
        "overlaps the CORDIC, -4 cycles + STAGE_PRODUCT=3 (3x3 split) + STAGE_NORMALIZE=2 + STAGE_PACK=1; engine "
        "half-rate, 2 cycles/iteration; LFE5U-25F)",
        top="zkf_cordic_rot_w8m36_synth_top",
        kind="cordic",
        mode=0,
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        unroll100=50,  # half-rate 2-cycle engine: the wide (WX=62) shift+add recurrence misses 100 MHz single-cycle.
        stage_product=3,  # row-sum staging for the shared correction multiply (depth/latency knob) -> 100 MHz.
        #   (Post-narrowing SP=2 flat 3x3 sum tried: Diamond 56 / Yosys 93 MHz -- the flat sum
        #   limits; reverted.)
        wmultiplier=18,  # 18-bit tile hint keeps the narrowed 42x45 product in a 3x3 grid (9 DSP); latency-neutral.
        stage_normalize=2,
        stage_pack=1,
        emit_schematic=False,
    ),
    # zkf_cordic fixed to vectoring: the folded engine + a folded radix-4 divider (zkf_divsqrt's digit step) + the
    # shared _zkf_pmul + one shared _zkf_fixed_to_float back-end (time-multiplexed over the magnitude then theta).
    ModuleSpec(
        name="zkf_cordic_vec",
        label="zkf_cordic (MODE=1: atan2(y, x) in turns + hypot(y, x); one datapath reused over "
        "ceil(N*100/UNROLL100) engine cycles + a ceil(XF/2)-cycle radix-4 divide; UNROLL100=100 full rate "
        "+ shared _zkf_pmul STAGE_PRODUCT=2 WMULTIPLIER=18 + STAGE_NORMALIZE=2 + STAGE_PACK=1)",
        top="zkf_cordic_vec_synth_top",
        kind="cordic",
        mode=1,
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        unroll100=100,  # the one-cycle CORDIC iteration is the limiter (Yosys ~109-119 MHz across seeds); only
        #   UNROLL100=50 splits it (~123-129 MHz, +11 cycles); the other stages sit outside that loop.
        stage_product=2,  # narrowed _zkf_pmul: a 2x2 grid (KINV->WMAN+5), so the flat 4-term column sum is trivial.
        wmultiplier=18,  # 18-bit DSP-tile grid (MULT18X18D) for the magnitude / correction products.
        stage_normalize=2,  # both fixed-to-float normshift splits are load-bearing (STAGE_NORMALIZE=1: 83 MHz).
        stage_pack=1,  # rounder pack register in the shared fixed-to-float back-end (load-bearing: 81 MHz without).
    ),
    # WEXP=8, WMAN=36: the wider datapath enables the optional stages needed to close 100 MHz on all flows.
    ModuleSpec(
        name="zkf_cordic_vec_w8m36",
        label="zkf_cordic (MODE=1, WEXP=8, WMAN=36; UNROLL100=50 (half-rate) + stock 1-phase folded radix-4 divider "
        "(ceil(XF/2) steps + a one-cycle 3*den setup) + shared _zkf_pmul (STAGE_PRODUCT=4, WMULTIPLIER=18, "
        "KINV/INV_TAU narrowed to WMAN+5 -> 61x41 product in a 4x3 grid) + STAGE_NORMALIZE=2 + STAGE_PACK=1 + "
        "STAGE_OUTPUT; LFE5U-25F)",
        top="zkf_cordic_vec_w8m36_synth_top",
        kind="cordic",
        mode=1,
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        unroll100=50,  # half-rate 2-cycle engine for the wide (WX=62) shift+add recurrence.
        stage_input=0,  # LATENCY EXPERIMENT (si 1->0, -1 cyc): Yosys-screened 112.7 MHz; Diamond ECP5 confirmed.
        stage_product=4,  # narrowed _zkf_pmul: 61x41 product in a 4x3 grid (KINV/INV_TAU->WMAN+5, WMAG 124->102).
        #   The row-pair staging keeps the product off the limiter after the registered public output stage changes
        #   the wide design's placement pressure (STAGE_PRODUCT=3: Diamond 89-93 MHz).
        wmultiplier=18,  # 18-bit DSP-tile grid -> the 61x41 products fit the default device.
        stage_normalize=2,
        stage_pack=1,
        stage_output=1,
        emit_schematic=False,
    ),
    # MODE=2, compared against the fixed-mode w8m36 pair on the same part: zkf_cordic_vec_w8m36's knobs leave the shared
    # rounder -> result select -> STAGE_OUTPUT register path placement-bound (Yosys ~91-105 MHz across seeds), so
    # STAGE_PACK=2 registers the rounded result and STAGE_INPUT=1 relieves the shared selects (+2 cycles); the radix-4
    # divider step (~102-110 MHz) sits behind it.
    ModuleSpec(
        name="zkf_cordic_w8m36",
        label="zkf_cordic (MODE=2, WEXP=8, WMAN=36, rotation or vectoring per transaction; one shared engine + "
        "_zkf_pmul + _zkf_fixed_to_float; zkf_cordic_vec_w8m36's knobs + STAGE_INPUT=1 + STAGE_PACK=2)",
        top="zkf_cordic_w8m36_synth_top",
        kind="cordic",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        unroll100=50,
        stage_input=1,
        stage_product=4,
        wmultiplier=18,
        stage_normalize=2,
        stage_pack=2,
        stage_output=1,
        emit_schematic=False,
    ),
]


def module_group(spec: ModuleSpec) -> str:
    """Identifier for grouping a module with its STAGE_* variants."""
    match = re.match(r"^(.+?)(?:_(?:si|sr|sp|sa|sd|sn|pa|so)\d+)+$", spec.name)
    return match.group(1) if match else spec.name


def rtl_sources(spec: ModuleSpec) -> list[Path]:
    hdl = REPO / "zkf" / "rtl"
    if spec.kind == "pack":
        return [hdl / "_zkf_pack.v"]
    if spec.kind == "mul":
        return [hdl / "_zkf_pack.v", hdl / "zkf_pipe.v", hdl / "_zkf_pmul.v", hdl / "zkf_mul.v"]
    if spec.kind == "add":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_rshift_sticky.v",
            hdl / "zkf_add.v",
        ]
    if spec.kind == "addsub":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_rshift_sticky.v",
            hdl / "zkf_add.v",
            hdl / "zkf_addsub.v",
        ]
    if spec.kind == "fma":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_pmul.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_rshift_sticky.v",
            hdl / "zkf_fma.v",
        ]
    if spec.kind == "divsqrt":
        return [hdl / "zkf_pipe.v", hdl / "_zkf_pack.v", hdl / "_zkf_divsqrt_step.v", hdl / "zkf_divsqrt.v"]
    if spec.kind == "cmp":
        return [hdl / "zkf_pipe.v", hdl / "zkf_cmp.v"]
    if spec.kind == "ilog2":
        return [hdl / "zkf_pipe.v", hdl / "zkf_ilog2.v"]
    if spec.kind == "mul_ilog2":
        return [hdl / "zkf_pipe.v", hdl / "zkf_mul_ilog2.v"]
    if spec.kind == "from_int":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_fixed_to_float.v",
            hdl / "zkf_from_int.v",
        ]
    if spec.kind == "resize":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "zkf_resize.v",
        ]
    if spec.kind == "rint":
        return [hdl / "zkf_pipe.v", hdl / "zkf_rint.v"]
    if spec.kind in {"exp2", "log2"}:
        # The generate-if selects the table whose name matches WMAN (the degree is a closed-form localparam inside the
        # table); the other WMAN branches reference undefined modules but are untaken, so synthesis prunes them (like
        # the _zkf_invalid_* sentinels). Yosys's hierarchy -check, however, also elaborates the *generic* zkf_<func>
        # (default WMAN), so that WMAN's table must be present too -- include both (deduped) and let synthesis prune
        # the unused generic.
        def table(wman: int) -> Path:
            return hdl / "_tables" / f"_zkf_{spec.kind}_m{wman}.v"

        DEFAULT_WMAN = 18  # the default WMAN of zkf_exp2 / zkf_log2
        tables = [table(w) for w in sorted({DEFAULT_WMAN, spec.wman})]
        sources = [hdl / "_zkf_pack.v", hdl / "zkf_pipe.v", hdl / "_zkf_pmul.v"]
        if spec.kind == "exp2":
            sources += [hdl / "_zkf_rshift_sticky.v"]
        if spec.kind == "log2":
            # log2's _zkf_fixed_to_float helper owns the _zkf_normshift instance, optional combine register, and
            # pack-input/output pipeline shared with zkf_from_int.
            sources += [hdl / "_zkf_normshift.v", hdl / "_zkf_fixed_to_float.v"]
        return sources + [hdl / "_zkf_horner.v", *tables, hdl / f"zkf_{spec.kind}.v"]
    if spec.kind == "cordic":
        # The default-WMAN (18) table is included as well so Yosys's hierarchy -check is satisfied for the generic
        # modules.
        tables = [hdl / "_tables" / f"_zkf_cordic_m{w}.v" for w in sorted({18, spec.wman})]
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_fixed_to_float.v",
            hdl / "_zkf_pmul.v",
            hdl / "_zkf_divsqrt_step.v",
            hdl / "_zkf_cordic_core.v",
            *tables,
            hdl / "_zkf_txn.v",
            hdl / "zkf_cordic.v",
        ]
    raise ValueError(f"unsupported module kind: {spec.kind}")


def model_for(spec: ModuleSpec) -> OperatorModel:
    fmt = ZkfFormat(spec.wexp_out, spec.wman_out) if spec.kind == "resize" else ZkfFormat(spec.wexp, spec.wman)
    values = {
        "wexp_unbiased": spec.wexp_unbiased or None,
        "wint": spec.wint or 32,
        "wk": spec.wk or None,
        "wexp_in": spec.wexp_in or None,
        "wman_in": spec.wman_in or None,
        "unroll100": spec.unroll100,
        "mode": spec.mode,
        "stage_input": spec.stage_input,
        "stage_reduce": spec.stage_reduce,
        "stage_product": spec.stage_product,
        "stage_product_final": spec.stage_product_final if spec.stage_product_final >= 0 else None,
        "stage_align": spec.stage_align,
        "stage_decode": spec.stage_decode,
        "stage_normalize": spec.stage_normalize,
        "stage_normalize_output": spec.stage_normalize_output,
        "stage_pack": spec.stage_pack,
        "stage_shift": spec.stage_shift,
        "stage_round": spec.stage_round,
        "stage_output": spec.stage_output,
        "wmultiplier": spec.wmultiplier,
    }
    factory = fmt.model_of(spec.kind)
    defaults = factory(**({"mode": spec.mode} if spec.kind == "cordic" else {}))  # the mode sets the format bounds
    return factory(**{name: values[name] for name in defaults.config.keys() if name in values})


def register_stages(spec: ModuleSpec) -> int | tuple[int, int]:
    timing = model_for(spec).timing
    if isinstance(timing, Timing):
        return timing.latency
    return timing[0].latency, timing[1].latency


def format_register_stages(stages: int | tuple[int, int]) -> str:
    if isinstance(stages, tuple):
        return f"{stages[0]} / {stages[1]} stages (rotation / vectoring)"
    suffix = "stage" if stages == 1 else "stages"
    return f"{stages} {suffix}"


def params(spec: ModuleSpec) -> str:
    return ", ".join(f"{name}={value}" for name, value in model_for(spec).params.items())


def selected_modules(names: str | None) -> list[ModuleSpec]:
    if not names:
        return MODULES
    selected = {name.strip() for name in names.split(",") if name.strip()}
    modules = [spec for spec in MODULES if spec.name in selected]
    missing = selected - {spec.name for spec in modules}
    if missing:
        raise ValueError(f"unknown module names: {', '.join(sorted(missing))}")
    return modules


def flow_modules(args_modules: str | None, flow_env_name: str) -> list[ModuleSpec]:
    names = (
        args_modules
        or os.environ.get(flow_env_name)
        or os.environ.get("FLOAT_SYNTH_MODULES")
        or os.environ.get("SYNTH_MODULES")
    )
    return selected_modules(names)

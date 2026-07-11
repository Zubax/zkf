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
from zkf import OperatorModel, ZkfFormat  # noqa: E402  (path set up immediately above)


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
    stage_input: int = 0  # zkf_div, zkf_from_int, zkf_to_int, zkf_resize, zkf_mul, zkf_fma: 0 or 1.
    stage_reduce: int = 0  # zkf_exp2: register reduced fixed-point i/f/flags before evaluator ROM input.
    stage_product: int = 0  # zkf_mul/fma/exp2/log2/sincos: _zkf_pmul pipeline depth / split 0..4.
    stage_product_final: int = -1  # zkf_log2 only: final f*C(f) split; -1 mirrors stage_product.
    stage_align: int = 0  # zkf_add, zkf_addsub, zkf_fma: 0 or 1 (alignment shifter split).
    stage_decode: int = 0  # zkf_add, zkf_addsub, zkf_mul_ilog2_const, zkf_fma, zkf_log2: 0 or 1.
    stage_normalize: int = 0  # zkf_add, zkf_addsub, zkf_fma, zkf_log2, zkf_from_int: 0/1/2 (normshift STAGE_SPLIT).
    stage_normalize_output: int = 0  # zkf_log2: 0/1 _zkf_normshift.STAGE_OUTPUT register.
    stage_pack: int = 0  # zkf_fma, zkf_log2, zkf_exp2, zkf_from_int: 0 or 1 (forwarded to _zkf_pack.STAGE_INPUT).
    stage_output: int = 0  # pack-based ops: 0 = combinational output (default); 1 = registered output (+1 cycle).
    unroll100: int = 100  # zkf_sincos: CORDIC iterations per engine cycle x100 (50 = half-rate; 100/200/300/400).
    wmultiplier: int = 0  # zkf_mul/fma/exp2/log2/sincos: _zkf_pmul DSP tile-width hint (0 = symmetric;
    #   >=8 -> slice grid).
    emit_schematic: bool = True  # wide flattened generic schematics can dominate runtime; timing does not need them.


MUL_ILOG2_CONST_K = 10  # representative midrange shift for the synthesis evaluation harness


MODULES = [
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
    ModuleSpec(
        name="_zkf_div_core",
        label="_zkf_div_core",
        top="_zkf_div_core_synth_top",
        kind="div_core",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_div",
        label="zkf_div",
        top="zkf_div_synth_top",
        kind="div",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_div_si1",
        label="zkf_div (STAGE_INPUT=1)",
        top="zkf_div_si1_synth_top",
        kind="div",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_input=1,
    ),
    ModuleSpec(
        name="zkf_div_w8m36",
        label="zkf_div (WEXP=8, WMAN=36, FPGA-optimal: quad 18x18; STAGE_INPUT=1 shields the wide input decode cone, "
        "STAGE_OUTPUT=1 registers the wide quotient round so Diamond closes timing)",
        top="zkf_div_w8m36_synth_top",
        kind="div",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_input=1,
        stage_output=1,
        emit_schematic=False,
    ),
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
        name="zkf_sort",
        label="zkf_sort",
        top="zkf_sort_synth_top",
        kind="sort",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_sort_w8m36",
        label="zkf_sort (WEXP=8, WMAN=36)",
        top="zkf_sort_w8m36_synth_top",
        kind="sort",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_mul_ilog2_const",
        label="zkf_mul_ilog2_const (K=+10)",
        top="zkf_mul_ilog2_const_synth_top",
        kind="mul_ilog2_const",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
    ),
    ModuleSpec(
        name="zkf_mul_ilog2_const_w8m36_sd1",
        label="zkf_mul_ilog2_const (WEXP=8, WMAN=36, K=+10, STAGE_DECODE=1)",
        top="zkf_mul_ilog2_const_w8m36_sd1_synth_top",
        kind="mul_ilog2_const",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_decode=1,
    ),
    # WK=44/SD0 pins the full cone that regressed to 97.37 MHz when accumulator width followed WK.
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
    ModuleSpec(
        name="zkf_mul_ilog2_w8m36_wk44",
        label="zkf_mul_ilog2 (runtime k; WEXP=8, WMAN=36, WK=44, STAGE_DECODE=0)",
        top="zkf_mul_ilog2_w8m36_wk44_synth_top",
        kind="mul_ilog2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wk=44,
        stage_decode=0,
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
        name="zkf_to_int",
        label="zkf_to_int (WINT=32)",
        top="zkf_to_int_synth_top",
        kind="to_int",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wint=32,
    ),
    ModuleSpec(
        name="zkf_to_int_si1",
        label="zkf_to_int (WINT=32, STAGE_INPUT=1)",
        top="zkf_to_int_si1_synth_top",
        kind="to_int",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        wint=32,
        stage_input=1,
    ),
    ModuleSpec(
        name="zkf_to_int_w8m36",
        label="zkf_to_int (WEXP=8, WMAN=36, WINT=32)",
        top="zkf_to_int_w8m36_synth_top",
        kind="to_int",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wint=32,
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
    # zkf_round (round-to-integer-valued float; runtime round_mode). The rounder is a variable-position
    # boundary-mask + guard/sticky reduction + increment adder feeding _zkf_pack as a pre-biased assembler
    # (EXP_IS_BIASED=1, no bias round-trip). Unpipelined the cone is ~21 ns, so the headline configs carry
    # STAGE_DECODE=1 (split mask generation from the reduction/add) and STAGE_PACK=1 (register the rounder->packer
    # cut). At 8/36 the wider 36-bit reduction also needs STAGE_OUTPUT=1 to hold 100 MHz on the Spartan/nextpnr flow.
    ModuleSpec(
        name="zkf_round",
        label="zkf_round (WEXP=6, WMAN=18, STAGE_DECODE=1 + STAGE_PACK=1)",
        top="zkf_round_synth_top",
        kind="round",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_decode=1,
        stage_pack=1,
    ),
    ModuleSpec(
        name="zkf_round_w8m36",
        label="zkf_round (WEXP=8, WMAN=36, STAGE_DECODE=1 + STAGE_PACK=1 + STAGE_OUTPUT=1)",
        top="zkf_round_w8m36_synth_top",
        kind="round",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_decode=1,
        stage_pack=1,
        stage_output=1,
    ),
    # zkf_exp2 / zkf_log2 (table + polynomial). Both close 100 MHz with margin on the LFE5U-12F at the 6/18
    # reference, but along opposite axes, so their headline entries differ (cf. how zkf_fma's plain entry carries
    # the knobs it needs to close while zkf_div's does not):
    #   - exp2's Horner argument is the full reduced fraction, so acc*w is a wide x wide product (35x10 at WMAN=18).
    #     The unsplit/native forms (STAGE_PRODUCT 0/1) leave the multi-DSP cascade's output sum unregistered and top
    #     out ~85 MHz on Yosys, so the headline carries STAGE_PRODUCT=2: each Horner multiply maps to a registered
    #     2x2 DSP grid with an operand-capture stage (in the shared _zkf_pmul the registered split starts at 2, since
    #     1 is operand-capture + native multiply). It then reaches ~125 MHz Yosys.
    #   - log2's argument is the narrow segment-local fraction, so acc*w is wide x narrow and the Horner multiply can
    #     stay unsplit. The generated tables insert their own mandatory post-ROM hold register, so STAGE_PRODUCT is not
    #     used merely to isolate the first multiply. The critical path is the final f*C(f) multiply + its carry chain;
    #     STAGE_NORMALIZE=2 (splitting the close-cancellation x->1 normalize) is the single knob that relieves the
    #     surrounding placement enough to close 100 MHz on both Yosys ECP5 (102.9 MHz) and Diamond (105.9 MHz). The
    #     formerly carried STAGE_NORMALIZE_OUTPUT=1 / STAGE_PACK=1 were redundant -- adding those stages only worsened
    #     placement congestion (e.g. si1+sn2+sno1+pa1 dropped to ~98 MHz) -- so they are removed. log2_so1 adds
    #     STAGE_OUTPUT=1 as the alternate registered-output boundary.
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
        "STAGE_PRODUCT_FINAL=1 operand-capture stage that shields the final unsigned |f|*C(f) multiply's DSP "
        "from the |f| magnitude-negate cone. The biased fixed-to-float back-end (EXP_IS_BIASED) and the "
        "direct-magnitude reconstruct freed enough slack to drop STAGE_NORMALIZE 2->1; closes 100 MHz on Yosys "
        "ECP5 (103.0 MHz) and Diamond)",
        top="zkf_log2_synth_top",
        kind="log2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_normalize=1,
        stage_product_final=1,
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
    # grids for the signed*unsigned products (3x3 for exp2 Horner, 3x2 for log2 Horner, 3x3 for log2's final f*C(f)),
    # so every multiply maps to DSP on both Yosys and Diamond. exp2 closes at STAGE_PRODUCT=3 without the optional
    # fixed-point split register. log2's final f*C(f) multiply is now fully UNSIGNED (|f|*C(f) with the sign folded into
    # the back-end add/subtract), which cuts its grid from a signed 3x3 to an unsigned 2x3 -- 27 DSPs -> 24 -- and that
    # relieves the LFE5U-12F placement bind. With the lighter DSP load and the biased-EXP/direct-magnitude back-end,
    # the formerly load-bearing STAGE_PRODUCT=4, STAGE_DECODE=1, STAGE_OUTPUT=1 and STAGE_NORMALIZE_OUTPUT all come back
    # out: STAGE_PRODUCT=3 + STAGE_DECODE=0 + no back-end output registers clears 100 MHz with margin (fewer FFs raise
    # f_max on a congestion-bound part). It still needs STAGE_NORMALIZE=2 (the x->1 normalize) and STAGE_PACK=1.
    ModuleSpec(
        name="zkf_exp2_w8m36",
        label="zkf_exp2 (WEXP=8, WMAN=36, STAGE_INPUT=1 + "
        "STAGE_PRODUCT=3 + WMULTIPLIER=18 18-bit DSP-tile grid + STAGE_OUTPUT=1)",
        top="zkf_exp2_w8m36_synth_top",
        kind="exp2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_input=1,
        stage_product=3,
        wmultiplier=18,
        stage_output=1,
        emit_schematic=False,
    ),
    ModuleSpec(
        name="zkf_log2_w8m36",
        label="zkf_log2 (WEXP=8, WMAN=36, STAGE_INPUT=1 + STAGE_PRODUCT=3 Horner grid + STAGE_PRODUCT_FINAL=3 final "
        "unsigned |f|*C(f) grid + WMULTIPLIER=18 18-bit DSP-tile grid + STAGE_NORMALIZE=2 (deep normshift split) "
        "+ STAGE_PACK=1; the unsigned final multiply cut the grid to 24 DSPs, so STAGE_DECODE, STAGE_OUTPUT and "
        "STAGE_NORMALIZE_OUTPUT all drop out -- fewer back-end FFs raise f_max on this congestion-bound part "
        "(116.2 MHz on Yosys ECP5))",
        top="zkf_log2_w8m36_synth_top",
        kind="log2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_input=1,
        stage_product=3,
        stage_product_final=3,
        wmultiplier=18,
        stage_normalize=2,
        stage_pack=1,
        emit_schematic=False,
    ),
    # sin/cos of a phase in turns: a turns-reduction front end, an iterative folded CORDIC (one datapath reused), the
    # tiny-input bypass multiply, and one shared _zkf_fixed_to_float back end. The rotation array is pure logic; the
    # only DSPs are the shared 2*pi linear-correction multiply. STAGE_NORMALIZE=2 + STAGE_PACK=1 keep the shared
    # fixed-to-float pre-pack cone under the 100 MHz gate across PNR seeds.
    ModuleSpec(
        name="zkf_sincos",
        label="zkf_sincos (sin/cos of x turns, iterative folded CORDIC; one datapath reused over ceil(K*100/UNROLL100) "
        "cycles + a shared linear-correction multiply; accept interval is latency+1. The only DSPs are the 2*pi "
        "correction; it "
        "fits the LFE5U-25F many times over. UNROLL100=100: one iteration per cycle, the shortest path)",
        top="zkf_sincos_synth_top",
        kind="sincos",
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
        name="zkf_sincos_w8m36",
        label="zkf_sincos (WEXP=8, WMAN=36, iterative folded CORDIC; UNROLL100=50 with the default decoupled "
        "z-path so the PHI correction overlaps the CORDIC, -4 cycles + STAGE_PRODUCT=3 "
        "(3x3 split) + STAGE_NORMALIZE=2 + STAGE_PACK=1; engine half-rate, 2 cycles/iteration; LFE5U-25F)",
        top="zkf_sincos_w8m36_synth_top",
        kind="sincos",
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
    # zkf_atan2 (two-input vectoring CORDIC): atan2(y, x) in turns + hypot(y, x). One folded engine + a folded radix-4
    # divider (the _zkf_div_core primitives) + the shared _zkf_pmul + one shared _zkf_fixed_to_float back-end
    # (time-multiplexed over the magnitude then theta). WEXP=6, WMAN=18 on LFE5U-25F (the same default device as
    # zkf_sincos).
    ModuleSpec(
        name="zkf_atan2",
        label="zkf_atan2 (atan2(y, x) in turns + hypot(y, x), iterative vectoring CORDIC; one datapath reused over "
        "ceil(N*100/UNROLL100) engine cycles + a ceil(XF/2)-cycle radix-4 divide; UNROLL100=50 "
        "half-rate + shared _zkf_pmul STAGE_PRODUCT=2 WMULTIPLIER=18 + STAGE_NORMALIZE=2 + "
        "STAGE_PACK)",
        top="zkf_atan2_synth_top",
        kind="atan2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        unroll100=50,  # half-rate 2-cycle engine: the full-rate vectoring shift+add+angle-LUT recurrence is the
        #   limiter on BOTH flows (Yosys ~100, Diamond ~75 -- 26 logic levels), insensitive to PAR;
        #   g_pipe splits the shift-sample from the add, clearing the cone. 2 cycles/iteration.
        stage_product=2,  # narrowed _zkf_pmul: now a 2x2 grid (KINV->WMAN+5), so the flat 4-term column sum is
        #   trivial
        #   and the row/column-sum split of SP=3 is no longer needed. Limiter is the radix-4 divider
        #   (Yosys) / fixed-to-float normshift (Diamond), not the product.
        wmultiplier=18,  # 18-bit DSP-tile grid (MULT18X18D) for the magnitude / correction products.
        stage_normalize=2,  # split the fixed-to-float close-cancellation normshift (the Diamond back-end limiter).
        stage_pack=1,  # rounder pack register in the shared fixed-to-float back-end
        #   (load-bearing: pack->output cone).
        stage_output=0,  # LATENCY (-1): the wide back-end datapath is mildly over-pipelined here, so dropping the
        #   packer output register relieves routing congestion (Yosys 112.6 MHz, Diamond >100).
    ),
    # WEXP=8, WMAN=36: the wider datapath enables the optional stages needed to close 100 MHz on all flows.
    ModuleSpec(
        name="zkf_atan2_w8m36",
        label="zkf_atan2 (WEXP=8, WMAN=36, vectoring CORDIC; UNROLL100=50 (half-rate) + stock 1-phase folded radix-4 "
        "divider (ceil(XF/2) steps + a one-cycle 3*den setup) + shared "
        "_zkf_pmul (STAGE_PRODUCT=4, WMULTIPLIER=18, KINV/INV_TAU narrowed to WMAN+5 -> 61x41 product "
        "in a 4x3 grid) + "
        "STAGE_NORMALIZE=2 + STAGE_PACK=1 + STAGE_OUTPUT; the same default LFE5U-25F as zkf_sincos_w8m36)",
        top="zkf_atan2_w8m36_synth_top",
        kind="atan2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        unroll100=50,  # half-rate 2-cycle engine for the wide (WX=62) shift+add recurrence.
        stage_input=0,  # LATENCY EXPERIMENT (si 1->0, -1 cyc): Yosys-screened 112.7 MHz; Diamond ECP5 confirmed.
        stage_product=4,  # narrowed _zkf_pmul: 61x41 product in a 4x3 grid (KINV/INV_TAU->WMAN+5,
        #   WMAG 124->102). The row-pair staging keeps the product off the limiter after the
        #   registered public output stage changes the wide design's placement pressure.
        wmultiplier=18,  # 18-bit DSP-tile grid -> the 61x41 products fit the default device.
        stage_normalize=2,
        stage_pack=1,
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
    if spec.kind == "div_core":
        return [hdl / "_zkf_div_core.v"]
    if spec.kind == "div":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_div_core.v",
            hdl / "zkf_div.v",
        ]
    if spec.kind == "cmp":
        return [hdl / "zkf_pipe.v", hdl / "zkf_cmp_comb.v", hdl / "zkf_cmp.v"]
    if spec.kind == "sort":
        return [hdl / "zkf_pipe.v", hdl / "zkf_cmp_comb.v", hdl / "zkf_sort.v"]
    if spec.kind == "mul_ilog2_const":
        return [hdl / "zkf_pipe.v", hdl / "zkf_mul_ilog2_const.v"]
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
    if spec.kind == "to_int":
        return [
            hdl / "zkf_pipe.v",
            hdl / "_zkf_rshift_sticky.v",
            hdl / "_zkf_to_fixpoint.v",
            hdl / "zkf_to_int.v",
        ]
    if spec.kind == "resize":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "zkf_resize.v",
        ]
    if spec.kind == "round":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "zkf_round.v",
        ]
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
            # exp2's _zkf_to_fixpoint helper uses _zkf_rshift_sticky for the right-shift path; the helper itself
            # owns the decode + folded-constant predicate cone shared with zkf_to_int.
            sources += [hdl / "_zkf_rshift_sticky.v", hdl / "_zkf_to_fixpoint.v"]
        if spec.kind == "log2":
            # log2's _zkf_fixed_to_float helper owns the _zkf_normshift instance, optional combine register, and
            # pack-input/output pipeline shared with zkf_from_int.
            sources += [hdl / "_zkf_normshift.v", hdl / "_zkf_fixed_to_float.v"]
        return sources + [hdl / "_zkf_horner.v", *tables, hdl / f"zkf_{spec.kind}.v"]
    if spec.kind == "sincos":
        # Left-shift turns reducer (inline) + octant fold + the shared CORDIC engine (_zkf_cordic) bound per WMAN
        # (_zkf_cordic_m<WMAN>) + the shared correction multiply (_zkf_pmul) + one shared _zkf_fixed_to_float back end.
        # Include both the default-WMAN (18) core and this spec's WMAN, deduped, so Yosys's hierarchy -check is
        # satisfied for the generic zkf_sincos too.
        def core(wman: int) -> Path:
            return hdl / "_tables" / f"_zkf_cordic_m{wman}.v"

        cores = [core(w) for w in sorted({18, spec.wman})]  # 18 = the default WMAN of zkf_sincos
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_fixed_to_float.v",
            hdl / "_zkf_pmul.v",
            hdl / "_zkf_cordic.v",
            *cores,
            hdl / "zkf_sincos.v",
        ]
    if spec.kind == "atan2":
        # Two-input vectoring CORDIC: the shared engine (_zkf_cordic) bound per WMAN, one shared _zkf_fixed_to_float
        # back-end (time-multiplexed over magnitude then theta), the folded radix-4 divider (the _zkf_div_core
        # primitives), and the shared _zkf_pmul (magnitude + correction products). Include both the default-WMAN (18)
        # core and this spec's WMAN, deduped, so Yosys's hierarchy -check is satisfied for the generic.
        def core(wman: int) -> Path:
            return hdl / "_tables" / f"_zkf_cordic_m{wman}.v"

        cores = [core(w) for w in sorted({18, spec.wman})]  # 18 = the default WMAN of zkf_atan2
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_fixed_to_float.v",
            hdl / "_zkf_cordic.v",
            hdl / "_zkf_div_core.v",
            hdl / "_zkf_pmul.v",
            *cores,
            hdl / "zkf_atan2.v",
        ]
    raise ValueError(f"unsupported module kind: {spec.kind}")


def model_for(spec: ModuleSpec) -> OperatorModel:
    fmt = ZkfFormat(spec.wexp_out, spec.wman_out) if spec.kind == "resize" else ZkfFormat(spec.wexp, spec.wman)
    values = {
        "wexp_unbiased": spec.wexp_unbiased or None,
        "wint": spec.wint or 32,
        "wk": spec.wk or None,
        "k": MUL_ILOG2_CONST_K,
        "wexp_in": spec.wexp_in or None,
        "wman_in": spec.wman_in or None,
        "unroll100": spec.unroll100,
        "stage_input": spec.stage_input,
        "stage_reduce": spec.stage_reduce,
        "stage_product": spec.stage_product,
        "stage_product_final": spec.stage_product_final if spec.stage_product_final >= 0 else None,
        "stage_align": spec.stage_align,
        "stage_decode": spec.stage_decode,
        "stage_normalize": spec.stage_normalize,
        "stage_normalize_output": spec.stage_normalize_output,
        "stage_pack": spec.stage_pack,
        "stage_output": spec.stage_output,
        "wmultiplier": spec.wmultiplier,
    }
    factory = fmt.model_of(spec.kind)
    defaults = factory()
    return factory(**{name: values[name] for name in defaults.config.keys() if name in values})


def register_stages(spec: ModuleSpec) -> int:
    return model_for(spec).latency


def format_register_stages(stages: int) -> str:
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

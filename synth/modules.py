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
    stage_product: int = 0  # _zkf_pmul depth / split 0..4; zkf_exp2/log2: of c1*w, the kernel's other products below.
    stage_product_square: int = 0
    stage_product_odd: int = 0
    stage_product_chain: int = 0
    stage_product_final: int = 0  # zkf_log2 only: depth of the final |f|*C(f) multiply.
    stage_align: int = 0  # zkf_add, zkf_fma: 0 or 1 (alignment shifter split); zkf_log2: kernel register.
    stage_reduce: int = 0  # zkf_exp2
    stage_decode: int = 0  # zkf_add, zkf_mul_ilog2, zkf_fma, zkf_divsqrt
    stage_normalize: int = 0  # zkf_add, zkf_fma, zkf_from_int, zkf_cordic
    stage_pack: int = 0  # _zkf_pack.STAGE_INPUT; zkf_exp2, zkf_log2: their own stage.
    stage_shift: int = 0  # zkf_rint
    stage_round: int = 0  # zkf_rint
    stage_output: int = 0
    unroll100: int = 100  # zkf_cordic: iterations per engine cycle x100 (50 = half-rate; 100/200/300/400).
    divide_radix: int = 4  # zkf_cordic
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
    # The stage knobs of the zkf_mul, zkf_add, zkf_fma, zkf_exp2, zkf_log2 and zkf_cordic presets are what the closure
    # procedure of AGENTS.md arrives at from every stage off; a preset whose name pins one knob (_si1, _so1, _sp1) adds
    # it to that base.
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
        label="zkf_mul (STAGE_OUTPUT=1)",
        top="zkf_mul_so1_synth_top",
        kind="mul",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_output=1,
    ),
    ModuleSpec(
        name="zkf_mul_w8m36_so1",
        label="zkf_mul (WEXP=8, WMAN=36, STAGE_PRODUCT=2 + WMULTIPLIER=18 + STAGE_OUTPUT=1)",
        top="zkf_mul_w8m36_so1_synth_top",
        kind="mul",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=2,
        wmultiplier=18,
        stage_output=1,
    ),
    ModuleSpec(
        name="zkf_mul_w8m36_sp2",
        label="zkf_mul (WEXP=8, WMAN=36, STAGE_PRODUCT=2 + WMULTIPLIER=18)",
        top="zkf_mul_w8m36_sp2_synth_top",
        kind="mul",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=2,
        wmultiplier=18,
    ),
    ModuleSpec(
        name="zkf_mul_w8m36_si1_sp2",
        label="zkf_mul (WEXP=8, WMAN=36, STAGE_INPUT=1 + STAGE_PRODUCT=2 + WMULTIPLIER=18)",
        top="zkf_mul_w8m36_si1_sp2_synth_top",
        kind="mul",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_input=1,
        stage_product=2,
        wmultiplier=18,
    ),
    ModuleSpec(
        name="zkf_mul_w8m25_sp2",
        label="zkf_mul (WEXP=8, WMAN=25, STAGE_PRODUCT=2 + STAGE_PACK=1)",
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
        name="zkf_add_w8m36_sn2",
        label="zkf_add (WEXP=8, WMAN=36, STAGE_NORMALIZE=2)",
        top="zkf_add_w8m36_sn2_synth_top",
        kind="add",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_normalize=2,
    ),
    ModuleSpec(
        name="zkf_fma",
        label="zkf_fma (a*b+c rounded once; STAGE_DECODE=1 + STAGE_NORMALIZE=2)",
        top="zkf_fma_synth_top",
        kind="fma",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_decode=1,
        stage_normalize=2,
    ),
    ModuleSpec(
        name="zkf_fma_w8m36_sp2_sd1_sa1_sn2_pa1",
        label="zkf_fma (WEXP=8, WMAN=36, STAGE_PRODUCT=2 + WMULTIPLIER=18 + STAGE_DECODE=1 + STAGE_ALIGN=1 + "
        "STAGE_NORMALIZE=2 + STAGE_PACK=1)",
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
        label="zkf_fma (WEXP=8, WMAN=36, STAGE_INPUT=1 + STAGE_PRODUCT=2 + WMULTIPLIER=18 + STAGE_DECODE=1 + "
        "STAGE_ALIGN=1 + STAGE_NORMALIZE=2 + STAGE_PACK=1)",
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
        label="zkf_rint (WEXP=8, WMAN=36, WINT=44, STAGE_SHIFT=1, STAGE_ROUND=1)",
        top="zkf_rint_w8m36_i44_synth_top",
        kind="rint",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        wint=44,
        stage_shift=1,
        stage_round=1,
    ),
    ModuleSpec(
        name="zkf_exp2",
        label="zkf_exp2 (2**x, table+polynomial; STAGE_PRODUCT=1 + STAGE_PRODUCT_CHAIN=1)",
        top="zkf_exp2_synth_top",
        kind="exp2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_product=1,
        stage_product_chain=1,
    ),
    ModuleSpec(
        name="zkf_log2",
        label="zkf_log2 (log2(x), table+polynomial; STAGE_PRODUCT=1 + STAGE_PRODUCT_CHAIN=1 + "
        "STAGE_PRODUCT_FINAL=2 + WMULTIPLIER=18)",
        top="zkf_log2_synth_top",
        kind="log2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_product=1,
        stage_product_chain=1,
        stage_product_final=2,
        wmultiplier=18,
    ),
    ModuleSpec(
        name="zkf_log2_so1",
        label="zkf_log2 (STAGE_PRODUCT=1 + STAGE_PRODUCT_CHAIN=1 + STAGE_PRODUCT_FINAL=2 + WMULTIPLIER=18 + "
        "STAGE_OUTPUT=1)",
        top="zkf_log2_so1_synth_top",
        kind="log2",
        wexp=6,
        wman=18,
        wexp_unbiased=0,
        stage_product=1,
        stage_product_chain=1,
        stage_product_final=2,
        wmultiplier=18,
        stage_output=1,
    ),
    ModuleSpec(
        name="zkf_exp2_w8m24",
        label="zkf_exp2 (WEXP=8, WMAN=24, STAGE_PRODUCT=1 + STAGE_PRODUCT_CHAIN=1)",
        top="zkf_exp2_w8m24_synth_top",
        kind="exp2",
        wexp=8,
        wman=24,
        wexp_unbiased=0,
        stage_product=1,
        stage_product_chain=1,
        emit_schematic=False,
    ),
    ModuleSpec(
        name="zkf_log2_w8m24",
        label="zkf_log2 (WEXP=8, WMAN=24, STAGE_PRODUCT=1 + STAGE_PRODUCT_CHAIN=1 + STAGE_PRODUCT_FINAL=2 + "
        "WMULTIPLIER=18)",
        top="zkf_log2_w8m24_synth_top",
        kind="log2",
        wexp=8,
        wman=24,
        wexp_unbiased=0,
        stage_product=1,
        stage_product_chain=1,
        stage_product_final=2,
        wmultiplier=18,
        emit_schematic=False,
    ),
    ModuleSpec(
        name="zkf_exp2_w8m36",
        label="zkf_exp2 (WEXP=8, WMAN=36, STAGE_PRODUCT=3 + STAGE_PRODUCT_SQUARE=1 + STAGE_PRODUCT_ODD=2 + "
        "STAGE_PRODUCT_CHAIN=2 + WMULTIPLIER=18)",
        top="zkf_exp2_w8m36_synth_top",
        kind="exp2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=3,
        stage_product_square=1,
        stage_product_odd=2,
        stage_product_chain=2,
        wmultiplier=18,
        emit_schematic=False,
    ),
    ModuleSpec(
        name="zkf_log2_w8m36",
        label="zkf_log2 (WEXP=8, WMAN=36, STAGE_PRODUCT=3 + STAGE_PRODUCT_SQUARE=1 + STAGE_PRODUCT_ODD=2 + "
        "STAGE_PRODUCT_CHAIN=2 + STAGE_ALIGN=1 + STAGE_PRODUCT_FINAL=2 + STAGE_PACK=1 + "
        "WMULTIPLIER=18)",
        top="zkf_log2_w8m36_synth_top",
        kind="log2",
        wexp=8,
        wman=36,
        wexp_unbiased=0,
        stage_product=3,
        stage_product_square=1,
        stage_product_odd=2,
        stage_product_chain=2,
        stage_align=1,
        stage_product_final=2,
        stage_pack=1,
        wmultiplier=18,
        emit_schematic=False,
    ),
    # zkf_cordic: at WMAN=36 the engine's 62-bit recurrence takes two cycles per iteration (UNROLL100=50).
    *[
        ModuleSpec(
            name=name,
            label=f"zkf_cordic (MODE={mode}, WEXP={wexp}, WMAN={wman}, WMULTIPLIER=18, "
            + " + ".join(f"{k.upper()}={v}" for k, v in knobs.items())
            + ")",
            top=f"{name}_synth_top",
            kind="cordic",
            mode=mode,
            wexp=wexp,
            wman=wman,
            wexp_unbiased=0,
            wmultiplier=18,
            emit_schematic=wman < 36,
            **knobs,
        )
        for name, mode, wexp, wman, knobs in (
            ("zkf_cordic_rot", 0, 6, 18, dict(stage_product=2, stage_normalize=2, stage_pack=1)),
            ("zkf_cordic_vec", 1, 6, 18, dict(stage_product=2, stage_normalize=2, stage_pack=1)),
            ("zkf_cordic_rot_w8m36", 0, 8, 36, dict(unroll100=50, stage_product=3, stage_normalize=2, stage_pack=2)),
            *(
                (
                    name,
                    mode,
                    8,
                    36,
                    dict(unroll100=50, divide_radix=2, stage_product=3, stage_normalize=3, stage_pack=2),
                )
                for name, mode in (("zkf_cordic_vec_w8m36", 1), ("zkf_cordic_w8m36", 2))
            ),
        )
    ],
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
            hdl / "_zkf_add_core.v",
            hdl / "zkf_add.v",
        ]
    if spec.kind == "fma":
        return [
            hdl / "_zkf_pack.v",
            hdl / "zkf_pipe.v",
            hdl / "_zkf_pmul.v",
            hdl / "_zkf_normshift.v",
            hdl / "_zkf_rshift_sticky.v",
            hdl / "_zkf_add_core.v",
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
        # The generate-if selects the table whose name matches WMAN; the untaken branches reference undefined modules
        # (like the _zkf_invalid_* sentinels). The generic zkf_<func> (default WMAN) is elaborated as well, so that
        # WMAN's table is listed too.
        def table(wman: int) -> Path:
            return hdl / "_tables" / f"_zkf_{spec.kind}_m{wman}.v"

        DEFAULT_WMAN = 18  # the default WMAN of zkf_exp2 / zkf_log2
        tables = [table(w) for w in sorted({DEFAULT_WMAN, spec.wman})]
        sources = [hdl / "zkf_pipe.v", hdl / "_zkf_pmul.v"]
        if spec.kind == "log2":
            sources += [hdl / "_zkf_normshift.v"]
        return sources + [*tables, hdl / f"zkf_{spec.kind}.v"]
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
        "divide_radix": spec.divide_radix,
        "mode": spec.mode,
        "stage_input": spec.stage_input,
        "stage_reduce": spec.stage_reduce,
        "stage_product": spec.stage_product,
        "stage_product_square": spec.stage_product_square,
        "stage_product_odd": spec.stage_product_odd,
        "stage_product_chain": spec.stage_product_chain,
        "stage_product_final": spec.stage_product_final,
        "stage_align": spec.stage_align,
        "stage_decode": spec.stage_decode,
        "stage_normalize": spec.stage_normalize,
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

#!/usr/bin/env python3
"""
Simulation target catalog: the source lists, toplevels, and cocotb modules that used to live in the
FuseSoC core file. test_float_matrix.py drives each Run through cocotb's native runner (get_runner)
using this table, so there is no FuseSoC dependency.

FILESETS maps a fileset name to its RTL/testbench sources (repo-relative). TARGETS maps a target base
(a Run.target with its trailing ``_<sim>`` stripped) to the toplevel module, the cocotb test module,
and the filesets to compile. The dependency lists mirror the historical FuseSoC filesets exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Shared table sources, spelled out once and reused by the transcendental filesets.
_EXP2_TABLES = [f"hdl/_tables/_zkf_exp2_m{w}.v" for w in (16, 18, 24, 27, 32, 36, 48, 53)]
_LOG2_TABLES = [f"hdl/_tables/_zkf_log2_m{w}.v" for w in (16, 18, 24, 27, 32, 36, 48, 53)]
_CORDIC_TABLES = [f"hdl/_tables/_zkf_cordic_m{w}.v" for w in (16, 18, 24, 27, 32, 36, 48, 53)]

FILESETS: dict[str, list[str]] = {
    "rtl_pack": ["hdl/zkf_pipe.v", "hdl/_zkf_pack.v"],
    "rtl_mul": ["hdl/_zkf_pack.v", "hdl/zkf_pipe.v", "hdl/_zkf_pmul.v", "hdl/zkf_mul.v"],
    "rtl_add": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_rshift_sticky.v",
        "hdl/zkf_add.v",
    ],
    "rtl_fma": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_pmul.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_rshift_sticky.v",
        "hdl/zkf_fma.v",
    ],
    "rtl_div": ["hdl/_zkf_pack.v", "hdl/zkf_pipe.v", "hdl/_zkf_div_core.v", "hdl/zkf_div.v"],
    "rtl_pipe": ["hdl/zkf_pipe.v"],
    "rtl_normshift": ["hdl/_zkf_normshift.v"],
    "rtl_rshift": ["hdl/_zkf_rshift_sticky.v"],
    "rtl_abs": ["hdl/zkf_abs.v"],
    "rtl_neg": ["hdl/zkf_neg.v"],
    "rtl_is_finite": ["hdl/zkf_is_finite.v"],
    "rtl_saturate": ["hdl/zkf_saturate.v"],
    "rtl_cmp": ["hdl/zkf_pipe.v", "hdl/zkf_cmp_comb.v", "hdl/zkf_cmp.v"],
    "rtl_sort": ["hdl/zkf_pipe.v", "hdl/zkf_cmp_comb.v", "hdl/zkf_sort.v"],
    "rtl_addsub": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_rshift_sticky.v",
        "hdl/zkf_add.v",
        "hdl/zkf_addsub.v",
    ],
    "rtl_mul_ilog2_const": ["hdl/zkf_pipe.v", "hdl/zkf_mul_ilog2_const.v"],
    "rtl_mul_ilog2": ["hdl/zkf_pipe.v", "hdl/zkf_mul_ilog2.v"],
    "rtl_from_int": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_fixed_to_float.v",
        "hdl/zkf_from_int.v",
    ],
    "rtl_to_int": ["hdl/zkf_pipe.v", "hdl/_zkf_rshift_sticky.v", "hdl/_zkf_to_fixpoint.v", "hdl/zkf_to_int.v"],
    "rtl_resize": ["hdl/_zkf_pack.v", "hdl/zkf_pipe.v", "hdl/zkf_resize.v"],
    "rtl_round": ["hdl/_zkf_pack.v", "hdl/zkf_pipe.v", "hdl/zkf_round.v"],
    "rtl_exp2": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_pmul.v",
        "hdl/_zkf_rshift_sticky.v",
        "hdl/_zkf_to_fixpoint.v",
        "hdl/_zkf_horner.v",
        *_EXP2_TABLES,
        "hdl/zkf_exp2.v",
    ],
    "rtl_log2": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_pmul.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_fixed_to_float.v",
        "hdl/_zkf_horner.v",
        *_LOG2_TABLES,
        "hdl/zkf_log2.v",
    ],
    "rtl_sincos": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_fixed_to_float.v",
        "hdl/_zkf_pmul.v",
        "hdl/_zkf_cordic.v",
        *_CORDIC_TABLES,
        "hdl/zkf_sincos.v",
    ],
    "rtl_atan2": [
        "hdl/_zkf_pack.v",
        "hdl/zkf_pipe.v",
        "hdl/_zkf_normshift.v",
        "hdl/_zkf_fixed_to_float.v",
        "hdl/_zkf_cordic.v",
        *_CORDIC_TABLES,
        "hdl/_zkf_div_core.v",
        "hdl/_zkf_pmul.v",
        "hdl/zkf_atan2.v",
    ],
    "tb_mul_ilog2_const_wrap": ["tb/zkf_mul_ilog2_const_wrap.v"],
}


@dataclass(frozen=True)
class Target:
    toplevel: str
    cocotb_module: str
    filesets: tuple[str, ...]

    def sources(self) -> list[str]:
        """Ordered, de-duplicated source list for this target."""
        seen: dict[str, None] = {}
        for fileset in self.filesets:
            for path in FILESETS[fileset]:
                seen.setdefault(path, None)
        return list(seen)


def _t(toplevel: str, cocotb_module: str, *filesets: str) -> Target:
    return Target(toplevel, cocotb_module, filesets)


TARGETS: dict[str, Target] = {
    "sim_pack": _t("_zkf_pack", "test_pack", "rtl_pack"),
    "sim_mul": _t("zkf_mul", "test_mul", "rtl_mul"),
    "sim_add": _t("zkf_add", "test_add", "rtl_add"),
    "sim_addsub": _t("zkf_addsub", "test_addsub", "rtl_addsub"),
    "sim_fma": _t("zkf_fma", "test_fma", "rtl_fma"),
    "sim_div": _t("zkf_div", "test_div", "rtl_div"),
    "sim_abs": _t("zkf_abs", "test_abs", "rtl_abs"),
    "sim_neg": _t("zkf_neg", "test_neg", "rtl_neg"),
    "sim_is_finite": _t("zkf_is_finite", "test_is_finite", "rtl_is_finite"),
    "sim_saturate": _t("zkf_saturate", "test_saturate", "rtl_saturate"),
    "sim_cmp": _t("zkf_cmp", "test_cmp", "rtl_cmp"),
    "sim_sort": _t("zkf_sort", "test_sort", "rtl_sort"),
    "sim_pipe": _t("zkf_pipe", "test_pipe", "rtl_pipe"),
    "sim_normshift": _t("_zkf_normshift", "test_normshift", "rtl_normshift"),
    "sim_rshift": _t("_zkf_rshift_sticky", "test_rshift", "rtl_rshift"),
    "sim_mul_ilog2_const": _t(
        "zkf_mul_ilog2_const_wrap", "test_mul_ilog2_const", "rtl_mul_ilog2_const", "tb_mul_ilog2_const_wrap"
    ),
    "sim_mul_ilog2": _t("zkf_mul_ilog2", "test_mul_ilog2", "rtl_mul_ilog2"),
    "sim_from_int": _t("zkf_from_int", "test_from_int", "rtl_from_int"),
    "sim_to_int": _t("zkf_to_int", "test_to_int", "rtl_to_int"),
    "sim_resize": _t("zkf_resize", "test_resize", "rtl_resize"),
    "sim_round": _t("zkf_round", "test_round", "rtl_round"),
    "sim_exp2": _t("zkf_exp2", "test_exp2", "rtl_exp2"),
    "sim_log2": _t("zkf_log2", "test_log2", "rtl_log2"),
    "sim_sincos": _t("zkf_sincos", "test_sincos", "rtl_sincos"),
    "sim_atan2": _t("zkf_atan2", "test_atan2", "rtl_atan2"),
    # Algebraic-property suite: same RTL as the direct operator targets, but the test_properties cocotb module.
    "sim_properties_mul": _t("zkf_mul", "test_properties", "rtl_mul"),
    "sim_properties_add": _t("zkf_add", "test_properties", "rtl_add"),
    "sim_properties_addsub": _t("zkf_addsub", "test_properties", "rtl_addsub"),
}

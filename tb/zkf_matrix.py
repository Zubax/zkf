#!/usr/bin/env python3
"""
Single source of truth for the float verification matrix.

Every simulation the suite runs is one Run here: a module, a simulator (icarus/verilator), a set of
vlog parameters, plusargs, and a tier. The tiers gate selection:

  pr          per-PR set (runs by default; what `nox -s tests` exercises)
  deep        full parameter-equivalence-class sweep (correctness on icarus, coverage on verilator)
  properties  algebraic-property tests (test_properties.py) on the add/addsub/mul toplevels
  fast        the smallest-config smoke set

test_float_matrix.py parametrizes pytest over build_matrix(), tagging each Run with its tier and
simulator as markers; pyproject.toml deselects deep/properties/fast by default, so the deep work skips
unless explicitly selected (pytest -m deep, etc.).

Running this module directly prints the matrix (counts per tier/sim, or the full list with --list).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SEED = os.environ.get("FLOAT_SEED", "0x9e3779b97f4a7c15")
CORE = "zubax:kulibin:float"

# pack:     (config, wexp, wman, wexp_unbiased, kind, count)
PACK = [
    ("w2_m4_u4_exhaustive", 2, 4, 4, "exhaustive", 0),
    ("w3_m4_u5_exhaustive", 3, 4, 5, "exhaustive", 0),
    ("w5_m8_u8_random", 5, 8, 8, "random", 768),
    ("w8_m24_u12_random", 8, 24, 12, "random", 2048),
]
# binary:   (config, wexp, wman, kind, count)
BINARY = [
    ("w2_m4_exhaustive", 2, 4, "exhaustive", 0),
    ("w3_m4_exhaustive", 3, 4, "exhaustive", 0),
    ("w3_m5_random", 3, 5, "random", 512),
    ("w4_m6_random", 4, 6, "random", 512),
    ("w5_m11_random", 5, 11, "random", 768),
    ("w6_m18_random", 6, 18, "random", 768),
    ("w7_m17_random", 7, 17, "random", 768),
    ("w8_m24_random", 8, 24, "random", 1024),
    ("w11_m53_random", 11, 53, "random", 384),
]
# fma (ternary a*b+c): exhaustive only at the smallest format; wider formats random (ternary-exhaustive explodes).
# (config, wexp, wman, kind, count)
FMA = [
    ("w2_m4_exhaustive", 2, 4, "exhaustive", 0),
    ("w3_m4_random", 3, 4, "random", 2048),
    ("w3_m5_random", 3, 5, "random", 768),
    ("w4_m6_random", 4, 6, "random", 768),
    ("w5_m11_random", 5, 11, "random", 768),
    ("w6_m18_random", 6, 18, "random", 1024),
    ("w8_m24_random", 8, 24, "random", 1024),
    ("w8_m36_random", 8, 36, "random", 1024),
    ("w11_m53_random", 11, 53, "random", 512),
    # Small WEXP, large WMAN: the close-cancellation corrected exponent underflows far below the product exponent
    # range, so this guards the sub-path exponent width (directed w4m30 cancellation witnesses run here).
    ("w4_m30_random", 4, 30, "random", 512),
]
# unary:    (config, wexp, wman, kind, count)
UNARY = [
    ("w2_m4_exhaustive", 2, 4, "exhaustive", 0),
    ("w3_m4_exhaustive", 3, 4, "exhaustive", 0),
    ("w5_m11_random", 5, 11, "random", 512),
    ("w8_m24_random", 8, 24, "random", 1024),
    ("w11_m53_random", 11, 53, "random", 384),
]
# exp2/log2: tables exist only for the supported WMAN {16,18,24,27,32,36,48,53} (min 16; see
# SUPPORTED_WMAN/WMAN_MIN in zkf_transcendental.py), so every config must use one of those.
# (config, wexp, wman, kind, count)
TRANS_EXPLOG = [
    ("w2_m16_exhaustive", 2, 16, "exhaustive", 0),  # exhaustive at the minimum WMAN
    ("w6_m16_random", 6, 16, "random", 512),  # min table width, wider exponent
    ("w8_m24_random", 8, 24, "random", 1024),
    ("w8_m32_random", 8, 32, "random", 768),
    ("w11_m53_random", 11, 53, "random", 384),
    # Wide-exponent guard: min WMAN keeps the datapath small while WEXP=20 and the directed overflow/underflow/inf/
    # pow2 corners exercise the wide-exponent reduction, OOR threshold, and clamp. The random supplement is kept small
    # on purpose -- extreme-exponent inputs cost ~0.5 s/vector in the high-precision oracle, so the directed corners
    # (not this sweep) carry the coverage here.
    ("w20_m16_random", 20, 16, "random", 128),
]
TRANS_EXPLOG_EXT = [
    (2, 16, "exhaustive", 0),
    (3, 16, "exhaustive", 0),
    (8, 27, "random", 512),
    (8, 32, "random", 512),
    (14, 16, "random", 128),  # wide-exponent guard (exhaustive infeasible)
]

# sincos/atan2 support down to WMAN=16 via the CORDIC generator (WMAN=11 was dropped -- XF-bound faithfulness), so
# their low-cost coverage uses WMAN=16.
TRANS_TRIG = [
    ("w2_m16_exhaustive", 2, 16, "exhaustive", 0),
    ("w3_m16_random", 3, 16, "random", 512),
    ("w5_m16_random", 5, 16, "random", 512),
    ("w8_m24_random", 8, 24, "random", 1024),
    ("w8_m32_random", 8, 32, "random", 768),
    ("w11_m53_random", 11, 53, "random", 384),
    ("w20_m16_random", 20, 16, "random", 128),
]
TRANS_TRIG_EXT = [
    (2, 16, "exhaustive", 0),
    (3, 16, "random", 512),
    (6, 16, "random", 512),
    (8, 27, "random", 512),
    (8, 32, "random", 512),
    (14, 16, "random", 128),
]

# zkf_atan2 is two-input, so joint-exhaustive is infeasible even at min WMAN: every format uses directed (the full
# special/axis/diagonal pair table) + random pairs. Covers the synthesized 6/18 and 8/36 plus the wide-exponent guard.
TRANS_ATAN2 = [
    ("w6_m18_random", 6, 18, "random", 1536),
    ("w8_m24_random", 8, 24, "random", 1024),
    ("w8_m36_random", 8, 36, "random", 768),
    ("w11_m53_random", 11, 53, "random", 384),
    ("w20_m16_random", 20, 16, "random", 128),
]
# pipe:     (config, width, stages, count)
PIPE = [("w8_n0", 8, 0, 64), ("w8_n4", 8, 4, 96), ("w24_n2", 24, 2, 96)]
# from_int/to_int: (config, wexp, wman, wint, kind, count)
FROM_INT = [
    ("w2_m4_int4_exhaustive", 2, 4, 4, "exhaustive", 0),
    ("w3_m4_int8_exhaustive", 3, 4, 8, "exhaustive", 0),
    ("w3_m5_int8_exhaustive", 3, 5, 8, "exhaustive", 0),
    ("w5_m11_int16_random", 5, 11, 16, "random", 512),
    ("w6_m18_int32_random", 6, 18, 32, "random", 768),
    ("w8_m24_int32_random", 8, 24, 32, "random", 1024),
    # Wide WINT makes the leading-one position + BIAS exceed a position-only-sized exponent field; the directed
    # extremes (int_max etc.) overflow to +inf and would regress to +0 if WEU is mis-sized. Exhaustive is infeasible
    # at this width, so directed covers the boundary deterministically.
    ("w6_m18_int128_directed", 6, 18, 128, "directed", 0),
]
TO_INT = FROM_INT + [("w11_m53_int32_random", 11, 53, 32, "random", 384)]
# resize: (config, wexp_in, wman_in, wexp_out, wman_out, kind, count). Covers every (WMAN, WEXP) quadrant so each
# zkf_resize elaboration-time branch runs: widen-only fast path, pack/g_widen (g_same_width, g_zero_pad), and
# pack/g_narrow (DROP=1/2/7).
RESIZE = [
    ("w3_m4_to_w3_m4_exhaustive", 3, 4, 3, 4, "exhaustive", 0),
    ("w3_m4_to_w4_m4_exhaustive", 3, 4, 4, 4, "exhaustive", 0),
    ("w3_m4_to_w3_m6_exhaustive", 3, 4, 3, 6, "exhaustive", 0),
    ("w3_m5_to_w3_m4_exhaustive", 3, 5, 3, 4, "exhaustive", 0),
    ("w3_m4_to_w4_m6_exhaustive", 3, 4, 4, 6, "exhaustive", 0),
    ("w4_m6_to_w3_m4_exhaustive", 4, 6, 3, 4, "exhaustive", 0),
    ("w5_m4_to_w3_m4_exhaustive", 5, 4, 3, 4, "exhaustive", 0),
    ("w5_m4_to_w3_m6_exhaustive", 5, 4, 3, 6, "exhaustive", 0),
    ("w6_m18_to_w5_m11_random", 6, 18, 5, 11, "random", 768),
    ("w5_m11_to_w6_m18_random", 5, 11, 6, 18, "random", 768),
    ("w8_m24_to_w6_m18_random", 8, 24, 6, 18, "random", 512),
    ("w11_m53_to_w8_m24_random", 11, 53, 8, 24, "random", 384),
]

# extended (deep) format lists.
BIN_EXT = [
    (2, 5, "exhaustive", 0),
    (4, 5, "exhaustive", 0),
    (2, 7, "exhaustive", 0),
    (3, 6, "exhaustive", 0),
    (5, 4, "exhaustive", 0),
    (4, 6, "exhaustive", 0),
    (3, 7, "random", 512),
    (6, 17, "random", 768),
    (8, 23, "random", 768),
    (7, 12, "random", 512),
    (9, 24, "random", 512),
    (6, 19, "random", 512),
]
DIV_EXT = [
    (2, 5, "exhaustive", 0),
    (4, 5, "exhaustive", 0),
    (3, 6, "exhaustive", 0),
    (5, 4, "exhaustive", 0),
    (6, 17, "random", 512),
    (8, 23, "random", 512),
    (7, 12, "random", 512),
]
UNARY_EXT = [
    (2, 5, "exhaustive", 0),
    (4, 5, "exhaustive", 0),
    (3, 6, "exhaustive", 0),
    (6, 17, "random", 512),
    (8, 23, "random", 512),
]
# fma deep formats: all random (ternary-exhaustive infeasible above wfull=6). (wexp, wman, kind, count)
FMA_EXT = [
    (4, 5, "random", 512),
    (3, 6, "random", 512),
    (5, 4, "random", 512),
    (3, 7, "random", 512),
    (6, 17, "random", 768),
    (8, 23, "random", 512),
    (7, 12, "random", 512),
    (9, 24, "random", 512),
]


@dataclass
class Run:
    module: str  # mul, add, pack, to_int, pipe, ...
    sim: str  # icarus | verilator
    tier: str  # pr | deep | properties | fast
    config: str  # config name including knob suffixes
    target: str  # simulation target, e.g. sim_mul_icarus
    root: str
    vlog: list  # [(name, value), ...]  -> --NAME value
    plus: list  # [(name, value), ...]  -> --ZKF_... value
    defines: list = field(default_factory=list)  # [(name, value), ...] -> vlogdefine parameters

    @property
    def id(self) -> str:
        return f"{self.tier}-{self.module}-{self.sim}-{self.config}"


def _root(tier: str, sim: str, module: str, config: str) -> str:
    if tier == "fast":
        return f"build/float/fast/{config}"
    base = {
        ("pr", "icarus"): "icarus",
        ("pr", "verilator"): "verilator",
        ("deep", "icarus"): "icarus-ext",
        ("deep", "verilator"): "verilator-toggle",
        ("properties", "icarus"): "properties",
    }[(tier, sim)]
    return f"build/float/{base}/{module}/{config}"


def _common(kind: str, count: int, config: str, with_kind: bool = True) -> list:
    parts = [("ZKF_KIND", kind)] if with_kind else []
    return parts + [("ZKF_COUNT", count), ("ZKF_SEED", SEED), ("ZKF_CONFIG", config)]


def _run(
    module,
    sim,
    tier,
    config,
    vlog,
    *,
    kind="exhaustive",
    count=0,
    target=None,
    with_kind=True,
    plus_names=None,
    root_module=None,
    defines=None,
) -> Run:
    """Assemble one Run. vlog params are mirrored to plusargs as ZKF_<name> unless plus_names overrides."""
    plus_names = plus_names or {}
    plus = [(plus_names.get(name, "ZKF_" + name), val) for name, val in vlog]
    plus += _common(kind, count, config, with_kind=with_kind)
    target = target or f"sim_{module}_{sim}"
    root = _root(tier, sim, root_module or module, config)
    return Run(module, sim, tier, config, target, root, vlog, plus, list(defines or []))


def _binary(
    module,
    sim,
    tier,
    base,
    w,
    m,
    kind,
    count,
    *,
    sp=None,
    si=None,
    sd=None,
    sa=None,
    sn=None,
    pa=None,
    so=None,
    wm=None,
    target=None,
    root_module=None,
) -> Run:
    # wm (WMULTIPLIER) is only meaningful for the multiply (zkf_mul); other _binary ops do not declare it.
    vlog = [("WEXP", w), ("WMAN", m)]
    suffix = ""
    if sp is not None:
        vlog.append(("STAGE_PRODUCT", sp))
        suffix += f"_sp{sp}"
    if wm is not None:
        vlog.append(("WMULTIPLIER", wm))
        suffix += f"_wm{wm}"
    if si is not None:
        vlog.append(("STAGE_INPUT", si))
        suffix += f"_si{si}"
    if sd is not None and sa is not None:
        vlog += [("STAGE_DECODE", sd), ("STAGE_ALIGN", sa)]
        suffix += f"_sd{sd}_sa{sa}"
    elif sd is not None:
        vlog.append(("STAGE_DECODE", sd))
        suffix += f"_sd{sd}"
    if sn is not None:
        vlog.append(("STAGE_NORMALIZE", sn))
        suffix += f"_sn{sn}"
    if pa is not None:
        vlog.append(("STAGE_PACK", pa))
        suffix += f"_pa{pa}"
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    return _run(module, sim, tier, base + suffix, vlog, kind=kind, count=count, target=target, root_module=root_module)


def _ilog2(sim, tier, base, w, m, kind, count, *, wk=None, si=None, sd=None) -> Run:
    # zkf_mul_ilog2: k is a runtime port of width WK (default WEXP+1). WK is always passed so the RTL parameter and
    # the bench (ZKF_WK) agree; the default width already reaches shifts that saturate to inf / flush to zero.
    wk = wk if wk is not None else w + 1
    vlog = [("WEXP", w), ("WMAN", m), ("WK", wk)]
    suffix = f"_wk{wk}"
    if si is not None:
        vlog.append(("STAGE_INPUT", si))
        suffix += f"_si{si}"
    if sd is not None:
        vlog.append(("STAGE_DECODE", sd))
        suffix += f"_sd{sd}"
    return _run("mul_ilog2", sim, tier, base + suffix, vlog, kind=kind, count=count, root_module="zkf_mul_ilog2")


def _fma(
    sim, tier, base, w, m, kind, count, *, sp=None, si=None, sd=None, sa=None, sn=None, pa=None, so=None, wm=None
) -> Run:
    vlog = [("WEXP", w), ("WMAN", m)]
    suffix = ""
    if sp is not None:
        vlog.append(("STAGE_PRODUCT", sp))
        suffix += f"_sp{sp}"
    if wm is not None:
        vlog.append(("WMULTIPLIER", wm))
        suffix += f"_wm{wm}"
    if si is not None:
        vlog.append(("STAGE_INPUT", si))
        suffix += f"_si{si}"
    if sd is not None:
        vlog.append(("STAGE_DECODE", sd))
        suffix += f"_sd{sd}"
    if sa is not None:
        vlog.append(("STAGE_ALIGN", sa))
        suffix += f"_sa{sa}"
    if sn is not None:
        vlog.append(("STAGE_NORMALIZE", sn))
        suffix += f"_sn{sn}"
    if pa is not None:
        vlog.append(("STAGE_PACK", pa))
        suffix += f"_pa{pa}"
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    return _run("fma", sim, tier, base + suffix, vlog, kind=kind, count=count)


def _pack(sim, tier, config, w, m, u, kind, count, *, so=None, eb=None, nov=None) -> Run:
    vlog = [("WEXP", w), ("WMAN", m), ("WEXP_UNBIASED", u)]
    suffix = ""
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    if eb is not None:
        vlog.append(("EXP_IS_BIASED", eb))
        suffix += f"_eb{eb}"
    if nov is not None:
        vlog.append(("ASSUME_NO_OVERFLOW", nov))
        suffix += f"_nov{nov}"
    return _run("pack", sim, tier, config + suffix, vlog, kind=kind, count=count)


def _cast(module, sim, tier, base, w, m, wint, kind, count, si, *, sn=None, pa=None, so=None) -> Run:
    vlog = [("WEXP", w), ("WMAN", m), ("WINT", wint), ("STAGE_INPUT", si)]
    suffix = f"_si{si}"
    if sn is not None:
        vlog.append(("STAGE_NORMALIZE", sn))
        suffix += f"_sn{sn}"
    if pa is not None:
        vlog.append(("STAGE_PACK", pa))
        suffix += f"_pa{pa}"
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    return _run(module, sim, tier, f"{base}{suffix}", vlog, kind=kind, count=count)


def _resize(sim, tier, base, wi, mi, wo, mo, kind, count, si, so=None) -> Run:
    vlog = [("WEXP_IN", wi), ("WMAN_IN", mi), ("WEXP_OUT", wo), ("WMAN_OUT", mo), ("STAGE_INPUT", si)]
    suffix = f"_si{si}"
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    return _run("resize", sim, tier, f"{base}{suffix}", vlog, kind=kind, count=count)


def _round(sim, tier, base, w, m, kind, count, *, si=None, sd=None, pa=None, so=None) -> Run:
    vlog = [("WEXP", w), ("WMAN", m)]
    suffix = ""
    if si is not None:
        vlog.append(("STAGE_INPUT", si))
        suffix += f"_si{si}"
    if sd is not None:
        vlog.append(("STAGE_DECODE", sd))
        suffix += f"_sd{sd}"
    if pa is not None:
        vlog.append(("STAGE_PACK", pa))
        suffix += f"_pa{pa}"
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    return _run("round", sim, tier, f"{base}{suffix}", vlog, kind=kind, count=count)


def _pipe(sim, tier, config, w, n, count) -> Run:
    return _run(
        "pipe",
        sim,
        tier,
        config,
        [("W", w), ("N", n)],
        count=count,
        with_kind=False,
        plus_names={"W": "ZKF_PIPE_W", "N": "ZKF_PIPE_N"},
    )


def _trans(
    module,
    sim,
    tier,
    base,
    w,
    m,
    kind,
    count,
    *,
    si=None,
    sr=None,
    sd=None,
    sp=None,
    spf=None,
    sn=None,
    sno=None,
    pa=None,
    so=None,
    un=None,
    parallel=None,
    wm=None,
) -> Run:
    # Each module takes only its own knobs (the driver passes only a module.s own declared parameters). Knob legend: si/sp/so =
    # STAGE_INPUT/PRODUCT/OUTPUT, wm = WMULTIPLIER (_zkf_pmul DSP-tile-grid hint), sr = STAGE_REDUCE (exp2),
    # sd/spf/sno = STAGE_DECODE/PRODUCT_FINAL/NORMALIZE_OUTPUT (log2), un = UNROLL100, parallel = PARALLEL
    # (decoupled z-path), sn = STAGE_NORMALIZE, pa = STAGE_PACK.
    vlog = [("WEXP", w), ("WMAN", m)]
    suffix = ""
    if un is not None:
        vlog.append(("UNROLL100", un))
        suffix += f"_un{un}"
    if parallel is not None:
        vlog.append(("PARALLEL", parallel))
        suffix += f"_par{parallel}"
    if si is not None:
        vlog.append(("STAGE_INPUT", si))
        suffix += f"_si{si}"
    if module == "exp2" and sr is not None:
        vlog.append(("STAGE_REDUCE", sr))
        suffix += f"_sr{sr}"
    if module == "log2" and sd is not None:
        vlog.append(("STAGE_DECODE", sd))
        suffix += f"_sd{sd}"
    if sp is not None:
        vlog.append(("STAGE_PRODUCT", sp))
        suffix += f"_sp{sp}"
    if module == "log2":
        spf_eff = sp if spf is None else spf
        if spf_eff is not None:
            vlog.append(("STAGE_PRODUCT_FINAL", spf_eff))
            suffix += f"_spf{spf_eff}" if spf is not None else ""
    if wm is not None:
        vlog.append(("WMULTIPLIER", wm))
        suffix += f"_wm{wm}"
    if sn is not None:
        vlog.append(("STAGE_NORMALIZE", sn))
        suffix += f"_sn{sn}"
    if module == "log2" and sno is not None:
        vlog.append(("STAGE_NORMALIZE_OUTPUT", sno))
        suffix += f"_sno{sno}"
    if pa is not None:
        vlog.append(("STAGE_PACK", pa))
        suffix += f"_pa{pa}"
    if so is not None:
        vlog.append(("STAGE_OUTPUT", so))
        suffix += f"_so{so}"
    return _run(module, sim, tier, base + suffix, vlog, kind=kind, count=count)


def _per_pr(sim, out: list) -> None:
    for cfg, w, m, u, k, c in PACK:
        out.append(_pack(sim, "pr", cfg, w, m, u, k, c))
    for op in ("cmp", "sort"):
        for cfg, w, m, k, c in BINARY:
            out.append(_binary(op, sim, "pr", cfg, w, m, k, c))
        out.append(_binary(op, sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1))
    for op in ("add", "addsub"):
        for sd in (0, 1):
            for sa in (0, 1):
                for cfg, w, m, k, c in BINARY:
                    out.append(_binary(op, sim, "pr", cfg, w, m, k, c, sd=sd, sa=sa))
        # si/pa knobs plus the all-on maxpipe row guard the latency bookkeeping against a register-stage change.
        out.append(_binary(op, sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1))
        out.append(_binary(op, sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, pa=1))
        out.append(_binary(op, sim, "pr", "w3_m4_maxpipe", 3, 4, "exhaustive", 0, sd=1, sa=1, sn=1, pa=1, si=2, so=1))
        # STAGE_INPUT>1 (dummy input stages) exercises the counted-latency bookkeeping and the multi-stage input pipe.
        out.append(_binary(op, sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=3))
        out.append(_binary(op, sim, "pr", "w8_m18", 8, 18, "random", 256, si=2))
        # STAGE_NORMALIZE forwards to _zkf_normshift.STAGE_SPLIT (close-cancel path); SN=2 adds an s2x catch-up cycle
        # and needs NL4 >= 3, i.e. NINPUT = WMAN+3 >= 11 -> WMAN >= 8 (hence the w8_m18 sn2 row).
        for sn in (1,):
            out.append(_binary(op, sim, "pr", "w4_m6_sn", 4, 6, "random", 256, sn=sn))
        out.append(_binary(op, sim, "pr", "w8_m18_sn2", 8, 18, "random", 256, sd=1, sa=1, sn=2))
    for sp in (0, 1):
        for si in (0, 1):
            for cfg, w, m, k, c in BINARY:
                out.append(_binary("mul", sim, "pr", cfg, w, m, k, c, sp=sp, si=si))
    # STAGE_PACK forwards to _zkf_pack.STAGE_INPUT.
    out.append(_binary("mul", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, pa=1))
    out.append(_binary("mul", sim, "pr", "w3_m4_maxpipe", 3, 4, "exhaustive", 0, sp=1, si=1, pa=1, so=1))
    # STAGE_PRODUCT 2/3 forward to _zkf_pmul's 2x2 / 3x3 split grids (bit-exact; checks latency bookkeeping).
    out.append(_binary("mul", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, sp=2))
    out.append(_binary("mul", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, sp=3))
    for si in (0, 1):
        for cfg, w, m, k, c in BINARY:
            out.append(_binary("div", sim, "pr", cfg, w, m, k, c, si=si))
    out.append(_binary("div", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, pa=1))
    out.append(_binary("div", sim, "pr", "w3_m4_maxpipe", 3, 4, "exhaustive", 0, si=1, pa=1, so=1))
    for cfg, w, m, k, c in FMA:
        out.append(_fma(sim, "pr", cfg, w, m, k, c))
    # Each pipeline knob once (plus all-on) on a fast format; results are staging-independent, so this checks the
    # out_valid timing of every STAGE_* register without re-running the slow formats.
    for si, sp, sd, sa, sn, pa, so in [
        (0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1),
    ]:
        out.append(_fma(sim, "pr", "w4_m6_stage", 4, 6, "random", 256, sp=sp, si=si, sd=sd, sa=sa, sn=sn, pa=pa, so=so))
    # STAGE_PRODUCT 2/3 forward to _zkf_pmul's 2x2 / 3x3 split grids (bit-exact; checks latency bookkeeping).
    out.append(_fma(sim, "pr", "w4_m6_stage", 4, 6, "random", 256, sp=2))
    out.append(_fma(sim, "pr", "w4_m6_stage", 4, 6, "random", 256, sp=3))
    # STAGE_NORMALIZE=2 (FMA-local 3-segment normalizer) needs NL4 >= 3, i.e. WMAN >= 7 (smaller is rejected at
    # elaboration), so it cannot use the w4/m6 format above: run the WMAN=7 guard edge and a wider WMAN=18.
    out.append(_fma(sim, "pr", "w4m7_sn2", 4, 7, "random", 384, sp=1, sd=1, sa=1, sn=2))
    out.append(_fma(sim, "pr", "w6m18_sn2", 6, 18, "random", 384, sp=1, sd=1, sa=1, sn=2))
    # sn=2 with so=1 is otherwise untested (other sn=2 rows have so=0): all-on guards the 3-segment normalizer's
    # payload realignment feeding the registered packer output.
    out.append(_fma(sim, "pr", "w6m18_maxpipe", 6, 18, "random", 384, sp=1, si=1, sd=1, sa=1, sn=2, so=1))
    # Shipped narrow synth config: si=1 + sa=1 + sn=2 (closes the W6/M18 cones on Yosys and Diamond/LSE with one
    # MULT18X18D; sp=1 would split the 18x18 into a 2x2 grid = 4 DSPs for no timing gain). Gated so it is tested
    # directly, not just inferred from the per-knob sweeps.
    out.append(_fma(sim, "pr", "w6m18_si1_sa1_sn2", 6, 18, "random", 384, si=1, sa=1, sn=2))
    for op in ("abs", "neg", "is_finite", "saturate"):
        for cfg, w, m, k, c in UNARY:
            out.append(_binary(op, sim, "pr", cfg, w, m, k, c))
    for op in ("exp2", "log2"):
        for cfg, w, m, k, c in TRANS_EXPLOG:
            out.append(_trans(op, sim, "pr", cfg, w, m, k, c))
    for cfg, w, m, k, c in TRANS_TRIG:
        for op in ("sincos",):
            out.append(_trans(op, sim, "pr", cfg, w, m, k, c))
    for op in ("exp2", "log2"):
        # STAGE_INPUT/PRODUCT/OUTPUT timing on the cheapest exhaustive format; results are staging-independent, so
        # these only exercise the register-stage bookkeeping. sincos's own knobs (UNROLL100/NORMALIZE/PACK) are below.
        out.append(_trans(op, sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, si=1))
        out.append(_trans(op, sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, so=1))
        out.append(_trans(op, sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, sp=1))
        out.append(_trans(op, sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, sp=2))
        out.append(_trans(op, sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, sp=3))
        out.append(_trans(op, sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, si=1, sp=1, so=1))
    out.append(_trans("exp2", sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, sr=1))
    out.append(_trans("log2", sim, "pr", "w6_m16_decode", 6, 16, "random", 256, sd=1))
    out.append(_trans("log2", sim, "pr", "w6_m16_split_final", 6, 16, "random", 256, sp=1, spf=2))
    # UNROLL100 (iterations/cycle x100): 50 = half-rate, 100 = synthesized M18 rate, 200 = 2/cycle. Each changes the
    # published latency; the test asserts measured == model.
    out.append(_trans("sincos", sim, "pr", "w5_m16_unroll", 5, 16, "random", 256, un=50))
    out.append(_trans("sincos", sim, "pr", "w5_m16_unroll", 5, 16, "random", 256, un=200))
    # sincos staging knobs, bit-transparent vs the unstaged path (the test checks bit-exactness + latency). si/so are
    # the standard sequential register stages; the decode and wide-datapath stages are always-on.
    out.append(_trans("sincos", sim, "pr", "w5_m16_stage", 5, 16, "random", 256, si=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_stage", 5, 16, "random", 256, so=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_stage", 5, 16, "random", 256, si=1, so=1))
    # STAGE_PRODUCT: shared _zkf_pmul depth (1 = native, 2 = 2x2, 3 = 3x3). Bit-transparent; each adds
    # 2*STAGE_PRODUCT cycles.
    out.append(_trans("sincos", sim, "pr", "w5_m16_prod", 5, 16, "random", 256, sp=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_prod", 5, 16, "random", 256, sp=2))
    out.append(_trans("sincos", sim, "pr", "w5_m16_prod", 5, 16, "random", 256, sp=3))
    out.append(_trans("sincos", sim, "pr", "w5_m16_un50_prod", 5, 16, "random", 256, un=50, parallel=0, sp=2))
    # PARALLEL decouples the z-recurrence (runs at full rate ahead of the half-rate x/y rotator, issues PHI early):
    # bit-identical to lock-step, lower latency. Only legal half-rate (full-rate + PARALLEL is rejected at
    # elaboration), so sweep un=50 with PARALLEL=1/0. The test asserts bit-exactness + latency.
    out.append(_trans("sincos", sim, "pr", "w5_m16_dec", 5, 16, "random", 256, un=50, parallel=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_dec", 5, 16, "random", 256, un=50, parallel=1, sp=2))
    out.append(_trans("sincos", sim, "pr", "w5_m16_dec", 5, 16, "random", 256, un=50, parallel=1, sp=3))
    out.append(_trans("sincos", sim, "pr", "w5_m16_dec", 5, 16, "random", 256, un=50, parallel=0, sp=3))
    # STAGE_NORMALIZE (log2/sincos) drives the normalizer's STAGE_SPLIT.
    out.append(_trans("log2", sim, "pr", "w6_m16_sncheck", 6, 16, "random", 256, sn=1))
    out.append(_trans("log2", sim, "pr", "w6_m16_sncheck", 6, 16, "random", 256, sp=1, sn=1))
    out.append(_trans("log2", sim, "pr", "w8_m24_sncheck", 8, 24, "random", 256, sn=2, pa=1))
    # log2 STAGE_NORMALIZE_OUTPUT: registered _zkf_normshift output + pole/domain sideband alignment; no other row
    # drives sno.
    out.append(_trans("log2", sim, "pr", "w6_m16_sno", 6, 16, "random", 256, sn=1, sno=1, pa=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_sncheck", 5, 16, "random", 256, sn=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_sncheck", 5, 16, "random", 256, sn=2))
    # STAGE_PACK forwards to _zkf_pack.STAGE_INPUT (exp2/log2/sincos), standalone and combined with other knobs.
    out.append(_trans("exp2", sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, pa=1))
    out.append(_trans("log2", sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, pa=1))
    out.append(_trans("sincos", sim, "pr", "w2_m16_exhaustive", 2, 16, "exhaustive", 0, pa=1))
    out.append(_trans("log2", sim, "pr", "w6_m16_sncheck", 6, 16, "random", 256, sn=1, pa=1))
    out.append(_trans("sincos", sim, "pr", "w5_m16_sncheck", 5, 16, "random", 256, sn=1, pa=1))
    # Shipped small log2 synth preset (6/18: sn=1 + STAGE_PRODUCT_FINAL=1): pins the latency and pole/domain-error
    # sideband alignment under the exact shipped knobs.
    out.append(_trans("log2", sim, "pr", "w6_m18_synth", 6, 18, "random", 512, sn=1, spf=1))
    out.append(_trans("log2", sim, "pr", "w6_m18_synth_so1", 6, 18, "random", 512, sn=2, spf=1, so=1))
    # zkf_atan2 (two-input vectoring CORDIC): directed pair table + random, then the shared knob sweeps. Each row
    # checks bit-exactness + latency; directed alone hits every special/axis/diagonal/bypass-boundary pair.
    for cfg, w, m, k, c in TRANS_ATAN2:
        out.append(_trans("atan2", sim, "pr", cfg, w, m, k, c))
    # Tiny WEXP (2, 3): random covers the narrow exponent-difference path plus the axis/diagonal turn constants that
    # underflow the normal range at small BIAS.
    out.append(_trans("atan2", sim, "pr", "w2_m16_random", 2, 16, "random", 512))
    out.append(_trans("atan2", sim, "pr", "w3_m16_random", 3, 16, "random", 512))
    out.append(_trans("atan2", sim, "pr", "w6_m18_directed", 6, 18, "directed", 0))
    out.append(_trans("atan2", sim, "pr", "w5_m16_unroll", 5, 16, "random", 256, un=50))
    out.append(_trans("atan2", sim, "pr", "w5_m16_unroll", 5, 16, "random", 256, un=200))
    out.append(_trans("atan2", sim, "pr", "w5_m16_unroll", 5, 16, "random", 256, un=400))
    out.append(_trans("atan2", sim, "pr", "w5_m16_stage", 5, 16, "random", 256, si=1))
    out.append(_trans("atan2", sim, "pr", "w5_m16_stage", 5, 16, "random", 256, so=1))
    out.append(_trans("atan2", sim, "pr", "w5_m16_stage", 5, 16, "random", 256, si=1, so=1))
    out.append(_trans("atan2", sim, "pr", "w5_m16_norm", 5, 16, "random", 256, sn=1))
    out.append(_trans("atan2", sim, "pr", "w5_m16_norm", 5, 16, "random", 256, sn=2))
    out.append(_trans("atan2", sim, "pr", "w5_m16_pack", 5, 16, "random", 256, pa=1))
    out.append(_trans("atan2", sim, "pr", "w5_m16_full", 5, 16, "random", 256, si=1, sn=2, pa=1, so=1))
    # STAGE_PRODUCT / WMULTIPLIER: shared _zkf_pmul (magnitude x_K*KINV, residual/bypass Q*INV_TAU); bit-transparent,
    # each adds STAGE_PRODUCT cycles. Native (sp=1), 2x2 (sp=2), plus the synthesized 6/18 operating point.
    out.append(_trans("atan2", sim, "pr", "w5_m16_prod", 5, 16, "random", 256, sp=1))
    out.append(_trans("atan2", sim, "pr", "w5_m16_prod", 5, 16, "random", 256, sp=2, wm=16))
    out.append(_trans("atan2", sim, "pr", "w6_m18_synth", 6, 18, "random", 256, un=50, sp=2, wm=18, sn=2, pa=1))
    # Shipped zkf_atan2_w8m36 synth config tested directly (correctness + data-independent latency, not just
    # inferred from the knob sweeps).
    out.append(_trans("atan2", sim, "pr", "w8_m36_synth", 8, 36, "random", 256, un=50, sp=4, wm=18, sn=2, pa=1, so=1))
    for sd in (0, 1):
        for cfg, w, m, k, c in UNARY:
            out.append(_binary("mul_ilog2_const", sim, "pr", cfg, w, m, k, c, sd=sd))
    out.append(_binary("mul_ilog2_const", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1))
    out.append(_binary("mul_ilog2_const", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1, sd=1))
    # zkf_mul_ilog2 (runtime k): same format sweep, both decode depths. Default WK=WEXP+1 already covers shifts that
    # overflow to inf / underflow to zero for every input class.
    for sd in (0, 1):
        for cfg, w, m, k, c in UNARY:
            out.append(_ilog2(sim, "pr", cfg, w, m, k, c, sd=sd))
    # STAGE_INPUT (incl. >1) and non-default WK: narrow (k cannot leave the normal range) and wide (large |k| hits
    # the saturating boundaries).
    out.append(_ilog2(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1))
    out.append(_ilog2(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1, sd=1))
    out.append(_ilog2(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=2))
    out.append(_ilog2(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, wk=2))
    out.append(_ilog2(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, wk=6, sd=1))
    for si in (0, 1):
        for cfg, w, m, wint, k, c in FROM_INT:
            out.append(_cast("from_int", sim, "pr", cfg, w, m, wint, k, c, si))
        for cfg, w, m, wint, k, c in TO_INT:
            out.append(_cast("to_int", sim, "pr", cfg, w, m, wint, k, c, si))
        for cfg, wi, mi, wo, mo, k, c in RESIZE:
            out.append(_resize(sim, "pr", cfg, wi, mi, wo, mo, k, c, si))
    # zkf_from_int knobs: STAGE_NORMALIZE -> _zkf_normshift.STAGE_SPLIT, STAGE_PACK -> _zkf_pack.STAGE_INPUT.
    out.append(_cast("from_int", sim, "pr", "w3_m4_int8_exhaustive", 3, 4, 8, "exhaustive", 0, si=0, sn=1))
    out.append(_cast("from_int", sim, "pr", "w3_m4_int8_exhaustive", 3, 4, 8, "exhaustive", 0, si=0, pa=1))
    out.append(_cast("from_int", sim, "pr", "w3_m4_int8_exhaustive", 3, 4, 8, "exhaustive", 0, si=1, sn=1, pa=1))
    # zkf_round: the bench sweeps every operand across all four rounding modes; UNARY covers the formats (w2_m4
    # reaches the round-up-overflows-to-inf corner). Stage knobs: STAGE_INPUT via zkf_pipe, STAGE_PACK/OUTPUT ->
    # _zkf_pack; exercised once each plus all-on to catch latency-bookkeeping regressions.
    for cfg, w, m, k, c in UNARY:
        out.append(_round(sim, "pr", cfg, w, m, k, c))
    out.append(_round(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=1))
    out.append(_round(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, sd=1))
    out.append(_round(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, pa=1))
    out.append(_round(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, so=1))
    out.append(_round(sim, "pr", "w3_m4_maxpipe", 3, 4, "exhaustive", 0, si=1, sd=1, pa=1, so=1))
    out.append(_binary("add", sim, "pr", "w6_m100_directed", 6, 100, "directed", 0))
    for cfg, w, n, c in PIPE:
        out.append(_pipe(sim, "pr", cfg, w, n, c))
    # STAGE_INPUT>1 across the generalized public modules: si=2 per module (+ si=3 on mul) checks widened input
    # pipes in the latency model. sincos/atan2 excluded (handshake-entangled input stage).
    for op in ("mul", "div", "cmp", "sort"):
        out.append(_binary(op, sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=2))
    out.append(_binary("mul", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=3))
    out.append(_binary("mul_ilog2_const", sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=2))
    out.append(_fma(sim, "pr", "w4_m6", 4, 6, "random", 256, si=2))
    out.append(_round(sim, "pr", "w3_m4", 3, 4, "exhaustive", 0, si=2))
    out.append(_cast("from_int", sim, "pr", "w3_m4_int8", 3, 4, 8, "exhaustive", 0, 2))
    out.append(_cast("to_int", sim, "pr", "w3_m4_int8", 3, 4, 8, "exhaustive", 0, 2))
    out.append(_resize(sim, "pr", "w3m4_to_w4m6", 3, 4, 4, 6, "exhaustive", 0, 2))
    for op in ("exp2", "log2"):
        out.append(_trans(op, sim, "pr", "w2_m16", 2, 16, "exhaustive", 0, si=2))


def _deep_correctness(out: list) -> None:
    # Full Cartesian product of each module's structural knobs across every format in its deep list (correctness;
    # coverage closure lives in _deep_coverage under merged-union). Knob axes: mul = STAGE_INPUT x PRODUCT x OUTPUT;
    # add/addsub = STAGE_DECODE x ALIGN x OUTPUT; div/from_int/resize = STAGE_INPUT x OUTPUT; to_int = STAGE_INPUT
    # only; pack = STAGE_OUTPUT x EXP_IS_BIASED; mul_ilog2_const = STAGE_DECODE x K.
    s = "icarus"
    for w, m, k, c in BIN_EXT:
        base = f"w{w}m{m}_{k}"
        for sp in (0, 1, 2, 3):  # _zkf_pmul depth: single / capture+native / 2x2 / 3x3
            for si in (0, 1):
                for so in (0, 1):
                    out.append(_binary("mul", s, "deep", base, w, m, k, c, sp=sp, si=si, so=so))
        for op in ("add", "addsub"):
            for sd in (0, 1):
                for sa in (0, 1):
                    for so in (0, 1):
                        out.append(_binary(op, s, "deep", base, w, m, k, c, sd=sd, sa=sa, so=so))
        out.append(_binary("cmp", s, "deep", base, w, m, k, c))
        out.append(_binary("sort", s, "deep", base, w, m, k, c))
    # fma: each deep format once (results are staging-independent), the full pipeline-knob cartesian on one fast
    # format, the WEXP=8/WMAN=36 synth config, and the smallest format at the staging extremes.
    for w, m, k, c in FMA_EXT:
        out.append(_fma(s, "deep", f"w{w}m{m}_{k}", w, m, k, c))
    for sp in (0, 1):
        for si in (0, 1):
            for sd in (0, 1):
                for sa in (0, 1):
                    for sn in (0, 1):
                        for so in (0, 1):
                            out.append(
                                _fma(
                                    s,
                                    "deep",
                                    "w4m6_knobs",
                                    4,
                                    6,
                                    "random",
                                    256,
                                    sp=sp,
                                    si=si,
                                    sd=sd,
                                    sa=sa,
                                    sn=sn,
                                    so=so,
                                )
                            )
    out.append(_fma(s, "deep", "w8m36", 8, 36, "random", 768, sp=1, sd=1, sa=1, sn=2))
    out.append(_fma(s, "deep", "w8m36_si1", 8, 36, "random", 768, sp=1, si=1, sd=1, sa=1, sn=2))
    # STAGE_PRODUCT 2/3 (2x2/3x3) on wide WMAN=36 with WMULTIPLIER=18 so _zkf_pmul derives the 18-bit DSP-tile grid.
    # sp=2 is what zkf_fma_w8m36 ships; sp=3 also exercises the 3x3 grid end to end.
    out.append(_fma(s, "deep", "w8m36", 8, 36, "random", 768, sp=2, wm=18, sd=1, sa=1, sn=2))
    out.append(_fma(s, "deep", "w8m36", 8, 36, "random", 768, sp=3, wm=18, sd=1, sa=1, sn=2))
    for si, sp, sd, sa, sn, so in ((0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)):
        out.append(_fma(s, "deep", "w2m4_exhaustive", 2, 4, "exhaustive", 0, sp=sp, si=si, sd=sd, sa=sa, sn=sn, so=so))
    for w, m, k, c in DIV_EXT:
        for si in (0, 1):
            for so in (0, 1):
                out.append(_binary("div", s, "deep", f"w{w}m{m}_{k}", w, m, k, c, si=si, so=so))
    # div at WMAN=48: no transcendental tables, so a wide format is cheap and otherwise untested.
    out.append(_binary("div", s, "deep", "w8m48_random", 8, 48, "random", 384, si=0, so=0))
    for w, m, k, c in UNARY_EXT:
        base = f"w{w}m{m}_{k}"
        for op in ("abs", "neg", "is_finite", "saturate"):
            out.append(_binary(op, s, "deep", base, w, m, k, c))
        for sd in (0, 1):
            out.append(_binary("mul_ilog2_const", s, "deep", base, w, m, k, c, sd=sd))
            out.append(_ilog2(s, "deep", base, w, m, k, c, sd=sd))
    # exp2/log2 staging (one knob at a time off the baseline) across the deep transcendental formats.
    for w, m, k, c in TRANS_EXPLOG_EXT:
        base = f"w{w}m{m}_{k}"
        for op in ("exp2", "log2"):
            for si, sp, so in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
                out.append(_trans(op, s, "deep", base, w, m, k, c, si=si, sp=sp, so=so))
        out.append(_trans("exp2", s, "deep", base, w, m, k, c, sr=1))
        out.append(_trans("log2", s, "deep", base + "_decode", w, m, k, c, sd=1))
        out.append(_trans("log2", s, "deep", base + "_split_final", w, m, k, c, sp=1, spf=2))

    # sincos: STAGE_PRODUCT (_zkf_pmul split), UNROLL100, STAGE_NORMALIZE/PACK; the testbench asserts latency.
    for w, m, k, c in TRANS_TRIG_EXT:
        base = f"w{w}m{m}_{k}"
        for un in (50, 100, 200, 400):
            out.append(_trans("sincos", s, "deep", base, w, m, k, c, un=un))
        for sp in (1, 2, 3):
            out.append(_trans("sincos", s, "deep", base, w, m, k, c, sp=sp))
        out.append(_trans("sincos", s, "deep", base, w, m, k, c, si=1, so=1, sn=1, pa=1))
        # Synthesized wide profile: half-rate + 3x3 split pinned to WMULTIPLIER=18 (as zkf_sincos_w8m36 ships),
        # bit-transparent, across the deep wide formats; focus is the wide multiplier grid.
        out.append(_trans("sincos", s, "deep", base, w, m, k, c, un=50, sp=3, wm=18))
    # Exact synthesized WEXP=8/WMAN=36 operating points with WMULTIPLIER=18 (the 18-bit DSP-tile grid the Diamond/LSE
    # flow needs). WMULTIPLIER is bit-transparent, but pinning the shipped grid exercises the full datapath at the
    # operating point synthesis actually builds, not just the symmetric default.
    out.append(_binary("mul", s, "deep", "w8m36", 8, 36, "random", 512, sp=2, wm=18, pa=1))
    out.append(_trans("exp2", s, "deep", "w8m36", 8, 36, "random", 512, si=1, sp=3, wm=18, so=1))
    out.append(_trans("log2", s, "deep", "w8m36", 8, 36, "random", 512, si=1, sp=3, spf=3, wm=18, sn=2, pa=1))
    # zkf_atan2 deep: baseline per format, UNROLL100 sweep + full staging on 5/16, and the synthesized 6/18 + 8/36
    # operating points. Each asserts latency.
    for cfg, w, m, k, c in TRANS_ATAN2:
        out.append(_trans("atan2", s, "deep", f"atan2_{cfg}", w, m, k, c))
    for un in (50, 100, 200, 400):
        out.append(_trans("atan2", s, "deep", "atan2_w5m16_un", 5, 16, "random", 512, un=un))
    out.append(_trans("atan2", s, "deep", "atan2_w5m16_stage", 5, 16, "random", 512, si=1, so=1, sn=2, pa=1))
    out.append(_trans("atan2", s, "deep", "atan2_w6m18_op", 6, 18, "random", 512, un=50, sp=2, wm=18, sn=2, pa=1))
    out.append(
        _trans("atan2", s, "deep", "atan2_w8m36_op", 8, 36, "random", 512, un=50, si=0, sp=4, wm=18, sn=2, pa=1, so=1)
    )
    # pack: STAGE_OUTPUT x EXP_IS_BIASED. EXP_IS_BIASED=1 is exhaustive-only (test_pack iterates the biased field);
    # random formats stay EXP_IS_BIASED=0 (also exercised transitively via add/from_int).
    for w, m, u, k, c in [
        (2, 5, 3, "exhaustive", 0),
        (2, 5, 5, "exhaustive", 0),
        (3, 5, 5, "exhaustive", 0),
        (4, 5, 8, "random", 768),
        (6, 17, 10, "random", 1024),
        (4, 4, 8, "random", 512),
    ]:
        base = f"w{w}m{m}u{u}_{k}"
        for so in (0, 1):
            out.append(_pack(s, "deep", base, w, m, u, k, c, so=so))
            if k == "exhaustive":
                out.append(_pack(s, "deep", base, w, m, u, k, c, so=so, eb=1))
    for w, m, i, k, c in [
        (3, 5, 5, "exhaustive", 0),
        (2, 5, 3, "exhaustive", 0),
        (4, 5, 7, "exhaustive", 0),
        (4, 6, 5, "exhaustive", 0),
        (5, 11, 9, "random", 512),
        (6, 17, 33, "random", 512),
        (8, 24, 17, "random", 512),
    ]:
        base = f"w{w}m{m}i{i}_{k}"
        for si in (0, 1):
            out.append(_cast("to_int", s, "deep", base, w, m, i, k, c, si))
            for so in (0, 1):
                out.append(_cast("from_int", s, "deep", base, w, m, i, k, c, si, so=so))
    # from_int at the widest WMAN and sn=2: its WX/WEU sizing, carry-to-inf wiring, and the sn=2 normalize-shift
    # split are otherwise unexercised (the loop above tops out at WMAN=24, sn<=1).
    out.append(_cast("from_int", s, "deep", "w11m53i32", 11, 53, 32, "random", 384, 0))
    out.append(_cast("from_int", s, "deep", "w6m18i32_sn2", 6, 18, 32, "random", 256, 0, sn=2))
    for wi, mi, wo, mo, k, c in [
        (4, 5, 4, 4, "exhaustive", 0),
        (4, 4, 4, 5, "exhaustive", 0),
        (2, 5, 4, 7, "exhaustive", 0),
        (4, 7, 2, 5, "exhaustive", 0),
        (5, 5, 3, 4, "exhaustive", 0),
        (3, 4, 5, 5, "exhaustive", 0),
        (6, 17, 4, 11, "random", 512),
        (5, 11, 6, 17, "random", 512),
    ]:
        base = f"w{wi}m{mi}_to_w{wo}m{mo}_{k}"
        for si in (0, 1):
            for so in (0, 1):
                out.append(_resize(s, "deep", base, wi, mi, wo, mo, k, c, si, so=so))
    # resize at wide WMAN=48 (narrow then widen): the WMAN-shrink GRS rounding and WMAN-grow zero-fill, otherwise
    # untested.
    out.append(_resize(s, "deep", "w8m48_to_w8m24", 8, 48, 8, 24, "random", 384, 0))
    out.append(_resize(s, "deep", "w8m24_to_w8m48", 8, 24, 8, 48, "random", 384, 0))
    # round: each unary deep format once, the full si x pa x so knob cartesian on a small exhaustive format, and the
    # WEXP=8/WMAN=36 wide format.
    for w, m, k, c in UNARY_EXT:
        out.append(_round(s, "deep", f"w{w}m{m}_{k}", w, m, k, c))
    for si in (0, 1):
        for sd in (0, 1):
            for pa in (0, 1):
                for so in (0, 1):
                    out.append(_round(s, "deep", "w3m6_knobs", 3, 6, "exhaustive", 0, si=si, sd=sd, pa=pa, so=so))
    out.append(_round(s, "deep", "w8m36", 8, 36, "random", 768))
    out.append(_round(s, "deep", "w8m36_maxpipe", 8, 36, "random", 768, si=1, sd=1, pa=1, so=1))
    out.append(_round(s, "deep", "w8m48", 8, 48, "random", 384))


def _deep_coverage(out: list) -> None:
    s = "verilator"
    for w, m in [(4, 5), (3, 6), (5, 4), (3, 5), (2, 6)]:
        base = f"w{w}m{m}"
        for sp in (0, 1):
            out.append(_binary("mul", s, "deep", base, w, m, "exhaustive", 0, sp=sp))
        out.append(_binary("add", s, "deep", base, w, m, "exhaustive", 0, sd=0, sa=0))
        out.append(_binary("add", s, "deep", base, w, m, "exhaustive", 0, sd=1, sa=1))
        out.append(_binary("add", s, "deep", base, w, m, "exhaustive", 0, si=2))
        out.append(_binary("addsub", s, "deep", base, w, m, "exhaustive", 0, sd=1, sa=1))
        out.append(_binary("addsub", s, "deep", base, w, m, "exhaustive", 0, si=2))
        out.append(_binary("cmp", s, "deep", base, w, m, "exhaustive", 0))
        out.append(_binary("sort", s, "deep", base, w, m, "exhaustive", 0))
    # fma coverage: W2/M4 exhaustive (the only feasible ternary-exhaustive) at default + all-on toggles the
    # product/decode/align/normalize/output split registers; wider random runs toggle the wide shifters and the
    # far-shift saturation path the tiny W2/M4 exponent range cannot reach.
    out.append(_fma(s, "deep", "w2m4", 2, 4, "exhaustive", 0, sp=0, sd=0, sa=0, sn=0, so=0))
    out.append(_fma(s, "deep", "w2m4", 2, 4, "exhaustive", 0, sp=1, sd=1, sa=1, sn=1, so=1))
    out.append(_fma(s, "deep", "w3m4", 3, 4, "random", 4096, sp=1, sd=1, sa=1, sn=1, so=1))
    out.append(_fma(s, "deep", "w6m18", 6, 18, "random", 1024, sp=1, sd=1, sa=1, sn=1, so=0))
    out.append(_fma(s, "deep", "w6m18_sn2", 6, 18, "random", 1024, sp=1, sd=1, sa=1, sn=2, so=0))
    out.append(_fma(s, "deep", "w8m36", 8, 36, "random", 1024, sp=1, sd=1, sa=1, sn=2, so=0))
    for w, m in [(4, 5), (3, 6), (3, 5), (2, 6)]:
        for si in (0, 1):
            out.append(_binary("div", s, "deep", f"w{w}m{m}", w, m, "exhaustive", 0, si=si))
    for w, m in [(4, 5), (3, 6), (2, 6)]:
        base = f"w{w}m{m}"
        for op in ("abs", "neg", "is_finite", "saturate"):
            out.append(_binary(op, s, "deep", base, w, m, "exhaustive", 0))
        for sd in (0, 1):
            out.append(_binary("mul_ilog2_const", s, "deep", base, w, m, "exhaustive", 0, sd=sd))
            out.append(_ilog2(s, "deep", base, w, m, "exhaustive", 0, sd=sd))
    # exp2/log2 coverage: WMAN=16 exhaustive toggles the ROM/Horner; so=1 covers the registered pack output;
    # sp=2/3/4 toggle the shared _zkf_pmul split-product paths; split-final exercises log2's final f*C(f) multiply.
    for w, m in [(2, 16), (3, 16)]:
        for op in ("exp2", "log2"):
            out.append(_trans(op, s, "deep", f"w{w}m{m}", w, m, "exhaustive", 0))
    out.append(_trans("exp2", s, "deep", "w2m16", 2, 16, "exhaustive", 0, so=1))
    out.append(_trans("exp2", s, "deep", "w2m16", 2, 16, "exhaustive", 0, sr=1))
    out.append(_trans("log2", s, "deep", "w2m16", 2, 16, "exhaustive", 0, so=1))
    for sp in (2, 3, 4):
        out.append(_trans("exp2", s, "deep", "w3m16", 3, 16, "exhaustive", 0, sp=sp))
        out.append(_trans("log2", s, "deep", "w3m16", 3, 16, "exhaustive", 0, sp=sp))
    out.append(_trans("log2", s, "deep", "w3m16_split_final", 3, 16, "exhaustive", 0, sp=2, spf=3))
    # Wide-format split-product coverage (bona fide): w3m16 sp=2/3/4 instruments _zkf_pmul's g_flat/g_rows/g_rows2
    # trees but leaves the high accumulator bits (csum/r_p/rowc/s_row/s_col MSB) dark; w8m36 fills the full WP.
    # exp2 drives the unsigned grids, log2 the signed grids. Also widens the significand so the hidden-bit MSB toggles.
    for sp in (2, 3, 4):
        out.append(_trans("exp2", s, "deep", "w8m36_grid", 8, 36, "random", 512, sp=sp, wm=18))
    for sp in (3, 4):
        out.append(_trans("log2", s, "deep", "w8m36_grid", 8, 36, "random", 512, sp=sp, wm=18))
    # sincos WMAN=16 coverage via the CORDIC table family (WMAN=11 dropped); w5_m16 also reaches the tiny-input
    # bypass (e <= -(GUARD_FF+2), needs a wide enough exponent field). w2_m16 stays exhaustive (2**18 codes); the
    # wider-WEXP rows are random because the total code space 2**(WEXP+WMAN) grows past exhaustive reach.
    out.append(_trans("sincos", s, "deep", "w2m16", 2, 16, "exhaustive", 0))
    out.append(_trans("sincos", s, "deep", "w3m16", 3, 16, "random", 4000))
    # sincos: w5_m16 (random) reaches the bypass path; un=200 and sn=1/pa=1 cover UNROLL100 and the
    # normshift-barrier / pack-register toggles.
    out.append(_trans("sincos", s, "deep", "w5m16", 5, 16, "random", 4000))
    out.append(_trans("sincos", s, "deep", "w5m16", 5, 16, "random", 4000, un=200))
    out.append(_trans("sincos", s, "deep", "w5m16", 5, 16, "random", 4000, sn=1, pa=1))
    # Lock-step (un=50) and decoupled (PARALLEL=1) sincos exercise both CORDIC handoff modes (coupled g_zadv +
    # half-rate sigma-replay); each mode's branches (and the phi_seen if/else legs) are reachable only in its own row.
    # The structurally-dead P_PHI implicit-else and FSM default arm are coverage_off in zkf_sincos.v.
    out.append(_trans("sincos", s, "deep", "w5m16", 5, 16, "random", 4000, un=50))
    out.append(_trans("sincos", s, "deep", "w5m16_par1", 5, 16, "random", 4000, un=50, parallel=1))
    # Wide-format sincos: at the small format some CORDIC X/Y-carry / local-mag / local-exp high bits sit above the
    # format ceiling; w8m24 (table _zkf_cordic_m24) makes them ordinary toggling mid-bits. (The exact net set shifts
    # with WMAN, so this row contributes advisory-toggle coverage, not a fixed net contract.)
    out.append(_trans("sincos", s, "deep", "w8m24", 8, 24, "random", 768))
    # zkf_atan2 coverage: random + directed pair table at 5/16 reach the small-ratio bypass, the residual divide, and
    # every special/axis/diagonal pair; un=200 and sn/pa toggle throughput + back-end staging. (Joint-exhaustive is
    # infeasible for a two-input op, so coverage is random + directed.)
    out.append(_trans("atan2", s, "deep", "atan2_w5m16", 5, 16, "random", 4000))
    out.append(_trans("atan2", s, "deep", "atan2_w5m16_directed", 5, 16, "directed", 0))
    out.append(_trans("atan2", s, "deep", "atan2_w5m16", 5, 16, "random", 2000, un=200))
    out.append(_trans("atan2", s, "deep", "atan2_w5m16", 5, 16, "random", 2000, sn=1, pa=1))
    # Wide-format atan2: at the small format some divider / shamt / significand / magnitude high bits sit above the
    # format ceiling; w8m24 makes them ordinary toggling mid-bits. (The exact net set shifts with WMAN, so this row
    # contributes advisory-toggle coverage, not a fixed net contract.)
    out.append(_trans("atan2", s, "deep", "atan2_w8m24", 8, 24, "random", 4000))
    for cfg, w, n in [("w8_n2", 8, 2), ("w8_n4", 8, 4), ("w24_n3", 24, 3)]:
        out.append(_pipe(s, "deep", cfg, w, n, 96))
    # w56s1 (wide directed): one-hot/low-magnitude vectors drive the full leading-zero-count range, toggling the high
    # count bits, the split digit registers, and the top-level z3 detect (its group only fits for W>=49). w130s1
    # pushes the radix-4 count to cnt[7] (CNTW=8 only for clog2(W) in {7,8}; count W-1=129 sets it, unreachable at
    # narrower W or any embedded instance). STAGE_OUTPUT and standalone STAGE_SPLIT=2 (w32s2, NL4>=3); the bench
    # checks out_valid/sb_out are delayed by STAGE_SPLIT + STAGE_OUTPUT.
    for cfg, w, split, output, kind in [
        ("w8s0", 8, 0, 0, "exhaustive"),
        ("w8s1", 8, 1, 0, "exhaustive"),
        ("w9s1", 9, 1, 0, "exhaustive"),
        ("w32s1", 32, 1, 0, "directed"),
        ("w56s1", 56, 1, 0, "directed"),
        ("w130s1", 130, 1, 0, "directed"),
        ("w8s0o1", 8, 0, 1, "exhaustive"),
        ("w8s1o1", 8, 1, 1, "exhaustive"),
        ("w32s2o0", 32, 2, 0, "directed"),
        ("w32s2o1", 32, 2, 1, "directed"),
    ]:
        out.append(
            _run(
                "normshift",
                s,
                "deep",
                cfg,
                [("W", w), ("STAGE_SPLIT", split), ("STAGE_OUTPUT", output)],
                kind=kind,
                count=0,
                plus_names={"W": "ZKF_NS_W", "STAGE_SPLIT": "ZKF_NS_SPLIT", "STAGE_OUTPUT": "ZKF_NS_OUTPUT"},
            )
        )
    for cfg, w, split, kind in [
        ("w8s0", 8, 0, "exhaustive"),
        ("w8s1", 8, 1, "exhaustive"),
        ("w16s0", 16, 0, "directed"),
        ("w16s1", 16, 1, "directed"),
    ]:
        out.append(
            _run(
                "rshift",
                s,
                "deep",
                cfg,
                [("W", w), ("STAGE_SPLIT", split)],
                kind=kind,
                count=0,
                plus_names={"W": "ZKF_RSH_W", "STAGE_SPLIT": "ZKF_RSH_SPLIT"},
            )
        )
    for w, m, u in [(4, 5, 6), (4, 5, 7), (3, 6, 5), (4, 4, 6)]:
        out.append(_pack(s, "deep", f"w{w}m{m}u{u}", w, m, u, "exhaustive", 0))
    for w, m, i in [(4, 5, 7), (4, 6, 5), (3, 6, 4), (5, 4, 8)]:
        for si in (0, 1):
            out.append(_cast("to_int", s, "deep", f"w{w}m{m}i{i}", w, m, i, "exhaustive", 0, si))
            out.append(_cast("from_int", s, "deep", f"w{w}m{m}i{i}", w, m, i, "exhaustive", 0, si))
    for wi, mi, wo, mo in [(3, 4, 5, 6), (5, 6, 3, 4), (4, 5, 4, 4), (4, 4, 4, 5), (5, 4, 3, 6), (3, 6, 5, 4)]:
        for si in (0, 1):
            out.append(_resize(s, "deep", f"w{wi}m{mi}_to_w{wo}m{mo}", wi, mi, wo, mo, "exhaustive", 0, si))
    # round coverage: cheap exhaustive formats toggle the rounder + specials path (w2_m4 reaches round-up overflow);
    # all-on toggles the input/pack/output registers; wide w8_m36 toggles the boundary-mask decoder and
    # exponent-difference bits the tiny formats cannot reach.
    for w, m in [(2, 4), (4, 5), (3, 6)]:
        out.append(_round(s, "deep", f"w{w}m{m}", w, m, "exhaustive", 0))
    out.append(_round(s, "deep", "w4m5_maxpipe", 4, 5, "exhaustive", 0, si=1, pa=1, so=1))
    out.append(_round(s, "deep", "w8m36", 8, 36, "random", 1024))
    # STAGE_OUTPUT=1 / EXP_IS_BIASED=1 elaborate branches dark under the defaults: _zkf_pack g_out_reg, zkf_pipe
    # g_registered (div, via _zkf_pack_delay), zkf_resize g_owr (widen path), and the standalone packer's
    # registered-output / biased-exponent cones. One config per branch suffices under merged-union.
    out.append(_binary("mul", s, "deep", "w3m5", 3, 5, "exhaustive", 0, sp=0, so=1))
    out.append(_binary("add", s, "deep", "w3m5", 3, 5, "exhaustive", 0, sd=0, sa=0, so=1))
    out.append(_binary("addsub", s, "deep", "w3m5", 3, 5, "exhaustive", 0, sd=1, sa=1, so=1))
    out.append(_binary("div", s, "deep", "w3m5", 3, 5, "exhaustive", 0, si=0, so=1))
    out.append(_cast("from_int", s, "deep", "w4m5i7", 4, 5, 7, "exhaustive", 0, 0, so=1))
    # Identity widen (FRAC_PAD=0, BIAS_OFFSET=0): registered s_y has no structurally-zero padding, so every bit
    # toggles; a padding-bearing widen would leave low s_y bits permanently 0.
    out.append(_resize(s, "deep", "w4m5_to_w4m5", 4, 5, 4, 5, "exhaustive", 0, 0, so=1))  # widen-only -> g_owr
    out.append(_resize(s, "deep", "w5m6_to_w3m4", 5, 6, 3, 4, "exhaustive", 0, 0, so=1))  # narrow -> g_out_reg
    out.append(_pack(s, "deep", "w4m5u6", 4, 5, 6, "exhaustive", 0, so=1))
    out.append(_pack(s, "deep", "w4m5u6", 4, 5, 6, "exhaustive", 0, eb=1))
    # ASSUME_NO_OVERFLOW=1 prunes the overflow detector (exp_overflow forced to 0): the case generator drops
    # out-of-range exponents, so the surviving in-range / force_inf / round-carry cases must still match the reference.
    out.append(_pack(s, "deep", "w4m5u6", 4, 5, 6, "exhaustive", 0, nov=1))


def _properties(out: list) -> None:
    s = "icarus"
    for op in ("add", "addsub"):
        tgt = f"sim_properties_{op}_icarus"
        for sd in (0, 1):
            for sa in (0, 1):
                for cfg, w, m, k, c in BINARY:
                    out.append(_binary(op, s, "properties", cfg, w, m, k, c, sd=sd, sa=sa, target=tgt, root_module=op))
    for sp in (0, 1):
        for cfg, w, m, k, c in BINARY:
            out.append(
                _binary(
                    "mul",
                    s,
                    "properties",
                    cfg,
                    w,
                    m,
                    k,
                    c,
                    sp=sp,
                    target="sim_properties_mul_icarus",
                    root_module="mul",
                )
            )


# Smoke set: (name, module, extra-vlog).
_FAST = [
    ("pack", "pack", [("WEXP", 2), ("WMAN", 4), ("WEXP_UNBIASED", 4)]),
    ("cmp", "cmp", [("WEXP", 2), ("WMAN", 4)]),
    ("sort", "sort", [("WEXP", 2), ("WMAN", 4)]),
    ("abs", "abs", [("WEXP", 2), ("WMAN", 4)]),
    ("neg", "neg", [("WEXP", 2), ("WMAN", 4)]),
    ("is_finite", "is_finite", [("WEXP", 2), ("WMAN", 4)]),
    ("saturate", "saturate", [("WEXP", 2), ("WMAN", 4)]),
    ("add_sd0_sa0", "add", [("WEXP", 2), ("WMAN", 4), ("STAGE_DECODE", 0), ("STAGE_ALIGN", 0)]),
    ("add_sd1_sa1", "add", [("WEXP", 2), ("WMAN", 4), ("STAGE_DECODE", 1), ("STAGE_ALIGN", 1)]),
    ("addsub_sd0_sa0", "addsub", [("WEXP", 2), ("WMAN", 4), ("STAGE_DECODE", 0), ("STAGE_ALIGN", 0)]),
    ("addsub_sd1_sa1", "addsub", [("WEXP", 2), ("WMAN", 4), ("STAGE_DECODE", 1), ("STAGE_ALIGN", 1)]),
    ("ilog2_sd0", "mul_ilog2_const", [("WEXP", 2), ("WMAN", 4), ("STAGE_DECODE", 0)]),
    ("ilog2_sd1", "mul_ilog2_const", [("WEXP", 2), ("WMAN", 4), ("STAGE_DECODE", 1)]),
    ("ilog2rt_sd0", "mul_ilog2", [("WEXP", 2), ("WMAN", 4), ("WK", 3), ("STAGE_DECODE", 0)]),
    ("ilog2rt_sd1", "mul_ilog2", [("WEXP", 2), ("WMAN", 4), ("WK", 3), ("STAGE_DECODE", 1)]),
    ("mul_sp0", "mul", [("WEXP", 2), ("WMAN", 4), ("STAGE_PRODUCT", 0)]),
    ("mul_sp1", "mul", [("WEXP", 2), ("WMAN", 4), ("STAGE_PRODUCT", 1)]),
    ("mul_so1", "mul", [("WEXP", 2), ("WMAN", 4), ("STAGE_OUTPUT", 1)]),
    ("mul_si1", "mul", [("WEXP", 2), ("WMAN", 4), ("STAGE_INPUT", 1)]),
    ("div_si0", "div", [("WEXP", 2), ("WMAN", 4), ("STAGE_INPUT", 0)]),
    ("div_si1", "div", [("WEXP", 2), ("WMAN", 4), ("STAGE_INPUT", 1)]),
    ("from_int_si0", "from_int", [("WEXP", 2), ("WMAN", 4), ("WINT", 4), ("STAGE_INPUT", 0)]),
    ("from_int_si1", "from_int", [("WEXP", 2), ("WMAN", 4), ("WINT", 4), ("STAGE_INPUT", 1)]),
    ("to_int_si0", "to_int", [("WEXP", 2), ("WMAN", 4), ("WINT", 4), ("STAGE_INPUT", 0)]),
    ("to_int_si1", "to_int", [("WEXP", 2), ("WMAN", 4), ("WINT", 4), ("STAGE_INPUT", 1)]),
    ("resize_si0", "resize", [("WEXP_IN", 3), ("WMAN_IN", 4), ("WEXP_OUT", 3), ("WMAN_OUT", 4), ("STAGE_INPUT", 0)]),
    ("resize_si1", "resize", [("WEXP_IN", 3), ("WMAN_IN", 4), ("WEXP_OUT", 3), ("WMAN_OUT", 4), ("STAGE_INPUT", 1)]),
    ("exp2", "exp2", [("WEXP", 2), ("WMAN", 16), ("STAGE_OUTPUT", 0)]),
    ("log2", "log2", [("WEXP", 2), ("WMAN", 16), ("STAGE_OUTPUT", 0)]),
    ("sincos", "sincos", [("WEXP", 2), ("WMAN", 16), ("UNROLL100", 50)]),
]


def _fast(out: list) -> None:
    for name, module, vlog in _FAST:
        out.append(_run(module, "icarus", "fast", name, vlog, kind="exhaustive", count=0))
    # zkf_atan2 is two-input (joint-exhaustive infeasible): the smoke uses the directed special/axis/diagonal pairs.
    out.append(
        _run(
            "atan2", "icarus", "fast", "atan2", [("WEXP", 5), ("WMAN", 16), ("UNROLL100", 50)], kind="directed", count=0
        )
    )


# The exhaustive sincos run (2**18 codes through the multi-cycle CORDIC) is by far the slowest case in the suite -- on
# the CI runner a single one takes ~30 min, so it strands one worker while the rest of the pool sits idle behind it.
# Split every exhaustive sincos into this many strided shards (union == the full sweep, coverage unchanged) so worksteal
# spreads them across the idle workers. Bench support lives in test_sincos.cases_for + ZKF_SHARD_INDEX/COUNT.
SINCOS_SHARDS = 8


def _shard_long_cases(runs: list) -> list:
    out = []
    for r in runs:
        if r.module == "sincos" and dict(r.plus).get("ZKF_KIND") == "exhaustive" and SINCOS_SHARDS > 1:
            for k in range(SINCOS_SHARDS):
                suffix = f"_sh{k}of{SINCOS_SHARDS}"
                plus = r.plus + [("ZKF_SHARD_INDEX", k), ("ZKF_SHARD_COUNT", SINCOS_SHARDS)]
                out.append(
                    Run(r.module, r.sim, r.tier, r.config + suffix, r.target, r.root + suffix, r.vlog, plus, r.defines)
                )
        else:
            out.append(r)
    return out


def build_matrix() -> list:
    """The complete suite as a flat list of Runs."""
    out: list = []
    _per_pr("icarus", out)
    _per_pr("verilator", out)
    _deep_correctness(out)
    _deep_coverage(out)
    _properties(out)
    _fast(out)
    return _shard_long_cases(out)


if __name__ == "__main__":
    import sys
    from collections import Counter

    runs = build_matrix()
    if "--list" in sys.argv:
        for r in sorted(runs, key=lambda r: r.id):
            print(r.id, "->", r.target, r.root)
    counts = Counter((r.tier, r.sim) for r in runs)
    print("counts per (tier, sim):")
    for key in sorted(counts):
        print(f"  {key[0]:11s} {key[1]:9s} {counts[key]}")
    print(f"  {'TOTAL':11s} {'':9s} {len(runs)}")
    ids = [r.id for r in runs]
    dups = [i for i, n in Counter(ids).items() if n > 1]
    print("duplicate ids:", dups if dups else "none")

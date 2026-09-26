#!/usr/bin/env python3
"""
Constant generator for the ZKF CORDIC trigonometric operators: zkf_sincos, zkf_atan2.

Phase ``x`` is in turns (``sin = sin(2*pi*x)``). The module reduces ``x`` mod 1 to ``frac(x) in [0,1)``, takes the top
two bits as the ``quadrant`` and folds the rest to one octant angle ``t' in [0, pi/4]``, runs a fixed-point CORDIC
rotation, then unmaps and packs. Why CORDIC: faithful sin/cos needs relative accuracy near each zero, and CORDIC gets it
with adds/shifts only (no small-coefficient cancellation; inverse-gain folded into the seed). KEY DESIGN: the SAME
engine (``zkf/rtl/_zkf_cordic_core.v``), run in *vectoring* mode, computes ``atan2`` + the vector magnitude -- reusing
the arctan LUT (stored in *turns*), datapath, iteration count, and quadrant pre/post-processing; only MODE differs.

Angle units: the arctan LUT is ``L[i] = atan(2**-i)/(2*pi)`` at scale ``2**-ZF``, ``ZF = WT + 2 + GUARD_ZF``. The octant
coordinate ``t'`` is at scale ``2**-WT``, so the angle accumulator seed is ``z0 = t' << GUARD_ZF`` (no multiply).

Tiny angles: below the CORDIC's smallest resolvable step a linear path returns ``sin ~= 2*pi*x`` (one multiply by a
generated ``2*pi``), ``cos = +1``; ``GUARD_FF(WMAN)`` places that handoff where it holds <= 1 ULP. After the octant fold
``cos`` is never small, so only sin needs it.

``--emit`` writes the per-WMAN Verilog cores and the Python data table and ``--check-emit`` fails if the checked-in
copies have drifted from a fresh run; ``--check`` verifies that both operators (and the quadrant) are faithfully
rounded against the unrounded ``mpmath`` truth (``zkf_accuracy.py``).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil
from pathlib import Path
from textwrap import dedent

import mpmath as mp

if __package__:  # python -m tools.<generator>
    from . import zkf_accuracy, zkf_emit
else:  # python tools/<generator>.py, or tools/ on sys.path
    import zkf_accuracy
    import zkf_emit

mp.mp.prec = 400  # generous headroom for the gain/LUT constants and ground-truth rounding

REPO = Path(__file__).resolve().parents[1]
HDL = REPO / "zkf" / "rtl"
TABLES = HDL / "_tables"
PKG_TABLES = REPO / "zkf" / "_tables"

# MIXED CORDIC: run K ~ WMAN/2 rotation iterations, then ONE linear rotation for the residual (cos = x_K - y_K*phi,
# sin = y_K + x_K*phi). The dropped phi^2/2 term is < 1 ULP once the residual is <= ~2**-(WMAN/2), so the pipeline is
# ~WMAN/2 stages instead of ~1.5*WMAN, traded for two correction multiplies. (The correction fixes only the ANGLE
# residual; the iteration array's truncation still bounds the small sines, so the datapath keeps ~1.5*WMAN frac bits.)
GUARD_XY = 6  # x/y fractional bits past 1.5*WMAN (round/sticky + iteration-rounding headroom)
# GOTCHA: GUARD_XY is SHARED by both operators (sizes WX/KINV and atan2's divider F / INV_TAU scale). sincos is faithful
# at 3; atan2's theta is XF-bound (more iterations don't help, only XF does), so it needs the larger value. GUARD_XY=6
# keeps atan2 faithful across the supported range (WMAN >= 16, which is the smallest trig format -- see SUPPORTED_WMAN
# below). WMAN=11 would have needed GUARD_XY=7 (XF=24) and was therefore dropped from the trig operators rather than
# widen the shared engine for every format; it is still supported by the algebraic operators, which use no CORDIC table.
# Do not lower GUARD_XY, or add a smaller trig WMAN, without re-running --check for all supported WMAN.
# Iterations before termination: N = (WMAN+1)//2 + GUARD_ITER_*. Named per operator but kept EQUAL (n_iters()
# enforces it): one table bakes one LUT depth, and zkf_cordic's runtime-mode engine has one stopping count.
# Both are 2 because all three outputs miss faithful rounding at 1, each for its own dropped term -- sincos and
# atan2's MAGNITUDE a quadratic one, atan2's THETA the cubic one dropped by the RESIDUAL correction's arctan
# series. (The small-ratio bypass drops a cubic term too, but more CORDIC iterations cannot move a bypass result;
# GUARD_DIV governs that path.) The regression corpora pin one input of each kind.
GUARD_ITER_SINCOS = 2
GUARD_ITER_ATAN2 = 2
# Angle accumulator integer headroom above the ZF fractional bits (z stays within +-(1/4 turn) through the rotation).
GUARD_Z = 3
# Extra angle fractional bits past the coordinate's (WT+2) turns bits: the rotation sums K LUT entries each rounded to
# 2**-ZF (~K*2**-ZF error) and the residual feeds the correction multiply, so it must keep full small-angle precision.
GUARD_ZF = 6
# atan2 residual-divide guard: extra quotient fractional bits so the divide-termination and small-ratio bypass round
# to <= 1 ULP. Consumed by the atan2 model and _zkf_cordic_unit.v.
# NOT MONOTONE -- raising it does not buy accuracy. It also sets the small-ratio bypass threshold, and the bypass
# drops a term cubic in the ratio (at 9, (11,53) theta exceeds 2 ULP). This, not GUARD_ITER_ATAN2, binds theta, the
# output with the least headroom.
GUARD_DIV = 8

# Inputs each once more than 1 ULP off. Keyed (wexp, wman); always swept, so a regression cannot hide behind the
# random draws.
SINCOS_REGRESSIONS = {(8, 24): [0x3EFFFBD1], (6, 18): [0x3E0044]}
# Same, for atan2, keyed (wexp, wman) -> [(y, x)]. Pinned on BOTH output axes, since a pair can miss on one and
# pass on the other: at GUARD_ITER 1, (6,16) misses on theta and the first (6,18) entry on magnitude. The next three
# (6,18) entries sit just below the overflow threshold, where a spurious +inf would break the README's overflow
# contract. The last (6,18) entry and the (5,16) one straddle the min_normal/2 midpoint from BELOW and ABOVE
# respectively, so narrowing zkf_accuracy.faithful's midpoint band to one side fails here.
ATAN2_REGRESSIONS = {
    (6, 16): [(0x38967F, 0x1DEA13)],
    (6, 18): [
        (0x7BEC6A, 0x7D7C61),
        (0x6BFFFF, 0x7DFFFF),
        (0x786922, 0x7DF44F),
        (0x6BF359, 0x7DFFFF),
        (0x05F9A3, 0x3E87D9),
    ],
    (5, 16): [(0x016E79, 0x0797D1)],
}

WMAN_MIN, WMAN_MAX = 16, 53
SUPPORTED_WMAN = [16, 18, 24, 27, 32, 36, 48, 53]
# --check formats: every supported WMAN. Each operator appends its own narrow-exponent case.
CHECK_CASES = [(6, 16), (6, 18), (8, 24), (8, 27), (8, 32), (8, 36), (8, 48), (8, 53), (11, 53)]

# Random --check samples per (format, operator) for non-exhaustive formats. UNSEEDED, so repeated runs accumulate
# coverage; override with ZKF_CHECK_SAMPLES=<n>.
RANDOM_CHECK_SAMPLES = int(os.environ.get("ZKF_CHECK_SAMPLES", "1000000"))


def guard_ff(wman: int) -> int:
    # Small-angle handoff guard: places the linear-path boundary where |1 - cos(2*pi*2**e_b)| <= 2**-WMAN, i.e.
    # GUARD_FF >= WMAN//2 + 2; floored at 12. Mirrored in _zkf_cordic_unit.
    return max(12, wman // 2 + 2)


def ff_bits(wman: int) -> int:
    """Reduced fraction width FF: frac(x) at scale 2**-FF; top 2 bits = quadrant, low FF-2 = t."""
    return wman + guard_ff(wman)


def wt_bits(wman: int) -> int:
    """Quadrant-local coordinate width WT = FF - 2 (t in [0,1) at scale 2**-WT)."""
    return ff_bits(wman) - 2


def n_sincos(wman: int) -> int:
    """N for zkf_sincos: rotation iterations before the linear final rotation (see the GUARD_ITER_* comment)."""
    return (wman + 1) // 2 + GUARD_ITER_SINCOS


def n_atan2(wman: int) -> int:
    """N for zkf_atan2: vectoring iterations before the residual divide (see the GUARD_ITER_* comment)."""
    return (wman + 1) // 2 + GUARD_ITER_ATAN2


def n_iters(wman: int) -> int:
    """The one CORDIC depth both operators run to, baked into the shared per-WMAN table as its LUT length."""
    ns, na = n_sincos(wman), n_atan2(wman)
    if ns != na:
        raise ValueError(
            f"GUARD_ITER_SINCOS/GUARD_ITER_ATAN2 diverge at WMAN={wman} ({ns} vs {na}); see the GUARD_ITER_* comment."
        )
    return ns


def tsa_bits(wman: int) -> int:
    """
    Small-angle handoff: octant coordinate t' below 2**TSA_BITS takes the linear bypass (sin = 2*pi*t', cos = 1).
    Bound by the cos=1 limit theta'(rad) < 2**-(WMAN/2): TSA_BITS = (WT+2) - ceil(WMAN/2) - 3.
    """
    return (wt_bits(wman) + 2) - ((wman + 1) // 2) - 3


def cordic_module(wman: int) -> str:
    return f"_zkf_cordic_m{wman}"


@dataclass(kw_only=True)
class Spec:
    wman: int
    n: int  # shared CORDIC depth baked into the table (== n_sincos == n_atan2; see GUARD_ITER_*)
    xf: int  # x/y fractional bits baked into the table (scale 2**-xf); == max operator XF (table width)
    xw: int  # x/y signed width (1 sign + 1 integer + xf)
    wt: int  # quadrant-local coordinate width (FF - 2); angle in turns is t'/4 at scale 2**-(WT+2)
    zf: int  # angle accumulator fractional bits (scale 2**-zf) = WT + 2 + GUARD_ZF (finer than WT+2)
    zw: int  # angle signed width
    kinv: int  # round(1/gain * 2**xf), gain = prod sqrt(1+2**-2i)
    n_sincos: int  # zkf_sincos iterations: (WMAN+1)//2 + GUARD_ITER_SINCOS
    n_atan2: int  # zkf_atan2 iterations: (WMAN+1)//2 + GUARD_ITER_ATAN2
    xf_atan2: int  # zkf_atan2 x/y fractional bits (drives the divider F/STEPS); == xf atm (shared engine width)
    tsa: int = 0  # small-angle handoff: t' < tsa uses the linear path (TSA_BITS = log2)
    lut: list = field(default_factory=list)  # L[i] = round(atan(2**-i)/(2*pi) * 2**zf), i = 0..n-1
    # The multiplier constants are EMITTED PRE-NARROWED to WMAN+5 bits, each at its own native scale 2**-S; every
    # dependent shift/exp-offset derives from that scale, so the datapath needs no DROP correction tokens.
    const2pi: int = 0  # round(2*pi * 2**CONST2PI_S), narrowed to WMAN+5 bits (sincos small-angle / linear-rotation)
    const2pi_s: int = 0  # native scale of the narrowed const2pi == WMAN+2
    inv_tau: int = 0  # round(2**INVTAU_S / (2*pi)), narrowed to WMAN+5 bits (atan2 residual/bypass turns scaling)
    invtau_s: int = 0  # native scale of the narrowed inv_tau == WMAN+7
    kinv_mag: int = 0  # round(1/gain * 2**KINV_S), narrowed to WMAN+5 bits (atan2 magnitude descale)
    kinv_s: int = 0  # native scale of the narrowed kinv_mag == WMAN+5


def cordic_gain(n: int):
    g = mp.mpf(1)
    for i in range(n):
        g *= mp.sqrt(1 + mp.mpf(2) ** (-2 * i))
    return g


def choose_spec(wman: int) -> Spec:
    """
    All-closed-form: the CORDIC depth and widths are functions of WMAN (so the pipeline depth is too). The arctan
    LUT is in turns and the inverse gain folds into the x seed; --check validates the resulting faithfulness.
    """
    if not (WMAN_MIN <= wman <= WMAN_MAX):
        raise ValueError(f"Bad {wman=}")
    n = n_iters(wman)  # shared depth (asserts n_sincos == n_atan2)
    xf = ceil(3 * wman / 2) + GUARD_XY  # shared engine XF
    zf = wt_bits(wman) + 2 + GUARD_ZF
    xw = xf + 2
    zw = zf + GUARD_Z
    kinv = int(mp.nint((1 / cordic_gain(n)) * (mp.mpf(2) ** xf)))  # full-precision inverse gain (the sincos seed)
    lut = [int(mp.nint(mp.atan(mp.mpf(2) ** (-i)) / (2 * mp.pi) * (mp.mpf(2) ** zf))) for i in range(n)]
    # Each constant is round-narrowed to its top WMAN+5 bits at its native scale 2**-S; consuming shifts are then plain
    # "product-scale minus target-scale" (no DROP correction).
    const2pi_s = wman + 2  # narrowed 2*pi scale (was XF; XF - DROP == WMAN+2)
    invtau_s = wman + 7  # narrowed 1/(2*pi) scale (XF - DROP_IT   == WMAN+7)
    kinv_s = wman + 5  # narrowed 1/gain scale (XF - DROP_K      == WMAN+5)
    const2pi = int(mp.nint(2 * mp.pi * (mp.mpf(2) ** const2pi_s)))  # narrowed 2*pi, WMAN+5 bits
    inv_tau = int(mp.nint((mp.mpf(1) / (2 * mp.pi)) * (mp.mpf(2) ** invtau_s)))  # narrowed 1/(2*pi), WMAN+5 bits
    kinv_mag = int(mp.nint((1 / cordic_gain(n)) * (mp.mpf(2) ** kinv_s)))  # narrowed 1/gain, WMAN+5 bits
    return Spec(
        wman=wman,
        n=n,
        xf=xf,
        xw=xw,
        wt=wt_bits(wman),
        zf=zf,
        zw=zw,
        kinv=kinv,
        n_sincos=n_sincos(wman),
        n_atan2=n_atan2(wman),
        xf_atan2=xf,
        tsa=1 << tsa_bits(wman),
        lut=lut,
        const2pi=const2pi,
        const2pi_s=const2pi_s,
        inv_tau=inv_tau,
        invtau_s=invtau_s,
        kinv_mag=kinv_mag,
        kinv_s=kinv_s,
    )


def generate_all() -> dict[int, Spec]:
    return {wman: choose_spec(wman) for wman in SUPPORTED_WMAN}


# --------------------------------------------------------------------------------------------------
# Verilog emission
# --------------------------------------------------------------------------------------------------
class _Writer:
    """Accumulates 4-space-indented lines; ``w(...)`` accepts single lines or dedented multiline blocks."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def __call__(self, *texts: str) -> None:
        for text in texts:
            if "\n" in text:
                block = dedent(text).removeprefix("\n").removesuffix("\n")
                for line in block.split("\n"):
                    self._append(line)
            else:
                self._append(text)

    def _append(self, text: str) -> None:
        self._lines.append(("    " * self._depth + text) if text else "")

    def push(self) -> None:
        self._depth += 1

    def pop(self) -> None:
        assert self._depth > 0
        self._depth -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def _emit_consts(s: Spec) -> str:
    mod = cordic_module(s.wman)
    cwb = s.const2pi.bit_length()  # narrowed 2*pi width == WMAN+5 (scale 2**-CONST2PI_S)
    itwb = s.inv_tau.bit_length()  # narrowed 1/(2*pi) width == WMAN+5 (scale 2**-INVTAU_S)
    kmb = s.kinv_mag.bit_length()  # narrowed 1/gain width == WMAN+5 (scale 2**-KINV_S)
    w = _Writer()
    w("/// GENERATED by zkf_trig.py -- DO NOT EDIT.")
    w(f"/// Per-WMAN CORDIC constants (WMAN={s.wman}): arctan(2^-i)/2pi LUT in turns, inverse-gain seed, widths.")
    w("/// Binds _zkf_cordic_core (MODE as there); both modes run to the same depth N, so one LUT and one inverse gain")
    w("/// serve either.")
    w("")
    w("`default_nettype none")
    w("")
    w(f"module {mod} #(")
    w.push()
    w("parameter integer MODE      = 0,")
    w("parameter integer UNROLL100 = 100,")
    w("parameter integer PARALLEL  = (MODE != 1) && (UNROLL100 < 100),")
    w(f"parameter integer EXPECT_WMAN       = {s.wman},")
    w(f"parameter integer EXPECT_N          = {s.n},")
    w(f"parameter integer EXPECT_XF         = {s.xf},")
    w(f"parameter integer EXPECT_WX         = {s.xw},")
    w(f"parameter integer EXPECT_WT         = {s.wt},")
    w(f"parameter integer EXPECT_ZF         = {s.zf},")
    w(f"parameter integer EXPECT_WZ         = {s.zw},")
    w(f"parameter integer EXPECT_WCONST2PI  = {cwb},")
    w(f"parameter integer EXPECT_CONST2PI_S = {s.const2pi_s},")
    w(f"parameter integer EXPECT_WINVTAU    = {itwb},")
    w(f"parameter integer EXPECT_INVTAU_S   = {s.invtau_s},")
    w(f"parameter integer EXPECT_WKINV_MAG  = {kmb},")
    w(f"parameter integer EXPECT_KINV_S     = {s.kinv_s}")
    w.pop()
    w(") (")
    w.push()
    w(f"""
        input  wire                clk,
        input  wire                rst,
        input  wire                start,
        input  wire                vectoring,
        input  wire signed [{s.xw - 1:3}:0] x0,
        input  wire signed [{s.xw - 1:3}:0] y0,
        input  wire signed [{s.zw - 1:3}:0] z0,
        output wire                busy,
        output wire                done,
        output wire                z_done,
        output wire signed [{s.xw - 1:3}:0] xn,
        output wire signed [{s.xw - 1:3}:0] yn,
        output wire signed [{s.zw - 1:3}:0] zn,
        output wire        [{cwb - 1:3}:0] const2pi,   // round(2*pi * 2**CONST2PI_S), WMAN+5 bits (sin/cos to radians)
        output wire        [{itwb - 1:3}:0] inv_tau,    // round(2**INVTAU_S / (2*pi)), WMAN+5 bits (atan2 turns scale)
        output wire        [{kmb - 1:3}:0] kinv_mag    // round(2**KINV_S / gain), WMAN+5 bits (atan2 magnitude descale)
    """)
    w.pop()
    w(");")
    w.push()
    w(f"localparam integer N    = {s.n};   // iterations (folded over the cycles selected by UNROLL100)")
    w(f"localparam integer WX   = {s.xw};")
    w(f"localparam integer WZ   = {s.zw};")
    w(f"localparam integer XF   = {s.xf};   // x/y fractional scale")
    w(f"localparam integer WT   = {s.wt};   // quadrant-local coordinate width used to derive ZF")
    w(f"localparam integer ZF   = {s.zf};   // angle (turns) fractional scale == WT + 2 + GUARD_ZF")
    w(f"localparam integer WCONST2PI  = {cwb};")
    w(f"localparam integer WINVTAU    = {itwb};")
    w(f"localparam integer WKINV_MAG  = {kmb};")
    # Native fixed-point scales of the pre-narrowed multiplier constants. Each consumer derives its shift / exp-offset
    # from these directly: the constant's top WMAN+5 bits at scale 2**-S ARE the value to the kept precision.
    w(f"localparam integer CONST2PI_S = {s.const2pi_s};   // scale of const2pi (== WMAN+2)")
    w(f"localparam integer INVTAU_S   = {s.invtau_s};   // scale of inv_tau  (== WMAN+7)")
    w(f"localparam integer KINV_S     = {s.kinv_s};   // scale of kinv_mag (== WMAN+5)")
    w(f"localparam signed [WX-1:0] KINV = {s.xw}'sd{s.kinv};   // round(1/gain * 2**XF), the sin/cos seed")
    w("")
    w("""
        // Geometry contract: the consuming RTL passes its locally-computed dimensions and constant scales here. If a
        // formula drifts away from zkf_trig.py, elaboration fails before any port truncation/extension can hide it.
        generate
            if ((EXPECT_WMAN      != %-2d) || (EXPECT_N  != N)  || (EXPECT_XF != XF) || (EXPECT_WX != WX) ||
                (EXPECT_WT        != WT) || (EXPECT_ZF != ZF) || (EXPECT_WZ != WZ) ||
                (EXPECT_WCONST2PI != WCONST2PI) || (EXPECT_CONST2PI_S != CONST2PI_S) ||
                (EXPECT_WINVTAU   != WINVTAU)   || (EXPECT_INVTAU_S   != INVTAU_S)   ||
                (EXPECT_WKINV_MAG != WKINV_MAG) || (EXPECT_KINV_S     != KINV_S)) begin : g_invalid_geometry_contract
                _zkf_invalid_cordic_geometry_contract u_invalid();
            end
        endgenerate
    """ % s.wman)
    w("")
    w(f"assign const2pi = {cwb}'d{s.const2pi};")
    w(f"assign inv_tau  = {itwb}'d{s.inv_tau};")
    w(f"assign kinv_mag = {kmb}'d{s.kinv_mag};")
    w("")
    w("// arctan(2^-i)/(2*pi) in turns, scale 2**-ZF, packed L[0] in the low WZ bits.")
    w("wire [N*WZ-1:0] LUT = {")
    w.push()
    # high index first in the concatenation literal (MSBs), L[0] last (LSBs)
    for i in reversed(range(s.n)):
        sep = "" if i == 0 else ","
        w(f"{s.zw}'d{s.lut[i]}{sep}   // atan(2^-{i})/2pi")
    w.pop()
    w("};")
    w("")
    w("""
        // Rotation seeds (1/gain, 0), so (xn, yn) = (cos z0, sin z0).
        wire                 rotate = (MODE == 0) || ((MODE == 2) && !vectoring);
        wire signed [WX-1:0] seed_x = rotate ? KINV       : x0;
        wire signed [WX-1:0] seed_y = rotate ? {WX{1'b0}} : y0;
        _zkf_cordic_core #(
            .N(N), .UNROLL100(UNROLL100), .PARALLEL(PARALLEL), .WX(WX), .WZ(WZ), .MODE(MODE)
        ) u_cordic (
            .clk(clk), .rst(rst), .start(start), .vectoring(vectoring),
            .x0(seed_x), .y0(seed_y), .z0(z0), .lut(LUT),
            .busy(busy), .done(done), .z_done(z_done), .xn(xn), .yn(yn), .zn(zn)
        );
    """)
    w.pop()
    w("endmodule")
    w("")
    w("`default_nettype wire")
    return w.render()


def _emit_python(all_specs: dict[int, Spec]) -> str:
    w = _Writer()
    w("# GENERATED by zkf_trig.py -- DO NOT EDIT.")
    w('"""Bit-exact CORDIC constants for zkf_sincos (and the shared atan2 engine), consumed by the zkf package."""')
    w("")
    w("SPECS = {")
    w.push()
    for wman in sorted(all_specs):
        s = all_specs[wman]
        w(f"{wman}: dict(")
        w.push()
        w(
            f"n={s.n}, xf={s.xf}, xw={s.xw}, wt={s.wt}, zf={s.zf}, zw={s.zw}, kinv={s.kinv}, tsa={s.tsa}, "
            f"const2pi={s.const2pi}, const2pi_s={s.const2pi_s}, inv_tau={s.inv_tau}, invtau_s={s.invtau_s},"
        )
        w(
            f"kinv_mag={s.kinv_mag}, kinv_s={s.kinv_s}, n_sincos={s.n_sincos}, n_atan2={s.n_atan2}, "
            f"xf_atan2={s.xf_atan2},"
        )
        w(f"lut={s.lut!r},")
        w.pop()
        w("),")
    w.pop()
    w("}")
    w("")
    w(f"GUARD_DIV = {GUARD_DIV}")
    w("# FF (reduced-fraction width) = WMAN + max(12, WMAN//2 + 2); WT = FF - 2; ZF = WT + 2 + GUARD_ZF.")
    return w.render()


def _artifacts(all_specs: dict[int, Spec]) -> dict[Path, str]:
    arts = {TABLES / f"{cordic_module(wman)}.v": _emit_consts(s) for wman, s in sorted(all_specs.items())}
    return arts | {PKG_TABLES / "trig.py": _emit_python(all_specs)}


def emit(all_specs: dict[int, Spec]) -> None:
    zkf_emit.write(_artifacts(all_specs))


def check_emit(all_specs: dict[int, Spec]) -> None:
    zkf_emit.verify(_artifacts(all_specs), sorted(TABLES.glob("_zkf_cordic_m*.v")), "zkf_trig.py --emit")


# --------------------------------------------------------------------------------------------------
# Reporting and accuracy check
# --------------------------------------------------------------------------------------------------
def _report(all_specs: dict[int, Spec]) -> None:
    print(f"{'WMAN':>4} {'N':>4} {'XF':>4} {'WX':>4} {'ZF':>4} {'WZ':>4} {'WT':>4} {'lut_bits':>8}")
    for wman, s in sorted(all_specs.items()):
        lut_bits = s.n * s.zw
        print(f"{wman:>4} {s.n:>4} {s.xf:>4} {s.xw:>4} {s.zf:>4} {s.zw:>4} {s.wt:>4} {lut_bits:>8}")


# fork lets the pool inherit the freshly-emitted tables; fall back to the default context where fork is unavailable.
_MP_CONTEXT = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)


# One worker process per (format) case over a fork pool spreads the mpmath sweep across all cores. Each unit
# regenerates its own inputs (random draws are unseeded either way) and returns only its summary; the helper
# functions and freshly-emitted tables come from the forked parent.
def _sincos_unit(case: tuple[int, int]) -> tuple[float, float, int, int, int, str, tuple | None]:
    """One (wexp, wman) sweep -> (worst_sin, worst_cos, quad_mismatches, violations, count, tag, first_bad)."""
    import zkf
    import zkf.oracle
    from zkf import ZkfFormat

    wexp, wman = case
    fmt = ZkfFormat(wexp, wman)
    n = 1 << fmt.wfull
    exhaustive = n <= (1 << 22)
    inputs = list(range(n)) if exhaustive else _stratified_inputs(fmt)
    worst_sin = worst_cos = 0.0
    nq = nbad = 0
    bad = None
    for b in inputs:
        z = zkf.Zkf(fmt, b)
        r = z.sincos()
        if z.is_inf or z.is_zero:  # exact contract values, not transcendental truth
            t = zkf.oracle.sincos(z)
            tq = t.quadrant
            ok = (r.sin.bits, r.cos.bits) == (t.sin.bits, t.cos.bits)
            ds = dc = 0.0 if ok else float("inf")
            sok = cok = ok
        else:
            ts, tc, tq = zkf.oracle._sincos_exact(z)
            ds, sok, _ = zkf_accuracy.faithful(fmt, r.sin.bits, ts)
            dc, cok, _ = zkf_accuracy.faithful(fmt, r.cos.bits, tc)
        qok = r.quadrant == tq
        if not (sok and cok and qok):
            nbad += 1
            if bad is None:
                bad = (hex(b), ds, dc, r.quadrant, tq)
        worst_sin, worst_cos = max(worst_sin, ds), max(worst_cos, dc)
        nq += int(not qok)
    tag = "exhaustive" if exhaustive else f"sampled({len(inputs)})"
    return worst_sin, worst_cos, nq, nbad, len(inputs), tag, bad


def _check() -> list[str]:
    """End-to-end faithful-rounding check vs mpmath via the bit-exact model. Returns the per-format failures."""
    import sys
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(REPO))
    # zkf is first-imported here, after --emit has written the tables, so the fork pool inherits the freshly-emitted
    # data without reaching into package internals to reload/clear caches.
    import zkf  # noqa: F401
    import zkf.oracle  # noqa: F401

    zkf_accuracy.self_check()
    print("end-to-end faithful-rounding check (model vs mpmath):")
    # (6,36) carries sincos's underflow decision, which no case above reaches: the smallest representable |sin| is
    # ~2*pi*2**-WFRAC (at x = 1/2 +- 1 ULP), so it drops below min_normal only when BIAS < WFRAC. (8,36) yields zero
    # such truths; (6,36) yields 280, 148 of them flushing to +0.
    sincos_cases = [*CHECK_CASES, (6, 36)]
    # A regression keyed to a format nothing sweeps would never run and never say so.
    assert not (
        SINCOS_REGRESSIONS.keys() - set(sincos_cases)
    ), f"SINCOS_REGRESSIONS entries for unchecked formats: {sorted(SINCOS_REGRESSIONS.keys() - set(sincos_cases))}"
    failures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=_MP_CONTEXT) as ex:
        for (wexp, wman), (worst_sin, worst_cos, nq, nbad, count, tag, bad) in zip(
            sincos_cases, ex.map(_sincos_unit, sincos_cases)
        ):
            ok = nbad == 0
            print(
                f"  {'OK ' if ok else 'BAD'} {wexp}/{wman:<3} sin_ulp={worst_sin:.3f} cos_ulp={worst_cos:.3f} "
                f"quad_mismatch={nq} not_faithful={nbad} ({tag})"
            )
            if not ok:
                failures.append(
                    f"{wexp}/{wman}: sin_ulp={worst_sin:.4f} cos_ulp={worst_cos:.4f} quad_mismatch={nq} "
                    f"not_faithful={nbad}/{count} first_bad={bad}"
                )
    return [f"sincos {f}" for f in failures]


def _stratified_inputs(fmt) -> list[int]:
    import numpy as np

    rng = np.random.default_rng()  # unseeded: true randomness, fresh inputs each run

    def rand_bits() -> int:
        v = 0
        for _ in range((fmt.wfull + 31) // 32):
            v = (v << 32) | int(rng.integers(0, 1 << 32))
        return v & ((1 << fmt.wfull) - 1)

    out = [rand_bits() for _ in range(RANDOM_CHECK_SAMPLES)]
    quarter_fracs = [0, 1, fmt.frac_mask, (1 << (fmt.wfrac - 1)) | 1]
    for exp in range(1, fmt.exp_inf):
        for sign in (0, 1):
            for _ in range(6):
                out.append(fmt.pack(sign, exp, int(rng.integers(0, 1 << fmt.wfrac))).bits)
            for fr in quarter_fracs:
                out.append(fmt.pack(sign, exp, fr).bits)

    # --- near-zero-result regression guard (deterministic, swept every run) ---
    # sin/cos -> 0 at the quadrant/octant turns, where the ULP -> 0 and uniform-random z can't land within ~2**-wman.
    # Densely sweep the tiny-angle binades and the +/-K*ULP neighborhoods of every result-zero/octant turn so a future
    # guard reduction that regresses near-zero rounding is caught. (Hardening: sincos is clean today.)
    import math

    K, nf = 96, 1 << fmt.wfrac
    for exp in range(1, min(7, fmt.exp_inf)):  # z -> 0: tiny angles, dense low/high fracs, both signs
        for sign in (0, 1):
            for fr in set(list(range(min(K, nf))) + list(range(max(0, nf - K), nf))):
                out.append(fmt.pack(sign, exp, fr).bits)
    for T in (
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
    ):  # result-zero (1/4,1/2,3/4,1) + octant (1/8,3/8,...) turns
        eT = math.floor(math.log2(T))
        for e in (eT - 1, eT):
            biased = e + fmt.bias
            if not (1 <= biased < fmt.exp_inf):
                continue
            f0 = round((T - 2.0**e) / (2.0 ** (e - fmt.wfrac)))  # frac of z = T within binade e
            for f in range(f0 - K, f0 + K + 1):
                if 0 <= f < nf:
                    for sign in (0, 1):
                        out.append(fmt.pack(sign, biased, f).bits)

    # Quadrant boundary x octant-local BINADE sweep. The +-K-ULP neighborhoods above probe only the boundary
    # itself, while the corner where the CORDIC path meets the small-angle handoff sits whole binades away, at
    # k/4 +- 2**-j for j around the TSA threshold.
    j0 = (wt_bits(fmt.wman) + 2) - tsa_bits(fmt.wman)  # turns exponent of the small-angle handoff
    for k in range(5):  # k = 4 is not k = 0 again: only it approaches a full turn from below with a positive input
        # Past WMAN+1 the offset is below an ULP of 1/4 and the point collapses back onto k/4, which the +-K-ULP
        # block above already sweeps.
        for j in range(max(1, j0 - 4), min(j0 + 10, fmt.wman + 2)):
            for off in (1, -1):
                offsets = [Fraction(1, 1 << j)]
                offsets += [Fraction(int(rng.integers(1 << 20, 1 << 21)), 1 << (20 + j)) for _ in range(4)]
                for u in offsets:
                    v = Fraction(k, 4) + off * u
                    if v > 0:  # both input signs: the model reduces negative turns down a different branch
                        out.extend(fmt.encode(sgn * v).bits for sgn in (1, -1))

    out.extend(SINCOS_REGRESSIONS.get((fmt.wexp, fmt.wman), []))
    return out


def _atan2_pairs(fmt) -> list[tuple[int, int]]:
    """
    Stratified (y, x) pairs for the atan2 check (joint-exhaustive is infeasible). Random pairs plus two single-operand
    "fans" (each operand swept vs central-binade anchors of the other): since atan2 depends on the exponent DIFFERENCE,
    the fans cross every bypass threshold and quadrant boundary, and the diagonals exercise the |y|==|x| octant edge.
    """
    import sys

    import numpy as np

    if str(REPO / "tb") not in sys.path:  # saturating_y is a bench fixture
        sys.path.append(str(REPO / "tb"))
    from zkf_operands import saturating_y

    rng = np.random.default_rng()  # unseeded: true randomness, fresh pairs each run

    def rand_bits() -> int:
        v = 0
        for _ in range((fmt.wfull + 31) // 32):
            v = (v << 32) | int(rng.integers(0, 1 << 32))
        return v & ((1 << fmt.wfull) - 1)

    sgn = 1 << fmt.sign_shift
    specials = [0, sgn, fmt.exp_inf << fmt.wfrac, sgn | (fmt.exp_inf << fmt.wfrac)]
    fracs = [0, 1, fmt.frac_mask, 1 << (fmt.wfrac - 1)]
    anchors = [fmt.normal(s, fmt.bias, fr).bits for s in (0, 1) for fr in (0, fmt.frac_mask)]

    pairs: set[tuple[int, int]] = set()
    for _ in range(RANDOM_CHECK_SAMPLES):
        pairs.add((rand_bits(), rand_bits()))
    swept = list(specials)
    for s in (0, 1):
        for e in range(1, fmt.exp_inf):
            for fr in fracs:
                swept.append(fmt.normal(s, e, fr).bits)
    for w in swept:
        for a in anchors:
            pairs.add((w, a))  # operand swept on the y side, x anchored
            pairs.add((a, w))  # operand swept on the x side, y anchored
    for s in (0, 1):
        for e in range(1, fmt.exp_inf):
            base = fmt.normal(s, e, 0).bits
            pairs.add((base, base))  # |y| == |x| (octant edge)
            pairs.add((base, base ^ sgn))

    # --- near-axis / bypass-seam regression guard (deterministic, swept every run) ---
    # theta -> 0 near the +x axis (small |y|/x); the bypass divide must faithfully round the smallest thetas, which
    # random pairs never hit. Sweep the smallest |y| against large x and straddle the bypass cutoff. (Hardening.)
    nf = 1 << fmt.wfrac
    yfr = sorted(set(list(range(96)) + list(range(max(0, nf - 48), nf)) + [nf // 2]))
    # smallest-NONZERO theta band: theta ~ 2**(ey-xe)/(2pi) reaches the smallest nonzero turn (and the underflow-to-0
    # seam) at xe - ey ~ bias -- the danger zone (tiny-but-nonzero, ULP -> 0), not xe-ey huge where it underflows to 0.
    for ey in (1, 2, 3):
        for sy in (0, 1):
            for yf in yfr:
                y = fmt.normal(sy, ey, yf).bits
                for dsh in range(fmt.bias - 6, fmt.bias + 3):  # xe-ey across smallest-nonzero theta .. underflow seam
                    xe = ey + dsh
                    if 1 <= xe < fmt.exp_inf:
                        for xf in (0, nf // 2, nf - 1):
                            pairs.add((y, fmt.normal(0, xe, xf).bits))
    ts = choose_spec(fmt.wman).zf - fmt.wman - GUARD_DIV
    for off in (-1, 0, 1, 2):
        for ey in (1, max(1, fmt.bias // 2), max(1, fmt.bias - 3)):
            xe = ey + ts + off
            if 1 <= xe < fmt.exp_inf:
                for sy in (0, 1):
                    for yf in yfr:
                        for xf in (0, nf // 2, nf - 1):
                            pairs.add((fmt.normal(sy, ey, yf).bits, fmt.normal(0, xe, xf).bits))

    # Near-axis ladder |y| = |x| * 2**-j over the whole range, with random mantissas. The straddles above pin the two
    # known seams; this walks the corridor between them, where theta's relative accuracy degrades smoothly rather
    # than at a named threshold.
    xe = fmt.bias
    for j in range(0, fmt.wman + 7):
        ye = xe - j
        if ye < 1:  # a narrow WEXP runs out of exponents before the ladder runs out of decades
            break
        for _ in range(6):
            yf, xf = int(rng.integers(0, nf)), int(rng.integers(0, nf))
            for sy in (0, 1):
                for sx in (0, 1):
                    pairs.add((fmt.normal(sy, ye, yf).bits, fmt.normal(sx, xe, xf).bits))

    # --- the two ends of the range, which random pairs reach with probability ~0 but the gate has rules for ---
    # OVERFLOW: with x = max_finite, hypot exceeds max_finite for ANY y > 0, and stays below the round-to-inf
    # threshold while y**2 < max*ulp, i.e. y < 2**(emax+1-wman/2). The sweep starts a few binades under that bound
    # and runs to the largest finite exponent, so it covers both sides: the low end has RN == max_finite (a +inf
    # there is the spurious kind the README's overflow contract rules out) and the high end genuinely overflows.
    top = fmt.normal(0, fmt.exp_inf - 1, fmt.frac_mask).bits
    for ye in range(max(1, fmt.exp_inf - 1 - (fmt.wman // 2) - 3), fmt.exp_inf):
        for yf in (0, 1, nf // 2, nf - 1):
            pairs.add((fmt.normal(0, ye, yf).bits, top))
    pairs.add((top, top))  # hypot = sqrt(2)*max: a GENUINE overflow, which must still come back +inf
    # Four mantissas per exponent reaches the packer's round-carry window only by luck -- at several formats it
    # never fired. Solve for one instead, so SATURATE_ROUND_CARRY is what keeps the pair finite.
    pairs.add((saturating_y(fmt), top))

    # UNDERFLOW: theta inside [min_normal/2, min_normal), where the two bracketing encodings are +0 and min_normal.
    for k in (Fraction(1, 2), Fraction(3, 4), Fraction(1, 1)):
        for xe in {e for e in (fmt.bias, fmt.bias + 2, fmt.exp_inf - 2) if 1 <= e < fmt.exp_inf}:
            x = fmt.normal(0, xe, 0)
            target = k * Fraction(1, 1 << (fmt.bias - 1))  # k * min_normal, in turns
            ratio = mp.tan(2 * mp.pi * mp.mpf(target.numerator) / target.denominator)
            y = fmt.encode(x.to_fraction() * Fraction(mp.nstr(ratio, 40)))
            if not (y.is_zero or y.is_inf):
                for sy in (0, 1):
                    pairs.add((fmt.pack(sy, y.exp, y.bits & fmt.frac_mask).bits, x.bits))

    pairs.update(ATAN2_REGRESSIONS.get((fmt.wexp, fmt.wman), []))
    return list(pairs)


def _atan2_unit(case: tuple[int, int]) -> tuple[float, float, int, int, tuple | None]:
    """One (wexp, wman) sweep -> (worst_theta_ulp, worst_mag_ulp, violations, count, first_bad)."""
    import zkf
    import zkf.oracle
    from zkf import ZkfFormat
    from zkf.oracle import _to_mpf

    wexp, wman = case
    fmt = ZkfFormat(wexp, wman)
    pairs = _atan2_pairs(fmt)
    worst_t = worst_m = 0.0
    nbad = 0
    bad = None
    for yb, xb in pairs:
        y, x = zkf.Zkf(fmt, yb), zkf.Zkf(fmt, xb)
        r = y.atan2(x)
        # The wrap-aware metric folds whole turns, so a theta off by a full turn would score 0. Pin the range here.
        assert not r.theta.is_inf and abs(_to_mpf(r.theta)) <= mp.mpf(0.5), f"theta out of range at {yb:#x},{xb:#x}"
        if zkf.oracle.atan2_special(fmt, yb, xb) is not None:  # exact axis/infinity constants, not truth
            t = zkf.oracle.atan2(y, x)
            ok = (r.theta.bits, r.magnitude.bits) == (t.theta.bits, t.magnitude.bits)
            dt = dm = 0.0 if ok else float("inf")
            tok = mok = ok
        else:
            tt, tm = zkf.oracle._atan2_exact(y, x)
            dt, tok, _ = zkf_accuracy.faithful(
                fmt, r.theta.bits, tt, wrap=True
            )  # turns wrap: +1/2 and -1/2 are one angle
            dm, mok, _ = zkf_accuracy.faithful(fmt, r.magnitude.bits, tm)
        if not (tok and mok):
            nbad += 1
            if bad is None:
                bad = (hex(yb), hex(xb), dt, dm)
        worst_t, worst_m = max(worst_t, dt), max(worst_m, dm)
    return worst_t, worst_m, nbad, len(pairs), bad


def _check_atan2() -> list[str]:
    """Same for zkf_atan2 (theta and mag). Returns the per-format failures."""
    import sys
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(REPO))
    # zkf is first-imported here, after --emit has written the tables, so the fork pool inherits the freshly-emitted
    # data without reaching into package internals to reload/clear caches.
    import zkf  # noqa: F401
    import zkf.oracle  # noqa: F401

    print("atan2 end-to-end faithful-rounding check (model vs mpmath):")
    # (5,16) is atan2's WEXP floor, so the accuracy of the narrowest configuration it will elaborate is measured
    # rather than assumed. It stays cheap because the corpus fans over the exponent range, i.e. it is O(2**WEXP) --
    # which is also why nothing above WEXP=11 belongs here: zkf_accuracy.faithful alone costs ~2.6 s/call at WEXP=30.
    atan2_cases = [*CHECK_CASES, (5, 16)]
    assert not (
        ATAN2_REGRESSIONS.keys() - set(atan2_cases)
    ), f"ATAN2_REGRESSIONS entries for unchecked formats: {sorted(ATAN2_REGRESSIONS.keys() - set(atan2_cases))}"
    # The diagonals are the representable thetas the generic path reaches: they must come exact (zkf_accuracy.faithful).
    for wexp, wman in atan2_cases:
        fmt = zkf.ZkfFormat(wexp, wman)
        for e, f in [(fmt.bias, 0), (1, 1), (fmt.exp_inf - 1, fmt.frac_mask), (fmt.bias + 1, fmt.frac_mask // 3)]:
            for ys, xs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                theta, _ = zkf.oracle._atan2_exact(fmt.pack(ys, e, f), fmt.pack(xs, e, f))
                assert theta == mp.mpf((-1) ** ys * (3 if xs else 1)) / 8, f"inexact diagonal theta at {fmt}"
    failures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=_MP_CONTEXT) as ex:
        for (wexp, wman), (worst_t, worst_m, nbad, count, bad) in zip(atan2_cases, ex.map(_atan2_unit, atan2_cases)):
            ok = nbad == 0
            print(
                f"  {'OK ' if ok else 'BAD'} {wexp}/{wman:<3} theta_ulp={worst_t:.3f} mag_ulp={worst_m:.3f} "
                f"not_faithful={nbad} (sampled({count}))"
            )
            if not ok:
                failures.append(
                    f"{wexp}/{wman}: theta_ulp={worst_t:.4f} mag_ulp={worst_m:.4f} not_faithful={nbad}/{count} "
                    f"first_bad={bad}"
                )
    return [f"atan2 {f}" for f in failures]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    gen = ap.add_mutually_exclusive_group()  # --check-emit after --emit would compare the files just written
    gen.add_argument("--emit", action="store_true", help="write the per-WMAN CORDIC constant cores and Python data")
    ap.add_argument(
        "--check", action="store_true", help="verify sincos AND atan2 accuracy vs mpmath (uses the bit-exact model)"
    )
    gen.add_argument(
        "--check-emit", action="store_true", help="fail if the checked-in generated artifacts differ from a fresh run"
    )
    ap.add_argument("--report", action="store_true", help="print the chosen CORDIC shapes (iterations, widths, LUT)")
    args = ap.parse_args()
    if not (args.emit or args.check or args.check_emit or args.report):
        ap.error("nothing to do: pass --emit, --check, --check-emit, and/or --report")

    all_specs = generate_all()
    if args.report or args.emit:
        _report(all_specs)
    if args.emit:
        emit(all_specs)
    if args.check_emit:
        check_emit(all_specs)
    if args.check:
        failures = _check() + _check_atan2()  # both run to completion first: one bad format must hide nothing
        if failures:
            raise SystemExit("not faithfully rounded:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()

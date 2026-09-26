#!/usr/bin/env python3
"""
Coefficient generator for the ZKF transcendental operators ``zkf_log2`` and ``zkf_exp2``.

Both reduce to a per-segment polynomial (truncating fixed-point Horner, ``zkf/rtl/_zkf_horner.v``) on the unit interval:

  * exp2 evaluates ``H(s) = 2**s`` for ``s in [0,1)`` -> the significand ``2**f in [1,2)``.

  * log2 uses SYMMETRIC argument reduction of ``x = m*2**e`` (m in [1,2)): if ``m >= sqrt(2)`` halve m and increment e,
    so ``m' in [sqrt(1/2), sqrt(2))`` and the reduced fraction ``f = m'-1`` is exact (Sterbenz). Fit the smooth kernel
    ``C(f) = log2(1+f)/f`` and return ``log2(m') = f*C(f)``. GOTCHA: the symmetric reduction is what removes the
    catastrophic cancellation of the naive ``m in [1,2)`` reduction as ``x -> 1`` (where ``e`` and ``log2(m)`` cancel).

Degree ``D`` is a closed-form function of ``WMAN`` (so is the pipeline depth); ``K`` (segment count) is then the
smallest meeting the accuracy target. Table content depends on ``WMAN`` only -- the helpers live on the unit interval;
the exponent/integer part is handled by the renormalize/pack stage. ``--emit`` writes the per-WMAN Verilog cores and the
Python data table; ``--check`` verifies both are faithfully rounded against the unrounded ``mpmath`` truth.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from textwrap import dedent

import mpmath as mp

if __package__:  # python -m tools.<generator>
    from . import zkf_accuracy, zkf_emit
else:  # python tools/<generator>.py, or tools/ on sys.path
    import zkf_accuracy
    import zkf_emit

mp.mp.prec = 280  # generous headroom for coefficient fitting and ground-truth rounding

REPO = Path(__file__).resolve().parents[1]
HDL = REPO / "zkf" / "rtl"
TABLES = HDL / "_tables"
PKG_TABLES = REPO / "zkf" / "_tables"

FUNCS = ("exp2", "log2")

# Fixed-point fractional headroom below the WMAN significand: reduced-argument width (exp2 FF = WMAN+GUARD) and the
# coefficient/result scale (CF = WMAN+GUARD). GUARD = ERR_GUARD + 4 = 12 puts the round bit ~7 bits clear of the fit +
# truncating-Horner error floor, keeping the operators faithfully rounded.
GUARD = 12
ERR_GUARD = 8  # helper relative-error budget exponent: target < 2**-(WMAN + ERR_GUARD)

# Max segment-index bits. choose_spec() picks the smallest K <= K_CAP meeting accuracy; K_CAP also sets the closed-form
# degree scale (larger cap -> lower Horner degree/shorter pipeline, wider ROM).
K_CAP = 11
ACC_MARGIN = 1  # extra accumulator bits above the measured maximum, guarding against wrap

# exp2/log2 tables start at WMAN=16 (narrower formats fit the K_CAP=11 geometry but are outside the shipped contract).
WMAN_MIN, WMAN_MAX = 16, 53
SUPPORTED_WMAN = [16, 18, 24, 27, 32, 36, 48, 53]  # FPGA-friendly sizes + the standard IEEE ones

# Random --check draws per non-exhaustive format, per operator (see _random_inputs). UNSEEDED, so repeated runs
# accumulate coverage; ZKF_CHECK_SAMPLES=<n> overrides both.
RANDOM_CHECK_SAMPLES = {
    func: int(os.environ.get("ZKF_CHECK_SAMPLES", default))
    for func, default in (("exp2", "300000"), ("log2", "500000"))
}


def ff_bits(wman: int) -> int:
    return wman + GUARD


def cf_bits(wman: int) -> int:
    return wman + GUARD


# log2 index geometry: the table is indexed by the unsigned v = f + 1/2 in [0.207, 0.914) (strictly inside [0,1), so
# the unit-interval top-K-bits segmenting is reused); per segment we fit C(f) = C(v - 1/2). RTL builds v exactly from
# the stored fraction (no irrational subtraction; model<->RTL identical): m < sqrt(2) -> v = 2**WFRAC + 2*frac, else
# v = frac; the signed combine operand is F = v - 2**WFRAC (= f at scale 2**-(WFRAC+1)).
LOG2_V_OFFSET = mp.mpf(1) / 2  # v = f + LOG2_V_OFFSET maps the signed reduced fraction f into [0,1) for indexing


def log2_sqrt2_threshold(wfrac: int) -> int:
    """
    Integer significand threshold THR for the re-center test m >= sqrt(2) (sig >= THR, sig the WMAN-bit significand).
    Rounding is not critical -- both branches stay within the fitted f range -- it only picks where the split lands.
    GOTCHA: the phase-2 RTL re-center stage must mirror this constant.
    """
    return int(mp.nint(mp.sqrt(2) * (1 << wfrac)))


def degree(wman: int) -> int:
    """
    Per-segment polynomial degree, closed-form in WMAN (so is the Horner pipeline depth). At the finest allowed
    segmentation (2**-K_CAP wide) a segment must approximate the helper to B = WMAN + ERR_GUARD bits, giving
    D = ceil(B/K_CAP) - 1. choose_spec() may pick a smaller K, but D stays fixed by K_CAP so latency tracks WMAN only.
    """
    if not (WMAN_MIN <= wman <= WMAN_MAX):
        raise ValueError(f"Bad {wman=}")
    d = ceil((wman + ERR_GUARD) / K_CAP) - 1
    assert d >= 2, "Maybe WMAN is too small or K_CAP is too large?"
    return d


def table_module(func: str, wman: int) -> str:
    return f"_zkf_{func}_m{wman}"


@dataclass
class Spec:
    func: str  # "exp2" | "log2"
    wman: int
    k: int  # segment-index bits
    seg_base: int  # first ROM segment represented by coeffs; zero for exp2, compacted for log2
    d: int  # polynomial degree
    cf: int  # coefficient fractional bits (scale 2**-cf)
    rw: int  # reduced-argument bits (wn = w / 2**rw); the total argument width is k + rw
    cw: int  # signed coefficient width
    accw: int  # signed Horner accumulator width
    coeffs: list = field(default_factory=list)  # [2**k][d+1] signed ints, low degree first

    @property
    def nseg(self) -> int:
        return len(self.coeffs)


# --------------------------------------------------------------------------------------------------
# Coefficient fitting (high precision via mpmath)
# --------------------------------------------------------------------------------------------------
def helper_true(func: str, arg):
    """
    Exact helper value (mpf). exp2: H(s)=2**s at s in [0,1). log2: the kernel C(f)=log2(1+f)/f at the signed reduced
    fraction ``arg`` = f, with the removable singularity C(0)=1/ln2.
    """
    if func == "exp2":
        return mp.power(2, arg)
    if arg == 0:
        return 1 / mp.log(2)
    return mp.log(1 + arg) / (mp.log(2) * arg)


def helper_arg(func: str, v):
    """Map the unit-interval index coordinate v to the helper argument: exp2 uses v; log2 uses f = v - 1/2."""
    return v if func == "exp2" else v - LOG2_V_OFFSET


def segment_coeffs(func: str, k: int, d: int, cf: int, idx: int) -> list[int]:
    """Near-minimax degree-d coefficients (low degree first, scaled by 2**cf, rounded) for one segment."""
    base = mp.mpf(idx) / (1 << k)
    width = mp.mpf(1) / (1 << k)
    cheb = mp.chebyfit(lambda wn: helper_true(func, helper_arg(func, base + width * wn)), [mp.mpf(0), mp.mpf(1)], d + 1)
    scale = mp.mpf(1 << cf)
    return [int(mp.nint(c * scale)) for c in reversed(cheb)]  # chebyfit is highest-degree first


def horner(coeffs_idx: list[int], w: int, rw: int) -> int:
    """Truncating fixed-point Horner, bit-identical to zkf/rtl/_zkf_horner.v. Returns acc at scale 2**-cf."""
    acc = coeffs_idx[-1]
    for c in reversed(coeffs_idx[:-1]):
        acc = c + ((acc * w) >> rw)  # Python >> floors, matching arithmetic >>> on signed
    return acc


def _log2_reachable_v_bounds(wman: int) -> tuple[int, int]:
    """Inclusive bounds for the log2 table-index coordinate v over all positive finite inputs."""
    wfrac = wman - 1
    half = 1 << wfrac
    thr = log2_sqrt2_threshold(wfrac)
    min_v = thr - half
    max_v = (2 * thr) - half - 2
    assert 0 <= min_v <= half <= max_v < (1 << wman)
    return min_v, max_v


def _segment_span(func: str, wman: int, k: int) -> tuple[int, int]:
    """Return (first segment index, number of segment rows) needed by this function."""
    if func == "exp2":
        return 0, 1 << k
    min_v, max_v = _log2_reachable_v_bounds(wman)
    rw = wman - k
    seg_base = min_v >> rw
    seg_last = max_v >> rw
    return seg_base, seg_last - seg_base + 1


def _arg_grid(func: str, wman: int, width: int, k: int, seg_base: int, nseg: int):
    """Argument values to probe: exhaustive when small, else dense segment-local sampling."""
    if func == "log2":
        min_v, max_v = _log2_reachable_v_bounds(wman)
        if width <= 16:
            return range(min_v, max_v + 1)
        rw = width - k
        span = 1 << rw
        probes = sorted({0, span // 7, span // 4, span // 2, (5 * span) // 7, (3 * span) // 4, span - 1})
        out = []
        for idx in range(seg_base, seg_base + nseg):
            for w in probes:
                a = (idx << rw) | w
                if min_v <= a <= max_v:
                    out.append(a)
        return out
    if width <= 16:
        return range(1 << width)
    rw = width - k
    span = 1 << rw
    probes = sorted({0, span // 7, span // 4, span // 2, (5 * span) // 7, (3 * span) // 4, span - 1})
    return [(idx << rw) | w for idx in range(1 << k) for w in probes]


def measure(func: str, wman: int, k: int, seg_base: int, cf: int, width: int, coeffs: list[list[int]]):
    """
    Return (max relative helper error, max |accumulator| seen) over the probe grid, exercising the truncating
    Horner so that the accuracy target covers the arithmetic the hardware performs.
    """
    rw = width - k
    scale = mp.mpf(1 << cf)
    max_rel, max_acc = mp.mpf(0), 0
    for a in _arg_grid(func, wman, width, k, seg_base, len(coeffs)):
        idx, w = (a >> rw) - seg_base, a & ((1 << rw) - 1)
        ci = coeffs[idx]
        acc = ci[-1]
        max_acc = max(max_acc, abs(acc))
        for c in reversed(ci[:-1]):
            acc = c + ((acc * w) >> rw)
            max_acc = max(max_acc, abs(acc))
        approx = mp.mpf(acc) / scale
        true = helper_true(func, helper_arg(func, mp.mpf(a) / (1 << width)))
        max_rel = max(max_rel, abs(approx / true - 1))
    return max_rel, max_acc


def choose_spec(func: str, wman: int) -> Spec:
    """
    Degree is fixed by degree() (closed-form in WMAN); search K in 1..K_CAP for the smallest ROM meeting the accuracy
    target. Accuracy is measured THROUGH the truncating Horner (on a sampled grid above 16 bits); --check verifies
    faithful rounding end to end.
    """
    cf = cf_bits(wman)
    # Reduced-argument (index coordinate) width: exp2 FF = WMAN+GUARD; log2 uses WMAN (v is a WFRAC+1 = WMAN-bit fraction).
    width = ff_bits(wman) if func == "exp2" else wman
    d = degree(wman)
    target = mp.mpf(2) ** (-(wman + ERR_GUARD))  # relative helper-error budget

    for k in range(1, K_CAP + 1):
        seg_base, nseg = _segment_span(func, wman, k)
        coeffs = [segment_coeffs(func, k, d, cf, idx) for idx in range(seg_base, seg_base + nseg)]
        rel, max_acc = measure(func, wman, k, seg_base, cf, width, coeffs)
        if rel < target:
            maxabs = max(abs(c) for seg in coeffs for c in seg)
            cw = maxabs.bit_length() + 2  # +1 sign, +1 margin
            accw = max(max_acc, maxabs).bit_length() + 1 + ACC_MARGIN
            if func == "log2":
                # The final multiply trims the Horner acc to ACCM = CF+2 bits, lossless only because C(f) < 2 (acc <
                # 2**(CF+1)). Assert here so a kernel/range change that broke the bound fails at generation, not silently.
                assert max_acc < (1 << (cf + 2)), (
                    f"log2 WMAN={wman}: max Horner acc {max_acc} >= 2**ACCM (2**{cf + 2}); "
                    f"the CF+2 final-multiply trim would lose bits -- widen ACCM in _emit_table"
                )
            if func == "exp2":
                # exp2's Horner is emitted UNSIGNED (ACC_SIGNED=0), valid only if every coefficient is >= 0 (then acc
                # stays >= 0 by induction). chebyfit is minimax, not Taylor, so it COULD emit a negative coefficient --
                # prove the premise here; a fired assert means this format needs the signed grid.
                assert all(c >= 0 for seg in coeffs for c in seg), (
                    f"exp2 WMAN={wman}: a fitted coefficient is negative; the unsigned Horner grid "
                    f"(ACC_SIGNED=0) would produce wrong bits -- make _zkf_horner signed for exp2 again"
                )
                # The significand is the slice acc[CF -: WMAN], so 2**s must stay below 2 (acc < 2**(CF+1)) as s -> 1;
                # a carry into bit CF+1 would drop the hidden bit silently in both model and RTL.
                assert max_acc < (1 << (cf + 1)), (
                    f"exp2 WMAN={wman}: max Horner acc {max_acc} >= 2**(CF+1) (2**{cf + 1}); "
                    f"the acc[CF -: WMAN] significand slice would wrap"
                )
            return Spec(func, wman, k, seg_base, d, cf, width - k, cw, accw, coeffs)
    raise RuntimeError(f"degree {d} needs K>K_CAP={K_CAP} for {func} WMAN={wman}: raise K_CAP")


def generate_all() -> dict[tuple[str, int], Spec]:
    return {(func, wman): choose_spec(func, wman) for func in FUNCS for wman in SUPPORTED_WMAN}


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


def _rom_word(s: Spec, coeffs: list[int]) -> str:
    """Return one packed coefficient word {c[D], ..., c[0]} as a Verilog concatenation."""
    return "{" + ", ".join(f"{s.cw}'h{c & ((1 << s.cw) - 1):0{(s.cw + 3) // 4}x}" for c in reversed(coeffs)) + "}"


def _rom_read_pipeline(
    w: _Writer,
    s: Spec,
    *,
    valid_expr: str = "in_valid",
    idx_expr: str = "idx",
    arg_expr: str = "w",
    sb_load: str,
    sb_width: str,
) -> None:
    """
    Emit the registered coefficient lookup. The ROM itself is a single initialized array. Its read register is followed
    by a mandatory fabric register that isolates the ROM clk-to-q from the first Horner multiply without changing the
    staging of every Horner product.
    """
    source_valid = valid_expr
    source_arg = arg_expr
    source_sb = sb_load
    select_expr = idx_expr
    sb_range = "" if sb_width == "1" else f"[{sb_width}-1:0] "
    roww = max(1, len(str(s.nseg - 1)))
    # exp2's coefficients are all non-negative and the truncating Horner preserves non-negativity, so its accumulator
    # multiply maps to the cheaper fully-unsigned DSP grid; log2 keeps the signed accumulator (sign-alternating coeffs).
    acc_signed = 0 if s.func == "exp2" else 1

    w("`ZKF_ATTRIBUTE_ROM_PRE reg [(D+1)*CW-1:0] rom [0:NSEG-1] `ZKF_ATTRIBUTE_ROM_POST;")
    w("initial begin")
    w.push()
    for row, coeffs in enumerate(s.coeffs):
        w(f"rom[{row:{roww}d}] = {_rom_word(s, coeffs)};")
    w.pop()
    w("end")
    w(f"""
        reg [(D+1)*CW-1:0] r_co1, r_co2;
        reg [RW-1:0] r_w1, r_w2;
        reg r_rv1, r_rv2;
        reg {sb_range}r_rsb1, r_rsb2;
        always @(posedge clk) begin
    """)
    w.push()
    w("if (rst) begin r_rv1 <= 1'b0; r_rv2 <= 1'b0; end")
    w(f"else begin r_rv1 <= {source_valid}; r_rv2 <= r_rv1; end")
    w(f"r_co1  <= rom[{select_expr}];")
    w(f"r_w1   <= {source_arg};")
    w(f"r_rsb1 <= {source_sb};")
    w("r_co2  <= r_co1;")
    w("r_w2   <= r_w1;")
    w("r_rsb2 <= r_rsb1;")
    w.pop()
    w("end")
    wout = "()" if s.func == "exp2" else "(ew)"
    ew_decl = "wire          [RW-1:0] ew;" if s.func == "log2" else ""
    w(f"""
        wire signed [ACCW-1:0] acc;
        wire                   ev;
        wire         {sb_range}esb;
        {ew_decl}
        _zkf_horner #(
            .D(D), .WCOEF(CW), .WRARG(RW), .WACC(ACCW), .WSB({sb_width}), .ACC_SIGNED({acc_signed}),
            .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
        ) u_h (
            .clk(clk), .rst(rst), .in_valid(r_rv2), .sb_in(r_rsb2), .coeffs(r_co2), .w(r_w2),
            .out_valid(ev), .sb_out(esb), .w_out{wout}, .acc(acc)
        );
    """)


def _emit_table(s: Spec) -> str:
    """
    One self-contained per-WMAN evaluation core. Shape (K/CF/RW/CW/ACCW) is baked; the degree D is a parameter
    defaulting to this ROM's fitted degree, which zkf_<func>.v drives with its own closed-form degree and the table
    asserts (mirroring the LATENCY parameter). zkf_<func>.v selects the module whose name matches its WMAN.
    """
    mod = table_module(s.func, s.wman)
    w = _Writer()
    w("/// GENERATED by zkf_transcendental.py -- DO NOT EDIT.")
    if s.func == "exp2":
        w(
            f"/// Table+polynomial core for zkf_exp2 at WMAN={s.wman} (degree {s.d}); zero-bubble, see _zkf_horner.",
            "/// Evaluates the significand 2**f in [1,2) from the reduced fractional argument f (FF = WMAN + 12 bits).",
            "/// Register stages: two-stage ROM read + D*(2+STAGE_PRODUCT) (Horner).",
        )
    else:
        w(
            f"/// Table+polynomial core for zkf_log2 at WMAN={s.wman} (degree {s.d}); zero-bubble, see _zkf_horner.",
            "/// Symmetric reduction: indexes C(f)=log2(1+f)/f by unsigned v (WFRAC+1 bits) and returns log2(m')=f*C(f) as",
            "/// the unsigned magnitude l_mag = |f|*C(f) plus the sign l_neg (set when m>=sqrt(2)), at scale 2**-F2,",
            "/// F2 = WFRAC+1+CF. The final f*C(f) multiply is fully unsigned (|f|*C(f)); the caller folds l_neg into its",
            "/// reconstruction add/subtract. Register stages: two-stage ROM read + Horner + final multiply.",
        )
    w("")
    w("// verilog_lint: waive-start line-length  (the ROM rows are wide one-liners)")
    w("")
    w("`default_nettype none")
    w("")
    w("`ifndef ZKF_ATTRIBUTE_ROM_PRE")
    w("`define ZKF_ATTRIBUTE_ROM_PRE")
    w("`define ZKF_ATTRIBUTE_ROM_PRE_DEFAULTED")
    w("`endif")
    w("`ifndef ZKF_ATTRIBUTE_ROM_POST")
    w("`define ZKF_ATTRIBUTE_ROM_POST")
    w("`define ZKF_ATTRIBUTE_ROM_POST_DEFAULTED")
    w("`endif")
    w("")
    if s.func == "exp2":
        w(f"module {mod} #(")
        w.push()
        w(f"parameter D             = {s.d},")
        w("parameter WSB           = 1,")
        w("parameter WMULTIPLIER   = 0,")
        w("parameter STAGE_PRODUCT = 0")
        w.pop()
        w(") (")
    else:
        w(f"module {mod} #(")
        w.push()
        w(f"parameter D                   = {s.d},")
        w("parameter WSB                 = 1,")
        w("parameter WMULTIPLIER         = 0,")
        w("parameter STAGE_PRODUCT       = 0,")
        w("parameter STAGE_PRODUCT_FINAL = STAGE_PRODUCT")
        w.pop()
        w(") (")
    w.push()
    if s.func == "exp2":
        w(f"""
            input  wire           clk,
            input  wire           rst,
            input  wire           in_valid,
            input  wire [WSB-1:0] sb_in,
            input  wire [{s.wman + 12:3}-1:0] f,            // FF = WMAN + 12 reduced-argument fraction bits, in [0,1)
            output wire           out_valid,
            output wire [WSB-1:0] sb_out,
            output wire [{s.wman:3}-1:0] significand,  // 2**f in [1,2): hidden bit + WFRAC fraction
            output wire           guard,
            output wire           round,
            output wire           sticky
        """)
    else:
        w(f"""
            input  wire               clk,
            input  wire               rst,
            input  wire               in_valid,
            input  wire     [WSB-1:0] sb_in,
            input  wire     [{s.wman:3}-1:0] v,  // index coordinate v = f + 2**WFRAC (WFRAC+1 = WMAN bits)
            output wire               out_valid,
            output wire     [WSB-1:0] sb_out,
            output wire        [{2 * s.wman + 12}:0] l_mag,  // |log2(m')| = |f|*C(f) magnitude at scale 2**-F2, F2 = WFRAC+1+CF
            output wire               l_neg   // sign of log2(m') (= sign of reduced f); 1 when m >= sqrt(2)
        """)
    w.pop()
    w(");")
    w.push()
    # Degree contract (mirrors the LATENCY parameter): D defaults to this ROM's fitted degree and zkf_<func>.v drives
    # it with its own closed-form degree; a mismatch fails elaboration, so the Horner pipeline depth -- hence the
    # operator latency -- cannot silently drift from the degree the ROM was actually fitted for.
    w(f"generate if (D != {s.d}) begin : g_degree_mismatch  _zkf_invalid_degree_mismatch u_invalid(); end endgenerate")
    # Shape localparams (baked); the public module hard-codes the matching FF/CF so it need not know K.
    w(f"localparam integer WMAN = {s.wman};")
    if s.func == "exp2":
        w("localparam integer FF   = WMAN + 12;")
    else:
        w(
            "localparam integer WFRAC = WMAN - 1;",
            "localparam integer CF    = WMAN + 12;",
            "localparam integer F2    = WFRAC + 1 + CF;   // f is at scale 2**-(WFRAC+1): one bit wider than WFRAC+CF",
            "localparam integer WF    = WMAN;             // signed reduced-argument width = WFRAC + 1",
            "localparam integer ACCM  = CF + 2;           // C(f) < 2 -> acc < 2**(CF+1); trim the Horner guard bits so",
        )
        w("                                            // the final multiply's acc operand fits a smaller DSP grid")
    w(f"localparam integer K    = {s.k};")
    if s.func == "exp2":
        w(f"localparam integer CF   = {s.cf};")
    if s.func == "log2":
        w(f"localparam integer SEG_BASE = {s.seg_base};")
    w(f"localparam integer RW   = {s.rw};")
    w(f"localparam integer CW   = {s.cw};")
    w(f"localparam integer ACCW = {s.accw};")
    w(f"localparam integer NSEG = {s.nseg};")
    w("localparam integer WIDX = (NSEG <= 1) ? 1 : $clog2(NSEG);")
    w("")
    if s.func == "exp2":
        # NSEG == 2**K for every exp2 table, so WIDX == K and idx is the top WIDX bits of f directly.
        w("wire [WIDX-1:0] idx = f[FF-1 -: WIDX];")
        w("wire [RW-1:0]   w   = f[RW-1:0];")
        # The external sideband rides the two ROM-read registers and the Horner sideband; that delay is exactly
        # 2 + D*(2+STAGE_PRODUCT) cycles, so esb lands aligned with out_valid (= ev) -- no separate delay line needed.
        _rom_read_pipeline(w, s, sb_load="sb_in", sb_width="WSB")
        # acc scale 2^-CF, value 2^f in [1,2): bit CF is the hidden one. Output is combinational after the Horner.
        w("""
            assign sb_out      = esb;
            assign significand = acc[CF -: WMAN];
            assign guard       = acc[CF-WMAN];
            assign round       = acc[CF-WMAN-1];
            assign sticky      = |acc[CF-WMAN-2:0];
            assign out_valid   = ev;
        """)
    else:
        # The unsigned coordinate v (WFRAC+1 = WMAN bits) selects the segment by its top K bits and feeds the in-segment
        # Horner argument w by its low RW bits; idx_raw/w mirror the model's `v >> rw` / `v & mask(rw)` exactly. The ROM
        # stores only the continuous span of reachable log2 rows, so idx subtracts SEG_BASE. idx_raw rides the Horner
        # sideband (low K bits) packed together with the external sideband (high WSB bits); _zkf_horner returns delayed
        # ew so the full delayed v, then signed f, can be reconstructed. The external sideband continues through the
        # final multiply so sb_out lands aligned with out_valid; no separate delay line is needed.
        w("wire [K-1:0]    idx_raw = v[WMAN-1 -: K];")
        w("wire [K-1:0]    idx_ofs = idx_raw - SEG_BASE[K-1:0];")
        w("wire [WIDX-1:0] idx     = idx_ofs[WIDX-1:0];")
        w("wire [RW-1:0]   w       = v[RW-1:0];")
        _rom_read_pipeline(w, s, sb_load="{sb_in, idx_raw}", sb_width="(K + WSB)")
        # log2(m') = f * C(f), SIGNED (f < 0 when m >= sqrt(2); acc = C(f) > 0), at scale 2^-F2. acc is trimmed to
        # ACCM = CF+2 bits (its high guard bits are structurally zero since C(f) < 2) so the multiply maps to a smaller
        # DSP grid. Instead of a signed*unsigned product (whose signed slice grid wastes a bit per tile, costing ~1/3
        # more DSPs), multiply the UNSIGNED magnitude |f|*C(f) on the cheaper fully-unsigned grid, carry f's sign bit
        # through the pmul sideband (same latency), and restore it combinationally: two's-complement negation commutes
        # with the [F2:0] truncation, so the result is bit-identical to the old signed product. This adds a same-cycle
        # negate cone after the final product but no latency and no ROM. The final multiply has its own split-depth knob
        # because its operands are wider than the Horner acc*w product, and it carries the external sideband to its
        # output. |f| < 2**(WF-1) over the reduced range, so the WF-bit unsigned magnitude has a structural top zero.
        w("""
            wire        [K-1:0]    idx_p = esb[K-1:0];
            wire        [WSB-1:0]  sb_p  = esb[K +: WSB];
            wire        [WMAN-1:0] v_p   = {idx_p, ew};
            wire signed [WF-1:0]   f_p   = $signed({~v_p[WMAN-1], v_p[WMAN-2:0]});
            wire                   f_neg = f_p[WF-1];
            wire        [WF-1:0]   mag_f = f_neg ? (-f_p) : f_p;   // |f|, fits WF-1 bits (top bit structurally 0)

            wire [WF+ACCM-1:0] umag_p;   // unsigned |f| * C(f)
            wire [WSB:0]       fsb;      // {delayed f sign, external sideband} riding the pmul
            _zkf_pmul #(
                .WA(WF), .WB(ACCM), .A_SIGNED(0), .B_SIGNED(0),
                .WSB(WSB + 1), .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT_FINAL)
            ) u_final_pmul (
                .clk(clk), .rst(rst), .in_valid(ev), .sb_in({f_neg, sb_p}),
                .a(mag_f), .b(acc[ACCM-1:0]),
                .out_valid(out_valid), .sb_out(fsb), .p(umag_p)
            );
            assign sb_out = fsb[WSB-1:0];
            // Emit the magnitude and the sign SEPARATELY rather than a signed product: the caller folds the sign into
            // its e + log2(m') reconstruction as a single add/subtract, so the post-product cone carries one fewer wide
            // negate carry chain (a standalone two's-complement negate here would otherwise sit in series with that
            // adder and the magnitude abs, the critical path on wide formats). umag_p is the exact |f|*C(f): C(f) < 2
            // and |f| < 2**(WF-1) bound it below 2**(F2-1), so the F2+1-bit slice is lossless. The sign rode the pmul
            // sideband (fsb[WSB]) and so is aligned with out_valid.
            assign l_mag = umag_p[F2:0];
            assign l_neg = fsb[WSB];
        """)
    w.pop()
    w("endmodule")
    w("")
    w("`ifdef ZKF_ATTRIBUTE_ROM_PRE_DEFAULTED")
    w("`undef ZKF_ATTRIBUTE_ROM_PRE")
    w("`undef ZKF_ATTRIBUTE_ROM_PRE_DEFAULTED")
    w("`endif")
    w("`ifdef ZKF_ATTRIBUTE_ROM_POST_DEFAULTED")
    w("`undef ZKF_ATTRIBUTE_ROM_POST")
    w("`undef ZKF_ATTRIBUTE_ROM_POST_DEFAULTED")
    w("`endif")
    w("")
    w("// verilog_lint: waive-stop line-length")
    w("`default_nettype wire")
    return w.render()


def _emit_python(all_specs: dict[tuple[str, int], Spec]) -> str:
    w = _Writer()
    w("# GENERATED by zkf_transcendental.py -- DO NOT EDIT.")
    w('"""Bit-exact table+polynomial data for zkf_log2 / zkf_exp2, consumed by the zkf package."""')
    w("")
    w("SPECS = {")
    w.push()
    for func, wman in sorted(all_specs):
        s = all_specs[(func, wman)]
        w(f"({func!r}, {wman}): dict(")
        w.push()
        w(f"k={s.k}, seg_base={s.seg_base}, d={s.d}, cf={s.cf}, rw={s.rw}, cw={s.cw}, accw={s.accw},")
        w(f"coeffs={s.coeffs!r},")
        w.pop()
        w("),")
    w.pop()
    w("}")
    return w.render()


def _artifacts(all_specs: dict[tuple[str, int], Spec]) -> dict[Path, str]:
    arts = {TABLES / f"{table_module(func, wman)}.v": _emit_table(s) for (func, wman), s in sorted(all_specs.items())}
    return arts | {PKG_TABLES / "trans.py": _emit_python(all_specs)}


def emit(all_specs: dict[tuple[str, int], Spec]) -> None:
    zkf_emit.write(_artifacts(all_specs))


def check_emit(all_specs: dict[tuple[str, int], Spec]) -> None:
    found = sorted(p for f in FUNCS for p in TABLES.glob(f"_zkf_{f}_m*.v"))
    zkf_emit.verify(_artifacts(all_specs), found, "zkf_transcendental.py --emit")


# --------------------------------------------------------------------------------------------------
# Reporting and accuracy check
# --------------------------------------------------------------------------------------------------
def _report(all_specs: dict[tuple[str, int], Spec]) -> None:
    print(
        f"{'func':5} {'WMAN':>4} {'K':>3} {'base':>5} {'rows':>5} {'D':>3} {'CF':>4} {'RW':>4} "
        f"{'CW':>4} {'ACCW':>5} {'entries':>8} {'ROM_kbit':>9}"
    )
    for (func, wman), s in sorted(all_specs.items()):
        entries = s.nseg * (s.d + 1)
        print(
            f"{func:5} {wman:>4} {s.k:>3} {s.seg_base:>5} {s.nseg:>5} {s.d:>3} {s.cf:>4} "
            f"{s.rw:>4} {s.cw:>4} {s.accw:>5} {entries:>8} {entries * s.cw / 1024.0:>9.1f}"
        )

    # zkf_<func>.v derives D = (WMAN+18)/11 - 1 closed-form at elaboration (matching degree()); this map cross-checks it,
    # identical for both operators.
    d_map = " ".join(f"{wman}:{degree(wman)}" for wman in range(WMAN_MIN, WMAN_MAX + 1))
    print(
        f"\nclosed-form degree D = (WMAN+18)/11 - 1 derived in zkf_<func>.v "
        f"({WMAN_MAX - WMAN_MIN + 1} values, same for both):\n  {d_map}"
    )


# fork lets the pool inherit the freshly-emitted tables; fall back to the default context where fork is unavailable.
_MP_CONTEXT = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)

# --check formats: the narrow exhaustive instruments, every supported WMAN, and IEEE binary64. Formats of at most 2**22
# encodings run exhaustively; the rest draw RANDOM_CHECK_SAMPLES stratified inputs per operator.
CHECK_CASES = [(2, 16), (3, 16), (6, 18), (8, 24), (8, 27), (8, 32), (8, 36), (8, 48), (8, 53), (11, 53)]
# The deterministic units (table seams, exp2 binade crossings, log2 near 1) run at these formats.
UNIT_CASES = [(8, wman) for wman in SUPPORTED_WMAN] + [(11, 53)]
# Worst error allowed on top of faithfulness, in ULP. Faithfulness already implies < 1, so this fires on its own only
# once lowered -- to make a drift toward the faithful bound fail rather than just print. Inclusive, since the error is a
# float: a faithful result just under 1 ULP away can read 1.0.
ERR_CEILING = 1.0
_SHARD = 1 << 16  # inputs per pool work unit


def _exp2_truth(v):
    """
    exp2's unrounded value for the metric. Below 2**-(2*WMAN+40), 2**x lies within 2**-(2*WMAN+39) of 1 -- far inside
    one rounding interval on either side -- and deep enough (at (11,53), below 2**-291) the fixed working precision
    collapses it onto 1, which would accept only 1. A same-sign proxy of that magnitude stands in: same RN, faithful set
    and verdicts. More precision is no fix: the metric accepts at most 4*WMAN+80 bits.
    """
    import zkf.oracle

    wman = v.fmt.wman
    if v.exp - v.fmt.bias >= -(2 * wman + 40):
        return zkf.oracle._exp2_exact(v)
    with mp.workprec(4 * wman + 80):
        return mp.power(2, mp.ldexp(-1 if v.negative else 1, -(2 * wman + 40)))


def _verdict(func: str, v, got_bits: int, flags: tuple[bool, bool] = (False, False)) -> tuple[float, bool, bool]:
    """
    (error in ULP, faithful, differs from RN) of the result ``got_bits`` -- and, for log2, its (domain_error, pole) --
    for the input ``v``. Contract inputs (exp2: +-inf, zero, |x| >= 2**(WEXP-1); log2: +inf, zero, x < 0) must match
    the oracle exactly; the rest are judged against the unrounded truth.
    """
    import zkf.oracle

    fmt = v.fmt
    if func == "exp2":
        if v.is_inf or v.is_zero or v.exp - fmt.bias >= fmt.wexp - 1:
            ok = got_bits == zkf.oracle.exp2(v).bits
            return (0.0 if ok else float("inf")), ok, not ok
        truth = _exp2_truth(v)
    else:
        if v.is_inf or v.is_zero or v.negative:
            want = zkf.oracle.log2(v)
            ok = (got_bits, *flags) == (want.value.bits, want.domain_error, want.pole)
            return (0.0 if ok else float("inf")), ok, not ok
        truth = zkf.oracle._log2_exact(v)
    err, ok, rn = zkf_accuracy.faithful(fmt, got_bits, truth)
    return err, ok and not any(flags), got_bits != rn


def _near(fmt, bases, span: int) -> list[int]:
    """Every encoding within ``span`` steps of each base, staying on the base's side of zero."""
    out = set()
    for b in bases:
        sign = b & (1 << fmt.sign_shift)
        mag = b ^ sign
        out.update(
            sign | m for m in range(max(1 << fmt.wfrac, mag - span), min(mag + span + 1, fmt.exp_inf << fmt.wfrac))
        )
    return sorted(out)


def _edge_inputs(fmt) -> list[int]:
    """+-inf, +-0, zero and infinity exponents carrying a fraction, and exp2's range cut with its in-range side."""
    top = fmt.bias + fmt.wexp - 1  # exp2: |x| >= 2**(WEXP-1) is out of range
    out = []
    for s in (0, 1):
        for e, f in [
            (0, 0),
            (0, 1),
            (fmt.exp_inf, 0),
            (fmt.exp_inf, fmt.frac_mask),
            (top, 0),
            (top - 1, fmt.frac_mask),
        ]:
            out.append(fmt.pack(s, e, f).bits)
    return out


def _random_inputs(func: str, fmt, n: int) -> list[int]:
    """
    ``n`` draws, uniform in the fraction and stratified in the exponent. exp2: both signs over [-FF, WEXP-2], where the
    reduced argument is nonzero (below it _boundary_inputs keeps a stratum). log2: x > 0, half the draws at |e| <= 2,
    where the kernel's error shows, half over every normal exponent.
    """
    import numpy as np

    rng = np.random.default_rng()
    frac = rng.integers(0, 1 << fmt.wfrac, n, dtype=np.int64)
    if func == "exp2":
        e = rng.integers(max(-ff_bits(fmt.wman), 1 - fmt.bias), fmt.wexp - 1, n)
        sign = rng.integers(0, 2, n)
    else:
        e = np.where(rng.random(n) < 0.5, rng.integers(-2, 3, n), rng.integers(1 - fmt.bias, fmt.bias + 1, n))
        e = np.clip(e, 1 - fmt.bias, fmt.bias)
        sign = np.zeros(n, dtype=np.int64)
    return [(int(s) << fmt.sign_shift) | ((int(x) + fmt.bias) << fmt.wfrac) | int(f) for s, x, f in zip(sign, e, frac)]


def _boundary_inputs(fmt) -> list[int]:
    """
    exp2: a band around every integer x (binade crossings; the ends include the saturation edges); bands at the
    crossings of the representables next to 1 (x = log2(1 + k*2**-WFRAC), log2(1 - k*2**-WMAN)), where a faithfulness
    gate is sharpest; x = -2**-FF, where the Horner accumulator peaks; and a stratum per exponent below -FF.
    """
    from zkf.oracle import _mpf_to_fraction

    eb = int(os.environ.get("ZKF_EXP2_BAND", "48"))
    ff = ff_bits(fmt.wman)
    bases = [fmt.encode(n).bits for n in range(1 - (1 << (fmt.wexp - 1)), 1 << (fmt.wexp - 1)) if n]
    with mp.workprec(4 * fmt.wman + 80):
        for k in range(1, 5):
            for x in (mp.log(1 + mp.ldexp(k, -fmt.wfrac), 2), mp.log(1 - mp.ldexp(k, -fmt.wman), 2)):
                bases.append(fmt.encode(_mpf_to_fraction(x)).bits)
    out = _near(fmt, bases, eb) + _near(fmt, [fmt.pack(1, fmt.bias - ff, 0).bits], 4)
    for e in range(1 - fmt.bias, -ff):
        out += [fmt.pack(s, fmt.bias + e, f).bits for s in (0, 1) for f in (0, 1, 1 << (fmt.wfrac - 1), fmt.frac_mask)]
    return out


def _seam_inputs(func: str, fmt) -> list[int]:
    """
    Neighborhoods of every reachable table-segment boundary, where adjacent fits meet, and of log2's sqrt(2) reduction
    threshold. exp2 selects the segment by frac(x) (boundaries at x = +-j/2**K); log2 by v = 2**WFRAC + 2*frac below
    the threshold and v = frac at or above it, in the binades on either side of 1.
    """
    from fractions import Fraction

    from zkf._reference import trans_spec, trans_sqrt2_threshold

    spec = trans_spec(func, fmt.wman)
    k, rw = spec["k"], spec["rw"]
    if func == "exp2":
        bases = [fmt.encode(Fraction(j, 1 << k) - d).bits for j in range(1, 1 << k) for d in (0, 1)]
    else:
        half, thr = 1 << fmt.wfrac, trans_sqrt2_threshold(fmt.wfrac)
        fracs = {thr - half}
        for vb in range(0, 1 << fmt.wman, 1 << rw):
            if half <= vb and (vb - half) // 2 < thr - half:
                fracs.add((vb - half) // 2)
            if thr - half <= vb < half:
                fracs.add(vb)
        bases = [fmt.pack(0, fmt.bias + e, f).bits for e in (-1, 0) for f in sorted(fracs)]
    return _near(fmt, bases, 16)


def _near1_inputs(fmt) -> list[int]:
    """
    log2: the fractions straddling x = 1 (the cancellation the symmetric reduction removed -- random draws never hit
    them), a log-uniform ladder x = 1 +- 2**-j and x = 2**(+-2**-j) out to where they meet, and x = 2**(+-2**j),
    min_normal and max_finite, whose results cross a power of two.
    """
    from fractions import Fraction

    from zkf.oracle import _mpf_to_fraction

    near1 = int(os.environ.get("ZKF_NEAR1_SAMPLES", str(1 << 16)))
    span = min(near1, 1 << fmt.wfrac)
    fmax = (1 << fmt.wfrac) - 1
    out = [((fmt.bias - 1) << fmt.wfrac) | f for f in range(fmax - span + 1, fmax + 1)]
    out += [(fmt.bias << fmt.wfrac) | f for f in range(span)]
    bases = [fmt.pack(0, 1, 0).bits, fmt.pack(0, fmt.exp_inf - 1, fmt.frac_mask).bits]
    bases += [fmt.encode(Fraction(2) ** (s * (1 << j))).bits for j in range(fmt.wexp - 1) for s in (1, -1)]
    with mp.workprec(4 * fmt.wman + 80):
        for j in range(1, fmt.wman):
            for x in (
                1 + mp.ldexp(1, -j),
                1 - mp.ldexp(1, -j),
                mp.power(2, mp.ldexp(1, -j)),
                mp.power(2, -mp.ldexp(1, -j)),
            ):
                bases.append(fmt.encode(_mpf_to_fraction(x)).bits)
    return out + _near(fmt, bases, 4)


def _inputs(func: str, wexp: int, wman: int, kind: str, shard: int, nshards: int) -> list[int]:
    from zkf import ZkfFormat

    fmt = ZkfFormat(wexp, wman)
    if kind == "random":
        return _random_inputs(func, fmt, RANDOM_CHECK_SAMPLES[func] // nshards)
    full = {
        "exhaustive": lambda: range(1 << fmt.wfull),
        "edges": lambda: _edge_inputs(fmt),
        "boundary": lambda: _boundary_inputs(fmt),
        "seam": lambda: _seam_inputs(func, fmt),
        "near1": lambda: _near1_inputs(fmt),
    }[kind]()
    return full[shard::nshards]


# The --check sweep runs one worker process per work unit over a fork pool, spreading the mpmath evaluations across all
# cores. Each unit builds its own inputs and returns only its summary, keeping IPC tiny.
def _check_unit(unit: tuple) -> tuple:
    """One work unit -> (unit key, worst ULP, not faithful, differs from RN, count, first bad)."""
    import zkf
    from zkf import ZkfFormat

    func, wexp, wman, kind, shard, nshards = unit
    fmt = ZkfFormat(wexp, wman)
    worst, nbad, nrn, count, bad = 0.0, 0, 0, 0, None
    for b in _inputs(*unit):
        v = zkf.Zkf(fmt, b)
        if func == "exp2":
            err, ok, differs = _verdict(func, v, v.exp2().bits)
        else:
            r = v.log2()
            err, ok, differs = _verdict(func, v, r.value.bits, (r.domain_error, r.pole))
        count += 1
        nrn += differs
        worst = max(worst, err)
        if not ok:
            nbad += 1
            bad = bad or (hex(b), err)
    return (func, wexp, wman, kind), worst, nbad, nrn, count, bad


def _units() -> list[tuple]:
    units = []
    for wexp, wman in CHECK_CASES:
        n = 1 << (wexp + wman)  # 2**WFULL
        kind = "exhaustive" if n <= (1 << 22) else "random"
        for func in FUNCS:
            nshards = max(1, (n if kind == "exhaustive" else RANDOM_CHECK_SAMPLES[func]) // _SHARD)
            units += [(func, wexp, wman, kind, i, nshards) for i in range(nshards)]
            units.append((func, wexp, wman, "edges", 0, 1))
    for wexp, wman in UNIT_CASES:
        for func, kind in [("exp2", "boundary"), ("exp2", "seam"), ("log2", "seam"), ("log2", "near1")]:
            units += [(func, wexp, wman, kind, i, 4) for i in range(4)]
    return units


def _passes(worst: float, not_faithful: int) -> bool:
    return not_faithful == 0 and worst <= ERR_CEILING


def _self_check() -> None:
    """
    The verdict path must reject what the pre-0.7.0 integer-ULP gate passed, and accept what is faithful. Each case
    guards a specific way the metric could silently decay.
    """
    import zkf.oracle
    from zkf import Zkf, ZkfFormat

    zkf_accuracy.self_check()
    f24, f53 = ZkfFormat(8, 24), ZkfFormat(11, 53)
    cases = [
        # Issue #12: one encoding step past RN on the far side, 1.5 ULP from the truth; RN and the near side pass.
        ("exp2", f24, 0x3F0005A1, 0x3FB507B7, False),
        ("exp2", f24, 0x3F0005A1, 0x3FB507B6, True),
        ("exp2", f24, 0x3F0005A1, 0x3FB507B5, True),
        ("log2", f24, 0x3FC00000, 0x3F15C019, False),  # 1.23 ULP, one step from RN
        ("log2", f24, 0x3FC00000, 0x3F15C01A, True),
        # Unfaithful at under 1 ULP across a power of two: catches a metric decayed to err < 1.
        ("exp2", f24, 0x30000000, 0x3F7FFFFF, False),
        ("log2", f24, 0x5F800001, 0x427FFFFF, False),
        # An exact result admits nothing else: log2(8) = 3.
        ("log2", f24, 0x41000000, 0x403FFFFF, False),
        ("log2", f24, 0x41000000, 0x40400001, False),
        # Tiny x at (11,53): 2**x sits just above (below) 1, so 1 and its neighbor on that side are faithful.
        ("exp2", f53, 0x0010000000000000, 0x3FF0000000000001, True),
        ("exp2", f53, 0x0010000000000000, 0x3FEFFFFFFFFFFFFF, False),
        ("exp2", f53, 0x8010000000000000, 0x3FEFFFFFFFFFFFFF, True),
        ("exp2", f53, 0x8010000000000000, 0x3FF0000000000001, False),
        ("exp2", f24, 0xC2FF0000, 0x80000000, False),  # exp2(-127.5) rounds to +0; -0 is not canonical
        ("exp2", f24, 0xC2FF0000, 0x00000000, True),
    ]
    for func, fmt, x, got, want in cases:
        err, ok, _ = _verdict(func, Zkf(fmt, x), got)
        assert ok == want, f"self-check: {func}({x:#x}) -> {got:#x} judged {ok} ({err:.4f} ULP), expected {want}"
    _, ok, _ = _verdict("log2", Zkf(f24, 0x41000000), 0x40400000, (True, False))
    assert not ok, "self-check: a finite log2 result carrying domain_error passed"
    assert _passes(ERR_CEILING, 0), "self-check: a unit at exactly ERR_CEILING fails (see ERR_CEILING)"
    # Representable truths must come exact (see zkf_accuracy.faithful).
    for wexp, wman in sorted(set(CHECK_CASES + UNIT_CASES)):
        fmt = ZkfFormat(wexp, wman)
        for n in range(1 - (1 << (wexp - 1)), 1 << (wexp - 1)):
            if n:
                assert _exp2_truth(fmt.encode(n)) == mp.ldexp(1, n), f"self-check: inexact exp2({n}) at {fmt}"
        for e in range(1 - fmt.bias, fmt.bias + 1):
            assert (
                zkf.oracle._log2_exact(fmt.pack(0, fmt.bias + e, 0)) == e
            ), f"self-check: inexact log2(2**{e}) at {fmt}"


def _check() -> list[str]:
    """End-to-end faithful-rounding check vs mpmath via the bit-exact model. Returns the failures."""
    import sys
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(REPO))
    # zkf is first-imported here, after --emit has written the tables, so the fork pool below inherits the
    # freshly-emitted data without reaching into package internals to reload/clear caches.
    import zkf  # noqa: F401
    import zkf.oracle  # noqa: F401

    _self_check()
    agg = defaultdict(lambda: [0.0, 0, 0, 0, None])
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=_MP_CONTEXT) as ex:
        for key, worst, nbad, nrn, count, bad in ex.map(_check_unit, _units()):
            a = agg[key]
            a[0], a[1], a[2], a[3], a[4] = max(a[0], worst), a[1] + nbad, a[2] + nrn, a[3] + count, a[4] or bad
    print(f"end-to-end faithful-rounding check (model vs mpmath; worst error <= {ERR_CEILING} ULP):")
    failures = []
    for (func, wexp, wman, kind), (worst, nbad, nrn, count, bad) in sorted(agg.items()):
        ok = _passes(worst, nbad)
        print(
            f"  {'OK ' if ok else 'BAD'} {func} {f'{wexp}/{wman}':<5} {kind:<10} worst={worst:.4f} not_faithful={nbad} "
            f"misrounded={nrn}/{count} ({nrn / max(count, 1):.1e})"
        )
        if not ok:
            failures.append(
                f"{func} {wexp}/{wman} {kind}: worst={worst:.4f} not_faithful={nbad}/{count} first_bad={bad}"
            )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    gen = ap.add_mutually_exclusive_group()  # --check-emit after --emit would compare the files just written
    gen.add_argument("--emit", action="store_true", help="write the per-table Verilog cores and Python data table")
    ap.add_argument("--check", action="store_true", help="verify accuracy vs mpmath (uses the bit-exact model)")
    gen.add_argument(
        "--check-emit", action="store_true", help="fail if the checked-in generated artifacts differ from a fresh run"
    )
    ap.add_argument("--report", action="store_true", help="print the chosen table shapes and the degree map")
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
        failures = _check()
        if failures:
            raise SystemExit("not faithfully rounded:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Table generator for the ZKF transcendental operators ``zkf_exp2`` and ``zkf_log2``.

Both evaluate a per-segment polynomial of a helper (exp2: ``2**f``, half an ULP included; log2: the kernel
``C(f) = log2(1+f)/f`` of the re-centered fraction, see ``zkf_log2.v``), by the arithmetic that
``zkf._reference.trans_kernel`` defines.

Degree ``D`` is closed-form in ``WMAN``; ``K`` (segment-index bits) and the operand widths of the products are chosen
here against the error budget. ``--emit`` writes the per-WMAN Verilog cores and the Python data table, after
``zkf_trans_certify.py`` has proven them; ``--check`` verifies the model end to end against ``mpmath``.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil
from pathlib import Path
from textwrap import dedent, wrap

import mpmath as mp

if __package__:  # python -m tools.<generator>
    from . import zkf_accuracy, zkf_emit
    from . import zkf_trans_certify as cert
else:  # python tools/<generator>.py, or tools/ on sys.path
    import zkf_accuracy
    import zkf_emit
    import zkf_trans_certify as cert

mp.mp.prec = 280  # generous headroom for coefficient fitting and ground-truth rounding

REPO = Path(__file__).resolve().parents[1]
HDL = REPO / "zkf" / "rtl"
TABLES = HDL / "_tables"
PKG_TABLES = REPO / "zkf" / "_tables"

FUNCS = ("exp2", "log2")

GUARD, ERR_GUARD = cert.GUARD, cert.ERR_GUARD
LOG2_G2 = 2  # zkf_log2.v's WG2
LOG2_GB = 12  # zkf_log2.v's WGB

# The shape knobs: the table size against fabric and latency. Moving one takes regenerating the tables, which the
# certificate then has to prove again; the arithmetic never depends on a module parameter.
K_CAP = 11  # max segment-index bits; also sets the closed-form degree scale, and with it the length of the chain
K_BASE = 9  # min segment-index bits: one more doubles the table and takes a bit off every operand above c0
# Operand width assumed of one hardware multiplier. Up to degree 3 the operands of the products are cut or fitted to
# it where the error budget allows; no staging depends on it.
NARROW = 18
SQUARE_GUARD = 3  # bits a two-tile square carries below the bits taken from it
SELECT = 0.9  # share of the budget the estimate used for choosing K may take; the certificate proves the whole

WMAN_MIN, WMAN_MAX = 16, 53
SUPPORTED_WMAN = [16, 18, 24, 27, 32, 36, 48, 53]  # FPGA-friendly sizes + the standard IEEE ones

# Random --check draws per non-exhaustive format, per operator; ZKF_CHECK_SAMPLES=<n> overrides both.
RANDOM_CHECK_SAMPLES = {
    func: int(os.environ.get("ZKF_CHECK_SAMPLES", default))
    for func, default in (("exp2", "300000"), ("log2", "500000"))
}


def degree(wman: int) -> int:
    """
    Per-segment polynomial degree, closed-form in WMAN. At the finest allowed segmentation (2**-K_CAP wide) a segment
    must approximate the helper to B = WMAN + ERR_GUARD bits.
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
    """The arithmetic of one table; zkf._reference.trans_kernel explains the operand-width fields."""

    func: str  # "exp2" | "log2"
    wman: int
    k: int  # segment-index bits
    rw: int  # in-segment argument bits
    cf: int  # coefficient fractional bits (scale 2**-cf)
    d: int  # polynomial degree
    t: list[int]
    t2: int
    q: list[int]
    tc: int = 0
    sc: int = 0
    z3: int = 0  # degree 3: low bits that the table's c3 lacks, so that c3*w is one multiplier
    th: int = 0
    sg: int = 0
    thr: int = 0  # log2: first re-centered segment
    coeffs: list = field(default_factory=list)  # [2**k][d+1] magnitudes, low degree first; constants folded into c0

    @property
    def widths(self) -> list[int]:
        return [max(row[j] for row in self.coeffs).bit_length() for j in range(self.d + 1)]

    def as_dict(self) -> dict:
        out = dict(k=self.k, rw=self.rw, cf=self.cf, d=self.d, t=self.t, t2=self.t2, q=self.q)
        if self.tc:
            out |= dict(tc=self.tc, sc=self.sc)
        if self.d == 3:
            out |= dict(z3=self.z3)
        if self.th:
            out |= dict(th=self.th, sg=self.sg)
        if self.func == "log2":
            out |= dict(thr=self.thr, g2=LOG2_G2, gb=LOG2_GB)
        return out | dict(wman=self.wman, coeffs=self.coeffs)


# --------------------------------------------------------------------------------------------------
# Coefficient fitting (high precision via mpmath)
# --------------------------------------------------------------------------------------------------
def log2_threshold(k: int) -> int:
    """First re-centered segment: the segment boundary nearest sqrt(2) - 1 in units of 2**-k."""
    return int(mp.nint((mp.sqrt(2) - 1) * (1 << k)))


def helper_arg(func: str, k: int, thr: int, seg: int, wn):
    """The helper's argument f at the in-segment position wn in [0, 1]."""
    x = (seg + wn) / (1 << k)
    if func == "exp2" or seg < thr:
        return x
    return (x - 1) / 2  # re-centered: f = m/2 - 1 with m = 1 + x


def helper_true(func: str, arg):
    """Exact helper value (mpf)."""
    if func == "exp2":
        return mp.power(2, arg)
    if arg == 0:
        return 1 / mp.log(2)
    return mp.log(1 + arg) / (mp.log(2) * arg)


def segment_coeffs(func: str, k: int, d: int, cf: int, thr: int, seg: int) -> list[int]:
    """Near-minimax degree-d coefficient magnitudes (low degree first, scaled by 2**cf) for one segment."""
    cheb = mp.chebyfit(lambda wn: helper_true(func, helper_arg(func, k, thr, seg, wn)), [mp.mpf(0), mp.mpf(1)], d + 1)
    scale = mp.mpf(1 << cf)
    c = [int(mp.nint(v * scale)) for v in reversed(cheb)]  # chebyfit is highest-degree first
    # The kernel stores magnitudes: exp2's coefficients are all positive, log2's alternate starting positive.
    assert all((v > 0) != (func == "log2" and j % 2) for j, v in enumerate(c)), f"{func} segment {seg}: signs {c}"
    return [abs(v) for v in c]


def _estimate(func: str, s: Spec, raw: list[list[int]]) -> tuple[float, list[list[int]]]:
    """
    The coefficients with the constants folded into c0 (exp2's half ULP; the center of the floor span), and an
    estimate of the worst helper error they leave as a share of the budget: the fit error at the equioscillation
    points plus half the floor span, which is the certificate's arithmetic bound. A shape the certificate refuses
    takes the whole budget and more.
    """
    spec = s.as_dict()
    budget = mp.mpf(2) ** -(s.wman + ERR_GUARD)
    scale = mp.mpf(1 << s.cf)
    half = (1 << (s.cf - s.wman)) if func == "exp2" else 0
    probes = [(1 - mp.cos(mp.pi * i / (s.d + 1))) / 2 for i in range(s.d + 2)]
    worst = mp.mpf(0)
    coeffs = []
    for seg, row in enumerate(raw):
        try:
            lo, hi = cert.arith_range(func, spec, row)
        except cert.CertificateError:
            return float("inf"), []
        if func == "exp2":
            hi += Fraction(7, 10) * 2  # the reduced argument is low by up to 2**-ff: ln2 * 2**f < 0.7 * 2 of an LSB
        bias = round((lo + hi) / 2)
        coeffs.append([row[0] + half + bias, *row[1:]])
        fit = mp.mpf(0)
        hmin = None
        for wn in probes:
            true = helper_true(func, helper_arg(func, s.k, s.thr, seg, wn))
            p = sum((-c if (func == "log2" and j % 2) else c) * wn**j for j, c in enumerate(row))
            fit = max(fit, abs(p / scale - true))
            hmin = true if hmin is None else min(hmin, true)
        worst = max(worst, (fit + (hi - lo) / 2 / scale + 1 / scale) / hmin / budget)
    return float(worst), coeffs


def _chain_bits(func: str, d: int, rows: list[list[int]]) -> tuple[list[int], list[int]]:
    """Bit widths of the w**2 chain value that starts at pair i (index 1..m; 0 elsewhere), and of each pair."""
    m = d // 2
    pair = [
        max(r[2 * i] + (r[2 * i + 1] if (2 * i + 1 <= d and func == "exp2") else 0) for r in rows) for i in range(m + 1)
    ]
    chain, total = [0] * (m + 2), 0
    for i in range(m, 0, -1):
        total += pair[i]
        chain[i] = total.bit_length()
    return chain, [p.bit_length() for p in pair]


def choose_spec(func: str, wman: int, d: int | None = None, k_base: int = K_BASE, narrow: int = NARROW) -> Spec:
    """
    Degree is fixed by degree(); K is the least from K_BASE up whose operands fit the multipliers and whose estimated
    error fits the budget. The certificate, not this estimate, carries the contract. ``d``, ``k_base`` and ``narrow``
    are overridden only by INSTRUMENTS.
    """
    cf = wman + GUARD
    d = d or degree(wman)
    width = cf if func == "exp2" else wman - 1  # argument bits: exp2 FF = WMAN+GUARD; log2 the fraction
    for k in range(min(k_base, width - 1), K_CAP + 1):
        rw = width - k
        thr = log2_threshold(k) if func == "log2" else 0
        raw = [segment_coeffs(func, k, d, cf, thr, seg) for seg in range(1 << k)]
        cw = [max(r[j] for r in raw).bit_length() for j in range(d + 1)]
        m = d // 2
        # Operand widths: each product keeps the bits that hold its error to about an LSB of the result scale.
        chain, _ = _chain_bits(func, d, raw)
        t = [rw] + [min(rw, cw[2 * i + 1]) for i in range(1, (d + 1) // 2)]
        t2 = min(rw, chain[1])
        q = [min(2 * t2, chain[i + 1]) for i in range(m)]
        s = Spec(func, wman, k, rw, cf, d, t, t2, q, thr=thr)
        if d == 2:
            if max(t2, cw[2], q[0]) > narrow:
                continue
            if rw > narrow:
                # The wide product takes one multiplier's width of argument; the bits below multiply the top of c1,
                # unless dropping them costs about an LSB only.
                s.t[0] = narrow
                if cw[1] - narrow > 1:
                    s.tc, s.sc = rw - narrow, max(0, min(narrow - 2, cw[1] - 2))
        elif d == 3:
            # c3*w is one multiplier: c3 gives up its low bits if it is wider.
            s.z3 = max(0, cw[3] - narrow)
            for r in raw:
                r[3] = ((r[3] + ((1 << s.z3) >> 1)) >> s.z3) << s.z3
            s.t[1] = min(rw, narrow)
            if t2 > narrow:
                # The square takes two multipliers and a short sum: the low part's own square is left out.
                s.th, s.sg = narrow, SQUARE_GUARD
                if t2 > 2 * narrow or 2 * t2 - q[0] - s.sg < 2 * (t2 - s.th):
                    continue
            # A few argument bits fewer in the wide product can save a multiplier, if the budget allows it.
            short = narrow * (ceil(rw / narrow) - 1)
            if 0 < rw - short <= 3:
                s.t[0] = short
                if _estimate(func, s, raw)[0] > SELECT:
                    s.t[0] = rw
        share, s.coeffs = _estimate(func, s, raw)
        if share > SELECT:
            continue
        _check_ranges(s)
        return s
    raise RuntimeError(f"no K <= {K_CAP} fits {func} WMAN={wman}")


def _check_ranges(s: Spec) -> None:
    """What the emitted RTL relies on structurally; the certificate proves the numeric ranges."""
    cf = s.cf
    for row in s.coeffs:
        assert (row[0] >> cf) == 1, f"{s.func} WMAN={s.wman}: c0 outside [2**cf, 2**(cf+1))"
    assert not s.tc or s.d == 2, "the uneven split is degree 2's"
    assert not (s.th or s.z3) or s.d == 3, "the split square and the short c3 are degree 3's"
    assert s.d != 3 or (s.z3 <= s.t[1] and all(row[3] % (1 << s.z3) == 0 for row in s.coeffs))
    if s.func == "log2":
        assert 0 < s.thr < 1 << s.k, "log2: a re-centered fraction is nonzero"


def generate_all() -> dict[tuple[str, int], Spec]:
    return {(func, wman): choose_spec(func, wman) for func in FUNCS for wman in SUPPORTED_WMAN}


# Kernel structures that the supported formats reach only at widths no check can enumerate, rebuilt at WMAN 16 for
# --check: degree 3 with a two-tile square, a shortened c3 and a shortened wide product, degree 4 (a last pair of one
# coefficient, which no supported format has), and degree 5.
INSTRUMENTS = {
    ("exp2", "deg3"): dict(wman=16, d=3, k_base=4, narrow=11),
    ("log2", "deg3"): dict(wman=16, d=3, k_base=4, narrow=12),
    ("exp2", "deg4"): dict(wman=16, d=4, k_base=3),
    ("log2", "deg4"): dict(wman=16, d=4, k_base=4),
    ("exp2", "deg5"): dict(wman=16, d=5, k_base=2),
    ("log2", "deg5"): dict(wman=16, d=5, k_base=3),
}


def instrument_specs() -> dict[tuple[str, str], Spec]:
    out = {key: choose_spec(key[0], **kw) for key, kw in INSTRUMENTS.items()}
    x3, l3 = out[("exp2", "deg3")], out[("log2", "deg3")]
    assert x3.th and x3.z3 and x3.t[0] < x3.rw and l3.z3, "the degree-3 instruments lost a structure"
    return out


# --------------------------------------------------------------------------------------------------
# Verilog emission
# --------------------------------------------------------------------------------------------------
class _Writer:
    """
    Accumulates 4-space-indented lines, except conditional-compilation directives, which stay at column 0 as in the
    hand-written RTL; ``w(...)`` accepts single lines or dedented multiline blocks.
    """

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
        flush = not text or text.startswith(("`ifdef", "`ifndef", "`else", "`elsif", "`endif"))
        self._lines.append(text if flush else "    " * self._depth + text)

    def push(self) -> None:
        self._depth += 1

    def pop(self) -> None:
        assert self._depth > 0
        self._depth -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def _pmul(
    w: _Writer, name: str, wa: str, wb: str, a: str, b: str, p: str, stage: str, vin: str, vout="", so: str = ""
) -> None:
    """
    One product and the wire it lands on, with the valid of its operands; `so` overrides the multiplier's output
    stage, which is on by default.
    """
    output = f", .STAGE_OUTPUT({so})" if so else ""
    w(f"""
        wire [{wa}+{wb}-1:0] {p};
        _zkf_pmul #(
            .WA({wa}), .WB({wb}), .SIGNED(0), .WSB(1),
            .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT({stage}){output}
        ) {name} (
            .clk(clk), .rst(rst), .in_valid({vin}), .sb_in(1'b0), .a({a}), .b({b}),
            .out_valid({vout}), .sb_out(), .p({p})
        );
    """)


def _pipe(w: _Writer, width: str, n: str, name: str, src: str, dst: str, vin: str = "", vout: str = "") -> None:
    """A delay of a payload, or with `vin` of the stream: its valid goes along, to `vout`."""
    head = f".clk(clk), .rst({'rst' if vin else "1'b0"}), .in_valid({vin or "1'b0"}), .in({src}),"
    tail = f".out_valid({vout}), .out({dst})"
    w(f"zkf_pipe #(.W({width}), .N({n})) {name} (")
    w(*([f"    {head} {tail}"] if len(head) + len(tail) < 100 else [f"    {head}", f"    {tail}"]))
    w(");")


def emit_table(s: Spec) -> str:
    """
    One self-contained per-WMAN core: the coefficient ROM and the kernel. zkf_<func>.v selects the module whose name
    matches its WMAN and passes the stage count it derived in closed form and the generator's constants it uses, which
    the core asserts (mirroring the LATENCY parameter).
    """
    exp2 = s.func == "exp2"
    d, cw = s.d, s.widths
    sign = "+" if exp2 else "-"
    fw = list(cw)  # ROM field widths: c0 without its constant leading one, c3 without its zero low bits
    fw[0] -= 1
    if d == 3:
        fw[3] -= s.z3
    wrom = sum(fw)
    w = _Writer()
    w("// GENERATED by zkf_transcendental.py -- DO NOT EDIT.")
    head = f"Table+polynomial core of zkf_{s.func} at WMAN={s.wman} (degree {d}): " + (
        "s = 2**f * 2**WCF plus half an ULP of the significand, in [2**WCF, 2**(WCF+2)), from the reduced argument f."
        if exp2
        else "c = C(f) * 2**WCF from the fraction of x, and rc, whether that fraction is re-centered; see zkf_log2.v."
    )
    head += (
        " Arithmetic: zkf._reference.trans_kernel. The ROM is addressed by the combinational argument; the first "
        "register is the ROM's own, and STAGES more follow, as the header of zkf_exp2.v counts them; the last sum is "
        "combinational."
    )
    w(*(f"// {line}" for line in wrap(head, 117)))
    w(
        "",
        "// verilog_lint: waive-start line-length  (the ROM rows are wide one-liners)",
        "",
        "`default_nettype none",
        "",
    )
    w(f"module {table_module(s.func, s.wman)} #(")
    w.push()
    w(f"parameter STAGES               = {d // 2},")
    w(f"parameter WCF                  = {s.cf},")
    if not exp2:
        w(f"parameter WG2                  = {LOG2_G2},")
        w(f"parameter WGB                  = {LOG2_GB},")
    w("parameter WMULTIPLIER          = 0,")
    w("parameter STAGE_PRODUCT        = 0,")
    w("parameter STAGE_PRODUCT_SQUARE = 0,")
    w("parameter STAGE_PRODUCT_ODD    = 0,")
    w("parameter STAGE_PRODUCT_CHAIN  = 0")
    w.pop()
    w(") (")
    w.push()
    w("input  wire           clk,")
    w("input  wire           rst,")
    w("input  wire           in_valid,")
    if exp2:
        w(
            f"input  wire [{s.cf:3}-1:0] f,",
            "output wire           out_valid,",
            f"output wire [{s.cf + 1:3}:0]   s",
        )
    else:
        w(
            f"input  wire [{s.wman - 1:3}-1:0] frac,",
            "output wire           rc,",
            "output wire           out_valid,",
            f"output wire [{s.cf:3}:0]   c",
        )
    w.pop()
    w(");")
    w.push()
    w("// Stages until the chain's feeders are ready (NF), until the chain is (NL), and of the kernel (NK).")
    later = "(STAGE_PRODUCT_SQUARE > STAGE_PRODUCT_ODD) ? STAGE_PRODUCT_SQUARE : STAGE_PRODUCT_ODD"
    w(f"localparam integer NF = {'STAGE_PRODUCT_SQUARE' if d < 3 else later};")
    links = "1 + STAGE_PRODUCT_CHAIN" if d < 4 else f"{d // 2} * (1 + STAGE_PRODUCT_CHAIN)"
    w(f"localparam integer NL = NF + {links};")
    w("localparam integer NK = (1 + STAGE_PRODUCT > NL) ? (1 + STAGE_PRODUCT) : NL;")
    w("localparam integer SQO = (STAGE_PRODUCT_SQUARE != 0) ? 1 : 0;  // the square's register, then its multiplier's")
    w("localparam integer SQP = STAGE_PRODUCT_SQUARE - SQO;")
    same = f"(STAGES != NK) || (WCF != {s.cf})" + ("" if exp2 else f" || (WG2 != {LOG2_G2}) || (WGB != {LOG2_GB})")
    w("generate")
    w(f"    if ({same}) begin : g_mismatch")
    w("        _zkf_invalid_generator_mismatch u_invalid();")
    w("    end")
    if d < 3:
        w("    // A knob that no product of this degree takes has the range it has where one does.")
        w("    if ((STAGE_PRODUCT_ODD < 0) || (STAGE_PRODUCT_ODD > 4)) begin : g_invalid_stage_product_odd")
        w("        _zkf_invalid_stage_product_out_of_range u_invalid();")
        w("    end")
    w("endgenerate")
    w(f"localparam integer K  = {s.k};")
    w(f"localparam integer RW = {s.rw};")
    for j in range(d + 1):
        w(f"localparam integer W{j} = {cw[j]};")
    for i, v in enumerate(s.t):
        w(f"localparam integer T{2 * i + 1} = {v};")
    w(f"localparam integer TS = {s.t2};")
    for i, v in enumerate(s.q):
        w(f"localparam integer Q{i} = {v};")
    if s.tc:
        w(f"localparam integer TC = {s.tc};")
        w(f"localparam integer SC = {s.sc};")
    if d == 3:
        w(f"localparam integer Z3 = {s.z3};")
    if s.th:
        w(f"localparam integer TH = {s.th};")
        w(f"localparam integer SG = {s.sg};")
    w(f"localparam integer WROM = {wrom};")
    w("")
    arg = "f" if exp2 else "frac"
    w("`ifdef ZKF_ATTRIBUTE_ROM")
    w("`ZKF_ATTRIBUTE_ROM")
    w("`endif")
    w(f"reg [WROM-1:0] rom [0:{(1 << s.k) - 1}];")
    w("initial begin")
    w.push()
    roww = len(str((1 << s.k) - 1))
    for seg, row in enumerate(s.coeffs):
        vals = [row[0] & ((1 << fw[0]) - 1), *row[1:]]
        if d == 3:
            vals[3] >>= s.z3
        fields = [f"{fw[j]}'h{vals[j]:0{(fw[j] + 3) // 4}x}" for j in range(d, -1, -1)]
        w(f"rom[{seg:{roww}d}] = {{{', '.join(fields)}}};")
    w.pop()
    w("end")
    if not exp2:
        w(f"assign rc = {arg}[RW+K-1 -: K] >= {s.thr};")
    w("reg [WROM-1:0] rom_q;")
    w("reg [RW-1:0]   w_r;")
    w("reg            v_r;")
    w("always @(posedge clk) begin")
    w.push()
    w("if (rst) v_r <= 1'b0;")
    w("else     v_r <= in_valid;")
    w(f"rom_q <= rom[{arg}[RW+K-1 -: K]];")
    w(f"w_r   <= {arg}[RW-1:0];")
    w.pop()
    w("end")
    off = fw[0]
    w("wire [W0-1:0] c0 = {1'b1, rom_q[W0-2:0]};")
    for j in range(1, d + 1):
        zs = "-Z3" if (d == 3 and j == 3) else ""
        w(f"wire [W{j}{zs}-1:0] c{j} = rom_q[{off} +: W{j}{zs}];")
        off += fw[j]
    w("")
    pair0 = _emit_wide(w, s, sign)
    w(f"assign {'s' if exp2 else 'c'} = {pair0} + {_emit_chain(w, s, sign)};")
    w.pop()
    w("endmodule")
    w("")
    w("// verilog_lint: waive-stop line-length")
    w("`default_nettype wire")
    return w.render()


def _emit_wide(w: _Writer, s: Spec, sign: str) -> str:
    """
    c0 +/- c1*w, which runs beside the chain and carries the stream's valid. Returns the pair as the last sum takes
    it: its two operands, each held until the chain is done, so that one sum reduces the whole kernel.
    """
    w(f"// c0 {sign} c1*w, the wide product, held if the chain takes longer.")
    w("wire v_p;")
    _pmul(w, "u_p1", "W1", "T1", "c1", "w_r[RW-1 -: T1]", "p1", "STAGE_PRODUCT", "v_r", "v_p")
    w("wire [W0-1:0] c0_d;")
    _pipe(w, "W0", "NK", "u_c0", "c0", "c0_d")
    w("wire [W1-1:0] p1_d;")
    _pipe(w, "W1", "NK - 1 - STAGE_PRODUCT", "u_p1_d", "p1[W1+T1-1:T1]", "p1_d", "v_p", "out_valid")
    w("")
    return f"c0_d {sign} p1_d"


def _admits(*operands: int) -> int:
    """The STAGE_PRODUCT that _zkf_pmul's guard admits on any grid: a symmetric split takes 4 bits per slice."""
    return 1 if min(operands) < 8 else 2 if min(operands) < 12 else 4


def _share(w: _Writer, name: str, knob: str, *operands: int) -> str:
    """
    The stages that a product narrower than the widest of its knob takes of it, as an expression: what _zkf_pmul
    admits at its width; the hold behind it takes the rest, so that no value the widest one admits is refused.
    """
    cap = _admits(*operands)
    if cap == 4:
        return knob
    w(f"localparam integer {name} = ({knob} > {cap}) ? {cap} : {knob};  // what a product this narrow admits")
    return name


def knob_limits(func: str, spec: dict) -> tuple[int, int, int, int]:
    """
    The largest STAGE_PRODUCT, _SQUARE, _ODD and _CHAIN that the widest product of each kind admits at WMULTIPLIER=0,
    for a table as Spec.as_dict gives it.
    """
    d, t, rows = spec["d"], spec["t"], spec["coeffs"]
    cw = [max(r[j] for r in rows).bit_length() for j in range(d + 1)]
    odd = [_admits(cw[j] - (spec["z3"] if d == 3 else 0), t[j // 2]) for j in range(3, d + 1, 2)]
    square = 1 + _admits(spec.get("th", spec["t2"]))
    return _admits(cw[1], t[0]), square, max(odd, default=4), _admits(_chain_bits(func, d, rows)[0][1], spec["q"][0])


def _emit_chain(w: _Writer, s: Spec, sign: str) -> str:
    """
    The chain w**2 * (pair1 + w**2 * (pair2 + ...)) of the pairs c[2i] +/- c[2i+1]*w above the wide one. A product
    with the square takes the pair below it, which waits in a delay line, and is registered as the bits the next sum
    takes. Returns the chain as the last sum takes it.
    """
    d, m, cw = s.d, s.d // 2, s.widths
    vw, pw = _chain_bits(s.func, d, s.coeffs)
    feed = "NF" if d >= 3 else "STAGE_PRODUCT_SQUARE"  # the square waits for an odd product only where there is one

    def bits(i: int, n: int) -> str:
        return f"W{2 * i}" + (f"+{n - cw[2 * i]}" if n != cw[2 * i] else "")

    w("wire [TS-1:0] ws = w_r[RW-1 -: TS];")
    w("wire v_s;  // the valid of the square")
    if d >= 3:
        w(f"// The pairs c[2i] {sign} c[2i+1]*w above the wide one, each held for its turn.")
    odd = {j: min(cw[j] - (s.z3 if j == 3 else 0), s.t[j // 2]) for j in range(3, d + 1, 2)}
    pair = {}
    for i in range(1, m + 1):
        j = 2 * i + 1
        turn = feed if i == m else f"{feed} + {m - i} * (1 + STAGE_PRODUCT_CHAIN)"
        if j > d:
            pair[i] = f"c{2 * i}_s"
            w(f"wire [W{2 * i}-1:0] {pair[i]};")
            _pipe(w, f"W{2 * i}", turn, f"u_c{2 * i}", f"c{2 * i}", pair[i])
            continue
        zs = "-Z3" if d == 3 else ""
        knob = "STAGE_PRODUCT_ODD"
        stage = knob if odd[j] == max(odd.values()) else _share(w, f"NP{j}", knob, odd[j])
        _pmul(w, f"u_p{j}", f"W{j}{zs}", f"T{j}", f"c{j}", f"w_r[RW-1 -: T{j}]", f"p{j}", stage, "v_r", so="0")
        w(f"wire [W{2 * i}-1:0] c{2 * i}_d;")
        _pipe(w, f"W{2 * i}", stage, f"u_c{2 * i}", f"c{2 * i}", f"c{2 * i}_d")
        pair[i] = f"pair{i}"
        w(f"wire [{bits(i, pw[i])}-1:0] pair{i}_0 = c{2 * i}_d {sign} p{j}[W{j}{zs}+T{j}-1:T{j}{zs}];")
        w(f"wire [{bits(i, pw[i])}-1:0] pair{i};")
        _pipe(w, bits(i, pw[i]), f"{turn} - {stage}", f"u_pair{i}", f"pair{i}_0", f"pair{i}")
    if m == 1:
        first = "c2" if d == 2 else f"(c2 {sign} c3*w)"
        w(f"// chain = {first} * w**2, registered as the bits the last sum takes.")
    else:
        nest = f"pair{m}"
        for i in range(m - 1, 0, -1):
            nest = f"(pair{i} + w**2 * {nest})"
        w(f"// chain = w**2 * {nest}, each product registered as the bits the next sum takes.")
    if s.th:
        w("// The square takes two products, the low part's own square left out.")
        w("localparam integer TL = TS - TH;")
        w("localparam integer BASE = 2*TS - Q0 - SG;")
        hh, hl, cross = "ws[TS-1:TL]", "ws[TL-1:0]", "sq_hl[TH+TL-1 : BASE-TL-1]"
        _pmul(w, "u_sq_hh", "TH", "TH", hh, hh, "sq_hh", "SQP", "v_r", "v_s", so="SQO")
        stage = _share(w, "NHL", "SQP", s.th, s.t2 - s.th)
        _pmul(w, "u_sq_hl", "TH", "TL", hh, hl, "sq_hl", stage, "v_r", so="SQO")
        if stage != "SQP":
            w("wire [TH+2*TL-BASE:0] sq_hl_d;")
            _pipe(w, "TH+2*TL-BASE+1", f"SQP - {stage}", "u_sq_hl_d", cross, "sq_hl_d")
            cross = "sq_hl_d"
        w(f"wire [Q0+SG-1:0] sq_s = sq_hh[2*TH-1 : BASE-2*TL] + {cross};")
        sq = "sq_s[Q0+SG-1:SG]"
    else:
        _pmul(w, "u_sq", "TS", "TS", "ws", "ws", "sq", "SQP", "v_r", "v_s", so="SQO")
        sq = "sq[2*TS-1 -: Q0]"
    valid = "v_s"
    if d >= 3:
        w("wire [Q0-1:0] sq_0;")
        w("wire v_0;")
        _pipe(w, "Q0", "NF - STAGE_PRODUCT_SQUARE", "u_sq_0", sq, "sq_0", "v_s", "v_0")
        sq, valid = "sq_0", "v_0"
    value = pair[m]
    for step in range(1, m):
        i, n = m - step, 2 * (m - step + 1)
        wide, knob = bits(i + 1, vw[i + 1]), "STAGE_PRODUCT_CHAIN"
        stage = _share(w, f"NP{n}", knob, vw[i + 1], s.q[i])
        w(f"wire v_p{n}, v_{step};")
        _pmul(w, f"u_p{n}", wide, f"Q{i}", value, f"{sq}[Q0-1 -: Q{i}]", f"p{n}", stage, valid, f"v_p{n}", so="0")
        w(f"wire [{wide}-1:0] p{n}_r;")
        hold = "1" if stage == knob else f"1 + {knob} - {stage}"
        _pipe(w, wide, hold, f"u_p{n}_r", f"p{n}[{wide}+Q{i}-1:Q{i}]", f"p{n}_r", f"v_p{n}", f"v_{step}")
        w(f"wire [{bits(i, vw[i])}-1:0] chain{i} = pair{i} + p{n}_r;")
        w(f"wire [Q0-1:0] sq_{step};")
        _pipe(w, "Q0", "1 + STAGE_PRODUCT_CHAIN", f"u_sq_{step}", sq, f"sq_{step}")
        value, sq, valid = f"chain{i}", f"sq_{step}", f"v_{step}"
    wide = bits(1, vw[1])
    _pmul(w, "u_p2", wide, "Q0", value, sq, "p2", "STAGE_PRODUCT_CHAIN", valid, so="0")
    w(f"wire [{wide}-1:0] p2_d;")
    _pipe(w, wide, "NK - NL", "u_p2_d", f"p2[{wide}+Q0-1:Q0]", "p2_d")
    if not s.tc:
        w(f"reg  [{wide}-1:0] chain;")
        w("always @(posedge clk) chain <= p2_d;")
        return "chain"
    assert d == 2 and s.func == "exp2", "log2's argument fits one multiplier at degree 2"
    wl = cw[1] + s.tc - s.rw
    w("// The argument bits below the wide product's operand times the top of c1: the rest of c1*w, at its")
    w("// depth. It joins the chain in that register.")
    stage = _share(w, "NPL", "STAGE_PRODUCT", cw[1] - s.sc, s.tc)
    _pmul(w, "u_pl", "W1-SC", "TC", "c1[W1-1:SC]", "w_r[TC-1:0]", "pl", stage, "v_r", so="0")
    w(f"wire [{wl - 1}:0] pl_d;")
    _pipe(w, str(wl), f"NK - 1 - {stage}", "u_pl_d", "pl[W1-SC+TC-1 : RW-SC]", "pl_d")
    w(f"reg  [{max(cw[2], wl)}:0] chain;")
    w("always @(posedge clk) chain <= p2_d + pl_d;")
    return "chain"


def _emit_python(all_specs: dict[tuple[str, int], Spec]) -> str:
    w = _Writer()
    w("# GENERATED by zkf_transcendental.py -- DO NOT EDIT.")
    w('"""Bit-exact table+polynomial data for zkf_log2 / zkf_exp2, consumed by the zkf package."""')
    w("")
    w("SPECS = {")
    w.push()
    for func, wman in sorted(all_specs):
        fields = all_specs[(func, wman)].as_dict()
        coeffs = fields.pop("coeffs")
        w(f"({func!r}, {wman}): dict(")
        w.push()
        w(", ".join(f"{k}={v!r}" for k, v in fields.items()) + ",")
        w(f"coeffs={coeffs!r},")
        w.pop()
        w("),")
    w.pop()
    w("}")
    return w.render()


def _artifacts(all_specs: dict[tuple[str, int], Spec]) -> dict[Path, str]:
    arts = {TABLES / f"{table_module(func, wman)}.v": emit_table(s) for (func, wman), s in sorted(all_specs.items())}
    return arts | {PKG_TABLES / "trans.py": _emit_python(all_specs)}


def emit(all_specs: dict[tuple[str, int], Spec]) -> None:
    with _MP_CONTEXT.Pool(os.cpu_count() or 1) as pool:
        for (func, wman), s in sorted(all_specs.items()):
            cert.certify(func, wman, s.as_dict(), pool)
    zkf_emit.write(_artifacts(all_specs))


def check_emit(all_specs: dict[tuple[str, int], Spec]) -> None:
    found = sorted(p for f in FUNCS for p in TABLES.glob(f"_zkf_{f}_m*.v"))
    zkf_emit.verify(_artifacts(all_specs), found, "zkf_transcendental.py --emit")


# --------------------------------------------------------------------------------------------------
# Reporting and accuracy check
# --------------------------------------------------------------------------------------------------
def ff_bits(wman: int) -> int:
    return wman + GUARD


def _report(all_specs: dict[tuple[str, int], Spec]) -> None:
    print(f"{'func':5} {'WMAN':>4} {'K':>3} {'D':>3} {'CF':>4} {'RW':>4} {'ROM bits':>9} {'ROM_kbit':>9}  operand bits")
    for (func, wman), s in sorted(all_specs.items()):
        bits = sum(s.widths) - 1 - s.z3  # WROM of emit_table
        print(
            f"{func:5} {wman:>4} {s.k:>3} {s.d:>3} {s.cf:>4} {s.rw:>4} {bits:>9} {(bits << s.k) / 1024.0:>9.1f}  "
            f"c={s.widths} t={s.t} t2={s.t2} q={s.q}"
        )


# fork lets the pool inherit the freshly-emitted tables; fall back to the default context where fork is unavailable.
_MP_CONTEXT = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)

# --check formats: the narrow ones checked exhaustively, every supported WMAN, and IEEE binary64.
CHECK_CASES = [(2, 16), (3, 16), (6, 18), (8, 24), (8, 27), (8, 32), (8, 36), (8, 48), (8, 53), (11, 53)]
# The deterministic units (table seams, exp2 binade crossings, log2 near 1) run at these formats.
UNIT_CASES = [(8, wman) for wman in SUPPORTED_WMAN] + [(11, 53)]
# Wide exponents: log2's alignment follows |e|; 17/16 is the widest zkf_log2 is built for, and beyond it only the
# reference is checked. exp2 is drawn below 2**EXP2_TOP there, the exact value of a larger result being impractical to
# hold; its reduction above that is of an integer.
WIDE_CASES = [(17, 16), (20, 16), (30, 16)]
EXP2_TOP = 17
# Worst error allowed, in ULP: half an ULP of rounding plus the helper budget. Faithfulness alone would also pass an
# operator that does not round, so this ceiling and MISROUND_CEILING gate the rounding itself.
ERR_CEILING = 0.5 + 2.0**-ERR_GUARD
MISROUND_CEILING = 2.0**-7  # share of results differing from round-to-nearest, per check unit of MISROUND_MIN inputs up
MISROUND_MIN = 4096
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
    # 2 to an integer is exact, the underflow threshold included: only the correct rounding will do. So is log2 of
    # a power of two, save where it is a tie, which zkf_log2 rounds away from zero (and refuses to be built for).
    if func == "exp2" and v.to_fraction().denominator == 1:
        return err, got_bits == rn and not any(flags), got_bits != rn
    if func == "log2" and v.frac == 0:
        e = v.exp - fmt.bias
        drop = max(abs(e).bit_length() - fmt.wman, 0)  # bits of |e| below the significand
        mag = ((abs(e) + ((1 << drop) >> 1)) >> drop) << drop
        return err, got_bits == fmt.encode(mag if e >= 0 else -mag).bits and not any(flags), got_bits != rn
    return err, ok and _underflow_ok(fmt, truth, got_bits) and not any(flags), got_bits != rn


def _underflow_ok(fmt, truth, got_bits: int) -> bool:
    """
    Below MIN_NORMAL the shared metric tolerates either neighbor in a band around MIN_NORMAL/2. These operators
    classify as _zkf_pack does, so the result is MIN_NORMAL from MIN_NORMAL/2 up, and only the certified error can
    move that threshold.
    """
    with mp.workprec(4 * fmt.wman + 80):
        min_normal = mp.ldexp(1, 1 - fmt.bias)
        if not 0 < abs(truth) < min_normal:
            return True
        at = abs(truth) - min_normal / 2
        want = fmt.pack(int(truth < 0), 1, 0).bits if at >= 0 else fmt.zero().bits
        return got_bits == want or abs(at) <= mp.ldexp(min_normal, -(fmt.wman + ERR_GUARD))


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


_SEED = 0


def _set_state(seed: int, variants: dict) -> None:
    global _SEED
    _SEED = seed
    _VARIANTS.update(variants)


def _random_inputs(func: str, fmt, n: int, shard: int) -> list[int]:
    """
    ``n`` draws, uniform in the fraction and stratified in the exponent. exp2: both signs over [-FF, WEXP-2], where the
    reduced argument is nonzero (below it _boundary_inputs keeps a stratum). log2: x > 0, half the draws at |e| <= 2,
    where the kernel's error shows, the rest over every normal exponent, uniform and log-uniform in |e|.
    """
    import numpy as np

    rng = np.random.default_rng([_SEED, shard, fmt.wexp, fmt.wman, func == "exp2"])
    frac = rng.integers(0, 1 << fmt.wfrac, n, dtype=np.int64)
    if func == "exp2":
        e = rng.integers(max(-ff_bits(fmt.wman), 1 - fmt.bias), min(fmt.wexp - 1, EXP2_TOP), n)
        sign = rng.integers(0, 2, n)
    else:
        top = np.int64(1) << rng.integers(0, fmt.wexp - 1, n)
        log = (top | (rng.integers(0, 1 << 30, n) & (top - 1))) * rng.choice([-1, 1], n)
        u = rng.random(n)
        e = np.select([u < 0.5, u < 0.75], [rng.integers(-2, 3, n), rng.integers(1 - fmt.bias, fmt.bias + 1, n)], log)
        e = np.clip(e, 1 - fmt.bias, fmt.bias)
        sign = np.zeros(n, dtype=np.int64)
    return [(int(s) << fmt.sign_shift) | ((int(x) + fmt.bias) << fmt.wfrac) | int(f) for s, x, f in zip(sign, e, frac)]


def _boundary_inputs(fmt) -> list[int]:
    """
    exp2: a band around every integer x (binade crossings; the ends include the saturation edges); bands at the
    crossings of the representables next to 1 (x = log2(1 + k*2**-WFRAC), log2(1 - k*2**-WMAN)), where a faithfulness
    gate is sharpest; x = -2**-FF, the largest reduced argument; and a stratum per exponent below -FF. At a wide
    exponent the integers and the exponents are those around the powers of two, the rest being too many.
    """
    from zkf.oracle import _mpf_to_fraction

    eb = int(os.environ.get("ZKF_EXP2_BAND", "48"))
    ff = ff_bits(fmt.wman)
    ints, exps = range(1, 1 << (fmt.wexp - 1)), range(1 - fmt.bias, -ff)
    if fmt.wexp > 11:
        ints = sorted({(1 << j) + d for j in range(min(fmt.wexp - 1, fmt.wman - 1)) for d in (-1, 0, 1)} - {0})
        exps = sorted({max(1 - fmt.bias, -ff - (1 << j)) for j in range(fmt.wexp)})
    bases = [fmt.encode(s * n).bits for n in ints for s in (1, -1)]
    with mp.workprec(4 * fmt.wman + 80):
        for k in range(1, 5):
            for x in (mp.log(1 + mp.ldexp(k, -fmt.wfrac), 2), mp.log(1 - mp.ldexp(k, -fmt.wman), 2)):
                bases.append(fmt.encode(_mpf_to_fraction(x)).bits)
    out = _near(fmt, bases, eb) + _near(fmt, [fmt.pack(1, fmt.bias - ff, 0).bits], 4)
    for e in exps:
        out += [fmt.pack(s, fmt.bias + e, f).bits for s in (0, 1) for f in (0, 1, 1 << (fmt.wfrac - 1), fmt.frac_mask)]
    return out


def _seam_inputs(func: str, fmt) -> list[int]:
    """
    Neighborhoods of every table-segment boundary, where adjacent fits meet. exp2 selects the segment by frac(x)
    (boundaries at x = +-j/2**K); log2 by the top K fraction bits, in the binades on either side of 1 and one more
    on each side, plus the true sqrt(2), where |log2(m')| crosses 1/2 next to the re-center boundary.
    """
    from zkf._reference import trans_spec

    spec = trans_spec(func, fmt.wman)
    k, rw = spec["k"], spec["rw"]
    if func == "exp2":
        bases = [
            fmt.encode(s * (Fraction(j, 1 << k) - d)).bits for j in range(1, 1 << k) for d in (0, 1) for s in (1, -1)
        ]
    else:
        fracs = {j << rw for j in range(1 << k)} | {int(mp.sqrt(2) * (1 << fmt.wfrac)) - (1 << fmt.wfrac)}
        exps = [e for e in (-2, -1, 0, 1) if 1 <= fmt.bias + e < fmt.exp_inf]
        bases = [fmt.pack(0, fmt.bias + e, f).bits for e in exps for f in sorted(fracs)]
    return _near(fmt, bases, 16)


def _near1_inputs(fmt, near1: int | None = None) -> list[int]:
    """
    log2: the fractions straddling x = 1 (random draws never hit them), a log-uniform ladder x = 1 +- 2**-j and
    x = 2**(+-2**-j) out to where they meet, and x = 2**(+-2**j), min_normal and max_finite, whose results cross a
    power of two.
    """
    from zkf.oracle import _mpf_to_fraction

    span = min(near1 or int(os.environ.get("ZKF_NEAR1_SAMPLES", str(1 << 16))), 1 << fmt.wfrac)
    fmax = (1 << fmt.wfrac) - 1
    out = [((fmt.bias - 1) << fmt.wfrac) | f for f in range(fmax - span + 1, fmax + 1)]
    out += [(fmt.bias << fmt.wfrac) | f for f in range(span)]
    bases = [fmt.pack(0, 1, 0).bits, fmt.pack(0, fmt.exp_inf - 1, fmt.frac_mask).bits]
    exps = {s * ((1 << j) + d) for j in range(fmt.wexp - 1) for s in (1, -1) for d in range(-3, 4)}  # exact ties too
    bases += [fmt.pack(0, fmt.bias + e, 0).bits for e in sorted(exps) if 1 - fmt.bias <= e <= fmt.bias]
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


def directed_strata(func: str, fmt, near1: int = 1 << 12) -> dict[str, list[int]]:
    """The directed strata of --check by name, for tb/ to run through the RTL (``near1`` caps log2's widest one)."""
    if func == "exp2":
        return {"edges": _edge_inputs(fmt), "boundary": _boundary_inputs(fmt), "seam": _seam_inputs(func, fmt)}
    return {"edges": _edge_inputs(fmt), "seam": _seam_inputs(func, fmt), "near1": _near1_inputs(fmt, near1)}


def _fraction_exponents(func: str, fmt) -> list[int]:
    """Unbiased exponents of the "fractions" sweep: exp2's whole in-range span, log2's |e| <= 4."""
    lo, hi = (-ff_bits(fmt.wman), fmt.wexp - 2) if func == "exp2" else (-4, 4)
    return list(range(max(lo, 1 - fmt.bias), min(hi, fmt.bias) + 1))


def _inputs(func: str, wexp: int, wman: int, kind: str, shard: int, nshards: int):
    from zkf import ZkfFormat

    fmt = ZkfFormat(wexp, wman)
    if kind == "random":
        return _random_inputs(func, fmt, RANDOM_CHECK_SAMPLES[func] // nshards, shard)
    if kind == "fractions":  # every fraction at one exponent (and, for exp2, one sign) per shard
        exps = _fraction_exponents(func, fmt)
        sign, e = (shard % 2, exps[shard // 2]) if func == "exp2" else (0, exps[shard])
        base = (sign << fmt.sign_shift) | ((e + fmt.bias) << fmt.wfrac)
        return (base | f for f in range(1 << fmt.wfrac))
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
_VARIANTS: dict[tuple[str, str], dict] = {}  # (func, tag) -> the spec a unit runs the model on; set before the fork


def _defects(specs: dict) -> dict[tuple[str, str], dict]:
    """Planted defects the end-to-end check must trip, on the WMAN-16 tables."""
    import copy

    out = {}
    s = copy.deepcopy(specs[("exp2", 16)])
    for row in s["coeffs"]:
        row[0] -= 1 << (s["cf"] - 16)  # the half ULP missing: faithful still, but truncated instead of rounded
    out[("exp2", "defect:truncates")] = s
    s = copy.deepcopy(specs[("exp2", 16)])
    for row in s["coeffs"]:
        row[1] += 256  # 1/32 ULP at the segment's end (an ULP is 2**(cf-wfrac) = 8192 units)
    out[("exp2", "defect:coefficient")] = s
    s = copy.deepcopy(specs[("exp2", 16)])
    for row in s["coeffs"]:
        row[1] += 32  # 1/256 ULP there: inside the general ceiling, outside the table's certified bound
    out[("exp2", "defect:1/256 ULP")] = s
    s = copy.deepcopy(specs[("log2", 16)])
    s["gb"] = 3  # the final sum keeps 3 bits below its lower rounding position, not 12
    out[("log2", "defect:final sum")] = s
    s = copy.deepcopy(specs[("log2", 16)])
    for row in s["coeffs"]:
        row[0] += 1 << 12  # C high by 2**-16: up to 0.8 ULP of the result at e = 0
    out[("log2", "defect:coefficient")] = s
    s = copy.deepcopy(specs[("log2", 16)])
    for row in s["coeffs"]:
        row[0] += 24  # C high by about 2**-24: 1/256 ULP of the result
    out[("log2", "defect:1/256 ULP")] = s
    return out


def _check_unit(unit: tuple) -> tuple:
    """One work unit -> (unit key, worst ULP, not faithful, differs from RN, count, first bad)."""
    import zkf
    from zkf import ZkfFormat
    from zkf._reference import trans_specs

    func, wexp, wman, kind, _, _, variant = unit
    fmt = ZkfFormat(wexp, wman)
    worst, nbad, nrn, count, bad = 0.0, 0, 0, 0, None
    specs = trans_specs()
    shipped = specs.get((func, wman))
    if variant:
        specs[(func, wman)] = _VARIANTS[(func, variant)]
    try:
        for b in _inputs(*unit[:6]):
            v = zkf.Zkf(fmt, b)
            try:
                if func == "exp2":
                    err, ok, differs = _verdict(func, v, v.exp2().bits)
                else:
                    r = v.log2()
                    err, ok, differs = _verdict(func, v, r.value.bits, (r.domain_error, r.pole))
            except AssertionError:  # a range the model asserts (only a planted defect gets here)
                err, ok, differs = float("inf"), False, True
            count += 1
            nrn += differs
            worst = max(worst, err)
            if not ok:
                nbad += 1
                bad = bad or (hex(b), err)
    finally:
        if variant:
            specs[(func, wman)] = shipped
    return (func, wexp, wman, kind, variant), worst, nbad, nrn, count, bad


def _units() -> list[tuple]:
    units = []
    for wexp, wman in CHECK_CASES + WIDE_CASES:
        n = 1 << (wexp + wman)  # 2**WFULL
        kind = "exhaustive" if n <= (1 << 24) else "random"
        for func in FUNCS:
            nshards = max(1, (n if kind == "exhaustive" else RANDOM_CHECK_SAMPLES[func]) // _SHARD)
            units += [(func, wexp, wman, kind, i, nshards, "") for i in range(nshards)]
            units.append((func, wexp, wman, "edges", 0, 1, ""))
    for wexp, wman in UNIT_CASES + WIDE_CASES:
        for func, kind in [("exp2", "boundary"), ("exp2", "seam"), ("log2", "seam"), ("log2", "near1")]:
            units += [(func, wexp, wman, kind, i, 4, "") for i in range(4)]
    for func, tag in _VARIANTS:  # the instruments on both narrow formats; a defect needs only one
        wman = _VARIANTS[(func, tag)]["wman"]
        for wexp in (2,) if tag.startswith("defect:") else (2, 3):
            nshards = (1 << (wexp + wman)) // _SHARD
            units += [(func, wexp, wman, "exhaustive", i, nshards, tag) for i in range(nshards)]
    return units


def _fraction_units(wexp: int, wman: int) -> list[tuple]:
    from zkf import ZkfFormat

    fmt = ZkfFormat(wexp, wman)
    return [
        (func, wexp, wman, "fractions", i, 0, "")
        for func in FUNCS
        for i in range(len(_fraction_exponents(func, fmt)) * (2 if func == "exp2" else 1))
    ]


def _passes(worst: float, not_faithful: int, misrounded: int = 0, count: int = 0, kind: str = "random") -> bool:
    # The rate gate is for the strata that sample the input space; the directed ones crowd the hard cases by design.
    sampled = kind in ("random", "exhaustive", "fractions") and count >= MISROUND_MIN
    faithful = count > 0 and not_faithful == 0 and worst <= ERR_CEILING
    return faithful and (not sampled or misrounded <= MISROUND_CEILING * count)


def _self_check() -> None:
    """
    The verdict path must reject what the pre-0.7.0 integer-ULP gate passed, and accept what is faithful. Each case
    guards a specific way the metric could silently decay.
    """
    import zkf.oracle
    from zkf import Zkf, ZkfFormat

    zkf_accuracy.self_check()
    f24, f53, f20 = ZkfFormat(8, 24), ZkfFormat(11, 53), ZkfFormat(20, 16)
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
        # 2**-BIAS is MIN_NORMAL/2 exactly, which rounds up.
        ("exp2", f24, f24.encode(-f24.bias).bits, f24.pack(0, 1, 0).bits, True),
        ("exp2", f24, f24.encode(-f24.bias).bits, 0x00000000, False),
        # An exact tie goes away from zero: log2(2**65537) = 65537 lies midway between 65536 and 65538 at WMAN 16.
        ("log2", f20, f20.pack(0, f20.bias + 65537, 0).bits, f20.encode(65538).bits, True),
        ("log2", f20, f20.pack(0, f20.bias + 65537, 0).bits, f20.encode(65536).bits, False),
    ]
    for func, fmt, x, got, want in cases:
        err, ok, _ = _verdict(func, Zkf(fmt, x), got)
        assert ok == want, f"self-check: {func}({x:#x}) -> {got:#x} judged {ok} ({err:.4f} ULP), expected {want}"
    _, ok, _ = _verdict("log2", Zkf(f24, 0x41000000), 0x40400000, (True, False))
    assert not ok, "self-check: a finite log2 result carrying domain_error passed"
    assert _passes(ERR_CEILING, 0, count=1), "self-check: a unit at exactly ERR_CEILING fails (see ERR_CEILING)"
    assert not _passes(0.0, 0), "self-check: a unit that judged nothing passed"
    # The fraction sweep: one shard per exponent, and per sign for exp2; log2 takes positive x only.
    for func in FUNCS:
        heads = [next(iter(_inputs(*u[:6]))) for u in _fraction_units(8, 24) if u[0] == func]
        got = sorted((b >> f24.sign_shift, ((b >> f24.wfrac) & f24.exp_inf) - f24.bias) for b in heads)
        signs = (0, 1) if func == "exp2" else (0,)
        assert got == sorted((s, e) for e in _fraction_exponents(func, f24) for s in signs), f"self-check: {func} sweep"
    # The zero / MIN_NORMAL threshold: MIN_NORMAL/2, movable only by the certified error.
    with mp.workprec(4 * f24.wman + 80):
        mn, zero = mp.ldexp(1, 1 - f24.bias), f24.zero().bits
        ulp = mp.ldexp(mn, -(f24.wman + 1))
        for sign in (1, -1):
            got_mn = f24.pack(int(sign < 0), 1, 0).bits
            for truth, got, want in [
                (mn / 2 + ulp / 4, got_mn, True),
                (mn / 2 + ulp / 4, zero, False),
                (mn / 2 - ulp / 4, zero, True),
                (mn / 2 - ulp / 4, got_mn, False),
                (mn / 2 + mp.ldexp(ulp, -10), zero, True),
                (mn / 2 - mp.ldexp(ulp, -10), got_mn, True),
                (mn * 3 / 4, zero, False),
            ]:
                assert _underflow_ok(f24, sign * truth, got) == want, f"self-check: underflow threshold, {truth}"
    # Representable truths must come exact (see zkf_accuracy.faithful).
    for wexp, wman in sorted(set(CHECK_CASES + UNIT_CASES + WIDE_CASES)):
        fmt = ZkfFormat(wexp, wman)
        # Every integer, or, at a wide exponent, those around the powers of two.
        some = {s * ((1 << j) + d) for j in range(wexp - 1) for d in (-1, 0, 1) for s in (-1, 1)} | {fmt.bias}
        for n in range(-fmt.bias, fmt.bias + 1) if wexp <= 11 else sorted(some):
            if 0 < abs(n) < min(1 << (wexp - 1), 1 << (wman - 1)):
                assert _exp2_truth(fmt.encode(n)) == mp.ldexp(1, n), f"self-check: inexact exp2({n}) at {fmt}"
            if 1 - fmt.bias <= n <= fmt.bias:
                assert (
                    zkf.oracle._log2_exact(fmt.pack(0, fmt.bias + n, 0)) == n
                ), f"self-check: inexact log2(2**{n}) at {fmt}"


def _kernel_unit(job: tuple) -> tuple:
    """
    One table: the integer kernel against the exact polynomial, which must stay inside the certificate's arithmetic
    bound. ``samples`` = 0 enumerates every (segment, argument) in int64/float64 (narrow tables only); otherwise that
    many arguments per segment in exact arithmetic. Returns (label, violations, largest share of the bound used).
    """
    import random

    import numpy as np

    from zkf._reference import trans_kernel

    label, func, spec, samples = job
    rw, d = spec["rw"], spec["d"]
    sign = -1 if func == "log2" else 1
    rng = random.Random(label)
    bad, used = 0, 0.0
    for row in spec["coeffs"]:
        lo, hi = cert.arith_range(func, spec, row)
        if samples:
            ws = [0, 1, (1 << rw) - 1, (1 << rw) - 2] + [rng.getrandbits(rw) for _ in range(samples)]
            diffs = []
            for w in ws:
                wn = Fraction(w, 1 << rw)
                diffs.append(sum(sign**j * row[j] * wn**j for j in range(d + 1)) - trans_kernel(func, spec, row, w))
            dmin, dmax = min(diffs), max(diffs)
        else:
            dmin, dmax = float("inf"), float("-inf")
            for start in range(0, 1 << rw, 1 << 20):
                w = np.arange(start, min(start + (1 << 20), 1 << rw), dtype=np.int64)
                wn = w.astype(np.float64) / (1 << rw)
                poly = np.zeros(len(w))
                for j in range(d, 0, -1):  # c1..cd by Horner; c0 cancels against the kernel's exactly
                    poly = (poly + sign**j * row[j]) * wn
                diff = poly - (trans_kernel(func, spec, row, w) - row[0])
                dmin, dmax = min(dmin, float(diff.min())), max(dmax, float(diff.max()))
        eps = 0 if samples else 1e-6  # float64 keeps these magnitudes to about 1e-8
        bad += not (lo - eps <= dmin and dmax <= hi + eps)
        used = max(used, float((dmax - dmin) / (hi - lo)))
    return label, bad, used


def _check(fractions: list[tuple[int, int]] | None = None) -> list[str]:
    """
    End-to-end faithful-rounding check vs mpmath via the bit-exact model. Returns the failures. ``fractions`` runs
    only the deep sweep over every fraction of those formats instead.
    """
    import sys
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(REPO))
    # zkf is first-imported here, after --emit has written the tables, so the fork pool below inherits the
    # freshly-emitted data without reaching into package internals to reload/clear caches.
    import zkf  # noqa: F401
    import zkf.oracle  # noqa: F401
    from zkf._reference import trans_specs

    _self_check()
    failures = []
    specs = trans_specs()
    kernel_jobs = []
    bound = {}  # (func, wman, variant) -> the certified error beyond half an ULP
    with _MP_CONTEXT.Pool(os.cpu_count() or 1) as pool:
        for (func, wman), spec in sorted(specs.items()):
            if not fractions or (func, wman) in {(f, w) for f in FUNCS for _, w in fractions}:
                bound[(func, wman, "")] = cert.certify(func, wman, spec, pool)["ulp"]
        if not fractions:
            for (func, tag), s in instrument_specs().items():
                _VARIANTS[(func, tag)] = s.as_dict()
                try:
                    bound[(func, s.wman, tag)] = cert.certify(func, s.wman, s.as_dict(), pool)["ulp"]
                except cert.CertificateError as ex:
                    failures.append(f"instrument {func} {tag}: {ex}")
                kernel_jobs.append((f"{func} {tag}", func, s.as_dict(), 0))
            _VARIANTS.update(_defects(specs))
            for (func, wman), spec in sorted(specs.items()):
                kernel_jobs.append((f"{func} m{wman}", func, spec, 0 if wman <= 18 else 28))
    units = [u for f in fractions for u in _fraction_units(*f)] if fractions else _units()
    agg = defaultdict(lambda: [0.0, 0, 0, 0, None])
    state = (_SEED, _VARIANTS)  # workers that are not forked start without it
    with ProcessPoolExecutor(os.cpu_count() or 1, _MP_CONTEXT, _set_state, state) as ex:
        kernel = list(ex.map(_kernel_unit, kernel_jobs))
        for key, worst, nbad, nrn, count, bad in ex.map(_check_unit, units):
            a = agg[key]
            a[0], a[1], a[2], a[3], a[4] = max(a[0], worst), a[1] + nbad, a[2] + nrn, a[3] + count, a[4] or bad
    if kernel:
        print("integer kernel vs exact polynomial, inside the certificate's arithmetic bound (share of it used):")
        for label, bad, used in kernel:
            print(f"  {'BAD' if bad else 'OK '} {label:<10} {used:.2f}")
            if bad:
                failures.append(f"kernel {label}: {bad} segments leave the certified arithmetic bound")
    print(
        f"end-to-end faithful-rounding check (model vs mpmath; worst error <= {ERR_CEILING} ULP and within each "
        f"table's certified bound; random strata seed {_SEED}):"
    )
    for (func, wexp, wman, kind, variant), (worst, nbad, nrn, count, bad) in sorted(agg.items()):
        defect = variant.startswith("defect:")  # judged against the bound of the table it was planted in
        certified = worst <= 0.5 + float(bound.get((func, wman, "" if defect else variant), 0)) + 1e-9
        ok = _passes(worst, nbad, nrn, count, kind) and certified
        print(
            f"  {('TRIPPED' if not ok else 'MISSED ') if defect else ('OK     ' if ok else 'BAD    ')} {func} "
            f"{f'{wexp}/{wman}':<5} {kind:<10} {variant:<18} worst={worst:.4f} not_faithful={nbad} "
            f"misrounded={nrn}/{count} ({nrn / max(count, 1):.1e})"
        )
        if defect and ok:
            failures.append(f"{func} {variant}: the planted defect passed")
        if not defect and not ok:
            failures.append(
                f"{func} {wexp}/{wman} {kind} {variant}: worst={worst:.4f} not_faithful={nbad}/{count} first_bad={bad}"
            )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    gen = ap.add_mutually_exclusive_group()  # --check-emit after --emit would compare the files just written
    gen.add_argument("--emit", action="store_true", help="write the per-table Verilog cores and Python data table")
    ap.add_argument("--check", action="store_true", help="verify accuracy vs mpmath (uses the bit-exact model)")
    ap.add_argument(
        "--check-fractions",
        action="append",
        metavar="WEXP/WMAN",
        help="deep sweep vs mpmath: every fraction at each exponent the kernel shows at (core-hours at WMAN 24)",
    )
    gen.add_argument(
        "--check-emit", action="store_true", help="fail if the checked-in generated artifacts differ from a fresh run"
    )
    ap.add_argument("--seed", type=int, help="seed of the random strata of --check (default: a fresh one, printed)")
    ap.add_argument("--report", action="store_true", help="print the chosen table shapes")
    args = ap.parse_args()
    if not (args.emit or args.check or args.check_emit or args.report or args.check_fractions):
        ap.error("nothing to do: pass --emit, --check, --check-fractions, --check-emit, and/or --report")

    if args.report or args.emit or args.check_emit:
        all_specs = generate_all()
        if not args.check_emit:
            _report(all_specs)
        if args.emit:
            emit(all_specs)
        if args.check_emit:
            check_emit(all_specs)
    global _SEED
    _SEED = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    fractions = [tuple(map(int, f.split("/"))) for f in args.check_fractions or []]
    failures = (_check() if args.check else []) + (_check(fractions) if fractions else [])
    if failures:
        raise SystemExit("not faithfully rounded:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()

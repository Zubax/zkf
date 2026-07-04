#!/usr/bin/env python3
"""
Targeted dense/exhaustive boundary sweep for zkf_exp2 (read-only discovery; no RTL/source edits).

Mirrors the log2 near-1 sweep methodology, targeting exp2's hard regions -- which are BOUNDARIES not zeros (2^x has no
near-zero result at finite x), where random sampling rarely lands:

  A. x near each representable INTEGER N   -> result crosses a power-of-two / binade boundary (2^N exact; the seam is
                                              just-below/above)
  B. x -> 0                                -> result -> 1.0 (the 1.0 / binade boundary)
  C. under/overflow saturation thresholds  -> x near +-2^(WEXP-1) (the e >= WEXP-1 gate): overflow-to-+inf and
                                              underflow-to-min-normal/zero seams
  D. top/bottom fracs of EACH input binade -> reduced-argument extremes feeding the polynomial (f->0, f->1)

For all 8 supported exp2/log2 WMAN values, compares exp2_reference (bit-exact RTL) vs exp2_true (mp.prec>=280 oracle,
faithful rounding), reporting the worst ULP per region per WMAN and flagging any > 1 ULP.
"""

from __future__ import annotations

import os
import sys

import mpmath as mp

REPO_FLOAT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../float
TB = os.path.join(REPO_FLOAT, "tb")
sys.path.insert(0, TB)
sys.path.insert(0, REPO_FLOAT)   # the zkf package lives directly under float/

mp.mp.prec = 320  # >= the generator's 280; extra headroom for the oracle

from zkf import Zkf, ZkfFormat  # noqa: E402
from zkf.oracle import exp2 as _exp2_true  # noqa: E402
from zkf_bits import hex_bits, mask  # noqa: E402


def _mpf_to_fraction(x):  # local mpf -> exact dyadic Fraction (dev sweep)
    from fractions import Fraction
    sign, man, exp, _bc = mp.mpf(x)._mpf_
    value = Fraction(int(man)) * (Fraction(2) ** int(exp))
    return -value if sign else value


def exp2_reference(fmt, b):
    return Zkf(fmt, b).exp2().bits


def exp2_true(fmt, b):
    return _exp2_true(Zkf(fmt, b)).bits

SUPPORTED_WMAN = [16, 18, 24, 27, 32, 36, 48, 53]

# Representative WEXP per WMAN (same pairing the --check uses, so the gate e>=WEXP-1 and the bias match what ships).
WEXP_FOR = {11: 5, 16: 6, 18: 6, 24: 8, 27: 8, 32: 8, 36: 8, 48: 8, 53: 8}


def _ordered_index(fmt: ZkfFormat, bits: int) -> int:
    bits = Zkf(fmt, bits).canonicalize().bits
    sign = (bits >> fmt.sign_shift) & 1
    mag = bits & ((1 << fmt.sign_shift) - 1)
    return -mag if sign else mag


def ulp_diff(fmt: ZkfFormat, a_bits: int, b_bits: int) -> int:
    return 0 if a_bits == b_bits else abs(_ordered_index(fmt, a_bits) - _ordered_index(fmt, b_bits))


def code(fmt: ZkfFormat, sign: int, exp_biased: int, frac: int) -> int:
    return fmt.pack(sign, exp_biased, frac & fmt.frac_mask).bits


def value_of(fmt: ZkfFormat, bits: int):
    """Exact value of a finite ZKF code as an mpf (for diagnostics only)."""
    d = Zkf(fmt, bits)
    if d.is_zero:
        return mp.mpf(0)
    sig = d.significand()
    v = mp.mpf(sig) * mp.power(2, d.exp - fmt.bias - fmt.wfrac)
    return -v if d.negative else v


def in_range_exps(fmt: ZkfFormat):
    """Unbiased exponents e that are NOT auto-saturated by the gate e >= WEXP-1 (i.e. fed to the polynomial)."""
    out = []
    for eb in range(1, fmt.exp_max_finite + 1):
        e = eb - fmt.bias
        if e < fmt.wexp - 1:
            out.append((eb, e))
    return out


def codes_around_value(fmt: ZkfFormat, target, half_band: int):
    """
    All finite ZKF codes within +-half_band encoding-ULP of target on the ordered line: the code nearest target
    plus its encoding neighbors, densely covering the just-below/just-above seam around any real target.
    """
    if target == 0:
        nearest = fmt.zero().bits
    else:
        nearest = fmt.encode(_mpf_to_fraction(mp.mpf(target))).bits
    base = _ordered_index(fmt, nearest)
    out = set()
    for d in range(-half_band, half_band + 1):
        idx = base + d
        bits = idx if idx >= 0 else ((-idx) | (1 << fmt.sign_shift))
        dd = Zkf(fmt, bits)
        if dd.is_inf:
            continue  # keep the sweep on finite inputs (saturation handled by region C)
        out.add(bits & mask(fmt.wfull))
    return out


def top_bottom_fracs(span: int, wfrac: int):
    """Indices for the densest top and bottom fracs of a binade (reduced-argument extremes f->1 and f->0)."""
    fmax = (1 << wfrac) - 1
    s = min(span, 1 << wfrac)
    lo = list(range(s))                          # frac near 0 (binade bottom, f -> 0)
    hi = list(range(fmax - s + 1, fmax + 1))     # frac near max (binade top, f -> 1)
    return sorted(set(lo) | set(hi))


def _eval(fmt: ZkfFormat, inputs):
    worst = ne = 0
    worst_ex = None
    for b in inputs:
        got = exp2_reference(fmt, b)
        want = exp2_true(fmt, b)
        u = ulp_diff(fmt, got, want)
        if u > 0:
            ne += 1
        if u > worst:
            worst = u
            worst_ex = (b, got, want)
    return worst, ne, len(inputs), worst_ex


def region_A_integers(fmt: ZkfFormat, band: int):
    """
    x near each representable integer N (both signs) in the polynomial-fed range |x| < 2^(WEXP-1). Densely sweeps
    the ZKF codes straddling each such N; N=0 is region B, skipped here.
    """
    limit = 1 << (fmt.wexp - 1)  # |x| below this is polynomial-fed
    inputs = set()
    for n in range(1, limit):
        for tgt in (n, -n):
            inputs |= codes_around_value(fmt, mp.mpf(tgt), band)
    return _eval(fmt, sorted(inputs))


def region_B_near_zero(fmt: ZkfFormat, span: int):
    """
    x -> 0 (result -> 1.0). Two complementary attacks:
       (1) the smallest-magnitude finite codes: smallest binade(s), densest fracs, both signs (x literally near 0);
       (2) the seam in the OUTPUT around 1.0: codes whose 2^x rounds to just-below / at / just-above 1.0.
    """
    inputs = set()
    # (1) smallest finite inputs: bottom two binades, all fracs up to span, both signs
    for eb in (1, 2):  # biased exp 1, 2 -> the two smallest normal binades
        if eb > fmt.exp_max_finite:
            continue
        for f in range(min(span, 1 << fmt.wfrac)):
            inputs.add(code(fmt, 0, eb, f))
            inputs.add(code(fmt, 1, eb, f))
    # top fracs of the smallest binade (still ~0)
    for f in top_bottom_fracs(span, fmt.wfrac):
        inputs.add(code(fmt, 0, 1, f))
        inputs.add(code(fmt, 1, 1, f))
    # (2) output seam around 1.0: dense codes near x=0 on the ordered input line
    inputs |= codes_around_value(fmt, mp.mpf(0), span)
    # magnitudes ~2^-WMAN .. 2^-1 so 2^x finely straddles 1.0's neighbors
    for k in range(1, fmt.wman + 4):
        e = -k
        eb = e + fmt.bias
        if 1 <= eb <= fmt.exp_max_finite:
            for f in top_bottom_fracs(min(span, 1 << fmt.wfrac), fmt.wfrac):
                inputs.add(code(fmt, 0, eb, f))
                inputs.add(code(fmt, 1, eb, f))
    return _eval(fmt, sorted(inputs))


def region_C_saturation(fmt: ZkfFormat, span: int):
    """
    Under/overflow saturation seams. The hard polynomial-active edge is e = WEXP-2 (the largest in-range binade):
    x in [2^(WEXP-2), 2^(WEXP-1)) feeds the largest finite results and the overflow-to-+inf (x>0) /
    underflow-to-min-normal-or-zero (x<0) seams. Sweeps that binade densely (both signs) plus a band of integers near
    +-2^(WEXP-1), then verifies the gate e >= WEXP-1 saturates as specified.
    """
    inputs = set()
    e_edge = fmt.wexp - 2
    eb_edge = e_edge + fmt.bias
    if 1 <= eb_edge <= fmt.exp_max_finite:
        for f in top_bottom_fracs(span, fmt.wfrac):
            inputs.add(code(fmt, 0, eb_edge, f))  # large positive -> overflow seam
            inputs.add(code(fmt, 1, eb_edge, f))  # large negative -> underflow seam
        # exhaustive over the edge binade when it is small enough
        if (1 << fmt.wfrac) <= (1 << 18):
            for f in range(1 << fmt.wfrac):
                inputs.add(code(fmt, 0, eb_edge, f))
                inputs.add(code(fmt, 1, eb_edge, f))
    # dense integer band straddling the threshold magnitude 2^(WEXP-1)
    thr = 1 << (fmt.wexp - 1)
    for n in range(thr - max(4, span // 64), thr + 1):
        inputs |= codes_around_value(fmt, mp.mpf(n), max(2, span // 256))
        inputs |= codes_around_value(fmt, mp.mpf(-n), max(2, span // 256))
    worst, ne, ninp, ex = _eval(fmt, sorted(inputs))

    # gate verification: every finite code with e >= WEXP-1 must saturate (+inf for x>0, +0 for x<0)
    gate_bad = []
    for eb in range(1, fmt.exp_max_finite + 1):
        e = eb - fmt.bias
        if e < fmt.wexp - 1:
            continue
        for f in (0, fmt.frac_mask):           # representative low/high frac in each saturated binade
            for s in (0, 1):
                b = code(fmt, s, eb, f)
                got = exp2_reference(fmt, b)
                want_pos = fmt.inf(0).bits
                want_neg = fmt.zero().bits
                want = want_neg if s else want_pos
                # confirm the oracle agrees with this saturation classification
                got_t = exp2_true(fmt, b)
                if got != want or got_t != want:
                    gate_bad.append((b, s, e, f, got, got_t, want))
    return worst, ne, ninp, ex, gate_bad


def region_D_binade_fracs(fmt: ZkfFormat, span: int):
    """Top/bottom fracs of EVERY in-range input binade, both signs (the reduced-argument extremes f->0 / f->1)."""
    inputs = set()
    for eb, _e in in_range_exps(fmt):
        for f in top_bottom_fracs(span, fmt.wfrac):
            inputs.add(code(fmt, 0, eb, f))
            inputs.add(code(fmt, 1, eb, f))
    return _eval(fmt, sorted(inputs))


# Driver
def fmt_ex(fmt: ZkfFormat, ex):
    if ex is None:
        return ""
    b, got, want = ex
    return (f"  x_bits={hex_bits(b, fmt.wfull)} x={mp.nstr(value_of(fmt, b), 12)} "
            f"got={hex_bits(got, fmt.wfull)} want={hex_bits(want, fmt.wfull)}")


def main():
    band = int(os.environ.get("EXP2_BAND", "48"))      # half-band of codes swept around each integer / zero
    span = int(os.environ.get("EXP2_SPAN", str(1 << 14)))  # top/bottom frac depth per binade (mirrors near1 default)

    print(f"exp2 BOUNDARY sweep | mp.prec={mp.mp.prec} | integer/zero half-band={band} | binade frac span={span}\n")
    header = f"{'WMAN':>4} {'WEXP':>4}  {'A:integers':>26} {'B:near-0/1.0':>22} {'C:sat-seam':>20} {'D:binade-fracs':>22}"
    print(header)
    print("-" * len(header))

    overall_worst = 0
    any_gate_bad = False
    detail = []
    for wman in SUPPORTED_WMAN:
        wexp = WEXP_FOR[wman]
        fmt = ZkfFormat(wexp, wman)

        aw, an, ai, aex = region_A_integers(fmt, band)
        bw, bn, bi, bex = region_B_near_zero(fmt, span)
        cw, cn, ci, cex, gate_bad = region_C_saturation(fmt, span)
        dw, dn, di, dex = region_D_binade_fracs(fmt, span)

        overall_worst = max(overall_worst, aw, bw, cw, dw)
        if gate_bad:
            any_gate_bad = True

        def cell(w, n, i):
            return f"ulp={w} ({n}/{i})"

        print(f"{wman:>4} {wexp:>4}  {cell(aw, an, ai):>26} {cell(bw, bn, bi):>22} "
              f"{cell(cw, cn, ci):>20} {cell(dw, dn, di):>22}")

        detail.append((wman, wexp, ("A", aw, aex), ("B", bw, bex), ("C", cw, cex), ("D", dw, dex), gate_bad))

    print("\nworst example per region per WMAN (only regions with worst ULP > 0 shown):")
    for wman, wexp, *regions in detail:
        gate_bad = regions[-1]
        regions = regions[:-1]
        for tag, w, ex in regions:
            if w > 0:
                print(f"  m{wman} region {tag}: ulp={w}{fmt_ex(ZkfFormat(wexp, wman), ex)}")
        if gate_bad:
            print(f"  m{wman} GATE SATURATION MISMATCHES: {len(gate_bad)}")
            for (b, s, e, f, got, got_t, want) in gate_bad[:6]:
                print(f"      x_bits={hex_bits(b, ZkfFormat(wexp, wman).wfull)} sign={s} e={e} frac={f} "
                      f"got={got:#x} oracle={got_t:#x} want={want:#x}")

    print("\n" + "=" * 80)
    print(f"OVERALL worst ULP across ALL regions, ALL WMAN: {overall_worst}")
    print(f"gate saturation: {'MISMATCHES FOUND' if any_gate_bad else 'all correct (+inf for x>0, +0 for x<0)'}")
    if overall_worst <= 1 and not any_gate_bad:
        print("RESULT: exp2 is CLEAN across all swept boundary regions, all WMAN (worst <= 1 ULP).")
    else:
        print("RESULT: VIOLATIONS present -- see per-region detail above.")


if __name__ == "__main__":
    main()

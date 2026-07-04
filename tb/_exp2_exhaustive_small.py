#!/usr/bin/env python3
"""
Fully exhaustive exp2 check for the small formats: every input code is tested, so every boundary (integers, x->0,
the 1.0 seam, saturation, all binade fracs) is covered with zero sampling gaps -- airtight evidence that the dense
targeted sweep missed nothing.

WMAN16 has 2**18 codes at WEXP2, 2**19 at WEXP3, 2**22 at WEXP6. Reports worst ULP and how many mismatches land in each
hard region.
"""

from __future__ import annotations

import os
import sys

import mpmath as mp

REPO_FLOAT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_FLOAT, "tb"))
sys.path.insert(0, REPO_FLOAT)   # the zkf package lives directly under float/
mp.mp.prec = 320

from zkf import Zkf, ZkfFormat  # noqa: E402
from zkf.oracle import exp2 as _exp2_true  # noqa: E402
from zkf_bits import hex_bits  # noqa: E402


def exp2_reference(fmt, b):
    return Zkf(fmt, b).exp2().bits


def exp2_true(fmt, b):
    return _exp2_true(Zkf(fmt, b)).bits


def _ordered_index(fmt, bits):
    bits = Zkf(fmt, bits).canonicalize().bits
    sign = (bits >> fmt.sign_shift) & 1
    mag = bits & ((1 << fmt.sign_shift) - 1)
    return -mag if sign else mag


def ulp_diff(fmt, a, b):
    return 0 if a == b else abs(_ordered_index(fmt, a) - _ordered_index(fmt, b))


def classify(fmt, bits):
    """Which hard region this input belongs to (for tallying where the 1-ULP cases land)."""
    d = Zkf(fmt, bits)
    if d.is_inf:
        return "inf"
    if d.is_zero:
        return "zero->1.0"
    e = d.exp - fmt.bias
    if e >= fmt.wexp - 1:
        return "saturated-gate"
    sig = Zkf(fmt, bits).significand()
    x = mp.mpf(sig) * mp.power(2, e - fmt.wfrac)
    if d.negative:
        x = -x
    nearest_int = mp.nint(x)
    if abs(x - nearest_int) <= mp.mpf(2) ** (e - fmt.wfrac):  # within ~1 ULP-of-x of an integer
        return f"near-int({int(nearest_int)})"
    if e <= -1:
        return "small-|x|(near 1.0)"
    return "interior"


def run(wexp, wman):
    fmt = ZkfFormat(wexp, wman)
    n = 1 << fmt.wfull
    worst = 0
    worst_ex = None
    region_hits = {}
    for b in range(n):
        got = exp2_reference(fmt, b)
        want = exp2_true(fmt, b)
        u = ulp_diff(fmt, got, want)
        if u > 0:
            r = classify(fmt, b)
            region_hits[r] = region_hits.get(r, 0) + 1
            if u > worst:
                worst = u
                worst_ex = (b, got, want)
        if u > 1:
            print(f"  !!! >1 ULP: m{wman} x_bits={hex_bits(b, fmt.wfull)} ulp={u} "
                  f"got={got:#x} want={want:#x} region={classify(fmt, b)}")
    total_mismatch = sum(region_hits.values())
    print(f"EXHAUSTIVE m{wman}/WEXP{wexp}: {n} codes  worst_ulp={worst}  total 1-ULP mismatches={total_mismatch}")
    for r in sorted(region_hits):
        print(f"    {r:>22}: {region_hits[r]}")
    if worst_ex:
        b, got, want = worst_ex
        print(f"    worst example: x_bits={hex_bits(b, fmt.wfull)} got={got:#x} want={want:#x}")
    assert worst <= 1, f"m{wman}: worst ULP {worst} > 1"
    return worst


if __name__ == "__main__":
    cases = [(2, 16), (3, 16), (6, 16)]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        cases = [(2, 16)]
    overall = 0
    for wexp, wman in cases:
        overall = max(overall, run(wexp, wman))
    print(f"\nEXHAUSTIVE overall worst ULP: {overall}  ->  "
          f"{'CLEAN (<=1 ULP, every code)' if overall <= 1 else 'VIOLATION'}")

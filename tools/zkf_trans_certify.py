#!/usr/bin/env python3
"""
Certificate for the ``zkf_exp2`` / ``zkf_log2`` tables: a proof, from the emitted integers alone, that the kernel each
operator evaluates is within the helper error budget for EVERY argument, and that every range the output stage relies
on holds. Its bounds are derived here from the kernel's definition, apart from the model: ``--selftest`` and the
generator's ``--check`` test the arithmetic bound against the model's kernel, and ``--selftest`` plants defects that
it must reject. The generator chooses each table's shape by the same arithmetic bound.

Per table segment the kernel result S(w) is compared with its target T (exp2: ``2**cf * 2**f`` plus half an ULP of the
significand; log2: ``2**cf * C(f)``, ``C(f) = log2(1+f)/f``) in two rigorous steps:

  * fit: the range of ``P(t) - T(t)`` over the whole closed segment, P the polynomial with the stored coefficients.
    T is replaced by a Taylor polynomial about the segment center plus an analytic remainder bound, and the
    difference polynomial is enclosed by its Bernstein coefficients on a subdivision, in interval arithmetic.

  * arithmetic: the range of ``P(w) - S(w)``, propagated through every truncated operand and floored product.

The sum of the two, with the argument truncation of exp2 and the dropped product bits of log2, is the error bound;
it must stay below ``2**-(WMAN+ERR_GUARD)`` of the helper, and the result's error beyond its half ULP of rounding below
``2**-ERR_GUARD`` ULP.

The ranges proven here are those the result assembly assumes: exp2's sum in ``[2**cf, 2**(cf+2))`` with a zero fraction
from ``2**(cf+1)`` up, which the budget alone takes the last argument to (a negative integer's: an LSB or two short
of 2, the half ULP added); log2's kernel in ``[2**cf, 2**(cf+1))`` and ``|log2(m')| <= 0.74``. The rest of the chain:
``proof/`` (the front ends and result assemblies, under these ranges), ``zkf_transcendental.py --check`` (the model
against ``mpmath``) and ``tb/`` (the RTL against the model).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

from mpmath import iv, mp

REPO = Path(__file__).resolve().parents[1]
ERR_GUARD = 8
GUARD = ERR_GUARD + 4  # cf - WMAN
PREC = 400  # bits; the kernel's Taylor coefficients lose about K bits per order to the division by f
PIECES = 8  # Bernstein subdivision of a segment


class CertificateError(Exception):
    pass


def _lo(x):
    return mp.mpf(x.a)


def _hi(x):
    return mp.mpf(x.b)


def _iv(x: Fraction):
    return iv.mpf(x.numerator) / x.denominator


def _sym(x):
    """[-|x|max, +|x|max] for a non-negative interval x."""
    return iv.mpf([-_hi(x), _hi(x)])


def _taylor_exp2(a: Fraction, h: Fraction, order: int):
    """2**(a + h*t), t in [0,1], as sum g[n]*u**n in u = t - 1/2, and an interval holding the remainder."""
    ln2 = iv.log(iv.mpf(2))
    step = _iv(h) * ln2
    g, term = [], iv.exp(_iv(a + h / 2) * ln2)
    for n in range(order + 1):
        g.append(term)
        term = term * step / (n + 1)
    # Lagrange: |R| <= max|G^(order+1)| / (order+1)! / 2**(order+1); the derivative peaks at the right end.
    top = iv.exp(_iv(a + h) * ln2) * step ** (order + 1)
    for n in range(2, order + 2):
        top = top / n
    return g, _sym(abs(top) / iv.mpf(2) ** (order + 1))


def _taylor_log2_kernel(a: Fraction, h: Fraction, order: int):
    """
    C(a + h*t) = log2(1 + f)/f as sum g[n]*u**n in u = t - 1/2, and an interval holding the remainder.

    The coefficients come from exact series arithmetic about the center f0 != 0: the series of log(1 + f0 + s) divided
    by (f0 + s). The remainder is a Cauchy estimate on the disc |z - f0| <= RHO: C(z) = sum (-z)**k / ((k+1) ln 2)
    for |z| < 1 -- which also defines C across the removable singularity at 0 -- so |C(z)| <= 1 / (ln2 (1 - |z|)).
    """
    ln2 = iv.log(iv.mpf(2))
    f0 = _iv(a + h / 2)
    if a <= 0 <= a + h and a + h / 2 == 0:
        raise CertificateError("segment centered on f = 0")
    num = [iv.log(1 + f0)]
    r, pw = 1 / (1 + f0), iv.mpf(1)
    for k in range(1, order + 1):
        pw = pw * r
        num.append(pw / k if k % 2 else -pw / k)
    q = []
    for n in range(order + 1):
        q.append((num[n] - (q[n - 1] if n else 0)) / f0)
    hh = _iv(h)
    g = [q[n] / ln2 * hh**n for n in range(order + 1)]
    rho = iv.mpf(1) / 4
    zmax = abs(f0) + rho
    ratio = abs(hh) / 2 / rho
    if not (_hi(zmax) < 1 and _hi(ratio) < 1):
        raise CertificateError("segment too far from 0 for the kernel's power series")
    return g, _sym(ratio ** (order + 1) / (1 - ratio) / ((1 - zmax) * ln2))


def _poly_range(p: list, pieces: int):
    """Interval enclosing p(u) over u in [-1/2, 1/2], from the Bernstein coefficients on equal subintervals."""
    n = len(p) - 1
    lo = hi = None
    width = iv.mpf(1) / pieces
    for k in range(pieces):
        q = list(p)
        left = iv.mpf(-1) / 2 + width * k
        for i in range(n):  # Taylor shift to the left end
            for j in range(n - 1, i - 1, -1):
                q[j] = q[j] + left * q[j + 1]
        s = iv.mpf(1)
        for i in range(n + 1):  # scale the subinterval to [0, 1]
            q[i] = q[i] * s
            s = s * width
        for m in range(n + 1):
            b = iv.mpf(0)
            for i in range(m + 1):
                b = b + q[i] * comb(m, i) / comb(n, i)
            lo = _lo(b) if lo is None else min(lo, _lo(b))
            hi = _hi(b) if hi is None else max(hi, _hi(b))
    return iv.mpf([lo, hi])


def _segment_arg(func: str, spec: dict, seg: int) -> tuple[Fraction, Fraction]:
    """(argument at the segment's left end, argument span) of the helper."""
    k = spec["k"]
    if func == "exp2":
        return Fraction(seg, 1 << k), Fraction(1, 1 << k)
    if seg >= spec["thr"]:  # re-centered: f = (frac/2**WFRAC - 1) / 2
        return (Fraction(seg, 1 << k) - 1) / 2, Fraction(1, 1 << (k + 1))
    return Fraction(seg, 1 << k), Fraction(1, 1 << k)


def _fit_range(func: str, spec: dict, seg: int):
    """Interval enclosing P(t) - T(t) over the closed segment, in units of 2**-cf."""
    d, cf = spec["d"], spec["cf"]
    row = spec["coeffs"][seg]
    a, h = _segment_arg(func, spec, seg)
    order = d + 8
    g, rem = (_taylor_exp2 if func == "exp2" else _taylor_log2_kernel)(a, h, order)
    scale = iv.mpf(1 << cf)
    half = iv.mpf(1) / 2
    diff = [-(g[i] * scale) for i in range(order + 1)]
    for j, c in enumerate(row):
        cj = iv.mpf(-c if (func == "log2" and j % 2) else c)
        for i in range(j + 1):  # (u + 1/2)**j
            diff[i] = diff[i] + cj * comb(j, i) * half ** (j - i)
    if func == "exp2":
        diff[0] = diff[0] - (1 << (cf - spec["wman"]))  # the half ULP folded into c0 belongs to the target
    return _poly_range(diff, PIECES) + rem * scale


def arith_range(func: str, spec: dict, row: list[int]) -> tuple[Fraction, Fraction]:
    """
    Range [lo, hi] of (exact polynomial) - (integer kernel) over every w, in units of 2**-cf. Derived from the kernel's
    definition (zkf._reference.trans_kernel): each product is short by its dropped operand bits times the other
    operand's bound, plus less than one for its floor.
    """
    rw, d, t, q, t2 = spec["rw"], spec["d"], spec["t"], spec["q"], spec["t2"]
    sub, q2 = func == "log2", max(q)
    if not (0 < t2 <= rw and all(0 < x <= rw for x in t) and all(0 < x <= q2 for x in q) and q2 <= 2 * t2):
        raise CertificateError("operand widths out of range")
    lsb = Fraction(1, 1 << rw)
    # wn**2 - w2/2**q2 in [0, d2]: the squared operand is short by under 2**-t2, the square is floored to q2 bits.
    d2 = 2 * (Fraction(1, 1 << t2) - lsb) + (Fraction(1, 1 << q2) if 2 * t2 > q2 else 0)
    if "th" in spec:
        # wt**2 = hh * 4**tl + hl * 2**(tl+1) + ll. At scale 2**base the two kept terms are floored (under one unit
        # each, 2**-sg of the result's), ll < 4**tl is dropped, and the sum is floored to q2 bits.
        th, sg = spec["th"], spec["sg"]
        tl, base = t2 - th, 2 * t2 - q2 - sg
        if not (d == 3 and 0 < tl <= th and sg > 0 and base >= 2 * tl):
            raise CertificateError("split square fields")
        d2 += (Fraction(2, 1 << sg) + Fraction(1 << (2 * tl), 1 << (2 * t2 - q2))) / (1 << q2)

    def odd(i: int) -> Fraction:
        """c[2i+1]*wn minus its floored product on the top t[i] argument bits, at most."""
        return row[2 * i + 1] * (Fraction(1, 1 << t[i]) - lsb) + 1

    lin = odd(0)
    if "tc" in spec:
        # c1*wn = c1*wh/2**t0 + c1*wl/2**rw exactly; the second product drops c1's low sc bits. Two floors.
        tc, sc = spec["tc"], spec["sc"]
        if not (tc == rw - t[0] and 0 <= sc <= rw and sc < row[1].bit_length()):
            raise CertificateError("split of the wide product")
        lin = 2 + ((1 << sc) - 1) * ((1 << tc) - 1) * lsb
    lo, hi = (-lin, Fraction(0)) if sub else (Fraction(0), lin)
    # Pairs chained by wn**2: V*W - floor(v*w) = (V - v)*W + v*(W - w) + frac, with W in [0, 1) and v in [0, vmax].
    m = d // 2
    clo, chi, vmax = Fraction(0), Fraction(0), row[2 * m]
    if 2 * m + 1 <= d:
        clo, chi = (-odd(m), Fraction(0)) if sub else (Fraction(0), odd(m))
        vmax += 0 if sub else row[2 * m + 1]
    for i in range(m, 0, -1):
        if sub and 2 * i + 1 <= d and not row[2 * i] > row[2 * i + 1]:
            raise CertificateError("a pair may go non-positive")
    for i in range(m - 1, 0, -1):
        plo, phi = (-odd(i), Fraction(0)) if sub else (Fraction(0), odd(i))
        clo, chi = plo + min(clo, 0), phi + max(chi, 0) + vmax * (d2 + Fraction(1, 1 << q[i])) + 1
        vmax += row[2 * i] + (0 if sub else row[2 * i + 1])
    return lo + min(clo, 0), hi + max(chi, 0) + vmax * d2 + 1


def _check_shape(func: str, wman: int, spec: dict) -> None:
    d, k, rw, cf = spec["d"], spec["k"], spec["rw"], spec["cf"]
    rows = spec["coeffs"]
    if spec["wman"] != wman or cf != wman + GUARD or len(rows) != 1 << k or any(len(r) != d + 1 for r in rows):
        raise CertificateError("table shape")
    if any(c <= 0 for r in rows for c in r):
        raise CertificateError("non-positive coefficient magnitude")
    if len(spec["t"]) != (d + 1) // 2 or len(spec["q"]) != d // 2:
        raise CertificateError("operand width lists")
    if func == "exp2" and k + rw != cf:
        raise CertificateError("exp2 argument width")
    if func == "log2" and (k + rw != wman - 1 or not 0 < spec["thr"] < 1 << k):
        raise CertificateError("log2 argument width or threshold")


def certify_segment(func: str, wman: int, spec: dict, seg: int) -> dict:
    """
    Bounds for one segment as interval upper ends: the kernel error relative to the helper, the operator's error in
    ULP beyond the half ULP of its rounding, the kernel's range in units of 2**-cf, and log2's largest |log2(m')|.
    """
    cf = spec["cf"]
    row = spec["coeffs"][seg]
    alo, ahi = arith_range(func, spec, row)
    err = _fit_range(func, spec, seg) - iv.mpf([_lo(_iv(alo)), _hi(_iv(ahi))])  # S - T
    a, h = _segment_arg(func, spec, seg)
    ln2 = iv.log(iv.mpf(2))
    scale = iv.mpf(1 << cf)
    if func == "exp2":
        helper = iv.mpf([_lo(iv.exp(_iv(a) * ln2)), _hi(iv.exp(_iv(a + h) * ln2))]) * scale
        # The reduced argument is low by at most 2**-ff, ff = cf: the target is up to ln2 * 2**f of an LSB higher
        # (the argument may then lie just past the segment's end; the factor covers that).
        err = err - iv.mpf([0, 1]) * ln2 * (_hi(helper) / scale) * (1 + iv.mpf(2) ** -(cf - 2))
    else:
        ends = [iv.log(1 + _iv(x)) / (_iv(x) * ln2) if x else 1 / ln2 for x in (a, a + h)]
        helper = iv.mpf([_lo(ends[1]), _hi(ends[0])]) * scale  # C decreases
        lmax = max(_hi(abs(iv.log(1 + _iv(x)) / ln2)) for x in (a, a + h))
    rel = abs(err) / _lo(helper)
    out = {"rel": _hi(rel), "smin": _lo(helper + err), "smax": _hi(helper + err)}
    if func == "exp2":
        out["ulp"] = _hi(abs(err) / (1 << (cf - wman + 1)))  # one ULP of the significand is 2**(cf - wfrac)
    else:
        # e = 0: the result is l = |f|*C, so its relative error is the kernel's: at most rel * 2**wman ULP.
        # e != 0: |e| - l >= 1 - lmax; below 1/2 (lmax > 1/2, next to the moved threshold) the ULP halves again.
        e_ne0 = rel * lmax * iv.mpf(2) ** (wman + 1 if lmax > mp.mpf(1) / 2 else wman)
        # The shifted C drops under 2**-(GUARD + g2) ULP of the lower candidate, the final sum under 2**-(gb + 1).
        cuts = mp.mpf(2) ** -(GUARD + spec["g2"]) + mp.mpf(2) ** -(spec["gb"] + 1)
        out["ulp"] = max(_hi(rel * iv.mpf(2) ** wman), _hi(e_ne0)) + cuts
        out["lmax"] = lmax
    return out


def _segment_job(args):
    func, wman, spec, seg = args
    iv.prec = PREC
    mp.prec = PREC
    return certify_segment(func, wman, spec, seg)


def certify(func: str, wman: int, spec: dict, pool=None) -> dict:
    """Prove one table; returns the worst bounds or raises CertificateError."""
    saved = mp.prec, iv.prec
    mp.prec = iv.prec = PREC
    try:
        _check_shape(func, wman, spec)
        jobs = [(func, wman, spec, seg) for seg in range(len(spec["coeffs"]))]
        segs = list(pool.imap(_segment_job, jobs, chunksize=8)) if pool else [_segment_job(j) for j in jobs]
        worst = {key: max(s[key] for s in segs) for key in segs[0] if key != "smin"}
        worst["smin"] = min(s["smin"] for s in segs)
        _gates(func, wman, spec, worst)
    finally:
        mp.prec, iv.prec = saved
    return worst


def _gates(func: str, wman: int, spec: dict, worst: dict) -> None:
    """The budget and every range the result assembly assumes, of the worst bounds over the segments."""
    cf = spec["cf"]
    budget = mp.mpf(2) ** -(wman + ERR_GUARD)
    if not worst["rel"] < budget:
        raise CertificateError(f"helper error 2^{float(mp.log(worst['rel'], 2)):.3f} exceeds 2^-{wman + ERR_GUARD}")
    if not worst["ulp"] <= mp.mpf(2) ** -ERR_GUARD:
        raise CertificateError(f"result error {float(worst['ulp']):.6f} ULP exceeds 2^-{ERR_GUARD}")
    if func == "exp2":
        # The significand slice needs 2**cf <= S (hidden bit) and S < 2**(cf+2) (one carry bit).
        half = 1 << (cf - wman)
        if not (worst["smin"] + half >= 1 << cf and worst["smax"] + half < 1 << (cf + 2)):
            raise CertificateError("exp2 sum leaves [2**cf, 2**(cf+2))")
        # A sum of 2 or more is the round-up to the next binade; zkf_exp2 takes its fraction bits as they are.
        if not worst["smax"] + half < (1 << (cf + 1)) + 2 * half:
            raise CertificateError("exp2 round-up with a nonzero fraction")
    else:
        if not (worst["smin"] >= 1 << cf and worst["smax"] < 1 << (cf + 1)):
            raise CertificateError("log2 kernel leaves [2**cf, 2**(cf+1))")
        # Leading-one prediction of the result frame (value 1 at bit FB; candidates hi = [1, 2), lo = [1/2, 1)):
        # e = 0 gives |f|n * C with |f|n in [1/2, 1); |e| = 1 with a subtract gives 2 - 2l; the other subtracts are at
        # least 1 - l / 2. The half ULP added before the select is far below the slack demanded here.
        cmin, cmax = worst["smin"] / (1 << cf), worst["smax"] / (1 << cf)  # exact: powers of two
        if not (cmin >= mp.mpf("1.01") and cmax <= mp.mpf("1.99") and worst["lmax"] <= mp.mpf("0.74")):
            raise CertificateError("log2 leading-one prediction")


def load_specs() -> dict:
    sys.path.insert(0, str(REPO))
    from zkf._tables import trans

    return trans.SPECS


def selftest(specs: dict, pool) -> None:
    """Each planted defect must be rejected."""
    import copy

    func, wman = "exp2", min(w for f, w in specs if f == "exp2")
    base = specs[(func, wman)]
    cases = []
    s = copy.deepcopy(base)
    s["coeffs"][3][1] += 64  # c1 high by 64 units: a few times the budget at the segment's end
    cases.append(("coefficient", func, wman, s))
    s = copy.deepcopy(base)
    s["t2"] -= 6  # a squarer operand too short
    s["q"] = [min(x, 2 * s["t2"]) for x in s["q"]]
    cases.append(("squarer width", func, wman, s))
    s = copy.deepcopy(base)
    for r in s["coeffs"]:
        r[0] -= 1 << (s["cf"] - wman)  # the half ULP missing from c0
    cases.append(("rounding constant", func, wman, s))
    lfunc, lwman = "log2", min(w for f, w in specs if f == "log2")
    lbase = specs[(lfunc, lwman)]
    s = copy.deepcopy(lbase)
    s["coeffs"][-1][0] += 40
    cases.append(("log2 coefficient", lfunc, lwman, s))
    s = copy.deepcopy(lbase)
    s["thr"] += 3  # re-centering moved without refitting
    cases.append(("log2 threshold", lfunc, lwman, s))
    s = copy.deepcopy(lbase)
    s["gb"] = 3  # the final sum keeps too few bits
    cases.append(("log2 sum guard", lfunc, lwman, s))
    for name, f, w, s in cases:
        try:
            certify(f, w, s, pool)
        except CertificateError as ex:
            print(f"  selftest {name}: rejected ({ex})")
        else:
            raise SystemExit(f"selftest: planted defect '{name}' was NOT rejected")
    # The range gates, which a table inside the budget cannot reach: each on bounds moved just past it.
    good = {func: certify(func, wman, base, pool), lfunc: certify(lfunc, lwman, lbase, pool)}
    cf, lcf = base["cf"], lbase["cf"]
    ranges = [
        (func, "smin", (1 << cf) - (1 << (cf - wman)) - 1, "sum leaves"),
        (func, "smax", 1 << (cf + 2), "sum leaves"),
        (func, "smax", (1 << (cf + 1)) + (1 << (cf - wman)), "round-up"),
        (lfunc, "smin", (1 << lcf) - 1, "kernel leaves"),
        (lfunc, "smax", 1 << (lcf + 1), "kernel leaves"),
        (lfunc, "smin", mp.mpf("1.009") * (1 << lcf), "leading-one"),
        (lfunc, "smax", mp.mpf("1.991") * (1 << lcf), "leading-one"),
        (lfunc, "lmax", mp.mpf("0.741"), "leading-one"),
    ]
    for f, key, value, gate in ranges:
        w, s = (wman, base) if f == func else (lwman, lbase)
        try:
            _gates(f, w, s, good[f] | {key: value})
        except CertificateError as ex:
            if gate not in str(ex):
                raise SystemExit(f"selftest: {f} {key} past the '{gate}' gate tripped another ({ex})") from None
        else:
            raise SystemExit(f"selftest: {f} {key} past the '{gate}' gate was NOT rejected")
    print(f"  selftest range gates: {len(ranges)} rejected")
    # At the budget itself: c0 moved by half of what the proven margin allows must still certify, and by 2.4 times
    # it (either way, the bound being two-sided) must not. The helper spans [2**cf, 2**(cf+1)), so only a move of
    # more than twice the margin is sure to leave the budget.
    margin = (mp.mpf(2) ** -(wman + ERR_GUARD) - good[func]["rel"]) * (1 << base["cf"])
    for share, accept in ((mp.mpf(1) / 2, True), (mp.mpf("2.4"), False)):
        for sign in (1, -1):
            s = copy.deepcopy(base)
            for r in s["coeffs"]:
                r[0] += sign * int(share * margin)
            try:
                certify(func, wman, s, pool)
                accepted = True
            except CertificateError:
                accepted = False
            if accepted != accept:
                raise SystemExit(f"selftest: c0 moved by {float(sign * share):+.1f} of the margin, accepted={accepted}")
    print(f"  selftest budget edge: c0 +-{int(margin / 2)} accepted, +-{int(margin * mp.mpf('2.4'))} rejected")


def selftest_arithmetic(count: int = 4000) -> None:
    """
    The arithmetic bound against the kernel it describes: random narrow kernels of every structure, each compared with
    its exact polynomial over every argument. Whatever arith_range accepts must enclose zkf._reference.trans_kernel.
    """
    import random

    sys.path.insert(0, str(REPO))
    from zkf._reference import trans_kernel

    rng = random.Random(1)
    kinds: dict[str, int] = {}
    while sum(kinds.values()) < count:
        func, d, rw = rng.choice(["exp2", "log2"]), rng.choice([2, 2, 3, 3, 3, 4, 5]), rng.randrange(3, 10)
        t2 = rng.randrange(1, rw + 1)
        q = [rng.randrange(1, 2 * t2 + 1) for _ in range(d // 2)]
        q[0] = max(q)
        spec = dict(rw=rw, d=d, t=[rng.randrange(1, rw + 1) for _ in range((d + 1) // 2)], t2=t2, q=q)
        row = [
            rng.getrandbits(b) | (1 << (b - 1))
            for b in sorted((rng.randrange(2, 22) for _ in range(d + 1)), reverse=True)
        ]
        if func == "log2" and d >= 3:
            row.sort(reverse=True)
        kind = f"degree {d}"
        if d == 2 and rng.random() < 0.5:
            spec |= dict(tc=rw - spec["t"][0], sc=rng.randrange(0, rw + 2))
            kind += ", uneven split"
        if d == 3 and t2 >= 2 and rng.random() < 0.7:
            spec |= dict(th=rng.randrange((t2 + 1) // 2, t2), sg=rng.randrange(1, 4))
            spec["q"] = [max(1, min(q[0], 2 * spec["th"] - spec["sg"]))]
            kind += ", two-tile square"
        try:
            lo, hi = arith_range(func, spec, row)
        except CertificateError:
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
        for w in range(1 << rw):
            wn = Fraction(w, 1 << rw)
            exact = sum((-c if (func == "log2" and j % 2) else c) * wn**j for j, c in enumerate(row))
            if not lo <= exact - trans_kernel(func, spec, row, w) <= hi:
                raise SystemExit(f"selftest: the arithmetic bound does not hold: {func} {spec} {row} w={w}")
    print(
        "  selftest arithmetic bound vs kernel, every argument: "
        + "; ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="also check the checker: planted defects, random kernels")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    specs = load_specs()
    ctx = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)
    failures = []
    with ctx.Pool(args.jobs) as pool:
        print("certified bound of the kernel error, every argument (budget 2^-(WMAN+8) of the helper, 2^-8 ULP):")
        for (func, wman), spec in sorted(specs.items()):
            try:
                w = certify(func, wman, spec, pool)
                print(
                    f"  OK  {func} m{wman:<2} K={spec['k']:<2} D={spec['d']} helper 2^{float(mp.log(w['rel'], 2)):.3f} "
                    f"(budget 2^-{wman + ERR_GUARD}), result {float(w['ulp'] * (1 << ERR_GUARD)):.3f} / "
                    f"{1 << ERR_GUARD} ULP"
                )
            except CertificateError as ex:
                print(f"  BAD {func} m{wman}: {ex}")
                failures.append(f"{func} m{wman}: {ex}")
        if args.selftest:
            selftest(specs, pool)
            selftest_arithmetic()
    if failures:
        raise SystemExit("not certified:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()

"""
Faithful-rounding verdicts shared by the ``--check`` gates of ``zkf_trig.py`` and ``zkf_transcendental.py`` (faithful
rounding is defined in the README).
"""

from __future__ import annotations

import mpmath as mp


def faithful(fmt, got_bits: int, truth, wrap: bool = False) -> tuple[float, bool, int]:
    """
    (error in ULP of the truth's binade, contract-satisfied, correctly-rounded encoding) of the encoding ``got_bits``
    against the UNROUNDED truth: it must be the correctly-rounded value, or its neighbor on the truth's side unless the
    truth is representable. Only canonical encodings pass (every accepted one is canonical by construction).

    The two range ends are judged by ENCODING, not by ratio, and report 0.0 on a pass -- so a caller's worst-ULP
    headline structurally excludes them and the violation COUNT is the load-bearing number, not the maximum.

    Measured against the UNROUNDED truth: comparing encodings with the rounded oracle yields only whole ULPs, which
    cannot express the margin left. `wrap` measures the circular distance for turns, where +1/2 and -1/2 are one
    angle.

    The truth must carry at most 4*WMAN+80 bits (asserted): the rounding below would otherwise judge a different value
    than the exactness test and accept the wrong side. A truth rounded onto a representable that the exact value is not
    only narrows the accepted set to that value, a false failure; conversely, an exactly representable result must come
    as an EXACT truth, or the neighbor on the truth's side passes.
    """
    from zkf import Zkf
    from zkf.oracle import _round_mpf, _to_mpf, atan2_canon_half

    got = Zkf(fmt, got_bits)
    with mp.workprec(4 * fmt.wman + 80):
        assert +truth == truth, "the truth carries more than 4*WMAN+80 bits"
        rn = _round_mpf(fmt, truth)
        min_normal = mp.power(2, 1 - fmt.bias)
        # A ULP at the truth's own binade is meaningless at the two ends of the range, so compare encodings there.
        if rn.is_inf or got.is_zero or got.is_inf or abs(truth) < min_normal:
            ok = got_bits == rn.bits
            if not ok and rn.is_inf:
                # Per the README's overflow contract, a truth up to one ULP past the overflow threshold
                # (max_finite + 1/2 ULP) may come back as max_finite.
                top = fmt.pack(int(rn.negative), fmt.exp_inf - 1, fmt.frac_mask)
                tm = abs(_to_mpf(top))
                ok = got_bits == top.bits and abs(truth) < tm + mp.mpf(1.5) * (
                    tm - abs(_to_mpf(_adjacent(fmt, top, False)))
                )
            elif not ok and truth != 0 and abs(truth) < min_normal:
                # No subnormals, so +0 and +-min_normal are the only representables bracketing a truth in
                # (0, min_normal) and both are faithful -- but accepting the pair across the WHOLE band would hide a
                # result that is wildly wrong yet technically adjacent. Relax only NEAR the midpoint (see the
                # README). Narrowing by SIDE instead does not work: the operator lands on either, and
                # zkf_trig.ATAN2_REGRESSIONS pins one case of each.
                near_midpoint = abs(abs(truth) - min_normal / 2) < 2 * min_normal * mp.power(2, -fmt.wfrac)
                span = {fmt.zero().bits, fmt.pack(int(truth < 0), 1, 0).bits} if near_midpoint else {rn.bits}
                ok = got_bits in span
            return (0.0 if ok else float("inf")), ok, rn.bits
        d = _to_mpf(got) - truth
        if wrap:
            d -= mp.nint(d)
        err = float(abs(d) / mp.power(2, mp.mag(truth) - fmt.wman))  # mag-1 is floor(log2|truth|)
        # Faithful (see the README) is NOT err < 1: across a power-of-two boundary the spacing halves, so
        # a result two steps away can measure half an ULP of the truth's coarser binade. Compare encodings instead.
        span = {rn.bits} if _to_mpf(rn) == truth else {rn.bits, _adjacent(fmt, rn, abs(truth) > abs(_to_mpf(rn))).bits}
        if wrap:  # -1/2 and +1/2 are one angle, and the contract emits the positive encoding
            span = {atan2_canon_half(fmt, b) for b in span}
    return err, got_bits in span, rn.bits


def self_check() -> None:
    """
    The contract-governed range ends must also REJECT: the min_normal/2 relaxation holds only near the midpoint, and an
    in-range truth must not come back infinite. Both rules are otherwise pinned only in the accepting direction.
    """
    from zkf import ZkfFormat
    from zkf.oracle import _to_mpf

    fmt = ZkfFormat(8, 24)
    zero, mn, inf = fmt.zero().bits, fmt.pack(0, 1, 0).bits, fmt.inf(0).bits
    top = fmt.pack(0, fmt.exp_inf - 1, fmt.frac_mask)
    with mp.workprec(4 * fmt.wman + 80):
        tn, tm = _to_mpf(fmt.pack(0, 1, 0)), _to_mpf(top)
        ulp = tm - _to_mpf(_adjacent(fmt, top, False))
        cases = [
            (tn * 3 / 10, zero, True),
            (tn * 3 / 10, mn, False),
            (tn * 8 / 10, mn, True),
            (tn * 8 / 10, zero, False),
            (tn / 2 * (1 + mp.ldexp(1, -30)), zero, True),
            (tn / 2 * (1 + mp.ldexp(1, -30)), mn, True),
            (tm + ulp / 4, top.bits, True),
            (tm + ulp / 4, inf, False),
            (tm + ulp * 3 / 4, inf, True),
            (tm + ulp * 3 / 4, top.bits, True),
            (tm + ulp * 8 / 5, top.bits, False),
        ]
    for truth, got, want in cases:
        assert (
            faithful(fmt, got, truth)[1] == want
        ), f"metric self-check: {got:#x} for {mp.nstr(truth, 10)}, want {want}"


def _adjacent(fmt, z, outward: bool):
    """The neighboring representable of a finite nonzero value, away from zero or toward it (ZKF has no subnormals)."""
    sign, e, f = int(z.negative), z.exp, z.bits & fmt.frac_mask
    if outward:
        if f < fmt.frac_mask:
            return fmt.pack(sign, e, f + 1)
        return fmt.inf(sign) if e + 1 >= fmt.exp_inf else fmt.pack(sign, e + 1, 0)
    if f > 0:
        return fmt.pack(sign, e, f - 1)
    return fmt.zero() if e <= 1 else fmt.pack(sign, e - 1, fmt.frac_mask)

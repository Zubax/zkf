"""
`exp2`, `log2` and `cordic` end to end against mpmath: the reference, which the hardware is tested to equal, is
faithful.
"""

import random
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
import pytest
from mpmath import mp

import zkf._exp2
import zkf._log2
from _cordic import ROTATED, VECTORED, rotation_strata, vectoring_strata
from _trans import boundaries, build, edges, near_one, seams
from test_table import INSTRUMENTS, instrument
from zkf import Cordic, Exp2, Log2, Zkf, ZkfFormat
from _certificate import certify
from test_table import TABLES
from zkf._cordic import _atan2_canon_half, _atan2_special
from zkf._trans import _ERR_GUARD, _GUARD, Spec, _arith_range, spec as table

pytestmark = pytest.mark.accuracy

# The narrow formats are walked whole; then every table, and binary64.
FORMATS = [(2, 16), (3, 16), (6, 18), (8, 20), (8, 24), (8, 27), (8, 32), (8, 36), (8, 40), (8, 48), (8, 53), (11, 53)]
DIRECTED = [(8, wman) for wman in (16, 18, 20, 24, 27, 32, 36, 40, 48, 53)] + [(11, 53)]
SAMPLES = {"exp2": 300000, "log2": 500000}
SHARD = 1 << 16
# The worst error allowed, in ULP: the half of rounding and the budget of the helper. Faithfulness alone would pass
# an operator that does not round, so this and the share of results that differ from the nearest gate the rounding.
ERR_CEILING = 0.5 + 2.0**-_ERR_GUARD
MISROUNDED = 2.0**-7
SAMPLED = 4096
"""Inputs from which a unit that samples its input space has that share gated."""
# `cordic`: every significand at one exponent, and what 0.9.0 was checked at beside those. A rotation is walked whole
# at 6/16 and narrower; at 6/36 its sine reaches the underflow, the bias being below the fraction; 5 is the narrowest
# exponent of a vectoring.
ROTATIONS = [(2, 16), (3, 16), (6, 16), (6, 18), (6, 36), (11, 53)] + [(8, wman) for wman in range(16, 54)]
VECTORINGS = [(5, 16), (6, 16), (6, 18), (11, 53)] + [(8, wman) for wman in range(16, 54)]
TURNED = 1 << 17
"""Random operands of a format and a mode."""


def adjacent(fmt: ZkfFormat, z: Zkf, outward: bool) -> Zkf:
    """The float next to a finite nonzero one, away from zero or toward it; there are no subnormals."""
    sign, exp, frac = int(z.negative), z.exp, z.bits & fmt.frac_mask
    if outward:
        if frac < fmt.frac_mask:
            return fmt.pack(sign, exp, frac + 1)
        return fmt.inf(sign) if exp + 1 >= fmt.exp_inf else fmt.pack(sign, exp + 1, 0)
    if frac > 0:
        return fmt.pack(sign, exp, frac - 1)
    return Zkf(fmt, 0) if exp <= 1 else fmt.pack(sign, exp - 1, fmt.frac_mask)


def faithful(fmt: ZkfFormat, got_bits: int, truth: Any, wrap: bool = False) -> tuple[float, bool, int]:
    """
    The error of the encoding `got_bits` in ULP of the binade of the unrounded `truth`, whether it is faithful, and
    the correctly rounded encoding. A faithful one is the correctly rounded value, or its neighbor on the side of
    the truth unless the truth is a float. Only canonical encodings pass.

    At the two ends of the range the verdict is on the encoding and not on a ratio, and a pass reports no error: the
    count of violations is what to read, not the worst error. `wrap` measures the distance on a circle, for turns,
    where a half and minus a half are one angle.

    The truth carries 4 * WMAN + 80 bits at most, or the rounding would judge another value than the test of
    exactness does. A result that is a float must come as an exact truth, or its neighbor on the truth's side passes.
    """
    got = Zkf(fmt, got_bits)
    with mp.workprec(4 * fmt.wman + 80):
        assert +truth == truth, "the truth carries more than 4 * WMAN + 80 bits"
        nearest = fmt.encode(truth)
        least = mp.power(2, 1 - fmt.bias)
        if nearest.is_inf or got.is_zero or got.is_inf or abs(truth) < least:
            ok = got_bits == nearest.bits
            if not ok and nearest.is_inf:
                # A truth up to an ULP past the threshold of overflow may come back as the largest finite.
                top = fmt.pack(int(nearest.negative), fmt.exp_inf - 1, fmt.frac_mask)
                largest = abs(top.to_mpf())
                step = largest - abs(adjacent(fmt, top, False).to_mpf())
                ok = got_bits == top.bits and abs(truth) < largest + mp.mpf(1.5) * step
            elif not ok and truth != 0 and abs(truth) < least:
                # Zero and the least normal bracket such a truth and both are faithful, but only near their midpoint
                # is either accepted: across the whole band that would hide a result that is far off yet adjacent.
                middle = abs(abs(truth) - least / 2) < 2 * least * mp.power(2, -fmt.wfrac)
                ok = got_bits in ({Zkf(fmt, 0).bits, fmt.pack(int(truth < 0), 1, 0).bits} if middle else {nearest.bits})
            return (0.0 if ok else float("inf")), ok, nearest.bits
        distance = got.to_mpf() - truth
        if wrap:
            distance -= mp.nint(distance)
        error = float(abs(distance) / mp.power(2, mp.mag(truth) - fmt.wman))
        # Not an error below 1: across a power of two the spacing halves, and a result two steps away can measure
        # half an ULP of the truth's coarser binade. The encodings are compared instead.
        span = {nearest.bits}
        if nearest.to_mpf() != truth:
            span.add(adjacent(fmt, nearest, abs(truth) > abs(nearest.to_mpf())).bits)
        if wrap:
            span = {_atan2_canon_half(fmt, bits) for bits in span}
    return error, got_bits in span, nearest.bits


def exp2_truth(model: Exp2, v: Zkf) -> Any:
    """
    Unrounded. Below `2**-(2 * WMAN + 40)`, `2**x` is far inside one rounding interval on either side of 1, and deep
    enough the working precision collapses it onto 1, which would accept 1 alone: a value of that magnitude and sign
    stands in, with the same nearest float, faithful set and verdicts.
    """
    wman = v.fmt.wman
    if v.exp - v.fmt.bias >= -(2 * wman + 40):
        return model.evaluate(v, oracle=True)
    with mp.workprec(4 * wman + 80):
        return mp.power(2, mp.ldexp(-1 if v.negative else 1, -(2 * wman + 40)))


def underflow_ok(fmt: ZkfFormat, truth: Any, got_bits: int) -> bool:
    """
    Below the least normal the metric takes either neighbor in a band around its half. These operators classify as
    the packer does: the result is the least normal from its half up, and only the certified error moves that.
    """
    with mp.workprec(4 * fmt.wman + 80):
        least = mp.ldexp(1, 1 - fmt.bias)
        if not 0 < abs(truth) < least:
            return True
        at = abs(truth) - least / 2
        want = fmt.pack(int(truth < 0), 1, 0).bits if at >= 0 else Zkf(fmt, 0).bits
        return bool(got_bits == want or abs(at) <= mp.ldexp(least, -(fmt.wman + _ERR_GUARD)))


def verdict(
    model: Exp2 | Log2, v: Zkf, got_bits: int, flags: tuple[bool, bool] = (False, False)
) -> tuple[float, bool, bool]:
    """
    The error in ULP, whether the result is right, and whether it differs from the nearest float. Where the result
    is by contract (`exp2` of an infinity, a zero or an `x` out of range; `log2` of the positive infinity, a zero or
    a negative `x`) it must be the oracle's; elsewhere it is judged against the unrounded truth.
    """
    fmt, function = v.fmt, model.operator
    if isinstance(model, Exp2):
        if v.is_inf or v.is_zero or v.exp - fmt.bias >= fmt.wexp - 1:
            ok = Zkf(fmt, got_bits) == fmt.encode(model.evaluate(v, oracle=True))
            return (0.0 if ok else float("inf")), ok, not ok
        truth = exp2_truth(model, v)
    else:
        oracle = model.evaluate(v, oracle=True)
        if v.is_inf or v.is_zero or v.negative:
            ok = (Zkf(fmt, got_bits), *flags) == (fmt.encode(oracle.y), oracle.domain_error, oracle.pole)
            return (0.0 if ok else float("inf")), ok, not ok
        truth = oracle.y
    error, ok, nearest = faithful(fmt, got_bits, truth)
    # 2 to an integer is exact, the threshold of underflow included: only the right rounding will do. So is the
    # logarithm of a power of two.
    if function == "exp2" and v.to_fraction().denominator == 1:
        return error, got_bits == nearest and not any(flags), got_bits != nearest
    if function == "log2" and v.frac == 0:
        exact = fmt.encode(v.exp - fmt.bias).bits
        return error, got_bits == exact and not any(flags), got_bits != nearest
    return error, ok and underflow_ok(fmt, truth, got_bits) and not any(flags), got_bits != nearest


def drawn(function: str, fmt: ZkfFormat, count: int, shard: int) -> list[int]:
    """
    Uniform in the fraction and stratified in the exponent. `exp2`: both signs wherever the reduced argument is not
    zero. `log2`: positive, half at `|e| <= 2`, where the error of the kernel shows, the rest over every exponent,
    uniformly in `e` and in its logarithm.
    """
    rng = np.random.default_rng([0, shard, fmt.wexp, fmt.wman, function == "exp2"])
    frac = rng.integers(0, 1 << fmt.wfrac, count, dtype=np.int64)
    if function == "exp2":
        e = rng.integers(max(-(fmt.wman + _GUARD), 1 - fmt.bias), fmt.wexp - 1, count)
        sign = rng.integers(0, 2, count)
    else:
        top = np.int64(1) << rng.integers(0, fmt.wexp - 1, count)
        by_log = (top | (rng.integers(0, 1 << 30, count) & (top - 1))) * rng.choice([-1, 1], count)
        u = rng.random(count)
        near, anywhere = rng.integers(-2, 3, count), rng.integers(1 - fmt.bias, fmt.bias + 1, count)
        e = np.clip(np.select([u < 0.5, u < 0.75], [near, anywhere], by_log), 1 - fmt.bias, fmt.bias)
        sign = np.zeros(count, dtype=np.int64)
    return [(int(s) << fmt.sign_shift) | ((int(x) + fmt.bias) << fmt.wfrac) | int(f) for s, x, f in zip(sign, e, frac)]


def inputs(function: str, fmt: ZkfFormat, kind: str, shard: int, shards: int) -> Any:
    if kind == "random":
        return drawn(function, fmt, SAMPLES[function] // shards, shard)
    whole: Any
    if kind == "every":
        whole = range(1 << fmt.wfull)
    elif kind == "seam":
        whole = seams(function, fmt)
    elif kind == "near1":
        whole = near_one(fmt, 1 << 16)
    else:
        whole = edges(fmt) if kind == "edges" else boundaries(fmt)
    return whole[shard::shards]


def judged(function: str, fmt: ZkfFormat, xs: Any) -> tuple[float, int, int, int, object]:
    """The worst error, the results that are wrong, those that differ from the nearest, the count, the first wrong."""
    worst, wrong, differing, count, first = 0.0, 0, 0, 0, None
    model = build(function, fmt)
    for bits in xs:
        v = Zkf(fmt, bits)
        found = model.evaluate(v)
        if isinstance(found, Zkf):
            error, ok, differs = verdict(model, v, found.bits)
        else:
            error, ok, differs = verdict(model, v, found.y.bits, (found.domain_error, found.pole))
        count += 1
        differing += differs
        worst = max(worst, error)
        if not ok:
            wrong += 1
            first = first or (hex(bits), error)
    return worst, wrong, differing, count, first


def passes(worst: float, wrong: int, differing: int = 0, count: int = 0, kind: str = "random") -> bool:
    """The share of misrounded results is gated where a unit samples its inputs: directed ones crowd the hard cases."""
    sampled = kind in ("random", "every") and count >= SAMPLED
    return count > 0 and wrong == 0 and worst <= ERR_CEILING and (not sampled or differing <= MISROUNDED * count)


@cache
def certified(spec: Spec) -> float:
    """What the certificate proves of a table's error beyond the half ULP of rounding."""
    return float(certify(spec)["ulp"])


def units() -> list[tuple[str, int, int, str, int, int]]:
    out = []
    for wexp, wman in FORMATS:
        kind = "every" if wexp + wman <= 24 else "random"
        for function in ("exp2", "log2"):
            shards = max(1, (1 << (wexp + wman) if kind == "every" else SAMPLES[function]) // SHARD)
            out += [(function, wexp, wman, kind, shard, shards) for shard in range(shards)]
            out.append((function, wexp, wman, "edges", 0, 1))
    for wexp, wman in DIRECTED:
        for function, kind in (("exp2", "boundary"), ("exp2", "seam"), ("log2", "seam"), ("log2", "near1")):
            out += [(function, wexp, wman, kind, shard, 4) for shard in range(4)]
    return out


@pytest.mark.parametrize("unit", units(), ids=lambda unit: "-".join(map(str, unit)))
def test_faithful(unit: tuple[str, int, int, str, int, int]) -> None:
    function, wexp, wman, kind, shard, shards = unit
    fmt = ZkfFormat(wexp, wman)
    worst, wrong, differing, count, first = judged(function, fmt, inputs(function, fmt, kind, shard, shards))
    assert passes(worst, wrong, differing, count, kind), (worst, wrong, differing, count, first)
    assert worst <= 0.5 + certified(table(function, wman, 18)) + 1e-9, worst


@pytest.mark.parametrize("wexp", [2, 3])
@pytest.mark.parametrize("name", INSTRUMENTS)
def test_instruments_are_faithful(name: str, wexp: int, monkeypatch: pytest.MonkeyPatch) -> None:
    function, fmt, spec = name[:4], ZkfFormat(wexp, 16), instrument(name)
    bound = certified(spec)
    monkeypatch.setattr(zkf._exp2 if function == "exp2" else zkf._log2, "table", lambda *key: spec)
    worst, wrong, differing, count, first = judged(function, fmt, range(1 << fmt.wfull))
    assert passes(worst, wrong, differing, count, "every") and worst <= 0.5 + bound + 1e-9, (worst, wrong, first)


def every_table() -> dict[str, Spec]:
    return {name: instrument(name) for name in INSTRUMENTS} | {f"{f} at {w}": table(f, w, m) for f, w, m in TABLES}


@pytest.mark.parametrize("name", [*INSTRUMENTS, *(f"{f} at {w}" for f, w, m in sorted(TABLES))])
def test_the_kernel_is_inside_the_arithmetic_bound(name: str) -> None:
    """
    The integer kernel against the exact polynomial: every argument of every segment of a narrow table, in int64
    and float64; of a wide one, the ends and a few arguments of each segment in exact arithmetic.
    """
    spec = every_table()[name]
    rw, d = spec.rw, spec.d
    sign = -1 if spec.function == "log2" else 1
    rng = random.Random(name)
    for row in spec.coeffs:
        lo, hi = _arith_range(spec, row)
        if spec.wman > 18:
            ws = [0, 1, (1 << rw) - 1, (1 << rw) - 2] + [rng.getrandbits(rw) for _ in range(28)]
            exact = [sum(sign**j * row[j] * Fraction(w, 1 << rw) ** j for j in range(d + 1)) for w in ws]
            differences = [value - spec._polynomial(row, w) for value, w in zip(exact, ws)]
            assert lo <= min(differences) and max(differences) <= hi, (name, row)
            continue
        for start in range(0, 1 << rw, 1 << 20):
            w: Any = np.arange(start, min(start + (1 << 20), 1 << rw), dtype=np.int64)
            wn = w.astype(np.float64) / (1 << rw)
            polynomial = np.zeros(len(w))
            for j in range(d, 0, -1):  # c1 and above by Horner; c0 cancels against the kernel's exactly
                polynomial = (polynomial + sign**j * row[j]) * wn
            difference = polynomial - (spec._polynomial(row, w) - row[0])
            # float64 keeps these magnitudes to about 1e-8.
            assert lo - 1e-6 <= float(difference.min()) and float(difference.max()) <= hi + 1e-6, (name, row)


def turned(vectoring: bool, fmt: ZkfFormat, kind: str, shard: int, shards: int) -> Any:
    """Raw bits, for a rotation, or pairs `(y, x)` of them."""
    if kind == "every":
        return range(1 << fmt.wfull)[shard::shards]
    if kind == "strata":
        return (vectoring_strata(fmt) if vectoring else rotation_strata(fmt))[shard::shards]
    rng = np.random.default_rng([1, shard, fmt.wexp, fmt.wman, vectoring])
    raw = [
        int.from_bytes(rng.bytes(16)) & fmt.pack(1, fmt.exp_inf, fmt.frac_mask).bits
        for _ in range(TURNED // shards * 2)
    ]
    return list(zip(raw[::2], raw[1::2])) if vectoring else raw[::2]


def rotated(model: Cordic, x: int) -> tuple[float, bool]:
    """The worse error of the two results, and whether they and the quadrant are right."""
    fmt = model.fmt
    v = Zkf(fmt, x)
    found = model.evaluate(0, v, Zkf(fmt, 0))
    sin, cos, quadrant = model.evaluate(0, v, Zkf(fmt, 0), oracle=True)
    if v.is_inf or v.is_zero:  # by contract
        ok = found == (fmt.encode(sin), fmt.encode(cos), quadrant)
        return (0.0 if ok else float("inf")), ok
    of_sin, of_cos = faithful(fmt, found.r0.bits, sin), faithful(fmt, found.r1.bits, cos)
    return max(of_sin[0], of_cos[0]), of_sin[1] and of_cos[1] and found.quadrant == quadrant


def vectored(model: Cordic, y: int, x: int) -> tuple[float, bool]:
    """As `rotated`. A distance on the circle would take an angle a whole turn off, so its range is held here."""
    fmt = model.fmt
    a, b = Zkf(fmt, y), Zkf(fmt, x)
    found = model.evaluate(1, a, b)
    with mp.workprec(4 * fmt.wman + 80):
        inside = not found.r0.is_inf and abs(found.r0.to_mpf()) <= mp.mpf(1) / 2
    theta, length, _ = model.evaluate(1, a, b, oracle=True)
    if _atan2_special(a, b) is not None:  # by contract
        ok = inside and found == (fmt.encode(theta), fmt.encode(length), 0)
        return (0.0 if ok else float("inf")), ok
    of_theta = faithful(fmt, found.r0.bits, theta, wrap=True)
    of_length = faithful(fmt, found.r1.bits, length)
    return max(of_theta[0], of_length[0]), inside and of_theta[1] and of_length[1]


def turns(vectoring: bool, fmt: ZkfFormat, operands: Any) -> tuple[float, int, int, object]:
    """The worst error, the results that are wrong, the count, and the first wrong."""
    worst, wrong, count, first = 0.0, 0, 0, None
    model = Cordic(fmt, stage_core_mul=3, wmultiplier=18, mode=int(vectoring))
    for operand in operands:
        error, ok = vectored(model, *operand) if vectoring else rotated(model, operand)
        count += 1
        worst = max(worst, error)
        if not ok:
            wrong += 1
            first = first or (operand, error)
    return worst, wrong, count, first


def trig_units() -> list[tuple[str, int, int, str, int, int]]:
    assert set(ROTATED) <= set(ROTATIONS) and set(VECTORED) <= set(VECTORINGS), "operands of a format that is not run"
    out = []
    for name, formats in (("sincos", ROTATIONS), ("atan2", VECTORINGS)):
        for wexp, wman in formats:
            fmt, vectoring = ZkfFormat(wexp, wman), name == "atan2"
            whole = not vectoring and fmt.wfull <= 22
            sizes = {"every": 1 << fmt.wfull} if whole else {"random": TURNED}
            sizes["strata"] = len(vectoring_strata(fmt) if vectoring else rotation_strata(fmt))
            for kind, size in sizes.items():
                shards = -(-size // SHARD)
                out += [(name, wexp, wman, kind, shard, shards) for shard in range(shards)]
    return out


@pytest.mark.parametrize("unit", trig_units(), ids=lambda unit: "-".join(map(str, unit)))
def test_cordic_is_faithful(unit: tuple[str, int, int, str, int, int]) -> None:
    name, wexp, wman, kind, shard, shards = unit
    fmt, vectoring = ZkfFormat(wexp, wman), name == "atan2"
    worst, wrong, count, first = turns(vectoring, fmt, turned(vectoring, fmt, kind, shard, shards))
    assert count and not wrong, (worst, wrong, count, first)

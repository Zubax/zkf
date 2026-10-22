"""The bench of `cordic`, whose rows have one shape, and its directed operands."""

import math
import zlib
from fractions import Fraction
from functools import reduce
from itertools import zip_longest
from operator import or_
from typing import Any

import numpy as np
from mpmath import mp

from _bench import Case, run_cuts, run_handshake
from _catalog import CATALOG
from _operands import (
    SEED,
    directed_numbers,
    random_bits,
    random_inf,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf import Timing, Zkf, ZkfFormat
from zkf._cordic import (
    _GUARD_DIV,
    _LACKED,
    Cordic,
    _atan2_bypass_shift,
    _constants,
    _cordic_vector,
    _geometry,
    _vector_residual,
)
from zkf._pmul import counts

SHORT = {
    "un": "unroll100",
    "si": "stage_input",
    "sm": "stage_core_mul",
    "sc": "stage_core",
    "so": "stage_output",
    "fa": "fast",
}
"""The knobs as the rows name them."""


def synthesized(wman: int, mode: int = 2) -> dict[str, Any]:
    """
    The knobs of the catalog's specs at 6/18, or at 8/36 above WMAN 24: the rate, the product's count and the packers of
    the mode's spec, or of them all, which agree, where it has none, the count brought within what the width takes; the
    registers of them all that the mode builds.
    """
    width = 18 if wman <= 24 else 36
    found = [model for model in CATALOG.values() if isinstance(model, Cordic) and model.fmt.wman == width]
    own = [model for model in found if model.mode == mode] or found
    ((un, sm, fa),) = {(model.unroll100, model.stage_core_mul, model.fast) for model in own}
    taken = counts(*_geometry(wman).operands(mode), signed=mode != 1, wmultiplier=18)
    sm = min(max(sm, taken.start), taken.stop - 1)
    return {"un": un, "sm": sm, "fa": fa, "sc": reduce(or_, (model.stage_core for model in found)) & ~_LACKED[mode]}


# Operands that were once more than an ULP off, by wexp and wman: a rotation's, and `(y, x)` on either result. Of
# the latter at 6/18, three are just below where the length overflows, and the last, with that at 5/16, has its
# angle on either side of half the least normal.
ROTATED = {(8, 24): [0x3EFFFBD1], (6, 18): [0x3E0044]}
VECTORED = {
    (6, 16): [(0x38967F, 0x1DEA13)],
    (6, 18): [(0x7BEC6A, 0x7D7C61), (0x6BFFFF, 0x7DFFFF), (0x786922, 0x7DF44F), (0x6BF359, 0x7DFFFF)]
    + [(0x05F9A3, 0x3E87D9)],
    (5, 16): [(0x016E79, 0x0797D1)],
}


def rotations(fmt: ZkfFormat) -> list[int]:
    """
    Zeros and infinities with and without a fraction; the directed numbers; every quarter of a turn up to two and the
    odd eighths below one, the floats next to a quarter, whole turns; both sides of where the linear path takes over,
    at either end of every quadrant; what is too small for the reduction to resolve, and the exponents at which it
    starts and stops shifting; and angles inside the octants, which the iterations and their correction serve.
    """
    sign_bit = 1 << fmt.sign_shift
    out = [0, sign_bit, fmt.exp_inf << fmt.wfrac, sign_bit | (fmt.exp_inf << fmt.wfrac), (1 << fmt.wfull) - 1]
    if fmt.wexp < 3:
        return out
    out += directed_numbers(fmt).values()
    spec = _constants(fmt.wman)
    edge = Fraction(spec.tsa, 1 << (spec.wt + 2))
    for sign in (0, 1):
        for k in range(1, 8):
            top = k.bit_length() - 1
            exp = fmt.bias + top - 2
            if 1 <= exp <= fmt.exp_max_finite:
                out.append(fmt.pack(sign, exp, (k << (fmt.wfrac - top)) & fmt.frac_mask).bits)
        out += [fmt.encode(Fraction(k, 8)).bits | (sign << fmt.sign_shift) for k in (1, 3, 5, 7)]
        quarter = fmt.bias - 2
        if quarter >= 2:
            out.append(fmt.pack(sign, quarter - 1, fmt.frac_mask).bits)
        if quarter >= 1:
            out.append(fmt.pack(sign, quarter, 1).bits)
        out += [fmt.pack(sign, fmt.bias + k, 0).bits for k in (0, 1, 2) if fmt.bias + k <= fmt.exp_max_finite]
        out.append(fmt.pack(sign, fmt.exp_max_finite, 0).bits)
        for quadrant in range(4):
            for center in (Fraction(quadrant, 4) + edge, Fraction(quadrant + 1, 4) - edge):
                bits = fmt.encode(center).bits
                if bits >> fmt.wfrac >= 2:
                    out += [(bits + step) | (sign << fmt.sign_shift) for step in (-2, -1, 0, 1, 2)]
        for exp in (1, 2, 3):
            out += [fmt.pack(sign, exp, frac).bits for frac in (1, fmt.frac_mask) if exp <= fmt.exp_max_finite]
        wff = spec.wt + 2
        least = fmt.bias - (wff - fmt.wman) - 1  # of the exponents whose significand reaches the reduced fraction
        clamps = [exp for exp in range(least - 1, least + 2)] + [exp for exp in range(least + wff - 1, least + wff + 2)]
        out += [fmt.pack(sign, e, f).bits for e in clamps if 1 <= e <= fmt.exp_max_finite for f in (0, fmt.frac_mask)]
        inside = [fmt.frac_mask // 3, fmt.frac_mask // 5 * 3, fmt.frac_mask // 7 * 6]
        exponents = range(max(1, fmt.bias - 6), min(fmt.bias + 3, fmt.exp_max_finite + 1))
        out += [fmt.pack(sign, exp, frac).bits for exp in exponents for frac in inside]
    return out


def pairs(fmt: ZkfFormat) -> list[tuple[int, int]]:
    """
    `(y, x)`: every pair of the zeros and infinities; the quadrants and the diagonals of the octants; the axes; one
    far smaller than the other, where the angle rounds to an end of its range; a ratio across the bypass at every
    exponent, and bypassed ones whose magnitude has the bit of a half turn set; lengths up to the largest finite one
    and past it; and angles either side of where they round onto a half turn, and one at a half turn or past it.
    """
    sign_bit, inf = 1 << fmt.sign_shift, fmt.exp_inf << fmt.wfrac
    raw = [0, sign_bit, inf, sign_bit | inf, (1 << fmt.wfull) - 1]
    out = [(y, x) for y in raw for x in raw]
    numbers = directed_numbers(fmt)
    one, minus_one, two = numbers["one"], numbers["minus_one"], numbers["two"]
    big, tiny = fmt.pack(0, fmt.exp_max_finite, 0).bits, fmt.pack(0, 1, 0).bits
    for sy in (0, sign_bit):
        for sx in (0, sign_bit):
            out += [(one | sy, one | sx), (two | sy, one | sx)]
        out += [(0, one), (0, minus_one), (one | sy, 0), (tiny | sy, big), (big | sy, tiny)]
        out += [(tiny | sy, big | sign_bit), (tiny | sy, minus_one), (big | sy, one), (one | sy, big | sy)]
    out += [(one, sign_bit | inf), (minus_one, sign_bit | inf)]
    for exp in range(1, fmt.exp_inf):
        out += [(fmt.pack(0, exp, fmt.frac_mask).bits, one), (one, fmt.pack(0, exp, 1).bits)]
    # Bypassed next to its threshold, `y` negative, its significand the largest whose magnitude has the bit of a half
    # turn set, which a half turn's test must not read on the bypass.
    c = _constants(fmt.wman)
    sig_x = Zkf(fmt, one).significand()

    def half_bit(frac: int) -> bool:
        r, rest = divmod((frac | 1 << fmt.wfrac) << 2 * c.steps, sig_x)
        return bool((r * c.inv_tau | bool(rest)) >> (c.zf - 1) & 1)

    frac = next(frac for frac in range(fmt.frac_mask, -1, -1) if half_bit(frac))
    below = fmt.bias - _atan2_bypass_shift(fmt)
    out += [(fmt.pack(1, exp, frac).bits, one) for exp in range(max(1, below - 3), max(1, below))]
    top = fmt.pack(0, fmt.exp_max_finite, fmt.frac_mask).bits
    for exp in range(max(1, fmt.exp_max_finite - fmt.wman // 2 - 3), fmt.exp_max_finite + 1):
        out += [(fmt.pack(0, exp, frac).bits, top) for frac in (0, fmt.frac_mask)]
    # The saturation is what this pair is for: the length rounds to the infinity and the operator gives the largest.
    edge, model = _saturating(fmt), build(fmt, mode=1)
    assert fmt.encode(model.evaluate(1, Zkf(fmt, edge), Zkf(fmt, top), oracle=True).r1).is_inf
    assert model.evaluate(1, Zkf(fmt, edge), Zkf(fmt, top)).r1.bits == top
    out.append((edge, top))
    # Either side of where an angle next to a half turn rounds onto it, `y` of both signs, `x` minus the least power of
    # two from 1 up that leaves them normal, where there is one.
    half = fmt.encode(Fraction(1, 2)).bits
    k = next(
        (
            k
            for k in range(fmt.exp_max_finite - fmt.bias + 1)
            if fmt.encode(Fraction(math.pi) * Fraction(2) ** (k - 3 - fmt.wfrac)).exp
        ),
        None,
    )
    if k is not None:
        x = fmt.pack(1, fmt.bias + k, 0).bits
        lo, hi = (
            fmt.encode(Fraction(math.pi) * Fraction(2) ** (k - 2 - fmt.wfrac) * m).bits for m in (Fraction(1, 2), 2)
        )

        def rounds_to_half(y: int) -> bool:
            return model.evaluate(1, Zkf(fmt, y), Zkf(fmt, x)).r0.bits == half

        assert rounds_to_half(lo) and not rounds_to_half(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if rounds_to_half(mid) else (lo, mid)
        out += [(y | sy, x) for y in (lo, hi) for sy in (0, sign_bit)]
    # The angle at a half turn or past it before its rounding, where the engine leaves a negative residual: a first
    # one among small angles, where the format has one.
    rng = np.random.default_rng(fmt.wman)

    def random_frac() -> int:
        return int(rng.integers(0, fmt.frac_mask + 1))

    small = (
        (fmt.pack(0, exp - shift, random_frac()).bits, fmt.pack(1, exp, random_frac()).bits)
        for shift in range(c.xf, 0, -1)
        for exp in [max(fmt.bias, shift + 1)]
        if exp <= fmt.exp_max_finite
        for _ in range(64)
    )
    past = next(((y | sign_bit, x) for y, x in small if _engine_angle(fmt, y, x) <= 0), None)
    if past is None:
        return out
    assert model.evaluate(1, Zkf(fmt, past[0]), Zkf(fmt, past[1])).r0.bits == half
    return [*out, past]


def operands(
    fmt: ZkfFormat, vectoring: bool, kind: str | int, shard: int = 0, shards: int = 1, seed: int = SEED
) -> list[tuple[int, int]]:
    """
    `(a, b)` of one mode: "every" operand of a rotation, of one shard of so many; the "directed" ones; or those and
    random ones up to a count. A rotation's `b` is noise.
    """
    rng = np.random.default_rng(seed + vectoring)
    full = (1 << fmt.wfull) - 1
    if kind == "every":
        assert not vectoring
        return [(x, random_bits(fmt.wfull, rng)) for x in range(1 << fmt.wfull)[shard::shards]]
    count = 0 if kind == "directed" else int(kind)
    if vectoring:
        found = dict.fromkeys((y & full, x & full) for y, x in pairs(fmt))
        while len(found) < count:
            found.setdefault(_random_pair(fmt, rng))
        return list(found)
    xs = dict.fromkeys(x & full for x in rotations(fmt))
    while len(xs) < count:
        xs.setdefault(random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng))
    # Next to a half and to a whole turn, where the sine is smallest and its exponent is decided.
    for exp in (fmt.bias - 1, fmt.bias) if fmt.wexp >= 3 else ():
        xs.update(dict.fromkeys((exp << fmt.wfrac) + sign * ulps for ulps in (1, 2, 3, 5, 8) for sign in (-1, 1)))
    return [(x, random_bits(fmt.wfull, rng)) for x in xs]


def case(model: Cordic, vectoring: bool, a: int, b: int, noise: int = 0) -> Case:
    """Of one mode; where the model has one mode only, `vectoring` is noise."""
    fmt, timing = model.fmt, model.timing
    latency = (timing if isinstance(timing, Timing) else timing[int(vectoring)]).latency
    selected = int(vectoring) if model.mode == 2 else noise
    found = model.evaluate(selected, Zkf(fmt, a), Zkf(fmt, b))
    return Case({"a": a, "b": b, "vectoring": selected}, found._asdict(), latency)


def build(fmt: ZkfFormat, mode: int = 2, **short: Any) -> Cordic:
    """The model, its knobs in short; `wm` is a multiplier width other than 18, and `sm` the fewest unless given."""
    width = short.pop("wm", 18)
    fewest = counts(*_geometry(fmt.wman).operands(mode), signed=mode != 1, wmultiplier=width).start
    knobs = {SHORT[name]: value for name, value in ({"sm": fewest} | short).items()}
    return Cordic(fmt, mode=mode, wmultiplier=width, **knobs)


def cases(
    model: Cordic, kind: str | int, shard: int = 0, shards: int = 1, seed: int = SEED, alternate: bool = False
) -> list[Case]:
    """Of every mode that the model has, shuffled, or under `alternate` one of each mode in turn while both last."""
    rng = np.random.default_rng(seed + 2)
    by_mode = [
        [
            case(model, vectoring, a, b, int(rng.integers(0, 2)))
            for a, b in operands(model.fmt, vectoring, kind, shard, shards, seed)
        ]
        for vectoring in (False, True)
        if model.mode in (2, int(vectoring))
    ]
    if alternate:
        return [found for turn in zip_longest(*by_mode) for found in turn if found is not None]
    out = [found for of_mode in by_mode for found in of_mode]
    return [out[i] for i in rng.permutation(len(out))]


def run(wexp: int, wman: int, kind: str | int, mode: int = 2, shard: int = 0, shards: int = 1, **short: Any) -> None:
    simulate(build(ZkfFormat(wexp, wman), mode, **short), kind, shard, shards)


def simulate(model: Cordic, kind: str | int, shard: int = 0, shards: int = 1, alternate: bool = False) -> None:
    """The bench's seed follows the configuration."""
    found = cases(model, kind, shard, shards, alternate=alternate)
    run_handshake(model.component(), found, zlib.crc32(model.name.encode()) % 3, overlap=overlaps(model))


def cut(model: Cordic) -> None:
    """
    A transaction of each mode cut by reset in every cycle and followed by one of the other mode; at one mode, by
    another of its own.
    """
    fmt = model.fmt
    numbers = directed_numbers(fmt)
    # The two of a mode differ in exponent, significand, octant and path, so that what one left shows in the other.
    rotations = [(Fraction(13, 32), numbers["two"], 0), (Fraction(-75, 64), numbers["one"], 1)]
    vectorings = [(Fraction(11, 8), Fraction(-5, 2), 0), (Fraction(-3, 16), Fraction(7, 2), 1)]
    turned = [case(model, False, fmt.encode(x).bits, b, noise) for x, b, noise in rotations if model.mode != 1]
    angled = [
        case(model, True, fmt.encode(y).bits, fmt.encode(x).bits, noise)
        for y, x, noise in vectorings
        if model.mode != 0
    ]
    first, then = (turned[0], angled[0]) if model.mode == 2 else (turned or angled)
    run_cuts(model.component(), [(first, then), (then, first)], overlap=overlaps(model))


def overlaps(model: Cordic) -> bool:
    """Whether the next transaction is taken in the cycle the previous one is handed over."""
    timing = model.timing
    return all(
        t.initiation_interval == t.latency for t in ([timing] if isinstance(timing, Timing) else timing.values())
    )


def rotation_strata(fmt: ZkfFormat, seed: int = SEED) -> list[int]:
    """
    Where a uniform draw does not land: every exponent, with the ends of its fractions; the smallest angles; a band
    around every eighth of a turn, where a result goes to zero or the octant folds; and, binades away from every
    quarter, where the linear path takes over from the iterations.
    """
    rng = np.random.default_rng(seed)
    count, band = 1 << fmt.wfrac, 96
    out = []
    for exp in range(1, fmt.exp_inf):
        fractions = [int(f) for f in rng.integers(0, count, 6)] + [0, 1, fmt.frac_mask, (count >> 1) | 1]
        out += [fmt.pack(sign, exp, frac).bits for sign in (0, 1) for frac in fractions]
    ends = sorted({*range(min(band, count)), *range(max(0, count - band), count)})
    out += [fmt.pack(s, exp, f).bits for exp in range(1, min(7, fmt.exp_inf)) for s in (0, 1) for f in ends]
    for eighth in range(1, 9):
        turn = Fraction(eighth, 8)
        top = _floor_log2(turn)
        for e in (top - 1, top):
            if 1 <= e + fmt.bias < fmt.exp_inf:
                at = round((turn - Fraction(2) ** e) / Fraction(2) ** (e - fmt.wfrac))
                fractions = [f for f in range(at - band, at + band + 1) if 0 <= f < count]
                out += [fmt.pack(sign, e + fmt.bias, f).bits for f in fractions for sign in (0, 1)]
    spec = _constants(fmt.wman)
    handoff = spec.wt + 2 - (spec.tsa.bit_length() - 1)  # the exponent, in turns, of where the paths meet
    for quarter in range(5):  # the fourth is not the first again: it alone nears a whole turn from below
        for j in range(max(1, handoff - 4), min(handoff + 10, fmt.wman + 2)):
            offsets = [Fraction(1, 1 << j)]
            offsets += [Fraction(int(rng.integers(1 << 20, 1 << 21)), 1 << (20 + j)) for _ in range(4)]
            values = [Fraction(quarter, 4) + side * offset for side in (1, -1) for offset in offsets]
            out += [fmt.encode(sign * value).bits for value in values if value > 0 for sign in (1, -1)]
    return out + ROTATED.get((fmt.wexp, fmt.wman), [])


def vectoring_strata(fmt: ZkfFormat, seed: int = SEED) -> list[tuple[int, int]]:
    """
    `(y, x)`. The angle follows the difference of the exponents, so each operand is swept over every exponent
    against the other near 1, which crosses the bypass and every quadrant; the diagonals; the smallest `|y|` against
    an `x` around where the angle underflows, and on both sides of the bypass; a ladder `|y| = |x| * 2**-j` between
    those; and the two ends of the range, lengths around the largest finite one and angles under the least normal.
    """
    rng = np.random.default_rng(seed)
    count, sign_bit = 1 << fmt.wfrac, 1 << fmt.sign_shift

    def normal(sign: int, exp: int, frac: int) -> int:
        return fmt.pack(sign, exp, frac).bits

    exponents = range(1, fmt.exp_inf)
    swept = [0, sign_bit, fmt.exp_inf << fmt.wfrac, sign_bit | (fmt.exp_inf << fmt.wfrac)]
    swept += [normal(s, e, f) for s in (0, 1) for e in exponents for f in (0, 1, fmt.frac_mask, count >> 1)]
    anchors = [normal(s, fmt.bias, f) for s in (0, 1) for f in (0, fmt.frac_mask)]
    out = {pair for w in swept for a in anchors for pair in ((w, a), (a, w))}
    out |= {(normal(s, e, 0), normal(s ^ flip, e, 0)) for s in (0, 1) for e in exponents for flip in (0, 1)}
    small = sorted({*range(96), *range(max(0, count - 48), count), count // 2})
    wide = (0, count // 2, count - 1)
    for ey in (1, 2, 3):
        for xe in range(ey + fmt.bias - 6, ey + fmt.bias + 3):
            if 1 <= xe < fmt.exp_inf:
                out |= {(normal(sy, ey, yf), normal(0, xe, xf)) for sy in (0, 1) for yf in small for xf in wide}
    bypass = _constants(fmt.wman).zf - fmt.wman - _GUARD_DIV
    for ey in (1, max(1, fmt.bias // 2), max(1, fmt.bias - 3)):
        for xe in range(ey + bypass - 1, ey + bypass + 3):
            if 1 <= xe < fmt.exp_inf:
                out |= {(normal(sy, ey, yf), normal(0, xe, xf)) for sy in (0, 1) for yf in small for xf in wide}
    for ye in range(fmt.bias, max(0, fmt.bias - fmt.wman - 7), -1):
        for _ in range(6):
            yf, xf = int(rng.integers(0, count)), int(rng.integers(0, count))
            out |= {(normal(sy, ye, yf), normal(sx, fmt.bias, xf)) for sy in (0, 1) for sx in (0, 1)}
    # With the largest finite `x` the length passes it for any `y`, and rounds to the infinity from a `y` of about
    # `2**(emax + 1 - WMAN / 2)` up: below that an infinity is spurious, above it is due.
    top = normal(0, fmt.exp_inf - 1, fmt.frac_mask)
    for ye in range(max(1, fmt.exp_inf - 1 - fmt.wman // 2 - 3), fmt.exp_inf):
        out |= {(normal(0, ye, yf), top) for yf in (0, 1, count // 2, count - 1)}
    out |= {(top, top), (_saturating(fmt), top)}
    # An angle from half the least normal to the whole of it, where a zero and the least normal are its neighbors.
    with mp.workprec(4 * fmt.wman + 80):
        for k in (Fraction(1, 2), Fraction(3, 4), Fraction(1)):
            for xe in {e for e in (fmt.bias, fmt.bias + 2, fmt.exp_inf - 2) if 1 <= e < fmt.exp_inf}:
                x = fmt.pack(0, xe, 0)
                slope = mp.tan(2 * mp.pi * mp.mpf(k.numerator) / k.denominator * mp.ldexp(1, 1 - fmt.bias))
                y = fmt.encode(x.to_fraction() * Fraction(mp.nstr(slope, 40)))
                if not (y.is_zero or y.is_inf):
                    out |= {(normal(sy, y.exp, y.bits & fmt.frac_mask), x.bits) for sy in (0, 1)}
    return sorted(out | set(VECTORED.get((fmt.wexp, fmt.wman), [])))


def _floor_log2(value: Fraction) -> int:
    exp = value.numerator.bit_length() - value.denominator.bit_length()  # the floor or one above it
    return exp - (Fraction(2) ** exp > value)


def _saturating(fmt: ZkfFormat) -> int:
    """
    The `y` that with the largest finite `x` puts the length in the upper half of the last step below the infinity,
    where rounding carries and only the packer's saturation keeps the result finite. In units of the last place, so
    that nothing grows with WEXP: with `x = U`, `(U + 3/4)**2 - U**2 = 3U/2 + 9/16`, so `y = sqrt(24U + 9) / 4`.
    """
    shift = 64
    steps = Fraction(math.isqrt((24 * ((1 << fmt.wman) - 1) + 9) << (2 * shift)), 4 << shift)
    k = _floor_log2(steps)
    frac, exp = round((steps / Fraction(2) ** k - 1) * (1 << fmt.wfrac)), k + fmt.exp_max_finite - fmt.wfrac
    if frac > fmt.frac_mask:
        frac, exp = 0, exp + 1
    return fmt.pack(0, exp, frac).bits


def _random_pair(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int]:
    kind = int(rng.integers(0, 11))
    special = {0: (random_zero, random_operand), 1: (random_operand, random_zero), 2: (random_zero, random_zero)}
    special |= {3: (random_inf, random_operand), 4: (random_operand, random_inf), 5: (random_inf, random_inf)}
    if kind in special:
        return special[kind][0](fmt, rng), special[kind][1](fmt, rng)
    # Magnitudes that are alike, at the edge of an octant, and far apart, where the bypass is.
    fractions = [0, fmt.frac_mask, fmt.frac_mask >> 1]
    near, low, high = [fmt.bias], [1, 2, 3], [fmt.exp_max_finite, fmt.exp_max_finite - 1]
    exponents = {6: (near, near), 7: (low, high), 8: (high, low)}
    if kind in exponents:
        return tuple(random_normal_near(fmt, rng, exps, fractions) for exps in exponents[kind])  # type: ignore
    return random_operand(fmt, rng), random_operand(fmt, rng)


def _engine_angle(fmt: ZkfFormat, y: int, x: int) -> int:
    """The angle in turns of `(|x|, |y|)` that the reference sums before it unfolds, for `|y| < |x|` off the bypass."""
    spec, a, b = _constants(fmt.wman), Zkf(fmt, x), Zkf(fmt, y)
    up = spec.xf - fmt.wfrac - 2
    vector = _cordic_vector(spec, a.significand() << up, (b.significand() << up) >> (a.exp - b.exp))
    return _vector_residual(spec, *vector)

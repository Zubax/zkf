"""Operands the benches of several operators share: the corner patterns of a format and its random draws."""

import numpy as np

from zkf import ZkfFormat

SEED = 0x9E3779B97F4A7C15


def signed_range(width: int) -> range:
    return range(-(1 << (width - 1)), 1 << (width - 1))


def raw_corners(fmt: ZkfFormat) -> list[int]:
    """Zeros and infinities of both signs, with and without a payload, as raw bits."""
    sign = 1 << fmt.sign_shift
    inf = fmt.exp_inf << fmt.wfrac
    return [0, 1, fmt.frac_mask, sign, sign | min(fmt.frac_mask, 1), inf, sign | inf, (1 << fmt.wfull) - 1]


def directed_numbers(fmt: ZkfFormat) -> dict[str, int]:
    if fmt.bias - 1 < 1 or fmt.bias + 1 > fmt.exp_max_finite:
        raise ValueError(f"format too small for generic directed values: {fmt}")
    return {
        "zero": 0,
        "neg_zero": fmt.pack(1, 0, min(fmt.frac_mask, 1)).bits,
        "one": fmt.pack(0, fmt.bias, 0).bits,
        "minus_one": fmt.pack(1, fmt.bias, 0).bits,
        "half": fmt.pack(0, fmt.bias - 1, 0).bits,
        "one_and_half": fmt.pack(0, fmt.bias, 1 << (fmt.wfrac - 1)).bits,
        "one_and_quarter": fmt.pack(0, fmt.bias, 1 << (fmt.wfrac - 2)).bits,
        "one_and_three_quarters": fmt.pack(0, fmt.bias, 3 << (fmt.wfrac - 2)).bits,
        "two": fmt.pack(0, fmt.bias + 1, 0).bits,
        "min_normal": fmt.pack(0, 1, 0).bits,
        "neg_min_normal": fmt.pack(1, 1, 0).bits,
        "max_finite": fmt.pack(0, fmt.exp_max_finite, fmt.frac_mask).bits,
        "neg_max_finite": fmt.pack(1, fmt.exp_max_finite, fmt.frac_mask).bits,
        "pos_inf": fmt.inf(0).bits,
        "neg_inf": fmt.inf(1).bits,
        "noncanonical_pos_inf": fmt.pack(0, fmt.exp_inf, min(fmt.frac_mask, 1)).bits,
        "noncanonical_neg_inf": fmt.pack(1, fmt.exp_inf, fmt.frac_mask).bits,
    }


def random_normal(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    return fmt.pack(
        int(rng.integers(0, 2)), int(rng.integers(1, fmt.exp_max_finite + 1)), int(rng.integers(0, fmt.frac_mask + 1))
    ).bits


def random_normal_near(fmt: ZkfFormat, rng: np.random.Generator, exponents: list[int], fractions: list[int]) -> int:
    exp = int(np.clip(int(rng.choice(exponents)) + int(rng.integers(-1, 2)), 1, fmt.exp_max_finite))
    frac = int(np.clip(int(rng.choice(fractions)) + int(rng.integers(-16, 17)), 0, fmt.frac_mask))
    return fmt.pack(int(rng.integers(0, 2)), exp, frac).bits


def random_zero(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    frac = 0 if int(rng.integers(0, 3)) else int(rng.integers(0, fmt.frac_mask + 1))
    return fmt.pack(int(rng.integers(0, 2)), 0, frac).bits


def random_inf(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    frac = 0 if int(rng.integers(0, 3)) else int(rng.integers(0, fmt.frac_mask + 1))
    return fmt.pack(int(rng.integers(0, 2)), fmt.exp_inf, frac).bits


def random_operand(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    mode = int(rng.integers(0, 12))
    if mode == 0:
        return random_zero(fmt, rng)
    if mode == 1:
        return random_inf(fmt, rng)
    if mode == 2:
        return random_normal_near(fmt, rng, [1, 2, 3], [0, 1, fmt.frac_mask])
    if mode == 3:
        return random_normal_near(fmt, rng, [fmt.bias - 1, fmt.bias, fmt.bias + 1], [0, 1, 2])
    if mode == 4:
        top = fmt.exp_max_finite
        return random_normal_near(fmt, rng, [top - 2, top - 1, top], [0, 1 << (fmt.wfrac - 1), fmt.frac_mask])
    return random_normal(fmt, rng)


def unary_operands(fmt: ZkfFormat, count: int, seed: int = SEED) -> list[int]:
    """`count` 0 is every operand; otherwise the corners and the directed numbers made up to `count` at random."""
    if not count:
        return list(range(1 << fmt.wfull))
    out = raw_corners(fmt) + (list(directed_numbers(fmt).values()) if fmt.wexp >= 3 else [])
    out = list(dict.fromkeys(out))
    rng = np.random.default_rng(seed)
    seen = set(out)
    while len(out) < min(count, 1 << fmt.wfull):  # no more than the format has
        x = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def directed_integers(wint: int) -> list[int]:
    """The ends of the range and what is next to them, the small integers, and the neighbors of powers of two."""
    low, high = -(1 << (wint - 1)), (1 << (wint - 1)) - 1
    half = 1 << max(0, wint - 2)
    out = [0, 1, -1, high, low, high - 1, low + 1, 2, -2, half, -half, half - 1, 1 - half]
    out += [sign * n for sign in (1, -1) for n in (3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 127, 129)]
    return list(dict.fromkeys(n for n in out if low <= n <= high))


def random_integer(wint: int, rng: np.random.Generator) -> int:
    low, high = -(1 << (wint - 1)), (1 << (wint - 1)) - 1
    mode = int(rng.integers(0, 8))
    if mode == 0:
        return 0
    if mode == 1:
        magnitude = int(rng.integers(1, min(16, high + 1)))
        return -magnitude if int(rng.integers(0, 2)) else magnitude
    if mode == 2:
        return int(rng.choice([low, high, low + 1, high - 1]))
    if mode == 3 and wint >= 4:  # next to a power of two
        near = (1 << int(rng.integers(1, wint - 1))) + int(rng.integers(-4, 5))
        near = max(low, min(high, near))
        return -near if int(rng.integers(0, 2)) else near
    raw = random_bits(wint, rng)
    return raw - (1 << wint) if raw >> (wint - 1) else raw


def random_bits(width: int, rng: np.random.Generator) -> int:
    value = 0
    offset = 0
    while offset < width:
        chunk = min(31, width - offset)
        value |= int(rng.integers(0, 1 << chunk)) << offset
        offset += chunk
    return value


def random_pack_mag_scale(fmt: ZkfFormat, rng: np.random.Generator) -> tuple[int, int, int]:
    wmag = 2 * fmt.wman
    sign = int(rng.integers(0, 2))
    mode = int(rng.integers(0, 12))
    if mode == 0:
        mag = 0
    elif mode <= 3:
        width = int(rng.integers(1, wmag + 1))
        mag = (1 << (width - 1)) | random_bits(width - 1, rng)
    elif mode <= 5:
        centers = [1, 1 << max(fmt.wfrac - 1, 0), 1 << fmt.wfrac, 1 << fmt.wman, (1 << wmag) - 1]
        mag = min(max(centers[int(rng.integers(0, len(centers)))] + int(rng.integers(-16, 17)), 0), ((1 << wmag) - 1))
    else:
        mag = random_bits(wmag, rng)
    scale_min = -(1 << (fmt.wexp + 1))
    scale_max = (1 << (fmt.wexp + 1)) - 1
    if mode in (1, 4):
        scale = int(rng.integers(scale_min, fmt.min_exp_unbiased + 4))
    elif mode in (2, 5):
        scale = int(rng.integers(fmt.max_exp_unbiased - wmag - 4, scale_max + 1))
    elif mode == 3:
        scale = int(rng.choice([fmt.min_exp_unbiased, fmt.min_exp_unbiased - 1, -1, 0, 1, fmt.max_exp_unbiased]))
    else:
        scale = int(rng.integers(scale_min, scale_max + 1))
    return sign, mag, scale


def exp2_operands(fmt: ZkfFormat, count: int, cf: int, seed: int = SEED) -> list[int]:
    """
    The corners; powers of two of both signs, whose results are exact and meet the overflow and the underflow; the
    ends of the result's range, where zero, the least normal and the normals meet below and the largest finite
    result sits above; then random ones, half of them at an exponent that leaves a reduced argument of `cf` bits
    nonzero, the others saturating at a wide exponent.
    """
    out = raw_corners(fmt)
    if fmt.wexp >= 3:
        out += directed_numbers(fmt).values()
        for k in (1, 2, 3, -1, -2, -3):
            if 1 <= fmt.bias + k <= fmt.exp_max_finite:
                out += [fmt.pack(sign, fmt.bias + k, 0).bits for sign in (0, 1)]
        shift, top = fmt.wfrac - (fmt.wexp - 2), fmt.bias + fmt.wexp - 2
        half = 1 << (shift - 1)
        ends = [(1, ((fmt.bias + 1) << shift) - 1), (1, (fmt.bias << shift) + half), (1, (fmt.bias << shift) + 1)]
        ends += [(1, fmt.bias << shift), (1, (fmt.bias << shift) - half), (1, ((fmt.bias - 1) << shift) + 1)]
        ends += [(1, (fmt.bias - 1) << shift), (0, ((fmt.bias + 1) << shift) - 1)]
        out += [fmt.pack(sign, top, q - (1 << fmt.wfrac)).bits for sign, q in ends]
        out += [fmt.pack(0, fmt.bias - 1, 1 << (fmt.wfrac - 1)).bits, fmt.pack(0, 1, 0).bits, fmt.pack(1, 1, 0).bits]
    out = list(dict.fromkeys(out))
    rng = np.random.default_rng(seed)
    seen = set(out)
    exponents = (max(fmt.bias - cf, 1), fmt.bias + fmt.wexp - 1)
    while len(out) < min(count, 1 << fmt.wfull):
        mode = int(rng.integers(0, 4))
        if mode >= 2:
            x = fmt.pack(int(rng.integers(0, 2)), int(rng.integers(*exponents)), random_bits(fmt.wfrac, rng)).bits
        else:
            x = random_operand(fmt, rng) if mode else random_bits(fmt.wfull, rng)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def log2_operands(fmt: ZkfFormat, count: int, seed: int = SEED) -> list[int]:
    """
    The corners; powers of two, whose logarithms are integers, with those whose result is a power of two itself and
    their neighbors, where the result's exponent turns; the largest exponent, where re-centering takes `e` to its
    bound; the two neighbors of one, where the result cancels; then random ones, half of them positive.
    """
    out = raw_corners(fmt)
    if fmt.wexp >= 3:
        out += directed_numbers(fmt).values()
        steps = {k for k in (-2, -1, 0, 1, 2)}
        steps |= {sign * ((1 << p) + near) for p in range(fmt.wexp) for near in (-1, 0, 1) for sign in (1, -1)}
        for k in sorted(steps):
            if 1 <= fmt.bias + k <= fmt.exp_max_finite:
                out += [fmt.pack(sign, fmt.bias + k, 0).bits for sign in (0, 1) if not sign or abs(k) <= 2]
        ones = fmt.frac_mask
        out += [fmt.pack(0, fmt.exp_max_finite, frac).bits for frac in sorted({0, 1, ones >> 1, ones - 1, ones})]
        out += [fmt.pack(0, fmt.bias, 1).bits, fmt.pack(0, fmt.bias - 1, ones).bits]
    out = list(dict.fromkeys(out))
    rng = np.random.default_rng(seed)
    seen = set(out)
    while len(out) < min(count, 1 << fmt.wfull):
        mode = int(rng.integers(0, 4))
        if mode >= 2:
            x = fmt.pack(0, int(rng.integers(1, fmt.exp_inf)), random_bits(fmt.wfrac, rng)).bits
        else:
            x = random_operand(fmt, rng) if mode else random_bits(fmt.wfull, rng)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

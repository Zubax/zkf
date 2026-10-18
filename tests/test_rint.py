from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import (
    SEED,
    directed_numbers,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
    raw_corners,
)
from zkf import Rint, RoundMode, Zkf, ZkfFormat


def directed(fmt: ZkfFormat, wint: int) -> list[int]:
    """The corners, the ties between small integers with their neighbors, and the values around the integer's rail."""
    out = raw_corners(fmt)
    values: list[Fraction] = []
    if fmt.wexp >= 3:
        named = directed_numbers(fmt)
        out += list(named.values())
        out += [named[n] | (1 << fmt.sign_shift) for n in ("half", "one_and_quarter", "one_and_half", "two")]
        out.append(named["one_and_three_quarters"] | (1 << fmt.sign_shift))
        for tie in (Fraction(1, 2), Fraction(3, 2), Fraction(5, 2)):
            values += [tie - fmt.epsilon, tie + fmt.epsilon]
    rail_exp = wint - 1
    if fmt.min_exp_unbiased <= rail_exp <= fmt.max_exp_unbiased:
        rail = Fraction(2) ** (rail_exp)
        below = max(Fraction(1), Fraction(2) ** (rail_exp - 1 - fmt.wfrac))
        above = max(Fraction(1), Fraction(2) ** (rail_exp - fmt.wfrac))
        values += [rail, rail - Fraction(1, 2), rail + Fraction(1, 2), rail - below, rail + above]
    return out + [fmt.encode(sign * value).bits for value in values for sign in (1, -1)]


def drawn(fmt: ZkfFormat, wint: int, rng: np.random.Generator) -> int:
    bias, top, ones, half = fmt.bias, fmt.exp_max_finite, fmt.frac_mask, 1 << (fmt.wfrac - 1)
    choice = int(rng.integers(0, 10))
    if choice == 0:
        return random_zero(fmt, rng)
    if choice == 1:
        return random_inf(fmt, rng)
    if choice == 2:  # magnitudes around one
        return random_normal_near(fmt, rng, [bias - 2, bias - 1, bias, bias + 1], [0, 1, ones])
    if choice == 3:  # halves between small integers
        return random_normal_near(fmt, rng, [bias, bias + 1, bias + 2], [0, half, half | 1, ones])
    if choice == 4:  # the largest exponents, where the float rounds up into an infinity in a tiny format
        return random_normal_near(fmt, rng, [top - 2, top - 1, top], [0, half, ones])
    if choice == 5:  # around the integer's rails
        return random_normal_near(fmt, rng, [bias + wint - 2, bias + wint - 1], [0, 1, ones])
    if choice == 6:  # around the exponent at which the fraction ends
        return random_normal_near(fmt, rng, [bias + fmt.wfrac - 1, bias + fmt.wfrac], [0, 1, ones])
    if choice == 7:
        return random_normal(fmt, rng)
    return random_operand(fmt, rng)


def run(wexp: int, wman: int, wint: int, count: int, **knobs: Any) -> None:
    simulate(Rint(ZkfFormat(wexp, wman), wint=wint, **knobs), count)


def simulate(model: Rint, count: int) -> None:
    """Every operand in every rounding mode; `count` 0 is every operand, otherwise the directed and random ones."""
    fmt, wint = model.fmt, model.wint
    operands = list(dict.fromkeys(directed(fmt, wint))) if count else list(range(1 << fmt.wfull))
    rng = np.random.default_rng(SEED)
    seen = set(operands)
    while len(operands) * len(RoundMode) < count:
        operand = drawn(fmt, wint, rng)
        if operand not in seen:
            seen.add(operand)
            operands.append(operand)
    cases = []
    for bits in operands:
        for mode in RoundMode:
            cases.append(Case({"a": bits, "round_mode": int(mode)}, model.evaluate(Zkf(fmt, bits), mode)._asdict()))
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp, wman, wint, count. Of the tiny ones: a float that rounds up into an infinity while its integer is in range;
# wide shifts and the exponent's rail; a round-up into the positive rail with an integer narrower than the
# significand; an integer that takes the whole significand; more fraction bits than exponent or integer hold.
FORMATS = [
    (2, 4, 4, 0),
    (3, 4, 8, 0),
    (3, 5, 8, 0),
    (5, 11, 16, 512),
    (6, 18, 32, 768),
    (8, 24, 32, 1024),
    (6, 18, 128, 256),
    (2, 4, 2, 0),
    (2, 4, 8, 0),
    (5, 4, 12, 0),
    (4, 6, 5, 0),
    (3, 4, 5, 0),
    (2, 9, 2, 0),
    (6, 18, 24, 768),
    (8, 36, 44, 768),
    (11, 53, 32, 384),
    (11, 53, 64, 384),
]


@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int, int]) -> None:
    run(*row)


EVERY: dict[str, Any] = {"stage_input": 1, "stage_core": Rint.Core.MASK, "stage_output": 1}


@pytest.mark.parametrize(
    "knobs",
    [
        *({knob: value} for knob, value in EVERY.items()),
        EVERY,
        {"stage_input": 2},
        {"stage_output": 2},
    ],
    ids=str,
)
def test_stages(knobs: dict[str, Any]) -> None:
    run(3, 4, 8, 0, **knobs)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Rint)])
def test_catalog(model: Rint) -> None:
    simulate(model, 512)


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    assert not Rint(fmt).clocked
    with pytest.raises(ValueError):
        Rint(fmt, wint=1)


@pytest.mark.deep
@pytest.mark.parametrize("stages", range(8))
def test_deep_stages(stages: int) -> None:
    run(3, 6, 5, 0, **{knob: value for bit, (knob, value) in enumerate(EVERY.items()) if (stages >> bit) & 1})


@pytest.mark.deep
@pytest.mark.parametrize(
    "row",
    [
        (2, 5, 3, 0, {}),
        (4, 5, 7, 0, {}),
        (3, 5, 5, 0, {}),
        (5, 4, 8, 0, {}),
        (3, 6, 4, 0, {}),
        (5, 11, 9, 512, {}),
        (6, 17, 33, 512, {}),
        (8, 24, 17, 512, {}),
        (8, 36, 32, 768, {}),
        (8, 48, 64, 384, {}),
        (8, 36, 44, 1024, {}),
        (8, 36, 44, 768, EVERY),
        (4, 5, 7, 0, EVERY),
    ],
    ids=str,
)
def test_deep_rows(row: tuple[int, int, int, int, dict[str, Any]]) -> None:
    run(row[0], row[1], row[2], row[3], **row[4])

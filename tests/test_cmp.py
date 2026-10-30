import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import SEED, directed_numbers, random_bits, random_operand, raw_corners
from zkf import Cmp, Zkf, ZkfFormat


def classes(fmt: ZkfFormat) -> list[int]:
    """Zeros and infinities of both signs at several fractions: all zeros are equal, and so are infinities of a sign."""
    fractions = sorted({0, 1, fmt.frac_mask >> 1, fmt.frac_mask})
    inf = fmt.exp_inf << fmt.wfrac
    return [(sign << fmt.sign_shift) | body | frac for sign in (0, 1) for frac in fractions for body in (0, inf)]


def directed(fmt: ZkfFormat) -> list[tuple[int, int]]:
    numbers = list(directed_numbers(fmt).values()) if fmt.wexp >= 3 else []
    values = raw_corners(fmt) + numbers
    out = [(a, b) for a in values for b in values]
    out += [(a, b) for a in classes(fmt) for b in classes(fmt)]
    # Each number against itself and against its neighbor, which is never equal to it.
    for value in numbers:
        out += [(value, value), (value, (value ^ 1) & ((1 << fmt.wfull) - 1))]
    return out


def pairs(fmt: ZkfFormat, count: int) -> list[tuple[int, int]]:
    """`count` 0 is every pair; otherwise the directed ones made up to `count` with random ones."""
    if not count:
        return [(a, b) for a in range(1 << fmt.wfull) for b in range(1 << fmt.wfull)]
    out = list(dict.fromkeys(directed(fmt)))
    rng = np.random.default_rng(SEED)
    seen = set(out)
    while len(out) < count:
        mode = int(rng.integers(0, 8))
        a = random_bits(fmt.wfull, rng) if mode in (5, 7) else random_operand(fmt, rng)
        b = random_bits(fmt.wfull, rng) if mode in (6, 7) else random_operand(fmt, rng)
        if (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def run(wexp: int, wman: int, count: int, **knobs: int) -> None:
    simulate(Cmp(ZkfFormat(wexp, wman), **knobs), count)


def simulate(model: Cmp, count: int) -> None:
    fmt = model.fmt
    cases = [Case({"a": a, "b": b}, model.evaluate(Zkf(fmt, a), Zkf(fmt, b))._asdict()) for a, b in pairs(fmt, count)]
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp, wman, count; the catalog has more.
FORMATS = [
    (2, 4, 0),
    (3, 4, 0),
    (3, 5, 512),
    (4, 6, 512),
    (5, 11, 768),
    (7, 17, 768),
    (8, 24, 1024),
    (11, 53, 384),
]


@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int]) -> None:
    run(*row)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Cmp)])
def test_catalog(model: Cmp) -> None:
    simulate(model, 768)


@pytest.mark.parametrize(
    "knobs",
    [
        {"stage_input": 1},
        {"stage_input": 2},
        {"stage_output": 1},
        {"stage_input": 1, "stage_output": 1},
        {"stage_input": 2, "stage_output": 1},
        {"stage_output": 2},
        {"stage_input": 2, "stage_output": 2},
    ],
    ids=str,
)
def test_stages(knobs: dict[str, int]) -> None:
    run(3, 4, 0, **knobs)


def test_the_reference_orders_the_classes() -> None:
    """Of equal operands `min` is `b` and `max` is `a`; a zero is below every positive number, above every negative."""
    fmt = ZkfFormat(4, 6)
    cmp = Cmp(fmt).evaluate
    zero, one, minus_one = Zkf(fmt, 1 << fmt.sign_shift | 5), fmt.pack(0, fmt.bias, 0), fmt.pack(1, fmt.bias, 0)
    inf, minus_inf = fmt.inf(0), fmt.inf(1)
    assert cmp(Zkf(fmt, 0), zero) == (False, True, False, zero, Zkf(fmt, 0))
    assert cmp(zero, one).a_lt_b and cmp(minus_one, zero).a_lt_b and cmp(one, inf).a_lt_b
    assert cmp(minus_inf, minus_one).a_lt_b and cmp(inf, Zkf(fmt, inf.bits | 3)).a_eq_b and cmp(minus_inf, inf).a_lt_b


DEEP = [
    (2, 5, 0),
    (4, 5, 0),
    (2, 7, 0),
    (3, 6, 0),
    (5, 4, 0),
    (4, 6, 0),
    (3, 7, 512),
    (6, 17, 768),
    (8, 23, 768),
    (7, 12, 512),
    (9, 24, 512),
    (6, 19, 512),
]


@pytest.mark.deep
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int]) -> None:
    run(*row)


@pytest.mark.deep
def test_deep_registered() -> None:
    run(2, 5, 0, stage_output=1)

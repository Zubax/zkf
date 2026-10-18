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
from zkf import Resize, Zkf, ZkfFormat


def directed(source: ZkfFormat, out: ZkfFormat) -> list[int]:
    """The corners of the format read, and what lands on half the smallest normal of the other and beside it."""
    values = (list(directed_numbers(source).values()) if source.wexp >= 3 else []) + raw_corners(source)
    half = out.min_exp_unbiased - 1
    if source.min_exp_unbiased <= half <= source.max_exp_unbiased:
        exp = half + source.bias
        values += [source.pack(sign, exp, frac).bits for sign in (0, 1) for frac in (0, 1 << (source.wfrac - 1))]
    if source.min_exp_unbiased <= half - 1 <= source.max_exp_unbiased:
        values.append(source.pack(0, half - 1 + source.bias, source.frac_mask).bits)
    return values


def drawn(fmt: ZkfFormat, rng: np.random.Generator) -> int:
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    mode = int(rng.integers(0, 9))
    if mode == 0:
        return random_zero(fmt, rng)
    if mode == 1:
        return random_inf(fmt, rng)
    if mode == 2:
        return random_normal_near(fmt, rng, [1, 2, 3], [0, 1, ones])
    if mode == 3:
        return random_normal_near(fmt, rng, [fmt.bias - 1, fmt.bias, fmt.bias + 1], [0, 1, ones])
    if mode == 4:
        return random_normal_near(fmt, rng, [top - 2, top - 1, top], [0, 1 << (fmt.wfrac - 1), ones])
    if mode == 5:
        return random_normal(fmt, rng)
    return random_operand(fmt, rng)


def run(wexp_in: int, wman_in: int, wexp: int, wman: int, count: int, **knobs: int) -> None:
    simulate(Resize(ZkfFormat(wexp, wman), fmt_in=ZkfFormat(wexp_in, wman_in), **knobs), count)


def simulate(model: Resize, count: int) -> None:
    """`count` 0 is every operand; otherwise the directed ones made up to `count` with random ones."""
    source, out = model.fmt_in, model.fmt
    operands = list(dict.fromkeys(directed(source, out))) if count else list(range(1 << source.wfull))
    rng = np.random.default_rng(SEED)
    seen = set(operands)
    while len(operands) < count:
        value = drawn(source, rng)
        if value not in seen:
            seen.add(value)
            operands.append(value)
    cases = [Case({"a": a}, {"y": model.evaluate(Zkf(source, a))}) for a in operands]
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp_in, wman_in, wexp, wman, count: each of the two dimensions the same, wider and narrower; the catalog has more.
FORMATS = [
    (3, 4, 3, 4, 0),
    (3, 4, 4, 4, 0),
    (3, 4, 3, 6, 0),
    (3, 5, 3, 4, 0),
    (3, 4, 4, 6, 0),
    (4, 6, 3, 4, 0),
    (5, 4, 3, 4, 0),
    (5, 4, 3, 6, 0),
    (8, 24, 6, 18, 512),
    (11, 53, 8, 24, 384),
]


@pytest.mark.parametrize("stage_input", [0, 1])
@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int, int, int], stage_input: int) -> None:
    run(*row, stage_input=stage_input)


# Stages that do nothing, and the registered result of each path.
@pytest.mark.parametrize(
    "row",
    [
        (3, 4, 4, 6, 0, {"stage_input": 2}),
        (4, 5, 4, 5, 0, {"stage_output": 1}),
        (5, 6, 3, 4, 0, {"stage_output": 1}),
        (5, 6, 3, 4, 0, {"stage_output": 3}),
    ],
    ids=str,
)
def test_rows(row: tuple[int, int, int, int, int, dict[str, int]]) -> None:
    run(*row[:5], **row[5])


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Resize)])
def test_catalog(model: Resize) -> None:
    simulate(model, 768)


def test_no_register_is_no_clock() -> None:
    fmt = ZkfFormat(3, 4)
    assert Resize(fmt, fmt_in=fmt).name == "zkf_resize_e3m4_e3m4_input0_output0_latency0"
    assert not Resize(fmt, fmt_in=fmt).clocked


DEEP = [
    (4, 5, 4, 4, 0),
    (4, 4, 4, 5, 0),
    (2, 5, 4, 7, 0),
    (4, 7, 2, 5, 0),
    (5, 5, 3, 4, 0),
    (3, 4, 5, 5, 0),
    (6, 17, 4, 11, 512),
    (5, 11, 6, 17, 512),
]


@pytest.mark.deep
@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize("stage_input", [0, 1])
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int, int, int], stage_input: int, stage_output: int) -> None:
    run(*row, stage_input=stage_input, stage_output=stage_output)


@pytest.mark.deep
@pytest.mark.parametrize(
    "row",
    [
        (8, 48, 8, 24, 384),
        (8, 24, 8, 48, 384),
        (3, 4, 5, 6, 0),
        (5, 6, 3, 4, 0),
        (3, 6, 5, 4, 0),
    ],
    ids=str,
)
def test_deep_rows(row: tuple[int, int, int, int, int]) -> None:
    for stage_input in (0, 1) if not row[4] else (0,):
        run(*row, stage_input=stage_input)

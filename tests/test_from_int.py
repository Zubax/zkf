from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import SEED, directed_integers, random_integer, signed_range
from zkf import FromInt, ZkfFormat


def integers(wint: int, count: int | None) -> list[int]:
    """`count` 0 is every integer; otherwise the directed ones made up to `count` at random, or alone for None."""
    if count == 0:
        return list(signed_range(wint))
    out = directed_integers(wint)
    rng = np.random.default_rng(SEED)
    seen = set(out)
    while len(out) < (count or 0):
        value = random_integer(wint, rng)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def run(wexp: int, wman: int, wint: int, count: int | None, **knobs: Any) -> None:
    simulate(FromInt(ZkfFormat(wexp, wman), wint=wint, **knobs), count)


def simulate(model: FromInt, count: int | None) -> None:
    cases = [Case({"a": value}, {"y": model.evaluate(value)}) for value in integers(model.wint, count)]
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp, wman, wint, count. The last has integers whose exponent an exponent sized for the integer alone would wrap.
FORMATS = [
    (2, 4, 4, 0),
    (3, 4, 8, 0),
    (3, 5, 8, 0),
    (5, 11, 16, 512),
    (6, 18, 32, 768),
    (8, 24, 32, 1024),
    (6, 18, 128, None),
]


@pytest.mark.parametrize("stage_input", [0, 1])
@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int, int | None], stage_input: int) -> None:
    run(*row, stage_input=stage_input)


@pytest.mark.parametrize(
    "knobs",
    [
        *({"stage_core": member} for member in FromInt.Core),
        {"stage_core": ~FromInt.Core(0)},
        {"stage_input": 1, "stage_core": ~FromInt.Core(0), "stage_output": 1},
        {"stage_input": 2},
        {"stage_output": 1},
        {"stage_output": 3},
    ],
    ids=str,
)
def test_stages(knobs: dict[str, Any]) -> None:
    run(3, 4, 8, 0, **knobs)


# Each register alone and all of them, before a normalizer of three levels.
@pytest.mark.parametrize("stage_core", [*FromInt.Core, ~FromInt.Core(0)], ids=str)
def test_registers(stage_core: FromInt.Core) -> None:
    run(6, 18, 32, 512, stage_core=stage_core)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, FromInt)])
def test_catalog(model: FromInt) -> None:
    simulate(model, 512)


DEEP = [
    (3, 5, 5, 0),
    (2, 5, 3, 0),
    (4, 5, 7, 0),
    (4, 6, 5, 0),
    (5, 11, 9, 512),
    (6, 17, 33, 512),
    (8, 24, 17, 512),
]


@pytest.mark.deep
@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize("stage_input", [0, 1])
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int, int], stage_input: int, stage_output: int) -> None:
    run(*row, stage_input=stage_input, stage_output=stage_output)


@pytest.mark.deep
@pytest.mark.parametrize(
    "row",
    [
        (11, 53, 32, 384, {}),
        (6, 18, 32, 256, {"stage_core": FromInt.Core.ABS | FromInt.Core.NORMALIZE}),
        (3, 6, 4, 0, {}),
        (3, 6, 4, 0, {"stage_input": 1}),
        (5, 4, 8, 0, {}),
        (5, 4, 8, 0, {"stage_input": 1}),
        (2, 4, 8, 0, {}),
        (5, 4, 12, 0, {}),
        (2, 9, 2, 0, {}),
    ],
    ids=str,
)
def test_deep_rows(row: tuple[int, int, int, int, dict[str, Any]]) -> None:
    run(row[0], row[1], row[2], row[3], **row[4])

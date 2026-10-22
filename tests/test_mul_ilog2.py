from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import SEED, directed_numbers, random_bits, random_operand, signed_range
from zkf import MulIlog2, Zkf, ZkfFormat


def shifts(fmt: ZkfFormat, wk: int) -> list[int]:
    """Those that take an input across every boundary between classes, and the ends of the range of `k`."""
    low, high = signed_range(wk)[0], signed_range(wk)[-1]
    candidates = {
        0,
        1,
        -1,
        2,
        -2,
        fmt.bias,
        -fmt.bias,
        -(fmt.bias - 1),
        -(fmt.bias + 1),
        fmt.exp_max_finite,
        -fmt.exp_max_finite,
        -(1 << fmt.wexp) - 1,
        -(1 << fmt.wexp),
        (1 << fmt.wexp) - 1,
        1 << fmt.wexp,
        low,
        high,
        low + 1,
        high - 1,
    }
    return sorted(k for k in candidates if low <= k <= high)


def operands(fmt: ZkfFormat) -> list[int]:
    raw = [
        0,
        1,
        fmt.frac_mask,
        1 << fmt.sign_shift,
        fmt.exp_inf << fmt.wfrac,
        (1 << fmt.sign_shift) | (fmt.exp_inf << fmt.wfrac) | min(fmt.frac_mask, 1),
        ((1 << fmt.wfull) - 1),
    ]
    return raw + (list(directed_numbers(fmt).values()) if fmt.wexp >= 3 else [])


def pairs(fmt: ZkfFormat, wk: int, count: int) -> list[tuple[int, int]]:
    """`count` 0 is every pair; otherwise the directed ones made up to `count` with random ones."""
    if not count:
        return [(a, k) for a in range(1 << fmt.wfull) for k in signed_range(wk)]
    out = list(dict.fromkeys((a, k) for a in operands(fmt) for k in shifts(fmt, wk)))
    rng = np.random.default_rng(SEED)
    seen = set(out)
    while len(out) < max(count, len(seen)):
        a = random_operand(fmt, rng) if int(rng.integers(0, 4)) else random_bits(fmt.wfull, rng)
        pair = (a, int(rng.integers(signed_range(wk)[0], signed_range(wk)[-1] + 1)))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def run(wexp: int, wman: int, count: int, wk: int | None = None, **knobs: Any) -> None:
    simulate(MulIlog2(ZkfFormat(wexp, wman), wk=wk, **knobs), count)


def simulate(model: MulIlog2, count: int) -> None:
    fmt, width = model.fmt, model.config["wk"]
    cases = [Case({"a": a, "k": k}, {"y": model.evaluate(Zkf(fmt, a), k)}) for a, k in pairs(fmt, width, count)]
    run_pipeline(model.component(), model.timing.latency, cases)


@pytest.mark.parametrize("stage_core", [MulIlog2.CoreStages(0), MulIlog2.CoreStages.DECODE], ids=str)
@pytest.mark.parametrize("row", [(2, 4, 0), (3, 4, 0), (5, 11, 512), (8, 24, 1024), (11, 53, 384)], ids=str)
def test_formats(row: tuple[int, int, int], stage_core: MulIlog2.CoreStages) -> None:
    run(*row, stage_core=stage_core)


# A narrow `k` cannot leave the normal range; a wide one reaches the boundaries where it saturates.
@pytest.mark.parametrize(
    "knobs",
    [
        {"stage_input": 1},
        {"stage_input": 1, "stage_core": MulIlog2.CoreStages.DECODE},
        {"stage_input": 2},
        {"stage_output": 1},
        {"stage_input": 1, "stage_core": MulIlog2.CoreStages.DECODE, "stage_output": 2},
        {"wk": 2},
        {"wk": 6, "stage_core": MulIlog2.CoreStages.DECODE},
    ],
    ids=str,
)
def test_knobs(knobs: dict[str, Any]) -> None:
    run(3, 4, 0, **knobs)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, MulIlog2)])
def test_catalog(model: MulIlog2) -> None:
    simulate(model, 512)


# The scaling at 8/36 with no register, of a wide `k`; that of a narrow one is the catalog's.
def test_single_cycle() -> None:
    run(8, 36, 512, wk=44)


@pytest.mark.deep
@pytest.mark.parametrize("stage_core", [MulIlog2.CoreStages(0), MulIlog2.CoreStages.DECODE], ids=str)
@pytest.mark.parametrize("row", [(2, 5, 0), (4, 5, 0), (3, 6, 0), (2, 6, 0), (6, 17, 512), (8, 23, 512)], ids=str)
def test_deep(row: tuple[int, int, int], stage_core: MulIlog2.CoreStages) -> None:
    run(*row, stage_core=stage_core)

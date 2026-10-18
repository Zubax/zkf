import random

import pytest

from _bench import Case, run_pipeline
from _operands import SEED
from zkf import Ilog2, Zkf, ZkfFormat


def run(wexp: int, wman: int, wint: int, count: int, **knobs: int) -> None:
    """`count` 0 is every operand; otherwise every exponent at three fractions in both signs, and random bits."""
    fmt = ZkfFormat(wexp, wman)
    model = Ilog2(fmt, wint=wint, **knobs)
    rng = random.Random(SEED)
    operands = [
        (sign << fmt.sign_shift) | (exp << fmt.wfrac) | frac
        for exp in range(1 << fmt.wexp)
        for frac in (0, 1, fmt.frac_mask)
        for sign in (0, 1)
    ] + [rng.getrandbits(fmt.wfull) for _ in range(count)]
    if not count:
        operands = list(range(1 << fmt.wfull))
    cases = [Case({"a": bits}, model.evaluate(Zkf(fmt, bits))._asdict()) for bits in operands]
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp, wman, wint, count: the narrowest integer, a wider one, and the catalog's.
FORMATS = [(2, 4, 3, 0), (3, 4, 8, 0), (6, 18, 7, 512), (8, 24, 9, 512), (8, 24, 32, 512), (11, 53, 64, 512)]


@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize("stage_input", [0, 1, 2])
@pytest.mark.parametrize("row", [*FORMATS, (6, 18, 32, 512)], ids=str)
def test_formats(row: tuple[int, int, int, int], stage_input: int, stage_output: int) -> None:
    run(*row, stage_input=stage_input, stage_output=stage_output)


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    assert Ilog2(fmt, wint=7).timing.latency == 0 and not Ilog2(fmt).clocked
    with pytest.raises(ValueError):
        Ilog2(fmt, wint=6)


DEEP = [(2, 5, 0), (4, 5, 0), (3, 6, 0), (2, 6, 0), (6, 17, 512), (8, 23, 512)]


@pytest.mark.deep
@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize("stage_input", [0, 1, 2])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int], wide: bool, stage_input: int, stage_output: int) -> None:
    run(row[0], row[1], 32 if wide else row[0] + 1, row[2], stage_input=stage_input, stage_output=stage_output)

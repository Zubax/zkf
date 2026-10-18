import numpy as np
import pytest
from amaranth.lib import data

from _bench import Case, run_pipeline
from _operands import SEED
from zkf._shift import Normshift


def inputs(width: int, count: int) -> list[int]:
    """`count` 0 is every input; otherwise every position of the leading one, alone and over ones, and random ones."""
    if not count:
        return list(range(1 << width)) + [0]
    xs = {0, 1, 1 << (width - 1), ((1 << width) - 1)}
    xs |= {1 << i for i in range(width)} | {(1 << i) - 1 for i in range(1, width + 1)}
    rng = np.random.default_rng(SEED)
    while len(xs) < count:
        xs.add(int(rng.integers(0, 1 << min(width, 62))) << int(rng.integers(0, max(1, width - 61))))
    return sorted(xs)


# stage_input, stage_split
STAGES = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def run(width: int, stages: tuple[int, int], count: int = 0) -> None:
    side = data.StructLayout({"tag": 7})
    dut = Normshift(width, stage_input=stages[0], stage_split=stages[1], sideband=side)
    cases = []
    for index, x in enumerate(inputs(width, count)):
        leading = width - x.bit_length()
        tag = side.const({"tag": index & 127})
        # The count and the shifted value of a zero mean nothing.
        want = {"zero": 0, "count": leading, "y": (x << leading) & ((1 << width) - 1)} if x else {"zero": 1}
        cases.append(Case({"x": x, "sb_in": tag}, want | {"sb_out": tag}))
    run_pipeline(dut, sum(stages), cases)


# Every width from the narrowest through the first with three levels, both parities of levels; the second split
# needs three.
@pytest.mark.parametrize(
    "width, stages",
    [
        (width, stages)
        for width in (2, 3, 4, 5, 7, 8, 9, 13, 15, 16, 17)
        for stages in STAGES
        if stages[1] < 2 or width >= 17
    ],
    ids=str,
)
def test_every_input(width: int, stages: tuple[int, int]) -> None:
    run(width, stages)


@pytest.mark.parametrize("width", [21, 27, 32, 39, 56, 64, 65, 75, 108, 109, 130])
@pytest.mark.parametrize("stages", STAGES, ids=str)
def test_wide(width: int, stages: tuple[int, int]) -> None:
    run(width, stages, 600)

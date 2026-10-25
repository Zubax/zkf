from typing import Any

import numpy as np
import pytest
from amaranth.lib import data

from _bench import Case, run_pipeline
from _operands import SEED
from zkf._shift import Normshift, normshift_reference


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


# stage_input, stage_split, stage_wide
STAGES = [(i, s, w) for i in (0, 1) for s in (0, 1) for w in (0, 1)]


def run(width: int, stages: tuple[int, int, int], count: int = 0, most: int | None = None, wsticky: int = 0) -> None:
    side = data.StructLayout({"tag": 7})
    knobs: dict[str, Any] = {"stage_input": stages[0], "stage_split": stages[1], "stage_wide": stages[2]}
    knobs |= {"sideband": side, "most": most}
    dut = Normshift(width, wsticky=wsticky, **knobs)
    cases = []
    xs = inputs(width, count)
    for index, x in enumerate(x for x in xs if most is None or not x or width - x.bit_length() <= most):
        _, leading, y = normshift_reference(width, x)
        tag = side.const({"tag": index & 127})
        # The count and the shifted value of a zero mean nothing.
        want = {"zero": 0, "count": leading, "y": y} if x else {"zero": 1}
        want |= {"sticky": int(y & ((1 << wsticky) - 1) != 0)}
        cases.append(Case({"x": x, "sb_in": tag}, want | {"sb_out": tag}))
    run_pipeline(dut, sum(stages), cases)


# Every width from the narrowest through the first with three levels, both parities of levels, a tree of the width and
# one wider, odd counts, whose widest digit is one bit; no sticky, one bit of it, about half and all.
@pytest.mark.parametrize(
    "width, stages, wsticky",
    [
        (width, stages, wsticky)
        for width in (2, 3, 4, 5, 7, 8, 9, 13, 15, 16, 17)
        for stages in STAGES
        for wsticky in sorted({0, 1, (width + 1) // 2, width})
    ],
    ids=str,
)
def test_every_input(width: int, stages: tuple[int, int, int], wsticky: int) -> None:
    run(width, stages, wsticky=wsticky)


@pytest.mark.parametrize(
    "width, wsticky", [(21, 3), (27, 0), (32, 0), (39, 0), (57, 37), (64, 30), (65, 41), (102, 64), (108, 0), (130, 70)]
)
@pytest.mark.parametrize("stages", STAGES, ids=str)
def test_wide(width: int, wsticky: int, stages: tuple[int, int, int]) -> None:
    run(width, stages, 600, wsticky=wsticky)


# A bound on the leading zeros that leaves out the widest level: at the reach of the rest, and below it, as `cordic`'s
# packers have them; one of none, which keeps a level.
@pytest.mark.parametrize(
    "width, most, wsticky, stages",
    [
        (w, m, ws, s)
        for w, m, ws in (
            (52, 28, 32),
            (57, 34, 37),
            (96, 54, 58),
            (102, 61, 64),
            (61, 1, 23),
            (105, 63, 50),
            (65, 39, 9),
            (32, 0, 5),
        )
        for s in STAGES
    ],
    ids=str,
)
def test_bounded(width: int, most: int, wsticky: int, stages: tuple[int, int, int]) -> None:
    run(width, stages, 600, most, wsticky)


def test_bound_builds_fewer_levels() -> None:
    side = data.StructLayout({"tag": 7})
    levels = [(len(Normshift(105, sideband=side, most=most).count) + 1) // 2 for most in (None, 64, 63, 15, 0)]
    assert levels == [4, 4, 3, 2, 1]

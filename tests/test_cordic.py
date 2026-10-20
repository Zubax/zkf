import re
from functools import reduce
from operator import or_
from typing import Any

import mpmath
import numpy as np
import pytest

from _catalog import CATALOG
from _cordic import build, cut, run, simulate, synthesized
from _operands import SEED
from zkf import Timing, UnsupportedFormat, ZkfFormat
from zkf._cordic import _LACKED, _ROTATION, _VECTORING, Cordic, _constants

Core = Cordic.Core
SHARED = Core.NORMALIZE_INPUT | Core.NORMALIZE | Core.PACK

# wexp, wman and a count. A rotation is walked whole at 2/16 in the deep tier and in a thirteenth here, which takes
# as long as the other formats together.
ROTATIONS = [(3, 16, 512), (5, 16, 512), (8, 24, 1024), (8, 32, 768), (11, 53, 384)]
# Every format has the directed pairs: two operands are too many to walk at any. 5 is the narrowest exponent, where
# the difference of the exponents is short.
VECTORINGS = [(6, 18, 1536), (8, 24, 1024), (8, 36, 768), (11, 53, 384), (5, 16, 512), (5, 53, 384)]
# The knobs at 5/16 unless a row has a format, where they change no result; of both modes unless it has one; the
# product's count the fewest unless a row has one. Half a rate with the recurrence of `z` ahead, several iterations a
# cycle with a last group that is not whole, the product from none of its registers at a width where it is one
# multiplier to the pairs of rows of a grid on narrower multipliers, and the registers of the rest one by one and all.
KNOBS: list[dict[str, Any]] = [
    {"un": 50},
    {"un": 200},
    {"un": 400, "mode": 1},
    {"un": 200, "wexp": 6, "wman": 18},
    {"un": 300, "wexp": 6, "wman": 18, "mode": 2},
    {"si": 1},
    {"so": 1},
    {"si": 1, "so": 1},
    {"si": 2},
    {"so": 3},
    {"sm": 0, "wm": 32},
    {"sm": 1, "wm": 32},
    {"sm": 0, "un": 50, "wm": 32, "mode": 0},
    {"sm": 3},
    {"sm": 4, "mode": 2},
    {"sm": 4, "wm": 10},
    {"sm": 5, "wm": 10, "mode": 2},
    {"sm": 3, "wm": 16, "mode": 1},
    {"sm": 4, "wm": 16, "mode": 2},
    {"un": 50, "sm": 3, "mode": 0},
    {"un": 50, "sm": 4, "mode": 0},
    {"un": 50, "sm": 4, "wm": 16, "mode": 2},
    {"un": 50, "sm": 3, "wexp": 6, "wman": 18, "mode": 1},
    *({"sc": member, "mode": mode} for mode in (0, 1, 2) for member in ~_LACKED[mode]),
    *({"sc": ~_LACKED[mode], "sm": 0, "wm": 32, "mode": mode} for mode in (0, 1, 2)),
    {"sc": Core.REDUCED | Core.NORMALIZE_INPUT | Core.NORMALIZE, "sm": 4, "wm": 10, "mode": 0},
    {"sc": ~_LACKED[1], "un": 50, "mode": 1},
    {"si": 1, "sm": 3, "sc": SHARED | Core.TAKEN | Core.ORDERED, "so": 1, "mode": 1},
    {
        "si": 1,
        "sm": 3,
        "sc": SHARED | Core.REDUCED | Core.FOLDED | Core.TAKEN | Core.ORDERED | Core.ANGLE,
        "so": 1,
        "mode": 2,
    },
    {"si": 1, "un": 50, "sm": 4, "sc": ~Core(0), "so": 2, "mode": 2},
]
# In a rotation at 5/18 the sine reaches the packer's underflow, the bias being below the fraction; at 8/48 the
# width of what the modes share crosses a power of two; 4 is an exponent that a rotation alone takes.
BOTH = [(5, 16, {}), (5, 18, {}), (8, 48, {"un": 50}), (11, 53, {}), (4, 16, {"mode": 0})]


# The deep tier: the formats of the default one and others at more operands, under the knobs one by one.
DEEP_KNOBS: list[dict[str, Any]] = [{}, {"un": 50}, {"un": 200}, {"un": 400}, {"sm": 3}, {"sm": 4}, {"sm": 4, "wm": 10}]
DEEP_KNOBS += [{"si": 1, "so": 1, "sc": Core.REDUCED | Core.NORMALIZE_INPUT | Core.NORMALIZE | Core.PACK}]
DEEP_KNOBS += [{"un": 50, "sm": 4}, {"sc": ~_LACKED[0]}]
DEEP: list[tuple[int, int, int | str, dict[str, Any]]] = [
    *((3, 16, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((6, 16, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((8, 27, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((8, 32, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((14, 16, 128, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    (3, 16, 4000, {"mode": 0}),
    (5, 16, 4000, {"mode": 0}),
    (5, 16, 4000, {"mode": 0, "un": 200}),
    (5, 16, 4000, {"mode": 0, "sc": Core.REDUCED | Core.NORMALIZE}),
    (5, 16, 4000, {"mode": 0, "un": 50}),
    (5, 16, 2000, {"mode": 0, "sm": 0, "wm": 32}),
    (5, 16, 512, {"mode": 0, "sc": ~_LACKED[0]}),
    (5, 16, 1000, {"sm": 1, "wm": 32}),
    (8, 24, 768, {"mode": 0}),
    *((wexp, wman, count, {"mode": 2}) for wexp, wman, count in VECTORINGS[:4]),
    *((5, 16, 512, {"mode": 1, "un": u}) for u in (50, 200, 400)),
    (5, 16, 512, {"mode": 1, "si": 1, "so": 1, "sc": Core.ORDERED | Core.NORMALIZE_INPUT | Core.NORMALIZE}),
    (5, 16, 512, {"mode": 1, "sc": ~_LACKED[1]}),
    (5, 16, 4000, {"mode": 1}),
    (5, 16, "directed", {"mode": 1}),
    (5, 16, 2000, {"mode": 1, "un": 200}),
    (6, 18, 2000, {"mode": 1, "un": 300}),
    (5, 16, 2000, {"mode": 1, "sc": Core.ORDERED | Core.NORMALIZE}),
    (5, 16, 2000, {"mode": 1, "sm": 0, "wm": 32}),
    (5, 16, 512, {"un": 50}),
    (8, 36, 2048, {"mode": 1} | synthesized(36, 1)),
]


def timings(model: Cordic) -> list[tuple[int, int]]:
    timing = model.timing
    return [(t.latency, t.initiation_interval) for t in ([timing] if isinstance(timing, Timing) else timing.values())]


@pytest.mark.parametrize("row", ROTATIONS, ids=str)
def test_rotation(row: tuple[int, int, int]) -> None:
    run(*row, mode=0)


@pytest.mark.parametrize("row", VECTORINGS, ids=str)
def test_vectoring(row: tuple[int, int, int]) -> None:
    run(*row, mode=1)


def test_rotation_sampled() -> None:
    run(2, 16, "every", mode=0, shard=5, shards=13)


@pytest.mark.deep
@pytest.mark.parametrize("shard", range(16))
@pytest.mark.parametrize("stages", [{"sc": Core.REDUCED | Core.NORMALIZE}, *DEEP_KNOBS], ids=str)
def test_rotation_whole(stages: dict[str, Any], shard: int) -> None:
    run(2, 16, "every", mode=0, shard=shard, shards=16, **stages)


@pytest.mark.deep
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: tuple[int, int, int | str, dict[str, Any]]) -> None:
    run(row[0], row[1], row[2], **row[3])


@pytest.mark.parametrize("row", KNOBS, ids=str)
def test_knobs(row: dict[str, Any]) -> None:
    knobs = dict(row)
    fmt = knobs.pop("wexp", 5), knobs.pop("wman", 16)
    for mode in (knobs.pop("mode"),) if "mode" in knobs else (0, 1, 2):
        run(*fmt, 256, mode=mode, **knobs)


@pytest.mark.parametrize("row", BOTH, ids=str)
def test_either_mode(row: tuple[int, int, dict[str, int]]) -> None:
    run(row[0], row[1], 256, **row[2])


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Cordic)])
def test_catalog(model: Cordic) -> None:
    simulate(model, 1024)
    cut(model)


def subsets(seed: int = SEED) -> list[dict[str, Any]]:
    """Random sets of the registers that a mode builds, the most of both modes, each at a rate of its own."""
    rng = np.random.default_rng(seed)
    out = []
    for mode, count in ((0, 6), (1, 6), (2, 20)):
        members = list(~_LACKED[mode])
        for _ in range(count):
            taken = [member for member, coin in zip(members, rng.integers(0, 2, len(members))) if coin]
            out.append({"sc": reduce(or_, taken, Core(0)), "un": int(rng.choice([50, 100, 200])), "mode": mode})
    return out


# A few of each mode per change, the rest in the deep tier.
@pytest.mark.parametrize(
    "row", [pytest.param(row, marks=() if k % 8 == 0 else pytest.mark.deep) for k, row in enumerate(subsets())], ids=str
)
def test_subsets(row: dict[str, Any]) -> None:
    """The modes in turn: what one mode leaves in what both share shows in the other."""
    model = build(ZkfFormat(5, 16), **row)
    simulate(model, 192, alternate=True)
    cut(model)


# A front ahead of the engine in one mode and none in the other: a transaction of the other starts the engine in the
# cycle it is taken in, unless the input is registered. The cut without the input stage is `test_cut_each_register`'s.
@pytest.mark.parametrize("si", [0, 1])
@pytest.mark.parametrize("front", [Core.SHIFT, Core.FOLDED, Core.TAKEN, Core.SHIFTED], ids=str)
def test_one_front(front: Cordic.Core, si: int) -> None:
    model = build(ZkfFormat(5, 16), sc=front, si=si)
    simulate(model, 256, alternate=True)
    if si:
        cut(model)


# A width changes the parity of the divider, the count of iterations and the widths of what the modes share, which
# a mode alone has narrower.
@pytest.mark.parametrize("wman", range(16, 54))
def test_every_width(wman: int) -> None:
    run(8, wman, "directed", **synthesized(wman))


@pytest.mark.deep
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("wman", range(16, 54))
def test_every_width_of_a_mode(wman: int, mode: int) -> None:
    run(8, wman, "directed", mode=mode, **synthesized(wman, mode))


@pytest.mark.parametrize("stages", [{}, {"un": 50, "sm": 3, "sc": ~Core(0), "so": 2}, {"un": 200, "si": 1}], ids=str)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cut_by_reset(mode: int, stages: dict[str, Any]) -> None:
    knobs = stages | {"sc": stages.get("sc", Core(0)) & ~_LACKED[mode]}
    cut(build(ZkfFormat(5, 16), mode, **knobs))
    if mode == 0:
        cut(build(ZkfFormat(3, 16), mode, **knobs))  # an exponent that a vectoring does not take


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cut_each_register(mode: int) -> None:
    for member in ~_LACKED[mode]:
        cut(build(ZkfFormat(5, 16), mode, sc=member))


def test_timing() -> None:
    """A register is a cycle of the modes on whose path it is."""
    fmt = ZkfFormat(6, 18)
    assert timings(Cordic(fmt, stage_core_mul=3, wmultiplier=18)) == [(15, 15), (26, 26)]
    assert timings(Cordic(fmt, unroll100=50, stage_core_mul=3, wmultiplier=18)) == [(23, 23), (33, 33)]
    steps = {member: (1, 0) for member in _ROTATION} | {member: (0, 1) for member in _VECTORING}
    for member in Core:
        rot, vec = steps.get(member, (1, 1))
        found = timings(Cordic(fmt, stage_core_mul=3, wmultiplier=18, stage_core=member))
        assert found == [(15 + rot, 15 + rot), (26 + vec, 26 + vec)], member


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    assert [len(~_LACKED[mode]) for mode in (0, 1, 2)] == [8, 8, 13]
    # Both modes' product is a grid of 3x2 multipliers, a rotation's of 2x2, and one multiplier at a width of 36.
    refused: list[dict[str, Any]] = [
        {"unroll100": 0},
        {"unroll100": 150},
        {"stage_core_mul": 2},
        {"stage_core_mul": 6},
        {"stage_core_mul": 1, "mode": 0},
        {"stage_core_mul": 5, "mode": 0},
        {"stage_core_mul": 4, "wmultiplier": 36},
        *({"stage_core": member, "mode": mode} for mode in (0, 1) for member in _LACKED[mode]),
        {"mode": 3},
    ]
    for knobs in refused:
        with pytest.raises(ValueError):
            Cordic(fmt, **{"wmultiplier": 18, "stage_core_mul": 3} | knobs)
    with pytest.raises(ValueError, match=r"^stage_core_mul=0: set 3\.\.4 "):
        Cordic(fmt, wmultiplier=18)
    refusal = "stage_core=Core.SHIFT|TAKEN|MEET: leave out TAKEN|MEET, which mode=0 lacks"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        Cordic(fmt, wmultiplier=18, stage_core_mul=3, stage_core=Core.SHIFT | Core.TAKEN | Core.MEET, mode=0)
    refusal = "stage_core=Core.FOLDED|ANGLE: leave out FOLDED, which mode=1 lacks"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        Cordic(fmt, wmultiplier=18, stage_core_mul=3, stage_core=Core.FOLDED | Core.ANGLE, mode=1)
    Cordic(fmt, wmultiplier=36)
    for wexp, wman, mode in ((4, 16, 1), (4, 16, 2), (8, 15, 0), (8, 54, 2)):
        with pytest.raises(UnsupportedFormat):
            Cordic(ZkfFormat(wexp, wman), wmultiplier=18, mode=mode)


def test_numbers_do_not_follow_the_precision_of_the_caller() -> None:
    before = mpmath.mp.prec
    try:
        found = []
        for precision in (24, 53, 80, 2000):
            mpmath.mp.prec = precision
            found.append(_constants.__wrapped__(53))
            assert mpmath.mp.prec == precision
    finally:
        mpmath.mp.prec = before
    assert all(numbers == found[0] for numbers in found)

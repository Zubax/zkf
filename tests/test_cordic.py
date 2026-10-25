import re
from dataclasses import replace
from functools import reduce
from operator import or_
from typing import Any

import mpmath
import numpy as np
import pytest

from _catalog import CATALOG
from _cordic import build, cut, run, simulate, synthesized
from _operands import SEED
from amaranth.sim import Simulator
from zkf import Timing, UnsupportedFormat, ZkfFormat

from zkf._cordic import _GUARD_ZF, _LACKED, _ROTATION, _SEED, _VECTORING, Cordic, _ahead, _constants, _Engine, _Seed
from zkf._value import bits_to_signed

CoreStages = Cordic.CoreStages
SHARED = CoreStages.NORMALIZE_INPUT | CoreStages.NORMALIZE | CoreStages.PACK

# wexp, wman and a count. A rotation is walked whole at 2/16 in the deep tier and in a thirteenth here, which takes
# as long as the other formats together.
ROTATIONS = [(3, 16, 512), (5, 16, 512), (8, 24, 1024), (8, 32, 768), (11, 53, 384)]
# Every format has the directed pairs: two operands are too many to walk at any. 5 is the narrowest exponent, where
# the difference of the exponents is short.
VECTORINGS = [(6, 18, 1536), (8, 24, 1024), (8, 36, 768), (11, 53, 384), (5, 16, 512), (5, 53, 384)]
# The knobs at 5/16 unless a row has a format, where they change no result; of both modes unless it has one; the
# product's count the fewest unless a row has one. The product from none of its registers at a width where it is one
# multiplier to the pairs of rows of a grid on narrower multipliers, the registers of the rest one by one and all, the
# second packer, the seeds off and in part, and a product as long as the iterations that the seed leaves.
KNOBS: list[dict[str, Any]] = [
    {"wexp": 6, "wman": 18},
    {"rs": 0},
    {"rs": 1},
    {"sm": 6, "wm": 9, "mode": 2},
    {"si": 1},
    {"so": 1},
    {"si": 1, "so": 1},
    {"si": 2},
    {"so": 3},
    {"sm": 0, "wm": 32},
    {"sm": 1, "wm": 32},
    {"sm": 3},
    {"sm": 4, "mode": 2},
    {"sm": 4, "wm": 10},
    {"sm": 5, "wm": 10, "mode": 2},
    {"sm": 3, "wm": 16, "mode": 1},
    {"sm": 4, "wm": 16, "mode": 2},
    {"sm": 4, "mode": 0},
    {"sm": 5, "wm": 9, "mode": 0},
    {"sm": 3, "wexp": 6, "wman": 18, "mode": 1},
    *({"sc": member, "mode": mode} for mode in (0, 1, 2) for member in ~_LACKED[mode]),
    *({"sc": ~_LACKED[mode], "sm": 0, "wm": 32, "mode": mode} for mode in (0, 1, 2)),
    {"sc": CoreStages.REDUCED | CoreStages.NORMALIZE_INPUT | CoreStages.NORMALIZE, "sm": 4, "wm": 10, "mode": 0},
    {"si": 1, "sm": 3, "sc": SHARED | CoreStages.TAKEN | CoreStages.ORDERED, "so": 1, "mode": 1},
    {
        "si": 1,
        "sm": 3,
        "sc": SHARED
        | CoreStages.REDUCED
        | CoreStages.FOLDED
        | CoreStages.TAKEN
        | CoreStages.ORDERED
        | CoreStages.ANGLE,
        "so": 1,
        "mode": 2,
    },
    {"si": 1, "sm": 4, "sc": ~CoreStages(0), "so": 2, "mode": 2},
    {"fa": True, "mode": 0},
    {"fa": True, "sm": 4, "mode": 2},
    {"fa": True, "si": 1, "sm": 0, "wm": 32, "sc": ~_LACKED[0], "so": 2, "mode": 0},
    {"fa": True, "sc": ~_LACKED[2], "mode": 2},
    {"fa": True, "sc": ~_LACKED[2] & ~CoreStages.CORRECTIONS, "mode": 2},
]
# In a rotation at 5/18 the sine reaches the packer's underflow, the bias being below the fraction; at 8/48 the
# width of what the modes share crosses a power of two; 4 is an exponent that a rotation alone takes; and the second
# packer at a catalog width.
BOTH = [(5, 16, {}), (5, 18, {}), (8, 48, {}), (11, 53, {}), (4, 16, {"mode": 0}), (8, 36, {"fa": True})]


# The deep tier: the formats of the default one and others at more operands, under the knobs one by one.
DEEP_KNOBS: list[dict[str, Any]] = [{}, {"rs": 1}, {"sm": 3}, {"sm": 4}, {"sm": 4, "wm": 10}]
DEEP_KNOBS += [
    {"si": 1, "so": 1, "sc": CoreStages.REDUCED | CoreStages.NORMALIZE_INPUT | CoreStages.NORMALIZE | CoreStages.PACK}
]
DEEP_KNOBS += [{"sc": ~_LACKED[0]}]
DEEP: list[tuple[int, int, int | str, dict[str, Any]]] = [
    *((3, 16, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((6, 16, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((8, 27, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((8, 32, 512, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    *((14, 16, 128, {"mode": 0} | knobs) for knobs in DEEP_KNOBS),
    (3, 16, 4000, {"mode": 0}),
    (5, 16, 4000, {"mode": 0}),
    (5, 16, 4000, {"mode": 0, "sc": CoreStages.REDUCED | CoreStages.NORMALIZE}),
    (5, 16, 2000, {"mode": 0, "sm": 0, "wm": 32}),
    (5, 16, 512, {"mode": 0, "sc": ~_LACKED[0]}),
    (5, 16, 1000, {"sm": 1, "wm": 32}),
    (8, 24, 768, {"mode": 0}),
    *((wexp, wman, count, {"mode": 2}) for wexp, wman, count in VECTORINGS[:4]),
    (
        5,
        16,
        512,
        {"mode": 1, "si": 1, "so": 1, "sc": CoreStages.ORDERED | CoreStages.NORMALIZE_INPUT | CoreStages.NORMALIZE},
    ),
    (5, 16, 512, {"mode": 1, "sc": ~_LACKED[1]}),
    (5, 16, 4000, {"mode": 1}),
    (5, 16, "directed", {"mode": 1}),
    (6, 18, 2000, {"mode": 1}),
    (5, 16, 2000, {"mode": 1, "sc": CoreStages.ORDERED | CoreStages.NORMALIZE}),
    (5, 16, 2000, {"mode": 1, "sm": 0, "wm": 32}),
    (5, 16, 512, {}),
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
@pytest.mark.parametrize(
    "stages", [{"sc": CoreStages.REDUCED | CoreStages.NORMALIZE}, *DEEP_KNOBS, {"fa": True}], ids=str
)
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
    """
    Random sets of the registers that a mode builds, the most of both modes; and every third with a rotation again with
    two packers.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for mode, count in ((0, 6), (1, 6), (2, 20)):
        members = list(~_LACKED[mode])
        for _ in range(count):
            taken = [member for member, coin in zip(members, rng.integers(0, 2, len(members))) if coin]
            out.append({"sc": reduce(or_, taken, CoreStages(0)), "mode": mode})
    return out + [row | {"fa": True} for row in out[::3] if row["mode"] != 1]


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
@pytest.mark.parametrize("front", [CoreStages.SHIFT, CoreStages.FOLDED, CoreStages.TAKEN, CoreStages.SHIFTED], ids=str)
def test_one_front(front: Cordic.CoreStages, si: int) -> None:
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


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_the_attributes_reach_the_tables(mode: int) -> None:
    model = Cordic(ZkfFormat(5, 16), wmultiplier=18, mode=mode, stage_core_mul=3)
    attribute = '(* syn_romstyle = "logic" *)'
    lines = replace(model, rom_attrs={"syn_romstyle": "logic"}).compile().splitlines()
    marked = {lines[at + 1].split()[-1].rstrip(";") for at, line in enumerate(lines) if line.strip() == attribute}
    assert marked == {"angle"} | ({"taken"} if _ahead(model) else set())
    assert [line for line in lines if line.strip() != attribute] == model.compile().splitlines()


@pytest.mark.parametrize("stages", [{}, {"sm": 3, "sc": ~CoreStages(0), "so": 2}, {"si": 1}], ids=str)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cut_by_reset(mode: int, stages: dict[str, Any]) -> None:
    knobs = stages | {"sc": stages.get("sc", CoreStages(0)) & ~_LACKED[mode]}
    cut(build(ZkfFormat(5, 16), mode, **knobs))
    if mode == 0:
        cut(build(ZkfFormat(3, 16), mode, **knobs))  # an exponent that a vectoring does not take


# Without CORRECTIONS a packer out of step with the other takes a value that has moved on.
@pytest.mark.parametrize("held", [True, False])
@pytest.mark.parametrize("mode", [0, 2])
def test_cut_fast(mode: int, held: bool) -> None:
    cut(
        build(ZkfFormat(5, 16), mode, fa=True, sc=~_LACKED[mode] & ~(CoreStages(0) if held else CoreStages.CORRECTIONS))
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cut_each_register(mode: int) -> None:
    for member in ~_LACKED[mode]:
        cut(build(ZkfFormat(5, 16), mode, sc=member))


def test_timing() -> None:
    """A register is a cycle of the modes on whose path it is; a seed takes its iterations off the engine."""
    fmt = ZkfFormat(6, 18)
    plain: dict[str, Any] = {"rotation_seed": 0, "stage_core_mul": 3, "wmultiplier": 18}
    assert timings(Cordic(fmt, **plain)) == [(15, 15), (25, 25)]
    steps = {member: (1, 0) for member in _ROTATION} | {member: (0, 1) for member in _VECTORING}
    for member in CoreStages:
        rot, vec = steps.get(member, (1, 1))
        found = timings(Cordic(fmt, **plain, stage_core=member))
        assert found == [(15 + rot, 15 + rot), (25 + vec, 25 + vec)], member
    assert timings(Cordic(fmt, **plain, fast=True)) == [(14, 14), (25, 25)]
    for seed in (1, 2):
        assert timings(Cordic(fmt, **plain | {"rotation_seed": seed})) == [(15 - seed,) * 2, (25, 25)]
    # No effect in the mode that does not take it.
    one: dict[str, Any] = {"stage_core_mul": 2, "wmultiplier": 18}
    assert Cordic(fmt, **one, mode=1, rotation_seed=2).timing == Cordic(fmt, **one, mode=1, rotation_seed=0).timing


@pytest.mark.parametrize("wman", range(16, 54))
def test_seed_tables(wman: int) -> None:
    """Each interval of the seed, and `halfway`, give the state of the reference's iterations at the engine's start."""
    c = _constants(wman)

    def iterated(tpw: int, count: int) -> tuple[int, int, int]:
        x, y, z = c.kinv, 0, tpw << _GUARD_ZF
        for i in range(count):
            x, y, z = (
                (x + (y >> i), y - (x >> i), z + c.lut[i]) if z < 0 else (x - (y >> i), y + (x >> i), z - c.lut[i])
            )
        return x, y, z

    half = 1 << (c.wt - 1)
    for signs in range(3):
        seed = _Seed(c, signs, merged=False)
        ends = [entry[0] for entry in seed.entries[1:]] + [half]
        assert ends == sorted(ends) and seed.entries[0][0] == 0
        for (low, x, y, take), high in zip(seed.entries, ends):
            for tpw in {low, (low + high) // 2, high - 1}:
                assert iterated(tpw, _SEED + signs) == (x, y, (tpw << _GUARD_ZF) - c.lut[0] + take), (signs, tpw)
        assert iterated(half, _SEED + signs) == seed.halfway_state, signs


@pytest.mark.parametrize("wman", [16, 36])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_engine_from_its_seeds(wman: int, mode: int) -> None:
    """
    The engine started at the state of each mode's seed ends where the reference's own iterations from that state do:
    what the outputs cannot show.
    """
    c = _constants(wman)
    rng = np.random.default_rng(wman * 100 + mode)
    seed = _SEED + 2

    def iterate(x: int, y: int, z: int, start: int, vectoring: bool) -> tuple[int, int, int]:
        for i in range(start, c.n):
            up = y >= 0 if vectoring else z >= 0
            x, y = (x + (y >> i), y - (x >> i)) if up == vectoring else (x - (y >> i), y + (x >> i))
            z = z + c.lut[i] if up == vectoring else z - c.lut[i]
            x, y, z = (bits_to_signed(v & ((1 << w) - 1), w) for v, w in ((x, c.wx), (y, c.wx), (z, c.wz)))
        return x, y, z

    cases = []
    for _ in range(12):
        for vectoring in [(0,), (1,), (0, 1)][mode]:
            if vectoring:
                x = int(rng.integers(1 << (c.xf - 2), 1 << (c.xf - 1)))
                x0, y0, z0 = x, int(rng.integers(-x, x + 1)), 0
                for i in range(_SEED):
                    up = y0 >= 0
                    x0, y0 = (x0 + (y0 >> i), y0 - (x0 >> i)) if up else (x0 - (y0 >> i), y0 + (x0 >> i))
                    z0 = z0 + c.lut[i] if up else z0 - c.lut[i]
                start = _SEED
                expected = iterate(x0, y0, z0, start, True)
            else:
                tpw = int(rng.integers(1, 1 << (c.wt - 1)))
                x0, y0, z0 = c.kinv, 0, tpw << _GUARD_ZF
                for i in range(seed):
                    up = z0 >= 0
                    x0, y0 = (x0 - (y0 >> i), y0 + (x0 >> i)) if up else (x0 + (y0 >> i), y0 - (x0 >> i))
                    z0 = z0 - c.lut[i] if up else z0 + c.lut[i]
                start = seed
                expected = iterate(x0, y0, z0, start, False)
            cases.append((vectoring, x0, y0, z0, expected))
    dut = _Engine(c.lut, wx=c.wx, wz=c.wz, mode=mode, seed=seed)

    async def bench(ctx: Any) -> None:
        for vectoring, x0, y0, z0, expected in cases:
            for port, value in ((dut.vectoring, vectoring), (dut.x0, x0), (dut.y0, y0), (dut.z0, z0)):
                ctx.set(port, value)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            for _ in range(4 * c.n):
                if ctx.get(dut.done):
                    break
                await ctx.tick()
            assert (ctx.get(dut.xn), ctx.get(dut.yn), ctx.get(dut.zn)) == expected, (vectoring, x0, y0, z0)
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(bench)
    sim.run()


@pytest.mark.parametrize("wman", [16, 18, 36, 53])
def test_seed_decoder(wman: int) -> None:
    """At the coordinates next to each threshold, on both sides of the fold, and at the ends and half a quadrant."""
    c = _constants(wman)
    wt, half = c.wt, 1 << (c.wt - 1)
    for signs in (1, 2):
        for merged in (False, True):
            dut = _Seed(c, signs, merged)
            ends = [entry[0] for entry in dut.entries[1:]]
            folded = {0, 1, half - 1, half}
            for end in ends:
                folded |= {end - 1, end, end + 1}
            coordinates = sorted(
                {tpw for tpw in folded if 0 <= tpw <= half} | {(1 << wt) - tpw for tpw in folded if 0 < tpw < half}
            )

            def expected(t: int) -> tuple[int, int | None, int]:
                tpw = (1 << wt) - t if t > half else t
                if tpw == half:
                    return (
                        (1 << len(dut.entries)) if merged else 1 << len(ends),
                        dut.halfway_state[2] if merged else None,
                        1,
                    )
                k = sum(tpw >= end for end in ends)
                return 1 << k, dut.entries[k][3], 0

            async def bench(ctx: Any) -> None:
                for t in coordinates:
                    ctx.set(dut.t, t)
                    picks, take, halfway = expected(t)
                    assert ctx.get(dut.halfway) == halfway, (signs, merged, t)
                    if picks is not None and (merged or not halfway):
                        assert ctx.get(dut.picks) == picks, (signs, merged, t)
                    if take is not None:
                        assert ctx.get(dut.z_take) == take, (signs, merged, t)

            sim = Simulator(dut)
            sim.add_testbench(bench)
            sim.run()


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    assert [len(~_LACKED[mode]) for mode in (0, 1, 2)] == [8, 8, 13]
    # Both modes' product is a grid of 3x2 multipliers, a rotation's of 2x2, and one multiplier at a width of 36.
    refused: list[dict[str, Any]] = [
        {"rotation_seed": -1},
        {"rotation_seed": 3},
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
    refusal = "stage_core=CoreStages.SHIFT|TAKEN|MEET: leave out TAKEN|MEET, which mode=0 lacks"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        Cordic(
            fmt,
            wmultiplier=18,
            stage_core_mul=3,
            stage_core=CoreStages.SHIFT | CoreStages.TAKEN | CoreStages.MEET,
            mode=0,
        )
    refusal = "stage_core=CoreStages.FOLDED|ANGLE: leave out FOLDED, which mode=1 lacks"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        Cordic(fmt, wmultiplier=18, stage_core_mul=3, stage_core=CoreStages.FOLDED | CoreStages.ANGLE, mode=1)
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

from typing import Any

import pytest

from _catalog import CATALOG
from _trans import build, others, run, simulate, staging
from zkf import ZkfFormat
from zkf._exp2 import Exp2

Core = Exp2.Core

# wexp, wman, kind: every table of a degree.
FORMATS: list[tuple[int, int, str | int]] = [
    (2, 16, "every"),
    (6, 16, 512),
    (8, 24, 1024),
    (8, 32, 768),
    (11, 53, 384),
]


@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, str | int]) -> None:
    run("exp2", *row)


@pytest.mark.parametrize("stages", staging(Core), ids=str)
def test_stages(stages: dict[str, Any]) -> None:
    """Stages change no result: the format that `test_formats` walks whole is sampled here."""
    run("exp2", 2, 16, 4096, **stages)


# Every count of the kernel at degree 3, whose square takes two multipliers, and the core beside it; the least is the
# catalog's.
@pytest.mark.parametrize("stages", [{"sk": 4, "sc": Core.SUM}, {"sk": 4, "sc": ~Core(0)}], ids=str)
def test_degree_3(stages: dict[str, Any]) -> None:
    run("exp2", 8, 36, 256, **stages)


@pytest.mark.parametrize("row", others(Core), ids=str)
def test_generated_for_the_format(row: tuple[int, int, dict[str, Any]]) -> None:
    run("exp2", row[0], row[1], 256, **row[2])


def test_other_degrees() -> None:
    run("exp2", 8, 27, 256, sk=3)
    run("exp2", 8, 48, 256, sk=4)
    run("exp2", 11, 53, 256, sk=5)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Exp2)])
def test_catalog(model: Exp2) -> None:
    simulate(model, 512)


def test_strata() -> None:
    run("exp2", 6, 18, "strata")


def test_timing() -> None:
    """
    The kernel's count adds the registers on its longest path of products; each register of the core and of the ends
    a cycle.
    """
    fmt = ZkfFormat(6, 18)
    assert [Exp2(fmt, wmultiplier=24, stage_core_kernel=k).timing.latency for k in range(3)] == [1, 3, 5]
    assert [Exp2(fmt, wmultiplier=18, stage_core_kernel=k).timing.latency for k in (2, 3)] == [5, 5]
    assert [build("exp2", ZkfFormat(8, 36), stage_core_kernel=k).timing.latency for k in (3, 4)] == [6, 6]
    assert [build("exp2", ZkfFormat(8, 36), stage_core=member).timing.latency for member in Core] == [7, 7]
    ends: dict[str, Any] = {"stage_input": 2, "stage_core": ~Core(0), "stage_output": 3}
    assert build("exp2", ZkfFormat(8, 36), **ends).timing.latency == 13


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    with pytest.raises(ValueError, match="stage_core_kernel=0: set 2..3 for the exp2 kernel"):
        Exp2(fmt, wmultiplier=18)  # c1*w takes two multipliers
    for knobs in (
        {"stage_core_kernel": 1},
        {"stage_core_kernel": 4},
    ):
        with pytest.raises(ValueError):
            build("exp2", fmt, **knobs)
    with pytest.raises(ValueError, match="set 3..4"):
        build("exp2", ZkfFormat(8, 36), stage_core_kernel=2)  # c1*w is a grid of 3x2


# wexp, wman, kind, shard, shards, stages
type Row = tuple[int, int, str | int, int, int, dict[str, Any]]
EACH: tuple[dict[str, Any], ...] = ({}, {"si": 1}, {"sk": 3}, {"sc": ~Core(0)}, {"so": 1}, {"sk": 3, "sc": Core.SUM})
DEEP: list[Row] = []
DEEP += [(2, 16, "every", 0, 1, stages) for stages in EACH[1:]]  # bare, it is a row of `test_formats`
DEEP += [(3, 16, "every", shard, 2, stages) for shard in (0, 1) for stages in EACH]
DEEP += [(w, m, 512 if m > 16 else 128, 0, 1, stages) for w, m in ((8, 27), (8, 32), (14, 16)) for stages in EACH]
DEEP += [(8, 36, 512, 0, 1, {"sk": 4})]  # 3 is the catalog's
DEEP += [
    (w, m, "strata", 0, 1, {})
    for w, m in ((8, 16), (8, 18), (8, 24), (8, 27), (8, 32), (8, 36), (8, 48), (8, 53), (11, 53))
]


@pytest.mark.deep
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: Row) -> None:
    wexp, wman, kind, shard, shards, stages = row
    run("exp2", wexp, wman, kind, shard, shards, **stages)

from typing import Any

import pytest

from _catalog import CATALOG
from _trans import build, others, run, simulate, staging
from zkf import ZkfFormat
from zkf._log2 import Log2

CoreStages = Log2.CoreStages

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
    run("log2", *row)


@pytest.mark.parametrize("stages", [*staging(CoreStages), {"sm": 3}], ids=str)
def test_stages(stages: dict[str, Any]) -> None:
    """Stages change no result: the format that `test_formats` walks whole is sampled here."""
    run("log2", 2, 16, 4096, **stages)


# The last product's count apart from the kernel's, and the side path laid out by the kernel's stages: none, two, three.
@pytest.mark.parametrize(
    "stages",
    [{"wm": 24, "sk": 0}, {"sk": 3, "sm": 2}, {"sk": 2, "sm": 3, "sc": CoreStages.ALIGN}, {"wm": 12, "sk": 3, "sm": 4}],
    ids=str,
)
def test_last_product(stages: dict[str, Any]) -> None:
    run("log2", 6, 16, 256, **stages)


# Every count of the kernel at degree 3, whose square takes two multipliers, and the rest beside it.
@pytest.mark.parametrize(
    "stages", [{"sk": 3}, {"sk": 4, "sm": 4, "sc": CoreStages.ALIGN}, {"sk": 4, "sc": ~CoreStages(0)}], ids=str
)
def test_degree_3(stages: dict[str, Any]) -> None:
    run("log2", 8, 36, 256, **stages)


@pytest.mark.parametrize("row", others(CoreStages), ids=str)
def test_generated_for_the_format(row: tuple[int, int, dict[str, Any]]) -> None:
    run("log2", row[0], row[1], 256, **row[2])


def test_other_degrees() -> None:
    run("log2", 8, 27, 256, sk=3, sm=4)
    run("log2", 8, 48, 256, sk=4, sm=4)
    run("log2", 11, 53, 256, sk=4)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Log2)])
def test_catalog(model: Log2) -> None:
    simulate(model, 512)


def test_strata() -> None:
    run("log2", 6, 18, "strata")


def test_timing() -> None:
    fmt = ZkfFormat(6, 18)
    assert build("log2", fmt, 24, stage_core_kernel=0).timing.latency == 3  # a kernel of single multipliers
    assert build("log2", fmt, stage_core_kernel=2, stage_core_mul=2).timing.latency == 7
    assert build("log2", fmt, stage_core_kernel=3, stage_core_mul=3).timing.latency == 8
    assert [build("log2", ZkfFormat(8, 36), stage_core_kernel=k).timing.latency for k in (3, 4)] == [9, 9]
    assert [build("log2", ZkfFormat(8, 36), stage_core=member).timing.latency for member in CoreStages] == [10, 10]
    ends: dict[str, Any] = {"stage_input": 2, "stage_core": ~CoreStages(0), "stage_output": 3}
    assert build("log2", ZkfFormat(8, 36), **ends).timing.latency == 16


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    with pytest.raises(ValueError, match="stage_core_kernel=0: set 2..3 for the log2 kernel"):
        Log2(fmt, wmultiplier=18, stage_core_mul=2)
    with pytest.raises(ValueError, match="stage_core_mul=0: set 2..3 on 18-bit multipliers"):
        Log2(fmt, wmultiplier=18, stage_core_kernel=2)
    for knobs in (
        {"stage_core_kernel": 4},
        {"stage_core_mul": 1},
        {"stage_core_mul": 4},
    ):
        with pytest.raises(ValueError):
            build("log2", fmt, **knobs)
    with pytest.raises(ValueError, match="stage_core_mul=2: set 3..4 on 18-bit multipliers"):
        build("log2", ZkfFormat(8, 24), stage_core_mul=2)  # its grid is 2x3


# wexp, wman, kind, shard, shards, stages
EACH: tuple[dict[str, Any], ...] = (
    {},
    {"si": 1},
    {"sk": 3},
    {"sm": 3},
    {"sc": ~CoreStages(0)},
    {"so": 1},
    {"sk": 3, "sc": CoreStages.ALIGN},
)
type Row = tuple[int, int, str | int, int, int, dict[str, Any]]
DEEP: list[Row] = []
DEEP += [(2, 16, "every", 0, 1, stages) for stages in EACH[1:]]  # bare, it is a row of `test_formats`
DEEP += [(3, 16, "every", shard, 2, stages) for shard in (0, 1) for stages in EACH]
DEEP += [(w, m, 512 if m > 16 else 128, 0, 1, stages) for w, m in ((8, 27), (8, 32), (14, 16)) for stages in EACH]
DEEP += [
    (w, m, "strata", 0, 1, {})
    for w, m in ((8, 16), (8, 18), (8, 24), (8, 27), (8, 32), (8, 36), (8, 48), (8, 53), (11, 53))
]


@pytest.mark.deep
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: Row) -> None:
    wexp, wman, kind, shard, shards, stages = row
    run("log2", wexp, wman, kind, shard, shards, **stages)

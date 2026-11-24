from typing import Any

import pytest

from _catalog import CATALOG
from _trans import build, others, run, simulate, staging
from zkf import Log2, ZkfFormat

CoreStages = Log2.CoreStages

# wexp, wman, kind: every table of a degree, the floor walked whole.
FORMATS: list[tuple[int, int, str | int]] = [
    (2, 10, "every"),
    (6, 16, 512),
    (8, 24, 1024),
    (8, 32, 768),
    (11, 53, 384),
]


@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, str | int]) -> None:
    run("log2", *row)


def test_every_compact() -> None:
    """The compact table of degree 2, which `fast` passes over at WMAN 16."""
    run("log2", 2, 16, "every", fa=False)


@pytest.mark.parametrize("stages", [*staging(CoreStages), {"sm": 10}], ids=str)
def test_stages(stages: dict[str, Any]) -> None:
    """Stages change no result: the format that `test_every_compact` walks whole is sampled here."""
    run("log2", 2, 16, 4096, fa=False, **stages)


# The last product's level apart from the kernel's, and the side path laid out by the kernel's stages: none, two, three.
@pytest.mark.parametrize(
    "stages",
    [
        {"wm": 24, "kl": 0},
        {"kl": 5, "sm": 6},
        {"kl": 2, "sm": 10, "sc": CoreStages.ALIGN},
        {"wm": 16, "kl": 5, "sm": 10},
    ],
    ids=str,
)
def test_last_product(stages: dict[str, Any]) -> None:
    run("log2", 6, 16, 256, fa=False, **stages)


# The kernel at degree 3, whose square takes two multipliers, and the rest beside it.
@pytest.mark.parametrize(
    "stages", [{"kl": 8}, {"kl": 9, "sm": 12, "sc": CoreStages.ALIGN}, {"kl": 9, "sc": ~CoreStages.NONE}], ids=str
)
def test_degree_3(stages: dict[str, Any]) -> None:
    run("log2", 8, 36, 256, **stages)


@pytest.mark.parametrize("row", others(CoreStages), ids=str)
def test_generated_for_the_format(row: tuple[int, int, dict[str, Any]]) -> None:
    run("log2", row[0], row[1], 256, **row[2])


def test_other_degrees() -> None:
    run("log2", 8, 27, 256, kl=6, sm=4, fa=False)
    run("log2", 8, 48, 256, kl=7, sm=4, fa=False)
    run("log2", 11, 53, 256, kl=7, fa=False)


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Log2)])
def test_catalog(model: Log2) -> None:
    simulate(model, 512)


def test_strata() -> None:
    run("log2", 6, 18, "strata")


def test_timing() -> None:
    fmt = ZkfFormat(6, 18)
    assert build("log2", fmt, 24, stage_core_kernel_level=0, fast=False).timing.latency == 1  # of single multipliers
    assert build("log2", fmt, stage_core_mul_level=6, fast=False).timing.latency == 3
    mixed = [build("log2", fmt, stage_core_kernel_level=k, stage_core_mul_level=6, fast=False) for k in range(14)]
    assert [model.timing.latency for model in mixed] == [3, 4, 4, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 11]
    assert build("log2", fmt, stage_core_kernel_level=6, stage_core_mul_level=10, fast=False).timing.latency == 8
    assert build("log2", fmt, stage_core_kernel_level=6, stage_core_mul_level=6).timing.latency == 5  # degree 1
    latency = [build("log2", ZkfFormat(8, 36), stage_core_kernel_level=k).timing.latency for k in range(14)]
    assert latency == [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9]
    assert [build("log2", ZkfFormat(8, 36), stage_core=member).timing.latency for member in CoreStages] == [2, 2]
    ends: dict[str, Any] = {"stage_input": 2, "stage_core": ~CoreStages.NONE, "stage_output": 3}
    assert build("log2", ZkfFormat(8, 36), **ends).timing.latency == 8


def test_refusals() -> None:
    fmt = ZkfFormat(6, 18)
    with pytest.raises(ValueError, match=r"^wmultiplier=15 is below 16$"):
        Log2(fmt, wmultiplier=15)
    # Each level knob outside its range, as the model is made, by its name.
    with pytest.raises(ValueError, match=r"^stage_core_kernel_level=14 is outside 0\.\.13$"):
        Log2(fmt, wmultiplier=18, stage_core_kernel_level=14, stage_core_mul_level=6)
    with pytest.raises(ValueError, match=r"^stage_core_mul_level=14 is outside 0\.\.13$"):
        Log2(fmt, wmultiplier=18, stage_core_mul_level=14)


# wexp, wman, kind, shard, shards, stages
EACH: tuple[dict[str, Any], ...] = (
    {},
    {"si": 1},
    {"kl": 8},
    {"sm": 10},
    {"sc": ~CoreStages.NONE},
    {"so": 1},
    {"kl": 8, "sc": CoreStages.ALIGN},
)
type Row = tuple[int, int, str | int, int, int, dict[str, Any]]
DEEP: list[Row] = []
DEEP += [(2, 16, "every", 0, 1, {"fa": False} | stages) for stages in EACH[1:]]  # bare, it is `test_every_compact`
DEEP += [(3, 16, "every", shard, 2, {"fa": False} | stages) for shard in (0, 1) for stages in EACH]
DEEP += [(w, m, 512 if m > 16 else 128, 0, 1, {"fa": False} | s) for w, m in ((8, 27), (8, 32), (14, 16)) for s in EACH]
DEEP += [
    (w, m, "strata", 0, 1, {})
    for w, m in ((8, 16), (8, 18), (8, 24), (8, 27), (8, 32), (8, 36), (8, 48), (8, 53), (11, 53))
]


@pytest.mark.deep
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(row: Row) -> None:
    wexp, wman, kind, shard, shards, stages = row
    run("log2", wexp, wman, kind, shard, shards, **stages)

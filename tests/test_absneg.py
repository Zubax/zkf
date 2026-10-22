import pytest

from _bench import Case, run_pipeline
from _operands import unary_operands
from zkf import AbsNeg, Finite, Zkf, ZkfFormat

# wexp, wman, count
FORMATS = [(2, 4, 0), (3, 4, 0), (5, 11, 512), (8, 24, 1024), (11, 53, 384)]
STAGES = [
    {"stage_input": 1},
    {"stage_output": 1},
    {"stage_input": 1, "stage_output": 1},
    {"stage_input": 2, "stage_output": 1},
    {"stage_output": 2},
    {"stage_input": 2, "stage_output": 2},
]
DEEP = [(2, 5, 0), (4, 5, 0), (3, 6, 0), (2, 6, 0), (6, 17, 512), (8, 23, 512)]


def run(cls: type[AbsNeg] | type[Finite], wexp: int, wman: int, count: int, **knobs: int) -> None:
    fmt = ZkfFormat(wexp, wman)
    model = cls(fmt, **knobs)
    cases = [Case({"x": x}, model.evaluate(Zkf(fmt, x))._asdict()) for x in unary_operands(fmt, count)]
    run_pipeline(model.component(), model.timing.latency, cases)


@pytest.mark.parametrize("cls", [AbsNeg, Finite], ids=lambda cls: cls.operator)
@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(cls: type[AbsNeg] | type[Finite], row: tuple[int, int, int]) -> None:
    run(cls, *row)


@pytest.mark.parametrize("cls", [AbsNeg, Finite], ids=lambda cls: cls.operator)
@pytest.mark.parametrize("knobs", STAGES, ids=str)
def test_stages(cls: type[AbsNeg] | type[Finite], knobs: dict[str, int]) -> None:
    run(cls, 3, 4, 0, **knobs)


@pytest.mark.parametrize("cls", [AbsNeg, Finite], ids=lambda cls: cls.operator)
def test_no_register_is_no_clock(cls: type[AbsNeg] | type[Finite]) -> None:
    fmt = ZkfFormat(3, 4)
    assert not cls(fmt).clocked and cls(fmt, stage_output=1).clocked


@pytest.mark.deep
@pytest.mark.parametrize("cls", [AbsNeg, Finite], ids=lambda cls: cls.operator)
@pytest.mark.parametrize("row", DEEP, ids=str)
def test_deep(cls: type[AbsNeg] | type[Finite], row: tuple[int, int, int]) -> None:
    run(cls, *row)

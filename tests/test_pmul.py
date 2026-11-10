import numpy as np
import pytest
from amaranth.lib import data

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from zkf import Pmul
from zkf._pmul import positions

CORNERS = (0, 1, -1, 0xAAAAAAAAAAAAAAAAAAAAAAAA, 0x555555555555555555555555, 0x10000, 0xFFFF)


def product(a: int, b: int, wa: int, wb: int, signed: bool) -> int:
    if signed:
        a, b = a - ((a >> (wa - 1)) << wa), b - ((b >> (wb - 1)) << wb)
    return (a * b) & ((1 << wa + wb) - 1)


def operands(wa: int, wb: int, count: int) -> list[tuple[int, int]]:
    """The corner patterns and the extremes of each operand against those of the other, then random pairs."""
    rng = np.random.default_rng(SEED)

    def patterns(width: int) -> list[int]:
        return [c & ((1 << width) - 1) for c in CORNERS] + [((1 << width - 1) - 1), 1 << (width - 1)]

    pairs = [(a, b) for a in patterns(wa) for b in patterns(wb)]
    return pairs + [(random_bits(wa, rng), random_bits(wb, rng)) for _ in range(count)]


def run(
    wa: int,
    wb: int,
    signed: bool,
    stages: int,
    wmultiplier: int,
    *,
    pairs: list[tuple[int, int]] | None = None,
) -> None:
    sideband = data.StructLayout({"tag": 5, "flag": 1})
    dut = Pmul(wa, wb, signed=signed, wmultiplier=wmultiplier, stages=stages, sideband=sideband)
    cases = []
    for index, (a, b) in enumerate(pairs or operands(wa, wb, 160)):
        tag = {"tag": index & 31, "flag": (index >> 5) & 1}
        want = {"p": product(a, b, wa, wb, signed), "sb_out": sideband.const(tag)}
        cases.append(Case({"a": a, "b": b, "sb_in": tag}, want))
    run_pipeline(dut, dut.latency, cases)


# Uneven, matched and narrow products, the widest of binary64, and a 4x3 grid, at every count that an operator takes.
SIZES = [(29, 20), (39, 24), (13, 25), (18, 18), (12, 12), (36, 36), (53, 53), (66, 41)]


@pytest.mark.deep
@pytest.mark.parametrize(
    "size, signed, stages",
    [
        (size, signed, stages)
        for size in SIZES
        for signed in (False, True)
        for stages in range(len(positions(*size, signed=signed, wmultiplier=18)) + 1)
    ],
    ids=str,
)
def test_deep_depths(size: tuple[int, int], signed: bool, stages: int) -> None:
    run(*size, signed, stages, 18)


# wa, wb, signed, wmultiplier: the grids and sums each row is there for, each at every count that an operator takes.
GRIDS = [
    (66, 41, False, 18),  # 4x3
    (30, 50, True, 18),  # signed 2x3
    (40, 20, False, 18),  # 3x2
    (50, 50, False, 12),  # 5x5
    (53, 28, True, 18),  # four rows: two pairs
    (35, 52, False, 18),  # two rows
    (29, 20, True, 11),  # 3x2 signed
    (29, 20, False, 10),  # 3x2 unsigned
    (23, 40, False, 18),  # uneven diagonals
    (40, 36, False, 18),
    (43, 49, False, 18),
    (64, 20, False, 18),
    (19, 55, False, 18),
    (22, 43, True, 8),  # more than four rows
    (9, 70, True, 8),
    (61, 41, True, 9),
    (53, 53, False, 8),  # 13 diagonals: four levels
    (66, 41, True, 8),  # ten rows
    (25, 16, True, 18),  # one column
    (53, 18, False, 18),  # three rows of one
    (53, 17, True, 18),  # four rows of one
]


@pytest.mark.deep
@pytest.mark.parametrize(
    "row, stages",
    [
        (row, stages)
        for row in GRIDS
        for stages in range(len(positions(*row[:2], signed=row[2], wmultiplier=row[3])) + 1)
    ],
    ids=str,
)
def test_deep_grids(row: tuple[int, int, bool, int], stages: int) -> None:
    wa, wb, signed, wmultiplier = row
    run(wa, wb, signed, stages, wmultiplier)


# 3x3, the smallest grid that has every kind of position.
@pytest.mark.parametrize("stages", [3, 4])
@pytest.mark.parametrize("signed", [False, True])
def test_every_pair_on_the_smallest_grid(signed: bool, stages: int) -> None:
    pairs = [(a, b) for a in range(256) for b in range(256)]
    run(8, 8, signed, stages, 3 + signed, pairs=pairs)


def test_a_count_of_stages_without_a_register_is_refused() -> None:
    """One multiplier, a 2x1 grid, and a 4x3 one with two levels of sums: each takes up to its registers."""
    for wa, wb, signed, top in ((18, 18, False, 3), (36, 18, False, 3), (66, 41, True, 5)):
        Pmul(wa, wb, signed=signed, wmultiplier=18, stages=top)
        for stages in (-1, top + 1):
            with pytest.raises(ValueError, match=rf"^stages={stages} is outside 0\.\.{top}$"):
                Pmul(wa, wb, signed=signed, wmultiplier=18, stages=stages)


def test_a_multiplier_narrower_than_2_bits_is_refused() -> None:
    for wmultiplier in (1, 0, -1):
        with pytest.raises(ValueError, match=rf"^wmultiplier={wmultiplier} is below 2$"):
            Pmul(8, 8, signed=True, wmultiplier=wmultiplier, stages=0)

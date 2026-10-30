import numpy as np
import pytest
from amaranth.lib import data

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from zkf._pmul import Pmul, counts, positions, slices

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


# Default tier: one product of each kind at every count; the deep tier walks every size and grid below.
KINDS = [
    (12, 12, False, 4),  # unsigned 3x3: diagonals with a pair level
    (66, 41, True, 18),  # signed 4x3: rows, then the rows in pairs
    (29, 20, True, 11),  # signed 3x2
    (18, 53, False, 18),  # 1x3: one row
]


@pytest.mark.parametrize(
    "kind, stages",
    [
        (kind, stages)
        for kind in KINDS
        for stages in range(len(positions(*kind[:2], signed=kind[2], wmultiplier=kind[3])) + 1)
    ],
    ids=str,
)
def test_kinds(kind: tuple[int, int, bool, int], stages: int) -> None:
    wa, wb, signed, wmultiplier = kind
    run(wa, wb, signed, stages, wmultiplier)


# Uneven, matched and narrow products, the widest of binary64, at every count, those that an operator refuses included;
# the signed 66x41 is a kind.
SIZES = [(29, 20), (39, 24), (13, 25), (18, 18), (12, 12), (36, 36), (53, 53)]


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


# wa, wb, signed, wmultiplier: the grids and sums each row is there for, each at every count.
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
    (12, 12, False, 6),  # 2x2 of the narrowest slices
    (8, 9, True, 6),
    (9, 8, False, 5),
    (11, 8, False, 6),
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


# wa, wb, signed, wmultiplier: slices of a, slices of b; the levels of the sum that a register can follow; the counts
# that an operator takes.
CUTS = {
    (18, 18, False, 18): ((1, 1), 0, range(4)),
    (18, 18, True, 18): ((1, 1), 0, range(4)),
    (19, 18, True, 18): ((2, 1), 0, range(2, 4)),
    (18, 53, False, 18): ((1, 3), 0, range(3, 4)),
    (53, 18, False, 18): ((3, 1), 0, range(3, 4)),
    (36, 36, False, 18): ((2, 2), 0, range(2, 4)),
    (20, 40, False, 18): ((2, 3), 1, range(3, 5)),
    (29, 20, False, 11): ((3, 2), 1, range(3, 5)),
    (36, 36, True, 18): ((3, 3), 1, range(3, 5)),
    (66, 41, True, 18): ((4, 3), 2, range(3, 6)),
    (53, 53, False, 8): ((7, 7), 3, range(3, 7)),
}


def test_how_a_product_is_cut() -> None:
    """The grid is that of the widths at every count, and the latency does not show it."""
    for (wa, wb, signed, wmultiplier), (grid, levels, taken) in CUTS.items():
        assert slices(wa, wb, signed=signed, wmultiplier=wmultiplier) == grid
        sums = tuple(f"sum{k}" for k in range(1, levels + 1))
        assert positions(wa, wb, signed=signed, wmultiplier=wmultiplier) == ("output", "capture", "grid", *sums)
        assert counts(wa, wb, signed=signed, wmultiplier=wmultiplier) == taken
        for stages in range(levels + 4):
            dut = Pmul(wa, wb, signed=signed, wmultiplier=wmultiplier, stages=stages)
            assert (dut._ga, dut._gb) == grid, (wa, wb, signed, wmultiplier, stages)
            assert dut.latency == stages

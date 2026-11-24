import numpy as np
import pytest
from amaranth.lib import data

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from zkf import Pmul
from zkf._pmul import levels

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
    level: int,
    wmultiplier: int,
    *,
    pairs: list[tuple[int, int]] | None = None,
) -> None:
    sideband = data.StructLayout({"tag": 5, "flag": 1})
    dut = Pmul(wa, wb, signed=signed, wmultiplier=wmultiplier, level=level, sideband=sideband)
    cases = []
    for index, (a, b) in enumerate(pairs or operands(wa, wb, 160)):
        tag = {"tag": index & 31, "flag": (index >> 5) & 1}
        want = {"p": product(a, b, wa, wb, signed), "sb_out": sideband.const(tag)}
        cases.append(Case({"a": a, "b": b, "sb_in": tag}, want))
    run_pipeline(dut, dut.latency, cases)


# Uneven, matched and narrow products, the widest of binary64, and a 4x3 grid, at every level that an operator takes.
SIZES = [(29, 20), (39, 24), (13, 25), (18, 18), (12, 12), (36, 36), (53, 53), (66, 41)]


@pytest.mark.deep
@pytest.mark.parametrize(
    "size, signed, level",
    [
        (size, signed, level)
        for size in SIZES
        for signed in (False, True)
        for level in range(len(levels(*size, signed=signed, wmultiplier=18)))
    ],
    ids=str,
)
def test_deep_depths(size: tuple[int, int], signed: bool, level: int) -> None:
    run(*size, signed, level, 18)


# wa, wb, signed, wmultiplier: the grids and sums each row is there for, each at every level that an operator takes.
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
    "row, level",
    [(row, level) for row in GRIDS for level in range(len(levels(*row[:2], signed=row[2], wmultiplier=row[3])))],
    ids=str,
)
def test_deep_grids(row: tuple[int, int, bool, int], level: int) -> None:
    wa, wb, signed, wmultiplier = row
    run(wa, wb, signed, level, wmultiplier)


# 3x3, the smallest grid that has every kind of position: every register at 13, the sum's first level alone at 3.
@pytest.mark.parametrize("level", [3, 13])
@pytest.mark.parametrize("signed", [False, True])
def test_every_pair_on_the_smallest_grid(signed: bool, level: int) -> None:
    pairs = [(a, b) for a in range(256) for b in range(256)]
    run(8, 8, signed, level, 3 + signed, pairs=pairs)


def code(registers: frozenset[str]) -> str:
    names = {"capture": "C", "grid": "G", "grid2": "G2", "sum1": "S"} | {f"sum{k}": f"S{k}" for k in range(2, 9)}
    return "".join(c for name, c in (names | {"output": "O"}).items() if name in registers) or "-"


def test_the_levels() -> None:
    """
    On a lone multiplier G, G2 and O are one chain; on a grid without a level of its sum to cut, S is O; a grid with one
    takes every level as its row has it; past 13, each further cut comes first in place of G2, then with it.
    """
    for size, signed, want in (
        ((18, 18), False, "- O O O GO CO CO GO GO GO CGO CGO CGO CGG2O"),
        ((36, 36), False, "- O G O GO CO CO GG2 GO GO CGO CGO CGO CGG2O"),
        ((53, 53), False, "- O G S GO CS CO GG2 GS GSO CGO CGS CGSO CGG2SO"),
        ((66, 41), True, "- O G S GO CS CO GG2 GS GSO CGO CGS CGSO CGG2SO CGSS2O CGG2SS2O"),
        ((90, 90), True, "- O G S GO CS CO GG2 GS GSO CGO CGS CGSO CGG2SO CGSS2O CGG2SS2O CGSS2S3O CGG2SS2S3O"),
    ):
        assert " ".join(code(at) for at in levels(*size, signed=signed, wmultiplier=18)) == want


@pytest.mark.parametrize("signed", [False, True])
def test_every_product_takes_its_capture_and_output_alone(signed: bool) -> None:
    """At two cycles, whether or not its grid's sum has a level to cut."""
    for wa in range(2, 80, 3):
        for wb in range(2, 80, 7):
            for wmultiplier in (4, 9, 18, 27):
                at = levels(wa, wb, signed=signed, wmultiplier=wmultiplier)
                assert frozenset({"capture", "output"}) in at, (wa, wb, wmultiplier)


@pytest.mark.parametrize("signed", [False, True])
def test_no_level_takes_fewer_cycles_than_the_one_below(signed: bool) -> None:
    for wa in range(2, 80, 3):
        for wb in range(2, 80, 7):
            for wmultiplier in (4, 9, 18, 27):
                at = levels(wa, wb, signed=signed, wmultiplier=wmultiplier)
                assert all(len(lower) <= len(upper) for lower, upper in zip(at, at[1:])), (wa, wb, wmultiplier)


def test_a_level_past_the_last_is_refused() -> None:
    """One multiplier, a 2x1 grid, and a 4x3 one with two levels of sums to cut: each takes up to its levels."""
    for wa, wb, signed, top in ((18, 18, False, 13), (36, 18, False, 13), (66, 41, True, 15)):
        assert Pmul(wa, wb, signed=signed, wmultiplier=18, level=top).latency >= 4
        for level in (-1, top + 1):
            with pytest.raises(ValueError, match=rf"^level={level} is outside 0\.\.{top}$"):
                Pmul(wa, wb, signed=signed, wmultiplier=18, level=level)


def test_a_multiplier_narrower_than_2_bits_is_refused() -> None:
    for wmultiplier in (1, 0, -1):
        with pytest.raises(ValueError, match=rf"^wmultiplier={wmultiplier} is below 2$"):
            Pmul(8, 8, signed=True, wmultiplier=wmultiplier, level=0)

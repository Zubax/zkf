"""
`FixedToFloat` as `cordic`'s packers have it, against its reference: every magnitude at small widths, and directed
ones at the widths of those packers, where Yosys also proves its normalizer against a plain one.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from amaranth.back import rtlil
from amaranth.hdl import Module, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from _bench import Case, run_pipeline
from _operands import SEED
from _yosys import YOSYS
from zkf import ZkfFormat
from zkf._fixed_to_float import FixedToFloat, fixed_to_float_reference
from zkf._shift import Normshift

type Txn = tuple[int, int, int, int]
"""sign, force_inf, exp_offset, mag"""

# NORMALIZE, PACK, ROUNDED: none, the catalog's `cordic`, and all.
STAGES = [(0, 0, 0), (1, 1, 0), (1, 1, 1)]
# The packers of `cordic`: a rotation's, which takes no magnitude that overflows, and a vectoring's, which saturates.
KINDS = {"rotation": {"assume_no_overflow": True}, "vectoring": {"saturate_round_carry": True}}


def weu_of(fmt: ZkfFormat, wmag: int) -> int:
    return fmt.wexp + wmag.bit_length() + 3


def run(
    fmt: ZkfFormat,
    wmag: int,
    txns: list[Txn],
    *,
    kind: str,
    stages: tuple[int, int, int],
    most: int | None = None,
    weu: int | None = None,
) -> None:
    options = KINDS[kind]
    dut = FixedToFloat(
        fmt,
        wmag=wmag,
        weu=weu or weu_of(fmt, wmag),
        classify_ahead=True,
        exp_is_biased=True,
        stage_normalize=stages[0],
        stage_pack=stages[1],
        stage_output=stages[2],
        most=most,
        **options,
    )
    saturate = options.get("saturate_round_carry", False)
    cases = []
    for sign, inf, exp, mag in txns:
        found = fixed_to_float_reference(fmt, sign, mag, exp, wmag, saturate_round_carry=saturate)
        if found.is_inf and not inf and options.get("assume_no_overflow"):
            continue
        want = fmt.inf(sign) if inf else found
        cases.append(Case({"sign": sign, "force_inf": inf, "exp_offset": exp + fmt.bias, "mag": mag}, {"y": want}))
    run_pipeline(dut, sum(stages), cases)


def leading(wmag: int, most: int | None) -> range:
    return range(wmag if most is None else most + 1)


def every(fmt: ZkfFormat, wmag: int, most: int | None) -> list[Txn]:
    """
    Every magnitude that `most` allows, at an exponent that keeps every count apart, then at those of a result next to
    the least normal, to the largest finite and past it.
    """
    mags = [mag for mag in range(1 << wmag) if not mag or most is None or wmag - mag.bit_length() <= most]
    edges = [fmt.min_exp_unbiased - 1, fmt.min_exp_unbiased, fmt.max_exp_unbiased, fmt.max_exp_unbiased + 1]
    txns = [(mag & 1, 0, wmag, mag) for mag in mags]
    for k, mag in enumerate(mags):
        count = wmag - mag.bit_length()
        txns.append((k & 1, int(k % 7 == 0), edges[k % len(edges)] + count, mag))
    return txns


# A tree of the width, one narrower and one wider; an odd count, whose widest digit is one bit; the sticky of one bit
# and of most; the bound of the second packer of `cordic` and one of none.
SMALL = [(5, 4, 7, None), (5, 4, 8, None), (5, 4, 9, None), (6, 5, 12, None), (6, 4, 12, 5), (5, 4, 8, 1), (5, 4, 9, 0)]


@pytest.mark.parametrize("stages", STAGES, ids=str)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("row", SMALL, ids=str)
def test_every_magnitude(row: tuple[int, int, int, int | None], kind: str, stages: tuple[int, int, int]) -> None:
    wexp, wman, wmag, most = row
    fmt = ZkfFormat(wexp, wman)
    run(fmt, wmag, every(fmt, wmag, most), kind=kind, stages=stages, most=most)


def directed(fmt: ZkfFormat, wmag: int, most: int | None) -> list[Txn]:
    """
    At every count: the leading one alone; one more set bit at the last place of the significand, the guard, the round,
    each end of the sticky and one place between; a tie to either parity, and broken by the lowest bit; an all-ones
    significand with the guard, which rounds into the next binade, at a middle exponent, at the largest finite, past it
    and next to the least normal. Then zeros and random magnitudes.
    """
    low, wman = wmag - fmt.wman, fmt.wman
    top = fmt.max_exp_unbiased
    rng = np.random.default_rng(SEED)
    txns: list[Txn] = [(0, 0, 0, 0), (1, 0, top, 0), (0, 1, 0, 0), (1, 1, top + 5, 0)]
    for count in leading(wmag, most):
        lead = wmag - 1 - count
        one = 1 << lead
        mid = count - (count & 3)
        txns.append((count & 1, 0, mid, one))
        for place in {low, low - 1, low - 2, low - 3, 0, int(rng.integers(0, low - 2))}:
            if place >= count:
                txns.append((0, 0, mid, one | 1 << (place - count)))
        if low - 1 >= count:
            for parity in (0, 1):
                tie = one | 1 << (low - 1 - count) | parity << (low - count)
                txns += [(parity, 0, mid, tie), (parity, 0, mid, tie | 1)]
        full = ((1 << (wman + 1)) - 1) << max(lead - wman, 0) >> max(wman - lead, 0)
        for exp in (mid, top + count, fmt.min_exp_unbiased - 1 + count, top + 1 + count):
            txns.append((count & 1, 0, exp, full))
    for _ in range(400):
        count = int(rng.integers(0, len(leading(wmag, most))))
        length = wmag - count
        mag = (1 << (length - 1)) | int(rng.integers(0, 1 << min(length - 1, 62))) << max(length - 63, 0)
        if rng.random() < 0.3:
            mag &= ~((1 << int(rng.integers(0, length))) - 1) | (1 << (length - 1))
        txns.append((int(rng.integers(0, 2)), 0, int(rng.integers(-40, 40)), mag))
    return txns


# The catalog's packers of `cordic`, with their widths of the magnitude and the exponent: a rotation's, a vectoring's or
# both modes', and the second packer.
CATALOG_WIDTHS = [(6, 18, 52, 28, "rotation", 9), (6, 18, 57, 34, "vectoring", 15), (8, 36, 96, 54, "rotation", 11)]
CATALOG_WIDTHS += [(8, 36, 102, 61, "vectoring", 18), (6, 18, 34, 1, "rotation", 8), (8, 36, 61, 1, "rotation", 10)]


@pytest.mark.parametrize("stages", STAGES, ids=str)
@pytest.mark.parametrize("row", CATALOG_WIDTHS, ids=str)
def test_directed(row: tuple[int, int, int, int, str, int], stages: tuple[int, int, int]) -> None:
    wexp, wman, wmag, most, kind, weu = row
    fmt = ZkfFormat(wexp, wman)
    run(fmt, wmag, directed(fmt, wmag, most), kind=kind, stages=stages, most=most, weu=weu)


class Miter(wiring.Component):
    """`bad`: `Normshift`'s results differ from a plain count, shift and sticky on an `x` within `most`; no register."""

    x: Signal
    bad: Signal

    def __init__(self, wmag: int, most: int | None, wsticky: int) -> None:
        self._wmag, self._most, self._wsticky = wmag, most, wsticky
        super().__init__({"x": In(wmag), "bad": Out(1)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wmag, most, x = self._wmag, self._most, self.x
        m.submodules.normalize = normalize = Normshift(wmag, wsticky=self._wsticky, sideband=1, most=most)
        m.d.comb += normalize.x.eq(x)
        count = Signal(len(normalize.count), name="count")
        for k in range(wmag):  # the last assignment, of the leading one, takes priority
            with m.If(x[k]):
                m.d.comb += count.eq(wmag - 1 - k)
        y = Signal(wmag, name="y")
        m.d.comb += y.eq(x << count)
        differs = (normalize.count != count) | (normalize.y != y) | (normalize.sticky != y[: self._wsticky].any())
        within = x[wmag - most - 1 :].any() if most is not None else x.any()
        m.d.comb += self.bad.eq(((x == 0) & ~normalize.zero) | (within & (normalize.zero | differs)))
        return m


@pytest.mark.skipif(YOSYS is None, reason="no Yosys of the system")
@pytest.mark.parametrize("row", CATALOG_WIDTHS, ids=str)
def test_proved_against_a_plain_normalizer(row: tuple[int, int, int, int, str, int], tmp_path: Path) -> None:
    assert YOSYS is not None
    _, wman, wmag, most, _, _ = row
    miter = Miter(wmag, most, wmag - wman - 2)
    (tmp_path / "miter.il").write_text(rtlil.convert(miter, name="miter"))
    (tmp_path / "prove.ys").write_text(
        "read_rtlil miter.il\nhierarchy -top miter\nproc\nflatten\nopt -fast\nsat -prove bad 0 -verify\n"
    )
    proved = subprocess.run([YOSYS, "-q", "-s", "prove.ys"], cwd=tmp_path, capture_output=True, text=True)
    assert proved.returncode == 0, proved.stdout + proved.stderr

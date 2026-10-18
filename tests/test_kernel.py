from dataclasses import replace

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from zkf import Exp2, Log2, ZkfFormat
from zkf._pmul import slices
from zkf._trans import Kernel, Spec, _schedule, _sizes, kernel_counts, spec as table


def arguments(width: int, rw: int, count: int) -> list[int]:
    """The ends of the range and of a few segments, where a row gives way to the next, then random ones."""
    out = [
        0,
        1,
        ((1 << width) - 1),
        (1 << width) - 2,
        ((1 << rw) - 1),
        1 << rw,
        (1 << rw) + 1,
        ((1 << width) - 1) ^ ((1 << rw) - 1),
    ]
    rng = np.random.default_rng(SEED)
    for _ in range(12):
        segment = int(rng.integers(0, 1 << (width - rw)))
        out += [(segment << rw) | low for low in (0, 1, ((1 << rw) - 1), ((1 << rw) - 1) - 1, 1 << (rw - 1))]
    return out + [random_bits(width, rng) for _ in range(count)]


def run(function: str, wman: int, stages: int, count: int = 400, wmultiplier: int = 18) -> None:
    spec = table(function, wman, wmultiplier)
    dut = Kernel(spec, wmultiplier=wmultiplier, stages=stages)
    rw, cf = spec.rw, spec.cf
    name, result, width = ("f", "s", cf) if function == "exp2" else ("frac", "c", wman - 1)
    cases = []
    for argument in arguments(width, rw, count):
        cases.append(Case({name: argument}, {result: spec.at(argument)}))
    run_pipeline(dut, dut.latency, cases)


@pytest.mark.parametrize("wman", [16, 18, 24, 27, 32, 36, 48, 53])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_every_table(function: str, wman: int) -> None:
    """At the least count and at the most, which cut every product as far as it goes."""
    taken = kernel_counts(table(function, wman, 18), 18)
    run(function, wman, taken.start, 200)
    run(function, wman, taken.stop - 1, 200)


# Every count at other widths: single multipliers at 24, from no register of the kernel on, and cut ones at 8 and 12.
@pytest.mark.parametrize("wmultiplier", [8, 12, 24])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_other_multipliers(function: str, wmultiplier: int) -> None:
    for stages in kernel_counts(table(function, 18, wmultiplier), wmultiplier):
        run(function, 18, stages, 200, wmultiplier)


@pytest.mark.parametrize("wman", [16, 18, 24, 27, 32, 36, 48, 53])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_every_row(function: str, wman: int) -> None:
    spec = table(function, wman, 18)
    every_row(spec, Kernel(spec, wmultiplier=18, stages=kernel_counts(spec, 18).start))


def test_what_a_product_counts_as_its_capture() -> None:
    """
    The square takes the register of `w`, and the top of the chain those of the square and of its pair, which `c3*w`
    of one multiplier gives after the pair's sum; a product of the chain fed by a sum takes none.
    """
    assert _schedule(table("exp2", 36, 18), 18, 3).captured == {"sq_hh", "sq_hl", "p2"}
    assert _schedule(table("exp2", 40, 18), 18, 3).captured == {"sq", "p4"}
    assert _schedule(table("log2", 53, 18), 18, 3).captured == {"sq", "p4"}


@pytest.mark.parametrize("wman", [18, 36, 40, 53])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_no_product_is_given_more_than_its_positions(function: str, wman: int) -> None:
    """Nor a product of one multiplier its pipeline register."""
    spec = table(function, wman, 18)
    for stages in kernel_counts(spec, 18):
        built = Kernel(spec, wmultiplier=18, stages=stages).elaborate(None)
        for name, (a, b) in _sizes(spec).items():
            product, lone = built.submodules[name], slices(a, b, signed=False, wmultiplier=18) == (1, 1)
            assert not (lone and product._stages >= 3), (name, stages)


def every_row(spec: Spec, dut: Kernel) -> None:
    """The kernel of the table against its reference at both ends of every row and inside it."""
    function = spec.function
    rw, rng = spec.rw, np.random.default_rng(SEED)
    name, result = ("f", "s") if function == "exp2" else ("frac", "c")
    cases = []
    for segment, row in enumerate(spec.coeffs):
        for low in (0, ((1 << rw) - 1), random_bits(rw, rng)):
            cases.append(Case({name: (segment << rw) | low}, {result: spec._polynomial(row, low)}))
    run_pipeline(dut, dut.latency, cases)


@pytest.mark.parametrize(
    "model",
    [
        Exp2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel=2),
        Log2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel=2, stage_core_mul=2),
    ],
)
def test_the_attributes_of_an_operator_reach_its_table(model: Exp2 | Log2) -> None:
    for attrs in (None, {"rom_style": "block"}):
        kernel = replace(model, rom_attrs=attrs).component().elaborate(None).submodules.kernel
        assert kernel.elaborate(None).submodules.rom.attrs == (attrs or {})

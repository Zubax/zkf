import re
from dataclasses import replace

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from _trans import INSTRUMENTS, TABLES, every_row, instrument
from zkf import Exp2, Log2, ZkfFormat
from zkf._trans import Kernel, kernel_counts, spec as table


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


def run(function: str, wman: int, stages: int, count: int = 400, wmultiplier: int = 18, fast: bool = False) -> None:
    spec = table(function, wman, wmultiplier, fast)
    dut = Kernel(spec, wmultiplier=wmultiplier, stages=stages)
    cases = []
    for argument in arguments(spec.k + spec.rw, spec.rw, count):
        cases.append(Case({"x": argument}, {"y": spec.at(argument)}))
    run_pipeline(dut, dut.latency, cases)


@pytest.mark.parametrize("function, wman, fast", [(f, w, fast) for f, w, _, fast in TABLES])
def test_every_table(function: str, wman: int, fast: bool) -> None:
    """At every count, the most of which cuts every product as far as it goes."""
    for stages in kernel_counts(table(function, wman, 18, fast), 18):
        run(function, wman, stages, 200, fast=fast)


# Every count at other widths: single multipliers at 24, from no register of the kernel on, and cut ones at 8 and 12.
@pytest.mark.parametrize("wmultiplier", [8, 12, 24])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_other_multipliers(function: str, wmultiplier: int) -> None:
    for stages in kernel_counts(table(function, 18, wmultiplier, False), wmultiplier):
        run(function, 18, stages, 200, wmultiplier)


@pytest.mark.parametrize("name", INSTRUMENTS)
def test_instruments(name: str) -> None:
    """Every row of each structure that formats reach only where no test can enumerate, at its least count and most."""
    spec, wmultiplier = instrument(name), INSTRUMENTS[name][1]
    taken = kernel_counts(spec, wmultiplier)
    for stages in (taken.start, taken.stop - 1):
        every_row(spec, Kernel(spec, wmultiplier=wmultiplier, stages=stages))


@pytest.mark.parametrize("function, wman, fast", [(f, w, fast) for f, w, _, fast in TABLES])
def test_every_row(function: str, wman: int, fast: bool) -> None:
    spec = table(function, wman, 18, fast)
    every_row(spec, Kernel(spec, wmultiplier=18, stages=kernel_counts(spec, 18).start))


@pytest.mark.parametrize(
    "model",
    [
        Exp2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel=2),
        Log2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel=2, stage_core_mul=2),
    ],
)
def test_the_attributes_reach_the_text(model: Exp2 | Log2) -> None:
    """`rom_attrs` mark the table, `mul_attrs` registers (the operands of the products), and nothing else changes."""
    plain = model.compile().splitlines()
    for marked, attribute, declaration in (
        (
            replace(model, rom_attrs={"rom_style": "block"}),
            '(* rom_style = "block" *)',
            r"reg \[\d+:0\] \w+ \[\d+:0\];",
        ),
        (replace(model, mul_attrs={"use_dsp": "yes"}), '(* use_dsp = "yes" *)', r"reg .*"),
    ):
        lines = marked.compile().splitlines()
        at = [k for k, line in enumerate(lines) if line.strip() == attribute]
        assert at and all(re.fullmatch(declaration, lines[k + 1].strip()) for k in at), attribute
        assert [line for line in lines if line.strip() != attribute] == plain

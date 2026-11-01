import re
from dataclasses import replace

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from _trans import TABLES, every_row
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


def run(function: str, wman: int, stages: int, count: int = 400, wmultiplier: int = 18) -> None:
    spec = table(function, wman, wmultiplier)
    dut = Kernel(spec, wmultiplier=wmultiplier, stages=stages)
    rw, cf = spec.rw, spec.cf
    name, result, width = ("f", "s", cf) if function == "exp2" else ("frac", "c", wman - 1)
    cases = []
    for argument in arguments(width, rw, count):
        cases.append(Case({name: argument}, {result: spec.at(argument)}))
    run_pipeline(dut, dut.latency, cases)


@pytest.mark.parametrize("function, wman", [key[:2] for key in TABLES])
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


@pytest.mark.parametrize("function, wman", [key[:2] for key in TABLES])
def test_every_row(function: str, wman: int) -> None:
    spec = table(function, wman, 18)
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

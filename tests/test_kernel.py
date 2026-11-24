import re
from dataclasses import replace

import numpy as np
import pytest
from amaranth.back import rtlil

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from _trans import INSTRUMENTS, OTHERS, TABLES, every_row, instrument, kernel_top, taken
from zkf import Exp2, Log2, ZkfFormat
from zkf._pmul import levels
from zkf._trans import Kernel, _ladder, _sizes, _staging, spec as table


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


def run(function: str, wman: int, level: int, count: int = 400, wmultiplier: int = 18, fast: bool = False) -> None:
    spec = table(function, wman, wmultiplier, fast)
    dut = Kernel(spec, wmultiplier=wmultiplier, level=level)
    cases = []
    for argument in arguments(spec.k + spec.rw, spec.rw, count):
        cases.append(Case({"x": argument}, {"y": spec.at(argument)}))
    run_pipeline(dut, dut.latency, cases)


@pytest.mark.parametrize("function, wman, fast", [(f, w, fast) for f, w, _, fast in TABLES])
def test_every_table(function: str, wman: int, fast: bool) -> None:
    """At every level that builds other registers than the one below, the last of which builds every one."""
    for level in taken(table(function, wman, 18, fast), 18):
        run(function, wman, level, 200, fast=fast)


# Every level at other widths: single multipliers at 24, from no register of the kernel on, and cut ones at 8 and 12.
@pytest.mark.parametrize("wmultiplier", [8, 12, 24])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_other_multipliers(function: str, wmultiplier: int) -> None:
    for level in taken(table(function, 18, wmultiplier, False), wmultiplier):
        run(function, 18, level, 200, wmultiplier)


def test_the_levels() -> None:
    """
    At 8/36 on 18-bit multipliers `p1` is cut into 3x2 multipliers and `p2` into 2x2, the rest single multipliers: each
    takes the level of `Pmul` of its column of `Exp2.stage_core_kernel_level`'s table.
    """
    spec = table("exp2", 36, 18, True)
    p1 = (0, 0, 2, 2, 8, 8, 8, 11, 11, 12, 12, 12, 13, 13)
    p2 = (0, 2, 2, 2, 2, 4, 4, 4, 10, 10, 13, 13, 13, 13)
    single = (0, 0, 0, 1, 1, 1, 6, 6, 6, 6, 6, 10, 10, 13)
    assert kernel_top(spec, 18) == 13
    for level in range(14):
        assert _staging(spec, 18, level) == {
            "p1": p1[level],
            "p3": single[level],
            "sq_hh": single[level],
            "sq_hl": single[level],
            "p2": p2[level],
        }


@pytest.mark.parametrize("function, wman, width, fast", TABLES + OTHERS)
def test_no_level_takes_fewer_cycles_than_the_one_below(function: str, wman: int, width: int, fast: bool) -> None:
    """Nor does any product; the last level builds every register of every product."""
    spec = table(function, wman, width, fast)
    ladder = _ladder(spec, width)
    for below, above in zip(ladder, ladder[1:]):
        assert below.total <= above.total and all(below.stages[name] <= above.stages[name] for name in below.stages)
    last = {name: len(levels(*size, signed=False, wmultiplier=width)) - 1 for name, size in _sizes(spec).items()}
    assert _staging(spec, width, kernel_top(spec, width)) == last


@pytest.mark.deep
@pytest.mark.parametrize("function, wman, width, fast", TABLES + OTHERS)
def test_no_level_takes_fewer_register_bits_than_the_one_below(
    function: str, wman: int, width: int, fast: bool
) -> None:
    """And one that `taken` skips builds the text of the level below."""
    spec = table(function, wman, width, fast)
    texts = [rtlil.convert(Kernel(spec, wmultiplier=width, level=k)) for k in range(kernel_top(spec, width) + 1)]
    bits = [sum(int(n) for n in re.findall(r"cell \$dff \S+\n\s+parameter \\WIDTH (\d+)", text)) for text in texts]
    assert bits == sorted(bits)
    for level in set(range(len(texts))) - set(taken(spec, width)):
        assert texts[level] == texts[level - 1], level


def test_a_level_past_the_last_is_refused() -> None:
    for function, wman, fast in (("exp2", 36, True), ("exp2", 18, True), ("log2", 53, False)):
        spec = table(function, wman, 18, fast)
        top = kernel_top(spec, 18)
        Kernel(spec, wmultiplier=18, level=top)
        for level in (-1, top + 1):
            with pytest.raises(ValueError, match=rf"^level={level} is outside 0\.\.{top}$"):
                Kernel(spec, wmultiplier=18, level=level)


@pytest.mark.parametrize("name", INSTRUMENTS)
def test_instruments(name: str) -> None:
    """Every row of each structure that formats reach only where no test can enumerate, at every level."""
    spec, wmultiplier = instrument(name), INSTRUMENTS[name][1]
    for level in taken(spec, wmultiplier):
        every_row(spec, Kernel(spec, wmultiplier=wmultiplier, level=level))


@pytest.mark.parametrize("function, wman, fast", [(f, w, fast) for f, w, _, fast in TABLES])
def test_every_row(function: str, wman: int, fast: bool) -> None:
    spec = table(function, wman, 18, fast)
    every_row(spec, Kernel(spec, wmultiplier=18, level=0))


@pytest.mark.parametrize(
    "model",
    [
        Exp2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel_level=6),
        Log2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel_level=6, stage_core_mul_level=6),
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
        lines = marked.compile().replace(marked.name, model.name).splitlines()  # the attributes name the module too
        at = [k for k, line in enumerate(lines) if line.strip() == attribute]
        assert at and all(re.fullmatch(declaration, lines[k + 1].strip()) for k in at), attribute
        assert [line for line in lines if line.strip() != attribute] == plain

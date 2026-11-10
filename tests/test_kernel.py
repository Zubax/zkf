import re
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from amaranth.back import rtlil

from _bench import Case, run_pipeline
from _operands import SEED, random_bits
from _trans import INSTRUMENTS, OTHERS, TABLES, every_row, instrument
from zkf import Exp2, Log2, ZkfFormat
from zkf._trans import Kernel, _schedule, _staging, kernel_levels, spec as table


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
    """At every level, the last of which cuts every product as far as it goes."""
    for level in kernel_levels(table(function, wman, 18, fast), 18):
        run(function, wman, level, 200, fast=fast)


# Every level at other widths: single multipliers at 24, from no register of the kernel on, and cut ones at 8 and 12.
@pytest.mark.parametrize("wmultiplier", [8, 12, 24])
@pytest.mark.parametrize("function", ["exp2", "log2"])
def test_other_multipliers(function: str, wmultiplier: int) -> None:
    for level in kernel_levels(table(function, 18, wmultiplier, False), wmultiplier):
        run(function, 18, level, 200, wmultiplier)


def test_the_levels() -> None:
    """
    At 8/36 on 18-bit multipliers `p1` is cut into 3x2 multipliers and `p2` into 2x2, the rest single multipliers;
    each level adds their registers in its order, to the sums last.
    """
    spec = table("exp2", 36, 18, True)
    p1 = (0, 1, 2, 2, 3, 3, 3, 4)
    p2 = (0, 1, 2, 2, 2, 3, 3, 3)
    single = (0, 0, 0, 1, 1, 1, 2, 2)
    for level in range(8):
        assert _staging(spec, 18, level) == {
            "p1": p1[level],
            "p3": single[level],
            "sq_hh": single[level],
            "sq_hl": single[level],
            "p2": p2[level],
        }


def test_the_levels_that_add_registers() -> None:
    """
    Of single multipliers alone, of one 2x2 grid alone, of grids of 3x3 and 2x2 without single multipliers, of every
    kind; and at 53 bits compact, where the only single multiplier's output is its pair's sum register, which the
    pair, waiting for the chain, has already at level 2.
    """
    for function, wman, wmultiplier, fast, levels in (
        ("exp2", 18, 18, True, [0, 3, 6]),
        ("exp2", 18, 17, True, [0, 1, 2, 5]),
        ("log2", 47, 18, False, [0, 1, 2, 4, 5, 7]),
        ("exp2", 36, 18, True, [0, 1, 2, 3, 4, 5, 6, 7]),
        ("exp2", 53, 18, False, [0, 1, 2, 4, 5, 6, 7]),
    ):
        assert kernel_levels(table(function, wman, wmultiplier, fast), wmultiplier) == levels


@pytest.mark.deep
@pytest.mark.parametrize("function, wman, fast", [(f, w, fast) for f, w, _, fast in TABLES + OTHERS])
def test_every_level_adds_a_register(function: str, wman: int, fast: bool, width: int = 18) -> None:
    """Each level builds other cells than the one below, and no fewer register bits."""
    spec = table(function, wman, width, fast)

    def built(level: int) -> tuple[Counter[tuple[str, tuple[str, ...]]], int]:
        text = rtlil.convert(Kernel(spec, wmultiplier=width, level=level))
        found = re.findall(r"cell (\S+) \S+\n(.*?)  end", text, re.S)
        cells = Counter((kind, tuple(sorted(re.findall(r"parameter (.*)", body)))) for kind, body in found)
        bits = sum(int(n) for n in re.findall(r"cell \$dff \S+\n\s+parameter \\WIDTH (\d+)", text))
        return cells, bits

    levels = [built(level) for level in kernel_levels(spec, width)]
    for (cells_below, bits_below), (cells_above, bits_above) in zip(levels, levels[1:]):
        assert cells_below != cells_above and bits_above >= bits_below
    # A refused level, built anyway from its counts, is the level below it.
    taken = kernel_levels(spec, width)
    for level in range(taken[-1]):
        if level not in taken:
            below = max(at for at in taken if at < level)
            kernel = Kernel(spec, wmultiplier=width, level=below)
            kernel._schedule = _schedule(spec, width, _staging(spec, width, level))
            text = rtlil.convert(kernel)
            found = re.findall(r"cell (\S+) \S+\n(.*?)  end", text, re.S)
            refused = Counter((kind, tuple(sorted(re.findall(r"parameter (.*)", body)))) for kind, body in found)
            assert refused == levels[taken.index(below)][0], level


def test_what_the_kernel_lacks_is_refused() -> None:
    for function, wman, fast, level, message in (
        ("exp2", 36, True, 8, r"level=8 is outside 0\.\.7"),
        ("exp2", 36, True, -1, r"level=-1 is outside 0\.\.7"),
        ("exp2", 18, True, 1, r"level=1 adds no register here; take 0 or 3"),
        ("exp2", 53, False, 3, r"level=3 adds no register here; take 2 or 4"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            Kernel(table(function, wman, 18, fast), wmultiplier=18, level=level)


@pytest.mark.parametrize("name", INSTRUMENTS)
def test_instruments(name: str) -> None:
    """Every row of each structure that formats reach only where no test can enumerate, at every level."""
    spec, wmultiplier = instrument(name), INSTRUMENTS[name][1]
    for level in kernel_levels(spec, wmultiplier):
        every_row(spec, Kernel(spec, wmultiplier=wmultiplier, level=level))


@pytest.mark.parametrize("function, wman, fast", [(f, w, fast) for f, w, _, fast in TABLES])
def test_every_row(function: str, wman: int, fast: bool) -> None:
    spec = table(function, wman, 18, fast)
    every_row(spec, Kernel(spec, wmultiplier=18, level=0))


@pytest.mark.parametrize(
    "model",
    [
        Exp2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel_level=6),
        Log2(ZkfFormat(6, 18), wmultiplier=18, stage_core_kernel_level=6, stage_core_mul=2),
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

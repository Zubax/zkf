"""Every operator, elaborated over its knobs, and what an operator compiles to, which linters take."""

import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import pytest
from amaranth.back import rtlil
from amaranth.hdl import Fragment, Shape
from amaranth.lib import wiring

from _catalog import CATALOG
from _cordic import LACKED
from _sweep import admits, configurations, make
from zkf import (
    AbsNeg,
    Add,
    Block,
    Cmp,
    Cordic,
    Divsqrt,
    Exp2,
    Finite,
    Fma,
    FromInt,
    Ilog2,
    Log2,
    Mul,
    MulIlog2,
    Pipe,
    Resize,
    Rint,
    ZkfFormat,
)

# The formats on which an operator's structure differs: per multiplier width, no split, an even one, an uneven
# one and a 3x3 one for a multiplier; a `k` narrower and wider than the exponent for the shift; for the adder a
# normalizer of two and three levels, and an exponent too narrow to reach the aligner's last level; an integer
# narrower and wider than the significand; a format that holds the other, and one narrower in either dimension;
# an integer of one magnitude bit, narrower than the significand and wider; for the divider each residue of the
# significand's width by 3 and both parities, and one whose frame is wider than 32 bits; for a table
# its degrees 1 to 5, and the narrowest and the widest exponent; for `cordic` each mode alone, an odd significand,
# the widest format, and an exponent that a rotation alone takes.
MODELS: list[tuple[type[Block], ZkfFormat, dict[str, object]]] = [
    (Pipe, ZkfFormat(3, 5), {}),
    (Pipe, ZkfFormat(3, 5), {"w": 1}),
    (Mul, ZkfFormat(3, 4), {}),
    (Mul, ZkfFormat(6, 18), {}),
    (Mul, ZkfFormat(8, 36), {}),
    (Mul, ZkfFormat(8, 25), {}),
    (Mul, ZkfFormat(11, 53), {}),
    (Mul, ZkfFormat(6, 18), {"wmultiplier": 8}),
    (MulIlog2, ZkfFormat(3, 4), {}),
    (MulIlog2, ZkfFormat(8, 36), {}),
    (MulIlog2, ZkfFormat(8, 36), {"wk": 3}),
    (MulIlog2, ZkfFormat(8, 36), {"wk": 44}),
    (Add, ZkfFormat(3, 4), {}),
    (Add, ZkfFormat(6, 18), {}),
    (Add, ZkfFormat(3, 36), {}),
    (Add, ZkfFormat(11, 53), {}),
    (FromInt, ZkfFormat(3, 4), {"wint": 8}),
    (FromInt, ZkfFormat(8, 24), {"wint": 5}),
    (FromInt, ZkfFormat(6, 18), {"wint": 128}),
    (FromInt, ZkfFormat(11, 53), {"wint": 64}),
    (AbsNeg, ZkfFormat(3, 4), {}),
    (Finite, ZkfFormat(3, 4), {}),
    (Cmp, ZkfFormat(3, 4), {}),
    (Cmp, ZkfFormat(8, 36), {}),
    (Ilog2, ZkfFormat(3, 4), {"wint": 4}),
    (Ilog2, ZkfFormat(8, 24), {}),
    (Resize, ZkfFormat(3, 4), {}),
    (Resize, ZkfFormat(4, 6), {"fmt_in": ZkfFormat(3, 4)}),
    (Resize, ZkfFormat(3, 4), {"fmt_in": ZkfFormat(5, 4)}),
    (Resize, ZkfFormat(3, 4), {"fmt_in": ZkfFormat(3, 5)}),
    (Resize, ZkfFormat(3, 4), {"fmt_in": ZkfFormat(3, 6)}),
    (Resize, ZkfFormat(5, 11), {"fmt_in": ZkfFormat(6, 18)}),
    (Resize, ZkfFormat(3, 6), {"fmt_in": ZkfFormat(5, 4)}),
    (Rint, ZkfFormat(2, 4), {"wint": 2}),
    (Rint, ZkfFormat(3, 4), {"wint": 8}),
    (Rint, ZkfFormat(3, 4), {"wint": 5}),
    (Rint, ZkfFormat(2, 9), {"wint": 2}),
    (Rint, ZkfFormat(8, 36), {"wint": 44}),
    (Fma, ZkfFormat(3, 4), {}),
    (Fma, ZkfFormat(6, 18), {}),
    (Fma, ZkfFormat(8, 36), {}),
    (Divsqrt, ZkfFormat(3, 9), {}),
    (Divsqrt, ZkfFormat(3, 10), {}),
    (Divsqrt, ZkfFormat(3, 11), {}),
    (Divsqrt, ZkfFormat(6, 18), {}),
    (Divsqrt, ZkfFormat(8, 27), {}),
    (Divsqrt, ZkfFormat(11, 53), {}),
    (Exp2, ZkfFormat(3, 16), {"wmultiplier": 18}),
    (Exp2, ZkfFormat(8, 36), {"wmultiplier": 18}),
    (Exp2, ZkfFormat(11, 53), {"wmultiplier": 18}),
    (Exp2, ZkfFormat(8, 40), {"wmultiplier": 27}),
    (Log2, ZkfFormat(2, 16), {"wmultiplier": 18}),
    (Log2, ZkfFormat(8, 36), {"wmultiplier": 18}),
    (Log2, ZkfFormat(11, 53), {"wmultiplier": 18}),
    (Log2, ZkfFormat(8, 40), {"wmultiplier": 27}),
    (Cordic, ZkfFormat(5, 16), {"wmultiplier": 18}),
    (Cordic, ZkfFormat(5, 16), {"wmultiplier": 18, "mode": 0}),
    (Cordic, ZkfFormat(5, 16), {"wmultiplier": 18, "mode": 1}),
    (Cordic, ZkfFormat(5, 17), {"wmultiplier": 18}),
    (Cordic, ZkfFormat(8, 36), {"wmultiplier": 18}),
    (Cordic, ZkfFormat(11, 53), {"wmultiplier": 18}),
    (Cordic, ZkfFormat(2, 16), {"wmultiplier": 18, "mode": 0}),
]


def models(cls: type[Block], fmt: ZkfFormat, fixed: dict[str, object], every_pair: bool = True) -> Iterator[Block]:
    """Every configuration of the sweep that the format admits; a refusal that the sweep does not expect fails."""
    for knobs in configurations(cls, every_pair=every_pair):
        try:
            model = make(cls, fmt, **(knobs | fixed))
        except ValueError as refusal:
            # A rotation's count of 0, and a kernel's level that adds no register or is past the last.
            expected = (
                r"stage_core_mul=0 is below 1, which a rotation takes"
                r"|level=\d+ (is outside 0\.\.\d+|adds no register here; take \d+ or \d+)"
            )
            assert re.fullmatch(expected, str(refusal)), refusal
            continue
        if admits(model):
            yield model


@pytest.mark.parametrize("shard", range(4))
@pytest.mark.parametrize("row", MODELS, ids=lambda row: f"{row[0].operator}-{row[1].wexp}/{row[1].wman}-{row[2]}")
def test_operators(row: tuple[type[Block], ZkfFormat, dict[str, object]], shard: int) -> None:
    """Each knob converted to a netlist, and every pair of them elaborated."""
    single = set(models(*row, every_pair=False))
    built = list(islice(dict.fromkeys(models(*row)), shard, None, 4))
    assert built or shard, "the format admits no configuration"
    for model in built:
        if model in single:
            rtlil.convert(model.component())
        else:
            Fragment.get(model.component(), None)


LINTERS = {
    "iverilog": ["iverilog", "-g2001", "-o", "/dev/null"],
    # Yosys prints operands of unequal widths, a `case` with no default where some values are not selected, and a
    # comparison with a constant that the operand cannot reach.
    "verilator": ["verilator", "--lint-only", "-Wno-WIDTH", "-Wno-CASEINCOMPLETE", "-Wno-CMPCONST"],
}


@pytest.mark.parametrize("proc_nomux", [True, False], ids=["nomux", "mux"])
@pytest.mark.parametrize("linter", LINTERS)
def test_lint(linter: str, proc_nomux: bool, tmp_path: Path) -> None:
    """The shipped text is Verilog-2001 to the tools a consumer may read it with."""
    if shutil.which(linter) is None:
        if os.environ.get("CI"):
            pytest.fail(f"no {linter} in CI")
        pytest.skip(f"no {linter}")
    # The catalog, each format of the sweep at its first configuration, every register of `cordic` in each mode, and
    # its vectorings without the length.
    specs = list(CATALOG.values()) + [next(models(cls, fmt, fixed)) for cls, fmt, fixed in MODELS]
    for mode in (0, 1, 2):
        specs.append(
            make(Cordic, ZkfFormat(5, 17), wmultiplier=18, mode=mode, stage_core_mul=3, stage_core=~LACKED[mode])
        )
    specs += [make(Cordic, ZkfFormat(5, 17), wmultiplier=18, mode=mode, hypot=False) for mode in (1, 2)]
    texts = [spec.compile(proc_nomux=proc_nomux) for spec in {spec.name: spec for spec in specs}.values()]
    # Without `-nomux` Yosys prints a one-hot select as a `casez` of overlapping patterns.
    command = LINTERS[linter] + (["-Wno-CASEOVERLAP"] if linter == "verilator" and not proc_nomux else [])
    for index, text in enumerate(texts):
        path = tmp_path / f"blob{index}.v"
        path.write_text(text)
        done = subprocess.run([*command, str(path)], capture_output=True, text=True, cwd=tmp_path)
        assert done.returncode == 0 and not done.stderr, f"{text.splitlines()[0]}\n{done.stdout}{done.stderr}"


# Reset held from time zero, or rising after it: SystemVerilog's time zero triggers no `always @*`, which only the text
# printed with `-nomux` has.
HELD, RISING = ["  reg clk = 0, rst = 1;"], ["  reg clk = 0, rst = 0;", "  initial #1 rst = 1;"]
READINGS = {"nomux-2005": (True, "2005", HELD), "nomux-2012": (True, "2012", RISING), "mux-2012": (False, "2012", HELD)}


@pytest.mark.parametrize("reading", READINGS)
@pytest.mark.parametrize("name", [name for name, spec in CATALOG.items() if spec.clocked])
def test_reset_alone_defines_the_text(name: str, reading: str, tmp_path: Path) -> None:
    """
    The text starts unknown, and a cycle of reset defines it: no flag is unknown afterwards, nor any output under
    `out_valid`, though every data input is unknown in each cycle that does not offer a transaction.
    """
    if shutil.which("iverilog") is None:
        if os.environ.get("CI"):
            pytest.fail("no iverilog in CI")
        pytest.skip("no iverilog")
    spec = CATALOG[name]
    ports = [
        (port, Shape.cast(member.shape).width, member.flow == wiring.Out)
        for port, member in spec.signature.members.items()
    ]
    flags = {"in_valid", "in_ready", "out_valid", "out_ready"}
    proc_nomux, standard, reset = READINGS[reading]
    lines = ["module bench;", *reset, "  integer seed = 1, cycle, results = 0;"]
    lines += [f"  {'wire' if out else 'reg'} [{width - 1}:0] {port};" for port, width, out in ports]
    connections = ", ".join(f".{port}({port})" for port in ["clk", "rst", *(port for port, _, _ in ports)])
    lines += [f"  {spec.name} dut({connections});", "  always #5 clk = ~clk;", "  initial begin"]
    lines += ["    for (cycle = 0; cycle < 600; cycle = cycle + 1) begin", "      @(negedge clk) rst = 0;"]
    for port, width, out in ports:
        if out:
            continue
        if port in flags:
            lines.append(f"      {port} = $random(seed);")
        else:
            words = ", ".join(["$random(seed)"] * -(-width // 32))
            lines.append(f"      {port} = in_valid ? {{{words}}} : {width}'bx;")
    lines.append("      #1;")
    for port, _, out in ports:
        if out and port in flags:
            lines.append(f'      if (^{port} === 1\'bx) $fatal(1, "unknown {port} in cycle %0d", cycle);')
        elif out:
            lines.append(f'      if (out_valid && ^{port} === 1\'bx) $fatal(1, "unknown {port} in cycle %0d", cycle);')
    lines += [
        "      results = results + out_valid;",
        "    end",
        '    if (!results) $fatal(1, "no result");',
        "    $finish;",
    ]
    lines += ["  end", "endmodule", ""]
    (tmp_path / "bench.v").write_text("\n".join(lines))
    (tmp_path / "dut.v").write_text(spec.compile(proc_nomux=proc_nomux))
    built = subprocess.run(
        ["iverilog", f"-g{standard}", "-o", "bench", "bench.v", "dut.v"], cwd=tmp_path, capture_output=True, text=True
    )
    assert built.returncode == 0, built.stdout + built.stderr
    done = subprocess.run(["vvp", "-n", "bench"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr

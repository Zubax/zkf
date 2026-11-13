import enum
import re
import subprocess
import sys
import types
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pytest
from amaranth.back import verilog
from amaranth.hdl import Shape

import zkf
from _sweep import configurations, make
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
    Operator,
    Pipe,
    Resize,
    Rint,
    RoundMode,
    Timing,
    UnsupportedFormat,
    ZkfFormat,
)

F18, F36 = ZkfFormat(6, 18), ZkfFormat(8, 36)
MODELS: tuple[type[Block], ...] = (
    AbsNeg,
    Add,
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
)


def test_names() -> None:
    f = ZkfFormat(7, 18)
    add = Add(f, stage_input=1, stage_core=Add.CoreStages.PACK | Add.CoreStages.SUM, stage_output=2)
    assert re.fullmatch(r"[0-9a-f]{12}", add.digest) and add.name == f"zkf_add_{add.digest}"
    assert Add(f, stage_input=True).name == Add(f, stage_input=1).name  # equal configurations, one name
    assert add.name != Add(f, stage_input=1, stage_core=Add.CoreStages.PACK, stage_output=2).name
    assert add.compile().splitlines()[1:8] == [
        "// add:",
        "//   fmt: e7m18",
        "//   stage_input: 1",
        "//   stage_core: CoreStages.SUM|PACK",
        "//   stage_output: 2",
        f"//   timing: {add.timing}",
        f"module {add.name}(clk, rst, in_valid, a, b, out_valid, y);",
    ]
    both = Cordic(F36, stage_core_mul=3, wmultiplier=18)
    assert f"//   timing: {both.timing}" in both.compile().splitlines()


def test_a_latency_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version that gives the knobs another latency gives the module another name."""
    add = Add(ZkfFormat(7, 18))
    before, later = add.name, Timing(add.timing.latency + 1, 1)
    monkeypatch.setattr(Add, "timing", property(lambda self: later))
    assert add.name != before


def test_a_new_register_renames_no_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version that adds a member to a set of registers, shifting the members' values, renames no module."""

    class CoreStages(enum.Flag):
        NEW = enum.auto()
        PACK = enum.auto()

    assert CoreStages.PACK.value != Add.CoreStages.PACK.value
    add = Add(ZkfFormat(7, 18), stage_core=Add.CoreStages.PACK)
    before, config = add.name, dict(add.config, stage_core=CoreStages.PACK)
    monkeypatch.setattr(Add, "config", property(lambda self: config))
    assert add.name == before


def test_the_top_of_the_text() -> None:
    """The ports of the signature, a signed one declared so, behind `clk` and `rst` where the model is clocked."""
    pipe = "my_" + Pipe(w=1).name
    assert Pipe(w=1).compile("my_").splitlines()[:12] == [
        f"// Produced by ZKF {zkf.__version__}, author: Zubax Robotics <sales@zubax.com>",
        "// pipe:",
        "//   w: 1",
        "//   n: 0",
        "//   timing: Timing(latency=0, initiation_interval=1)",
        f"module {pipe}(in_valid, x, out_valid, y);",
        "  input wire in_valid;",
        "  input wire x;",
        "  output wire out_valid;",
        "  output wire y;",
        f"  {pipe}_core core (.in_valid(in_valid), .x(x), .out_valid(out_valid), .y(y));",
        "endmodule",
    ]
    mul_ilog2 = "my_" + MulIlog2(ZkfFormat(3, 4), stage_output=1).name
    assert MulIlog2(ZkfFormat(3, 4), stage_output=1).compile("my_").splitlines()[8:17] == [
        f"module {mul_ilog2}(clk, rst, in_valid, a, k, out_valid, y);",
        "  input wire clk;",
        "  input wire rst;",
        "  input wire in_valid;",
        "  input wire [6:0] a;",
        "  input wire signed [3:0] k;",
        "  output wire out_valid;",
        "  output wire [6:0] y;",
        f"  {mul_ilog2}_core core "
        "(.clk(clk), .rst(rst), .in_valid(in_valid), .a(a), .k(k), .out_valid(out_valid), .y(y));",
    ]
    assert "  output wire signed [3:0] y;" in Ilog2(ZkfFormat(3, 4), wint=4).compile().splitlines()


def test_the_yosys_is_the_first_of_amaranth_use_yosys_that_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A system Yosys by `YOSYS`, a relative path too, else Amaranth's, or a Pyodide host's; what a failing one says is in
    the error; none.
    """
    pipe = Pipe(w=1, n=1)
    ran, system, broken = tmp_path / "ran", tmp_path / "yosys", tmp_path / "broken"
    system.write_text(f'#!/bin/sh\ntouch {ran}\nexec {sys.executable} -m amaranth_yosys "$@"\n')
    broken.write_text("#!/bin/sh\necho no such pass >&2\nexit 1\n")
    system.chmod(0o755)
    broken.chmod(0o755)
    monkeypatch.setenv("AMARANTH_USE_YOSYS", "builtin")
    text = pipe.compile()
    monkeypatch.delenv("AMARANTH_USE_YOSYS")
    monkeypatch.setenv("YOSYS", str(system))
    assert pipe.compile() == text and ran.exists()
    ran.unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YOSYS", "./yosys")
    assert pipe.compile() == text and ran.exists()
    monkeypatch.setenv("YOSYS", str(broken))
    with pytest.raises(verilog.YosysError, match="^no such pass$"):
        pipe.compile()
    monkeypatch.setenv("YOSYS", str(tmp_path / "absent"))
    assert pipe.compile() == text
    monkeypatch.setenv("AMARANTH_USE_YOSYS", "javascript")
    with pytest.raises(verilog.YosysError, match=r"^no Yosys found for AMARANTH_USE_YOSYS='javascript'"):
        pipe.compile()

    def run(args: list[str], stdin: str) -> list[object]:
        """As a Pyodide host's `runAmaranthYosys`, the bundled Yosys behind it."""
        done = subprocess.run(
            [sys.executable, "-m", "amaranth_yosys", *args], input=stdin, capture_output=True, text=True
        )
        return [done.returncode, done.stdout, done.stderr]

    host = types.ModuleType("js")
    monkeypatch.setitem(sys.modules, "js", host)
    with pytest.raises(verilog.YosysError, match=r"^no Yosys found for AMARANTH_USE_YOSYS='javascript'"):
        pipe.compile()
    host.runAmaranthYosys = run  # type: ignore[attr-defined]
    assert pipe.compile() == text


def test_names_are_distinct() -> None:
    seen: dict[str, object] = {}
    for cls in MODELS:
        for fmt in (F18, F36):
            for knobs in configurations(cls):
                try:
                    model = make(cls, fmt, **knobs)
                except ValueError:
                    # Every value of the sweep is taken but a rotation's count of 0 and a kernel's past its registers.
                    assert cls in (Cordic, Exp2, Log2), knobs
                    continue
                assert seen.setdefault(model.name, model) == model


@pytest.mark.parametrize("cls", [Mul, Fma, Exp2, Log2, Cordic], ids=lambda cls: cls.operator)
def test_multiplier_width_is_required(cls: type[Operator]) -> None:
    """The operators of tables and Cordic take 16 bits at least; the others leave the width to the product."""
    with pytest.raises(TypeError):
        cls(F36)
    if cls in (Mul, Fma):
        cls(F36, wmultiplier=7)
        for bad in (0, -1):
            with pytest.raises(ValueError, match=rf"^wmultiplier={bad} is below 2$"):
                cls(F36, wmultiplier=bad).compile()
        return
    for bad in (15, 0, -1):
        with pytest.raises(ValueError, match=rf"^wmultiplier={bad} is below 16$"):
            cls(F36, wmultiplier=bad)  # type: ignore[call-arg]
    cls(F36, wmultiplier=16)  # type: ignore[call-arg]


def test_a_set_of_another_type_is_refused() -> None:
    """The integer of a former ladder, or another operator's set."""
    with pytest.raises(TypeError, match=r"^stage_core=3 is not of type `Add.CoreStages`$"):
        Add(F18, stage_core=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^stage_core_add=<CoreStages.ALIGN: 1> is not of type `Add.CoreStages`$"):
        Fma(F18, wmultiplier=18, stage_core_add=Log2.CoreStages.ALIGN)  # type: ignore[arg-type]


def test_a_set_of_registers_is_none_by_default() -> None:
    sets = [knob.default for cls in MODELS for knob in fields(cls) if isinstance(knob.default, enum.Flag)]
    assert sets and all(not default and default.name == "NONE" for default in sets)


def test_a_negative_count_of_stages_is_refused() -> None:
    for cls in MODELS:
        for knob in ("n",) if cls is Pipe else ("stage_input", "stage_output"):
            with pytest.raises(ValueError, match=f"^{knob}=-1 is negative$"):
                make(cls, F18, **configurations(cls)[0], **{knob: -1})


def test_refusals() -> None:
    knobs: tuple[Callable[[], Block], ...] = (
        lambda: Exp2(F18, wmultiplier=18, stage_core_kernel_level=7),
        lambda: Exp2(F36, wmultiplier=18, stage_core_kernel_level=8),
        lambda: Log2(F18, wmultiplier=18, stage_core_kernel_level=1),
        lambda: Pipe(w=0),
    )
    for build in knobs:
        with pytest.raises(ValueError) as refusal:
            build()
        assert not isinstance(refusal.value, UnsupportedFormat)
    formats: tuple[Callable[[], Operator], ...] = (
        lambda: Exp2(ZkfFormat(6, 9), wmultiplier=18),
        lambda: Log2(ZkfFormat(20, 18), wmultiplier=18),
        lambda: MulIlog2(F18, wk=0),
        lambda: FromInt(F18, wint=1),
    )
    for build in formats:
        with pytest.raises(UnsupportedFormat):
            build()


def widths(model: Block) -> dict[str, tuple[str, int, bool]]:
    out = {}
    for name, member in model.signature.members.items():
        shape = Shape.cast(member.shape)
        out[name] = (member.flow.value, shape.width, shape.signed)
    return out


def test_ports() -> None:
    f, i, o = F18.wfull, "In", "Out"
    valid = {"in_valid": (i, 1, False), "out_valid": (o, 1, False)}
    assert widths(AbsNeg(F18)) == valid | {"x": (i, f, False), "absolute": (o, f, False), "negated": (o, f, False)}
    assert widths(Finite(F18)) == valid | {"x": (i, f, False), "finite": (o, 1, False), "saturated": (o, f, False)}
    assert widths(Cmp(F18)) == valid | {
        "a": (i, f, False),
        "b": (i, f, False),
        "a_gt_b": (o, 1, False),
        "a_eq_b": (o, 1, False),
        "a_lt_b": (o, 1, False),
        "min": (o, f, False),
        "max": (o, f, False),
    }
    binary = valid | {"a": (i, f, False), "b": (i, f, False), "y": (o, f, False)}
    assert widths(Add(F18)) == binary
    assert widths(Mul(F18, wmultiplier=18)) == binary
    assert widths(Fma(F18, wmultiplier=18)) == binary | {"c": (i, f, False)}
    assert widths(Divsqrt(F18)) == binary | {"op_sqrt": (i, 1, False), "error": (o, 1, False)}
    assert widths(Ilog2(F18, wint=12)) == valid | {
        "a": (i, f, False),
        "y": (o, 12, True),
        "zero": (o, 1, False),
        "infinity": (o, 1, False),
        "negative": (o, 1, False),
    }
    assert widths(MulIlog2(F18)) == valid | {"a": (i, f, False), "k": (i, 7, True), "y": (o, f, False)}
    assert widths(MulIlog2(F18, wk=44))["k"] == (i, 44, True)
    assert widths(FromInt(F18, wint=9)) == valid | {"a": (i, 9, True), "y": (o, f, False)}
    assert widths(Resize(F18, fmt_in=ZkfFormat(8, 36))) == valid | {"a": (i, 44, False), "y": (o, f, False)}
    assert widths(Rint(F18, wint=44)) == valid | {
        "a": (i, f, False),
        "round_mode": (i, 2, False),
        "y_float": (o, f, False),
        "y_int": (o, 44, True),
    }
    unary = valid | {"x": (i, f, False), "y": (o, f, False)}
    assert widths(Exp2(F18, wmultiplier=18, stage_core_kernel_level=6)) == unary
    log2 = Log2(F18, wmultiplier=18, stage_core_kernel_level=6, stage_core_mul=2)
    assert widths(log2) == unary | {"domain_error": (o, 1, False), "pole": (o, 1, False)}
    assert widths(Pipe(w=5)) == valid | {"x": (i, 5, False), "y": (o, 5, False)}
    assert widths(Cordic(F18, stage_core_mul=3, wmultiplier=18)) == valid | {
        "in_ready": (o, 1, False),
        "out_ready": (i, 1, False),
        "vectoring": (i, 1, False),
        "a": (i, f, False),
        "b": (i, f, False),
        "r0": (o, f, False),
        "r1": (o, f, False),
        "quadrant": (o, 2, False),
    }


def test_specialized() -> None:
    """
    A specialization lacks the selecting input and the ports that it leaves unread or undriven; each call starts from
    the default, and each build has its name.
    """
    cordic, divsqrt, rint = Cordic(F18, stage_core_mul=3, wmultiplier=18), Divsqrt(F18), Rint(F18)

    def lacks(model: Block, alone: Block, *ports: str) -> bool:
        return widths(alone) == {name: port for name, port in widths(model).items() if name not in ports}

    assert lacks(cordic, cordic.specialized(vectoring=False), "vectoring", "b")
    assert lacks(cordic, cordic.specialized(vectoring=True), "vectoring", "quadrant")
    assert lacks(cordic, cordic.specialized(vectoring=True, length=False), "vectoring", "r1", "quadrant")
    assert lacks(cordic, cordic.specialized(length=False))
    assert lacks(divsqrt, divsqrt.specialized(op_sqrt=False), "op_sqrt")
    assert lacks(divsqrt, divsqrt.specialized(op_sqrt=True), "op_sqrt", "b")
    assert lacks(rint, rint.specialized(round_mode=RoundMode.CEIL), "round_mode")
    assert cordic.specialized(vectoring=True).specialized() == cordic
    rotation = cordic.specialized(vectoring=False)
    assert cordic.specialized(vectoring=False, length=False) == rotation
    assert cordic.specialized(vectoring=0, length=False) == rotation  # type: ignore[arg-type]
    builds = {cordic.specialized(vectoring=v, length=n) for v in (None, False, True) for n in (False, True)}
    assert len(builds) == len({build.name for build in builds}) == 5


def test_one_configuration_is_one_model() -> None:
    assert MulIlog2(F18) == MulIlog2(F18, wk=F18.wexp + 1)
    assert len({MulIlog2(F18), MulIlog2(F18, wk=F18.wexp + 1), MulIlog2(F18, wk=5)}) == 2
    assert Add(F18) != Add(F36) and Add(F18) != Add(F18, stage_input=1) and Pipe(w=5, n=1) == Pipe(n=1, w=5)
    block = {"rom_style": "block"}
    table = Exp2(F18, wmultiplier=18, stage_core_kernel_level=6, rom_attrs=block)
    plain = Exp2(F18, wmultiplier=18, stage_core_kernel_level=6)
    assert table == plain and table.name != plain.name  # one configuration, two texts, two names
    assert replace(table, stage_output=1).rom_attrs == block and "rom_attrs" not in table.config
    keep: dict[str, int | str] = {"keep": "true", "syn_keep": 1}
    for model in (Mul(F36, stage_core_mul=3, wmultiplier=18), Cordic(F18, stage_core_mul=3, wmultiplier=18), table):
        kept = replace(model, mul_attrs=keep)
        assert kept == model and kept.name != model.name and "mul_attrs" not in kept.config
        assert kept.name == replace(model, mul_attrs=dict(reversed(list(keep.items())))).name
        assert replace(model, mul_attrs={}).name == model.name  # no attributes either way
        assert replace(model, mul_attrs={"keep": True}).name == replace(model, mul_attrs={"keep": 1}).name

    @dataclass(frozen=True, kw_only=True, eq=False)
    class Labeled(Pipe):
        label: str = field(default="", compare=False)

    assert Labeled(w=1) == Labeled(w=1, label="x") and Labeled(w=1).name != Labeled(w=1, label="x").name
    Labeled(w=1, label="x").compile()


def test_clocked() -> None:
    assert not Pipe(w=1).clocked and Pipe(w=1, n=1).clocked
    assert Mul(F18, stage_core_mul=1, wmultiplier=18).clocked and Cordic(F18, stage_core_mul=3, wmultiplier=18).clocked
    assert AbsNeg(F18, stage_output=1).clocked and Finite(F18, stage_output=1).clocked
    unclocked = (
        AbsNeg(F18),
        Finite(F18),
        Add(F18),
        Mul(F18, wmultiplier=18),
        Resize(F18, fmt_in=F18),
        Ilog2(F18),
        Rint(F18),
    )
    assert not any(model.clocked for model in unclocked)


def test_a_default_follows_the_format() -> None:
    """A knob left to its default is derived again when the model is copied to another format; a given one stays."""
    assert replace(MulIlog2(F18), fmt=F36) == MulIlog2(F36)
    assert replace(MulIlog2(F18, wk=5), fmt=F36) == MulIlog2(F36, wk=5)


TEXTS = """
import hashlib, sys, mpmath
from concurrent.futures import ThreadPoolExecutor
import zkf._trans
from zkf import Cordic, Exp2, Log2, ZkfFormat
mpmath.mp.prec = int(sys.argv[1])
fitting = zkf._trans._fitting.prec  # the search's context, which generations at once would leave moved
models = [
    lambda: Log2(ZkfFormat(6, 16), wmultiplier=18, stage_core_kernel_level=6, stage_core_mul=2),
    lambda: Log2(ZkfFormat(8, 36), wmultiplier=18, stage_core_kernel_level=6, stage_core_mul=3),
    lambda: Exp2(ZkfFormat(8, 24), wmultiplier=18, stage_core_kernel_level=6),
    lambda: Exp2(ZkfFormat(8, 36), wmultiplier=18, stage_core_kernel_level=6),
    lambda: Cordic(ZkfFormat(11, 53), wmultiplier=18, stage_core_mul=3),
]
compile = lambda model: model().compile()  # made where it is compiled, which generates its tables
texts = list(ThreadPoolExecutor(8).map(compile, models) if sys.argv[2] == "threads" else map(compile, models))
assert mpmath.mp.prec == int(sys.argv[1]) and zkf._trans._fitting.prec == fitting
print(hashlib.sha256("".join(texts).encode()).hexdigest())
"""


def test_the_text_follows_neither_the_precision_nor_the_threads_of_the_caller() -> None:
    """Each in a fresh interpreter, as the tables and the constants are generated once a process."""

    def digest(precision: int, how: str) -> str:
        root = Path(__file__).resolve().parents[1]
        done = subprocess.run(
            [sys.executable, "-c", TEXTS, str(precision), how], cwd=root, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    assert len({digest(20, "serial"), digest(53, "serial"), digest(2000, "serial"), digest(53, "threads")}) == 1

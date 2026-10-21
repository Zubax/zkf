from dataclasses import replace
from collections.abc import Callable
from typing import Any

import pytest
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
    add = Add(f, stage_input=1, stage_core=Add.Core.PACK | Add.Core.SUM, stage_output=2)
    assert add.name == "zkf_add_e7m18_input1_core_sum_pack_output2_latency5"
    mul = Mul(F36, stage_core_mul=3, wmultiplier=18)
    assert mul.name == "zkf_mul_e8m36_input0_core_mul3_output0_wmultiplier18_latency3"
    assert Pipe(w=24, n=4).name == "zkf_pipe_w24_n4_latency4"
    assert MulIlog2(F18).name == "zkf_mul_ilog2_e6m18_wk7_input0_core_none_output0_latency0"
    assert Resize(F18, fmt_in=ZkfFormat(8, 36)).name == "zkf_resize_e6m18_e8m36_input0_output0_latency0"
    both = Cordic(F36, stage_core_mul=3, wmultiplier=18)
    assert both.name.endswith("_mode2_fast0_latency_rotation24_latency_vectoring43")
    assert "_unroll100_100_" in both.name
    assert Cordic(F36, stage_core_mul=3, wmultiplier=18, mode=1).name.endswith("_mode1_fast0_latency43")
    assert Add(f, stage_input=True).name == Add(f, stage_input=1).name  # equal configurations, one name


def test_the_top_of_the_text() -> None:
    """The ports of the signature, a signed one declared so, behind `clk` and `rst` where the model is clocked."""
    pipe = "my_zkf_pipe_w1_n0_latency0"
    assert Pipe(w=1).compile("my_").splitlines()[:8] == [
        f"// Produced by ZKF {zkf.__version__}, author: Zubax Robotics",
        f"module {pipe}(in_valid, x, out_valid, y);",
        "  input wire in_valid;",
        "  input wire x;",
        "  output wire out_valid;",
        "  output wire y;",
        f"  {pipe}_core core (.in_valid(in_valid), .x(x), .out_valid(out_valid), .y(y));",
        "endmodule",
    ]
    assert MulIlog2(ZkfFormat(3, 4), stage_output=1).compile("my_").splitlines()[1:10] == [
        "module my_zkf_mul_ilog2_e3m4_wk4_input0_core_none_output1_latency1(clk, rst, in_valid, a, k, out_valid, y);",
        "  input wire clk;",
        "  input wire rst;",
        "  input wire in_valid;",
        "  input wire [6:0] a;",
        "  input wire signed [3:0] k;",
        "  output wire out_valid;",
        "  output wire [6:0] y;",
        "  my_zkf_mul_ilog2_e3m4_wk4_input0_core_none_output1_latency1_core core "
        "(.clk(clk), .rst(rst), .in_valid(in_valid), .a(a), .k(k), .out_valid(out_valid), .y(y));",
    ]
    assert "  output wire signed [3:0] y;" in Ilog2(ZkfFormat(3, 4), wint=4).compile().splitlines()


def test_names_are_distinct() -> None:
    seen: dict[str, object] = {}
    for cls in MODELS:
        for fmt in (F18, F36):
            for knobs in configurations(cls):
                try:
                    model = make(cls, fmt, **knobs)
                except ValueError:
                    # Every value of the sweep is taken but a product's own refusals and a nested register alone.
                    assert cls in (Add, Mul, Fma, Cordic, Exp2, Log2), knobs
                    continue
                if isinstance(model.timing, Timing):
                    assert model.name.endswith(f"_latency{model.timing.latency}")
                assert seen.setdefault(model.name, model) == model


@pytest.mark.parametrize("cls", [Mul, Fma, Exp2, Log2, Cordic], ids=lambda cls: cls.operator)
def test_multiplier_width_is_required(cls: type[Operator]) -> None:
    product: dict[str, Any] = {
        Mul: {"stage_core_mul": 3},
        Fma: {"stage_core_mul": 3},
        Cordic: {"stage_core_mul": 3},
        Exp2: {"stage_core_kernel": 3},
        Log2: {"stage_core_kernel": 3, "stage_core_mul": 3},
    }[cls]

    def build(*args: Any, **knobs: Any) -> Operator:
        return cls(*args, **(product | knobs))

    with pytest.raises(TypeError):
        build(F36)
    for bad in (0, 7, -1):
        with pytest.raises(ValueError):
            build(F36, wmultiplier=bad)
    build(F18 if cls in (Exp2, Log2) else F36, wmultiplier=8)  # no table of 8-bit multipliers is 36 bits wide


def test_a_set_of_another_type_is_refused() -> None:
    """The integer of a former ladder, or another operator's set."""
    with pytest.raises(TypeError, match=r"^stage_core=3: a set of `Add.Core` is due$"):
        Add(F18, stage_core=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^stage_core_add=<Core.ALIGN: 1>: a set of `Add.Core` is due$"):
        Fma(F18, wmultiplier=18, stage_core_add=Log2.Core.ALIGN)  # type: ignore[arg-type]


def test_a_negative_count_of_stages_is_refused() -> None:
    for cls in MODELS:
        for knob in ("n",) if cls is Pipe else ("stage_input", "stage_output"):
            with pytest.raises(ValueError, match=f"^{knob}=-1 is negative$"):
                make(cls, F18, **configurations(cls)[0], **{knob: -1})


def test_refusals() -> None:
    knobs: tuple[Callable[[], Block], ...] = (
        lambda: Add(F18, stage_core=Add.Core.NORMALIZE_WIDE),
        lambda: Add(ZkfFormat(4, 6), stage_core=Add.Core.NORMALIZE | Add.Core.NORMALIZE_WIDE),
        lambda: Mul(F18, wmultiplier=18, stage_core_mul=4),
        lambda: Mul(F18, wmultiplier=18, stage_core_mul=-1),
        lambda: Mul(F36, wmultiplier=18, stage_core_mul=1),
        lambda: Mul(F36, wmultiplier=18, stage_core_mul=5),
        lambda: Mul(ZkfFormat(11, 53), wmultiplier=18, stage_core_mul=2),
        lambda: Fma(F36, wmultiplier=18, stage_core_mul=1),
        lambda: Fma(F18, wmultiplier=18, stage_core_add=Add.Core.NORMALIZE_WIDE),
        lambda: Divsqrt(ZkfFormat(6, 17), stage_core=Divsqrt.Core.FIRST_DIGIT),
        lambda: Cordic(F18, wmultiplier=18, stage_core_mul=3, unroll100=75),
        lambda: Exp2(F18, wmultiplier=18),
        lambda: Exp2(F36, wmultiplier=18, stage_core_kernel=2),
        lambda: Cordic(F18, wmultiplier=18, stage_core_mul=3, mode=0, stage_core=Cordic.Core.MEET),
        lambda: Log2(F18, wmultiplier=18, stage_core_kernel=2),
        lambda: Pipe(w=0),
    )
    for build in knobs:
        with pytest.raises(ValueError) as refusal:
            build()
        assert not isinstance(refusal.value, UnsupportedFormat)
    formats: tuple[Callable[[], Operator], ...] = (
        lambda: Exp2(ZkfFormat(6, 15), wmultiplier=18),
        lambda: Log2(ZkfFormat(20, 18), wmultiplier=18),
        lambda: Cordic(ZkfFormat(4, 18), wmultiplier=18, mode=1),
        lambda: MulIlog2(F18, wk=0),
        lambda: FromInt(F18, wint=1),
    )
    for build in formats:
        with pytest.raises(UnsupportedFormat):
            build()
    Cordic(ZkfFormat(4, 18), stage_core_mul=3, wmultiplier=18, mode=0)


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
    assert widths(Exp2(F18, wmultiplier=18, stage_core_kernel=2)) == unary
    log2 = Log2(F18, wmultiplier=18, stage_core_kernel=2, stage_core_mul=2)
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


def test_one_configuration_is_one_model() -> None:
    assert MulIlog2(F18) == MulIlog2(F18, wk=F18.wexp + 1)
    assert len({MulIlog2(F18), MulIlog2(F18, wk=F18.wexp + 1), MulIlog2(F18, wk=5)}) == 2
    assert Add(F18) != Add(F36) and Add(F18) != Add(F18, stage_input=1) and Pipe(w=5, n=1) == Pipe(n=1, w=5)
    block = {"rom_style": "block"}
    table = Exp2(F18, wmultiplier=18, stage_core_kernel=2, rom_attrs=block)
    plain = Exp2(F18, wmultiplier=18, stage_core_kernel=2)
    assert table == plain and table.name == plain.name
    assert replace(table, stage_output=1).rom_attrs == block and "rom_attrs" not in table.config
    keep = {"keep": "true"}
    for model in (Mul(F36, stage_core_mul=3, wmultiplier=18), Cordic(F18, stage_core_mul=3, wmultiplier=18), table):
        kept = replace(model, mul_attrs=keep)
        assert kept == model and kept.name == model.name and "mul_attrs" not in kept.config


def test_clocked() -> None:
    assert not AbsNeg(F18).clocked and not Pipe(w=1).clocked and Pipe(w=1, n=1).clocked
    assert Mul(F18, stage_core_mul=1, wmultiplier=18).clocked and Cordic(F18, stage_core_mul=3, wmultiplier=18).clocked
    assert not Mul(F18, wmultiplier=18).clocked and not Add(F18).clocked


def test_a_default_follows_the_format() -> None:
    """A knob left to its default is derived again when the model is copied to another format; a given one stays."""
    assert replace(MulIlog2(F18), fmt=F36) == MulIlog2(F36)
    assert replace(MulIlog2(F18, wk=5), fmt=F36) == MulIlog2(F36, wk=5)

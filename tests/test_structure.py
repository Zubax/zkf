"""Every operator and private core, elaborated over its knobs, and what an operator compiles to, which linters take."""

import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from functools import partial
from itertools import islice, product
from pathlib import Path

import pytest
from amaranth.back import rtlil, verilog
from amaranth.lib import data, wiring

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
    Pipe,
    Resize,
    Rint,
    ZkfFormat,
)
from zkf._shift import Normshift
from zkf._pack import Pack
from zkf._cordic import _LACKED
from zkf._pmul import Pmul, positions
from zkf._trans import Kernel, spec as table
from zkf._txn import Txn

# The formats on which an operator's structure differs: per multiplier width, no split, an even one, an uneven
# one and a 3x3 one for a multiplier; a `k` narrower and wider than the exponent for the shift; for the adder a
# normalizer of two and three levels, and an exponent too narrow to reach the aligner's last level; an integer
# narrower and wider than the significand; a format that holds the other, and one narrower in either dimension;
# an integer of one magnitude bit, narrower than the significand and wider; for the divider an even significand,
# whose first digit folds into the decoding, an odd one, and one whose frame is wider than 32 bits; for a table
# its degrees 2 to 5, and the narrowest and the widest exponent; for `cordic` each mode alone, an odd significand,
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
    (Divsqrt, ZkfFormat(3, 4), {}),
    (Divsqrt, ZkfFormat(3, 5), {}),
    (Divsqrt, ZkfFormat(6, 18), {}),
    (Divsqrt, ZkfFormat(8, 27), {}),
    (Divsqrt, ZkfFormat(11, 53), {}),
    (Exp2, ZkfFormat(2, 16), {"wmultiplier": 18}),
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
    """Every configuration of the sweep that the format admits."""
    for knobs in configurations(cls, every_pair=every_pair):
        try:
            yield make(cls, fmt, **(knobs | fixed))
        except ValueError:
            continue


def cores() -> Iterator[Callable[[], wiring.Component]]:
    side = data.StructLayout({"tag": 3, "flag": 1})
    for stage_output in (0, 1):
        yield partial(Txn, side, stage_output)
    for fmt, unbiased in ((ZkfFormat(3, 4), 5), (ZkfFormat(4, 5), 4), (ZkfFormat(8, 36), 12)):
        for options, stages in product(range(8), range(4)):
            if not (options & 1 and unbiased == fmt.wexp):
                yield partial(
                    Pack,
                    fmt,
                    wexp_unbiased=unbiased,
                    exp_is_biased=bool(options & 1),
                    assume_no_overflow=bool(options & 2),
                    saturate_round_carry=bool(options & 4),
                    stage_input=stages & 1,
                    stage_output=stages >> 1,
                )
    # Signed 53x18: a pair level without row sums.
    sizes = ((66, 41, 18), (29, 20, 11), (18, 18, 18), (53, 53, 8), (36, 36, 18), (12, 12, 5), (53, 18, 18))
    for (wa, wb, wmultiplier), signed, sideband in product(sizes, (False, True), (None, side)):
        for stages in range(len(positions(wa, wb, signed=signed, wmultiplier=wmultiplier)) + 1):
            yield partial(
                Pmul,
                wa,
                wb,
                signed=signed,
                wmultiplier=wmultiplier,
                stages=stages,
                sideband=sideband,
            )


@pytest.mark.parametrize("shard", range(4))
@pytest.mark.parametrize("row", MODELS, ids=lambda row: f"{row[0].operator}-{row[1].wexp}/{row[1].wman}-{row[2]}")
def test_operators(row: tuple[type[Block], ZkfFormat, dict[str, object]], shard: int) -> None:
    built = list(islice(dict.fromkeys(models(*row)), shard, None, 4))
    assert built or shard, "the format admits no configuration"
    for model in built:
        rtlil.convert(model.component())


def test_cores() -> None:
    for build in cores():
        rtlil.convert(build())


LINTERS = {
    "iverilog": ["iverilog", "-g2001", "-o", "/dev/null"],
    # Yosys prints operands of unequal widths, a `case` with no default where some values are not selected, and a
    # comparison with a constant that the operand cannot reach.
    "verilator": ["verilator", "--lint-only", "-Wno-WIDTH", "-Wno-CASEINCOMPLETE", "-Wno-CMPCONST"],
}


@pytest.mark.parametrize("linter", LINTERS)
def test_lint(linter: str, tmp_path: Path) -> None:
    """The shipped text is Verilog-2001 to the tools a consumer may read it with."""
    if shutil.which(linter) is None:
        if os.environ.get("CI"):
            pytest.fail(f"no {linter} in CI")
        pytest.skip(f"no {linter}")
    side = data.StructLayout({"tag": 3, "flag": 1})
    convert = partial(verilog.convert, strip_internal_attrs=True)
    texts = [
        convert(Txn(side, 0), name="txn"),
        convert(Normshift(39, stage_split=1, stage_wide=1, sideband=side), name="normshift"),
    ]
    for function in ("exp2", "log2"):
        kernel = Kernel(table(function, 36, 18), wmultiplier=18, stages=4)
        texts.append(convert(kernel, name=f"kernel_{function}"))
    for cls, fmt, fixed in MODELS:
        swept = list(models(cls, fmt, fixed, every_pair=False))
        texts += [model.compile() for model in {swept[at].name: swept[at] for at in (0, len(swept) // 2, -1)}.values()]
    # What the ends of its sweep do not have: half a rate, where a sign is replayed, and one iteration a cycle, where
    # the angle's product starts ahead.
    for mode, rate in product((0, 1, 2), (50, 100)):
        every = ~_LACKED[mode]
        staged = make(
            Cordic, ZkfFormat(5, 17), wmultiplier=18, mode=mode, unroll100=rate, stage_core_mul=3, stage_core=every
        )
        texts.append(staged.compile())
    for index, text in enumerate(texts):
        path = tmp_path / f"blob{index}.v"
        path.write_text(text)
        done = subprocess.run([*LINTERS[linter], str(path)], capture_output=True, text=True, cwd=tmp_path)
        assert done.returncode == 0 and not done.stderr, f"{text.splitlines()[0]}\n{done.stdout}{done.stderr}"

import numpy as np
import pytest
from amaranth.hdl import Module
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from _bench import Case, run_pipeline
from _operands import SEED
from zkf._block import pipe
from zkf._shift import RshiftSticky


class Bench(wiring.Component):
    """The shifter between ports, beside a valid flag of its latency."""

    def __init__(self, width: int, wshift: int, stage_split: int) -> None:
        self.shifter = RshiftSticky(width, wshift=wshift, stage_split=stage_split)
        self.stage_split = stage_split
        ports = {"x": In(width), "shamt": In(len(self.shifter.shamt)), "y": Out(width)}
        super().__init__({"in_valid": In(1), "out_valid": Out(1), **ports})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        m.submodules.shifter = shifter = self.shifter
        m.d.comb += [
            shifter.x.eq(self.x),
            shifter.shamt.eq(self.shamt),
            self.y.eq(shifter.y),
            self.out_valid.eq(pipe(m, "stage", self.in_valid, 1, self.in_valid, self.stage_split)[0]),
        ]
        return m


def reference(x: int, shamt: int) -> int:
    return (x >> shamt) | int(x & ((1 << shamt) - 1) != 0)


def run(width: int, stage_split: int, wshift: int, count: int = 0) -> None:
    """`count` 0 is every input; otherwise every amount on every single bit and run of ones, and random inputs."""
    dut = Bench(width, wshift, stage_split)
    amounts = range(1 << len(dut.shamt))
    if count:
        rng = np.random.default_rng(SEED)
        xs = {0, ((1 << width) - 1)} | {1 << i for i in range(width)} | {((1 << i) - 1) for i in range(1, width)}
        near = [s for s in amounts if s <= width + 1 or s == amounts[-1] or s & (s - 1) == 0]
        pairs = [(x, s) for x in sorted(xs) for s in near]
        while len(pairs) < count:
            high = int(rng.integers(0, 1 << max(1, width - 62))) << 62
            pairs.append(((high | int(rng.integers(0, 1 << 62))) & ((1 << width) - 1), int(rng.choice(amounts))))
    else:
        pairs = [(x, s) for x in range(1 << width) for s in amounts]
    run_pipeline(dut, stage_split, [Case({"x": x, "shamt": s}, {"y": reference(x, s)}) for x, s in pairs])


# Every width from the narrowest through the first with three levels; an amount narrower than the cascade takes, as
# wide, and wider, which is what saturates.
@pytest.mark.parametrize("width", [2, 3, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("stage_split", [0, 1])
def test_every_input(width: int, stage_split: int) -> None:
    for wshift in (1, 2, 3, 4, 6):
        run(width, stage_split, wshift)
    run(width, stage_split, ceil_log2(width) + 1)


@pytest.mark.parametrize("width", [16, 17, 21, 27, 39, 56, 64, 65, 75, 109, 130])
@pytest.mark.parametrize("stage_split", [0, 1])
def test_wide(width: int, stage_split: int) -> None:
    run(width, stage_split, ceil_log2(width) + 1, count=3000)
    for wshift in (3, 5, 11):
        run(width, stage_split, wshift, count=1500)

import pytest
from amaranth.hdl import Module, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.sim import Simulator, SimulatorContext

from zkf import Zkf, ZkfFormat
from zkf._block import FloatLayout, FloatView

FMT = ZkfFormat(4, 6)


class Probe(wiring.Component):
    x: FloatView
    y: FloatView
    zero: Signal
    inf: Signal
    significand: Signal

    def __init__(self) -> None:
        layout = FloatLayout(FMT)
        super().__init__({"x": In(layout), "y": Out(layout), "zero": Out(1), "inf": Out(1), "significand": Out(6)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        assert self.x.fmt == FMT
        m.d.comb += [
            self.y.eq(self.x),
            self.zero.eq(self.x.is_zero),
            self.inf.eq(self.x.is_inf),
            self.significand.eq(self.x.significand),
        ]
        return m


def test_ports_exchange_zkf() -> None:
    dut = Probe()

    async def bench(ctx: SimulatorContext) -> None:
        for bits in range(1 << FMT.wfull):
            x = Zkf(FMT, bits)
            ctx.set(dut.x, x)
            assert ctx.get(dut.y) == x
            assert (ctx.get(dut.zero), ctx.get(dut.inf)) == (x.is_zero, x.is_inf)
            assert ctx.get(dut.significand) == (1 << FMT.wfrac) | x.frac

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_a_constant_is_a_value_of_the_format() -> None:
    with pytest.raises(ValueError, match="is not"):
        FloatLayout(FMT).const(Zkf(ZkfFormat(5, 5), 0))
    with pytest.raises(TypeError):
        FloatLayout(FMT).const(3)

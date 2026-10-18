from dataclasses import dataclass

from amaranth.hdl import Module, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ._block import Block, Timing, pipe


@dataclass(frozen=True, kw_only=True, eq=False)
class Pipe(Block):
    """`y` is `x`, of `w` bits, delayed by `n` register stages beside a valid flag; reset clears only the flag."""

    operator = "pipe"
    w: int
    n: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.w < 1:
            raise ValueError(f"w={self.w} is below 1")
        if self.n < 0:
            raise ValueError(f"n={self.n} is negative")

    @property
    def timing(self) -> Timing:
        return Timing(self.n, 1)

    @property
    def signature(self) -> wiring.Signature:
        return wiring.Signature({"in_valid": In(1), "x": In(self.w), "out_valid": Out(1), "y": Out(self.w)})

    def component(self) -> _Pipe:
        return _Pipe(self)

    def evaluate(self, x: int) -> int:
        if not 0 <= x < 1 << self.w:
            raise ValueError(f"x={x} does not fit {self.w} bits")
        return x


class _Pipe(wiring.Component):
    in_valid: Signal
    x: Signal
    out_valid: Signal
    y: Signal

    def __init__(self, model: Pipe) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        valid, payload = pipe(m, "stage", self.in_valid, len(self.x), self.x, self._model.n)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.y.eq(payload)
        return m

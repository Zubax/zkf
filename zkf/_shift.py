"""The normalizing shifters and the sticky right shifter."""

from amaranth.hdl import Cat, Const, Module, Mux, ShapeLike, Signal, Value
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import bundle, delay, pipe


class Normshift(wiring.Component):
    """
    `y = x << count`, `count` being the number of leading zeros of `x`; `zero = x == 0`, with `count` and `y`
    meaningless then; `sticky`: whether `y[:wsticky]` has a set bit. `sideband` is delayed with `y`.

    A tree counts the leading zeros, whose base-4 digits select a radix-4 shift, largest first; the sticky is found from
    `x` beside the shift. `most`, where given, is the most leading zeros of a nonzero `x`, and only the tree and the
    levels that reach it are built.

    `stage_input` registers `x`, `stage_split` `x` with its count ahead of the shift, and `stage_wide` the shift behind
    its widest level.
    """

    in_valid: Signal
    x: Signal
    out_valid: Signal
    zero: Signal
    count: Signal
    y: Signal
    sticky: Signal

    def __init__(
        self,
        width: int,
        *,
        wsticky: int = 0,
        stage_input: int = 0,
        stage_split: int = 0,
        stage_wide: int = 0,
        sideband: ShapeLike,
        most: int | None = None,
    ) -> None:
        self._width, self._wsticky = width, wsticky
        # The tree spans the top bits that can hold the leading one, as a power of two.
        self._span = 1 << ceil_log2(max(width if most is None else min(most + 1, width), 2))
        self._stage_input, self._stage_split, self._stage_wide = stage_input, stage_split, stage_wide
        self._sideband = sideband
        super().__init__(
            {
                "in_valid": In(1),
                "x": In(width),
                "out_valid": Out(1),
                "zero": Out(1),
                "count": Out(ceil_log2(self._span)),
                "y": Out(width),
                "sticky": Out(1),
                "sb_in": In(sideband),
                "sb_out": Out(sideband),
            }
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        width, span, wsticky = self._width, self._span, self._wsticky
        taken = data.StructLayout({"x": width, "sb": self._sideband})
        valid, given = pipe(m, "in", self.in_valid, taken, bundle(taken, x=self.x, sb=self.sb_in), self._stage_input)
        x = Signal(width, name="x_in")
        m.d.comb += x.eq(given.x)

        zero, count = _leading_zeros(m, Cat(Const(0, max(span - width, 0)), x[max(width - span, 0) :]), "lz")
        counted = data.StructLayout({"x": width, "count": len(count), "zero": 1, "sb": self._sideband})
        found = bundle(counted, x=x, count=count, zero=zero, sb=given.sb)
        valid, held = pipe(m, "counted", valid, counted, found, self._stage_split)
        x, count = Signal(width, name="x_counted"), Signal(len(count), name="count_counted")
        m.d.comb += [x.eq(held.x), count.eq(held.count)]

        zero, sb, sticky = held.zero, held.sb, None
        shifted: Value = x
        top = (len(count) + 1) // 2 - 1
        for k in reversed(range(top + 1)):
            digit = count[2 * k : 2 * k + 2]
            level = Signal(width, name=f"level{k}")
            with m.Switch(digit):
                for d in range(1 << len(digit)):
                    with m.Case(d):
                        m.d.comb += level.eq(shifted << (d << (2 * k)))
            shifted = level
            if k == top and self._stage_wide:
                behind = data.StructLayout(
                    {"y": width, "count": len(count), "zero": 1, "sticky": 1, "sb": self._sideband}
                )
                found = bundle(behind, y=shifted, count=count, zero=zero, sticky=_sticky(m, x, count, wsticky), sb=sb)
                valid, staged = pipe(m, "shifted", valid, behind, found, 1)
                shifted, count, zero, sticky, sb = staged.y, staged.count, staged.zero, staged.sticky, staged.sb

        if sticky is None:
            sticky = _sticky(m, x, count, wsticky)

        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.sb_out.eq(sb)
        m.d.comb += self.zero.eq(zero)
        m.d.comb += self.count.eq(count)
        m.d.comb += self.y.eq(shifted)
        m.d.comb += self.sticky.eq(sticky)
        return m


class RshiftSticky(wiring.Component):
    """
    `y = x >> shamt` with the dropped bits ORed into `y[0]`; a `shamt` of the width or more leaves only that sticky bit,
    whatever `wshift`, the width of `shamt`, is.

    A radix-4 cascade, smallest shift first, of one 4:1 multiplexer a level. `stage_split=1` puts a register in its
    middle.
    """

    x: Signal
    shamt: Signal
    y: Signal

    def __init__(self, width: int, *, wshift: int, stage_split: int) -> None:
        self._width = width
        self._wshift = wshift
        self._stage_split = stage_split
        super().__init__({"x": In(width), "shamt": In(self._wshift), "y": Out(width)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        width, wshift, split = self._width, self._wshift, self._stage_split
        levels = _levels(width)
        middle = levels // 2

        data: Value = self.x
        sticky: Value | None = None
        for i in range(levels):
            bits = self.shamt[2 * i : min(2 * i + 2, wshift)] if 2 * i < wshift else None
            if bits is not None:
                # A level behind the register reads its part of the amount a cycle later.
                late = split and i > middle
                sel = Signal(len(bits), name=f"sel{i}", reset_less=True)
                m.d["sync" if late else "comb"] += sel.eq(bits)
                m.submodules[f"level{i}"] = level = _RightLevel(width, len(bits), 1 << (2 * i))
                m.d.comb += [level.data.eq(data), level.sel.eq(sel)]
                if sticky is not None:
                    m.d.comb += level.sticky_in.eq(sticky)
                data, sticky = level.out, level.sticky
            if split and i == middle:
                assert sticky is not None
                data_r, sticky_r = Signal(width, name="data", reset_less=True), Signal(name="sticky", reset_less=True)
                m.d.sync += [data_r.eq(data), sticky_r.eq(sticky)]
                data, sticky = data_r, sticky_r
        assert sticky is not None

        if wshift <= 2 * levels:
            m.d.comb += self.y.eq(Cat(data[0] | sticky, data[1:]))
            return m
        # An amount with a bit above those the cascade takes shifts everything out.
        over = Signal(name="over", reset_less=True)
        m.d["sync" if split else "comb"] += over.eq(self.shamt[2 * levels :].any())
        low = Mux(over, delay(m, "any", 1, self.x.any(), split), data[0] | sticky)
        m.d.comb += self.y.eq(Cat(low, Mux(over, Const(0, width - 1), data[1:])))
        return m


def normshift_reference(width: int, value: int) -> tuple[int, int, int]:
    """`(zero, count, y)` that `Normshift` gives; `(1, 0, 0)` for a zero, where its `count` and `y` mean nothing."""
    value &= (1 << width) - 1
    if value == 0:
        return 1, 0, 0
    count = width - value.bit_length()
    return 0, count, value << count


class _RightLevel(wiring.Component):
    """
    `out = data >> sel * unit`, `sticky = sticky_in | (a set bit is dropped)`; `sel` is one bit wide where the amount
    ends within this level.
    """

    data: Signal
    sticky_in: Signal
    sel: Signal
    out: Signal
    sticky: Signal

    def __init__(self, width: int, wsel: int, unit: int) -> None:
        self._unit = unit
        ports = {"data": In(width), "sticky_in": In(1), "sel": In(wsel), "out": Out(width), "sticky": Out(1)}
        super().__init__(ports)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        data, sel, width = self.data, self.sel, len(self.data)
        by = [min(n * self._unit, width) for n in (1, 2, 3)]
        shifted = [Cat(data[n:], Const(0, n)) if n < width else Const(0, width) for n in by]
        # Shifts of the whole width or more drop the same bits and share a detector.
        lost = {n: Signal(name=f"lost{n}") for n in sorted(set(by))}
        m.d.comb += [dropped.eq(data[:n].any()) for n, dropped in lost.items()]
        dropped = Signal(name="dropped")
        with m.Switch(sel):
            with m.Case(0):
                m.d.comb += self.out.eq(data)
            for n in range(1, 1 << len(sel)):
                with m.Case(n):
                    m.d.comb += [self.out.eq(shifted[n - 1]), dropped.eq(lost[by[n - 1]])]
        m.d.comb += self.sticky.eq(self.sticky_in | dropped)
        return m


def _sticky(m: Module, x: Value, count: Value, wsticky: int) -> Value:
    """Whether `(x << count)[:wsticky]` has a set bit."""
    if not wsticky:
        return Const(0, 1)
    # A constant shifted by the count: compares of the count per bit map to carry chains.
    mask = Signal(wsticky, name="mask")
    m.d.comb += mask.eq(Const((1 << wsticky) - 1, wsticky) >> count)
    return (x[:wsticky] & mask).any()


def _levels(width: int) -> int:
    return (ceil_log2(width) + 1) // 2


def _leading_zeros(m: Module, v: Value, name: str) -> tuple[Value, Value]:
    """
    `(v == 0, leading zeros of v)` for a `v` of a power-of-two width, by halves. Each node is a signal: its parent
    reads it twice, which the netlist would expand otherwise.
    """
    if len(v) == 1:
        return ~v[0], Const(0, 0)
    half = len(v) // 2
    high, high_count = _leading_zeros(m, v[half:], f"{name}h")
    low, low_count = _leading_zeros(m, v[:half], f"{name}l")
    zero, count = Signal(name=f"{name}_zero"), Signal(len(high_count) + 1, name=f"{name}_count")
    m.d.comb += zero.eq(high & low)
    m.d.comb += count.eq(Cat(Mux(high, low_count, high_count), high) if half > 1 else high)
    return zero, count

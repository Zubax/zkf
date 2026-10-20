"""The normalizing shifter and the sticky right shifter."""

from amaranth.hdl import Cat, Const, Module, Mux, ShapeLike, Signal, Value
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import delay, pipe


def split_points(width: int) -> int:
    """How many registers `stage_split` of `Normshift` can place at `width`: two from three levels on, else one."""
    return 2 if _levels(width) >= 3 else 1


class Normshift(wiring.Component):
    """
    `y = x << count`, `count` being the number of leading zeros of `x`; `zero = x == 0`, with `count` and `y`
    meaningless then. `sideband` is delayed with `y`.

    A radix-4 cascade, largest shift first, at half the depth of radix 2: the level of weight `4**k` shifts by as many
    of the top one, two and three groups of `4**k` bits as are zero, which is the base-4 digit k of `count`.

    `stage_split=1` puts a register in the middle of the cascade; 2 adds one right behind the widest level, and needs
    three built levels (unbounded, a width of 17 or more). `stage_input=1` registers `x` with the zero-detects of the
    widest level, found ahead of the register, so that the shift of that level, whose select has the widest fan-out,
    starts at registers. `most`, where given, is the most leading zeros of a nonzero `x`, and only the levels that reach
    it are built.
    """

    in_valid: Signal
    x: Signal
    out_valid: Signal
    zero: Signal
    count: Signal
    y: Signal

    def __init__(
        self, width: int, *, stage_input: int = 0, stage_split: int = 0, sideband: ShapeLike, most: int | None = None
    ) -> None:
        self._width = width
        self._levels = _levels(width) if most is None else min(_levels(width), max(_levels(most + 1), 1))
        assert stage_split < 2 or self._levels >= 3, (width, most, stage_split)
        self._stage_input = stage_input
        self._stage_split = stage_split
        self._sideband = sideband
        super().__init__(
            {
                "in_valid": In(1),
                "x": In(width),
                "out_valid": Out(1),
                "zero": Out(1),
                "count": Out(ceil_log2(width)),
                "y": Out(width),
                "sb_in": In(sideband),
                "sb_out": Out(sideband),
            }
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        levels, split, depth = self._levels, self._stage_split, self._stage_input + self._stage_split
        middle = (levels - 1) // 2 + 1
        barriers = ({middle} if split else set()) | ({1} if split >= 2 else set())

        data = self.x
        digits: list[Value] = []
        for s in range(levels):
            staged = bool(self._stage_input) and not s
            m.submodules[f"level{s}"] = level = _NormLevel(self._width, 1 << (2 * (levels - 1 - s)), staged)
            m.d.comb += level.data.eq(data)
            data = level.out
            if s + 1 in barriers:
                data = Signal(self._width, name=f"data{s + 1}", reset_less=True)
                m.d.sync += data.eq(level.out)
            # A digit waits for the barriers its level is ahead of.
            digits.insert(0, delay(m, f"digit{s}_", 2, level.count, sum(barrier > s for barrier in barriers)))

        zero = Signal(name="zero_in")
        m.d.comb += zero.eq(~self.x.any())
        valid, side = pipe(m, "side", self.in_valid, self._sideband, self.sb_in, depth)
        m.d.comb += self.sb_out.eq(side)
        zero = delay(m, "zero", 1, zero, depth)
        m.d.comb += self.out_valid.eq(valid)
        m.d.comb += self.zero.eq(zero)
        m.d.comb += self.count.eq(Cat(*digits)[: len(self.count)])
        m.d.comb += self.y.eq(data)
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
    count = (width - 1) - (value.bit_length() - 1)
    return 0, count, (value << count) & ((1 << width) - 1)


class _NormLevel(wiring.Component):
    """
    `count` is how many of the leading `group`-bit groups of `data` are zero, 0 to 3, and `out = data << count * group`.
    A group reaching past the width is zero only for a zero input, whose count is not read, so it is not examined, which
    keeps the widest levels shallow. `staged` registers `data` and its zero-detects ahead of the shift.
    """

    data: Signal
    count: Signal
    out: Signal

    def __init__(self, width: int, group: int, staged: bool = False) -> None:
        self._group, self._staged = group, staged
        super().__init__({"data": In(width), "count": Out(2), "out": Out(width)})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        data, group, width = self.data, self._group, len(self.data)
        zeros: list[Value] = []
        for n in (1, 2, 3):
            if n * group < width:
                zero = Signal(name=f"z{n}")
                m.d.comb += zero.eq(~data[width - n * group :].any())
                zeros.append(zero)
            else:
                zeros.append(Const(0, 1))
        if self._staged:
            for n in (1, 2, 3):
                if n * group < width:
                    found = Signal(name=f"zr{n}", reset_less=True)
                    m.d.sync += found.eq(zeros[n - 1])
                    zeros[n - 1] = found
            held = Signal(width, name="held", reset_less=True)
            m.d.sync += held.eq(data)
            data = held
        shifted = [
            Cat(Const(0, n * group), data[: width - n * group]) if n * group < width else Const(0, width)
            for n in (1, 2, 3)
        ]
        count = Mux(zeros[0], Mux(zeros[1], Mux(zeros[2], Const(3, 2), Const(2, 2)), Const(1, 2)), Const(0, 2))
        m.d.comb += self.count.eq(count)
        # Selected by the zero-detects in priority order, not by the count, which keeps the select one level deep.
        m.d.comb += self.out.eq(Mux(zeros[2], shifted[2], Mux(zeros[1], shifted[1], Mux(zeros[0], shifted[0], data))))
        return m


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


def _levels(width: int) -> int:
    return (ceil_log2(width) + 1) // 2

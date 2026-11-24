import re
from collections.abc import Collection
from functools import cache, reduce
from operator import add, or_

from amaranth.hdl import Cat, Const, Module, Shape, ShapeLike, Signal, Value
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._block import Attrs, pipe, zext

_RUNGS = ("O", "G", "S", "GO", "CS", "CO", "GG2", "GS", "GSO", "CGO", "CGS", "CGSO", "CGG2SO")
"""The registers of levels 1 to 13, as `Mul.stage_core_mul_level` lists them."""
_NAMES = {"C": "capture", "G": "grid", "G2": "grid2", "O": "output"}


class Pmul(wiring.Component):
    """
    Integer `p = a * b`, exact, as `wa + wb` raw bits (two's complement where `signed`);
    the operands are both signed or both unsigned.
    `sideband` is the shape of an optional payload, `sb_in` delayed with `p` to `sb_out`.

    The operands are cut into slices that fit `wmultiplier`-bit multipliers, one row of the grid per slice of `a`,
    and the products are summed: unsigned ones by the diagonals of the grid, signed ones by rows, then the rows.
    `level` takes the registers as `Mul.stage_core_mul_level` does; `latency` follows from them.
    `attrs` go as `Mul.mul_attrs` do.

    This component is used to efficiently implement more complex floating-point arithmetic that uses integer
    multiplication under the hood. It is exposed as a public member because it is often useful in third-party circuits.
    """

    in_valid: Signal
    a: Signal
    b: Signal
    out_valid: Signal
    p: Signal

    def __init__(
        self,
        wa: int,
        wb: int,
        *,
        signed: bool,
        wmultiplier: int,
        level: int,
        sideband: ShapeLike | None = None,
        attrs: Attrs | None = None,
    ) -> None:
        self._ga, self._gb = slices(wa, wb, signed=signed, wmultiplier=wmultiplier)
        self._attrs = dict(attrs or {})
        self._wa, self._wb, self._signed, self._sideband = wa, wb, signed, sideband
        self._registers = registers(level, wa, wb, signed=signed, wmultiplier=wmultiplier)
        members = {"in_valid": In(1), "a": In(wa), "b": In(wb), "out_valid": Out(1), "p": Out(wa + wb)}
        if sideband is not None:
            members |= {"sb_in": In(sideband), "sb_out": Out(sideband)}
        super().__init__(members)

    @property
    def latency(self) -> int:
        return len(self._registers)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wp, ga, gb, at = self._wa + self._wb, self._ga, self._gb, self._registers
        cuts = sorted(int(name[3:]) - 1 for name in at if name.startswith("sum"))
        if self._sideband is None:
            valid = self.in_valid
            for k in range(self.latency):
                stage = Signal(name=f"side{k}_valid")
                m.d.sync += stage.eq(valid)
                valid = stage
        else:
            valid, side = pipe(m, "side", self.in_valid, self._sideband, self.sb_in, self.latency)
            m.d.comb += self.sb_out.eq(side)
        m.d.comb += self.out_valid.eq(valid)

        c_a, c_b = self.a, self.b
        if "capture" in at:
            c_a = Signal(self._wa, name="c_a", reset_less=True, attrs=self._attrs)
            c_b = Signal(self._wb, name="c_b", reset_less=True, attrs=self._attrs)
            m.d.sync += [c_a.eq(self.a), c_b.eq(self.b)]
        slices_a, slices_b = self._slices(self._wa, ga), self._slices(self._wb, gb)

        def factors(operand: Signal, slices: list[tuple[int, int, int]]) -> list[Value]:
            out = []
            for offset, bits, factor in slices:
                piece = operand[offset : offset + bits]
                if self._signed:
                    piece = zext(piece, factor).as_signed()
                out.append(piece)
            return out

        a_s, b_s = factors(c_a, slices_a), factors(c_b, slices_b)

        low = [[slices_a[i][0] + slices_b[j][0] for j in range(gb)] for i in range(ga)]
        high = [[low[i][j] + slices_a[i][2] + slices_b[j][2] for j in range(gb)] for i in range(ga)]
        pp: list[list[Signal]] = []
        for i in range(ga):
            pp.append([])
            for j in range(gb):
                bits = high[i][j] - low[i][j]
                shape = Shape(bits, signed=self._signed)
                q = Signal(shape, name=f"m{i}_{j}", reset_less=True)
                m.d["sync" if "grid" in at else "comb"] += q.eq(a_s[i] * b_s[j])
                if "grid2" in at:
                    q2 = Signal(shape, name=f"m{i}_{j}_r2", reset_less=True)
                    m.d.sync += q2.eq(q)
                    q = q2
                above = (q[-1] if self._signed else Const(0, 1)).replicate(max(0, wp - high[i][j]))
                placed = Signal(wp, name=f"pp{i}_{j}")
                m.d.comb += placed.eq(Cat(Const(0, low[i][j]), q, above))
                pp[i].append(placed)

        if not self._signed:
            # Unsigned products on one diagonal of the grid abut, so a diagonal is one summand by concatenation. The
            # diagonals are ordered by their lowest bit, the main one first; at a tie, the one of the higher index.
            def diagonal_low(d: int) -> int:
                return slices_a[d - (gb - 1)][0] if d > gb - 1 else slices_b[gb - 1 - d][0]

            def diagonal_high(d: int) -> int:
                return high[d][gb - 1] if d < ga else high[ga - 1][ga - 1 - d + gb - 1]

            order = sorted(range(ga + gb - 1), key=lambda d: (diagonal_low(d), d < gb - 1))
            operands = []
            for t, d in enumerate(order):
                abutting = [pp[i][i + gb - 1 - d] for i in range(ga) if 0 <= i + gb - 1 - d < gb]
                diagonal = Signal(wp, name=f"diagonal{t}")
                m.d.comb += diagonal.eq(reduce(or_, abutting))
                operands.append(diagonal)
            lows, highs = [diagonal_low(d) for d in order], [diagonal_high(d) for d in order]
            total = self._sum(m, "sum", operands, lows, highs, wp, cuts)
        else:
            # Sign-extended products overlap: they are summed by rows, then the rows.
            split = _levels(gb)
            rows = [
                self._sum(m, f"row{i}", pp[i], low[i], high[i], high[i][gb - 1], [c for c in cuts if c < split])
                for i in range(ga)
            ]
            lows, highs = [low[i][0] for i in range(ga)], [high[i][gb - 1] for i in range(ga)]
            total = self._sum(m, "sum", rows, lows, highs, wp, [c - split for c in cuts if c >= split])
        if "output" in at:
            p_r = Signal(wp, name="p_r0", reset_less=True)
            m.d.sync += p_r.eq(total)
            total = p_r
        m.d.comb += self.p.eq(total)
        return m

    def _slices(self, width: int, count: int) -> list[tuple[int, int, int]]:
        """The offset, width and factor width of each slice; the low slices take the remainder of an uneven split."""
        out, offset = [], 0
        for k in range(count):
            bits = (width + count - 1 - k) // count
            out.append((offset, bits, bits + (self._signed and k != count - 1)))
            offset += bits
        return out

    def _sum(
        self,
        m: Module,
        name: str,
        operands: list[Signal],
        low: list[int],
        high: list[int],
        top: int,
        registered: Collection[int],
    ) -> Signal:
        """
        The sum of `wp`-bit operands, operand k having bits `low[k]` to `high[k]` only, sign-extended above if signed;
        the sum fits below bit `top`. While more than three remain, neighbors are added in pairs, level by level; the
        rest are added at once, the last level. The bits below all operands but the lowest pass through, and no adder is
        wider than its operands' extents, so that the structure does not depend on what a tool infers. The levels in
        `registered`, counted from 0, are registered, each node on the bits of its extent only.
        """
        count, wp = len(operands), self._wa + self._wb
        assert all(low[k] > 0 and low[k] >= low[k - 1] for k in range(1, count)), low

        def stored(label: str, bits: Value, at: int) -> Signal:
            """The operand whose bits from `at` are `bits` registered, zero below and extended above."""
            latched = Signal(len(bits), name=f"{label}_q", reset_less=True)
            m.d.sync += latched.eq(bits)
            above = (latched[-1] if self._signed else Const(0, 1)).replicate(wp - at - len(bits))
            whole = Signal(wp, name=label)
            m.d.comb += whole.eq(Cat(Const(0, at), latched, above))
            return whole

        def joined(label: str, terms: list[tuple[Signal, int, int]], to: int, cut: bool) -> tuple[Signal, int, int]:
            """The sum of `terms`, below bit `to`; the bits below all but the lowest are those of the lowest."""
            (lowest, at, _), above = min(terms, key=lambda term: term[1]), sorted(term[1] for term in terms)[1]
            total = Signal(to - above, name=f"{label}_hi")
            m.d.comb += total.eq(reduce(add, (term[above:to] for term, _, _ in terms)))
            if cut:
                return stored(label, Cat(lowest[at:above], total), at), at, to
            whole = Signal(wp, name=label)
            m.d.comb += whole.eq(
                Cat(lowest[:above], total, (total[-1] if self._signed else Const(0, 1)).replicate(wp - to))
            )
            return whole, at, to

        nodes = [(operand, low[k], min(top, high[k])) for k, operand in enumerate(operands)]
        for level in range(_levels(count) - 1):
            cut, merged = level in registered, []
            for k in range(0, len(nodes), 2):
                label = f"{name}_l{level}_{k // 2}"
                if k + 1 < len(nodes):
                    pair = nodes[k : k + 2]
                    merged.append(joined(label, pair, min(top, max(hi for _, _, hi in pair) + 1), cut))
                elif cut:
                    operand, at, hi = nodes[k]
                    merged.append((stored(label, operand[at:hi], at), at, hi))
                else:
                    merged.append(nodes[k])
            nodes = merged
        if len(nodes) == 1:
            return nodes[0][0]
        return joined(f"{name}_last", nodes, top, _levels(count) - 1 in registered)[0]


class PmulWith(Pmul):
    """
    A product of the registers given, not those of a level: Cordic's, and a kernel's whose output follows its pair's
    sum.
    """

    def __init__(
        self,
        wa: int,
        wb: int,
        placed: frozenset[str],
        *,
        signed: bool,
        wmultiplier: int,
        sideband: ShapeLike | None = None,
        attrs: Attrs | None = None,
    ) -> None:
        assert placed <= levels(wa, wb, signed=signed, wmultiplier=wmultiplier)[-1], placed
        super().__init__(wa, wb, signed=signed, wmultiplier=wmultiplier, level=0, sideband=sideband, attrs=attrs)
        self._registers = placed


def slices(wa: int, wb: int, *, signed: bool, wmultiplier: int) -> tuple[int, int]:
    """
    How many slices `Pmul` cuts each operand into: the fewest of `wmultiplier` bits, or of `wmultiplier - 1` if signed
    since every slice but the top one is zero-extended; e.g. 66x41 gives 4x3 at 18.
    """
    if wmultiplier < 2:
        raise ValueError(f"wmultiplier={wmultiplier} is below 2")

    def count(width: int) -> int:
        return 1 if width <= wmultiplier else -(-width // (wmultiplier - signed))

    return count(wa), count(wb)


@cache
def levels(wa: int, wb: int, *, signed: bool, wmultiplier: int, cuts: int | None = None) -> tuple[frozenset[str], ...]:
    """
    The registers of `Pmul` at each level of `Mul.stage_core_mul_level`, from 0; with `cuts`, as if its sum had only
    that many levels to cut.
    """
    ga, gb = slices(wa, wb, signed=signed, wmultiplier=wmultiplier)
    sums = sum_cuts(wa, wb, signed=signed, wmultiplier=wmultiplier) if cuts is None else cuts
    out: list[frozenset[str]] = [frozenset()]
    for rung in _RUNGS:
        tokens = re.findall("G2|.", rung)
        built = frozenset(("sum1" if sums else "output") if token == "S" else _NAMES[token] for token in tokens)
        if ga == gb == 1:
            chain = len(built & {"grid", "grid2", "output"})
            built = (built - {"grid", "grid2", "output"}) | frozenset(("output", "grid", "grid2")[:chain])
        out.append(built)
    top = out[-1]
    for k in range(2, sums + 1):
        deeper = {f"sum{n}" for n in range(2, k + 1)}
        out += [top - {"grid2"} | deeper, top | deeper]
    return tuple(out)


def registers(
    level: int, wa: int, wb: int, *, signed: bool, wmultiplier: int, knob: str = "level", cuts: int | None = None
) -> frozenset[str]:
    """The registers of `levels` at the level, which it refuses outside their range, naming the knob that gave it."""
    out = levels(wa, wb, signed=signed, wmultiplier=wmultiplier, cuts=cuts)
    if not 0 <= level < len(out):
        raise ValueError(f"{knob}={level} is outside 0..{len(out) - 1}")
    return out[level]


def latency(level: int, wa: int, wb: int, *, signed: bool, wmultiplier: int, knob: str = "level") -> int:
    """The latency of `Pmul` at the level, refused as `registers` refuses it."""
    return len(registers(level, wa, wb, signed=signed, wmultiplier=wmultiplier, knob=knob))


def sum_cuts(wa: int, wb: int, *, signed: bool, wmultiplier: int) -> int:
    """The levels of the sum of `Pmul`'s grid that a register can cut: all but the last."""
    ga, gb = slices(wa, wb, signed=signed, wmultiplier=wmultiplier)
    return max(0, (_levels(gb) + _levels(ga) if signed else _levels(ga + gb - 1)) - 1)


def _levels(count: int) -> int:
    """The levels of `Pmul._sum` of `count` operands: pairs while more than three remain, then the rest at once."""
    return 0 if count == 1 else 1 + (ceil_log2((count + 2) // 3) if count > 3 else 0)

"""
The piecewise-polynomial kernels of `2**f` and `log2(1 + f) / f`: the table spec, its generation on demand, and the
component that evaluates it. The tests certify the tables.
"""

import functools
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil
from typing import Any

import mpmath
from amaranth.hdl import Cat, Const, Module, Mux, Signal, Value
from amaranth.lib import data, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

from ._block import Attrs, delay, pipe
from ._pack import pack_reference
from ._pmul import Pmul, counts, positions, slices
from ._value import UnsupportedFormat, Zkf, ZkfFormat

_ERR_GUARD = 8
"""A kernel is within `2**-(WMAN + _ERR_GUARD)` of its helper, relatively: `2**-_ERR_GUARD` ULP of a result."""
_GUARD = _ERR_GUARD + 4
"""Fraction bits of a coefficient below those of the significand."""
_K_CAP = 11
"""
The most bits that index a compact table's segment. The degree follows from it, and the length of the chain from the
degree.
"""
_K_FAST = 11
"""
The most bits that index a segment of `fast`'s table, four times the compact one's rows: one more would double the ROM
for a few kernel cycles less, and the argument into the table's address begins to bind the timing.
"""
_K_BASE = 9
"""The fewest bits that index a segment: one more doubles the table and takes a bit off every operand above `c0`."""
_SQUARE_GUARD = 3
"""Bits that a square of two multipliers carries below those taken from it."""
_SELECT = 0.9
"""The share of the budget that the estimate may take; the certificate proves the whole."""

# A context of its own, whose precision decides the coefficients: the caller's is neither read nor changed.
_fitting = mpmath.mp.clone()
_fitting.prec = 280
_GENERATING = threading.Lock()


@dataclass(frozen=True)
class Spec:
    """
    The table of `function` at one WMAN, and the arithmetic that evaluates it. The polynomial of a segment,
    `c0 +- c1*wn + c2*wn**2 +- ...` in `wn = w / 2**rw` at the scale `2**-cf`, is taken as `c0 +- c1*wn` beside the
    pairs `c[2i] +- c[2i+1]*wn` chained by `wn**2`, or by a square of fewer bits above `p2` (`tn`); degree 1 has no
    chain. Every product takes a fixed number of operand bits and is floored; the constants that center those floors,
    and the half ULP of `exp2`, are part of `c0`. The odd coefficients of `log2` subtract: a table holds magnitudes.
    """

    function: str
    wman: int
    k: int
    """Argument bits that select the segment; the `rw` below them are `w`."""
    rw: int
    cf: int
    d: int
    """The degree."""
    t: tuple[int, ...]
    """Bits from the top of `w` that multiply `c[2i+1]`."""
    t2: int
    """Bits from the top of `w` that are squared."""
    q: tuple[int, ...]
    """Bits of the square that each product by it takes, the outermost first."""
    tc: int = 0
    """Degree 2: the bits of `w` below `t[0]`, which multiply `c1 >> sc`: the wide product is then one multiplier's."""
    sc: int = 0
    z: tuple[int, ...] = ()
    """
    Of each coefficient from `c0`, the low bits that it lacks, all zero, so that its product spans a multiple of the
    multipliers' width.
    """
    th: int = 0
    """
    Degree 3, a square wider than a multiplier: its top `th` bits are squared and multiplied by the rest, the square of
    the rest is left out, and the sum carries `sg` bits below those taken.
    """
    sg: int = 0
    tn: int = 0
    """
    Degree 4 and up: the bits of `w` squared apart for the chain's products above `p2`, which take the top of that
    square instead of the full one; zero where they take the full one's.
    """
    thr: int = 0
    """`log2`: the first segment whose significand is re-centered."""
    coeffs: tuple[tuple[int, ...], ...] = ()
    """Of each segment, the low degree first."""

    def at(self, argument: int) -> int:
        """The kernel at an argument of `k + rw` bits."""
        return self._polynomial(self.coeffs[argument >> self.rw], argument & ((1 << self.rw) - 1))

    def _polynomial(self, row: Sequence[int], w: int) -> int:
        """The polynomial of the coefficients `row` at `w`, bit for bit what the hardware gives."""
        rw, t, q, t2 = self.rw, self.t, self.q, self.t2
        sign = -1 if self.function == "log2" else 1

        def odd(i: int) -> int:
            return (row[2 * i + 1] * (w >> (rw - t[i]))) >> t[i]

        linear = odd(0)
        if self.tc:
            linear += ((row[1] >> self.sc) * (w & ((1 << self.tc) - 1))) >> (rw - self.sc)
        if self.d == 1:
            return row[0] + sign * linear
        q2 = max(q)
        wt = w >> (rw - t2)
        if self.th:
            tl, base = t2 - self.th, 2 * t2 - q2 - self.sg
            wh, wl = wt >> tl, wt & ((1 << tl) - 1)
            w2 = (((wh * wh) >> (base - 2 * tl)) + ((wh * wl) >> (base - tl - 1))) >> self.sg
        else:
            w2 = (wt * wt) >> (2 * t2 - q2)
        m, tn = self.d // 2, self.tn
        value = row[2 * m] + (sign * odd(m) if 2 * m + 1 <= self.d else 0)
        for i in range(m - 1, 0, -1):
            taken = ((w >> (rw - tn)) ** 2) >> (2 * tn - q[i]) if tn else w2 >> (q2 - q[i])
            value = row[2 * i] + sign * odd(i) + ((value * taken) >> q[i])
        return row[0] + sign * linear + ((value * w2) >> q2)


def spec(function: str, wman: int, wmultiplier: int, fast: bool) -> Spec:
    """
    The table of `function` at a significand of `wman` bits, 10 to 53, for multipliers of `wmultiplier` bits,
    generated once: `_shortest`'s if `fast`, else the compact one.
    """
    if not 10 <= wman <= 53:
        raise UnsupportedFormat(f"no {function} table for WMAN={wman}: it is generated for 10 to 53")
    with _GENERATING:  # the fit has one context, and a flow compiles its specs from threads
        return _generated(function, wman, wmultiplier, fast)


def pack_fixed(fmt: ZkfFormat, sign: int, exp: int, v: int, point: int) -> Zkf:
    """
    The float nearest to `(-1)**sign * 2**exp * v / 2**point`, `v > 0`, a tie rounding away from zero, classified as
    `Pack` does.
    """
    drop = v.bit_length() - fmt.wman  # bits below the significand
    assert drop >= 1
    # With its round bit set the packer rounds up exactly when the guard bit is.
    return pack_reference(fmt, sign, 0, 0, exp + v.bit_length() - 1 - point, v >> drop, (v >> (drop - 1)) & 1, 1, 0)


def kernel_counts(spec: Spec, wmultiplier: int) -> range:
    """The counts that `Kernel` takes: from none, up to the most positions of one product."""
    return range(max(_caps(spec, wmultiplier).values()) + 1)


def kernel_check(stages: int, spec: Spec, wmultiplier: int) -> None:
    """Refuses, as `stage_core_kernel`, a count outside `kernel_counts`, saying what to set."""
    taken = kernel_counts(spec, wmultiplier)
    if stages not in taken:
        raise ValueError(
            f"stage_core_kernel={stages}: set {taken.start}..{taken.stop - 1} for the {spec.function} kernel at "
            f"WMAN={spec.wman} on {wmultiplier}-bit multipliers"
        )


def kernel_depth(spec: Spec, wmultiplier: int, stages: int) -> int:
    """Register stages of `Kernel` after the table's own."""
    return _schedule(spec, wmultiplier, stages).total


class Kernel(wiring.Component):
    """
    `y = spec.at(x)`, the table and the polynomial: the top `k` bits of the argument `x` select a row of coefficients,
    and the rest, `w`, is the variable. For `exp2` `x` is `f` and `y = 2**f` at the scale `2**-cf` with half an ULP of
    the significand added, in `[1, 4)`. For `log2` `x` is the fraction of the operand and `y = C(f) = log2(1 + f) / f`
    at that scale, `f` being the fraction re-centered from the segment `thr` up.

    The table is addressed by the combinational argument, and the first register is its own, `w` beside it; `depth`
    more follow, the terms of the sum added as they are formed, and the last sum is combinational. `stages` is the
    count of every product, as `Exp2.stage_core_kernel`; each product builds its own capture. `c[2i+1]*w` of one
    multiplier has its output after its pair's sum. `rom_attrs` goes on the table, `mul_attrs` on the products'
    captures.
    """

    in_valid: Signal
    x: Signal
    out_valid: Signal
    y: Signal

    def __init__(
        self,
        spec: Spec,
        *,
        wmultiplier: int,
        stages: int,
        rom_attrs: Attrs | None = None,
        mul_attrs: Attrs | None = None,
    ) -> None:
        assert stages in kernel_counts(spec, wmultiplier), stages
        self._spec, self._wmultiplier, self._schedule = spec, wmultiplier, _schedule(spec, wmultiplier, stages)
        self.depth = self._schedule.total
        self._rom_attrs = rom_attrs or {}
        self._mul_attrs = mul_attrs or {}
        wy = spec.cf + (2 if spec.function == "exp2" else 1)
        super().__init__({"in_valid": In(1), "x": In(spec.k + spec.rw), "out_valid": Out(1), "y": Out(wy)})

    @property
    def latency(self) -> int:
        return 1 + self.depth

    def elaborate(self, platform: object) -> Module:
        m = Module()
        spec, exp2, at = self._spec, self._spec.function == "exp2", self._schedule
        d, k, rw, t, t2, q = spec.d, spec.k, spec.rw, spec.t, spec.t2, spec.q
        th, sg, tc, sc = spec.th, spec.sg, spec.tc, spec.sc
        rows, half = spec.coeffs, d // 2
        cw, fields = _widths(spec)

        def product(name: str, a: Value, b: Value, stages: int | None = None) -> Value:
            m.submodules[name] = multiplier = Pmul(
                len(a),
                len(b),
                signed=False,
                wmultiplier=self._wmultiplier,
                stages=at.stages[name] if stages is None else stages,
                attrs=self._mul_attrs,
            )
            m.d.comb += multiplier.a.eq(a)
            m.d.comb += multiplier.b.eq(b)
            return multiplier.p

        layout = data.StructLayout({f"c{j}": width for j, width in enumerate(fields)})
        table = [
            {
                f"c{j}": (row[j] & ((1 << fields[0]) - 1)) if j == 0 else row[j] >> (cw[j] - fields[j])
                for j in range(d + 1)
            }
            for row in rows
        ]
        m.submodules.rom = rom = Memory(shape=layout, depth=1 << k, init=table, attrs=dict(self._rom_attrs))
        read = rom.read_port()
        m.d.comb += read.addr.eq(self.x[rw:])
        valid = Signal(name="v_r")
        m.d.sync += valid.eq(self.in_valid)
        w = Signal(rw, name="w_r", reset_less=True)
        m.d.sync += w.eq(self.x[:rw])
        c = [Cat(read.data.c0, Const(1, 1))] + [read.data[f"c{j}"] for j in range(1, d + 1)]

        # The terms after c0 by when each is formed, with whether it subtracts; c1*w, the wide product, first.
        terms: dict[int, list[tuple[bool, Value]]] = defaultdict(list)
        terms[at.ready["p1"]].append((not exp2, product("p1", c[1], w[rw - t[0] :])[t[0] - (cw[1] - fields[1]) :]))

        # The pairs c[2i] +- c[2i+1]*w above the wide one, each held for its turn.
        pair_total = _pairs(spec.function, d, rows)
        pair_bits = [value.bit_length() for value in pair_total]
        chain_bits = _chain_bits(pair_total)
        pair: dict[int, Value] = {}
        for i in range(1, half + 1):
            j = 2 * i + 1
            if j > d:
                pair[i] = delay(m, f"c{2 * i}_s", cw[2 * i], c[2 * i], at.turn[i])
                continue
            name, a, b, formed = f"p{j}", c[j], w[rw - t[i] :], at.ready[f"p{j}"]
            if name in at.post:
                # Its output after the pair's sum, for a multiplier's post-adder: the capture held, the register behind.
                attrs = self._mul_attrs
                a, b = delay(m, f"{name}_a", len(a), a, formed, attrs), delay(m, f"{name}_b", len(b), b, formed, attrs)
                pj = product(name, a, b, 0)
            else:
                pj = product(name, a, b)
            even = delay(m, f"c{2 * i}_d", cw[2 * i], c[2 * i], formed)
            term = pj[t[i] - (cw[j] - fields[j]) :]
            first = (even + term if exp2 else even - term)[: pair_bits[i]]
            pair[i] = delay(m, f"pair{i}_", pair_bits[i], first, at.turn[i] - formed)

        if half:
            # w**2: of two products where its operand is wider than a multiplier, the low part's own square left out.
            ws = w[rw - t2 :]
            if th:
                tl, base = t2 - th, 2 * t2 - q[0] - sg
                high, cross = product("sq_hh", ws[tl:], ws[tl:]), product("sq_hl", ws[tl:], ws[:tl])
                square = (high[base - 2 * tl :] + cross[base - tl - 1 :])[sg : q[0] + sg]
            else:
                square = product("sq", ws, ws)[2 * t2 - q[0] :]
            squares = {"sq": (square, at.ready["sq"])}
            if spec.tn:
                # The top of a square of fewer bits, for the products above p2.
                wn, top = w[rw - spec.tn :], max(q[1:])
                squares["sqn"] = (product("sqn", wn, wn)[2 * spec.tn - top :], at.ready["sqn"])

            # The chain: a product with a square takes the pair above, its sum with the pair below feeds the next.
            value, since = pair[half], at.turn[half]
            for i in range(half - 1, -1, -1):
                begin, name = at.begin[i], "sqn" if i and spec.tn else "sq"
                value = delay(m, f"chain{i}_", len(value), value, begin - since)
                taken_square, formed = squares[name]
                taken_square = delay(m, f"{name}{i}_", len(taken_square), taken_square, begin - formed)
                squares[name] = taken_square, begin
                if not i:
                    terms[at.ready["p2"]].append((False, product("p2", value, taken_square)[q[0] :]))
                    break
                taken = product(f"p{2 * (i + 1)}", value, taken_square[len(taken_square) - q[i] :])[q[i] :]
                taken = delay(m, f"taken{i}_", len(taken), taken, at.turn[i] - begin - at.stages[f"p{2 * (i + 1)}"])
                value, since = (pair[i] + taken)[: chain_bits[i]], at.turn[i]
        if tc:
            # The argument bits below the wide product's operand times the top of c1: the rest of c1*w.
            assert d == 2 and exp2, "log2's argument fits one multiplier at degree 2"
            terms[at.ready["pl"]].append((False, product("pl", c[1][sc:], w[:tc])[rw - sc :]))

        out_valid, partial, since = valid, c[0], 0
        for time, group in sorted(terms.items()):
            out_valid, partial = pipe(m, f"sum{time}_", out_valid, len(partial), partial, time - since)
            if time == max(terms) and len(group) == 1:
                # A carry select: the last term meets the low bits; their carry or borrow picks the high bits or a step.
                [(subtracted, term)] = group
                n = len(term)
                low = partial[:n] - term if subtracted else partial[:n] + term
                high = partial[n : len(self.y)]
                partial = Cat(low[:n], Mux(low[n], high - 1 if subtracted else high + 1, high))[: len(self.y)]
            else:
                for subtracted, term in group:
                    partial = (partial - term if subtracted else partial + term)[: len(self.y)]
            since = time
        m.d.comb += self.out_valid.eq(out_valid)
        m.d.comb += self.y.eq(partial)
        return m


def _arith_range(spec: Spec, row: Sequence[int]) -> tuple[Fraction, Fraction]:
    """
    The range of the exact polynomial less the integer kernel over every `w`, in units of `2**-cf`, from the definition
    of the kernel: each product is short by the operand bits it drops times the bound of its other operand, and by less
    than one for its floor.
    """
    rw, d, t, q, t2 = spec.rw, spec.d, spec.t, spec.q, spec.t2
    sub, q2 = spec.function == "log2", max(q, default=0)
    square = 0 < t2 <= rw and all(0 < x <= q2 for x in q) and q2 <= 2 * t2
    if not (all(0 < x <= rw for x in t) and (d == 1 or square)):
        raise ValueError("operand widths out of range")
    lsb = Fraction(1, 1 << rw)
    # wn**2 - w2 / 2**q2 is in [0, d2]: the squared operand is short by under 2**-t2, the square is floored.
    d2 = 2 * (Fraction(1, 1 << t2) - lsb) + (Fraction(1, 1 << q2) if 2 * t2 > q2 else 0)
    if spec.th:
        # wt**2 = hh * 4**tl + hl * 2**(tl + 1) + ll. At the scale 2**base the two terms kept are floored (under one
        # unit each, 2**-sg of the result's), ll < 4**tl is dropped, and the sum is floored to q2 bits.
        th, sg = spec.th, spec.sg
        tl, base = t2 - th, 2 * t2 - q2 - sg
        if not (d == 3 and 0 < tl <= th and sg > 0 and base >= 2 * tl):
            raise ValueError("split square fields")
        d2 += (Fraction(2, 1 << sg) + Fraction(1 << (2 * tl), 1 << (2 * t2 - q2))) / (1 << q2)

    def odd(i: int) -> Fraction:
        """`c[2i+1] * wn` less its floored product on the top `t[i]` argument bits, at most."""
        return row[2 * i + 1] * (Fraction(1, 1 << t[i]) - lsb) + 1

    linear = odd(0)
    if spec.tc:
        # c1 * wn = c1 * wh / 2**t0 + c1 * wl / 2**rw; the second product drops the low sc bits of c1; two floors.
        tc, sc = spec.tc, spec.sc
        if not (tc == rw - t[0] and 0 <= sc <= rw and sc < row[1].bit_length()):
            raise ValueError("split of the wide product")
        linear = 2 + ((1 << sc) - 1) * ((1 << tc) - 1) * lsb
    zero = Fraction(0)
    lo, hi = (-linear, zero) if sub else (zero, linear)
    if d == 1:
        return lo, hi
    # Pairs chained by wn**2: V*W - floor(v*w) = (V - v)*W + v*(W - w) + a fraction, W in [0, 1), v in [0, vmax].
    m = d // 2
    chain_lo, chain_hi, vmax = zero, zero, Fraction(row[2 * m])
    if 2 * m + 1 <= d:
        chain_lo, chain_hi = (-odd(m), zero) if sub else (zero, odd(m))
        vmax += 0 if sub else row[2 * m + 1]
    for i in range(m, 0, -1):
        if sub and 2 * i + 1 <= d and not row[2 * i] > row[2 * i + 1]:
            raise ValueError("a pair may go non-positive")
    tn = spec.tn
    if tn and not (d >= 4 and all(q[i] <= 2 * tn for i in range(1, m)) and tn <= rw):
        raise ValueError("narrow square fields")
    for i in range(m - 1, 0, -1):
        pair_lo, pair_hi = (-odd(i), zero) if sub else (zero, odd(i))
        # The square that the product takes is short of wn**2 by its operand's dropped bits and its floor.
        taken = 2 * (Fraction(1, 1 << tn) - lsb) + Fraction(1, 1 << q[i]) if tn else d2 + Fraction(1, 1 << q[i])
        chain_hi = pair_hi + max(chain_hi, zero) + vmax * taken + 1
        chain_lo = pair_lo + min(chain_lo, zero)
        vmax += row[2 * i] + (0 if sub else row[2 * i + 1])
    return lo + min(chain_lo, zero), hi + max(chain_hi, zero) + vmax * d2 + 1


def _argument(function: str, k: int, thr: int, segment: int, wn: Any) -> Any:
    """The helper's argument at the position `wn` in [0, 1] of a segment."""
    x = (segment + wn) / (1 << k)
    return x if function == "exp2" or segment < thr else (x - 1) / 2  # re-centered: f = m / 2 - 1, m = 1 + x


def _helper(function: str, argument: Any) -> Any:
    if function == "exp2":
        return _fitting.power(2, argument)
    return _fitting.log(1 + argument) / (_fitting.log(2) * argument) if argument else 1 / _fitting.log(2)


def _fit(function: str, k: int, d: int, cf: int, thr: int, segment: int) -> list[int]:
    """The magnitudes of a near-minimax polynomial of a segment, the low degree first, at the scale `2**cf`."""
    fitted = _fitting.chebyfit(
        lambda wn: _helper(function, _argument(function, k, thr, segment, wn)),
        [_fitting.mpf(0), _fitting.mpf(1)],
        d + 1,
        asc=True,
    )
    out = [int(_fitting.nint(c * (1 << cf))) for c in fitted]
    # The coefficients of exp2 are positive; those of log2 alternate from a positive one; a high one may round to zero.
    assert all(not c or (c > 0) != (function == "log2" and j % 2) for j, c in enumerate(out)), f"{segment}: {out}"
    return [abs(c) for c in out]


def _estimate(spec: Spec, raw: list[list[int]], segments: Sequence[int]) -> tuple[float, tuple[tuple[int, ...], ...]]:
    """
    The share of the budget that the worst error of the helper is estimated to take over the `segments` whose
    coefficients `raw` holds, and those coefficients with the constants folded into `c0`: the half ULP of `exp2`, and
    the center of the span of the floors. The estimate is the error of the fit at its equioscillation points plus half
    that span, which is the certificate's bound of the arithmetic.
    """
    exp2 = spec.function == "exp2"
    budget = _fitting.mpf(2) ** -(spec.wman + _ERR_GUARD)
    scale = _fitting.mpf(1 << spec.cf)
    probes = [(1 - _fitting.cos(_fitting.pi * i / (spec.d + 1))) / 2 for i in range(spec.d + 2)]
    worst = _fitting.mpf(0)
    rows = []
    for segment, row in zip(segments, raw):
        lo, hi = _arith_range(spec, row)
        if exp2:
            hi += Fraction(7, 10) * 2  # the reduced argument is low by up to 2**-cf: ln 2 * 2**f < 1.4 of an LSB
        rows.append((row[0] + (1 << (spec.cf - spec.wman) if exp2 else 0) + round((lo + hi) / 2), *row[1:]))
        true = [_helper(spec.function, _argument(spec.function, spec.k, spec.thr, segment, wn)) for wn in probes]
        fits = [sum((c if exp2 or j % 2 == 0 else -c) * wn**j for j, c in enumerate(row)) for wn in probes]
        fit = max(abs(p / scale - value) for p, value in zip(fits, true))
        half = (hi - lo) / 2
        worst = max(worst, (fit + (_fitting.mpf(half.numerator) / half.denominator + 1) / scale) / min(true) / budget)
    return float(worst), tuple(rows)


def _ends(function: str, k: int, thr: int) -> list[int]:
    """
    The segments where each coefficient is widest and the error largest: those of `exp2` grow with the segment, those
    of `log2` peak at its first and where re-centering starts.
    """
    n = 1 << k
    ends = {0, 1, n - 2, n - 1} if function == "exp2" else {0, 1, thr - 1, thr, thr + 1, n - 1}
    return sorted(segment for segment in ends if 0 <= segment < n)


def _pairs(function: str, d: int, rows: Sequence[Sequence[int]]) -> list[int]:
    """The largest value of each pair `c[2i] +- c[2i+1]*w`, its odd term counted where it adds."""
    return [
        max(r[2 * i] + (r[2 * i + 1] if 2 * i + 1 <= d and function == "exp2" else 0) for r in rows)
        for i in range(d // 2 + 1)
    ]


def _chain_bits(pairs: list[int]) -> list[int]:
    """The width of the value that the chain holds from pair `i` up, for `i` from 1; zero elsewhere."""
    return [0] + [sum(pairs[i:]).bit_length() for i in range(1, len(pairs))] + [0]


def _degree(wman: int) -> int:
    """
    The degree of a compact table's polynomial: `WMAN + _ERR_GUARD` bits at the narrowest segments, of `_K_CAP` bits.
    """
    return ceil((wman + _ERR_GUARD) / _K_CAP) - 1


def _generate(
    function: str,
    wman: int,
    wmultiplier: int,
    *,
    d: int | None = None,
    k_base: int = _K_BASE,
    k_cap: int = _K_CAP,
    cut: bool = False,
    probe: bool = False,
) -> Spec:
    """
    The table of degree `d` (the compact one's by default) with the fewest segments, from `2**k_base` up to `2**k_cap`,
    whose operands fit multipliers of `wmultiplier` bits and whose estimated error fits the budget. The operands of the
    products are cut or fitted to that width where the budget allows; with `cut`, also degree 1's and the narrow square.
    A `probe` fits only the segments that bound the widths and the estimate, for `_shortest` to rank tables by.
    """
    cf, narrow = wman + _GUARD, wmultiplier
    d = d or _degree(wman)
    width = cf if function == "exp2" else wman - 1

    def fitted(width: int) -> int:
        """The width, or the multiple of the multipliers' below it where it passes that by a few bits."""
        short = narrow * (ceil(width / narrow) - 1)
        return short if short and width - short <= 3 else width

    for k in range(min(k_base, width - 1), min(k_cap, width - 1) + 1):
        rw = width - k
        # The segment boundary nearest sqrt(2) - 1.
        thr = int(_fitting.nint((_fitting.sqrt(2) - 1) * (1 << k))) if function == "log2" else 0
        # A probe fits the segments where each coefficient peaks and the error is largest: the widths are the fit's.
        segments = _ends(function, k, thr) if probe else range(1 << k)
        raw = [_fit(function, k, d, cf, thr, segment) for segment in segments]
        if not all(row[j] for row in raw for j in range(1, d + 1)):  # a zero: the degree is too high for the segments
            continue
        cw = [max(row[j] for row in raw).bit_length() for j in range(d + 1)]
        chain = _chain_bits(_pairs(function, d, raw))
        # Each product keeps the operand bits that hold its error to about an LSB of the result's scale.
        t = [rw] + [min(rw, cw[2 * i + 1]) for i in range(1, (d + 1) // 2)]
        t2 = min(rw, chain[1])
        q = tuple(min(2 * t2, chain[i + 1]) for i in range(d // 2))
        spec = Spec(function, wman, k, rw, cf, d, tuple(t), t2, q, thr=thr)

        def tried(shorter: Spec) -> Spec:
            return shorter if shorter != spec and _estimate(shorter, raw, segments)[0] <= _SELECT else spec

        if d == 1 and cut:
            # c1 and w each give up the few bits past a multiple of the multipliers, if the budget allows.
            z1, t0 = cw[1] - fitted(cw[1]), fitted(rw)
            rounded = [[row[0], ((row[1] + ((1 << z1) >> 1)) >> z1) << z1] for row in raw]
            shorter = replace(spec, z=(0, z1), t=(t0,))
            fits = max(row[1] for row in rounded).bit_length() == cw[1] and z1 <= t0
            if (z1 or t0 < rw) and fits and _estimate(shorter, rounded, segments)[0] <= _SELECT:
                spec, raw = shorter, rounded
        elif d == 2:
            if max(t2, cw[2], q[0]) > narrow:
                continue
            if rw > narrow:
                # A multiplier's width of argument; the bits below multiply the top of c1, unless they cost an LSB only.
                tc, sc = (rw - narrow, max(0, min(narrow - 2, cw[1] - 2))) if cw[1] - narrow > 1 else (0, 0)
                spec = replace(spec, t=(narrow,), tc=tc, sc=sc)
        elif d == 3:
            # c3 * w is one multiplier: c3 gives up its low bits if it is wider.
            z3 = max(0, cw[3] - narrow)
            for row in raw:
                row[3] = ((row[3] + ((1 << z3) >> 1)) >> z3) << z3
            spec = replace(spec, z=(0, 0, 0, z3), t=(rw, min(rw, narrow)))
            if z3 > spec.t[1]:  # c3 keeps no bit that its product takes
                continue
            if t2 > narrow:
                # The square takes two multipliers and a short sum: the square of the low part is left out.
                spec = replace(spec, th=narrow, sg=_SQUARE_GUARD)
                if t2 > 2 * narrow or 2 * t2 - q[0] - _SQUARE_GUARD < 2 * (t2 - narrow):
                    continue
            # A few argument bits fewer in the wide product can save a multiplier, if the budget allows it.
            spec = tried(replace(spec, t=(fitted(rw), spec.t[1])))
        elif d >= 4:
            # The square's operand and the wide product's argument give up the few bits past a multiple of the
            # multipliers, each where the budget allows; with `cut`, the products above p2 take a square of fewer bits.
            spec = tried(replace(spec, t2=fitted(t2), q=tuple(min(2 * fitted(t2), x) for x in q)))
            spec = tried(replace(spec, t=(fitted(rw), *spec.t[1:])))
            tn = min(max(spec.q[1:]), narrow)
            if cut and tn < spec.t2 and max(spec.q[1:]) <= 2 * tn:
                spec = tried(replace(spec, tn=tn))
        share, coeffs = _estimate(spec, raw, segments)
        if share > _SELECT:
            continue
        if not all(row[0] >> cf == 1 for row in coeffs):  # c0, with its leading one, leaves [2**cf, 2**(cf + 1))
            continue
        spec = replace(spec, coeffs=coeffs)
        # What the kernel relies on structurally; the certificate proves the numeric ranges.
        assert function == "exp2" or 0 < thr < 1 << k, "log2: a re-centered fraction is nonzero"
        return spec
    raise UnsupportedFormat(
        f"no {function} table of {k_cap} bits of segment or fewer fits WMAN={wman} on {wmultiplier}-bit multipliers"
    )


@functools.cache
def _generated(function: str, wman: int, wmultiplier: int, fast: bool) -> Spec:
    if not fast:
        return _generate(function, wman, wmultiplier)
    return _shortest(_generated(function, wman, wmultiplier, False), wmultiplier)


def _shortest(compact: Spec, wmultiplier: int) -> Spec:
    """
    Of the tables of `2**_K_FAST` segments or fewer that make a kernel shorter at count 3 than the compact one, that of
    the least depth at count 3, then at count 2, then bits, then multipliers; or the compact one. Every degree up to the
    compact one's, at every segmentation, with its operands cut to the multipliers or not, is probed; the full fits
    follow the probes' order until the best fitted ranks no worse than the next probe, which ranks no worse than its
    full fit.
    """
    function, wman = compact.function, compact.wman
    width = compact.cf if function == "exp2" else wman - 1

    def rank(spec: Spec) -> tuple[int, int, int, int]:
        products = (slices(*size, signed=False, wmultiplier=wmultiplier) for size in _sizes(spec).values())
        depth = _schedule(spec, wmultiplier, 3).total
        return depth, _schedule(spec, wmultiplier, 2).total, _rom_bits(spec), sum(a * b for a, b in products)

    def table(d: int, k: int, cut: bool, probe: bool) -> Spec | None:
        try:
            return _generate(function, wman, wmultiplier, d=d, k_base=k, k_cap=k, cut=cut, probe=probe)
        except UnsupportedFormat:
            return None

    shortest = rank(compact)[0]
    ranked = []
    for d in range(1, compact.d + 1):
        for k in range(min(_K_BASE, width - 1), min(_K_FAST, width - 1) + 1):
            for cut in (False, True) if d == 1 or d >= 4 else (False,):
                probed = table(d, k, cut, True)
                if probed is not None and rank(probed)[0] < shortest:
                    ranked.append((rank(probed), d, k, cut))
    best: tuple[tuple[int, int, int, int], Spec] | None = None
    for predicted, d, k, cut in sorted(ranked):
        if best is not None and best[0] <= predicted:
            break
        found = table(d, k, cut, False)
        if found is not None and rank(found)[0] < shortest and (best is None or rank(found) < best[0]):
            best = rank(found), found
    return compact if best is None else best[1]


def _rom_bits(spec: Spec) -> int:
    """The bits of the kernel's table."""
    return (1 << spec.k) * sum(_widths(spec)[1])


def _widths(spec: Spec) -> tuple[list[int], list[int]]:
    """
    Of each coefficient, the low degree first: its width, and that in the table, where `c0` lacks its leading one,
    which is constant, and the others their zeros of `z`.
    """
    d, cw = spec.d, [max(row[j] for row in spec.coeffs).bit_length() for j in range(spec.d + 1)]
    z = spec.z + (0,) * (d + 1 - len(spec.z))
    return cw, [cw[0] - 1] + [cw[j] - z[j] for j in range(1, d + 1)]


def _sizes(spec: Spec) -> dict[str, tuple[int, int]]:
    """The operand widths of each product of the kernel, by name."""
    d, t, half = spec.d, spec.t, spec.d // 2
    cw, fields = _widths(spec)
    chain = _chain_bits(_pairs(spec.function, d, spec.coeffs))
    sizes = {"p1": (fields[1], t[0])} | ({"pl": (cw[1] - spec.sc, spec.tc)} if spec.tc else {})
    sizes |= {f"p{2 * i + 1}": (fields[2 * i + 1], t[i]) for i in range(1, half + 1) if 2 * i + 1 <= d}
    if not half:
        return sizes
    if spec.th:
        sizes |= {"sq_hh": (spec.th, spec.th), "sq_hl": (spec.th, spec.t2 - spec.th)}
    else:
        sizes["sq"] = (spec.t2, spec.t2)
    if spec.tn:
        sizes |= {"sqn": (spec.tn,) * 2}
    return sizes | {f"p{2 * (i + 1)}": (chain[i + 1], spec.q[i]) for i in range(half)}


def _caps(spec: Spec, wmultiplier: int) -> dict[str, int]:
    """The positions of each product of the kernel: those of `Pmul`, the output and the operands of one multiplier."""
    out = {}
    for name, (a, b) in _sizes(spec).items():
        lone = slices(a, b, signed=False, wmultiplier=wmultiplier) == (1, 1)
        out[name] = 2 if lone else len(positions(a, b, signed=False, wmultiplier=wmultiplier))
    return out


@dataclass(frozen=True)
class _Schedule:
    """
    The kernel's products in stages after the table's register: the count of each, by name; the odd ones whose output
    follows the pair's sum; when the result of each is formed (`sq` the square's, `sqn` the narrow one's); when pair
    `i` joins the chain, the top one at its start; when the chain's product below pair `i + 1` takes its operands; the
    total.
    """

    stages: dict[str, int]
    post: frozenset[str]
    ready: dict[str, int]
    turn: dict[int, int]
    begin: dict[int, int]
    total: int


def _schedule(spec: Spec, wmultiplier: int, stages: int) -> _Schedule:
    d, half, caps = spec.d, spec.d // 2, _caps(spec, wmultiplier)
    least = {name: counts(a, b, signed=False, wmultiplier=wmultiplier).start for name, (a, b) in _sizes(spec).items()}
    n = {name: min(max(stages, least[name]), cap) for name, cap in caps.items()}
    odd = [f"p{2 * i + 1}" for i in range(1, half + 1) if 2 * i + 1 <= d]
    post = frozenset(name for name in odd if caps[name] == 2)
    ready = {name: n[name] for name in caps if name in ("p1", "pl", *odd)}
    ready |= {name: max(0, n[name] - 1) for name in post}
    turn: dict[int, int] = {}
    begin: dict[int, int] = {}
    if half:
        squares = [name for name in caps if name in ("sq", "sq_hh", "sq_hl")]
        assert len({n[name] for name in squares}) == 1, "the square's products are of one size"
        ready["sq"] = n[squares[0]]
        ready["sqn"] = n["sqn"] if spec.tn else ready["sq"]
        # A pair forms at its product's count, a `post` one in its register; the chain starts at its top pair, a product
        # takes its operands once the pair above and its square are formed; a pair joins once it and that product are.
        formed = {i: n.get(f"p{2 * i + 1}", 0) for i in range(1, half + 1)}
        turn = {half: formed[half]}
        for i in range(half - 1, -1, -1):
            begin[i] = max(turn[i + 1], ready["sqn" if i else "sq"])
            done = begin[i] + n[f"p{2 * (i + 1)}"]
            if i:
                turn[i] = max(done, formed[i])
        ready["p2"] = done
    total = max(ready[name] for name in ("p1", "pl", "p2") if name in ready)
    return _Schedule(n, post, ready, turn, begin, total)

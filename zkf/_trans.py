"""
The piecewise-polynomial kernels of `2**f` and `log2(1 + f) / f`: the table spec, its generation on demand, and the
component that evaluates it. The tests certify the tables.
"""

import functools
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil
from typing import Any

import mpmath
from amaranth.hdl import Cat, Const, Module, Signal, Value
from amaranth.lib import data, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

from ._block import Attrs, delay, pipe
from ._pack import pack_reference
from ._pmul import Pmul, counts, positions, slices
from ._value import UnsupportedFormat, ZkfFormat

_ERR_GUARD = 8
"""A kernel is within `2**-(WMAN + _ERR_GUARD)` of its helper, relatively: `2**-_ERR_GUARD` ULP of a result."""
_GUARD = _ERR_GUARD + 4
"""Fraction bits of a coefficient below those of the significand."""
_K_CAP = 11
"""The most bits that index a segment. The degree follows from it, and the length of the chain from the degree."""
_K_BASE = 9
"""The fewest bits that index a segment: one more doubles the table and takes a bit off every operand above `c0`."""
_SQUARE_GUARD = 3
"""Bits that a square of two multipliers carries below those taken from it."""
_SELECT = 0.9
"""The share of the budget that the estimate may take; the certificate proves the whole."""

# A context of its own, whose precision decides the coefficients: the caller's is neither read nor changed.
_fitting = mpmath.mp.clone()
_fitting.prec = 280


def degree(wman: int) -> int:
    """The degree of a segment's polynomial: `WMAN + _ERR_GUARD` bits at the narrowest segments, of `_K_CAP` bits."""
    return ceil((wman + _ERR_GUARD) / _K_CAP) - 1


@dataclass(frozen=True)
class Spec:
    """
    The table of `function` at one WMAN, and the arithmetic that evaluates it. The polynomial of a segment,
    `c0 +- c1*wn + c2*wn**2 +- ...` in `wn = w / 2**rw` at the scale `2**-cf`, is taken as `c0 +- c1*wn` beside the
    pairs `c[2i] +- c[2i+1]*wn` chained by `wn**2`, which is formed once. Every product takes a fixed number of operand
    bits and is floored; the constants that center those floors, and the half ULP of `exp2`, are part of `c0`. The odd
    coefficients of `log2` subtract: a table holds magnitudes.
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
    z3: int = 0
    """Degree 3: low bits that `c3` lacks, all zero, so that `c3*wn` is one multiplier wide."""
    th: int = 0
    """
    Degree 3, a square wider than a multiplier: its top `th` bits are squared and multiplied by the rest, the square of
    the rest is left out, and the sum carries `sg` bits below those taken.
    """
    sg: int = 0
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
        q2 = max(q)
        wt = w >> (rw - t2)
        if self.th:
            tl, base = t2 - self.th, 2 * t2 - q2 - self.sg
            wh, wl = wt >> tl, wt & ((1 << tl) - 1)
            w2 = (((wh * wh) >> (base - 2 * tl)) + ((wh * wl) >> (base - tl - 1))) >> self.sg
        else:
            w2 = (wt * wt) >> (2 * t2 - q2)
        sign = -1 if self.function == "log2" else 1

        def odd(i: int) -> int:
            return (row[2 * i + 1] * (w >> (rw - t[i]))) >> t[i]

        linear = odd(0)
        if self.tc:
            linear += ((row[1] >> self.sc) * (w & ((1 << self.tc) - 1))) >> (rw - self.sc)
        m = self.d // 2
        value = row[2 * m] + (sign * odd(m) if 2 * m + 1 <= self.d else 0)
        for i in range(m - 1, 0, -1):
            value = row[2 * i] + sign * odd(i) + ((value * (w2 >> (q2 - q[i]))) >> q[i])
        return row[0] + sign * linear + ((value * w2) >> q2)


def spec(function: str, wman: int, wmultiplier: int) -> Spec:
    """
    The table of `function` at a significand of `wman` bits, 16 to 53, for multipliers of `wmultiplier` bits,
    generated once.
    """
    if not 16 <= wman <= 53:
        raise UnsupportedFormat(f"no {function} table for WMAN={wman}: it is generated for 16 to 53")
    with _GENERATING:  # the fit has one context, and a flow compiles its specs from threads
        return _generated(function, wman, wmultiplier)


def pack_fixed(fmt: ZkfFormat, sign: int, exp: int, v: int, point: int) -> int:
    """
    The float nearest to `(-1)**sign * 2**exp * v / 2**point`, `v > 0`, a tie rounding away from zero, classified as
    `Pack` does.
    """
    drop = v.bit_length() - fmt.wman  # bits below the significand
    assert drop >= 1
    # With its round bit set the packer rounds up exactly when the guard bit is.
    return pack_reference(fmt, sign, 0, 0, exp + v.bit_length() - 1 - point, v >> drop, (v >> (drop - 1)) & 1, 1, 0)


def kernel_counts(spec: Spec, wmultiplier: int) -> range:
    """The counts that `Kernel` takes: from the least that every product takes, up to the most positions of one."""
    least = max(counts(a, b, signed=False, wmultiplier=wmultiplier).start for a, b in _sizes(spec).values())
    return range(least, max(_caps(spec, wmultiplier).values()) + 1)


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
    The table and the polynomial: the top `k` bits of the argument select a row of coefficients, and the rest, `w`, is
    the variable. The arithmetic is that of `Spec.at`: `c0 +- c1*w` beside the chain
    `w**2 * (c2 +- c3*w + w**2 * (c4 +- c5*w))`, one shape at every degree.

    For `exp2` the argument is `f` and the result `s = 2**f` at the scale `2**-cf` with half an ULP of the significand
    added, in `[1, 4)`. For `log2` the argument is the fraction of `x`, the result `c = C(f) = log2(1 + f) / f` at that
    scale, `f` being the fraction re-centered where `rc`, which is combinational from the argument.

    The table is addressed by the combinational argument, and the first register is its own, `w` beside it; `depth`
    more follow, and the last sum is combinational. `stages` is the count of every product (see `zkf._pmul.Pmul`), each
    capped at its own positions, those of one multiplier at its output and operands; each product builds its own
    capture. `c[2i+1]*w` of one multiplier has its output after its pair's sum. `rom_attrs` goes on the table,
    `mul_attrs` on the products' captures.
    """

    in_valid: Signal
    out_valid: Signal

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
        cf = spec.cf
        if spec.function == "exp2":
            members = {"in_valid": In(1), "f": In(cf), "out_valid": Out(1), "s": Out(cf + 2)}
        else:
            members = {
                "in_valid": In(1),
                "frac": In(spec.wman - 1),
                "rc": Out(1),
                "out_valid": Out(1),
                "c": Out(cf + 1),
            }
        super().__init__(members)

    @property
    def latency(self) -> int:
        return 1 + self.depth

    def elaborate(self, platform: object) -> Module:
        m = Module()
        spec, exp2, at = self._spec, self._spec.function == "exp2", self._schedule
        d, k, rw, t, t2, q = spec.d, spec.k, spec.rw, spec.t, spec.t2, spec.q
        th, sg, tc, sc = spec.th, spec.sg, spec.tc, spec.sc
        rows, half, total = spec.coeffs, d // 2, at.total
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
        argument = self.f if exp2 else self.frac
        segment = argument[rw : rw + k]
        read = rom.read_port()
        m.d.comb += read.addr.eq(segment)
        if not exp2:
            m.d.comb += self.rc.eq(segment >= spec.thr)
        valid = Signal(name="v_r")
        m.d.sync += valid.eq(self.in_valid)
        w = Signal(rw, name="w_r", reset_less=True)
        m.d.sync += w.eq(argument[:rw])
        c = [Cat(read.data.c0, Const(1, 1))] + [read.data[f"c{j}"] for j in range(1, d + 1)]

        # c0 +- c1*w, the wide product, held for the chain.
        out_valid, c0 = pipe(m, "c0_d", valid, cw[0], c[0], total)
        m.d.comb += self.out_valid.eq(out_valid)
        p1 = delay(m, "p1_d", cw[1], product("p1", c[1], w[rw - t[0] :])[t[0] :], total - at.ready["p1"])

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
                # Its capture, and its output after the pair's sum, which a multiplier's post-adder takes: the capture
                # is held, with the pair's register behind the multiplier.
                attrs = self._mul_attrs
                a, b = delay(m, f"{name}_a", len(a), a, formed, attrs), delay(m, f"{name}_b", len(b), b, formed, attrs)
                pj = product(name, a, b, 0)
            else:
                pj = product(name, a, b)
            even = delay(m, f"c{2 * i}_d", cw[2 * i], c[2 * i], formed)
            term = pj[t[i] - (cw[j] - fields[j]) :]
            first = (even + term if exp2 else even - term)[: pair_bits[i]]
            pair[i] = delay(m, f"pair{i}_", pair_bits[i], first, at.turn[i] - formed)

        # w**2: of two products where its operand is wider than a multiplier, the low part's own square left out.
        ws = w[rw - t2 :]
        if th:
            tl, base = t2 - th, 2 * t2 - q[0] - sg
            high, cross = product("sq_hh", ws[tl:], ws[tl:]), product("sq_hl", ws[tl:], ws[:tl])
            square = (high[base - 2 * tl :] + cross[base - tl - 1 :])[sg : q[0] + sg]
        else:
            square = product("sq", ws, ws)[2 * t2 - q[0] :]
        square = delay(m, "sq_d", q[0], square, at.turn[half] - at.ready["sq"])

        # The chain: a product with the square takes the pair above, its sum with the pair below feeds the next.
        value = pair[half]
        for i in range(half - 1, 0, -1):
            taken = product(f"p{2 * (i + 1)}", value, square[q[0] - q[i] :])[q[i] :]
            value = (pair[i] + taken)[: chain_bits[i]]
            square = delay(m, f"sq_{i}_", q[0], square, at.turn[i] - at.turn[i + 1])
        last = delay(m, "p2_d", chain_bits[1], product("p2", value, square)[q[0] :], total - at.ready["p2"])
        chain: Value = last
        if tc:
            # The argument bits below the wide product's operand times the top of c1: the rest of c1*w.
            assert d == 2 and exp2, "log2's argument fits one multiplier at degree 2"
            rest = cw[1] + tc - rw
            low = delay(m, "pl_d", rest, product("pl", c[1][sc:], w[:tc])[rw - sc :], total - at.ready["pl"])
            chain = (last + low)[: max(cw[2], rest) + 1]
        if exp2:
            m.d.comb += self.s.eq((c0 + p1 + chain)[: len(self.s)])
        else:
            m.d.comb += self.c.eq((c0 - p1 + chain)[: len(self.c)])
        return m


def _arith_range(spec: Spec, row: Sequence[int]) -> tuple[Fraction, Fraction]:
    """
    The range of the exact polynomial less the integer kernel over every `w`, in units of `2**-cf`, from the definition
    of the kernel: each product is short by the operand bits it drops times the bound of its other operand, and by less
    than one for its floor.
    """
    rw, d, t, q, t2 = spec.rw, spec.d, spec.t, spec.q, spec.t2
    sub, q2 = spec.function == "log2", max(q)
    if not (0 < t2 <= rw and all(0 < x <= rw for x in t) and all(0 < x <= q2 for x in q) and q2 <= 2 * t2):
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
        # c1 * wn = c1 * wh / 2**t0 + c1 * wl / 2**rw exactly; the second product drops the low sc bits of c1. Two
        # floors.
        tc, sc = spec.tc, spec.sc
        if not (tc == rw - t[0] and 0 <= sc <= rw and sc < row[1].bit_length()):
            raise ValueError("split of the wide product")
        linear = 2 + ((1 << sc) - 1) * ((1 << tc) - 1) * lsb
    zero = Fraction(0)
    lo, hi = (-linear, zero) if sub else (zero, linear)
    # Pairs chained by wn**2: V*W - floor(v*w) = (V - v)*W + v*(W - w) + a fraction, W in [0, 1), v in [0, vmax].
    m = d // 2
    chain_lo, chain_hi, vmax = zero, zero, Fraction(row[2 * m])
    if 2 * m + 1 <= d:
        chain_lo, chain_hi = (-odd(m), zero) if sub else (zero, odd(m))
        vmax += 0 if sub else row[2 * m + 1]
    for i in range(m, 0, -1):
        if sub and 2 * i + 1 <= d and not row[2 * i] > row[2 * i + 1]:
            raise ValueError("a pair may go non-positive")
    for i in range(m - 1, 0, -1):
        pair_lo, pair_hi = (-odd(i), zero) if sub else (zero, odd(i))
        chain_hi = pair_hi + max(chain_hi, zero) + vmax * (d2 + Fraction(1, 1 << q[i])) + 1
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
    # The coefficients of exp2 are positive; those of log2 alternate from a positive one.
    assert all((c > 0) != (function == "log2" and j % 2) for j, c in enumerate(out)), f"segment {segment}: {out}"
    return [abs(c) for c in out]


def _estimate(spec: Spec, raw: list[list[int]]) -> tuple[float, tuple[tuple[int, ...], ...]]:
    """
    The share of the budget that the worst error of the helper is estimated to take, and the coefficients with the
    constants folded into `c0`: the half ULP of `exp2`, and the center of the span of the floors. The estimate is the
    error of the fit at its equioscillation points plus half that span, which is the certificate's bound of the
    arithmetic. A shape that the certificate refuses takes more than the budget.
    """
    exp2 = spec.function == "exp2"
    budget = _fitting.mpf(2) ** -(spec.wman + _ERR_GUARD)
    scale = _fitting.mpf(1 << spec.cf)
    probes = [(1 - _fitting.cos(_fitting.pi * i / (spec.d + 1))) / 2 for i in range(spec.d + 2)]
    worst = _fitting.mpf(0)
    rows = []
    for segment, row in enumerate(raw):
        try:
            lo, hi = _arith_range(spec, row)
        except ValueError:
            return float("inf"), ()
        if exp2:
            hi += Fraction(7, 10) * 2  # the reduced argument is low by up to 2**-cf: ln 2 * 2**f < 1.4 of an LSB
        rows.append((row[0] + (1 << (spec.cf - spec.wman) if exp2 else 0) + round((lo + hi) / 2), *row[1:]))
        true = [_helper(spec.function, _argument(spec.function, spec.k, spec.thr, segment, wn)) for wn in probes]
        fits = [sum((c if exp2 or j % 2 == 0 else -c) * wn**j for j, c in enumerate(row)) for wn in probes]
        fit = max(abs(p / scale - value) for p, value in zip(fits, true))
        half = (hi - lo) / 2
        worst = max(worst, (fit + (_fitting.mpf(half.numerator) / half.denominator + 1) / scale) / min(true) / budget)
    return float(worst), tuple(rows)


def _pairs(function: str, d: int, rows: Sequence[Sequence[int]]) -> list[int]:
    """The largest value of each pair `c[2i] +- c[2i+1]*w`, its odd term counted where it adds."""
    return [
        max(r[2 * i] + (r[2 * i + 1] if 2 * i + 1 <= d and function == "exp2" else 0) for r in rows)
        for i in range(d // 2 + 1)
    ]


def _chain_bits(pairs: list[int]) -> list[int]:
    """The width of the value that the chain holds from pair `i` up, for `i` from 1; zero elsewhere."""
    return [0] + [sum(pairs[i:]).bit_length() for i in range(1, len(pairs))] + [0]


def _generate(function: str, wman: int, wmultiplier: int, *, d: int | None = None, k_base: int = _K_BASE) -> Spec:
    """
    The table with the fewest segments, from `2**k_base` up, whose operands fit multipliers of `wmultiplier` bits and
    whose estimated error fits the budget. Up to degree 3 the operands of the products are cut or fitted to that width
    where the budget allows. `d` and `k_base` are given only to reach, at a width that can be enumerated, a structure
    that the formats reach at widths that cannot.
    """
    cf, narrow = wman + _GUARD, wmultiplier
    d = d or degree(wman)
    width = cf if function == "exp2" else wman - 1
    for k in range(min(k_base, width - 1), _K_CAP + 1):
        rw = width - k
        # The segment boundary nearest sqrt(2) - 1.
        thr = int(_fitting.nint((_fitting.sqrt(2) - 1) * (1 << k))) if function == "log2" else 0
        raw = [_fit(function, k, d, cf, thr, segment) for segment in range(1 << k)]
        cw = [max(row[j] for row in raw).bit_length() for j in range(d + 1)]
        chain = _chain_bits(_pairs(function, d, raw))
        # Each product keeps the operand bits that hold its error to about an LSB of the result's scale.
        t = [rw] + [min(rw, cw[2 * i + 1]) for i in range(1, (d + 1) // 2)]
        t2 = min(rw, chain[1])
        q = tuple(min(2 * t2, chain[i + 1]) for i in range(d // 2))
        spec = Spec(function, wman, k, rw, cf, d, tuple(t), t2, q, thr=thr)
        if d == 2:
            if max(t2, cw[2], q[0]) > narrow:
                continue
            if rw > narrow:
                # The wide product takes a multiplier's width of argument; the bits below multiply the top of c1, unless
                # dropping them costs about an LSB only.
                tc, sc = (rw - narrow, max(0, min(narrow - 2, cw[1] - 2))) if cw[1] - narrow > 1 else (0, 0)
                spec = replace(spec, t=(narrow,), tc=tc, sc=sc)
        elif d == 3:
            # c3 * w is one multiplier: c3 gives up its low bits if it is wider.
            z3 = max(0, cw[3] - narrow)
            for row in raw:
                row[3] = ((row[3] + ((1 << z3) >> 1)) >> z3) << z3
            spec = replace(spec, z3=z3, t=(rw, min(rw, narrow)))
            if z3 > spec.t[1]:  # c3 keeps no bit that its product takes
                continue
            if t2 > narrow:
                # The square takes two multipliers and a short sum: the square of the low part is left out.
                spec = replace(spec, th=narrow, sg=_SQUARE_GUARD)
                if t2 > 2 * narrow or 2 * t2 - q[0] - _SQUARE_GUARD < 2 * (t2 - narrow):
                    continue
            # A few argument bits fewer in the wide product can save a multiplier, if the budget allows it.
            short = narrow * (ceil(rw / narrow) - 1)
            if 0 < rw - short <= 3:
                shorter = replace(spec, t=(short, spec.t[1]))
                if _estimate(shorter, raw)[0] <= _SELECT:
                    spec = shorter
        share, coeffs = _estimate(spec, raw)
        if share > _SELECT:
            continue
        spec = replace(spec, coeffs=coeffs)
        # What the kernel relies on structurally; the certificate proves the numeric ranges.
        assert all(row[0] >> cf == 1 for row in coeffs), f"{function} at {wman}: c0 outside [2**cf, 2**(cf + 1))"
        assert function == "exp2" or 0 < thr < 1 << k, "log2: a re-centered fraction is nonzero"
        return spec
    raise UnsupportedFormat(
        f"no {function} table of {_K_CAP} bits of segment or fewer fits WMAN={wman} on {wmultiplier}-bit multipliers"
    )


@functools.cache
def _generated(function: str, wman: int, wmultiplier: int) -> Spec:
    return _generate(function, wman, wmultiplier)


_GENERATING = threading.Lock()


def _widths(spec: Spec) -> tuple[list[int], list[int]]:
    """
    Of each coefficient, the low degree first: its width, and that in the table, where `c0` lacks its leading one,
    which is constant, and a degree-3 `c3` its low zeros.
    """
    d, cw = spec.d, [max(row[j] for row in spec.coeffs).bit_length() for j in range(spec.d + 1)]
    return cw, [cw[0] - 1] + [cw[j] - (spec.z3 if d == 3 and j == 3 else 0) for j in range(1, d + 1)]


def _sizes(spec: Spec) -> dict[str, tuple[int, int]]:
    """The operand widths of each product of the kernel, by name."""
    d, t, half = spec.d, spec.t, spec.d // 2
    cw, fields = _widths(spec)
    chain = _chain_bits(_pairs(spec.function, d, spec.coeffs))
    sizes = {"p1": (cw[1], t[0])} | ({"pl": (cw[1] - spec.sc, spec.tc)} if spec.tc else {})
    sizes |= {f"p{2 * i + 1}": (fields[2 * i + 1], t[i]) for i in range(1, half + 1) if 2 * i + 1 <= d}
    sizes |= {"sq_hh": (spec.th, spec.th), "sq_hl": (spec.th, spec.t2 - spec.th)} if spec.th else {"sq": (spec.t2,) * 2}
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
    follows the pair's sum; when the result of each is formed (`sq` the square's); when pair `i` joins the chain, the
    top one at its start; the total.
    """

    stages: dict[str, int]
    post: frozenset[str]
    ready: dict[str, int]
    turn: dict[int, int]
    total: int


def _schedule(spec: Spec, wmultiplier: int, stages: int) -> _Schedule:
    d, half, caps = spec.d, spec.d // 2, _caps(spec, wmultiplier)
    n = {name: min(stages, cap) for name, cap in caps.items()}
    odd = [f"p{2 * i + 1}" for i in range(1, half + 1) if 2 * i + 1 <= d]
    post = frozenset(name for name in odd if caps[name] == 2)
    squares = [name for name in caps if name.startswith("sq")]
    assert len({n[name] for name in squares}) == 1, "the square's products are of one size"
    square = n[squares[0]]
    ready = {name: n[name] for name in caps if name in ("p1", "pl", *odd)}
    ready |= {name: max(0, n[name] - 1) for name in post} | {"sq": square}
    # The chain starts once the square and every pair are formed, that of a `post` product in its register.
    feed = max([ready["sq"], *(n[name] for name in odd)])
    turn, start = {half: feed}, feed
    for i in range(half - 1, -1, -1):
        start += n[f"p{2 * (i + 1)}"]
        if i:
            turn[i] = start
    ready["p2"] = start
    total = max(ready[name] for name in ("p1", "pl", "p2") if name in ready)
    return _Schedule(n, post, ready, turn, total)

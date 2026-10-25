import enum
import functools
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, overload

import amaranth.lib.enum
import mpmath
from amaranth.hdl import Array, Cat, Const, Module, Mux, Shape, Signal, signed, Value
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from ._fixed_to_float import FixedToFloat, fixed_to_float_reference
from ._block import Attrs, FloatLayout, FloatView, Operator, Timing, bundle, pipe
from ._pmul import Pmul, check, slices
from ._txn import Txn
from ._divsqrt import DivsqrtStep
from ._value import UnsupportedFormat, Zkf, ZkfFormat, bits_to_signed, context

_GUARD_DIV = 8
"""
Fraction bits of the quotient of vectoring's division beyond the significand's, so that the division and the bypass for
a small ratio round within an ULP; they set the division's digits. It also sets where that bypass starts, and the bypass
drops a term cubic in the ratio, so more is not better: this, and not the count of iterations, is what binds the angle.
"""
_GUARD_XY = 6
"""Fraction bits of `x` and `y` beyond one and a half significands: the angle of vectoring is bound by them."""
_GUARD_ZF, _GUARD_Z = 6, 3
"""Of the angle: fraction bits beyond those of the coordinate in turns, and integer bits above them."""


_SEED = 2
"""
The iteration that the engine starts at, before the seeds' own. A rotation's first two are constant, its folded angle
being in `[0, lut[0]]`; a vectoring's first adds (`y0 >= 0`) and its second subtracts unless `y0 == x0`. The load is
their result.
"""

_exact = context(400)
"""The context that computes the constants."""


class CordicResult[T](NamedTuple):
    r0: T
    r1: T
    quadrant: int


@dataclass(frozen=True, kw_only=True, eq=False)
class Cordic(Operator):
    """
    Rotation at `mode=0`, vectoring at `mode=1`, either by `vectoring` per transaction at `mode=2`, faithfully rounded;
    a fixed mode builds only its own and ignores `vectoring`.

    Rotation: `r0 = sin(2*pi*a)`, `r1 = cos(2*pi*a)`, `quadrant = floor(4 * (a mod 1))`; `b` is not read. A zero result
    is +0, and `a = +-inf` gives `r0 = r1 = +-inf`, of its sign, in quadrant 0.

    Vectoring: `r0 = atan2(a, b) / (2*pi)` in (-1/2, 1/2], `r1 = hypot(a, b)`, `quadrant = 0`. An angle below half the
    least normal is +0. A zero and an infinity give the angle of their axis, two infinities that of their diagonal, two
    zeros +0. `r1 = +inf` if an operand is infinite or the length as computed overflows the exponent; one that only
    rounds past `max` gives `max`.

    Not pipelined: a transaction is taken on `in_valid & in_ready`, its result appears under `out_valid` after the
    latency of its mode and stays until taken by `out_ready`. `in_ready` follows `out_ready` combinationally in that
    cycle, which can take the next, whatever the stages; where the product has no register, it rises a cycle later.

    One iterative CORDIC engine serves both modes, beside one multiplier, which the products take in turn, and one
    packer, which a rotation's two results take in turn unless `fast`; the constants are generated for the WMAN. About
    WMAN / 2 iterations suffice, as each mode finishes with one step linear in the residual angle, whose square is below
    an ULP.

    A rotation reduces `|a|` modulo a turn, exactly, to a quadrant and a coordinate within it, folds that into an eighth
    of a turn, and turns `(1/gain, 0)` by it, which leaves `(cos, sin)` and a small residual angle `z`. One linear
    rotation by `phi = 2*pi*z` finishes it with two corrections: `sin + cos*phi` and `cos - sin*phi`. An angle too small
    for the iterations gives `sin = 2*pi*angle` and `cos = 1` instead. The octant and the quadrant are then unfolded.

    A vectoring orders the operands by magnitude, aligns the smaller to the binade of the larger, and drives `y` to zero
    from a quarter of the vector, `z` gathering the angle and `x` growing to the gain times that length. The residual
    angle is finished by one division, `y / x` by the radix-4 digit step of `Divsqrt` at one digit a cycle, whose digits
    gather its product with `1/(2*pi)`; meanwhile the multiplier takes `x / gain`, which scaled back is the length.
    Near the positive x axis, where the angle is too small for the engine, the divider gives `|a| / |b|` from the
    significands, times `1/(2*pi)`. The octant and the quadrant are then unfolded into (-1/2, 1/2].
    """

    class CoreStages(enum.Flag):
        """The registers of `stage_core`, in path order: a rotation's, a vectoring's, then those of both."""

        SHIFT = enum.auto()
        """A rotation's shift amount, which reduces the operand modulo a turn."""
        REDUCED = enum.auto()
        """A rotation's reduced coordinate."""
        FOLDED = enum.auto()
        """A rotation's coordinate folded into an octant, ahead of the engine."""
        CORRECTIONS = enum.auto()
        """A rotation's corrections, ahead of their packing."""
        TAKEN = enum.auto()
        """A vectoring's operands as taken, with the differences of their exponents."""
        ORDERED = enum.auto()
        """A vectoring's smaller operand aligned to the larger."""
        SHIFTED = enum.auto()
        """What a vectoring's load takes, ahead of the engine."""
        ANGLE = enum.auto()
        """A vectoring's angle summed, ahead of the packer."""
        MEET = enum.auto()
        """Where the modes meet, ahead of the packers; at `mode=2` alone."""
        NORMALIZE_INPUT = enum.auto()
        """The magnitude to pack, ahead of each normalizer's count of its leading zeros."""
        NORMALIZE = enum.auto()
        """The magnitude with that count, ahead of each normalizer's shift."""
        PACK = enum.auto()
        """The packers' input: the class and exponent of each result, decided ahead of it, and the significand."""
        ROUNDED = enum.auto()
        """The packers' output."""

    operator = "cordic"
    unroll100: int = 200
    """Iterations per cycle in hundredths: 50 (two cycles each), 100 or 200."""
    stage_input: int = 0
    stage_core_mul: int = 0
    """As `Mul`'s; a rotation passes the product twice in series."""
    stage_core: CoreStages = CoreStages(0)
    """See `Cordic.CoreStages`; a fixed mode refuses those it lacks."""
    stage_output: int = 0
    wmultiplier: int
    """As `Mul`'s."""
    mode: int = 2
    fast: bool = False
    """A second packer, which packs a rotation's cosine of the octant beside its sine: a cycle less on a rotation."""
    rotation_seed: int = 2
    """
    Up to 2 of a rotation's iterations, resolved ahead of the engine by compares of its coordinate with constants; no
    effect at `mode=1`.
    """
    vectoring_seed: int = 1
    """1 takes a vectoring's first iteration into its load; no effect but at `unroll100=50` in a mode that vectors."""
    mul_attrs: Attrs | None = field(default=None, compare=False)
    """
    On the product's capture, where a register follows the multipliers: from `stage_core_mul=2` on for one multiplier,
    from 3 on for a grid.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")
        if self.wmultiplier < 8:
            raise ValueError(f"wmultiplier={self.wmultiplier} is below 8")
        if not 0 <= self.mode <= 2:
            raise ValueError(f"mode={self.mode} is outside 0..2")
        if self.mode and self.fmt.wexp < 5:
            raise UnsupportedFormat(f"WEXP={self.fmt.wexp} is below 5, which a vectoring takes")
        g = _geometry(self.fmt.wman)
        if self.unroll100 not in (50, 100, 200):
            raise ValueError(f"unroll100={self.unroll100} is not 50, 100 or 200")
        wa, wb = g.operands(self.mode)
        check("stage_core_mul", self.stage_core_mul, wa, wb, signed=self.mode != 1, wmultiplier=self.wmultiplier)
        if not 0 <= self.rotation_seed <= 2:
            raise ValueError(f"rotation_seed={self.rotation_seed} is outside 0..2")
        if not 0 <= self.vectoring_seed <= 1:
            raise ValueError(f"vectoring_seed={self.vectoring_seed} is outside 0..1")
        lacked = self.stage_core & _LACKED[self.mode]
        if lacked:
            raise ValueError(f"stage_core={self.stage_core}: leave out {lacked.name}, which mode={self.mode} lacks")

    @property
    def timing(self) -> Timing | Mapping[int, Timing]:
        if self.mode != 2:
            return self._timing_of(self.mode)
        return MappingProxyType({mode: self._timing_of(mode) for mode in (0, 1)})

    @property
    def signature(self) -> wiring.Signature:
        f = FloatLayout(self.fmt)
        return wiring.Signature(
            {
                "in_valid": In(1),
                "in_ready": Out(1),
                "vectoring": In(1),
                "a": In(f),
                "b": In(f),
                "out_valid": Out(1),
                "out_ready": In(1),
                "r0": Out(f),
                "r1": Out(f),
                "quadrant": Out(2),
            }
        )

    def component(self) -> _Cordic:
        return _Cordic(self)

    @overload
    def evaluate(self, vectoring: int, a: Zkf, b: Zkf, *, oracle: Literal[False] = False) -> CordicResult[Zkf]: ...
    @overload
    def evaluate(self, vectoring: int, a: Zkf, b: Zkf, *, oracle: Literal[True]) -> CordicResult[mpmath.mpf]: ...
    def evaluate(
        self, vectoring: int, a: Zkf, b: Zkf, *, oracle: bool = False
    ) -> CordicResult[Zkf] | CordicResult[mpmath.mpf]:
        """
        `oracle=True` gives the results themselves in place of the hardware's: mpf numbers of 4 * WMAN + 80 bits, which
        `fmt.encode` rounds correctly; a rounded angle of -1/2 is the operator's +1/2.
        """
        if a.fmt != self.fmt or b.fmt != self.fmt:
            raise ValueError(f"an operand is not of {self.fmt}")
        if not 0 <= vectoring <= 1:
            raise ValueError(f"vectoring={vectoring} is not 0 or 1")
        vectors = bool(vectoring) if self.mode == 2 else self.mode == 1
        if vectors:
            return _atan2_oracle(a, b) if oracle else _atan2(a, b)
        return _sincos_oracle(a) if oracle else _sincos(a)

    def _timing_of(self, mode: int) -> Timing:
        g = _geometry(self.fmt.wman)
        product = self.stage_core_mul
        registers = len(self.stage_core & ~(_VECTORING, _ROTATION)[mode])
        iterations = g.n - _SEED - _seeded(self)[mode]
        preloaded = int(self.unroll100 < 100)  # the load shifts for the first iteration
        cycles = (iterations * 100 + self.unroll100 - 1) // self.unroll100 - preloaded
        if mode == 0:
            held = not product  # the angle's product waits in `tprod`, which is then the multiplier's register
            saved = 0
            if self.unroll100 < 100:
                # The angle's product overlaps the engine, which runs `z` that many cycles ahead, and is held by its
                # end, when the sine is issued; one that is not back before the end delays the sine, a cycle more where
                # it is back in the end's own cycle, which the idle state does not see.
                lead = cycles - iterations
                saved = product + held if product + held < lead else lead - (product + held == lead)
            # The engine's load; the cosine a cycle behind the sine into the multiplier, and into the packer unless
            # `fast`.
            latency = 2 + (not self.fast) + held + cycles + 2 * product - saved - max(_ahead(self) - 1, 0)
        else:
            # The engine's load, the division's, and the last digit into the angle.
            latency = 3 + cycles + g.steps
        latency += registers + self.stage_input + self.stage_output
        return Timing(latency, latency if self.stage_core_mul else latency + 1)


_ROTATION = (
    Cordic.CoreStages.SHIFT | Cordic.CoreStages.REDUCED | Cordic.CoreStages.FOLDED | Cordic.CoreStages.CORRECTIONS
)
_VECTORING = Cordic.CoreStages.TAKEN | Cordic.CoreStages.ORDERED | Cordic.CoreStages.SHIFTED | Cordic.CoreStages.ANGLE
"""The members of `Cordic.CoreStages` on the path of one mode alone; `MEET` is on both."""
_LACKED = (_VECTORING | Cordic.CoreStages.MEET, _ROTATION | Cordic.CoreStages.MEET, Cordic.CoreStages(0))
"""The members of `Cordic.CoreStages` that each `mode` does not build."""


@dataclass(frozen=True)
class _Geometry:
    """The widths and counts that follow from WMAN in closed form."""

    wman: int
    n: int
    """Iterations: `ceil(WMAN / 2) + 2`, the fewest that keep each output faithful."""
    wff: int
    """Bits of the fraction of `x` in turns: the top two are the quadrant, the rest the coordinate within it."""
    xf: int
    zf: int

    @property
    def wt(self) -> int:
        return self.wff - 2

    @property
    def wx(self) -> int:
        return self.xf + 2

    @property
    def wz(self) -> int:
        return self.zf + _GUARD_Z

    @property
    def steps(self) -> int:
        """The radix-4 digits of the division of vectoring, which a bypass's quotient takes."""
        return (self.wman + _GUARD_DIV + 1) // 2

    @property
    def f2(self) -> int:
        """Fraction bits of the residual's quotient."""
        return 2 * ((self.xf + 1) // 2)

    @property
    def lead(self) -> int:
        """The residual quotient's top bits that the division skips, all zero: `|y|` starts shifted up by them."""
        return self.f2 - 2 * self.steps

    @property
    def wd(self) -> int:
        """
        Of the division's remainder and divisor: the quotient's digits and six bits more. Cut to it, `|y|` loses only
        zeros, and `x` raises the residual's quotient by under a twentieth of its ULP.
        """
        return 2 * self.steps + 6

    @property
    def wfactor(self) -> int:
        """Of each of the three factors."""
        return self.wman + 5

    @property
    def wop(self) -> int:
        """Of the coordinate that a rotation's linear path multiplies: it is below `2**(wop - 1)` there."""
        return max(self.wman, self.wt - (self.wman + 1) // 2)

    @property
    def wcp(self) -> int:
        """Of what a rotation's product with `const2pi` takes of the coordinate or of the residual angle."""
        return max(self.wop, self.zf - self.n) + 2

    @property
    def wphi(self) -> int:
        """Of the residual angle that a rotation's corrections take."""
        return min(self.wman + 6, max(self.xf - self.n + 2, 2))

    def operands(self, mode: int) -> tuple[int, int]:
        """
        Of the multiplier at `mode`: a rotation's are signed, a factor or a coordinate by an angle; a vectoring's
        unsigned, the length by a factor; both modes' as wide as either needs.
        """
        rotation = (max(self.wfactor + 1, self.wman + 6), max(self.wcp, self.wphi))
        vectoring = (self.wx - 1, self.wfactor)
        return (rotation, vectoring, (max(rotation[0], vectoring[0] + 1), max(rotation[1], vectoring[1] + 1)))[mode]


def _geometry(wman: int) -> _Geometry:
    if not 16 <= wman <= 53:
        raise UnsupportedFormat(f"no cordic for WMAN={wman}: its constants are generated for 16 to 53")
    wff = wman + max(12, wman // 2 + 2)  # where the linear path takes over, `1 - cos` being below an ULP there
    xf = (3 * wman + 1) // 2 + _GUARD_XY
    g = _Geometry(wman=wman, n=(wman + 1) // 2 + 2, wff=wff, xf=xf, zf=wff + _GUARD_ZF)
    # The engine leaves `|y|` within `x * 2**(1 - n)` and a unit an iteration, so a lead of `n - 2` keeps the first
    # remainder below the divisor.
    assert 0 <= g.lead <= g.n - 2 and g.wd < g.wx, g
    return g


@dataclass(frozen=True)
class _Constants(_Geometry):
    """
    `lut`, the angle of each iteration in turns at `2**-zf`; `kinv`, the inverse gain at `2**-xf`, which seeds a
    rotation; three factors of `wman + 5` bits, each at its own scale (`const2pi_s`, `invtau_s`, `kinv_s`):
    `const2pi = 2*pi`, `inv_tau = 1 / (2*pi)` and `kinv_mag`, the inverse gain that descales a length. `tsa` is the
    coordinate below which a rotation is linear. `least` is the fewest bits of a nonzero magnitude that the packer takes
    in rotation and in vectoring, which bounds what its normalizer shifts.
    """

    kinv: int
    lut: tuple[int, ...]
    const2pi: int
    const2pi_s: int
    inv_tau: int
    invtau_s: int
    kinv_mag: int
    kinv_s: int
    tsa: int
    least: tuple[int, int]


@functools.cache
def _constants(wman: int) -> _Constants:
    """The constants at WMAN, generated once."""
    g = _geometry(wman)
    gain = _exact.mpf(1)
    for i in range(g.n):
        gain *= _exact.sqrt(1 + _exact.mpf(2) ** (-2 * i))
    turn = 2 * _exact.pi

    def scaled(value: Any, scale: int) -> int:
        return int(_exact.nint(value * _exact.mpf(2) ** scale))

    def bits(value: Any) -> int:
        return int(_exact.floor(_exact.log(value, 2))) + 1

    const2pi, inv_tau, kinv_mag = scaled(turn, wman + 2), scaled(1 / turn, wman + 7), scaled(1 / gain, wman + 5)
    tsa = 1 << (g.wt + 2 - (wman + 1) // 2 - 3)
    slack = 4 * g.n  # units of what the engine and its corrections may lose
    # A rotation's least: the sine at the linear threshold, `const2pi` times the least coordinate past the tiny ones and
    # times a significand, and the cosine. A vectoring's: the angle at the bypass's threshold with the least ratio of
    # significands, the bypass's product with a ratio of a half, and the length of the least operand.
    sine = bits(_exact.sin(turn * tsa / _exact.mpf(2) ** (g.wt + 2)) * _exact.mpf(2) ** g.xf - slack)
    linear = min((const2pi << (g.wt - wman)).bit_length(), (const2pi << (wman - 1)).bit_length())
    ratio = _exact.mpf(2) ** (_GUARD_DIV + wman - g.zf) * (1 << (wman - 1)) / ((1 << wman) - 1)
    angle = bits(_exact.atan(ratio) / turn * _exact.mpf(2) ** g.zf - slack)
    length = bits(gain * _exact.mpf(2) ** (g.xf - 2) * kinv_mag * _exact.mpf(0.999))
    bypass = (inv_tau << (2 * g.steps - 1)).bit_length()
    return _Constants(
        **asdict(g),
        kinv=scaled(1 / gain, g.xf),
        lut=tuple(scaled(_exact.atan(_exact.mpf(2) ** -i) / turn, g.zf) for i in range(g.n)),
        const2pi=const2pi,
        const2pi_s=wman + 2,
        inv_tau=inv_tau,
        invtau_s=wman + 7,
        kinv_mag=kinv_mag,
        kinv_s=wman + 5,
        tsa=tsa,
        least=(min(sine, linear, g.xf), min(angle, bypass, length)),
    )


def _ahead(model: Cordic) -> int:
    """
    How many iterations ahead of the engine's end a rotation issues the angle's product at one iteration a cycle,
    `stage_core_mul + 1`, so that the angle, less a constant of the signs taken since, is held by the end; 0 where the
    multiplier has no register, the wider angle would take another slice, or the product would start ahead of the
    rotation's first iteration.
    """
    k = model.stage_core_mul + 1
    if model.unroll100 != 100 or not model.stage_core_mul or model.mode == 1:
        return 0
    c = _constants(model.fmt.wman)
    if c.n - k < _SEED + _seeded(model)[0]:
        return 0
    wa, wb = c.operands(model.mode)
    grid = functools.partial(slices, wa, signed=True, wmultiplier=model.wmultiplier)
    return k if grid(max(wb, _angle_bits(c, k))) == grid(wb) else 0


def _seeded(model: Cordic) -> tuple[int, int]:
    """The iterations that each mode resolves ahead of the engine, of `rotation_seed` and `vectoring_seed`."""
    return (
        model.rotation_seed if model.mode != 1 else 0,
        model.vectoring_seed if model.mode != 0 and model.unroll100 == 50 else 0,
    )


def _seed(c: _Constants, signs: int) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int]]:
    """
    A rotation's state at iteration `_SEED + signs`: for each interval of the folded coordinate `tpw` below half a
    quadrant, its lower end and the vector and the constant that `z1` takes then; and the vector and `z` of `halfway`.
    """
    lut, half = c.lut, 1 << (c.wt - 1)
    entries: list[tuple[int, int, int, int]] = []

    def follow(j: int, x: int, y: int, offset: int, low: int, high: int) -> None:
        if j == _SEED + signs:
            entries.append((low, x, y, lut[1] + offset))
            return
        cut = min(max(-((offset + lut[1] - lut[0]) >> _GUARD_ZF), low), high)  # where `z_j` turns nonnegative
        assert low < cut < high  # at two signs at the most, every interval splits
        follow(j + 1, x + (y >> j), y - (x >> j), offset + lut[j], low, cut)
        follow(j + 1, x - (y >> j), y + (x >> j), offset - lut[j], cut, high)

    follow(_SEED, c.kinv + (c.kinv >> 1), c.kinv - (c.kinv >> 1), 0, 0, half)
    x, y, z = c.kinv - (c.kinv >> 1), c.kinv + (c.kinv >> 1), -lut[1]
    for j in range(_SEED, _SEED + signs):
        x, y, z = (x + (y >> j), y - (x >> j), z + lut[j]) if z < 0 else (x - (y >> j), y + (x >> j), z - lut[j])
    return entries, (x, y, z)


def _angle_bits(c: _Constants, k: int) -> int:
    """Of the signed angle `k` iterations before the end, which is within the angles of those left plus the last."""
    return (sum(c.lut[c.n - k :]) + c.lut[-1]).bit_length() + 1


def _sincos(a: Zkf) -> CordicResult[Zkf]:
    """A rotation: `(sin(2 * pi * a), cos(2 * pi * a), quadrant)`."""
    fmt = a.fmt
    if a.is_inf:
        s = fmt.inf(a.negative).bits
        return CordicResult(Zkf(fmt, s), Zkf(fmt, s), 0)
    if a.is_zero:
        return CordicResult(Zkf(fmt, 0), fmt.pack(0, fmt.bias, 0), 0)

    spec = _constants(fmt.wman)
    xf, zf, wt = spec.xf, spec.zf, spec.wt
    const2pi, const2pi_s = spec.const2pi, spec.const2pi_s
    ff = wt + 2
    # One magnitude width, that of the widest product, `const2pi * t'`; the offsets of the exponent of a magnitude at
    # the scale 2**-xf and of a product with const2pi.
    wmag = const2pi.bit_length() + wt + 1
    eone_xf = wmag - 1 - xf
    eone_s = wmag - 1 - const2pi_s
    one = (1 << xf, eone_xf)
    tsa = spec.tsa
    sig = a.significand()
    e = a.exp - fmt.bias

    # |x| mod 1 in `ff` bits: sin(2*pi*x) = sign * sin(2*pi*|x|) and cos(2*pi*x) = cos(2*pi*|x|).
    sh = e - fmt.wfrac + ff
    tiny = sh < 0  # below the reducer's resolution -> small-angle path on |x|
    lshamt = 0 if tiny else min(sh, ff)
    frac_pos = (sig << lshamt) & ((1 << ff) - 1)
    quadrant_abs = 0 if tiny else (frac_pos >> wt) & 3  # |x| quadrant (0 for tiny: |x| < 1/4)
    t = sig if tiny else (frac_pos & ((1 << wt) - 1))  # quadrant-local coordinate, scale 2**-WT
    tzero = (not tiny) and t == 0  # frac(|x|)*4 integer: a quadrant boundary / exact magnitude

    # The local angle folded into [0, pi/4].
    half = 1 << (wt - 1)
    oct_flip = (not tiny) and t > half  # theta in (pi/4, pi/2): use the pi/2 - theta complement
    tp = ((1 << wt) - t) if oct_flip else t

    # (sin theta', cos theta') as pairs of a magnitude and an exponent offset.
    if tiny:
        sin_tp = (const2pi * sig, eone_s + e - fmt.wfrac)  # sin ~= 2 * pi * |x|
        cos_tp = one
    elif tzero:
        sin_tp, cos_tp = (0, eone_xf), one
    elif tp < tsa:
        sin_tp = (const2pi * tp, eone_s - ff)  # sin ~= 2 * pi * t'
        cos_tp = one
    else:
        # On operands of WMAN + 6 bits, which leaves under an ULP.
        xk, yk, zk = _cordic_rotate(spec, tp << (zf - ff))
        phi_trunc = max(0, xf - spec.n + 2 - (fmt.wman + 6))
        phi_s = xf - phi_trunc
        xk_trunc = xf + 2 - (fmt.wman + 6)
        phi = (const2pi * zk) >> ((const2pi_s + zf) - phi_s)
        corr_s = ((xk >> xk_trunc) * phi) >> (xf - xk_trunc - phi_trunc)
        corr_c = ((yk >> xk_trunc) * phi) >> (xf - xk_trunc - phi_trunc)
        sin_tp = (yk + corr_s, eone_xf)
        cos_tp = (xk - corr_c, eone_xf)

    # The octant unfolded, then the quadrant of |x|.
    sin_loc, cos_loc = (cos_tp, sin_tp) if oct_flip else (sin_tp, cos_tp)
    sin_m, cos_m = (cos_loc, sin_loc) if (quadrant_abs & 1) else (sin_loc, cos_loc)
    sin_sign = ((quadrant_abs >> 1) & 1) ^ a.negative  # sin is odd: negative x flips it
    cos_sign = ((quadrant_abs >> 1) ^ quadrant_abs) & 1  # cos is even: unchanged by the sign of x
    # floor(frac(x) * 4): for x < 0 that of |x| reflected about a turn.
    if not a.negative:
        quadrant = quadrant_abs
    else:
        quadrant = ((4 - quadrant_abs) & 3) if tzero else (3 - quadrant_abs)

    assert all(not mag or mag.bit_length() >= spec.least[0] for mag in (sin_m[0], cos_m[0])), "the packer's bound"
    assert 1 << (xf - 1) <= cos_tp[0] <= 1 << xf, "the bound of the cosine's packer"
    sin_bits = fixed_to_float_reference(fmt, sin_sign, sin_m[0], sin_m[1], wmag)
    cos_bits = fixed_to_float_reference(fmt, cos_sign, cos_m[0], cos_m[1], wmag)
    return CordicResult(Zkf(fmt, sin_bits), Zkf(fmt, cos_bits), quadrant)


def _atan2(y: Zkf, x: Zkf) -> CordicResult[Zkf]:
    """A vectoring: the angle of the vector `(x, y)` in turns, from -1/2 exclusive to 1/2, and its length."""
    fmt = y.fmt
    sp = _atan2_special(y, x)
    if sp is not None:
        return CordicResult(Zkf(fmt, sp[0]), Zkf(fmt, sp[1]), 0)

    spec = _constants(fmt.wman)
    xf, zf, wfrac = spec.xf, spec.zf, fmt.wfrac
    kinv_mag, kinv_s = spec.kinv_mag, spec.kinv_s
    inv_tau, invtau_s = spec.inv_tau, spec.invtau_s

    sx, sy = x.negative, y.negative
    sig_x, sig_y = x.significand(), y.significand()
    ex, ey = x.exp - fmt.bias, y.exp - fmt.bias

    # Order by magnitude: den = max(|x|,|y|), num = min. The reduced octant angle a0 = atan(num/den) in [0, 1/8].
    swap = (ey > ex) or (ey == ex and sig_y > sig_x)  # |y| > |x|
    if swap:
        den_sig, e_den, num_sig, e_num = sig_y, ey, sig_x, ex
    else:
        den_sig, e_den, num_sig, e_num = sig_x, ex, sig_y, ey
    shift_dn = e_den - e_num  # >= 0 (den >= num)
    # A quarter of the vector, so that x_K = gain * hypot stays inside the engine's width; the length takes it back.
    den_fixed = den_sig << (xf - wfrac - 2)  # in [0.25, 0.5) * 2**xf
    num_fixed = (num_sig << (xf - wfrac - 2)) >> shift_dn  # num aligned to den's scale 2**-xf

    x_k, y_k, z_k = _cordic_vector(spec, den_fixed, num_fixed)
    wmag = 2 * xf + 4
    exp_off_m = (wmag - 1) - (xf + kinv_s) + e_den + 2
    assert (x_k * kinv_mag).bit_length() >= spec.least[1], "the packer's bound"
    mag_bits = fixed_to_float_reference(fmt, 0, x_k * kinv_mag, exp_off_m, wmag, saturate_round_carry=True)

    # The bypass: the quotient truncated as the divider's, at the digits, and its sticky bit beside the product.
    if (not swap) and (sx == 0) and (shift_dn > _atan2_bypass_shift(fmt)):
        fb = 2 * spec.steps
        r, rest = divmod(num_sig << fb, den_sig)
        exp_off_b = (wmag - 1) + (e_num - e_den) - fb - invtau_s
        assert (r * inv_tau).bit_length() >= spec.least[1], "the packer's bound"
        theta_bits = fixed_to_float_reference(
            fmt, sy, (r * inv_tau) | bool(rest), exp_off_b, wmag, saturate_round_carry=True
        )
        return CordicResult(Zkf(fmt, theta_bits), Zkf(fmt, mag_bits), 0)

    a0 = _vector_residual(spec, x_k, y_k, z_k)

    # The octant and the quadrant unfolded.
    quarter = 1 << (zf - 2)
    half = 1 << (zf - 1)
    phi1 = (quarter - a0) if swap else a0
    theta_mag = (half - phi1) if sx else phi1
    wmag_t = zf + 2
    assert not theta_mag or theta_mag.bit_length() >= spec.least[1], "the packer's bound"
    theta_bits = fixed_to_float_reference(fmt, sy, theta_mag, wmag_t - 1 - zf, wmag_t, saturate_round_carry=True)
    return CordicResult(Zkf(fmt, _atan2_canon_half(fmt, theta_bits)), Zkf(fmt, mag_bits), 0)


def _cordic_rotate(spec: _Constants, z0: int) -> tuple[int, int, int]:
    """
    The engine's rotation: (x_K, y_K, z_K), the vector at the scale 2**-xf and the residual angle in
    turns at 2**-zf. The inverse gain is the seed of x; a shift floors.
    """
    kinv, lut = spec.kinv, spec.lut
    x, y, z = kinv, 0, z0
    for i in range(spec.n):
        if z < 0:  # sigma = -1
            x, y, z = x + (y >> i), y - (x >> i), z + lut[i]
        else:  # sigma = +1
            x, y, z = x - (y >> i), y + (x >> i), z - lut[i]
    return x, y, z


def _cordic_vector(spec: _Constants, x0: int, y0: int) -> tuple[int, int, int]:
    """
    The engine's vectoring, which drives y to 0: (x_K, y_K, z_K), x_K being the length times the gain
    at the scale 2**-xf and z_K the angle in turns at 2**-zf, each wrapped to the engine's width.
    """
    lut, xw, zw = spec.lut, spec.wx, spec.wz
    x = bits_to_signed(x0 & ((1 << xw) - 1), xw)
    y = bits_to_signed(y0 & ((1 << xw) - 1), xw)
    z = 0
    for i in range(spec.n):
        if y >= 0:
            nx, ny, nz = x + (y >> i), y - (x >> i), z + lut[i]
        else:
            nx, ny, nz = x - (y >> i), y + (x >> i), z - lut[i]
        x = bits_to_signed(nx & ((1 << xw) - 1), xw)
        y = bits_to_signed(ny & ((1 << xw) - 1), xw)
        z = bits_to_signed(nz & ((1 << zw) - 1), zw)
    return x, y, z


def _vector_residual(spec: _Constants, x_k: int, y_k: int, z_k: int) -> int:
    """
    The vectoring's reduced angle `z_K + (y_K / x_K) / (2 * pi)` at the scale of the angle, `x_K > 0`; the quotient is
    truncated as the divider's, at `f2` fraction bits, of operands cut to the division's width.
    """
    cut = spec.wx - spec.wd
    q = (((abs(y_k) << spec.lead) >> cut) << (2 * spec.steps)) // (x_k >> cut)
    delta = (q * spec.inv_tau) >> (spec.f2 + spec.invtau_s - spec.zf)
    return z_k - delta if y_k < 0 else z_k + delta


def _atan2_turn(fmt: ZkfFormat, sign: int, frac: Fraction) -> int:
    """A signed turn of a dyadic fraction."""
    return fmt.encode(-frac if sign else frac).bits


def _atan2_canon_half(fmt: ZkfFormat, theta_bits: int) -> int:
    """Minus a half turn, the limit on the negative x axis, as the half turn: the angle is in (-1/2, 1/2]."""
    pos_half = _atan2_turn(fmt, 0, Fraction(1, 2))
    return pos_half if theta_bits == (pos_half | (1 << fmt.sign_shift)) else theta_bits


def _atan2_special(y: Zkf, x: Zkf) -> tuple[int, int] | None:
    """
    The bits of (theta, length) where an operand is zero or infinite, for the reference and the oracle alike; None where
    both are finite and nonzero.
    """
    fmt = y.fmt
    if x.is_inf or y.is_inf:
        mag = fmt.inf(0).bits
        if x.is_inf and y.is_inf:
            return _atan2_turn(fmt, y.negative, Fraction(3, 8) if x.negative else Fraction(1, 8)), mag
        if y.is_inf:
            return _atan2_turn(fmt, y.negative, Fraction(1, 4)), mag
        if x.negative:
            return _atan2_turn(fmt, 0, Fraction(1, 2)), mag
        return 0, mag
    if x.is_zero and y.is_zero:
        return 0, 0
    if y.is_zero:
        theta = _atan2_turn(fmt, 0, Fraction(1, 2)) if x.negative else 0
        return theta, fmt.pack(0, x.exp, x.frac).bits
    if x.is_zero:
        return _atan2_turn(fmt, y.negative, Fraction(1, 4)), fmt.pack(0, y.exp, y.frac).bits
    return None


def _atan2_bypass_shift(fmt: ZkfFormat) -> int:
    return _geometry(fmt.wman).zf - fmt.wman - _GUARD_DIV


def _sincos_oracle(z: Zkf) -> CordicResult[mpmath.mpf]:
    """
    The oracle of a rotation. The phase is reduced mod 1 exactly and folded into an octant before mpmath sees it, which
    avoids the cancellation of a large argument and keeps the zeros at quadrant boundaries exact.
    """
    if z.is_inf:
        return CordicResult(z.to_mpf(), z.to_mpf(), 0)
    if z.is_zero:
        return CordicResult(mpmath.mpf(0), mpmath.mpf(1), 0)
    fmt = z.fmt
    e = z.exp - fmt.bias
    prec = 4 * fmt.wman + 80
    ctx = context(prec)
    rsh = fmt.wfrac - e  # |x| = sig / 2**rsh
    frac_abs = Fraction(0) if rsh <= 0 else Fraction(z.significand() % (1 << rsh), 1 << rsh)
    frac = (1 - frac_abs) if (z.negative and frac_abs != 0) else frac_abs  # frac(x) in [0,1)
    q4 = frac * 4
    quadrant = int(q4)
    t_local = q4 - quadrant  # in [0,1)
    if t_local <= Fraction(1, 2):
        theta = (ctx.pi / 2) * ctx.mpf(t_local.numerator) / ctx.mpf(t_local.denominator)
        s0, c0 = ctx.sin(theta), ctx.cos(theta)
    else:
        comp = 1 - t_local
        theta = (ctx.pi / 2) * ctx.mpf(comp.numerator) / ctx.mpf(comp.denominator)
        s0, c0 = ctx.cos(theta), ctx.sin(theta)
    sin_mag, cos_mag = (c0, s0) if (quadrant & 1) else (s0, c0)
    sin_v = -sin_mag if (quadrant >> 1) & 1 else sin_mag
    cos_v = -cos_mag if ((quadrant >> 1) ^ quadrant) & 1 else cos_mag
    return CordicResult(mpmath.mpf(sin_v, prec=prec), mpmath.mpf(cos_v, prec=prec), quadrant)


def _atan2_oracle(y: Zkf, x: Zkf) -> CordicResult[mpmath.mpf]:
    """The oracle of a vectoring: the angle in turns and the length."""
    fmt = y.fmt
    sp = _atan2_special(y, x)
    if sp is not None:
        return CordicResult(Zkf(fmt, sp[0]).to_mpf(), Zkf(fmt, sp[1]).to_mpf(), 0)
    prec = 4 * fmt.wman + 80
    ctx = context(prec)
    yv, xv = ctx.convert(y.to_mpf()), ctx.convert(x.to_mpf())
    if abs(yv) == abs(xv):  # +-1/8 or +-3/8 turn exactly; the generic quotient can miss 3/8 by an ULP of its own
        theta = ctx.mpf(1 if xv > 0 else 3) / 8
        theta = -theta if yv < 0 else theta
    else:
        theta = ctx.atan2(yv, xv) / (2 * ctx.pi)
    return CordicResult(mpmath.mpf(theta, prec=prec), mpmath.mpf(ctx.hypot(yv, xv), prec=prec), 0)


def _zext(value: Value, width: int) -> Value:
    """With zeros above, unsigned."""
    return Cat(value.as_unsigned(), Const(0, width - len(value)))


def _sext(value: Value, width: int) -> Value:
    """With the sign above, signed."""
    return Cat(value, value[-1].replicate(width - len(value))).as_signed()


class _Product(amaranth.lib.enum.Enum, shape=2):  # type: ignore[call-arg]
    """The sideband tags of a rotation's products: the linear path, the residual angle, and the two corrections."""

    LINEAR = 0
    ANGLE = 1
    SINE = 2
    COSINE = 3


@dataclass(frozen=True)
class _Share:
    """What a mode drives into the shared engine, multiplier and packer, at their widths, and its results."""

    start: Value
    x0: Value
    y0: Value
    z0: Value
    neg0: Value
    xa: Value
    ya: Value
    pm_iv: Value
    pm_sb: Value
    pm_a: Value
    pm_b: Value
    pk_iv: Value
    pk_sign: Value
    pk_inf: Value
    pk_exp: Value
    pk_mag: Value
    pk_sb: Value
    res: Value
    r0: Value
    r1: Value
    quadrant: Value


_RAISED = ("start", "pm_iv", "pk_iv", "res")
"""The members of `_Share` that either mode raises; the rest are of the mode that the transaction is in."""
_TAKING = ("x0", "y0", "z0", "neg0", "xa", "ya", "pm_sb", "pm_a", "pm_b")
"""The members of `_Share` that a transaction drives in the cycle it is taken in, where the engine can start then."""


class _Cordic(wiring.Component):
    in_valid: Signal
    in_ready: Signal
    vectoring: Signal
    a: FloatView
    b: FloatView
    out_valid: Signal
    out_ready: Signal
    r0: FloatView
    r1: FloatView
    quadrant: Signal

    def __init__(self, model: Cordic) -> None:
        self._model, self._numbers = model, _constants(model.fmt.wman)
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, fmt, mode, c = self._model, self._model.fmt, self._model.mode, self._numbers
        wexp, wman, wfrac, bias = fmt.wexp, fmt.wman, fmt.wfrac, fmt.bias
        we = wexp + 1  # of the signed unbiased exponent
        n, wff, wt, xf, wx, zf, wz = c.n, c.wff, c.wt, c.xf, c.wx, c.zf, c.wz
        wfactor = c.wfactor
        const2pi_s, invtau_s, kinv_s = c.const2pi_s, c.invtau_s, c.kinv_s
        rotates, vectors, both = mode != 1, mode != 0, mode == 2
        seeded = _seeded(model)
        f = FloatLayout(fmt)

        # A rotation's products keep the top bits of the residual angle and of what the engine leaves.
        tsa_bits = c.tsa.bit_length() - 1  # the coordinate below `2**tsa_bits` takes the linear path
        wop, wcp, wphi = c.wop, c.wcp, c.wphi
        wxc = wman + 6
        wa_rot, wb_rot = c.operands(0)
        ahead = _ahead(model)
        wza = max(wcp, _angle_bits(c, ahead)) if ahead else wcp  # of the angle that the product takes
        wb_rot = max(wb_rot, wza)
        wmag_rot = wfactor + wt + 1  # the full product of the linear path is the widest
        weu_rot = ceil_log2(wmag_rot + 1) + 2

        steps = c.steps
        wquo = 2 * steps + 1  # the quotient of two significands, which the bypass divides, can exceed 1
        wa_mag = wx - 1  # what the engine leaves of `x` in vectoring is positive
        wa_mul = c.operands(1)[0]
        wmag_vec = max(wa_mag + wfactor, wquo + wfactor, wz + 2)

        # What both modes share is as wide as either needs: a signed multiplier, with a bit more for the unsigned
        # operands of vectoring, and a packer that takes a biased exponent.
        wa_pm, wb_pm = c.operands(mode)
        wb_pm = max(wb_pm, wb_rot) if rotates else wb_pm
        # The tags beside a rotation's product and a packed result: each mode reads its own, in its own transactions.
        theta = data.StructLayout({"theta": 1})
        quadrant = data.StructLayout({"quadrant": 2} | ({} if model.fast else {"second": 1}))
        pk_tags = data.UnionLayout(
            ({"rotation": quadrant} if rotates else {}) | ({"vectoring": theta} if vectors else {})
        )
        wmag_pk = (wmag_rot, wmag_vec, max(wmag_rot, wmag_vec))[mode]
        weu_pk = wexp + ceil_log2(wmag_pk + 1) + 3 if vectors else max(weu_rot, wexp + 3)

        core, stages = model.stage_core, model.stage_core_mul
        CoreStages = Cordic.CoreStages

        result = data.StructLayout({"quadrant": 2, "r1": f, "r0": f})
        m.submodules.txn = txn = Txn(result, min(model.stage_output, 1), overlap=bool(stages))
        m.d.comb += txn.in_valid.eq(self.in_valid)
        m.d.comb += self.in_ready.eq(txn.in_ready)
        m.d.comb += txn.out_ready.eq(self.out_ready)
        m.d.comb += self.out_valid.eq(txn.out_valid)
        m.d.comb += self.r0.eq(txn.out.r0)
        m.d.comb += self.r1.eq(txn.out.r1)
        m.d.comb += self.quadrant.eq(txn.out.quadrant)

        # The tags of the two modes overlap, so each drives and reads what is shared in its own transactions only. What
        # a transaction gives back is selected by the mode held, which a transaction taken in that cycle does not reach.
        own_vec: Value = Signal(1, name="vec", reset_less=True) if both else Const(mode, 1)
        held_vec = own_vec
        if both:
            with m.If(txn.accept):
                m.d.sync += own_vec.eq(self.vectoring)
            fronts = (
                CoreStages.SHIFT | CoreStages.REDUCED | CoreStages.FOLDED,
                CoreStages.TAKEN | CoreStages.ORDERED | CoreStages.SHIFTED,
            )
            if not model.stage_input and not all(core & front for front in fronts):
                # A transaction can start the engine in the cycle it is taken in.
                now_vec = Signal(name="vec_now")
                m.d.comb += now_vec.eq(Mux(txn.accept, self.vectoring, own_vec))
                own_vec = now_vec

        # A product without a register comes back in the cycle it is issued in, maybe that of the transaction's taking.
        product_vec = held_vec if stages else own_vec
        taken = data.StructLayout({"b": f, "a": f, "vec": 1})
        offered = bundle(taken, b=self.b, a=self.a, vec=self.vectoring if both else Const(mode, 1))
        si_valid, si = pipe(m, "in", txn.accept, taken, offered, model.stage_input)

        m.submodules.engine = engine = _Engine(
            c.lut,
            wx=wx,
            wz=wz,
            unroll100=model.unroll100,
            mode=mode,
            ahead=ahead,
            seeds=(_SEED + seeded[0], _SEED + seeded[1]),
        )
        m.submodules.pmul = pmul = Pmul(
            wa_pm,
            wb_pm,
            signed=rotates,
            wmultiplier=model.wmultiplier,
            stages=stages,
            sideband=_Product if rotates else None,
            attrs=model.mul_attrs,
        )
        least = min(c.least[side] for side in ((0,), (1,), (0, 1))[mode])
        # The same stages keep the packers in step.
        packer = functools.partial(
            FixedToFloat,
            fmt,
            classify_ahead=True,
            exp_is_biased=True,
            stage_input=int(CoreStages.NORMALIZE_INPUT in core),
            stage_normalize=int(CoreStages.NORMALIZE in core),
            stage_pack=int(CoreStages.PACK in core),
            stage_output=int(CoreStages.ROUNDED in core),
        )
        m.submodules.f2f = f2f = packer(
            wmag=wmag_pk,
            weu=weu_pk,
            assume_no_overflow=not vectors,
            saturate_round_carry=vectors,
            sideband=pk_tags,
            most=wmag_pk - least,
        )
        a_zero = Signal(name="a_zero")
        m.d.comb += a_zero.eq(si.a.is_zero)
        a_inf = Signal(name="a_inf")
        m.d.comb += a_inf.eq(si.a.is_inf)
        pm_valid, pm_p = pmul.out_valid, pmul.p
        pm_sb = pmul.sb_out if rotates else None
        pk_valid, pk_y, pk_sb = f2f.out_valid, f2f.y, f2f.sb_out

        def rotation() -> _Share:
            wp = wa_rot + wb_rot
            # A magnitude is at the scale `2**-xf`, for a correction and for the exact 1, or at that of `const2pi`, for
            # its products; the exponents are biased here, so that the packer adds no bias behind its count.
            eone_xf = wmag_pk - 1 - xf + bias
            eone_s = wmag_pk - 1 - const2pi_s + bias
            phi_trunc = max(xf - n + 2 - wphi, 0)
            xk_trunc = wx - wxc
            corr_shift = xf - xk_trunc - phi_trunc
            sh_base = bias - (wff - wman) - 1
            wlsh = ceil_log2(wff + 1)

            # The reduction of `x` modulo a turn. The clamps of the shift compare the exponent with constants, beside
            # the subtraction and not behind it.
            e0 = si.a.exp
            low, high = Signal(name="e_lo"), Signal(name="e_hi")
            m.d.comb += [low.eq(e0 < sh_base), high.eq(e0 > sh_base + wff)]
            # A zero is shifted out whole rather than masked, which leaves no mask for a set or reset to take.
            lshamt = Signal(wlsh, name="lshamt")
            with m.Switch(Cat(high, low, a_zero)):
                with m.Case("1--", "0-1"):
                    m.d.comb += lshamt.eq(Const(wff, wlsh))
                with m.Case("010"):
                    m.d.comb += lshamt.eq(0)
                with m.Default():
                    m.d.comb += lshamt.eq((_zext(e0, max(wexp, wlsh)) - sh_base)[:wlsh])
            flags = {"sign": 1, "inf": 1, "tiny": 1}
            amount = data.StructLayout(flags | {"sig": wman, "lshamt": wlsh, "e": signed(we)})
            given = bundle(
                amount, sign=si.a.sign, inf=a_inf, tiny=low, sig=si.a.significand, lshamt=lshamt, e=(e0 - bias)[:we]
            )
            shift_valid, sa = pipe(m, "shift", si_valid & ~si.vec, amount, given, int(CoreStages.SHIFT in core))

            frac_pos = Signal(wff, name="frac_pos")
            m.d.comb += frac_pos.eq((_zext(sa.sig, wff) << sa.lshamt)[:wff])
            reduction = data.StructLayout(flags | {"quad": 2, "t": wt, "e": signed(we)})
            quad = Mux(sa.inf, Const(0, 2), frac_pos[wff - 2 :])
            given = bundle(reduction, sign=sa.sign, inf=sa.inf, tiny=sa.tiny, quad=quad, t=frac_pos[:wt], e=sa.e)
            reduced_valid, rd = pipe(m, "reduced", shift_valid, reduction, given, int(CoreStages.REDUCED in core))

            # The complement's increment keeps a small reflected sine accurate next to the fold.
            flip = Signal(name="oct_flip")
            m.d.comb += flip.eq(rd.t > (1 << (wt - 1)))
            # At `mode` 2 `halfway` is one more interval, which takes the mux behind the start's add out of the shared
            # load; alone the mode keeps it apart.
            merged = both
            m.submodules.seed = seed = _Seed(c, seeded[0], merged)
            m.d.comb += seed.t.eq(rd.t)
            entries, (x_half, y_half, z_half) = seed.entries, seed.halfway_state
            layout = {"flip": 1, "tpw": wt, "halfway": 1, "tzero": 1, "quad": 2, "e": signed(we)}
            layout |= {"picks": len(seed.picks), "z_take": signed(wz)}
            layout |= {"x_take": signed(wx), "y_take": signed(wx)} if merged else {}
            octant = data.StructLayout(flags | layout)
            given = bundle(
                octant,
                sign=rd.sign,
                inf=rd.inf,
                tiny=rd.tiny,
                flip=flip,
                tpw=Mux(flip, (~rd.t + 1)[:wt], rd.t),
                halfway=seed.halfway,
                tzero=~rd.t.any(),
                quad=rd.quad,
                e=rd.e,
                picks=seed.picks,
                z_take=seed.z_take,
                **({"x_take": seed.x_take, "y_take": seed.y_take} if merged else {}),
            )
            folded_valid, fd = pipe(m, "folded", reduced_valid, octant, given, int(CoreStages.FOLDED in core))
            # The angle after the first iteration, `tpw - lut[0]` with `lut[0]` half a quadrant, which bounds `tpw`.
            halfway = fd.halfway
            z1 = Cat(Const(0, _GUARD_ZF), fd.tpw[: wt - 1], (~halfway).replicate(wz - wt + 1 - _GUARD_ZF))
            z2 = (z1.as_signed() + fd.z_take)[:wz].as_signed()
            if merged:
                x_seed, y_seed = fd.x_take, fd.y_take
            else:
                x_seed = _chosen(m, fd.picks, [e[1] for e in entries], signed(wx), "seed_x")
                y_seed = _chosen(m, fd.picks, [e[2] for e in entries], signed(wx), "seed_y")
                z2 = Mux(halfway, Const(z_half, signed(wz)), z2)
                x_seed = Mux(halfway, Const(x_half, signed(wx)), x_seed)
                y_seed = Mux(halfway, Const(y_half, signed(wx)), y_seed)

            # What the back end takes of the transaction, held for it.
            e_o = Signal(signed(we), name="e_o", reset_less=True)
            quad_o = Signal(2, name="quad_o", reset_less=True)
            swap_o, tzero_o, tiny_o, sa_o, inf_o, sign_o = (
                Signal(1, name=f"{name}_o", reset_less=True) for name in ("swap", "tzero", "tiny", "sa", "inf", "sign")
            )
            with m.If(folded_valid):
                m.d.sync += e_o.eq(fd.e)
                m.d.sync += quad_o.eq(fd.quad)
                m.d.sync += swap_o.eq(fd.quad[0] ^ fd.flip)
                m.d.sync += tzero_o.eq(fd.tzero)
                m.d.sync += tiny_o.eq(fd.tiny)
                m.d.sync += inf_o.eq(fd.inf)
                m.d.sync += sign_o.eq(fd.sign)
                m.d.sync += sa_o.eq(fd.tiny | ~fd.tpw[tsa_bits:].any())
            done = Signal(name="rot_done")
            m.d.comb += done.eq(engine.done & ~held_vec if both else engine.done)
            z_done = done
            if engine.early:
                z_done = Signal(name="rot_zdone")
                m.d.comb += z_done.eq(engine.z_done & ~held_vec if both else engine.z_done)

            # The linear product is issued at the start, while the multiplier idles; that of the residual angle when the
            # engine has the angle; the two corrections when it has the vector, the sine's first.
            phi_seen = Signal(name="phi_seen")
            tprod = Signal(signed(wp), name="tprod", reset_less=True)
            corr_s = Signal(signed(wp), name="corr_s", reset_less=True)
            bypass_mag = Signal(wmag_rot, name="bypass_mag", reset_less=True)
            e_xn = Signal(signed(wx), name="e_xn", reset_less=True)
            e_yn = Signal(signed(wx), name="e_yn", reset_less=True)
            phi = Signal(signed(wphi), name="phi")
            m.d.comb += phi.eq((tprod >> (const2pi_s + zf - (xf - phi_trunc)))[:wphi].as_signed())
            xc = (e_xn >> xk_trunc)[:wxc].as_signed()
            yc = (e_yn >> xk_trunc)[:wxc].as_signed()
            starting = Signal(name="rot_first")
            z_issue = z_done
            if ahead:
                z_issue = Signal(name="rot_zahead")
                m.d.comb += z_issue.eq(engine.z_ahead & ~held_vec if both else engine.z_ahead)
            m.d.comb += starting.eq(folded_valid | z_issue)  # the two never coincide
            pm_a, pm_b = Signal(signed(wa_rot), name="rot_pm_a"), Signal(signed(wb_rot), name="rot_pm_b")
            pm_tag, pm_iv = Signal(_Product, name="rot_pm_sb"), Signal(1, name="rot_pm_iv")
            mul_valid = Signal(name="rot_mul_valid")
            m.d.comb += mul_valid.eq(pm_valid & ~product_vec if both else pm_valid)
            product = Signal(signed(wp), name="rot_product")
            m.d.comb += product.eq(pm_p[:wp].as_signed())  # exact: it fits its own width
            turned = Signal(name="rot_turned")
            m.d.comb += turned.eq(mul_valid & (pm_sb == _Product.ANGLE))
            # The sine is issued on the angle as the multiplier gives it, and from its hold where the multiplier has no
            # register, which would close the loop through it.
            issue, sine_phi = phi_seen, phi
            if stages and not ahead:
                phi_now = Signal(signed(wphi), name="phi_now")
                m.d.comb += phi_now.eq((product >> (const2pi_s + zf - (xf - phi_trunc)))[:wphi].as_signed())
                issue, sine_phi = phi_seen | turned, Mux(phi_seen, phi, phi_now)
            with m.FSM(name="mphase"):
                with m.State("IDLE"):
                    m.d.comb += pm_iv.eq(starting)
                    m.d.comb += pm_tag.eq(_Product(Mux(z_issue, _Product.ANGLE, _Product.LINEAR)))
                    m.d.comb += pm_a.eq(_zext(Const(c.const2pi, wfactor), wa_rot).as_signed())
                    m.d.comb += pm_b.eq(
                        Mux(
                            z_issue,
                            _sext(engine.zn[:wza], wb_rot),
                            _zext(fd.tpw[:wop], wb_rot).as_signed(),
                        )
                    )
                    with m.If(done):
                        m.next = "SINE"
                        # The angle held by then, the sine is issued from the engine's own `x`.
                        with m.If(phi_seen):
                            m.d.comb += pm_iv.eq(1)
                            m.d.comb += pm_tag.eq(_Product.SINE)
                            m.d.comb += pm_a.eq((engine.xn >> xk_trunc)[:wxc].as_signed())
                            m.d.comb += pm_b.eq(_sext(phi, wb_rot))
                            m.next = "COSINE"
                with m.State("SINE"):
                    m.d.comb += pm_iv.eq(issue)
                    m.d.comb += pm_tag.eq(_Product.SINE)
                    m.d.comb += pm_a.eq(xc)
                    m.d.comb += pm_b.eq(_sext(sine_phi, wb_rot))
                    with m.If(issue):
                        m.next = "COSINE"
                with m.State("COSINE"):
                    m.d.comb += pm_iv.eq(1)
                    m.d.comb += pm_tag.eq(_Product.COSINE)
                    m.d.comb += pm_a.eq(yc)
                    m.d.comb += pm_b.eq(_sext(phi, wb_rot))
                    m.next = "IDLE"
            last = Signal(name="rot_last")
            m.d.comb += last.eq(mul_valid & (pm_sb == _Product.COSINE))
            corrections_valid: Value = last
            if CoreStages.CORRECTIONS in core:
                corrections_valid = Signal(name="corrections_valid")
                m.d.sync += corrections_valid.eq(last)
            with m.If(folded_valid):
                m.d.sync += phi_seen.eq(0)
            with m.Elif(turned):
                m.d.sync += phi_seen.eq(1)
            with m.If(done):
                m.d.sync += e_xn.eq(engine.xn)
                m.d.sync += e_yn.eq(engine.yn)
            with m.If(mul_valid):
                with m.Switch(pm_sb):
                    with m.Case(_Product.LINEAR):
                        m.d.sync += bypass_mag.eq(_zext(product[:wmag_rot], wmag_rot))  # it is not negative
                    with m.Case(_Product.ANGLE):
                        if ahead:
                            # `const2pi` times the angles of the last iterations, by their signs, is taken back.
                            taken = [
                                c.const2pi
                                * sum(-lut if (pattern >> j) & 1 else lut for j, lut in enumerate(c.lut[n - ahead :]))
                                for pattern in range(1 << ahead)
                            ]
                            signs = Cat(engine.z_signs, engine.zn[-1])
                            table = Array(Const(value, signed(wp + 1)) for value in taken)
                            m.d.sync += tprod.eq((product - table[signs])[:wp].as_signed())
                        else:
                            m.d.sync += tprod.eq(product)
                    with m.Case(_Product.SINE):
                        m.d.sync += corr_s.eq(product)
            sin_now = (e_yn + (corr_s >> corr_shift))[:wx].as_signed()
            cos_now = (e_xn - (product >> corr_shift))[:wx].as_signed()
            sin_held, cos_held = sin_now, cos_now  # `fast` without CORRECTIONS packs both as they are made
            if CoreStages.CORRECTIONS in core or not model.fast:
                held_sin = Signal(signed(wx), name="held_sin", reset_less=True)
                held_cos = Signal(signed(wx), name="held_cos", reset_less=True)
                with m.If(last):
                    m.d.sync += held_sin.eq(sin_now)
                    m.d.sync += held_cos.eq(cos_now)
                # Without CORRECTIONS the first of them is packed as they are made, and the second from their hold.
                sin_held = held_sin if CoreStages.CORRECTIONS in core else Mux(last, sin_now, held_sin)
                cos_held = held_cos if CoreStages.CORRECTIONS in core else Mux(last, cos_now, held_cos)

            sin_mag = Signal(wmag_rot, name="sin_tp_mag")
            m.d.comb += sin_mag.eq(Mux(sa_o, bypass_mag, _zext(sin_held[: xf + 1], wmag_rot)))
            cos_mag = Signal(xf + 1, name="cos_tp_mag")  # in (1/2, 1]
            m.d.comb += cos_mag.eq(Mux(sa_o, Const(1 << xf, xf + 1), cos_held[: xf + 1]))
            sin_exp = Signal(signed(weu_pk), name="sin_tp_exp")
            with m.Switch(Cat(tiny_o, sa_o)):
                with m.Case("0-"):
                    m.d.comb += sin_exp.eq(Const(eone_xf, signed(weu_pk)))
                with m.Case("11"):  # the linear path of a tiny `x`: `const2pi` times its significand
                    m.d.comb += sin_exp.eq((_sext(e_o, weu_pk) + (eone_s - wfrac))[:weu_pk].as_signed())
                with m.Default():  # and of a small coordinate: `const2pi` times it
                    m.d.comb += sin_exp.eq(Const(eone_s - (wt + 2), signed(weu_pk)))

            pk_quad = Signal(2, name="rot_pk_quad")
            with m.Switch(Cat(tzero_o, sign_o, inf_o)):
                with m.Case("1--"):
                    m.d.comb += pk_quad.eq(0)
                with m.Case("00-"):
                    m.d.comb += pk_quad.eq(quad_o)
                with m.Case("011"):
                    m.d.comb += pk_quad.eq((-quad_o)[:2])
                with m.Default():
                    m.d.comb += pk_quad.eq((3 - quad_o)[:2])
            r0_sign, r1_sign = quad_o[1] ^ sign_o, Mux(inf_o, sign_o, quad_o[1] ^ quad_o[0])
            packed = Signal(name="rot_packed")
            m.d.comb += packed.eq(pk_valid & ~held_vec if both else pk_valid)

            if model.fast:
                # The sine and the cosine of the octant side by side, then unfolded.
                cosine_in = data.StructLayout({"sign": 1, "inf": 1, "mag": xf + 1})
                given = bundle(cosine_in, sign=Mux(swap_o, r0_sign, r1_sign), inf=inf_o, mag=cos_mag)
                cos_valid, cs = pipe(m, "meet_cos", corrections_valid, cosine_in, given, int(CoreStages.MEET in core))
                m.submodules.f2f_cos = f2c = packer(wmag=xf + 1, weu=wexp + 2, assume_no_overflow=True, most=1)
                m.d.comb += [f2c.in_valid.eq(cos_valid), f2c.sign.eq(cs.sign), f2c.force_inf.eq(cs.inf)]
                m.d.comb += [f2c.exp_offset.eq(bias), f2c.mag.eq(cs.mag)]
                pk_iv, pk_sign, pk_exp, pk_mag = corrections_valid, Mux(swap_o, r1_sign, r0_sign), sin_exp, sin_mag
                res, r0, r1 = packed, Mux(swap_o, f2c.y, pk_y), Mux(swap_o, pk_y, f2c.y)
                tags = bundle(quadrant, quadrant=pk_quad)
            else:
                # The sine, then the cosine a cycle later from the same state: the packer's input stage is their
                # register.
                second = Signal(name="rot_second")
                m.d.sync += second.eq(corrections_valid)
                cosine = Signal(name="rot_cosine")
                m.d.comb += cosine.eq(second ^ swap_o)
                sin_num = Signal(f, name="sin_num", reset_less=True)
                with m.If(packed & ~pk_sb.rotation.second):
                    m.d.sync += sin_num.eq(pk_y)
                pk_iv, pk_sign = corrections_valid | second, Mux(second, r1_sign, r0_sign)
                pk_exp, pk_mag = Mux(cosine, Const(eone_xf, signed(weu_pk)), sin_exp), Mux(cosine, cos_mag, sin_mag)
                res, r0, r1 = packed & pk_sb.rotation.second, sin_num, pk_y
                tags = bundle(quadrant, quadrant=pk_quad, second=second)
            return _Share(
                start=folded_valid,
                x0=x_seed,
                y0=y_seed,
                z0=z2,
                neg0=z2[-1],
                xa=Const(0, wx),
                ya=Const(0, wx),
                pm_iv=pm_iv,
                pm_sb=pm_tag.as_value(),
                pm_a=_sext(pm_a, wa_pm).as_unsigned(),
                pm_b=_sext(pm_b, wb_pm).as_unsigned(),
                pk_iv=pk_iv,
                pk_sign=pk_sign,
                pk_inf=inf_o,
                pk_exp=pk_exp,
                pk_mag=_zext(pk_mag, wmag_pk),
                pk_sb=tags,
                res=res,
                r0=Value.cast(r0),
                r1=Value.cast(r1),
                quadrant=pk_sb.rotation.quadrant,
            )

        def vectoring() -> _Share:
            wcnt, wd = ceil_log2(steps + 1), c.wd
            wpmul = wa_mul + wfactor
            fb = 2 * steps  # fraction bits of a bypass's quotient
            quarter, half = 1 << (zf - 2), 1 << (zf - 1)
            wfull = fmt.wfull

            # `|y| > |x|` from two compares of half the bits of exponent and fraction, whose order as unsigned numbers
            # is that of the magnitudes.
            wk = wfull - 1
            wkl = wk // 2
            key_x, key_y = Value.cast(si.b)[:wk], Value.cast(si.a)[:wk]
            mine_now = Signal(name="vec_taken")
            m.d.comb += mine_now.eq(si_valid & si.vec)
            above, high = key_y[wkl:] > key_x[wkl:], key_y[wkl:] == key_x[wkl:]
            level = high & (key_y[:wkl] > key_x[:wkl])
            at50 = model.unroll100 < 100
            # Both orders of the difference of the exponents are formed beside the compares, so that the order only
            # selects. A shift past `wx - 4` leaves nothing of a significand scaled as below; there the numerator is
            # zeroed.
            clamp = wx - 4
            wsh = ceil_log2(clamp + 1)
            wcmp = max(we + 1, wsh + 1) + 1
            operands = data.StructLayout(
                {"swap": 1, "y": f, "x": f, "xz": 1, "xi": 1, "yz": 1, "yi": 1, "equal": 1, "dxy": wcmp, "dyx": wcmp}
                | {"bxy": 1, "byx": 1}
            )
            dxy = Signal(wcmp, name="f1_dxy")
            m.d.comb += dxy.eq((_zext(si.b.exp, wcmp) - _zext(si.a.exp, wcmp))[:wcmp])
            dyx = Signal(wcmp, name="f1_dyx")
            m.d.comb += dyx.eq((_zext(si.a.exp, wcmp) - _zext(si.b.exp, wcmp))[:wcmp])
            given = bundle(
                operands,
                dxy=dxy,
                dyx=dyx,
                bxy=dxy > clamp,
                byx=dyx > clamp,
                swap=above | level,
                y=si.a,
                x=si.b,
                xz=si.b.is_zero,
                xi=si.b.is_inf,
                yz=a_zero,
                yi=a_inf,
                equal=high & (key_y[:wkl] == key_x[:wkl]),
            )
            taken_valid, tk = pipe(m, "taken", mine_now, operands, given, int(CoreStages.TAKEN in core))

            special = Signal(name="f1_special")
            m.d.comb += special.eq(tk.xz | tk.xi | tk.yz | tk.yi)
            den_sig = Signal(wman, name="f1_den_sig")
            m.d.comb += den_sig.eq(Mux(tk.swap, tk.y.significand, tk.x.significand))
            num_sig = Signal(wman, name="f1_num_sig")
            m.d.comb += num_sig.eq(Mux(tk.swap, tk.x.significand, tk.y.significand))
            den_exp = Signal(wexp, name="f1_den_exp")
            m.d.comb += den_exp.eq(Mux(tk.swap, tk.y.exp, tk.x.exp))  # biased, as the packer takes it
            shift_xy, shift_yx = tk.dxy, tk.dyx
            shamt = Signal(wsh, name="f1_shamt")
            m.d.comb += shamt.eq(Mux(tk.swap, shift_yx, shift_xy)[:wsh])
            beyond = Signal(name="f1_beyond")
            m.d.comb += beyond.eq(Mux(tk.swap, tk.byx, tk.bxy))
            # The significands, scaled by a quarter, top out two bits below the engine's sign; the numerator is aligned
            # to the denominator's binade.
            up = xf - wfrac - 2
            # Past the clamp it is zeroed ahead of the shift: a mask behind it would be a set/reset of the register.
            kept = Signal(wman, name="f1_num_kept")
            m.d.comb += kept.eq(
                Mux(tk.swap, tk.x.significand & ~tk.byx.replicate(wman), tk.y.significand & ~tk.bxy.replicate(wman))
            )
            aligned = Signal(signed(wx), name="f1_y0")
            m.d.comb += aligned.eq(Cat(Const(0, up), kept, Const(0, wx - wman - up)) >> shamt)
            # The first remainder of the bypass, worked out away from the divider's loading, in both orders.
            diffs = [Signal(wman + 1, name=f"f1_byp_diff_{name}") for name in ("yx", "xy")]
            m.d.comb += diffs[0].eq((tk.y.significand - tk.x.significand)[: wman + 1])
            m.d.comb += diffs[1].eq((tk.x.significand - tk.y.significand)[: wman + 1])
            byp_diff = Signal(wman + 1, name="f1_byp_diff")
            m.d.comb += byp_diff.eq(Mux(tk.swap, diffs[1], diffs[0]))
            byp_ibit = Signal(name="f1_byp_ibit")
            m.d.comb += byp_ibit.eq(~byp_diff[-1])

            # What the back end takes of the transaction, held for it. Where an operand is zero the other is the larger,
            # so the length is the denominator; the angle, in eighths of a turn, waits for the output.
            held_den_sig = Signal(wman, name="held_den_sig", reset_less=True)
            held_swap, held_sx, held_sy = (
                Signal(name="held_swap", reset_less=True),
                Signal(name="held_sx", reset_less=True),
                Signal(name="held_sy", reset_less=True),
            )
            held_special = Signal(name="held_special", reset_less=True)
            small = shift_xy > zf - wman - _GUARD_DIV
            held_bypass = Signal(name="held_bypass", reset_less=True)
            held_eden = Signal(signed(we), name="held_eden", reset_less=True)
            held_eydiff = Signal(signed(we + 1), name="held_eydiff", reset_less=True)
            held_sp_sign = Signal(name="held_sp_sign", reset_less=True)
            held_sp_mag = Signal(f, name="held_sp_mag", reset_less=True)
            held_spk = Signal(3, name="held_spk", reset_less=True)
            infinite = f.const(fmt.inf(0))
            ordered = bundle(f, frac=den_sig[:wfrac], exp=den_exp, sign=Const(0, 1))
            behind = Signal(3, name="f1_behind")
            m.d.comb += behind.eq(Mux(tk.x.sign, Const(4, 3), Const(0, 3)))  # half a turn where x < 0
            held_byp_irem = Signal(wman, name="held_byp_irem", reset_less=True)
            held_byp_ibit = Signal(name="held_byp_ibit", reset_less=True)
            with m.If(taken_valid):
                m.d.sync += held_den_sig.eq(den_sig)
                m.d.sync += [held_swap.eq(tk.swap), held_sx.eq(tk.x.sign), held_sy.eq(tk.y.sign)]
                m.d.sync += held_special.eq(special)
                m.d.sync += held_bypass.eq(~tk.swap & ~tk.x.sign & ~special & small)
                m.d.sync += held_eden.eq(Cat(den_exp, Const(0, 1)).as_signed())
                m.d.sync += held_eydiff.eq(shift_yx[: we + 1].as_signed())
                m.d.sync += held_sp_sign.eq(Mux(tk.yz, Const(0, 1), tk.y.sign))
                with m.Switch(Cat(tk.yz, tk.xz, tk.yi, tk.xi)):
                    with m.Case("11--"):
                        m.d.sync += held_spk.eq(Mux(tk.x.sign, Const(3, 3), Const(1, 3)))
                        m.d.sync += held_sp_mag.eq(infinite)
                    with m.Case("01--"):
                        m.d.sync += held_spk.eq(2)
                        m.d.sync += held_sp_mag.eq(infinite)
                    with m.Case("10--"):
                        m.d.sync += held_spk.eq(behind)
                        m.d.sync += held_sp_mag.eq(infinite)
                    with m.Case("0011"):
                        m.d.sync += held_spk.eq(0)
                        m.d.sync += held_sp_mag.eq(f.const(Zkf(fmt, 0)))
                    with m.Case("0001"):
                        m.d.sync += held_spk.eq(behind)
                        m.d.sync += held_sp_mag.eq(ordered)
                    with m.Case("0010"):
                        m.d.sync += held_spk.eq(2)
                        m.d.sync += held_sp_mag.eq(ordered)
                    with m.Default():
                        m.d.sync += held_spk.eq(0)
                        m.d.sync += held_sp_mag.eq(ordered)
                m.d.sync += held_byp_irem.eq(Mux(byp_ibit, byp_diff[:wman], num_sig))
                m.d.sync += held_byp_ibit.eq(byp_ibit)
            # With it the terms of `x0` that the load takes: `3 den`, and `-den` or, where `y0 == x0`, `-5 den`; at 50
            # what gives the sign of `y2`.
            sign2 = {"three": wman + 2, "shamt": wsh, "beyond": 1} if at50 else {}
            numerator = data.StructLayout(
                {"y0": signed(wx), "den3": wman + 2, "dneg": signed(wman + 4), "equal": 1} | sign2
            )
            given = bundle(
                numerator,
                y0=aligned,
                den3=den_sig + (den_sig << 1),
                dneg=Mux(tk.equal, -((den_sig << 2) + den_sig), -den_sig)[: wman + 4],
                equal=tk.equal,
                three=num_sig + (num_sig << 1),
                shamt=shamt,
                beyond=beyond,
            )
            ordered_valid, od = pipe(m, "ordered", taken_valid, numerator, given, int(CoreStages.ORDERED in core))

            # `x2 = 3/2 x0 + ceil(y0 / 2)` and `y2 = y0 + floor(y0 / 2) - x0 / 2`, `x0` being even; where `y0 == x0`,
            # `y2 = -x0`, which the same sum gives with `-5/2 x0`.
            y0, term = od.y0, Cat(Const(0, up - 1), od.dneg).as_signed()
            x2 = Cat(y0[0], Const(0, up - 2), od.den3)  # less `floor(y0 / 2)`
            if at50:
                # The sums ahead of the register, and the sign of `y2` from the significands: `3 y0 >= x0`, always
                # where `y0` is not shifted, never where by three or more.
                den = held_den_sig if CoreStages.ORDERED in core else den_sig
                rises = Array([Const(1, 1), od.three >= den << 1, od.three >= den << 2, Const(0, 1)])
                seed = data.StructLayout({"x2": signed(wx), "y2": signed(wx), "equal": 1, "rising": 1})
                given = bundle(
                    seed,
                    x2=(x2 + (y0 >> 1))[:wx],
                    y2=(y0 + (y0 >> 1) + term)[:wx],
                    equal=od.equal,
                    rising=~od.equal & ~od.beyond & ~od.shamt[2:].any() & rises[od.shamt[:2]],
                )
            else:
                # The first lane's adders sum the addends, `y2`'s as sums and carries.
                seed = data.StructLayout({"xl": signed(wx), "xa": wx, "yl": signed(wx), "ya": wx, "equal": 1})
                given = bundle(
                    seed,
                    xl=y0 >> 1,
                    xa=_zext(x2, wx),
                    yl=y0 ^ (y0 >> 1) ^ term,
                    ya=((y0 & (y0 >> 1) | (y0 ^ (y0 >> 1)) & term) << 1)[:wx],
                    equal=od.equal,
                )
            shifted_valid, sd = pipe(m, "shifted", ordered_valid, seed, given, int(CoreStages.SHIFTED in core))
            if at50:
                x_load, y_load, xa, ya = sd.x2, sd.y2, Const(0, wx), Const(0, wx)
            else:
                x_load, y_load, xa, ya = sd.xl, sd.yl, sd.xa, sd.ya
            z_load = Mux(sd.equal, Const(c.lut[0] + c.lut[1], signed(wz)), Const(c.lut[0] - c.lut[1], signed(wz)))
            rising: Value = sd.rising if at50 else Const(0, 1)
            if seeded[1]:
                # The engine's first iteration, ahead of where the modes meet; `z` of it is one of four constants. Each
                # input is a signal: both sums read it, which the simulator would expand otherwise.
                x2, y2, up = Signal(signed(wx), name="vec_x2"), Signal(signed(wx), name="vec_y2"), Signal(name="vec_up")
                m.d.comb += [x2.eq(x_load), y2.eq(y_load), up.eq(rising)]
                x3, y3 = Signal(signed(wx), name="vec_x3"), Signal(signed(wx), name="vec_y3")
                m.d.comb += x3.eq(_step(x2, (y2 >> _SEED).as_unsigned(), ~up))
                m.d.comb += y3.eq(_step(y2, (x2 >> _SEED).as_unsigned(), up))
                turns = [
                    Mux(
                        sd.equal,
                        *(Const(c.lut[0] + d + sign * c.lut[_SEED], signed(wz)) for d in (c.lut[1], -c.lut[1])),
                    )
                    for sign in (-1, 1)
                ]
                x_load, y_load, z_load, rising = x3, y3, Mux(up, turns[1], turns[0]), ~y3[-1]
            done = Signal(name="vec_done")
            m.d.comb += done.eq(engine.done & held_vec if both else engine.done)

            # A zero or an infinity divides garbage, which the output masks.
            yk_abs = Signal(wx, name="be_ykabs")
            m.d.comb += yk_abs.eq(Mux(engine.yn[-1], (-engine.yn)[:wx], engine.yn.as_unsigned()))
            dv_valid, dv_run, mag_issue = Signal(name="dv_valid"), Signal(name="dv_run"), Signal(name="mag_issue")
            dv_xn, dv_zn, dv_yneg = (
                Signal(signed(wx), name="dv_xn", reset_less=True),
                Signal(signed(wz), name="dv_zn", reset_less=True),
                Signal(1, name="dv_yneg", reset_less=True),
            )
            dv_cnt, dv_quo = Signal(wcnt, name="dv_cnt", reset_less=True), Signal(wquo, name="dv_quo", reset_less=True)
            dv_rem, dv_den, dv_den3 = (
                Signal(wd, name="dv_rem", reset_less=True),
                Signal(wd, name="dv_den", reset_less=True),
                Signal(wd + 2, name="dv_den3", reset_less=True),
            )
            den_arm = Signal(wd, name="den_arm")
            m.d.comb += den_arm.eq(Mux(held_bypass, _zext(held_den_sig, wd), engine.xn.as_unsigned()[wx - wd :]))
            m.submodules.step = step = DivsqrtStep(len(dv_rem), mode=0, iq=0)
            m.d.comb += [step.w.eq(dv_rem), step.d.eq(dv_den), step.d3.eq(dv_den3)]
            m.d.sync += dv_valid.eq(0)
            m.d.sync += mag_issue.eq(done)
            # What the engine leaves of `x` is positive in a transaction that divides, so the divisor needs no guard
            # against zero; the digit step takes a remainder below its divisor.
            with m.If(done):
                m.d.sync += dv_run.eq(1)
                m.d.sync += dv_cnt.eq(0)
                m.d.sync += dv_rem.eq(Mux(held_bypass, _zext(held_byp_irem, wd), (yk_abs << c.lead)[wx - wd : wx]))
                m.d.sync += dv_den.eq(den_arm)
                m.d.sync += dv_den3.eq((Cat(Const(0, 1), den_arm) + den_arm)[: wd + 2])
                m.d.sync += dv_quo.eq(_zext(held_bypass & held_byp_ibit, wquo))
                # The engine holds these too; the copies keep what follows off its recurrence.
                m.d.sync += dv_xn.eq(engine.xn)
                m.d.sync += dv_zn.eq(engine.zn)
                m.d.sync += dv_yneg.eq(engine.yn[-1])
            with m.Elif(dv_run):
                m.d.sync += dv_rem.eq(step.w_next)
                m.d.sync += dv_quo.eq(Cat(step.digit, dv_quo[: wquo - 2]))
                m.d.sync += dv_cnt.eq((dv_cnt + 1)[:wcnt])
                with m.If(dv_cnt == steps - 1):
                    m.d.sync += dv_run.eq(0)
                    m.d.sync += dv_valid.eq(1)

            # The multiplier takes the length; the quotient times `1/(2*pi)` is gathered a digit at a time, a cycle
            # behind the division, which is exact.
            mag_ready = Signal(name="mag_ov")
            m.d.comb += mag_ready.eq(pm_valid & product_vec if both else pm_valid)
            product = Signal(wmag_vec, name="vec_product")
            m.d.comb += product.eq(pm_p[:wpmul])
            wacc = wquo + wfactor
            acc = Signal(wacc, name="dv_acc", reset_less=True)
            acc_run, theta_ready = Signal(name="acc_run"), Signal(name="theta_ov")
            m.d.sync += [acc_run.eq(dv_run), theta_ready.eq(dv_valid)]
            multiple = Array(Const(k * c.inv_tau, wfactor + 2) for k in range(4))[dv_quo[:2]]
            with m.If(done):
                m.d.sync += acc.eq(Mux(held_bypass & held_byp_ibit, c.inv_tau, 0))
            with m.Elif(acc_run):
                m.d.sync += acc.eq(((acc << 2) + multiple)[:wacc])

            # `theta = unmap +- zn +- delta`: a constant of the octant, the angle that the engine gathered, and the late
            # term of the division, whose sign folds the octant's with that of the residual `y`.
            zn = _sext(dv_zn, wz + 2)  # it can be slightly negative
            unmap = Signal(signed(wz + 2), name="unmap_const")
            with m.Switch(Cat(held_sx, held_swap)):
                with m.Case("1-"):
                    m.d.comb += unmap.eq(Const(quarter, signed(wz + 2)))
                with m.Case("01"):
                    m.d.comb += unmap.eq(Const(half, signed(wz + 2)))
                with m.Default():
                    m.d.comb += unmap.eq(0)
            negates = Signal(name="un_neg_a0")
            m.d.comb += negates.eq(held_swap ^ held_sx)
            un_base = Signal(signed(wz + 2), name="un_base", reset_less=True)
            m.d.sync += un_base.eq(Mux(negates, unmap - zn, unmap + zn)[: wz + 2].as_signed())
            # The late term meets the rest of theta right behind the accumulator's register.
            late = Cat((acc >> (c.f2 + invtau_s - zf))[: wz + 2], Const(0, 1)).as_signed()
            delta = Signal(signed(wz + 3), name="un_delta")
            m.d.comb += delta.eq(late)
            un_tmag = Signal(wz + 2, name="un_tmag")
            m.d.comb += un_tmag.eq(Mux(negates ^ dv_yneg, (un_base - delta)[: wz + 2], (un_base + delta)[: wz + 2]))
            # It can decide a tie at WMAN 16 and 45, where `inv_tau` ends in four zeros.
            inexact = Signal(name="dv_inexact", reset_less=True)
            m.d.sync += inexact.eq(dv_rem.any())
            bypassed = _zext(acc, wmag_vec) | _zext(inexact, wmag_vec)
            # With the bias in it already, so that the packer adds none.
            byp_exp = (_sext(held_eydiff, weu_pk) + (wmag_pk - 1 - fb - invtau_s + bias))[:weu_pk].as_signed()
            angle = data.StructLayout({"tmag": wmag_vec, "texp": signed(weu_pk)})
            given = bundle(
                angle,
                tmag=Mux(held_bypass, bypassed, _zext(un_tmag, wmag_vec)),
                texp=Mux(held_bypass, byp_exp, Const(wmag_pk - 1 - zf + bias, signed(weu_pk))),
            )
            angle_valid, ag = pipe(m, "angle", theta_ready, angle, given, int(CoreStages.ANGLE in core))
            # Minus a half turn is packed as the canonical plus a half: what rounds to a half, from half an ULP below
            # it, takes the plus sign ahead of the packer, which leaves no compare behind its rounding.
            to_half = Signal(name="to_half")
            m.d.comb += to_half.eq(~held_bypass & (ag.tmag[zf - 1] | ag.tmag[zf - 3 - wfrac : zf - 1].all()))

            # The length is packed first and held until theta comes; a zero or an infinity overrides both results from
            # what is held of the operands.
            # The 2 is the seed's quarter.
            mag_exp = (_sext(held_eden, weu_pk) + (wmag_pk - 1 - (xf + kinv_s) + 2))[:weu_pk].as_signed()
            pk_iv = Signal(name="vec_pk_iv")
            m.d.comb += pk_iv.eq(mag_ready | angle_valid)
            packed = Signal(name="vec_packed")
            m.d.comb += packed.eq(pk_valid & held_vec if both else pk_valid)
            mag_num = Signal(f, name="mag_num", reset_less=True)
            with m.If(packed & ~pk_sb.vectoring.theta):
                m.d.sync += mag_num.eq(pk_y)
            # Eighths of a turn are constants: no carry chain on the output.
            turn8 = Signal(f, name="turn8")
            eighth = {1: (bias - 3, 0), 2: (bias - 2, 0), 3: (bias - 2, 1 << (wfrac - 1)), 4: (bias - 1, 0)}
            with m.Switch(held_spk):
                for k, (exp, frac) in eighth.items():
                    with m.Case(k):
                        sign = Const(0, 1) if k == 4 else held_sp_sign
                        m.d.comb += turn8.eq(bundle(f, frac=Const(frac, wfrac), exp=Const(exp, wexp), sign=sign))
                with m.Default():
                    m.d.comb += turn8.eq(f.const(Zkf(fmt, 0)))
            theta, length = Signal(f, name="vec_theta"), Signal(f, name="vec_mag")
            with m.If(held_special):
                m.d.comb += theta.eq(turn8)
                m.d.comb += length.eq(held_sp_mag)
            with m.Else():
                m.d.comb += theta.eq(pk_y)
                m.d.comb += length.eq(mag_num)
            return _Share(
                start=shifted_valid,
                x0=x_load,
                y0=y_load,
                z0=z_load,
                neg0=rising,
                xa=xa,
                ya=ya,
                pm_iv=mag_issue,
                pm_sb=Const(0, _Product.as_shape()),
                pm_a=_zext(dv_xn[:wa_mul], wa_pm),
                pm_b=_zext(Const(c.kinv_mag, wfactor), wb_pm),
                pk_iv=pk_iv,
                pk_sign=angle_valid & held_sy & ~to_half,
                pk_inf=Const(0, 1),
                pk_exp=Mux(angle_valid, ag.texp, mag_exp),
                pk_mag=_zext(Mux(angle_valid, ag.tmag, product), wmag_pk),
                pk_sb=_zext(angle_valid, pk_tags.size),
                res=packed & pk_sb.vectoring.theta,
                r0=theta.as_value(),
                r1=length.as_value(),
                quadrant=Const(0, 2),
            )

        sides = [side() for side, built in ((vectoring, vectors), (rotation, rotates)) if built]

        def shared(name: str) -> Value:
            values = [getattr(side, name) for side in sides]
            if not both:
                return values[0]
            return values[0] | values[1] if name in _RAISED else Mux(own_vec if name in _TAKING else held_vec, *values)

        now = _Share(**{field.name: shared(field.name) for field in fields(_Share)})
        m.d.comb += engine.start.eq(now.start)
        m.d.comb += engine.vectoring.eq(own_vec)
        m.d.comb += engine.x0.eq(now.x0)
        m.d.comb += engine.y0.eq(now.y0)
        m.d.comb += engine.z0.eq(now.z0)
        m.d.comb += engine.neg0.eq(now.neg0)
        m.d.comb += engine.xa.eq(now.xa)
        m.d.comb += engine.ya.eq(now.ya)
        m.d.comb += pmul.in_valid.eq(now.pm_iv)
        if rotates:
            m.d.comb += pmul.sb_in.eq(now.pm_sb)
        m.d.comb += pmul.a.eq(now.pm_a)
        m.d.comb += pmul.b.eq(now.pm_b)
        back = data.StructLayout({"sign": 1, "inf": 1, "exp": signed(weu_pk), "mag": wmag_pk, "sb": pk_tags})
        given = bundle(back, sign=now.pk_sign, inf=now.pk_inf, exp=now.pk_exp, mag=now.pk_mag, sb=now.pk_sb)
        pk_iv, pk = pipe(m, "meet", now.pk_iv, back, given, int(CoreStages.MEET in core))
        m.d.comb += f2f.in_valid.eq(pk_iv)
        m.d.comb += f2f.sign.eq(pk.sign)
        m.d.comb += f2f.force_inf.eq(pk.inf)
        m.d.comb += f2f.exp_offset.eq(pk.exp)
        m.d.comb += f2f.mag.eq(pk.mag)
        m.d.comb += f2f.sb_in.eq(pk.sb)
        # The output's register holds the result until it is taken; the stages past it are ahead of it.
        found = bundle(result, quadrant=now.quadrant, r1=now.r1, r0=now.r0)
        res_valid, res = pipe(m, "res", now.res, result, found, max(model.stage_output - 1, 0))
        m.d.comb += txn.res_valid.eq(res_valid)
        m.d.comb += txn.res.eq(res)
        return m


def _chosen(m: Module, of: Value, values: Sequence[int], shape: Shape, name: str) -> Value:
    """The value of `values` that the one-hot `of` picks."""
    value = Signal(shape, name=name)
    picked = (Mux(of[k], Const(v, shape), Const(0, shape)) for k, v in enumerate(values))
    m.d.comb += value.eq(functools.reduce(operator.or_, picked))
    return value


class _Seed(wiring.Component):
    """
    Of a rotation's coordinate `t`, before its fold into an octant: the interval of `_seed` that it is in, one-hot in
    `picks`, and the constant `z_take` that `z1` takes to give `z` at the engine's start; `halfway` that it is half a
    quadrant. Under `merged`, `halfway` is one more interval, last in `picks`, which overrides the others and which
    `z_take` covers, `z1` being 0 there; and the vector at the start comes out too, in `x_take` and `y_take`.
    """

    t: Signal
    picks: Signal
    z_take: Signal
    halfway: Signal

    def __init__(self, c: _Constants, signs: int, merged: bool) -> None:
        self._wt, self._merged = c.wt, merged
        self.entries, self.halfway_state = _seed(c, signs)
        shape = {"picks": Out(len(self.entries) + merged), "z_take": Out(signed(c.wz)), "halfway": Out(1)}
        if merged:
            shape |= {"x_take": Out(signed(c.wx)), "y_take": Out(signed(c.wx))}
        super().__init__({"t": In(c.wt)} | shape)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        wt, t, entries = self._wt, self.t, self.entries
        bounds = [entry[0] for entry in entries[1:]]
        # `tpw < bound` on the coordinate unfolded: `tpw` is `(t ^ s) + s` with `s` its top bit.
        top = t[wt - 1]
        unfolded = t ^ top.replicate(wt)
        below = Signal(len(bounds), name="below")
        m.d.comb += below.eq(Cat(unfolded < Mux(top, Const(bound - 1, wt), Const(bound, wt)) for bound in bounds))
        m.d.comb += self.halfway.eq(t == 1 << (wt - 1))
        spans = [Const(1, 1)]
        if bounds:
            spans = [
                below[0] if not k else ~below[k - 1] if k == len(bounds) else below[k] & ~below[k - 1]
                for k in range(len(entries))
            ]
        if self._merged:
            m.d.comb += self.picks.eq(Cat(*(~self.halfway & span for span in spans), self.halfway))
        else:
            m.d.comb += self.picks.eq(Cat(*spans))
        z_half = [self.halfway_state[2]] if self._merged else []
        m.d.comb += self.z_take.eq(
            _chosen(m, self.picks, [e[3] for e in entries] + z_half, self.z_take.shape(), "take")
        )
        if self._merged:
            for k, port in ((1, self.x_take), (2, self.y_take)):
                values = [e[k] for e in entries] + [self.halfway_state[k - 1]]
                m.d.comb += port.eq(_chosen(m, self.picks, values, port.shape(), f"take_{'xy'[k - 1]}"))
        return m


def _step(a: Value, b: Value, subtract: Value) -> Value:
    """`a - b` or `a + b` on one carry chain, wrapped to the width of `a`; `b` is unsigned and as wide."""
    return (a.as_unsigned() + (b ^ subtract.replicate(len(b))) + subtract)[: len(a)].as_signed()


class _Lane(wiring.Component):
    """
    One of the iterations chained in a cycle: the `index`-th after iteration `i`. One past the last leaves `z` as it is,
    the first lane being never past it while the engine runs, and one past the last or while the engine is idle adds
    nothing to `x` and `y`, which then load through the same adders. `xs` and `ys` are what is shifted: `x` and `y`, or
    the registers themselves in the first lane. `vec` is read at `mode` 2 alone. `seeding`, while the engine is idle,
    adds `xa` to `x` and `ya` to `y` instead.
    """

    run: Signal
    vec: Signal
    i: Signal
    x: Signal
    y: Signal
    z: Signal
    xs: Signal
    ys: Signal
    seeding: Signal
    xa: Signal
    ya: Signal
    x_next: Signal
    y_next: Signal
    z_next: Signal

    def __init__(self, index: int, lut: Array, n: int, mode: int, *, wi: int, wx: int, wz: int) -> None:
        self._index, self._lut, self._n, self._mode = index, lut, n, mode
        taken = {"run": In(1), "vec": In(1), "i": In(wi), "x": In(signed(wx)), "y": In(signed(wx))}
        taken |= {"z": In(signed(wz)), "xs": In(signed(wx)), "ys": In(signed(wx))}
        taken |= {"seeding": In(1), "xa": In(wx), "ya": In(wx)}
        super().__init__(taken | {"x_next": Out(signed(wx)), "y_next": Out(signed(wx)), "z_next": Out(signed(wz))})

    def elaborate(self, platform: object) -> Module:
        m = Module()
        x, y, z, xs, ys, wx = self.x, self.y, self.z, self.xs, self.ys, len(self.x)
        idx = Signal(len(self.i), name="idx")
        m.d.comb += idx.eq((self.i + self._index)[: len(self.i)])
        en_z = Signal(name="en_z")
        m.d.comb += en_z.eq(idx < self._n if self._index else 1)
        en = Signal(name="en")
        m.d.comb += en.eq(self.run & en_z)
        by_mode = (z[-1], ~ys[-1], Mux(self.vec, ~ys[-1], z[-1]))
        neg = Signal(name="neg")
        m.d.comb += neg.eq(by_mode[self._mode])
        sub_z = Signal(name="sub_z")
        m.d.comb += sub_z.eq(~neg)
        sub_x = Signal(name="sub_x")
        m.d.comb += sub_x.eq(en & sub_z)
        sub_y = Signal(name="sub_y")
        m.d.comb += sub_y.eq(en & neg)
        x_by = Mux(self.seeding, self.xa, (ys >> idx).as_unsigned() & en.replicate(wx))
        y_by = Mux(self.seeding, self.ya, (xs >> idx).as_unsigned() & en.replicate(wx))
        m.d.comb += self.x_next.eq(_step(x, x_by, sub_x))
        m.d.comb += self.y_next.eq(_step(y, y_by, sub_y))
        m.d.comb += self.z_next.eq(Mux(en_z, _step(z, self._lut[idx], sub_z), z))
        return m


class _Engine(wiring.Component):
    """
    One `(x, y, z)` datapath used over several cycles in place of a pipeline, so the area is that of a few iterations
    and the latency does not depend on the data.

    An iteration is `x' = x -/+ (y >> i)`, `y' = y +/- (x >> i)`, `z' = z -/+ lut[i]`, with arithmetic shifts. In
    rotation the sign follows `z`, which goes to zero while `(x, y)` turns by `z0`; in vectoring it follows `y`, which
    goes to zero while `z` gathers the angle of `(x0, y0)`. `mode` is 0 for rotation, 1 for vectoring and 2 for either,
    vectoring where `vectoring` is high at `start`. `lut` is the angle of each iteration, unsigned.

    `unroll100` is `Cordic`'s. At 50 and not for vectoring alone, the recurrence of `z`, one narrow addition, runs at an
    iteration a cycle ahead of `x` and `y` and replays its signs to them: `z_done` then comes before `done`, and a
    rotation starts on its residual angle early. At 100, `ahead`, where nonzero, raises `z_ahead` that many iterations
    before the end and gives in `z_signs` the signs of `z` from there on but the last's.

    `seeds` are the iterations that each mode, rotation first, starts at. `start` for a cycle takes `x0`, `y0` and `z0`,
    those of its mode's seed, and is ignored while the engine runs; at 100 and up a vectoring's load adds `xa` to `x0`
    and `ya` to `y0` on the first lane's adders. At 50 the load also shifts for its mode's first iteration, whose sign
    is `neg0`: that `z0` is negative in rotation, that `y0` is not in vectoring. `done` is high for one cycle with `xn`,
    `yn` and `zn`; `z_done` for one with `zn`.
    """

    start: Signal
    vectoring: Signal
    x0: Signal
    y0: Signal
    z0: Signal
    neg0: Signal
    xa: Signal
    ya: Signal
    done: Signal
    z_done: Signal
    xn: Signal
    yn: Signal
    zn: Signal

    def __init__(
        self,
        lut: Sequence[int],
        *,
        wx: int,
        wz: int,
        unroll100: int,
        mode: int,
        ahead: int = 0,
        seeds: tuple[int, int] = (_SEED, _SEED),
    ) -> None:
        self._lut, self._wx, self._wz, self._unroll100, self._mode = list(lut), wx, wz, unroll100, mode
        self._ahead, self._seeds = ahead, seeds
        super().__init__(
            {
                "start": In(1),
                "vectoring": In(1),
                "x0": In(signed(wx)),
                "y0": In(signed(wx)),
                "z0": In(signed(wz)),
                "neg0": In(1),
                "xa": In(wx),
                "ya": In(wx),
                "done": Out(1),
                "z_done": Out(1),
                "xn": Out(signed(wx)),
                "yn": Out(signed(wx)),
                "zn": Out(signed(wz)),
            }
            | ({"z_ahead": Out(1), "z_signs": Out(ahead - 1)} if ahead else {})
        )

    @property
    def early(self) -> bool:
        """Whether `z_done` comes before `done`; it is `done` otherwise."""
        return self._mode != 1 and self._unroll100 < 100

    def elaborate(self, platform: object) -> Module:
        m = Module()
        n, wx, wz, mode = len(self._lut), self._wx, self._wz, self._mode
        half = self._unroll100 < 100
        u = 1 if half else self._unroll100 // 100
        wi = ceil_log2(n + u + 1)
        # What is read past the last angle, by an idle lane or by the fetch ahead, is not used.
        lut = Array(Const(angle, wz) for angle in self._lut)

        run, done = Signal(name="run"), Signal(name="done")
        phase = Signal(name="phase") if half else None  # 0 shifts and samples, 1 adds and advances
        i = Signal(wi, name="i", reset_less=True)
        vec = Signal(1, name="vec", reset_less=True) if mode == 2 else None
        x, y, z = (
            Signal(signed(wx), name="x", reset_less=True),
            Signal(signed(wx), name="y", reset_less=True),
            Signal(signed(wz), name="z", reset_less=True),
        )
        i_next = Signal(wi, name="i_next")
        m.d.comb += i_next.eq((i + u)[:wi])
        last = Signal(name="last")
        m.d.comb += last.eq(i_next >= n)
        ending = last
        if phase is not None:
            ending = Signal(name="ending")
            m.d.comb += ending.eq(phase & last)
        z_done: Value = done

        def by_mode(of: Callable[[int], Value]) -> Value:
            rot, vec_seed = self._seeds
            if mode != 2 or rot == vec_seed:
                return of(vec_seed if mode == 1 else rot)
            return Mux(self.vectoring, of(vec_seed), of(rot))

        m.d.sync += done.eq(run & ending)
        with m.If(run if phase is None else run & phase):
            m.d.sync += i.eq(i_next)
            with m.If(last):
                m.d.sync += run.eq(0)
        with m.Elif(~run & self.start):
            m.d.sync += run.eq(1)
            m.d.sync += i.eq(by_mode(lambda seed: Const(seed, wi)))
            if vec is not None:
                m.d.sync += vec.eq(self.vectoring)

        if phase is None:
            # `z` keeps a load of its own: with a bare sum ahead of it, a tool may retime the register into the adder,
            # onto the path of the sign.
            x_in = Signal(signed(wx), name="x_in")
            m.d.comb += x_in.eq(Mux(run, x, self.x0))
            y_in = Signal(signed(wx), name="y_in")
            m.d.comb += y_in.eq(Mux(run, y, self.y0))
            feed = (x_in, y_in, z, x, y)
            for index in range(u):
                m.submodules[f"lane{index}"] = lane = _Lane(index, lut, n, mode, wi=wi, wx=wx, wz=wz)
                m.d.comb += [lane.run.eq(run), lane.i.eq(i), lane.x.eq(feed[0]), lane.y.eq(feed[1])]
                m.d.comb += [lane.z.eq(feed[2]), lane.xs.eq(feed[3]), lane.ys.eq(feed[4])]
                if vec is not None:
                    m.d.comb += lane.vec.eq(vec)
                if index == 0 and mode:
                    seeding = ~run & (1 if mode == 1 else self.vectoring)
                    m.d.comb += [lane.seeding.eq(seeding), lane.xa.eq(self.xa), lane.ya.eq(self.ya)]
                feed = (lane.x_next, lane.y_next, lane.z_next, lane.x_next, lane.y_next)
            with m.If(run | self.start):
                m.d.sync += x.eq(lane.x_next)
                m.d.sync += y.eq(lane.y_next)
            with m.If(run):
                m.d.sync += z.eq(lane.z_next)
            with m.Elif(self.start):
                m.d.sync += z.eq(self.z0)
            if self._ahead:
                k = self._ahead
                m.d.comb += self.z_ahead.eq(run & (i == n - k))
                signs = Signal(k - 1, name="z_signs", reset_less=True)
                with m.If(run):
                    m.d.sync += signs.eq(Cat(signs[1:], z[-1]))
                m.d.comb += self.z_signs.eq(signs)
        else:
            x_shifted = Signal(signed(wx), name="x_shifted", reset_less=True)
            y_shifted = Signal(signed(wx), name="y_shifted", reset_less=True)
            neg, angle = Signal(1, name="neg", reset_less=True), Signal(wz, name="angle", reset_less=True)
            # The sign in copies, so that each steers fewer adders: `neg` that of `y`, `sub` those of `x` and of
            # `z` but at `mode` 2, where `vsub` steers a vectoring's `z`.
            sub = Signal(name="sub", reset_less=True)
            vsub = Signal(name="vsub", reset_less=True) if mode == 2 else None
            # The sign of `z` ahead of each iteration, recorded from the sum that makes it.
            replayed = Signal(n + 1, name="signs", reset_less=True) if self.early else None
            ahead = z[-1] if replayed is None else replayed.bit_select(i, 1)
            fetched = None
            if replayed is None:
                fetched = Signal(wz, name="fetched")
                m.d.comb += fetched.eq(lut[i])
            sampled = Signal(name="sampled")
            m.d.comb += sampled.eq((ahead, ~y[-1], None if vec is None else Mux(vec, ~y[-1], ahead))[mode])
            with m.If(run):
                m.d.sync += phase.eq(~phase)
            with m.If(~run):
                with m.If(self.start):
                    m.d.sync += x.eq(self.x0)
                    m.d.sync += y.eq(self.y0)
                    m.d.sync += x_shifted.eq(by_mode(lambda seed: self.x0 >> seed))
                    m.d.sync += y_shifted.eq(by_mode(lambda seed: self.y0 >> seed))
                    m.d.sync += neg.eq(self.neg0)
                    m.d.sync += sub.eq(~self.neg0)
                    if vsub is not None:
                        m.d.sync += vsub.eq(~self.neg0 & self.vectoring)
                    m.d.sync += phase.eq(1)
                    if replayed is None:
                        m.d.sync += z.eq(self.z0)
                        m.d.sync += angle.eq(by_mode(lambda seed: lut[seed]))
            with m.Elif(~phase):
                m.d.sync += x_shifted.eq(x >> i)
                m.d.sync += y_shifted.eq(y >> i)
                m.d.sync += neg.eq(sampled)
                m.d.sync += sub.eq(~sampled)
                if vec is not None and vsub is not None:
                    m.d.sync += vsub.eq(y[-1] & vec)
                if fetched is not None:
                    m.d.sync += angle.eq(fetched)
            with m.Else():
                m.d.sync += x.eq(_step(x, y_shifted.as_unsigned(), sub))
                m.d.sync += y.eq(_step(y, x_shifted.as_unsigned(), neg))
                if replayed is None:
                    m.d.sync += z.eq(_step(z, angle, sub))

            if replayed is not None:
                # A vectoring's `z` steps with `x` and `y`, on the sign they sampled. The angle is fetched a cycle
                # early, which keeps the table's selection out of the addition.
                zi = Signal(wi, name="zi", reset_less=True)
                z_run, z_finished = Signal(name="z_run"), Signal(name="z_finished")
                zi_next = Signal(wi, name="zi_next")
                m.d.comb += zi_next.eq((zi + 1)[:wi])
                z_last = Signal(name="z_last")
                m.d.comb += z_last.eq(zi_next >= n)
                z_loading = Signal(name="z_load")
                m.d.comb += z_loading.eq(self.start & ~run)
                z_sub = Signal(name="z_sub")
                m.d.comb += z_sub.eq(~z[-1] if vec is None or vsub is None else vsub | ~vec & ~z[-1])
                ahead_angle = Signal(wz, name="fetched")
                m.d.comb += ahead_angle.eq(lut[zi_next])
                m.d.sync += z_finished.eq(z_run & z_last)
                with m.If(~z_run):
                    m.d.sync += z_run.eq(z_loading if vec is None else z_loading & ~self.vectoring)
                with m.Elif(z_last):
                    m.d.sync += z_run.eq(0)
                z_next = Signal(signed(wz), name="z_next")
                m.d.comb += z_next.eq(_step(z, angle, z_sub))
                with m.If(z_run if vec is None else z_run | (vec & run & phase)):
                    m.d.sync += z.eq(z_next)
                    m.d.sync += zi.eq(zi_next)
                    m.d.sync += replayed.bit_select(zi_next, 1).eq(z_next[-1])
                    m.d.sync += angle.eq(ahead_angle)
                with m.Elif(z_loading):
                    m.d.sync += z.eq(self.z0)
                    m.d.sync += zi.eq(by_mode(lambda seed: Const(seed, wi)))
                    m.d.sync += angle.eq(by_mode(lambda seed: lut[seed]))
                z_done = z_finished if vec is None else z_finished | (vec & done)

        m.d.comb += self.done.eq(done)
        m.d.comb += self.z_done.eq(z_done)
        m.d.comb += self.xn.eq(x)
        m.d.comb += self.yn.eq(y)
        m.d.comb += self.zn.eq(z)
        return m

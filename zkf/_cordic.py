import enum
import functools
import operator
from collections.abc import Mapping, Sequence
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
from ._divsqrt import DivideStep8, DivsqrtStep
from ._value import UnsupportedFormat, Zkf, ZkfFormat, context

_GUARD_DIV = 8
"""
Fraction bits of the quotient of vectoring's division beyond the significand's, so that the division and the bypass for
a small ratio round within an ULP; they set the division's digits. It also sets where that bypass starts, and the bypass
drops a term cubic in the ratio, so more is not better: this, and not the count of iterations, is what binds the angle.
"""
_GUARD_XY = 6
"""Fraction bits of `x` and `y` beyond one and a half significands: the angle of vectoring is bound by them."""
_GUARD_XY_MORE = frozenset({50})
"""The WMAN at which the angle takes one more of them, not faithful otherwise."""
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
    cycle, which can take the next, whatever the stages.

    One iterative CORDIC engine serves both modes, beside one multiplier, which the products take in turn, and one
    packer, which a rotation's two results take in turn unless `fast`; the constants are generated for the WMAN.
    About WMAN / 2 iterations suffice, as each mode finishes with one step linear in the residual angle, whose square is
    below an ULP.

    A rotation reduces `|a|` modulo a turn, exactly, to a quadrant and a coordinate within it, folds that into an eighth
    of a turn, and turns `(1/gain, 0)` by it, which leaves `(cos, sin)` and a small residual angle `z`. One linear
    rotation by `phi = 2*pi*z` finishes it with two corrections: `sin + cos*phi` and `cos - sin*phi`. An angle too small
    for the iterations gives `sin = 2*pi*angle` and `cos = 1` instead. The octant and the quadrant are then unfolded.

    A vectoring orders the operands by magnitude, aligns the smaller to the binade of the larger, and drives `y` to zero
    from a quarter of the vector, `z` gathering the angle and `x` growing to the gain times that length. The residual
    angle is finished by one division, `y / x` at a digit a cycle, whose digits gather its product with `1/(2*pi)`;
    meanwhile the multiplier takes `x / gain`, which scaled back is the length.
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
    stage_input: int = 0
    stage_core_mul: int
    """As `Mul`'s, from 1 up; a rotation passes the product twice in series."""
    stage_core: CoreStages = CoreStages(0)
    """See `Cordic.CoreStages`; a fixed mode refuses those it lacks."""
    stage_output: int = 0
    wmultiplier: int
    """As `Mul`'s."""
    mode: int = 2
    fast: bool = True
    """
    A second packer, which packs a rotation's cosine of the octant beside its sine, a cycle less on a rotation; and a
    vectoring's division at radix 8, a third fewer of its digits. Off saves fabric and shortens the division's loop.
    """
    rotation_seed: int = 2
    """
    Up to 2 of a rotation's iterations, resolved ahead of the engine by compares of its coordinate with constants; no
    effect at `mode=1`.
    """
    rom_attrs: Attrs | None = field(default=None, compare=False)
    """
    Synthesis attributes of the reads of the engine's angles and of what the angle's product issued ahead takes back,
    such as `{"syn_romstyle": "logic"}`.
    """
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
        wa, wb = g.operands(self.mode)
        if self.stage_core_mul < 1:
            raise ValueError(f"stage_core_mul={self.stage_core_mul} is below 1")
        check("stage_core_mul", self.stage_core_mul, wa, wb, signed=self.mode != 1, wmultiplier=self.wmultiplier)
        if not 0 <= self.rotation_seed <= 2:
            raise ValueError(f"rotation_seed={self.rotation_seed} is outside 0..2")
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
        `oracle=True` gives the results themselves in place of the hardware's, for a simulation of the ideal
        operator: mpf numbers of 4 * WMAN + 80 bits, which `fmt.encode` rounds correctly; a rounded angle of -1/2 is
        the operator's +1/2.
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
        iterations = g.n - _SEED - (self.rotation_seed if mode == 0 else 0)
        if mode == 0:
            # The engine's load; the cosine a cycle behind the sine into the multiplier, and into the packer unless
            # `fast`.
            latency = 2 + (not self.fast) + iterations + 2 * product - max(_ahead(self) - 1, 0)
        else:
            # The engine's load, the division's, and the last digit into the angle.
            latency = 3 + iterations + g.digits(_radix(self))
        latency += registers + self.stage_input + self.stage_output
        return Timing(latency, latency)


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
    def fb(self) -> int:
        """Fraction bits of a bypass's quotient."""
        return 2 * ((self.wman + _GUARD_DIV + 1) // 2)

    @property
    def bypass(self) -> int:
        """The difference of the exponents past which a vectoring near the positive x axis takes the bypass."""
        return self.zf - self.wman - _GUARD_DIV

    def digits(self, radix: int) -> int:
        """Of vectoring's division at `radix`, which skips the quotient's top bits that the engine leaves zero."""
        return -(-(self.f2 - (self.n - 2)) // (radix.bit_length() - 1))

    def digits_bypass(self, radix: int) -> int:
        """Of a bypass's division at `radix`, whose quotient has an integer bit too."""
        return -(-(self.fb + 1) // (radix.bit_length() - 1))

    def lead_div(self, radix: int) -> int:
        """The lead at which `digits` end on the quotient's last bit."""
        return self.f2 - (radix.bit_length() - 1) * self.digits(radix)

    @property
    def f2(self) -> int:
        """Fraction bits of the residual's quotient."""
        return 2 * ((self.xf + 1) // 2)

    @property
    def wd(self) -> int:
        """
        Of the division's remainder and divisor: `fb` and two bits more, with which the angle is faithful at every WMAN;
        cut to it, `|y|` loses only zeros.
        """
        return self.fb + 2

    @property
    def wfactor(self) -> int:
        """Of each of the three factors."""
        return self.wman + 5

    @property
    def const2pi_s(self) -> int:
        return self.wman + 2

    @property
    def invtau_s(self) -> int:
        return self.wman + 7

    @property
    def kinv_s(self) -> int:
        return self.wman + 5

    @property
    def wxc(self) -> int:
        """Of what a rotation's corrections take of the vector, signed."""
        return self.wman + 6

    @property
    def xk_trunc(self) -> int:
        """The bits that the corrections' cut of the vector drops."""
        return self.wx - self.wxc

    @property
    def phi_shift(self) -> int:
        """Of the product of `const2pi` and the residual angle, to the corrections' angle."""
        return self.const2pi_s + self.zf - (self.xf - self.phi_trunc)

    @property
    def corr_shift(self) -> int:
        """Of the product of the vector and the corrections' angle, to a correction at the scale of the vector."""
        return self.xf - self.xk_trunc - self.phi_trunc

    @property
    def wlin(self) -> int:
        """A rotation's coordinate below `2**wlin` takes the linear path."""
        return self.wff - (self.wman + 1) // 2 - 3

    @property
    def wop(self) -> int:
        """Of what a rotation's linear path multiplies: the coordinate there, or the significand of a tiny `x`."""
        return max(self.wman, self.wlin + 1)

    @property
    def wcp(self) -> int:
        """Of what a rotation's product with `const2pi` takes of the coordinate or of the residual angle."""
        return max(self.wop, self.zf - self.n) + 2

    @property
    def wphi(self) -> int:
        """
        Of the residual angle that a rotation's corrections take, signed: its fraction cut to at most `wman + 5` bits,
        and a bit above them, which the angle reaches where the last iteration leaves it just past a power of two.
        """
        return self.xf - self.n + 3 - self.phi_trunc

    @property
    def phi_trunc(self) -> int:
        """The bits that the residual angle's cut drops."""
        return max(0, self.xf - self.n + 2 - self.wxc)

    def operands(self, mode: int) -> tuple[int, int]:
        """
        Of the multiplier at `mode`: a rotation's are signed, a factor or a coordinate by an angle; a vectoring's
        unsigned, the length by a factor; both modes' as wide as either needs.
        """
        rotation = (max(self.wfactor + 1, self.wxc), max(self.wcp, self.wphi))
        vectoring = (self.wx - 1, self.wfactor)
        return (rotation, vectoring, (max(rotation[0], vectoring[0] + 1), max(rotation[1], vectoring[1] + 1)))[mode]


def _geometry(wman: int) -> _Geometry:
    if not 16 <= wman <= 53:
        raise UnsupportedFormat(f"no cordic for WMAN={wman}: its constants are generated for 16 to 53")
    wff = wman + max(12, wman // 2 + 2)  # where the linear path takes over, `1 - cos` being below an ULP there
    xf = (3 * wman + 1) // 2 + _GUARD_XY + (wman in _GUARD_XY_MORE)
    g = _Geometry(wman=wman, n=(wman + 1) // 2 + 2, wff=wff, xf=xf, zf=wff + _GUARD_ZF)
    # The engine leaves `|y|` within `x * 2**(1 - n)` and a unit an iteration, so a lead of `n - 2` keeps the first
    # remainder below the divisor.
    assert g.wd < g.wx, g
    for radix in (4, 8):
        assert (
            g.wx - g.wd <= g.lead_div(radix) <= g.n - 2 and 0 <= g.digits_bypass(radix) - g.digits(radix) < g.n - _SEED
        ), g
    return g


@dataclass(frozen=True)
class _Constants(_Geometry):
    """
    `lut`, the angle of each iteration in turns at `2**-zf`; `kinv`, the inverse gain at `2**-xf`, which seeds a
    rotation; three factors of `wman + 5` bits, each at its own scale (`const2pi_s`, `invtau_s`, `kinv_s`):
    `const2pi = 2*pi`, `inv_tau = 1 / (2*pi)` and `kinv_mag`, the inverse gain that descales a length. `least` is the
    fewest bits of a nonzero magnitude that the packer takes in rotation and in vectoring, which bounds what its
    normalizer shifts.
    """

    kinv: int
    lut: tuple[int, ...]
    const2pi: int
    inv_tau: int
    kinv_mag: int
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

    const2pi, inv_tau, kinv_mag = scaled(turn, g.const2pi_s), scaled(1 / turn, g.invtau_s), scaled(1 / gain, g.kinv_s)
    slack = 4 * g.n  # units of what the engine and its corrections may lose
    # The least of a rotation is its sine at the linear threshold; of a vectoring, its angle at the bypass's threshold
    # with the least ratio of significands.
    sine = bits(_exact.sin(turn * _exact.mpf(2) ** (g.wlin - g.wff)) * _exact.mpf(2) ** g.xf - slack)
    ratio = _exact.mpf(2) ** -g.bypass * (1 << (wman - 1)) / ((1 << wman) - 1)
    angle = bits(_exact.atan(ratio) / turn * _exact.mpf(2) ** g.zf - slack)
    return _Constants(
        **asdict(g),
        kinv=scaled(1 / gain, g.xf),
        lut=tuple(scaled(_exact.atan(_exact.mpf(2) ** -i) / turn, g.zf) for i in range(g.n)),
        const2pi=const2pi,
        inv_tau=inv_tau,
        kinv_mag=kinv_mag,
        least=(sine, angle),
    )


def _radix(model: Cordic) -> int:
    """Of vectoring's division."""
    return 8 if model.fast else 4


def _ahead(model: Cordic) -> int:
    """
    How many iterations ahead of the engine's end a rotation issues the angle's product, `stage_core_mul + 1`, so that
    the angle, less a constant of the signs taken since, is held by the end; 0 where the wider angle would take another
    slice, or the product would start ahead of the rotation's first iteration.
    """
    if model.mode == 1:
        return 0
    k = model.stage_core_mul + 1
    c = _constants(model.fmt.wman)
    if c.n - k < _SEED + model.rotation_seed:
        return 0
    wa, wb = c.operands(model.mode)
    grid = functools.partial(slices, wa, signed=True, wmultiplier=model.wmultiplier)
    return k if grid(max(wb, _angle_bits(c, k))) == grid(wb) else 0


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
        cut = -((offset + lut[1] - lut[0]) >> _GUARD_ZF)  # where `z_j` turns nonnegative
        assert low < cut < high  # at two signs at the most, every interval splits
        follow(j + 1, x + (y >> j), y - (x >> j), offset + lut[j], low, cut)
        follow(j + 1, x - (y >> j), y + (x >> j), offset - lut[j], cut, high)

    follow(_SEED, c.kinv + (c.kinv >> 1), c.kinv - (c.kinv >> 1), 0, 0, half)
    halfway = _iterate(c, c.kinv - (c.kinv >> 1), c.kinv + (c.kinv >> 1), -lut[1], False, _SEED, _SEED + signs)
    return entries, halfway


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
    ff = spec.wff
    one = (1 << xf, xf)
    sig = a.significand()
    e = a.exp - fmt.bias

    # |x| mod 1 in `ff` bits: sin(2*pi*x) = sign * sin(2*pi*|x|) and cos(2*pi*x) = cos(2*pi*|x|).
    sh = e - fmt.wfrac + ff
    tiny = sh < 0  # below the reducer's resolution -> small-angle path on |x|, whose `t` is its significand
    frac_pos = (sig << max(sh, 0)) & ((1 << ff) - 1)
    quadrant_abs = frac_pos >> wt  # |x| quadrant
    t = frac_pos & ((1 << wt) - 1)  # quadrant-local coordinate, scale 2**-WT
    tzero = t == 0  # frac(|x|)*4 integer: a quadrant boundary / exact magnitude

    # The local angle folded into [0, pi/4].
    half = 1 << (wt - 1)
    oct_flip = t > half  # theta in (pi/4, pi/2): use the pi/2 - theta complement
    tp = ((1 << wt) - t) if oct_flip else t

    # (sin theta', cos theta') as pairs of a magnitude and its scale.
    if tiny or tp < 1 << spec.wlin:
        sin_tp = (const2pi * tp, const2pi_s + ff - min(sh, 0))  # sin ~= 2 * pi * t'
        cos_tp = one
    else:
        # On operands of WMAN + 6 bits, which leaves under an ULP.
        xk, yk, zk = _iterate(spec, spec.kinv, 0, tp << (zf - ff), False)
        phi = (const2pi * zk) >> spec.phi_shift
        assert phi.bit_length() < spec.wphi, "the corrections' angle"
        corr_s = ((xk >> spec.xk_trunc) * phi) >> spec.corr_shift
        corr_c = ((yk >> spec.xk_trunc) * phi) >> spec.corr_shift
        sin_tp = (yk + corr_s, xf)
        cos_tp = (xk - corr_c, xf)

    # The octant unfolded, then the quadrant of |x|.
    sin_m, cos_m = (cos_tp, sin_tp) if (quadrant_abs & 1) ^ oct_flip else (sin_tp, cos_tp)
    sin_sign = (quadrant_abs >> 1) ^ a.negative  # sin is odd: negative x flips it
    cos_sign = ((quadrant_abs >> 1) ^ quadrant_abs) & 1  # cos is even: unchanged by the sign of x
    # floor(frac(x) * 4): for x < 0 that of |x| reflected about a turn.
    quadrant = (-quadrant_abs - (not tzero)) & 3 if a.negative else quadrant_abs

    assert all(not mag or mag.bit_length() >= spec.least[0] for mag in (sin_m[0], cos_m[0])), "the packer's bound"
    assert 1 << (xf - 1) <= cos_tp[0] <= 1 << xf, "the bound of the cosine's packer"
    return CordicResult(_packed(fmt, sin_sign, *sin_m), _packed(fmt, cos_sign, *cos_m), quadrant)


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
    swap = (y.exp, y.frac) > (x.exp, x.frac)
    den, num = (y, x) if swap else (x, y)
    den_sig, num_sig = den.significand(), num.significand()
    e_den = den.exp - fmt.bias
    shift_dn = den.exp - num.exp
    # A quarter of the vector, so that x_K = gain * hypot stays inside the engine's width; the length takes it back.
    den_fixed = den_sig << (xf - wfrac - 2)  # in [0.25, 0.5) * 2**xf
    num_fixed = (num_sig << (xf - wfrac - 2)) >> shift_dn  # num aligned to den's scale 2**-xf

    x_k, y_k, z_k = _iterate(spec, den_fixed, num_fixed, 0, True)
    assert (x_k * kinv_mag).bit_length() >= spec.least[1], "the packer's bound"
    length = _packed(fmt, 0, x_k * kinv_mag, xf + kinv_s - e_den - 2, saturate=True)

    # The bypass: the quotient truncated as the divider's, at the digits, and its sticky bit beside the product.
    if (not swap) and (sx == 0) and (shift_dn > spec.bypass):
        fb = spec.fb
        r, rest = divmod(num_sig << fb, den_sig)
        assert (r * inv_tau).bit_length() >= spec.least[1], "the packer's bound"
        theta = _packed(fmt, sy, (r * inv_tau) | bool(rest), fb + invtau_s + shift_dn)
        return CordicResult(theta, length, 0)

    a0 = _vector_residual(spec, x_k, y_k, z_k)

    # The octant and the quadrant unfolded.
    quarter = 1 << (zf - 2)
    half = 1 << (zf - 1)
    phi1 = (quarter - a0) if swap else a0
    theta_mag = (half - phi1) if sx else phi1
    assert not theta_mag or theta_mag.bit_length() >= spec.least[1], "the packer's bound"
    theta_bits = _packed(fmt, sy, theta_mag, zf).bits
    return CordicResult(Zkf(fmt, _atan2_canon_half(fmt, theta_bits)), length, 0)


def _packed(fmt: ZkfFormat, sign: int, mag: int, scale: int, *, saturate: bool = False) -> Zkf:
    """`(-1)**sign * mag * 2**-scale` as the packer rounds it."""
    wmag = max(mag.bit_length(), fmt.wman + 3)
    return Zkf(fmt, fixed_to_float_reference(fmt, sign, mag, wmag - 1 - scale, wmag, saturate_round_carry=saturate))


def _iterate(
    c: _Constants, x: int, y: int, z: int, vectoring: bool, start: int = 0, stop: int | None = None
) -> tuple[int, int, int]:
    """
    The engine's iterations from `start` to `stop`, the last by default, on `(x, y)` at the scale 2**-xf and `z` in
    turns at 2**-zf: a rotation drives `z` to 0, a vectoring `y`; a shift floors.
    """
    for i in range(start, c.n if stop is None else stop):
        if (y >= 0) if vectoring else (z < 0):
            x, y, z = x + (y >> i), y - (x >> i), z + c.lut[i]
        else:
            x, y, z = x - (y >> i), y + (x >> i), z - c.lut[i]
        assert not vectoring or abs(y) << (i + 1) < 1 << c.wx, "the engine's `u`"
    return x, y, z


def _vector_residual(spec: _Constants, x_k: int, y_k: int, z_k: int) -> int:
    """
    The vectoring's reduced angle `z_K + (y_K / x_K) / (2 * pi)` at the scale of the angle, `x_K > 0`; the quotient is
    truncated as the divider's, at `f2` fraction bits, of operands cut to the division's width.
    """
    cut = spec.wx - spec.wd
    q = (abs(y_k) << (spec.f2 - cut)) // (x_k >> cut)
    assert q < 1 << (spec.f2 - spec.n + 2), "the division's first remainder"
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


def _sincos_oracle(z: Zkf) -> CordicResult[mpmath.mpf]:
    """
    The oracle of a rotation. The phase is reduced mod 1 exactly and folded into an octant before mpmath sees it, which
    avoids the cancellation of a large argument and keeps the zeros at quadrant boundaries exact.
    """
    if z.is_inf:
        return CordicResult(z.to_mpf(), z.to_mpf(), 0)
    if z.is_zero:
        return CordicResult(mpmath.mpf(0), mpmath.mpf(1), 0)
    prec = 4 * z.fmt.wman + 80
    ctx = context(prec)
    q4 = z.to_fraction() % 1 * 4
    quadrant = int(q4)
    t_local = q4 - quadrant  # in [0,1)
    flip = t_local > Fraction(1, 2)
    folded = 1 - t_local if flip else t_local
    theta = (ctx.pi / 2) * ctx.mpf(folded.numerator) / ctx.mpf(folded.denominator)
    s0, c0 = ctx.sin(theta), ctx.cos(theta)
    sin_mag, cos_mag = (c0, s0) if (quadrant & 1) ^ flip else (s0, c0)
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
_TAKING = ("x0", "y0", "z0", "xa", "ya", "pm_sb", "pm_a", "pm_b")
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
        f = FloatLayout(fmt)

        # A rotation's products keep the top bits of the residual angle and of what the engine leaves.
        wop, wcp, wphi = c.wop, c.wcp, c.wphi
        wxc = c.wxc
        wa_rot, wb_rot = c.operands(0)
        ahead = _ahead(model)
        wza = max(wcp, _angle_bits(c, ahead)) if ahead else wcp  # of the angle that the product takes
        wb_rot = max(wb_rot, wza)
        wmag_rot = wfactor + wt + 1
        weu_rot = ceil_log2(wmag_rot + 1) + 2

        wquo = c.fb + 1  # the quotient of two significands, which the bypass divides, can exceed 1
        wa_mag = wx - 1  # what the engine leaves of `x` in vectoring is positive
        wmag_vec = wa_mag + wfactor  # the length's product is the widest

        # What both modes share is as wide as either needs: a signed multiplier, with a bit more for the unsigned
        # operands of vectoring, and a packer that takes a biased exponent.
        wa_pm, wb_pm = c.operands(mode)
        wb_pm = max(wb_pm, wb_rot) if rotates else wb_pm
        # The tags beside a packed result: each mode reads its own, in its own transactions.
        quadrant = data.StructLayout({"quadrant": 2} | ({} if model.fast else {"second": 1}))
        pk_tags = data.UnionLayout(
            ({"rotation": quadrant} if rotates else {})
            | ({"vectoring": data.StructLayout({"theta": 1})} if vectors else {})
        )
        wmag_pk = (wmag_rot, wmag_vec, max(wmag_rot, wmag_vec))[mode]
        weu_pk = wexp + ceil_log2(wmag_pk + 1) + 3 if vectors else max(weu_rot, wexp + 3)

        core, stages = model.stage_core, model.stage_core_mul
        CoreStages = Cordic.CoreStages

        result = data.StructLayout({"quadrant": 2, "r1": f, "r0": f})
        m.submodules.txn = txn = Txn(result, min(model.stage_output, 1))
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

        inputs = data.StructLayout({"b": f, "a": f, "vec": 1})
        offered = bundle(inputs, b=self.b, a=self.a, vec=self.vectoring if both else Const(mode, 1))
        si_valid, si = pipe(m, "in", txn.accept, inputs, offered, model.stage_input)

        early = c.digits_bypass(_radix(model)) - c.digits(_radix(model)) if mode else 0
        m.submodules.engine = engine = _Engine(
            c.lut,
            wx=wx,
            wz=wz,
            mode=mode,
            ahead=ahead,
            rom_attrs=model.rom_attrs,
            early=early,
            seed=_SEED + model.rotation_seed,
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
        # What the engine leaves of `x`, held for either mode off its recurrence.
        xn = Signal(signed(wx), name="xn_held", reset_less=True)
        with m.If(engine.done):
            m.d.sync += xn.eq(engine.xn)
        least = (*c.least, min(c.least))[mode]
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
            xk_trunc, corr_shift = c.xk_trunc, c.corr_shift
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
            m.submodules.seed = seed = _Seed(c, model.rotation_seed, merged)
            m.d.comb += seed.t.eq(rd.t)
            entries, (x_half, y_half, z_half) = seed.entries, seed.halfway_state
            layout = {"flip": 1, "tpw": wt, "halfway": 1, "tzero": 1, "quad": 2, "e": signed(we)}
            layout |= {"z_take": signed(wz)} | ({} if merged else {"picks": len(seed.picks)})
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
                **({} if merged else {"picks": seed.picks}),
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
                m.d.sync += sa_o.eq(fd.tiny | ~fd.tpw[c.wlin :].any())
            done = Signal(name="rot_done")
            m.d.comb += done.eq(engine.done & ~held_vec if both else engine.done)

            # The linear product is issued at the start, while the multiplier idles; that of the residual angle when the
            # engine has the angle; the two corrections when it has the vector, the sine's first.
            phi_seen = Signal(name="phi_seen")
            tprod = Signal(signed(wp), name="tprod", reset_less=True)
            corr_s = Signal(signed(wp), name="corr_s", reset_less=True)
            bypass_mag = Signal(wmag_rot, name="bypass_mag", reset_less=True)
            e_yn = Signal(signed(wx), name="e_yn", reset_less=True)
            phi = Signal(signed(wphi), name="phi")
            m.d.comb += phi.eq((tprod >> c.phi_shift)[:wphi].as_signed())
            xc = (xn >> xk_trunc)[:wxc].as_signed()
            yc = (e_yn >> xk_trunc)[:wxc].as_signed()
            starting = Signal(name="rot_first")
            z_issue = done
            if ahead:
                z_issue = Signal(name="rot_zahead")
                m.d.comb += z_issue.eq(engine.z_ahead & ~held_vec if both else engine.z_ahead)
            m.d.comb += starting.eq(folded_valid | z_issue)  # the two never coincide
            pm_a, pm_b = Signal(signed(wa_rot), name="rot_pm_a"), Signal(signed(wb_rot), name="rot_pm_b")
            pm_tag, pm_iv = Signal(_Product, name="rot_pm_sb"), Signal(1, name="rot_pm_iv")
            mul_valid = Signal(name="rot_mul_valid")
            m.d.comb += mul_valid.eq(pm_valid & ~held_vec if both else pm_valid)
            product = Signal(signed(wp), name="rot_product")
            m.d.comb += product.eq(pm_p[:wp].as_signed())  # exact: it fits its own width
            turned = Signal(name="rot_turned")
            m.d.comb += turned.eq(mul_valid & (pm_sb == _Product.ANGLE))
            # The sine is issued on the angle as the multiplier gives it, unless the angle is held by the engine's end.
            issue, sine_phi = phi_seen, phi
            if not ahead:
                phi_now = Signal(signed(wphi), name="phi_now")
                m.d.comb += phi_now.eq((product >> c.phi_shift)[:wphi].as_signed())
                issue, sine_phi = phi_seen | turned, Mux(phi_seen, phi, phi_now)
            with m.FSM(name="mphase"):
                with m.State("IDLE"):
                    m.d.comb += pm_iv.eq(starting)
                    m.d.comb += pm_tag.eq(_Product(Mux(z_issue, _Product.ANGLE, _Product.LINEAR)))
                    m.d.comb += pm_a.eq(Const(c.const2pi, signed(wa_rot)))
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
                            m.d.comb += pm_b.eq(phi)
                            m.next = "COSINE"
                with m.State("SINE"):
                    m.d.comb += pm_iv.eq(issue)
                    m.d.comb += pm_tag.eq(_Product.SINE)
                    m.d.comb += pm_a.eq(xc)
                    m.d.comb += pm_b.eq(sine_phi)
                    with m.If(issue):
                        m.next = "COSINE"
                with m.State("COSINE"):
                    m.d.comb += pm_iv.eq(1)
                    m.d.comb += pm_tag.eq(_Product.COSINE)
                    m.d.comb += pm_a.eq(yc)
                    m.d.comb += pm_b.eq(phi)
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
                m.d.sync += e_yn.eq(engine.yn)
            if ahead:
                # `const2pi` times the angles of the last iterations, by their signs, is taken back.
                taken = [
                    c.const2pi * sum(-lut if (pattern >> j) & 1 else lut for j, lut in enumerate(c.lut[n - ahead :]))
                    for pattern in range(1 << ahead)
                ]
                table = Array(Const(value, signed(wp + 1)) for value in taken)
                back = Signal(signed(wp + 1), name="taken", attrs=dict(model.rom_attrs or {}))
                m.d.comb += back.eq(table[Cat(engine.z_signs, engine.zn[-1])])
            with m.If(mul_valid):
                with m.Switch(pm_sb):
                    with m.Case(_Product.LINEAR):
                        m.d.sync += bypass_mag.eq(product[:wmag_rot])  # it is not negative
                    with m.Case(_Product.ANGLE):
                        m.d.sync += tprod.eq((product - back)[:wp].as_signed() if ahead else product)
                    with m.Case(_Product.SINE):
                        m.d.sync += corr_s.eq(product)
            sin_now = (e_yn + (corr_s >> corr_shift))[:wx].as_signed()
            cos_now = (xn - (product >> corr_shift))[:wx].as_signed()
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
            radix, wd = _radix(model), c.wd
            bits, digits, digits_bypass = radix.bit_length() - 1, c.digits(radix), c.digits_bypass(radix)
            wcnt = ceil_log2(digits_bypass)
            # Of a bypass's divisor, so that its digits end on its quotient's last bit, the integer bit among them.
            scale = bits * digits_bypass - c.fb
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

            # What the back end takes of the transaction, held for it; a special angle, in eighths of a turn, waits for
            # the output.
            held_den_sig = Signal(wman, name="held_den_sig", reset_less=True)
            held_swap, held_sx, held_sy, held_special, held_bypass = (
                Signal(name=f"held_{name}", reset_less=True) for name in ("swap", "sx", "sy", "special", "bypass")
            )
            small = shift_xy > c.bypass
            held_eden = Signal(signed(we), name="held_eden", reset_less=True)
            held_eydiff = Signal(signed(we + 1), name="held_eydiff", reset_less=True)
            held_sp_inf, held_sp_zero = (Signal(name=f"held_sp_{name}", reset_less=True) for name in ("inf", "zero"))
            held_spk = Signal(3, name="held_spk", reset_less=True)
            behind = Signal(3, name="f1_behind")
            m.d.comb += behind.eq(Mux(tk.x.sign, Const(4, 3), Const(0, 3)))  # half a turn where x < 0
            held_num_sig = Signal(wman, name="held_num_sig", reset_less=True)
            with m.If(taken_valid):
                m.d.sync += held_den_sig.eq(den_sig)
                m.d.sync += [held_swap.eq(tk.swap), held_sx.eq(tk.x.sign), held_sy.eq(tk.y.sign)]
                m.d.sync += held_special.eq(special)
                m.d.sync += held_bypass.eq(~tk.swap & ~tk.x.sign & ~special & small)
                m.d.sync += held_eden.eq(Cat(den_exp, Const(0, 1)).as_signed())
                m.d.sync += held_eydiff.eq(shift_yx[: we + 1].as_signed())
                m.d.sync += [held_sp_inf.eq(tk.xi | tk.yi), held_sp_zero.eq(tk.xz & tk.yz)]
                with m.Switch(Cat(tk.yz, tk.xz, tk.yi, tk.xi)):
                    with m.Case("11--"):
                        m.d.sync += held_spk.eq(Mux(tk.x.sign, Const(3, 3), Const(1, 3)))
                    with m.Case("01--", "0010"):
                        m.d.sync += held_spk.eq(2)
                    with m.Case("10--", "0001"):
                        m.d.sync += held_spk.eq(behind)
                    with m.Default():
                        m.d.sync += held_spk.eq(0)
                m.d.sync += held_num_sig.eq(num_sig)
            # With it the terms of `x0` that the load takes: `3 den`, and `-den` or, where `y0 == x0`, `-5 den`.
            numerator = data.StructLayout({"y0": signed(wx), "den3": wman + 2, "dneg": signed(wman + 4), "equal": 1})
            given = bundle(
                numerator,
                y0=aligned,
                den3=den_sig + (den_sig << 1),
                dneg=Mux(tk.equal, -((den_sig << 2) + den_sig), -den_sig)[: wman + 4],
                equal=tk.equal,
            )
            ordered_valid, od = pipe(m, "ordered", taken_valid, numerator, given, int(CoreStages.ORDERED in core))

            # `x2 = 3/2 x0 + ceil(y0 / 2)` and `y2 = y0 + floor(y0 / 2) - x0 / 2`, `x0` being even; where `y0 == x0`,
            # `y2 = -x0`, which the same sum gives with `-5/2 x0`.
            y0, term = od.y0, Cat(Const(0, up - 1), od.dneg).as_signed()
            x2 = Cat(y0[0], Const(0, up - 2), od.den3)  # less `floor(y0 / 2)`
            # The engine's adders sum the addends at its load, `y2`'s as sums and carries.
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
            x_load, y_load, xa, ya = sd.xl, sd.yl, sd.xa, sd.ya
            z_load = Mux(sd.equal, Const(c.lut[0] + c.lut[1], signed(wz)), Const(c.lut[0] - c.lut[1], signed(wz)))
            done = Signal(name="vec_done")
            m.d.comb += done.eq(engine.done & held_vec if both else engine.done)

            # A zero or an infinity divides garbage, which the output masks.
            yk_abs = Signal(wx, name="be_ykabs")
            m.d.comb += yk_abs.eq(Mux(engine.yn[-1], (-engine.yn)[:wx], engine.yn.as_unsigned()))
            dv_valid, dv_run, mag_issue = Signal(name="dv_valid"), Signal(name="dv_run"), Signal(name="mag_issue")
            dv_zn, dv_yneg = Signal(signed(wz), name="dv_zn", reset_less=True), Signal(name="dv_yneg", reset_less=True)
            dv_cnt, dv_digit = (
                Signal(wcnt, name="dv_cnt", reset_less=True),
                Signal(bits, name="dv_digit", reset_less=True),
            )
            dv_rem, dv_den, dv_den3 = (
                Signal(wd, name="dv_rem", reset_less=True),
                Signal(wd, name="dv_den", reset_less=True),
                Signal(wd + 2, name="dv_den3", reset_less=True),
            )
            den_arm = Signal(wd, name="den_arm")
            m.d.comb += den_arm.eq(
                Mux(held_bypass, _zext(held_den_sig << scale, wd), engine.xn.as_unsigned()[wx - wd :])
            )
            # The radix-8 step takes `5 den` and `7 den` too, which the first digit of either path can need.
            multiples = []
            if radix == 4:
                m.submodules.step = step = DivsqrtStep(wd, mode=0, iq=0)
            else:
                m.submodules.step = step = DivideStep8(wd)
                dv_den5, dv_den7 = (
                    Signal(wd + 3, name="dv_den5", reset_less=True),
                    Signal(wd + 3, name="dv_den7", reset_less=True),
                )
                m.d.comb += [step.d5.eq(dv_den5), step.d7.eq(dv_den7)]
                multiples = [
                    dv_den5.eq((Cat(Const(0, 2), den_arm) + den_arm)[: wd + 3]),
                    dv_den7.eq((Cat(Const(0, 3), den_arm) - den_arm)[: wd + 3]),
                ]
            m.d.comb += [step.w.eq(dv_rem), step.d.eq(dv_den), step.d3.eq(dv_den3)]
            # A bypass's division takes more digits, of operands held since TAKEN: it starts as many cycles ahead, so
            # that both end together.
            dv_load = Signal(name="dv_load")
            ahead_of_done = engine.early if early else engine.done
            m.d.comb += dv_load.eq(Mux(held_bypass, ahead_of_done & held_vec if both else ahead_of_done, done))
            m.d.sync += dv_valid.eq(0)
            m.d.sync += mag_issue.eq(done)
            # What the engine leaves of `x` is positive in a transaction that divides, so the divisor needs no guard
            # against zero; the digit step takes a remainder below its divisor.
            with m.If(dv_load):
                m.d.sync += dv_run.eq(1)
                m.d.sync += dv_cnt.eq(Mux(held_bypass, digits_bypass - 1, digits - 1))
                m.d.sync += dv_rem.eq(
                    Mux(held_bypass, _zext(held_num_sig, wd), (yk_abs << c.lead_div(radix))[wx - wd : wx])
                )
                m.d.sync += dv_den.eq(den_arm)
                m.d.sync += dv_den3.eq((Cat(Const(0, 1), den_arm) + den_arm)[: wd + 2])
                m.d.sync += multiples
            with m.Elif(dv_run):
                m.d.sync += dv_rem.eq(step.w_next)
                m.d.sync += dv_digit.eq(step.digit)
                m.d.sync += dv_cnt.eq((dv_cnt - 1)[:wcnt])
                with m.If(dv_cnt == 0):
                    m.d.sync += dv_run.eq(0)
                    m.d.sync += dv_valid.eq(1)
            # Copies of what the engine leaves keep what follows off its recurrence.
            with m.If(done):
                m.d.sync += dv_zn.eq(engine.zn)
                m.d.sync += dv_yneg.eq(engine.yn[-1])

            # The multiplier takes the length; the quotient times `1/(2*pi)` is gathered a digit at a time, a cycle
            # behind the division, which is exact.
            mag_ready = Signal(name="mag_ov")
            m.d.comb += mag_ready.eq(pm_valid & held_vec if both else pm_valid)
            product = Signal(wmag_vec, name="vec_product")
            m.d.comb += product.eq(pm_p[:wmag_vec])
            wacc = wquo + wfactor
            acc = Signal(wacc, name="dv_acc", reset_less=True)
            acc_run, theta_ready = Signal(name="acc_run"), Signal(name="theta_ov")
            m.d.sync += [acc_run.eq(dv_run), theta_ready.eq(dv_valid)]
            multiple = Array(Const(k * c.inv_tau, wfactor + bits) for k in range(radix))[dv_digit]
            with m.If(dv_load):
                m.d.sync += acc.eq(0)
            with m.Elif(acc_run):
                m.d.sync += acc.eq(((acc << bits) + multiple)[:wacc])

            # `theta = unmap +- zn +- delta`: a constant of the octant, the angle that the engine gathered, and the late
            # term of the division, whose sign folds the octant's with that of the residual `y`.
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
            m.d.sync += un_base.eq(Mux(negates, unmap - dv_zn, unmap + dv_zn))
            # The late term meets the rest of theta right behind the accumulator's register.
            delta = Signal(signed(wz + 3), name="un_delta")
            m.d.comb += delta.eq((acc >> (c.f2 + invtau_s - zf))[: wz + 2])
            un_tmag = Signal(wz + 2, name="un_tmag")
            m.d.comb += un_tmag.eq(Mux(negates ^ dv_yneg, (un_base - delta)[: wz + 2], (un_base + delta)[: wz + 2]))
            # It can decide a tie at WMAN 16 and 45, where `inv_tau` ends in four zeros.
            inexact = Signal(name="dv_inexact", reset_less=True)
            m.d.sync += inexact.eq(dv_rem.any())
            bypassed = acc | inexact
            # With the bias in it already, so that the packer adds none.
            byp_exp = (_sext(held_eydiff, weu_pk) + (wmag_pk - 1 - c.fb - invtau_s + bias))[:weu_pk].as_signed()
            angle = data.StructLayout({"tmag": wmag_vec, "texp": signed(weu_pk)})
            given = bundle(
                angle,
                tmag=_zext(Mux(held_bypass, bypassed, un_tmag), wmag_vec),
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
                        sign = Const(0, 1) if k == 4 else held_sy
                        m.d.comb += turn8.eq(bundle(f, frac=Const(frac, wfrac), exp=Const(exp, wexp), sign=sign))
                with m.Default():
                    m.d.comb += turn8.eq(f.const(Zkf(fmt, 0)))
            theta, length = Signal(f, name="vec_theta"), Signal(f, name="vec_mag")
            # Where an operand is zero the other is the larger, so the length is the denominator.
            unpacked = Signal(1, name="held_sp_unpacked")
            m.d.comb += unpacked.eq(held_sp_inf | held_sp_zero)
            sp_mag = bundle(
                f,
                frac=held_den_sig[:wfrac] & ~unpacked.replicate(wfrac),
                exp=held_eden[:wexp] | held_sp_inf.replicate(wexp),
                sign=Const(0, 1),
            )
            with m.If(held_special):
                m.d.comb += theta.eq(turn8)
                m.d.comb += length.eq(sp_mag)
            with m.Else():
                m.d.comb += theta.eq(pk_y)
                m.d.comb += length.eq(mag_num)
            return _Share(
                start=shifted_valid,
                x0=x_load,
                y0=y_load,
                z0=z_load,
                xa=xa,
                ya=ya,
                pm_iv=mag_issue,
                pm_sb=Const(0, _Product.as_shape()),
                pm_a=_zext(xn[:wa_mag], wa_pm),
                pm_b=Const(c.kinv_mag, wb_pm),
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


class _Engine(wiring.Component):
    """
    One `(x, y, z)` datapath at an iteration a cycle in place of a pipeline, so the area is that of one iteration and
    the latency does not depend on the data.

    An iteration is `x' = x -/+ (y >> i)`, `y' = y +/- (x >> i)`, `z' = z -/+ lut[i]`, with arithmetic shifts. In
    rotation the sign follows `z`, which goes to zero while `(x, y)` turns by `z0`; in vectoring it follows `y`, which
    goes to zero while `z` gathers the angle of `(x0, y0)`. `mode` is 0 for rotation, 1 for vectoring and 2 for either,
    vectoring where `vectoring` is high at `start`. `lut` is the angle of each iteration, unsigned.

    `y` is held as `u = y * 2**i`, so its step takes `x` with the low `i` bits cleared, not shifted; `x`'s takes
    `u >> 2i`, which a register holds a cycle ahead of `x`'s add. No cycle holds both a shift and an add.

    `seed` is the iteration that a rotation starts at, a vectoring starting at `_SEED`. `start` for a cycle takes `x0`,
    `y0` and `z0`, those of its mode's seed, and is ignored while the engine runs; the load adds `xa` to `x0` and `ya`,
    unsigned, to `y0` on the iteration's adders, sums that fit `wx` bits as integers, which a rotation gives as 0.
    `ahead`, where nonzero, raises `z_ahead` that many iterations before the end and gives in `z_signs` the signs of `z`
    from there on but the last's. `done` is high for a cycle with `xn`, `yn`, `zn`; `early`, where nonzero, raises
    `early` that many cycles before it. `rom_attrs` goes on the read of the angles.
    """

    start: Signal
    vectoring: Signal
    x0: Signal
    y0: Signal
    z0: Signal
    xa: Signal
    ya: Signal
    done: Signal
    xn: Signal
    yn: Signal
    zn: Signal

    def __init__(
        self,
        lut: Sequence[int],
        *,
        wx: int,
        wz: int,
        mode: int,
        ahead: int = 0,
        seed: int = _SEED,
        rom_attrs: Attrs | None = None,
        early: int = 0,
    ) -> None:
        self._lut, self._wx, self._wz, self._mode = list(lut), wx, wz, mode
        self._ahead, self._seed, self._rom_attrs, self._early = ahead, seed, dict(rom_attrs or {}), early
        super().__init__(
            {
                "start": In(1),
                "vectoring": In(1),
                "x0": In(signed(wx)),
                "y0": In(signed(wx)),
                "z0": In(signed(wz)),
                "xa": In(wx),
                "ya": In(wx),
                "done": Out(1),
                "xn": Out(signed(wx)),
                "yn": Out(signed(wx)),
                "zn": Out(signed(wz)),
            }
            | ({"z_ahead": Out(1), "z_signs": Out(ahead - 1)} if ahead else {})
            | ({"early": Out(1)} if early else {})
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        n, wx, wz, mode = len(self._lut), self._wx, self._wz, self._mode
        wi = ceil_log2(n + 1)
        run, done = Signal(name="run"), Signal(name="done")
        i = Signal(wi, name="i", reset_less=True)
        vec = Signal(1, name="vec", reset_less=True) if mode == 2 else None
        # A rotation's `y` does not shrink, so its `u` takes every iteration's bit.
        wu = wx + 1 if mode == 1 else wx + n
        u = Signal(signed(wu), name="u", reset_less=True)
        xr = Signal(signed(wx), name="xr", reset_less=True)
        d = Signal(wx, name="d", reset_less=True)
        d_carry = Signal(name="d_carry", reset_less=True)
        z = Signal(signed(wz), name="z", reset_less=True)
        i_next = Signal(wi, name="i_next")
        m.d.comb += i_next.eq((i + 1)[:wi])
        last = Signal(name="last")
        m.d.comb += last.eq(i_next >= n)
        # What the load is: a vectoring's, with its addends, or a rotation's.
        vectors = (Const(0), Const(1), self.vectoring)[mode]
        seed = Mux(vectors, Const(_SEED, wi), Const(self._seed, wi))

        m.d.sync += done.eq(run & last)
        with m.If(run):
            m.d.sync += i.eq(i_next)
            with m.If(last):
                m.d.sync += run.eq(0)
        with m.Elif(self.start):
            m.d.sync += run.eq(1)
            m.d.sync += i.eq(seed)
            if vec is not None:
                m.d.sync += vec.eq(self.vectoring)

        neg = Signal(name="neg")
        m.d.comb += neg.eq(Mux(vec, ~u[-1], z[-1]) if vec is not None else (z[-1], ~u[-1])[mode])
        sub_z = Signal(name="sub_z")
        m.d.comb += sub_z.eq(~neg)
        x_now = Signal(signed(wx), name="x_now")
        m.d.comb += x_now.eq((xr + d + d_carry)[:wx])
        index = Signal(wi, name="index")
        m.d.comb += index.eq(Mux(run, i_next, seed))
        kept = Signal(wx, name="kept", reset_less=True)
        m.d.sync += kept.eq(Cat(index <= k for k in range(wx)))
        cleared = Signal(wx, name="cleared")
        m.d.comb += cleared.eq(x_now.as_unsigned() & kept)
        # The loads, at their seed's scale, halved for the doubling that every step of `u` takes.
        y_load, y_add = Signal(signed(wu), name="y_load"), Signal(signed(wu), name="y_add")
        m.d.comb += y_load.eq(Mux(vectors, self.y0 << (_SEED - 1), self.y0 << (self._seed - 1)))
        m.d.comb += y_add.eq(self.ya << (_SEED - 1))
        u_in = Signal(signed(wu), name="u_in")
        m.d.comb += u_in.eq(Mux(run, u, y_load))
        u_by = Signal(signed(wu), name="u_by")
        m.d.comb += u_by.eq(Mux(run, cleared.as_signed() ^ neg.replicate(wx).as_signed(), y_add))
        u_next = Signal(signed(wu), name="u_next")
        m.d.comb += u_next.eq(((u_in + u_by + (run & neg)) << 1)[:wu])
        with m.If(run | self.start):
            m.d.sync += u.eq(u_next)
        with m.If(run):
            m.d.sync += xr.eq(x_now)
            m.d.sync += d.eq((u >> (2 * i))[:wx] ^ sub_z.replicate(wx))
            m.d.sync += d_carry.eq(sub_z)
        with m.Elif(self.start):
            m.d.sync += xr.eq(self.x0)
            m.d.sync += d.eq(self.xa)
            m.d.sync += d_carry.eq(0)
        # What is read past the last angle by the idle engine is not used. The angle of the iteration under way is read
        # a cycle ahead.
        lut = Array(Const(a, wz) for a in self._lut)
        angle = Signal(wz, name="angle", attrs=self._rom_attrs)
        m.d.comb += angle.eq(lut[index])
        angle_r = Signal(wz, name="angle_r", reset_less=True)
        m.d.sync += angle_r.eq(angle)
        # `z` keeps a load of its own: with a bare sum ahead of it, a tool may retime the register into the adder, onto
        # the path of the sign.
        with m.If(run):
            m.d.sync += z.eq(_step(z, angle_r, sub_z))
        with m.Elif(self.start):
            m.d.sync += z.eq(self.z0)
        if self._ahead:
            k = self._ahead
            m.d.comb += self.z_ahead.eq(run & (i == n - k))
            signs = Signal(k - 1, name="z_signs", reset_less=True)
            with m.If(run):
                m.d.sync += signs.eq(Cat(signs[1:], z[-1]))
            m.d.comb += self.z_signs.eq(signs)

        m.d.comb += self.done.eq(done)
        if self._early:
            m.d.comb += self.early.eq(run & (i == n - self._early))
        m.d.comb += self.xn.eq(x_now)
        m.d.comb += self.yn.eq((u >> n)[:wx])
        m.d.comb += self.zn.eq(z)
        return m

"""The bench of `exp2` and `log2`, whose rows have one shape, and the directed operands of their accuracy checks."""

import enum
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from mpmath import mp

from _bench import Case, run_pipeline
from _operands import exp2_operands, log2_operands
import zkf._exp2
import zkf._log2
from zkf import Exp2, Log2, Zkf, ZkfFormat
from zkf._pmul import counts
from zkf._trans import _GUARD, kernel_counts, spec as table

SHORT = {
    "si": "stage_input",
    "sk": "stage_core_kernel",
    "sm": "stage_core_mul",
    "sc": "stage_core",
    "so": "stage_output",
}
"""The stages as the rows name them."""


def others(core: type[enum.Flag]) -> list[tuple[int, int, dict[str, Any]]]:
    """
    wexp, wman and what else a row names, of the operator whose registers `core` has: formats and multipliers that
    0.9.0 had no table for, degree 4 among them.
    """
    return [
        (8, 20, {}),
        (8, 25, {"wm": 17, "sk": 3}),
        (8, 30, {"wm": 24, "sk": 3, "sc": next(iter(core))}),
        (8, 40, {"sk": 4}),
        (8, 44, {"wm": 27, "sk": 3, "sc": ~core(0)}),
        (6, 17, {"wm": 8, "sk": 3}),
    ]


def staging(core: type[enum.Flag]) -> list[dict[str, Any]]:
    """
    Stages on the cheapest format that can be walked whole, where they change no result: the kernel's grid, each
    register of `core` and all of them, the ends, and all of those.
    """
    every = ~core(0)
    return [
        {"sk": 3},
        *({"sc": member} for member in core),
        {"sc": every},
        {"si": 1},
        {"so": 1},
        {"si": 2, "so": 2},
        {"si": 1, "sk": 3, "sc": every, "so": 1},
    ]


def build(function: str, fmt: ZkfFormat, wmultiplier: int = 18, **stages: Any) -> Exp2 | Log2:
    """
    The model, its stages by their own names; the count of a product not given is the least that it takes, in the
    table of the operator's module, which a test may replace.
    """
    module = zkf._exp2 if function == "exp2" else zkf._log2
    spec = module.table(function, fmt.wman, wmultiplier)
    least = {"stage_core_kernel": kernel_counts(spec, wmultiplier).start}
    if function == "log2":
        least["stage_core_mul"] = counts(*zkf._log2._last(spec, fmt), signed=False, wmultiplier=wmultiplier).start
    cls: Callable[..., Exp2 | Log2] = Exp2 if function == "exp2" else Log2
    return cls(fmt, wmultiplier=wmultiplier, **(least | stages))


def run(function: str, wexp: int, wman: int, kind: str | int, shard: int = 0, shards: int = 1, **short: Any) -> None:
    """A row names the stages in short, and with `wm` a multiplier width other than 18."""
    width = short.pop("wm", 18)
    stages = {SHORT[name]: value for name, value in short.items()}
    simulate(build(function, ZkfFormat(wexp, wman), width, **stages), kind, shard, shards)


def simulate(model: Exp2 | Log2, kind: str | int, shard: int = 0, shards: int = 1) -> None:
    """`kind` is "every" operand, of one shard of so many; the "strata"; or a count of directed and random ones."""
    fmt, function = model.fmt, model.operator
    xs: list[int] | range
    if kind == "every":
        xs = range(1 << fmt.wfull)[shard::shards]
    elif kind == "strata":
        xs = list(dict.fromkeys(x for values in strata(function, fmt).values() for x in values))
    else:
        assert isinstance(kind, int)
        xs = exp2_operands(fmt, kind, fmt.wman + _GUARD) if function == "exp2" else log2_operands(fmt, kind)
    cases = []
    for x in xs:
        found = model.evaluate(Zkf(fmt, x))
        cases.append(Case({"x": x}, {"y": found} if isinstance(found, Zkf) else found._asdict()))
    run_pipeline(model.component(), model.timing.latency, cases)


def near(fmt: ZkfFormat, bases: list[int], span: int) -> list[int]:
    """Every normal encoding within `span` steps of a base, on its side of zero."""
    out: set[int] = set()
    for base in bases:
        sign = base & (1 << fmt.sign_shift)
        magnitude = base ^ sign
        low, high = max(1 << fmt.wfrac, magnitude - span), min(magnitude + span + 1, fmt.exp_inf << fmt.wfrac)
        out.update(sign | m for m in range(low, high))
    return sorted(out)


def edges(fmt: ZkfFormat) -> list[int]:
    """Infinities and zeros, with and without a fraction, and where `exp2` goes out of range, with its last inside."""
    top = fmt.bias + fmt.wexp - 1
    fields = [(0, 0), (0, 1), (fmt.exp_inf, 0), (fmt.exp_inf, fmt.frac_mask), (top, 0), (top - 1, fmt.frac_mask)]
    return [fmt.pack(sign, exp, frac).bits for sign in (0, 1) for exp, frac in fields]


def boundaries(fmt: ZkfFormat, band: int = 48) -> list[int]:
    """
    Of `exp2`: a band around every integer, where the result changes binade and, at the ends, saturates; bands
    where the result crosses the floats next to 1, where faithfulness is tested hardest; the largest reduced
    argument, `-2**-cf`; and a few fractions at each exponent below it.
    """
    cf = fmt.wman + _GUARD
    integers = range(1, 1 << (fmt.wexp - 1))
    exponents = range(1 - fmt.bias, -cf)
    bases = [fmt.encode(Fraction(sign * n)).bits for n in integers for sign in (1, -1)]
    with mp.workprec(4 * fmt.wman + 80):
        for k in range(1, 5):
            for x in (mp.log(1 + mp.ldexp(k, -fmt.wfrac), 2), mp.log(1 - mp.ldexp(k, -fmt.wman), 2)):
                bases.append(fmt.encode(x).bits)
    out = near(fmt, bases, band) + near(fmt, [fmt.pack(1, fmt.bias - cf, 0).bits], 4)
    fractions = (0, 1, 1 << (fmt.wfrac - 1), fmt.frac_mask)
    return out + [fmt.pack(s, fmt.bias + e, f).bits for e in exponents for s in (0, 1) for f in fractions]


def seams(function: str, fmt: ZkfFormat) -> list[int]:
    """
    Around every boundary of the table's segments, where two fits meet. `exp2` selects the segment by the fraction
    of `x`; `log2` by the top bits of the significand, taken here in the binades on either side of 1 and one more
    on each side, with the square root of 2, where `|log2(m')|` crosses a half next to where re-centering starts.
    """
    spec = table(function, fmt.wman, 18)
    if function == "exp2":
        steps = [Fraction(j, 1 << spec.k) - d for j in range(1, 1 << spec.k) for d in (0, 1)]
        return near(fmt, [fmt.encode(sign * step).bits for step in steps for sign in (1, -1)], 16)
    with mp.workprec(4 * fmt.wman + 80):
        root = int(mp.sqrt(2) * (1 << fmt.wfrac)) - (1 << fmt.wfrac)
    fractions = sorted({j << spec.rw for j in range(1 << spec.k)} | {root})
    exponents = [e for e in (-2, -1, 0, 1) if 1 <= fmt.bias + e < fmt.exp_inf]
    return near(fmt, [fmt.pack(0, fmt.bias + e, f).bits for e in exponents for f in fractions], 16)


def near_one(fmt: ZkfFormat, span: int) -> list[int]:
    """
    Of `log2`: the `span` floats on either side of 1, which random operands never hit; `1 +- 2**-j` and
    `2**(+-2**-j)` out to where they meet; and `2**(+-2**j)`, the least normal and the largest finite, whose results
    cross a power of two.
    """
    span = min(span, 1 << fmt.wfrac)
    ones = fmt.frac_mask
    out = [((fmt.bias - 1) << fmt.wfrac) | f for f in range(ones - span + 1, ones + 1)]
    out += [(fmt.bias << fmt.wfrac) | f for f in range(span)]
    bases = [fmt.pack(0, 1, 0).bits, fmt.pack(0, fmt.exp_inf - 1, ones).bits]
    exponents = {sign * ((1 << j) + d) for j in range(fmt.wexp - 1) for sign in (1, -1) for d in range(-3, 4)}
    bases += [fmt.pack(0, fmt.bias + e, 0).bits for e in sorted(exponents) if 1 - fmt.bias <= e <= fmt.bias]
    with mp.workprec(4 * fmt.wman + 80):
        for j in range(1, fmt.wman):
            step = mp.ldexp(1, -j)
            for x in (1 + step, 1 - step, mp.power(2, step), mp.power(2, -step)):
                bases.append(fmt.encode(x).bits)
    return out + near(fmt, bases, 4)


def strata(function: str, fmt: ZkfFormat, span: int = 1 << 12) -> dict[str, list[int]]:
    """By name; `span` is that of `near_one`, the widest of them."""
    if function == "exp2":
        return {"edges": edges(fmt), "boundary": boundaries(fmt), "seam": seams(function, fmt)}
    return {"edges": edges(fmt), "seam": seams(function, fmt), "near1": near_one(fmt, span)}

"""
What holds of any right adder and multiplier, whatever the reference says: a violation is a fault of the hardware
even where the reference shares it.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from amaranth.hdl import Value
from amaranth.sim import SimulatorContext

from _bench import _Dut
from _operands import SEED, directed_numbers, random_operand
from zkf import Add, Mul, Zkf, ZkfFormat
from zkf._pmul import counts

FORMATS = [(2, 4), (3, 4), (3, 5), (4, 6), (5, 11), (6, 18), (7, 17), (8, 24), (11, 53)]
STAGES: dict[str, list[dict[str, Any]]] = {
    "add": [
        {"stage_core": Add.CoreStages.SUM},
        {
            "stage_core": Add.CoreStages.OPERANDS
            | Add.CoreStages.ALIGNED
            | Add.CoreStages.SUM
            | Add.CoreStages.NORMALIZED
        },
        {"stage_core": ~Add.CoreStages(0)},
    ],
    "mul": [{"stage_core_mul": count} for count in (2, 3)],
}


def results(model: Add | Mul, pairs: Sequence[tuple[int, int]]) -> list[int]:
    """What the operator gives of each pair, held at its inputs for as long as its latency."""
    dut = model.component()
    bench = _Dut(dut, ("in_valid", "out_valid"), [])
    out: list[int] = []

    async def testbench(ctx: SimulatorContext) -> None:
        ctx.set(bench.domain.rst, 1)
        await ctx.tick()
        ctx.set(bench.domain.rst, 0)
        ctx.set(dut.in_valid, 1)
        for a, b in pairs:
            ctx.set(Value.cast(dut.a), a)
            ctx.set(Value.cast(dut.b), b)
            await ctx.tick().repeat(model.timing.latency)
            assert ctx.get(dut.out_valid)
            out.append(ctx.get(Value.cast(dut.y)))

    bench.sim.add_testbench(testbench)
    bench.sim.run()
    return out


def corners(fmt: ZkfFormat) -> list[int]:
    """Zeros and infinities, canonical and not, ones, and the ends of the range, at any exponent width."""
    top, ones, stray = fmt.exp_max_finite, fmt.frac_mask, min(fmt.frac_mask, 1)
    out = [
        0,
        fmt.pack(1, 0, 0).bits,
        fmt.pack(0, 0, stray).bits,
        fmt.inf(0).bits,
        fmt.inf(1).bits,
    ]
    out += [fmt.pack(0, fmt.exp_inf, stray).bits, fmt.pack(1, fmt.exp_inf, ones).bits]
    out += [fmt.pack(sign, exp, frac).bits for sign in (0, 1) for exp, frac in ((fmt.bias, 0), (1, 0), (top, ones))]
    return out + [fmt.pack(0, fmt.bias, 1 << (fmt.wfrac - 1)).bits, fmt.pack(1, fmt.bias + 1, 1).bits]


@pytest.mark.parametrize("fmt", [ZkfFormat(*row) for row in FORMATS], ids=str)
@pytest.mark.parametrize("operator", STAGES)
def test_properties(operator: str, fmt: ZkfFormat) -> None:
    rng = np.random.default_rng(SEED)
    directed = list(directed_numbers(fmt).values()) if fmt.wexp >= 3 else []
    commuted = [(a, b) for a in directed for b in directed]
    commuted += [(random_operand(fmt, rng), random_operand(fmt, rng)) for _ in range(64)]
    singles = corners(fmt) + [random_operand(fmt, rng) for _ in range(16)]
    signed = [(a, b) for a in corners(fmt) for b in corners(fmt)]
    signed += [(random_operand(fmt, rng), random_operand(fmt, rng)) for _ in range(24)]
    plus_zero, minus_zero, one = 0, fmt.pack(1, 0, 0).bits, fmt.pack(0, fmt.bias, 0).bits

    def negated(bits: int) -> int:
        return bits ^ (1 << fmt.sign_shift)

    def opposite(result: int) -> int:
        """Of a result, which is canonical: a zero has one sign."""
        return plus_zero if result == plus_zero else negated(result)

    def canonical(bits: int) -> int:
        return Zkf(fmt, bits).canonicalize().bits

    for stages in STAGES[operator]:
        taken = counts(fmt.wman, fmt.wman, signed=False, wmultiplier=18)
        if operator == "mul" and stages["stage_core_mul"] not in taken:
            continue
        model: Add | Mul = Add(fmt, **stages) if operator == "add" else Mul(fmt, wmultiplier=18, **stages)

        def given(pairs: Sequence[tuple[int, int]]) -> list[int]:
            return results(model, pairs)

        assert given(commuted) == given([(b, a) for a, b in commuted])
        direct = given(signed)
        if operator == "mul":
            assert given([(a, one) for a in singles]) == [canonical(a) for a in singles]
            assert given([(a, plus_zero) for a in singles]) == [plus_zero] * len(singles)
            assert given([(negated(a), b) for a, b in signed]) == [opposite(y) for y in direct]
        else:
            for nothing in (plus_zero, minus_zero):
                assert given([(a, nothing) for a in singles]) == [canonical(a) for a in singles]
            assert given([(a, negated(a)) for a in singles]) == [plus_zero] * len(singles)
            assert given([(negated(a), negated(b)) for a, b in signed]) == [opposite(y) for y in direct]

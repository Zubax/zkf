#!/usr/bin/env python3
"""
zkf_cordic against Zkf.sincos / Zkf.atan2, which the dedicated operators' benches pin to their RTL. At a fixed MODE
only that mode's cases run and `vectoring` is noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

from test_atan2 import cases_for as atan2_cases
from test_sincos import cases_for as sincos_cases
from zkf import CordicModel, Timing, ZkfFormat
from zkf_bits import hex_bits
from zkf_operands import random_bits
from zkf_params import check_width, float_context
from zkf_stream import drive_unsigned, expect_reaccept, reset_boundaries, start_clock


@dataclass(frozen=True)
class CordicCase:
    label: str
    vectoring: bool
    a: int
    b: int
    r0: int
    r1: int
    quadrant: int

    def describe(self, fmt: ZkfFormat) -> str:
        mode = "vectoring" if self.vectoring else "rotation"
        return f"{mode} {self.label} a={hex_bits(self.a, fmt.wfull)} b={hex_bits(self.b, fmt.wfull)}"


def rotation(fmt: ZkfFormat, label: str, x: int, noise: int) -> CordicCase:
    r = fmt.wrap(x).sincos()
    return CordicCase(label, False, x, noise, r.sin.bits, r.cos.bits, r.quadrant)


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int, mode: int = 2) -> list[CordicCase]:
    rng = np.random.default_rng(seed)
    cases = []
    if mode != 1:
        cases += [
            CordicCase(c.label, False, c.x, random_bits(fmt.wfull, rng), c.sin, c.cos, c.quadrant)
            for c in sincos_cases(fmt, kind, seed, count)
        ]
        # Rotation near k/2 turn, where |sin| is smallest: its underflow decision (reachable when BIAS < WFRAC) and
        # the union packer's re-based exponent both sit here.
        bias = (1 << (fmt.wexp - 1)) - 1
        for exp in (bias - 1, bias):
            center = exp << fmt.wfrac
            for ulps in (1, 2, 3, 5, 8):
                for bits in (center - ulps, center + ulps):
                    cases.append(rotation(fmt, "half-turn", bits, random_bits(fmt.wfull, rng)))
    if mode != 0:
        cases += [CordicCase(c.label, True, c.y, c.x, c.theta, c.mag, 0) for c in atan2_cases(fmt, kind, seed, count)]
    return [cases[i] for i in rng.permutation(len(cases))]


def expected_timing(context) -> Mapping[int, Timing]:
    """Keyed by the case's `vectoring` bool, which hashes as 0/1."""
    timing = CordicModel(
        ZkfFormat(context.wexp, context.wman),
        mode=context.mode,
        unroll100=context.unroll100,
        stage_input=context.stage_input,
        stage_product=context.stage_product,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    ).timing
    return {context.mode: timing} if isinstance(timing, Timing) else timing


def max_latency(context) -> int:
    return max(t.latency for t in expected_timing(context).values())


def _outputs(dut) -> dict[str, int]:
    return {"r0": int(dut.r0.value), "r1": int(dut.r1.value), "quadrant": int(dut.quadrant.value)}


def _expected(case: CordicCase) -> dict[str, int]:
    return {"r0": case.r0, "r1": case.r1, "quadrant": case.quadrant}


async def _reset(dut, out_ready: int) -> None:
    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = out_ready
    dut.vectoring.value = 0
    drive_unsigned(dut.a, 0)
    drive_unsigned(dut.b, 0)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def _issue(dut, case: CordicCase, rng: np.random.Generator, mode: int) -> None:
    dut.in_valid.value = 1
    dut.vectoring.value = int(case.vectoring if mode == 2 else rng.integers(0, 2))
    drive_unsigned(dut.a, case.a)
    drive_unsigned(dut.b, case.b)


def _scramble(dut, rng: np.random.Generator, wfull: int) -> None:
    dut.in_valid.value = 0
    dut.vectoring.value = int(rng.integers(0, 2))
    drive_unsigned(dut.a, random_bits(wfull, rng))
    drive_unsigned(dut.b, random_bits(wfull, rng))


@cocotb.test()
async def cordic_runtime_cases(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    timing = expected_timing(context)
    for name in ("a", "b", "r0", "r1"):
        check_width(name, getattr(dut, name), fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count, context.mode)
    transitions = {(a.vectoring, b.vectoring) for a, b in zip(cases, cases[1:])}
    assert len(transitions) == (4 if context.mode == 2 else 1), f"{context.prefix()}: mode transitions {transitions}"
    rng = np.random.default_rng(context.seed + 1)
    await _reset(dut, out_ready=1)

    timeout = max_latency(context) + 8
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: in_ready stuck low (case {index})"
        _issue(dut, case, rng, context.mode)
        await RisingEdge(dut.clk)
        _scramble(dut, rng, fmt.wfull)
        guard = 0
        while int(dut.out_valid.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: out_valid timeout (case {index})"
            assert int(dut.in_ready.value) == 0, f"{context.prefix()}: in_ready high while busy (case {index})"
        want = timing[case.vectoring]
        assert guard == want.latency, f"{context.prefix()} {case.describe(fmt)}: latency {guard} != model {want}"
        got, exp = _outputs(dut), _expected(case)
        assert got == exp, f"{context.prefix()} case={index} {case.describe(fmt)}: got {got} expected {exp}"
        await expect_reaccept(dut, f"{context.prefix()} case={index}", want.initiation_interval - want.latency)


@cocotb.test()
async def cordic_random_handshake(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    timing = expected_timing(context)
    cases = cases_for(fmt, context.kind, context.seed, context.count, context.mode)[:128]
    rng = np.random.default_rng(context.seed + 2)
    await _reset(dut, out_ready=0)

    issued, taken, cycle = 0, 0, 0
    pending: tuple[CordicCase, int] | None = None
    held: CordicCase | None = None
    budget = 8 * len(cases) * (max_latency(context) + 2)
    while taken < len(cases):
        assert cycle < budget, f"{context.prefix()}: {taken}/{len(cases)} results in {budget} cycles"
        offer = issued < len(cases) and rng.random() < 0.5
        if offer:
            _issue(dut, cases[issued], rng, context.mode)
        else:
            _scramble(dut, rng, fmt.wfull)
        dut.out_ready.value = out_ready = int(rng.random() < 0.5)
        await ReadOnly()
        if pending is not None or held is not None:
            assert int(dut.in_ready.value) == 0, f"{context.prefix()} cycle {cycle}: in_ready high while busy"
        if int(dut.out_valid.value):
            if held is None:
                assert pending is not None, f"{context.prefix()} cycle {cycle}: out_valid with nothing in flight"
                held, accepted_at = pending
                pending = None
                assert (
                    cycle - accepted_at == timing[held.vectoring].latency
                ), f"{context.prefix()} {held.describe(fmt)}: latency"
            got = _outputs(dut)
            assert got == _expected(held), f"{context.prefix()} {held.describe(fmt)}: got {got}"
            if out_ready:
                held, taken = None, taken + 1
        if pending is not None:
            case, accepted_at = pending
            assert (
                cycle - accepted_at < timing[case.vectoring].latency
            ), f"{context.prefix()} {case.describe(fmt)}: no result"
        if offer and int(dut.in_ready.value):
            pending, issued = (cases[issued], cycle), issued + 1
        await RisingEdge(dut.clk)
        cycle += 1


@cocotb.test()
async def cordic_reset_boundaries(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    directed = [c for c in cases_for(fmt, "directed", context.seed, 0, context.mode) if c.r0 and c.r1]
    one = directed[0]
    other = next(c for c in directed if c.vectoring != one.vectoring or (context.mode != 2 and c.r0 != one.r0))
    rng = np.random.default_rng(context.seed + 3)
    await reset_boundaries(
        dut,
        context.prefix(),
        [(one, other), (other, one)],
        lambda case: _issue(dut, case, rng, context.mode),
        lambda: _scramble(dut, rng, fmt.wfull),
        lambda: _outputs(dut),
        _expected,
        lambda case: case.describe(fmt),
        max_latency(context) + 8,
    )

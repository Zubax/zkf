#!/usr/bin/env python3
"""zkf_cordic against Zkf.sincos / Zkf.atan2, which the dedicated operators' benches pin to their RTL."""

from __future__ import annotations

from dataclasses import dataclass

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

from test_atan2 import cases_for as atan2_cases
from test_sincos import cases_for as sincos_cases
from zkf import ZkfFormat
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


def cases_for(fmt: ZkfFormat, kind: str, seed: int, count: int) -> list[CordicCase]:
    rng = np.random.default_rng(seed)
    cases = [
        CordicCase(c.label, False, c.x, random_bits(fmt.wfull, rng), c.sin, c.cos, c.quadrant)
        for c in sincos_cases(fmt, kind, seed, count)
    ]
    # Rotation near k/2 turn, where |sin| is smallest: its underflow decision (reachable when BIAS < WFRAC) and the
    # union packer's re-based exponent both sit here.
    bias = (1 << (fmt.wexp - 1)) - 1
    for exp in (bias - 1, bias):
        center = exp << fmt.wfrac
        for ulps in (1, 2, 3, 5, 8):
            for bits in (center - ulps, center + ulps):
                cases.append(rotation(fmt, "half-turn", bits, random_bits(fmt.wfull, rng)))
    cases += [CordicCase(c.label, True, c.y, c.x, c.theta, c.mag, 0) for c in atan2_cases(fmt, kind, seed, count)]
    return [cases[i] for i in rng.permutation(len(cases))]


def expected_latency(context) -> dict[bool, int]:
    model = ZkfFormat(context.wexp, context.wman).model_of("cordic")(
        unroll100=context.unroll100,
        stage_input=context.stage_input,
        stage_product=context.stage_product,
        stage_normalize=context.stage_normalize,
        stage_pack=context.stage_pack,
        stage_output=context.stage_output,
    )
    return {False: model.latency_rotation, True: model.latency_vectoring}


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


def _issue(dut, case: CordicCase) -> None:
    dut.in_valid.value = 1
    dut.vectoring.value = int(case.vectoring)
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
    latency = expected_latency(context)
    for name in ("a", "b", "r0", "r1"):
        check_width(name, getattr(dut, name), fmt.wfull, context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)
    assert {c.vectoring for c in cases} == {False, True}, f"{context.prefix()}: both modes must be exercised"
    rng = np.random.default_rng(context.seed + 1)
    await _reset(dut, out_ready=1)

    timeout = max(latency.values()) + 8
    for index, case in enumerate(cases):
        guard = 0
        while int(dut.in_ready.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: in_ready stuck low (case {index})"
        _issue(dut, case)
        await RisingEdge(dut.clk)
        _scramble(dut, rng, fmt.wfull)
        guard = 0
        while int(dut.out_valid.value) == 0:
            await RisingEdge(dut.clk)
            guard += 1
            assert guard < timeout, f"{context.prefix()}: out_valid timeout (case {index})"
        want = latency[case.vectoring]
        assert guard == want, f"{context.prefix()} {case.describe(fmt)}: latency {guard} != model {want}"
        got, exp = _outputs(dut), _expected(case)
        assert got == exp, f"{context.prefix()} case={index} {case.describe(fmt)}: got {got} expected {exp}"
        await expect_reaccept(dut, f"{context.prefix()} case={index}")


@cocotb.test()
async def cordic_random_handshake(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    latency = expected_latency(context)
    cases = cases_for(fmt, context.kind, context.seed, context.count)[:128]
    rng = np.random.default_rng(context.seed + 2)
    await _reset(dut, out_ready=0)

    issued, taken, cycle = 0, 0, 0
    pending: tuple[CordicCase, int] | None = None
    held: CordicCase | None = None
    budget = 8 * len(cases) * (max(latency.values()) + 2)
    while taken < len(cases):
        assert cycle < budget, f"{context.prefix()}: {taken}/{len(cases)} results in {budget} cycles"
        offer = issued < len(cases) and rng.random() < 0.5
        if offer:
            _issue(dut, cases[issued])
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
                    cycle - accepted_at == latency[held.vectoring]
                ), f"{context.prefix()} {held.describe(fmt)}: latency"
            got = _outputs(dut)
            assert got == _expected(held), f"{context.prefix()} {held.describe(fmt)}: got {got}"
            if out_ready:
                held, taken = None, taken + 1
        if pending is not None:
            case, accepted_at = pending
            assert cycle - accepted_at < latency[case.vectoring], f"{context.prefix()} {case.describe(fmt)}: no result"
        if offer and int(dut.in_ready.value):
            pending, issued = (cases[issued], cycle), issued + 1
        await RisingEdge(dut.clk)
        cycle += 1


@cocotb.test()
async def cordic_reset_boundaries(dut) -> None:
    context = float_context("cordic")
    fmt = ZkfFormat(context.wexp, context.wman)
    directed = [c for c in cases_for(fmt, "directed", context.seed, 0) if c.r0 and c.r1]
    rot = next(c for c in directed if not c.vectoring)
    vec = next(c for c in directed if c.vectoring)
    rng = np.random.default_rng(context.seed + 3)
    await reset_boundaries(
        dut,
        context.prefix(),
        [(rot, vec), (vec, rot)],
        lambda case: _issue(dut, case),
        lambda: _scramble(dut, rng, fmt.wfull),
        lambda: _outputs(dut),
        _expected,
        lambda case: case.describe(fmt),
        max(expected_latency(context).values()) + 8,
    )

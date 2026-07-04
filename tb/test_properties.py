#!/usr/bin/env python3
"""
Model-independent algebraic-property tests: self-consistency identities (commutativity, algebraic
identities) that hold for any correct implementation, so a failure flags an RTL bug even when the Python
model shares it. The scaffolding dispatches by DUT port shape, so one file serves any binary toplevel
exposing (clk, rst, in_valid, a, b, out_valid, y): zkf_mul, zkf_add, zkf_addsub.
"""

from __future__ import annotations

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, Timer

from zkf import ZkfFormat
from zkf_bits import hex_bits
from zkf_operands import canonical_inf, directed_numbers, normal, pack_bits, random_operand, zero
from zkf_latency import add_latency, mul_latency
from zkf_params import check_width, float_context
from zkf_stream import drive_unsigned, is_resolvable, start_clock


def operand_pairs(fmt: ZkfFormat, seed: int, count: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    if fmt.wexp >= 3:
        directed = list(directed_numbers(fmt).values())
        for a in directed:
            for b in directed:
                pairs.append((a, b))
    while len(pairs) < count:
        pairs.append((random_operand(fmt, rng), random_operand(fmt, rng)))
    return pairs[: max(count, len(pairs))]


async def reset_dut(dut, stages: int) -> None:
    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_unsigned(dut.a, 0)
    drive_unsigned(dut.b, 0)
    if hasattr(dut, "op_sub"):
        dut.op_sub.value = 0
    for _ in range(stages + 2):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def drive_and_capture(dut, a: int, b: int, stages: int, op_sub: int = 0) -> int:
    """
    Drive (a, b) one cycle; expect out_valid=1 after exactly stages clock edges.
    op_sub selects subtraction on toplevels that expose it (zkf_addsub); ignored otherwise.
    """
    drive_unsigned(dut.a, a)
    drive_unsigned(dut.b, b)
    if hasattr(dut, "op_sub"):
        dut.op_sub.value = op_sub
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.in_valid.value = 0
    drive_unsigned(dut.a, 0)
    drive_unsigned(dut.b, 0)
    for _ in range(stages - 1):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    assert is_resolvable(dut.out_valid), "out_valid is unresolved at expected result cycle"
    assert int(dut.out_valid.value) == 1, "out_valid was not asserted at expected result cycle"
    assert is_resolvable(dut.y), "y is unresolved at expected result cycle"
    return int(dut.y.value)


def infer_stages(dut, stage_product: int = 0, stage_decode: int = 0, stage_align: int = 0,
                 stage_output: int = 0) -> int:
    """
    Pipeline depth per toplevel (knobs hardcoded): zkf_mul has STAGE_PRODUCT; zkf_add/zkf_addsub have
    STAGE_DECODE and STAGE_ALIGN; all carry STAGE_OUTPUT (0=combinational, 1=registered).
    """
    name = str(dut._name)
    if "mul" in name:
        return mul_latency(stage_product=stage_product, stage_output=stage_output)
    if "addsub" in name or "add" in name:
        return add_latency(stage_decode=stage_decode, stage_align=stage_align, stage_output=stage_output)
    raise RuntimeError(f"unknown toplevel for property test: {name}")


@cocotb.test()
async def commutativity(dut) -> None:
    context = float_context("properties")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    stages = infer_stages(dut, context.stage_product, context.stage_decode, context.stage_align,
                          context.stage_output)

    start_clock(dut)
    await reset_dut(dut, stages)

    pairs = operand_pairs(fmt, context.seed, max(64, context.count or 64))
    failures = 0
    for index, (a, b) in enumerate(pairs):
        y_ab = await drive_and_capture(dut, a, b, stages)
        y_ba = await drive_and_capture(dut, b, a, stages)
        if y_ab != y_ba:
            failures += 1
            if failures <= 5:
                cocotb.log.error(
                    f"commutativity violation #{index}: "
                    f"a={hex_bits(a, fmt.wfull)} b={hex_bits(b, fmt.wfull)} "
                    f"y_ab={hex_bits(y_ab, fmt.wfull)} y_ba={hex_bits(y_ba, fmt.wfull)}"
                )
    assert failures == 0, (
        f"{context.prefix()} found {failures} commutativity violation(s); first five logged above"
    )


def special_operands(fmt: ZkfFormat) -> list[int]:
    """Corner operands valid for any WEXP >= 2 (directed_numbers needs WEXP >= 3)."""
    ops = [
        zero(fmt),                                              # +0
        pack_bits(fmt, 1, 0, 0),                                # -0 (non-canonical)
        pack_bits(fmt, 0, 0, min(fmt.frac_mask, 1)),            # +0 with stray fraction
        normal(fmt, 0, fmt.bias, 0),                            # +1
        normal(fmt, 1, fmt.bias, 0),                            # -1
        normal(fmt, 0, 1, 0),                                   # +min_normal
        normal(fmt, 1, 1, 0),                                   # -min_normal
        normal(fmt, 0, fmt.exp_max_finite, fmt.frac_mask),      # +max_finite
        normal(fmt, 1, fmt.exp_max_finite, fmt.frac_mask),      # -max_finite
        canonical_inf(fmt, 0),                                  # +inf
        canonical_inf(fmt, 1),                                  # -inf
        pack_bits(fmt, 0, fmt.exp_inf, min(fmt.frac_mask, 1)),  # +inf (non-canonical)
        pack_bits(fmt, 1, fmt.exp_inf, fmt.frac_mask),          # -inf (non-canonical)
    ]
    if fmt.wfrac >= 2:
        ops.append(normal(fmt, 0, fmt.bias, 1 << (fmt.wfrac - 1)))  # +1.5
        ops.append(normal(fmt, 1, fmt.bias + 1, 1))                # ~-2
    return ops


@cocotb.test()
async def algebraic_identities(dut) -> None:
    """
    Model-independent algebraic identities. Each holds for any correct implementation, so a
    violation flags an RTL bug even if the Python model shares it. Dispatched by toplevel.
    """
    context = float_context("properties")
    fmt = ZkfFormat(context.wexp, context.wman)
    check_width("a", dut.a, fmt.wfull, context)
    check_width("b", dut.b, fmt.wfull, context)
    check_width("y", dut.y, fmt.wfull, context)
    stages = infer_stages(dut, context.stage_product, context.stage_decode, context.stage_align,
                          context.stage_output)

    name = str(dut._name)
    is_mul = "mul" in name
    is_addsub = "addsub" in name
    is_add = ("add" in name) and not is_addsub
    has_sub = hasattr(dut, "op_sub")

    start_clock(dut)
    await reset_dut(dut, stages)

    rng = np.random.default_rng(context.seed)
    operands = special_operands(fmt)
    operands += [random_operand(fmt, rng) for _ in range(16)]
    pos_zero = zero(fmt)
    one = normal(fmt, 0, fmt.bias, 0)

    def neg_or_zero(y: int) -> int:
        # Flip the sign of a canonical result, but +0 stays canonical +0.
        return pos_zero if y == pos_zero else (-fmt.wrap(y)).bits

    fails: list[str] = []

    async def expect(label: str, a: int, b: int, op_sub: int, want: int) -> None:
        got = await drive_and_capture(dut, a, b, stages, op_sub)
        if got != want:
            fails.append(
                f"{label}: a={hex_bits(a, fmt.wfull)} b={hex_bits(b, fmt.wfull)} sub={op_sub} "
                f"got={hex_bits(got, fmt.wfull)} want={hex_bits(want, fmt.wfull)}"
            )

    for a in operands:
        if is_mul:
            await expect("mul_identity a*1==a", a, one, 0, fmt.wrap(a).canonicalize().bits)
            await expect("mul_annihilator a*0==+0", a, pos_zero, 0, pos_zero)
        if is_add or is_addsub:
            await expect("add_identity a+0==a", a, pos_zero, 0, fmt.wrap(a).canonicalize().bits)
            await expect("add_inverse a+(-a)==+0", a, (-fmt.wrap(a)).bits, 0, pos_zero)
        if is_addsub and has_sub:
            await expect("sub_self a-a==+0", a, a, 1, pos_zero)
            await expect("sub_identity a-0==a", a, pos_zero, 1, fmt.wrap(a).canonicalize().bits)

    # Pair identities over specials only (O(n^2)) plus a few random pairs.
    pairs = [(a, b) for a in special_operands(fmt) for b in special_operands(fmt)]
    pairs += [(random_operand(fmt, rng), random_operand(fmt, rng)) for _ in range(24)]
    for a, b in pairs:
        if is_mul:
            y_ab = await drive_and_capture(dut, a, b, stages, 0)
            await expect("mul_sign (-a)*b==-(a*b)", (-fmt.wrap(a)).bits, b, 0, neg_or_zero(y_ab))
        if is_add or is_addsub:
            y_ab = await drive_and_capture(dut, a, b, stages, 0)
            await expect("add_neg (-a)+(-b)==-(a+b)", (-fmt.wrap(a)).bits, (-fmt.wrap(b)).bits, 0,
                         neg_or_zero(y_ab))
        if is_addsub and has_sub:
            y_addnegb = await drive_and_capture(dut, a, (-fmt.wrap(b)).bits, stages, 0)
            await expect("sub_via_neg a-b==a+(-b)", a, b, 1, y_addnegb)

    for line in fails[:8]:
        cocotb.log.error(f"identity violation: {line}")
    assert not fails, f"{context.prefix()} found {len(fails)} algebraic-identity violation(s); first eight logged"

#!/usr/bin/env python3
"""
Standalone bench for the fused leading-zero-normalizing shifter _zkf_normshift.

Embedded callers (zkf_add, zkf_from_int) drive it with narrow, zero-padded inputs over a limited leading-one range, so
this bench sweeps x exhaustively at small widths -- every reachable cascade node toggles -- and checks (zero, count, y)
against the model across STAGE_SPLIT and STAGE_OUTPUT.

normshift_sideband_stream additionally pins the streaming sideband latency: it drives a fresh in_valid/sb_in every
cycle and asserts out_valid/sb_out are delayed by EXACTLY STAGE_SPLIT + STAGE_OUTPUT (with valid gaps and a flush),
which an input-holding sweep cannot.
"""

from __future__ import annotations

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, Timer

from zkf_bits import mask
from zkf_params import TestContext, plusarg_int, plusarg_str
from zkf_stream import (
    RegisterStageScoreboard,
    drive_unsigned,
    is_resolvable,
    run_stream_cases,
    start_clock,
)


def normshift_reference(width: int, value: int) -> tuple[int, int, int]:
    """
    Independent bench reference for _zkf_normshift: returns (zero, count, y). count = (width-1) - leading_one_pos,
    the left-shift bringing the leading 1 to the MSB; y = value << count. count and y are don't-care when zero.
    """
    value &= mask(width)
    if value == 0:
        return 1, 0, 0
    count = (width - 1) - (value.bit_length() - 1)
    return 0, count, (value << count) & mask(width)


def cases_for(width: int, kind: str, seed: int, count: int) -> list[int]:
    if kind == "exhaustive":
        # Trailing 0 after the all-ones value so every input bit also toggles 1->0.
        return list(range(1 << width)) + [0]
    xs = {0, 1, 1 << (width - 1), mask(width)}
    xs.update(1 << i for i in range(width))  # one-hot (each leading-one position)
    xs.update((1 << i) - 1 for i in range(1, width + 1))
    if kind == "directed":
        return sorted(xs)
    rng = np.random.default_rng(seed)
    while len(xs) < max(count, len(xs) + 1):
        xs.add(int(rng.integers(0, 1 << width)))
    return sorted(xs)


@cocotb.test()
async def normshift_runtime_cases(dut) -> None:
    width = plusarg_int("ZKF_NS_W")
    split = plusarg_int("ZKF_NS_SPLIT", 0)
    output = plusarg_int("ZKF_NS_OUTPUT", 0)
    kind = plusarg_str("ZKF_KIND", "exhaustive")
    seed = plusarg_int("ZKF_SEED", 0)
    count = plusarg_int("ZKF_COUNT", 0)
    cfg = plusarg_str("ZKF_CONFIG", "default")
    if len(dut.x) != width:
        raise AssertionError(f"{cfg}: x width {len(dut.x)} != ZKF_NS_W={width}")

    # Output latency = STAGE_SPLIT cascade barriers + optional STAGE_OUTPUT register. (out_valid/sb_out latency is
    # pinned separately by normshift_sideband_stream.)
    latency = split + output

    cases = cases_for(width, kind, seed, count)
    start_clock(dut)
    # (zero, count, y) are pure datapath (independent of in_valid/sb_in/rst), so this sweep leaves the streaming
    # controls idle and reads the settled result after the output latency.
    dut.rst.value = 0
    dut.in_valid.value = 0
    checked = 0
    for x in cases:
        drive_unsigned(dut.x, x)
        # Settle the combinational path (the latency==0 result), then advance latency clock edges holding x stable
        # across the register barriers. Settling before the first edge avoids a t=0 drive/clock race on the un-reset
        # datapath registers.
        await Timer(1, unit="ns")
        for _ in range(latency):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
        exp_zero, exp_count, exp_y = normshift_reference(width, x)
        assert is_resolvable(dut.zero), f"{cfg}: zero unresolved x={x:#x}"
        obs_zero = int(dut.zero.value)
        assert obs_zero == exp_zero, f"{cfg}: zero mismatch x={x:#x} got={obs_zero} exp={exp_zero}"
        if not exp_zero:
            assert is_resolvable(dut.count), f"{cfg}: count unresolved x={x:#x}"
            obs_count = int(dut.count.value)
            assert (
                obs_count == exp_count
            ), f"{cfg}: count mismatch x={x:#x} got={obs_count} exp={exp_count} (W={width} split={split})"
            assert is_resolvable(dut.y), f"{cfg}: y unresolved x={x:#x}"
            obs_y = int(dut.y.value)
            assert obs_y == exp_y, f"{cfg}: y mismatch x={x:#x} got={obs_y:#x} exp={exp_y:#x} (W={width} split={split})"
        checked += 1
    assert checked == len(cases), f"{cfg}: checked {checked} of {len(cases)}"


@cocotb.test()
async def normshift_sideband_stream(dut) -> None:
    """
    Pin the streaming sideband latency: out_valid/sb_out must be delayed by exactly STAGE_SPLIT + STAGE_OUTPUT. A
    fresh in_valid/sb_in every cycle makes a wrong delay mismatch the alternating sb_out (or the gated out_valid).
    """
    width = plusarg_int("ZKF_NS_W")
    split = plusarg_int("ZKF_NS_SPLIT", 0)
    output = plusarg_int("ZKF_NS_OUTPUT", 0)
    seed = plusarg_int("ZKF_SEED", 0)
    cfg = plusarg_str("ZKF_CONFIG", "default")
    sbw = len(dut.sb_in)
    latency = split + output

    context = TestContext(suite="normshift", config=cfg, seed=seed, stage_normalize=split, stage_output=output)
    start_clock(dut)
    rng = np.random.default_rng(seed ^ 0x5DEECE66D)
    sb_mask = (1 << sbw) - 1
    x_fixed = 1 << (width - 1)  # any nonzero magnitude; this test checks the sideband/valid channel, not (zero,count,y)
    sample_count = max(64, latency * 8)

    if latency == 0:
        # Purely combinational (no rst gating), so RegisterStageScoreboard's reset-flush model doesn't apply -- check
        # passthrough directly (mirrors test_pipe).
        dut.rst.value = 0
        for i in range(sample_count):
            valid = (i % 3) != 0
            value = int(rng.integers(0, sb_mask + 1))
            dut.in_valid.value = int(valid)
            dut.sb_in.value = value
            drive_unsigned(dut.x, x_fixed)
            await Timer(1, unit="ns")
            assert is_resolvable(dut.out_valid), f"{cfg}: out_valid unresolved i={i}"
            assert int(dut.out_valid.value) == int(valid), f"{cfg}: out_valid mismatch i={i}"
            if valid:
                assert is_resolvable(dut.sb_out), f"{cfg}: sb_out unresolved i={i}"
                assert (
                    int(dut.sb_out.value) == value
                ), f"{cfg}: sb_out mismatch i={i} got={int(dut.sb_out.value)} exp={value}"
            await RisingEdge(dut.clk)
        return

    scoreboard = RegisterStageScoreboard(dut, latency, context, {"sb_out": (dut.sb_out, sbw)})

    def drive_case(case: int) -> dict[str, int]:
        drive_unsigned(dut.x, x_fixed)
        dut.sb_in.value = case
        return {"sb_out": case}

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.x, x_fixed)
        dut.sb_in.value = 0

    cases = [int(rng.integers(0, sb_mask + 1)) for _ in range(sample_count)]
    await run_stream_cases(
        dut,
        scoreboard,
        cases,
        drive_case,
        invalid_drive,
        lambda index, case: f"{cfg}: i={index} sb={case:#x}",
    )
    assert scoreboard.checked >= sample_count, f"{cfg}: checked {scoreboard.checked} of {sample_count}"

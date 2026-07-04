#!/usr/bin/env python3
"""
Standalone bench for the sticky-folded right shifter _zkf_rshift_sticky.

Embedded callers (zkf_to_int, zkf_add) drive only a limited shift-amount range, leaving the upper radix-4 cascade
stages and the over-range saturation path under-toggled. This bench sweeps (x, shamt) across the FULL shamt range
(including shamt >= W, the saturation collapse) and checks y against the model, for both STAGE_SPLIT polarities.
"""

from __future__ import annotations

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, Timer

from zkf_bits import mask
from zkf_params import plusarg_int, plusarg_str
from zkf_stream import drive_unsigned, is_resolvable, start_clock


def rshift_sticky_reference(width: int, value: int, shamt: int) -> int:
    """
    Independent bench reference for _zkf_rshift_sticky: y = value >> shamt, with y[0] OR-collecting every dropped
    bit (and the bit landing at position 0). For shamt >= width the result is {0, |value}.
    """
    value &= mask(width)
    if shamt >= width:
        shifted, dropped = 0, value
    else:
        shifted, dropped = value >> shamt, value & mask(shamt)
    return (shifted | (1 if dropped else 0)) & mask(width)


def shamt_values(width: int, wshift: int) -> list[int]:
    # 0..W (every in-range shift plus the first saturating value) and the top of the shamt range.
    vals = set(range(0, width + 1))
    vals.add(mask(wshift))
    vals.update(v for v in (width + 1, width + 2, 1 << (wshift - 1)) if v <= mask(wshift))
    return sorted(vals)


def x_values(width: int, kind: str, rng) -> list[int]:
    if kind == "exhaustive":
        return list(range(1 << width)) + [0]  # trailing 0 so every bit also toggles 1->0
    xs = {0, 1, mask(width), 1 << (width - 1)}
    xs.update(1 << i for i in range(width))
    xs.update(int(rng.integers(0, 1 << width)) for _ in range(8))
    return sorted(xs)


@cocotb.test()
async def rshift_runtime_cases(dut) -> None:
    width = plusarg_int("ZKF_RSH_W")
    split = plusarg_int("ZKF_RSH_SPLIT", 0)
    kind = plusarg_str("ZKF_KIND", "exhaustive")
    seed = plusarg_int("ZKF_SEED", 0)
    cfg = plusarg_str("ZKF_CONFIG", "default")
    if len(dut.x) != width:
        raise AssertionError(f"{cfg}: x width {len(dut.x)} != ZKF_RSH_W={width}")
    wshift = len(dut.shamt)

    rng = np.random.default_rng(seed)
    xs = x_values(width, kind, rng)
    shamts = shamt_values(width, wshift)

    start_clock(dut)
    checked = 0
    for x in xs:
        for shamt in shamts:
            drive_unsigned(dut.x, x)
            drive_unsigned(dut.shamt, shamt)
            # Settle the combinational path first (the STAGE_SPLIT=0 result), then advance split clock edges holding
            # the inputs stable across the register barrier. Settling before the first edge avoids a t=0 drive/clock
            # race on the un-reset datapath register.
            await Timer(1, unit="ns")
            for _ in range(split):
                await RisingEdge(dut.clk)
                await Timer(1, unit="ns")
            exp = rshift_sticky_reference(width, x, shamt)
            assert is_resolvable(dut.y), f"{cfg}: y unresolved x={x:#x} shamt={shamt}"
            obs = int(dut.y.value)
            assert obs == exp, (
                f"{cfg}: y mismatch x={x:#x} shamt={shamt} got={obs:#x} exp={exp:#x} (W={width} split={split})"
            )
            checked += 1
    assert checked == len(xs) * len(shamts), f"{cfg}: checked {checked} of {len(xs) * len(shamts)}"

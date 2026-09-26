#!/usr/bin/env python3
"""The runtime-mode CORDIC engine (_zkf_cordic_core MODE 2) against its fixed modes, bit and cycle exact."""

from __future__ import annotations

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, Timer

from zkf_operands import random_bits
from zkf_params import plusarg_int, plusarg_str
from zkf_stream import drive_signed, drive_unsigned, int_value, start_clock

_SIGNALS = ("xn", "yn", "zn")


def _outputs(dut, mode: str) -> dict[str, int]:
    return {name: int_value(getattr(dut, f"{name}_{mode}")) for name in _SIGNALS}


def _below(rng: np.random.Generator, bound: int) -> int:
    """Uniform in [0, bound) at any width (NumPy integers stop at 64 bits)."""
    return random_bits(bound.bit_length() + 8, rng) % bound


def _draw(rng: np.random.Generator, wx: int, wz: int, xf: int, zf: int) -> tuple[int, int, int]:
    """(x0, y0, z0): an in-range vectoring vector and a first-octant rotation angle, occasionally full-width noise."""
    if rng.random() < 0.1:
        return random_bits(wx, rng), random_bits(wx, rng), random_bits(wz, rng)
    x0 = (1 << (xf - 2)) + _below(rng, 1 << (xf - 2))
    y0 = _below(rng, 2 * x0 + 1) - x0
    z0 = _below(rng, (1 << (zf - 3)) + 1)
    return x0, y0, z0


def _scramble(dut, rng: np.random.Generator) -> None:
    for handle in (dut.x0, dut.y0, dut.z0):
        drive_unsigned(handle, random_bits(len(handle), rng))
    dut.vectoring.value = int(rng.integers(0, 2))


@cocotb.test()
async def cordic_runtime_matches_fixed_modes(dut) -> None:
    cfg = plusarg_str("ZKF_CONFIG", "default")
    seed = plusarg_int("ZKF_SEED", 0)
    count = plusarg_int("ZKF_COUNT", 256)
    wx, wz = len(dut.x0), len(dut.z0)
    xf, zf = wx - 2, wz - 3
    rng = np.random.default_rng(seed)

    start_clock(dut)
    dut.rst.value = 1
    dut.start.value = 0
    _scramble(dut, rng)
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0

    driven = {False: 0, True: 0}
    checked = 0
    pending: bool | None = None
    resets = 0
    budget = count * 200
    for cycle in range(budget):
        await Timer(1, "ns")
        busy, done = int_value(dut.busy), int_value(dut.done)
        if busy not in (0, 7) or done not in (0, 7):
            raise AssertionError(f"{cfg}: cycle {cycle}: busy={busy:03b} done={done:03b} diverge across modes")
        z_done = int_value(dut.z_done)
        want_z = 0 if pending is None else (z_done >> (1 if pending else 0)) & 1
        if (z_done >> 2) & 1 != want_z:
            raise AssertionError(f"{cfg}: cycle {cycle}: z_done={z_done:03b} but the runtime reference is {want_z}")
        if want_z and int_value(dut.zn_run) != int_value(dut.zn_vec if pending else dut.zn_rot):
            raise AssertionError(f"{cfg}: cycle {cycle}: zn diverges at z_done")
        if done:
            assert pending is not None, f"{cfg}: cycle {cycle}: done without a transaction"
            want = _outputs(dut, "vec" if pending else "rot")
            got = _outputs(dut, "run")
            if got != want:
                raise AssertionError(f"{cfg}: vectoring={pending}: runtime {got} != fixed-mode {want}")
            checked += 1
            pending = None
            if checked >= count:
                break
        dut.start.value = 0
        dut.rst.value = 0
        if busy and rng.random() < 0.02:
            dut.rst.value = 1
            pending = None
            resets += 1
        elif not busy and pending is None:
            vec = bool(rng.integers(0, 2))
            x0, y0, z0 = _draw(rng, wx, wz, xf, zf)
            drive_signed(dut.x0, x0)
            drive_signed(dut.y0, y0)
            drive_signed(dut.z0, z0)
            dut.vectoring.value = int(vec)
            dut.start.value = 1
            driven[vec] += 1
            pending = vec
        else:
            _scramble(dut, rng)
            if busy and rng.random() < 0.05:
                dut.start.value = 1
        await RisingEdge(dut.clk)
    if checked < count:
        raise AssertionError(f"{cfg}: only {checked}/{count} transactions completed in {budget} cycles")
    if min(driven.values()) == 0:
        raise AssertionError(f"{cfg}: both modes must be driven, got {driven}")
    dut._log.info(f"{cfg}: {checked} transactions (vectoring {driven[True]}), {resets} resets")

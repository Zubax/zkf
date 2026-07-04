#!/usr/bin/env python3

from __future__ import annotations

import cocotb
from cocotb.triggers import RisingEdge, Timer

from zkf_params import check_width, pipe_context
from zkf_stream import RegisterStageScoreboard, drive_unsigned, is_resolvable, start_clock


@cocotb.test()
async def pipe_runtime_cases(dut) -> None:
    context = pipe_context()
    width = context.pipe_w
    stages = context.pipe_n
    sample_count = context.count
    assert width is not None
    assert stages is not None

    in_handle = dut["in"]
    check_width("in", in_handle, width, context)
    check_width("out", dut.out, width, context)

    start_clock(dut)
    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_unsigned(in_handle, 0)

    if stages == 0:
        for _ in range(2):
            await RisingEdge(dut.clk)
        dut.rst.value = 0
        for i in range(sample_count):
            value = ((i * 0xA5) + 1) & ((1 << width) - 1)
            valid = (i % 3) != 0
            dut.in_valid.value = valid
            drive_unsigned(in_handle, value)
            await Timer(1, unit="ns")
            assert is_resolvable(dut.out_valid), context.prefix() + " out_valid unresolved"
            observed_valid = int(dut.out_valid.value)
            assert observed_valid == int(valid), (
                f"{context.prefix()} passthrough out_valid mismatch i={i} expected={int(valid)} "
                f"observed={observed_valid}"
            )
            if valid:
                observed = int(dut.out.value)
                assert observed == value, (
                    f"{context.prefix()} passthrough out mismatch i={i} expected={value:0{(width + 3) // 4}x} "
                    f"observed={observed:0{(width + 3) // 4}x}"
                )
            await RisingEdge(dut.clk)
        return

    scoreboard = RegisterStageScoreboard(dut, stages, context, {"out": (dut.out, width)})

    def drive_idx(i: int) -> int:
        return ((i * 0xCAFE_BABE) ^ ((i + 1) * 0x9E37_79B9)) & ((1 << width) - 1)

    def drive_during_reset() -> None:
        dut.in_valid.value = 1
        drive_unsigned(in_handle, 0xDEAD_BEEF & ((1 << width) - 1))

    await scoreboard.reset(stages + 2, drive_during_reset=drive_during_reset)

    for i in range(sample_count):
        value = drive_idx(i)
        dut.in_valid.value = 1
        drive_unsigned(in_handle, value)
        await scoreboard.tick({"out": value}, f"i={i} value={value:0{(width + 3) // 4}x}")

    dut.in_valid.value = 0
    drive_unsigned(in_handle, 0)
    for flush_index in range(stages + 2):
        await scoreboard.tick(None, f"flush={flush_index}")

    assert scoreboard.checked >= sample_count, (
        f"{context.prefix()} checked {scoreboard.checked} outputs, expected at least {sample_count}"
    )

    for i in range(stages):
        dut.in_valid.value = 1
        drive_unsigned(in_handle, drive_idx(100 + i))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

    dut.rst.value = 1
    dut.in_valid.value = 0
    drive_unsigned(in_handle, 0)
    for cycle in range(stages + 2):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert is_resolvable(dut.out_valid), context.prefix() + f" out_valid unresolved during reset cycle={cycle}"
        assert int(dut.out_valid.value) == 0, f"{context.prefix()} reset failed to clear out_valid cycle={cycle}"
    dut.rst.value = 0

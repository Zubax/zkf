#!/usr/bin/env python3

from __future__ import annotations

import random

import cocotb

from zkf import ZkfFormat
from zkf_params import cast_context, check_width
from zkf_stream import RegisterStageScoreboard, drive_unsigned, run_stream_cases, start_clock


@cocotb.test()
async def ilog2_cases(dut) -> None:
    context = cast_context("ilog2")
    fmt = ZkfFormat(context.wexp, context.wman)
    wint = context.wint
    check_width("a", dut.a, fmt.wfull, context)
    check_width("y", dut.y, wint, context)
    if context.kind == "exhaustive":
        cases = list(range(1 << fmt.wfull))
    else:
        cases = [
            (sign << fmt.sign_shift) | (exp << fmt.wfrac) | frac
            for exp in range(1 << fmt.wexp)
            for frac in (0, 1, fmt.frac_mask)
            for sign in (0, 1)
        ]
        rng = random.Random(context.seed)
        cases.extend(rng.getrandbits(fmt.wfull) for _ in range(context.count))

    start_clock(dut)
    dut.in_valid.value = 0
    dut.a.value = 0
    stages = fmt.model_of("ilog2")(wint=wint, stage_input=context.stage_input).latency
    outputs = {"y": (dut.y, wint), **{name: (getattr(dut, name), 1) for name in ("zero", "infinity", "negative")}}
    scoreboard = RegisterStageScoreboard(dut, stages, context, outputs)

    def drive_case(bits: int) -> dict[str, int]:
        drive_unsigned(dut.a, bits)
        exponent = (bits >> fmt.wfrac) & ((1 << fmt.wexp) - 1)
        return {
            "y": (exponent - ((1 << (fmt.wexp - 1)) - 1)) & ((1 << wint) - 1),
            "zero": int(exponent == 0),
            "infinity": int(exponent == (1 << fmt.wexp) - 1),
            "negative": bits >> fmt.sign_shift,
        }

    def invalid_drive() -> None:
        dut.in_valid.value = 0
        drive_unsigned(dut.a, (1 << fmt.wfull) - 1)

    await scoreboard.reset(stages + 1)
    await run_stream_cases(dut, scoreboard, cases, drive_case, invalid_drive, lambda i, a: f"case={i} a={a:x}")
    assert scoreboard.checked == len(cases)

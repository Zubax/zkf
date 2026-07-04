#!/usr/bin/env python3
"""Shared Cocotb clocking, reset, drive, and stream scoreboard helpers."""

from __future__ import annotations

from collections import deque
from typing import Callable

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from zkf_bits import hex_bits, mask, signed_to_bits
from zkf_params import TestContext


def start_clock(dut, period_ns: int = 10) -> None:
    cocotb.start_soon(Clock(dut.clk, period_ns, unit="ns").start())


def is_resolvable(handle) -> bool:
    return bool(handle.value.is_resolvable)


def int_value(handle) -> int:
    if not is_resolvable(handle):
        raise ValueError(str(handle.value))
    return int(handle.value)


def drive_signed(handle, value: int) -> None:
    handle.value = signed_to_bits(value, len(handle))


def drive_unsigned(handle, value: int) -> None:
    handle.value = value & mask(len(handle))


class RegisterStageScoreboard:
    def __init__(
        self,
        dut,
        register_stages: int,
        context: TestContext,
        outputs: dict[str, tuple[object, int]],
        reset_passthrough: bool = False,
    ) -> None:
        if register_stages < 0:
            raise ValueError(f"register_stages must be non-negative, got {register_stages}")
        if reset_passthrough and register_stages != 0:
            raise ValueError("reset_passthrough is only valid for zero-register combinational paths")
        self._dut = dut
        # A combinational module (register_stages == 0) is observed like a single-stage one: the driver holds the inputs
        # across the sampling edge, so the held combinational output is still valid one edge after the drive.
        self._queue_delay = max(0, register_stages - 1)
        self._context = context
        self._outputs = outputs
        self._reset_passthrough = reset_passthrough
        self._queue: deque[tuple[dict[str, int], str] | None] = deque([None] * self._queue_delay)
        self.checked = 0

    @property
    def queue_delay(self) -> int:
        return self._queue_delay

    def clear(self) -> None:
        self._queue = deque([None] * self._queue_delay)

    def _message(self, detail: str, case_description: str = "") -> str:
        suffix = f" {case_description}" if case_description else ""
        return f"{self._context.prefix()} {detail}{suffix}"

    async def tick(self, expected: dict[str, int] | None, case_description: str = "") -> None:
        await RisingEdge(self._dut.clk)
        await Timer(1, unit="ns")

        current = (expected, case_description) if expected is not None else None
        due = current if self._queue_delay == 0 else self._queue.popleft()
        assert is_resolvable(self._dut.out_valid), self._message("out_valid is unresolved", case_description)
        observed_valid = int(self._dut.out_valid.value)

        if due is None:
            assert observed_valid == 0, self._message(
                f"expected out_valid=0 observed={observed_valid}",
                case_description,
            )
        else:
            expected_outputs, expected_case = due
            assert observed_valid == 1, self._message(
                f"expected out_valid=1 observed={observed_valid}",
                expected_case,
            )
            for name, expected_value in expected_outputs.items():
                handle, width = self._outputs[name]
                assert is_resolvable(handle), self._message(f"{name} is unresolved while out_valid=1", expected_case)
                observed_value = int(handle.value)
                assert observed_value == expected_value, self._message(
                    f"{name} mismatch expected={hex_bits(expected_value, width)} "
                    f"observed={hex_bits(observed_value, width)}",
                    expected_case,
                )
            self.checked += 1

        if self._queue_delay > 0:
            self._queue.append(current)

    async def reset(
        self,
        cycles: int,
        drive_during_reset: Callable[[], dict[str, int] | None] | None = None,
    ) -> None:
        self._dut.rst.value = 1
        self.clear()
        for _ in range(cycles):
            expected = None
            if drive_during_reset is not None:
                expected = drive_during_reset()
            await RisingEdge(self._dut.clk)
            await Timer(1, unit="ns")
            assert is_resolvable(self._dut.out_valid), self._message("out_valid is unresolved during reset")
            observed_valid = int(self._dut.out_valid.value)
            if self._reset_passthrough and expected is not None:
                assert observed_valid == 1, self._message("expected out_valid=1 during reset")
                for name, expected_value in expected.items():
                    handle, width = self._outputs[name]
                    assert is_resolvable(handle), self._message(f"{name} is unresolved during reset")
                    observed_value = int(handle.value)
                    assert observed_value == expected_value, self._message(
                        f"{name} reset mismatch expected={hex_bits(expected_value, width)} "
                        f"observed={hex_bits(observed_value, width)}"
                    )
            else:
                assert observed_valid == 0, self._message("out_valid asserted during reset")
            self.clear()
        self._dut.rst.value = 0
        self.clear()


async def run_stream_cases(
    dut,
    scoreboard: RegisterStageScoreboard,
    cases: list[object],
    drive_case: Callable[[object], dict[str, int]],
    invalid_drive: Callable[[], None],
    describe_case: Callable[[int, object], str],
) -> None:
    if cases:
        stress_count = min(len(cases), scoreboard.queue_delay + 3)
        for index, case in enumerate(cases[:stress_count]):
            dut.in_valid.value = 1
            expected = drive_case(case)
            await scoreboard.tick(expected, describe_case(index, case))

        def drive_reset_sample() -> dict[str, int]:
            dut.in_valid.value = 1
            return drive_case(cases[0])

        await scoreboard.reset(2, drive_during_reset=drive_reset_sample)
        scoreboard.checked = 0

    for index, case in enumerate(cases):
        dut.in_valid.value = 1
        expected = drive_case(case)
        await scoreboard.tick(expected, describe_case(index, case))

        if index % 17 == 3:
            invalid_drive()
            await scoreboard.tick(None, f"gap_after_case={index}")
        if index % 43 == 7:
            invalid_drive()
            await scoreboard.tick(None, f"gap_after_case={index}")
            await scoreboard.tick(None, f"gap_after_case={index}")

    invalid_drive()
    for flush_index in range(scoreboard.queue_delay + 2):
        await scoreboard.tick(None, f"flush={flush_index}")

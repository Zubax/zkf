#!/usr/bin/env python3
"""Cocotb plusarg parsing and configuration checks for ZKF tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import cocotb

VALID_KINDS = {"directed", "exhaustive", "random"}


@dataclass(frozen=True)
class TestContext:
    suite: str
    config: str
    seed: int
    kind: str = "directed"
    count: int = 0
    wexp: int | None = None
    wman: int | None = None
    wexp_unbiased: int | None = None
    pipe_w: int | None = None
    pipe_n: int | None = None
    wint: int | None = None
    wexp_in: int | None = None
    wman_in: int | None = None
    wexp_out: int | None = None
    wman_out: int | None = None
    wk: int | None = None  # zkf_mul_ilog2: width of the signed runtime shift k
    stage_input: int = 0  # input-register knob for sequential float operators
    stage_reduce: int = 0  # zkf_exp2: register the reduced fixed-point before the evaluator ROM
    stage_product: int = 0  # zkf_mul / zkf_fma / zkf_exp2 / zkf_log2 / zkf_sincos / zkf_atan2
    stage_product_final: int = 0  # zkf_log2 final f*C(f) multiply; defaults to stage_product in float_context()
    stage_align: int = 0  # zkf_add / zkf_addsub / zkf_fma
    stage_decode: int = 0  # zkf_mul_ilog2_const / zkf_fma / zkf_log2
    stage_normalize: int = 0  # zkf_add / zkf_addsub / zkf_fma / zkf_log2 / zkf_from_int
    stage_normalize_output: int = 0  # zkf_log2: register _zkf_normshift outputs before GRS/exponent combine
    stage_pack: int = 0  # zkf_fma / zkf_log2 / zkf_exp2 / zkf_from_int (forwarded to _zkf_pack.STAGE_INPUT)
    stage_output: int = 0  # pack-based ops: 0 = combinational (default), 1 = registered (+1 cycle)
    unroll100: int = 100  # zkf_sincos: CORDIC iterations/cycle x100 (mirrors the UNROLL100 vlogparam)
    parallel: int = 0  # zkf_sincos: run the z-path ahead of x/y (mirrors the PARALLEL vlogparam)
    exp_is_biased: int = 0  # _zkf_pack: 1 = exponent input already biased (packer skips its bias add)
    assume_no_overflow: int = 0  # _zkf_pack: 1 = overflow detector pruned (caller guarantees in-range exponent)
    shard_index: int = 0  # exhaustive sweep is split into shard_count strided slices; this run drives slice shard_index
    shard_count: int = 1  # >1 splits a long exhaustive case into parallel cocotb runs (union == the full sweep)

    @property
    def params(self) -> str:
        knob_suffix = ""
        if self.stage_input:
            knob_suffix += f" SI={self.stage_input}"
        if self.stage_reduce:
            knob_suffix += f" SR={self.stage_reduce}"
        if self.stage_product:
            knob_suffix += f" SP={self.stage_product}"
        if self.stage_product_final != self.stage_product:
            knob_suffix += f" SPF={self.stage_product_final}"
        if self.stage_align:
            knob_suffix += f" SA={self.stage_align}"
        if self.stage_decode:
            knob_suffix += f" SD={self.stage_decode}"
        if self.stage_normalize:
            knob_suffix += f" SN={self.stage_normalize}"
        if self.stage_normalize_output:
            knob_suffix += f" SNO={self.stage_normalize_output}"
        if self.stage_pack:
            knob_suffix += f" PA={self.stage_pack}"
        if self.stage_output == 0:
            knob_suffix += " SO=0"
        if self.parallel:
            knob_suffix += " PAR"
        if self.exp_is_biased:
            knob_suffix += f" EB={self.exp_is_biased}"
        if self.assume_no_overflow:
            knob_suffix += f" NOV={self.assume_no_overflow}"
        if self.wk is not None:
            knob_suffix += f" WK={self.wk}"
        if self.wexp_in is not None and self.wman_in is not None:
            return f"{self.config} {self.wexp_in}/{self.wman_in}->" f"{self.wexp_out}/{self.wman_out}{knob_suffix}"
        if self.wint is not None:
            return f"{self.config} WEXP={self.wexp} WMAN={self.wman} WINT={self.wint}{knob_suffix}"
        if self.wexp is not None and self.wman is not None:
            return f"{self.config} WEXP={self.wexp} WMAN={self.wman}{knob_suffix}"
        if self.pipe_w is not None and self.pipe_n is not None:
            return f"{self.config} W={self.pipe_w} N={self.pipe_n}"
        return self.config

    def prefix(self) -> str:
        return (
            f"suite={self.suite} params={self.params} kind={self.kind} " f"count={self.count} seed=0x{self.seed:016x}"
        )


def _runtime_plusargs() -> dict[str, object]:
    return getattr(cocotb, "plusargs", {})


def _get_text(name: str, default: str | None = None, aliases: Iterable[str] = ()) -> str | None:
    for candidate in (name, *aliases):
        value = _runtime_plusargs().get(candidate)
        if value is not None and value is not True:
            return str(value)
        if candidate in os.environ:
            return os.environ[candidate]
    return default


def plusarg_str(name: str, default: str | None = None, aliases: Iterable[str] = ()) -> str:
    value = _get_text(name, default, aliases)
    if value is None:
        raise ValueError(f"required plusarg +{name}=... is missing")
    return value


def plusarg_int(name: str, default: int | None = None, aliases: Iterable[str] = ()) -> int:
    text = _get_text(name, None, aliases)
    if text is None:
        if default is None:
            raise ValueError(f"required plusarg +{name}=... is missing")
        return default
    return int(text, 0)


def _seed() -> int:
    seed = plusarg_int("ZKF_SEED", 0)
    if seed < 0:
        raise ValueError(f"ZKF_SEED must be non-negative, got {seed}")
    return seed


def _kind() -> str:
    kind = plusarg_str("ZKF_KIND", "directed")
    if kind not in VALID_KINDS:
        raise ValueError(f"ZKF_KIND must be one of {sorted(VALID_KINDS)}, got {kind!r}")
    return kind


def _stage_input() -> int:
    value = plusarg_int("ZKF_STAGE_INPUT", 0)
    if value < 0:
        raise ValueError(f"ZKF_STAGE_INPUT must be non-negative, got {value}")
    return value


def _stage_reduce() -> int:
    value = plusarg_int("ZKF_STAGE_REDUCE", 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_STAGE_REDUCE must be 0 or 1, got {value}")
    return value


def _stage_product() -> int:
    value = plusarg_int("ZKF_STAGE_PRODUCT", 0)
    if value < 0:
        raise ValueError(f"ZKF_STAGE_PRODUCT must be non-negative, got {value}")
    return value


def _stage_product_final(stage_product: int) -> int:
    # Defaults to STAGE_PRODUCT when the plusarg is absent (the matrix always passes it; this fallback covers direct
    # cocotb runs outside the matrix driver).
    return plusarg_int("ZKF_STAGE_PRODUCT_FINAL", stage_product)


def _stage_align() -> int:
    value = plusarg_int("ZKF_STAGE_ALIGN", 0)
    if value < 0:
        raise ValueError(f"ZKF_STAGE_ALIGN must be non-negative, got {value}")
    return value


def _stage_decode() -> int:
    value = plusarg_int("ZKF_STAGE_DECODE", 0)
    if value < 0:
        raise ValueError(f"ZKF_STAGE_DECODE must be non-negative, got {value}")
    return value


def _stage_normalize() -> int:
    value = plusarg_int("ZKF_STAGE_NORMALIZE", 0)
    if value < 0:
        raise ValueError(f"ZKF_STAGE_NORMALIZE must be non-negative, got {value}")
    return value


def _stage_normalize_output() -> int:
    value = plusarg_int("ZKF_STAGE_NORMALIZE_OUTPUT", 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_STAGE_NORMALIZE_OUTPUT must be 0 or 1, got {value}")
    return value


def _stage_pack() -> int:
    value = plusarg_int("ZKF_STAGE_PACK", 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_STAGE_PACK must be 0 or 1, got {value}")
    return value


def _stage_output() -> int:
    value = plusarg_int("ZKF_STAGE_OUTPUT", 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_STAGE_OUTPUT must be 0 or 1, got {value}")
    return value


def _unroll100() -> int:
    # Mirror the RTL UNROLL100 vlogparam (iterations/cycle x100) so the latency model matches the engine.
    value = plusarg_int("ZKF_UNROLL100", 100)
    if value != 50 and (value < 100 or value % 100 != 0):
        raise ValueError(f"ZKF_UNROLL100 must be 50 or a positive multiple of 100, got {value}")
    return value


def _parallel(unroll100: int) -> int:
    # Mirror the RTL PARALLEL vlogparam; its default is (UNROLL100 < 100), so derive the same default here.
    value = plusarg_int("ZKF_PARALLEL", 1 if unroll100 < 100 else 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_PARALLEL must be 0 or 1, got {value}")
    if value and unroll100 != 50:
        raise ValueError(f"ZKF_PARALLEL requires ZKF_UNROLL100=50, got {unroll100}")
    return value


def _exp_is_biased() -> int:
    value = plusarg_int("ZKF_EXP_IS_BIASED", 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_EXP_IS_BIASED must be 0 or 1, got {value}")
    return value


def _assume_no_overflow() -> int:
    value = plusarg_int("ZKF_ASSUME_NO_OVERFLOW", 0)
    if value not in (0, 1):
        raise ValueError(f"ZKF_ASSUME_NO_OVERFLOW must be 0 or 1, got {value}")
    return value


def _shard() -> tuple[int, int]:
    count = plusarg_int("ZKF_SHARD_COUNT", 1)
    index = plusarg_int("ZKF_SHARD_INDEX", 0)
    if count < 1 or not (0 <= index < count):
        raise ValueError(f"ZKF_SHARD_INDEX/COUNT must satisfy 0 <= index < count, got {index}/{count}")
    return index, count


def float_context(suite: str, require_wexp_unbiased: bool = False) -> TestContext:
    wexp = plusarg_int("ZKF_WEXP")
    wman = plusarg_int("ZKF_WMAN")
    wexp_unbiased = plusarg_int("ZKF_WEXP_UNBIASED", wexp + 2) if require_wexp_unbiased else None
    if wexp < 2:
        raise ValueError(f"ZKF_WEXP must be at least 2, got {wexp}")
    if wman < 4:
        raise ValueError(f"ZKF_WMAN must be at least 4, got {wman}")
    if wexp_unbiased is not None and wexp_unbiased < wexp + 1:
        raise ValueError(f"ZKF_WEXP_UNBIASED={wexp_unbiased} is too narrow for ZKF_WEXP={wexp}")
    stage_product = _stage_product()
    unroll100 = _unroll100()
    shard_index, shard_count = _shard()
    return TestContext(
        suite=suite,
        config=plusarg_str("ZKF_CONFIG", "default"),
        seed=_seed(),
        kind=_kind(),
        count=plusarg_int("ZKF_COUNT", 0, aliases=("ZKF_RANDOM_COUNT",)),
        wexp=wexp,
        wman=wman,
        wexp_unbiased=wexp_unbiased,
        shard_index=shard_index,
        shard_count=shard_count,
        wk=(plusarg_int("ZKF_WK", 0) or None),  # only zkf_mul_ilog2 sets it; 0/absent -> None (RTL default WEXP+1)
        stage_input=_stage_input(),
        stage_reduce=_stage_reduce(),
        stage_product=stage_product,
        stage_product_final=_stage_product_final(stage_product),
        stage_align=_stage_align(),
        stage_decode=_stage_decode(),
        stage_normalize=_stage_normalize(),
        stage_normalize_output=_stage_normalize_output(),
        stage_pack=_stage_pack(),
        stage_output=_stage_output(),
        unroll100=unroll100,
        parallel=_parallel(unroll100) if suite == "sincos" else 0,
        exp_is_biased=_exp_is_biased(),
        assume_no_overflow=_assume_no_overflow(),
    )


def cast_context(suite: str) -> TestContext:
    wexp = plusarg_int("ZKF_WEXP")
    wman = plusarg_int("ZKF_WMAN")
    wint = plusarg_int("ZKF_WINT")
    if wexp < 2:
        raise ValueError(f"ZKF_WEXP must be at least 2, got {wexp}")
    if wman < 4:
        raise ValueError(f"ZKF_WMAN must be at least 4, got {wman}")
    if wint < 2:
        raise ValueError(f"ZKF_WINT must be at least 2, got {wint}")
    return TestContext(
        suite=suite,
        config=plusarg_str("ZKF_CONFIG", "default"),
        seed=_seed(),
        kind=_kind(),
        count=plusarg_int("ZKF_COUNT", 0, aliases=("ZKF_RANDOM_COUNT",)),
        wexp=wexp,
        wman=wman,
        wint=wint,
        stage_input=_stage_input(),
        stage_product=_stage_product(),
        stage_align=_stage_align(),
        stage_decode=_stage_decode(),
        stage_normalize=_stage_normalize(),
        stage_pack=_stage_pack(),
        stage_output=_stage_output(),
    )


def resize_context(suite: str) -> TestContext:
    wexp_in = plusarg_int("ZKF_WEXP_IN")
    wman_in = plusarg_int("ZKF_WMAN_IN")
    wexp_out = plusarg_int("ZKF_WEXP_OUT")
    wman_out = plusarg_int("ZKF_WMAN_OUT")
    if wexp_in < 2 or wexp_out < 2:
        raise ValueError(f"ZKF_WEXP_IN/OUT must each be at least 2, got in={wexp_in} out={wexp_out}")
    if wman_in < 4 or wman_out < 4:
        raise ValueError(f"ZKF_WMAN_IN/OUT must each be at least 4, got in={wman_in} out={wman_out}")
    return TestContext(
        suite=suite,
        config=plusarg_str("ZKF_CONFIG", "default"),
        seed=_seed(),
        kind=_kind(),
        count=plusarg_int("ZKF_COUNT", 0, aliases=("ZKF_RANDOM_COUNT",)),
        wexp_in=wexp_in,
        wman_in=wman_in,
        wexp_out=wexp_out,
        wman_out=wman_out,
        stage_input=_stage_input(),
        stage_product=_stage_product(),
        stage_align=_stage_align(),
        stage_decode=_stage_decode(),
        stage_output=_stage_output(),
    )


def pipe_context() -> TestContext:
    width = plusarg_int("ZKF_PIPE_W")
    stages = plusarg_int("ZKF_PIPE_N")
    if width < 1:
        raise ValueError(f"ZKF_PIPE_W must be positive, got {width}")
    if stages < 0:
        raise ValueError(f"ZKF_PIPE_N must be non-negative, got {stages}")
    return TestContext(
        suite="pipe",
        config=plusarg_str("ZKF_CONFIG", "default"),
        seed=_seed(),
        kind="random",
        count=plusarg_int("ZKF_COUNT", 64, aliases=("ZKF_PIPE_COUNT",)),
        pipe_w=width,
        pipe_n=stages,
    )


def signal_width(handle) -> int:
    try:
        return len(handle)
    except TypeError:
        return 1


def check_width(label: str, handle, expected: int, context: TestContext) -> None:
    observed = signal_width(handle)
    if observed != expected:
        raise AssertionError(f"{context.prefix()} {label} width mismatch expected={expected} observed={observed}")

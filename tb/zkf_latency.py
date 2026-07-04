#!/usr/bin/env python3
"""
Latency formulas shared by the cocotb scoreboard and the synthesis reports.

The tests use these values as the scoreboard delay (a wrong value fails simulation), and the reports import the same
helpers, so the published latency is exactly the one the suite verifies.
"""

from __future__ import annotations

from zkf import ZkfFormat


def _enabled(value: int) -> int:
    if value < 0:
        raise ValueError(f"stage value must be non-negative, got {value}")
    return 1 if value else 0


def _count(value: int) -> int:
    if value < 0:
        raise ValueError(f"stage count must be non-negative, got {value}")
    return value


def div_qfrac(wman: int) -> int:
    qfrac_base = wman + 2
    return qfrac_base + (qfrac_base % 2)


def pack_latency(*, stage_output: int = 0) -> int:
    return _enabled(stage_output)


def mul_latency(
    *,
    stage_input: int = 0,
    stage_product: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    # stage_product forwards to _zkf_pmul (latency 1 + stage_product), so it contributes its raw count.
    return 1 + _count(stage_input) + _count(stage_product) + _enabled(stage_pack) + _enabled(stage_output)


def add_latency(
    *,
    stage_input: int = 0,
    stage_decode: int = 0,
    stage_align: int = 0,
    stage_normalize: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    return (
        4
        + _count(stage_input)  # unbounded count of input stages (matches RTL LATENCY_REF)
        + _enabled(stage_decode)
        + _enabled(stage_align)
        + _count(stage_normalize)
        + _enabled(stage_pack)
        + _enabled(stage_output)
    )


def fma_latency(
    *,
    stage_input: int = 0,
    stage_product: int = 0,
    stage_decode: int = 0,
    stage_align: int = 0,
    stage_normalize: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    return (
        5
        + _count(stage_input)
        + _count(stage_product)  # forwards to _zkf_pmul (latency 1 + stage_product); contributes its raw count
        + _enabled(stage_decode)
        + _enabled(stage_align)
        + _count(stage_normalize)
        + _enabled(stage_pack)
        + _enabled(stage_output)
    )


def div_core_latency(wman: int) -> int:
    return 2 + (div_qfrac(wman) // 2)


def div_latency(
    wman: int,
    *,
    stage_input: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    return div_core_latency(wman) + _count(stage_input) + _enabled(stage_pack) + _enabled(stage_output)


def cmp_latency(*, stage_input: int = 0) -> int:
    return 1 + _count(stage_input)


def mul_ilog2_const_latency(*, stage_input: int = 0, stage_decode: int = 0) -> int:
    return 1 + _count(stage_input) + _enabled(stage_decode)


def mul_ilog2_latency(*, stage_input: int = 0, stage_decode: int = 0) -> int:
    return 1 + _count(stage_input) + _enabled(stage_decode)


def from_int_latency(
    *,
    stage_input: int = 0,
    stage_normalize: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    return 1 + _count(stage_input) + _count(stage_normalize) + _enabled(stage_pack) + _enabled(stage_output)


def to_int_latency(*, stage_input: int = 0) -> int:
    return 4 + _count(stage_input)


def resize_latency(*, stage_input: int = 0, stage_output: int = 0) -> int:
    return _count(stage_input) + _enabled(stage_output)


def round_latency(*, stage_input: int = 0, stage_decode: int = 0, stage_pack: int = 0, stage_output: int = 0) -> int:
    return _count(stage_input) + _enabled(stage_decode) + _enabled(stage_pack) + _enabled(stage_output)


def exp2_latency(
    fmt: ZkfFormat,
    *,
    stage_input: int = 0,
    stage_reduce: int = 0,
    stage_product: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    degree = fmt.exp2_poly_degree
    product_stages = _count(stage_product)
    return (
        _count(stage_input)
        + _enabled(stage_reduce)
        + 4
        + degree * (2 + product_stages)
        + _enabled(stage_pack)
        + _enabled(stage_output)
    )


def log2_latency(
    fmt: ZkfFormat,
    *,
    stage_input: int = 0,
    stage_decode: int = 0,
    stage_product: int = 0,
    stage_product_final: int | None = None,
    stage_normalize: int = 0,
    stage_normalize_output: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    degree = fmt.log2_poly_degree
    product_stages = _count(stage_product)
    final_product_raw = stage_product if stage_product_final in (None, -1) else stage_product_final
    final_product_stages = _count(final_product_raw)
    return (
        _count(stage_input)
        + _enabled(stage_decode)
        + 5
        + final_product_stages
        + _count(stage_normalize)
        + _enabled(stage_normalize_output)
        + _enabled(stage_pack)
        + degree * (2 + product_stages)
        + _enabled(stage_output)
    )


def sincos_latency(
    fmt: ZkfFormat,
    *,
    unroll100: int = 100,
    parallel: int = 0,
    stage_input: int = 0,
    stage_output: int = 0,
    stage_product: int = 0,
    stage_normalize: int = 0,
    stage_pack: int = 0,
    **_ignored: int,
) -> int:
    # Folded-CORDIC initiation interval = latency (accept -> out_valid). The cocotb bench asserts the RTL matches this,
    # and zkf_sincos.v's LATENCY parameter is the same closed form.
    # 11 = 1 (R1 decode) + 1 (R2 barrel shift) + 1 (engine start) + 1 (engine done) + 4 (shared-multiply micro-sequence)
    #      + 2 (octant-fold + merge-B3) + 1 (fixed-to-float back-end: cos one cycle after sin).
    # Each STAGE_PRODUCT adds one cycle to PHI and one across the pipelined S/C pair => 2*STAGE_PRODUCT.
    # iter_cycles = ceil(K*100/UNROLL100); UNROLL100 = iterations/cycle x100 (50 = half-rate, 100 = 1/cycle,
    # 200/300/400 = 2/3/4 per cycle). STAGE_INPUT/STAGE_OUTPUT add +1 each; STAGE_NORMALIZE + STAGE_PACK add counts.
    # Decoupled z-path (parallel, half-rate only): the z-recurrence finishes ZGAP = iter_cycles - k early, so the PHI
    # PMUL_L = 1+STAGE_PRODUCT pipeline overlaps the CORDIC and the back-end saves SAVED = min(PMUL_L, ZGAP).
    # Mirrors zkf_sincos LATENCY_REF.
    if stage_product not in (0, 1, 2, 3, 4):
        raise ValueError(f"stage_product must be 0..4, got {stage_product}")
    if unroll100 != 50 and (unroll100 < 100 or unroll100 % 100 != 0):
        raise ValueError(f"unroll100 must be 50 or a positive multiple of 100, got {unroll100}")
    k = fmt.sincos_iterations
    iter_cycles = (k * 100 + unroll100 - 1) // unroll100
    saved = 0
    if parallel:
        zgap = iter_cycles - k
        pmul_l = 1 + _count(stage_product)
        saved = min(pmul_l, zgap)
    return (
        11 + 2 * _count(stage_product) + iter_cycles - saved
        + _count(stage_input) + _count(stage_output)
        + _count(stage_normalize) + _count(stage_pack)
    )


def atan2_latency(
    fmt: ZkfFormat,
    *,
    unroll100: int = 100,
    stage_input: int = 0,
    stage_product: int = 0,
    stage_normalize: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
    **_ignored: int,
) -> int:
    # Vectoring-CORDIC initiation interval = latency (accept -> out_valid). The cocotb bench asserts the RTL matches
    # this, and zkf_atan2.v's LATENCY parameter is the same closed form.
    # 8 = front-end pipeline (3) + divide-setup (1) + QT-product base (1, the shared _zkf_pmul's first stage)
    #     + two post-divide (2) + output (1). The magnitude product shares _zkf_pmul but issues during the divide,
    #     so it never adds latency.
    # iter_cycles = ceil(N*100/UNROLL100) (UNROLL100 as in sincos). STEPS = ceil(XF/2) folded radix-4 divider cycles
    # (data-independent). STAGE_PRODUCT adds cycles in the shared _zkf_pmul (post-divide QT product, the only one on the
    # critical path); STAGE_INPUT +1; STAGE_NORMALIZE + STAGE_PACK in the shared _zkf_fixed_to_float back-end;
    # STAGE_OUTPUT on the public theta/mag outputs. Mirrors zkf_atan2 LATENCY_REF.
    if unroll100 != 50 and (unroll100 < 100 or unroll100 % 100 != 0):
        raise ValueError(f"unroll100 must be 50 or a positive multiple of 100, got {unroll100}")
    n, xf = fmt.atan2_iterations, fmt.atan2_divider_width
    iter_cycles = (n * 100 + unroll100 - 1) // unroll100
    steps = (xf + 1) // 2                                  # folded radix-4 divider: 2 quotient bits per cycle
    # STEPS divider digit-cycles + a one-cycle setup that forms 3*den. Mirrors ZKF_ATAN2_DIVCYC = STEPS + 1.
    div_cycles = steps + 1
    return (
        8 + iter_cycles + div_cycles + _count(stage_product)
        + _count(stage_input) + _count(stage_normalize)
        + _count(stage_pack) + _count(stage_output)
    )


def module_latency(
    kind: str,
    *,
    wexp: int = 2,   # valid minimum; transcendental latencies depend only on WMAN
    wman: int = 0,
    unroll100: int = 100,
    parallel: int = 0,
    stage_input: int = 0,
    stage_reduce: int = 0,
    stage_product: int = 0,
    stage_product_final: int | None = None,
    stage_align: int = 0,
    stage_decode: int = 0,
    stage_normalize: int = 0,
    stage_normalize_output: int = 0,
    stage_pack: int = 0,
    stage_output: int = 0,
) -> int:
    if kind == "pack":
        return pack_latency(stage_output=stage_output)
    if kind == "mul":
        return mul_latency(
            stage_input=stage_input,
            stage_product=stage_product,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    if kind in {"add", "addsub"}:
        return add_latency(
            stage_input=stage_input,
            stage_decode=stage_decode,
            stage_align=stage_align,
            stage_normalize=stage_normalize,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    if kind == "fma":
        return fma_latency(
            stage_input=stage_input,
            stage_product=stage_product,
            stage_decode=stage_decode,
            stage_align=stage_align,
            stage_normalize=stage_normalize,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    if kind == "div_core":
        return div_core_latency(wman)
    if kind == "div":
        return div_latency(wman, stage_input=stage_input, stage_pack=stage_pack, stage_output=stage_output)
    if kind in {"cmp", "sort"}:
        return cmp_latency(stage_input=stage_input)
    if kind == "mul_ilog2_const":
        return mul_ilog2_const_latency(stage_input=stage_input, stage_decode=stage_decode)
    if kind == "mul_ilog2":
        return mul_ilog2_latency(stage_input=stage_input, stage_decode=stage_decode)
    if kind == "from_int":
        return from_int_latency(
            stage_input=stage_input,
            stage_normalize=stage_normalize,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    if kind == "to_int":
        return to_int_latency(stage_input=stage_input)
    if kind == "resize":
        return resize_latency(stage_input=stage_input, stage_output=stage_output)
    if kind == "round":
        return round_latency(stage_input=stage_input, stage_decode=stage_decode,
                             stage_pack=stage_pack, stage_output=stage_output)
    if kind == "exp2":
        return exp2_latency(
            ZkfFormat(wexp, wman),
            stage_input=stage_input,
            stage_reduce=stage_reduce,
            stage_product=stage_product,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    if kind == "log2":
        return log2_latency(
            ZkfFormat(wexp, wman),
            stage_input=stage_input,
            stage_decode=stage_decode,
            stage_product=stage_product,
            stage_product_final=stage_product_final,
            stage_normalize=stage_normalize,
            stage_normalize_output=stage_normalize_output,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    if kind == "sincos":
        return sincos_latency(
            ZkfFormat(wexp, wman),
            unroll100=unroll100,
            parallel=parallel,
            stage_input=stage_input,
            stage_output=stage_output,
            stage_product=stage_product,
            stage_normalize=stage_normalize,
            stage_pack=stage_pack,
        )
    if kind == "atan2":
        return atan2_latency(
            ZkfFormat(wexp, wman),
            unroll100=unroll100,
            stage_input=stage_input,
            stage_product=stage_product,
            stage_normalize=stage_normalize,
            stage_pack=stage_pack,
            stage_output=stage_output,
        )
    raise ValueError(f"unsupported module kind: {kind}")

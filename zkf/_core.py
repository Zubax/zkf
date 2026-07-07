"""
Core of the ZKF reference model: the Zkf value class and its bit-exact operator engine.
"""

from __future__ import annotations

import enum
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from typing import ClassVar, NamedTuple


def _check_int_range(value: int, min: int | None, max: int | None, /, extra: set[int] | None = None) -> None:
    extra = extra or set()
    if not isinstance(value, int):
        raise ValueError(f"Expected an integer, found {value}")
    if (min is not None and value < min) or (max is not None and value > max):
        if value not in extra:
            raise ValueError(f"Value {value} is outside {min}..{max}")


@dataclass(frozen=True, slots=True)
class OperatorModel:
    fmt: "ZkfFormat"
    module: ClassVar[str]

    @property
    def config(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init and f.name != "fmt"}

    @property
    def params(self) -> dict[str, int]:
        raise NotImplementedError

    @property
    def verilog_params(self) -> str:
        return ", ".join(f".{name}({value})" for name, value in self.params.items())

    @property
    def latency(self) -> int:
        raise NotImplementedError

    def _params_with_latency(self, params: dict[str, int]) -> dict[str, int]:
        return {**params, "LATENCY": self.latency}


@dataclass(frozen=True, slots=True)
class AbsModel(OperatorModel):
    module: ClassVar[str] = "zkf_abs"

    @property
    def params(self) -> dict[str, int]:
        return {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman}

    @property
    def latency(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class NegModel(AbsModel):
    module: ClassVar[str] = "zkf_neg"


@dataclass(frozen=True, slots=True)
class IsFiniteModel(AbsModel):
    module: ClassVar[str] = "zkf_is_finite"


@dataclass(frozen=True, slots=True)
class SaturateModel(AbsModel):
    module: ClassVar[str] = "zkf_saturate"


@dataclass(frozen=True, slots=True)
class PipeModel(OperatorModel):
    module: ClassVar[str] = "zkf_pipe"
    w: int | None = None
    n: int = 0

    def __post_init__(self) -> None:
        if self.w is not None:
            _check_int_range(self.w, 1, None)
        _check_int_range(self.n, 0, None)

    @property
    def _w(self) -> int:
        return self.fmt.wfull if self.w is None else self.w

    @property
    def config(self) -> dict[str, int]:
        return {"w": self._w, "n": self.n}

    @property
    def params(self) -> dict[str, int]:
        return {"W": self._w, "N": self.n}

    @property
    def latency(self) -> int:
        return self.n


@dataclass(frozen=True, slots=True)
class PackModel(OperatorModel):
    module: ClassVar[str] = "_zkf_pack"
    wexp_unbiased: int | None = None
    exp_is_biased: int = 0
    assume_no_overflow: int = 0
    stage_input: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        if self.wexp_unbiased is not None:
            _check_int_range(self.wexp_unbiased, 1, None)
        _check_int_range(self._wexp_unbiased, self.fmt.wexp + 1, None)
        for value in (self.exp_is_biased, self.assume_no_overflow, self.stage_input, self.stage_output):
            _check_int_range(value, 0, 1)

    @property
    def _wexp_unbiased(self) -> int:
        return self.fmt.wexp + 2 if self.wexp_unbiased is None else self.wexp_unbiased

    @property
    def config(self) -> dict[str, int]:
        return {
            "wexp_unbiased": self._wexp_unbiased,
            "exp_is_biased": self.exp_is_biased,
            "assume_no_overflow": self.assume_no_overflow,
            "stage_input": self.stage_input,
            "stage_output": self.stage_output,
        }

    @property
    def params(self) -> dict[str, int]:
        return {
            "WEXP": self.fmt.wexp,
            "WMAN": self.fmt.wman,
            "WEXP_UNBIASED": self._wexp_unbiased,
            "EXP_IS_BIASED": self.exp_is_biased,
            "ASSUME_NO_OVERFLOW": self.assume_no_overflow,
            "STAGE_INPUT": self.stage_input,
            "STAGE_OUTPUT": self.stage_output,
        }

    @property
    def latency(self) -> int:
        return self.stage_input + self.stage_output


@dataclass(frozen=True, slots=True)
class CmpModel(OperatorModel):
    module: ClassVar[str] = "zkf_cmp"
    stage_input: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman, "STAGE_INPUT": self.stage_input}
        )

    @property
    def latency(self) -> int:
        return 1 + self.stage_input


@dataclass(frozen=True, slots=True)
class SortModel(CmpModel):
    module: ClassVar[str] = "zkf_sort"


@dataclass(frozen=True, slots=True)
class AddModel(OperatorModel):
    module: ClassVar[str] = "zkf_add"
    stage_input: int = 0
    stage_decode: int = 0
    stage_align: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)
        for value in (self.stage_decode, self.stage_align, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.stage_normalize, 0, 2)
        norm_width = self.fmt.wman + 3
        if self.stage_normalize == 2 and norm_width < 17:
            raise ValueError(f"STAGE_NORMALIZE=2 needs _zkf_normshift W>=17 (got {norm_width}); split=2 needs NL4>=3")

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "STAGE_INPUT": self.stage_input,
                "STAGE_DECODE": self.stage_decode,
                "STAGE_ALIGN": self.stage_align,
                "STAGE_NORMALIZE": self.stage_normalize,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return (
            4
            + self.stage_input
            + self.stage_decode
            + self.stage_align
            + self.stage_normalize
            + self.stage_pack
            + self.stage_output
        )


@dataclass(frozen=True, slots=True)
class AddSubModel(AddModel):
    module: ClassVar[str] = "zkf_addsub"


@dataclass(frozen=True, slots=True)
class MulModel(OperatorModel):
    module: ClassVar[str] = "zkf_mul"
    stage_input: int = 0
    stage_product: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_product, 0, 4)
        for value in (self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.wmultiplier, 0, None)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WMULTIPLIER": self.wmultiplier,
                "STAGE_INPUT": self.stage_input,
                "STAGE_PRODUCT": self.stage_product,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_product + self.stage_pack + self.stage_output


@dataclass(frozen=True, slots=True)
class FmaModel(OperatorModel):
    module: ClassVar[str] = "zkf_fma"
    stage_input: int = 0
    stage_product: int = 0
    stage_decode: int = 0
    stage_align: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_product, 0, 4)
        for value in (self.stage_decode, self.stage_align, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.stage_normalize, 0, 2)
        norm_width = 2 * self.fmt.wman + 3
        if self.stage_normalize == 2 and norm_width < 17:
            raise ValueError(f"STAGE_NORMALIZE=2 needs _zkf_normshift W>=17 (got {norm_width}); split=2 needs NL4>=3")
        _check_int_range(self.wmultiplier, 0, None)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WMULTIPLIER": self.wmultiplier,
                "STAGE_INPUT": self.stage_input,
                "STAGE_PRODUCT": self.stage_product,
                "STAGE_DECODE": self.stage_decode,
                "STAGE_ALIGN": self.stage_align,
                "STAGE_NORMALIZE": self.stage_normalize,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return (
            5
            + self.stage_input
            + self.stage_product
            + self.stage_decode
            + self.stage_align
            + self.stage_normalize
            + self.stage_pack
            + self.stage_output
        )


@dataclass(frozen=True, slots=True)
class MulIlog2Model(OperatorModel):
    module: ClassVar[str] = "zkf_mul_ilog2"
    wk: int | None = None
    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        if self.wk is not None:
            _check_int_range(self.wk, 1, None)
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_decode, 0, 1)

    @property
    def _wk(self) -> int:
        return self.fmt.wexp + 1 if self.wk is None else self.wk

    @property
    def config(self) -> dict[str, int]:
        return {
            "wk": self._wk,
            "stage_input": self.stage_input,
            "stage_decode": self.stage_decode,
        }

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WK": self._wk,
                "STAGE_INPUT": self.stage_input,
                "STAGE_DECODE": self.stage_decode,
            }
        )

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_decode


@dataclass(frozen=True, slots=True)
class MulIlog2ConstModel(OperatorModel):
    module: ClassVar[str] = "zkf_mul_ilog2_const"
    k: int = 0
    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        limit = (1 << self.fmt.wexp) - 2
        _check_int_range(self.k, -limit, limit - 1)
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_decode, 0, 1)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "K": self.k,
                "STAGE_INPUT": self.stage_input,
                "STAGE_DECODE": self.stage_decode,
            }
        )

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_decode


@dataclass(frozen=True, slots=True)
class DivCoreModel(OperatorModel):
    module: ClassVar[str] = "_zkf_div_core"

    @property
    def qfrac_base(self) -> int:
        return self.fmt.wman + 2

    @property
    def qfrac(self) -> int:
        return self.qfrac_base + (self.qfrac_base % 2)

    @property
    def params(self) -> dict[str, int]:
        return {
            "WEXP": self.fmt.wexp,
            "WMAN": self.fmt.wman,
            "QFRAC_BASE": self.qfrac_base,
            "QFRAC": self.qfrac,
            "WEXP_UNBIASED": self.fmt.wexp + 2,
        }

    @property
    def latency(self) -> int:
        return 2 + self.qfrac // 2


@dataclass(frozen=True, slots=True)
class DivModel(OperatorModel):
    module: ClassVar[str] = "zkf_div"
    stage_input: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)
        for value in (self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "STAGE_INPUT": self.stage_input,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return DivCoreModel(self.fmt).latency + self.stage_input + self.stage_pack + self.stage_output


@dataclass(frozen=True, slots=True)
class FromIntModel(OperatorModel):
    module: ClassVar[str] = "zkf_from_int"
    wint: int = 32
    stage_input: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.wint, 2, None)
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_normalize, 0, 2)
        norm_width = max(self.wint, self.fmt.wman + 3)
        if self.stage_normalize == 2 and norm_width < 17:
            raise ValueError(f"STAGE_NORMALIZE=2 needs _zkf_normshift W>=17 (got {norm_width}); split=2 needs NL4>=3")
        for value in (self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WINT": self.wint,
                "STAGE_INPUT": self.stage_input,
                "STAGE_NORMALIZE": self.stage_normalize,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_normalize + self.stage_pack + self.stage_output


@dataclass(frozen=True, slots=True)
class ToIntModel(OperatorModel):
    module: ClassVar[str] = "zkf_to_int"
    wint: int = 32
    stage_input: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.wint, 2, None)
        _check_int_range(self.stage_input, 0, None)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman, "WINT": self.wint, "STAGE_INPUT": self.stage_input}
        )

    @property
    def latency(self) -> int:
        return 4 + self.stage_input


@dataclass(frozen=True, slots=True)
class ResizeModel(OperatorModel):
    module: ClassVar[str] = "zkf_resize"
    wexp_in: int | None = None
    wman_in: int | None = None
    stage_input: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        ZkfFormat(self._wexp_in, self._wman_in)
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_output, 0, 1)

    @property
    def _wexp_in(self) -> int:
        return self.fmt.wexp if self.wexp_in is None else self.wexp_in

    @property
    def _wman_in(self) -> int:
        return self.fmt.wman if self.wman_in is None else self.wman_in

    @property
    def config(self) -> dict[str, int]:
        return {
            "wexp_in": self._wexp_in,
            "wman_in": self._wman_in,
            "stage_input": self.stage_input,
            "stage_output": self.stage_output,
        }

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP_IN": self._wexp_in,
                "WMAN_IN": self._wman_in,
                "WEXP_OUT": self.fmt.wexp,
                "WMAN_OUT": self.fmt.wman,
                "STAGE_INPUT": self.stage_input,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return self.stage_input + self.stage_output


@dataclass(frozen=True, slots=True)
class RoundModel(OperatorModel):
    module: ClassVar[str] = "zkf_round"
    stage_input: int = 0
    stage_decode: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)
        for value in (self.stage_decode, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "STAGE_INPUT": self.stage_input,
                "STAGE_DECODE": self.stage_decode,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        return self.stage_input + self.stage_decode + self.stage_pack + self.stage_output


@dataclass(frozen=True, slots=True)
class Exp2Model(OperatorModel):
    module: ClassVar[str] = "zkf_exp2"
    stage_input: int = 0
    stage_reduce: int = 0
    stage_product: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_product, 0, 4)
        for value in (self.stage_reduce, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.wmultiplier, 0, None)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WMULTIPLIER": self.wmultiplier,
                "STAGE_INPUT": self.stage_input,
                "STAGE_REDUCE": self.stage_reduce,
                "STAGE_PRODUCT": self.stage_product,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        degree = _trans_spec("exp2", self.fmt.wman)["d"]
        return (
            self.stage_input
            + self.stage_reduce
            + 4
            + degree * (2 + self.stage_product)
            + self.stage_pack
            + self.stage_output
        )


@dataclass(frozen=True, slots=True)
class Log2Model(OperatorModel):
    module: ClassVar[str] = "zkf_log2"
    stage_input: int = 0
    stage_decode: int = 0
    stage_product: int = 0
    stage_product_final: int | None = None
    stage_normalize: int = 0
    stage_normalize_output: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        spec = _trans_spec("log2", self.fmt.wman)
        _check_int_range(self.stage_input, 0, None)
        _check_int_range(self.stage_product, 0, 4)
        if self.stage_product_final is not None:
            _check_int_range(self.stage_product_final, 0, 4)
        _check_int_range(self.stage_decode, 0, 1)
        norm_width = self.fmt.wexp + self.fmt.wfrac + spec["cf"]
        _check_int_range(self.stage_normalize, 0, 2)
        if self.stage_normalize == 2 and norm_width < 17:
            raise ValueError(f"STAGE_NORMALIZE=2 needs _zkf_normshift W>=17 (got {norm_width}); split=2 needs NL4>=3")
        for value in (self.stage_normalize_output, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.wmultiplier, 0, None)

    @property
    def _stage_product_final(self) -> int:
        return self.stage_product if self.stage_product_final is None else self.stage_product_final

    @property
    def config(self) -> dict[str, int]:
        return {
            "stage_input": self.stage_input,
            "stage_decode": self.stage_decode,
            "stage_product": self.stage_product,
            "stage_product_final": self._stage_product_final,
            "stage_normalize": self.stage_normalize,
            "stage_normalize_output": self.stage_normalize_output,
            "stage_pack": self.stage_pack,
            "stage_output": self.stage_output,
            "wmultiplier": self.wmultiplier,
        }

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WMULTIPLIER": self.wmultiplier,
                "STAGE_INPUT": self.stage_input,
                "STAGE_DECODE": self.stage_decode,
                "STAGE_PRODUCT": self.stage_product,
                "STAGE_PRODUCT_FINAL": self._stage_product_final,
                "STAGE_NORMALIZE": self.stage_normalize,
                "STAGE_NORMALIZE_OUTPUT": self.stage_normalize_output,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        degree = _trans_spec("log2", self.fmt.wman)["d"]
        return (
            self.stage_input
            + self.stage_decode
            + 5
            + self._stage_product_final
            + self.stage_normalize
            + self.stage_normalize_output
            + self.stage_pack
            + degree * (2 + self.stage_product)
            + self.stage_output
        )


@dataclass(frozen=True, slots=True)
class SincosModel(OperatorModel):
    module: ClassVar[str] = "zkf_sincos"
    unroll100: int = 100
    stage_input: int = 0
    stage_product: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        spec = _trig_spec(self.fmt.wman)
        _check_int_range(self.unroll100, 100, None, {50})
        _check_int_range(self.stage_product, 0, 4)
        norm_width = spec["const2pi"].bit_length() + spec["wt"] + 1
        _check_int_range(self.stage_normalize, 0, 2)
        if self.stage_normalize == 2 and norm_width < 17:
            raise ValueError(f"STAGE_NORMALIZE=2 needs _zkf_normshift W>=17 (got {norm_width}); split=2 needs NL4>=3")
        for value in (self.stage_input, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.wmultiplier, 0, None)

    @property
    def config(self) -> dict[str, int]:
        return {
            "unroll100": self.unroll100,
            "stage_input": self.stage_input,
            "stage_product": self.stage_product,
            "stage_normalize": self.stage_normalize,
            "stage_pack": self.stage_pack,
            "stage_output": self.stage_output,
            "wmultiplier": self.wmultiplier,
        }

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WMULTIPLIER": self.wmultiplier,
                "UNROLL100": self.unroll100,
                "STAGE_INPUT": self.stage_input,
                "STAGE_PRODUCT": self.stage_product,
                "STAGE_NORMALIZE": self.stage_normalize,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        k = _trig_spec(self.fmt.wman)["n_sincos"]
        xycyc = (k * 100 + self.unroll100 - 1) // self.unroll100
        parallel = self.unroll100 < 100
        saved = min(1 + self.stage_product, xycyc - k) if parallel else 0
        return (
            11
            + (2 * self.stage_product)
            + xycyc
            - saved
            + self.stage_input
            + self.stage_normalize
            + self.stage_pack
            + self.stage_output
        )


@dataclass(frozen=True, slots=True)
class Atan2Model(OperatorModel):
    module: ClassVar[str] = "zkf_atan2"
    unroll100: int = 100
    stage_input: int = 0
    stage_product: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        spec = _trig_spec(self.fmt.wman)
        _check_int_range(self.unroll100, 100, None, {50})
        _check_int_range(self.stage_product, 0, 4)
        xf = spec["xf"]
        zf = spec["zf"]
        wx = xf + 2
        wquo = 2 * ((xf + 1) // 2) + 1
        norm_width = max((wx - 1) + self.fmt.wman + 5, wquo + self.fmt.wman + 5, zf + 5)
        _check_int_range(self.stage_normalize, 0, 2)
        if self.stage_normalize == 2 and norm_width < 17:
            raise ValueError(f"STAGE_NORMALIZE=2 needs _zkf_normshift W>=17 (got {norm_width}); split=2 needs NL4>=3")
        for value in (self.stage_input, self.stage_pack, self.stage_output):
            _check_int_range(value, 0, 1)
        _check_int_range(self.wmultiplier, 0, None)

    @property
    def params(self) -> dict[str, int]:
        return self._params_with_latency(
            {
                "WEXP": self.fmt.wexp,
                "WMAN": self.fmt.wman,
                "WMULTIPLIER": self.wmultiplier,
                "UNROLL100": self.unroll100,
                "STAGE_INPUT": self.stage_input,
                "STAGE_PRODUCT": self.stage_product,
                "STAGE_NORMALIZE": self.stage_normalize,
                "STAGE_PACK": self.stage_pack,
                "STAGE_OUTPUT": self.stage_output,
            }
        )

    @property
    def latency(self) -> int:
        spec = _trig_spec(self.fmt.wman)
        n = spec["n_atan2"]
        xf = spec["xf_atan2"]
        steps = (xf + 1) // 2
        div_cycles = steps + 1
        xycyc = (n * 100 + self.unroll100 - 1) // self.unroll100
        return (
            8
            + self.stage_input
            + xycyc
            + div_cycles
            + self.stage_product
            + self.stage_normalize
            + self.stage_pack
            + self.stage_output
        )


_MODEL_BY_NAME: dict[str, type[OperatorModel]] = {
    module.removeprefix("_").removeprefix("zkf_"): item
    for name in dir()
    if not name.startswith("_")
    for item in (globals()[name],)
    if isinstance(item, type) and is_dataclass(item)
    for module in (getattr(item, "module", None),)
    if isinstance(module, str)
}


@dataclass(frozen=True, slots=True)
class ZkfFormat:
    """A ZKF format (wexp, wman) plus the factory hub for Zkf values."""

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        _check_int_range(self.wexp, 2, None)
        _check_int_range(self.wman, 4, None)

    def model_of(self, name: str) -> Callable[..., OperatorModel]:
        try:
            model_cls = _MODEL_BY_NAME[name]
        except KeyError:
            raise KeyError(name) from None

        def factory(**config: object) -> OperatorModel:
            return model_cls(self, **config)

        return factory

    @property
    def wfrac(self) -> int:
        return self.wman - 1

    @property
    def wfull(self) -> int:
        return self.wexp + self.wman

    @property
    def sign_shift(self) -> int:
        return self.wexp + self.wfrac

    @property
    def bias(self) -> int:
        return (1 << (self.wexp - 1)) - 1

    @property
    def exp_inf(self) -> int:
        return (1 << self.wexp) - 1

    @property
    def exp_max_finite(self) -> int:
        return self.exp_inf - 1

    @property
    def frac_mask(self) -> int:
        return (1 << self.wfrac) - 1

    @property
    def min_exp_unbiased(self) -> int:
        return 1 - self.bias

    @property
    def max_exp_unbiased(self) -> int:
        return self.exp_max_finite - self.bias

    @property
    def lowest(self) -> Fraction:
        """Smallest representable positive magnitude (no subnormals, so this is also the smallest normal)."""
        return _pow2_fraction(self.min_exp_unbiased)

    @property
    def max(self) -> Fraction:
        """Largest finite magnitude."""
        return (Fraction(2) - _pow2_fraction(-self.wfrac)) * _pow2_fraction(self.max_exp_unbiased)

    @property
    def epsilon(self) -> Fraction:
        """Gap between 1.0 and the next representable value above it."""
        return _pow2_fraction(-self.wfrac)

    def wrap(self, bits: int) -> Zkf:
        """Wrap raw packed bits (masked to WFULL) as a Zkf."""
        return Zkf(self, bits)

    def zero(self, sign: int = 0) -> Zkf:
        return Zkf(self, _pack_bits(self, sign, 0, 0))

    def inf(self, sign: int = 0) -> Zkf:
        return Zkf(self, _pack_bits(self, sign, self.exp_inf, 0))

    def normal(self, sign: int, exp: int, frac: int) -> Zkf:
        return Zkf(self, _normal(self, sign, exp, frac))

    def pack(self, sign: int, exp: int, frac: int) -> Zkf:
        """
        Construct a value directly from raw field values (each masked to width); no canonicalization or range
        checks. For non-canonical test vectors; prefer zero/inf/normal/encode.
        """
        return Zkf(self, _pack_bits(self, sign, exp, frac))

    def from_int(self, wint: int, value: int) -> Zkf:
        """Convert a signed wint-bit integer to the nearest ZKF value (RNTE)."""
        if not _signed_int_min(wint) <= value <= _signed_int_max(wint):
            raise ValueError(f"value={value} out of signed {wint}-bit range")
        if value == 0:
            return Zkf(self, _zero(self))
        return Zkf(self, _round_fraction_to_zkf(self, int(value < 0), Fraction(abs(value))))

    def encode(self, value: float | int | Fraction) -> Zkf:
        """Round a real number to this format (RNTE). float NaN -> ValueError; float +/-inf -> inf."""
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError("NaN is not representable in ZKF")
            if math.isinf(value):
                return self.inf(int(value < 0))
        value = Fraction(value)
        if value == 0:
            return self.zero()
        return Zkf(self, _round_fraction_to_zkf(self, int(value < 0), abs(value)))


class DivResult(NamedTuple):
    quotient: "Zkf"
    div_by_zero: bool


class Log2Result(NamedTuple):
    value: "Zkf"
    domain_error: bool
    pole: bool


class SinCos(NamedTuple):
    sin: "Zkf"
    cos: "Zkf"
    quadrant: int


class Atan2Result(NamedTuple):
    theta: "Zkf"
    magnitude: "Zkf"


class CmpResult(NamedTuple):
    lt: bool
    eq: bool
    gt: bool


@dataclass(frozen=True, slots=True, repr=False)
class Zkf:
    """
    An immutable ZKF value: a ZkfFormat and the packed integer bits.

    Bit-exact model of the RTL operators, exposed as Python operators/methods. Equality and hashing are structural
    (same format and same bits) -- the correct semantics for a verification value type, and what catches a
    canonicalization mismatch that a numeric compare would hide. Numeric comparison is the explicit cmp method.
    Arithmetic requires both operands to share a format; a mismatch raises ValueError.
    """

    fmt: ZkfFormat
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", self.bits & _mask(self.fmt.wfull))

    @property
    def negative(self) -> bool:
        return bool((self.bits >> self.fmt.sign_shift) & 1)

    @property
    def exp(self) -> int:
        return (self.bits >> self.fmt.wfrac) & self.fmt.exp_inf

    @property
    def frac(self) -> int:
        return self.bits & self.fmt.frac_mask

    @property
    def is_zero(self) -> bool:
        return self.exp == 0

    @property
    def is_inf(self) -> bool:
        return self.exp == self.fmt.exp_inf

    @property
    def is_normal(self) -> bool:
        return 0 < self.exp < self.fmt.exp_inf

    @property
    def is_finite(self) -> bool:
        return not self.is_inf

    def significand(self) -> int:
        """WMAN-bit significand including the implicit leading one."""
        return _significand(self.fmt, self.bits)

    def to_fraction(self) -> Fraction:
        """Exact value as a Fraction. Raises on infinity (Fraction has no infinity)."""
        if self.is_inf:
            raise ValueError("infinity has no finite value; use is_inf/negative")
        if self.is_zero:
            return Fraction(0)
        mag = Fraction(self.significand()) * _pow2_fraction(self.exp - self.fmt.bias - self.fmt.wfrac)
        return -mag if self.negative else mag

    def to_int(self, wint: int) -> int:
        """The zkf_to_int hardware operator: round to a signed wint-bit integer (ties-to-even, saturating)."""
        int_max = _signed_int_max(wint)
        int_min = _signed_int_min(wint)
        if self.is_inf:
            return int_min if self.negative else int_max
        if self.is_zero:
            return 0
        rounded = _round_fraction_to_int_ties_even(self.to_fraction())
        if rounded > int_max:
            return int_max
        if rounded < int_min:
            return int_min
        return rounded

    def __float__(self) -> float:
        if self.is_inf:
            return -math.inf if self.negative else math.inf
        return float(self.to_fraction())

    def __int__(self) -> int:
        if self.is_inf:
            raise OverflowError("cannot convert ZKF infinity to integer")
        return math.trunc(self.to_fraction())

    def __bool__(self) -> bool:
        return not self.is_zero

    def __repr__(self) -> str:
        return f"Zkf({self.fmt!r}, {self:x})"

    def __str__(self) -> str:
        return format(self)

    def __format__(self, spec: str) -> str:
        if spec == "x":
            return f"0x{self.bits:0{(self.fmt.wfull + 3) // 4}x}"
        return format(float(self), spec)

    def _require_same(self, other: "Zkf") -> None:
        if other.fmt != self.fmt:
            raise ValueError(f"operands have different formats: {self.fmt} vs {other.fmt}")

    def __mul__(self, other: "Zkf") -> "Zkf":
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        fmt = self.fmt
        a = self
        b = other
        result_zero = a.is_zero or b.is_zero
        result_inf = (not result_zero) and (a.is_inf or b.is_inf)

        product = _significand(fmt, a.bits) * _significand(fmt, b.bits)
        product_high = (product >> ((2 * fmt.wman) - 1)) & 1
        exp_unbiased_base = a.exp + b.exp - (fmt.bias << 1)

        if product_high:
            exp_unbiased = exp_unbiased_base + 1
            significand_value = (product >> fmt.wman) & _mask(fmt.wman)
            guard = (product >> (fmt.wman - 1)) & 1
            round_bit = (product >> (fmt.wman - 2)) & 1
            sticky = _sticky_below(product, fmt.wman - 3)
        else:
            exp_unbiased = exp_unbiased_base
            significand_value = (product >> (fmt.wman - 1)) & _mask(fmt.wman)
            guard = (product >> (fmt.wman - 2)) & 1
            round_bit = (product >> (fmt.wman - 3)) & 1
            sticky = _sticky_below(product, fmt.wman - 4)

        return Zkf(
            fmt,
            _pack_reference(
                fmt,
                a.negative ^ b.negative,
                int(result_zero),
                int(result_inf),
                exp_unbiased,
                significand_value,
                guard,
                round_bit,
                sticky,
            ),
        )

    def __add__(self, other: "Zkf") -> "Zkf":
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        fmt = self.fmt
        a = self
        b = other

        if a.is_inf and b.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, a.negative) if a.negative == b.negative else _zero(fmt))
        if a.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, a.negative))
        if b.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, b.negative))

        result = a.to_fraction() + b.to_fraction()
        if result == 0:
            return Zkf(fmt, _zero(fmt))
        return Zkf(fmt, _round_fraction_to_zkf(fmt, int(result < 0), abs(result)))

    def __sub__(self, other: "Zkf") -> "Zkf":
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        return self + (-other)

    def __truediv__(self, other: "Zkf") -> "Zkf":
        if not isinstance(other, Zkf):
            return NotImplemented
        self._require_same(other)
        return self.div(other).quotient

    def __neg__(self) -> "Zkf":
        return Zkf(self.fmt, self.bits ^ (1 << self.fmt.sign_shift))  # Zkf.__post_init__ masks to WFULL

    def __abs__(self) -> "Zkf":
        return Zkf(self.fmt, self.bits & _mask(self.fmt.sign_shift))

    def div(self, other: "Zkf") -> DivResult:
        """Quotient plus the div-by-zero status flag. a / b is a.div(b).quotient."""
        self._require_same(other)
        fmt = self.fmt
        a = self
        b = other
        div0 = b.is_zero

        if a.is_zero or b.is_inf:
            return DivResult(Zkf(fmt, _zero(fmt)), div0)

        result_sign = a.negative if b.is_zero else (a.negative ^ b.negative)
        if b.is_zero or a.is_inf:
            return DivResult(Zkf(fmt, _canonical_inf(fmt, result_sign)), div0)

        value = abs(a.to_fraction() / b.to_fraction())
        return DivResult(Zkf(fmt, _round_fraction_to_zkf(fmt, result_sign, value)), div0)

    def fma(self, b: "Zkf", c: "Zkf") -> "Zkf":
        """Correctly-rounded fused multiply-add: round(self*b + c) with a single rounding."""
        self._require_same(b)
        self._require_same(c)
        fmt = self.fmt
        a = self

        p_zero = a.is_zero or b.is_zero
        p_inf = (not p_zero) and (a.is_inf or b.is_inf)
        p_sign = a.negative ^ b.negative

        if p_inf and c.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, p_sign) if p_sign == c.negative else _zero(fmt))
        if p_inf:
            return Zkf(fmt, _canonical_inf(fmt, p_sign))
        if c.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, c.negative))

        product = Fraction(0) if p_zero else a.to_fraction() * b.to_fraction()
        result = product + c.to_fraction()
        if result == 0:
            return Zkf(fmt, _zero(fmt))
        return Zkf(fmt, _round_fraction_to_zkf(fmt, int(result < 0), abs(result)))

    def mul_ilog2(self, k: int) -> "Zkf":
        """Multiply by 2**k for signed integer k (exponent add); models zkf_mul_ilog2 and zkf_mul_ilog2_const."""
        fmt = self.fmt
        a = self
        if a.is_zero:
            return Zkf(fmt, _zero(fmt))
        if a.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, a.negative))
        new_exp = a.exp + k
        if new_exp < 0:
            return Zkf(fmt, _zero(fmt))
        if new_exp == 0:
            return Zkf(fmt, _normal(fmt, a.negative, 1, 0))
        if new_exp > fmt.exp_max_finite:
            return Zkf(fmt, _canonical_inf(fmt, a.negative))
        return Zkf(fmt, _normal(fmt, a.negative, new_exp, a.frac))

    def saturate(self) -> "Zkf":
        """Clamp infinities to the largest finite magnitude; finite values pass through."""
        fmt = self.fmt
        if not self.is_inf:
            return Zkf(fmt, self.bits)
        return Zkf(fmt, _normal(fmt, self.negative, fmt.exp_max_finite, fmt.frac_mask))

    def canonicalize(self) -> "Zkf":
        """
        Fold non-canonical encodings to their canonical form: any zero -> +0 (sign and stray fraction cleared),
        any infinity -> sign-preserving canonical infinity (fraction cleared); finite values pass through unchanged.
        """
        return Zkf(self.fmt, _canonicalize_special(self.fmt, self.bits))

    def resize(self, fmt_out: ZkfFormat) -> "Zkf":
        """Re-encode into another format (RNTE)."""
        if self.is_zero:
            return Zkf(fmt_out, _zero(fmt_out))
        if self.is_inf:
            return Zkf(fmt_out, _canonical_inf(fmt_out, self.negative))
        return Zkf(fmt_out, _round_fraction_to_zkf(fmt_out, self.negative, abs(self.to_fraction())))

    def round(self) -> "Zkf":
        """Round to the nearest integral value, ties to even (the zkf_round RNTE mode)."""
        return self._round_to_integral(_RoundMode.NEAREST_EVEN)

    def floor(self) -> "Zkf":
        """Round toward -inf to an integral value (the zkf_round floor mode)."""
        return self._round_to_integral(_RoundMode.FLOOR)

    def ceil(self) -> "Zkf":
        """Round toward +inf to an integral value (the zkf_round ceil mode)."""
        return self._round_to_integral(_RoundMode.CEIL)

    def trunc(self) -> "Zkf":
        """Round toward zero to an integral value (the zkf_round trunc mode)."""
        return self._round_to_integral(_RoundMode.TRUNC)

    def _round_to_integral(self, mode: _RoundMode) -> "Zkf":
        """Round to an integral value in the same format per mode (the zkf_round operator)."""
        fmt = self.fmt
        if self.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, self.negative))
        if self.is_zero:
            return Zkf(fmt, _zero(fmt))
        rounded = _round_signed_fraction_to_int(self.to_fraction(), mode)
        if rounded == 0:
            return Zkf(fmt, _zero(fmt))
        return Zkf(fmt, _round_fraction_to_zkf(fmt, int(rounded < 0), Fraction(abs(rounded))))

    def cmp(self, other: "Zkf") -> CmpResult:
        """Numeric comparison mirroring the RTL zkf_cmp (total order over the format)."""
        self._require_same(other)
        a_key = _ordered_key(self.fmt, self.bits)
        b_key = _ordered_key(self.fmt, other.bits)
        return CmpResult(a_key < b_key, a_key == b_key, a_key > b_key)

    def sort(self, other: "Zkf") -> tuple["Zkf", "Zkf"]:
        """Return (min, max) of the pair by numeric order (the zkf_sort operator)."""
        self._require_same(other)
        lt = self.cmp(other).lt
        a_bits, b_bits = self.bits, other.bits
        lo, hi = (a_bits, b_bits) if lt else (b_bits, a_bits)
        return Zkf(self.fmt, lo), Zkf(self.fmt, hi)

    def exp2(self) -> "Zkf":
        """2 ** self (the zkf_exp2 operator)."""
        fmt, bits = self.fmt, self.bits
        d = self
        if d.is_inf:
            return Zkf(fmt, _canonical_inf(fmt, 0) if not d.negative else _zero(fmt))  # +inf -> +inf, -inf -> +0
        if d.is_zero:
            return Zkf(fmt, _normal(fmt, 0, fmt.bias, 0))  # 2**0 = 1.0
        e = d.exp - fmt.bias
        # |x| >= 2^(WEXP-1) is always out of range: overflow (x>0) or underflow (x<0).
        if e >= fmt.wexp - 1:
            return Zkf(fmt, _canonical_inf(fmt, 0) if not d.negative else _zero(fmt))

        spec = _trans_spec("exp2", fmt.wman)
        cf, rw = spec["cf"], spec["rw"]
        ff = spec["k"] + rw  # full reduced-argument width FF = K + RW
        sig = _significand(fmt, bits)
        shift = e - fmt.wfrac + ff
        if shift >= 0:
            mfix = sig << shift
            lost_sticky = 0
        else:
            rs = -shift
            mfix = sig >> rs
            lost_sticky = 1 if (sig & _mask(rs)) else 0
        v = -mfix if d.negative else mfix
        i = v >> ff  # arithmetic floor -> integer part of x
        f = v & _mask(ff)  # fractional part in [0, 2^FF)

        acc = _horner_eval(spec["coeffs"][(f >> rw) - spec.get("seg_base", 0)], f & _mask(rw), rw)
        significand_value = (acc >> (cf - fmt.wfrac)) & _mask(fmt.wman)
        guard = (acc >> (cf - fmt.wman)) & 1
        round_bit = (acc >> (cf - fmt.wman - 1)) & 1
        sticky = (1 if (acc & _mask(cf - fmt.wman - 1)) else 0) | lost_sticky
        return Zkf(fmt, _pack_reference(fmt, 0, 0, 0, i, significand_value, guard, round_bit, sticky))

    def log2(self) -> Log2Result:
        """log2(self) plus the domain-error (self<0) and pole (self==0) status flags."""
        fmt, bits = self.fmt, self.bits
        d = self
        if d.is_inf and not d.negative:
            return Log2Result(Zkf(fmt, _canonical_inf(fmt, 0)), False, False)  # log2(+inf) = +inf
        if d.is_zero:
            return Log2Result(Zkf(fmt, _canonical_inf(fmt, 1)), False, True)  # log2(+0) = -inf, pole
        if d.negative:
            return Log2Result(Zkf(fmt, _canonical_inf(fmt, 1)), True, False)  # log2(x<0) = -inf, domain error

        e = d.exp - fmt.bias
        spec = _trans_spec("log2", fmt.wman)
        cf, rw = spec["cf"], spec["rw"]

        # Symmetric argument reduction (mirrors the phase-2 RTL re-center; defines the bit-exact contract). x = m*2^e,
        # m = sig/2^WFRAC in [1,2). If m >= sqrt(2), halve m and increment e so m' in [sqrt(1/2), sqrt(2)). At scale
        # 2^-(WFRAC+1): v = f + 1/2 is the unsigned segment index, f_signed = m'-1 (= f) the signed combine operand.
        sig = _significand(fmt, bits)  # WMAN-bit significand, m = sig / 2^WFRAC in [1,2)
        if sig >= _trans_sqrt2_threshold(fmt.wfrac):  # m >= sqrt(2): re-center into [sqrt(1/2), sqrt(2))
            e += 1
            v = d.frac  # = sig - 2^WFRAC
            f_signed = d.frac - (1 << fmt.wfrac)  # < 0
        else:
            v = (1 << fmt.wfrac) + (d.frac << 1)  # = 2*sig - 2^WFRAC
            f_signed = d.frac << 1  # = 2*frac, >= 0

        acc = _horner_eval(spec["coeffs"][(v >> rw) - spec.get("seg_base", 0)], v & _mask(rw), rw)
        f2 = fmt.wfrac + 1 + cf  # f is at scale 2^-(WFRAC+1)
        l_signed = f_signed * acc  # log2(m') = f * C(f), signed, at scale 2^-f2
        r = (e << f2) + l_signed  # signed fixed point e + log2(m')
        sign_out = 1 if r < 0 else 0
        magnitude = -r if r < 0 else r
        w_norm = fmt.wexp + f2 + 1
        zero_flag, count, aligned = _normshift_reference(w_norm, magnitude)
        significand_value = (aligned >> (w_norm - fmt.wman)) & _mask(fmt.wman)
        guard = (aligned >> (w_norm - fmt.wman - 1)) & 1
        round_bit = (aligned >> (w_norm - fmt.wman - 2)) & 1
        sticky = 1 if (aligned & _mask(w_norm - fmt.wman - 2)) else 0
        exp_unbiased = (w_norm - 1 - count) - f2
        y = _pack_reference(fmt, sign_out, zero_flag, 0, exp_unbiased, significand_value, guard, round_bit, sticky)
        return Log2Result(Zkf(fmt, y), False, False)

    def sincos(self) -> SinCos:
        """(sin(2*pi*self), cos(2*pi*self), quadrant) -- the zkf_sincos operator."""
        fmt, bits = self.fmt, self.bits
        d = self
        if d.is_inf:
            s = _canonical_inf(fmt, d.negative)
            return SinCos(Zkf(fmt, s), Zkf(fmt, s), 0)  # +inf -> (+inf,+inf,0); -inf -> (-inf,-inf,0)
        if d.is_zero:
            return SinCos(Zkf(fmt, _zero(fmt)), Zkf(fmt, _normal(fmt, 0, fmt.bias, 0)), 0)  # sin(0)=+0, cos(0)=+1

        spec = _trig_spec(fmt.wman)
        xf, zf = spec["xf"], spec["zf"]
        # const2pi arrives PRE-NARROWED from the table (top WMAN+5 bits == round(2*pi * 2**const2pi_s)); const2pi_s is
        # the single source of scale for every consuming shift / exp-offset. Mirrors zkf/rtl/zkf_sincos.v.
        const2pi, const2pi_s = spec["const2pi"], spec["const2pi_s"]
        n_sincos = spec["n_sincos"]  # sincos iterations (linear-rotation termination); == table n
        wt = spec["wt"]  # quadrant-local coordinate width (FF - 2)
        ff = wt + 2
        zg = zf - (wt + 2)  # extra angle-accumulator fractional bits (GUARD_ZF)
        # Uniform magnitude width wmag sized for the widest product (const2pi * t'). CORDIC magnitudes are at scale
        # 2**-xf (exp offset eone_xf); const2pi products at 2**-const2pi_s (eone_s). Mirrors RTL width+exp.
        cwb = const2pi.bit_length()  # narrowed 2*pi width == WMAN+5
        wmag = cwb + wt + 1
        eone_xf = wmag - 1 - xf  # exp_offset for a magnitude at scale 2**-xf (corr / +1 path)
        eone_s = wmag - 1 - const2pi_s  # exp_offset for a const2pi product at scale 2**-const2pi_s
        one = (1 << xf, eone_xf)  # value +1.0
        tsa = spec["tsa"]
        sig = _significand(fmt, bits)
        e = d.exp - fmt.bias

        # -- Reduce |x| mod 1 to the FF-bit fraction; SH = e - WFRAC + FF places |sig| at scale 2**-FF. Using
        # sin(2*pi*x) = sign*sin(2*pi*|x|), cos(2*pi*x) = cos(2*pi*|x|) collapses the x<0 negate to a sin sign flip.
        sh = e - fmt.wfrac + ff
        tiny = sh < 0  # below the reducer's resolution -> small-angle path on |x|
        lshamt = 0 if tiny else min(sh, ff)
        frac_pos = (sig << lshamt) & _mask(ff)
        quadrant_abs = 0 if tiny else (frac_pos >> wt) & 3  # |x| quadrant (0 for tiny: |x| < 1/4)
        t = sig if tiny else (frac_pos & _mask(wt))  # quadrant-local coordinate, scale 2**-WT
        tzero = (not tiny) and t == 0  # frac(|x|)*4 integer: a quadrant boundary / exact magnitude

        # -- Octant fold: bring the local angle into [0, pi/4] (t' <= 1/2). theta' in turns at scale 2**-zf is just t'.
        half = 1 << (wt - 1)
        oct_flip = (not tiny) and t > half  # theta in (pi/4, pi/2): use the pi/2 - theta complement
        tp = ((1 << wt) - t) if oct_flip else t

        # -- Octant-local (sin theta', cos theta') as (magnitude, exp_offset) pairs at the uniform WMAG scale.
        if tiny:
            # Under-resolution: sin ~= 2*pi*|x| = 2*pi*|sig|*2**(e-wfrac); cos = +1. const2pi*|sig| sits at scale
            # 2**-const2pi_s, so its exp_offset is eone_s plus the data binade (e - wfrac). No correction token.
            sin_tp = (const2pi * sig, eone_s + e - fmt.wfrac)
            cos_tp = one
        elif tzero:
            sin_tp, cos_tp = (0, eone_xf), one  # exact quadrant boundary: sin theta' = 0, cos theta' = 1
        elif tp < tsa:
            # Small octant-local angle (cos=1): sin theta' ~= 2*pi*tp (tp at scale 2**-(WT+2)); const2pi*tp is at
            # 2**-const2pi_s, so exp_offset = eone_s - (wt + 2).
            sin_tp = (const2pi * tp, eone_s - (wt + 2))
            cos_tp = one
        else:
            # K CORDIC iterations then one linear rotation by residual z_K (phi = 2*pi*z_K): sin theta' = y_K + x_K*phi,
            # cos theta' = x_K - y_K*phi.
            xk, yk, zk = _cordic_rotate(spec, tp << zg, n_sincos)  # seed z0 = t' shifted into the finer 2**-zf scale
            # phi and x_K/y_K are narrowed to ~18-bit operands (one 18x18 DSP each); dropping the low bits of this small
            # fix-up stays < 1 ULP (--check confirms).
            n = n_sincos  # phi's natural width XF-N+2 uses the sincos iteration count
            phiw = min(fmt.wman + 6, max(2, xf - n + 2))  # phi top bits (natural width XF-N+2, capped at WMAN+6)
            phi_trunc = max(0, (xf - n + 2) - phiw)
            phi_s = xf - phi_trunc  # scale of the narrowed phi (== 2*pi*z_K at 2**-phi_s)
            xcw = fmt.wman + 6  # x_K/y_K correction-operand top bits
            xk_trunc = (xf + 2) - xcw  # XW = XF+2; keep the top XCW bits
            # phi = const2pi*z_K (scale 2**-(const2pi_s + zf)) narrowed to PHIW bits at scale 2**-phi_s by a single
            # right-shift (const2pi_s + zf) - phi_s. const2pi is the pre-narrowed operand, so no correction token.
            phi = (const2pi * zk) >> ((const2pi_s + zf) - phi_s)  # signed, PHIW bits, scale 2**-phi_s
            corr_s = ((xk >> xk_trunc) * phi) >> (xf - xk_trunc - phi_trunc)  # x_K*phi at scale 2**-xf
            corr_c = ((yk >> xk_trunc) * phi) >> (xf - xk_trunc - phi_trunc)
            sin_tp = (yk + corr_s, eone_xf)
            cos_tp = (xk - corr_c, eone_xf)

        # -- Unmap the octant (sin theta = cos theta', cos theta = sin theta' when folded), then the |x| quadrant.
        sin_loc, cos_loc = (cos_tp, sin_tp) if oct_flip else (sin_tp, cos_tp)
        sin_m, cos_m = (cos_loc, sin_loc) if (quadrant_abs & 1) else (sin_loc, cos_loc)
        sin_sign = ((quadrant_abs >> 1) & 1) ^ d.negative  # sin is odd: negative x flips it
        cos_sign = ((quadrant_abs >> 1) ^ quadrant_abs) & 1  # cos is even: unchanged by the sign of x
        # Output quadrant = floor(frac(x)*4): for x >= 0 the |x| quadrant; for x < 0 it reflects about a turn.
        if not d.negative:
            quadrant = quadrant_abs
        else:
            quadrant = ((4 - quadrant_abs) & 3) if tzero else (3 - quadrant_abs)

        sin_bits = _fixed_to_float_ref(fmt, sin_sign, sin_m[0], sin_m[1], wmag)
        cos_bits = _fixed_to_float_ref(fmt, cos_sign, cos_m[0], cos_m[1], wmag)
        return SinCos(Zkf(fmt, sin_bits), Zkf(fmt, cos_bits), quadrant)

    def atan2(self, x: "Zkf") -> Atan2Result:
        """
        (theta, magnitude) of atan2(self, x): theta in turns (-0.5, 0.5], magnitude = hypot(self, x).

        Reuses the sincos CORDIC engine in vectoring mode: it yields z_K ~= atan2 (turns) and x_K ~= gain*hypot; a
        residual divide finishes the small angle, a small-ratio bypass covers the near-+x-axis, and the magnitude
        descales x_K by 1/gain (== KINV).
        """
        self._require_same(x)
        fmt, y_bits, x_bits = self.fmt, self.bits, x.bits
        sp = atan2_special(fmt, y_bits, x_bits)
        if sp is not None:
            return Atan2Result(Zkf(fmt, sp[0]), Zkf(fmt, sp[1]))

        spec = _trig_spec(fmt.wman)
        # xf is the SHARED engine width (table WX/KINV/INV_TAU and the returned x_K/y_K all at 2**-xf). xf_atan2 is
        # atan2's own x/y width, driving only the divider quotient budget F; equal today, read separately to decouple.
        xf, zf = spec["xf"], spec["zf"]
        n = spec["n_atan2"]  # atan2 iterations (residual-divide termination)
        xf_div = spec["xf_atan2"]  # divider x/y fractional width (== xf today)
        wfrac = fmt.wfrac

        # The shared _zkf_pmul multiplies x_K*kinv_mag (magnitude) and Q*inv_tau (residual + bypass theta). Both
        # constants are PRE-NARROWED to WMAN+5 bits at their native scales (kinv_s, invtau_s), so every dependent shift
        # is "product-scale minus target-scale" with no fold-back. Mirrors zkf/rtl/zkf_atan2.v.
        kinv_mag, kinv_s = spec["kinv_mag"], spec["kinv_s"]  # narrowed 1/gain (MAG product) + its native scale
        inv_tau, invtau_s = spec["inv_tau"], spec["invtau_s"]  # narrowed 1/(2*pi) (residual + bypass) + native scale

        dy = self
        dx = x
        sx, sy = dx.negative, dy.negative
        sig_x = _significand(fmt, x_bits)
        sig_y = _significand(fmt, y_bits)
        ex = dx.exp - fmt.bias
        ey = dy.exp - fmt.bias

        # Order by magnitude: den = max(|x|,|y|), num = min. The reduced octant angle a0 = atan(num/den) in [0, 1/8].
        swap = (ey > ex) or (ey == ex and sig_y > sig_x)  # |y| > |x|
        if swap:
            den_sig, e_den, num_sig, e_num = sig_y, ey, sig_x, ex
        else:
            den_sig, e_den, num_sig, e_num = sig_x, ex, sig_y, ey
        shift_dn = e_den - e_num  # >= 0 (den >= num)
        # Pre-scale the vector by 1/4 (den_fixed in [0.25, 0.5)*2**xf) so x_K = gain*hypot stays inside the engine's
        # signed width WX=xf+2; the 1/4 is folded into the magnitude exponent (+2) and the angle is invariant.
        den_fixed = den_sig << (xf - wfrac - 2)  # in [0.25, 0.5) * 2**xf
        num_fixed = (num_sig << (xf - wfrac - 2)) >> shift_dn  # num aligned to den's scale 2**-xf

        # Magnitude (always via the engine): x_K = gain*hypot(num_fixed, den_fixed); descale by 1/gain == KINV (scale
        # 2**-xf each), so M = x_K*KINV is hypot at scale 2**-2xf, then carry the den binade e_den plus the +2 of 1/4.
        x_k, y_k, z_k = _cordic_vector(spec, den_fixed, num_fixed, n)
        mag_prod = x_k * kinv_mag  # x_K * kinv_mag -> M at 2**-(xf+kinv_s)
        wmag_m = 2 * xf + 4  # holds M for normshift (value-invariant to field width)
        exp_off_m = (wmag_m - 1) - (xf + kinv_s) + e_den + 2  # read M back; den binade + 1/4 pre-scale undone
        mag_bits = _fixed_to_float_ref(fmt, 0, mag_prod, exp_off_m, wmag_m)

        # Quotient fractional budget F = 2*ceil(xf/2): q = floor(|y_K|*2**F/x_K) must carry >= wman significant bits.
        # The divide is truncating floor division (radix-independent), so Python //,% match the radix-4 divider exactly.
        f_bits = 2 * ((xf_div + 1) // 2)  # divider F from atan2's OWN xf (== xf today)

        # theta. Bypass only the near-+x-axis tiny-theta corner; everywhere else theta sits near a boundary
        # (+-1/4, +-1/2) and the fixed-turns path (scale 2**-zf, zf >> wman) has ample relative precision.
        tiny_shift = _atan2_bypass_shift(fmt)
        if (not swap) and (sx == 0) and (shift_dn > tiny_shift):
            # theta ~= (|y|/|x|)*INV_TAU (atan(r) ~= r): one truncating divide (F frac bits + sticky), *INV_TAU, then a
            # single RTNE pack consuming the sticky. F >= wman+GUARD_DIV keeps the truncation below the round bit.
            r = (num_sig << f_bits) // den_sig  # floor((|y|/|x|)*2**F), F frac bits (+ possible int bit)
            sticky = 1 if ((num_sig << f_bits) % den_sig) else 0
            prod_j = (r * inv_tau) | sticky  # r * inv_tau; sticky in bit 0 (RTNE)
            wmag_b = 2 * xf + 4  # same renormalize field as residual / magnitude paths
            exp_off_b = (wmag_b - 1) + (e_num - e_den) - f_bits - invtau_s  # read r*inv_tau back at 2**-(F+invtau_s)
            return Atan2Result(Zkf(fmt, _fixed_to_float_ref(fmt, sy, prod_j, exp_off_b, wmag_b)), Zkf(fmt, mag_bits))

        # Residual correction: a0 = z_K + (y_K/x_K)*INV_TAU at the angle scale 2**-zf. The divide is truncating (matches
        # the folded radix-4 fixed-point divider); x_K > 0 always, y_K is signed (vectoring drives y through 0).
        qf = f_bits  # quotient fractional bits (>= wman+GUARD_DIV)
        aq = -y_k if y_k < 0 else y_k
        q = (aq << qf) // x_k
        # q*inv_tau is at scale 2**-(qf + invtau_s); the right-shift to the angle scale 2**-zf is the difference
        # qf + invtau_s - zf (>= 0 for every supported WMAN; the (-sh_d) guard mirrors the RTL).
        sh_d = qf + invtau_s - zf
        qti = q * inv_tau
        delta = (qti >> sh_d) if sh_d >= 0 else (qti << (-sh_d))  # |delta| at scale 2**-zf
        a0 = z_k - delta if y_k < 0 else z_k + delta

        # Octant/quadrant unmap into theta (turns, scale 2**-zf): phi1 in [0, 1/4], theta_mag in [0, 1/2], sign = sy.
        quarter = 1 << (zf - 2)
        half = 1 << (zf - 1)
        phi1 = (quarter - a0) if swap else a0
        theta_mag = (half - phi1) if sx else phi1
        wmag_t = zf + 2
        exp_off_t = wmag_t - 1 - zf  # reads theta_mag at scale 2**-zf back as itself
        theta_bits = _fixed_to_float_ref(fmt, sy, theta_mag, exp_off_t, wmag_t)
        # fold the generic negative-x-axis limit -1/2 -> +1/2
        return Atan2Result(Zkf(fmt, atan2_canon_half(fmt, theta_bits)), Zkf(fmt, mag_bits))


def bits_to_signed(value: int, width: int) -> int:
    value &= _mask(width)
    sign_bit = 1 << (width - 1)
    return value - (1 << width) if value & sign_bit else value


def atan2_canon_half(fmt: ZkfFormat, theta_bits: int) -> int:
    """
    Fold the out-of-range -1/2-turn to the canonical +1/2. The generic atan2 path packs -1/2 (sign set) as the
    negative-x-axis limit (finite x<0, |y|->0), but the documented output range is the half-open (-0.5, 0.5], which
    excludes -1/2. This is the inverse of _atan2_turn's frac==1/2 -> sign 0 rule, mirrored in the RTL turn8 k==4 clamp.
    -1/2 is the only out-of-range value the generic path (|theta| <= 1/2) can produce, so nothing else is affected.
    """
    pos_half = _atan2_turn(fmt, 0, Fraction(1, 2))
    return pos_half if theta_bits == (pos_half | (1 << fmt.sign_shift)) else theta_bits


def atan2_special(fmt: ZkfFormat, y_bits: int, x_bits: int) -> tuple[int, int] | None:
    """
    Shared special-case table for atan2 (no NaN; only +0; tiny negatives flush to +0). Returns (theta, mag) bits or
    None for the both-finite-nonzero generic path. Used by BOTH the reference and the mpmath oracle so they agree on
    the exact dyadic constants at the axes/diagonals.
    """
    dy = Zkf(fmt, y_bits)
    dx = Zkf(fmt, x_bits)
    if dx.is_inf or dy.is_inf:
        mag = _canonical_inf(fmt, 0)  # hypot with any inf operand is +inf
        if dx.is_inf and dy.is_inf:  # diagonals: +-pi/4 (x>0) / +-3pi/4 (x<0) -> +-1/8 / +-3/8
            return _atan2_turn(fmt, dy.negative, Fraction(3, 8) if dx.negative else Fraction(1, 8)), mag
        if dy.is_inf:  # |y|=inf, x finite -> +-1/4 (vertical)
            return _atan2_turn(fmt, dy.negative, Fraction(1, 4)), mag
        if dx.negative:  # x=-inf -> half-turn endpoint, canonicalized to +1/2
            return _atan2_turn(fmt, 0 if dy.is_zero else dy.negative, Fraction(1, 2)), mag
        return _zero(fmt), mag  # x=+inf, y finite -> +-0 -> +0 (no -0)
    if dx.is_zero and dy.is_zero:
        return _zero(fmt), _zero(fmt)  # atan2(0,0)=+0, hypot=+0
    if dy.is_zero:  # y=0, x finite nonzero: +0 (x>0) / 1/2 (x<0); mag=|x|
        theta = _atan2_turn(fmt, 0, Fraction(1, 2)) if dx.negative else _zero(fmt)
        return theta, (x_bits & _mask(fmt.sign_shift))
    if dx.is_zero:  # x=0, y finite nonzero -> +-1/4; mag=|y|
        return _atan2_turn(fmt, dy.negative, Fraction(1, 4)), (y_bits & _mask(fmt.sign_shift))
    return None


def _mask(width: int) -> int:
    return (1 << width) - 1


def _pack_bits(fmt: ZkfFormat, sign: int, exp: int, frac: int) -> int:
    return ((sign & 1) << fmt.sign_shift) | ((exp & _mask(fmt.wexp)) << fmt.wfrac) | (frac & fmt.frac_mask)


def _zero(fmt: ZkfFormat) -> int:
    return _pack_bits(fmt, 0, 0, 0)


def _canonical_inf(fmt: ZkfFormat, sign: int) -> int:
    return _pack_bits(fmt, sign, fmt.exp_inf, 0)


def _normal(fmt: ZkfFormat, sign: int, exp: int, frac: int) -> int:
    if not 1 <= exp <= fmt.exp_max_finite:
        raise ValueError(f"normal exponent out of range: {exp}")
    if not 0 <= frac <= fmt.frac_mask:
        raise ValueError(f"fraction out of range: {frac}")
    return _pack_bits(fmt, sign, exp, frac)


def _significand(fmt: ZkfFormat, bits: int) -> int:
    return (1 << fmt.wfrac) | (bits & fmt.frac_mask)


def _pow2_fraction(exp: int) -> Fraction:
    return Fraction(1 << exp, 1) if exp >= 0 else Fraction(1, 1 << -exp)


def _floor_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 is defined for positive values only")
    exp = value.numerator.bit_length() - value.denominator.bit_length()
    while _pow2_fraction(exp + 1) <= value:
        exp += 1
    while _pow2_fraction(exp) > value:
        exp -= 1
    return exp


def _round_fraction_to_zkf(fmt: ZkfFormat, sign: int, value: Fraction) -> int:
    if value <= 0:
        return _zero(fmt)

    exp_unbiased = _floor_log2_fraction(value)
    if exp_unbiased < fmt.min_exp_unbiased:
        return _normal(fmt, sign, 1, 0) if value >= _pow2_fraction(fmt.min_exp_unbiased - 1) else _zero(fmt)

    scaled = value / _pow2_fraction(exp_unbiased) * (1 << fmt.wfrac)
    quotient = scaled.numerator // scaled.denominator
    remainder = scaled.numerator % scaled.denominator

    increment = (2 * remainder) > scaled.denominator
    increment = increment or ((2 * remainder) == scaled.denominator and (quotient & 1) != 0)
    if increment:
        quotient += 1

    if quotient >= (1 << fmt.wman):
        quotient >>= 1
        exp_unbiased += 1

    if exp_unbiased > fmt.max_exp_unbiased:
        return _canonical_inf(fmt, sign)

    return _normal(fmt, sign, exp_unbiased + fmt.bias, quotient & fmt.frac_mask)


def _pack_reference(
    fmt: ZkfFormat,
    sign: int,
    force_zero: int,
    force_inf: int,
    exp_unbiased: int,
    significand_value: int,
    guard: int,
    round_bit: int,
    sticky: int,
) -> int:
    exp_biased = exp_unbiased + fmt.bias
    exp_underflow_zero = exp_unbiased < (fmt.min_exp_unbiased - 1)
    exp_one_below_min = exp_unbiased == (fmt.min_exp_unbiased - 1)
    exp_overflow = exp_unbiased > fmt.max_exp_unbiased

    round_increment = bool(guard and (round_bit or sticky or (significand_value & 1)))
    rounded_ext = (significand_value & _mask(fmt.wman)) + (1 if round_increment else 0)
    round_carry = (rounded_ext >> fmt.wman) & 1
    rounded_significand = (rounded_ext >> 1) if round_carry else (rounded_ext & _mask(fmt.wman))
    exp_round_overflow = (exp_biased == fmt.exp_max_finite) and bool(round_carry)
    infinity = bool(force_inf or exp_overflow or exp_round_overflow)

    result_zero = bool(force_zero or ((not force_inf) and exp_underflow_zero))
    result_infinity = (not result_zero) and infinity
    result_min_normal = (not result_zero) and (not result_infinity) and (not force_inf) and exp_one_below_min

    if result_zero:
        return _zero(fmt)
    if result_infinity:
        return _canonical_inf(fmt, sign)
    if result_min_normal:
        return _normal(fmt, sign, 1, 0)

    exp_rounded = (exp_biased + round_carry) & _mask(fmt.wexp)
    return _pack_bits(fmt, sign, exp_rounded, rounded_significand & fmt.frac_mask)


def _sticky_below(value: int, high_bit: int) -> int:
    if high_bit < 0:
        return 0
    return 1 if (value & _mask(high_bit + 1)) != 0 else 0


def _canonicalize_special(fmt: ZkfFormat, bits: int) -> int:
    bits &= _mask(fmt.wfull)
    exp = (bits >> fmt.wfrac) & fmt.exp_inf
    if exp == 0:
        return _zero(fmt)
    if exp == fmt.exp_inf:
        return _canonical_inf(fmt, (bits >> fmt.sign_shift) & 1)
    return bits


def _ordered_key(fmt: ZkfFormat, bits: int) -> int:
    canonical = _canonicalize_special(fmt, bits)
    sign = (canonical >> fmt.sign_shift) & 1
    return (~canonical & _mask(fmt.wfull)) if sign else (canonical | (1 << fmt.sign_shift))


def _signed_int_min(wint: int) -> int:
    if wint < 2:
        raise ValueError(f"wint must be at least 2, got {wint}")
    return -(1 << (wint - 1))


def _signed_int_max(wint: int) -> int:
    if wint < 2:
        raise ValueError(f"wint must be at least 2, got {wint}")
    return (1 << (wint - 1)) - 1


def _round_fraction_to_int_ties_even(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    frac_part = value - Fraction(floor, 1)
    half = Fraction(1, 2)
    if frac_part < half:
        return floor
    if frac_part > half:
        return floor + 1
    return floor if (floor % 2 == 0) else floor + 1


@enum.unique
class _RoundMode(enum.IntEnum):
    """Private round-to-integer modes; the integer values match zkf/rtl/zkf_round.v and its 2-bit round_mode port."""

    NEAREST_EVEN = 0  # round to nearest integer, ties to even (the IEEE default)
    FLOOR = 1  # round toward -inf
    CEIL = 2  # round toward +inf
    TRUNC = 3  # round toward zero (truncate)


def _round_signed_fraction_to_int(value: Fraction, mode: int) -> int:
    """Round an exact signed value to an integer according to the selected zkf_round mode."""
    if mode == _RoundMode.NEAREST_EVEN:
        return _round_fraction_to_int_ties_even(value)  # floor-based helper is already symmetric for negatives
    if mode == _RoundMode.FLOOR:
        return math.floor(value)
    if mode == _RoundMode.CEIL:
        return math.ceil(value)
    if mode == _RoundMode.TRUNC:
        return math.trunc(value)
    raise ValueError(f"invalid round mode: {mode}")


@functools.cache
def _trans_specs() -> dict:
    from ._tables import trans

    return trans.SPECS


def _trans_spec(func: str, wman: int) -> dict:
    try:
        return _trans_specs()[(func, wman)]
    except KeyError:
        raise KeyError(f"no {func} table for WMAN={wman}; run zkf_transcendental.py --emit")


def _trans_sqrt2_threshold(wfrac: int) -> int:
    """
    Integer significand threshold for the log2 symmetric-reduction re-center test m >= sqrt(2) (m = sig/2^WFRAC,
    sig the WMAN-bit significand): re-center iff sig >= THR with THR = round(sqrt(2) * 2**WFRAC). Computed
    exactly with integer isqrt (round-to-nearest), and MUST equal the generator's log2_sqrt2_threshold and the phase-2
    RTL constant. round(sqrt(S)) for S = 2**(2*WFRAC+1) is (floor(sqrt(4*S)) + 1) // 2 = (isqrt(4*S) + 1) // 2.
    """
    return (math.isqrt(1 << (2 * wfrac + 3)) + 1) // 2  # 4*S = 2**(2*wfrac+3)


def _horner_eval(coeffs_idx: list[int], w: int, rw: int) -> int:
    """Truncating fixed-point Horner, bit-identical to zkf/rtl/_zkf_horner.v and the generator."""
    acc = coeffs_idx[-1]
    for j in range(len(coeffs_idx) - 2, -1, -1):
        acc = coeffs_idx[j] + ((acc * w) >> rw)  # Python >> floors, matching arithmetic >>> on signed
    return acc


@functools.cache
def _trig_specs() -> dict:
    from ._tables import trig

    return trig.SPECS


def _trig_spec(wman: int) -> dict:
    try:
        return _trig_specs()[wman]
    except KeyError:
        raise KeyError(f"no sincos table for WMAN={wman}; run zkf_trig.py --emit")


def _atan2_bypass_shift(fmt: ZkfFormat) -> int:
    from ._tables import trig

    return _trig_spec(fmt.wman)["zf"] - fmt.wman - trig.GUARD_DIV


def _fixed_to_float_ref(fmt: ZkfFormat, sign: int, mag: int, exp_offset: int, wmag: int, *, force_inf: int = 0) -> int:
    """
    Mirror zkf/rtl/_zkf_fixed_to_float.v: normalize the unsigned magnitude, extract G/R/S, exp = exp_offset - count,
    and pack (RTNE). mag == 0 forces +0 unless force_inf is set.
    """
    zero_flag, count, aligned = _normshift_reference(wmag, mag)
    significand_value = (aligned >> (wmag - fmt.wman)) & _mask(fmt.wman)
    guard = (aligned >> (wmag - fmt.wman - 1)) & 1
    round_bit = (aligned >> (wmag - fmt.wman - 2)) & 1
    sticky = 1 if (aligned & _mask(wmag - fmt.wman - 2)) else 0
    exp_unbiased = exp_offset - count
    force_zero = 0 if force_inf else (1 if zero_flag else 0)
    return _pack_reference(fmt, sign, force_zero, force_inf, exp_unbiased, significand_value, guard, round_bit, sticky)


def _cordic_rotate(spec: dict, z0: int, n: int) -> tuple[int, int, int]:
    """
    Fixed-point CORDIC rotation (n ~= WMAN/2 iterations), mirroring zkf/rtl/_zkf_cordic.v. Returns (x_K, y_K, z_K): the
    partially rotated vector at scale 2**-xf and the residual angle z_K at scale 2**-zf (turns); the caller finishes
    with one linear step. Inverse gain is folded into the x seed. Shifts truncate toward -inf (Verilog >>>); the
    N-iteration truncation bias stays below the result ULP.
    """
    kinv, lut = spec["kinv"], spec["lut"]
    x, y, z = kinv, 0, z0
    for i in range(n):
        if z < 0:  # sigma = -1
            x, y, z = x + (y >> i), y - (x >> i), z + lut[i]
        else:  # sigma = +1
            x, y, z = x - (y >> i), y + (x >> i), z - lut[i]
    return x, y, z


def _cordic_vector(spec: dict, x0: int, y0: int, n: int) -> tuple[int, int, int]:
    """
    Fixed-point CORDIC in VECTORING mode (n = N_atan2 iterations), mirroring zkf/rtl/_zkf_cordic.v MODE=1. Drives y -> 0
    and returns (x_K, y_K, z_K): the residual vector at scale 2**-xf (x_K ~= gain*hypot) and the accumulated angle z_K
    at scale 2**-zf (turns, ~= atan2(y0, x0)). Same update as _cordic_rotate but sigma follows sign(y), and there is no
    inverse-gain seed (the gain stays in x_K for the magnitude path). x/y/z wrap to the engine widths (xw/zw); the
    caller pre-scales the seed by 1/4 so that wrap never fires.
    """
    lut, xw, zw = spec["lut"], spec["xw"], spec["zw"]
    x = bits_to_signed(x0 & _mask(xw), xw)
    y = bits_to_signed(y0 & _mask(xw), xw)
    z = 0
    for i in range(n):
        if y >= 0:  # sigma = -1 (drive y down): mirrors neg = ~y[msb]
            nx, ny, nz = x + (y >> i), y - (x >> i), z + lut[i]
        else:  # sigma = +1
            nx, ny, nz = x - (y >> i), y + (x >> i), z - lut[i]
        x = bits_to_signed(nx & _mask(xw), xw)
        y = bits_to_signed(ny & _mask(xw), xw)
        z = bits_to_signed(nz & _mask(zw), zw)
    return x, y, z


def _atan2_turn(fmt: ZkfFormat, sign: int, frac: Fraction) -> int:
    """A signed exact-dyadic turn constant as a ZKF float; the half-turn endpoint canonicalizes to +1/2."""
    if frac == Fraction(1, 2):
        sign = 0
    return _round_fraction_to_zkf(fmt, sign, frac)


def _normshift_reference(width: int, value: int) -> tuple[int, int, int]:
    """
    Reference for _zkf_normshift: returns (zero, count, y). count = (width-1) - leading_one_position, i.e. the
    left-shift that brings the leading 1 to the MSB; y = value << count, the normalized vector. count and y are
    don't-care when zero is asserted.
    """
    value &= _mask(width)
    if value == 0:
        return 1, 0, 0
    count = (width - 1) - (value.bit_length() - 1)
    return 0, count, (value << count) & _mask(width)

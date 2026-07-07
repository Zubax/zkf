from __future__ import annotations

from dataclasses import dataclass
from ._format import OperatorModel, ZkfFormat
from ._reference import trans_spec, trig_spec


@dataclass(frozen=True)
class AbsModel(OperatorModel):
    module = "zkf_abs"

    @property
    def params(self) -> dict[str, int]:
        return {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman}

    @property
    def latency(self) -> int:
        return 0


@dataclass(frozen=True)
class NegModel(AbsModel):
    module = "zkf_neg"


@dataclass(frozen=True)
class IsFiniteModel(AbsModel):
    module = "zkf_is_finite"


@dataclass(frozen=True)
class SaturateModel(AbsModel):
    module = "zkf_saturate"


@dataclass(frozen=True)
class PipeModel(OperatorModel):
    module = "zkf_pipe"
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


@dataclass(frozen=True)
class PackModel(OperatorModel):
    module = "_zkf_pack"
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


@dataclass(frozen=True)
class CmpModel(OperatorModel):
    module = "zkf_cmp"
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


@dataclass(frozen=True)
class SortModel(CmpModel):
    module = "zkf_sort"


@dataclass(frozen=True)
class AddModel(OperatorModel):
    module = "zkf_add"
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


@dataclass(frozen=True)
class AddSubModel(AddModel):
    module = "zkf_addsub"


@dataclass(frozen=True)
class MulModel(OperatorModel):
    module = "zkf_mul"
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


@dataclass(frozen=True)
class FmaModel(OperatorModel):
    module = "zkf_fma"
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


@dataclass(frozen=True)
class MulIlog2Model(OperatorModel):
    module = "zkf_mul_ilog2"
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


@dataclass(frozen=True)
class MulIlog2ConstModel(OperatorModel):
    module = "zkf_mul_ilog2_const"
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


@dataclass(frozen=True)
class DivCoreModel(OperatorModel):
    module = "_zkf_div_core"

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


@dataclass(frozen=True)
class DivModel(OperatorModel):
    module = "zkf_div"
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


@dataclass(frozen=True)
class FromIntModel(OperatorModel):
    module = "zkf_from_int"
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


@dataclass(frozen=True)
class ToIntModel(OperatorModel):
    module = "zkf_to_int"
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


@dataclass(frozen=True)
class ResizeModel(OperatorModel):
    module = "zkf_resize"
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


@dataclass(frozen=True)
class RoundModel(OperatorModel):
    module = "zkf_round"
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


@dataclass(frozen=True)
class Exp2Model(OperatorModel):
    module = "zkf_exp2"
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
        degree = trans_spec("exp2", self.fmt.wman)["d"]
        return (
            self.stage_input
            + self.stage_reduce
            + 4
            + degree * (2 + self.stage_product)
            + self.stage_pack
            + self.stage_output
        )


@dataclass(frozen=True)
class Log2Model(OperatorModel):
    module = "zkf_log2"
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
        spec = trans_spec("log2", self.fmt.wman)
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
        degree = trans_spec("log2", self.fmt.wman)["d"]
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


@dataclass(frozen=True)
class SincosModel(OperatorModel):
    module = "zkf_sincos"
    unroll100: int = 100
    stage_input: int = 0
    stage_product: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        spec = trig_spec(self.fmt.wman)
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
        k = trig_spec(self.fmt.wman)["n_sincos"]
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


@dataclass(frozen=True)
class Atan2Model(OperatorModel):
    module = "zkf_atan2"
    unroll100: int = 100
    stage_input: int = 0
    stage_product: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    wmultiplier: int = 0

    def __post_init__(self) -> None:
        spec = trig_spec(self.fmt.wman)
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
        spec = trig_spec(self.fmt.wman)
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


def _check_int_range(value: int, min: int | None, max: int | None, /, extra: set[int] | None = None) -> None:
    extra = extra or set()
    if not isinstance(value, int):
        raise ValueError(f"Expected an integer, found {value}")
    if (min is not None and value < min) or (max is not None and value > max):
        if value not in extra:
            raise ValueError(f"Value {value} is outside {min}..{max}")

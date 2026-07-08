from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import ClassVar

from ._reference import (
    normal as normal_bits,
    pack_bits,
    pow2_fraction,
    round_fraction_to_zkf,
    signed_int_max,
    signed_int_min,
    zero as zero_bits,
)


@dataclass(frozen=True)
class OperatorModel:
    """
    This is the centerpiece for modeling and instantiating operators.
    Instantiate RTL as:

        f"{model.module} #({model.verilog_params}) {instance_name} ({nets...});"
    """

    fmt: ZkfFormat
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

    @property
    def initiation_interval(self) -> int:
        return 1

    def _params_with_latency(self, params: dict[str, int]) -> dict[str, int]:
        return {**params, "LATENCY": self.latency}


@dataclass(frozen=True, slots=True)
class ZkfFormat:
    """A ZKF format (wexp, wman) plus the factory hub for Zkf values."""

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        if not isinstance(self.wexp, int) or self.wexp < 2:
            raise ValueError(f"Bad exponent: {self.wexp}")
        if not isinstance(self.wman, int) or self.wman < 4:
            raise ValueError(f"Bad mantissa: {self.wman}")

    def model_of(self, name: str) -> Callable[..., OperatorModel]:
        for model_cls in _operator_model_descendants():
            module = getattr(model_cls, "module", None)
            if isinstance(module, str) and module.removeprefix("_").removeprefix("zkf_") == name:
                break
        else:
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
        return pow2_fraction(self.min_exp_unbiased)

    @property
    def max(self) -> Fraction:
        """Largest finite magnitude."""
        return (Fraction(2) - pow2_fraction(-self.wfrac)) * pow2_fraction(self.max_exp_unbiased)

    @property
    def epsilon(self) -> Fraction:
        """Gap between 1.0 and the next representable value above it."""
        return pow2_fraction(-self.wfrac)

    def wrap(self, bits: int) -> Zkf:
        """Wrap raw packed bits (masked to WFULL) as a Zkf."""
        return Zkf(self, bits)

    def zero(self, sign: int = 0) -> Zkf:
        return Zkf(self, pack_bits(self, sign, 0, 0))

    def inf(self, sign: int = 0) -> Zkf:
        return Zkf(self, pack_bits(self, sign, self.exp_inf, 0))

    def normal(self, sign: int, exp: int, frac: int) -> Zkf:
        return Zkf(self, normal_bits(self, sign, exp, frac))

    def pack(self, sign: int, exp: int, frac: int) -> Zkf:
        """
        Construct a value directly from raw field values (each masked to width); no canonicalization or range
        checks. For non-canonical test vectors; prefer zero/inf/normal/encode.
        """
        return Zkf(self, pack_bits(self, sign, exp, frac))

    def from_int(self, wint: int, value: int) -> Zkf:
        """Convert a signed wint-bit integer to the nearest ZKF value (RNTE)."""
        if not signed_int_min(wint) <= value <= signed_int_max(wint):
            raise ValueError(f"value={value} out of signed {wint}-bit range")
        if value == 0:
            return Zkf(self, zero_bits(self))
        return Zkf(self, round_fraction_to_zkf(self, int(value < 0), Fraction(abs(value))))

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
        return Zkf(self, round_fraction_to_zkf(self, int(value < 0), abs(value)))


def _operator_model_descendants(cls: type[OperatorModel] = OperatorModel):
    for sub in cls.__subclasses__():
        yield sub
        yield from _operator_model_descendants(sub)


from ._value import Zkf

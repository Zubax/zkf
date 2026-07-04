#!/usr/bin/env python3
"""
Bit/integer utilities for the ZKF testbenches, kept out of the vendorable zkf package so its public API stays
minimal. The package keeps its own private copies internally; this is an independent copy for the benches, so drift is
caught by the RTL comparisons.
"""

from __future__ import annotations

from fractions import Fraction


def hex_bits(value: int, width: int) -> str:
    """Format the low width bits of value as a zero-padded 0x... hex string."""
    return f"0x{value & ((1 << width) - 1):0{(width + 3) // 4}x}"


def mask(width: int) -> int:
    """Low width-bit all-ones mask."""
    return (1 << width) - 1


def signed_to_bits(value: int, width: int) -> int:
    """Two's-complement encoding of value into width bits."""
    return value & mask(width)


def signed_range(width: int) -> range:
    """The full inclusive range of signed width-bit integers."""
    return range(-(1 << (width - 1)), 1 << (width - 1))


def signed_int_min(wint: int) -> int:
    """Most negative signed wint-bit integer."""
    if wint < 2:
        raise ValueError(f"wint must be at least 2, got {wint}")
    return -(1 << (wint - 1))


def signed_int_max(wint: int) -> int:
    """Most positive signed wint-bit integer."""
    if wint < 2:
        raise ValueError(f"wint must be at least 2, got {wint}")
    return (1 << (wint - 1)) - 1


def pow2_fraction(exp: int) -> Fraction:
    """Exact 2**exp as a fractions.Fraction for any signed exp."""
    return Fraction(1 << exp, 1) if exp >= 0 else Fraction(1, 1 << -exp)

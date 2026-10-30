"""
Usage:
    python -m zkf WEXP WMAN [VALUE]
"""

import sys
from fractions import Fraction

import mpmath

from ._value import ZkfFormat, bits_to_signed


def main(argv: list[str]) -> None:
    usage = f"usage: {argv[0]} WEXP WMAN [VALUE]"
    if len(argv) not in (3, 4):
        raise SystemExit(usage)

    try:
        fmt = ZkfFormat(int(argv[1]), int(argv[2]))
    except ValueError as exc:
        raise SystemExit(f"{usage}  ({exc})")
    print(f"WEXP={fmt.wexp} WMAN={fmt.wman} WFRAC={fmt.wfrac} WFULL={fmt.wfull} BIAS={fmt.bias}")
    lowest, top, wfrac = fmt.min_exp_unbiased, fmt.max_exp_unbiased, fmt.wfrac
    print(f"lowest     = 2**{lowest} ≈ {approx(1, lowest)}")
    print(f"max        = (2 - 2**-{wfrac}) * 2**{top} ≈ {approx((1 << fmt.wman) - 1, top - wfrac)}")
    print(f"ε          = 2**-{wfrac} ≈ {approx(1, -wfrac)}")

    bit_diagram = "s" + "e" * fmt.wexp + "f" * fmt.wfrac
    print(bit_diagram)
    for place in range(len(str(fmt.wfull - 1))):
        print("".join(str(bit // 10**place % 10) for bit in reversed(range(fmt.wfull))))

    if len(argv) > 3:
        try:
            z = fmt.encode(number(argv[3]))
        except ValueError as exc:
            raise SystemExit(f"{usage}  ({exc})")
        print(f"Value {argv[3]} represented in this format:")
        print(f"signed decimal: {bits_to_signed(z.bits, fmt.wfull):+d}")
        print(f"twos-complement: {z.bits:0{fmt.wfull}b}")
        print(f"                 {bit_diagram}")


def number(text: str) -> Fraction | float:
    """`text` exactly, or as a float where it is an infinity or a NaN."""
    try:
        return Fraction(text)
    except ValueError:
        return float(text)


def approx(significand: int, exp: int) -> str:
    """`significand * 2**exp` to four digits, at any exponent."""
    return mpmath.nstr(mpmath.ldexp(significand, exp), 4, strip_zeros=False, min_fixed=0, max_fixed=0)


if __name__ == "__main__":
    main(sys.argv)

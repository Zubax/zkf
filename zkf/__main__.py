"""
Usage:
    python -m zkf WEXP WMAN [VALUE]
"""

import sys

from ._core import ZkfFormat, bits_to_signed


def main(argv: list[str]) -> None:
    usage = f"usage: {argv[0]} WEXP WMAN [VALUE]"
    if len(argv) not in (3, 4):
        raise SystemExit(usage)

    try:
        fmt = ZkfFormat(int(argv[1]), int(argv[2]))
    except ValueError as exc:
        raise SystemExit(f"{usage}  ({exc})")
    print(f"WEXP={fmt.wexp} WMAN={fmt.wman} WFRAC={fmt.wfrac} WFULL={fmt.wfull} BIAS={fmt.bias}")
    print(f"lowest     = {fmt.lowest} ≈ {float(fmt.lowest):.3e}")
    print(f"max        = {fmt.max} ≈ {float(fmt.max):.3e}")
    print(f"ε          = {fmt.epsilon} ≈ {float(fmt.epsilon):.3e}")

    bit_diagram = "s" + "e" * fmt.wexp + "f" * fmt.wfrac
    print(bit_diagram)
    print(("0123456789" * ((fmt.wfull + 10) // 10))[:fmt.wfull][::-1])
    print("".join((f"{x}" * 10) for x in range(10))[:fmt.wfull][::-1])

    if len(argv) > 3:
        value = float(argv[3])
        z = fmt.encode(value)
        print(f"Value {value} represented in this format:")
        print(f"signed decimal: {bits_to_signed(z.bits, fmt.wfull):+d}")
        print(f"twos-complement: {z.bits:0{fmt.wfull}b}")
        print(f"                 {bit_diagram}")


if __name__ == "__main__":
    main(sys.argv)

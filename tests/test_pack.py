import numpy as np
import pytest

from _bench import Case, run_pipeline
from _operands import SEED, random_pack_mag_scale, signed_range
from zkf import Zkf, ZkfFormat
from zkf._pack import Pack, pack_reference

type Inputs = tuple[int, int, int, int, int, int, int, int]
"""sign, force_zero, force_inf, exp_unbiased, significand, guard, round, sticky"""


def from_magnitude(fmt: ZkfFormat, sign: int, mag: int, scale: int) -> Inputs:
    """The packer's inputs for the value `(-1)**sign * mag * 2**scale`."""
    if mag == 0:
        return sign, 1, 0, scale, 0, 0, 0, 0
    top = mag.bit_length() - 1
    aligned = (mag << (fmt.wman + 1)) >> top
    tail = top - fmt.wman - 1
    sticky = int(tail > 0 and mag & ((1 << tail) - 1) != 0)
    return sign, 0, 0, scale + top, (aligned >> 2) & ((1 << fmt.wman) - 1), (aligned >> 1) & 1, aligned & 1, sticky


def directed(fmt: ZkfFormat) -> list[Inputs]:
    low, high = fmt.min_exp_unbiased, fmt.max_exp_unbiased
    one, full = 1 << fmt.wfrac, ((1 << fmt.wman) - 1)
    return [
        (1, 1, 1, high + 2, full, 1, 1, 1),
        (1, 0, 1, low - 3, one, 0, 0, 0),
        (0, 0, 0, low - 2, full, 1, 1, 1),
        (0, 0, 0, low - 1, one, 0, 0, 0),
        (0, 0, 0, low - 1, one + (one >> 1), 0, 0, 0),
        (1, 0, 0, low - 1, full - 1, 1, 0, 0),
        (1, 0, 0, low, one, 0, 0, 0),
        (0, 0, 0, 0, one, 1, 0, 0),
        (0, 0, 0, 0, one + 1, 1, 0, 0),
        (0, 0, 0, 0, one + 1, 0, 1, 1),
        (0, 0, 0, 0, one + 1, 1, 0, 1),
        (0, 0, 0, high, full, 0, 0, 0),
        (0, 0, 0, high, full, 1, 0, 0),
        (1, 0, 0, high + 1, one, 0, 0, 0),
    ]


# sign, magnitude, scale and the result, worked by hand at 5/8.
MANUAL = [
    (1, 0, 31, 0x0000),
    (0, 511, -24, 0x0000),
    (0, 1, -15, 0x0080),
    (0, 255, -22, 0x0080),
    (0, 511, -23, 0x0080),
    (1, 511, -23, 0x1080),
    (0, 1, -14, 0x0080),
    (1, 1, -14, 0x1080),
    (0, 1, 0, 0x0780),
    (1, 1, 0, 0x1780),
    (0, 3, -1, 0x07C0),
    (0, 1, 1, 0x0800),
    (1, 1, 1, 0x1800),
    (0, 255, -7, 0x07FF),
    (0, 513, -9, 0x0780),
    (0, 514, -9, 0x0780),
    (0, 515, -9, 0x0781),
    (0, 518, -9, 0x0782),
    (0, 1021, -9, 0x07FF),
    (0, 511, -8, 0x0800),
    (1, 511, -8, 0x1800),
    (0, 0x8000, 0, 0x0F00),
    (0, 0xFFFF, 0, 0x0F80),
    (0, 255, 8, 0x0F7F),
    (0, 1021, 6, 0x0F7F),
    (0, 511, 7, 0x0F80),
    (1, 511, 7, 0x1F80),
    (0, 1, 16, 0x0F80),
]


def test_reference_by_hand() -> None:
    fmt = ZkfFormat(5, 8)
    for sign, mag, scale, want in MANUAL:
        assert pack_reference(fmt, *from_magnitude(fmt, sign, mag, scale)) == want


def exhaustive(fmt: ZkfFormat, wexp_unbiased: int, exp_is_biased: bool) -> list[Inputs]:
    """Over the exponent field, which is the biased exponent where the packer takes that."""
    return [
        (sign, force_zero, force_inf, field - fmt.bias * exp_is_biased, significand, grs >> 2, (grs >> 1) & 1, grs & 1)
        for sign in (0, 1)
        for force_zero in (0, 1)
        for force_inf in (0, 1)
        for field in signed_range(wexp_unbiased)
        for significand in range(1 << fmt.wman)
        for grs in range(8)
    ]


def run(wexp: int, wman: int, wexp_unbiased: int, count: int = 0, **options: int) -> None:
    """`count` 0 is every input; otherwise the directed ones made up to `count` with random ones."""
    fmt = ZkfFormat(wexp, wman)
    exp_is_biased = bool(options.get("exp_is_biased"))
    assume_no_overflow = bool(options.get("assume_no_overflow"))
    saturate_round_carry = bool(options.get("saturate_round_carry"))
    if count:
        inputs = directed(fmt)
        if (wexp, wman) == (5, 8):
            inputs += [from_magnitude(fmt, sign, mag, scale) for sign, mag, scale, _ in MANUAL]
        rng = np.random.default_rng(SEED)
        seen = set(inputs)
        while len(inputs) < count:
            drawn = from_magnitude(fmt, *random_pack_mag_scale(fmt, rng))
            if drawn not in seen:
                seen.add(drawn)
                inputs.append(drawn)
        # An exponent that the port does not hold is no input of this configuration.
        inputs = [i for i in inputs if i[3] + fmt.bias * exp_is_biased in signed_range(wexp_unbiased)]
    else:
        inputs = exhaustive(fmt, wexp_unbiased, exp_is_biased)
    if assume_no_overflow:
        # The caller's promise: an exponent above the finite range comes with a force flag.
        inputs = [i for i in inputs if i[1] or i[2] or i[3] <= fmt.max_exp_unbiased]
    names = ("sign", "force_zero", "force_inf", "exp_unbiased", "significand", "guard", "round", "sticky")
    cases = []
    for i in inputs:
        want = pack_reference(fmt, *i, saturate_round_carry=saturate_round_carry)
        driven = dict(zip(names, i)) | {"exp_unbiased": i[3] + fmt.bias * exp_is_biased}
        cases.append(Case(driven, {"y": Zkf(fmt, want)}))
    dut = Pack(
        fmt,
        wexp_unbiased=wexp_unbiased,
        exp_is_biased=exp_is_biased,
        assume_no_overflow=assume_no_overflow,
        saturate_round_carry=saturate_round_carry,
        stage_input=options.get("stage_input", 0),
        stage_output=options.get("stage_output", 0),
    )
    run_pipeline(dut, options.get("stage_input", 0) + options.get("stage_output", 0), cases)


# wexp, wman, wexp_unbiased (WEXP at the least), count
ROWS = [
    (2, 4, 4, 0),
    (3, 4, 5, 0),
    (4, 5, 4, 0),
    (5, 8, 8, 768),
    (8, 24, 12, 2048),
    (6, 18, 8, 768),
    (8, 36, 12, 768),
    (11, 53, 15, 768),
]


@pytest.mark.parametrize("row", ROWS, ids=str)
def test_pack(row: tuple[int, int, int, int]) -> None:
    run(*row)


@pytest.mark.parametrize("stage_input", [0, 1])
@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize(
    "options", [{}, {"exp_is_biased": 1}, {"saturate_round_carry": 1}, {"assume_no_overflow": 1}], ids=str
)
def test_pack_stages(stage_input: int, stage_output: int, options: dict[str, int]) -> None:
    run(3, 4, 5, stage_input=stage_input, stage_output=stage_output, **options)


@pytest.mark.deep
@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize(
    "row", [(2, 5, 3, 0), (2, 5, 5, 0), (3, 5, 5, 0), (4, 5, 8, 768), (6, 17, 10, 1024), (4, 4, 8, 512)], ids=str
)
def test_pack_deep(row: tuple[int, int, int, int], stage_output: int) -> None:
    run(*row, stage_output=stage_output)
    if not row[3] and row[2] > row[0]:
        run(*row, stage_output=stage_output, exp_is_biased=1)


@pytest.mark.deep
@pytest.mark.parametrize("row", [(4, 5, 6), (4, 5, 7), (3, 6, 5), (4, 4, 6)], ids=str)
def test_pack_deep_exhaustive(row: tuple[int, int, int]) -> None:
    run(*row)


@pytest.mark.deep
@pytest.mark.parametrize(
    "options",
    [
        {"stage_output": 1},
        {"exp_is_biased": 1},
        {"assume_no_overflow": 1},
        {"saturate_round_carry": 1},
        {"saturate_round_carry": 1, "assume_no_overflow": 1},
        {"saturate_round_carry": 1, "exp_is_biased": 1},
    ],
    ids=str,
)
def test_pack_deep_options(options: dict[str, int]) -> None:
    run(4, 5, 6, **options)

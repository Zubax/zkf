from typing import Any

import numpy as np
import pytest

from _bench import Case, run_pipeline
from _catalog import CATALOG
from _operands import (
    SEED,
    directed_numbers,
    random_inf,
    random_normal,
    random_normal_near,
    random_operand,
    random_zero,
)
from zkf import Add, Fma, Zkf, ZkfFormat
from zkf._pmul import counts

# Operands at 6/18 where one rounding differs from that of a multiplier and an adder in a row, and at 4/30 where a
# deep cancellation leaves an exponent far below the product's range.
WITNESSES = {
    (6, 18): [(0x128B2F, 0xD23F08, 0x892F90), (0x922766, 0x4EF8AA, 0x8F6D05), (0x1A61DB, 0x94E3BF, 0x923A73)],
    (4, 30): [(0x08EA99F89, 0x07AEC4FCE, 0x22AF60440), (0x28DD00C89, 0x08A4ABE3C, 0x03C8C11EA)],
}


def directed(model: Fma) -> list[tuple[int, int, int]]:
    fmt = model.fmt
    v = directed_numbers(fmt)
    one, zero, top = v["one"], v["zero"], v["max_finite"]
    out = [
        (one, one, zero),
        (v["two"], one, one),
        (one, one, one),
        (one, one, v["minus_one"]),
        (v["half"], v["half"], zero),
        (top, one, one),
        (top, top, zero),
        (top, top, v["neg_max_finite"]),
        (v["neg_max_finite"], top, zero),
        (v["min_normal"], v["half"], zero),
        (v["min_normal"], v["min_normal"], v["min_normal"]),
        # A zero product, that of a zero and an infinity too: the result is `c`.
        (zero, one, one),
        (zero, v["pos_inf"], v["two"]),
        (v["neg_zero"], v["noncanonical_pos_inf"], one),
        (one, zero, v["minus_one"]),
        (zero, zero, zero),
        (v["pos_inf"], one, one),
        (v["pos_inf"], one, v["neg_inf"]),
        (v["neg_inf"], one, v["pos_inf"]),
        (v["pos_inf"], v["minus_one"], top),
        (v["two"], v["two"], v["noncanonical_neg_inf"]),
        (v["pos_inf"], v["pos_inf"], v["neg_inf"]),
        (v["pos_inf"], zero, v["pos_inf"]),
        (v["minus_one"], v["minus_one"], zero),
        (one, v["minus_one"], zero),
        (v["one_and_half"], one, v["one_and_quarter"]),
        (v["one_and_three_quarters"], v["one_and_three_quarters"], zero),
    ]
    # `c` the negated rounded product: what is left is the rounding error, deep in the low half of the product.
    for a, b in (
        (v["one_and_half"], v["one_and_quarter"]),
        (v["one_and_three_quarters"], v["one_and_three_quarters"]),
        (top, v["one_and_half"]),
        (v["min_normal"], v["one_and_three_quarters"]),
    ):
        product = model.evaluate(Zkf(fmt, a), Zkf(fmt, b), Zkf(fmt, 0))
        out.append((a, b, product.bits ^ (1 << fmt.sign_shift)))
    # Every distance between the exponents of the product and of `c`, in both signs.
    high = min(fmt.exp_max_finite, fmt.bias + 4)
    for apart in range(min(fmt.wman + 3, fmt.exp_max_finite)):
        out += [(one, one, fmt.pack(sign, max(1, high - apart), fmt.frac_mask).bits) for sign in (0, 1)]
    return out + WITNESSES.get((fmt.wexp, fmt.wman), [])


def drawn(model: Fma, rng: np.random.Generator) -> tuple[int, int, int]:
    fmt = model.fmt
    top, ones = fmt.exp_max_finite, fmt.frac_mask
    mode = int(rng.integers(0, 12))
    if mode < 3:
        operands = [random_operand(fmt, rng) for _ in range(3)]
        operands[mode] = random_zero(fmt, rng)
        return operands[0], operands[1], operands[2]
    if mode == 3:
        return random_inf(fmt, rng), random_operand(fmt, rng), random_operand(fmt, rng)
    if mode == 4:
        return random_operand(fmt, rng), random_operand(fmt, rng), random_inf(fmt, rng)
    if mode == 5:
        return random_inf(fmt, rng), random_operand(fmt, rng), random_inf(fmt, rng)
    if mode in (6, 7):
        a, b = random_normal(fmt, rng), random_normal(fmt, rng)
        if mode == 6:  # `c` the product or its negation: the closest cancellation
            product = model.evaluate(Zkf(fmt, a), Zkf(fmt, b), Zkf(fmt, 0)).bits
            return a, b, product ^ (int(rng.integers(0, 2)) << fmt.sign_shift)
        exps = ((a >> fmt.wfrac) & fmt.exp_inf) + ((b >> fmt.wfrac) & fmt.exp_inf) - fmt.bias
        return a, b, fmt.pack(int(rng.integers(0, 2)), int(np.clip(exps, 1, top)), int(rng.integers(0, ones + 1))).bits
    if mode == 8:
        largest = [random_normal_near(fmt, rng, [top], [ones]) for _ in range(2)]
        return largest[0], largest[1], random_operand(fmt, rng)
    if mode == 9:
        least = [random_normal_near(fmt, rng, [1, 2], [0, 1]) for _ in range(3)]
        return least[0], least[1], least[2]
    return random_operand(fmt, rng), random_operand(fmt, rng), random_operand(fmt, rng)


def run(wexp: int, wman: int, count: int, wmultiplier: int = 18, **knobs: Any) -> None:
    """The product takes the lowest count that it takes by default."""
    knobs = {"stage_core_mul": counts(wman, wman, signed=False, wmultiplier=wmultiplier).start} | knobs
    simulate(Fma(ZkfFormat(wexp, wman), wmultiplier=wmultiplier, **knobs), count)


def simulate(model: Fma, count: int) -> None:
    """`count` 0 is every triple; otherwise the directed ones made up to `count` with random ones."""
    fmt = model.fmt
    every = range(1 << fmt.wfull)
    triples = (
        (directed(model) if fmt.wexp >= 3 else [])
        if count
        else [(a, b, c) for a in every for b in every for c in every]
    )
    triples = list(dict.fromkeys(triples))
    rng = np.random.default_rng(SEED)
    seen = set(triples)
    while len(triples) < count:
        triple = drawn(model, rng)
        if triple not in seen:
            seen.add(triple)
            triples.append(triple)
    cases = []
    for a, b, c in triples:
        cases.append(Case({"a": a, "b": b, "c": c}, {"y": model.evaluate(Zkf(fmt, a), Zkf(fmt, b), Zkf(fmt, c))}))
    run_pipeline(model.component(), model.timing.latency, cases)


# wexp, wman, count
FORMATS = [
    (3, 4, 2048),
    (3, 5, 768),
    (4, 6, 768),
    (5, 11, 768),
    (6, 18, 1024),
    (8, 24, 1024),
    (8, 36, 1024),
    (11, 53, 512),
    (4, 30, 256),
]


@pytest.mark.parametrize("row", FORMATS, ids=str)
def test_formats(row: tuple[int, int, int]) -> None:
    run(*row)


@pytest.mark.parametrize(
    "knobs",
    [
        {"stage_input": 1},
        {"stage_core_mul": 1},
        {"stage_core_add": Add.CoreStages.SUM},
        {"stage_output": 1},
        {
            "stage_input": 2,
            "stage_core_mul": 2,
            "stage_core_add": Add.CoreStages.SUM | Add.CoreStages.OPERANDS,
            "stage_output": 2,
        },
        {"stage_core_mul": 3, "stage_core_add": ~Add.CoreStages(0)},
    ],
    ids=str,
)
def test_stages(knobs: dict[str, Any]) -> None:
    run(4, 6, 256, **knobs)


# Each register of the sum alone and all of them.
@pytest.mark.parametrize("stage_core", [*Add.CoreStages, ~Add.CoreStages(0)], ids=str)
def test_registers(stage_core: Add.CoreStages) -> None:
    run(4, 7, 256, stage_core_add=stage_core)


# The product cut in two and in three by 8-bit multipliers, at the ends of the counts that each takes, and every stage
# at once.
@pytest.mark.parametrize(
    "row",
    [
        (4, 12, 256, 8, {"stage_core_mul": 2}),
        (4, 12, 256, 8, {"stage_core_mul": 3}),
        (4, 17, 256, 8, {"stage_core_mul": 3}),
        (4, 17, 256, 8, {"stage_core_mul": 4}),
        (6, 18, 384, 18, {"stage_core_mul": 2, "stage_core_add": ~Add.CoreStages(0)}),
        (
            6,
            18,
            384,
            18,
            {"stage_input": 1, "stage_core_mul": 3, "stage_core_add": ~Add.CoreStages(0), "stage_output": 1},
        ),
    ],
    ids=str,
)
def test_rows(row: tuple[int, int, int, int, dict[str, Any]]) -> None:
    run(row[0], row[1], row[2], row[3], **row[4])


@pytest.mark.parametrize("model", [pytest.param(m, id=n) for n, m in CATALOG.items() if isinstance(m, Fma)])
def test_catalog(model: Fma) -> None:
    simulate(model, 384)


def test_every_triple() -> None:
    run(2, 4, 0)


def test_refusals() -> None:
    fmt = ZkfFormat(4, 6)
    refused: list[dict[str, Any]] = [
        {"stage_core_mul": 4},
        {"stage_core_mul": -1},
    ]
    for knobs in refused:
        with pytest.raises(ValueError):
            Fma(fmt, **({"wmultiplier": 18} | knobs))
    # A cut product without its capture or past its positions, and a grid with a side longer than 2 summed with its
    # multipliers.
    for wman, stage_core_mul in ((12, 1), (12, 5), (17, 2)):
        with pytest.raises(ValueError):
            Fma(ZkfFormat(4, wman), wmultiplier=8, stage_core_mul=stage_core_mul)


@pytest.mark.deep
@pytest.mark.parametrize(
    "row", [(4, 5, 512), (3, 6, 512), (5, 4, 512), (3, 7, 512), (6, 17, 768), (8, 23, 512)], ids=str
)
def test_deep(row: tuple[int, int, int]) -> None:
    run(*row)


@pytest.mark.deep
@pytest.mark.parametrize("stages", [stages for stages in range(16) if stages & (stages - 1)])  # two knobs or more
def test_deep_stages(stages: int) -> None:
    core = (
        Add.CoreStages.OPERANDS
        | Add.CoreStages.ALIGNED
        | Add.CoreStages.SUM
        | Add.CoreStages.NORMALIZE
        | Add.CoreStages.NORMALIZED
    )
    values: dict[str, Any] = {"stage_core_mul": 2, "stage_input": 1, "stage_core_add": core, "stage_output": 1}
    run(4, 6, 256, **{name: value for bit, (name, value) in enumerate(values.items()) if (stages >> bit) & 1})


@pytest.mark.deep
@pytest.mark.parametrize(
    "row",
    [
        (7, 12, 512, {}),
        (9, 24, 512, {}),
        (8, 36, 768, {"stage_core_mul": 2, "stage_core_add": ~Add.CoreStages(0)}),
        (8, 36, 768, {"stage_core_mul": 3, "stage_core_add": ~Add.CoreStages(0)}),
        (
            2,
            4,
            0,
            {
                "stage_input": 1,
                "stage_core_mul": 2,
                "stage_core_add": ~Add.CoreStages(0),
                "stage_output": 1,
            },
        ),
        (3, 4, 4096, {"stage_core_mul": 2, "stage_core_add": ~Add.CoreStages(0), "stage_output": 1}),
        (
            6,
            18,
            1024,
            {
                "stage_core_mul": 2,
                "stage_core_add": Add.CoreStages.SUM | Add.CoreStages.NORMALIZE | Add.CoreStages.PACK,
            },
        ),
    ],
    ids=str,
)
def test_deep_rows(row: tuple[int, int, int, dict[str, Any]]) -> None:
    run(row[0], row[1], row[2], **row[3])

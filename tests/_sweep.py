"""The configurations of an operator that a sweep over its knobs visits."""

import enum
from collections.abc import Mapping
from dataclasses import fields
from itertools import combinations

import zkf._cordic
from zkf import Block, Cordic, Operator, Resize, RoundMode, ZkfFormat
from zkf._pmul import levels
from zkf._trans import _KERNEL

WIDTHS = {"fmt", "wmultiplier", "w", "wk", "wint", "fmt_in"}
"""What a sweep is given and does not vary."""
# Every legal value of a knob, or the ends and one inside of an unbounded one.
KNOBS: Mapping[str, tuple[object, ...]] = {
    "stage_input": (0, 1, 3),
    "stage_output": (0, 1, 3),
    "stage_core_mul_level": tuple(range(16)),
    "stage_core_kernel_level": tuple(range(len(_KERNEL) + 2)),
    "rotation_seed": (0, 1, 2),
    "n": (0, 1, 7),
    "fast": (False, True),
    "_vectoring": (None, False, True),
    "_length": (False, True),
    "_op_sqrt": (None, False, True),
    "_round_mode": (None, *RoundMode),
}


def make(cls: type[Block], fmt: ZkfFormat, **knobs: object) -> Block:
    """A configuration of the class, in the format if it has one; a resize is from that format unless told."""
    if cls is Resize:
        knobs = {"fmt_in": fmt} | knobs
    return cls(fmt, **knobs) if issubclass(cls, Operator) else cls(**knobs)


def admits(model: Block) -> bool:
    """
    Whether Cordic's modes take its knobs, which only its timing and elaboration refuse: a build that rotates takes the
    levels of the rotation's own product, a vectoring alone those of the length's.
    """
    if not isinstance(model, Cordic):
        return True
    mode = zkf._cordic._mode(model)
    if (mode != 1 and model.stage_core_mul_level < 1) or (mode and model.fmt.wexp < 5):
        return False
    served = zkf._cordic._served(model)
    if served is None:
        return True
    wa, wb = zkf._cordic._geometry(model.fmt.wman).operands(int(served == 1))
    return model.stage_core_mul_level < len(levels(wa, wb, signed=served != 1, wmultiplier=model.wmultiplier))


def configurations(cls: type[Block], wmultiplier: int = 18, every_pair: bool = False) -> list[dict[str, object]]:
    """
    Every value of every knob with the others at their defaults, and if told, every pair of values of two knobs. A set
    of registers takes each member and all of them; if told, also every pair of its members, with the others at
    their defaults.
    """
    knobs = [field.name for field in fields(cls) if field.compare]
    fixed: dict[str, object] = {
        name: width for name, width in (("wmultiplier", wmultiplier), ("w", 8)) if name in knobs
    }
    values: dict[str, tuple[object, ...]] = dict(KNOBS)
    within: list[dict[str, object]] = []
    for field in fields(cls):
        if isinstance(field.default, enum.Flag):
            members = list(type(field.default))
            values[field.name] = (*members, ~type(field.default)(0))
            if every_pair:
                within += [{**fixed, field.name: a | b} for a, b in combinations(members, 2)]
    unswept = set(knobs) - set(values) - WIDTHS
    assert not unswept, f"{cls.__name__}: {unswept} are in no sweep"
    names = [name for name in knobs if name in values]
    out = [dict(fixed)]
    for name in names:
        out += [{**fixed, name: value} for value in values[name]]
    if every_pair:
        for a, b in combinations(names, 2):
            out += [{**fixed, a: x, b: y} for x in values[a] for y in values[b]]
    return out + within

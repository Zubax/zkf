"""The configuration base classes, the float port layout, and the register-stage helpers."""

import enum
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from amaranth.back import verilog
from amaranth.hdl import Cat, Const, Module, Shape, ShapeLike, Signal, Value, ValueLike
from amaranth.lib import data, wiring

from ._value import Zkf, ZkfFormat

type Attrs = Mapping[str, int | str]
"""Synthesis attributes: name to value."""


class FloatView(data.View):
    """
    A ZKF value on a float port. Every read of `is_zero`, `is_inf` or `significand` builds logic: take it into a signal
    if it is used more than once.
    """

    frac: Value
    exp: Value
    sign: Value

    @property
    def fmt(self) -> ZkfFormat:
        layout = self.shape()
        assert isinstance(layout, FloatLayout)
        return layout.fmt

    @property
    def is_zero(self) -> Value:
        return self.exp == 0

    @property
    def is_inf(self) -> Value:
        return self.exp.all()

    @property
    def significand(self) -> Value:
        return Cat(self.frac, Const(1, 1))


class FloatLayout(data.StructLayout):
    """The layout of a float port; in simulation it reads as and accepts `Zkf`."""

    def __init__(self, fmt: ZkfFormat) -> None:
        self.fmt = fmt
        super().__init__({"frac": fmt.wfrac, "exp": fmt.wexp, "sign": 1})

    def __call__(self, target: ValueLike) -> FloatView:
        return FloatView(self, target)

    def const(self, init: object) -> data.Const:
        if isinstance(init, Zkf):
            if init.fmt != self.fmt:
                raise ValueError(f"{init.fmt} is not {self.fmt}")
            return data.Const(self, init.bits)
        return super().const(init)

    def from_bits(self, raw: int) -> Zkf:
        return Zkf(self.fmt, raw)


@dataclass(frozen=True, slots=True)
class Timing:
    latency: int
    initiation_interval: int


@dataclass(frozen=True, eq=False)
class Block(ABC):
    """
    A configuration that compiles to one Verilog top module. Its dataclass fields are its knobs, except those declared
    `compare=False`. A subclass is declared `@dataclass(frozen=True, kw_only=True, eq=False)`; two instances of one
    class are equal iff their `config` is, whether a knob was given or defaulted.
    """

    operator: ClassVar[str]

    def __post_init__(self) -> None:
        """Refuses a set of registers of another type."""
        for knob in fields(self):
            value = getattr(self, knob.name)
            if isinstance(knob.default, enum.Flag) and not isinstance(value, type(knob.default)):
                raise TypeError(f"{knob.name}={value!r}: a set of `{type(knob.default).__qualname__}` is due")

    @property
    def config(self) -> dict[str, Any]:
        """The knob values, defaults resolved, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.compare}

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.config == self.config

    def __hash__(self) -> int:
        return hash((type(self), *self.config.items()))

    @property
    def name(self) -> str:
        """
        The Verilog module name: the operator and a digest of `config` and the latency, so distinct configurations never
        share a name, nor does one whose latency changes, at a length that vendor tools take; attributes are no part of
        it, so two texts of one name can differ by them (`compile(prefix)` tells such modules apart).
        """
        # `True` and 1 are one knob value.
        knobs = [(knob, int(value) if isinstance(value, bool) else value) for knob, value in self.config.items()]
        digest = hashlib.sha256(str((self.operator, knobs, self.timing)).encode()).hexdigest()
        return f"zkf_{self.operator}_{digest[:12]}"

    @property
    @abstractmethod
    def timing(self) -> Timing | Mapping[int, Timing]:
        """
        A `Timing` if the timing does not depend on the mode (even if a mode input selects the arithmetic); otherwise a
        mapping from each legal value of the mode-selecting input to its `Timing`. The initiation interval of a mode is
        the number of cycles from accepting a transaction in that mode to accepting the next one of any mode, with no
        output back-pressure.
        """

    @property
    def clocked(self) -> bool:
        """Whether the module has `clk` and `rst`: iff some mode has a nonzero latency."""
        timing = self.timing
        return any(t.latency for t in ([timing] if isinstance(timing, Timing) else timing.values()))

    @property
    @abstractmethod
    def signature(self) -> wiring.Signature:
        """The ports, less `clk` and `rst`, which are present iff `clocked`."""

    @abstractmethod
    def component(self) -> wiring.Component:
        """
        The hardware, for an Amaranth design or simulation. Converted by Amaranth directly, it lacks the top module that
        `compile` adds.
        """

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """
        The bit-exact reference. The parameters are the input data ports, by name and in signature order: a `Zkf` of the
        port's format, an `int`, or the port's enumeration. The result is the output data port, or a named tuple of them
        if there are several, with flags as `bool`. A float of another format or an integer that its port cannot hold
        raises `ValueError`, even on a port that a fixed mode ignores.
        """

    def compile(self, prefix: str = "") -> str:
        """
        Verilog of this configuration and every module under it, each named `prefix + self.name` or prefixed by it,
        headed by a line naming ZKF, its version and its author, and a table of the knobs and the timing. The top
        module has the ports of `signature` in order, after `clk` and `rst` if `clocked`, with signed ports declared
        signed, which Amaranth cannot do; it wraps the Amaranth output as the instance `core`.
        """
        from . import __version__

        name = prefix + self.name
        ports = [("clk", "input wire"), ("rst", "input wire")] if self.clocked else []
        for port, member in self.signature.members.items():
            shape = Shape.cast(member.shape)
            kind = ("output" if member.flow == wiring.Out else "input") + " wire" + " signed" * shape.signed
            ports.append((port, kind + f" [{shape.width - 1}:0]" * (shape.width > 1)))
        top = [
            f"// Produced by ZKF {__version__}, author: Zubax Robotics",
            f"// {self.operator}:",
            *(f"//   {knob}: {value}" for knob, value in self.config.items()),
            f"//   timing: {self.timing}",
            f"module {name}({', '.join(port for port, _ in ports)});",
        ]
        top += [f"  {kind} {port};" for port, kind in ports]
        top += [f"  {name}_core core ({', '.join(f'.{port}({port})' for port, _ in ports)});", "endmodule", ""]
        core = verilog.convert(self.component(), name=f"{name}_core", emit_src=False, strip_internal_attrs=True)
        return "\n".join(top) + core


@dataclass(frozen=True, eq=False)
class Operator(Block):
    """
    An operator on floats of the format `fmt`, which `Resize` converts to from another. Its `stage_input` and
    `stage_output` are register stages at its ports; where >0, no combinational path is exposed at that port.
    """

    fmt: ZkfFormat


def delay[T: ValueLike](m: Module, name: str, shape: ShapeLike, value: T, stages: int, attrs: Attrs | None = None) -> T:
    """`value` delayed by `stages` reset-less registers, `attrs` on the last."""
    for k in range(stages):
        q = Signal(shape, name=f"{name}{k}", reset_less=True, attrs=dict(attrs or {}) if k == stages - 1 else {})
        m.d.sync += q.eq(value)
        value = q
    return value


def pipe[T: ValueLike](
    m: Module, name: str, valid: ValueLike, shape: ShapeLike, payload: T, stages: int, attrs: Attrs | None = None
) -> tuple[ValueLike, T]:
    """`valid` and `payload` delayed by `stages` registers, `attrs` on the last payload; reset clears only the flags."""
    for k in range(stages):
        kept = dict(attrs or {}) if k == stages - 1 else {}
        flag, q = Signal(name=f"{name}{k}_valid"), Signal(shape, name=f"{name}{k}", reset_less=True, attrs=kept)
        m.d.sync += [flag.eq(valid), q.eq(payload)]
        valid, payload = flag, q
    return valid, payload


def bundle(layout: data.StructLayout, **fields: ValueLike) -> data.View:
    """A view of `layout` built from its fields by name, each as wide as its member."""
    assert all(len(Value.cast(fields[name])) == Shape.cast(shape).width for name, shape in layout.members.items())
    return layout(Cat(*(fields[name] for name in layout.members)))

"""Operator base classes and their Verilog output, the float port layout, and register helpers."""

import enum
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from amaranth.back import rtlil, verilog
from amaranth.hdl import Cat, Const, Module, Shape, ShapeLike, Signal, Value, ValueLike
from amaranth.lib import data, wiring

from ._value import Zkf, ZkfFormat

type Attrs = Mapping[str, int | str]
"""Synthesis attributes: name to value."""


class FloatView(data.View):
    """
    A ZKF value on a float port. Each read of `is_zero`, `is_inf` or `significand` builds new logic: assign it to a
    signal if it is used more than once.
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
    """The layout of a float port; in simulation its values are read and written as `Zkf`."""

    def __init__(self, fmt: ZkfFormat) -> None:
        self.fmt = fmt
        super().__init__({"frac": fmt.wfrac, "exp": fmt.wexp, "sign": 1})

    def __call__(self, target: ValueLike) -> FloatView:
        return FloatView(self, target)

    def const(self, init: object) -> data.Const:
        if isinstance(init, Zkf):
            if init.fmt != self.fmt:
                raise ValueError(f"expected {self.fmt}, got {init.fmt}")
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
    A configuration that compiles to one Verilog top module. The dataclass fields are its knobs, except those declared
    `compare=False` (such as synthesis attributes), which are left out of `config` but still change `name`. Subclasses
    are declared `@dataclass(frozen=True, kw_only=True, eq=False)`; two instances are equal if their `config` is.
    """

    operator: ClassVar[str]

    def __post_init__(self) -> None:
        """Rejects a register-set knob of the wrong type."""
        for knob in fields(self):
            value = getattr(self, knob.name)
            if isinstance(knob.default, enum.Flag) and not isinstance(value, type(knob.default)):
                raise TypeError(f"{knob.name}={value!r} is not of type `{type(knob.default).__qualname__}`")

    @property
    def config(self) -> dict[str, Any]:
        """The knob values, defaults included, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.compare}

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.config == self.config

    def __hash__(self) -> int:
        return hash((type(self), *self.config.items()))

    @property
    def name(self) -> str:
        """The Verilog module name: the operator and `digest`."""
        return f"zkf_{self.operator}_{self.digest}"

    @property
    def digest(self) -> str:
        """
        A 12-digit hex hash of the operator, `config`, the attributes and the timing, so that different configurations,
        or a configuration whose latency changes, get different names short enough for vendor tools.
        """

        def plain(value: Any) -> Any:
            if isinstance(value, enum.Flag):  # By name: member values shift when a member is added.
                return sorted(member.name for member in value)
            if isinstance(value, Mapping):  # None and {} both mean no attributes.
                return sorted((key, plain(item)) for key, item in value.items())
            return int(value) if isinstance(value, bool) else value  # `True` and 1 are the same value.

        knobs = sorted((knob, plain(value)) for knob, value in self.config.items())
        attrs = sorted((f.name, plain(getattr(self, f.name) or {})) for f in fields(self) if not f.compare)
        return hashlib.sha256(str((self.operator, knobs, attrs, self.timing)).encode()).hexdigest()[:12]

    @property
    @abstractmethod
    def timing(self) -> Timing | Mapping[int, Timing]:
        """
        A `Timing` if it is the same in every mode, else a mapping from each value of the mode input to its `Timing`.
        The initiation interval of a mode is the number of cycles from accepting a transaction in that mode until the
        next one can be accepted, with no output back-pressure.
        """

    @property
    def clocked(self) -> bool:
        """Whether the module has `clk` and `rst`: true if any mode has a nonzero latency."""
        timing = self.timing
        return any(t.latency for t in ([timing] if isinstance(timing, Timing) else timing.values()))

    @property
    @abstractmethod
    def signature(self) -> wiring.Signature:
        """The ports, except `clk` and `rst`, which exist only if `clocked`."""

    @abstractmethod
    def component(self) -> wiring.Component:
        """The hardware, for an Amaranth design or simulation; it lacks the top module that `compile` adds."""

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """
        The bit-exact reference. Arguments are the input data ports, in signature order: a `Zkf` of the port's format,
        an `int`, or the port's enum. Returns the output data port, or a named tuple if there are several, with flags as
        `bool`. A float of another format or an integer too wide for its port raises `ValueError`, even if unused.
        """

    def compile(self, prefix: str = "", *, proc_nomux: bool | None = None) -> str:
        """
        The Verilog of this configuration: top module `prefix + self.name`, submodules prefixed with it, and a header
        with the ZKF version, the knobs and the timing. The top module has the ports of `signature` in order (after
        `clk` and `rst` if `clocked`), declares signed ports signed, and wraps the Amaranth design as `core`.
        No flip-flop has an initial value (ROM contents are kept), so synthesis can prune logic that constant inputs
        leave unused. Yosys is chosen as in Amaranth: by `AMARANTH_USE_YOSYS`, default the system one (`YOSYS`), else
        the bundled one; `javascript` runs a Pyodide host's.

        `proc_nomux` is Yosys's `proc -nomux`: on (the default, as in Amaranth), switches become `always @*` `case`
        blocks; off, comparators and multiplexers, which some tools map better. It is not a stable API.
        """
        from . import __version__

        name = prefix + self.name
        ports = [("clk", "input wire"), ("rst", "input wire")] if self.clocked else []
        for port, member in self.signature.members.items():
            shape = Shape.cast(member.shape)
            kind = ("output" if member.flow == wiring.Out else "input") + " wire" + " signed" * shape.signed
            ports.append((port, kind + f" [{shape.width - 1}:0]" * (shape.width > 1)))
        top = [
            f"// Produced by ZKF {__version__}, author: Zubax Robotics <sales@zubax.com>",
            f"// {self.operator}:",
            *(f"//   {knob}: {value}" for knob, value in self.config.items()),
            f"//   timing: {self.timing}",
            f"module {name}({', '.join(port for port, _ in ports)});",
        ]
        top += [f"  {kind} {port};" for port, kind in ports]
        top += [f"  {name}_core core ({', '.join(f'.{port}({port})' for port, _ in ports)});", "endmodule", ""]
        nomux = True if proc_nomux is None else proc_nomux
        return "\n".join(top) + _verilog(self.component(), f"{name}_core", nomux=nomux)


@dataclass(frozen=True, eq=False)
class Operator(Block):
    """
    An operator on floats of format `fmt` (`Resize` converts between formats). `stage_input` and `stage_output` are
    register stages at the ports; when nonzero, no combinational path reaches that port.
    """

    fmt: ZkfFormat


def delay[T: ValueLike](m: Module, name: str, shape: ShapeLike, value: T, stages: int, attrs: Attrs | None = None) -> T:
    """`value` delayed by `stages` registers without reset, `attrs` on the last."""
    for k in range(stages):
        q = Signal(shape, name=f"{name}{k}", reset_less=True, attrs=dict(attrs or {}) if k == stages - 1 else {})
        m.d.sync += q.eq(value)
        value = q
    return value


def pipe[T: ValueLike](
    m: Module, name: str, valid: ValueLike, shape: ShapeLike, payload: T, stages: int
) -> tuple[ValueLike, T]:
    """`valid` and `payload` delayed by `stages` registers; reset clears only the valid flags."""
    for k in range(stages):
        flag, q = Signal(name=f"{name}{k}_valid"), Signal(shape, name=f"{name}{k}", reset_less=True)
        m.d.sync += [flag.eq(valid), q.eq(payload)]
        valid, payload = flag, q
    return valid, payload


def zext(value: Value, width: int) -> Value:
    """Zero-extended to `width` bits, unsigned."""
    return Cat(value, Const(0, width - len(value)))


def sext(value: Value, width: int) -> Value:
    """Sign-extended to `width` bits, signed."""
    return Cat(value, value[-1].replicate(width - len(value))).as_signed()


def ge(left: Value, right: Value) -> Value:
    """`left >= right` from the borrow of a subtraction, so that it maps onto a carry chain."""
    return ~(left - right)[-1]


def add_carry(a: Value, b: Value, carry: Value) -> Value:
    """`a + b + carry`, unsigned, as wide as `a + b`; built so that every tool makes one carry chain of it."""
    return (Cat(carry, a) + Cat(carry, b))[1:]


def bundle(layout: data.StructLayout, **fields: ValueLike) -> data.View:
    """A view of `layout` from its fields by name, each exactly as wide as its member."""
    assert all(len(Value.cast(fields[name])) == Shape.cast(shape).width for name, shape in layout.members.items())
    return layout(Cat(*(fields[name] for name in layout.members)))


_STRIPPED = ("generator", "top", "src", "amaranth.hierarchy", "amaranth.decoding")
"""The attributes that `verilog.convert(strip_internal_attrs=True)` removes."""


def _verilog(component: wiring.Component, name: str, *, nomux: bool) -> str:
    """`component` as Verilog module `name`: with `nomux`, what `verilog.convert` gives, minus the initial values."""
    removed = " ".join(f"-remove {attribute}" for attribute in _STRIPPED)
    script = [
        f"read_rtlil <<rtlil\n{rtlil.convert(component, name=name, emit_src=False)}\nrtlil",
        "proc -norom" + " -nomux" * nomux,
        "memory_collect",
        f"attrmap -remove init {removed}",
        f"attrmap -modattr {removed}",
        # Without `-nomux`, `-norename` would keep mux functions named `$...`, which Icarus takes for system functions.
        "write_verilog" + " -norename" * nomux,
    ]
    code, out, err = _yosys(["-q", "-"], "\n".join(script))
    if code:
        raise verilog.YosysError(err.strip())
    return out


def _yosys(args: list[str], stdin: str) -> tuple[int, str, str]:
    """Runs the Yosys that `Block.compile` picks; returns the exit code, stdout and stderr."""
    clauses = os.environ.get("AMARANTH_USE_YOSYS", "system,builtin")
    for clause in clauses.split(","):
        if clause == "system" and (found := shutil.which(os.environ.get("YOSYS", "yosys"))):
            return _run([found, *args], stdin)
        if clause == "builtin" and importlib.util.find_spec("amaranth_yosys"):
            # `-P`: files in the current directory cannot shadow its imports.
            return _run([sys.executable, "-P", "-m", "amaranth_yosys", *args], stdin)
        if clause == "javascript":
            try:  # A Pyodide host's `runAmaranthYosys(args, stdin)` returns `[exit_code, stdout, stderr]`.
                run = importlib.import_module("js").runAmaranthYosys
            except ImportError, AttributeError:
                continue
            code, out, err = run(args, stdin)
            return code, out, err
    raise verilog.YosysError(f"no Yosys found for AMARANTH_USE_YOSYS={clauses!r} (system, builtin or javascript)")


def _run(command: list[str], stdin: str) -> tuple[int, str, str]:
    done = subprocess.run(command, input=stdin, capture_output=True, encoding="utf-8")
    return done.returncode, done.stdout, done.stderr

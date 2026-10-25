"""
The top that a flow synthesizes: an operator between registers, so that each of its paths is timed from a flip-flop
to a flip-flop and none is lost to an unconstrained pin. The registers are counted in every figure of the suite.
"""

import dataclasses
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from amaranth.hdl import Shape
from amaranth.lib import wiring

from _catalog import CoreSpec
from zkf import Attrs, Block, Cordic, Divsqrt, Exp2, Log2, Timing

type Row = dict[str, str]
"""What a flow found of a spec, as its report shows it."""

FREQ_MHZ = 100.0
"""The clock that every flow is constrained to."""

HANDSHAKE = ("in_valid", "in_ready", "out_valid", "out_ready")
KEEP = '(* keep = "true", syn_preserve = "true" *)'


@dataclass(frozen=True)
class Port:
    name: str
    width: int
    signed: bool
    output: bool

    @property
    def kind(self) -> str:
        return ("signed " if self.signed else "") + (f"[{self.width - 1}:0] " if self.width > 1 else "")


def ports(signature: wiring.Signature) -> list[Port]:
    out = []
    for name, member in signature.members.items():
        shape = Shape.cast(member.shape)
        out.append(Port(name, shape.width, shape.signed, member.flow == wiring.Out))
    return out


def harness(
    top: str,
    dut: str,
    ports: Sequence[Port],
    *,
    clocked: bool,
    unused: Mapping[str, int | None] | None = None,
) -> str:
    """
    A module `top` that holds `dut` with a register on every port; reset clears the registers of the stream control
    only. `clocked` is whether the DUT has `clk` and `rst`. A port in `unused` gets no pin and no register, as an
    instance that does not use it would have it: an input is tied to the constant given, an output (None) is left
    open.
    """
    unused = unused or {}
    shown = [p for p in ports if p.name not in unused]
    captured, driven = [p for p in shown if not p.output], [p for p in shown if p.output]
    control = [p for p in shown if p.name in HANDSHAKE]

    def source(p: Port) -> str:
        return f"dut_{p.name}" if p.output else p.name

    lines = ["`default_nettype none", "", f"module {top} ("]
    lines += ["    input  wire clk,", "    input  wire rst,"]
    lines += [f"    {'output' if p.output else 'input '} wire {p.kind}{p.name}," for p in shown]
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    for p in captured:
        lines += [f"    {KEEP}", f"    reg {p.kind}r_{p.name};"]
    lines += [f"    wire {p.kind}dut_{p.name};" for p in driven]
    for p in driven:
        lines += [f"    {KEEP}", f"    reg {p.kind}r_{p.name};"]
    lines += [f"    assign {p.name} = r_{p.name};" for p in driven]

    connections = [".clk(clk)", ".rst(rst)"] if clocked else []
    for p in ports:
        if p.name in unused:
            tie = unused[p.name]
            net = "" if tie is None else f"{p.width}'d{tie % (1 << p.width)}"
        else:
            net = f"dut_{p.name}" if p.output else f"r_{p.name}"
        connections.append(f".{p.name}({net})")
    lines.append(f"    {dut} dut (")
    lines += [f"        {c}," for c in connections]
    lines[-1] = lines[-1].rstrip(",")
    lines.append("    );")

    lines.append("    always @(posedge clk) begin")
    if control:
        lines.append("        if (rst) begin")
        lines += [f"            r_{p.name} <= 1'b0;" for p in control]
        lines.append("        end else begin")
        lines += [f"            r_{p.name} <= {source(p)};" for p in control]
        lines.append("        end")
    lines += [f"        r_{p.name} <= {source(p)};" for p in shown if p not in control]
    lines += ["    end", "endmodule", "", "`default_nettype wire", ""]
    return "\n".join(lines)


def unused(model: Block | CoreSpec) -> Mapping[str, int | None]:
    """
    The ports that a spec of one fixed mode leaves out of its harness, as an instance of that mode in a design ties
    the inputs off and leaves the outputs open: the constant an input is tied to, None for an output.
    """
    ties: Mapping[str, int | None] = {}
    if isinstance(model, Divsqrt):
        ties = ({"op_sqrt": 0}, {"op_sqrt": 0, "b": 0}, {})[model.mode]
    if isinstance(model, Cordic):
        ties = ({"vectoring": 0, "b": 0}, {"vectoring": 1, "quadrant": None}, {})[model.mode]
    return ties


@dataclass(frozen=True)
class Sources:
    rtl: Path
    harness: Path


@dataclass(frozen=True)
class Design:
    """A spec as a flow is handed it."""

    name: str
    model: Block | CoreSpec
    large: bool = False
    """Placed on the larger part, by the flows that offer one."""
    faster: bool = False
    """Placed on the faster speed grade, by the flows that offer one."""

    @property
    def top(self) -> str:
        return f"{self.name}_synth_top"

    @property
    def latency(self) -> int | tuple[int, ...]:
        timing = self.model.timing
        return timing.latency if isinstance(timing, Timing) else tuple(t.latency for t in timing.values())

    @property
    def row(self) -> Row:
        """What every row of the spec begins with: its name, which its directory takes, its knobs, latency and group."""
        latency = self.latency
        if isinstance(latency, tuple):
            stages = f"{' / '.join(map(str, latency))} stages (rotation / vectoring)"
        else:
            stages = f"{latency} {'stage' if latency == 1 else 'stages'}"
        group = f"{type(self.model).__name__} {getattr(self.model, 'fmt', '')}"
        config = self.model.config if isinstance(self.model, Block) else {}
        return {
            "name": self.name,
            "knobs": ", ".join(f"{knob}={value}" for knob, value in config.items()),
            "register_stages": stages,
            "group": group,
        }

    def write(self, directory: Path, *, rom_attrs: Attrs | None = None) -> Sources:
        """The compiled text, with the attributes that the flow puts on a table's memory, and the harness."""
        model = self.model
        if isinstance(model, Exp2 | Log2):
            model = dataclasses.replace(model, rom_attrs=rom_attrs)
        sources = Sources(directory / "rtl.v", directory / "harness.v")
        sources.rtl.write_text(model.compile())
        pins, ties = ports(model.signature), unused(model)
        text = harness(self.top, model.name, pins, clocked=model.clocked, unused=ties)
        sources.harness.write_text(text)
        return sources


def run(command: Sequence[str | Path], log: Path, timeout: float | None = None) -> None:
    """A tool's command in the directory of its log, which takes what it prints."""
    rendered = [str(item) for item in command]
    with log.open("w") as out:
        out.write("$ " + " ".join(shlex.quote(item) for item in rendered) + "\n\n")
        out.flush()
        try:
            subprocess.run(rendered, cwd=log.parent, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            out.write(f"\n\n[run] command exceeded {timeout:g}s timeout and was killed\n")
            raise

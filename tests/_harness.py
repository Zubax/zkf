"""
The top that a flow synthesizes: an operator between registers, so that each of its paths is timed from a flip-flop
to a flip-flop and none is lost to an unconstrained pin. The registers are counted in every figure of the suite.
"""

import dataclasses
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from amaranth.hdl import Shape
from amaranth.lib import wiring

from zkf import Attrs, Block, Exp2, Log2, Timing

type Row = dict[str, str]
"""What a flow found of a spec, as its report shows it."""

FREQ_MHZ = 100.0
"""The clock every flow but Vivado's is constrained to; its targets set their own."""

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
) -> str:
    """
    A module `top` that holds `dut` with a register on every port; reset clears the registers of the stream control
    only. `clocked` is whether the DUT has `clk` and `rst`.
    """
    captured, driven = [p for p in ports if not p.output], [p for p in ports if p.output]
    control = [p for p in ports if p.name in HANDSHAKE]

    def source(p: Port) -> str:
        return f"dut_{p.name}" if p.output else p.name

    lines = ["`default_nettype none", "", f"module {top} ("]
    lines += ["    input  wire clk,", "    input  wire rst,"]
    lines += [f"    {'output' if p.output else 'input '} wire {p.kind}{p.name}," for p in ports]
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
        connections.append(f".{p.name}({f'dut_{p.name}' if p.output else f'r_{p.name}'})")
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
    lines += [f"        r_{p.name} <= {source(p)};" for p in ports if p not in control]
    lines += ["    end", "endmodule", "", "`default_nettype wire", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class Sources:
    rtl: Path
    harness: Path


@dataclass(frozen=True)
class Design:
    """A spec as a flow is handed it."""

    name: str
    model: Block
    large: bool = False
    """Placed on the larger part, by the flows that offer one."""
    ecp5_grade: int = 6
    """The speed grade, by the ECP5 flows."""
    artix7_grade: int = 1
    """The speed grade, by the Artix-7 flow."""
    zynquplus_grade: int = 1
    """The speed grade, by the Zynq UltraScale+ flow."""
    cyclone4_grade: int = 8
    """The speed grade, by the Cyclone IV E flow."""
    cyclone5_grade: int = 8
    """The speed grade, by the Cyclone V flow."""

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
        return {
            "name": self.name,
            "knobs": ", ".join(f"{knob}={value}" for knob, value in self.model.config.items()),
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
        text = harness(self.top, model.name, ports(model.signature), clocked=model.clocked)
        sources.harness.write_text(text)
        return sources


@dataclass(frozen=True)
class Device:
    """The device of a vendor's flow."""

    template: str
    """The part, its speed grade left as `{grade}`."""
    grade: Callable[[Design], int]

    def part(self, design: Design) -> str:
        return self.template.format(grade=self.grade(design))


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

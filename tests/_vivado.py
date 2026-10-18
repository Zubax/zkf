"""The Vivado flow on the Artix-7: synthesis, placement and routing out of context, once."""

import re
import shutil
from pathlib import Path

from _harness import Design, Row, run

TOOLS = ("vivado",)
PART = "xc7a100tcsg324-1"
"""The slowest speed grade."""
FREQ_MHZ = 150.0
TIMEOUT_S = 3600


def tcl(design: Design, rtl: Path, harness: Path, xdc: Path) -> str:
    return f"""read_verilog {{{rtl}}} {{{harness}}}
read_xdc {{{xdc}}}
synth_design -top {design.top} -part {PART} -mode out_of_context
opt_design
place_design
route_design
report_timing_summary -file timing.rpt
report_utilization -file utilization.rpt
report_route_status -file route.rpt
"""


def synthesize(design: Design, directory: Path) -> Row:
    """Writes the spec into the directory, cleared first, runs the flow and reads the row from what it left."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    sources = design.write(directory)
    xdc, script = directory / "clock.xdc", directory / "run.tcl"
    xdc.write_text(f"create_clock -period {1000 / FREQ_MHZ:.3f} -name clk [get_ports clk]\n")
    script.write_text(tcl(design, sources.rtl, sources.harness, xdc))
    run(["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", script], directory / "vivado.log", TIMEOUT_S)

    def text(name: str) -> str:
        path = directory / name
        return path.read_text(errors="replace") if path.is_file() else ""

    timing, utilization, route = text("timing.rpt"), text("utilization.rpt"), text("route.rpt")
    summary = re.search(r"WNS\(ns\).*?\n\s*-+.*?\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+\d+\s+\d+\s+(-?[0-9.]+)", timing, re.S)
    wns, whs = (float(summary.group(1)), float(summary.group(3))) if summary else (None, None)
    errors = re.search(r"# of nets with routing errors\.*\s*:\s*(\d+)", route)
    routed = errors is not None and int(errors.group(1)) == 0
    period = 1000 / FREQ_MHZ

    def used(label: str) -> str:
        found = re.search(rf"\|\s*{label}\s*\|\s*([0-9.]+)\s*\|", utilization)
        return found.group(1) if found else "not reported"

    return design.row | {
        "target": f"{FREQ_MHZ:g} MHz",
        "fmax": "not reported" if wns is None else f"{1000 / (period - wns):.2f} MHz",
        "slack": "not reported" if wns is None else f"{wns:.3f} ns",
        "status": "PASS" if routed and wns is not None and wns >= 0 and whs is not None and whs >= 0 else "FAIL",
        "lut": used("Slice LUTs"),
        "ff": used("Slice Registers"),
        "dsp": used("DSPs"),
        "bram": used("Block RAM Tile"),
        "vivado_log": str((directory / "vivado.log").relative_to(directory.parent)),
        "timing_report": str((directory / "timing.rpt").relative_to(directory.parent)),
    }

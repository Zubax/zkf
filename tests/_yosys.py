"""
The Yosys flow on the Lattice ECP5: Yosys synthesizes the harness of a spec and nextpnr-ecp5 places and routes it.
"""

import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _harness import FREQ_MHZ, Design, Row, Sources, run

type Report = dict[str, Any]
"""The JSON report of nextpnr-ecp5."""

ROM_ATTRS = {"rom_style": "block"}
YOSYS_COMMAND = os.environ.get("YOSYS", "yosys")
"""The Yosys of the system, as Amaranth finds it: the one that comes with Amaranth reads no Verilog, has no solver."""
YOSYS = shutil.which(YOSYS_COMMAND)
"""Where `YOSYS_COMMAND` is, if anywhere."""
NEXTPNR = "nextpnr-ecp5"
TOOLS = (YOSYS_COMMAND, NEXTPNR)
DEVICE, LARGE_DEVICE = "LFE5U-12F", "LFE5U-45F"
PACKAGE = "CABGA381"
UTIL_RESOURCES = ("TRELLIS_COMB", "TRELLIS_FF", "TRELLIS_IO", "MULT18X18D", "ALU54B", "DP16KD", "TRELLIS_RAMW", "DCCA")
"""The resources that a row lists even where none is used."""
# A diverging place-and-route fails its spec instead of stalling the run.
YOSYS_TIMEOUT_S = 900.0
NEXTPNR_TIMEOUT_S = 1200.0


def part(design: Design) -> str:
    return f"{LARGE_DEVICE if design.large else DEVICE}-{design.ecp5_grade}"


def script(design: Design, sources: Sources, netlist: Path) -> str:
    return "\n".join(
        [f"read_verilog {sources.rtl} {sources.harness}", f"hierarchy -check -top {design.top}"]
        + ["proc", "opt"]
        # No -dff: ABC in sequential mode retimes registers across the multipliers, which the wide specs lose by.
        + [f"synth_ecp5 -top {design.top} -noabc9 -abc2 -json {netlist}", "stat", ""]
    )


def nextpnr_args(design: Design, netlist: Path, report: Path) -> list[str | Path]:
    return [
        "--45k" if design.large else "--12k",
        "--package",
        PACKAGE,
        "--speed",
        str(design.ecp5_grade),
        "--freq",
        f"{FREQ_MHZ:g}",
        "--timing-allow-fail",
        # No pads: placed with them, a small module's registers follow the pads around the die, and its timing is
        # the distance between them rather than its own logic.
        "--out-of-context",
        "--json",
        netlist,
        "--report",
        report,
    ]


def synthesize(design: Design, directory: Path) -> Row:
    """Writes the spec into the directory, cleared first, runs the flow and reads the row from what it left."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    ys, netlist, report = directory / "synth.ys", directory / "netlist.json", directory / "nextpnr.json"
    sources = design.write(directory, rom_attrs=ROM_ATTRS)
    ys.write_text(script(design, sources, netlist))
    run([YOSYS_COMMAND, "-s", ys], directory / "yosys.log", YOSYS_TIMEOUT_S)
    run([NEXTPNR, *nextpnr_args(design, netlist, report)], directory / "nextpnr.log", NEXTPNR_TIMEOUT_S)

    nextpnr_text = (directory / "nextpnr.log").read_text()
    data: Report = json.loads(report.read_text())
    cells = netlist_cells(netlist, design.top)

    def link(path: Path) -> str:
        return str(path.relative_to(directory.parent))

    return design.row | {
        "fmax": fmax(data),
        "target": f"{FREQ_MHZ:g} MHz",
        "status": "PASS" if timing_met(data) else "FAIL",
        "yosys_cells": format_cells(cells),
        "utilization": utilization(nextpnr_text),
        "slack": slack(data),
        "yosys_log": link(directory / "yosys.log"),
        "nextpnr_log": link(directory / "nextpnr.log"),
        "nextpnr_json": link(report),
        "lut": str(cells.get("LUT4", 0)),
        "lut_total": used(*total_lut4(nextpnr_text)),
        "ff": str(cells.get("TRELLIS_FF", 0)),
        "comb": utilized(data, "TRELLIS_COMB"),
        "carry": str(cells.get("CCU2C", 0)),
        "pfumx": str(cells.get("PFUMX", 0)),
        "l6mux21": str(cells.get("L6MUX21", 0)),
        "dsp": utilized(data, "MULT18X18D"),
        "alu54": utilized(data, "ALU54B"),
        "bram": utilized(data, "DP16KD"),
        "io": utilized(data, "TRELLIS_IO"),
    }


def netlist_cells(netlist: Path, top: str) -> dict[str, int]:
    return dict(Counter(cell["type"] for cell in json.loads(netlist.read_text())["modules"][top]["cells"].values()))


def clocks(report: Report) -> dict[str, tuple[float, float, bool]]:
    """Each clock's achieved frequency and target, in MHz, and whether it meets the target."""
    return {
        name: (float(clock["achieved"]), float(clock["constraint"]), clock["achieved"] >= clock["constraint"])
        for name, clock in report["fmax"].items()
    }


def fmax(report: Report) -> str:
    found = clocks(report)
    return f"{min(achieved for achieved, _, _ in found.values()):.2f} MHz" if found else "not reported"


def timing_met(report: Report) -> bool:
    """The gate: every clock meets its target; with no clock stated, the spec misses."""
    found = clocks(report)
    return bool(found) and all(met for _, _, met in found.values())


def slack(report: Report) -> str:
    lines = [
        f"{name}: {1000.0 / constraint - 1000.0 / achieved:.3f} ns at {constraint:.2f} MHz target "
        f"(achieved {achieved:.2f} MHz)"
        for name, (achieved, constraint, _) in clocks(report).items()
        if achieved > 0
    ]
    return "\n".join(lines) if lines else "not reported"


def used(count: int, available: int) -> str:
    return f"{count}/{available} ({100.0 * count / available:.2f}%)"


def utilized(report: Report, key: str) -> str:
    item = report["utilization"][key]
    return used(item["used"], item["available"])


def total_lut4(nextpnr_log: str) -> tuple[int, int]:
    """The LUT4s that the netlist takes before packing, its carries and memories included, and the device's."""
    match = re.search(r"Total LUT4s:\s*([0-9]+)/([0-9]+)", nextpnr_log)
    assert match, "nextpnr stated no total of LUT4s"
    return int(match.group(1)), int(match.group(2))


def format_cells(cells: Mapping[str, int]) -> str:
    keys = ["LUT4", "TRELLIS_FF", "CCU2C", "PFUMX", "L6MUX21", "MULT18X18D", "ALU54B", "DP16KD"]
    keys.extend(sorted(key for key, value in cells.items() if value and key not in keys and not key.startswith("$")))
    lines = [f"{key}: {cells[key]}" for key in keys if cells.get(key, 0)]
    return "\n".join(lines) if lines else "not reported"


def utilization(nextpnr_log: str) -> str:
    """nextpnr's total of LUT4s, then its table of the placed resources: those listed and any other in use."""
    tables = re.findall(r"Device utilisation:\n((?:Info:\s+\w+:\s+[0-9]+/\s*[0-9]+.*\n)+)", nextpnr_log)
    rows = re.findall(r"(\w+):\s+([0-9]+)/\s*([0-9]+)", tables[-1])
    lines = [f"Total LUT4s: {used(*total_lut4(nextpnr_log))}"]
    lines += [f"{key}: {used(int(n), int(of))}" for key, n, of in sorted(rows) if key in UTIL_RESOURCES or int(n)]
    return "\n".join(lines)

"""
The Yosys flows: Yosys synthesizes the harness of a spec and nextpnr places and routes it, on the Lattice ECP5 or on
the AMD Spartan-7. A target is what tells one device from the other.
"""

import fcntl
import json
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from _harness import FREQ_MHZ, Design, Row, Sources, run

type Report = dict[str, object]
"""The JSON report of nextpnr-ecp5; nextpnr-xilinx gives none."""

ROM_ATTRS = {"rom_style": "block"}
# A diverging place-and-route fails its spec instead of stalling the run.
YOSYS_TIMEOUT_S = 900.0
NEXTPNR_TIMEOUT_S = 1200.0


@dataclass(frozen=True)
class Target:
    nextpnr: str
    tools: tuple[str, ...]
    """What the flow runs, all on the path."""
    synth: Callable[[str, Path], str]
    """The lines of the Yosys script that synthesize a top into a netlist."""
    args: Callable[[Path, Path], list[str | Path]]
    """The arguments of nextpnr, given the netlist and the report it writes."""
    util_resources: tuple[str, ...]
    """The resources that a row lists even where none is used."""
    resources: Callable[[Mapping[str, int], Report, str], Row]
    """The figures of a row from the cells of the netlist, the report and the log of nextpnr."""


def script(target: Target, design: Design, sources: Sources, netlist: Path) -> str:
    return "\n".join(
        [f"read_verilog {sources.rtl} {sources.harness}", f"hierarchy -check -top {design.top}"]
        # The README's remedy for "Conflicting init values".
        + ["attrmap -remove init", "proc", "opt"]
        + [target.synth(design.top, netlist), "stat", ""]
    )


def synthesize(target: Target, design: Design, directory: Path) -> Row:
    """Writes the spec into the directory, cleared first, runs the flow and reads the row from what it left."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    ys, netlist, report = directory / "synth.ys", directory / "netlist.json", directory / "nextpnr.json"
    sources = design.write(directory, rom_attrs=ROM_ATTRS)
    ys.write_text(script(target, design, sources, netlist))
    run(["yosys", "-s", ys], directory / "yosys.log", YOSYS_TIMEOUT_S)
    run([target.nextpnr, *target.args(netlist, report)], directory / "nextpnr.log", NEXTPNR_TIMEOUT_S)

    nextpnr_text = (directory / "nextpnr.log").read_text()
    data = read_report(report)
    cells = netlist_cells(netlist, design.top) or log_cells((directory / "yosys.log").read_text())

    def link(path: Path) -> str:
        return str(path.relative_to(directory.parent)) if path.is_file() else ""

    row = design.row | {
        "fmax": fmax(nextpnr_text, data),
        "target": f"{FREQ_MHZ:g} MHz",
        "status": "PASS" if timing_met(nextpnr_text, data) else "FAIL",
        "yosys_cells": format_cells(cells),
        "utilization": utilization(nextpnr_text, data, target.util_resources),
        "slack": slack(nextpnr_text, data),
        "json": (
            f"nextpnr JSON report keys: {', '.join(sorted(data))}" if data else "nextpnr did not emit a JSON report"
        ),
        "yosys_log": link(directory / "yosys.log"),
        "nextpnr_log": link(directory / "nextpnr.log"),
        "nextpnr_json": link(report),
    }
    return row | target.resources(cells, data, nextpnr_text)


def log_cells(yosys_log: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in yosys_log.splitlines():
        match = re.match(r"\s+([A-Za-z0-9_.$]+)\s+([0-9]+)\s*$", line)
        if match:
            counts[match.group(1)] = int(match.group(2))
        match = re.match(r"\s+([0-9]+)\s+([A-Za-z0-9_.$]+)\s*$", line)
        if match:
            counts[match.group(2)] = int(match.group(1))
    return counts


def netlist_cells(netlist: Path, top: str) -> dict[str, int]:
    try:
        data = json.loads(netlist.read_text())
    except json.JSONDecodeError, FileNotFoundError:
        return {}
    modules = data.get("modules")
    if not isinstance(modules, dict):
        return {}
    module = modules.get(top)
    if not isinstance(module, dict):
        return {}
    cells = module.get("cells")
    if not isinstance(cells, dict):
        return {}
    counts: Counter[str] = Counter()
    for cell in cells.values():
        if isinstance(cell, dict) and isinstance(cell.get("type"), str):
            counts[cell["type"]] += 1
    return dict(counts)


def read_report(path: Path) -> Report:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def fmax(nextpnr_log: str, report: Report) -> str:
    clocks = report.get("fmax")
    if isinstance(clocks, dict) and clocks:
        achieved = []
        for clock in clocks.values():
            if isinstance(clock, dict) and isinstance(clock.get("achieved"), (int, float)):
                achieved.append(float(clock["achieved"]))
        if achieved:
            return f"{min(achieved):.2f} MHz"

    # nextpnr-xilinx has no report. nextpnr prints "Max frequency for clock 'X': Y MHz" after placement and again after
    # routing; the last value of each clock is the one after routing, and the slowest clock is the figure.
    per_clock: dict[str, float] = {}
    for clock_name, value in re.findall(r"Max frequency for clock\s+'([^']+)':\s*([0-9.]+)\s*MHz", nextpnr_log):
        per_clock[clock_name] = float(value)
    if per_clock:
        return f"{min(per_clock.values()):.2f} MHz"

    matches = re.findall(r"Max frequency[^:]*:\s*([0-9.]+)\s*MHz", nextpnr_log)
    if matches:
        return f"{min(float(item) for item in matches):.2f} MHz"
    return "not reported"


def timing_met(nextpnr_log: str, report: Report) -> bool:
    """The gate: every clock of the report achieves its constraint, or else passes in the log; neither is a miss."""
    clocks = report.get("fmax")
    if isinstance(clocks, dict) and clocks:
        for clock in clocks.values():
            if not isinstance(clock, dict):
                return False
            achieved = clock.get("achieved")
            constraint = clock.get("constraint")
            if not isinstance(achieved, (int, float)) or not isinstance(constraint, (int, float)):
                return False
            if float(achieved) < float(constraint):
                return False
        return True
    # The last verdict of each clock is the one after routing.
    per_clock: dict[str, str] = {}
    for clock_name, verdict in re.findall(r"Max frequency for clock\s+'([^']+)':[^()]*\((PASS|FAIL) at", nextpnr_log):
        per_clock[clock_name] = verdict
    if per_clock:
        return all(verdict == "PASS" for verdict in per_clock.values())
    return False


def slack(nextpnr_log: str, report: Report) -> str:
    clocks = report.get("fmax")
    if isinstance(clocks, dict) and clocks:
        lines = []
        for clock_name, clock in clocks.items():
            if not isinstance(clock, dict):
                continue
            achieved = clock.get("achieved")
            constraint = clock.get("constraint")
            if isinstance(achieved, (int, float)) and isinstance(constraint, (int, float)) and achieved > 0:
                slack_ns = (1000.0 / float(constraint)) - (1000.0 / float(achieved))
                lines.append(
                    f"{clock_name}: {slack_ns:.3f} ns at {float(constraint):.2f} MHz target "
                    f"(achieved {float(achieved):.2f} MHz)"
                )
        if lines:
            return "\n".join(lines)

    slack_lines = [line.strip() for line in nextpnr_log.splitlines() if "slack" in line.lower()]
    return "\n".join(slack_lines[-8:]) if slack_lines else "not reported"


def used(count: int, available: int | None = None) -> str:
    if available is not None and available > 0:
        return f"{count}/{available} ({100.0 * count / available:.2f}%)"
    return str(count)


def resource(report: Report, key: str) -> tuple[int, int | None] | None:
    table = report.get("utilization")
    if isinstance(table, dict):
        item = table.get(key)
        if isinstance(item, dict):
            count = item.get("used")
            available = item.get("available")
            if isinstance(count, int):
                return count, available if isinstance(available, int) else None
    return None


def format_resource(report: Report, key: str, fallback: int | None = None) -> str:
    found = resource(report, key)
    if found is not None:
        return used(*found)
    if fallback is not None:
        return str(fallback)
    return "not reported"


def placed_lut4(nextpnr_log: str) -> tuple[int, int] | None:
    match = re.search(r"Total LUT4s:\s*([0-9]+)/([0-9]+)", nextpnr_log)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def format_cells(cells: Mapping[str, int]) -> str:
    keys = ["LUT4", "TRELLIS_FF", "CCU2C", "PFUMX", "L6MUX21", "MULT18X18D", "ALU54B", "DP16KD"]
    keys.extend(sorted(key for key, value in cells.items() if value and key not in keys and not key.startswith("$")))
    lines = [f"{key}: {cells[key]}" for key in keys if cells.get(key, 0)]
    return "\n".join(lines) if lines else "not reported"


def utilization(nextpnr_log: str, report: Report, util_resources: tuple[str, ...]) -> str:
    lines = []
    total = placed_lut4(nextpnr_log)
    if total is not None:
        lines.append(f"Total LUT4s: {used(*total)}")

    table = report.get("utilization")
    if isinstance(table, dict):
        keys = set(util_resources)
        keys.update(
            key
            for key, item in table.items()
            if isinstance(key, str)
            and isinstance(item, dict)
            and isinstance(item.get("used"), int)
            and item["used"] > 0
        )
        for key in sorted(keys):
            found = resource(report, key)
            if found is not None:
                lines.append(f"{key}: {used(*found)}")
        if lines:
            return "\n".join(lines)

    useful = []
    for line in nextpnr_log.splitlines():
        if any(cell in line for cell in ("TRELLIS_SLICE", "TRELLIS_FF", "LUT4", "PFU", "MULT18X18D", "DP16KD")):
            useful.append(line.strip())
    return "\n".join(useful[-12:]) if useful else "not reported"


ECP5_DEVICE = "LFE5U-12F"
ECP5_PACKAGE = "CABGA381"
ECP5_SPEED_GRADE = "6"


def ecp5_args(netlist: Path, report: Path) -> list[str | Path]:
    return [
        "--12k",
        "--package",
        ECP5_PACKAGE,
        "--speed",
        ECP5_SPEED_GRADE,
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


def ecp5_resources(cells: Mapping[str, int], report: Report, nextpnr_log: str) -> Row:
    flops = resource(report, "TRELLIS_FF")
    total = placed_lut4(nextpnr_log)
    return {
        "lut": str(cells.get("LUT4", cells.get("$_LUT_", "not reported"))),
        "lut_placed": "not reported" if total is None else used(*total),
        "ff": str(cells.get("TRELLIS_FF", flops[0] if flops is not None else "not reported")),
        "comb": format_resource(report, "TRELLIS_COMB"),
        "carry": str(cells.get("CCU2C", 0)),
        "pfumx": str(cells.get("PFUMX", 0)),
        "l6mux21": str(cells.get("L6MUX21", 0)),
        "dsp": format_resource(report, "MULT18X18D", cells.get("MULT18X18D", 0)),
        "alu54": format_resource(report, "ALU54B", cells.get("ALU54B", 0)),
        "bram": format_resource(report, "DP16KD", cells.get("DP16KD", 0)),
        "io": format_resource(report, "TRELLIS_IO"),
    }


def ecp5_synth(top: str, netlist: Path) -> str:
    # No -dff: ABC in sequential mode retimes registers across the multipliers, which the wide specs lose by.
    return f"synth_ecp5 -top {top} -noabc9 -abc2 -json {netlist}"


ECP5 = Target(
    nextpnr="nextpnr-ecp5",
    tools=("yosys", "nextpnr-ecp5"),
    synth=ecp5_synth,
    args=ecp5_args,
    util_resources=(
        "TRELLIS_COMB",
        "TRELLIS_FF",
        "TRELLIS_IO",
        "MULT18X18D",
        "ALU54B",
        "DP16KD",
        "TRELLIS_RAMW",
        "DCCA",
    ),
    resources=ecp5_resources,
)


# The open databases of the Spartan-7 cover the xc7s50 alone; the slowest speed grade.
SPARTAN_DEVICE = "xc7s50csga324-1"


def chipdb(directory: Path) -> Path:
    """
    The chip database of nextpnr-xilinx, built once by the first worker that needs it and kept, as it is not portable
    across versions. The databases of prjxray that bbaexport.py reads ship beside it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    database, bba = directory / f"{SPARTAN_DEVICE}.bin", directory / f"{SPARTAN_DEVICE}.bba"
    log = directory / f"{SPARTAN_DEVICE}_build.log"
    with (directory / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not database.is_file():
            # pypy3 runs bbaexport.py markedly faster.
            python = shutil.which("pypy3") or sys.executable
            bbaexport = shutil.which("bbaexport.py", mode=os.F_OK) or "bbaexport.py"
            try:
                run([python, bbaexport, "--device", SPARTAN_DEVICE, "--bba", bba], log)
                run(["bbasm", "--le", bba, database], log)
            except BaseException:
                database.unlink(missing_ok=True)
                raise
    return database


def spartan_args(netlist: Path, report: Path) -> list[str | Path]:
    """
    nextpnr-xilinx has no report: the figures come from its log. It refuses a top port with no IOSTANDARD, so every one
    is given one and no pin; the harness registers every port, so where the pads land is not timed.
    """
    xdc = netlist.with_suffix(".xdc")
    modules = json.loads(netlist.read_text()).get("modules", {})
    top = next((m for m in modules.values() if str(m.get("attributes", {}).get("top", "0")).strip("0")), None)
    lines = []
    if isinstance(top, dict):
        for name, port in top.get("ports", {}).items():
            width = len(port.get("bits", []))
            bits = [name] if width <= 1 else [f"{name}[{i}]" for i in range(width)]
            lines += [f"set_property IOSTANDARD LVCMOS33 [get_ports {{{bit}}}]" for bit in bits]
    xdc.write_text("\n".join(lines) + "\n")
    database = chipdb(netlist.parent.parent / "chipdb")
    return ["--chipdb", database, "--xdc", xdc, "--json", netlist, "--freq", f"{FREQ_MHZ:g}", "--timing-allow-fail"]


def spartan_resources(cells: Mapping[str, int], report: Report, nextpnr_log: str) -> Row:
    def total(*keys: str) -> int:
        return sum(cells.get(key, 0) for key in keys)

    io = total("IBUF", "OBUF", "OBUFT", "IOBUF")
    return {
        "lut": str(total("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")),
        "ff": str(total("FDRE", "FDSE", "FDCE", "FDPE", "FDCPE")),
        "carry": str(total("CARRY4")),
        "wide": str(total("MUXF7", "MUXF8")),
        "dsp": format_resource(report, "DSP48E1", total("DSP48E1")),
        "bram": str(total("RAMB18E1", "RAMB36E1")),
        "io": str(io) if io else format_resource(report, "IOB33"),
    }


def spartan_synth(top: str, netlist: Path) -> str:
    # The scope cells are deleted, which nextpnr-xilinx cannot place.
    return f"synth_xilinx -flatten -family xc7 -top {top}\ndelete t:$scopeinfo\nwrite_json {netlist}"


SPARTAN = Target(
    nextpnr="nextpnr-xilinx",
    tools=("yosys", "nextpnr-xilinx", "bbasm", "bbaexport.py"),
    synth=spartan_synth,
    args=spartan_args,
    util_resources=("SLICE_LUTX", "SLICE_FFX", "CARRY4", "DSP48E1", "RAMB18E1", "RAMB36E1"),
    resources=spartan_resources,
)

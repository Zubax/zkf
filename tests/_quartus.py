"""
The Quartus flows: Quartus Prime Lite synthesizes, fits and times the harness of a spec once, on the Cyclone IV E or on
the Cyclone V. A target is what tells one device from the other.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from _harness import FREQ_MHZ, Design, Device, Row, Sources, ports, run

TOOLS = ("quartus_map", "quartus_fit", "quartus_sta")
TIMEOUT_S = 3600
PROJECT = "spec"


@dataclass(frozen=True)
class Target(Device):
    family: str
    resources: tuple[tuple[str, str], ...]
    """The key of each figure of a row, and the label of its line in the fitter's report."""


def settings(target: Target, design: Design, sources: Sources) -> str:
    """
    Every pin of the harness but its clock is a virtual one, which the fitter places in the fabric: the spec is placed
    and timed out of context, as Vivado's and nextpnr's are.
    """
    model = design.model
    pins = ["rst", *(f"{p.name}[*]" if p.width > 1 else p.name for p in ports(model.signature))]
    lines = [
        f'set_global_assignment -name FAMILY "{target.family}"',
        f"set_global_assignment -name DEVICE {target.part(design)}",
        f"set_global_assignment -name TOP_LEVEL_ENTITY {design.top}",
        *(f"set_global_assignment -name VERILOG_FILE {source.name}" for source in (sources.rtl, sources.harness)),
        "set_global_assignment -name SDC_FILE clock.sdc",
        # One processor a run: the suite runs a spec a worker.
        "set_global_assignment -name NUM_PARALLEL_PROCESSORS 1",
        *(f"set_instance_assignment -name VIRTUAL_PIN ON -to {pin}" for pin in pins),
    ]
    return "\n".join(lines) + "\n"


def synthesize(target: Target, design: Design, directory: Path) -> Row:
    """Writes the spec into the directory, cleared first, runs the flow and reads the row from what it left."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    sources = design.write(directory)
    (directory / f"{PROJECT}.qpf").write_text(f'PROJECT_REVISION = "{PROJECT}"\n')
    (directory / f"{PROJECT}.qsf").write_text(settings(target, design, sources))
    period = 1000 / FREQ_MHZ
    (directory / "clock.sdc").write_text(
        f"create_clock -name clk -period {period:.3f} [get_ports clk]\nderive_clock_uncertainty\n"
    )
    run(["quartus_map", PROJECT], directory / "map.log", TIMEOUT_S)
    run(["quartus_fit", PROJECT], directory / "fit.log", TIMEOUT_S)
    run(["quartus_sta", PROJECT, "--do_report_timing"], directory / "sta.log", TIMEOUT_S)
    # The databases, which nothing reads after the timing, are most of what the flow leaves.
    for database in ("db", "incremental_db"):
        shutil.rmtree(directory / database, ignore_errors=True)

    def text(name: str) -> str:
        path = directory / name
        return path.read_text(errors="replace") if path.is_file() else ""

    fit, sta = text(f"{PROJECT}.fit.rpt"), text(f"{PROJECT}.sta.rpt")
    # Over every corner of the device.
    worst = re.search(r"^; Worst-case Slack\s*;\s*(-?[0-9.]+)\s*;\s*(-?[0-9.]+)\s*;", sta, re.M)
    wns, whs = (float(worst.group(1)), float(worst.group(2))) if worst else (None, None)

    def link(name: str) -> str:
        return str((directory / name).relative_to(directory.parent)) if (directory / name).is_file() else ""

    return (
        design.row
        | {
            "target": f"{FREQ_MHZ:g} MHz",
            "fmax": "not reported" if wns is None else f"{1000 / (period - wns):.2f} MHz",
            "slack": "not reported" if wns is None else f"{wns:.3f} ns",
            "hold_slack": "not reported" if whs is None else f"{whs:.3f} ns",
            "status": "PASS" if wns is not None and wns >= 0 and whs is not None and whs >= 0 else "FAIL",
            "map_report": link(f"{PROJECT}.map.rpt"),
            "fit_report": link(f"{PROJECT}.fit.rpt"),
            "timing_report": link(f"{PROJECT}.sta.rpt"),
        }
        | {key: used(fit, label) for key, label in target.resources}
    )


def used(report: str, label: str) -> str:
    """The figure of the report's line, of the device's total where it states one."""
    match = re.search(rf"^; {re.escape(label)}\s*;\s*([0-9,]+)(?:\s*/\s*([0-9,]+))?", report, re.M)
    if not match:
        return "not reported"
    count = int(match.group(1).replace(",", ""))
    if match.group(2) is None:
        return str(count)
    available = int(match.group(2).replace(",", ""))
    return f"{count}/{available} ({100.0 * count / available:.2f}%)"


CYCLONE4 = Target(
    family="Cyclone IV E",
    template="EP4CE22F17C{grade}",
    grade=lambda design: design.cyclone4_grade,
    resources=(
        ("le", "Total logic elements"),
        ("registers", "Total registers"),
        ("multipliers", "Embedded Multiplier 9-bit elements"),
        ("m9k", "M9Ks"),
        ("memory_bits", "Total memory bits"),
    ),
)

CYCLONE5 = Target(
    family="Cyclone V",
    template="5CEFA5F23C{grade}",
    grade=lambda design: design.cyclone5_grade,
    resources=(
        ("alm", "Logic utilization (in ALMs)"),
        ("registers", "Total registers"),
        ("dsp", "Total DSP Blocks"),
        ("m10k", "M10K blocks"),
        ("memory_bits", "Total block memory bits"),
    ),
)

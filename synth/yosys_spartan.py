#!/usr/bin/env python3
"""
Yosys + nextpnr-xilinx synthesis of the float modules, targeting the AMD/Xilinx Spartan-7.

A thin entry point (sibling of yosys_ecp5.py) that defines the Spartan-7 device profile (synth_xilinx
command, nextpnr-xilinx device flags, resource columns) and hands it to the shared engine in yosys.py.

Unlike the ECP5 flow, this target is OPTIONAL and NON-FATAL: it runs only when nextpnr-xilinx and a
chip database are available locally, and a synthesis or timing failure never breaks the build. It is
therefore wired into CI on the deep-verify path only, not into the per-PR gate.

Device: the open prjxray / nextpnr-xilinx Spartan-7 database coverage is xc7s50 only (there are no
databases for the smaller xc7s6/xc7s15/xc7s25), so xc7s50 is the smallest *supported* Spartan-7. We
target it at the slowest speed grade (-1; Spartan-7 has no -2L).

nextpnr-xilinx selects the device through a binary chip database (--chipdb) built from prjxray with
bbaexport.py + bbasm. The database is not portable across nextpnr-xilinx versions, so we build it on
first use and cache it under the build directory. bbaexport.py ships its prjxray-db and metadata
submodules alongside (e.g. under /opt/nextpnr-xilinx/xilinx in the toolchain container), so its own
argument defaults locate them and no separate database checkout is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yosys
from common import REPO, find_executable, find_file, format_mhz, metric_cell, run, table_cell
from modules import ModuleSpec, flow_modules


BUILD_DIR = REPO / "build" / "float_synth_yosys_spartan7"
CHIPDB_DIR = BUILD_DIR / "chipdb"
DEVICE = "xc7s50csga324-1"
TARGET_FREQ_MHZ = float(os.environ.get("YOSYS_TARGET_FREQ_MHZ", "100"))

# Yosys cell types summed into each resource column. Reading the netlist gives deterministic counts;
# the per-module details "Utilization" block additionally shows whatever nextpnr-xilinx reports.
_LUT_KEYS = ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
_FF_KEYS = ("FDRE", "FDSE", "FDCE", "FDPE", "FDCPE")
_BRAM_KEYS = ("RAMB18E1", "RAMB36E1")
_IO_KEYS = ("IBUF", "OBUF", "OBUFT", "IOBUF")

# Utilization keys always shown in the per-module details section. nextpnr-xilinx bel-type spellings
# are version-dependent; unknown keys are silently skipped by the engine, and any nonzero reported
# resource is added automatically, so this list is best-effort and need not be exhaustive.
IMPORTANT_UTILIZATION_RESOURCES = (
    "SLICE_LUTX",
    "SLICE_FFX",
    "CARRY4",
    "DSP48E1",
    "RAMB18E1",
    "RAMB36E1",
)

# Device-specific resource columns shown in the table, between Status and Schematic.
_RESOURCE_HEADERS = (
    "<th>Yosys LUT</th><th>FF</th><th>CARRY4</th><th>MUXF7/8</th>"
    "<th>DSP48E1</th><th>BRAM</th><th>IO</th>"
)


def _synth_command(spec: ModuleSpec, netlist) -> str:
    # synth_xilinx's own netlist writer is -edif; emit JSON for nextpnr with a separate write_json so
    # the line works regardless of whether this Yosys build accepts synth_xilinx -json. The engine
    # joins script lines with newlines, so returning a multi-line string is fine.
    # delete t:$scopeinfo strips Yosys scope-metadata cells that nextpnr-xilinx cannot place.
    return (
        f"synth_xilinx -flatten -family xc7 -top {spec.top}\n"
        "delete t:$scopeinfo\n"
        f"write_json {netlist}"
    )


def _sum_cells(cells: dict, keys: tuple[str, ...]) -> int:
    return sum(int(cells.get(key, 0)) for key in keys)


def _extract_resources(cells: dict, report_data: dict, nextpnr_text: str) -> dict[str, str]:
    io = _sum_cells(cells, _IO_KEYS)
    return {
        "lut": str(_sum_cells(cells, _LUT_KEYS)),
        "ff": str(_sum_cells(cells, _FF_KEYS)),
        "carry": str(int(cells.get("CARRY4", 0))),
        "wide": str(int(cells.get("MUXF7", 0)) + int(cells.get("MUXF8", 0))),
        "dsp": yosys.format_nextpnr_resource(report_data, "DSP48E1", int(cells.get("DSP48E1", 0))),
        "bram": str(_sum_cells(cells, _BRAM_KEYS)),
        "io": str(io) if io else yosys.format_nextpnr_resource(report_data, "IOB33"),
    }


def _resource_row(result: dict, bounds_by_key: dict) -> str:
    return (
        metric_cell(result["lut"], bounds_by_key.get("lut"), higher_is_better=False, class_name="resource")
        + table_cell(result["ff"], "resource")
        + table_cell(result["carry"], "resource")
        + table_cell(result["wide"], "resource")
        + table_cell(result["dsp"], "resource")
        + table_cell(result["bram"], "resource")
        + table_cell(result["io"], "resource")
    )


_FLOW_DESCRIPTION = (
    "<p>Flow: Yosys synth_xilinx -flatten -family xc7, "
    f"nextpnr-xilinx for {DEVICE} at {format_mhz(TARGET_FREQ_MHZ)} (no pin constraints; --freq sets the "
    "clock target).</p>"
)

_NOTES = (
    '<p class="note"><strong>Note:</strong> the open nextpnr-xilinx / prjxray Spartan-7 database covers '
    "<code>xc7s50</code> only, so this is the smallest <em>supported</em> Spartan-7 rather than the smallest "
    "in the family. Every top-level harness port is given an <code>IOSTANDARD</code> (but no fixed pin) so the "
    "placer can route the registered harness; reported f max is the post-route, register-to-register limit. "
    "A few DSP-heavy multipliers can trip a nextpnr-xilinx DSP-packing assertion and surface as FAIL rows, "
    "which is tolerated by this optional, non-gating flow.</p>"
)


def _write_port_xdc(netlist: Path, xdc_path: Path) -> None:
    """
    Write an XDC giving every top-level port an IOSTANDARD so nextpnr-xilinx will place its IOBs.

    nextpnr-xilinx refuses to run unless every top port carries an IOSTANDARD (and, unlike nextpnr-ecp5,
    has no --lpf-allow-unconstrained escape hatch). We assign IOSTANDARD but no package pin, leaving the
    placer free to pick IOB sites. The measurement harness registers every DUT input and output, so the
    reported f max is a register-to-register limit and is unaffected by which pads the ports land on.
    """
    data = json.loads(netlist.read_text())
    modules = data.get("modules", {})
    top = next((m for m in modules.values() if str(m.get("attributes", {}).get("top", "0")).strip("0")), None)
    lines = []
    if isinstance(top, dict):
        for name, port in top.get("ports", {}).items():
            width = len(port.get("bits", []))
            bits = [name] if width <= 1 else [f"{name}[{i}]" for i in range(width)]
            lines += [f"set_property IOSTANDARD LVCMOS33 [get_ports {{{bit}}}]" for bit in bits]
    xdc_path.write_text("\n".join(lines) + "\n")


def _make_nextpnr_args(chipdb: Path):
    # nextpnr-xilinx (openXC7 fork) has no --report, so fmax/timing are recovered from the log. The
    # clock target comes from --freq (timing is allowed to miss so the run still completes), and a
    # generated per-port IOSTANDARD XDC satisfies nextpnr's requirement that every top port be
    # constrained without pinning the harness to a board.
    def _nextpnr_args(target: yosys.YosysTarget, paths: yosys.NextpnrPaths) -> list:
        xdc = paths.module_dir / f"{paths.name}.xdc"
        _write_port_xdc(paths.netlist, xdc)
        return [
            "--chipdb",
            chipdb,
            "--xdc",
            xdc,
            "--json",
            paths.netlist,
            "--freq",
            f"{paths.target_freq_mhz:g}",
            "--timing-allow-fail",
        ]

    return _nextpnr_args


def build_target(chipdb: Path) -> yosys.YosysTarget:
    return yosys.YosysTarget(
        name="spartan7",
        build_dir=BUILD_DIR,
        nextpnr_tool="nextpnr-xilinx",
        nextpnr_args=_make_nextpnr_args(chipdb),
        target_freq_mhz=TARGET_FREQ_MHZ,
        report_title=f"Kulibin Float Yosys Synthesis Report (Spartan-7 {DEVICE})",
        flow_description_html=_FLOW_DESCRIPTION,
        notes_html=_NOTES,
        util_resources=IMPORTANT_UTILIZATION_RESOURCES,
        synth_command=_synth_command,
        extract_resources=_extract_resources,
        resource_headers=_RESOURCE_HEADERS,
        resource_row=_resource_row,
        metric_keys=(("lut", False),),
        area_key="lut",
        area_label="Yosys LUT",
    )


def resolve_chipdb() -> tuple[Path | None, str]:
    """
    Locate or build the nextpnr-xilinx chip database for DEVICE; return (path, "") or (None, reason).

    A previously built database is reused from the cache; failing that, one is built with bbaexport.py
    + bbasm. The bundled bbaexport.py ships its prjxray-db and metadata submodules alongside, so its own
    argument defaults locate them and we pass only the device and output paths.
    """
    cached = CHIPDB_DIR / f"{DEVICE}.bin"
    if cached.is_file():
        return cached, ""

    bbaexport = find_file("bbaexport.py")
    if bbaexport is None:
        return None, "bbaexport.py was not found under /opt or /usr"
    bbasm = find_executable("bbasm")
    if bbasm is None:
        return None, "bbasm executable was not found on PATH or under /opt, /usr"

    # pypy3 runs bbaexport markedly faster than CPython; fall back to the current interpreter.
    python_bin = find_executable("pypy3") or Path(sys.executable)

    CHIPDB_DIR.mkdir(parents=True, exist_ok=True)
    bba = CHIPDB_DIR / f"{DEVICE}.bba"
    build_log = CHIPDB_DIR / f"{DEVICE}_build.log"
    try:
        run([python_bin, bbaexport, "--device", DEVICE, "--bba", bba], build_log)
        # bbasm assembles the .bba into the runtime binary database; --le selects little-endian, which
        # matches every host we target (x86-64 / arm64). Plain binary output is the default mode.
        run([bbasm, "--le", bba, cached], build_log)
    except Exception as exc:  # noqa: BLE001 -- a build failure must skip, not crash, the optional flow
        cached.unlink(missing_ok=True)
        return None, f"chip database build failed ({type(exc).__name__}); see {build_log}"

    if not cached.is_file():
        return None, f"chip database build produced no output; see {build_log}"
    return cached, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modules",
        help="comma-separated module names to synthesize; defaults to all configured float modules",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Check the placer first: building a chip database is pointless if nextpnr-xilinx is unavailable,
    # and this yields the most relevant skip message when the toolchain is absent entirely.
    if find_executable("nextpnr-xilinx") is None:
        print("skipping Spartan-7 synthesis: 'nextpnr-xilinx' was not found on PATH or under /opt, /usr")
        return
    chipdb, reason = resolve_chipdb()
    if chipdb is None:
        print(f"skipping Spartan-7 synthesis: {reason}")
        return
    yosys.run_flow(
        build_target(chipdb),
        flow_modules(args.modules, "YOSYS_MODULES"),
        gate=False,
        optional=True,
    )


if __name__ == "__main__":
    main()

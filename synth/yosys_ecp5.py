#!/usr/bin/env python3
"""
Yosys + nextpnr-ecp5 synthesis of the float modules, targeting the Lattice ECP5.

This is a thin entry point: it defines the ECP5 device profile (synth_ecp5 command, nextpnr-ecp5
device flags, resource columns) and hands it to the shared engine in yosys.py. A future Spartan-7
target would be a sibling yosys_spartan7.py with its own profile and the same run_flow() call.
"""

from __future__ import annotations

import argparse
import os

import yosys
from common import REPO, format_mhz, metric_cell, table_cell
from modules import ModuleSpec, flow_modules


BUILD_DIR = REPO / "build" / "float_synth_yosys_ecp5"
DEVICE_SPEED_GRADE = "6"
DEVICE_PACKAGE = "CABGA381"
TARGET_FREQ_MHZ = float(os.environ.get("YOSYS_TARGET_FREQ_MHZ", "100"))

# Utilization keys always shown in the per-module details section, even when zero.
IMPORTANT_UTILIZATION_RESOURCES = (
    "TRELLIS_COMB",
    "TRELLIS_FF",
    "TRELLIS_IO",
    "MULT18X18D",
    "ALU54B",
    "DP16KD",
    "TRELLIS_RAMW",
    "DCCA",
)

# Device-specific resource columns shown in the table, between Status and Schematic.
_RESOURCE_HEADERS = (
    "<th>Yosys LUT4</th><th>Placed LUT4</th><th>FF</th><th>TRELLIS_COMB</th>"
    "<th>CCU2C</th><th>PFUMX</th><th>L6MUX21</th><th>DSP MULT18X18D</th>"
    "<th>ALU54B</th><th>BRAM DP16KD</th><th>IO</th>"
)


DEFAULT_DEVICE_SIZE = "12k"  # LFE5U-12F/25F die; the representative small part.


# -dff is intentionally NOT passed: it runs ABC in sequential mode, which retimes/moves flops across the
# DSP multiplies. On this DSP-heavy float library that hurts the wide configs (the MULT18X18D is mapped
# combinationally either way, so retiming just disturbs placement of the reg->DSP->reg cones). Disabling it
# lifts the timing-critical wide transcendentals (e.g. zkf_log2_w8m36 +3.5 MHz, zkf_exp2_w8m36 +2.6 MHz) with
# no module dropping below the 100 MHz gate. -abc2 (extra ABC pass) and -noabc9 are kept (both measured best).
def _synth_command(spec: ModuleSpec, netlist) -> str:
    return f"synth_ecp5 -top {spec.top} -noabc9 -abc2 -json {netlist}"


def _nextpnr_args(target: yosys.YosysTarget, paths: yosys.NextpnrPaths) -> list:
    return [
        f"--{DEFAULT_DEVICE_SIZE}",
        "--package",
        DEVICE_PACKAGE,
        "--speed",
        DEVICE_SPEED_GRADE,
        "--freq",
        f"{paths.target_freq_mhz:g}",
        "--timing-allow-fail",
        "--lpf-allow-unconstrained",
        "--json",
        paths.netlist,
        "--textcfg",
        paths.module_dir / f"{paths.name}.config",
        "--report",
        paths.report,
    ]


def _extract_resources(cells: dict, report_data: dict, nextpnr_text: str) -> dict[str, str]:
    utilization = report_data.get("utilization")
    nextpnr_ff = None
    if isinstance(utilization, dict):
        ff_util = utilization.get("TRELLIS_FF")
        if isinstance(ff_util, dict) and isinstance(ff_util.get("used"), int):
            nextpnr_ff = ff_util["used"]
    return {
        "lut": str(cells.get("LUT4", cells.get("$_LUT_", "not reported"))),
        "lut_placed": yosys.format_nextpnr_total_lut4(nextpnr_text),
        "ff": str(cells.get("TRELLIS_FF", nextpnr_ff if nextpnr_ff is not None else "not reported")),
        "comb": yosys.format_nextpnr_resource(report_data, "TRELLIS_COMB"),
        "carry": str(cells.get("CCU2C", 0)),
        "pfumx": str(cells.get("PFUMX", 0)),
        "l6mux21": str(cells.get("L6MUX21", 0)),
        "dsp": yosys.format_nextpnr_resource(report_data, "MULT18X18D", cells.get("MULT18X18D", 0)),
        "alu54": yosys.format_nextpnr_resource(report_data, "ALU54B", cells.get("ALU54B", 0)),
        "bram": yosys.format_nextpnr_resource(report_data, "DP16KD", cells.get("DP16KD", 0)),
        "io": yosys.format_nextpnr_resource(report_data, "TRELLIS_IO"),
    }


def _resource_row(result: dict, bounds_by_key: dict) -> str:
    return (
        table_cell(result["lut"], "resource")
        + metric_cell(result["lut_placed"], bounds_by_key.get("lut_placed"), higher_is_better=False, class_name="resource")
        + table_cell(result["ff"], "resource")
        + table_cell(result["comb"], "resource")
        + table_cell(result["carry"], "resource")
        + table_cell(result["pfumx"], "resource")
        + table_cell(result["l6mux21"], "resource")
        + table_cell(result["dsp"], "resource")
        + table_cell(result["alu54"], "resource")
        + table_cell(result["bram"], "resource")
        + table_cell(result["io"], "resource")
    )


_FLOW_DESCRIPTION = (
    "<p>Flow: Yosys synth_ecp5 with -noabc9 -abc2 (sequential -dff retiming disabled), "
    f"nextpnr-ecp5 for LFE5U-12F {DEVICE_PACKAGE} speed grade "
    f"{DEVICE_SPEED_GRADE} at {format_mhz(TARGET_FREQ_MHZ)}.</p>"
)

_NOTES = (
    '<p class="note"><strong>Note on LUT4 counts:</strong> nextpnr-ecp5 <code>--12k</code> targets the '
    "LFE5U-25F fabric &mdash; the LFE5U-12F is the same silicon die, marketed with a reduced capacity. "
    "All LUT4, slice, and utilization figures below (including the <code>/24288</code> denominators) therefore "
    "reflect the 25F array, not the 12F&rsquo;s 12096-LUT4 marketing limit, and nextpnr will not enforce the "
    "smaller limit. For true 12F fit/capacity, use the Lattice Diamond report.</p>"
)


ECP5_TARGET = yosys.YosysTarget(
    name="ecp5",
    build_dir=BUILD_DIR,
    nextpnr_tool="nextpnr-ecp5",
    nextpnr_args=_nextpnr_args,
    target_freq_mhz=TARGET_FREQ_MHZ,
    report_title="Kulibin Float Yosys Synthesis Report",
    flow_description_html=_FLOW_DESCRIPTION,
    notes_html=_NOTES,
    util_resources=IMPORTANT_UTILIZATION_RESOURCES,
    synth_command=_synth_command,
    extract_resources=_extract_resources,
    resource_headers=_RESOURCE_HEADERS,
    resource_row=_resource_row,
    metric_keys=(("lut_placed", False),),
    area_key="lut_placed",
    area_label="placed LUT4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modules",
        help="comma-separated module names to synthesize; defaults to all configured float modules",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yosys.run_flow(ECP5_TARGET, flow_modules(args.modules, "YOSYS_MODULES"))


if __name__ == "__main__":
    main()

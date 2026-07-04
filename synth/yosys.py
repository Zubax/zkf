"""
Device-agnostic Yosys + nextpnr synthesis engine.

The flow skeleton (emit harness, run Yosys, run nextpnr, parse the report JSON, drive the worker pool,
emit HTML, gate on PASS) is identical across nextpnr targets; only a small device profile differs. That
profile is the YosysTarget below: it carries the synth command, nextpnr binary/args, target frequency,
and the device-specific resource extraction + report columns. Concrete targets (e.g. yosys_ecp5.py, and
a future yosys_spartan7.py) build a YosysTarget and call run_flow(). Not runnable on its own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable
import json
import os
import re
import subprocess

from common import (
    artifact_link,
    clean_module_dir,
    find_executable,
    format_mhz,
    generated_local_time,
    joined_links,
    metric_bounds,
    metric_cell,
    print_run_summary,
    require_executable,
    require_passing_results,
    run,
    synthesize_with_progress,
)
from modules import ModuleSpec, format_register_stages, module_group, params, register_stages, rtl_sources
from wrappers import write_wrapper


# Per-invocation wall-clock limits. nextpnr place-and-route runtime is workload- and seed-dependent and
# can occasionally diverge (especially the openXC7 nextpnr-xilinx router); yosys synthesis is usually
# quick. Bounding each means one stuck module fails fast - a non-fatal FAIL row in the optional flow -
# instead of stalling the whole run until the CI job ceiling. Override via env for slow hosts / huge designs.
NEXTPNR_TIMEOUT_S = float(os.environ.get("YOSYS_NEXTPNR_TIMEOUT_S", "1200"))
YOSYS_TIMEOUT_S = float(os.environ.get("YOSYS_SYNTH_TIMEOUT_S", "900"))


@dataclass(frozen=True)
class NextpnrPaths:
    """The per-module file paths a nextpnr command line needs, handed to YosysTarget.nextpnr_args."""

    name: str                                 # module name, e.g. "zkf_mul"; used to name device output files
    module_dir: Path                          # per-module artifact directory (where outputs may be written)
    netlist: Path                             # Yosys JSON netlist fed to nextpnr via --json
    report: Path                              # JSON utilization/timing report nextpnr writes via --report
    target_freq_mhz: float                    # clock target passed via --freq


@dataclass(frozen=True)
class YosysTarget:
    """Everything that distinguishes one nextpnr device from another within the shared flow."""

    name: str                                 # short label used in console progress, e.g. "ecp5"
    build_dir: Path                           # report + per-module artifact root
    nextpnr_tool: str                         # nextpnr executable name to locate (PATH, then /opt, /usr)
    nextpnr_args: Callable[["YosysTarget", NextpnrPaths], list]  # full post-binary nextpnr argv for one module
    target_freq_mhz: float
    report_title: str                         # HTML <title>/<h1>
    flow_description_html: str                # "Flow: ..." sentence for the report header
    notes_html: str                           # extra device-specific <p> notes; may be empty
    util_resources: tuple[str, ...]           # utilization keys always shown in the details section
    synth_command: Callable[[ModuleSpec, Path], str]               # the `synth_*` Yosys line
    extract_resources: Callable[[dict, dict, str], dict[str, str]] # cells, report, nextpnr_log -> resource columns
    resource_headers: str                     # device resource <th> cells (placed after Status)
    resource_row: Callable[[dict, "dict[str, tuple[float, float] | None]"], str]  # result, bounds -> <td> cells
    metric_keys: tuple[tuple[str, bool], ...]  # (result_key, higher_is_better) device columns that get a heatmap
    area_key: str                             # result key ranked in the console area summary (e.g. "lut_placed")
    area_label: str                           # human label for that area metric (e.g. "placed LUT4")


def write_yosys_script(
    spec: ModuleSpec,
    target: YosysTarget,
    wrapper: Path,
    netlist: Path,
    schematic_prefix: Path,
    script: Path,
) -> None:
    # The schematic is emitted from a pushed copy of the post-opt, pre-techmap design so the diagram shows
    # generic operators (adders, muxes, registers) rather than device primitives; flatten first so submodules
    # show their internals instead of opaque boxes. The original design is then popped back for the actual
    # synthesis pass, leaving its results unaffected.
    defines = script.parent / "zkf_yosys_defines.vh"
    defines.write_text(
        '`define ZKF_ATTRIBUTE_ROM_PRE (* rom_style = "block" *)\n'
        "`define ZKF_ATTRIBUTE_ROM_POST\n"
    )
    rtl = [str(defines)] + [str(path) for path in rtl_sources(spec)] + [str(wrapper)]
    schematic_commands = []
    if spec.emit_schematic:
        schematic_commands = [
            "design -push-copy",
            "flatten",
            "opt -fast",
            f"show -prefix {schematic_prefix} -format svg -notitle -stretch -enum {spec.top}",
            "design -pop",
        ]
    script.write_text(
        "\n".join(
            ["read_verilog " + " ".join(rtl)]
            + [
                f"hierarchy -check -top {spec.top}",
                "proc",
                "opt",
            ]
            + schematic_commands
            + [
                target.synth_command(spec, netlist),
                "stat",
                "",
            ]
        )
    )

def parse_cell_counts(yosys_log: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in yosys_log.splitlines():
        match = re.match(r"\s+([A-Za-z0-9_.$]+)\s+([0-9]+)\s*$", line)
        if match:
            counts[match.group(1)] = int(match.group(2))
        match = re.match(r"\s+([0-9]+)\s+([A-Za-z0-9_.$]+)\s*$", line)
        if match:
            counts[match.group(2)] = int(match.group(1))
    return counts


def read_yosys_cell_counts(netlist: Path, top: str) -> dict[str, int]:
    try:
        data = json.loads(netlist.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
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


def read_nextpnr_report(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_yosys_fmax(nextpnr_log: str, report: dict[str, object]) -> str:
    fmax = report.get("fmax")
    if isinstance(fmax, dict) and fmax:
        achieved = []
        for clock in fmax.values():
            if isinstance(clock, dict) and isinstance(clock.get("achieved"), (int, float)):
                achieved.append(float(clock["achieved"]))
        if achieved:
            return f"{min(achieved):.2f} MHz"

    # Log fallback (nextpnr-xilinx has no JSON report). nextpnr prints "Max frequency for clock 'X':
    # Y MHz" once after placement and again after routing; the post-route value is the authoritative
    # one, so keep the last seen per clock and report the slowest clock among them.
    per_clock: dict[str, float] = {}
    for clock_name, value in re.findall(r"Max frequency for clock\s+'([^']+)':\s*([0-9.]+)\s*MHz", nextpnr_log):
        per_clock[clock_name] = float(value)
    if per_clock:
        return f"{min(per_clock.values()):.2f} MHz"

    matches = re.findall(r"Max frequency[^:]*:\s*([0-9.]+)\s*MHz", nextpnr_log)
    if matches:
        return f"{min(float(item) for item in matches):.2f} MHz"
    return "not reported"


def yosys_timing_met(nextpnr_log: str, report: dict[str, object]) -> bool:
    # Report path (nextpnr-ecp5): authoritative per-clock achieved vs constraint from the JSON report.
    fmax = report.get("fmax")
    if isinstance(fmax, dict) and fmax:
        for clock in fmax.values():
            if not isinstance(clock, dict):
                return False
            achieved = clock.get("achieved")
            constraint = clock.get("constraint")
            if not isinstance(achieved, (int, float)) or not isinstance(constraint, (int, float)):
                return False
            if float(achieved) < float(constraint):
                return False
        return True
    # Log path (nextpnr-xilinx / openXC7 has no --report): nextpnr prints
    # "Max frequency for clock 'X': ... (PASS|FAIL at Y MHz)" after placement and again after routing.
    # Keep the post-route (last) verdict per clock and require every clock to pass.
    per_clock: dict[str, str] = {}
    for clock_name, verdict in re.findall(
        r"Max frequency for clock\s+'([^']+)':[^()]*\((PASS|FAIL) at", nextpnr_log
    ):
        per_clock[clock_name] = verdict
    if per_clock:
        return all(verdict == "PASS" for verdict in per_clock.values())
    return False


def parse_yosys_slack(nextpnr_log: str, report: dict[str, object]) -> str:
    fmax = report.get("fmax")
    if isinstance(fmax, dict) and fmax:
        lines = []
        for clock_name, clock in fmax.items():
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


def format_used_available(used: int, available: int | None = None) -> str:
    if available is not None and available > 0:
        return f"{used}/{available} ({100.0 * used / available:.2f}%)"
    return str(used)


def nextpnr_resource(report: dict[str, object], key: str) -> tuple[int, int | None] | None:
    utilization = report.get("utilization")
    if isinstance(utilization, dict):
        item = utilization.get(key)
        if isinstance(item, dict):
            used = item.get("used")
            available = item.get("available")
            if isinstance(used, int):
                return used, available if isinstance(available, int) else None
    return None


def format_nextpnr_resource(
    report: dict[str, object],
    key: str,
    fallback_used: int | None = None,
) -> str:
    resource = nextpnr_resource(report, key)
    if resource is not None:
        return format_used_available(resource[0], resource[1])
    if fallback_used is not None:
        return str(fallback_used)
    return "not reported"


def parse_nextpnr_total_lut4(nextpnr_log: str) -> tuple[int, int | None] | None:
    match = re.search(r"Total LUT4s:\s*([0-9]+)/([0-9]+)", nextpnr_log)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def format_nextpnr_total_lut4(nextpnr_log: str) -> str:
    resource = parse_nextpnr_total_lut4(nextpnr_log)
    if resource is None:
        return "not reported"
    return format_used_available(resource[0], resource[1])


def format_yosys_cell_counts(cells: dict[str, int]) -> str:
    keys = [
        "LUT4",
        "TRELLIS_FF",
        "CCU2C",
        "PFUMX",
        "L6MUX21",
        "MULT18X18D",
        "ALU54B",
        "DP16KD",
    ]
    keys.extend(
        sorted(
            key
            for key, value in cells.items()
            if value and key not in keys and not key.startswith("$")
        )
    )
    lines = [f"{key}: {cells[key]}" for key in keys if cells.get(key, 0)]
    return "\n".join(lines) if lines else "not reported"


def parse_nextpnr_utilization(nextpnr_log: str, report: dict[str, object], util_resources: tuple[str, ...]) -> str:
    lines = []
    total_lut4 = parse_nextpnr_total_lut4(nextpnr_log)
    if total_lut4 is not None:
        lines.append(f"Total LUT4s: {format_used_available(total_lut4[0], total_lut4[1])}")

    utilization = report.get("utilization")
    if isinstance(utilization, dict):
        keys = set(util_resources)
        keys.update(
            key
            for key, item in utilization.items()
            if isinstance(key, str)
            and isinstance(item, dict)
            and isinstance(item.get("used"), int)
            and item["used"] > 0
        )
        for key in sorted(keys):
            resource = nextpnr_resource(report, key)
            if resource is not None:
                lines.append(f"{key}: {format_used_available(resource[0], resource[1])}")
        if lines:
            return "\n".join(lines)

    useful = []
    for line in nextpnr_log.splitlines():
        if any(cell in line for cell in ("TRELLIS_SLICE", "TRELLIS_FF", "LUT4", "PFU", "MULT18X18D", "DP16KD")):
            useful.append(line.strip())
    return "\n".join(useful[-12:]) if useful else "not reported"


def summarize_report_json(report: dict[str, object]) -> str:
    if not report:
        return "nextpnr did not emit a JSON report"
    keys = ", ".join(sorted(report.keys()))
    return f"nextpnr JSON report keys: {keys}"


def synthesize(spec: ModuleSpec, target: YosysTarget, yosys_bin: Path, nextpnr_bin: Path) -> dict[str, str]:
    module_dir = target.build_dir / spec.name
    clean_module_dir(module_dir)

    wrapper = module_dir / f"{spec.name}_wrapper.v"
    yosys_script = module_dir / f"{spec.name}.ys"
    netlist = module_dir / f"{spec.name}.json"
    nextpnr_report = module_dir / f"{spec.name}_nextpnr.json"
    yosys_log = module_dir / "yosys.log"
    nextpnr_log = module_dir / "nextpnr.log"
    schematic_prefix = module_dir / f"{spec.name}_schematic"
    schematic_svg = schematic_prefix.with_suffix(".svg")

    write_wrapper(spec, wrapper)
    write_yosys_script(spec, target, wrapper, netlist, schematic_prefix, yosys_script)

    nextpnr_paths = NextpnrPaths(
        name=spec.name,
        module_dir=module_dir,
        netlist=netlist,
        report=nextpnr_report,
        target_freq_mhz=target.target_freq_mhz,
    )
    run([yosys_bin, "-s", yosys_script], yosys_log, timeout=YOSYS_TIMEOUT_S)
    run([nextpnr_bin, *target.nextpnr_args(target, nextpnr_paths)], nextpnr_log, timeout=NEXTPNR_TIMEOUT_S)

    yosys_text = yosys_log.read_text()
    nextpnr_text = nextpnr_log.read_text()
    report_data = read_nextpnr_report(nextpnr_report)
    cells = read_yosys_cell_counts(netlist, spec.top) or parse_cell_counts(yosys_text)

    result = {
        "name": spec.name,
        "label": spec.label,
        "params": params(spec),
        "register_stages": format_register_stages(register_stages(spec)),
        "fmax": parse_yosys_fmax(nextpnr_text, report_data),
        "target": format_mhz(target.target_freq_mhz),
        "status": "PASS" if yosys_timing_met(nextpnr_text, report_data) else "FAIL",
        "yosys_cells": format_yosys_cell_counts(cells),
        "utilization": parse_nextpnr_utilization(nextpnr_text, report_data, target.util_resources),
        "slack": parse_yosys_slack(nextpnr_text, report_data),
        "json": summarize_report_json(report_data),
        "yosys_log": str(yosys_log.relative_to(target.build_dir)),
        "nextpnr_log": str(nextpnr_log.relative_to(target.build_dir)),
        "nextpnr_json": str(nextpnr_report.relative_to(target.build_dir)),
        "schematic": str(schematic_svg.relative_to(target.build_dir)) if schematic_svg.is_file() else "",
        "group": module_group(spec),
    }
    result.update(target.extract_resources(cells, report_data, nextpnr_text))
    return result


def write_html(target: YosysTarget, results: list[dict[str, str]]) -> None:
    rows = []
    details = []
    generated_at = generated_local_time()
    fmax_bounds = metric_bounds(results, "fmax")
    bounds_by_key = {key: metric_bounds(results, key) for key, _ in target.metric_keys}
    last_group: str | None = None
    for result in results:
        status_class = "pass" if result["status"] == "PASS" else "fail"
        group = result.get("group") or result.get("name", "")
        tr_class = ' class="group-start"' if (last_group is not None and group != last_group) else ""
        last_group = group
        rows.append(
            f"<tr{tr_class}>"
            f"<td>{escape(result['label'])}</td>"
            f"<td>{escape(result['params'])}</td>"
            f"<td>{escape(result['register_stages'])}</td>"
            f"<td>{escape(result['target'])}</td>"
            + metric_cell(result["fmax"], fmax_bounds, higher_is_better=True)
            + f"<td><span class=\"status {status_class}\">{escape(result['status'])}</span></td>"
            + target.resource_row(result, bounds_by_key)
            + "<td>"
            + (artifact_link(result, "schematic", "SVG", new_tab=True) or "—")
            + "</td>"
            + "<td>"
            + joined_links(
                artifact_link(result, "nextpnr_log", "nextpnr"),
                artifact_link(result, "yosys_log", "Yosys"),
                artifact_link(result, "nextpnr_json", "JSON"),
            )
            + "</td>"
            "</tr>"
        )
        details.append(
            f"<h2>{escape(result['label'])}</h2>"
            "<h3>Artifacts</h3>"
            "<p>"
            + joined_links(
                artifact_link(result, "nextpnr_log", "nextpnr log"),
                artifact_link(result, "yosys_log", "Yosys log"),
                artifact_link(result, "nextpnr_json", "nextpnr JSON"),
            )
            + "</p>"
            "<h3>Worst Slack</h3>"
            f"<pre>{escape(result['slack'])}</pre>"
            "<h3>Utilization</h3>"
            f"<pre>{escape(result['utilization'])}</pre>"
            "<h3>Yosys Cell Counts</h3>"
            f"<pre>{escape(result['yosys_cells'])}</pre>"
            "<h3>Report JSON</h3>"
            f"<pre>{escape(result['json'])}</pre>"
        )

    (target.build_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(target.report_title)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #111; }}
table {{ border-collapse: collapse; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #bbb; padding: 0.35rem 0.6rem; text-align: left; }}
th {{ background: #eee; }}
td.resource {{ white-space: nowrap; }}
tbody tr.group-start td {{ border-top: 3px solid #555; }}
.status {{ border-radius: 999px; display: inline-block; font-weight: 700; padding: 0.2rem 0.6rem; }}
.status.pass {{ background: #11823b; color: #fff; }}
.status.fail {{ background: #c82424; color: #fff; }}
pre {{ background: #f6f6f6; border: 1px solid #ddd; padding: 0.8rem; overflow-x: auto; }}
.note {{ background: #fff7e0; border: 1px solid #e3c75a; border-radius: 6px; padding: 0.6rem 0.9rem; max-width: 72rem; }}
.note code {{ background: #f0e6c0; padding: 0 0.25rem; border-radius: 3px; }}
</style>
</head>
<body>
<h1>{escape(target.report_title)}</h1>
"""
        + f"<p>Generated: {escape(generated_at)}</p>"
        + target.flow_description_html
        + target.notes_html
        + """
<p>Each row is measured through a registered synthesis harness: every DUT input is driven by a wrapper register and
every DUT output is captured by a wrapper register. This makes the reported f max a register-to-register limit instead
of ignoring primary-input or primary-output paths. The harness registers are included in utilization numbers, but the
DUT latency column excludes the harness and is computed by the same helper used by the cocotb scoreboards.</p>
<p>Helper-module rows are standalone out-of-context builds. Parent-module rows are flattened and context-optimized, so
helper and parent resource counts are not additive. Schematic links open the pre-techmap generic-cell diagram for the
module in a new tab.</p>
<table>
<thead><tr>
<th>Module</th><th>Parameters</th><th>DUT latency</th><th>Target</th><th>f max</th><th>Status</th>
"""
        + target.resource_headers
        + "<th>Schematic</th><th>Logs</th>"
        + """
</tr></thead>
<tbody>
"""
        + "\n".join(rows)
        + """
</tbody>
</table>
"""
        + "\n".join(details)
        + """
</body>
</html>
"""
    )


def failed_result(spec: ModuleSpec, target: YosysTarget, note: str, module_dir: Path) -> dict[str, str]:
    """
    A FAIL result for a module whose toolchain run crashed, so one bad module cannot abort the flow.

    Used only by the optional (non-gating) path; it links whatever logs were written before the crash and
    fills the device resource columns with the extractor's empty-input defaults so write_html stays happy.
    """
    def rel(name: str) -> str:
        path = module_dir / name
        return str(path.relative_to(target.build_dir)) if path.is_file() else ""

    result = {
        "name": spec.name,
        "label": spec.label,
        "params": params(spec),
        "register_stages": format_register_stages(register_stages(spec)),
        "fmax": "not reported",
        "target": format_mhz(target.target_freq_mhz),
        "status": "FAIL",
        "yosys_cells": "not reported",
        "utilization": note,
        "slack": note,
        "json": note,
        "yosys_log": rel("yosys.log"),
        "nextpnr_log": rel("nextpnr.log"),
        "nextpnr_json": rel(f"{spec.name}_nextpnr.json"),
        "schematic": rel(f"{spec.name}_schematic.svg"),
        "group": module_group(spec),
    }
    result.update(target.extract_resources({}, {}, ""))
    return result


def run_flow(target: YosysTarget, modules: list[ModuleSpec], *, gate: bool = True, optional: bool = False) -> None:
    """
    Run the Yosys + nextpnr flow for one device target.

    gate=True (default) exits nonzero if any module fails or misses timing. optional=True downgrades a
    missing nextpnr binary to a graceful skip and turns a per-module toolchain crash into a FAIL row
    instead of aborting; together gate=False, optional=True make the target safe to run in CI where the
    toolchain may be absent and synthesis must never break the build.
    """
    yosys_bin = require_executable("yosys")
    if optional:
        nextpnr_bin = find_executable(target.nextpnr_tool)
        if nextpnr_bin is None:
            print(
                f"skipping {target.name} synthesis: '{target.nextpnr_tool}' was not found on PATH or under /opt, /usr"
            )
            return
    else:
        nextpnr_bin = require_executable(target.nextpnr_tool)

    def synthesize_module(spec: ModuleSpec) -> dict[str, str]:
        if not optional:
            return synthesize(spec, target, yosys_bin, nextpnr_bin)
        try:
            return synthesize(spec, target, yosys_bin, nextpnr_bin)
        except subprocess.CalledProcessError as exc:
            note = f"toolchain command failed (exit {exc.returncode}); see logs"
            return failed_result(spec, target, note, target.build_dir / spec.name)
        except Exception as exc:  # noqa: BLE001 -- optional flow must survive any single-module failure
            return failed_result(spec, target, f"synthesis raised {type(exc).__name__}: {exc}", target.build_dir / spec.name)

    target.build_dir.mkdir(parents=True, exist_ok=True)
    results = synthesize_with_progress("yosys", modules, synthesize_module)
    write_html(target, results)
    report_path = target.build_dir / "index.html"
    print(f"wrote {report_path}")
    print_run_summary("yosys", results, target.area_key, target.area_label)
    if gate:
        require_passing_results("Yosys", results, report_path)


if __name__ == "__main__":
    raise SystemExit("yosys.py is a library module; run yosys_ecp5.py (or another yosys_<device>.py).")

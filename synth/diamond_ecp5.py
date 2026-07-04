#!/usr/bin/env python3
"""
Lattice Diamond LSE synthesis of the float modules, targeting the Lattice ECP5.

Diamond is Lattice-only, so unlike the Yosys flow there is no second-device profile to abstract; the whole flow lives
here. The flow uses LSE exclusively and shares the module catalog, harness generators, and HTML/plumbing toolkit with
the rest of the suite via modules / wrappers / common.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import argparse
import os
import re
import shlex
from common import (
    REPO,
    artifact_link,
    clean_module_dir,
    find_executable,
    format_mhz,
    generated_local_time,
    joined_links,
    metric_bounds,
    metric_cell,
    print_run_summary,
    read_text,
    relative_or_missing,
    require_passing_results,
    run,
    synthesize_with_progress,
)
from modules import (
    ModuleSpec,
    flow_modules,
    format_register_stages,
    module_group,
    params,
    register_stages,
    rtl_sources,
)
from wrappers import write_wrapper

DIAMOND_BUILD = REPO / "build" / "float_synth_diamond_ecp5"
DIAMOND_DEVICE = "LFE5U-12F-6BG381C"
DIAMOND_TARGET_FREQ_MHZ = float(os.environ.get("DIAMOND_TARGET_FREQ_MHZ", "100"))
DIAMOND_ROUTE_PASSES = int(os.environ.get("DIAMOND_ROUTE_PASSES", "3"))
DIAMOND_PAR_EFFORT = int(os.environ.get("DIAMOND_PAR_EFFORT", "3"))
DIAMOND_ROUTER = os.environ.get("DIAMOND_ROUTER", "NBR")


@dataclass(frozen=True)
class DiamondTools:
    diamond: Path
    diamond_env: Path | None


@dataclass(frozen=True)
class DiamondReportPaths:
    twr: Path | None
    lse_twr: Path | None
    mrp: Path | None
    par: Path | None


def resolve_diamond() -> tuple[DiamondTools | None, str]:
    diamond = find_executable("diamond")
    if diamond is None:
        return None, "diamond executable was not found"

    diamond = diamond.resolve()
    diamond_env = diamond.parent / "diamond_env"
    pnmainc = find_executable("pnmainc")
    if not diamond_env.is_file() and pnmainc is None:
        return None, f"neither {diamond_env} nor pnmainc is available"

    return DiamondTools(diamond=diamond, diamond_env=diamond_env if diamond_env.is_file() else None), ""


def project_name(spec: ModuleSpec) -> str:
    return (spec.name.lstrip("_") or spec.name.replace("_", "")) + "_diamond"


def path_for_xml(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def xml_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def write_diamond_lpf(path: Path) -> None:
    path.write_text(f"""BLOCK RESETPATHS ;
BLOCK ASYNCPATHS ;
USE PRIMARY NET "clk_c" ;
FREQUENCY NET "clk_c" {DIAMOND_TARGET_FREQ_MHZ:.6f} MHz ;
""")


def write_diamond_strategy(path: Path) -> None:
    properties = {
        "PROP_LST_CarryChain": "True",
        "PROP_LST_CarryChainLength": "0",
        "PROP_LST_DSPStyle": "DSP",
        "PROP_LST_DSPUtil": "100",
        "PROP_LST_EBRUtil": "100",
        "PROP_LST_EdfFrequency": f"{DIAMOND_TARGET_FREQ_MHZ:.0f}",
        "PROP_LST_FIXGATEDCLKS": "True",
        "PROP_LST_FSMEncodeStyle": "Auto",
        "PROP_LST_ForceGSRInfer": "Auto",
        "PROP_LST_IOInsertion": "True",
        "PROP_LST_LoopLimit": "1950",
        "PROP_LST_MaxFanout": "1000",
        "PROP_LST_MuxStyle": "Auto",
        "PROP_LST_NumCriticalPaths": "10",
        "PROP_LST_OptimizeGoal": "Timing",
        "PROP_LST_PropagatConst": "True",
        "PROP_LST_RAMStyle": "Auto",
        "PROP_LST_ROMStyle": "EBR",
        "PROP_LST_RemoveDupRegs": "True",
        "PROP_LST_ResourceShare": "True",
        "PROP_LST_UseIOReg": "Auto",
        "PROP_LST_UseLPF": "True",
        "PROP_MAPSTA_AnalysisOption": "Standard Setup and Hold Analysis",
        "PROP_MAPSTA_AutoTiming": "True",
        "PROP_MAPSTA_CheckUnconstrainedConns": "False",
        "PROP_MAPSTA_CheckUnconstrainedPaths": "False",
        "PROP_MAPSTA_NumUnconstrainedPaths": "0",
        "PROP_MAPSTA_ReportStyle": "Verbose Timing Report",
        "PROP_MAP_MAPIORegister": "Auto",
        "PROP_MAP_MAPInferGSR": "True",
        "PROP_MAP_RegRetiming": "False",
        "PROP_MAP_TimingDriven": "True",
        "PROP_MAP_TimingDrivenNodeRep": "True",
        "PROP_MAP_TimingDrivenPack": "True",
        "PROP_PARSTA_AnalysisOption": "Standard Setup and Hold Analysis",
        "PROP_PARSTA_AutoTiming": "True",
        "PROP_PARSTA_CheckUnconstrainedConns": "False",
        "PROP_PARSTA_CheckUnconstrainedPaths": "False",
        "PROP_PARSTA_NumUnconstrainedPaths": "0",
        "PROP_PARSTA_ReportStyle": "Verbose Timing Report",
        "PROP_PARSTA_SpeedForHoldAnalysis": "m",
        "PROP_PARSTA_SpeedForSetupAnalysis": "default",
        "PROP_PARSTA_WordCasePaths": "10",
        "PROP_PAR_DisableTDParDes": "False",
        "PROP_PAR_EffortParDes": str(DIAMOND_PAR_EFFORT),
        "PROP_PAR_MultiSeedSortMode": "Worst Slack",
        "PROP_PAR_NewRouteParDes": DIAMOND_ROUTER,
        "PROP_PAR_PARClockSkew": "Off",
        "PROP_PAR_PlcIterParDes": "5",
        "PROP_PAR_PlcStCostTblParDes": "1",
        "PROP_PAR_PrefErrorOut": "False",
        "PROP_PAR_RoutePassParDes": str(DIAMOND_ROUTE_PASSES),
        "PROP_PAR_RoutingCDP": "Auto",
        "PROP_PAR_RoutingCDR": "1",
        "PROP_PAR_RunParWithTrce": "True",
        "PROP_PAR_RunTimeReduction": "False",
        "PROP_PAR_SaveBestRsltParDes": "1",
        "PROP_PAR_StopZero": "False",
        "PROP_PAR_parHold": "On",
        "PROP_PAR_parPathBased": "On",
        "PROP_SYN_EdfArea": "False",
        "PROP_SYN_EdfFrequency": f"{DIAMOND_TARGET_FREQ_MHZ:.0f}",
        "PROP_SYN_EdfGSR": "False",
        "PROP_SYN_EdfInsertIO": "False",
        "PROP_SYN_EdfRunRetiming": "Pipelining and Retiming",
        "PROP_SYN_EdfVerilogInput": "Verilog 2001",
        "PROP_SYN_UseLPF": "True",
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE strategy>",
        '<Strategy version="1.0" predefined="0" description="" label="DiamondLseMaxTiming">',
    ]
    lines.extend(
        f'    <Property name="{xml_attr(name)}" value="{xml_attr(value)}" time="0"/>'
        for name, value in properties.items()
    )
    lines.append("</Strategy>")
    path.write_text("\n".join(lines) + "\n")


def write_diamond_ldf(
    spec: ModuleSpec,
    wrapper: Path,
    lpf: Path,
    sty: Path,
    ldf: Path,
) -> None:
    project_dir = ldf.parent
    sources = [wrapper] + rtl_sources(spec)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<BaliProject version="3.2" title="{xml_attr(project_name(spec))}" '
            f'device="{xml_attr(DIAMOND_DEVICE)}" default_implementation="impl1">'
        ),
        "    <Options/>",
        '    <Implementation title="impl1" dir="impl1" description="impl1" synthesis="lse" default_strategy="Strategy1">',
        f'        <Options def_top="{xml_attr(spec.top)}">',
        f'            <Option name="top" value="{xml_attr(spec.top)}"/>',
        "        </Options>",
    ]
    for source in sources:
        rel = xml_attr(path_for_xml(source, project_dir))
        if source == wrapper:
            lines.extend(
                [
                    f'        <Source name="{rel}" type="Verilog" type_short="Verilog">',
                    f'            <Options top_module="{xml_attr(spec.top)}"/>',
                    "        </Source>",
                ]
            )
        else:
            lines.extend(
                [
                    f'        <Source name="{rel}" type="Verilog" type_short="Verilog">',
                    "            <Options/>",
                    "        </Source>",
                ]
            )
    lines.extend(
        [
            f'        <Source name="{xml_attr(path_for_xml(lpf, project_dir))}" '
            'type="Logic Preference" type_short="LPF">',
            "            <Options/>",
            "        </Source>",
            "    </Implementation>",
            f'    <Strategy name="Strategy1" file="{xml_attr(path_for_xml(sty, project_dir))}"/>',
            "</BaliProject>",
        ]
    )
    ldf.write_text("\n".join(lines) + "\n")


def write_diamond_tcl(project_file: Path, tcl: Path) -> None:
    project = str(project_file).replace("\\", "/")
    tcl.write_text(f"""proc fail {{message}} {{
    puts stderr $message
    exit 1
}}
if {{[catch {{prj_project open "{project}"}} result]}} {{
    fail $result
}}
if {{[catch {{prj_run PAR -impl impl1 -forceAll}} result]}} {{
    catch {{prj_project close}}
    fail $result
}}
if {{[catch {{prj_project close}} result]}} {{
    fail $result
}}
exit 0
""")


def run_diamond_console(tools: DiamondTools, tcl: Path, log_path: Path) -> None:
    bindir = shlex.quote(str(tools.diamond.parent))
    env_path = shlex.quote(str(tools.diamond_env)) if tools.diamond_env is not None else ""
    source_env = f"source {env_path}" if env_path else ":"
    script = f"""
set -euo pipefail
bindir={bindir}
export PATH="$bindir:$PATH"
set +u
{source_env}
set -u
command -v pnmainc >/dev/null 2>&1 || {{
    echo "error: Diamond Tcl console 'pnmainc' was not found" >&2
    exit 1
}}
pnmainc < {shlex.quote(str(tcl))}
"""
    run(["bash", "-lc", script], log_path)


def find_diamond_report_paths(module_dir: Path) -> DiamondReportPaths:
    impl_dir = module_dir / "impl1"
    twrs = sorted(impl_dir.glob("*.twr"))
    lse_twrs = [path for path in twrs if path.name.endswith("_lse.twr")]
    post_twrs = [path for path in twrs if path not in lse_twrs]
    return DiamondReportPaths(
        twr=post_twrs[-1] if post_twrs else None,
        lse_twr=lse_twrs[-1] if lse_twrs else None,
        mrp=next(iter(sorted(impl_dir.glob("*.mrp"))), None),
        par=next(iter(sorted(impl_dir.glob("*.par"))), None),
    )


def parse_diamond_fmax_mhz(twr_text: str) -> float | None:
    matches = re.findall(
        r"(?:Report|Warning):\s+([0-9.]+)\s*MHz is the maximum frequency",
        twr_text,
        re.IGNORECASE,
    )
    if matches:
        return min(float(match) for match in matches)
    return None


def parse_diamond_timing_errors(twr_text: str) -> tuple[int | None, int | None]:
    match = re.search(r"Timing errors:\s+([0-9]+)\s+\(setup\),\s+([0-9]+)\s+\(hold\)", twr_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"Timing errors:\s+([0-9]+)\s+Score:", twr_text)
    if match:
        return int(match.group(1)), None
    return None, None


def parse_diamond_resource_counts(pattern: str, text: str) -> tuple[int | None, int | None]:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def format_diamond_resource(counts: tuple[int | None, int | None]) -> str:
    used, available = counts
    if used is None or available is None:
        return "not reported"
    return f"{used}/{available} ({100.0 * used / available:.2f}%)"


def parse_diamond_resource(pattern: str, text: str) -> str:
    return format_diamond_resource(parse_diamond_resource_counts(pattern, text))


def parse_diamond_par_summary(par_text: str, key: str) -> int | None:
    match = re.search(rf"PAR_SUMMARY::{re.escape(key)}\s*=\s*([0-9]+)", par_text)
    return int(match.group(1)) if match else None


def format_optional_fmax(value: float | None) -> str:
    return f"{value:.2f} MHz" if value is not None else "not reported"


def format_diamond_slack_from_fmax(fmax_mhz: float | None) -> str:
    if fmax_mhz is None or fmax_mhz <= 0.0:
        return "not reported"
    slack_ns = (1000.0 / DIAMOND_TARGET_FREQ_MHZ) - (1000.0 / fmax_mhz)
    return f"{slack_ns:.3f} ns"


def synthesize_diamond(spec: ModuleSpec, tools: DiamondTools) -> dict[str, str]:
    module_dir = DIAMOND_BUILD / spec.name
    clean_module_dir(module_dir)

    wrapper = module_dir / f"{spec.name}_wrapper.v"
    lpf = module_dir / f"{project_name(spec)}.lpf"
    sty = module_dir / f"{project_name(spec)}.sty"
    ldf = module_dir / f"{project_name(spec)}.ldf"
    tcl = module_dir / "run_diamond.tcl"
    diamond_log = module_dir / "diamond.log"

    write_wrapper(spec, wrapper)
    write_diamond_lpf(lpf)
    write_diamond_strategy(sty)
    write_diamond_ldf(spec, wrapper, lpf, sty, ldf)
    write_diamond_tcl(ldf, tcl)
    run_diamond_console(tools, tcl, diamond_log)

    reports = find_diamond_report_paths(module_dir)
    twr_text = read_text(reports.twr)
    mrp_text = read_text(reports.mrp)
    par_text = read_text(reports.par)
    fmax = parse_diamond_fmax_mhz(twr_text)
    setup_errors, hold_errors = parse_diamond_timing_errors(twr_text)
    unrouted = parse_diamond_par_summary(par_text, "Number of unrouted conns")
    par_errors = parse_diamond_par_summary(par_text, "Number of errors")
    bram_counts = parse_diamond_resource_counts(r"Number of block RAMs:\s+([0-9]+) out of ([0-9]+)", mrp_text)
    route_clean = unrouted in {None, 0} and par_errors in {None, 0}
    timing_clean = (
        fmax is not None and fmax >= DIAMOND_TARGET_FREQ_MHZ and setup_errors in {None, 0} and hold_errors in {None, 0}
    )

    return {
        "name": spec.name,
        "label": spec.label,
        "params": params(spec),
        "register_stages": format_register_stages(register_stages(spec)),
        "target": format_mhz(DIAMOND_TARGET_FREQ_MHZ),
        "fmax": format_optional_fmax(fmax),
        "slack": format_diamond_slack_from_fmax(fmax),
        "status": "PASS" if route_clean and timing_clean else "FAIL",
        "registers": parse_diamond_resource(r"Number of registers:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "lut4": parse_diamond_resource(r"Number of LUT4s:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "bram": format_diamond_resource(bram_counts),
        "dsp_mult": parse_diamond_resource(r"Number of Used DSP MULT Sites:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "slice": parse_diamond_resource(r"SLICE\s+([0-9]+)/([0-9]+)", par_text),
        "pio": parse_diamond_resource(r"PIO \(prelim\)\s+([0-9]+)/([0-9]+)", par_text),
        "diamond_log": relative_or_missing(diamond_log, DIAMOND_BUILD),
        "twr": relative_or_missing(reports.twr, DIAMOND_BUILD),
        "lse_twr": relative_or_missing(reports.lse_twr, DIAMOND_BUILD),
        "mrp": relative_or_missing(reports.mrp, DIAMOND_BUILD),
        "par": relative_or_missing(reports.par, DIAMOND_BUILD),
        "ldf": relative_or_missing(ldf, DIAMOND_BUILD),
        "sty": relative_or_missing(sty, DIAMOND_BUILD),
        "lpf": relative_or_missing(lpf, DIAMOND_BUILD),
        "setup_errors": "not reported" if setup_errors is None else str(setup_errors),
        "hold_errors": "not reported" if hold_errors is None else str(hold_errors),
        "unrouted": "not reported" if unrouted is None else str(unrouted),
        "par_errors": "not reported" if par_errors is None else str(par_errors),
        "group": module_group(spec),
    }


def write_diamond_html(results: list[dict[str, str]]) -> None:
    rows = []
    details = []
    generated_at = generated_local_time()
    fmax_bounds = metric_bounds(results, "fmax")
    slice_bounds = metric_bounds(results, "slice")
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
            + f"<td>{escape(result['slack'])}</td>"
            + f"<td><span class=\"status {status_class}\">{escape(result['status'])}</span></td>"
            + f"<td class=\"resource\">{escape(result['lut4'])}</td>"
            + f"<td class=\"resource\">{escape(result['registers'])}</td>"
            + f"<td class=\"resource\">{escape(result['bram'])}</td>"
            + f"<td class=\"resource\">{escape(result['dsp_mult'])}</td>"
            + metric_cell(result["slice"], slice_bounds, higher_is_better=False, class_name="resource")
            + f"<td class=\"resource\">{escape(result['pio'])}</td>"
            + "<td>"
            + joined_links(
                artifact_link(result, "twr", "TRACE"),
                artifact_link(result, "par", "PAR"),
                artifact_link(result, "mrp", "MAP"),
                artifact_link(result, "diamond_log", "Diamond"),
            )
            + "</td>"
            "</tr>"
        )
        details.append(
            f"<h2>{escape(result['label'])}</h2>"
            "<h3>Status</h3>"
            "<pre>"
            f"setup errors: {escape(result['setup_errors'])}\n"
            f"hold errors:  {escape(result['hold_errors'])}\n"
            f"unrouted:     {escape(result['unrouted'])}\n"
            f"PAR errors:   {escape(result['par_errors'])}"
            "</pre>"
            "<h3>Artifacts</h3>"
            "<p>"
            + joined_links(
                artifact_link(result, "ldf", "project"),
                artifact_link(result, "sty", "strategy"),
                artifact_link(result, "lpf", "preferences"),
                artifact_link(result, "twr", "TRACE"),
                artifact_link(result, "lse_twr", "LSE timing"),
                artifact_link(result, "par", "PAR"),
                artifact_link(result, "mrp", "MAP"),
                artifact_link(result, "diamond_log", "Diamond log"),
            )
            + "</p>"
        )

    (DIAMOND_BUILD / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kulibin Float Diamond Synthesis Report</title>
<style>
body { font-family: sans-serif; margin: 2rem; color: #111; }
table { border-collapse: collapse; margin-bottom: 2rem; }
th, td { border: 1px solid #bbb; padding: 0.35rem 0.6rem; text-align: left; }
th { background: #eee; }
td.resource { white-space: nowrap; }
tbody tr.group-start td { border-top: 3px solid #555; }
.status { border-radius: 999px; display: inline-block; font-weight: 700; padding: 0.2rem 0.6rem; }
.status.pass { background: #11823b; color: #fff; }
.status.fail { background: #c82424; color: #fff; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 0.8rem; overflow-x: auto; }
</style>
</head>
<body>
<h1>Kulibin Float Diamond Synthesis Report</h1>
"""
        + f"<p>Generated: {escape(generated_at)}</p>"
        + f"<p>Flow: Lattice Diamond LSE ({escape(DIAMOND_DEVICE)}) at "
        + f"{format_mhz(DIAMOND_TARGET_FREQ_MHZ)}. Synthesis optimization goal is Timing, "
        + f"MAP register retiming is disabled, PAR placement effort is {DIAMOND_PAR_EFFORT} with 5 placement seeds, "
        + f"router is {escape(DIAMOND_ROUTER)}, and routing passes are {DIAMOND_ROUTE_PASSES}.</p>"
        + """
<p>Each row is measured through a registered synthesis harness: every DUT input is driven by a wrapper register and
every DUT output is captured by a wrapper register. This makes the reported f max a register-to-register limit instead
of ignoring primary-input or primary-output paths. The harness registers are included in utilization numbers, but the
DUT latency column excludes the harness and is computed by the same helper used by the cocotb scoreboards.</p>
<p>Helper-module rows are standalone out-of-context builds. Parent-module rows are flattened and context-optimized, so
helper and parent resource counts are not additive.</p>
<table>
<thead><tr>
<th>Module</th><th>Parameters</th><th>DUT latency</th><th>Target</th><th>f max</th><th>Slack</th><th>Status</th>
<th>LUT4</th><th>Registers</th><th>Block RAM</th><th>DSP MULT Sites</th><th>Slice</th><th>PIO</th><th>Logs</th>
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


def run_diamond_flow(modules: list[ModuleSpec]) -> bool:
    tools, reason = resolve_diamond()
    if tools is None:
        print(f"skipping Diamond synthesis: {reason}")
        return False

    DIAMOND_BUILD.mkdir(parents=True, exist_ok=True)
    results = synthesize_with_progress("diamond", modules, lambda spec: synthesize_diamond(spec, tools))
    write_diamond_html(results)
    report_path = DIAMOND_BUILD / "index.html"
    print(f"wrote {report_path}")
    print_run_summary("diamond", results, "slice", "slices")
    require_passing_results("Diamond", results, report_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modules",
        help="comma-separated module names to synthesize; defaults to all configured float modules",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diamond_flow(flow_modules(args.modules, "DIAMOND_MODULES"))


if __name__ == "__main__":
    main()

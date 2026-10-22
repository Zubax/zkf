"""The Lattice Diamond flow on the ECP5: synthesis by LSE, then map, placement and routing once."""

import os
import re
import shlex
import shutil
from pathlib import Path

from _harness import FREQ_MHZ, Design, Row, Sources, run

TOOLS = ("diamond",)
DEVICE, LARGE_DEVICE = "LFE5U-12F-{grade}BG381C", "LFE5U-45F-{grade}BG381C"
SPEED_GRADE, FASTER_GRADE = 6, 7
ROUTE_PASSES = 3
PAR_EFFORT = 3
ROUTER = "NBR"


PROJECT = "spec"
"""The project's name, which Diamond repeats in every file it writes: short, since its map refuses long paths and the
spec's directory already names it."""


def part(design: Design) -> str:
    return (LARGE_DEVICE if design.large else DEVICE).format(grade=FASTER_GRADE if design.faster else SPEED_GRADE)


def xml_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def preferences() -> str:
    return f"""BLOCK RESETPATHS ;
BLOCK ASYNCPATHS ;
USE PRIMARY NET "clk_c" ;
FREQUENCY NET "clk_c" {FREQ_MHZ:.6f} MHz ;
"""


def strategy() -> str:
    properties = {
        "PROP_LST_CarryChain": "True",
        "PROP_LST_CarryChainLength": "0",
        "PROP_LST_DSPStyle": "DSP",
        "PROP_LST_DSPUtil": "100",
        "PROP_LST_EBRUtil": "100",
        "PROP_LST_EdfFrequency": f"{FREQ_MHZ:.0f}",
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
        "PROP_PAR_EffortParDes": str(PAR_EFFORT),
        "PROP_PAR_MultiSeedSortMode": "Worst Slack",
        "PROP_PAR_NewRouteParDes": ROUTER,
        "PROP_PAR_PARClockSkew": "Off",
        "PROP_PAR_PlcIterParDes": "1",
        "PROP_PAR_PlcStCostTblParDes": "1",
        "PROP_PAR_PrefErrorOut": "False",
        "PROP_PAR_RoutePassParDes": str(ROUTE_PASSES),
        "PROP_PAR_RoutingCDP": "Auto",
        "PROP_PAR_RoutingCDR": "1",
        "PROP_PAR_RunParWithTrce": "True",
        "PROP_PAR_RunTimeReduction": "False",
        "PROP_PAR_SaveBestRsltParDes": "1",
        "PROP_PAR_StopZero": "False",
        "PROP_PAR_parHold": "On",
        "PROP_PAR_parPathBased": "On",
        "PROP_SYN_EdfArea": "False",
        "PROP_SYN_EdfFrequency": f"{FREQ_MHZ:.0f}",
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
    lines += [f'    <Property name="{xml_attr(k)}" value="{xml_attr(v)}" time="0"/>' for k, v in properties.items()]
    lines.append("</Strategy>")
    return "\n".join(lines) + "\n"


def project(design: Design, sources: Sources, lpf: Path, sty: Path) -> str:
    """The project, in its directory, which the paths in it are relative to; the harness is the top."""

    def relative(path: Path) -> str:
        return xml_attr(os.path.relpath(path, lpf.parent).replace(os.sep, "/"))

    top = xml_attr(design.top)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<BaliProject version="3.2" title="{PROJECT}" '
            f'device="{xml_attr(part(design))}" default_implementation="impl1">'
        ),
        "    <Options/>",
        '    <Implementation title="impl1" dir="impl1" description="impl1" synthesis="lse"'
        ' default_strategy="Strategy1">',
        f'        <Options def_top="{top}">',
        f'            <Option name="top" value="{top}"/>',
        "        </Options>",
        f'        <Source name="{relative(sources.harness)}" type="Verilog" type_short="Verilog">',
        f'            <Options top_module="{top}"/>',
        "        </Source>",
        f'        <Source name="{relative(sources.rtl)}" type="Verilog" type_short="Verilog">',
        "            <Options/>",
        "        </Source>",
        f'        <Source name="{relative(lpf)}" type="Logic Preference" type_short="LPF">',
        "            <Options/>",
        "        </Source>",
        "    </Implementation>",
        f'    <Strategy name="Strategy1" file="{relative(sty)}"/>',
        "</BaliProject>",
    ]
    return "\n".join(lines) + "\n"


def tcl(ldf: Path) -> str:
    return f"""proc fail {{message}} {{
    puts stderr $message
    exit 1
}}
if {{[catch {{prj_project open "{str(ldf).replace("\\", "/")}"}} result]}} {{
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
"""


def console(tcl: Path) -> str:
    """The shell that runs the Tcl console of Diamond, with the environment that the installation gives it."""
    diamond = Path(shutil.which("diamond") or "diamond").resolve()
    env = diamond.parent / "diamond_env"
    return f"""
set -euo pipefail
bindir={shlex.quote(str(diamond.parent))}
export PATH="$bindir:$PATH"
set +u
{f"source {shlex.quote(str(env))}" if env.is_file() else ":"}
set -u
command -v pnmainc >/dev/null 2>&1 || {{
    echo "error: Diamond Tcl console 'pnmainc' was not found" >&2
    exit 1
}}
pnmainc < {shlex.quote(str(tcl))}
"""


def synthesize(design: Design, directory: Path) -> Row:
    """Writes the spec into the directory, cleared first, runs the flow and reads the row from what it left."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    stem = directory / PROJECT
    lpf, sty, ldf = stem.with_suffix(".lpf"), stem.with_suffix(".sty"), stem.with_suffix(".ldf")
    sources = design.write(directory)
    lpf.write_text(preferences())
    sty.write_text(strategy())
    ldf.write_text(project(design, sources, lpf, sty))
    (directory / "run_diamond.tcl").write_text(tcl(ldf))
    run(["bash", "-lc", console(directory / "run_diamond.tcl")], directory / "diamond.log")

    impl = directory / "impl1"
    twrs = sorted(impl.glob("*.twr"))
    lse_twrs = [path for path in twrs if path.name.endswith("_lse.twr")]
    post_twrs = [path for path in twrs if path not in lse_twrs]
    twr = post_twrs[-1] if post_twrs else None
    mrp, par = next(iter(sorted(impl.glob("*.mrp"))), None), next(iter(sorted(impl.glob("*.par"))), None)
    twr_text, mrp_text, par_text = (path.read_text(errors="replace") if path else "" for path in (twr, mrp, par))

    fmax = parse_fmax(twr_text)
    setup_errors, hold_errors = parse_timing_errors(twr_text)
    unrouted = par_summary(par_text, "Number of unrouted conns")
    par_errors = par_summary(par_text, "Number of errors")
    route_clean = unrouted == 0 and par_errors == 0
    timing_clean = fmax is not None and fmax >= FREQ_MHZ and setup_errors == 0 and hold_errors == 0

    def link(path: Path | None) -> str:
        return "" if path is None else str(path.relative_to(directory.parent))

    def counted(value: int | None) -> str:
        return "not reported" if value is None else str(value)

    return design.row | {
        "target": f"{FREQ_MHZ:g} MHz",
        "fmax": "not reported" if fmax is None else f"{fmax:.2f} MHz",
        "slack": "not reported" if fmax is None or fmax <= 0 else f"{1000 / FREQ_MHZ - 1000 / fmax:.3f} ns",
        "status": "PASS" if route_clean and timing_clean else "FAIL",
        "registers": parse_resource(r"Number of registers:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "lut4": parse_resource(r"Number of LUT4s:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "lut4_ram": parse_count(r"Number used as distributed RAM:\s+([0-9]+)", mrp_text),
        "bram": parse_resource(r"Number of block RAMs:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "dsp_mult": parse_resource(r"Number of Used DSP MULT Sites:\s+([0-9]+) out of ([0-9]+)", mrp_text),
        "slice": parse_resource(r"SLICE\s+([0-9]+)/([0-9]+)", par_text),
        "pio": parse_resource(r"PIO \(prelim\)\s+([0-9]+)/([0-9]+)", par_text),
        "diamond_log": link(directory / "diamond.log"),
        "twr": link(twr),
        "lse_twr": link(lse_twrs[-1] if lse_twrs else None),
        "mrp": link(mrp),
        "par": link(par),
        "ldf": link(ldf),
        "sty": link(sty),
        "lpf": link(lpf),
        "setup_errors": counted(setup_errors),
        "hold_errors": counted(hold_errors),
        "unrouted": counted(unrouted),
        "par_errors": counted(par_errors),
    }


def parse_fmax(twr_text: str) -> float | None:
    matches = re.findall(r"(?:Report|Warning):\s+([0-9.]+)\s*MHz is the maximum frequency", twr_text, re.IGNORECASE)
    return min(float(match) for match in matches) if matches else None


def parse_timing_errors(twr_text: str) -> tuple[int | None, int | None]:
    match = re.search(r"Timing errors:\s+([0-9]+)\s+\(setup\),\s+([0-9]+)\s+\(hold\)", twr_text)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def parse_resource(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return "not reported"
    used, available = int(match.group(1)), int(match.group(2))
    return f"{used}/{available} ({100.0 * used / available:.2f}%)"


def parse_count(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else "not reported"


def par_summary(par_text: str, key: str) -> int | None:
    match = re.search(rf"PAR_SUMMARY::{re.escape(key)}\s*=\s*([0-9]+)", par_text)
    return int(match.group(1)) if match else None

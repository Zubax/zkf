#!/usr/bin/env python3
"""
Merge Verilator coverage, generate an HTML report, and optionally gate uncovered ZKF RTL coverage.

Three dimensions, all emitted by Verilator:

  * v_line   - basic-block / line coverage (--coverage-line instruments per basic block)
  * v_branch - branch coverage (each if/else/case arm)
  * v_toggle - net-bit toggle coverage (0->1 and 1->0 per bit)

Gating modes:

  * --gate (per-PR): fail on any uncovered LINE point only.
  * --full (deep): fail on any uncovered LINE or BRANCH point. TOGGLE is advisory and never fatal.

Points are merged by (file, page-type, line, net) -- dropping the parameter mangling and the hierarchy instance,
taking the max hit count across every coverage.dat -- so a point counts as covered if any configuration in any
instantiation hit it (the net/o field is parameter- and instance-independent). This is the right semantic for
a reusable library: each piece of RTL must be exercised somewhere (a shared submodule via its own standalone test), and
it lets a diverse matrix close toggle coverage that no single format reaches alone.

Genuinely-unreachable points are suppressed at the source with Verilator's // verilator coverage_off /
coverage_on pragmas; there is deliberately no external waiver list (a list of line numbers rots as the RTL changes,
whereas in-source pragmas move with the code).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
RTL_DIR = REPO_ROOT / "zkf" / "rtl"
TB_DIR = REPO_ROOT / "tb"

_RECORD = re.compile(r"^C '(.*)' (\d+)\s*$")

# Top-level DUT ports excluded from the TOGGLE gate: primary I/O is testbench-driven (stimulus, not DUT logic) and
# outputs are covered by line+branch, so their toggle reflects stimulus. Standard policy. These bare names are never
# internal-net names here, so internal nets stay gated. (pack's data inputs are NOT excluded -- sim_pack covers them.)
_TB_DRIVEN_PORTS = {"clk", "rst", "in_valid", "out_valid", "a", "b", "x", "y", "in", "op_sub", "shamt"}


def is_zkf_source(path_text: str) -> bool:
    path = Path(path_text)
    return path.parent.name == "rtl" and path.name.endswith(".v") and path.name.startswith(("zkf_", "_zkf_"))


def normalized_source(path_text: str) -> str:
    """
    Rewrite an SF: path from the per-run staged copy to one that exists in the workspace, so genhtml can find the
    source (Verilator emits build-CWD-relative paths that don't resolve after the build dir is cleaned).
    """
    path = Path(path_text)
    if path.parent.name == "rtl":
        source = RTL_DIR / path.name
        if source.is_file():
            return str(source)
    if path.parent.name == "_tables":
        source = RTL_DIR / "_tables" / path.name
        if source.is_file():
            return str(source)
    if path.parent.name == "tb":
        source = TB_DIR / path.name
        if source.is_file():
            return str(source)
    return path_text


def normalize_info_sources(info_path: Path) -> None:
    lines = []
    with info_path.open() as fp:
        for raw in fp:
            if raw.startswith("SF:"):
                lines.append(f"SF:{normalized_source(raw[3:].strip())}\n")
            else:
                lines.append(raw)
    info_path.write_text("".join(lines), encoding="utf-8")


def merge_coverage(build_dir: Path, output_dir: Path) -> Path:
    tool = shutil.which("verilator_coverage")
    if tool is None:
        raise RuntimeError("verilator_coverage is not on PATH")

    dat_files = sorted(build_dir.rglob("coverage.dat"))
    if not dat_files:
        raise RuntimeError(f"no coverage.dat files found under {build_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "merged.info"
    subprocess.run(
        [tool, "--write-info", str(info_path), *(str(path) for path in dat_files)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    normalize_info_sources(info_path)
    return info_path


# Raw coverage.dat parsing for branch and toggle points.


@dataclass(frozen=True)
class Point:
    file: str
    ptype: str  # "v_line" | "v_branch" | "v_toggle"
    line: int
    net: str  # the 'o' field: parameter- and instance-independent


def _parse_fields(key: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in key.split("\x01"):
        name, sep, value = part.partition("\x02")
        if sep:
            fields[name] = value
    return fields


def merged_points(build_dir: Path) -> dict[Point, int]:
    """Merge coverage.dat by parameter-independent point identity (max hit count); covered iff hit by >=1 config."""
    merged: dict[Point, int] = defaultdict(int)
    for dat in sorted(build_dir.rglob("coverage.dat")):
        with dat.open(encoding="latin-1") as fp:
            for raw in fp:
                m = _RECORD.match(raw)
                if not m:
                    continue
                fields = _parse_fields(m.group(1))
                count = int(m.group(2))
                src = fields.get("f", "")
                if not is_zkf_source(src):
                    continue
                page = fields.get("page", "")
                ptype = page.split("/", 1)[0]
                if ptype not in ("v_line", "v_branch", "v_toggle"):
                    continue
                try:
                    line = int(fields.get("l", "0"))
                except ValueError:
                    line = 0
                net = fields.get("o", "")
                # Exclude testbench-driven top-level ports from the toggle gate (see _TB_DRIVEN_PORTS).
                if ptype == "v_toggle" and net.split(":", 1)[0].split("[", 1)[0] in _TB_DRIVEN_PORTS:
                    continue
                point = Point(file=Path(src).name, ptype=ptype, line=line, net=net)
                if count > merged[point]:
                    merged[point] = count
    return dict(merged)


@dataclass
class Stats:
    total: int = 0
    covered: int = 0
    uncovered: list[Point] = field(default_factory=list)


def summarize(points: dict[Point, int]) -> dict[str, dict[str, Stats]]:
    """Return {file: {ptype: Stats}}. Line/branch are gated; toggle is advisory."""
    out: dict[str, dict[str, Stats]] = defaultdict(lambda: defaultdict(Stats))
    for point, count in points.items():
        st = out[point.file][point.ptype]
        st.total += 1
        if count > 0:
            st.covered += 1
        else:
            st.uncovered.append(point)
    return out


PTYPES = ("v_line", "v_branch", "v_toggle")
PTYPE_LABEL = {"v_line": "Line/Block", "v_branch": "Branch", "v_toggle": "Toggle"}


def write_report(output_dir: Path, summary: dict[str, dict[str, Stats]], genhtml_ok: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def bar(st: Stats) -> str:
        denom = st.total or 1
        pct = 100.0 * st.covered / denom
        color = "#86efac" if not st.uncovered else "#fca5a5"
        n_un = len(st.uncovered)
        return (
            f"<div class='cell'><div class='barwrap'><div class='bar' style='width:{pct:.1f}%;"
            f"background:{color}'></div></div><span class='pctn'>{pct:.1f}% "
            f"({st.covered}/{st.total}){' · ' + str(n_un) + ' UNCOVERED' if n_un else ''}</span></div>"
        )

    rows = []
    for fname in sorted(summary):
        cells = "".join(f"<td>{bar(summary[fname].get(pt, Stats()))}</td>" for pt in PTYPES)
        rows.append(f"<tr><td class='fn'>{escape(fname)}</td>{cells}</tr>")

    detail_rows = []
    for fname in sorted(summary):
        for pt in PTYPES:
            st = summary[fname].get(pt)
            if not st or not st.uncovered:
                continue
            for p in sorted(st.uncovered, key=lambda q: (q.line, q.net))[:200]:
                detail_rows.append(
                    f"<tr><td>{escape(fname)}</td><td>{PTYPE_LABEL[pt]}</td><td>{p.line}</td>"
                    f"<td class='net'>{escape(p.net)}</td></tr>"
                )
    detail = (
        "<h2>Uncovered points</h2><table><thead><tr><th>Source</th><th>Kind</th><th>Line</th>"
        "<th>Net / block</th></tr></thead><tbody>" + "\n".join(detail_rows) + "</tbody></table>"
        if detail_rows
        else "<h2>Uncovered points</h2><p class='good'>None — every line and branch covered (toggle advisory).</p>"
    )

    genhtml_link = (
        "<p><a href='lcov/index.html'>Detailed line drill-down (genhtml) &rarr;</a></p>" if genhtml_ok else ""
    )
    note = (
        "<p class='sub'>Genuinely-unreachable points are suppressed in the RTL with "
        "<code>// verilator coverage_off</code> / <code>coverage_on</code> and so never appear here.</p>"
    )

    (output_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Kulibin Float Coverage</title>
<style>
body {{ margin:0; font-family:system-ui,sans-serif; background:#0b1020; color:#e5e7eb; }}
main {{ max-width:1100px; margin:0 auto; padding:40px 24px; }}
h1 {{ font-size:30px; margin:0 0 6px; }}
h2 {{ margin:34px 0 12px; color:#93c5fd; }}
.sub {{ color:#94a3b8; margin:0 0 20px; }}
table {{ width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden; box-shadow:0 0 0 1px #1f2937; }}
th,td {{ padding:10px 12px; border-bottom:1px solid #1f2937; text-align:left; vertical-align:middle; }}
th {{ background:#111827; color:#93c5fd; font-size:13px; letter-spacing:.04em; text-transform:uppercase; }}
td {{ background:#0e1426; font-size:14px; }}
.fn {{ font-family:ui-monospace,monospace; color:#fbbf24; }}
.net,.hier {{ font-family:ui-monospace,monospace; font-size:12px; color:#cbd5e1; }}
.barwrap {{ background:#1f2937; border-radius:6px; height:10px; width:130px; overflow:hidden; display:inline-block; vertical-align:middle; }}
.bar {{ height:10px; }}
.pctn {{ font-size:12px; margin-left:8px; color:#cbd5e1; }}
code {{ font-family:ui-monospace,monospace; background:#111827; padding:1px 5px; border-radius:4px; font-size:12px; }}
.good {{ color:#86efac; font-weight:700; }}
.bad {{ color:#fca5a5; font-weight:700; }}
</style></head><body><main>
<h1>Kulibin Float — Verilator Coverage (line · branch · toggle)</h1>
<p class="sub">Merged across the parameter matrix by parameter-independent point identity; covered = hit by at least one configuration.</p>
<table><thead><tr><th>Source</th><th>Line/Block</th><th>Branch</th><th>Toggle</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
{note}
{genhtml_link}
{detail}
</main></body></html>
""",
        encoding="utf-8",
    )


def run_genhtml(info_path: Path, output_dir: Path) -> bool:
    tool = shutil.which("genhtml")
    if tool is None:
        return False
    lcov_dir = output_dir / "lcov"
    subprocess.run(
        [
            tool,
            "--legend",
            "--show-details",
            "--title",
            "Kulibin Float Line Coverage",
            "--output-directory",
            str(lcov_dir),
            str(info_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build/float/verilator"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/coverage"))
    parser.add_argument(
        "--gate",
        action="store_true",
        help="per-PR tier: fail on uncovered LINE points only (branch is gated by --full; " "toggle is advisory)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="deep tier: fail on uncovered LINE or BRANCH points; report toggle as advisory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        info_path = merge_coverage(args.build_dir, args.output_dir)
        genhtml_ok = run_genhtml(info_path, args.output_dir)

        points = merged_points(args.build_dir)
        summary = summarize(points)
        write_report(args.output_dir, summary, genhtml_ok)
    except Exception as ex:  # noqa: BLE001 - report any tooling failure as a hard error
        print(f"[float-coverage] failed: {ex}", file=sys.stderr)
        return 2

    uncovered: dict[str, list[Point]] = {pt: [] for pt in PTYPES}
    for fname, by_type in summary.items():
        for pt, st in by_type.items():
            uncovered[pt].extend(st.uncovered)

    def report(label: str, pts: list[Point]) -> None:
        print(f"[float-coverage] uncovered {label} points: {len(pts)}", file=sys.stderr)
        for p in sorted(pts, key=lambda q: (q.file, q.line, q.net))[:80]:
            print(f"    {p.file}:{p.line} {p.net}", file=sys.stderr)

    if args.full:
        # Line and branch are mandatory; toggle is ADVISORY -- reported, never fatal (100% toggle on wide datapaths is
        # impractical, so it is a dev aid, not a quality gate).
        gated = ("v_line", "v_branch")
        for pt in PTYPES:
            if uncovered[pt]:
                report(PTYPE_LABEL[pt] + (" [advisory]" if pt == "v_toggle" else ""), uncovered[pt])
        if any(uncovered[pt] for pt in gated):
            print(
                "[float-coverage] To close: add a config/vector that exercises the uncovered line/branch, or "
                "suppress a genuinely-unreachable line/branch in the RTL with // verilator coverage_off/"
                "coverage_on.",
                file=sys.stderr,
            )
            return 1
        ntog = len(uncovered["v_toggle"])
        print(
            f"[float-coverage] PASS: line+branch covered ({ntog} toggle point(s) uncovered -- advisory, "
            f"non-fatal). Report: {args.output_dir / 'index.html'}"
        )
        return 0

    # Default / --gate (per-PR): gate on uncovered LINE points only, ptype-aware. Branch is enforced by --full; toggle
    # is advisory everywhere. Gating on the raw merged LCOV info instead would be ptype-blind -- it collapses
    # line/branch/toggle into one DA record per source line, so an uncovered toggle point would fail the line gate.
    line_uncovered = uncovered["v_line"]
    if line_uncovered:
        report("line", line_uncovered)
        if args.gate:
            print(
                "[float-coverage] To close: add a config/vector that exercises the uncovered line, or suppress a "
                "genuinely-unreachable line in the RTL with // verilator coverage_off/coverage_on.",
                file=sys.stderr,
            )
        return 1 if args.gate else 0

    nbr = len(uncovered["v_branch"])
    ntog = len(uncovered["v_toggle"])
    print(
        f"[float-coverage] PASS: line covered ({nbr} branch + {ntog} toggle point(s) uncovered -- not gated at "
        f"this tier; run --full for the branch gate). Report: {args.output_dir / 'index.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

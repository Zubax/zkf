#!/usr/bin/env python3
"""
Render a colourful HTML report from the JSON summary produced by run_proofs.py.

Style matches tb/zkf_coverage.py's dark-theme fallback for visual consistency.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Kulibin Float Formal Verification")
    return parser.parse_args()


STATUS_CLASS = {
    "PASS": "good",
    "FAIL": "bad",
    "TIMEOUT": "warn",
    "ERROR": "bad",
}


def render(results: list[dict], title: str) -> str:
    rows: list[str] = []
    for r in results:
        status = r.get("status", "ERROR")
        cls = STATUS_CLASS.get(status, "bad")
        engine = escape(r.get("engine", "") or "-")
        params = escape(r.get("parameters", "") or "-")
        wall = f"{r.get('wall_seconds', 0.0):.1f}s"
        trace = r.get("trace_vcd", "") or ""
        detail = r.get("detail", "") or ""
        trace_cell = ""
        if trace:
            trace_cell = f"<a href='{escape(trace)}'>trace.vcd</a>"
        rows.append(
            f"<tr>"
            f"<td class='name'>{escape(r['name'])}</td>"
            f"<td class='{cls}'><strong>{status}</strong></td>"
            f"<td>{engine}</td>"
            f"<td>{params}</td>"
            f"<td>{wall}</td>"
            f"<td>{trace_cell}</td>"
            f"<td class='detail'>{escape(detail)}</td>"
            f"</tr>"
        )

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    timed_out = sum(1 for r in results if r.get("status") == "TIMEOUT")
    errored = sum(1 for r in results if r.get("status") == "ERROR")
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{escape(title)}</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 48px 24px; }}
h1 {{ margin: 0 0 8px; font-size: 32px; }}
.subtitle {{ color: #94a3b8; margin-bottom: 32px; }}
.summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
.summary div {{ padding: 12px 24px; border-radius: 8px; background: #1e293b; font-size: 16px; }}
.summary .good {{ background: #14532d; color: #bbf7d0; }}
.summary .bad {{ background: #7f1d1d; color: #fecaca; }}
.summary .warn {{ background: #78350f; color: #fde68a; }}
table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }}
th, td {{ padding: 12px 14px; border-bottom: 1px solid #334155; text-align: left; vertical-align: top; }}
th {{ background: #1e293b; color: #93c5fd; font-size: 14px; }}
td {{ background: #0f172a; font-size: 14px; }}
td.name {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
td.detail {{ font-family: ui-monospace, monospace; font-size: 12px; max-width: 320px; white-space: pre-wrap; }}
.good {{ color: #86efac; font-weight: 700; }}
.bad {{ color: #fca5a5; font-weight: 700; }}
.warn {{ color: #fde68a; font-weight: 700; }}
a {{ color: #93c5fd; }}
</style>
</head>
<body><main>
<h1>{escape(title)}</h1>
<div class='subtitle'>{total} proofs total · {passed} pass · {failed} fail · {timed_out} timeout · {errored} error</div>
<div class='summary'>
<div class='good'>PASS: {passed}/{total}</div>
{'<div class="bad">FAIL: ' + str(failed) + '</div>' if failed else ''}
{'<div class="warn">TIMEOUT: ' + str(timed_out) + '</div>' if timed_out else ''}
{'<div class="bad">ERROR: ' + str(errored) + '</div>' if errored else ''}
</div>
<table>
<thead><tr>
<th>Module / proof</th>
<th>Status</th>
<th>Engine</th>
<th>Parameters</th>
<th>Wall clock</th>
<th>Trace</th>
<th>Detail</th>
</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</main></body></html>
"""


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"summary not found: {args.summary}")
    results = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results, args.title), encoding="utf-8")
    print(f"[float-formal-report] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

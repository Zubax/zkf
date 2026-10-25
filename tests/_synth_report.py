"""The reports of a session that synthesized: each flow's table of its rows."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import _diamond
import _vivado
import _yosys
from _catalog import CATALOG
from _harness import FREQ_MHZ, Row

BUILD = Path(__file__).resolve().parents[1] / "build" / "synth"
"""Each flow's report and a directory per spec of what the flow was handed and what it left."""


@dataclass(frozen=True)
class Table:
    title: str
    about: str
    """HTML: the flow and what to know of its figures."""
    columns: tuple[tuple[str, str, bool | None], ...]
    """The figures after the status: a heading, the key of the row, and for a heat map whether more is better."""
    links: tuple[tuple[str, str], ...]
    """A label and the key of the row that holds the path of the file."""
    details: tuple[tuple[str, str], ...]
    """A heading and the key of the row that holds the text, shown under the table."""


HARNESS = """<p>Each row is measured through a registered synthesis harness: every DUT input is driven by a wrapper
register and every DUT output is captured by a wrapper register. This makes the reported f max a register-to-register
limit instead of ignoring primary-input or primary-output paths. The harness registers are included in utilization
numbers, but the DUT latency column excludes the harness: it is the latency that the model states.</p>
<p>Helper-module rows are standalone out-of-context builds. Parent-module rows are flattened and context-optimized, so
helper and parent resource counts are not additive.</p>"""

YOSYS_LINKS = (("nextpnr", "nextpnr_log"), ("Yosys", "yosys_log"), ("JSON", "nextpnr_json"))
YOSYS_DETAILS = (
    ("Worst Slack", "slack"),
    ("Utilization", "utilization"),
    ("Yosys Cell Counts", "yosys_cells"),
    ("Report JSON", "json"),
)

TABLES = {
    "yosys_ecp5": Table(
        "Kulibin Float Yosys Synthesis Report",
        f"""<p>Flow: Yosys synth_ecp5 with -noabc9 -abc2 (sequential -dff retiming disabled), nextpnr-ecp5 out of
context (no I/O pads) for {_yosys.ECP5_DEVICE}, or {_yosys.ECP5_LARGE_DEVICE} where the row says so,
{_yosys.ECP5_PACKAGE} speed grade {_yosys.ECP5_SPEED_GRADE}, or {_yosys.ECP5_FASTER_GRADE} where the row says so, at
{FREQ_MHZ:g} MHz.</p>
<p class="note"><strong>Note on LUT4 counts:</strong> nextpnr-ecp5 <code>--12k</code> targets the LFE5U-25F fabric
&mdash; the LFE5U-12F is the same silicon die, marketed with a reduced capacity. The utilization figures of its rows
(the <code>/24288</code> denominators) therefore reflect the 25F array, not the 12F&rsquo;s 12096-LUT4 marketing limit,
and nextpnr will not enforce the smaller limit; for true 12F fit/capacity, use the Lattice Diamond report. The rows on
the 45F (<code>--45k</code>) show its own array, <code>/43848</code>.</p>""",
        (
            ("Part", "part", None),
            ("Yosys LUT4", "lut", None),
            ("Placed LUT4", "lut_placed", False),
            ("FF", "ff", None),
            ("TRELLIS_COMB", "comb", None),
            ("CCU2C", "carry", None),
            ("PFUMX", "pfumx", None),
            ("L6MUX21", "l6mux21", None),
            ("DSP MULT18X18D", "dsp", None),
            ("ALU54B", "alu54", None),
            ("BRAM DP16KD", "bram", None),
            ("IO", "io", None),
        ),
        YOSYS_LINKS,
        YOSYS_DETAILS,
    ),
    "diamond_ecp5": Table(
        "Kulibin Float Diamond Synthesis Report",
        f"""<p>Flow: Lattice Diamond LSE ({_diamond.DEVICE.format(grade=_diamond.SPEED_GRADE)}, or another size or speed
grade where the row says so) at {FREQ_MHZ:g} MHz. Synthesis optimization goal is Timing, MAP register retiming is
disabled, PAR placement effort is {_diamond.PAR_EFFORT} with one placement, router is {_diamond.ROUTER}, and routing
passes are {_diamond.ROUTE_PASSES}.</p>""",
        (
            ("Part", "part", None),
            ("Slack", "slack", None),
            ("LUT4", "lut4", None),
            ("Registers", "registers", None),
            ("Block RAM", "bram", None),
            ("DSP MULT Sites", "dsp_mult", None),
            ("Slice", "slice", False),
            ("PIO", "pio", None),
            ("Setup errors", "setup_errors", None),
            ("Hold errors", "hold_errors", None),
            ("Unrouted", "unrouted", None),
            ("PAR errors", "par_errors", None),
        ),
        (
            ("TRACE", "twr"),
            ("PAR", "par"),
            ("MAP", "mrp"),
            ("Diamond", "diamond_log"),
            ("project", "ldf"),
            ("strategy", "sty"),
            ("preferences", "lpf"),
            ("LSE timing", "lse_twr"),
        ),
        (),
    ),
    "yosys_spartan": Table(
        f"Kulibin Float Yosys Synthesis Report (Spartan-7 {_yosys.SPARTAN_DEVICE})",
        f"""<p>Flow: Yosys synth_xilinx -flatten -family xc7, nextpnr-xilinx for {_yosys.SPARTAN_DEVICE} at
{FREQ_MHZ:g} MHz (no pin constraints; --freq sets the clock target).</p>
<p class="note"><strong>Note:</strong> the open nextpnr-xilinx / prjxray Spartan-7 database covers <code>xc7s50</code>
only, so this is the smallest <em>supported</em> Spartan-7 rather than the smallest in the family. Every top-level
harness port is given an <code>IOSTANDARD</code> (but no fixed pin) so the placer can route the registered harness;
reported f max is the post-route, register-to-register limit.</p>""",
        (
            ("Yosys LUT", "lut", False),
            ("FF", "ff", None),
            ("CARRY4", "carry", None),
            ("MUXF7/8", "wide", None),
            ("DSP48E1", "dsp", None),
            ("BRAM", "bram", None),
            ("IO", "io", None),
        ),
        YOSYS_LINKS,
        YOSYS_DETAILS,
    ),
    "vivado_artix7": Table(
        "Kulibin Float Vivado Synthesis Report",
        f"""<p>Flow: Vivado synthesis, placement and routing out of context for {_vivado.PART} at
{_vivado.FREQ_MHZ:g} MHz, once.</p>""",
        (
            ("Slack", "slack", None),
            ("LUT", "lut", None),
            ("FF", "ff", None),
            ("DSP48E1", "dsp", None),
            ("Block RAM", "bram", None),
        ),
        (("Vivado", "vivado_log"), ("timing", "timing_report")),
        (),
    ),
}

STYLE = """
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
.note { background: #fff7e0; border: 1px solid #e3c75a; border-radius: 6px; padding: 0.6rem 0.9rem; max-width: 72rem; }
.note code { background: #f0e6c0; padding: 0 0.25rem; border-radius: 3px; }
"""

# Five saturated stops from the best value (0) to the worst (1): green, yellow-green, amber, orange, red.
PALETTE = (
    (0.00, (46, 125, 50)),
    (0.25, (154, 205, 50)),
    (0.50, (255, 213, 79)),
    (0.75, (251, 140, 0)),
    (1.00, (198, 40, 40)),
)


def number(text: str) -> float | None:
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", text)
    return float(match.group(0)) if match else None


def heat(text: str, values: Sequence[float], higher_is_better: bool) -> str:
    """The style of a cell of a heat map over the values of its column."""
    value = number(text)
    if value is None or not values:
        return ""
    low, high = min(values), max(values)
    worst = 0.0 if high <= low else ((high - value) if higher_is_better else (value - low)) / (high - low)
    stop = min(int(worst * 4), 3)
    (a, rgb_a), (b, rgb_b) = PALETTE[stop], PALETTE[stop + 1]
    rgb = tuple(round(x + (y - x) * (worst - a) / (b - a)) for x, y in zip(rgb_a, rgb_b))
    # The luminance of ITU-R BT.601: amber and yellow keep black text, the darker stops take white.
    dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 160
    return f' style="background-color: #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}; color: {"#fff" if dark else "#000"};"'


def links(row: Row, pairs: Sequence[tuple[str, str]]) -> str:
    """The files of the row, and its directory, which holds all that the flow was handed and left."""
    found = [f'<a href="{escape(row[key])}">{escape(label)}</a>' for label, key in pairs if row.get(key)]
    return " | ".join([*found, f'<a href="{escape(row["name"])}/">all</a>'])


def table(spec: Table, rows: Sequence[Row]) -> str:
    def column(key: str) -> list[float]:
        return [value for row in rows if (value := number(row.get(key, ""))) is not None]

    fmax = column("fmax")
    figures = {key: column(key) for _, key, better in spec.columns if better is not None}
    body, details = [], []
    for index, row in enumerate(rows):
        start = ' class="group-start"' if index and row["group"] != rows[index - 1]["group"] else ""
        status = row["status"]
        cells = [escape(row[key]) for key in ("name", "knobs", "register_stages")]
        line = f"<tr{start}>" + "".join(f"<td>{cell}</td>" for cell in cells)
        line += f"<td>{escape(row.get('target', ''))}</td>"
        line += f"<td{heat(row.get('fmax', ''), fmax, True)}>{escape(row.get('fmax', ''))}</td>"
        line += f'<td><span class="status {status.lower()}">{escape(status)}</span></td>'
        for _, key, better in spec.columns:
            style = "" if better is None else heat(row.get(key, ""), figures[key], better)
            line += f'<td class="resource"{style}>{escape(row.get(key, ""))}</td>'
        body.append(line + f"<td>{links(row, spec.links)}</td></tr>")
        shown = [("Note", "note"), *spec.details]
        details.append(
            f"<h2>{escape(row['name'])}</h2><p>{escape(row['knobs'])}</p><p>{links(row, spec.links)}</p>"
            + "".join(f"<h3>{heading}</h3><pre>{escape(row[key])}</pre>" for heading, key in shown if row.get(key))
        )
    headings = ["Module", "Knobs", "DUT latency", "Target", "f max", "Status"]
    headings += [heading for heading, _, _ in spec.columns] + ["Files"]
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%z)")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(spec.title)}</title>
<style>{STYLE}</style>
</head>
<body>
<h1>{escape(spec.title)}</h1>
<p>Generated: {escape(generated)}</p>
{spec.about}
{HARNESS}
<table>
<thead><tr>{"".join(f"<th>{escape(heading)}</th>" for heading in headings)}</tr></thead>
<tbody>
{"\n".join(body)}
</tbody>
</table>
{"\n".join(details)}
</body>
</html>
"""


def write(rows: Mapping[str, Sequence[Row]], directory: Path) -> None:
    """The reports of the rows of a session, by flow, each table in the order of the catalog; none of another."""
    order = list(CATALOG)
    for flow in TABLES:
        (directory / flow / "index.html").unlink(missing_ok=True)
    for flow, found in rows.items():
        (directory / flow).mkdir(parents=True, exist_ok=True)
        ordered = sorted(found, key=lambda row: order.index(row["name"]))
        (directory / flow / "index.html").write_text(table(TABLES[flow], ordered))

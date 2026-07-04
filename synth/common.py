"""
Shared, tool-agnostic plumbing for the float synthesis suite.

Holds the repo anchor, subprocess/executable helpers, the parallel worker pool, the pass/fail gate,
and the HTML reporting toolkit (heatmap palette, metric/table cells, artifact links) used by both the
Yosys and Diamond flows. Imports nothing from the rest of the suite, so it sits at the bottom of the
dependency graph. Not runnable on its own.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html import escape
from pathlib import Path
import functools
import os
import re
import shlex
import shutil
import subprocess
import threading


# common.py lives at <repo>/float/synth/common.py, so the repo root is three parents up.
REPO = Path(__file__).resolve().parents[2]
SYNTH_WORKERS = max(1, int(os.environ.get("SYNTH_WORKERS", str(os.cpu_count() or 1))))

# Attribute applied to the measurement-harness registers so synthesis keeps them as real I/O boundary
# flops instead of optimizing them away; this is what makes the reported f max a register-to-register limit.
SYNTH_REG_ATTR = '(* keep = "true", syn_preserve = "true" *)'


def format_mhz(value: float) -> str:
    return f"{value:g} MHz"


def generated_local_time() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%z)")


def run(command: list[str | Path], log_path: Path, cwd: Path = REPO, timeout: float | None = None) -> None:
    rendered = [str(item) for item in command]
    with log_path.open("w") as log:
        log.write("$ " + " ".join(shlex.quote(item) for item in rendered) + "\n\n")
        log.flush()
        try:
            subprocess.run(rendered, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Killed for running too long (e.g. a diverging nextpnr place-and-route). Note it in the log
            # and re-raise; flows that tolerate per-module failure (the optional synth) turn it into a
            # FAIL row instead of letting one module stall the whole run.
            log.write(f"\n\n[run] command exceeded {timeout:g}s timeout and was killed\n")
            raise


def clean_module_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# Roots searched (recursively, in order) in addition to PATH when locating a tool or a bundled script.
# Deliberately broad: toolchains land in assorted prefixes (/usr/bin, /usr/local/bin, /opt/<tool>/...),
# a missing tool simply yields None, and results are cached, so over-searching costs little and spares
# every flow a pile of "set X to override" environment knobs.
_SEARCH_ROOTS = (Path("/opt"), Path("/usr"), Path("/home"))


def _walk_for(name: str, require_exec: bool) -> Path | None:
    for root in _SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None, followlinks=False):
            if name in filenames:
                candidate = Path(dirpath) / name
                if not require_exec or os.access(candidate, os.X_OK):
                    return candidate
    return None


@functools.lru_cache(maxsize=None)
def find_executable(name: str) -> Path | None:
    """Locate an executable by name on PATH first, then by recursive search under /opt, /usr, and so on."""
    found = shutil.which(name)
    if found:
        return Path(found)
    return _walk_for(name, require_exec=True)


@functools.lru_cache(maxsize=None)
def find_file(name: str) -> Path | None:
    """Locate a bundled (non-PATH) file by name via recursive search under /opt, /usr, and so on."""
    return _walk_for(name, require_exec=False)


def require_executable(name: str) -> Path:
    path = find_executable(name)
    if path is None:
        raise SystemExit(f"required executable '{name}' was not found")
    return path


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(errors="replace")


def relative_or_missing(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    return str(path.relative_to(base))


def artifact_link(result: dict[str, str], key: str, label: str, new_tab: bool = False) -> str:
    target = result.get(key, "")
    if not target:
        return ""
    attrs = ' target="_blank" rel="noopener"' if new_tab else ""
    return f'<a href="{escape(target)}"{attrs}>{escape(label)}</a>'


def joined_links(*links: str) -> str:
    return " | ".join(link for link in links if link)


def parse_metric_value(text: str) -> float | None:
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", text)
    return float(match.group(0)) if match else None


def metric_bounds(results: list[dict[str, str]], key: str) -> tuple[float, float] | None:
    values = [value for result in results if (value := parse_metric_value(result.get(key, ""))) is not None]
    return (min(values), max(values)) if values else None


# Five saturated stops from best (worstness=0) to worst (worstness=1). Adjacent regions are
# visually distinct without relying on dim pastels, and the hue progression is easy to scan in
# either direction. Approximate Material-design palette: green / yellow-green / amber / orange / red.
_PALETTE_STOPS = (
    (0.00, ( 46, 125,  50)),   # dark green
    (0.25, (154, 205,  50)),   # yellow-green
    (0.50, (255, 213,  79)),   # amber yellow
    (0.75, (251, 140,   0)),   # orange
    (1.00, (198,  40,  40)),   # dark red
)


def _interpolate_palette(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    for i in range(len(_PALETTE_STOPS) - 1):
        a_t, a_rgb = _PALETTE_STOPS[i]
        b_t, b_rgb = _PALETTE_STOPS[i + 1]
        if t <= b_t:
            frac = 0.0 if b_t <= a_t else (t - a_t) / (b_t - a_t)
            return tuple(round(a + (b - a) * frac) for a, b in zip(a_rgb, b_rgb))
    return _PALETTE_STOPS[-1][1]


def _readable_text_color(rgb: tuple[int, int, int]) -> str:
    # ITU-R BT.601 perceived luminance. Threshold tuned so amber/yellow keeps black text and
    # the darker green/orange/red use white.
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#000" if lum >= 160 else "#fff"


def metric_color_style(
    value_text: str,
    bounds: tuple[float, float] | None,
    higher_is_better: bool,
) -> str:
    value = parse_metric_value(value_text)
    if value is None or bounds is None:
        return ""

    lowest, highest = bounds
    if highest <= lowest:
        normalized_worstness = 0.0
    elif higher_is_better:
        normalized_worstness = (highest - value) / (highest - lowest)
    else:
        normalized_worstness = (value - lowest) / (highest - lowest)

    rgb = _interpolate_palette(normalized_worstness)
    bg = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    fg = _readable_text_color(rgb)
    return f"background-color: {bg}; color: {fg};"


def table_cell(text: str, class_name: str = "", style: str = "") -> str:
    attributes = []
    if class_name:
        attributes.append(f'class="{class_name}"')
    if style:
        attributes.append(f'style="{style}"')
    attribute_text = " " + " ".join(attributes) if attributes else ""
    return f"<td{attribute_text}>{escape(text)}</td>"


def metric_cell(
    text: str,
    bounds: tuple[float, float] | None,
    higher_is_better: bool,
    class_name: str = "",
) -> str:
    return table_cell(text, class_name, metric_color_style(text, bounds, higher_is_better))


def synthesize_with_progress(flow_name: str, modules: list, synthesize_module) -> list[dict[str, str]]:
    total = len(modules)
    workers = max(1, min(SYNTH_WORKERS, total))
    print_lock = threading.Lock()
    started = {"n": 0}
    completed = {"n": 0}

    def run_one(spec) -> dict[str, str]:
        with print_lock:
            started["n"] += 1
            index = started["n"]
            print(f"[{flow_name}] start {index}/{total}: {spec.name}", flush=True)
        result = synthesize_module(spec)
        with print_lock:
            completed["n"] += 1
            done = completed["n"]
            print(
                f"[{flow_name}] done {done}/{total}: {spec.name}: "
                f"{result['status']}, fmax {result.get('fmax', 'not reported')}",
                flush=True,
            )
        return result

    if workers == 1:
        ordered = [run_one(spec) for spec in modules]
    else:
        print(f"[{flow_name}] running {total} modules with {workers} parallel workers", flush=True)
        results_by_name: dict[str, dict[str, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(run_one, spec): spec for spec in modules}
            for future in as_completed(futures):
                spec = futures[future]
                results_by_name[spec.name] = future.result()
        ordered = [results_by_name[spec.name] for spec in modules]
    return ordered


def print_run_summary(flow_label: str, results: list[dict[str, str]], area_key: str, area_label: str) -> None:
    """Print the synthesized targets ranked by f max (high to low) and by area (low to high)."""
    if not results:
        return

    def metric(result: dict[str, str], key: str) -> float | None:
        return parse_metric_value(result.get(key, ""))

    def tag(result: dict[str, str]) -> str:
        return "" if result.get("status") == "PASS" else "  [FAIL]"

    by_fmax = sorted(results, key=lambda r: (metric(r, "fmax") is None, -(metric(r, "fmax") or 0.0), r["name"]))
    by_area = sorted(results, key=lambda r: (metric(r, area_key) is None, metric(r, area_key) or 0.0, r["name"]))
    fmax_w = max(len(r.get("fmax", "")) for r in results)
    area_w = max(len(r.get(area_key, "")) for r in results)

    print(f"\n[{flow_label}] ranked by f_max (high to low):")
    for result in by_fmax:
        print(f"  {result.get('fmax', ''):>{fmax_w}}  {result['name']}{tag(result)}")
    print(f"\n[{flow_label}] ranked by area, {area_label} (low to high):")
    for result in by_area:
        print(f"  {result.get(area_key, ''):>{area_w}}  {result['name']}{tag(result)}")


def require_passing_results(flow_name: str, results: list[dict[str, str]], report_path: Path) -> None:
    failed = [result for result in results if result["status"] != "PASS"]
    if not failed:
        return

    print(f"{flow_name} synthesis failed; see {report_path}")
    for result in failed:
        print(
            f"  {result['name']}: status={result['status']} "
            f"target={result.get('target', 'not reported')} fmax={result.get('fmax', 'not reported')}"
        )
    raise SystemExit(1)

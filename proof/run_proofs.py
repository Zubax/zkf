#!/usr/bin/env python3
"""
Drive every .sby flow under proof/sby/ and aggregate the results.

For each .sby file (skipping *_cover.sby and *_explore.sby which are run separately when needed),
this script:

  1. Invokes sby with a per-proof wall-clock timeout.
  2. Records status: PASS, FAIL (with VCD path), TIMEOUT, or ERROR.
  3. Writes a machine-readable JSON summary alongside the HTML report.

A FAIL with a counterexample propagates to the script's exit code as 2; any other failure mode
(TIMEOUT, ERROR) exits with 1. PASS-only runs exit 0.

`nox -s formal` invokes this driver. The HTML report is rendered by proof/report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from zkf import ZkfFormat  # noqa: E402

LATENCY_KIND_BY_HARNESS = {
    "zkf_pack_eq": "pack",
    "zkf_cmp_eq": "cmp",
    "zkf_sort_eq": "sort",
    "zkf_mul_eq": "mul",
    "zkf_add_eq": "add",
    "zkf_div_eq": "div",
}


@dataclass
class ProofResult:
    name: str
    sby_path: str
    status: str  # PASS, FAIL, TIMEOUT, ERROR
    wall_seconds: float
    engine: str = ""
    parameters: str = ""
    trace_vcd: str = ""
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sby-dir", type=Path, default=Path("proof/sby"))
    parser.add_argument("--build-dir", type=Path, default=Path("build/float/formal"))
    parser.add_argument("--report", type=Path, default=Path("build/float/formal/report.html"))
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="run this many proofs in parallel (0 = os.cpu_count()); each sby runs in its own "
        "build subdir, so they are independent",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path; defaults to <build-dir>/summary.json",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=3 * 60 * 60, help="Per-proof wall-clock soft timeout (default 3 hours)"
    )
    parser.add_argument(
        "--include-explore", action="store_true", help="Also run *_explore.sby flows after the primary set"
    )
    parser.add_argument("--include-cover", action="store_true", help="Also run *_cover.sby flows")
    return parser.parse_args()


def discover_proofs(sby_dir: Path, include_explore: bool, include_cover: bool) -> list[Path]:
    proofs = []
    for path in sorted(sby_dir.glob("*.sby")):
        name = path.stem
        if name.endswith("_cover") and not include_cover:
            continue
        if name.endswith("_explore") and not include_explore:
            continue
        proofs.append(path)
    return proofs


def parse_sby_summary(build_dir: Path) -> tuple[str, str, str]:
    """Read the sby logfile to extract status, engine name, and trace path if any."""
    log_path = build_dir / "logfile.txt"
    if not log_path.exists():
        return ("ERROR", "", "")
    status = "ERROR"
    engine = ""
    trace = ""
    with log_path.open() as fp:
        for raw in fp:
            line = raw.strip()
            if "DONE (PASS" in line:
                status = "PASS"
            elif "DONE (FAIL" in line:
                status = "FAIL"
            elif "DONE (ERROR" in line:
                if status == "ERROR":
                    status = "ERROR"
            elif "DONE (UNKNOWN" in line or "DONE (TIMEOUT" in line:
                status = "TIMEOUT"
            elif "counterexample trace:" in line:
                # form: "... counterexample trace: build/.../trace.vcd"
                trace = line.split("counterexample trace:")[-1].strip()
            elif "summary: engine_0 (smtbmc " in line and "returned" in line:
                # form: "... engine_0 (smtbmc yices) returned pass"
                between = line.split("(", 1)[-1].split(")", 1)[0]
                engine = between.strip()
    return (status, engine, trace)


def parse_sby_chparams(sby_path: Path) -> list[tuple[str, dict[str, int]]]:
    out: list[tuple[str, dict[str, int]]] = []
    with sby_path.open() as fp:
        for raw in fp:
            line = raw.strip()
            if line.startswith("chparam"):
                tokens = line.split()
                pairs: dict[str, int] = {}
                index = 1
                while index < len(tokens):
                    if tokens[index] == "-set" and index + 2 < len(tokens):
                        pairs[tokens[index + 1]] = int(tokens[index + 2], 0)
                        index += 3
                    else:
                        index += 1
                if pairs:
                    out.append((tokens[-1], pairs))
    return out


def parse_sby_parameters(sby_path: Path) -> str:
    chparam_terms: list[str] = []
    for _module, pairs in parse_sby_chparams(sby_path):
        chparam_terms.append(" ".join(f"{key}={value}" for key, value in pairs.items()))
    return "; ".join(chparam_terms)


def proof_latency(harness: str, params: dict[str, int]) -> int | None:
    kind = LATENCY_KIND_BY_HARNESS.get(harness)
    if kind is None:
        return None
    wexp = params.get("WEXP", 6)
    wman = params.get("WMAN", 18)
    values = {
        "wexp_unbiased": params.get("WEXP_UNBIASED"),
        "stage_input": params.get("STAGE_INPUT", 0),
        "stage_product": params.get("STAGE_PRODUCT", 0),
        "stage_align": params.get("STAGE_ALIGN", 0),
        "stage_decode": params.get("STAGE_DECODE", 0),
        "stage_normalize": params.get("STAGE_NORMALIZE", 0),
        "stage_pack": params.get("STAGE_PACK", 0),
        "stage_output": params.get("STAGE_OUTPUT", 0),
    }
    fmt = ZkfFormat(wexp, wman)
    factory = fmt.model_of(kind)
    defaults = factory()
    model = factory(**{name: values[name] for name in defaults.config.keys() if name in values})
    return model.latency


def _inject_chparam_latency(line: str, latency: int) -> str:
    tokens = line.split()
    if "LATENCY" in tokens:
        index = tokens.index("LATENCY")
        tokens[index + 1] = str(latency)
        return " ".join(tokens)
    return " ".join([*tokens[:-1], "-set", "LATENCY", str(latency), tokens[-1]])


def prepare_sby(sby_path: Path, build_root: Path) -> Path:
    chparams = parse_sby_chparams(sby_path)
    if not chparams:
        return sby_path
    latencies = {
        harness: latency for harness, params in chparams if (latency := proof_latency(harness, params)) is not None
    }
    if not latencies:
        return sby_path

    generated_dir = build_root / "_generated_sby"
    generated_dir.mkdir(parents=True, exist_ok=True)
    out_path = generated_dir / sby_path.name
    min_depth = max(latencies.values()) + 4
    in_files = False
    in_options = False
    lines: list[str] = []
    for raw in sby_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_files = stripped == "[files]"
            in_options = stripped == "[options]"
            lines.append(raw)
        elif in_options and stripped.startswith("depth "):
            depth = max(int(stripped.split()[1], 0), min_depth)
            lines.append(f"depth {depth}")
        elif stripped.startswith("chparam"):
            harness = stripped.split()[-1]
            lines.append(_inject_chparam_latency(raw, latencies[harness]) if harness in latencies else raw)
        elif in_files and stripped and not stripped.startswith("#"):
            lines.append(str((REPO / stripped).resolve()))
        else:
            lines.append(raw)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_one_proof(sby_path: Path, build_root: Path, timeout: int) -> ProofResult:
    name = sby_path.stem
    build_dir = build_root / name
    if build_dir.exists():
        shutil.rmtree(build_dir)
    prepared_sby = prepare_sby(sby_path, build_root)
    cmd = ["sby", "-f", "-d", str(build_dir), str(prepared_sby)]
    start = time.monotonic()
    try:
        completed = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - start
        return ProofResult(
            name=name,
            sby_path=str(sby_path),
            status="TIMEOUT",
            wall_seconds=wall,
            engine="",
            parameters=parse_sby_parameters(prepared_sby),
            trace_vcd="",
            detail=f"Wall-clock soft timeout after {timeout}s",
        )
    wall = time.monotonic() - start
    status, engine, trace = parse_sby_summary(build_dir)
    detail = ""
    if status not in {"PASS", "FAIL", "TIMEOUT"}:
        # Likely an SBY tool error; capture stderr tail.
        detail = (completed.stderr or completed.stdout)[-2048:].strip()
    return ProofResult(
        name=name,
        sby_path=str(sby_path),
        status=status,
        wall_seconds=wall,
        engine=engine,
        parameters=parse_sby_parameters(prepared_sby),
        trace_vcd=trace,
        detail=detail,
    )


def main() -> int:
    args = parse_args()
    if not args.sby_dir.is_dir():
        print(f"[run_proofs] sby directory not found: {args.sby_dir}", file=sys.stderr)
        return 1
    args.build_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or (args.build_dir / "summary.json")

    proofs = discover_proofs(args.sby_dir, args.include_explore, args.include_cover)
    if not proofs:
        print(f"[run_proofs] no proofs found under {args.sby_dir}", file=sys.stderr)
        return 1

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(proofs)))
    results: list[ProofResult] = []
    if jobs == 1:
        for sby_path in proofs:
            print(f"[run_proofs] starting {sby_path.stem}", flush=True)
            result = run_one_proof(sby_path, args.build_dir, args.timeout_seconds)
            results.append(result)
            print(f"[run_proofs] {result.name}: {result.status} ({result.wall_seconds:.1f}s)", flush=True)
    else:
        # Proofs are independent (each sby runs in build_dir/<name>), so fan them across cores and
        # reassemble in discovery order for a deterministic summary/report.
        print(f"[run_proofs] running {len(proofs)} proofs with {jobs} parallel workers", flush=True)
        print_lock = threading.Lock()

        def _run(sby_path: Path) -> ProofResult:
            with print_lock:
                print(f"[run_proofs] starting {sby_path.stem}", flush=True)
            r = run_one_proof(sby_path, args.build_dir, args.timeout_seconds)
            with print_lock:
                print(f"[run_proofs] {r.name}: {r.status} ({r.wall_seconds:.1f}s)", flush=True)
            return r

        done: dict[Path, ProofResult] = {}
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            future_to_path = {ex.submit(_run, p): p for p in proofs}
            for fut in as_completed(future_to_path):
                done[future_to_path[fut]] = fut.result()
        results = [done[p] for p in proofs]

    summary_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )

    # Render HTML report.
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "report.py"),
                "--summary",
                str(summary_path),
                "--output",
                str(args.report),
            ],
            check=False,
        )
    except FileNotFoundError:
        print("[run_proofs] report.py not invocable; skipping HTML render", file=sys.stderr)

    any_fail = any(r.status == "FAIL" for r in results)
    any_other = any(r.status not in {"PASS", "FAIL"} for r in results)
    if any_fail:
        return 2
    if any_other:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

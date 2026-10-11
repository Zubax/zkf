#!/usr/bin/env python3
"""Runs the benches of tb/_zkf_pmul_tb.v, of _zkf_pmul alone, under Icarus; invoked by ``nox -s tests``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build" / "float" / "pmul"
RTL = REPO_ROOT / "zkf" / "rtl"
SOURCES = [RTL / "zkf_pipe.v", RTL / "_zkf_pmul.v", REPO_ROOT / "tb" / "_zkf_pmul_tb.v"]
TOPLEVELS = ["_zkf_pmul_tb", "_zkf_pmul_min_tb"]


def _run(top: str) -> int:
    sim = BUILD_DIR / f"{top}.vvp"
    compile_cmd = [
        "iverilog",
        "-g2012",
        "-gstrict-expr-width",  # as synthesis sizes expressions: a one-bit parameter value stays one bit
        "-Wall",
        "-Wno-timescale",
        "-s",
        top,
        "-o",
        str(sim),
        *[str(s) for s in SOURCES],
    ]
    build = subprocess.run(compile_cmd, cwd=REPO_ROOT)
    if build.returncode != 0:
        print(f"\033[31m{top}: COMPILE FAILED\033[0m", file=sys.stderr)
        return build.returncode
    run = subprocess.run(["vvp", str(sim)], cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(run.stdout)
    sys.stderr.write(run.stderr)
    ok = run.returncode == 0 and "PASS" in run.stdout and "FAIL" not in run.stdout
    if ok:
        print(f"\033[32m{top}: PASS\033[0m")
        return 0
    print(f"\033[31m{top}: FAILED (rc={run.returncode})\033[0m", file=sys.stderr)
    return 1


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return max(_run(top) for top in TOPLEVELS)


if __name__ == "__main__":
    raise SystemExit(main())

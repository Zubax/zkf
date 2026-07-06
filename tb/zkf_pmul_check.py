#!/usr/bin/env python3
"""
Self-checking unit bench for the shared pipelined multiply _zkf_pmul.

Compiles zkf/rtl/_zkf_pmul.v + tb/_zkf_pmul_tb.v under Icarus and runs it: the bench proves p == a*b exactly
across STAGE_PRODUCT in {0..4}, both signedness flags, and several operand widths, then prints PASS and
$finish (or FAIL and $fatal). This is a plain-Verilog bench, not cocotb, so it runs standalone rather than
through the cocotb matrix. Formerly the FuseSoC target sim_pmul_icarus; now invoked by ``nox -s tests``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build" / "float" / "pmul"
SOURCES = [REPO_ROOT / "zkf" / "rtl" / "_zkf_pmul.v", REPO_ROOT / "tb" / "_zkf_pmul_tb.v"]
TOPLEVEL = "_zkf_pmul_tb"


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sim = BUILD_DIR / "sim.vvp"
    compile_cmd = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-Wno-timescale",
        "-DSIMULATION=1",
        "-s",
        TOPLEVEL,
        "-o",
        str(sim),
        *[str(s) for s in SOURCES],
    ]
    build = subprocess.run(compile_cmd, cwd=REPO_ROOT)
    if build.returncode != 0:
        print("\033[31m_zkf_pmul_tb: COMPILE FAILED\033[0m", file=sys.stderr)
        return build.returncode
    run = subprocess.run(["vvp", str(sim)], cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(run.stdout)
    sys.stderr.write(run.stderr)
    ok = run.returncode == 0 and "PASS" in run.stdout and "FAIL" not in run.stdout
    if ok:
        print("\033[32m_zkf_pmul_tb: PASS\033[0m")
        return 0
    print(f"\033[31m_zkf_pmul_tb: FAILED (rc={run.returncode})\033[0m", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

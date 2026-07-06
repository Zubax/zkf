#!/usr/bin/env python3
"""
Pytest orchestrator for the float verification matrix.

This is NOT a cocotb test - it is a thin driver that turns every entry of zkf_matrix.build_matrix()
into a pytest test case, builds and runs it through cocotb's native runner (cocotb_tools.runner), and
checks the cocotb results.xml. Each case is tagged with its tier (pr/deep/properties/fast) and simulator
(icarus/verilator) as markers; pyproject.toml deselects deep/properties/fast by default, so a bare pytest
(or ``nox -s tests``) runs only the per-PR set and the heavy work skips unless explicitly selected
(pytest -m deep, -m properties, ...).

The source lists, toplevels, and cocotb modules that FuseSoC used to supply now live in zkf_targets.py.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
import cocotb_tools.runner as _cocotb_runner
from cocotb_tools.runner import get_runner

# xdist provides case-level parallelism; keep each Verilator build single-threaded so workers*make-j does not
# oversubscribe the runner -- the dominant cause of CFS-throttled, worse-than-serial CI wall time.
_cocotb_runner.MAX_PARALLEL_BUILD_JOBS = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
TB_DIR = REPO_ROOT / "tb"
MODEL_DIR = REPO_ROOT  # parent of the zkf reference-model package (the repo root, flat layout)

# The simulator subprocess imports the reference-model package (zkf, at the repo root) and the cocotb harness
# modules (in tb/). cocotb's runner derives the child PYTHONPATH from os.pathsep.join(sys.path), so both
# directories must be on this process's sys.path.
for _p in (str(TB_DIR), str(MODEL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zkf_matrix import build_matrix  # noqa: E402
from zkf_results import check_results  # noqa: E402
from zkf_targets import TARGETS  # noqa: E402

# Per-tool build flags. -DSIMULATION=1 is passed as a define (below) so it applies to both tools; these lists
# mirror the historical FuseSoC flow_options. Verilator collects line+toggle coverage; the runtime coverage file
# is selected per run via a +verilator+coverage+file+... plusarg pointing inside the build dir.
ICARUS_BUILD_ARGS = ["-Wall", "-Wno-timescale"]
VERILATOR_BUILD_ARGS = [
    "--timing",
    "-Wno-TIMESCALEMOD",
    "-Wno-WIDTHEXPAND",
    "-Wno-WIDTHTRUNC",
    "-Wno-DECLFILENAME",
    "-Wno-UNOPTFLAT",
    "--coverage-line",
    "--coverage-toggle",
]

# Environment for the simulator subprocess. Disabling cocotb's pytest-based assertion rewriting keeps a
# globally-installed pytest plugin from leaking arguments into the child (the tests carry explicit failure
# messages), and disabling plugin autoload keeps the inner run hermetic. Mirrors the old FuseSoC subprocess env.
SIM_ENV = {"COCOTB_REWRITE_ASSERTION_FILES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def _target_base(run) -> str:
    suffix = "_" + run.sim
    assert run.target.endswith(suffix), f"unexpected target {run.target!r} for sim {run.sim!r}"
    return run.target[: -len(suffix)]


def _run_cocotb(run) -> None:
    """Build + run one sim via cocotb's runner, then validate its results.xml."""
    spec = TARGETS[_target_base(run)]
    build_dir = REPO_ROOT / run.root
    sources = [REPO_ROOT / src for src in spec.sources()]
    parameters = {name: value for name, value in run.vlog}
    defines = {"SIMULATION": 1, **{name: value for name, value in run.defines}}
    plusargs = [f"+{name}={value}" for name, value in run.plus]
    if run.sim == "verilator":
        build_args = VERILATOR_BUILD_ARGS
        plusargs.append(f"+verilator+coverage+file+{build_dir / 'coverage.dat'}")
    else:
        build_args = ICARUS_BUILD_ARGS

    runner = get_runner(run.sim)
    runner.build(
        sources=sources,
        hdl_toplevel=spec.toplevel,
        defines=defines,
        parameters=parameters,
        build_args=build_args,
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    # Under pytest, runner.test() validates the results and raises on any failing testcase or missing results.
    runner.test(
        test_module=spec.cocotb_module,
        hdl_toplevel=spec.toplevel,
        test_dir=str(TB_DIR),
        plusargs=plusargs,
        extra_env=SIM_ENV,
        build_dir=str(build_dir),
        results_xml=str(build_dir / "results.xml"),
    )
    # Belt and suspenders: also fail if nothing was recorded (results.xml with zero testcases).
    assert check_results(build_dir) == 0, f"{run.id}: results check failed (root={run.root})"


def _shard_keep():
    """
    Optional deterministic sharding for parallel CI jobs.

    FLOAT_SHARD='k/N' keeps only configs whose stable hash falls in shard k (1-indexed) of N; unset
    keeps everything. The partition hashes run.id with a stable hash -- NOT Python's salted hash()
    -- so every shard process partitions identically and the shards tile the matrix with no overlap or
    gaps. It composes with any -m marker selection: each shard runs its slice of the selected tier/sim.
    """
    spec = os.environ.get("FLOAT_SHARD", "").strip()
    if not spec:
        return lambda run: True
    k_text, _, n_text = spec.partition("/")
    k, n = int(k_text), int(n_text)
    if not (n >= 1 and 1 <= k <= n):
        raise ValueError(f"FLOAT_SHARD must be 'k/N' with 1 <= k <= N, got {spec!r}")
    return lambda run: int(hashlib.md5(run.id.encode()).hexdigest(), 16) % n == (k - 1)


def _parametrized():
    keep = _shard_keep()
    for run in build_matrix():
        if not keep(run):
            continue
        marks = [getattr(pytest.mark, run.tier), getattr(pytest.mark, run.sim)]
        yield pytest.param(run, id=run.id, marks=marks)


@pytest.mark.parametrize("run", list(_parametrized()))
def test_float(run) -> None:
    _run_cocotb(run)

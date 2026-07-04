#!/usr/bin/env python3
"""
Pytest orchestrator for the float verification matrix.

This is NOT a cocotb test - it is a thin driver that turns every entry of zkf_matrix.build_matrix()
into a pytest test case, runs it through FuseSoC, and checks the cocotb results.xml. Each case is tagged
with its tier (pr/deep/properties/fast) and simulator (icarus/verilator) as markers; pytest.ini
deselects deep/properties/fast by default, so a bare pytest (or make verify-float) runs only the
per-PR set and the heavy work skips unless explicitly selected (pytest -m deep, -m properties, ...).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zkf_matrix import CORE, build_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TB_DIR = str(REPO_ROOT / "float" / "tb")
MODEL_DIR = str(REPO_ROOT / "float")   # parent of the zkf package, so cocotb sims can from zkf import ...
FUSESOC = os.environ.get("FUSESOC", "fusesoc")
PYTHON = os.environ.get("PYTHON", sys.executable)
PRUNE_BUILDS = os.environ.get("FLOAT_PRUNE_BUILDS", "0") not in ("", "0", "false", "False", "no", "No")


def _subprocess_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([MODEL_DIR, TB_DIR]) + (os.pathsep + existing if existing else "")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"   # keep cocotb's pytest plugin out of the inner sim run
    env["COCOTB_REWRITE_ASSERTION_FILES"] = ""
    return env


def _run_fusesoc(run) -> None:
    """Build + run one sim via FuseSoC, then validate its results.xml. Fails the test on either error."""
    root = REPO_ROOT / run.root
    if root.exists():
        shutil.rmtree(root)
    cmd = [FUSESOC, "run", f"--build-root={run.root}", f"--target={run.target}", CORE]
    for name, value in run.vlog:
        cmd += [f"--{name}", str(value)]
    for name, value in run.defines:
        cmd += [f"--{name}"] if value is True else [f"--{name}", str(value)]
    for name, value in run.plus:
        cmd += [f"--{name}", str(value)]
    env = _subprocess_env()
    # stdout/stderr inherit -> pytest's capture shows them only when the test fails.
    sim = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    results = subprocess.run([PYTHON, "float/tb/zkf_results.py", run.root], cwd=REPO_ROOT, env=env)
    assert sim.returncode == 0 and results.returncode == 0, (
        f"{run.id}: FuseSoC rc={sim.returncode}, results rc={results.returncode} "
        f"(target={run.target}, root={run.root})"
    )
    if PRUNE_BUILDS:
        _prune_successful_build(root, run.sim == "verilator")


def _prune_successful_build(root: Path, keep_coverage: bool) -> None:
    """Free disk after a successful run; Verilator coverage merge needs only coverage.dat files."""
    if not root.exists():
        return
    if not keep_coverage:
        shutil.rmtree(root)
        return

    dat_files = sorted(root.rglob("coverage.dat"))
    if not dat_files:
        return

    tmp = root.with_name(f".{root.name}.coverage")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for idx, dat in enumerate(dat_files):
        dst_dir = tmp if len(dat_files) == 1 else tmp / f"coverage_{idx}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dat, dst_dir / "coverage.dat")

    shutil.rmtree(root)
    shutil.move(str(tmp), str(root))


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
    _run_fusesoc(run)

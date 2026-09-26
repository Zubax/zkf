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
import re
import subprocess
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
from zkf_targets import FILESETS, TARGETS  # noqa: E402

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


def _elaborate(tmp_path, top: str, params: dict, sources, *, valid: bool = True, marker: str | None = None) -> None:
    """Icarus elaboration of `top` must succeed iff `valid`; a refusal must name `marker` when given."""
    result = subprocess.run(
        ["iverilog", "-s", top, "-o", str(tmp_path / "top.vvp"), *[f"-P{top}.{k}={v}" for k, v in params.items()]]
        + [str(REPO_ROOT / src) for src in sources],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == valid, result.stderr
    assert valid or marker is None or marker in result.stderr, result.stderr


_TRIG_SOURCES = TARGETS["sim_cordic"].sources()


@pytest.mark.parametrize(
    "overrides,valid",
    [
        ({}, True),
        ({"WEXP": 8, "WINT": 9, "LATENCY": 1}, True),
        ({"WEXP": 8, "WINT": 8}, False),
        ({"WEXP": 1}, False),
        ({"WMAN": 3}, False),
        ({"STAGE_INPUT": -1}, False),
        ({"STAGE_INPUT": 2, "LATENCY": 3}, True),
        ({"STAGE_INPUT": 2, "LATENCY": 2}, False),
    ],
)
def test_ilog2_elaboration(tmp_path, overrides, valid) -> None:
    _elaborate(tmp_path, "zkf_ilog2", overrides, ["zkf/rtl/zkf_pipe.v", "zkf/rtl/zkf_ilog2.v"], valid=valid)


@pytest.mark.parametrize("top", ["zkf_atan2", "zkf_cordic"])
@pytest.mark.parametrize("wexp,valid", [(2, False), (4, False), (5, True), (6, True)])
def test_vectoring_wexp_floor(tmp_path, top, wexp, valid) -> None:
    """Vectoring requires WEXP >= 5; 4 pins the boundary, 2 the case where theta's codomain is sub-normal."""
    _elaborate(
        tmp_path, top, {"WEXP": wexp, "WMAN": 16}, _TRIG_SOURCES, valid=valid, marker="_zkf_invalid_wexp_or_wman"
    )


_TRIG_LATENCY_KNOBS = [
    ((6, 18), {}),
    ((5, 16), {"unroll100": 50, "stage_input": 1, "stage_product": 1}),
    ((8, 36), {"unroll100": 50, "stage_product": 4, "wmultiplier": 18, "stage_normalize": 2, "stage_pack": 1}),
    ((5, 16), {"stage_pack": 2, "stage_output": 1}),
]


@pytest.mark.parametrize(
    "model,field",
    [
        ("SincosModel", "LATENCY"),
        ("Atan2Model", "LATENCY"),
        ("CordicModel", "LATENCY_ROTATION"),
        ("CordicModel", "LATENCY_VECTORING"),
    ],
)
@pytest.mark.parametrize("knobs", range(len(_TRIG_LATENCY_KNOBS)))
@pytest.mark.parametrize("delta", [0, -1, 1])
def test_trig_latency_guard(tmp_path, model, field, knobs, delta) -> None:
    import zkf

    fmt, config = _TRIG_LATENCY_KNOBS[knobs]
    m = getattr(zkf, model)(zkf.ZkfFormat(*fmt), **config)
    params = {**m.params, field: m.params[field] + delta}
    _elaborate(tmp_path, m.module, params, _TRIG_SOURCES, valid=delta == 0, marker="_zkf_invalid_latency_mismatch")


@pytest.mark.parametrize("top", ["zkf_sincos", "zkf_atan2", "zkf_cordic"])
@pytest.mark.parametrize(
    "knob,value,marker",
    [
        ("STAGE_PACK", 2, None),
        ("STAGE_PACK", 3, "_zkf_invalid_stage_input"),
        ("STAGE_PACK", -1, "_zkf_invalid_stage_input"),
        ("STAGE_INPUT", 2, None),
        ("STAGE_OUTPUT", 2, "_zkf_invalid_stage_output"),
    ],
)
def test_trig_stage_range(tmp_path, top, knob, value, marker) -> None:
    _elaborate(tmp_path, top, {knob: value}, _TRIG_SOURCES, valid=marker is None, marker=marker)


_RTL_SOURCES = sorted((REPO_ROOT / "zkf" / "rtl").rglob("*.v"))
_RTL_MODULES = sorted(re.findall(r"^module\s+(\w+)", "\n".join(p.read_text() for p in _RTL_SOURCES), re.MULTILINE))


@pytest.mark.parametrize("top", _RTL_MODULES)
def test_module_elaborates_with_defaults(tmp_path, top) -> None:
    """Yosys, and so Holoso's flow, elaborates every module it reads at its defaults."""
    _elaborate(tmp_path, top, {}, _RTL_SOURCES)


@pytest.mark.parametrize("mode,parallel,valid", [(0, 1, True), (1, 1, False), (2, 1, True), (2, 0, True)])
def test_cordic_parallel_needs_rotation_mode(tmp_path, mode, parallel, valid) -> None:
    params = {"UNROLL100": 50, "MODE": mode, "PARALLEL": parallel}
    _elaborate(
        tmp_path,
        "_zkf_cordic_m18",
        params,
        FILESETS["rtl_cordic_modes"],
        valid=valid,
        marker="_zkf_parallel_needs_rotation_mode",
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cordic_unit_nets_driven(tmp_path, mode) -> None:
    """The absent datapath's tie-off must name every net it would drive; simulators and synthesis let a miss pass."""
    result = subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            "_zkf_cordic_unit",
            f"-GMODE={mode}",
            *[str(REPO_ROOT / src) for src in _TRIG_SOURCES],
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    undriven = re.findall(r"^%Warning-\w+: .*_zkf_cordic_unit\.v:\d+:\d+: .*not driven.*$", result.stderr, re.MULTILINE)
    assert not undriven, "\n".join(undriven)


@pytest.mark.parametrize("wexp,valid", [(4, False), (5, True)])
def test_atan2_wexp_floor_model(wexp, valid) -> None:
    """The RTL guard, the operator model and the public entry point must agree on the floor, not just the RTL."""
    import zkf
    from zkf._operators import Atan2Model

    fmt = zkf.ZkfFormat(wexp, 16)
    one = fmt.encode(1)
    if valid:
        Atan2Model(fmt=fmt)
        one.atan2(one)
    else:
        with pytest.raises(ValueError):
            Atan2Model(fmt=fmt)
        with pytest.raises(ValueError):
            one.atan2(one)


def test_cordic_modes_rows_cover_every_sigma_arm() -> None:
    # A dropped row would pass every coverage gate (see _cordic_modes).
    rows = {
        (r.tier, dict(r.vlog)["UNROLL100"] >= 100, dict(r.vlog)["PARALLEL"])
        for r in build_matrix()
        if r.module == "cordic_modes"
    }
    assert rows >= {(tier, full, par) for tier in ("pr", "deep") for full, par in ((False, 0), (False, 1), (True, 0))}


@pytest.mark.parametrize("model", ["SincosModel", "Atan2Model", "Exp2Model", "Log2Model"])
def test_model_wexp_ceiling(model) -> None:
    """A model describes a buildable instance, so it refuses the WEXP >= 31 its RTL refuses."""
    import zkf
    import zkf._operators

    cls = getattr(zkf._operators, model)
    cls(fmt=zkf.ZkfFormat(30, 16))
    with pytest.raises(ValueError):
        cls(fmt=zkf.ZkfFormat(31, 16))


@pytest.mark.parametrize("generator", ["zkf_trig", "zkf_transcendental"])
def test_generator_module_entry_point(generator) -> None:
    """`python -m tools.<generator>` must work too; nox runs the generators only by script path."""
    cmd = [sys.executable, "-m", f"tools.{generator}", "--help"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# _zkf_pack's WEXP_UNBIASED floor at WEXP=4: 4 bits unbiased, 5 biased.
_PACK_WU_CASES = [(4, 0, True), (3, 0, False), (5, 1, True), (4, 1, False)]


@pytest.mark.parametrize("wu,eb,valid", _PACK_WU_CASES)
def test_pack_wexp_unbiased_floor(tmp_path, wu, eb, valid) -> None:
    params = {"WEXP": 4, "WMAN": 5, "WEXP_UNBIASED": wu, "EXP_IS_BIASED": eb}
    _elaborate(
        tmp_path,
        "_zkf_pack",
        params,
        ["zkf/rtl/_zkf_pack.v"],
        valid=valid,
        marker="_zkf_invalid_wexp_unbiased_too_narrow",
    )


@pytest.mark.parametrize("wu,eb,valid", _PACK_WU_CASES)
def test_pack_wexp_unbiased_floor_model(wu, eb, valid) -> None:
    import zkf
    from zkf._operators import PackModel

    fmt = zkf.ZkfFormat(4, 5)
    if valid:
        PackModel(fmt=fmt, wexp_unbiased=wu, exp_is_biased=eb)
    else:
        with pytest.raises(ValueError):
            PackModel(fmt=fmt, wexp_unbiased=wu, exp_is_biased=eb)

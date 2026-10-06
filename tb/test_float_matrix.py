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
import itertools
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

from zkf_matrix import ROTATION_SHARDS, build_matrix  # noqa: E402
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


_CORDIC_SOURCES = TARGETS["sim_cordic"].sources()


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


_RINT_PIPELINED = {"STAGE_INPUT": 2, "STAGE_SHIFT": 1, "STAGE_ROUND": 1, "STAGE_OUTPUT": 1}


@pytest.mark.parametrize(
    "overrides,valid",
    [
        ({}, True),
        ({"STAGE_SHIFT": 1, "LATENCY": 1}, True),
        ({"STAGE_SHIFT": 1, "LATENCY": 2}, False),
        ({**_RINT_PIPELINED, "LATENCY": 5}, True),
        ({**_RINT_PIPELINED, "LATENCY": 4}, False),
    ],
)
def test_rint_elaboration(tmp_path, overrides, valid) -> None:
    sources = ["zkf/rtl/zkf_pipe.v", "zkf/rtl/zkf_rint.v"]
    _elaborate(tmp_path, "zkf_rint", overrides, sources, valid=valid, marker="_zkf_invalid_latency_mismatch")


@pytest.mark.parametrize("module", ["zkf_cmp", "zkf_finite"])
@pytest.mark.parametrize(
    "overrides,marker",
    [
        ({}, None),
        ({"STAGE_INPUT": 2, "STAGE_OUTPUT": 1}, None),  # LATENCY=0 disables the check
        ({"STAGE_INPUT": 2, "STAGE_OUTPUT": 1, "LATENCY": 3}, None),
        ({"STAGE_INPUT": 2, "STAGE_OUTPUT": 1, "LATENCY": 2}, "_zkf_invalid_latency_mismatch"),
        ({"STAGE_OUTPUT": 2}, "_zkf_invalid_stage_output"),
        ({"WMAN": 3}, "_zkf_invalid_wexp_or_wman"),
    ],
)
def test_in_out_staged_elaboration(tmp_path, module, overrides, marker) -> None:
    sources = ["zkf/rtl/zkf_pipe.v", f"zkf/rtl/{module}.v"]
    _elaborate(tmp_path, module, overrides, sources, valid=marker is None, marker=marker)


_CORDIC_LATENCY_KNOBS = [
    ((6, 18), {}),
    ((5, 16), {"unroll100": 50, "stage_input": 1, "stage_product": 1}),
    ((8, 36), {"unroll100": 50, "stage_product": 4, "wmultiplier": 18, "stage_normalize": 2, "stage_pack": 1}),
    ((5, 16), {"stage_pack": 2, "stage_output": 1}),
]


@pytest.mark.parametrize(
    "mode,field",
    [(2, "LATENCY_ROTATION"), (2, "LATENCY_VECTORING"), (0, "LATENCY_ROTATION"), (1, "LATENCY_VECTORING")],
)
@pytest.mark.parametrize("knobs", range(len(_CORDIC_LATENCY_KNOBS)))
@pytest.mark.parametrize("delta", [0, -1, 1])
def test_cordic_latency_guard(tmp_path, mode, field, knobs, delta) -> None:
    import zkf

    fmt, config = _CORDIC_LATENCY_KNOBS[knobs]
    m = zkf.CordicModel(zkf.ZkfFormat(*fmt), **config, mode=mode)
    params = {**m.params, field: m.params[field] + delta}
    _elaborate(tmp_path, m.module, params, _CORDIC_SOURCES, valid=delta == 0, marker="_zkf_invalid_latency_mismatch")


@pytest.mark.parametrize("mode", [0, 1, 2])
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
def test_cordic_stage_range(tmp_path, mode, knob, value, marker) -> None:
    params = {"MODE": mode, knob: value}
    _elaborate(tmp_path, "zkf_cordic", params, _CORDIC_SOURCES, valid=marker is None, marker=marker)


_RTL_SOURCES = sorted((REPO_ROOT / "zkf" / "rtl").rglob("*.v"))
_RTL_MODULES = sorted(re.findall(r"^module\s+(\w+)", "\n".join(p.read_text() for p in _RTL_SOURCES), re.MULTILINE))


@pytest.mark.parametrize("top", _RTL_MODULES)
def test_module_elaborates_with_defaults(tmp_path, top) -> None:
    """Yosys, and so Holoso's flow, elaborates every module it reads at its defaults."""
    _elaborate(tmp_path, top, {}, _RTL_SOURCES)


@pytest.mark.parametrize("mode,valid", [(0, True), (1, True), (2, True), (3, False), (-1, False)])
def test_cordic_mode_range(tmp_path, mode, valid) -> None:
    _elaborate(tmp_path, "zkf_cordic", {"MODE": mode}, _CORDIC_SOURCES, valid=valid, marker="_zkf_invalid_cordic_mode")


@pytest.mark.parametrize("mode", [0, 1])
def test_cordic_absent_mode_latency_ignored(tmp_path, mode) -> None:
    """Fixing the mode of an instance that pins both latencies must not break it."""
    import zkf

    params = {**zkf.CordicModel(zkf.ZkfFormat(5, 16), unroll100=50).params, "MODE": mode}
    params["LATENCY_ROTATION" if mode else "LATENCY_VECTORING"] = 1
    _elaborate(tmp_path, "zkf_cordic", params, _CORDIC_SOURCES)


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
def test_cordic_nets_driven(tmp_path, mode) -> None:
    """The absent datapath's tie-off must name every net it would drive; simulators and synthesis let a miss pass."""
    result = subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            "zkf_cordic",
            f"-GMODE={mode}",
            *[str(REPO_ROOT / src) for src in _CORDIC_SOURCES],
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    undriven = re.findall(r"^%Warning-\w+: .*zkf_cordic\.v:\d+:\d+: .*not driven.*$", result.stderr, re.MULTILINE)
    assert not undriven, "\n".join(undriven)


_DIVSQRT_SOURCES = TARGETS["sim_divsqrt"].sources()
_DIVSQRT_KNOBS = [
    ((6, 18), {}),
    ((6, 18), {"stage_decode": 1}),
    ((8, 27), {"stage_input": 2, "stage_decode": 1}),
    ((2, 4), {"stage_pack": 1, "stage_output": 1}),
]


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("knobs", range(len(_DIVSQRT_KNOBS)))
@pytest.mark.parametrize("delta", [0, -1, 1])
def test_divsqrt_latency_guard(tmp_path, mode, knobs, delta) -> None:
    import zkf

    fmt, config = _DIVSQRT_KNOBS[knobs]
    m = zkf.DivsqrtModel(zkf.ZkfFormat(*fmt), **config, mode=mode)
    params = {**m.params, "LATENCY": m.params["LATENCY"] + delta}
    _elaborate(tmp_path, m.module, params, _DIVSQRT_SOURCES, valid=delta == 0, marker="_zkf_invalid_latency_mismatch")


@pytest.mark.parametrize(
    "params,marker",
    [
        ({"MODE": 3}, "_zkf_invalid_divsqrt_mode"),
        ({"MODE": -1}, "_zkf_invalid_divsqrt_mode"),
        ({"STAGE_DECODE": 2}, "_zkf_invalid_stage_decode"),
        ({"STAGE_OUTPUT": 2}, "_zkf_invalid_stage_output"),
        ({"MODE": 0, "STAGE_INPUT": 3, "STAGE_DECODE": 1}, None),
        ({"MODE": 1, "WMAN": 5, "STAGE_DECODE": 1}, None),
    ],
)
def test_divsqrt_parameter_range(tmp_path, params, marker) -> None:
    _elaborate(tmp_path, "zkf_divsqrt", params, _DIVSQRT_SOURCES, valid=marker is None, marker=marker)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("wman,stage_decode", [(4, 0), (4, 1), (5, 0), (18, 0), (18, 1), (27, 0), (53, 0)])
def test_divsqrt_lints_clean(tmp_path, mode, wman, stage_decode) -> None:
    """Every elaboration: no undriven net, no out-of-range select (a simulator or synthesis lets either pass)."""
    result = subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "-Wno-DECLFILENAME",
            "--top-module",
            "zkf_divsqrt",
            f"-GWMAN={wman}",
            f"-GMODE={mode}",
            f"-GSTAGE_DECODE={stage_decode}",
            *[str(REPO_ROOT / src) for src in _DIVSQRT_SOURCES],
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    warnings = re.findall(r"^%Warning-(?!UNUSED)\w+: .*zkf_divsqrt\w*\.v:\d+:\d+: .*$", result.stderr, re.MULTILINE)
    assert not warnings, "\n".join(warnings)


def test_cordic_modes_rows_cover_every_sigma_arm() -> None:
    # A dropped row would pass every coverage gate (see _cordic_modes).
    rows = {
        (r.tier, dict(r.vlog)["UNROLL100"] >= 100, dict(r.vlog)["PARALLEL"])
        for r in build_matrix()
        if r.module == "cordic_modes"
    }
    assert rows >= {(tier, full, par) for tier in ("pr", "deep") for full, par in ((False, 0), (False, 1), (True, 0))}


def test_exhaustive_rotation_rows_are_sharded() -> None:
    # An unsharded sweep passes just the same, an order of magnitude slower.
    rows = [
        dict(r.plus)
        for r in build_matrix()
        if r.module == "cordic" and dict(r.vlog).get("MODE") == 0 and dict(r.plus)["ZKF_KIND"] == "exhaustive"
    ]
    assert rows and all(row.get("ZKF_SHARD_COUNT") == ROTATION_SHARDS for row in rows)
    assert {row["ZKF_SHARD_INDEX"] for row in rows} == set(range(ROTATION_SHARDS))


def test_synth_model_takes_spec_mode(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "synth"))
    from modules import ModuleSpec, model_for

    spec = ModuleSpec(name="", label="", top="", kind="cordic", wexp=2, wman=16, wexp_unbiased=0, mode=0)
    assert model_for(spec).params["MODE"] == 0  # below the floor of the other modes


def _format_bounds() -> list:
    """
    (model, wexp, wman, format knobs, RTL refusal marker or None) on both sides of every format bound a model mirrors.
    Table-backed WMANs come from the generated specs: the smallest supported one and the next one without a table.
    """
    from zkf._reference import trans_specs, trig_specs

    def span(wmans) -> tuple[int, int]:
        return min(wmans), next(w for w in itertools.count(min(wmans)) if w not in wmans)

    exp2, exp2_absent = span({w for f, w in trans_specs() if f == "exp2"})
    log2, log2_absent = span({w for f, w in trans_specs() if f == "log2"})
    trig, trig_absent = span(set(trig_specs()))
    wexp_or_wman = "_zkf_invalid_wexp_or_wman"
    rows = [
        ("Exp2Model", 1, exp2, {}, wexp_or_wman),  # ZkfFormat's own floor
        ("Exp2Model", 30, exp2, {}, None),
        ("Exp2Model", 31, exp2, {}, "_zkf_invalid_exp2_wexp_too_wide_unportable"),
        ("Exp2Model", 8, exp2_absent, {}, f"_zkf_exp2_m{exp2_absent}"),
        ("Log2Model", 30, log2, {}, None),
        ("Log2Model", 31, log2, {}, wexp_or_wman),
        ("Log2Model", 8, log2_absent, {}, f"_zkf_log2_m{log2_absent}"),
        ("FromIntModel", 30, 16, {"wint": 2}, None),
        ("FromIntModel", 31, 16, {}, "_zkf_invalid_from_int_wexp_too_wide_unportable"),
        ("FromIntModel", 8, 16, {"wint": 1}, wexp_or_wman),
        ("RintModel", 31, 16, {"wint": 2}, None),
        ("RintModel", 32, 16, {}, "_zkf_invalid_rint_wexp_too_wide_unportable"),
        ("RintModel", 8, 16, {"wint": 1}, wexp_or_wman),
        ("Ilog2Model", 8, 16, {"wint": 9}, None),
        ("Ilog2Model", 8, 16, {"wint": 8}, "_zkf_invalid_ilog2_wint"),
        ("MulIlog2Model", 8, 16, {"wk": 1}, None),
        ("MulIlog2Model", 8, 16, {"wk": 0}, "_zkf_invalid_mul_ilog2_wk"),
    ]
    # Vectoring's floor is 5 (2 is where theta's codomain is sub-normal); rotation alone goes down to the packer's 2.
    for mode in (1, 2):
        rows += [
            ("CordicModel", w, trig, {"mode": mode}, None if 5 <= w <= 30 else wexp_or_wman) for w in (2, 4, 5, 30, 31)
        ]
    rows += [("CordicModel", w, trig, {"mode": 0}, None if w <= 30 else wexp_or_wman) for w in (2, 4, 30, 31)]
    rows += [("CordicModel", 8, trig_absent, {"mode": mode}, f"_zkf_cordic_m{trig_absent}") for mode in (0, 1, 2)]
    for mode in (0, 1, 2):  # the smallest format folds into no digit stage at all
        rows += [
            ("DivsqrtModel", 2, wman, {"mode": mode, "stage_decode": sd}, None) for wman in (4, 5) for sd in (0, 1)
        ]
    return rows


@pytest.mark.parametrize("model,wexp,wman,knobs,marker", _format_bounds())
def test_model_format_bounds(tmp_path, model, wexp, wman, knobs, marker) -> None:
    """A model describes a buildable instance: it constructs iff its RTL elaborates, else raises UnsupportedFormat."""
    import zkf

    cls = getattr(zkf, model)
    if marker is None:
        m = cls(zkf.ZkfFormat(wexp, wman), **knobs)
        _elaborate(tmp_path, m.module, m.params, _RTL_SOURCES)
    else:
        with pytest.raises(zkf.UnsupportedFormat):
            cls(zkf.ZkfFormat(wexp, wman), **knobs)
        params = {"WEXP": wexp, "WMAN": wman, **{k.upper(): v for k, v in knobs.items()}}
        _elaborate(tmp_path, cls.module, params, _RTL_SOURCES, valid=False, marker=marker)


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
        with pytest.raises(ValueError) as refusal:
            PackModel(fmt=fmt, wexp_unbiased=wu, exp_is_biased=eb)
        assert not isinstance(refusal.value, zkf.UnsupportedFormat), "WEXP_UNBIASED is a knob, not a format"

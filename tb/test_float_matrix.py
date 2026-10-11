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
# mirror the historical FuseSoC flow_options. Verilator collects line coverage, and toggle coverage at the deep tier;
# the runtime coverage file is selected per run via a +verilator+coverage+file+... plusarg pointing inside the build
# dir.
ICARUS_BUILD_ARGS = ["-Wall", "-Wno-timescale"]
VERILATOR_BUILD_ARGS = [
    "--timing",
    "-Wno-TIMESCALEMOD",
    "-Wno-WIDTHEXPAND",
    "-Wno-WIDTHTRUNC",
    "-Wno-DECLFILENAME",
    "-Wno-UNOPTFLAT",
    "--coverage-line",
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
        # Toggle coverage is advisory and costs build time.
        build_args = VERILATOR_BUILD_ARGS + (["--coverage-toggle"] if run.tier == "deep" else [])
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


@pytest.mark.parametrize("module", ["zkf_cmp", "zkf_finite", "zkf_absneg"])
@pytest.mark.parametrize(
    "overrides,marker",
    [
        ({}, None),
        ({"STAGE_INPUT": 2, "STAGE_OUTPUT": 1}, None),  # LATENCY=0 disables the check
        ({"STAGE_INPUT": 2, "STAGE_OUTPUT": 1, "LATENCY": 3}, None),
        ({"STAGE_INPUT": 2, "STAGE_OUTPUT": 1, "LATENCY": 2}, "_zkf_invalid_latency_mismatch"),
        ({"STAGE_OUTPUT": 2, "LATENCY": 2}, None),
    ],
)
def test_in_out_staged_elaboration(tmp_path, module, overrides, marker) -> None:
    sources = ["zkf/rtl/zkf_pipe.v", f"zkf/rtl/{module}.v"]
    _elaborate(tmp_path, module, overrides, sources, valid=marker is None, marker=marker)


@pytest.mark.parametrize("module,valid", [("zkf_cmp", False), ("zkf_finite", False), ("zkf_absneg", True)])
def test_in_out_staged_format_bounds(tmp_path, module, valid) -> None:
    sources = ["zkf/rtl/zkf_pipe.v", f"zkf/rtl/{module}.v"]
    _elaborate(tmp_path, module, {"WMAN": 3}, sources, valid=valid, marker="_zkf_invalid_wexp_or_wman")


def test_rtl_bundle(tmp_path) -> None:
    """The bundle holds every packaged source verbatim and compiles as a whole."""
    import zkf

    bundle = zkf.get_rtl_bundle()
    assert bundle.startswith(f"// GENERATED BY ZKF v{zkf.__version__}\n// Public modules: zkf_absneg, zkf_add, ")
    rtl = zkf.get_rtl()
    for path, source in rtl.items():
        assert bundle.count(f"\n// BEGIN {path}\n\n{source.strip()}\n\n// END {path}\n") == 1, path
    assert re.findall(r"^// BEGIN (.+)$", bundle, re.M) == sorted(rtl, key=lambda path: (path.startswith("_"), path))
    (tmp_path / "zkf.v").write_text(bundle)
    result = subprocess.run(
        ["iverilog", "-o", str(tmp_path / "zkf.vvp"), str(tmp_path / "zkf.v")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


_CORDIC_LATENCY_KNOBS = [
    ((6, 18), {}),
    ((5, 16), {"unroll100": 50, "stage_input": 1, "stage_product": 1}),
    ((8, 36), {"unroll100": 50, "divide_radix": 2, "stage_product": 3, "wmultiplier": 18, "stage_normalize": 3}),
    ((5, 16), {"stage_pack": 2, "stage_output": 1}),
    ((5, 16), {"unroll100": 50, "stage_product": 4}),
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
        ("DIVIDE_RADIX", 2, None),
        ("DIVIDE_RADIX", 3, "_zkf_invalid_divide_radix"),
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
    ((8, 36), {"stage_input": 1, "stage_decode": 1}),
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
        ({"MODE": 1, "WMAN": 5, "STAGE_DECODE": 1}, None),  # no effect at odd WMAN
        ({"MODE": 1, "WMAN": 6, "STAGE_DECODE": 1}, None),
    ],
)
def test_divsqrt_parameter_range(tmp_path, params, marker) -> None:
    _elaborate(tmp_path, "zkf_divsqrt", params, _DIVSQRT_SOURCES, valid=marker is None, marker=marker)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("wman,stage_decode", [(4, 0), (4, 1), (5, 0), (5, 1), (18, 0), (18, 1), (27, 0), (53, 0)])
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
    log2_wide = min(w for f, w in trans_specs() if f == "log2" and w >= 30)
    trig, trig_absent = span(set(trig_specs()))
    wexp_or_wman = "_zkf_invalid_wexp_or_wman"
    rows = [
        ("Exp2Model", 1, exp2, {}, wexp_or_wman),  # ZkfFormat's own floor
        ("Exp2Model", 30, exp2, {}, None),
        ("Exp2Model", 31, exp2, {}, "_zkf_invalid_exp2_wexp_too_wide_unportable"),
        ("Exp2Model", 8, exp2_absent, {}, f"_zkf_exp2_m{exp2_absent}"),
        ("Log2Model", log2 + 1, log2, {}, None),  # the widest exponent whose results cannot tie
        ("Log2Model", log2 + 2, log2, {}, "_zkf_invalid_log2_wexp_admits_ties"),
        ("Log2Model", 30, log2_wide, {}, None),
        ("Log2Model", 31, log2_wide, {}, wexp_or_wman),
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


# zkf_exp2 / zkf_log2: the model's latency is the RTL's at every degree, whichever of the wide product and the chain is
# the longer and whichever feeder of the chain the later, and with each other stage knob.
_PRODUCT_KNOBS = ["stage_product"] + [f"stage_product_{term}" for term in ("square", "odd", "chain")]


def _products(*stages: int) -> dict[str, int]:
    return dict(zip(_PRODUCT_KNOBS, stages, strict=True))


_TRANS_KNOBS = [
    ("exp2", (6, 18), {}),
    ("exp2", (6, 18), _products(1, 0, 3, 1)),  # no product takes the odd knob at this degree
    (
        "exp2",
        (8, 36),
        {"stage_input": 2, "stage_pack": 1, "stage_output": 1, "wmultiplier": 18} | _products(4, 1, 2, 1),
    ),
    ("exp2", (11, 53), {"stage_reduce": 1} | _products(1, 2, 0, 1)),
    ("log2", (6, 18), _products(0, 2, 0, 1)),
    ("log2", (8, 36), {"stage_input": 1, "stage_align": 1, "stage_product_final": 3} | _products(2, 3, 1, 2)),
    ("log2", (11, 53), {"stage_product_final": 1, "stage_pack": 1, "stage_output": 1} | _products(4, 0, 1, 0)),
]


@pytest.mark.parametrize("knobs", range(len(_TRANS_KNOBS)))
@pytest.mark.parametrize("delta", [0, -1, 1])
def test_trans_latency_guard(tmp_path, knobs, delta) -> None:
    import zkf

    func, fmt, config = _TRANS_KNOBS[knobs]
    m = zkf.ZkfFormat(*fmt).model_of(func)(**config)
    params = {**m.params, "LATENCY": m.params["LATENCY"] + delta}
    sources = TARGETS[f"sim_{func}"].sources()
    _elaborate(tmp_path, m.module, params, sources, valid=delta == 0, marker="_zkf_invalid_latency_mismatch")


def test_log2_final_product_default(tmp_path) -> None:
    """STAGE_PRODUCT alone leaves the final multiply at its own default depth of zero."""
    _elaborate(tmp_path, "zkf_log2", {"STAGE_PRODUCT": 1, "LATENCY": 4}, TARGETS["sim_log2"].sources())


@pytest.mark.parametrize("func", ["exp2", "log2"])
@pytest.mark.parametrize("wman", [18, 27, 53])
@pytest.mark.parametrize("knob", _PRODUCT_KNOBS)
def test_trans_product_knob_range(tmp_path, func, wman, knob) -> None:
    """
    Each product knob has one range at every degree, the RTL's and the model's, also where no product takes it. On a
    grid that WMULTIPLIER shapes: the symmetric one refuses the products too narrow for it.
    """
    import zkf

    model = zkf.ZkfFormat(8, wman).model_of(func)
    sources = TARGETS[f"sim_{func}"].sources()
    top = 5 if knob == "stage_product_square" else 4
    _elaborate(tmp_path, f"zkf_{func}", model(**{knob: top}, wmultiplier=18).params, sources)
    with pytest.raises(ValueError):
        model(**{knob: top + 1})
    params = {"WEXP": 8, "WMAN": wman, "WMULTIPLIER": 18, knob.upper(): top + 1}
    _elaborate(tmp_path, f"zkf_{func}", params, sources, valid=False, marker="_zkf_invalid_stage_product_out_of_range")
    params = {"WEXP": 8, "WMAN": wman, "WMULTIPLIER": 18, knob.upper(): -1}
    _elaborate(tmp_path, f"zkf_{func}", params, sources, valid=False, marker="_zkf_invalid_stage_product_out_of_range")


@pytest.mark.parametrize(
    "wa,wb,sp,wm,valid",
    [
        (8, 8, 2, 0, True),
        (7, 8, 2, 0, False),
        (40, 7, 2, 0, False),
        (7, 7, 1, 0, True),
        (12, 12, 4, 0, True),
        (12, 11, 3, 0, False),
        (11, 40, 4, 0, False),
        (2, 18, 4, 18, True),
        (9, 70, 2, 8, True),
    ],
)
def test_pmul_slice_floor(tmp_path, wa, wb, sp, wm, valid) -> None:
    """A split leaves each operand 4 bits per slice at least, which a grid that WMULTIPLIER shapes always does."""
    params = {"WA": wa, "WB": wb, "STAGE_PRODUCT": sp, "WMULTIPLIER": wm}
    sources = ["zkf/rtl/zkf_pipe.v", "zkf/rtl/_zkf_pmul.v"]
    _elaborate(tmp_path, "_zkf_pmul", params, sources, valid=valid, marker="_zkf_invalid_pmul_slice_too_narrow")


@pytest.mark.parametrize(
    "func,wman,knobs,valid",
    [
        ("exp2", 16, {"STAGE_PRODUCT": 4}, True),  # c1*w is 20x18; the side product beside it is 4x1
        ("exp2", 27, {"STAGE_PRODUCT_SQUARE": 5}, True),  # the square's tiles are 18x18 and 18x2
        ("exp2", 48, {"STAGE_PRODUCT_ODD": 4}, True),  # c3*w is 30x30, c5*w 7x7
        ("log2", 16, {"STAGE_PRODUCT": 2}, False),  # c1*w is 19x6
        ("exp2", 16, {"STAGE_PRODUCT_CHAIN": 3}, False),  # c2 * w**2 is 9x9
        ("exp2", 27, {"STAGE_PRODUCT_ODD": 3}, False),  # c3*w is 9x18
        ("log2", 16, {"STAGE_PRODUCT": 2, "WMULTIPLIER": 18}, True),
    ],
)
def test_trans_narrow_product(tmp_path, func, wman, knobs, valid) -> None:
    """A knob is refused where the widest product of its kind is too narrow to split, and only there."""
    import zkf

    params = {"WEXP": 8, "WMAN": wman} | knobs
    names = {k.lower(): v for k, v in knobs.items()}
    params["LATENCY"] = zkf.ZkfFormat(8, wman).model_of(func)(**names).timing.latency
    sources = TARGETS[f"sim_{func}"].sources()
    _elaborate(tmp_path, f"zkf_{func}", params, sources, valid=valid, marker="_zkf_invalid_pmul_slice_too_narrow")


def _latency_models() -> list:
    from zkf._format import _operator_model_descendants

    return [cls for cls in _operator_model_descendants() if getattr(cls, "module", "_").startswith("zkf_")]


@pytest.mark.parametrize("cls", _latency_models(), ids=lambda cls: cls.module)
def test_header_latency_formula(tmp_path, cls) -> None:
    """The LATENCY formula that ends an operator's header, evaluated by Icarus, is the model's latency at every knob."""
    import dataclasses
    import zkf

    variants = {"unroll100": (50, 200), "divide_radix": (2,), "mode": (0, 1)}
    models = []
    for wexp, wman in ((8, 24), (8, 27), (11, 53)):
        try:
            bases = [cls(zkf.ZkfFormat(wexp, wman))]
        except zkf.UnsupportedFormat:
            continue
        names = [f.name for f in dataclasses.fields(cls)]
        bases += [dataclasses.replace(bases[0], **{k: v}) for k in names if k in variants for v in variants[k]]
        models += bases
        for base, knob in itertools.product(bases, [k for k in names if k.startswith("stage_")]):
            for step in (1, 2, 3):  # the smallest step the model accepts
                try:
                    models.append(dataclasses.replace(base, **{knob: step}))
                except ValueError:
                    continue
                break
        # Two knobs at once, in either order: a formula that takes the longer of two paths needs both.
        for a, b in itertools.permutations([k for k in names if k.startswith("stage_")], 2):
            try:
                models.append(dataclasses.replace(bases[0], **{a: 2, b: 1}))
            except ValueError:
                continue
    if not any(name.startswith("LATENCY") for name in models[0].params):
        return  # zkf_pipe
    source = (REPO_ROOT / "zkf" / "rtl" / f"{cls.module}.v").read_text()
    header, _, body = source.partition("`default_nettype")
    formula = re.findall(r"^// (\w+) += (.+)$", header.rstrip().rsplit("\n//\n", 1)[1], re.M)
    defaults = re.findall(r"^ +parameter +(\w+) *= *(.+?),? *(?://.*)?$", body.partition("\n) (")[0], re.M)
    text, wanted = [], {}
    for index, model in enumerate(models):
        latency = {k: v for k, v in model.params.items() if k.startswith("LATENCY")}
        given = {k: v for k, v in model.params.items() if k not in latency}
        given |= {k: v for k, v in defaults if k not in model.params and not k.startswith("LATENCY")}
        text.append(f"module t{index};")
        stages = "(" + " + ".join(k for k in given if k.startswith("STAGE_")) + ")"  # what STAGE_* stands for
        text += [f"    localparam integer {k} = {v};" for k, v in given.items()]
        text += [f"    localparam integer {k} = {v.replace('STAGE_*', stages)};" for k, v in formula]
        text += [f'    initial $display("{index} {k} %0d", {k});' for k in latency]
        text.append("endmodule")
        wanted |= {(str(index), k): str(v) for k, v in latency.items()}
    (tmp_path / "t.v").write_text("\n".join(text) + "\n")
    tops = [arg for index in range(len(models)) for arg in ("-s", f"t{index}")]
    subprocess.run(["iverilog", "-o", str(tmp_path / "t.vvp"), *tops, str(tmp_path / "t.v")], check=True)
    out = subprocess.run(["vvp", "-n", str(tmp_path / "t.vvp")], check=True, capture_output=True, text=True).stdout
    assert {tuple(line.split()[:2]): line.split()[2] for line in out.splitlines()} == wanted


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

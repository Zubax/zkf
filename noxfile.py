"""
Central verification entry point for the Zubax Kulibin float engine.

The verification matrix drives Icarus/Verilator through cocotb's native runner (no FuseSoC, no make).
Tests may take a long time to run; if there is no output, assume they are still running, not stuck.
"""

from pathlib import Path
import shutil

import nox

nox.options.reuse_existing_virtualenvs = True

BLACK_TARGETS = ("zkf", "tb", "synth", "proof", "noxfile.py", "zkf_transcendental.py", "zkf_trig.py")


@nox.session(python=False, default=False)
def clean(session):
    pats = [
        "build",
        ".nox",
        ".*cache",
        ".coverage*",
        "*.egg-info",
        "*.log",
        "*.tmp",
        "*.history",
    ]
    for w in pats:
        for f in Path.cwd().glob(w):
            session.log(f"Removing: {f}")
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)
    for f in Path.cwd().rglob("__pycache__"):
        session.log(f"Removing: {f}")
        shutil.rmtree(f, ignore_errors=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Per-PR gate: model layout, the pmul unit bench, the per-PR sim matrix (both simulators), coverage gate."""
    session.install("-e", ".[test]")
    session.run("python", "tb/test_zkf_model_layout.py")
    session.run("python", "tb/zkf_pmul_check.py")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-n", "auto", *session.posargs)
    session.run(
        "python",
        "tb/zkf_coverage.py",
        "--build-dir",
        "build/float/verilator",
        "--output-dir",
        "build/float/coverage",
        "--gate",
    )


@nox.session
def fast(session: nox.Session) -> None:
    """Smallest-config smoke set (Icarus only); runs in well under a minute. For interactive use between edits."""
    session.install("-e", ".[test]")
    session.run("python", "tb/test_zkf_model_layout.py")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-m", "fast", "-n", "auto", *session.posargs)


@nox.session
def properties(session: nox.Session) -> None:
    """Algebraic-property tests (add/addsub/mul) on Icarus."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-m", "properties", "-n", "auto", *session.posargs)


@nox.session
def deep(session: nox.Session) -> None:
    """Full parameter-equivalence sweep (correctness on Icarus, coverage on Verilator) plus the branch coverage gate."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-m", "deep", "-n", "auto", *session.posargs)
    session.run(
        "python",
        "tb/zkf_coverage.py",
        "--build-dir",
        "build/float/verilator-toggle",
        "--output-dir",
        "build/float/coverage-full",
        "--full",
    )


@nox.session
def accuracy(session: nox.Session) -> None:
    """Transcendental/trig accuracy gate: sweep the fixed-point references against the mpmath oracle (<= 1 ULP)."""
    session.install("-e", ".[test]")
    session.run("python", "zkf_transcendental.py", "--check")
    session.run("python", "zkf_trig.py", "--check")


@nox.session
def formal(session: nox.Session) -> None:
    """SymbiYosys equivalence proofs for the modules under proof/sby/."""
    session.install("-e", ".[test]")
    session.run(
        "python",
        "proof/run_proofs.py",
        "--sby-dir",
        "proof/sby",
        "--build-dir",
        "build/float/formal",
        "--report",
        "build/float/formal/report.html",
        "--jobs",
        "0",
    )


@nox.session
def synth(session: nox.Session) -> None:
    """Out-of-context FPGA synthesis (Yosys + nextpnr-ecp5) of the float modules."""
    session.install("-e", ".[test]")
    session.run("python", "synth/yosys_ecp5.py", *session.posargs)


@nox.session
def black(session: nox.Session) -> None:
    session.install("black~=26.5")
    default = ("--check", *BLACK_TARGETS)
    session.run("python", "-m", "black", *(session.posargs or default))

"""
Central verification entry point for the Zubax Kulibin Float (ZKF) engine.
"""

from pathlib import Path
import os
import shutil

import nox

nox.options.reuse_existing_virtualenvs = True

BLACK_TARGETS = ("zkf", "tb", "synth", "proof", "noxfile.py", "tools")

# A cocotb case hands off between the simulator and its Python driver by blocking, not spinning, so it occupies ~1
# core (measured: build and run both ~1.0 core for icarus and verilator). One worker per core is therefore correct;
# worksteal load-balances the heterogeneous-duration matrix. The earlier ~6h tail was NOT oversubscription -- it was a
# single atan2 case generating ~2e6 vectors via an O(2**WEXP) sweep (fixed in test_atan2.py); the incidental ~180%-CPU
# reading that motivated a cores/2 cap came from that pathological case's rapid handoffs and does not generalize, so
# the cap merely left half the runner idle.
PYTEST_DIST = ("-n", str(os.cpu_count() or 2), "--dist", "worksteal")


@nox.session(python=False, default=False)
def clean(session):
    pats = [
        "build",
        "dist",
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
    session.install("-e", ".[test]")
    session.run("python", "tb/test_zkf_model_layout.py")
    session.run("python", "tb/zkf_pmul_check.py")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", *PYTEST_DIST, *session.posargs)
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
    session.install("-e", ".[test]")
    session.run("python", "tb/test_zkf_model_layout.py")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-m", "fast", *PYTEST_DIST, *session.posargs)


@nox.session
def properties(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-m", "properties", *PYTEST_DIST, *session.posargs)


@nox.session
def deep(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "tb/test_float_matrix.py", "-m", "deep", *PYTEST_DIST, *session.posargs)
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
    session.install("-e", ".[test]")
    session.run("python", "tools/zkf_transcendental.py", "--check")
    session.run("python", "tools/zkf_trig.py", "--check")


# Runs in the ambient toolchain environment, not a fresh virtualenv: run_proofs.py/report.py are stdlib-only,
# and sby (a Python program) must keep resolving its own interpreter -- a venv on PATH would shadow the system
# Python that carries sby's dependencies (click, etc.), breaking every proof with "No module named 'click'".
@nox.session(python=False)
def formal(session: nox.Session) -> None:
    """SymbiYosys equivalence proofs for the modules under proof/sby/."""
    session.run(
        "python3",
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
    session.install("-e", ".[test]")
    session.run("python", "synth/yosys_ecp5.py", *session.posargs)


@nox.session
def black(session: nox.Session) -> None:
    session.install("black~=26.5")
    default = ("--check", *BLACK_TARGETS)
    session.run("python", "-m", "black", *(session.posargs or default))

"""
Central verification entry point for the Zubax Kulibin Float (ZKF) engine.
"""

from pathlib import Path
import os
import shutil

import nox

nox.options.reuse_existing_virtualenvs = True

BLACK_TARGETS = ("zkf", "tests", "noxfile.py")

PYTEST_DIST = ("-n", str(max(1, round(2 * (os.cpu_count() or 2) / 3))), "--dist", "worksteal")


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
    """The per-PR tier, with the gate on the branch coverage of the hardware's structural variants."""
    session.install("-e", ".[test]")
    coverage = ("--cov", "--cov-report=term-missing:skip-covered", "--cov-report=html:build/coverage")
    session.run("python", "-m", "pytest", *PYTEST_DIST, *(session.posargs or coverage))


@nox.session(default=False)
def deep(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-m", "deep", *PYTEST_DIST, *session.posargs)


@nox.session
def accuracy(session: nox.Session) -> None:
    """`exp2`, `log2` and `cordic` against mpmath."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-m", "accuracy", *PYTEST_DIST, *session.posargs)


@nox.session
def synth(session: nox.Session) -> None:
    """The specs of the catalog through the Yosys flow on the ECP5, each once; the reports go to `build/synth/`."""
    session.install("-e", ".[test]")
    workers = ("-n", str(os.cpu_count() or 1), "--dist", "worksteal")
    session.run("python", "-m", "pytest", "-m", "synth", "--synth-flows", "yosys_ecp5", *workers, *session.posargs)


@nox.session
def synth_vendor(session: nox.Session) -> None:
    """The specs of the catalog through the flows of the vendors, each once; the reports go to `build/synth/`."""
    session.install("-e", ".[test]")
    names = ("diamond_ecp5", "vivado_artix7", "vivado_zynquplus", "quartus_cyclone4", "quartus_cyclone5")
    flows = tuple(option for name in names for option in ("--synth-flows", name))
    # A worker's place and route takes up to 5 GB, Vivado's on the UltraScale+; more than the memory holds get killed.
    with open("/proc/meminfo") as meminfo:
        available_kib = next(int(line.split()[1]) for line in meminfo if line.startswith("MemAvailable:"))
    workers = ("-n", str(max(1, min(int(PYTEST_DIST[1]), available_kib // (6 << 20)))), "--dist", "worksteal")
    # The catalog closes on the texts that Amaranth's own Yosys prints; another Yosys prints others, placed otherwise.
    env = {"AMARANTH_USE_YOSYS": "builtin"}
    session.run("python", "-m", "pytest", "-m", "synth", *flows, *workers, *session.posargs, env=env)


@nox.session
def mypy(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "mypy")


@nox.session
def black(session: nox.Session) -> None:
    session.install("black~=26.5")
    default = ("--check", *BLACK_TARGETS)
    session.run("python", "-m", "black", *(session.posargs or default))

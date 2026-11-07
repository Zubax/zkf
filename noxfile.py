"""
Central verification entry point for the Zubax Kulibin Float (ZKF) engine.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import threading

import nox

nox.options.reuse_existing_virtualenvs = True

BLACK_TARGETS = ("zkf", "tests", "noxfile.py")


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
    with watched(session):
        session.run("python", "-m", "pytest", *workers(gib=1.5), *(session.posargs or coverage))


@nox.session(default=False)
def deep(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    with watched(session):
        session.run("python", "-m", "pytest", "-m", "deep", *workers(gib=2), *session.posargs)


@nox.session
def accuracy(session: nox.Session) -> None:
    """`exp2`, `log2` and `cordic` against mpmath."""
    session.install("-e", ".[test]")
    with watched(session):
        session.run("python", "-m", "pytest", "-m", "accuracy", *workers(gib=0.4), *session.posargs)


@nox.session
def synth_yosys(session: nox.Session) -> None:
    """The specs of the catalog through the Yosys flow on the ECP5, each once; the reports go to `build/synth/`."""
    session.install("-e", ".[test]")
    flows = ("--synth-flows", "yosys_ecp5")
    with watched(session):
        session.run("python", "-m", "pytest", "-m", "synth", *flows, *workers(gib=1), *session.posargs)


@nox.session
def synth_vivado(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    flows = ("--synth-flows", "vivado_artix7", "--synth-flows", "vivado_zynquplus")
    # The catalog closes on the texts that Amaranth's own Yosys prints; another Yosys prints others, placed otherwise.
    env = {"AMARANTH_USE_YOSYS": "builtin"}
    with watched(session):
        session.run("python", "-m", "pytest", "-m", "synth", *flows, *workers(gib=3), *session.posargs, env=env)


@nox.session
def synth_diamond(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    flows = ("--synth-flows", "diamond_ecp5")
    env = {"AMARANTH_USE_YOSYS": "builtin"}
    with watched(session):
        session.run("python", "-m", "pytest", "-m", "synth", *flows, *workers(gib=1), *session.posargs, env=env)


@nox.session
def synth_quartus(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    flows = ("--synth-flows", "quartus_cyclone4", "--synth-flows", "quartus_cyclone5")
    env = {"AMARANTH_USE_YOSYS": "builtin"}
    with watched(session):
        session.run("python", "-m", "pytest", "-m", "synth", *flows, *workers(gib=1.5), *session.posargs, env=env)


@nox.session
def mypy(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "mypy")


@nox.session
def black(session: nox.Session) -> None:
    session.install("black~=26.5")
    default = ("--check", *BLACK_TARGETS)
    session.run("python", "-m", "black", *(session.posargs or default))


def meminfo(key: str) -> int:
    """A figure of `/proc/meminfo`, in bytes."""
    with open("/proc/meminfo") as table:
        return next(int(line.split()[1]) << 10 for line in table if line.startswith(f"{key}:"))


def workers(*, gib: float) -> tuple[str, ...]:
    """
    The workers of pytest-xdist that take `gib` GiB each: as many as the available memory holds, one CPU spared, and
    at least two.
    """
    count = min((os.process_cpu_count() or 2) - 1, int(meminfo("MemAvailable") // (gib * 2**30)))
    return "-n", str(max(2, count)), "--dist", "worksteal"


@contextmanager
def watched(session: nox.Session) -> Iterator[None]:
    """Logs the most memory that the machine had in use while the block ran, sampled each second."""
    total, peak, done = meminfo("MemTotal"), [0], threading.Event()

    def sample() -> None:
        while not done.wait(1):
            peak[0] = max(peak[0], total - meminfo("MemAvailable"))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
        session.log(f"memory in use at most: {peak[0] / 2**30:.1f} GiB of {total / 2**30:.1f}, {peak[0] / total:.0%}")

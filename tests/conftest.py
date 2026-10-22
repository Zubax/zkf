"""The option of the synthesis tests, and their reports, which the controller writes from the rows of the session."""

from typing import cast

import pytest

from _harness import Row
from _synth_report import BUILD, TABLES, write

ROWS: dict[str, list[Row]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--synth-flows",
        action="append",
        choices=list(TABLES),
        help="a flow that the synthesis tests run, failing where its tools are absent; by default every flow is run "
        "where its tools are found",
    )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """A worker of pytest-xdist sends the controller the properties that a test records with its report."""
    if report.when == "call":
        for name, row in report.user_properties:
            if name in TABLES:
                ROWS.setdefault(name, []).append(cast(Row, row))


def pytest_sessionfinish(session: pytest.Session) -> None:
    if ROWS and not hasattr(session.config, "workerinput"):
        write(ROWS, BUILD)

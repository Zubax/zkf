"""Every spec of the catalog through every synthesis flow, marked `synth`."""

import os
import shutil
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

import _diamond
import _vivado
import _yosys
from _catalog import CATALOG, ECP5_LARGE, GRADES
from _harness import Design, Row
from _synth_report import BUILD

FLOWS: dict[str, tuple[tuple[str, ...], Callable[[Design, Path], Row], Callable[[Design], str]]] = {
    "yosys_ecp5": (_yosys.ECP5.tools, partial(_yosys.synthesize, _yosys.ECP5), _yosys.ECP5.part),
    "diamond_ecp5": (_diamond.TOOLS, _diamond.synthesize, _diamond.part),
    "yosys_spartan": (_yosys.SPARTAN.tools, partial(_yosys.synthesize, _yosys.SPARTAN), _yosys.SPARTAN.part),
    "vivado_artix7": (_vivado.TOOLS, _vivado.synthesize, _vivado.part),
}
"""The tools of a flow, all found on the path, what runs it on a spec in a directory, and the device it places it on."""


@pytest.mark.synth
@pytest.mark.parametrize("name", CATALOG)
@pytest.mark.parametrize("flow", FLOWS)
def test_synth(
    flow: str, name: str, request: pytest.FixtureRequest, record_property: Callable[[str, object], None]
) -> None:
    """
    The flow, run once, closes timing on the spec: on Yosys every clock meets its target in the report of nextpnr, or
    else in its log, and with neither the spec misses; on Diamond and on Vivado there is no setup or hold error and
    every net is routed. The row goes to the reports, that of a tool that failed with what failed.
    """
    named = request.config.getoption("synth_flows")
    if named and flow not in named:
        pytest.skip(f"--synth-flows does not name {flow}")
    tools, synthesize, part = FLOWS[flow]
    # A script that Python runs need not be executable.
    absent = [tool for tool in tools if shutil.which(tool, mode=os.F_OK) is None]
    if absent and named:
        pytest.fail(f"not on the path: {', '.join(absent)}")
    if absent:
        pytest.skip(f"not on the path: {', '.join(absent)}")
    design = Design(name, CATALOG[name], large=name in ECP5_LARGE, **GRADES.get(name, {}))
    directory = BUILD / flow / design.name
    placed = {"part": part(design)}
    try:
        row = synthesize(design, directory) | placed
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        record_property(flow, design.row | placed | {"status": "FAIL", "note": str(error)})
        raise
    record_property(flow, row)
    assert row["status"] == "PASS", f"{row['fmax']} at {row['target']}; see {directory}"

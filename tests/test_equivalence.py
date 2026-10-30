"""
The compiled text of every spec of the catalog is the design that the simulations check: Yosys proves the core that it
wraps equivalent to the Amaranth design, and its top module only hands its ports to the core.
"""

import json
import subprocess
from pathlib import Path

import pytest
from amaranth.back import rtlil
from amaranth.hdl import Shape
from amaranth.lib import wiring

from _catalog import CATALOG, CoreSpec
from _yosys import YOSYS

# Proved from equal states: the initial values are not compared.
PREPARE = "proc\nmemory\nflatten\nopt_clean\n"

pytestmark = pytest.mark.skipif(YOSYS is None, reason="no Yosys of the system")


@pytest.mark.parametrize("name", CATALOG)
def test_the_compiled_text_is_the_design(name: str, tmp_path: Path) -> None:
    assert YOSYS is not None
    spec = CATALOG[name]
    wrapped = not isinstance(spec, CoreSpec)
    core = f"{name}_core" if wrapped else name
    (tmp_path / "gold.il").write_text(rtlil.convert(spec.component(), name="gold", emit_src=False))
    (tmp_path / "gate.v").write_text(spec.compile())
    top = f"select {name}\nwrite_json -selected top.json\nselect -clear\n" if wrapped else ""
    (tmp_path / "prove.ys").write_text(
        f"read_verilog gate.v\n{top}hierarchy -check -top {core}\n{PREPARE}write_json gate.json\nrename {core} gate\n"
        "design -stash gate\n"
        f"read_rtlil gold.il\nhierarchy -check -top gold\n{PREPARE}design -copy-from gate -as gate gate\n"
        "equiv_make gold gate equiv\nhierarchy -top equiv\n"
        # Alike cell for cell, the two merge almost whole before the solver is asked anything.
        "opt -fast\nequiv_struct\nequiv_simple -seq 2\nequiv_induct\nequiv_status -assert\n"
    )
    proved = subprocess.run([YOSYS, "-q", "-s", "prove.ys"], cwd=tmp_path, capture_output=True, text=True)
    assert proved.returncode == 0, proved.stdout + proved.stderr
    # What the proof does not see: every register on the rising edge of `clk`.
    gate = json.loads((tmp_path / "gate.json").read_text())["modules"][core]
    for cell in gate["cells"].values():
        if "CLK" in cell["connections"]:
            assert int(cell["parameters"]["CLK_POLARITY"], 2) == 1
            assert cell["connections"]["CLK"] == gate["ports"]["clk"]["bits"]
    if wrapped:
        # The top module hands each port of the signature, in order, to the core's, and drives nothing itself.
        module = json.loads((tmp_path / "top.json").read_text())["modules"][name]
        (cell,) = module["cells"].values()
        assert cell["type"] == core
        ports = module["ports"]
        due = {"clk": ("input", 1, False), "rst": ("input", 1, False)} if spec.clocked else {}
        for port, member in spec.signature.members.items():
            shape = Shape.cast(member.shape)
            due[port] = ("output" if member.flow == wiring.Out else "input", shape.width, shape.signed)
        found = [(port, e["direction"], len(e["bits"]), bool(e.get("signed"))) for port, e in ports.items()]
        assert found == [(port, *d) for port, d in due.items()]
        bits = [bit for entry in ports.values() for bit in entry["bits"]]
        assert all(isinstance(bit, int) for bit in bits) and len(set(bits)) == len(bits)
        assert cell["connections"] == {port: entry["bits"] for port, entry in ports.items()}

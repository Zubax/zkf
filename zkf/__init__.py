"""ZKF (Zubax Kulibin float) engine: the bit-exact reference model plus the packaged RTL sources.

The value model is re-exported here; ``get_rtl()`` returns the Verilog modules shipped as package data.
"""

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType

from ._core import (
    Atan2Result as Atan2Result,
    CmpResult as CmpResult,
    DivResult as DivResult,
    Log2Result as Log2Result,
    SinCos as SinCos,
    Zkf as Zkf,
    ZkfFormat as ZkfFormat,
)

__all__ = [
    "Atan2Result",
    "CmpResult",
    "DivResult",
    "Log2Result",
    "SinCos",
    "Zkf",
    "ZkfFormat",
    "get_rtl",
    "__version__",
]

# Changing the version causes a new release to be deployed and tagged when pushed to the main branch.
__version__ = "0.1.0"


@cache
def get_rtl() -> Mapping[str, str]:
    """Every packaged RTL module as a read-only mapping from POSIX path relative to ``zkf/rtl``
    (e.g. ``zkf_add.v``, ``_tables/_zkf_exp2_m18.v``) to its Verilog source text."""
    out: dict[str, str] = {}

    def walk(node: Traversable, prefix: str) -> None:
        for child in node.iterdir():
            name = f"{prefix}{child.name}"
            if child.is_dir():
                walk(child, f"{name}/")
            elif child.name.endswith(".v"):
                out[name] = child.read_text(encoding="utf-8")

    walk(files(__name__) / "rtl", "")
    return MappingProxyType(dict(sorted(out.items())))

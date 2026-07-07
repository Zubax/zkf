from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType


@cache
def get_rtl() -> Mapping[str, str]:
    """Packaged RTL modules as a read-only mapping from POSIX path to Verilog source text."""
    out: dict[str, str] = {}

    def walk(node: Traversable, prefix: str) -> None:
        for child in node.iterdir():
            name = f"{prefix}{child.name}"
            if child.is_dir():
                walk(child, f"{name}/")
            elif child.name.endswith(".v"):
                out[name] = child.read_text(encoding="utf-8")

    walk(files("zkf") / "rtl", "")
    return MappingProxyType(dict(sorted(out.items())))

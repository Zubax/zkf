#!/usr/bin/env python3
"""Validate cocotb results.xml files emitted by native Edalize simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def check_results(root: Path) -> int:
    result_files = sorted(root.rglob("results.xml"))
    if not result_files:
        print(f"[float-results] no results.xml under {root}", file=sys.stderr)
        return 2

    rc = 0
    for result_file in result_files:
        tree = ET.parse(result_file)
        testcases = tree.findall(".//testcase")
        failures = tree.findall(".//failure") + tree.findall(".//error")
        if not testcases:
            print(f"[float-results] {result_file}: no testcases recorded", file=sys.stderr)
            rc = 1
        if failures:
            print(f"[float-results] {result_file}: {len(failures)} failure(s)", file=sys.stderr)
            for failure in failures:
                message = failure.attrib.get("error_msg") or failure.attrib.get("message") or ET.tostring(
                    failure,
                    encoding="unicode",
                )
                print(f"  {message}", file=sys.stderr)
            rc = 1
    return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> int:
    return check_results(parse_args().root)


if __name__ == "__main__":
    raise SystemExit(main())

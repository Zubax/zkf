#!/usr/bin/env python3
"""
Core-level check of the generated zkf_exp2 / zkf_log2 cores: the raw kernel value of the RTL against
``zkf._reference.trans_kernel`` on random and directed arguments, for every core and several product depths.

The cocotb benches compare rounded results, through which a kernel defect of a few LSBs, or one that fires on a rare
carry, is all but invisible; the table certificate, however, is a statement about ``trans_kernel``, so this is the
link that carries it to the RTL. The instrument tables of tools/zkf_transcendental.py get cores of their own here,
enumerated where the argument is narrow enough.

Plain Verilog under Icarus, like zkf_pmul_check.py.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
BUILD = REPO / "build" / "float" / "trans_core"
RTL = REPO / "zkf" / "rtl"
ENUMERATE = 20  # argument bits up to which a core is enumerated instead of sampled

# (STAGE_PRODUCT, _SQUARE, _ODD, _CHAIN, WMULTIPLIER): nothing; the wide product the longest; the chain the longest
# with the square ready last, every product on a tile grid; the odd products ready last, on the symmetric grids, where
# a core takes of each knob what its widest product admits.
KNOBS = [(0, 0, 0, 0, 0), (4, 0, 0, 1, 18), (2, 3, 2, 2, 18), (2, 1, 2, 2, 0), (4, 5, 4, 4, 0)]
KNOBS_DEEP = KNOBS + [
    (1, 0, 0, 1, 0),
    (3, 3, 3, 3, 0),
    (4, 5, 4, 4, 18),
    (0, 1, 4, 0, 18),
    (2, 5, 0, 3, 0),
    (3, 1, 2, 2, 18),
]

# Planted defects: (what, pattern, replacement), each a way an emitter slip would look in the RTL.
MUTANTS = [
    ("the last sum off by one", r"\+ chain;", "+ chain + 1'b1;"),
    (
        "a product's low bit dropped",
        re.escape(".in(pl[W1-SC+TC-1 : RW-SC])"),
        ".in({pl[W1-SC+TC-1 : RW-SC+1], 1'b0})",
    ),
    (
        "a carry lost inside the two-tile square",
        re.escape("sq_s = sq_hh[2*TH-1 : BASE-2*TL] + sq_hl_d;"),
        "sq_s = {sq_hh[2*TH-1 : TH+6], sq_hh[TH+5 : BASE-2*TL] + sq_hl_d};",
    ),
    ("c3*w cut short", re.escape("p3[W3-Z3+T3-1:T3-Z3];"), "{p3[W3-Z3+T3-1:T3-Z3+2], 2'b00};"),
    (
        "the narrow tile of the square held too long",
        re.escape(".N(SQP - NHL)) u_sq_hl_d ("),
        ".N(SQP - NHL + 1)) u_sq_hl_d (",
    ),
    ("a pair with a narrow product held too long", re.escape(".N(NF - NP5)) u_pair2 ("), ".N(NF - NP5 + 1)) u_pair2 ("),
    ("the re-centering boundary off by one", r">= (\d+);", r"> \1;"),
    (
        "a coefficient held one stage too long",
        re.escape(".N(STAGE_PRODUCT_SQUARE)) u_c2 ("),
        ".N(STAGE_PRODUCT_SQUARE + 1)) u_c2 (",
    ),
    (
        "a chain operand one bit short",
        re.escape(".b(sq_0[Q0-1 -: Q1])"),
        ".b({sq_0[Q0-1 -: Q1-1], 1'b0})",
    ),
]


def arguments(spec: dict, count: int, seed: str) -> list[tuple[int, int]]:
    """(segment, in-segment argument) pairs: every pair if they are few, else directed patterns plus random draws."""
    k, rw = spec["k"], spec["rw"]
    if k + rw <= ENUMERATE:
        return [(a >> rw, a & ((1 << rw) - 1)) for a in range(1 << (k + rw))]
    rng = random.Random(seed)
    top = (1 << rw) - 1
    walk = {0, 1, top, top - 1} | {1 << i for i in range(rw)} | {top ^ ((1 << i) - 1) for i in range(rw)}
    walk |= {(1 << i) - 1 for i in range(rw)} | {top ^ (1 << i) for i in range(rw)}
    segs = {0, 1, (1 << k) - 1, (1 << k) - 2, spec.get("thr", 1) - 1, spec.get("thr", 1)}
    segs |= {rng.randrange(1 << k) for _ in range(10)}
    out = [(s, w) for s in sorted(segs) for w in sorted(walk)]
    return out + [(rng.randrange(1 << k), rng.getrandbits(rw)) for _ in range(count)]


def run(job: tuple) -> tuple[str, bool, str]:
    """One core at one knob set -> (label, passed, bench output)."""
    from zkf._operators import _trans_kernel_stages
    from zkf._reference import trans_kernel

    label, func, spec, core, (sp, sq, od, ch, wm), count = job
    k, rw, cf, d = spec["k"], spec["rw"], spec["cf"], spec["d"]
    warg, wout = k + rw, cf + (2 if func == "exp2" else 1)
    stages = _trans_kernel_stages(d, sp, sq, od, ch)  # the model's count: the core checks it and the bench times it
    tag = f"{label}_sp{sp}_sq{sq}_od{od}_ch{ch}_wm{wm}".replace(" ", "_").replace(":", "_")
    vectors = BUILD / f"{tag}.hex"
    args = arguments(spec, count, label)
    with vectors.open("w") as fh:
        for seg, w in args:
            value = trans_kernel(func, spec, spec["coeffs"][seg], w)
            rc = int(func == "log2" and seg >= spec["thr"])
            fh.write(f"{(((rc << wout) | value) << warg) | (seg << rw) | w:x}\n")
    sim = BUILD / f"{tag}.vvp"
    module = f"_zkf_{func}_m{spec['wman']}"
    compile_cmd = ["iverilog", "-g2012", "-Wno-timescale", "-s", "_zkf_trans_core_tb", "-o", str(sim)]
    compile_cmd += [f"-DZKF_CORE={module}", f'-DZKF_VECTORS="{vectors}"'] + (
        ["-DZKF_CORE_LOG2"] if func == "log2" else []
    )
    for name, value in (
        ("WARG", warg),
        ("WOUT", wout),
        ("N", len(args)),
        ("STAGES", stages),
        ("SP", sp),
        ("SQ", sq),
        ("OD", od),
        ("CH", ch),
        ("WM", wm),
    ):
        compile_cmd.append(f"-P_zkf_trans_core_tb.{name}={value}")
    compile_cmd += [
        str(p) for p in (RTL / "zkf_pipe.v", RTL / "_zkf_pmul.v", core, REPO / "tb" / "_zkf_trans_core_tb.v")
    ]
    build = subprocess.run(compile_cmd, capture_output=True, text=True)
    if build.returncode != 0:
        return tag, False, build.stdout + build.stderr
    out = subprocess.run(["vvp", "-n", str(sim)], capture_output=True, text=True).stdout
    vectors.unlink()
    sim.unlink()
    return tag, any(line.startswith("PASS") for line in out.splitlines()), out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", action="store_true", help="about 2**20 arguments per core, more product depths")
    ap.add_argument("--selftest", action="store_true", help="planted RTL defects must each fail")
    args = ap.parse_args()
    import zkf_transcendental as gen
    from zkf._reference import trans_specs

    BUILD.mkdir(parents=True, exist_ok=True)
    count = (1 << 20) if args.deep else (1 << 17)
    knobs = KNOBS_DEEP if args.deep else KNOBS

    def admitted(func: str, spec: dict) -> list[tuple]:
        limits = (*gen.knob_limits(func, spec), 0)
        return sorted({k if k[-1] else tuple(map(min, k, limits)) for k in knobs})

    jobs = []
    for (func, wman), spec in sorted(trans_specs().items()):
        core = RTL / "_tables" / f"_zkf_{func}_m{wman}.v"
        jobs += [(f"{func} m{wman}", func, spec, core, knob, count) for knob in admitted(func, spec)]
    for (func, name), s in gen.instrument_specs().items():
        core = BUILD / f"instrument_{func}_{name}.v"
        core.write_text(gen.emit_table(s))
        jobs += [(f"{func} {name}", func, s.as_dict(), core, knob, count) for knob in admitted(func, s.as_dict())]
    expect = [True] * len(jobs)
    if args.selftest:
        specs = trans_specs()
        tables = [RTL / "_tables" / f"_zkf_{func}_m{wman}.v" for func, wman in sorted(specs)]
        for what, old, new in MUTANTS:
            table = next((t for t in tables if len(re.findall(old, t.read_text())) == 1), None)
            assert table, f"mutant '{what}': no table has the text to replace"
            core = BUILD / f"mutant_{len(jobs)}_{table.name}"
            core.write_text(re.sub(old, new, table.read_text()))
            func, wman = table.stem.split("_")[2], int(table.stem.split("_m")[1])
            jobs.append((f"defect: {what}", func, specs[(func, wman)], core, KNOBS[2], 1 << 17))
            expect.append(False)
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2)) as pool:
        for (tag, passed, out), job, want in zip(pool.map(run, jobs), jobs, expect):
            ok = passed
            if not want:  # a defect trips only by the bench's own verdict, not by failing to build
                verdict = next((line for line in out.splitlines() if line.startswith(("PASS", "FAIL"))), "no verdict")
                ok = verdict.startswith("FAIL")
                print(f"  {'TRIPPED' if ok else 'MISSED '} {job[0]}: {verdict}")
            if not ok:
                failures.append(f"{tag}: {out.strip()[-400:]}")
    if failures:
        print("\033[31mzkf_trans_core_check: FAILED\033[0m\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print(f"\033[32mzkf_trans_core_check: PASS\033[0m ({len(jobs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

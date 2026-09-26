"""
Shared artifact write/verify policy for the table generators (zkf_trig.py, zkf_transcendental.py).

Deliberately under tools/ rather than in the zkf package: this is developer tooling and zkf/ ships in the wheel, so
putting it there would widen the consumer import surface.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def write(arts: dict[Path, str]) -> None:
    for path, text in arts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(REPO)}")


def verify(arts: dict[Path, str], found: list[Path], regen: str) -> None:
    """`found` is the caller's OWN on-disk tables: _tables/ is shared, so each generator owns only its prefixes."""
    # Compared as bytes: read_text would translate CRLF and hide a line-ending difference.
    bad = [
        str(p.relative_to(REPO))
        for p, text in arts.items()
        if not (p.exists() and p.read_bytes() == text.encode("utf-8"))
    ]
    bad += [f"{p.relative_to(REPO)} (orphan)" for p in found if p not in arts]
    if bad:
        # --emit rewrites but never unlinks, so an orphan needs a delete rather than a regeneration.
        raise SystemExit(
            f"generated artifacts out of date -- run {regen}, and delete any (orphan):\n  " + "\n  ".join(bad)
        )
    print(f"{len(arts)} generated artifacts match a fresh run")

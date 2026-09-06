#!/usr/bin/env python3
"""Check that every docs page's nav matches the canonical structure — stdlib only.

Thin CI-facing wrapper around site_nav.py: reuses its canonical link set and
renderer instead of re-deriving them, so the check can never drift from the
generator itself. Exits non-zero (and lists the stale pages) instead of
rewriting anything — use `python3 scripts/site_nav.py` to fix.

Usage: python3 scripts/check_nav.py [docs_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_nav  # noqa: E402


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    stale = []
    for path in sorted(docs.glob("*.html")):
        if path.name in site_nav.SKIP:
            continue
        before = path.read_text(encoding="utf-8")
        after = site_nav.apply_to_text(before, path.name)
        if after != before:
            stale.append(path.name)
    if stale:
        print("nav out of sync on: " + ", ".join(stale))
        print("run: python3 scripts/site_nav.py")
        return 1
    print(f"{docs}: nav in sync on all pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

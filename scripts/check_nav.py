#!/usr/bin/env python3
"""Check every docs page's shared chrome: nav, footer, table wrappers — stdlib only.

Thin CI-facing wrapper around site_nav.py and wrap_tables.py: reuses their
canonical structure and renderers instead of re-deriving them, so the check can
never drift from the generators themselves. Exits non-zero (and lists the stale
pages) instead of rewriting anything — run the script it names to fix.

Usage: python3 scripts/check_nav.py [docs_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_nav  # noqa: E402
import wrap_tables  # noqa: E402


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    stale: list[str] = []
    footerless: list[str] = []
    unwrapped: list[str] = []
    for path in sorted(docs.glob("*.html")):
        if path.name in site_nav.SKIP:
            continue
        before = path.read_text(encoding="utf-8")
        if not site_nav.has_footer(before):
            footerless.append(path.name)
        if site_nav.apply_to_text(before, path.name) != before:
            stale.append(path.name)
        if path.name not in wrap_tables.SKIP and wrap_tables.apply_to_text(before) != before:
            unwrapped.append(path.name)
    rc = 0
    if stale:
        print("nav out of sync on: " + ", ".join(stale))
        print("run: python3 scripts/site_nav.py")
        rc = 1
    if footerless:
        # A page with no footer has no licence line and no repo link — chrome a
        # third of the site was silently missing until this check existed.
        print("no footer on: " + ", ".join(footerless))
        print("run: python3 scripts/site_nav.py")
        rc = 1
    if unwrapped:
        print("tables outside a .table-scroll region on: " + ", ".join(unwrapped))
        print("run: python3 scripts/wrap_tables.py")
        rc = 1
    if rc == 0:
        print(f"{docs}: nav, footer and table wrappers in sync on all pages")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

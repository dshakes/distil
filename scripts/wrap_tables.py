#!/usr/bin/env python3
"""Wrap every docs table in the `.table-scroll` scroll region — stdlib only.

`.table-scroll` (docs/site.css) is what stops a 5-7 column table overflowing the
viewport on a phone, and site.js wraps tables at runtime — but only for readers
who run JavaScript, and only after first paint, so the first frame on a phone is
still the broken one. 104 tables across 34 pages shipped with no wrapper in the
HTML at all. This puts the wrapper in the source, where it costs nothing and
needs no JS.

The wrapper carries the same `tabindex`/`role`/`aria-label` site.js adds, because
a scroll region that keyboard users cannot focus is not reachable at all — and
site.js deliberately skips any table already inside `.table-scroll`, so a bare
`<div class="table-scroll">` here would *remove* those attributes rather than
add them. The label is the nearest preceding heading, same rule as site.js.

Idempotent: a table whose immediately-preceding tag already carries the
`table-scroll` class token is left exactly as it is, so running this twice is
a no-op. A table nested inside another table's cell is not wrapped — the
naive first-`</table>`-after-the-open scan would match the inner table's
close and corrupt the outer one, so nesting is detected and raises instead.

Usage: python3 scripts/wrap_tables.py [docs_dir]
Checked by tests/test_table_scroll.py (regenerate-and-compare, same pattern as
scripts/site_nav.py / scripts/build_search_index.py).
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# index.html has its own inline CSS and its own wrapper script (it has no
# `.content` container, so the shared `.content .table-scroll` rule never
# applies to it) — same carve-out as site_nav.py and build_search_index.py.
SKIP = {"index.html"}

_TABLE_OPEN_RE = re.compile(r"<table\b")
_TABLE_CLOSE_RE = re.compile(r"</table\s*>")
_HEADING_RE = re.compile(r"<h[1-4]\b[^>]*>(.*?)</h[1-4]\s*>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")
_CLASS_RE = re.compile(r'class="([^"]*)"')


def _already_wrapped(before: str) -> bool:
    """True when the tag immediately preceding the table is the scroll wrapper.

    Matches the `table-scroll` class token exactly (same as site.js's
    `closest(".table-scroll")`) rather than a bare substring check, so a class
    like `not-table-scroll` is not mistaken for the wrapper.
    """
    stripped = before.rstrip()
    lt = stripped.rfind("<")
    if lt == -1:
        return False
    m = _CLASS_RE.search(stripped[lt:])
    return m is not None and "table-scroll" in m.group(1).split()


def _label(before: str) -> str:
    """The nearest preceding heading's text — the same label site.js derives."""
    matches = _HEADING_RE.findall(before)
    if not matches:
        return "Scrollable table"
    text = html.unescape(_TAG_RE.sub(" ", matches[-1]))
    text = _WS_RE.sub(" ", text.replace("#", "")).strip()
    return text or "Scrollable table"


def apply_to_text(text: str) -> str:
    """Return *text* with every unwrapped `<table>` inside a scroll region."""
    out: list[str] = []
    pos = 0
    while True:
        m = _TABLE_OPEN_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        close = _TABLE_CLOSE_RE.search(text, m.end())
        if close is None:  # unbalanced markup: leave the rest untouched
            out.append(text[pos:])
            return "".join(out)
        if _TABLE_OPEN_RE.search(text, m.end(), close.start()):
            # A <table> opens again before this </table> — the close we just
            # found belongs to the nested table, not this one. Wrapping here
            # would close the outer <div> at the wrong point and corrupt the
            # page. Nested tables aren't supported; fail loudly rather than
            # silently emitting invalid HTML.
            raise ValueError(
                f"nested <table> found at offset {m.start()} in the doc being wrapped — "
                "wrap_tables.py cannot safely wrap it; wrap the nested table by hand."
            )
        before = text[: m.start()]
        if _already_wrapped(before):
            out.append(text[pos : close.end()])
            pos = close.end()
            continue
        # Re-use the table's own indentation for the wrapper so the diff reads as
        # two added lines rather than a reflow of the whole block.
        line_start = before.rfind("\n") + 1
        indent = before[line_start:]
        indent = indent if not indent.strip() else ""
        label = html.escape(_label(before), quote=True)
        out.append(text[pos : m.start()])
        out.append(
            f'<div class="table-scroll" tabindex="0" role="region" aria-label="{label}">\n{indent}'
        )
        out.append(text[m.start() : close.end()])
        out.append(f"\n{indent}</div>")
        pos = close.end()


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    changed = 0
    for path in sorted(docs.glob("*.html")):
        if path.name in SKIP:
            continue
        before = path.read_text(encoding="utf-8")
        after = apply_to_text(before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed += 1
    print(f"{docs}: wrapped tables on {changed} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

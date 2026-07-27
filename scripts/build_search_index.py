#!/usr/bin/env python3
"""Build the docs search index — stdlib only, run at commit time.

The docs site is 22 pages with no way to search them, so finding "how do I set
an admin token" meant knowing which page it lived on. This produces a small
static JSON the site loads on first ⌘K; there is no search service, no runtime
dependency, and nothing to keep running.

Indexed per page: the title, every heading, and the lead paragraph. Headings are
what people actually search for, and each one carries its anchor so a hit lands
on the section rather than the top of the page.

Usage:  python3 scripts/build_search_index.py [docs_dir]
Checked by tests/test_search_index.py, which fails if the committed index is
stale relative to the HTML — otherwise it silently rots the first time someone
adds a page.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# Pages that are not documentation: the landing page has its own navigation and
# would swamp results with marketing copy.
SKIP = {"index.html"}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(raw: str) -> str:
    """Strip tags and collapse whitespace. Entities are unescaped by the parser."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()


class _Extractor(HTMLParser):
    """Pulls <title>, headings (with ids) and the first <p class="lead">.

    Deliberately a parser rather than a regex sweep: headings carry inline markup
    (<span class="g">, <code>, anchor links) that a naive regex mangles into the
    search text, and a mangled heading is a heading nobody finds.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.headings: list[tuple[str, str]] = []  # (text, anchor)
        self.lead = ""
        self.body: list[str] = []
        self._buf: list[str] = []
        self._mode = ""
        self._anchor = ""
        self._depth = 0
        # Body text is collected only inside <main>, so the sidebar (identical on
        # all 21 pages) does not make every page match every navigation word.
        self._in_main = 0
        self._skip = 0  # inside <script>/<style>: markup, not prose

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "main":
            self._in_main += 1
        # `not self.title` — the FIRST title wins. Inline <svg> uses <title> for
        # accessible names, and several diagrams have one, so without this the
        # page title became whatever the last diagram was called ("Learned
        # keep-model" for architecture.html).
        if tag == "title" and not self._mode and not self.title:
            self._mode, self._buf = "title", []
        elif tag in ("h1", "h2", "h3") and not self._mode:
            # Sidebar section labels ("Getting started", "Learn", "Reference")
            # are navigation chrome repeated on all 21 pages. Indexing them puts
            # the same four meaningless hits under every query.
            if "sidebar-section" in cls:
                return
            self._mode, self._buf = "h", []
            self._anchor = a.get("id") or ""
            self._depth = 1
        elif tag == "p" and not self._mode and "lead" in (a.get("class") or ""):
            self._mode, self._buf = "lead", []
            self._depth = 1
        elif self._mode in ("h", "lead"):
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag == "main" and self._in_main:
            self._in_main -= 1
        if not self._mode:
            return
        if self._mode in ("h", "lead"):
            self._depth -= 1
            if self._depth > 0:
                return
        text = _clean("".join(self._buf))
        if self._mode == "title":
            self.title = text
        elif self._mode == "h" and text:
            # The "#" permalink glyph is markup, not content.
            self.headings.append((text.rstrip("# ").strip(), self._anchor))
        elif self._mode == "lead":
            self.lead = text
        self._mode, self._buf, self._anchor = "", [], ""

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)
        if self._in_main and not self._skip:
            t = data.strip()
            if t:
                self.body.append(t)


def build(docs: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(docs.glob("*.html")):
        if path.name in SKIP:
            continue
        p = _Extractor()
        p.feed(path.read_text(encoding="utf-8", errors="replace"))
        title = (p.title or path.stem).split("—")[0].strip()
        out.append(
            {
                "u": path.name,
                "t": title,
                "d": p.lead[:180],
                # Anchor-bearing headings only for the deep links; the rest still
                # contribute their words to the page's searchable text.
                "h": [{"t": t, "a": a} for t, a in p.headings if t],
                # Body prose, lowercased and capped. Without it, only headings
                # were searchable — so a real query like "admin token" (which
                # lives in a paragraph, not a heading) returned nothing at all.
                # Lowercased at build time so the browser never re-lowercases
                # ~300 KB on every keystroke.
                "b": _WS_RE.sub(" ", " ".join(p.body))[:14000].lower(),
            }
        )
    return out


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    index = build(docs)
    dest = docs / "search-index.json"
    # Compact: this is fetched by the browser, and it is regenerated not edited.
    dest.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    pages = len(index)
    headings = sum(len(p["h"]) for p in index)
    print(f"{dest}: {pages} pages, {headings} headings, {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

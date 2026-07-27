"""The docs search index must match the docs.

A committed, generated artifact rots the first time someone adds a page and
forgets to rebuild — and it rots silently, because search simply never returns
the new page. This test is the thing that makes the artifact safe to commit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_INDEX = _DOCS / "search-index.json"
_BUILDER = _ROOT / "scripts" / "build_search_index.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_search_index", _BUILDER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not present in this checkout")


def test_committed_index_is_current():
    """Regenerating must be a no-op. If this fails, run:

    python3 scripts/build_search_index.py
    """
    built = _load_builder().build(_DOCS)
    committed = json.loads(_INDEX.read_text(encoding="utf-8"))
    built_pages = {p["u"] for p in built}
    committed_pages = {p["u"] for p in committed}
    assert built_pages == committed_pages, (
        "search index is stale — pages added/removed since it was last built: "
        f"missing={sorted(built_pages - committed_pages)} "
        f"extra={sorted(committed_pages - built_pages)}. "
        "Run: python3 scripts/build_search_index.py"
    )
    assert built == committed, "search index content is stale — rebuild it"


def test_every_docs_page_is_searchable():
    """A page absent from the index is a page search cannot find."""
    mod = _load_builder()
    on_disk = {p.name for p in _DOCS.glob("*.html")} - mod.SKIP
    indexed = {p["u"] for p in json.loads(_INDEX.read_text(encoding="utf-8"))}
    assert on_disk == indexed, f"not searchable: {sorted(on_disk - indexed)}"


def test_index_carries_real_content_not_just_titles():
    """Headings-only indexing made real queries return nothing (the bug that
    prompted body indexing: "admin token" lives in prose, not a heading)."""
    idx = json.loads(_INDEX.read_text(encoding="utf-8"))
    assert idx, "empty index"
    for page in idx:
        assert page["t"], f"{page['u']} has no title"
    # The corpus must actually contain prose, or search degrades to a nav menu.
    assert sum(len(p["b"]) for p in idx) > 50_000
    joined = " ".join(p["b"] for p in idx)
    for term in ("admin", "token", "prometheus", "compression"):
        assert term in joined, f"{term!r} is not findable anywhere in the docs index"


def test_navigation_chrome_is_not_indexed():
    """The sidebar is identical on every page; indexing it made all 22 pages
    match navigation words and buried the real hits."""
    idx = json.loads(_INDEX.read_text(encoding="utf-8"))
    nav_labels = {"Getting started", "Learn", "Reference", "More"}
    for page in idx:
        for h in page["h"]:
            assert h["t"] not in nav_labels, f"{page['u']} indexed nav label {h['t']!r}"


def test_page_titles_are_not_taken_from_svg_titles():
    """Inline <svg> uses <title> for accessible names. Taking the last one made
    architecture.html's title "Learned keep-model"."""
    idx = json.loads(_INDEX.read_text(encoding="utf-8"))
    by_url = {p["u"]: p for p in idx}
    if "architecture.html" in by_url:
        assert by_url["architecture.html"]["t"] == "Architecture"

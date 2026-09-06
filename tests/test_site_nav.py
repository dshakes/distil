"""The docs site's shared topbar + sidebar must match the canonical structure.

The nav is hand-duplicated on every page (there is no templating layer) and it
had drifted into a dozen variants: cli.html was missing its own Library API
link, benchmark.html dropped its two sibling benchmark pages, 9 pages never
got the "Which Mode?" topbar link, and benchmarks.html was not reachable from
any page's nav at all. scripts/site_nav.py renders the canonical nav from one
source of truth; this test fails the moment a page's nav diverges from it,
the same regenerate-and-compare pattern as tests/test_search_index.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_BUILDER = _ROOT / "scripts" / "site_nav.py"

pytestmark = pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not present in this checkout")


def _load_site_nav():
    spec = importlib.util.spec_from_file_location("site_nav", _BUILDER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_page_nav_matches_canonical():
    """Regenerating must be a no-op. If this fails, run:

    python3 scripts/site_nav.py
    """
    mod = _load_site_nav()
    stale = []
    for path in sorted(_DOCS.glob("*.html")):
        if path.name in mod.SKIP:
            continue
        before = path.read_text(encoding="utf-8")
        after = mod.apply_to_text(before, path.name)
        if after != before:
            stale.append(path.name)
    assert not stale, f"nav out of sync on: {stale}. Run: python3 scripts/site_nav.py"


def test_canonical_set_covers_every_page_but_landing_and_which_mode():
    """which-mode.html is deliberately topbar-only; every other non-index page
    must be reachable from the sidebar, or it is an orphan (the bug that hid
    benchmarks.html from every page's nav)."""
    mod = _load_site_nav()
    on_disk = {p.name for p in _DOCS.glob("*.html")} - mod.SKIP - {"which-mode.html"}
    missing = sorted(on_disk - mod.ALL_SIDEBAR_HREFS)
    assert not missing, f"orphaned from the sidebar: {missing}"


def test_which_mode_is_on_every_topbar():
    """9 pages had a shorter topbar variant that silently dropped this link."""
    for path in sorted(_DOCS.glob("*.html")):
        if path.name in {"index.html", "which-mode.html"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert 'href="which-mode.html"' in text, f"{path.name} topbar is missing Which Mode?"

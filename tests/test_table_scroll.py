"""Every docs table must ship inside a `.table-scroll` region, and every page
must ship the shared footer.

Both are site chrome that had drifted almost completely: 104 tables across 34
pages had no wrapper in the HTML (site.js added one at runtime, so a reader
without JavaScript — or looking at the first frame — got a table wider than the
phone), and 16 of 44 pages ended right after `</main>` with no footer at all.

Same regenerate-and-compare pattern as tests/test_site_nav.py and
tests/test_search_index.py: the generator is the source of truth, and this test
fails the moment a page diverges from it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_SCRIPTS = _ROOT / "scripts"

pytestmark = pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not present in this checkout")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_table_is_wrapped():
    """Regenerating must be a no-op. If this fails, run:

    python3 scripts/wrap_tables.py
    """
    mod = _load("wrap_tables")
    stale = []
    for path in sorted(_DOCS.glob("*.html")):
        if path.name in mod.SKIP:
            continue
        before = path.read_text(encoding="utf-8")
        if mod.apply_to_text(before) != before:
            stale.append(path.name)
    assert not stale, f"unwrapped tables on: {stale}. Run: python3 scripts/wrap_tables.py"


def test_wrapping_is_idempotent_and_keeps_the_a11y_attributes():
    """site.js deliberately skips any table already inside `.table-scroll`, so a
    bare `<div class="table-scroll">` here would *remove* the focusability the
    runtime wrapper used to add. The static wrapper must carry it itself, and a
    second pass must not nest a wrapper inside a wrapper."""
    mod = _load("wrap_tables")
    src = "<h2>Flags</h2>\n<table><tr><td>x</td></tr></table>\n"
    once = mod.apply_to_text(src)
    assert once != src
    assert 'class="table-scroll"' in once
    assert 'tabindex="0"' in once
    assert 'role="region"' in once
    assert 'aria-label="Flags"' in once
    assert mod.apply_to_text(once) == once
    assert once.count("table-scroll") == 1


def test_wrapper_label_is_attribute_escaped():
    """A heading is arbitrary page text going into a quoted HTML attribute."""
    mod = _load("wrap_tables")
    out = mod.apply_to_text('<h2>The "big" table</h2>\n<table><tr><td>x</td></tr></table>')
    assert 'aria-label="The &quot;big&quot; table"' in out


def test_every_page_has_the_shared_footer():
    """16 of 44 pages had no footer element at all — no licence line, no repo
    link, and nothing closing the page."""
    mod = _load("site_nav")
    missing = [
        p.name
        for p in sorted(_DOCS.glob("*.html"))
        if p.name not in mod.SKIP and not mod.has_footer(p.read_text(encoding="utf-8"))
    ]
    assert not missing, f"no footer on: {missing}. Run: python3 scripts/site_nav.py"


def test_no_page_grew_a_second_footer():
    """The insertion is guarded by `has_footer`, but a second footer would be
    invisible in a nav diff and wrong on every page it reached."""
    mod = _load("site_nav")
    for path in sorted(_DOCS.glob("*.html")):
        if path.name in mod.SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        assert len(re.findall(r'<div class="site-footer">', text)) == 1, path.name

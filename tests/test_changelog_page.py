"""docs/changelog.html must be exactly what scripts/build_changelog_page.py renders.

CHANGELOG.md was previously invisible from the site — no page linked it, no page
rendered it. The generated page must never drift from its source: the same
regenerate-and-compare pattern as tests/test_search_index.py and
tests/test_site_nav.py.
"""

from __future__ import annotations

import importlib.util
from html.parser import HTMLParser
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CHANGELOG = _ROOT / "CHANGELOG.md"
_OUT = _ROOT / "docs" / "changelog.html"
_BUILDER = _ROOT / "scripts" / "build_changelog_page.py"

pytestmark = pytest.mark.skipif(
    not (_CHANGELOG.is_file() and _OUT.is_file()),
    reason="CHANGELOG.md or docs/changelog.html not present in this checkout",
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_changelog_page", _BUILDER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_changelog_page_matches_generator():
    """Regenerating must be a no-op. If this fails, run:

    python3 scripts/build_changelog_page.py
    """
    mod = _load_builder()
    committed = _OUT.read_text(encoding="utf-8")
    regenerated = mod.build()
    assert regenerated == committed, (
        "docs/changelog.html is stale. Run: python3 scripts/build_changelog_page.py"
    )


def test_every_version_heading_is_present():
    """Spot-check: every `## [x.y.z]` entry in CHANGELOG.md gets its own anchored
    section, so a reader following a #vX-Y-Z link never hits a missing id."""
    mod = _load_builder()
    changelog_md = _CHANGELOG.read_text(encoding="utf-8")
    versions = [
        mod._VERSION_RE.match(line).group(1)
        for line in changelog_md.splitlines()
        if mod._VERSION_RE.match(line)
    ]
    assert versions, "no '## [version]' headers found in CHANGELOG.md"
    page = _OUT.read_text(encoding="utf-8")
    missing = [v for v in versions if f'id="{mod._slug(v)}"' not in page]
    assert not missing, f"versions missing an anchored section on the page: {missing}"


def test_changelog_page_is_well_formed_html():
    """The hand-rolled inline-markdown renderer has already produced unbalanced
    tags once (a literal `*` inside a `` `code span` `` was read as emphasis and
    swallowed real markup downstream) — catch any regression of that class."""

    class _BalanceChecker(HTMLParser):
        VOID = {
            "br",
            "img",
            "hr",
            "meta",
            "link",
            "input",
            "area",
            "base",
            "col",
            "embed",
            "source",
            "track",
            "wbr",
        }

        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in self.VOID:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in self.VOID:
                return
            if not self.stack or self.stack[-1] != tag:
                self.errors.append(f"</{tag}> at {self.getpos()} does not match {self.stack[-3:]}")
            else:
                self.stack.pop()

    checker = _BalanceChecker()
    checker.feed(_OUT.read_text(encoding="utf-8"))
    assert not checker.errors, f"unbalanced tags: {checker.errors[:5]}"
    assert not checker.stack, f"unclosed tags at EOF: {checker.stack}"

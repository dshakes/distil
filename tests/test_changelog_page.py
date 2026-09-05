"""docs/changelog.html must be exactly what scripts/build_changelog_page.py renders.

CHANGELOG.md was previously invisible from the site — no page linked it, no page
rendered it. The generated page must never drift from its source: the same
regenerate-and-compare pattern as tests/test_search_index.py and
tests/test_site_nav.py.
"""

from __future__ import annotations

import importlib.util
import re
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


def test_ids_on_page_are_unique():
    """CHANGELOG.md has reused a version number across two entries before (two
    `## [1.13.0]` sections), which collided on the same `id="v1-13-0"` — an
    HTML validity bug and a broken deep link (the browser jumps to whichever
    one comes first). Every id on the page must be distinct."""
    mod = _load_builder()

    class _IdCollector(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.ids: list[str] = []

        def handle_starttag(self, tag, attrs):
            for name, value in attrs:
                if name == "id" and value:
                    self.ids.append(value)

    collector = _IdCollector()
    collector.feed(mod.build())
    dupes = {i for i in collector.ids if collector.ids.count(i) > 1}
    assert not dupes, f"duplicate ids on the page: {sorted(dupes)}"


def test_changelog_page_is_well_formed_html():
    """The hand-rolled inline-markdown renderer has already produced unbalanced
    tags once (a literal `*` inside a `` `code span` `` was read as emphasis and
    swallowed real markup downstream) — catch any regression of that class.

    Also asserts the three structural facts a Codex cross-audit of PR #161 found
    broken: nested emphasis leaving literal `**` in the page, and CHANGELOG.md's
    indented code blocks / GFM table being silently flattened to plain text.
    """

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

    text = _OUT.read_text(encoding="utf-8")
    checker = _BalanceChecker()
    checker.feed(text)
    assert not checker.errors, f"unbalanced tags: {checker.errors[:5]}"
    assert not checker.stack, f"unclosed tags at EOF: {checker.stack}"

    no_code = re.sub(r"<code>.*?</code>", "", text, flags=re.DOTALL)
    assert "**" not in no_code, "a literal '**' survived outside a <code> span"
    assert "<pre><code>" in text, "no indented code block was rendered"
    assert "<table>" in text, "no GFM table was rendered"


def test_nested_emphasis_renders_correctly():
    """CHANGELOG.md has `**Extra descriptors were leaked *and* hung.**` — a bold
    span containing a nested italic. The old `_BOLD` regex (`\\*\\*([^*]+)\\*\\*`)
    can never match this, because its content class excludes `*` outright."""
    mod = _load_builder()
    rendered = mod._inline("**Extra descriptors were leaked *and* hung.**")
    assert rendered == "<strong>Extra descriptors were leaked <em>and</em> hung.</strong>"


def test_href_rejects_attribute_breakout_and_dangerous_schemes():
    """A markdown link target is written into an HTML attribute, not text
    content — the one place in this renderer where raw source text can break
    out of a quoted string, and where a `javascript:` URI needs no breakout
    at all to execute on click."""
    mod = _load_builder()

    breakout = mod._inline('[x](" onmouseover="alert(1))')
    assert 'onmouseover="alert' not in breakout, breakout
    assert "&quot;" in breakout, breakout

    js = mod._inline("[x](javascript:alert(1))")
    assert "<a href" not in js, js

    # legitimate link forms actually used in CHANGELOG.md must still work
    assert mod._inline("[docs/CACHE.md](docs/CACHE.md)") == (
        '<a href="docs/CACHE.md">docs/CACHE.md</a>'
    )
    assert mod._inline("[semver.org](https://semver.org/)") == (
        '<a href="https://semver.org/">semver.org</a>'
    )


def test_indented_code_block_renders_as_pre_code():
    """CHANGELOG.md's 4-space-indented blocks (e.g. the json-in-prose numbers)
    were being flattened into plain paragraph/list text; they must render as
    a real <pre><code> block, dedented."""
    mod = _load_builder()
    lines = [
        "Some intro text.",
        "",
        "    json-in-prose   2405 -> 507 tok   78.9% saved   (was 0.0%)",
        "    gh api dump     2083 -> 371 tok   82.2% saved   (was 0.0%)",
        "",
        "Wired into both fold chains.",
    ]
    rendered = mod._render_block(lines)
    assert "<pre><code>" in rendered
    # Code-block content is HTML-escaped, so `->` becomes `-&gt;` like every
    # other `>` in the page — this is correct, not a bug in the fix.
    assert "json-in-prose   2405 -&gt; 507 tok   78.9% saved   (was 0.0%)" in rendered
    assert "gh api dump     2083 -&gt; 371 tok   82.2% saved   (was 0.0%)" in rendered


def test_gfm_table_renders_as_table():
    """CHANGELOG.md's one real GFM pipe table (the htmlx.py before/after numbers)
    was being flattened to plain text; it must render as a real <table> with a
    <thead> header row and a <tbody> body."""
    mod = _load_builder()
    lines = [
        "Intro line.",
        "",
        "| page | before | after | saved | facts lost |",
        "|---|---|---|---|---|",
        "| Wikipedia article | 281,093 tok | 14,260 tok | **94.9%** | **0** |",
        "| Python docs page | 32,322 tok | 4,229 tok | **86.9%** | 0 |",
        "",
        "Deliberately recall-biased.",
    ]
    rendered = mod._render_block(lines)
    assert "<table>" in rendered
    assert "<thead>" in rendered and "<tbody>" in rendered
    assert "<th>page</th>" in rendered
    assert "<td>Wikipedia article</td>" in rendered
    assert "<td><strong>94.9%</strong></td>" in rendered

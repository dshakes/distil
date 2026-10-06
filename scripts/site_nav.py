#!/usr/bin/env python3
"""Regenerate the shared topbar + sidebar navigation across the docs site — stdlib only.

The topbar and sidebar are hand-duplicated on every page and had drifted: cli.html
was missing its own Library API link, benchmark.html dropped its two sibling
benchmark pages, 9 pages never got the "Which Mode?" topbar link (so it lived as
a workaround duplicate in metrics.html's sidebar instead), and benchmarks.html was
not reachable from any page's nav at all — it took a link from faq.html's prose to
find it. This renders both blocks from one canonical structure per page so that
class of drift cannot happen again silently.

Usage: python3 scripts/site_nav.py [docs_dir]
Checked by tests/test_site_nav.py (regenerate-and-compare, same pattern as
build_search_index.py / test_search_index.py).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pages with their own bespoke navigation, not the shared template.
SKIP = {"index.html"}  # extended with REDIRECTS below

# The release that introduced each page (from `git log --diff-filter=A` + the first
# tag containing it). A page shows a "New" badge only while that release is within
# NEW_WINDOW minor versions of the current one, so badges retire themselves as
# releases ship instead of every page staying "New" forever. Add a row for each
# new page; pages without a row never get the badge.
ADDED: dict[str, str] = {
    "cache.html": "1.41.0",
    "which-mode.html": "1.48.1",
    "ab.html": "1.55.0",
    "hooks.html": "1.55.0",
    "mcp.html": "1.55.0",
    "scoreboard.html": "1.57.0",
}
NEW_WINDOW = 1  # the current minor release only


def _minor(version: str) -> tuple[int, int]:
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()  # type: ignore[union-attr]
    return int(major), int(minor)


def current_version() -> str:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)  # type: ignore[union-attr]


def is_new(href: str, version: str | None = None) -> bool:
    added = ADDED.get(href)
    if added is None:
        return False
    (cmaj, cmin), (amaj, amin) = _minor(version or current_version()), _minor(added)
    return amaj == cmaj and 0 <= cmin - amin < NEW_WINDOW


# (href, label_html, badge|None)
GETTING_STARTED: list[tuple[str, str, str | None]] = [
    ("getting-started.html", "Install &amp; Quickstart", None),
]

LEARN: list[tuple[str, str, str | None]] = [
    ("token-economics.html", "Token Economics", "Start here"),
    ("concepts.html", "Concepts &amp; Course", None),
    ("techniques.html", "Techniques", None),
    ("architecture.html", "Architecture", None),
    ("cache.html", "Prompt Caching", None),
    ("research.html", "Research &amp; Frontier", None),
]

EVALUATION: list[tuple[str, str, str | None]] = [
    ("scoreboard.html", "Scoreboard", None),
    ("benchmark.html", "Benchmarks &amp; Evals", None),
    ("ab.html", "Task-level A/B", None),
    ("compare.html", "Compare", None),
    ("adoption.html", "Adoption &amp; Metrics", "Live"),
]

REFERENCE: list[tuple[str, str, str | None]] = [
    ("cli.html", "CLI Reference", None),
]

AGENT_TOOLING: list[tuple[str, str, str | None]] = [
    ("hooks.html", "Post-tool Hooks", None),
    ("mcp.html", "MCP Compressor", None),
    ("integrations.html", "Integrations &amp; Library", None),
]

MORE: list[tuple[str, str, str | None]] = [
    ("faq.html", "FAQ", None),
    ("security.html", "Security &amp; Deploy", None),
    ("changelog.html", "Changelog", None),
]

# Sidebar sections in order: (title, entries, open by default). Every section is a
# native <details> — no JS, keyboard and screen-reader accessible — and the one
# holding the current page is always open, so a flat link list reads as 6 groups.
SECTIONS: list[tuple[str, list[tuple[str, str, str | None]], bool]] = [
    ("Getting started", GETTING_STARTED, True),
    ("Learn", LEARN, True),
    ("Evaluation &amp; benchmarks", EVALUATION, False),
    ("Reference", REFERENCE, False),
    ("Agent tooling", AGENT_TOOLING, False),
    ("More", MORE, False),
]

# Pages that were merged into another (2026-10 surface cleanup). Each old URL stays
# alive as a tiny redirect stub so no inbound link 404s; the stub is written by
# main() and skipped by every nav/table/search generator and check.
REDIRECTS: dict[str, str] = {
    "adapters.html": "integrations.html#adapters",
    "agno.html": "integrations.html#agno-page",
    "anthropic-sdk.html": "integrations.html#anthropic-sdk",
    "asgi.html": "integrations.html#asgi-page",
    "autogen.html": "integrations.html#autogen-page",
    "benchmark-independent.html": "benchmark.html#benchmark-independent",
    "benchmarks.html": "benchmark.html#benchmarks",
    "cache-contract.html": "cache.html#cache-contract",
    "code-skeletons.html": "techniques.html#code-skeletons",
    "corpus.html": "benchmark.html#corpus-page",
    "crewai.html": "integrations.html#crewai-page",
    "deploy-security.html": "security.html#deploy-security",
    "evals.html": "benchmark.html#evals",
    "langchain.html": "integrations.html#langchain",
    "langgraph.html": "integrations.html#langgraph",
    "learn-compression.html": "concepts.html#learn-compression",
    "learn-distil.html": "concepts.html#learn-distil",
    "learn-tokens.html": "concepts.html#learn-tokens",
    "library.html": "integrations.html#library",
    "litellm.html": "integrations.html#litellm",
    "llamaindex.html": "integrations.html#llamaindex-page",
    "metrics.html": "adoption.html#metrics",
    "model-migration.html": "ab.html#model-migration",
    "openai-sdk.html": "integrations.html#openai-sdk",
    "output.html": "techniques.html#output",
    "provider-compaction.html": "research.html#provider-compaction",
    "strands.html": "integrations.html#strands-page",
    "subscription.html": "which-mode.html#subscription",
    "threat-model.html": "security.html#threat-model",
    "vercel-ai-sdk.html": "integrations.html#vercel-ai-sdk",
}

# Every href the canonical sidebar/topbar can render — used by check_nav.py and
# by the completeness self-test below.
ALL_SIDEBAR_HREFS = {h for _, entries, _ in SECTIONS for h, _, _ in entries}

SKIP = SKIP | set(REDIRECTS)  # stubs carry no sidebar, footer or tables


def _badge(text: str | None) -> str:
    return f' <span class="nav-badge">{text}</span>' if text else ""


def _li(
    active: str, href: str, label: str, badge: str | None = None, indent: str = "        "
) -> str:
    cls = ' class="active" aria-current="page"' if href == active else ""
    badge = badge or ("New" if is_new(href) else None)
    return f'{indent}<li><a href="{href}"{cls}>{label}{_badge(badge)}</a></li>'


def render_topbar_links(active: str) -> str:
    wm_cls = ' class="active" aria-current="page"' if active == "which-mode.html" else ""
    return (
        '  <nav class="topbar-links">\n'
        f'    <a href="which-mode.html"{wm_cls}>Which Mode?{_badge("New" if is_new("which-mode.html") else None)}</a>\n'
        '    <a href="getting-started.html">Docs</a>\n'
        '    <a href="https://github.com/dshakes/distil" target="_blank" rel="noopener">GitHub →</a>\n'
        "  </nav>"
    )


def render_sidebar(active: str) -> str:
    lines = ['  <aside class="sidebar" id="sidebar">', '    <nav aria-label="Documentation">']
    for title, entries, default_open in SECTIONS:
        hrefs = {h for h, _, _ in entries}
        is_open = default_open or active in hrefs
        lines.append(f'      <details class="sidebar-group"{" open" if is_open else ""}>')
        lines.append(f'        <summary class="sidebar-section">{title}</summary>')
        lines.append("        <ul>")
        for href, label, badge in entries:
            lines.append(_li(active, href, label, badge, indent="          "))
        if entries is MORE:
            lines.append(_li(active, "index.html", "← Landing page", indent="          "))
        lines.append("        </ul>")
        lines.append("      </details>")
    lines.append("    </nav>")
    lines.append("  </aside>")
    return "\n".join(lines)


_TOPBAR_RE = re.compile(r'  <nav class="topbar-links">.*?\n  </nav>', re.S)
_SIDEBAR_RE = re.compile(r'  <aside class="sidebar" id="sidebar">.*?\n  </aside>', re.S)

# The third piece of shared chrome, and the one that drifted furthest: 16 of 44
# pages ended right after </main> with no footer at all, so over a third of the
# site had no licence line and no repo link. Same markup as the 28 pages that do
# have one — this closes the drift, it does not redesign the footer.
FOOTER = (
    '    <div class="site-footer">\n'
    "      Distil · compression with a quality contract · Apache-2.0 · "
    '<a href="https://github.com/dshakes/distil">github.com/dshakes/distil</a>\n'
    "    </div>"
)
_FOOTER_RE = re.compile(r'<div class="site-footer">.*?</div>', re.S)
_MAIN_CLOSE_RE = re.compile(r"\n([ \t]*)</main>")


def has_footer(text: str) -> bool:
    return _FOOTER_RE.search(text) is not None


def apply_to_text(text: str, active: str) -> str:
    text = _TOPBAR_RE.sub(lambda _m: render_topbar_links(active), text, count=1)
    text = _SIDEBAR_RE.sub(lambda _m: render_sidebar(active), text, count=1)
    if not has_footer(text):
        # Inserted just inside </main>, where every existing footer already sits.
        text = _MAIN_CLOSE_RE.sub(lambda m: f"\n\n{FOOTER}\n{m.group(1)}</main>", text, count=1)
    return text


_LINK_BADGE_RE = re.compile(
    r'(<a href="(?P<href>[a-z0-9-]+\.html)"[^>]*>[^<]*?)(?: <span class="nav-badge">New</span>)?(</a>)'
)


def refresh_new_badges(text: str) -> str:
    """Re-derive every "New" badge on links to pages in ADDED (for the bespoke landing
    page, whose hand-written nav the shared template does not render)."""

    def fix(m: re.Match[str]) -> str:
        if m.group("href") not in ADDED or m.group(1).rstrip().endswith(("&rarr;", "→")):
            return m.group(0)  # not a tracked page, or a call-to-action arrow link
        badge = ' <span class="nav-badge">New</span>' if is_new(m.group("href")) else ""
        return f"{m.group(1).rstrip()}{badge}{m.group(3)}"

    return _LINK_BADGE_RE.sub(fix, text)


_STUB = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moved - Distil</title>
<meta name="robots" content="noindex">
<meta name="distil-redirect" content="{to}">
<link rel="canonical" href="https://dshakes.github.io/distil/{to}">
<meta http-equiv="refresh" content="0; url={to}">
</head>
<body>
<p>This page moved to <a href="{to}">{to}</a>.</p>
</body>
</html>
"""


def is_redirect(text: str) -> bool:
    return 'name="distil-redirect"' in text


def write_redirects(docs: Path) -> int:
    n = 0
    for old, to in REDIRECTS.items():
        stub = _STUB.format(to=to)
        p = docs / old
        if not p.exists() or p.read_text(encoding="utf-8") != stub:
            p.write_text(stub, encoding="utf-8")
            n += 1
    return n


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    changed = write_redirects(docs)
    for path in sorted(docs.glob("*.html")):
        before = path.read_text(encoding="utf-8")
        if path.name in SKIP:
            after = refresh_new_badges(before)
        else:
            after = apply_to_text(before, path.name)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed += 1
    print(f"{docs}: synced nav on {changed} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

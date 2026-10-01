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
SKIP = {"index.html"}

# The release that introduced each page (from `git log --diff-filter=A` + the first
# tag containing it). A page shows a "New" badge only while that release is within
# NEW_WINDOW minor versions of the current one, so badges retire themselves as
# releases ship instead of every page staying "New" forever. Add a row for each
# new page; pages without a row never get the badge.
ADDED: dict[str, str] = {
    "provider-compaction.html": "1.33.0",
    "cache.html": "1.41.0",
    "library.html": "1.42.0",
    "subscription.html": "1.48.0",
    "which-mode.html": "1.48.1",
    "benchmark-independent.html": "1.50.1",
    "cache-contract.html": "1.52.0",
    "threat-model.html": "1.52.0",
    "ab.html": "1.55.0",
    "hooks.html": "1.55.0",
    "code-skeletons.html": "1.55.0",
    "mcp.html": "1.55.0",
    "model-migration.html": "1.56.0",
}
NEW_WINDOW = 2  # the current minor release and the one before it


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
    ("concepts.html", "Concepts", None),
    ("techniques.html", "Techniques", None),
    ("architecture.html", "Architecture", None),
    ("research.html", "Research &amp; Frontier", None),
    ("cache-contract.html", "Cache Contract", None),
    ("subscription.html", "Subscription", None),
    ("provider-compaction.html", "Provider Compaction", None),
]

# Nested under Token Economics — the 3-module course, previously two clicks deep
# (sidebar -> hub -> module) on every other page.
COURSE_MODULES: list[tuple[str, str]] = [
    ("learn-tokens.html", "1. Fundamentals"),
    ("learn-compression.html", "2. Compression"),
    ("learn-distil.html", "3. Distil &amp; Proof"),
]

EVALUATION: list[tuple[str, str, str | None]] = [
    ("evals.html", "Evaluation", None),
    ("ab.html", "Task-level A/B", None),
    ("model-migration.html", "Model Migration", None),
    ("benchmark-independent.html", "Independent Benchmark", None),
    ("benchmark.html", "Live Benchmark", None),
    ("benchmarks.html", "Reproduce Benchmarks", None),
    ("compare.html", "Compare", None),
    ("adoption.html", "Adoption", "Live"),
]

REFERENCE: list[tuple[str, str, str | None]] = [
    ("library.html", "Library API", None),
    ("cli.html", "CLI Reference", None),
    ("metrics.html", "Metrics &amp; Observability", None),
    ("cache.html", "Prompt Caching", None),
    ("output.html", "Output &amp; I/O", None),
    ("corpus.html", "Corpus", None),
]

AGENT_TOOLING: list[tuple[str, str, str | None]] = [
    ("hooks.html", "Post-tool Hooks", None),
    ("code-skeletons.html", "Code Skeletons", None),
    ("mcp.html", "MCP Compressor", None),
]

INTEGRATIONS: list[tuple[str, str, str | None]] = [
    ("integrations.html", "Overview", None),
    ("adapters.html", "Adapters", None),
    ("anthropic-sdk.html", "Anthropic SDK", None),
    ("openai-sdk.html", "OpenAI SDK", None),
    ("litellm.html", "LiteLLM", None),
    ("langchain.html", "LangChain", None),
    ("langgraph.html", "LangGraph", None),
    ("vercel-ai-sdk.html", "Vercel AI SDK", None),
    ("agno.html", "Agno", None),
    ("strands.html", "Strands", None),
    ("autogen.html", "AutoGen", None),
    ("llamaindex.html", "LlamaIndex", None),
    ("crewai.html", "CrewAI", None),
    ("asgi.html", "ASGI Middleware", None),
]

MORE: list[tuple[str, str, str | None]] = [
    ("faq.html", "FAQ", None),
    ("security.html", "Security", None),
    ("deploy-security.html", "Deploy &amp; Security", None),
    ("threat-model.html", "Threat Model", None),
    ("changelog.html", "Changelog", None),
]

# Sidebar sections in order: (title, entries, open by default). Every section is a
# native <details> — no JS, keyboard and screen-reader accessible — and the one
# holding the current page is always open, so a flat ~60-link list reads as 7 groups.
SECTIONS: list[tuple[str, list[tuple[str, str, str | None]], bool]] = [
    ("Getting started", GETTING_STARTED, True),
    ("Learn", LEARN, True),
    ("Evaluation &amp; benchmarks", EVALUATION, False),
    ("Reference", REFERENCE, False),
    ("Agent tooling", AGENT_TOOLING, False),
    ("Integrations", INTEGRATIONS, False),
    ("More", MORE, False),
]

# Every href the canonical sidebar/topbar can render — used by check_nav.py and
# by the completeness self-test below.
ALL_SIDEBAR_HREFS = {h for _, entries, _ in SECTIONS for h, _, _ in entries} | {
    h for h, _ in COURSE_MODULES
}


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
        if entries is LEARN:
            hrefs |= {h for h, _ in COURSE_MODULES}
        is_open = default_open or active in hrefs
        lines.append(f'      <details class="sidebar-group"{" open" if is_open else ""}>')
        lines.append(f'        <summary class="sidebar-section">{title}</summary>')
        lines.append("        <ul>")
        for href, label, badge in entries:
            if href == "token-economics.html":
                cls = ' class="active" aria-current="page"' if href == active else ""
                lines.append(f'          <li><a href="{href}"{cls}>{label}{_badge(badge)}</a>')
                lines.append('            <ul class="sidebar-sub">')
                for mhref, mlabel in COURSE_MODULES:
                    mcls = ' class="active" aria-current="page"' if mhref == active else ""
                    lines.append(f'              <li><a href="{mhref}"{mcls}>{mlabel}</a></li>')
                lines.append("            </ul></li>")
            else:
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


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    changed = 0
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

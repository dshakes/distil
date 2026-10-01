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
    ("cache-contract.html", "Cache Contract", "New"),
    ("subscription.html", "Subscription", "New"),
    ("provider-compaction.html", "Provider Compaction", "New"),
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
    ("ab.html", "Task-level A/B", "New"),
    ("model-migration.html", "Model Migration", "New"),
    ("benchmark-independent.html", "Independent Benchmark", "New"),
    ("benchmark.html", "Live Benchmark", None),
    ("benchmarks.html", "Reproduce Benchmarks", None),
    ("compare.html", "Compare", None),
    ("adoption.html", "Adoption", "Live"),
]

REFERENCE: list[tuple[str, str, str | None]] = [
    ("library.html", "Library API", "New"),
    ("cli.html", "CLI Reference", None),
    ("metrics.html", "Metrics &amp; Observability", None),
    ("cache.html", "Prompt Caching", "New"),
    ("output.html", "Output &amp; I/O", None),
    ("corpus.html", "Corpus", None),
]

AGENT_TOOLING: list[tuple[str, str, str | None]] = [
    ("hooks.html", "Post-tool Hooks", "New"),
    ("code-skeletons.html", "Code Skeletons", "New"),
    ("mcp.html", "MCP Compressor", "New"),
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
    ("threat-model.html", "Threat Model", "New"),
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
    return f'{indent}<li><a href="{href}"{cls}>{label}{_badge(badge)}</a></li>'


def render_topbar_links(active: str) -> str:
    wm_cls = ' class="active" aria-current="page"' if active == "which-mode.html" else ""
    return (
        '  <nav class="topbar-links">\n'
        f'    <a href="which-mode.html"{wm_cls}>Which Mode? <span class="nav-badge">New</span></a>\n'
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


def main(argv: list[str]) -> int:
    docs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "docs"
    changed = 0
    for path in sorted(docs.glob("*.html")):
        if path.name in SKIP:
            continue
        before = path.read_text(encoding="utf-8")
        after = apply_to_text(before, path.name)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed += 1
    print(f"{docs}: synced nav on {changed} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

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
    ("cache-contract.html", "Cache Contract", "New"),
    ("subscription.html", "Subscription", "New"),
    ("research.html", "Research &amp; Frontier", None),
    ("provider-compaction.html", "Provider Compaction", "New"),
    ("evals.html", "Evaluation", None),
    ("benchmark-independent.html", "Independent Benchmark", "New"),
    ("benchmark.html", "Live Benchmark", None),
    ("benchmarks.html", "Reproduce Benchmarks", None),
    ("compare.html", "Compare", None),
    ("adoption.html", "Adoption", "Live"),
]

# Nested under Token Economics — the 3-module course, previously two clicks deep
# (sidebar -> hub -> module) on every other page.
COURSE_MODULES: list[tuple[str, str]] = [
    ("learn-tokens.html", "1. Fundamentals"),
    ("learn-compression.html", "2. Compression"),
    ("learn-distil.html", "3. Distil &amp; Proof"),
]

REFERENCE: list[tuple[str, str, str | None]] = [
    ("library.html", "Library API", "New"),
    ("cli.html", "CLI Reference", None),
    ("adapters.html", "Adapters", None),
    ("metrics.html", "Metrics &amp; Observability", None),
    ("cache.html", "Prompt Caching", "New"),
    ("corpus.html", "Corpus", None),
    ("output.html", "Output &amp; I/O", None),
]

INTEGRATIONS_SUB: list[tuple[str, str]] = [
    ("anthropic-sdk.html", "Anthropic SDK"),
    ("openai-sdk.html", "OpenAI SDK"),
    ("litellm.html", "LiteLLM"),
    ("langchain.html", "LangChain"),
    ("vercel-ai-sdk.html", "Vercel AI SDK"),
    ("agno.html", "Agno"),
    ("strands.html", "Strands"),
    ("autogen.html", "AutoGen"),
    ("crewai.html", "CrewAI"),
    ("asgi.html", "ASGI Middleware"),
]

# faq.html, then integrations.html (spliced in with its sublist), then these.
MORE: list[tuple[str, str, str | None]] = [
    ("faq.html", "FAQ", None),
    ("security.html", "Security", None),
    ("changelog.html", "Changelog", None),
    ("deploy-security.html", "Deploy &amp; Security", None),
    ("threat-model.html", "Threat Model", "New"),
]

# Every href the canonical sidebar/topbar can render — used by check_nav.py and
# by the completeness self-test below.
ALL_SIDEBAR_HREFS = (
    {h for h, _, _ in GETTING_STARTED}
    | {h for h, _, _ in LEARN}
    | {h for h, _ in COURSE_MODULES}
    | {h for h, _, _ in REFERENCE}
    | {h for h, _ in INTEGRATIONS_SUB}
    | {h for h, _, _ in MORE}
    | {"integrations.html"}
)


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

    lines.append('      <h2 class="sidebar-section">Getting started</h2>')
    lines.append("      <ul>")
    for href, label, badge in GETTING_STARTED:
        lines.append(_li(active, href, label, badge))
    lines.append("      </ul>")
    lines.append("")

    lines.append('      <h2 class="sidebar-section">Learn</h2>')
    lines.append("      <ul>")
    for href, label, badge in LEARN:
        if href == "token-economics.html":
            cls = ' class="active" aria-current="page"' if href == active else ""
            lines.append(f'        <li><a href="{href}"{cls}>{label}{_badge(badge)}</a>')
            lines.append('          <ul class="sidebar-sub">')
            for mhref, mlabel in COURSE_MODULES:
                mcls = ' class="active" aria-current="page"' if mhref == active else ""
                lines.append(f'            <li><a href="{mhref}"{mcls}>{mlabel}</a></li>')
            lines.append("          </ul></li>")
        else:
            lines.append(_li(active, href, label, badge))
    lines.append("      </ul>")
    lines.append("")

    lines.append('      <h2 class="sidebar-section">Reference</h2>')
    lines.append("      <ul>")
    for href, label, badge in REFERENCE:
        lines.append(_li(active, href, label, badge))
    lines.append("      </ul>")
    lines.append("")

    lines.append('      <h2 class="sidebar-section">More</h2>')
    lines.append("      <ul>")
    for href, label, badge in MORE:
        lines.append(_li(active, href, label, badge))
        if href == "faq.html":
            icls = ' class="active" aria-current="page"' if active == "integrations.html" else ""
            lines.append(f'        <li><a href="integrations.html"{icls}>Integrations</a>')
            lines.append('          <ul class="sidebar-sub">')
            for ihref, ilabel in INTEGRATIONS_SUB:
                ic = ' class="active" aria-current="page"' if ihref == active else ""
                lines.append(f'            <li><a href="{ihref}"{ic}>{ilabel}</a></li>')
            lines.append("          </ul></li>")
    lines.append(_li(active, "index.html", "← Landing page"))
    lines.append("      </ul>")
    lines.append("    </nav>")
    lines.append("  </aside>")
    return "\n".join(lines)


_TOPBAR_RE = re.compile(r'  <nav class="topbar-links">.*?\n  </nav>', re.S)
_SIDEBAR_RE = re.compile(r'  <aside class="sidebar" id="sidebar">.*?\n  </aside>', re.S)


def apply_to_text(text: str, active: str) -> str:
    text = _TOPBAR_RE.sub(lambda _m: render_topbar_links(active), text, count=1)
    text = _SIDEBAR_RE.sub(lambda _m: render_sidebar(active), text, count=1)
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

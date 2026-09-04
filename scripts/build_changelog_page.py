#!/usr/bin/env python3
"""Render CHANGELOG.md into docs/changelog.html — stdlib only.

CHANGELOG.md was invisible to anyone who didn't already know to look in the repo
root; the site had 35+ pages and no changelog. This renders it into the same
template shell as every other page (topbar/sidebar via scripts/site_nav.py, so
it can never drift from the canonical nav), one `<section>` per `## [version]`
entry with a stable anchor id.

The converter below is a small hand-rolled subset of Markdown, not a general
parser: it supports exactly the constructs CHANGELOG.md actually uses (`##`/`###`
headers, fenced code blocks, blockquotes, one level of nested bullet lists with
wrapped continuation lines, `` `code` ``, **bold**, *italic*, and `[text](url)`
links). Checked by tests/test_changelog_page.py (regenerate-and-compare, same
pattern as build_search_index.py / test_search_index.py).

Usage: python3 scripts/build_changelog_page.py [changelog_md] [out_html]
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_nav  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_CHANGELOG = _ROOT / "CHANGELOG.md"
_OUT = _ROOT / "docs" / "changelog.html"

_VERSION_RE = re.compile(r"^## \[([^\]]+)\](?: — (.*))?\s*$")
_H3_RE = re.compile(r"^### (.*)$")
_BULLET_RE = re.compile(r"^(\s*)- (.*)$")

_INLINE_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")


def _inline(text: str) -> str:
    """Render inline Markdown (code/link/bold/italic) inside already-plain text.

    Code spans are pulled out and stashed before bold/italic run, and restored
    verbatim afterward — a literal `*` or `_` inside `` `usage.*` `` must never
    be read as an emphasis marker for text outside the span.
    """
    text = html.escape(text, quote=False)
    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        stashed.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(stashed) - 1}\x00"

    text = _INLINE_CODE.sub(_stash, text)
    text = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)
    text = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", text)
    for idx, code_html in enumerate(stashed):
        text = text.replace(f"\x00{idx}\x00", code_html)
    return text


def _slug(version: str) -> str:
    return "v" + re.sub(r"[^a-zA-Z0-9]+", "-", version).strip("-")


def _join_wrapped(parts: list[str]) -> str:
    """Join wrapped source lines back into one, without inserting a space that
    would land inside an inline `code span` a hard line-wrap happened to split."""
    result = parts[0] if parts else ""
    for part in parts[1:]:
        sep = "" if result.count("`") % 2 == 1 else " "
        result += sep + part
    return result


def _render_block(lines: list[str]) -> str:
    """Render one version entry's body lines (below the `## [x.y.z]` header)."""
    out: list[str] = []
    para: list[str] = []
    # Each open <ul> tracked as [indent_of_its_bullets, pending_li_text_or_None,
    # li_already_opened_in_out]. `opened` is set when a nested <ul> had to start
    # while this <li>'s text was still pending, so its <li> tag was written early
    # (unclosed, to contain the nested list) instead of in one shot at close time.
    list_stack: list[list] = []
    i, n = 0, len(lines)

    def flush_para() -> None:
        if para:
            out.append(f"<p>{_inline(_join_wrapped(para))}</p>")
            para.clear()

    def close_li(level: list) -> None:
        if level[2]:
            out.append("</li>")
        elif level[1] is not None:
            out.append("<li>" + _inline(level[1]) + "</li>")
        level[1] = None
        level[2] = False

    def close_lists(min_indent: int = -1) -> None:
        while list_stack and list_stack[-1][0] >= min_indent:
            close_li(list_stack[-1])
            out.append("</ul>")
            list_stack.pop()

    while i < n:
        line = lines[i]

        if line.strip() == "```":
            flush_para()
            close_lists()
            code: list[str] = []
            i += 1
            while i < n and lines[i].strip() != "```":
                code.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        if line.startswith("> "):
            flush_para()
            close_lists()
            quote = [line[2:]]
            i += 1
            while i < n and lines[i].startswith("> "):
                quote.append(lines[i][2:])
                i += 1
            out.append(f"<blockquote><p>{_inline(_join_wrapped(quote))}</p></blockquote>")
            continue

        m3 = _H3_RE.match(line)
        if m3:
            flush_para()
            close_lists()
            out.append(f"<h3>{_inline(m3.group(1))}</h3>")
            i += 1
            continue

        mb = _BULLET_RE.match(line)
        if mb:
            flush_para()
            indent = len(mb.group(1))
            close_lists(indent + 1)
            if not list_stack or list_stack[-1][0] < indent:
                # Starting a nested list: the parent <li>'s text (if any) must be
                # opened in `out` now, so the nested <ul> lands inside it.
                if list_stack and list_stack[-1][1] is not None and not list_stack[-1][2]:
                    out.append("<li>" + _inline(list_stack[-1][1]))
                    list_stack[-1][2] = True
                list_stack.append([indent, None, False])
                out.append("<ul>")
            close_li(list_stack[-1])
            list_stack[-1][1] = mb.group(2)
            i += 1
            continue

        if not line.strip():
            flush_para()
            close_lists()
            i += 1
            continue

        # Continuation of the current list item (wrapped bullet text) or a plain
        # paragraph line, depending on whether a list is currently open.
        if list_stack and list_stack[-1][1] is not None:
            list_stack[-1][1] = _join_wrapped([list_stack[-1][1], line.strip()])
        else:
            para.append(line.strip())
        i += 1

    flush_para()
    close_lists()
    return "\n".join(out)


def render_entries(changelog_md: str) -> str:
    lines = changelog_md.splitlines()
    i, n = 0, len(lines)
    while i < n and not _VERSION_RE.match(lines[i]):
        i += 1

    sections: list[str] = []
    while i < n:
        m = _VERSION_RE.match(lines[i])
        assert m, f"expected a '## [version]' header at line {i + 1}"
        version, title = m.group(1), m.group(2)
        i += 1
        body_start = i
        while i < n and not _VERSION_RE.match(lines[i]):
            i += 1
        body_html = _render_block(lines[body_start:i])
        heading = f"{version} — {title}" if title else version
        sections.append(
            f'<section class="changelog-entry" id="{_slug(version)}">\n'
            f"<h2>{_inline(heading)}</h2>\n"
            f"{body_html}\n"
            "</section>"
        )
    return "\n\n".join(sections)


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Changelog — Distil</title>
<link rel="stylesheet" href="site.css"/>
<meta property="og:type" content="website"/>
<meta property="og:site_name" content="Distil"/>
<meta property="og:title" content="Changelog — Distil"/>
<meta property="og:description" content="Every notable Distil change, in order, generated from CHANGELOG.md."/>
<meta property="og:url" content="https://dshakes.github.io/distil/changelog.html"/>
<meta property="og:image" content="https://dshakes.github.io/distil/og.png"/>
<meta property="og:image:width" content="1200"/>
<meta property="og:image:height" content="630"/>
<meta name="twitter:card" content="summary_large_image"/>
<meta name="twitter:title" content="Changelog — Distil"/>
<meta name="twitter:description" content="Every notable Distil change, in order, generated from CHANGELOG.md."/>
<meta name="twitter:image" content="https://dshakes.github.io/distil/og.png"/>
  <link rel="icon" type="image/svg+xml" href="assets/logo.svg">
<script>(function(){try{var t=localStorage.getItem("distil-theme");if(t!=="light"&&t!=="dark")t=window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.setAttribute("data-theme",t);}catch(e){}})();</script>
</head>
<body>
<a class="skip-link" href="#content">Skip to main content</a>

<header class="topbar">
  <button class="sidebar-toggle" onclick="toggleSidebar()" aria-label="Toggle navigation" aria-expanded="false" aria-controls="sidebar">☰</button>
  <a href="index.html" class="topbar-logo"><img src="assets/logo.svg" alt="" width="22" height="22" style="border-radius:6px;vertical-align:-6px;margin-right:8px"/>Dist<b>il</b></a>
  <span class="topbar-pill">compression with a quality contract</span>
%(topbar)s
</header>

<div class="shell">
%(sidebar)s

  <main class="content" id="content" tabindex="-1">

    <h1><span class="g">Changelog</span></h1>
    <p class="lead">Every notable change to Distil, newest first — generated from <a href="https://github.com/dshakes/distil/blob/main/CHANGELOG.md"><code>CHANGELOG.md</code></a>, so this page can never drift from the source of truth.</p>

%(entries)s

    <div class="site-footer">
      Distil · compression with a quality contract · Apache-2.0 · <a href="https://github.com/dshakes/distil">github.com/dshakes/distil</a>
    </div>
  </main>
</div>

<script src="site.js" defer></script>
</body>
</html>
"""


def build(changelog_path: Path = _CHANGELOG) -> str:
    md = changelog_path.read_text(encoding="utf-8")
    entries = render_entries(md)
    page = _PAGE_TEMPLATE % {
        "topbar": site_nav.render_topbar_links("changelog.html"),
        "sidebar": site_nav.render_sidebar("changelog.html"),
        "entries": entries,
    }
    return page


def main(argv: list[str]) -> int:
    changelog_path = Path(argv[1]) if len(argv) > 1 else _CHANGELOG
    out_path = Path(argv[2]) if len(argv) > 2 else _OUT
    out_path.write_text(build(changelog_path), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

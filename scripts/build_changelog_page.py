#!/usr/bin/env python3
"""Render CHANGELOG.md into docs/changelog.html — stdlib only.

CHANGELOG.md was invisible to anyone who didn't already know to look in the repo
root; the site had 35+ pages and no changelog. This renders it into the same
template shell as every other page (topbar/sidebar via scripts/site_nav.py, so
it can never drift from the canonical nav), one `<section>` per `## [version]`
entry with a stable anchor id.

The converter below is a small hand-rolled subset of Markdown, not a general
parser: it supports exactly the constructs CHANGELOG.md actually uses (`##`/`###`
headers, fenced code blocks, 4-space-indented code blocks, GFM pipe tables,
blockquotes, one level of nested bullet lists with wrapped continuation lines,
`` `code` ``, **bold**, *italic*/_italic_, and `[text](url)` links — the last
scheme-checked and attribute-escaped before it reaches `href`). Checked by
tests/test_changelog_page.py (regenerate-and-compare, same pattern as
build_search_index.py / test_search_index.py).

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
_DOCS = _ROOT / "docs"
_OUT = _DOCS / "changelog.html"
_GITHUB_BLOB = "https://github.com/dshakes/distil/blob/main/"
# Lookup keyed by lowercased path -> real on-disk name. Built from a directory
# listing (not `Path.is_file()`): this repo has both docs/CACHE.md and
# docs/cache.html (different case), and a case-insensitive dev filesystem (the
# macOS/Windows default) would report docs/CACHE.html as existing -- and emit
# that wrong-case href -- when only docs/cache.html actually exists. GitHub
# Pages serves from a case-sensitive Linux filesystem, so a wrong-case href
# 404s in production while looking fine locally.
_DOCS_HTML_BY_LOWER = (
    {str(p.relative_to(_DOCS)).lower(): str(p.relative_to(_DOCS)) for p in _DOCS.rglob("*.html")}
    if _DOCS.is_dir()
    else {}
)

_VERSION_RE = re.compile(r"^## \[([^\]]+)\](?: — (.*))?\s*$")
_H3_RE = re.compile(r"^### (.*)$")
_BULLET_RE = re.compile(r"^(\s*)- (.*)$")

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_URI_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_SAFE_URI_SCHEMES = {"http", "https", "mailto"}


def _rewrite_repo_link(href: str) -> str:
    """CHANGELOG.md's relative links are repo-root-relative, the way GitHub
    renders it (``docs/CACHE.md``, ``CHANGELOG.md``, ...) — but this page lives
    *in* ``docs/``, so left alone a link like ``docs/CACHE.md`` resolves to the
    nonexistent ``docs/docs/CACHE.md``.

    - ``docs/<x>.html`` -> ``<x>.html``: already a site page, just drop the
      now-redundant ``docs/`` prefix.
    - ``docs/<x>.md`` -> ``<x>.html`` if that site page exists, else the
      GitHub blob URL for the source markdown (most ``docs/*.md`` files, e.g.
      the ADRs, have no rendered HTML counterpart).
    - anything else repo-root-relative (``CHANGELOG.md``, ``distil/...``,
      ``tests/...``) -> the GitHub blob URL, since nothing on the site can
      serve it.

    Absolute URLs, ``mailto:``, and bare ``#anchor`` fragments need no
    rewriting and are returned unchanged.
    """
    if _URI_SCHEME_RE.match(href) or href.startswith("#"):
        return href
    if href.startswith("docs/"):
        rest = href[len("docs/") :]
        if rest.endswith(".html"):
            return rest
        if rest.endswith(".md"):
            real = _DOCS_HTML_BY_LOWER.get(f"{rest[:-3]}.html".lower())
            if real is not None:
                return real
    return _GITHUB_BLOB + href


def _safe_href(url: str) -> str | None:
    """Reject dangerous URI schemes (``javascript:``, ``data:``, ...).

    A URL with no scheme at all — every relative path, ``#anchor``, ``./x``,
    ``/x``, and the rewritten (see `_rewrite_repo_link`) site-relative links —
    is left alone rather than rejected.
    """
    m = _URI_SCHEME_RE.match(url)
    if m is None:
        return url
    return url if m.group(1).lower() in _SAFE_URI_SCHEMES else None


def _intraword_underscore(text: str, pos: int) -> bool:
    """CommonMark's rule: `_` inside a word is never an emphasis delimiter.

    Without this, `distil/proxy.py _post_upstream` (a real, unbacktick-quoted
    CHANGELOG.md line) or `edit_file / run_tests` would have the first `_` of
    one identifier greedily paired with a later `_` of some unrelated
    identifier, wrapping the text between them in `<em>` -- snake_case is far
    more common in this file than genuine `_italic_` (which nothing here uses).
    """
    before = text[pos - 1] if pos > 0 else ""
    after = text[pos + 1] if pos + 1 < len(text) else ""
    return before.isalnum() and after.isalnum()


def _inline(text: str) -> str:
    """Render inline Markdown (code/bold/italic/links) into escaped HTML.

    A small recursive-descent scanner, not global regex substitution over the
    whole string: code spans are matched first and their contents are never
    re-parsed (a literal `*` inside `` `usage.*` `` is never read as emphasis),
    and bold/italic recurse into their own contents so `**leaked *and* hung**`
    nests correctly instead of the outer `**` failing to match because its
    content contains a `*`. Link targets are HTML-attribute-escaped (quote=True)
    and scheme-checked before being written into `href` -- markdown link syntax
    is the one place raw source text ends up inside an attribute rather than
    text content, so it is the injection-relevant path.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]

        if ch == "`":
            j = text.find("`", i + 1)
            if j == -1:
                out.append("`")
                i += 1
                continue
            out.append(f"<code>{html.escape(text[i + 1 : j], quote=False)}</code>")
            i = j + 1
            continue

        if text[i : i + 2] == "**":
            j = text.find("**", i + 2)
            if j == -1:
                out.append("**")
                i += 2
                continue
            out.append(f"<strong>{_inline(text[i + 2 : j])}</strong>")
            i = j + 2
            continue

        if ch == "*":
            j = i + 1
            while j < n and text[j] != "*":
                j += 1
            if j < n and j > i + 1:
                out.append(f"<em>{_inline(text[i + 1 : j])}</em>")
                i = j + 1
                continue
            out.append("*")
            i += 1
            continue

        if ch == "_":
            j = i + 1
            while j < n and (text[j] != "_" or _intraword_underscore(text, j)):
                j += 1
            if j < n and j > i + 1 and not _intraword_underscore(text, i):
                out.append(f"<em>{_inline(text[i + 1 : j])}</em>")
                i = j + 1
                continue
            out.append("_")
            i += 1
            continue

        if ch == "[":
            m = _LINK_RE.match(text, i)
            if m:
                inner = _inline(m.group(1))
                href = _safe_href(_rewrite_repo_link(m.group(2).strip()))
                out.append(
                    f'<a href="{html.escape(href, quote=True)}">{inner}</a>'
                    if href is not None
                    else inner
                )
                i = m.end()
                continue
            out.append("[")
            i += 1
            continue

        j = i
        while j < n and text[j] not in "`*_[":
            j += 1
        out.append(html.escape(text[i:j], quote=False))
        i = j
    return "".join(out)


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


def _looks_like_table_row(line: str) -> bool:
    s = line.strip()
    return len(s) > 1 and s.startswith("|") and s.endswith("|")


def _looks_like_table_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and "-" in s and set(s) <= set("|:- ")


def _table_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip()[1:-1].split("|")]


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

        # Top-level-only constructs: both only ever occur in CHANGELOG.md right
        # after a blank line, outside any list -- gate on that so a 4-space
        # indented *continuation line* of an open nested bullet is never
        # misread as a code block.
        if (
            not list_stack
            and not para
            and _looks_like_table_row(line)
            and i + 1 < n
            and _looks_like_table_separator(lines[i + 1])
        ):
            header = _table_cells(line)
            i += 2
            body_rows: list[list[str]] = []
            while i < n and _looks_like_table_row(lines[i]):
                body_rows.append(_table_cells(lines[i]))
                i += 1
            thead = "<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>"
            tbody = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
                for row in body_rows
            )
            out.append(f"<table>\n<thead>{thead}</thead>\n<tbody>{tbody}</tbody>\n</table>")
            continue

        if not list_stack and not para and (line.startswith("    ") or line.startswith("\t")):
            code = []
            while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                code.append(lines[i][4:] if lines[i].startswith("    ") else lines[i][1:])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
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

    # CHANGELOG.md sometimes reuses a version number across two entries (e.g. an
    # rc build and its GA promotion both titled "1.13.0"), which would otherwise
    # collide on the same slug and produce a duplicate id. Second and later
    # occurrences get -2, -3, ... appended, same convention as HTML heading
    # anchors elsewhere.
    seen: dict[str, int] = {}
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
        base = _slug(version)
        seen[base] = seen.get(base, 0) + 1
        slug = base if seen[base] == 1 else f"{base}-{seen[base]}"
        # The id lives on the <h2>, not the <section>, because
        # scripts/build_search_index.py only reads anchors off h1/h2/h3.
        sections.append(
            f'<section class="changelog-entry">\n'
            f'<h2 id="{slug}">{_inline(heading)}</h2>\n'
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

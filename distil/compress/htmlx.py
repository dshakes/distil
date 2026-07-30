"""Reversible HTML content extraction — the web-fetch blind spot.

An agent with a fetch or browser tool receives raw HTML, and distil compressed **none
of it**: measured on a realistic 8.3k-token page, savings were 0.0%. The reason is
structural rather than a missing heuristic — minified HTML arrives as one enormous
line, so Tier-1's line-folding has nothing to fold and the JSON/record folds do not
recognise markup. A page is mostly `<script>`, `<style>`, nav, and footer chrome, so
that 0% was distil paying full price for tokens the model can never use.

This extracts the content and keeps the original behind an expand handle. Two
consequences worth stating plainly:

* **Recall-biased on purpose.** Only tags that cannot carry article content are
  dropped outright (`script`, `style`, `svg`, …) plus four unambiguous chrome
  landmarks (`nav`, `footer`, `aside`, `form`). `<header>` is deliberately KEPT —
  it usually holds the `<h1>` — because for a compressor a dropped fact is a wrong
  answer while a kept-but-unneeded one is merely fewer tokens saved.
* **Reversible, which lossy extractors are not.** The exact original bytes go into
  the restore table under the marker's handle, so anything this heuristic gets wrong
  is one ``distil_expand`` call away rather than gone. That is what lets the
  extraction be aggressive without being risky, and `distil retention` measures the
  difference (visible vs recoverable) instead of asserting it.

Output is line-oriented (block-level tags become newlines), which also re-arms the
downstream line-based machinery — keep-policy error pinning and the retention
probe's per-line error extraction — on content that previously had no lines at all.

Stdlib only: ``html.parser``. No lxml, no bs4, no readability port.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Tags whose contents are never article text. Dropped with their subtree.
_DROP_SUBTREE = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "object",
        "embed",
        "template",
        "map",
        "audio",
        "video",
        "select",
        "option",
        "datalist",
    }
)

# Structural chrome. Only landmarks whose semantic meaning is unambiguous — `header`
# is NOT here (it commonly wraps the title), and neither is `div`/`section`, whose
# meaning depends entirely on the site.
_DROP_CHROME = frozenset({"nav", "footer", "aside", "form"})

# Block-level tags become a newline, so the result has lines for the rest of the
# pipeline (and for a human) to work with.
_BLOCK = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "br",
        "hr",
        "li",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "tr",
        "td",
        "th",
        "table",
        "thead",
        "tbody",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "figure",
        "figcaption",
        "title",
    }
)

_VOID = frozenset({"br", "hr", "img", "input", "meta", "link", "source", "col", "area", "base"})

# Primary content landmarks. Chrome skipping keys off a matching close tag, and real
# HTML frequently never sends one — so an unclosed <nav>/<aside> would otherwise skip
# the entire remainder of the document, including the article. These end any active
# skip: whatever an unclosed sidebar is, it does not legitimately contain the page's
# <article>/<main>. (Audit finding: content before an unclosed chrome tag was emitted
# while everything after it was dropped.)
_CONTENT_LANDMARK = frozenset({"article", "main"})
# Backstop for pages with no landmark to recover on (chrome built from <div>s). Real
# nav/footer text is a few hundred characters; this much skipped data means the close
# tag is missing, so abandon the skip rather than eat the rest of the page.
_SKIP_DATA_BUDGET = 8_000

_TAG_RE = re.compile(r"<\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")
_CLOSE_RE = re.compile(r"</\s*([a-zA-Z][a-zA-Z0-9]*)\s*>")
_DOCTYPE_RE = re.compile(r"<!doctype\s+html|<html\b|<body\b|<head\b", re.IGNORECASE)
_WS_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_RUN = re.compile(r"\n{2,}")

# Below this, extraction is not worth a marker + a restore entry: a fragment that is
# already mostly text should pass through untouched rather than churn the cache.
_MIN_CHARS = 400
_MIN_TAGS = 8
_MIN_GAIN = 0.15  # require a real win, not a rounding one
# Fraction of closeable open tags that must actually be closed for a doctype-less
# fragment to count as markup. Well below 1.0: real-world HTML omits </li>, </p>, </td>.
_MIN_CLOSE_RATIO = 0.5


def looks_like_html(text: str) -> bool:
    """True if `text` is plausibly an HTML document or a substantial fragment.

    Deliberately conservative: a stray ``<div>`` in a log line or a code sample that
    mentions markup must not be routed through an HTML extractor.
    """
    if len(text) < _MIN_CHARS:
        return False
    if _DOCTYPE_RE.search(text):
        return True
    # No document landmarks. Tag density alone is NOT enough: a log line reading
    # "parse failed near <div>" repeated 40 times clears any density bar. Real markup
    # closes its elements, so require the open tags to be substantially balanced by
    # closing tags — which prose mentioning a tag name never is.
    opens = _TAG_RE.findall(text)
    if len(opens) < _MIN_TAGS:
        return False
    closes = _CLOSE_RE.findall(text)
    voidish = sum(1 for t in opens if t.lower() in _VOID)
    closeable = len(opens) - voidish
    return bool(closes) and closeable > 0 and len(closes) >= closeable * _MIN_CLOSE_RATIO


class _Extractor(HTMLParser):
    """Collects visible text, skipping non-content subtrees."""

    def __init__(self) -> None:
        # convert_charrefs resolves &amp;/&#39; for us, so facts compare equal to the
        # decoded form a model would read.
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._skipping: str | None = None
        self._skipped_chars = 0
        self.dropped_tags = 0

    def _end_skip(self) -> None:
        self._skipping = None
        self._skip_depth = 0
        self._skipped_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.dropped_tags += 1
        if self._skipping is not None:
            # A content landmark cannot be inside the sidebar we think we are in: the
            # chrome tag was never closed. Resume collecting rather than dropping the
            # rest of the document.
            if tag in _CONTENT_LANDMARK and self._skipping not in _DROP_SUBTREE:
                self._end_skip()
            elif tag == self._skipping and tag not in _VOID:
                self._skip_depth += 1
                return
            else:
                return
        if tag in _DROP_SUBTREE or tag in _DROP_CHROME:
            if tag not in _VOID:
                self._skipping = tag
                self._skip_depth = 1
            return
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag == "img":
            # Alt text is real content and often the only description of a figure.
            alt = next((v for k, v in attrs if k.lower() == "alt" and v), None)
            if alt:
                self.parts.append(f"[image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skipping is not None:
            if tag == self._skipping:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._end_skip()
            return
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skipping is not None:
            # Budget backstop for chrome with no close tag and no landmark after it
            # (a <div>-built sidebar): stop skipping instead of eating the article.
            # Never applies to script/style, whose payload is legitimately huge.
            if self._skipping in _DROP_SUBTREE:
                return
            self._skipped_chars += len(data)
            if self._skipped_chars > _SKIP_DATA_BUDGET:
                self._end_skip()
            return
        if data.strip():
            self.parts.append(data)

    def error(self, message: str) -> None:  # pragma: no cover - py<3.10 ABC shim
        return None


def extract(text: str) -> str | None:
    """Extract readable content from HTML, or None if that is not worthwhile.

    Returns None when the input is not HTML, when parsing fails, or when the result
    is not meaningfully smaller — so the caller's reject-if-bigger invariant is never
    the thing that has to catch it.
    """
    if not looks_like_html(text):
        return None
    parser = _Extractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception:  # noqa: BLE001 — malformed markup must never break a request
        return None

    body = "".join(parser.parts)
    body = _WS_RUN.sub(" ", body)
    body = "\n".join(line.strip() for line in body.split("\n"))
    body = _BLANK_RUN.sub("\n", body).strip()

    if not body:
        return None
    if len(body) > len(text) * (1.0 - _MIN_GAIN):
        return None  # not a real win: leave the block alone
    return body

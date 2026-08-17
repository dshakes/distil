"""The docs sidebar is a contract, not 30 independent copies.

Every page renders its own copy of the nav. A generated page whose sidebar drifts
— a mislabelled link, a missing entry, a dead href — is invisible to every other
test in this suite and to anyone not looking at that exact page. It happened:
templating the per-framework pages renamed the "Metrics & Observability" sidebar
link to the framework's own name on all eight, while still pointing at
metrics.html.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = sorted(p for p in DOCS.glob("*.html") if p.name != "index.html")
LINK = re.compile(r'<a href="([a-z0-9\-]+\.html)"[^>]*>(.*?)</a>', re.S)


def _nav_links(html: str) -> dict[str, str]:
    aside = html.split("<aside", 1)[-1].split("</aside>", 1)[0]
    out = {}
    for href, label in LINK.findall(aside):
        # Strip nav badges ("Start here", "New", "Live"): the same link legitimately
        # carries a badge on some pages and not on the course sub-pages, and
        # flagging that would make this test noise rather than a signal.
        text = re.sub(r"<span class=\"nav-badge\">.*?</span>", "", label, flags=re.S)
        text = re.sub(r"<[^>]+>", "", text).strip()
        text = re.sub(r"\s+", " ", text)
        out.setdefault(href, text)
    return out


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_sidebar_link_resolves(page):
    for href in _nav_links(page.read_text(encoding="utf-8")):
        assert (DOCS / href).is_file(), f"{page.name}: dead sidebar link -> {href}"


def test_sidebar_labels_agree_across_every_page():
    """The same href must carry the same label everywhere. Templating a new page
    off an existing one is exactly how this drifts."""
    seen: dict[str, tuple[str, str]] = {}
    problems = []
    for page in PAGES:
        for href, label in _nav_links(page.read_text(encoding="utf-8")).items():
            # The landing-page back-link is deliberately worded per context.
            if href == "index.html":
                continue
            if href in seen and seen[href][0] != label:
                problems.append(
                    f"{href}: {seen[href][0]!r} in {seen[href][1]} vs {label!r} in {page.name}"
                )
            seen.setdefault(href, (label, page.name))
    assert not problems, "sidebar labels disagree:\n  " + "\n  ".join(problems[:8])


def test_no_orphan_pages():
    """A page nothing links to is a page nobody finds.

    Reachability is checked across ALL links, not just the sidebar: the Token
    Economics course modules and the reproduce-the-numbers page are deliberately
    reached from their parent page's body rather than from the nav, and demanding
    a sidebar entry for each would be inventing a rule the site does not follow.
    """
    linked = set()
    for page in PAGES + [DOCS / "index.html"]:
        html = page.read_text(encoding="utf-8")
        linked |= set(re.findall(r'href="([a-z0-9\-]+\.html)"', html))
    orphans = {p.name for p in PAGES} - linked
    assert not orphans, f"no page links to: {sorted(orphans)}"


def test_no_nav_list_item_holds_two_links():
    """One nav <li>, one nav link.

    Nav links get inserted into 35+ pages by script, and those pages do not share a
    single nav markup: most use bare <a> in a <div>, but nine use <li> lists. A
    regex that appends a link without closing the <li> puts two anchors on one
    bullet — they render on the same line, under one marker. Review caught exactly
    that on #143, after the identical defect had already shipped in the
    subscription nav, so this guards the class rather than the instance.

    Scoped to <nav>/<aside> so it cannot fire on prose lists (which legitimately
    carry several inline links) or on a parent item with a nested sub-menu.
    """
    import re
    from pathlib import Path

    docs = Path(__file__).resolve().parent.parent / "docs"
    offenders = []
    for page in sorted(docs.glob("*.html")):
        text = page.read_text(encoding="utf-8")
        for region in re.finditer(r"<(nav|aside)\b.*?</\1>", text, re.S):
            # stop each item at a nested <ul>: a parent link plus a sub-menu is fine
            for item in re.finditer(r"<li>(?:(?!</li>).)*?(?=<ul|</li>)", region.group(0), re.S):
                if item.group(0).count("<a href=") > 1:
                    offenders.append(f"{page.name}: {' '.join(item.group(0).split())[:110]}")
    assert not offenders, "nav <li> holding more than one link:\n" + "\n".join(offenders[:5])

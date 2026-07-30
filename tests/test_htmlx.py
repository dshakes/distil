"""Reversible HTML extraction.

The transform is lossy-but-recoverable, which makes two properties load-bearing: it
must be *recall-biased* (never drop something that could be article content) and it
must be *reversible* (the exact original recoverable from the handle). It must also
decline politely — on non-HTML, on fragments too small to be worth a restore entry,
and on anything it cannot parse.
"""

from __future__ import annotations

import json

from distil.adapters.anthropic import compress_messages
from distil.compress.htmlx import extract, looks_like_html
from distil.compress.tier1 import Tier1Reversible
from distil.tokenizer import DEFAULT as TOK
from distil.trajectory import Block, Kind, Stability


def _page(body: str, *, head: str = "", chrome: bool = True) -> str:
    noise = (
        "<script>var x=1;" + ("/*pad*/" * 200) + "</script>"
        "<style>" + (".c{color:red}" * 200) + "</style>"
    )
    nav = "<nav>" + ('<a href="/x">NavLink</a>' * 100) + "</nav>" if chrome else ""
    foot = "<footer>" + ("<span>footjunk</span>" * 200) + "</footer>" if chrome else ""
    return (
        f"<!DOCTYPE html><html><head>{head}{noise}</head>"
        f"<body>{nav}<article>{body}</article>{foot}</body></html>"
    )


# --------------------------------------------------------------------------- detection


def test_detects_documents() -> None:
    assert looks_like_html(_page("<p>hello world and some more text to clear the floor</p>"))


def test_rejects_short_input() -> None:
    assert looks_like_html("<html><body><p>hi</p></body></html>") is False


def test_rejects_prose_and_logs() -> None:
    assert not looks_like_html("Just a long paragraph of prose. " * 40)
    log = "\n".join(f"2026-07-29 INFO worker-{i} latency_ms={i} ok" for i in range(60))
    assert not looks_like_html(log)


def test_rejects_a_stray_tag_in_a_log() -> None:
    """A log line mentioning markup must not be routed through an HTML extractor."""
    log = "\n".join(f"2026-07-29 ERROR parse failed near <div> at line {i}" for i in range(40))
    assert not looks_like_html(log)


def test_tag_density_accepts_a_real_fragment() -> None:
    fragment = "<ul>" + "".join(f"<li>Item number {i} with text</li>" for i in range(30)) + "</ul>"
    assert looks_like_html(fragment)


# --------------------------------------------------------------------------- extraction


def test_strips_noise_keeps_content() -> None:
    html = _page("<h1>Real Title</h1><p>Body sentence with latency_ms=42 in it.</p>" * 20)
    out = extract(html)
    assert out is not None
    assert "Real Title" in out and "latency_ms=42" in out
    assert "/*pad*/" not in out and "color:red" not in out
    assert "NavLink" not in out and "footjunk" not in out


def test_header_is_kept_because_it_holds_titles() -> None:
    """Recall bias: `header` is ambiguous chrome and commonly wraps the h1, so unlike
    nav/footer/aside it must NOT be dropped."""
    html = _page("<header><h1>The Headline</h1></header><p>Body text here.</p>" * 20)
    out = extract(html)
    assert out is not None and "The Headline" in out


def test_image_alt_text_survives() -> None:
    """Alt text is often the only description of a figure — real content."""
    html = _page('<p>Intro text.</p><img src="/a.png" alt="Chart of latency by region">' * 20)
    out = extract(html)
    assert out is not None and "Chart of latency by region" in out


def test_entities_are_decoded() -> None:
    html = _page("<p>Fish &amp; Chips cost &lt;5 &#39;quid&#39; today.</p>" * 20)
    out = extract(html)
    assert out is not None
    assert "Fish & Chips" in out and "<5" in out and "'quid'" in out


def test_output_is_line_oriented() -> None:
    """Block tags become newlines, which re-arms the line-based machinery downstream."""
    html = _page("".join(f"<p>Paragraph number {i} of the body.</p>" for i in range(40)))
    out = extract(html)
    assert out is not None and out.count("\n") > 20


def test_nested_dropped_subtree_does_not_swallow_the_rest() -> None:
    """A nested same-named tag inside a dropped subtree must not unbalance the skip."""
    html = _page(
        "<p>Before the noise.</p><nav><div><nav>inner</nav></div></nav>"
        "<p>After the noise, still content with fact=7.</p>" * 20
    )
    out = extract(html)
    assert out is not None
    assert "After the noise" in out and "fact=7" in out
    assert "inner" not in out


def test_unclosed_chrome_does_not_swallow_the_article() -> None:
    """Audit finding. Chrome skipping keyed off a matching close tag, and real HTML
    often never sends one — so content BEFORE an unclosed <aside> was emitted while the
    actual article after it was silently dropped. A content landmark ends the skip."""
    html = (
        "<!DOCTYPE html><html><body>"
        "<article><h1>Kept Title</h1><p>Intro paragraph.</p></article>"
        "<aside>sidebar"  # never closed
        "<article><h2>Second Section</h2>"
        + "<p>Body fact=42 that must survive.</p>" * 40
        + "</article></body></html>"
    )
    out = extract(html)
    assert out is not None
    assert "Kept Title" in out
    assert "Second Section" in out  # was lost before the fix
    assert "fact=42" in out


def test_unclosed_chrome_without_a_landmark_hits_the_budget() -> None:
    """Backstop for <div>-built chrome, where no <article>/<main> follows to recover on:
    the skip is abandoned once it has eaten more than a plausible nav's worth of text."""
    html = (
        "<!DOCTYPE html><html><body><nav>"
        + ("<span>navjunk</span>" * 60)
        + ("<div><p>Real body fact=7 in a div-built page.</p></div>" * 400)
        + "</body></html>"
    )
    out = extract(html)
    assert out is not None and "fact=7" in out


def test_well_formed_chrome_is_still_dropped() -> None:
    """The fixes must not cost the savings: closed nav/footer still go."""
    html = _page("<h1>T</h1>" + "<p>Article body sentence.</p>" * 40)
    out = extract(html)
    assert out is not None
    assert "NavLink" not in out and "footjunk" not in out
    assert "Article body" in out


def test_script_payload_is_never_budget_capped() -> None:
    """A huge <script> is legitimately huge — the budget backstop must not resume
    collecting inside it and leak minified JS into the output."""
    html = (
        "<!DOCTYPE html><html><head><script>"
        + ("var averylongvariablename=1;" * 4000)
        + "</script></head><body><article>"
        + "<p>Body text here.</p>" * 30
        + "</article></body></html>"
    )
    out = extract(html)
    assert out is not None
    assert "averylongvariablename" not in out
    assert "Body text" in out


def test_declines_when_not_worthwhile() -> None:
    """Markup that is already almost all text is not worth a marker + restore entry."""
    html = "<div><p>" + ("Almost entirely readable prose content here. " * 40) + "</p></div>"
    assert extract(html) is None


def test_declines_on_non_html() -> None:
    assert extract("Just prose, no markup at all. " * 40) is None


def test_empty_body_declines() -> None:
    html = "<!DOCTYPE html><html><head><script>" + ("x=1;" * 400) + "</script></head></html>"
    assert extract(html) is None


def test_malformed_markup_is_survived() -> None:
    """Never raise: unclosed tags, stray brackets, truncated documents."""
    for junk in (
        "<!DOCTYPE html><html><body><p>unclosed" + ("<div>" * 500),
        "<html><body>" + ("<<<>>>" * 300) + "<p>text</p>",
        "<!DOCTYPE html><html><body><p>" + ("a" * 2000),
    ):
        extract(junk)  # must not raise


# --------------------------------------------------------------------------- integration


def test_tier1_digests_html_reversibly() -> None:
    html = _page("<h1>T</h1><p>Body with fact=99 inside.</p>" * 30)
    block = Block(id="b", kind=Kind.TOOL_OUTPUT, text=html, stability=Stability.VOLATILE)
    result = Tier1Reversible().compress([block])
    out = result.blocks[0].text
    assert "handle=" in out
    assert len(out) < len(html)
    assert list(result.restore.values()) == [html]  # byte-exact original


def test_live_path_saves_and_stays_recoverable() -> None:
    """The measured gap this closes: 0% savings on HTML through the real adapter."""
    html = _page("<h1>Real Title</h1><p>Body sentence with latency_ms=42.</p>" * 40)

    def tr(text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": text}],
        }

    messages = [tr(html)] + [tr(f"later tool result {i}") for i in range(4)]
    compressed, store = compress_messages(messages)

    before = TOK.count(json.dumps(messages[0]))
    after = TOK.count(json.dumps(compressed[0]))
    assert after < before * 0.4  # measured ~91% on this shape

    handles = list(store.handles)
    assert len(handles) == 1
    assert store.expand(handles[0]) == html  # exact original, one expand away

    seen = json.dumps(compressed[0])
    assert "Real Title" in seen and "latency_ms=42" in seen
    assert "/*pad*/" not in seen and "NavLink" not in seen


def test_recent_tool_result_stays_byte_exact() -> None:
    """The recency rule outranks the transform: the agent's latest output is untouched."""
    html = _page("<h1>T</h1><p>Body fact=1.</p>" * 40)

    def tr(text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": text}],
        }

    compressed, store = compress_messages([tr(html)])
    assert json.dumps(compressed[0]) == json.dumps(tr(html))
    assert not list(store.handles)


def test_verbatim_mode_never_extracts() -> None:
    """Verbatim means no expand tool is available, so an unrecoverable elision is
    forbidden — even a big win."""
    html = _page("<h1>T</h1><p>Body fact=1.</p>" * 40)

    def tr(text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": text}],
        }

    messages = [tr(html)] + [tr(f"later {i}") for i in range(4)]
    compressed, store = compress_messages(messages, verbatim=True)
    assert "html chrome elided" not in json.dumps(compressed[0])
    assert not list(store.handles)


def test_retention_on_html_loses_nothing() -> None:
    """Grade the new transform with the retention probe: facts may move behind a
    handle, but none may be lost."""
    from distil.retention import measure_live

    html = _page(
        "<h1>Report</h1>"
        + "".join(
            f"<p>Row {i}: latency_ms={i * 3} path=/srv/data/file-{i}.csv</p>" for i in range(60)
        )
        + "<p>ERROR: upstream refused connection</p>"
    )

    def tr(text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": text}],
        }

    messages = [tr(html)] + [tr(f"later tool result {i}") for i in range(4)]
    compressed, store = compress_messages(messages)
    dims = measure_live(messages, compressed, store)
    assert sum(t.total for t in dims.values()) > 0
    assert sum(t.lost for t in dims.values()) == 0  # nothing unrecoverable

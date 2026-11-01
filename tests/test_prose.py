"""Sentence-level prose condensing (distil/compress/prose.py) and its adapter path."""

from __future__ import annotations

import pytest

from distil.compress import prose

_FILLER = (
    "This section explains the general idea behind the approach in plain words. "
    "It is written for readers who are new to the project and want an overview. "
    "Nothing here is required reading for day to day work on the codebase."
)


def _doc(paragraphs: int = 12) -> str:
    out = ["# Guide", ""]
    for i in range(paragraphs):
        out += [
            f"Paragraph {i} opens with its lead sentence about the topic at hand. {_FILLER} "
            f"The timeout is set in config/server/limits.toml as read_timeout_ms = {400 + i}.",
            "",
        ]
    out += ["```python", "def keep_me():", "    return 1", "```", "", "| a | b |", "|---|---|"]
    return "\n".join(out)


def test_detects_prose_and_rejects_code_and_logs():
    assert prose.is_prose(_doc())
    code = "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(20))
    logs = "\n".join(f"2026-10-31T12:00:{i:02d} INFO worker-3 batch={i} ok" for i in range(40))
    assert not prose.is_prose(code)
    assert not prose.is_prose(logs)


def test_keeps_structure_leads_and_specific_sentences():
    doc = _doc()
    out = prose.condense(doc)
    assert out is not None and len(out) < len(doc)
    assert "# Guide" in out
    assert "def keep_me():" in out and "| a | b |" in out
    for i in range(12):
        assert f"Paragraph {i} opens with its lead sentence" in out
        assert f"read_timeout_ms = {400 + i}" in out  # a named value survives
    assert "Nothing here is required reading" not in out
    assert prose.ELIDED in out


def test_a_sentence_wrapped_across_lines_is_kept_or_dropped_whole():
    para = [
        "The lead sentence of this paragraph stays in view for the reader. This plain",
        "remark wraps across a line break and carries nothing specific at all, and",
        "it ends here. The limit lives in /etc/app/conf.d/limits.conf and it",
        "wraps too, ending at value 4096.",
    ]
    doc = "\n".join(["# T", ""] + (para + [""]) * 10)
    out = prose.condense(doc, keep_share=0.6)
    assert out is not None
    assert "remark wraps" not in out and "it ends here." not in out  # dropped whole
    assert "/etc/app/conf.d/limits.conf and it" in out
    assert "wraps too, ending at value 4096." in out  # its wrapped tail kept with it


def test_intent_terms_keep_an_otherwise_plain_sentence():
    doc = _doc()
    out = prose.condense(doc, intent=frozenset({"overview"}))
    assert out is not None and "want an overview" in out


def test_is_deterministic_and_none_when_nothing_to_elide():
    doc = _doc()
    assert prose.condense(doc) == prose.condense(doc)
    assert prose.condense("# Title\n\nOne lead sentence only, nothing else to drop here.") is None


@pytest.mark.real_digest_floor
def test_adapter_condenses_large_prose_recoverably():
    from distil.adapters.anthropic import _DIGEST_MIN_TOKENS, compress_messages
    from distil.tokenizer import DEFAULT as tok

    doc = _doc(60)
    assert tok.count(doc) >= _DIGEST_MIN_TOKENS
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "WebFetch", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": doc}]},
    ]
    for i in range(3):  # push it out of the recency window
        msgs += [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"p{i}", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"p{i}", "content": "ok"}],
            },
        ]
    out, store = compress_messages(msgs, persist=False)
    seen = out[2]["content"][0]["content"]
    assert "prose condensed" in seen and "handle=" in seen
    assert store.expand(next(iter(store.handles))) == doc

"""``<distil:keep>`` spans are never compressed, and the tags themselves survive."""

from __future__ import annotations

import json

import pytest

from distil.adapters.anthropic import compress_messages
from distil.compress import keeptags
from distil.compress.tier1 import Tier1Reversible
from distil.trajectory import Block, Kind

LOG = "\n".join(f"2026-09-25 INFO worker-{i % 5} step {i} ok in {i % 17}ms" for i in range(300))
KEPT = (
    keeptags.OPEN
    + "\n"
    + "\n".join(f"secret-config line {i}: value={i * 31}" for i in range(80))
    + "\n"
    + keeptags.CLOSE
)


@pytest.fixture(autouse=True)
def _metered(monkeypatch):
    """These tests exercise the digest, which a subscription login keeps off by default
    (tests/test_hook_tiers.py covers that branch); pin the billing so the developer's
    own environment cannot flip them."""
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")


def _upper(s: str) -> str:
    return s.upper()


@pytest.mark.parametrize(
    "text,want",
    [
        ("plain", "PLAIN"),
        ("a<distil:keep>b</distil:keep>c", "A<distil:keep>b</distil:keep>C"),
        ("<distil:keep>x</distil:keep>", "<distil:keep>x</distil:keep>"),
        (
            "a<distil:keep>b</distil:keep><distil:keep>c</distil:keep>d",
            "A<distil:keep>b</distil:keep><distil:keep>c</distil:keep>D",
        ),
        ("a<distil:keep>unclosed tail", "A<distil:keep>unclosed tail"),
        ("a</distil:keep>b", "A</DISTIL:KEEP>B"),  # a stray close tag protects nothing
        # Non-nesting: the first close ends the span; what follows is outside.
        (
            "<distil:keep>a<distil:keep>b</distil:keep>c</distil:keep>",
            "<distil:keep>a<distil:keep>b</distil:keep>C</DISTIL:KEEP>",
        ),
    ],
    ids=["none", "middle", "whole", "two", "unclosed", "stray-close", "nested"],
)
def test_apply(text, want):
    assert keeptags.apply(text, _upper) == want


def _tool_result_messages(text: str) -> list[dict]:
    """A tool_result old enough to be outside the recency carve-out."""
    msgs: list[dict] = [{"role": "user", "content": "run it"}]
    msgs.append(
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t0", "name": "bash", "input": {"command": "make"}}
            ],
        }
    )
    msgs.append(
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": text}]}
    )
    for n in range(1, 6):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{n}", "name": "bash", "input": {"command": "ls"}}
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": "ok"}],
            }
        )
    return msgs


def test_proxy_adapter_keeps_the_span_and_compresses_around_it():
    text = LOG + "\n" + KEPT + "\n" + LOG
    out, store = compress_messages(_tool_result_messages(text))
    got = out[2]["content"][0]["content"]
    assert KEPT in got, "the span, tags included, is byte-exact"
    assert len(got) < len(text) / 2, "the rest still compresses"
    # Every stretch outside the span that was digested is recoverable.
    for h in store.handles:
        assert store.expand(h) in text


def test_proxy_adapter_same_text_without_tag_digests_the_span():
    bare = KEPT.replace(keeptags.OPEN, "").replace(keeptags.CLOSE, "")
    out, _ = compress_messages(_tool_result_messages(LOG + bare + LOG))
    assert bare not in out[2]["content"][0]["content"]


def test_user_text_span_survives_tier0():
    spaced = json.dumps({"k": list(range(50))}, indent=2)
    text = f"please keep {keeptags.OPEN}{spaced}{keeptags.CLOSE} and this {spaced}"
    out, _ = compress_messages([{"role": "user", "content": text}])
    got = out[0]["content"]
    got = got if isinstance(got, str) else got[0]["text"]
    assert f"{keeptags.OPEN}{spaced}{keeptags.CLOSE}" in got


def test_hook_keeps_the_span():
    from distil.hook import compress_text

    text = LOG + "\n" + KEPT + "\n" + LOG
    out = compress_text(text)
    assert out is not None and KEPT in out and len(out) < len(text) / 2


def test_offline_tier1_leaves_a_tagged_block_whole():
    blk = Block(id="b", kind=Kind.TOOL_OUTPUT, text=LOG + KEPT)
    res = Tier1Reversible().compress([blk])
    assert res.blocks[0].text == LOG + KEPT and not res.restore

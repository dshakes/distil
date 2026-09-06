"""Cache-delta context coding — cross-turn dedup, cross-version delta, monotonicity.

The invention vs exact-only dedup: a file RE-READ after an edit is a near-duplicate,
not identical, so it is delta-encoded (reference + diff) rather than re-sent whole.
"""

from __future__ import annotations

from distil.cachedelta import (
    CacheStats,
    DeltaSession,
    delta_encode,
    get_session,
    reset_sessions,
    session_key,
)
from distil.pricing import get as pricing_get

V1 = "\n".join(f"line {i}: value {i} status=ok" for i in range(60))
V2 = V1.replace("line 30: value 30 status=ok", "line 30: value THIRTY status=EDITED")
assert V1 != V2 and len(V1) > 400


def _tr(text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": text}],
    }


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _block_text(msg: dict) -> str:
    return msg["content"][0]["content"]


# --- exact cross-turn dedup ------------------------------------------------ #


def test_exact_resend_becomes_reference_and_is_recoverable():
    s = DeltaSession()
    delta_encode([_u("q"), _tr(V1)], session=s)  # turn 1 — remembers V1
    out, store, stats = delta_encode([_u("q"), _tr(V1), _u("again"), _tr(V1)], session=s)
    assert stats.prefix_msgs == 2  # first two messages are the cache-stable prefix
    assert stats.exact_refs == 1
    seen = _block_text(out[3])
    assert "«distil-ref handle=" in seen and len(seen) < len(V1)
    # Reversible: the original is recoverable byte-exact via expand.
    assert any(store.expand(h) == V1 for h in store.handles)


# --- cross-version delta (the invention) ----------------------------------- #


def test_reread_after_edit_is_delta_encoded():
    s = DeltaSession()
    delta_encode([_u("q"), _tr(V1)], session=s)  # turn 1 — remembers V1
    out, store, stats = delta_encode([_u("q"), _tr(V1), _u("edit"), _tr(V2)], session=s)
    assert stats.delta_refs == 1  # near-duplicate, NOT exact -> delta, not full re-send
    seen = _block_text(out[3])
    assert "«distil-delta base=" in seen
    assert "THIRTY" in seen  # the diff carries exactly what changed
    assert len(seen) < len(V2)  # smaller than re-sending the whole file
    assert stats.tokens_saved > 0
    # The full current version is recoverable byte-exact.
    assert any(store.expand(h) == V2 for h in store.handles)


def test_exact_only_dedup_would_miss_the_reread():
    # Sanity: V2 is genuinely not identical to V1, so exact-only dedup (the state of
    # the art elsewhere) cannot dedup it — only cross-version delta can.
    s = DeltaSession()
    delta_encode([_u("q"), _tr(V1)], session=s)
    _out, _store, stats = delta_encode([_u("q"), _tr(V1), _u("x"), _tr(V2)], session=s)
    assert stats.exact_refs == 0 and stats.delta_refs == 1


# --- cache-monotonicity ---------------------------------------------------- #


def test_stable_prefix_is_never_mutated():
    s = DeltaSession()
    t1 = [_u("q"), _tr(V1)]
    delta_encode(t1, session=s)
    t2 = [_u("q"), _tr(V1), _u("more"), _tr(V2)]
    out, _store, stats = delta_encode(t2, session=s)
    # Prefix messages are returned as the SAME objects — byte-identical, cache-safe.
    for i in range(stats.prefix_msgs):
        assert out[i] is t2[i]


# --- first turn / small blocks / robustness -------------------------------- #


def test_first_turn_changes_nothing():
    s = DeltaSession()
    out, _store, stats = delta_encode([_u("q"), _tr(V1)], session=s)
    assert stats.exact_refs == 0 and stats.delta_refs == 0
    assert _block_text(out[1]) == V1


def test_small_blocks_untouched():
    s = DeltaSession()
    small = "short tool output"
    delta_encode([_u("q"), _tr(small)], session=s)
    out, _store, stats = delta_encode([_u("q"), _tr(small), _u("z"), _tr(small)], session=s)
    assert stats.exact_refs == 0  # below the size threshold
    assert _block_text(out[3]) == small


def test_malformed_messages_do_not_crash():
    s = DeltaSession()
    out, _store, _stats = delta_encode([None, 1, "x", {"role": "user"}], session=s)
    assert isinstance(out, list) and len(out) == 4


# --- session registry + economics ------------------------------------------ #


def test_session_key_stable_and_distinct():
    a = [_u("project A bug")]
    b = [_u("project B bug")]
    assert session_key(a) == session_key(a)
    assert session_key(a) != session_key(b)


def test_get_session_is_per_key():
    reset_sessions()
    s1 = get_session("k1")
    assert get_session("k1") is s1
    assert get_session("k2") is not s1


def test_dollars_saved_uses_input_rate():
    p = pricing_get("claude-opus-4-8")
    stats = CacheStats(tokens_saved=1000)
    assert stats.dollars_saved(p) == 1000 * p.input


def test_cache_monotonic_emitted_bytes_are_stable():
    """A re-read encoded as a delta in one turn must encode to the SAME bytes when
    it later sits in the prefix — else the prompt cache busts (the bug the coding
    benchmark caught)."""
    s = DeltaSession()
    turn_n = [_u("q"), _tr(V1), _u("edit"), _tr(V2)]  # V2 (delta) at index 3
    out_n, _s1, _st1 = delta_encode(turn_n, session=s)
    turn_n1 = turn_n + [_u("more"), _tr("small")]  # index 3 is now inside the prefix
    out_n1, _s2, _st2 = delta_encode(turn_n1, session=s)
    assert _block_text(out_n[3]) == _block_text(out_n1[3])  # byte-identical emission


def test_delta_encode_is_pure_without_session():
    # No session needed for correctness — the cumulative conversation is the memory.
    msgs = [_u("q"), _tr(V1), _u("e"), _tr(V2)]
    a, _sa, sta = delta_encode(msgs)
    b, _sb, stb = delta_encode(msgs)
    assert sta.delta_refs == stb.delta_refs == 1
    assert _block_text(a[3]) == _block_text(b[3])  # deterministic


# --- the exact-quote guarantee outranks the delta ------------------------- #


def _tr_id(text: str, tid: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def test_delta_never_touches_an_exact_quote_block():
    """--session-delta ran BEFORE the compressor's exemption and replaced file reads
    with delta references, silently voiding the 1.49.0 byte-exact guarantee whenever the
    flag was on: a reference is not a copy, so the agent's next Edit could not match."""
    msgs = [_u("q"), _tr_id(V1, "read1"), _u("edit"), _tr_id(V2, "read2")]
    out, _store, _stats = delta_encode(msgs, keep_ids=frozenset({"read2"}))
    assert _block_text(out[3]) == V2, "an exempt read must reach the wire byte-exact"


def test_an_exempt_block_is_still_available_as_a_delta_base():
    """Skipping the transform must not remove the block from the session's memory —
    later re-reads still dedup against it, which is the whole point of the mechanism."""
    msgs = [_u("q"), _tr_id(V1, "read1"), _u("edit"), _tr_id(V2, "read2")]
    _out, _store, stats = delta_encode(msgs, keep_ids=frozenset({"read1"}))
    assert stats.delta_refs == 1, "V2 should still delta against the exempt V1"


def test_the_proxy_wires_the_exemption_into_the_delta_pass():
    """End to end on the real ids: a shell read stays byte-exact with delta on."""
    from distil.adapters.anthropic import exact_quote_tool_use_ids

    msgs = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r1", "name": "Bash", "input": {"command": "cat /a.py"}}
            ],
        },
        _tr_id(V1, "r1"),
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r2", "name": "Bash", "input": {"command": "cat /a.py"}}
            ],
        },
        _tr_id(V2, "r2"),
    ]
    keep = frozenset(exact_quote_tool_use_ids(msgs))
    assert keep == {"r2"}, "only the latest read of the path needs to stay verbatim"
    out, _store, _stats = delta_encode(msgs, keep_ids=keep)
    assert _block_text(out[5]) == V2


def test_azure_chat_completions_resolves_the_keep_list_with_the_openai_extractor():
    """The keep-list and the compressor must agree on the body's shape.

    The delta pass picked its id extractor from a literal `/v1/chat/completions` test
    while the compressor below it used the Azure-aware predicate. On an Azure path the
    Anthropic extractor ran against OpenAI-shaped messages, returned nothing, and delta
    rewrote the very reads the exemption was about to keep byte-exact — the
    guarantee-voiding bug this fix exists to close, through a second door.
    """
    import json as _json

    from distil.adapters.openai import exact_quote_tool_call_ids
    from distil.httpguard import is_chat_completions_path

    azure = "/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"
    assert is_chat_completions_path(azure), "Azure path must classify as chat completions"

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": _json.dumps({"command": "cat /app/main.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": V1},
        {"role": "user", "content": "now edit it"},
        {"role": "tool", "tool_call_id": "c2", "content": V2},
    ]
    keep = frozenset(exact_quote_tool_call_ids(messages))
    assert keep == {"c1"}
    out, _store, _stats = delta_encode(messages, keep_ids=keep)
    assert out[1]["content"] == V1, "the exempt read must reach the wire byte-exact"

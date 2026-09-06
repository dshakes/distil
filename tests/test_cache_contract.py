"""The cache contract — the bytes distil forwards must be stable across turns.

ADR 0008. Prompt caching is the single largest lever in distil's cost model, and it is
all-or-nothing: rewriting one byte at or before the provider's cache boundary throws
away the entry for the *whole* prefix, which costs far more than any digest saves. That
failure is silent — the request still succeeds, it just costs 2x — so it needs a test
rather than a code review.

The contract, stated as it is asserted here:

  (a) prefix stability   — for every message the client re-sends byte-identical, distil
                           forwards it byte-identical, for every index at or before the
                           provider's cache boundary.
  (b) suffix-only        — anything that does change lies strictly after that boundary.
  (c) handle determinism — a block digested at turn N carries the same handle at turn
                           N+1, because handles are content-addressed (sha256 of the
                           block, not a per-request nonce).
  (d) not guaranteed     — a client that rewrites its own history gets no promise, and
                           a client that sends no cache marker at all has no prefix to
                           protect, so the last-k recency window slides freely there.

Each provider shape is driven through the same public entry point the proxy uses, so a
regression in the adapter surfaces here and not only in production billing.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

import pytest

from distil.adapters.anthropic import compress_messages
from distil.adapters.gemini import compress_generate_request
from distil.adapters.openai import compress_chat_completions, compress_responses_input
from distil.compress.recency import cached_prefix_end

_HANDLE_RE = re.compile(r"handle=([0-9a-f]{8})")
TURNS = 6


def _log(n: int, tag: str) -> str:
    """A verbose, digestible tool output — the block class the contract is about."""
    return "\n".join(
        f"2026-09-04 10:00:{i:02d} INFO {tag} worker-{i} handled request id={i} ok"
        for i in range(n)
    )


def _key(obj: Any) -> str:
    """Serialize the way the proxy actually forwards a changed body.

    Deliberately NOT ``sort_keys=True``. The provider's cache matches on exact bytes, so
    key order is part of the prefix — normalising it here would let a transform that
    reorders a dict bust the cache in production while every assertion below stayed
    green. Same separators and ``ensure_ascii`` as ``proxy._serialize_if_changed``, so
    what this compares is what actually goes on the wire.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- shapes
# Each builder returns the request items for a session of `turns` tool round-trips,
# exactly as the client would re-send them on that turn.


def _anthropic(turns: int, mark: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "kick off the run"}]}
    ]
    for t in range(turns):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"t{t}",
                        "name": "bash",
                        "input": {"cmd": f"run {t}"},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{t}", "content": _log(40, f"step{t}")}
                ],
            }
        )
    if mark == "first":
        # Pinned at the system-ish head and never moved.
        msgs[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
    elif mark == "moving":
        # What Claude Code does: pin the newest turn so the whole history is cached.
        msgs[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}
    return msgs


def _openai_chat(turns: int, mark: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "you are a build agent"}]
    for t in range(turns):
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{t}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": _log(40, f"step{t}")})
    return msgs


def _openai_responses(turns: int, mark: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "kick off the run"}],
        }
    ]
    for t in range(turns):
        items.append(
            {"type": "function_call", "call_id": f"c{t}", "name": "bash", "arguments": "{}"}
        )
        items.append(
            {"type": "function_call_output", "call_id": f"c{t}", "output": _log(40, f"step{t}")}
        )
    return items


def _gemini(turns: int, mark: str) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "kick off the run"}]}]
    for t in range(turns):
        contents.append(
            {"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {"cmd": t}}}]}
        )
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "bash",
                            "response": {"stdout": _log(40, f"step{t}")},
                        }
                    }
                ],
            }
        )
    return contents


def _fwd_anthropic(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compress_messages(msgs)[0]


def _fwd_openai_chat(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compress_chat_completions(msgs)[0]


def _fwd_openai_responses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compress_responses_input(items)[0]


def _fwd_gemini(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compress_generate_request({"contents": contents})[0]["contents"]


Builder = Callable[[int, str], list[dict[str, Any]]]
Forwarder = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

# `mark` is the client's cache_control placement. Only Anthropic has an explicit
# marker; OpenAI and Gemini cache implicitly, so their boundary is "everything sent"
# and the adapters give them no recency carve-out at all.
SHAPES: list[tuple[str, Builder, Forwarder, str]] = [
    ("anthropic/marker-at-head", _anthropic, _fwd_anthropic, "first"),
    ("anthropic/marker-moves", _anthropic, _fwd_anthropic, "moving"),
    ("anthropic/no-marker", _anthropic, _fwd_anthropic, "none"),
    ("openai/chat-completions", _openai_chat, _fwd_openai_chat, "none"),
    ("openai/responses", _openai_responses, _fwd_openai_responses, "none"),
    ("gemini/generateContent", _gemini, _fwd_gemini, "none"),
]


def _boundary(shape: str, items: list[dict[str, Any]]) -> int:
    """The index the provider has cached through, on this turn.

    Anthropic caches only what the client marks. OpenAI and Gemini cache implicitly and
    commit everything they are sent, so their boundary is the last index — which is why
    those adapters exempt nothing.
    """
    if shape.startswith("anthropic"):
        return cached_prefix_end(items)
    return len(items) - 1


def _walk(shape: str, build: Builder, forward: Forwarder, mark: str):
    """Replay a growing session, yielding per-turn (index, same_input, same_output, hw)."""
    high_water = -1
    prev_in: list[str] | None = None
    prev_out: list[str] | None = None
    for turns in range(1, TURNS + 1):
        items = build(turns, mark)
        sent = forward(items)
        cur_in = [_key(m) for m in items]
        cur_out = [_key(m) for m in sent]
        assert len(cur_out) == len(cur_in), "compression must not add or drop messages"
        if prev_in is not None:
            for i in range(len(prev_in)):
                yield i, prev_in[i] == cur_in[i], prev_out[i] == cur_out[i], high_water
        high_water = max(high_water, _boundary(shape, items))
        prev_in, prev_out = cur_in, cur_out


@pytest.mark.parametrize(("shape", "build", "forward", "mark"), SHAPES, ids=[s[0] for s in SHAPES])
def test_cached_prefix_is_byte_stable(shape: str, build, forward, mark: str) -> None:
    """(a) + (b): same bytes in, same bytes out — for everything the provider has cached.

    The high-water mark is the furthest index any earlier turn committed to a cached
    prefix, not just this turn's marker: once the provider has cached through index i,
    rewriting i on any later turn invalidates the entry.
    """
    violations = [
        (i, hw)
        for i, same_in, same_out, hw in _walk(shape, build, forward, mark)
        if same_in and not same_out and i <= hw
    ]
    assert not violations, (
        f"{shape}: distil rewrote {len(violations)} message(s) the client re-sent "
        f"byte-identical, at or before the provider's cache boundary "
        f"(indices {sorted({i for i, _ in violations})}) — this busts the prompt cache "
        "for the whole prefix"
    )


@pytest.mark.parametrize(("shape", "build", "forward", "mark"), SHAPES, ids=[s[0] for s in SHAPES])
def test_implicit_cache_providers_never_drift(shape: str, build, forward, mark: str) -> None:
    """OpenAI and Gemini cache everything they are sent, so for them the contract is
    absolute: no same-input message may ever change, at any index. Anthropic with a
    marker at the head is exempt from this stronger form — see clause (d) and the
    dedicated tail test below."""
    if shape.startswith("anthropic"):
        pytest.skip("Anthropic caches only what the client marks; covered by the tail test")
    drift = [
        i
        for i, same_in, same_out, _ in _walk(shape, build, forward, mark)
        if same_in and not same_out
    ]
    assert not drift, f"{shape}: implicitly-cached content drifted at indices {sorted(set(drift))}"


def test_uncached_tail_is_the_only_place_that_moves() -> None:
    """(d), stated as a measurement rather than a promise.

    A client that sends no cache marker has no prefix to protect, so the last-k recency
    window slides and a block goes verbatim on one turn and digested on the next. That
    is intended. This test exists so the exception stays *bounded*: it must happen only
    strictly after the boundary, and it must actually happen — if it stopped, the
    recency carve-out would be silently dead.
    """
    tail = [
        i
        for i, same_in, same_out, hw in _walk(
            "anthropic/no-marker", _anthropic, _fwd_anthropic, "none"
        )
        if same_in and not same_out
    ]
    assert tail, "the recency carve-out never fired — is it still wired up?"
    inside = [
        i
        for i, same_in, same_out, hw in _walk(
            "anthropic/no-marker", _anthropic, _fwd_anthropic, "none"
        )
        if same_in and not same_out and i <= hw
    ]
    assert not inside, f"tail churn leaked into the cached prefix at {sorted(set(inside))}"


def test_a_moving_marker_pins_everything() -> None:
    """The realistic client shape (Claude Code pins the newest turn) must be totally
    stable: with the whole history cached there is no uncached tail, so nothing may
    move. This is the configuration that actually bills, so it gets its own assertion."""
    drift = [
        i
        for i, same_in, same_out, _ in _walk(
            "anthropic/marker-moves", _anthropic, _fwd_anthropic, "moving"
        )
        if same_in and not same_out
    ]
    assert not drift, f"fully-cached session drifted at indices {sorted(set(drift))}"


def test_handles_are_content_addressed_not_per_request() -> None:
    """(c): the handle is sha256(block)[:8], so the same block digests to the same bytes
    on every turn. A random per-request handle would rewrite every digest stub on every
    turn and bust the cache by itself, while every other assertion here still passed."""
    text = _log(80, "deploy")
    seen: set[str] = set()
    for _ in range(3):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t0", "name": "bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t0", "content": text}],
            },
            {"role": "user", "content": [{"type": "text", "text": "and now something else"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}],
            },
        ]
        sent, _store = compress_messages(msgs)
        found = set(_HANDLE_RE.findall(json.dumps(sent)))
        assert found, "the verbose block was not digested — the fixture no longer exercises (c)"
        seen |= found
    assert len(seen) == 1, f"handle is not deterministic across requests: {sorted(seen)}"
    assert seen == {hashlib.sha256(text.encode()).hexdigest()[:8]}, (
        "handle is not the content address of the block it replaces"
    )


def test_key_reordering_is_caught_by_the_comparison() -> None:
    """The tests above only mean something if `_key` can see key order.

    With `sort_keys=True` a transform that rebuilt a dict in a different order would
    forward different bytes, bust the provider's prefix cache, and still pass every
    assertion here. This drives that exact transform and asserts the comparison rejects
    it — a negative test for the harness rather than for the compressor.
    """
    original = {"role": "user", "content": "hello", "cache_control": {"type": "ephemeral"}}
    reordered = {k: original[k] for k in reversed(list(original))}

    assert original == reordered, "same mapping — only the serialized key order differs"
    assert _key(original) != _key(reordered), (
        "_key normalises key order, so a reordering transform would go undetected"
    )
    assert json.dumps(original, sort_keys=True) == json.dumps(reordered, sort_keys=True), (
        "sort_keys is what would have hidden it — if this fails the fixture is wrong"
    )


def test_key_serializes_the_way_the_proxy_forwards() -> None:
    """`_key` must not drift from the proxy's encoder, or the contract is asserted
    against bytes nobody sends."""
    from distil.proxy import _serialize_if_changed

    body = {"z": "café", "a": [1, {"b": 2}]}
    # A body that differs from `raw` forces the re-serialize branch — the one that
    # decides the bytes actually forwarded when a transform changed something.
    assert _key(body).encode() == _serialize_if_changed(b'{"different":1}', body)

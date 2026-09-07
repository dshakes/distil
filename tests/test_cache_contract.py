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


# ===========================================================================
# ADR 0011 — forwarded-bytes prefix replay
#
# Clause (d) above says a client that rewrites its own history gets no promise. Real
# agentic clients rewrite it on every turn without changing a token the model reads,
# so "no promise" was costing the whole prefix. Replay closes that: for the longest
# canonically-equal leading prefix, distil forwards the bytes it forwarded last turn.
#
# The assertion is relative, and has to be. distil already moves some bytes on purpose
# (the recency carve-out, clause (d) again), so "nothing drifted" would fail on a
# healthy session and tell us nothing about replay. What replay promises is narrower
# and exactly measurable: **a non-semantic client rewrite adds no drift the same
# session would not have had anyway.** Each test below therefore compares three arms —
# the un-rewritten session, the rewritten session, and the rewritten session with
# replay on — so the control proves the rewrite was expensive before the fix claims to
# have made it free.
# ===========================================================================

from distil import prefixreplay  # noqa: E402


def _strip_marks(node: Any) -> Any:
    """Same payload with every ``cache_control`` removed.

    The comparison is "did the content bytes hold", not "did the marker stay put". The
    marker tells the provider where to cut the cached span; it is not part of the span,
    and ``prefix._flatten`` has skipped it since 1.41 for the same reason. Replay lets
    it follow the client, so comparing with it in would flag the one movement that is
    supposed to happen.
    """
    if isinstance(node, dict):
        return {k: _strip_marks(v) for k, v in node.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_strip_marks(x) for x in node]
    return node


def _clone(item: Any) -> Any:
    return json.loads(_key(item))


def _churn_marker(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """The client advances its ``cache_control`` breakpoint to the newest block."""
    out = [_clone(_strip_marks(m)) for m in items]
    if out:
        blocks = out[-1].get("content")
        if isinstance(blocks, list) and blocks and isinstance(blocks[-1], dict):
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        else:
            out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def _churn_index(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """An SDK shim stamps positional ``index`` fields, renumbered every turn.

    Stamped wherever that SDK would put them: content/parts blocks on the block-shaped
    providers, and ``tool_calls`` entries on Chat Completions, which is where the OpenAI
    streaming types actually carry an index.
    """
    out = [_clone(m) for m in items]
    for m in out:
        for key in ("content", "parts", "tool_calls"):
            for j, b in enumerate(m.get(key) or []):
                if isinstance(b, dict):
                    b["index"] = j + turn
    return out


def _churn_sugar(items: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """String content and a single text block, flipped on alternating turns.

    Both spellings are legal wherever the other is, and SDK round-trips flip between
    them. To the model they are the same bytes; to a byte-exact prefix cache they are
    not.
    """
    to_list = turn % 2 == 0

    def flip(node: Any) -> Any:
        if isinstance(node, list):
            return [flip(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {k: flip(v) for k, v in node.items()}
        c = out.get("content")
        if to_list and isinstance(c, str):
            out["content"] = [{"type": "text", "text": c}]
        elif (
            not to_list
            and isinstance(c, list)
            and len(c) == 1
            and isinstance(c[0], dict)
            and set(c[0]) == {"type", "text"}
        ):
            out["content"] = c[0]["text"]
        return out

    return [flip(_clone(m)) for m in items]


# Gemini's `parts` have no string/single-block sugar to flip — there is no second
# spelling — so that pairing is skipped by name rather than silently passing on a churn
# that did nothing.
_NOT_APPLICABLE = {("gemini/generateContent", "string-block-sugar")}

CHURNS = [
    ("marker-advances", _churn_marker),
    ("index-stamped", _churn_index),
    ("string-block-sugar", _churn_sugar),
]


def _stable_prefix(
    shape: str, build: Builder, forward: Forwarder, mark: str, churn: Any, *, replay: bool
) -> list[int]:
    """Per turn, how many LEADING messages went out byte-identical to the previous turn.

    A provider cache entry covers a byte prefix, so this is the number that bills: what
    happens after the first changed byte is already uncached and costs nothing extra to
    change again. Counting total drifted indices instead would score churn in the
    volatile tail — which distil creates on purpose (clause (d)) — as a regression.

    ``churn=None`` is the client that does not rewrite its history.
    """
    prefixreplay.reset()
    key = f"{shape}/{getattr(churn, '__name__', 'none')}/{replay}"
    prev: list[str] | None = None
    lens: list[int] = []
    for turns in range(1, TURNS + 1):
        items = build(turns, mark)
        if churn is not None:
            items = churn(items, turns)
        sent = forward(items)
        if replay:
            sent, _stats = prefixreplay.replay(key, items, sent)
        cur = [_key(_strip_marks(m)) for m in sent]
        if prev is not None:
            n = 0
            while n < min(len(prev), len(cur)) and prev[n] == cur[n]:
                n += 1
            lens.append(n)
        prev = cur
    prefixreplay.reset()
    return lens


@pytest.mark.parametrize(("churn_name", "churn"), CHURNS, ids=[c[0] for c in CHURNS])
@pytest.mark.parametrize(("shape", "build", "forward", "mark"), SHAPES, ids=[s[0] for s in SHAPES])
def test_a_client_rewrite_costs_bytes_without_replay(
    shape: str, build, forward, mark: str, churn_name: str, churn
) -> None:
    """The control arm: with replay off, each rewrite shape really does move bytes the
    un-rewritten session would have held. Without this, the test below could stay green
    through a total removal of the mechanism.

    ``marker-advances`` is exempt and that is a finding, not a hole: anchoring the
    recency carve-out to the client's breakpoint (ADR 0008) already made a moving marker
    free, so there is nothing left for replay to repair there. The test below still runs
    it, to catch replay *re-introducing* the drift.
    """
    if (shape, churn_name) in _NOT_APPLICABLE:
        pytest.skip("this provider has no second spelling for that field")
    if churn_name == "marker-advances":
        pytest.skip("already neutralised by breakpoint-anchored recency (ADR 0008)")
    baseline = _stable_prefix(shape, build, forward, mark, None, replay=False)
    control = _stable_prefix(shape, build, forward, mark, churn, replay=False)
    assert any(c < b for c, b in zip(control, baseline)), (
        f"{shape} under a {churn_name} rewrite keeps as much prefix as the un-rewritten "
        f"session ({control} vs {baseline}) — this fixture proves nothing"
    )


@pytest.mark.parametrize(("churn_name", "churn"), CHURNS, ids=[c[0] for c in CHURNS])
@pytest.mark.parametrize(("shape", "build", "forward", "mark"), SHAPES, ids=[s[0] for s in SHAPES])
def test_replay_holds_the_prefix_through_a_client_rewrite(
    shape: str, build, forward, mark: str, churn_name: str, churn
) -> None:
    """The headline: a non-semantic client rewrite costs no byte-stable prefix.

    Compared with the proxy's own serializer (no ``sort_keys`` — see ``_key``), so a
    transform that merely re-ordered keys would still be caught.
    """
    if (shape, churn_name) in _NOT_APPLICABLE:
        pytest.skip("this provider has no second spelling for that field")
    baseline = _stable_prefix(shape, build, forward, mark, None, replay=False)
    with_replay = _stable_prefix(shape, build, forward, mark, churn, replay=True)
    short = [(t, r, b) for t, (r, b) in enumerate(zip(with_replay, baseline), start=2) if r < b]
    assert not short, (
        f"{shape} under a {churn_name} rewrite forwarded a SHORTER byte-stable prefix "
        f"than the un-rewritten session at (turn, with-replay, baseline) {short} — the "
        "provider re-bills the difference"
    )


def test_a_semantic_edit_breaks_the_prefix_at_exactly_that_index() -> None:
    """Replay stops at the first divergence, and the first divergence is where the client
    actually changed something. Not one index earlier (a lost cache hit), and emphatically
    not one later (stale bytes forwarded over a real edit)."""
    prefixreplay.reset()
    base = _anthropic(4, "moving")
    prefixreplay.replay("edit", base, _fwd_anthropic(base))

    edited = _clone(base)
    target = 3  # an assistant tool_use in the middle of the history
    edited[target]["content"][0]["input"] = {"cmd": "something else entirely"}
    fresh = _fwd_anthropic(edited)
    out, stats = prefixreplay.replay("edit", edited, fresh)

    assert stats.hits == target, f"replay stopped at {stats.hits}, expected exactly {target}"
    assert stats.misses == len(edited) - target
    assert _key(out[target]) == _key(fresh[target]), (
        "the edited message was overlaid with older bytes"
    )
    prefixreplay.reset()


def test_a_changed_tool_result_is_never_overlaid_with_old_bytes() -> None:
    """The failure that would matter: a tool_result whose bytes changed keeps the new
    bytes. A file re-read after an edit is exactly this shape, and forwarding the old
    version would hand the agent a stale view of a file it just wrote."""
    prefixreplay.reset()
    first = _anthropic(3, "moving")
    prefixreplay.replay("tr", first, _fwd_anthropic(first))

    second = _clone(first)
    idx = 2  # the first tool_result
    assert second[idx]["content"][0]["type"] == "tool_result", "fixture drifted"
    second[idx]["content"][0]["content"] = _log(40, "step0") + "\nNEW LINE AFTER THE EDIT"
    fresh = _fwd_anthropic(second)
    out, stats = prefixreplay.replay("tr", second, fresh)

    assert stats.hits == idx, f"replay reached index {stats.hits}, past the changed result"
    assert _key(out[idx]) == _key(fresh[idx]), "the changed tool_result was replayed"
    blob = json.dumps(out[idx])
    assert "NEW LINE AFTER THE EDIT" in blob or "handle=" in blob, (
        "the changed tool_result was neither forwarded verbatim nor digested"
    )
    prefixreplay.reset()


def test_thinking_signatures_are_never_touched() -> None:
    """Signed thinking blocks are cryptographic: a signature that does not match its text
    is rejected by the provider, and a replay that paired one turn's signature with
    another turn's text would do exactly that. Replaying identical bytes is safe; a
    changed signature must break the prefix like any other content change. (Replays have
    died on signed thinking blocks before — see the 1.51.1 shadow fix.)"""
    prefixreplay.reset()

    def convo(sig: str) -> list[dict[str, Any]]:
        return [
            {"role": "user", "content": [{"type": "text", "text": "think about it"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "step one, step two", "signature": sig},
                    {"type": "redacted_thinking", "data": "AAAA" + sig},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "go on"}]},
        ]

    turn1 = convo("SIGNATURE-ONE")
    out1, _ = prefixreplay.replay("sig", turn1, _fwd_anthropic(turn1))
    assert "SIGNATURE-ONE" in json.dumps(out1)

    # Same signature, non-semantic churn only: replayed byte-for-byte, signature intact.
    turn2 = _churn_index(convo("SIGNATURE-ONE"), 1)
    out2, stats2 = prefixreplay.replay("sig", turn2, _fwd_anthropic(turn2))
    assert stats2.hits >= 2, "a purely non-semantic rewrite should have held the prefix"
    assert _key(out2[1]) == _key(out1[1]), "the signed block's bytes moved"

    # A re-signed block is a content change: it must break the prefix at its index.
    turn3 = convo("SIGNATURE-TWO")
    out3, stats3 = prefixreplay.replay("sig", turn3, _fwd_anthropic(turn3))
    assert stats3.hits == 1, f"replay held a re-signed block (hits={stats3.hits})"
    assert "SIGNATURE-TWO" in json.dumps(out3[1]), "the new signature was overwritten by the old"
    assert "SIGNATURE-ONE" not in json.dumps(out3)
    prefixreplay.reset()


def test_replay_never_crosses_a_model_or_tools_change() -> None:
    """A cached prefix belongs to one model + system + tool set. Replaying across a change
    would forward a history the provider holds no entry for and the client did not send."""
    msgs = _anthropic(3, "moving")
    a = prefixreplay.lineage_key({"model": "claude-opus-4-8", "tools": []}, msgs)
    assert a != prefixreplay.lineage_key({"model": "claude-sonnet-5", "tools": []}, msgs)
    assert a != prefixreplay.lineage_key(
        {"model": "claude-opus-4-8", "tools": [{"name": "bash"}]}, msgs
    )
    assert a != prefixreplay.lineage_key(
        {"model": "claude-opus-4-8", "tools": [], "system": "be brief"}, msgs
    )
    assert a == prefixreplay.lineage_key({"model": "claude-opus-4-8", "tools": []}, msgs)


def test_a_moving_marker_does_not_fork_the_lineage() -> None:
    """The lineage is pinned by the conversation's head message, and on the Claude Code
    shape that head is where a one-turn session puts its marker. Keying on the raw head
    would start a fresh lineage the moment the marker advanced — every session's first
    replay, lost."""
    head_marked = _anthropic(1, "first")
    head_bare = _anthropic(1, "none")
    body = {"model": "claude-opus-4-8"}
    assert prefixreplay.lineage_key(body, head_marked) == prefixreplay.lineage_key(body, head_bare)


def test_lineage_state_is_bounded() -> None:
    """Unbounded per-session state in a long-lived proxy is a leak with a nice name."""
    prefixreplay.reset()
    for n in range(prefixreplay._MAX_LINEAGES * 3):
        msgs = [{"role": "user", "content": f"session {n}"}]
        prefixreplay.replay(f"k{n}", msgs, msgs)
    assert len(prefixreplay._LINEAGES) == prefixreplay._MAX_LINEAGES
    prefixreplay.reset()


def test_an_oversized_history_is_forgotten_rather_than_held() -> None:
    """The byte ceiling has to release the memory, not merely stop replaying."""
    prefixreplay.reset()
    small = [{"role": "user", "content": "hello"}]
    prefixreplay.replay("big", small, small)
    assert "big" in prefixreplay._LINEAGES
    huge = [{"role": "user", "content": "x" * (prefixreplay._MAX_STATE_BYTES + 1)}]
    out, stats = prefixreplay.replay("big", huge, huge)
    assert out == huge and stats.hits == 0
    assert "big" not in prefixreplay._LINEAGES, "an oversized session stayed resident"
    prefixreplay.reset()


def test_canonical_only_ignores_the_fields_it_names() -> None:
    """The strip set, pinned from the other side.

    Everything ``prefixreplay`` does rests on its comparison key being *narrow*: each
    field it ignores is a promise that two items differing only in that field are the
    same input, and a key that ignores too much declares different inputs equal and
    forwards stale content. A dropped ``else`` in the recursion once made every message
    canonicalise to ``{}`` — caught by the parametrized tests above, but only because
    they happened to cover it. This says it directly.
    """
    base = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t0", "content": "the output"},
            {"type": "thinking", "thinking": "reasoning", "signature": "SIG"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
            {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
        ],
    }
    key = prefixreplay.canonical(base)

    def mutate(path: list[Any], value: Any) -> dict[str, Any]:
        out = json.loads(_key(base))
        node: Any = out
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value
        return out

    for path, value, what in [
        (["role"], "assistant", "role"),
        (["content", 0, "content"], "different output", "tool_result content"),
        (["content", 0, "tool_use_id"], "t9", "tool_use_id"),
        (["content", 1, "thinking"], "other reasoning", "thinking text"),
        (["content", 1, "signature"], "OTHER", "thinking signature"),
        (["content", 2, "input"], {"cmd": "rm"}, "tool input"),
        (["content", 2, "name"], "python", "tool name"),
        (["content", 3, "source"], {"type": "base64", "data": "BBBB"}, "image data"),
        (["content", 0, "type"], "text", "block type"),
    ]:
        assert prefixreplay.canonical(mutate(path, value)) != key, (
            f"canonical() ignores {what} — a change there would be replayed away"
        )

    # ...and the two it does ignore, on the same fixture.
    marked = json.loads(_key(base))
    marked["content"][0]["cache_control"] = {"type": "ephemeral"}
    marked["content"][1]["index"] = 7
    assert prefixreplay.canonical(marked) == key


def test_a_breakpoint_with_nowhere_to_sit_stops_the_replay() -> None:
    """The client re-spells a bare string as one text block *and* puts its breakpoint on
    it. The comparison key calls those two the same input — that is the sugar rule, and
    it is what makes the rewrite repairable — but last turn's form is a string with no
    block to carry the marker. Forwarding it anyway would delete the breakpoint the
    client asked for, and on Anthropic the breakpoint IS the cache entry: a hit bought by
    destroying the thing being hit. Replay stops at that index instead.
    """
    prefixreplay.reset()
    plain: list[Any] = [{"role": "user", "content": "hello"}]
    marked: list[Any] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
        }
    ]

    prefixreplay.replay("mark", plain, _clone(plain))
    out, stats = prefixreplay.replay("mark", marked, _clone(marked))
    assert stats.hits == 0, "replay held a message whose breakpoint it could not carry"
    assert _key(out) == _key(marked), (
        f"the client's breakpoint was dropped on the way out: forwarded {_key(out)}"
    )

    # The other direction is a genuine repair and must still happen: the client has no
    # marker this turn, so last turn's block spelling goes out with the marker stripped.
    prefixreplay.reset()
    prefixreplay.replay("mark", marked, _clone(marked))
    back, stats2 = prefixreplay.replay("mark", plain, _clone(plain))
    assert stats2.hits == 1, "a re-spelling with no marker to carry is still replayable"
    assert "cache_control" not in _key(back), "last turn's breakpoint was left behind"
    prefixreplay.reset()


def test_a_tool_payload_is_never_canonicalised_away() -> None:
    """The strip set applies to MESSAGE structure, not one level down inside a tool
    payload. Gemini's ``functionResponse.response`` is arbitrary tool output, and a key
    in it called ``index`` is data — stripping it there would declare two different
    results equal and forward the first one's bytes for the second. Same for the
    Responses API's ``output``. A stale tool result is the one failure this module must
    never produce, so the payload keys are compared opaquely.
    """
    shapes = {
        "gemini functionResponse": lambda n: [
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "rows", "response": {"index": n, "v": "A"}}}
                ],
            }
        ],
        "responses function_call_output": lambda n: [
            {"type": "function_call_output", "call_id": "c1", "output": {"index": n, "v": "A"}}
        ],
    }
    for what, convo in shapes.items():
        prefixreplay.reset()
        first = convo(1)
        prefixreplay.replay(what, first, _clone(first))
        second = convo(2)
        out, stats = prefixreplay.replay(what, second, _clone(second))
        assert stats.hits == 0, f"{what}: a changed tool payload was treated as unchanged"
        assert _key(out) == _key(second), (
            f"{what}: the client sent index=2 and distil forwarded {_key(out)} — "
            f"the previous turn's tool output was replayed over a different one"
        )
    prefixreplay.reset()


def test_replay_never_writes_back_into_the_state_it_replayed_from() -> None:
    """Re-placing the client's markers must build a new item, not write through to the
    stored one. The stored item is shared with the lineage and, in a threaded server,
    with any concurrent request on the same conversation; mutating it in place would
    hand that request this request's marker placement — and would leave the lineage
    holding bytes it never actually forwarded.
    """
    prefixreplay.reset()

    def convo(mark: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": "kick off the run"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "on it"}]},
        ]
        msgs[mark]["content"][0]["cache_control"] = {"type": "ephemeral"}
        return msgs

    first = convo(0)
    prefixreplay.replay("cow", first, _clone(first))
    held = prefixreplay._LINEAGES["cow"].forwarded[0]
    before = _key(held)
    assert "cache_control" in before, "fixture drifted — nothing for the marker to move off"

    second = convo(1)  # the client advanced its breakpoint to the newest block
    out, stats = prefixreplay.replay("cow", second, _clone(second))
    assert stats.hits == 2, "a marker move is not a content change"
    assert "cache_control" not in _key(out[0]), "the replayed item kept last turn's marker"
    assert _key(held) == before, (
        "replay mutated the item it replayed from — the stored bytes now carry this "
        "turn's marker placement, which no request ever sent"
    )
    prefixreplay.reset()


def test_wire_serializes_the_way_the_proxy_forwards() -> None:
    """``prefixreplay._wire`` decides whether a repair is counted as one. If it drifts
    from the proxy's encoder it is measuring bytes nobody sends."""
    from distil.proxy import _serialize_if_changed

    body = {"z": "café", "a": [1, {"b": 2}]}
    assert prefixreplay._wire(body).encode() == _serialize_if_changed(b'{"different":1}', body)


def test_replay_forwards_unchanged_on_shapes_it_does_not_understand() -> None:
    """Fail-open, on the hottest path in the product. Anything replay cannot reason
    about — a non-list payload, a non-dict message, a block that is a bare string — is
    forwarded exactly as the compressor produced it, and the marker re-placement copes
    with a message-level ``cache_control`` as well as a per-block one."""
    prefixreplay.reset()
    out, stats = prefixreplay.replay("odd", {"not": "a list"}, {"not": "a list"})  # type: ignore[arg-type]
    assert out == {"not": "a list"} and stats.hits == 0

    weird: list[Any] = [
        "a bare string where a message should be",
        {"role": "user", "content": ["a bare block", {"type": "text", "text": "hi"}]},
        {"role": "user", "content": "flat", "cache_control": {"type": "ephemeral"}},
    ]
    prefixreplay.replay("odd", weird, weird)
    again = json.loads(_key(weird))
    again[2].pop("cache_control")  # the client moved its message-level marker off
    again[1]["cache_control"] = {"type": "ephemeral"}  # ...and onto the one before it
    out2, stats2 = prefixreplay.replay("odd", again, again)
    assert stats2.hits == len(again), "a marker move is not a content change"
    assert "cache_control" not in out2[2]
    assert out2[1]["cache_control"] == {"type": "ephemeral"}
    assert out2[1]["content"][0] == "a bare block"
    prefixreplay.reset()


# --------------------------------------------------------------------------- servers
# distil ships three servers, and a default-on property that only one of them has is a
# property users do not have. These drive the Anthropic shape through each server's own
# entry point, against a stub upstream, and assert on the bytes the upstream received —
# so compression, replay and serialization are all inside the measurement.


def _churned_body(turn: int) -> dict[str, Any]:
    """One conversation, re-spelled the way a real client re-spells it every turn: an
    SDK stamps a positional `index` on each block and renumbers it, and the
    cache_control breakpoint advances to the newest block."""
    msgs = _anthropic(3, "moving")
    for m in msgs:
        for j, b in enumerate(m["content"]):
            b["index"] = j + turn
    return {"model": "claude-test", "max_tokens": 64, "messages": msgs}


def _stub_upstream():
    """A threaded HTTP server that records the last body it was posted."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: dict[str, bytes] = {}
    payload = json.dumps(
        {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-test",
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
    ).encode()

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["raw"] = self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _post_json(port: int, body: dict[str, Any]) -> None:
    import urllib.request

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def _serve_threaded(handler_cls):
    import threading
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_the_threaded_proxy_holds_the_prefix() -> None:
    from distil.proxy import build_handler

    prefixreplay.reset()
    up, seen = _stub_upstream()
    px = _serve_threaded(build_handler(f"http://127.0.0.1:{up.server_address[1]}"))
    try:
        _post_json(px.server_address[1], _churned_body(1))
        first = seen["raw"]
        _post_json(px.server_address[1], _churned_body(2))
        assert seen["raw"] == first, "the threaded proxy re-billed the prefix"
    finally:
        px.shutdown()
        up.shutdown()
        prefixreplay.reset()


# ----------------------------------------------------------------- re-read × replay
# The re-read delta (ADR 0010, clause (e)) replaces a run of lines in a re-read with a
# reference to the earlier read that still carries them. It is prefix-deterministic by
# construction, and replay is the thing that decides whether that determinism reaches
# the wire — so the two are driven together, through the server, not asserted apart.

_HANDLERS = 60


def _module_lines() -> list[str]:
    """A file of many same-shaped, individually identifiable blocks (3 lines each)."""
    out: list[str] = []
    for i in range(_HANDLERS):
        out += [
            f"def handler_{i}(request):  # MARK",
            "    payload = request.json()",
            f"    return {{'ok': True, 'n': {i}, 'payload': payload}}",
        ]
    return out


def _handler(i: int) -> str:
    return "\n".join(_module_lines()[3 * i : 3 * i + 3])


def _reread_body(turn: int, *, extra: int = 0, edit: str | None = None) -> dict[str, Any]:
    """A read → re-read session (the delta fires on the second one), re-spelled the way a
    real client re-spells its history every turn: `index` stamps renumbered, breakpoint
    advanced to the newest block."""
    lines = _module_lines()

    def use(tid: str, name: str, inp: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
        }

    def res(tid: str, text: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
        }

    path = {"file_path": "/app/handlers.py"}
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "refactor the handlers"}]},
        use("r1", "Read", path),
        res("r1", "\n".join(lines[:120])),  # the base
        use("r2", "Read", path),
        res("r2", "\n".join(lines[60:])),  # the re-read: overlaps the base by 60 lines
    ]
    if edit is not None:
        msgs += [
            use("e1", "Edit", {**path, "old_string": edit, "new_string": edit + "  # patched"}),
            res("e1", "applied"),
        ]
    for k in range(extra):
        msgs += [use(f"b{k}", "bash", {"cmd": "pytest -q"}), res(f"b{k}", _log(30, f"run{k}"))]
    for m in msgs:
        for j, b in enumerate(m["content"]):
            b["index"] = j + turn
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {"model": "claude-test", "max_tokens": 64, "messages": msgs}


def _through_the_proxy(*bodies: dict[str, Any], replay: bool = True) -> list[list[dict[str, Any]]]:
    """Post each body to the threaded proxy in order; return the messages the upstream
    actually received for each."""
    from distil.proxy import build_handler

    prefixreplay.reset()
    up, seen = _stub_upstream()
    px = _serve_threaded(
        build_handler(f"http://127.0.0.1:{up.server_address[1]}", prefix_replay=replay)
    )
    out: list[list[dict[str, Any]]] = []
    try:
        for body in bodies:
            _post_json(px.server_address[1], body)
            out.append(json.loads(seen["raw"])["messages"])
    finally:
        px.shutdown()
        up.shutdown()
        prefixreplay.reset()
    return out


def test_a_re_read_stub_is_forwarded_byte_identical_on_the_next_turn() -> None:
    """The two transforms have to agree about determinism. The re-read delta emits a
    stub whose bytes are a pure function of the message prefix (ADR 0010), and replay
    forwards a message only when its canonical form has not changed — so the stub the
    provider cached at turn N must be the stub it is shown at turn N+1, right through
    the client renumbering its `index` stamps and advancing its breakpoint underneath.
    """
    turns = (_reread_body(1), _reread_body(2, extra=1))
    first, second = _through_the_proxy(*turns)

    assert "«distil-reread" in _key(first), "the re-read delta never fired — fixture is stale"
    # The control arm, same session with replay off: the rewrite really does move these
    # bytes, so the assertion below cannot pass through a dead mechanism.
    off1, off2 = _through_the_proxy(*turns, replay=False)
    assert [i for i in range(5) if _key(_strip_marks(off2[i])) != _key(_strip_marks(off1[i]))], (
        "the fixture stopped churning — a stability assertion with no churn proves nothing"
    )
    # Marks stripped for the same reason as `_stable_prefix`: the client moved its own
    # breakpoint this turn and replay re-places it where the client now wants it, which
    # is the one difference that is not a rewrite of the cached span.
    drift = [i for i in range(5) if _key(_strip_marks(second[i])) != _key(_strip_marks(first[i]))]
    assert not drift, f"the re-read session was re-billed at {drift}"


def test_replay_never_overlays_a_stub_the_compressor_has_taken_back() -> None:
    """The guard, on the transform most able to trip it. An `Edit` arriving at turn N+1
    whose `old_string` straddles the elision cut and runs past the base's last line is
    quotable from neither copy, so `_guard_quotes` withdraws the re-read delta and the
    block goes back to verbatim. Replay restores bytes, never decisions: it must forward
    that verbatim block, not last turn's stub, even though the client re-sent the message
    byte-identical and the whole prefix is otherwise a hit.
    """
    quote = "\n".join(_handler(i) for i in range(32, 42))
    first, second = _through_the_proxy(_reread_body(1), _reread_body(2, edit=quote))

    assert "«distil-reread" in _key(first[4]), "the re-read delta never fired — fixture is stale"
    assert "«distil-reread" not in _key(second[4]), (
        "replay overlaid the stub the compressor had just taken back — the agent's Edit "
        "now has no byte-exact copy of the lines it quotes"
    )
    assert quote in _key(second[4]).replace("\\n", "\n"), "the withdrawn lines did not come back"
    # ...and it stopped at exactly that message: the base read before it still replays.
    assert _key(second[2]) == _key(first[2]), "the divergence was applied to the wrong index"


def test_the_plain_proxy_never_shares_a_lineage_between_two_credentials() -> None:
    """The gateway scopes by tenant because it knows its tenants. The plain proxies
    forward the client's own key, so they scope by the key itself, hashed — the same
    boundary drawn with the only identity available. A proxy bound wider than loopback
    can serve two credentials, and a cached prefix belongs to one of them.
    """
    from distil.proxy import build_handler

    prefixreplay.reset()
    up, seen = _stub_upstream()
    px = _serve_threaded(build_handler(f"http://127.0.0.1:{up.server_address[1]}"))
    port = px.server_address[1]

    def post(api_key: str, turn: int) -> bytes:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages",
            data=json.dumps(_churned_body(turn)).encode(),
            headers={"content-type": "application/json", "x-api-key": api_key},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return seen["raw"]

    try:
        first = post("sk-one", 1)
        assert post("sk-one", 2) == first, "the same credential lost its own prefix"

        # `index` is ignored by the comparison, so a shared lineage would call the second
        # credential's request equal to the first's and overlay the first's stored bytes.
        other = post("sk-two", 3)
        served = json.loads(other)["messages"][0]["content"][0]["index"]
        assert served == 3, (
            f"cross-credential leak: sk-two sent index=3 and the proxy forwarded index="
            f"{served}, which is sk-one's — the two keys are sharing replay state"
        )
    finally:
        px.shutdown()
        up.shutdown()
        prefixreplay.reset()


def test_the_credential_scope_is_a_hash_and_nothing_else() -> None:
    """It goes in a lineage key that lives in memory and in no log, but it is derived
    from a secret, so it is hashed rather than carried. No credential header at all is
    an empty scope, which is the single-user proxy and must not become a fourth lineage."""
    key = "sk-ant-secret-value"
    scope = prefixreplay.credential_scope({"x-api-key": key})
    assert key not in scope and scope, "the raw credential reached the lineage key"
    assert scope == prefixreplay.credential_scope({"X-Api-Key": key}), "header case forked it"
    assert scope != prefixreplay.credential_scope({"x-api-key": key + "2"})
    assert prefixreplay.credential_scope({"authorization": f"Bearer {key}"}) not in ("", scope)
    assert prefixreplay.credential_scope({"content-type": "application/json"}) == ""


def test_the_gateway_holds_the_prefix_and_never_shares_it_between_tenants() -> None:
    """Plus the property that only the gateway has: a cached prefix belongs to one
    credential at the provider, so two tenants posting the identical conversation must
    not share replay state."""
    from distil.gateway import GatewayState, build_gateway_handler
    from distil.pricing import get as pricing_get

    prefixreplay.reset()
    up, seen = _stub_upstream()
    price = pricing_get("claude-opus-4-8")
    gw = _serve_threaded(
        build_gateway_handler(
            f"http://127.0.0.1:{up.server_address[1]}",
            GatewayState(price),
            price,
            trust_tenant_header=True,
        )
    )
    port = gw.server_address[1]

    def post(tenant: str, turn: int) -> bytes:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages",
            data=json.dumps(_churned_body(turn)).encode(),
            headers={"content-type": "application/json", "x-distil-tenant": tenant},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return seen["raw"]

    try:
        first = post("acme", 1)
        assert post("acme", 2) == first, "the gateway re-billed the prefix"

        # Cross-tenant isolation, stated as the value that would differ. globex posts
        # the same conversation with its own turn-3 `index` stamps. The canonical
        # comparison IGNORES `index`, so a shared lineage would find globex's request
        # canonically equal to acme's and overlay acme's stored bytes — `index` would
        # come back 1, acme's number, in a request globex sent with 3. Verified by
        # mutation: dropping `scope=tenant` from gateway.py fails both asserts below.
        other = post("globex", 3)
        served = json.loads(other)["messages"][0]["content"][0]["index"]
        assert served == 3, (
            f"cross-tenant leak: globex sent index=3 and the gateway forwarded index="
            f"{served}, which is acme's — the tenants are sharing replay state"
        )
        assert other != first, "one tenant's forwarded bytes leaked into another's request"
    finally:
        gw.shutdown()
        up.shutdown()
        prefixreplay.reset()


def test_the_async_proxy_holds_the_prefix() -> None:
    """The async proxy re-serializes every body, so its prefix is byte-stable only as
    long as the ITEMS are — which is what replay restores. It ran without this until the
    managed-install path made a default-on feature missing from one server a real gap."""
    aiohttp = pytest.importorskip("aiohttp")
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from distil.aproxy import make_app

    seen: dict[str, bytes] = {}

    async def echo(request):
        seen["raw"] = await request.read()
        return web.Response(body=b'{"ok":1}', content_type="application/json")

    async def go() -> None:
        up = web.Application()
        up.router.add_post("/v1/messages", echo)
        up_srv = TestServer(up)
        await up_srv.start_server()
        try:
            base = f"http://127.0.0.1:{up_srv.port}"
            client = TestClient(TestServer(make_app(base)))
            await client.start_server()
            try:
                await (await client.post("/v1/messages", json=_churned_body(1))).read()
                first = seen["raw"]
                await (await client.post("/v1/messages", json=_churned_body(2))).read()
                assert seen["raw"] == first, "the async proxy re-billed the prefix"
            finally:
                await client.close()
        finally:
            await up_srv.close()

    prefixreplay.reset()
    try:
        asyncio.run(go())
    finally:
        prefixreplay.reset()
    assert aiohttp is not None

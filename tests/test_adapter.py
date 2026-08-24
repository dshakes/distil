"""Tests for distil.adapters.anthropic — Phase 3 runtime adapter."""

from __future__ import annotations

import pytest

from distil.adapters.anthropic import (
    RestoreStore,
    compress_messages,
    place_cache_control,
    wrap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_TOOL_RESULT = "\n".join(
    [
        "Result from bash tool execution on the remote host:",
        "total disk usage: 48 GB across 12 partitions",
        "filesystem /dev/sda1: 32 GB used of 100 GB available",
        "filesystem /dev/sdb1: 16 GB used of 200 GB available",
        "warning: /tmp is 89% full — consider cleaning up old build artefacts",
        "warning: inode count on /var/log approaching limit (91% used)",
        "no errors detected in kernel ring buffer",
        "last boot: 2026-06-20T03:14:22Z (uptime 18h 42m)",
        "load averages: 0.23 0.31 0.29 (1m/5m/15m)",
        "memory: 14.2 GB used / 31.9 GB total, 0 GB swap",
        "top process: python3 pid=8821 cpu=4.1% mem=2.3%",
        "all health checks passed",
    ]
)  # 12 verbose lines — well above the 6-line threshold; dropped middle is longer than marker

SHORT_TOOL_RESULT = "\n".join(["line one", "line two", "line three"])  # 3 lines — below threshold

# Two trailing user turns push an earlier tool_result out of the recency-exempt
# window (anthropic._RECENCY_KEEP_TURNS = 2), so it is eligible for Tier-1 digestion
# again. The most recent K turns are always kept verbatim (see test_recency_*).
_PAD = [{"role": "user", "content": "next"}, {"role": "user", "content": "next"}]


def _make_tool_result_message(content: str | list) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": content}],
    }


# ---------------------------------------------------------------------------
# RestoreStore
# ---------------------------------------------------------------------------


class TestRestoreStore:
    def test_expand_returns_original(self) -> None:
        store = RestoreStore()
        store._record("abcd1234", "original text")
        assert store.expand("abcd1234") == "original text"

    def test_handles_property(self) -> None:
        store = RestoreStore()
        store._record("aaa", "x")
        store._record("bbb", "y")
        assert store.handles == frozenset({"aaa", "bbb"})

    def test_expand_missing_key_raises(self) -> None:
        store = RestoreStore()
        with pytest.raises(KeyError):
            store.expand("nonexistent")


# ---------------------------------------------------------------------------
# compress_messages — long tool_result is digested
# ---------------------------------------------------------------------------


class TestCompressMessagesLongToolResult:
    def setup_method(self) -> None:
        self.msg = _make_tool_result_message(LONG_TOOL_RESULT)
        # _PAD keeps self.msg out of the recency-exempt window so it still digests.
        self.new_messages, self.store = compress_messages([self.msg, *_PAD])

    def test_output_is_new_list(self) -> None:
        assert self.new_messages != [self.msg]

    def test_input_not_mutated(self) -> None:
        # Original message content must be unchanged.
        original_content = self.msg["content"][0]["content"]
        assert original_content == LONG_TOOL_RESULT

    def test_content_is_shrunk(self) -> None:
        compressed_content = self.new_messages[0]["content"][0]["content"]
        assert len(compressed_content) < len(LONG_TOOL_RESULT)

    def test_handle_marker_present(self) -> None:
        compressed_content = self.new_messages[0]["content"][0]["content"]
        assert "handle=" in compressed_content

    def test_store_has_handle(self) -> None:
        assert len(self.store.handles) == 1

    def test_original_recoverable(self) -> None:
        handle = next(iter(self.store.handles))
        assert self.store.expand(handle) == LONG_TOOL_RESULT


# ---------------------------------------------------------------------------
# compress_messages — short tool_result is passed through
# ---------------------------------------------------------------------------


class TestCompressMessagesShortToolResult:
    def setup_method(self) -> None:
        self.msg = _make_tool_result_message(SHORT_TOOL_RESULT)
        self.new_messages, self.store = compress_messages([self.msg])

    def test_content_unchanged(self) -> None:
        compressed_content = self.new_messages[0]["content"][0]["content"]
        # Short content: no digest marker expected (may have tier0 transforms but same
        # text since it's not JSON and has no repeated lines).
        assert "handle=" not in compressed_content

    def test_store_is_empty(self) -> None:
        assert len(self.store.handles) == 0


# ---------------------------------------------------------------------------
# compress_messages — tool_use block passed through unchanged
# ---------------------------------------------------------------------------


class TestCompressMessagesToolUseUnchanged:
    def test_tool_use_not_touched(self) -> None:
        tool_use_block = {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "bash",
            "input": {"command": "ls -la"},
        }
        msg = {"role": "assistant", "content": [tool_use_block]}
        new_messages, store = compress_messages([msg])
        assert new_messages[0]["content"][0] is tool_use_block
        assert len(store.handles) == 0


# ---------------------------------------------------------------------------
# compress_messages — image block passed through unchanged
# ---------------------------------------------------------------------------


class TestCompressMessagesImageUnchanged:
    def test_image_not_touched(self) -> None:
        image_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }
        msg = {"role": "user", "content": [image_block]}
        new_messages, store = compress_messages([msg])
        assert new_messages[0]["content"][0] is image_block
        assert len(store.handles) == 0


# ---------------------------------------------------------------------------
# compress_messages — assistant text passed through unchanged
# ---------------------------------------------------------------------------


class TestCompressMessagesAssistantTextUnchanged:
    def test_assistant_text_not_touched(self) -> None:
        text_block = {"type": "text", "text": "I will help you with that."}
        msg = {"role": "assistant", "content": [text_block]}
        new_messages, store = compress_messages([msg])
        assert new_messages[0]["content"][0] is text_block


# ---------------------------------------------------------------------------
# compress_messages — input list not mutated
# ---------------------------------------------------------------------------


class TestInputNotMutated:
    def test_original_messages_unchanged(self) -> None:
        import copy

        msgs = [_make_tool_result_message(LONG_TOOL_RESULT)]
        original = copy.deepcopy(msgs)
        compress_messages(msgs)
        assert msgs == original


# ---------------------------------------------------------------------------
# compress_messages — verbatim (Tier-0 only) vs default (digest)
# ---------------------------------------------------------------------------


class TestVerbatimParam:
    def test_verbatim_does_not_digest(self) -> None:
        """Verbatim mode is lossless-IN-CONTEXT: a large tool_result is never
        replaced by a Tier-1 digest stub the model can't recover."""
        msg = _make_tool_result_message(LONG_TOOL_RESULT)
        new_messages, store = compress_messages([msg], verbatim=True)
        assert len(store.handles) == 0  # nothing digested
        seen = new_messages[0]["content"][0]["content"]
        assert "<< +" not in seen  # no digest stub marker
        # Every non-empty original line still present (Tier-0 is semantically lossless).
        for line in LONG_TOOL_RESULT.splitlines():
            if line.strip():
                assert line in seen

    def test_default_does_digest(self) -> None:
        """Default mode keeps the aggressive reversible digest — the moat."""
        msg = _make_tool_result_message(LONG_TOOL_RESULT)
        _new, store = compress_messages([msg, *_PAD], verbatim=False)
        assert len(store.handles) == 1  # digested, recoverable via the store


# ---------------------------------------------------------------------------
# compress_messages — list-typed tool_result content
# ---------------------------------------------------------------------------


class TestListToolResultContent:
    def test_long_list_content_digested(self) -> None:
        content_list = [{"type": "text", "text": LONG_TOOL_RESULT}]
        msg = _make_tool_result_message(content_list)
        new_messages, store = compress_messages([msg, *_PAD])
        compressed_sub = new_messages[0]["content"][0]["content"][0]["text"]
        assert "handle=" in compressed_sub
        assert len(store.handles) == 1

    def test_original_recoverable_from_list_content(self) -> None:
        content_list = [{"type": "text", "text": LONG_TOOL_RESULT}]
        msg = _make_tool_result_message(content_list)
        _, store = compress_messages([msg, *_PAD])
        handle = next(iter(store.handles))
        assert store.expand(handle) == LONG_TOOL_RESULT


# ---------------------------------------------------------------------------
# place_cache_control
# ---------------------------------------------------------------------------


class TestPlaceCacheControl:
    def test_string_system_promoted_to_block(self) -> None:
        result = place_cache_control("You are helpful.", [])
        system = result["system"]
        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are helpful."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_list_system_last_block_marked(self) -> None:
        blocks = [
            {"type": "text", "text": "block one"},
            {"type": "text", "text": "block two"},
        ]
        result = place_cache_control(blocks, [])
        system = result["system"]
        assert "cache_control" not in system[0]
        assert system[1]["cache_control"] == {"type": "ephemeral"}

    def test_list_system_original_not_mutated(self) -> None:
        blocks = [{"type": "text", "text": "block one"}]
        place_cache_control(blocks, [])
        assert "cache_control" not in blocks[0]

    def test_messages_passed_through(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = place_cache_control("sys", msgs)
        assert result["messages"] is msgs


# ---------------------------------------------------------------------------
# wrap — delegates to fake client and compresses first
# ---------------------------------------------------------------------------


class TestWrap:
    def _make_fake_client(self) -> object:
        """Return a minimal duck-typed fake Anthropic client."""
        calls: list[dict] = []

        class FakeMessages:
            def create(self, **kwargs):
                calls.append(kwargs)
                return {"id": "msg_fake", "content": []}

        class FakeClient:
            def __init__(self):
                self.messages = FakeMessages()
                self._calls = calls

        return FakeClient()

    def test_wrap_delegates_to_real_client(self) -> None:
        fake = self._make_fake_client()
        client = wrap(fake)
        result = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result["id"] == "msg_fake"
        assert len(fake._calls) == 1  # type: ignore[attr-defined]

    def test_wrap_compresses_verbatim_no_unrecoverable_stub(self) -> None:
        # wrap() runs in verbatim mode: it must NEVER emit a Tier-1 digest stub, since
        # the in-process wrapper has no server-side expand loop to recover it (F2). Plain
        # text (no lossless transform applies) is therefore passed through byte-exact.
        fake = self._make_fake_client()
        client = wrap(fake)
        long_msg = _make_tool_result_message(LONG_TOOL_RESULT)
        client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[long_msg, *_PAD],
        )
        received_msgs = fake._calls[0]["messages"]  # type: ignore[attr-defined]
        compressed_content = received_msgs[0]["content"][0]["content"]
        assert "handle=" not in compressed_content  # no unrecoverable digest stub
        assert compressed_content == LONG_TOOL_RESULT  # verbatim: byte-exact passthrough

    def test_wrap_applies_lossless_tier0(self) -> None:
        # Verbatim still shrinks where it is provably lossless: a whitespace-padded JSON
        # tool_result is minified (semantically identical, fewer tokens), never stubbed.
        fake = self._make_fake_client()
        client = wrap(fake)
        padded = '{\n    "a": 1,\n    "b": [1, 2, 3]\n}'
        msg = _make_tool_result_message(padded)
        client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[msg, *_PAD],
        )
        received = fake._calls[0]["messages"][0]["content"][0]["content"]  # type: ignore[attr-defined]
        assert received == '{"a":1,"b":[1,2,3]}'

    def test_wrap_applies_cache_control_when_system_present(self) -> None:
        fake = self._make_fake_client()
        client = wrap(fake)
        client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "hi"}],
        )
        received_system = fake._calls[0]["system"]  # type: ignore[attr-defined]
        assert isinstance(received_system, list)
        assert received_system[0]["cache_control"] == {"type": "ephemeral"}

    def test_wrap_proxies_other_attributes(self) -> None:
        fake = self._make_fake_client()
        fake.api_key = "sk-test"  # type: ignore[attr-defined]
        client = wrap(fake)
        assert client.api_key == "sk-test"

    def test_input_not_mutated_via_wrap(self) -> None:
        import copy

        fake = self._make_fake_client()
        client = wrap(fake)
        msgs = [_make_tool_result_message(LONG_TOOL_RESULT)]
        original = copy.deepcopy(msgs)
        client.messages.create(model="claude-opus-4-5", max_tokens=1024, messages=msgs)
        assert msgs == original


def test_openai_tool_message_is_digested_and_reversible():
    # OpenAI shape: {"role":"tool","content": "<long string>"}
    from distil.adapters.anthropic import compress_messages

    long = "DECISION: keep this\n" + "\n".join(f"verbose log line {i}" for i in range(20))
    messages = [
        {"role": "user", "content": "investigate"},
        {"role": "tool", "tool_call_id": "c1", "content": long},
        *_PAD,  # keep the tool turn out of the recency-exempt window so it digests
    ]
    out, store = compress_messages(messages)
    tool_msg = out[1]
    assert len(tool_msg["content"]) < len(long)  # digested
    assert "DECISION: keep this" in tool_msg["content"]  # decision preserved
    assert any(store.expand(h) == long for h in store.handles)  # reversible


def test_tier0_never_inflates_tokens_on_blank_runs():
    """collapse_runs must not turn near-free blank-line runs into a count marker
    that costs MORE tokens — Tier-0 is reject-if-bigger by tokens."""
    from distil.tokenizer import DEFAULT as _tok

    text = "alpha\n\n\n\n\nbeta\n\n\n\n\ngamma\n\n\n\n\ndelta"
    new_messages, _store = compress_messages([{"role": "user", "content": text}], verbatim=True)
    seen = new_messages[0]["content"]
    assert _tok.count(seen) <= _tok.count(text)  # never inflates


# ---------------------------------------------------------------------------
# FIX 1 — recency exemption: latest tool outputs are never digested
# ---------------------------------------------------------------------------


def _tr(tid: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}],
    }


def _has_handle(msg: dict) -> bool:
    c = msg["content"][0]["content"]
    return isinstance(c, str) and "handle=" in c


def _log_body(tag: str) -> str:
    return "\n".join(f"{tag} verbose log line {i}" for i in range(20))


def test_recent_toolresults_kept_verbatim_older_are_digested() -> None:
    """The last _RECENCY_KEEP_TURNS tool-output turns must stay byte-exact so the
    agent always sees its most recent outputs; older ones still digest."""
    asst = {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]}
    msgs = [
        _tr("t1", _log_body("t1")),
        asst,
        _tr("t2", _log_body("t2")),
        asst,
        _tr("t3", _log_body("t3")),
        asst,
        _tr("t4", _log_body("t4")),
    ]  # tool turns at indices 0, 2, 4, 6; the last two (4, 6) are recency-exempt
    out, store = compress_messages(msgs)
    assert _has_handle(out[0]) and _has_handle(out[2])  # older → digested
    assert not _has_handle(out[4]) and not _has_handle(out[6])  # recent → verbatim
    assert out[6]["content"][0]["content"] == _log_body("t4")  # byte-exact


def test_compression_is_append_only_across_turns() -> None:
    """The cached prefix must never be rewritten as the conversation grows.

    A recency window counted back from the END slides forward, so each block is
    kept verbatim while fresh and digested one turn later — rewriting a message
    the client has by then committed to its cached prefix. Measured against the
    real API that cost every cache read on every turn (2x the cost of sending
    nothing compressed at all), so it is pinned here as an invariant.
    """
    import json as _json

    prev: list[str] | None = None
    for turns in range(1, 6):
        msgs: list[dict] = []
        for i in range(turns):
            msgs.append(_tr(f"t{i}", _log_body(f"t{i}")))
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": "thinking"}]})
        # The client marks history through the last tool_result as cacheable.
        msgs[-2] = _json.loads(_json.dumps(msgs[-2]))
        msgs[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}
        out, _ = compress_messages(msgs)

        def _content_only(m: dict) -> str:
            # The breakpoint marker itself moves each turn by design; what must
            # not move is the content the provider hashed on the previous turn.
            stripped = _json.loads(_json.dumps(m))
            for b in (
                stripped.get("content", []) if isinstance(stripped.get("content"), list) else []
            ):
                if isinstance(b, dict):
                    b.pop("cache_control", None)
            return _json.dumps(stripped, sort_keys=True)

        ser = [_content_only(m) for m in out]
        if prev is not None:
            assert ser[: len(prev)] == prev, (
                f"turn {turns} rewrote a message the previous turn had already "
                "committed to the cached prefix — that invalidates the provider's "
                "cache entry for the whole prefix"
            )
        prev = ser


def test_cached_history_does_not_depend_on_the_current_question() -> None:
    """Query-aware salience must not reach back into content already cached.

    Intent terms come from the newest user turn and change every turn by design,
    so letting them shape an already-sent block rewrites the cached prefix on
    every question — a second, independent cause of the same cache bust the
    recency window caused. Measured before the fix: asking a different follow-up
    rewrote cached message #0.
    """
    import json as _json

    big = "\n".join(f"config value alpha_{i} = {i * 7} beta_{i} = {i * 13}" for i in range(40))

    def convo(last_user: str) -> list[dict]:
        msgs: list[dict] = []
        for i in range(4):
            msgs.append(_tr(f"t{i}", f"{big}\nfile {i}"))
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
        msgs[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}
        msgs.append({"role": "user", "content": last_user})
        return msgs

    a, _ = compress_messages(convo("tell me about alpha_3 and the beta values"))
    b, _ = compress_messages(convo("what is the deployment rollback procedure"))
    assert [_json.dumps(m, sort_keys=True) for m in a[:-1]] == [
        _json.dumps(m, sort_keys=True) for m in b[:-1]
    ]


def test_no_cache_control_keeps_the_sliding_recency_window() -> None:
    """A client that never marks a prefix has no cache to invalidate, so the
    plain last-k carve-out (and its byte-exact freshest output) is preserved."""
    asst = {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]}
    msgs = [
        _tr("t1", _log_body("t1")),
        asst,
        _tr("t2", _log_body("t2")),
        asst,
        _tr("t3", _log_body("t3")),
    ]
    out, _ = compress_messages(msgs)
    assert _has_handle(out[0])  # older → digested
    assert not _has_handle(out[4])  # freshest → byte-exact
    assert out[4]["content"][0]["content"] == _log_body("t3")


def test_recency_is_anchored_to_the_cache_breakpoint() -> None:
    from distil.adapters.anthropic import _RECENCY_KEEP_TURNS, _recent_verbatim_indices

    msgs: list[dict] = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
    ]
    # No breakpoint: the last k turns are exempt, as before.
    assert _recent_verbatim_indices(msgs, _RECENCY_KEEP_TURNS) == {1, 2}
    # Client caches through index 1 -> only the uncached tail stays exempt.
    msgs[1]["content"][0]["cache_control"] = {"type": "ephemeral"}
    assert _recent_verbatim_indices(msgs, _RECENCY_KEEP_TURNS) == {2}
    # Client caches everything -> nothing is exempt; it must all reach the wire
    # in final form, which is what the certified strategy already does.
    msgs[2]["content"][0]["cache_control"] = {"type": "ephemeral"}
    assert _recent_verbatim_indices(msgs, _RECENCY_KEEP_TURNS) == set()


def test_recency_helper_counts_only_user_and_tool_turns() -> None:
    from distil.adapters.anthropic import _RECENCY_KEEP_TURNS, _recent_verbatim_indices

    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},  # never counts toward recency
        {"role": "user", "content": "b"},
        {"role": "tool", "content": "c"},
    ]
    assert _recent_verbatim_indices(msgs, _RECENCY_KEEP_TURNS) == {2, 3}
    assert _recent_verbatim_indices(msgs, 0) == set()


def test_verbatim_never_emits_handles_across_all_turns() -> None:
    """FIX 2: lossless-only maps to verbatim at every proxy; verbatim must yield
    zero Tier-1 handles anywhere — no stub the agent cannot recover."""
    asst = {"role": "assistant", "content": [{"type": "text", "text": "x"}]}
    msgs = [
        _tr("t1", _log_body("t1")),
        asst,
        _tr("t2", _log_body("t2")),
        asst,
        _tr("t3", _log_body("t3")),
    ]
    _out, store = compress_messages(msgs, verbatim=True)
    assert store.handles == frozenset()


def _census_fixture() -> list[dict]:
    """A transcript whose blocks are deliberately claimed by DIFFERENT gates: long
    tool output (digestible, but the last turns are recency-exempt), assistant prose
    (never rewritten), and user text (Tier-0 only)."""
    asst = {"role": "assistant", "content": [{"type": "text", "text": "reasoning. " * 200}]}
    msgs: list[dict] = []
    for turn in range(6):
        msgs.append({"role": "user", "content": "continue please"})
        msgs.append(asst)
        msgs.append(_tr(f"t{turn}", _log_body(f"t{turn}")))
    return msgs


def test_census_accounts_for_the_entire_payload() -> None:
    """The census must be exhaustive, or it is worse than nothing.

    A partial census reads as a complete explanation — the missing tokens are
    invisible, so whatever bucket happens to be largest gets blamed. Every token in
    the request must land in exactly one bucket.
    """
    from distil.adapters.anthropic import take_census
    from distil.proxy import _count_messages

    msgs = _census_fixture()
    compress_messages(msgs, verbatim=False)
    census = take_census()
    assert census is not None

    total = sum(census.values())
    payload = _count_messages(msgs)
    assert abs(total - payload) / payload < 0.005, f"census {total} vs payload {payload}"


def test_census_names_the_gate_that_claimed_each_block() -> None:
    """Distinguishing the gates is the entire point: "mostly assistant prose" is
    working as designed, "the digester declined" is a defect, and "mostly recent" is
    transient. A savings percentage alone cannot tell them apart."""
    from distil.adapters.anthropic import take_census

    compress_messages(_census_fixture(), verbatim=False)
    census = take_census() or {}

    # Assistant prose is never rewritten, so it must be attributed to policy...
    assert census.get("assistant_text", 0) > 0
    # ...the older tool turns are digested...
    assert census.get("tool_result_digested", 0) > 0
    # ...and the freshest ones are held back by the recency carve-out, not by failure.
    assert census.get("tool_result_recent", 0) > 0
    # Nothing reached the digester and came back empty-handed.
    assert census.get("tool_result_declined", 0) == 0


def test_census_is_reopened_per_call_and_never_accumulates() -> None:
    """Two compressions on one thread must not sum. The proxy reads the census after
    the pass, so a stale carry-over would attribute one request's content to another —
    exactly the class of bug where a number from the wrong window is quoted as current."""
    from distil.adapters.anthropic import take_census

    compress_messages(_census_fixture(), verbatim=False)
    first = sum((take_census() or {}).values())
    compress_messages(_census_fixture(), verbatim=False)
    second = sum((take_census() or {}).values())
    assert first == second, f"census accumulated across calls: {first} -> {second}"


def test_census_is_content_free() -> None:
    """Keys are a fixed vocabulary of reasons and values are integers. This record is
    written to disk on every request; anything else here would be a content leak."""
    from distil.adapters.anthropic import take_census

    compress_messages(_census_fixture(), verbatim=False)
    census = take_census() or {}
    assert census
    for key, value in census.items():
        assert isinstance(key, str) and key.replace("_", "").isalnum(), key
        assert isinstance(value, int), (key, value)


def test_census_is_a_no_op_when_none_is_open() -> None:
    """The adapter's helpers are imported and called directly by the OpenAI adapter and
    by tests, outside any ``compress_messages`` call. Counting there would both cost
    tokenizer work nobody asked for and attribute blocks to whichever request happened
    to run last on this thread."""
    import distil.adapters.anthropic as A

    A._census_tls.counts = None
    A._census("assistant_text", "some text that must not be counted")
    assert A.take_census() is None


# --- read-lifecycle exemption -------------------------------------------------
def _read_convo(n_filler_turns: int = 6):
    """A conversation where a file was Read early, then many turns pass."""
    src = "\n".join(f"def f{i}():\n    return {i}" for i in range(40))
    msgs = [
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_read",
                    "name": "Read",
                    "input": {"file_path": "/a.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_read", "content": src}],
        },
    ]
    for i in range(n_filler_turns):
        msgs += [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"tu_b{i}",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tu_b{i}",
                        "content": "\n".join(f"line {j} of routine log output" for j in range(40)),
                    }
                ],
            },
        ]
    return msgs, src


def test_file_read_stays_byte_exact_however_old():
    """A Read result must survive verbatim so Edit(old_string=...) can still match.

    Recency is positional, so a read from many turns ago used to be digestible —
    but the agent must still reproduce that text character-for-character to edit
    the file. Digesting it makes every subsequent edit unmatchable, which is how a
    run ends with the agent reporting success and nothing written to disk.
    """
    from distil.adapters.anthropic import compress_messages

    msgs, src = _read_convo()
    out, _store = compress_messages(msgs, verbatim=False)
    read_result = out[2]["content"][0]["content"]
    assert read_result == src, "file content must not be digested"
    assert "def f7():" in read_result, "exact-match anchor must survive"


def test_non_read_tool_output_still_digests():
    """The exemption is provenance-scoped: ordinary tool output still compresses."""
    from distil.adapters.anthropic import compress_messages

    msgs, _src = _read_convo()
    out, _store = compress_messages(msgs, verbatim=False)
    # The oldest Bash result is far outside the recency window and must be digested.
    bash_results = [
        blk["content"]
        for m in out
        for blk in (m.get("content") or [])
        if isinstance(blk, dict) and str(blk.get("tool_use_id", "")).startswith("tu_b")
    ]
    assert any("handle=" in r for r in bash_results), "log output should still digest"


# --- extended thinking --------------------------------------------------------
def test_thinking_blocks_are_counted_but_never_rewritten():
    """Thinking is billed on Claude 4.6+ and cannot be compressed — so it must at
    least be VISIBLE.

    The payload lives under `thinking`, not `text`, so it was counted by neither the
    before nor the after side: real billed tokens absent from every percentage distil
    reports. It stays byte-identical (the provider pins the block by signature and
    re-expands it server-side, so editing it achieves nothing and risks rejection on
    replay), but it is now censused.
    """
    from distil.adapters.anthropic import compress_messages, take_census
    from distil.proxy import _count_messages

    block = {"type": "thinking", "thinking": "deliberating " * 200, "signature": "sig"}
    msgs = [{"role": "assistant", "content": [block]}]

    assert _count_messages(msgs) > 0, "billed thinking must appear in the baseline"

    out, _store = compress_messages(msgs, verbatim=False)
    assert out[0]["content"][0] == block, "thinking must never be rewritten"
    assert (take_census() or {}).get("thinking_billed", 0) > 0


def test_redacted_thinking_is_also_counted_and_preserved():
    from distil.adapters.anthropic import compress_messages
    from distil.proxy import _count_messages

    block = {"type": "redacted_thinking", "data": "opaque", "redacted_thinking": "r" * 400}
    msgs = [{"role": "assistant", "content": [block]}]
    assert _count_messages(msgs) > 0
    out, _store = compress_messages(msgs, verbatim=False)
    assert out[0]["content"][0] == block

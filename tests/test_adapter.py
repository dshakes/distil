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

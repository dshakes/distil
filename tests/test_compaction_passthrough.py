"""Contract tests: Anthropic server-side compaction survives distil untouched.

Anthropic's Messages API compacts context server-side — threshold (beta
``compact-2026-01-12``, ``context_management.edits[].type: "compact_20260112"``) and
on-demand (beta ``compact-2026-09-04``). The response carries a ``compaction`` content
block with a ``signature`` the provider re-validates on replay: alter the block, move
it, or even re-encode it losslessly, and the NEXT request 400s with
``compaction_signature_invalid``. Every layer distil's Anthropic path runs a request
through — ``compress_messages`` (digest/recency/provenance/rereaddelta), the SDK
``wrap()`` adapter, the HTTP proxy (headers + ``context_management``), and the
streaming splice (``streamexpand``) — must leave that block, its ``signature``, the
``context_management`` request field, and the ``anthropic-beta`` header byte-equivalent.

Two more invariants this suite pins:
  * content AFTER a compaction block (same message or a later one) must still compress
    normally — a compaction block must not accidentally halt or exempt what follows it.
  * a compaction block's own payload (however it is shaped) must never be mistaken for
    a digestible ``tool_result`` — dispatch is by ``type``, not by duck-typing a
    ``content``/``text`` key.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from distil.adapters.anthropic import (
    _RECENCY_KEEP_TURNS,
    _cached_prefix_end,
    _recent_verbatim_indices,
    compress_messages,
    take_census,
    wrap,
)
from distil.proxy import _count_messages, build_handler
from distil.streamexpand import stream_with_expand
from tests.test_streamexpand import _Handler, _Resp, _Store, _evt, _sender

# ---------------------------------------------------------------------------
# Fixtures
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
    ]
)  # 10 lines — well above the 6-line digest threshold

# Two trailing pad turns push a tool_result out of the recency-exempt window
# (_RECENCY_KEEP_TURNS = 2) so it is eligible for Tier-1 digestion in these tests.
_PAD = [{"role": "user", "content": "next"}, {"role": "user", "content": "next"}]


def _compaction_block(signature: str = "sig-abc123==", **extra: Any) -> dict[str, Any]:
    """A minimal but realistic ``compaction`` content block."""
    return {
        "type": "compaction",
        "content": "Summary of the compacted turns: read 3 files, ran 2 tests, all passed.",
        "signature": signature,
        **extra,
    }


def _tool_result(text: str, tool_use_id: str = "toolu_01") -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


def _has_handle(msg: dict[str, Any]) -> bool:
    content = msg["content"][0]["content"]
    return isinstance(content, str) and "handle=" in content


# ---------------------------------------------------------------------------
# compress_messages: the compaction block itself is byte-equivalent
# ---------------------------------------------------------------------------


class TestCompactionBlockImmutable:
    def test_passthrough_identical_object_verbatim_mode(self) -> None:
        block = _compaction_block()
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        out, _store = compress_messages(msgs, verbatim=True)
        assert out[0]["content"][0] is block  # not merely equal — untouched

    def test_passthrough_identical_object_digest_mode(self) -> None:
        block = _compaction_block()
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        out, _store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0] is block

    def test_signature_bytes_exact_including_unicode(self) -> None:
        # A signature is opaque provider bytes; anything distil does to it (case-fold,
        # whitespace-normalise, base64 re-encode) breaks the provider's own check.
        sig = "AQIDsig==é☃\n\t"
        block = _compaction_block(signature=sig)
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        out, _store = compress_messages(msgs, verbatim=False)
        got = out[0]["content"][0]
        assert got["signature"] == sig
        assert json.dumps(got, sort_keys=True) == json.dumps(block, sort_keys=True)

    def test_not_recorded_in_restore_store(self) -> None:
        # A compaction block must never become a digest handle — it has no marker to
        # digest FROM (it is not a tool_result) and nothing must ever claim to "recover"
        # provider-pinned bytes via distil_expand.
        block = _compaction_block()
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        _out, store = compress_messages(msgs, verbatim=False)
        assert store.handles == frozenset()

    def test_generic_signed_block_fallback_for_unknown_future_type(self) -> None:
        # Not every provider-signed opaque block will be named "compaction" forever —
        # the guard is keyed on the presence of a `signature`, not an allowlist of
        # type strings, so a new signed type is safe by construction.
        block = {"type": "future_signed_block", "content": "x" * 500, "signature": "s"}
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        out, _store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0] is block

    def test_generic_signed_block_census_bucket_is_signed_block_billed(self) -> None:
        # Distinct from "compaction_billed": an unknown signed type is real provider
        # cost too, just not one distil has a dedicated name for yet.
        block = {"type": "future_signed_block", "content": "x" * 500, "signature": "s"}
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        compress_messages(msgs, verbatim=False)
        census = take_census() or {}
        assert census.get("signed_block_billed", 0) > 0
        assert census.get("compaction_billed", 0) == 0

    def test_tool_result_with_stray_signature_key_is_compressed_normally(self) -> None:
        # A `signature` key is not exclusive to provider-opaque blocks — a tool_result
        # could carry one incidentally (e.g. an upstream that stamps every block). It
        # must still go through tool_result's own handling (digestion, exact-quote
        # exemption), not be swallowed by the opaque-block fallback: that would skip
        # compression AND drop it from the digest census silently.
        signed_result = {**_tool_result(LONG_TOOL_RESULT), "signature": "not-actually-opaque"}
        msgs = [{"role": "user", "content": [signed_result]}, *_PAD]
        out, store = compress_messages(msgs, verbatim=False)
        digested = out[0]["content"][0]["content"]
        assert "handle=" in digested
        assert store.handles
        census = take_census() or {}
        assert census.get("signed_block_billed", 0) == 0


# ---------------------------------------------------------------------------
# Content around a compaction block is still compressible
# ---------------------------------------------------------------------------


class TestSurroundingContentStillCompresses:
    def test_tool_result_after_compaction_in_same_message_still_digests(self) -> None:
        msg = {
            "role": "user",
            "content": [_compaction_block(), _tool_result(LONG_TOOL_RESULT)],
        }
        msgs = [msg, *_PAD]
        out, store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0] == _compaction_block()  # compaction untouched
        digested = out[0]["content"][1]["content"]
        assert "handle=" in digested
        assert store.handles  # the tool_result really was recorded

    def test_tool_result_before_compaction_in_same_message_still_digests(self) -> None:
        msg = {
            "role": "user",
            "content": [_tool_result(LONG_TOOL_RESULT), _compaction_block()],
        }
        msgs = [msg, *_PAD]
        out, _store = compress_messages(msgs, verbatim=False)
        assert "handle=" in out[0]["content"][0]["content"]
        assert out[0]["content"][1] == _compaction_block()

    def test_tool_result_in_later_message_still_digests(self) -> None:
        compaction_msg = {"role": "assistant", "content": [_compaction_block()]}
        msgs = [compaction_msg, _tool_result(LONG_TOOL_RESULT), *_PAD]
        # _tool_result() is bare content; wrap it in a user message like the real shape.
        msgs = [
            compaction_msg,
            {"role": "user", "content": [_tool_result(LONG_TOOL_RESULT)]},
            *_PAD,
        ]
        out, _store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0] is compaction_msg["content"][0]
        assert "handle=" in out[1]["content"][0]["content"]


# ---------------------------------------------------------------------------
# A compaction block's own content is never digested as if it were a tool_result
# ---------------------------------------------------------------------------


class TestCompactionContentNotDigestedAsToolResult:
    def test_large_inline_content_field_not_folded_or_digested(self) -> None:
        # Shaped to trip both gates a tool_result would trip: >= 6 lines AND a JSON
        # array a columnar fold would normally flatten. Dispatch is by `type`, so
        # neither must ever fire here.
        big = "\n".join(f"compacted turn {i}: did something" for i in range(20))
        block = _compaction_block()
        block["content"] = big
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        out, store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0] is block
        assert out[0]["content"][0]["content"] == big
        assert store.handles == frozenset()

    def test_nested_list_shaped_content_not_digested(self) -> None:
        # The real wire shape may nest text parts under `content` rather than a bare
        # string. Either way the top-level dispatch never even looks inside it.
        block = _compaction_block()
        block["content"] = [{"type": "text", "text": "compacted " * 200}]
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        out, store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0] is block
        assert store.handles == frozenset()


# ---------------------------------------------------------------------------
# Census: compaction is billed and counted (mirrors thinking/redacted_thinking)
# ---------------------------------------------------------------------------


class TestCompactionCensus:
    def test_compaction_tokens_are_censused_not_invisible(self) -> None:
        block = _compaction_block()
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        assert _count_messages(msgs) > 0
        compress_messages(msgs, verbatim=False)
        census = take_census() or {}
        assert census.get("compaction_billed", 0) > 0

    def test_baseline_and_census_agree_on_compaction_tokens(self) -> None:
        block = _compaction_block()
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        baseline = _count_messages(msgs)
        compress_messages(msgs, verbatim=False)
        censused = (take_census() or {}).get("compaction_billed", 0)
        # baseline also includes the two pad user turns' text; isolate the compaction
        # contribution by diffing against a census of the pad alone.
        pad_only_baseline = _count_messages(_PAD)
        assert baseline - pad_only_baseline == censused


# ---------------------------------------------------------------------------
# Prompt-cache breakpoint logic with a compaction block at message start
# ---------------------------------------------------------------------------


class TestCacheBreakpointWithCompactionAtStart:
    def test_cached_prefix_end_does_not_crash_and_finds_the_marker(self) -> None:
        msgs: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [{**_compaction_block(), "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
        ]
        assert _cached_prefix_end(msgs) == 0

    def test_recency_window_still_exempts_the_uncached_tail(self) -> None:
        # A compaction block bearing the cache breakpoint at index 0 must behave like
        # any other cache_control marker: only what follows it is recency-exempt.
        msgs: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [{**_compaction_block(), "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ]
        assert _recent_verbatim_indices(msgs, _RECENCY_KEEP_TURNS) == {1, 2}

    def test_tool_result_after_a_compaction_cache_breakpoint_still_digests(self) -> None:
        msgs: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [{**_compaction_block(), "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": [_tool_result(LONG_TOOL_RESULT)]},
            *_PAD,
        ]
        out, store = compress_messages(msgs, verbatim=False)
        assert out[0]["content"][0]["type"] == "compaction"
        assert store.handles  # the later tool_result still digested normally


# ---------------------------------------------------------------------------
# SDK wrap(): context_management / betas kwargs pass through untouched
# ---------------------------------------------------------------------------


class TestWrapPassesThroughCompactionKwargs:
    def _make_fake_client(self) -> Any:
        calls: list[dict] = []

        class FakeMessages:
            def create(self, **kwargs: Any) -> Any:
                calls.append(kwargs)
                return {"id": "msg_fake", "content": []}

        class FakeClient:
            def __init__(self) -> None:
                self.messages = FakeMessages()
                self._calls = calls

        return FakeClient()

    def test_context_management_and_betas_kwargs_untouched(self) -> None:
        fake = self._make_fake_client()
        client = wrap(fake)
        cm = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            context_management=cm,
            betas=["compact-2026-01-12"],
            extra_headers={"anthropic-beta": "compact-2026-01-12"},
        )
        sent = fake._calls[0]  # type: ignore[attr-defined]
        assert sent["context_management"] == cm
        assert sent["context_management"] is cm
        assert sent["betas"] == ["compact-2026-01-12"]
        assert sent["extra_headers"] == {"anthropic-beta": "compact-2026-01-12"}

    def test_compaction_block_in_history_untouched_through_wrap(self) -> None:
        fake = self._make_fake_client()
        client = wrap(fake)
        block = _compaction_block()
        msgs = [{"role": "assistant", "content": [block]}, *_PAD]
        client.messages.create(model="claude-opus-4-5", max_tokens=1024, messages=msgs)
        received = fake._calls[0]["messages"]  # type: ignore[attr-defined]
        assert received[0]["content"][0] == block


# ---------------------------------------------------------------------------
# Proxy: context_management field + anthropic-beta header survive the wire
# ---------------------------------------------------------------------------


class _HeaderCapturingEcho(BaseHTTPRequestHandler):
    """Fake upstream: echoes the POST body back, and mirrors selected request headers
    into response headers (prefixed) so the test can assert on what actually left the
    proxy — not just what the proxy's own JSON logic produced."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name in ("anthropic-beta", "x-api-key"):
            value = self.headers.get(name)
            if value is not None:
                self.send_header(f"x-echo-{name}", value)
        self.end_headers()
        self.wfile.write(body)


def test_proxy_preserves_context_management_and_beta_header_alongside_real_compression() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _HeaderCapturingEcho)
    up_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    up_thread.start()
    up_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    handler_cls = build_handler(up_url)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        context_management = {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {}}]}
        block = _compaction_block()
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 256,
            "context_management": context_management,
            "messages": [
                {"role": "assistant", "content": [block]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": LONG_TOOL_RESULT}
                    ],
                },
                *_PAD,
            ],
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "anthropic-beta": "compact-2026-01-12,compact-2026-09-04",
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            saved_hdr = resp.headers.get("x-distil-tokens-saved")
            echoed_beta = resp.headers.get("x-echo-anthropic-beta")
            echoed_body: dict[str, Any] = json.loads(resp.read())
    finally:
        proxy.shutdown()
        upstream.shutdown()

    # Real compression still happened on the surrounding tool_result.
    assert saved_hdr is not None and int(saved_hdr) > 0
    # The anthropic-beta header reached upstream byte-identical.
    assert echoed_beta == "compact-2026-01-12,compact-2026-09-04"
    # context_management reached upstream byte-identical (not reordered/rewritten).
    assert echoed_body["context_management"] == context_management
    # The compaction block itself reached upstream byte-identical.
    assert echoed_body["messages"][0]["content"][0] == block


# ---------------------------------------------------------------------------
# Streaming: a compaction block survives the distil_expand splice untouched
# ---------------------------------------------------------------------------


def _compaction_stream_frame(index: int, block: dict[str, Any]) -> bytes:
    return b"".join(
        [
            _evt(type="content_block_start", index=index, content_block=block),
            _evt(type="content_block_stop", index=index),
        ]
    )


def _plain_response_with_compaction(block: dict[str, Any]) -> bytes:
    return b"".join(
        [
            _evt(
                type="message_start",
                message={
                    "id": "msg_c",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 50, "output_tokens": 1},
                },
            ),
            _compaction_stream_frame(0, block),
            _evt(
                type="message_delta",
                delta={"stop_reason": "end_turn"},
                usage={"output_tokens": 3},
            ),
            _evt(type="message_stop"),
        ]
    )


def _expand_response_with_leading_compaction(handle: str, block: dict[str, Any]) -> bytes:
    """compaction block at index 0, distil_expand tool_use at index 1 — the shape that
    would exercise `_accumulate`'s reconstruction if the block arrived via deltas."""
    return b"".join(
        [
            _evt(
                type="message_start",
                message={
                    "id": "msg_x",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 50, "output_tokens": 1},
                },
            ),
            _compaction_stream_frame(0, block),
            _evt(
                type="content_block_start",
                index=1,
                content_block={
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "distil_expand",
                    "input": {},
                },
            ),
            _evt(
                type="content_block_delta",
                index=1,
                delta={"type": "input_json_delta", "partial_json": json.dumps({"handle": handle})},
            ),
            _evt(type="content_block_stop", index=1),
            _evt(
                type="message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 5}
            ),
            _evt(type="message_stop"),
        ]
    )


def _text_response(text: str) -> bytes:
    return b"".join(
        [
            _evt(
                type="message_start",
                message={
                    "id": "msg_a",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            ),
            _evt(type="content_block_start", index=0, content_block={"type": "text", "text": ""}),
            _evt(type="content_block_delta", index=0, delta={"type": "text_delta", "text": text}),
            _evt(type="content_block_stop", index=0),
            _evt(
                type="message_delta",
                delta={"stop_reason": "end_turn"},
                usage={"output_tokens": 4},
            ),
            _evt(type="message_stop"),
        ]
    )


def test_streaming_relays_compaction_block_untouched_when_no_expand() -> None:
    block = _compaction_block()
    send, _bodies = _sender([_Resp(_plain_response_with_compaction(block))])
    h = _Handler()
    st = stream_with_expand(
        h, send, {"messages": [{"role": "user", "content": "hi"}]}, _Store(), hop_by_hop=frozenset()
    )
    out = bytes(h.wfile.buf).decode()
    assert st == 200
    # The compaction block reaches the client exactly as the upstream sent it.
    assert json.dumps(block) in out or json.dumps(block, separators=(",", ": ")) in out
    start_frame = next(
        line for line in out.split("\n\n") if '"type": "content_block_start"' in line
    )
    payload = json.loads(start_frame.split("data: ", 1)[1])
    assert payload["content_block"] == block
    assert payload["index"] == 0  # unchanged: nothing suppressed before it


def test_streaming_splice_forwards_compaction_block_byte_identical_in_requery() -> None:
    # compaction (index 0) precedes the distil_expand tool_use (index 1) in one turn.
    # The re-query must resend the compaction block to upstream exactly as received —
    # this is the literal "clients send it back in later requests" contract.
    block = _compaction_block()
    store = _Store()
    send, bodies = _sender(
        [
            _Resp(_expand_response_with_leading_compaction("h1", block)),
            _Resp(_text_response("resumed")),
        ]
    )
    h = _Handler()
    st = stream_with_expand(
        h, send, {"messages": [{"role": "user", "content": "go"}]}, store, hop_by_hop=frozenset()
    )
    assert st == 200
    assert store.calls == ["h1"]
    requery = bodies[1]
    assistant_turn = requery["messages"][-2]
    assert assistant_turn["role"] == "assistant"
    compaction_sent_back = next(
        b for b in assistant_turn["content"] if b.get("type") == "compaction"
    )
    assert compaction_sent_back == block

    # The client also saw the compaction block untouched before the splice occurred.
    out = bytes(h.wfile.buf).decode()
    start_frame = next(
        line for line in out.split("\n\n") if '"type": "content_block_start"' in line
    )
    payload = json.loads(start_frame.split("data: ", 1)[1])
    assert payload["content_block"] == block

"""Tests for distil.adapters.openai — Chat Completions and Responses API adapters.

Covered:
* Chat Completions: role:"tool" string content → Tier-1 reversible digest
* Chat Completions: role:"tool" list content (text parts) → Tier-1 per part
* Chat Completions: role:"assistant" (with tool_calls) → passthrough unchanged
* Chat Completions: role:"system" → passthrough (Tier-0 at most)
* Chat Completions: recency carve-out — last K tool turns stay verbatim
* Chat Completions: reject-if-bigger — short content never inflated
* Chat Completions: fail-open on malformed entries
* Chat Completions: RestoreStore round-trip (handle in digest → expand → original)
* Responses API: function_call_output.output → Tier-1 reversible digest
* Responses API: message/user input_text → Tier-0 lossless transforms
* Responses API: message/assistant output_text → passthrough
* Responses API: function_call → passthrough
* Responses API: recency carve-out — last K function_call_output items verbatim
* Responses API: fail-open on non-string output field
* Responses API: RestoreStore round-trip
* Proxy dispatch: /v1/chat/completions uses OpenAI adapter (different from /v1/messages)
* Proxy dispatch: /v1/responses compresses function_call_output items
* Proxy dispatch: /v1/messages still uses Anthropic adapter (unchanged)

No real model or network required.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


from distil.adapters.openai import (
    compress_chat_completions,
    compress_responses_input,
    count_responses_tokens,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Minimum lines for Tier-1 digest to kick in (matches _MIN_LINES = 6 in anthropic.py)
_MIN_LINES = 6


def _big_tool_output(tag: str = "tool") -> str:
    """Return a string long enough (>= _MIN_LINES lines) to trigger Tier-1 digestion."""
    return "\n".join(
        [
            f"=== {tag} result ===",
            "total disk usage: 48 GB across 12 partitions",
            "filesystem /dev/sda1: 32 GB used of 100 GB available",
            "filesystem /dev/sdb1: 16 GB used of 200 GB available",
            "warning: /tmp is 89% full — consider cleaning up old build artefacts",
            "warning: inode count on /var/log approaching limit (91% used)",
            "no errors detected in kernel ring buffer",
            "last boot: 2026-06-20T03:14:22Z (uptime 18h 42m)",
        ]
    )  # 8 lines — safely above the 6-line digest threshold


def _has_handle(text: str) -> bool:
    """True if `text` contains a distil digest handle marker."""
    import re

    return bool(re.search(r"handle=[0-9a-fA-F]{6,}", text))


# ---------------------------------------------------------------------------
# Chat Completions: compress_chat_completions
# ---------------------------------------------------------------------------


class TestChatCompletionsToolStringContent:
    """role:"tool" with bare string content must be Tier-1 digested."""

    def test_large_tool_message_is_digested(self) -> None:
        output = _big_tool_output()
        # Add two extra user turns so the tool message falls outside the recency window
        # (RECENCY_KEEP_TURNS = 2, which exempts the last 2 user/tool messages).
        msgs = [
            {"role": "tool", "tool_call_id": "call_1", "content": output},
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, store = compress_chat_completions(msgs)
        result = compressed[0]["content"]
        assert _has_handle(result), f"Expected digest handle in output, got: {result!r}"
        assert len(result) < len(output), "Compressed content should be shorter than original"
        assert store.handles, "Store should have at least one handle"

    def test_restore_store_round_trip(self) -> None:
        output = _big_tool_output()
        msgs = [
            {"role": "tool", "tool_call_id": "call_1", "content": output},
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, store = compress_chat_completions(msgs)
        handle = next(iter(store.handles))
        recovered = store.expand(handle)
        assert recovered == output, "Expanded handle must return original byte-for-byte"

    def test_small_tool_message_not_digested(self) -> None:
        # Under _MIN_LINES → Tier-0 only, never a digest handle
        small = "just a tiny result"
        msgs = [{"role": "tool", "tool_call_id": "call_1", "content": small}]
        compressed, store = compress_chat_completions(msgs)
        assert not _has_handle(compressed[0]["content"])
        assert not store.handles


class TestChatCompletionsToolListContent:
    """role:"tool" with list of text parts — the OpenAI-specific case.
    Each text part must be Tier-1 digested, not Tier-0 as user text would be."""

    def test_tool_list_content_text_part_digested(self) -> None:
        output = _big_tool_output("list-tool")
        # Two extra turns to push the tool message outside the recency window.
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [{"type": "text", "text": output}],
            },
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, store = compress_chat_completions(msgs)
        part = compressed[0]["content"][0]
        assert _has_handle(part["text"]), (
            f"Expected digest handle in text part, got: {part['text']!r}"
        )
        assert store.handles

    def test_restore_from_list_content(self) -> None:
        output = _big_tool_output("restore-list")
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [{"type": "text", "text": output}],
            },
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, store = compress_chat_completions(msgs)
        handle = next(iter(store.handles))
        assert store.expand(handle) == output

    def test_non_text_parts_passed_through(self) -> None:
        # image_url parts in a tool message should pass through unchanged
        image_part: dict[str, Any] = {"type": "image_url", "image_url": {"url": "http://x/img"}}
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "content": [image_part],
            }
        ]
        compressed, store = compress_chat_completions(msgs)
        assert compressed[0]["content"][0] == image_part
        assert not store.handles


class TestChatCompletionsAssistantPassthrough:
    """role:"assistant" messages — including tool_calls — must never be modified."""

    def test_assistant_string_content_passthrough(self) -> None:
        big = _big_tool_output("assistant")
        msgs = [{"role": "assistant", "content": big}]
        compressed, store = compress_chat_completions(msgs)
        assert compressed[0]["content"] == big
        assert not store.handles

    def test_assistant_with_tool_calls_passthrough(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                }
            ],
        }
        compressed, store = compress_chat_completions([msg])
        assert compressed[0] == msg
        assert not store.handles

    def test_assistant_list_content_passthrough(self) -> None:
        big = _big_tool_output("asst-list")
        msg = {"role": "assistant", "content": [{"type": "text", "text": big}]}
        compressed, store = compress_chat_completions([msg])
        assert compressed[0]["content"][0]["text"] == big
        assert not store.handles


class TestChatCompletionsSystemPassthrough:
    """role:"system" is not a tool result — receives Tier-0 at most, never Tier-1."""

    def test_system_content_not_digested(self) -> None:
        sys_msg = {"role": "system", "content": _big_tool_output("system")}
        compressed, store = compress_chat_completions([sys_msg])
        # System messages use _compress_text_content (Tier-0 only) — no handle
        assert not _has_handle(compressed[0]["content"])
        assert not store.handles


class TestChatCompletionsRecency:
    """Last RECENCY_KEEP_TURNS tool-bearing messages must stay verbatim."""

    def test_tool_message_is_digested_even_when_freshest(self) -> None:
        """OpenAI caches prefixes implicitly, so there is no recency carve-out.

        Keeping the freshest tool message verbatim only defers the rewrite by one
        turn — and by then OpenAI has cached it, so digesting it later invalidates
        the whole cached prefix. Digesting on first sight keeps every later request
        byte-identical.
        """
        output = _big_tool_output("recent")
        msgs = [
            {"role": "tool", "tool_call_id": "call_1", "content": output},
        ]
        compressed, store = compress_chat_completions(msgs)
        assert _has_handle(compressed[0]["content"])
        assert store.handles

    def test_old_tool_message_outside_recency_is_digested(self) -> None:
        output = _big_tool_output("old")
        # Three tool messages; only the last 2 are in the recency window (RECENCY_KEEP_TURNS=2).
        # The first one must be digested.
        msgs = [
            {"role": "tool", "tool_call_id": "call_0", "content": output},
            {"role": "user", "content": "follow-up 1"},
            {"role": "user", "content": "follow-up 2"},
            {"role": "tool", "tool_call_id": "call_1", "content": _big_tool_output("mid")},
            {"role": "tool", "tool_call_id": "call_2", "content": _big_tool_output("newest")},
        ]
        compressed, store = compress_chat_completions(msgs)
        # First tool message (index 0) is outside the recency window → must be digested
        assert _has_handle(compressed[0]["content"]), (
            "Old tool message outside recency window should be digested"
        )
        assert store.handles

    def test_recency_verbatim_mode_unchanged(self) -> None:
        """In verbatim mode, recent messages are verbatim by virtue of verbatim=True."""
        output = _big_tool_output("verbatim-recent")
        msgs = [
            {"role": "tool", "tool_call_id": "call_0", "content": output},
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, store = compress_chat_completions(msgs, verbatim=True)
        assert not _has_handle(compressed[0]["content"])
        assert not store.handles


class TestChatCompletionsRejectIfBigger:
    """Compression must never inflate content (reject-if-bigger invariant)."""

    def test_short_content_not_inflated(self) -> None:
        # A 3-line tool result is below _MIN_LINES → Tier-0 only (whitespace collapse),
        # and should never be longer than the input.
        short = "line 1\nline 2\nline 3"
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": short}]
        compressed, store = compress_chat_completions(msgs)
        result = compressed[0]["content"]
        assert len(result) <= len(short), (
            f"Compressed content ({len(result)}) must not exceed original ({len(short)})"
        )
        assert not store.handles

    def test_input_not_mutated(self) -> None:
        output = _big_tool_output("mutation-check")
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": output}]
        original_content = msgs[0]["content"]
        compress_chat_completions(msgs)
        assert msgs[0]["content"] == original_content, "Input list must not be mutated"


class TestChatCompletionsFailOpen:
    """Malformed entries in the messages list must pass through without error."""

    def test_non_dict_entry_passes_through(self) -> None:
        # A bare string instead of a dict — should be forwarded untouched.
        msgs: list[Any] = [
            "not-a-dict",
            {"role": "tool", "tool_call_id": "c1", "content": _big_tool_output()},
        ]
        compressed, store = compress_chat_completions(msgs)
        assert compressed[0] == "not-a-dict"

    def test_none_content_passes_through(self) -> None:
        # Tool message with content=None (valid in some OpenAI shapes)
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": None}]
        compressed, store = compress_chat_completions(msgs)
        assert compressed[0]["content"] is None

    def test_missing_content_key_passes_through(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "c1"}]
        compressed, _ = compress_chat_completions(msgs)
        assert "content" not in compressed[0]


class TestChatCompletionsMixedRoles:
    """Integration: mixed-role conversation — only tool content gets digested."""

    def test_only_old_tool_content_digested(self) -> None:
        output = _big_tool_output("mixed")
        msgs = [
            {"role": "system", "content": _big_tool_output("sys")},
            {"role": "user", "content": _big_tool_output("user")},
            {"role": "assistant", "content": _big_tool_output("asst")},
            {"role": "tool", "tool_call_id": "c1", "content": output},  # index 3
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, store = compress_chat_completions(msgs)
        # system: Tier-0 only
        assert not _has_handle(compressed[0]["content"])
        # user: Tier-0 only
        assert not _has_handle(compressed[1]["content"])
        # assistant: passthrough
        assert compressed[2]["content"] == msgs[2]["content"]
        # tool at index 3: outside recency window (two more user messages after it) → digested
        assert _has_handle(compressed[3]["content"])


# ---------------------------------------------------------------------------
# Responses API: compress_responses_input
# ---------------------------------------------------------------------------


class TestResponsesFunctionCallOutput:
    """function_call_output items: output field → Tier-1 reversible digest."""

    def test_large_output_is_digested(self) -> None:
        output = _big_tool_output("fn-call-out")
        # Two extra function_call_output items push call_1 outside the recency window.
        items = [
            {"type": "function_call_output", "call_id": "call_1", "output": output},
            {"type": "function_call_output", "call_id": "c1", "output": _big_tool_output("c1")},
            {"type": "function_call_output", "call_id": "c2", "output": _big_tool_output("c2")},
        ]
        compressed, store = compress_responses_input(items)
        result = compressed[0]["output"]
        assert _has_handle(result), f"Expected digest handle in output, got: {result!r}"
        assert len(result) < len(output)
        assert store.handles

    def test_restore_store_round_trip(self) -> None:
        output = _big_tool_output("fn-restore")
        items = [
            {"type": "function_call_output", "call_id": "call_1", "output": output},
            {"type": "function_call_output", "call_id": "c1", "output": _big_tool_output("c1")},
            {"type": "function_call_output", "call_id": "c2", "output": _big_tool_output("c2")},
        ]
        compressed, store = compress_responses_input(items)
        # Every item digests now that implicit prefix caching leaves no recency
        # carve-out, so the store holds a handle per item — take the one belonging
        # to the block under test rather than an arbitrary member of the set.
        handle = re.search(r"handle=([0-9a-f]{8})", compressed[0]["output"]).group(1)
        assert store.expand(handle) == output

    def test_non_string_output_passthrough(self) -> None:
        # output field is not a string (unusual but must be fail-open)
        items = [{"type": "function_call_output", "call_id": "c1", "output": {"key": "val"}}]
        compressed, store = compress_responses_input(items)
        assert compressed[0]["output"] == {"key": "val"}
        assert not store.handles

    def test_missing_output_key_passthrough(self) -> None:
        items = [{"type": "function_call_output", "call_id": "c1"}]
        compressed, _ = compress_responses_input(items)
        assert "output" not in compressed[0]


class TestResponsesMessageItems:
    """message items: user input_text → Tier-0; assistant output_text → passthrough."""

    def test_user_input_text_tier0(self) -> None:
        # Tier-0 transforms (JSON minification, collapse): no handle emitted
        big_json = json.dumps({"key": "val", "data": "x" * 200}) + "\n" * 10
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": big_json}],
            }
        ]
        compressed, store = compress_responses_input(items)
        # No handle (Tier-0 only for user text)
        assert not _has_handle(compressed[0]["content"][0]["text"])
        assert not store.handles

    def test_assistant_output_text_passthrough(self) -> None:
        big = _big_tool_output("asst-resp")
        items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": big}],
            }
        ]
        compressed, store = compress_responses_input(items)
        assert compressed[0]["content"][0]["text"] == big
        assert not store.handles

    def test_non_input_text_parts_passthrough(self) -> None:
        part: dict[str, Any] = {"type": "image_file", "file_id": "file_123"}
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [part],
            }
        ]
        compressed, _ = compress_responses_input(items)
        assert compressed[0]["content"][0] == part


class TestResponsesFunctionCallPassthrough:
    """function_call items are model outputs — must pass through unchanged."""

    def test_function_call_passthrough(self) -> None:
        fn_call = {
            "type": "function_call",
            "id": "call_abc",
            "name": "bash",
            "arguments": '{"cmd": "ls -la"}',
        }
        items = [fn_call]
        compressed, store = compress_responses_input(items)
        assert compressed[0] == fn_call
        assert not store.handles


class TestResponsesRecency:
    """Last RECENCY_KEEP_TURNS function_call_output items must stay verbatim."""

    def test_function_call_output_is_digested_even_when_freshest(self) -> None:
        """No recency carve-out on the Responses path either — same cache reason."""
        output = _big_tool_output("recent-resp")
        items = [{"type": "function_call_output", "call_id": "c1", "output": output}]
        compressed, store = compress_responses_input(items)
        assert _has_handle(compressed[0]["output"])
        assert store.handles

    def test_old_function_call_output_digested(self) -> None:
        output = _big_tool_output("old-resp")
        # Three function_call_output items; first one is outside recency window.
        items = [
            {"type": "function_call_output", "call_id": "c0", "output": output},
            {"type": "function_call_output", "call_id": "c1", "output": _big_tool_output("mid")},
            {"type": "function_call_output", "call_id": "c2", "output": _big_tool_output("new")},
        ]
        compressed, store = compress_responses_input(items)
        assert _has_handle(compressed[0]["output"]), (
            "Old function_call_output outside recency window should be digested"
        )
        assert store.handles

    def test_compression_is_stable_as_items_are_appended(self) -> None:
        """Appending items must never change how an earlier item compresses.

        The cache-safety invariant: OpenAI caches the prefix it was sent, so an
        item whose bytes change between requests invalidates that entry.
        """
        import json

        items = [
            {"type": "function_call_output", "call_id": "c0", "output": _big_tool_output("a")},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ok"}]},
            {"type": "function_call_output", "call_id": "c1", "output": _big_tool_output("b")},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {"type": "function_call_output", "call_id": "c2", "output": _big_tool_output("c")},
        ]
        prev = None
        for n in range(1, len(items) + 1):
            compressed, _ = compress_responses_input(items[:n])
            ser = [json.dumps(i, sort_keys=True) for i in compressed]
            if prev is not None:
                assert ser[: len(prev)] == prev, (
                    f"appending item {n} rewrote an earlier item OpenAI had already cached"
                )
            prev = ser


class TestResponsesFailOpen:
    """Malformed entries must pass through without raising."""

    def test_non_dict_entry_passthrough(self) -> None:
        items: list[Any] = [
            "not-a-dict",
            {"type": "function_call_output", "call_id": "c1", "output": _big_tool_output()},
        ]
        compressed, _ = compress_responses_input(items)
        assert compressed[0] == "not-a-dict"

    def test_unknown_type_passthrough(self) -> None:
        item = {"type": "computer_call_output", "output": _big_tool_output("cpu")}
        compressed, _ = compress_responses_input([item])
        assert compressed[0] == item

    def test_input_not_mutated(self) -> None:
        output = _big_tool_output("mut-check")
        items = [{"type": "function_call_output", "call_id": "c1", "output": output}]
        orig = items[0]["output"]
        compress_responses_input(items)
        assert items[0]["output"] == orig


# ---------------------------------------------------------------------------
# count_responses_tokens
# ---------------------------------------------------------------------------


class TestCountResponsesTokens:
    def test_counts_function_call_output_output(self) -> None:
        items = [{"type": "function_call_output", "call_id": "c1", "output": "hello world"}]
        count = count_responses_tokens(items)
        assert count > 0

    def test_counts_user_input_text(self) -> None:
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ]
        count = count_responses_tokens(items)
        assert count > 0

    def test_excludes_assistant_and_function_call(self) -> None:
        items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x" * 500}],
            },
            {"type": "function_call", "id": "c", "name": "bash", "arguments": "x" * 500},
        ]
        assert count_responses_tokens(items) == 0


# ---------------------------------------------------------------------------
# Proxy-level integration: dispatch by path
# ---------------------------------------------------------------------------


def _make_echo_server() -> tuple[HTTPServer, int]:
    """An HTTP server that echoes the POST body back as JSON."""

    class _EchoHandler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _post_to_proxy(port: int, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    import urllib.request

    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=raw,
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


class TestProxyDispatch:
    """Verify proxy.py routes the right paths to the right adapter."""

    def test_chat_completions_tool_message_digested(self) -> None:
        """POST /v1/chat/completions with a large tool message → OpenAI adapter → handle in body."""
        from distil.proxy import build_handler
        from http.server import HTTPServer

        echo_server, echo_port = _make_echo_server()
        try:
            upstream = f"http://127.0.0.1:{echo_port}"
            handler = build_handler(upstream)
            proxy = HTTPServer(("127.0.0.1", 0), handler)
            proxy_port = proxy.server_address[1]
            t = threading.Thread(target=proxy.serve_forever, daemon=True)
            t.start()
            try:
                output = _big_tool_output("proxy-chat")
                body = {
                    "model": "gpt-4o",
                    "messages": [
                        # Old tool message (outside recency window)
                        {"role": "tool", "tool_call_id": "c0", "content": output},
                        {"role": "user", "content": "next 1"},
                        {"role": "user", "content": "next 2"},
                    ],
                }
                status, echoed = _post_to_proxy(proxy_port, "/v1/chat/completions", body)
                assert status == 200
                forwarded_content = echoed["messages"][0]["content"]
                assert _has_handle(forwarded_content), (
                    f"/v1/chat/completions: expected digest handle in tool message, "
                    f"got: {forwarded_content!r}"
                )
            finally:
                proxy.shutdown()
        finally:
            echo_server.shutdown()

    def test_responses_function_call_output_digested(self) -> None:
        """POST /v1/responses with function_call_output → Responses adapter → handle in output."""
        from distil.proxy import build_handler
        from http.server import HTTPServer

        echo_server, echo_port = _make_echo_server()
        try:
            upstream = f"http://127.0.0.1:{echo_port}"
            handler = build_handler(upstream)
            proxy = HTTPServer(("127.0.0.1", 0), handler)
            proxy_port = proxy.server_address[1]
            t = threading.Thread(target=proxy.serve_forever, daemon=True)
            t.start()
            try:
                output = _big_tool_output("proxy-resp")
                body = {
                    "model": "gpt-4o",
                    "input": [
                        # Old function_call_output (outside recency window)
                        {"type": "function_call_output", "call_id": "c0", "output": output},
                        # Two more to push c0 out of recency window
                        {
                            "type": "function_call_output",
                            "call_id": "c1",
                            "output": _big_tool_output("c1"),
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "c2",
                            "output": _big_tool_output("c2"),
                        },
                    ],
                }
                status, echoed = _post_to_proxy(proxy_port, "/v1/responses", body)
                assert status == 200
                forwarded_output = echoed["input"][0]["output"]
                assert _has_handle(forwarded_output), (
                    f"/v1/responses: expected digest handle in function_call_output, "
                    f"got: {forwarded_output!r}"
                )
            finally:
                proxy.shutdown()
        finally:
            echo_server.shutdown()

    def test_messages_path_still_uses_anthropic_adapter(self) -> None:
        """POST /v1/messages with Anthropic tool_result → Anthropic adapter → handle in content."""
        from distil.proxy import build_handler
        from http.server import HTTPServer

        echo_server, echo_port = _make_echo_server()
        try:
            upstream = f"http://127.0.0.1:{echo_port}"
            handler = build_handler(upstream)
            proxy = HTTPServer(("127.0.0.1", 0), handler)
            proxy_port = proxy.server_address[1]
            t = threading.Thread(target=proxy.serve_forever, daemon=True)
            t.start()
            try:
                output = _big_tool_output("proxy-anthropic")
                body = {
                    "model": "claude-opus-4-5",
                    "max_tokens": 256,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_01",
                                    "content": output,
                                }
                            ],
                        },
                        # Push tool_result outside recency window
                        {"role": "user", "content": "next 1"},
                        {"role": "user", "content": "next 2"},
                    ],
                }
                status, echoed = _post_to_proxy(proxy_port, "/v1/messages", body)
                assert status == 200
                forwarded_content = echoed["messages"][0]["content"][0]["content"]
                assert _has_handle(forwarded_content), (
                    f"/v1/messages: expected digest handle in tool_result, "
                    f"got: {forwarded_content!r}"
                )
            finally:
                proxy.shutdown()
        finally:
            echo_server.shutdown()

    def test_responses_non_json_body_forwarded_unchanged(self) -> None:
        """A non-JSON body on /v1/responses must be forwarded without crashing."""
        from distil.proxy import build_handler
        from http.server import HTTPServer
        import urllib.request

        received_bodies: list[bytes] = []

        class _RawEcho(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length)
                received_bodies.append(raw)
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        raw_server = HTTPServer(("127.0.0.1", 0), _RawEcho)
        raw_port = raw_server.server_address[1]
        threading.Thread(target=raw_server.serve_forever, daemon=True).start()
        try:
            upstream = f"http://127.0.0.1:{raw_port}"
            handler = build_handler(upstream)
            proxy = HTTPServer(("127.0.0.1", 0), handler)
            proxy_port = proxy.server_address[1]
            t = threading.Thread(target=proxy.serve_forever, daemon=True)
            t.start()
            try:
                bad_body = b"not json at all"
                req = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/responses",
                    data=bad_body,
                    headers={"Content-Length": str(len(bad_body))},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    assert resp.status == 200
                assert received_bodies and received_bodies[0] == bad_body, (
                    "Non-JSON body must be forwarded byte-for-byte (fail-open)"
                )
            finally:
                proxy.shutdown()
        finally:
            raw_server.shutdown()


# ---------------------------------------------------------------------------
# _compress_openai_message: list content with non-dict parts (lines 105-106)
# ---------------------------------------------------------------------------


class TestChatCompletionsListContentNonDictPart:
    """A non-dict element inside list content must be passed through unchanged."""

    def test_user_list_content_non_dict_part_passthrough(self) -> None:
        # An integer inside the content list — not a dict, so pass through
        msgs: list[Any] = [
            {"role": "user", "content": [42, {"type": "text", "text": "actual text"}]}
        ]
        compressed, store = compress_chat_completions(msgs)
        # The integer part passed through; the text part may be compressed
        assert compressed[0]["content"][0] == 42

    def test_tool_list_content_non_dict_part_passthrough(self) -> None:
        msgs: list[Any] = [
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": ["string-not-dict", {"type": "text", "text": _big_tool_output()}],
            },
            {"role": "user", "content": "next 1"},
            {"role": "user", "content": "next 2"},
        ]
        compressed, _ = compress_chat_completions(msgs)
        assert compressed[0]["content"][0] == "string-not-dict"


# ---------------------------------------------------------------------------
# _compress_openai_message: user role with list content (line 113)
# ---------------------------------------------------------------------------


class TestChatCompletionsUserListContent:
    """role:'user' with list content — text parts get Tier-0 (not Tier-1)."""

    def test_user_list_text_part_not_digested(self) -> None:
        # User content uses _compress_text_content (Tier-0), never a digest handle
        big_text = _big_tool_output("user-list")
        msgs = [{"role": "user", "content": [{"type": "text", "text": big_text}]}]
        compressed, store = compress_chat_completions(msgs)
        # Tier-0 only — no Tier-1 digest handle
        assert not _has_handle(compressed[0]["content"][0]["text"])
        assert not store.handles


# ---------------------------------------------------------------------------
# _recent_chat_verbatim_indices with k=0 (line 134)
# _recent_response_verbatim_indices with k=0 (line 208)
# ---------------------------------------------------------------------------


def test_recent_chat_verbatim_k_zero() -> None:
    """k=0 means no recency carve-out — returns empty set immediately."""
    from distil.adapters.openai import _recent_chat_verbatim_indices

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "result"},
    ]
    assert _recent_chat_verbatim_indices(msgs, 0) == set()


def test_recent_response_verbatim_k_zero() -> None:
    """k=0 means no recency carve-out for Responses items."""
    from distil.adapters.openai import _recent_response_verbatim_indices

    items = [
        {"type": "function_call_output", "call_id": "c1", "output": "data"},
    ]
    assert _recent_response_verbatim_indices(items, 0) == set()


# ---------------------------------------------------------------------------
# _compress_response_item: message with non-list content (line 249)
# ---------------------------------------------------------------------------


def test_compress_response_message_non_list_content_passthrough() -> None:
    """A 'message' item with a string (not list) content field passes through unchanged."""
    item = {"type": "message", "role": "user", "content": "plain string content"}
    items = [item]
    compressed, store = compress_responses_input(items)
    assert compressed[0] == item
    assert not store.handles


# ---------------------------------------------------------------------------
# _compress_response_item: user message list with non-dict parts (lines 258-259)
# ---------------------------------------------------------------------------


def test_compress_response_user_message_non_dict_part_passthrough() -> None:
    """Non-dict elements in a user message content list pass through unchanged."""
    items: list[Any] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                42,  # not a dict
                {"type": "input_text", "text": "valid text part"},
            ],
        }
    ]
    compressed, _ = compress_responses_input(items)
    assert compressed[0]["content"][0] == 42  # non-dict passed through


# ---------------------------------------------------------------------------
# count_responses_tokens: edge cases
# ---------------------------------------------------------------------------


class TestCountResponsesTokensEdgeCases:
    def test_non_dict_item_skipped(self) -> None:
        """Non-dict items in the list are skipped without error."""
        items: list[Any] = ["not-a-dict", {"type": "function_call_output", "output": "hello"}]
        count = count_responses_tokens(items)
        # Only the dict item counted
        assert count > 0

    def test_message_non_list_content_not_counted(self) -> None:
        """User message with string content (not list) → 0 tokens counted (schema mismatch)."""
        items = [{"type": "message", "role": "user", "content": "plain string"}]
        count = count_responses_tokens(items)
        assert count == 0

    def test_function_call_output_non_string_output_not_counted(self) -> None:
        """function_call_output with non-string output → 0 tokens counted."""
        items = [{"type": "function_call_output", "output": {"key": "value"}}]
        count = count_responses_tokens(items)
        assert count == 0


# ---------------------------------------------------------------------------
# Intent extraction for Responses API
# ---------------------------------------------------------------------------


class TestExtractResponsesIntent:
    """_extract_responses_intent pulls terms from latest user text + function_call args."""

    def test_extracts_from_user_input_text(self) -> None:
        from distil.adapters.openai import _extract_responses_intent

        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "find the authentication token"}],
            }
        ]
        terms = _extract_responses_intent(items)
        assert "authentication" in terms or "token" in terms

    def test_extracts_from_function_call_args(self) -> None:
        from distil.adapters.openai import _extract_responses_intent

        items = [
            {
                "type": "function_call",
                "id": "fc_1",
                "name": "bash",
                "arguments": '{"cmd": "grep tenant_id /var/log/app.log"}',
            }
        ]
        terms = _extract_responses_intent(items)
        assert "bash" in terms or "tenant_id" in terms

    def test_uses_latest_user_message_only(self) -> None:
        from distil.adapters.openai import _extract_responses_intent

        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "find alpha_needle_xyz"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "find beta_needle_xyz"}],
            },
        ]
        terms = _extract_responses_intent(items)
        # Second user message is the latest; its terms must be present
        assert "beta_needle_xyz" in terms

    def test_empty_items_returns_empty(self) -> None:
        from distil.adapters.openai import _extract_responses_intent

        assert _extract_responses_intent([]) == frozenset()

    def test_malformed_function_call_args_fail_open(self) -> None:
        from distil.adapters.openai import _extract_responses_intent

        items = [{"type": "function_call", "name": "bash", "arguments": "not-valid-json{{{"}]
        # Must not raise; terms from name still extracted
        terms = _extract_responses_intent(items)
        assert "bash" in terms

    def test_non_dict_items_skipped(self) -> None:
        from distil.adapters.openai import _extract_responses_intent

        items: list[Any] = ["not-a-dict", 42, None]
        terms = _extract_responses_intent(items)
        assert terms == frozenset()


# ---------------------------------------------------------------------------
# Responses API expand-tool injection
# ---------------------------------------------------------------------------


class TestInjectExpandToolResponses:
    """inject_expand_tool_responses uses the flat Responses function-tool schema."""

    def test_injects_into_body_with_no_tools(self) -> None:
        from distil.expand import EXPAND_TOOL_NAME, inject_expand_tool_responses

        body: dict[str, Any] = {"input": [], "model": "gpt-4o"}
        out = inject_expand_tool_responses(body)
        assert isinstance(out["tools"], list)
        names = [t.get("name") for t in out["tools"] if isinstance(t, dict)]
        assert EXPAND_TOOL_NAME in names

    def test_flat_schema_no_nested_function_key(self) -> None:
        """Responses schema is flat — no 'function' wrapper key like Chat Completions."""
        from distil.expand import EXPAND_TOOL_NAME, inject_expand_tool_responses

        out = inject_expand_tool_responses({})
        tool = next(
            t for t in out["tools"] if isinstance(t, dict) and t.get("name") == EXPAND_TOOL_NAME
        )
        # Flat: name/description/parameters at top level, NOT nested under "function"
        assert "name" in tool
        assert "parameters" in tool
        assert "function" not in tool

    def test_idempotent(self) -> None:
        from distil.expand import EXPAND_TOOL_NAME, inject_expand_tool_responses

        body: dict[str, Any] = {}
        once = inject_expand_tool_responses(body)
        twice = inject_expand_tool_responses(once)
        count = sum(
            1 for t in twice["tools"] if isinstance(t, dict) and t.get("name") == EXPAND_TOOL_NAME
        )
        assert count == 1

    def test_preserves_existing_tools(self) -> None:
        from distil.expand import EXPAND_TOOL_NAME, inject_expand_tool_responses

        existing = {"type": "function", "name": "get_logs", "parameters": {}}
        out = inject_expand_tool_responses({"tools": [existing]})
        names = {t["name"] for t in out["tools"] if isinstance(t, dict)}
        assert names == {"get_logs", EXPAND_TOOL_NAME}


# ---------------------------------------------------------------------------
# Output shaping: shape="responses"
# ---------------------------------------------------------------------------


class TestShapeRequestResponses:
    """shape_request with shape='responses' appends directive to instructions."""

    def test_appends_to_existing_instructions(self) -> None:
        from distil.output import shape_request

        body = {"instructions": "You are a helpful assistant.", "input": []}
        out = shape_request(body, level="light", allow=True, shape="responses")
        assert out["instructions"].startswith("You are a helpful assistant.")
        assert "concise" in out["instructions"].lower()

    def test_creates_instructions_when_absent(self) -> None:
        from distil.output import shape_request

        body: dict[str, Any] = {"input": []}
        out = shape_request(body, level="light", allow=True, shape="responses")
        assert "concise" in out["instructions"].lower()

    def test_noop_when_off(self) -> None:
        from distil.output import shape_request

        body: dict[str, Any] = {"input": []}
        assert shape_request(body, level="off", allow=True, shape="responses") is body

    def test_noop_when_disallowed(self) -> None:
        from distil.output import shape_request

        body: dict[str, Any] = {"input": []}
        assert shape_request(body, level="light", allow=False, shape="responses") is body

    def test_aggressive_level(self) -> None:
        from distil.output import shape_request

        body: dict[str, Any] = {"input": []}
        out = shape_request(body, level="aggressive", allow=True, shape="responses")
        assert "concise" in out["instructions"].lower()

    def test_does_not_mutate_input(self) -> None:
        from distil.output import shape_request

        body: dict[str, Any] = {"instructions": "sys", "input": []}
        shape_request(body, level="light", allow=True, shape="responses")
        assert body["instructions"] == "sys"


# ---------------------------------------------------------------------------
# Proxy integration: Responses expand loop end-to-end
# ---------------------------------------------------------------------------


class TestProxyResponsesExpandLoop:
    """Full path: proxy digests a function_call_output → fake upstream returns
    distil_expand function_call → proxy resolves and re-queries → final answer."""

    def test_expand_loop_responses_end_to_end(self, tmp_path: Any) -> None:
        import re
        import threading
        import urllib.request
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from distil.proxy import build_handler

        call_count: list[int] = [0]

        class _Upstream(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                call_count[0] += 1
                input_items = body.get("input", [])
                # After the first call: if input contains a function_call_output,
                # the proxy resolved our expand request — return the final answer.
                has_fco = any(
                    isinstance(i, dict)
                    and i.get("type") == "function_call_output"
                    and "load-bearing" in (i.get("output") or "")
                    for i in input_items
                )
                if has_fco:
                    resp = {"output": [{"type": "output_text", "text": "done with detail"}]}
                else:
                    # First call: find the handle in the input and ask to expand it
                    m = re.search(r"handle=([0-9a-f]{8})", json.dumps(input_items))
                    handle = m.group(1) if m else "00000000"
                    resp = {
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "distil_expand",
                                "arguments": json.dumps({"handle": handle}),
                            }
                        ]
                    }
                out = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a: object) -> None:
                pass

        import os

        os.environ.setdefault("DISTIL_HOME", str(tmp_path))

        up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        handler = build_handler(f"http://127.0.0.1:{up.server_address[1]}", expand=True)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            big = "\n".join(
                ("load-bearing line 5" if i == 5 else f"verbose log line {i}") for i in range(40)
            )
            payload = json.dumps(
                {
                    "model": "gpt-4o",
                    "input": [
                        {"type": "function_call_output", "call_id": "c0", "output": big},
                        # Two more to push c0 outside the recency window so it digests.
                        {
                            "type": "function_call_output",
                            "call_id": "c1",
                            "output": _big_tool_output("c1"),
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "c2",
                            "output": _big_tool_output("c2"),
                        },
                    ],
                }
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/v1/responses",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                assert r.headers.get("x-distil-expanded") == "1"
                final = json.loads(r.read())
            assert final["output"][0]["text"] == "done with detail"
            assert call_count[0] == 2  # initial + one recovery round-trip
        finally:
            proxy.shutdown()
            up.shutdown()


class TestProxyResponsesOutputShaping:
    """Proxy wires output shaping into the /v1/responses branch."""

    def test_shaping_appends_instructions(self) -> None:
        from http.server import HTTPServer

        from distil.proxy import build_handler

        echo_server, echo_port = _make_echo_server()
        try:
            handler = build_handler(
                f"http://127.0.0.1:{echo_port}",
                lossless_only=False,
                shape_output="light",
            )
            proxy = HTTPServer(("127.0.0.1", 0), handler)
            proxy_port = proxy.server_address[1]
            t = threading.Thread(target=proxy.serve_forever, daemon=True)
            t.start()
            try:
                body = {
                    "model": "gpt-4o",
                    "instructions": "You are helpful.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        }
                    ],
                }
                status, echoed = _post_to_proxy(proxy_port, "/v1/responses", body)
                assert status == 200
                instructions = echoed.get("instructions", "")
                assert "concise" in instructions.lower(), (
                    f"Expected shaping directive in instructions, got: {instructions!r}"
                )
            finally:
                proxy.shutdown()
        finally:
            echo_server.shutdown()

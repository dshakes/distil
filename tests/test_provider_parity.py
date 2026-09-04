"""Every server, every provider: the same compression, the same recovery.

The seams this file guards were all "works on Anthropic, silently wrong elsewhere":

* the Responses API was compressed by ``proxy.py`` and forwarded raw by ``aproxy.py``
  and ``gateway.py``, because each server kept its own hand-written path set;
* ``/v1/chat/completions`` got the ANTHROPIC expand-tool schema injected, which
  OpenAI rejects outright, and a ``distil_expand`` tool_call it did return was never
  resolved (the Anthropic loop looks for ``tool_use`` blocks, not ``tool_calls``);
* the streaming splice parses Anthropic SSE only, and ran anyway on OpenAI and
  Gemini streams — re-labelling every frame and appending a bogus ``message_stop``;
* Azure OpenAI's paths carry a deployment name, so none of them matched at all.

Each test below fails on exactly one of those.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distil import expand as _expand
from distil.httpguard import (
    is_chat_completions_path,
    is_compressible_path,
    is_responses_path,
)
from distil.proxy import _unstream_path, build_handler

_HANDLE_RE = re.compile(r"handle=([0-9a-f]{8})")

# 40 lines of tool output — comfortably past the Tier-1 digest threshold.
_LOG = "\n".join(f"log line number {i} here, with enough text to matter" for i in range(40))


# ---------------------------------------------------------------------------
# Scriptable upstream
# ---------------------------------------------------------------------------


class _Upstream:
    """A fake provider. Queue response bodies; every request is recorded.

    A queued value may be a callable ``(path, body) -> dict``, which is how the
    expand tests answer with a tool call naming a handle they could not know in
    advance (the proxy mints it during compression, exactly as in production).
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.queue: list = []
        self.default: dict = {"ok": True}
        self.port = 0

    def next_body(self, path: str, body: dict) -> dict:
        self.requests.append((path, body))
        if not self.queue:
            return self.default
        item = self.queue.pop(0)
        return item(path, body) if callable(item) else item


@pytest.fixture()
def upstream():
    up = _Upstream()

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw)
            except ValueError:
                body = {}
            payload = json.dumps(up.next_body(self.path, body)).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # silence
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    up.port = srv.server_address[1]
    yield up
    srv.shutdown()


@pytest.fixture()
def proxy(upstream):
    servers = []

    def make(**kw):
        h = build_handler(f"http://127.0.0.1:{upstream.port}", **kw)
        px = ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=px.serve_forever, daemon=True).start()
        servers.append(px)
        return px.server_address[1]

    yield make
    for s in servers:
        s.shutdown()


def _post(port, path: str, payload: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return r.status, dict(r.headers), r.read()


def _chat_body() -> dict:
    return {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "you are a test agent"},
            {"role": "user", "content": "read the log"},
            {"role": "tool", "tool_call_id": "c1", "content": _LOG},
            {"role": "user", "content": "summarise it"},
        ],
    }


def _responses_body() -> dict:
    return {
        "model": "gpt-test",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {"type": "function_call_output", "call_id": "c1", "output": _LOG},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "?"}]},
        ],
    }


# ---------------------------------------------------------------------------
# Path matching (item 5): one definition, Azure included
# ---------------------------------------------------------------------------


class TestPathMatching:
    @pytest.mark.parametrize(
        "path",
        [
            "/v1/messages",
            "/v1/chat/completions",
            "/v1/responses",
            "/openai/deployments/gpt4o-prod/chat/completions",
            "/openai/deployments/gpt4o-prod/chat/completions?api-version=2024-10-21",
            "/openai/v1/chat/completions?api-version=preview",
            "/openai/responses?api-version=2025-04-01-preview",
            "/openai/v1/responses",
        ],
    )
    def test_compressible(self, path):
        assert is_compressible_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/models",
            "/v1/embeddings",
            "/openai/deployments/gpt4o/embeddings",
            "/v1/chat/completions/extra",
            "/openai/deployments/a/b/chat/completions",  # deployment names have no slash
            "/evil/v1/messages",
        ],
    )
    def test_not_compressible(self, path):
        assert not is_compressible_path(path)

    def test_azure_chat_is_chat_not_responses(self):
        p = "/openai/deployments/d1/chat/completions"
        assert is_chat_completions_path(p)
        assert not is_responses_path(p)

    def test_azure_responses_is_responses_not_chat(self):
        assert is_responses_path("/openai/responses?api-version=x")
        assert not is_chat_completions_path("/openai/responses")

    def test_query_string_never_defeats_the_match(self):
        # strip_query is applied inside the matcher; a caller that forgets to strip
        # must not silently fall through to passthrough.
        assert is_compressible_path("/v1/messages?beta=true")


class TestUnstreamPath:
    def test_gemini_stream_becomes_unary(self):
        assert (
            _unstream_path("/v1beta/models/gemini-2.0:streamGenerateContent?alt=sse&key=k")
            == "/v1beta/models/gemini-2.0:generateContent?key=k"
        )

    def test_alt_sse_alone_leaves_no_dangling_question_mark(self):
        assert _unstream_path("/v1beta/models/m:streamGenerateContent?alt=sse") == (
            "/v1beta/models/m:generateContent"
        )

    def test_openai_path_is_untouched(self):
        assert _unstream_path("/v1/chat/completions") == "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Expand tool schemas (item 2)
# ---------------------------------------------------------------------------


class TestExpandToolSchemas:
    def test_chat_tool_is_the_nested_function_form(self):
        out = _expand.inject_expand_tool_chat({"messages": []})
        tool = out["tools"][-1]
        assert tool["type"] == "function"
        # The Anthropic spec's keys must NOT appear — that shape 400s on OpenAI.
        assert "input_schema" not in tool and "name" not in tool
        assert tool["function"]["name"] == "distil_expand"
        assert tool["function"]["parameters"]["required"] == ["handle"]

    def test_chat_injection_is_idempotent(self):
        once = _expand.inject_expand_tool_chat({"messages": []})
        twice = _expand.inject_expand_tool_chat(once)
        assert len(twice["tools"]) == 1
        assert twice is once  # unchanged object: the cached prefix stays byte-stable

    def test_has_expand_tool_sees_the_nested_form(self):
        assert _expand._has_expand_tool([_expand.EXPAND_TOOL_CHAT])
        assert _expand._has_expand_tool([_expand.EXPAND_TOOL])
        assert _expand._has_expand_tool([_expand.EXPAND_TOOL_RESPONSES])
        assert not _expand._has_expand_tool([{"type": "function", "function": {"name": "other"}}])
        assert not _expand._has_expand_tool(["not a dict", None])

    def test_existing_client_tools_are_preserved(self):
        client = {"type": "function", "function": {"name": "bash", "parameters": {}}}
        out = _expand.inject_expand_tool_chat({"tools": [client]})
        assert out["tools"][0] == client
        assert len(out["tools"]) == 2


# ---------------------------------------------------------------------------
# Chat Completions expand loop (item 2)
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.asked: list[str] = []

    def expand(self, handle: str) -> str:
        self.asked.append(handle)
        return self.mapping[handle]  # KeyError for an unknown handle, as in production


def _chat_tool_call(handle: str, *, name: str = "distil_expand", cid: str = "call_1") -> dict:
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"handle": handle})},
    }


def _chat_resp(message: dict) -> dict:
    return {"id": "r", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


class TestRunExpandLoopChat:
    def test_resolves_and_requeries(self):
        store = _Store({"aabbccdd": "THE ORIGINAL"})
        posted: list[dict] = []

        def post(b):
            posted.append(b)
            return _chat_resp({"role": "assistant", "content": "answer"})

        first = _chat_resp({"role": "assistant", "tool_calls": [_chat_tool_call("aabbccdd")]})
        final = _expand.run_expand_loop_chat(_chat_body(), first, store, post, on_signal=None)
        assert final["choices"][0]["message"]["content"] == "answer"
        assert store.asked == ["aabbccdd"]
        # The recovered text goes back as a role:"tool" message keyed by tool_call_id.
        tail = posted[0]["messages"][-2:]
        assert tail[0]["tool_calls"][0]["id"] == "call_1"
        assert tail[1] == {"role": "tool", "tool_call_id": "call_1", "content": "THE ORIGINAL"}

    def test_client_tool_call_is_never_swallowed(self):
        # A turn that also asks the CLIENT to run something must reach the agent
        # untouched, or its finish_reason is hidden behind the continuation's.
        store = _Store({"aabbccdd": "x"})
        first = _chat_resp(
            {
                "role": "assistant",
                "tool_calls": [
                    _chat_tool_call("aabbccdd"),
                    _chat_tool_call("z", name="bash", cid="call_2"),
                ],
            }
        )

        def post(b):  # pragma: no cover — must not run
            raise AssertionError("mixed turn was replayed")

        assert _expand.run_expand_loop_chat(_chat_body(), first, store, post) is first

    def test_no_tool_call_returns_the_first_response(self):
        first = _chat_resp({"role": "assistant", "content": "hi"})

        def post(b):  # pragma: no cover
            raise AssertionError("plain answer was replayed")

        assert _expand.run_expand_loop_chat(_chat_body(), first, _Store(), post) is first

    def test_unknown_handle_answers_with_a_miss_not_a_500(self):
        store = _Store()  # empty: every expand raises
        seen: list[tuple[str, str]] = []
        posted: list[dict] = []

        def post(b):
            posted.append(b)
            return _chat_resp({"role": "assistant", "content": "ok"})

        first = _chat_resp({"role": "assistant", "tool_calls": [_chat_tool_call("deadbeef")]})
        _expand.run_expand_loop_chat(
            _chat_body(), first, store, post, on_signal=lambda h, o: seen.append((h, o))
        )
        assert _expand.is_miss(posted[0]["messages"][-1]["content"])
        assert seen and _expand.is_miss(seen[0][1])

    def test_malformed_arguments_do_not_raise(self):
        store = _Store()
        first = _chat_resp(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "distil_expand", "arguments": "{not json"},
                    }
                ],
            }
        )
        posted: list[dict] = []

        def post(b):
            posted.append(b)
            return _chat_resp({"role": "assistant", "content": "ok"})

        _expand.run_expand_loop_chat(_chat_body(), first, store, post, on_signal=None)
        assert _expand.is_miss(posted[0]["messages"][-1]["content"])

    def test_bounded_by_max_iters(self):
        store = _Store({"aabbccdd": "x"})
        calls = {"n": 0}

        def post(b):
            calls["n"] += 1
            return _chat_resp({"role": "assistant", "tool_calls": [_chat_tool_call("aabbccdd")]})

        first = _chat_resp({"role": "assistant", "tool_calls": [_chat_tool_call("aabbccdd")]})
        _expand.run_expand_loop_chat(_chat_body(), first, store, post, max_iters=3, on_signal=None)
        assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Sync proxy: every path compresses, every path recovers
# ---------------------------------------------------------------------------


class TestProxyChatCompletions:
    def test_expand_injects_the_openai_schema(self, proxy, upstream):
        port = proxy(expand=True)
        _post(port, "/v1/chat/completions", _chat_body())
        _, sent = upstream.requests[0]
        tool = sent["tools"][-1]
        assert tool["function"]["name"] == "distil_expand"
        assert "input_schema" not in tool

    def test_anthropic_still_gets_the_anthropic_schema(self, proxy, upstream):
        port = proxy(expand=True)
        _post(
            port,
            "/v1/messages",
            {
                "model": "claude-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t", "content": _LOG}],
                    }
                ],
            },
        )
        _, sent = upstream.requests[0]
        tool = sent["tools"][-1]
        assert tool["name"] == "distil_expand"
        assert "input_schema" in tool

    def test_tool_call_is_resolved_and_requeried(self, proxy, upstream):
        # The upstream plays the model: it reads the handle out of the digest stub it
        # was actually sent, asks for it back, and then answers.
        def ask_for_handle(path, body):
            m = _HANDLE_RE.search(json.dumps(body))
            assert m, "nothing was digested — the test body is too small"
            return _chat_resp({"role": "assistant", "tool_calls": [_chat_tool_call(m.group(1))]})

        upstream.queue = [
            ask_for_handle,
            _chat_resp({"role": "assistant", "content": "final answer"}),
        ]
        port = proxy(expand=True)
        _, headers, body = _post(port, "/v1/chat/completions", _chat_body())

        assert json.loads(body)["choices"][0]["message"]["content"] == "final answer"
        assert headers.get("x-distil-expanded") == "1"
        assert len(upstream.requests) == 2  # the recovery round-trip never reached the client
        replay = upstream.requests[1][1]["messages"]
        assert replay[-1]["role"] == "tool"
        assert "log line number 39" in replay[-1]["content"]

    def test_body_is_compressed_and_headed(self, proxy, upstream):
        port = proxy()
        _, headers, _ = _post(port, "/v1/chat/completions", _chat_body())
        assert headers["x-distil-compressed"] == "1"
        assert int(headers["x-distil-tokens-saved"]) > 0


class TestProxyResponsesApi:
    def test_azure_responses_path_is_compressed(self, proxy, upstream):
        port = proxy()
        _, headers, _ = _post(
            port, "/openai/responses?api-version=2025-04-01-preview", _responses_body()
        )
        assert headers["x-distil-compressed"] == "1"
        assert int(headers["x-distil-tokens-saved"]) > 0
        _, sent = upstream.requests[0]
        assert _HANDLE_RE.search(json.dumps(sent["input"]))

    def test_azure_chat_path_is_compressed(self, proxy, upstream):
        port = proxy()
        _, headers, _ = _post(port, "/openai/deployments/gpt4o-prod/chat/completions", _chat_body())
        assert headers["x-distil-compressed"] == "1"
        assert int(headers["x-distil-tokens-saved"]) > 0

    def test_responses_expand_tool_is_the_flat_form(self, proxy, upstream):
        port = proxy(expand=True)
        _post(port, "/v1/responses", _responses_body())
        _, sent = upstream.requests[0]
        tool = sent["tools"][-1]
        assert tool["type"] == "function" and tool["name"] == "distil_expand"
        assert "function" not in tool


# ---------------------------------------------------------------------------
# Streaming (item 3): the splice is Anthropic-only
# ---------------------------------------------------------------------------


def _sse_upstream(frames: bytes):
    """An upstream that answers with a real SSE stream (for the Anthropic path)."""

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, *a):
            pass

    return _H


class TestStreamingProviderGate:
    def test_openai_stream_gets_no_anthropic_frames(self, proxy, upstream):
        # Before the gate, the Anthropic splice ran here: every chunk came back
        # re-labelled `event: message` and the body ended with a fabricated
        # `message_stop` no OpenAI client can parse.
        upstream.queue = [_chat_resp({"role": "assistant", "content": "streamed answer"})]
        port = proxy(expand=True)
        body = dict(_chat_body(), stream=True)
        _, headers, out = _post(port, "/v1/chat/completions", body)

        assert b"message_stop" not in out
        assert headers["Content-Type"] == "text/event-stream"
        frames = [ln for ln in out.decode().splitlines() if ln.startswith("data: ")]
        assert frames[-1] == "data: [DONE]"
        first = json.loads(frames[0][6:])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["content"] == "streamed answer"

    def test_openai_stream_still_resolves_expands(self, proxy, upstream):
        # Losing the splice must not lose the recovery: the buffered loop runs instead.
        def ask_for_handle(path, body):
            m = _HANDLE_RE.search(json.dumps(body))
            assert m
            return _chat_resp({"role": "assistant", "tool_calls": [_chat_tool_call(m.group(1))]})

        upstream.queue = [
            ask_for_handle,
            _chat_resp({"role": "assistant", "content": "recovered"}),
        ]
        port = proxy(expand=True)
        _, headers, out = _post(port, "/v1/chat/completions", dict(_chat_body(), stream=True))
        assert headers.get("x-distil-expanded") == "1"
        assert b"recovered" in out
        assert b"message_stop" not in out

    def test_upstream_never_sees_the_stream_flag(self, proxy, upstream):
        upstream.queue = [_chat_resp({"role": "assistant", "content": "x"})]
        port = proxy(expand=True)
        _post(port, "/v1/chat/completions", dict(_chat_body(), stream=True))
        assert "stream" not in upstream.requests[0][1]

    def test_gemini_stream_gets_no_anthropic_frames(self, proxy, upstream):
        gem_resp = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}]}
        upstream.queue = [gem_resp]
        port = proxy(expand=True)
        path = "/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "look"}]},
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": "read", "response": {"content": _LOG}}}
                    ],
                },
            ]
        }
        _, headers, out = _post(port, path, body)
        assert b"message_stop" not in out
        assert headers["Content-Type"] == "text/event-stream"
        assert json.loads(out.decode().split("data: ", 1)[1]) == gem_resp
        # The URL, not just the body, carries Gemini's streaming mode.
        fwd_path = upstream.requests[0][0]
        assert ":generateContent" in fwd_path and "alt=sse" not in fwd_path

    def test_anthropic_stream_still_takes_the_splice(self):
        # The regression guard for the gate itself: /v1/messages must keep the
        # frame-by-frame path (its SSE flows through untouched, message_stop and all).
        frames = (
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{}}}\n\n'
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        up = ThreadingHTTPServer(("127.0.0.1", 0), _sse_upstream(frames))
        threading.Thread(target=up.serve_forever, daemon=True).start()
        h = build_handler(f"http://127.0.0.1:{up.server_address[1]}", expand=True)
        px = ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=px.serve_forever, daemon=True).start()
        try:
            _, _, out = _post(
                px.server_address[1],
                "/v1/messages",
                {
                    "model": "claude-test",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "t", "content": _LOG}
                            ],
                        }
                    ],
                },
            )
        finally:
            px.shutdown()
            up.shutdown()
        assert b"message_stop" in out
        assert b"chat.completion.chunk" not in out


class TestSseSynthesis:
    def test_chat_carries_tool_calls_with_an_index(self):
        from distil.streamexpand import sse_from_response

        resp = _chat_resp(
            {"role": "assistant", "tool_calls": [_chat_tool_call("aabbccdd", name="bash")]}
        )
        out = sse_from_response("chat", resp).decode()
        head = json.loads(out.split("data: ", 1)[1].split("\n", 1)[0])
        assert head["choices"][0]["delta"]["tool_calls"][0]["index"] == 0

    def test_responses_emits_the_completed_event(self):
        from distil.streamexpand import sse_from_response

        resp = {"id": "resp_1", "output": [{"type": "message"}]}
        out = sse_from_response("responses", resp).decode()
        assert "event: response.created" in out
        assert "event: response.completed" in out
        assert out.count("resp_1") == 2

    def test_gemini_is_one_frame(self):
        from distil.streamexpand import sse_from_response

        resp = {"candidates": []}
        assert sse_from_response("gemini", resp) == b'data: {"candidates": []}\n\n'

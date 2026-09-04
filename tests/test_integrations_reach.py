"""AutoGen + ASGI integrations.

Both are duck-typed/framework-free and must never import their target
(``autogen_core``, ``starlette``/an ASGI server) — the tests run with neither
installed, which is also how CI runs them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from distil.integrations.asgi import DistilMiddleware
from distil.integrations.autogen import (
    DistilModelClient,
    compress_messages as autogen_compress,
    compress_tool_result,
    compressing_tool,
)

BIG = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog" for i in range(400))
_BIG_OUTPUT = "\n".join(f"row {i}: value_{i} status=ok detail=lorem ipsum dolor" for i in range(40))


def test_neither_framework_is_imported() -> None:
    assert "autogen_core" not in sys.modules
    assert "starlette" not in sys.modules


# --- AutoGen: fakes (duck-typed pydantic-v2-shaped messages) ----------------


class _FakeMsg:
    def __init__(self, type_: str, content: Any) -> None:
        self.type = type_
        self.content = content

    def model_copy(self, update: dict[str, Any]) -> "_FakeMsg":
        new = _FakeMsg(self.type, self.content)
        for k, v in update.items():
            setattr(new, k, v)
        return new


class _FakeFuncResult:
    def __init__(self, content: str, name: str = "tool", call_id: str = "1") -> None:
        self.content = content
        self.name = name
        self.call_id = call_id

    def model_copy(self, update: dict[str, Any]) -> "_FakeFuncResult":
        new = _FakeFuncResult(self.content, self.name, self.call_id)
        for k, v in update.items():
            setattr(new, k, v)
        return new


class _FakeFERMsg:
    type = "FunctionExecutionResultMessage"

    def __init__(self, content: list[_FakeFuncResult]) -> None:
        self.content = content

    def model_copy(self, update: dict[str, Any]) -> "_FakeFERMsg":
        new = _FakeFERMsg(self.content)
        for k, v in update.items():
            setattr(new, k, v)
        return new


# --- AutoGen: compress_tool_result / compressing_tool -----------------------


def test_compress_tool_result_shrinks_big_output() -> None:
    out = compress_tool_result(BIG)
    assert len(out) < len(BIG)


def test_compressing_tool_wraps_sync_function() -> None:
    def get_data() -> str:
        return BIG

    wrapped = compressing_tool(get_data)
    assert len(wrapped()) < len(BIG)
    assert wrapped.__name__ == "get_data"


def test_compressing_tool_wraps_async_function() -> None:
    async def get_data() -> str:
        return BIG

    wrapped = compressing_tool(get_data)
    result = asyncio.run(wrapped())
    assert len(result) < len(BIG)


def test_compressing_tool_passes_through_non_string_return() -> None:
    def get_data() -> dict:
        return {"a": 1}

    assert compressing_tool(get_data)() == {"a": 1}


# --- AutoGen: compress_messages ---------------------------------------------


def test_autogen_assistant_message_never_rewritten() -> None:
    msgs = [_FakeMsg("AssistantMessage", BIG)]
    assert autogen_compress(msgs)[0].content == BIG


def test_autogen_system_and_user_message_lossless() -> None:
    msgs = [_FakeMsg("SystemMessage", BIG), _FakeMsg("UserMessage", "short")]
    out = autogen_compress(msgs)
    assert out[1].content == "short"  # untouched, nothing to compress
    assert isinstance(out[0].content, str)


def test_autogen_function_execution_result_is_digested() -> None:
    msg = _FakeFERMsg([_FakeFuncResult(BIG)])
    out = autogen_compress([msg])
    assert len(out[0].content[0].content) < len(BIG)
    assert out[0].content[0].name == "tool"  # sibling fields preserved


def test_autogen_non_string_content_passes_through() -> None:
    msg = _FakeMsg("UserMessage", ["not", "a", "string"])
    out = autogen_compress([msg])
    assert out[0].content == ["not", "a", "string"]


def test_autogen_does_not_mutate_input() -> None:
    msg = _FakeFERMsg([_FakeFuncResult(BIG)])
    autogen_compress([msg])
    assert msg.content[0].content == BIG  # original untouched


# --- AutoGen: DistilModelClient ----------------------------------------------


class _FakeChatClient:
    model_info = {"vision": False}

    def __init__(self) -> None:
        self.seen: Any = None
        self.seen_stream: Any = None

    async def create(self, messages: list[Any], **kw: Any) -> str:
        self.seen = messages
        return "ok"

    def create_stream(self, messages: list[Any], **kw: Any) -> Any:
        self.seen_stream = messages

        async def _gen() -> Any:
            yield "chunk"

        return _gen()


def test_model_client_delegates_unknown_attrs() -> None:
    fc = _FakeChatClient()
    wrapped = DistilModelClient(fc)
    assert wrapped.model_info == {"vision": False}


def test_model_client_compresses_create_messages_positional() -> None:
    fc = _FakeChatClient()
    wrapped = DistilModelClient(fc)
    msg = _FakeFERMsg([_FakeFuncResult(BIG)])
    result = asyncio.run(wrapped.create([msg]))
    assert result == "ok"
    assert len(fc.seen[0].content[0].content) < len(BIG)


def test_model_client_compresses_create_messages_kwarg() -> None:
    fc = _FakeChatClient()
    wrapped = DistilModelClient(fc)
    msg = _FakeFERMsg([_FakeFuncResult(BIG)])
    asyncio.run(wrapped.create(messages=[msg]))
    assert len(fc.seen[0].content[0].content) < len(BIG)


def test_model_client_compresses_create_stream() -> None:
    fc = _FakeChatClient()
    wrapped = DistilModelClient(fc)
    msg = _FakeFERMsg([_FakeFuncResult(BIG)])

    async def _drain() -> list[Any]:
        return [chunk async for chunk in wrapped.create_stream([msg])]

    chunks = asyncio.run(_drain())
    assert chunks == ["chunk"]
    assert len(fc.seen_stream[0].content[0].content) < len(BIG)


# --- ASGI middleware ---------------------------------------------------------


def _receive_for(body: bytes):
    events = iter(
        [{"type": "http.request", "body": body, "more_body": False}, {"type": "http.disconnect"}]
    )

    async def _receive() -> dict:
        return next(events)

    return _receive


async def _noop_send(_message: dict) -> None:
    pass


def _run_through(app, scope: dict, body: bytes) -> None:
    asyncio.run(DistilMiddleware(app)(scope, _receive_for(body), _noop_send))


def test_asgi_ignores_non_post() -> None:
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["scope"] = scope
        captured["msg"] = await receive()

    body = b"hello"
    scope = {"type": "http", "method": "GET", "path": "/v1/messages", "headers": []}
    _run_through(app, scope, body)
    assert captured["msg"]["body"] == body  # untouched — passthrough


def test_asgi_ignores_non_compressible_path() -> None:
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["msg"] = await receive()

    body = json.dumps({"messages": [{"role": "tool", "content": BIG}]}).encode()
    scope = {"type": "http", "method": "POST", "path": "/health", "headers": []}
    _run_through(app, scope, body)
    assert captured["msg"]["body"] == body


def test_asgi_compresses_anthropic_shaped_body() -> None:
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["scope"] = scope
        captured["body"] = (await receive())["body"]

    messages = [
        {"role": "tool", "content": BIG},
        # keep the tool turn out of the recency-exempt window so it digests
        # (distil.adapters.anthropic.compress_messages keeps the freshest tool
        # turns byte-exact — see tests/test_adapter.py's `_PAD`).
        {"role": "user", "content": "next"},
        {"role": "user", "content": "next"},
    ]
    original = json.dumps({"model": "x", "messages": messages}).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/messages",
        "headers": [(b"content-length", str(len(original)).encode())],
    }
    _run_through(app, scope, original)

    assert len(captured["body"]) < len(original)
    headers = dict(captured["scope"]["headers"])
    assert int(headers[b"content-length"]) == len(captured["body"])
    json.loads(captured["body"])  # still valid JSON


def test_asgi_compresses_gemini_shaped_body() -> None:
    """Older turns outside the recency window get digested (mirrors test_gemini.py)."""
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["body"] = (await receive())["body"]

    contents = []
    for i in range(3):
        contents.append({"role": "user", "parts": [{"text": f"round {i}"}]})
        contents.append(
            {"role": "model", "parts": [{"functionCall": {"name": "fetch", "args": {"id": i}}}]}
        )
        contents.append(
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "fetch", "response": {"output": _BIG_OUTPUT}}}
                ],
            }
        )
    original = json.dumps({"contents": contents}).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1beta/models/gemini-1.5-pro:generateContent",
        "headers": [],
    }
    _run_through(app, scope, original)
    assert len(captured["body"]) < len(original)


def test_asgi_passes_through_malformed_json() -> None:
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["body"] = (await receive())["body"]

    body = b"not json"
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    _run_through(app, scope, body)
    assert captured["body"] == body


def test_asgi_skips_oversized_body(monkeypatch) -> None:
    import distil.integrations.asgi as asgi_mod

    monkeypatch.setattr(asgi_mod, "MAX_BODY_BYTES", 4)
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["body"] = (await receive())["body"]

    original = json.dumps({"messages": [{"role": "tool", "content": BIG}]}).encode()
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    _run_through(app, scope, original)
    assert captured["body"] == original  # too big to inspect — forwarded unchanged


def test_asgi_compression_error_fails_open(monkeypatch) -> None:
    import distil.integrations.asgi as asgi_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(asgi_mod, "compress_messages", _boom)
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        captured["body"] = (await receive())["body"]

    original = json.dumps({"messages": [{"role": "tool", "content": BIG}]}).encode()
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    _run_through(app, scope, original)
    assert captured["body"] == original


def test_asgi_second_receive_call_reaches_real_channel() -> None:
    """After the replayed body, a later receive() (e.g. disconnect) hits the real one."""
    captured = {}

    async def app(scope: dict, receive, send) -> None:
        await receive()  # consumes the replayed body
        captured["second"] = await receive()  # should be the real http.disconnect

    body = json.dumps({"messages": []}).encode()
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    _run_through(app, scope, body)
    assert captured["second"]["type"] == "http.disconnect"

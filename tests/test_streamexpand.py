"""Unit tests for the streaming distil_expand interceptor (distil/streamexpand.py).

Validated in isolation with a mock upstream + client: the interceptor must (1) relay a
plain streamed response untouched, (2) intercept a distil_expand call — suppress it,
resolve the handle, re-query, and splice the continuation — so the client sees ONE
coherent message and never the internal tool, (3) survive SSE frames split across read
boundaries, and (4) re-index spliced blocks so indices stay coherent.
"""

from __future__ import annotations

import json

from distil.streamexpand import stream_with_expand


# --- test doubles -------------------------------------------------------------
class _WFile:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, b: bytes) -> None:
        self.buf += b

    def flush(self) -> None:
        pass


class _Handler:
    def __init__(self) -> None:
        self.wfile = _WFile()
        self.close_connection = False
        self.status = None
        self.headers_sent: list[tuple[str, str]] = []

    def send_response(self, s: int) -> None:
        self.status = s

    def send_header(self, k: str, v: str) -> None:
        self.headers_sent.append((k, v))

    def end_headers(self) -> None:
        pass


class _Hdrs:
    def __init__(self, pairs):
        self._p = pairs

    def items(self):
        return list(self._p)


class _Resp:
    """A fake upstream SSE response; read1 dribbles at most `chunk` bytes (to exercise
    frame reassembly across boundaries)."""

    def __init__(
        self, sse: bytes, *, status: int = 200, ctype: str = "text/event-stream", chunk=8192
    ):
        self._d = sse
        self._i = 0
        self._chunk = chunk
        self.status = status
        self.headers = _Hdrs([("content-type", ctype)])

    def read1(self, n: int) -> bytes:
        n = min(n, self._chunk)
        out = self._d[self._i : self._i + n]
        self._i += len(out)
        return out


class _Store:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def expand(self, handle: str) -> str:
        self.calls.append(handle)
        return f"ORIGINAL[{handle}]"


def _sender(responses):
    bodies: list[dict] = []
    it = iter(responses)

    def send(body):
        bodies.append(body)
        return next(it)

    return send, bodies


def _evt(**d) -> bytes:
    return b"event: " + d["type"].encode() + b"\ndata: " + json.dumps(d).encode() + b"\n\n"


def _text_response(text: str, msg_id: str = "msg", out_tokens: int = 5) -> bytes:
    return b"".join(
        [
            _evt(
                type="message_start",
                message={
                    "id": msg_id,
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            ),
            _evt(type="content_block_start", index=0, content_block={"type": "text", "text": ""}),
            _evt(type="content_block_delta", index=0, delta={"type": "text_delta", "text": text}),
            _evt(type="content_block_stop", index=0),
            _evt(
                type="message_delta",
                delta={"stop_reason": "end_turn"},
                usage={"output_tokens": out_tokens},
            ),
            _evt(type="message_stop"),
        ]
    )


def _expand_response(handle: str, *, preamble: str | None = None) -> bytes:
    frames = [
        _evt(
            type="message_start",
            message={
                "id": "msg_x",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        )
    ]
    idx = 0
    if preamble is not None:
        frames += [
            _evt(type="content_block_start", index=idx, content_block={"type": "text", "text": ""}),
            _evt(
                type="content_block_delta",
                index=idx,
                delta={"type": "text_delta", "text": preamble},
            ),
            _evt(type="content_block_stop", index=idx),
        ]
        idx += 1
    frames += [
        _evt(
            type="content_block_start",
            index=idx,
            content_block={"type": "tool_use", "id": "tu_1", "name": "distil_expand", "input": {}},
        ),
        _evt(
            type="content_block_delta",
            index=idx,
            delta={"type": "input_json_delta", "partial_json": json.dumps({"handle": handle})},
        ),
        _evt(type="content_block_stop", index=idx),
        _evt(type="message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 7}),
        _evt(type="message_stop"),
    ]
    return b"".join(frames)


# --- tests --------------------------------------------------------------------
def test_passthrough_when_no_expand_call():
    send, bodies = _sender([_Resp(_text_response("Hello world"))])
    h = _Handler()
    st = stream_with_expand(
        h, send, {"messages": [{"role": "user", "content": "hi"}]}, _Store(), hop_by_hop=frozenset()
    )
    out = bytes(h.wfile.buf)
    assert st == 200
    assert len(bodies) == 1  # no re-query
    assert b"Hello world" in out
    assert out.count(b'"type": "message_start"') == 1
    assert b"message_stop" in out


def test_expand_is_intercepted_resolved_and_spliced():
    store = _Store()
    send, bodies = _sender(
        [_Resp(_expand_response("abcd1234")), _Resp(_text_response("The answer is 42"))]
    )
    h = _Handler()
    st = stream_with_expand(
        h, send, {"messages": [{"role": "user", "content": "go"}]}, store, hop_by_hop=frozenset()
    )
    out = bytes(h.wfile.buf)
    assert st == 200
    # the client NEVER sees distil's internal recovery tool or its arg deltas
    assert b"distil_expand" not in out
    assert b"input_json_delta" not in out
    # the client DOES see the spliced answer, as one coherent message
    assert b"The answer is 42" in out
    assert out.count(b'"type": "message_start"') == 1
    assert b"message_stop" in out
    assert out.endswith(b"0\r\n\r\n")  # proper chunked-transfer terminator
    # the handle was resolved and fed back into a re-query
    assert store.calls == ["abcd1234"]
    assert len(bodies) == 2
    requery = bodies[1]
    assert requery["messages"][-2]["role"] == "assistant"  # carries the distil_expand tool_use
    assert requery["messages"][-1]["role"] == "user"  # carries the tool_result
    assert "ORIGINAL[abcd1234]" in json.dumps(requery["messages"][-1])
    # the assistant turn we reconstructed includes the tool call with the parsed handle
    tu = [b for b in requery["messages"][-2]["content"] if b.get("type") == "tool_use"]
    assert tu and tu[0]["input"] == {"handle": "abcd1234"}


def test_expand_with_preamble_text_reindexes_continuation():
    # Model streams visible text, THEN calls distil_expand; the continuation's blocks
    # must re-index to continue after the preamble so the client sees a coherent message.
    store = _Store()
    send, _ = _sender(
        [_Resp(_expand_response("h8", preamble="Let me check. ")), _Resp(_text_response("Done."))]
    )
    h = _Handler()
    stream_with_expand(
        h, send, {"messages": [{"role": "user", "content": "go"}]}, store, hop_by_hop=frozenset()
    )
    out = bytes(h.wfile.buf).decode()
    assert "Let me check. " in out and "Done." in out  # both preamble and answer reach the client
    assert "distil_expand" not in out
    # the continuation's text block was re-indexed to 1 (preamble took index 0)
    assert '"index": 1' in out


def test_survives_sse_frames_split_across_read_boundaries():
    # read1 returns ONE byte at a time — the parser must reassemble frames.
    send, bodies = _sender([_Resp(_text_response("dribble"), chunk=1)])
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, _Store(), hop_by_hop=frozenset())
    assert st == 200
    assert b"dribble" in bytes(h.wfile.buf)


def test_nonstream_first_response_is_relayed_verbatim():
    # A non-SSE (e.g. JSON error) first response is relayed as-is, no interception.
    body = b'{"error":"bad"}'
    send, _ = _sender([_Resp(body, status=400, ctype="application/json")])
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, _Store(), hop_by_hop=frozenset())
    assert st == 400
    assert b'{"error":"bad"}' in bytes(h.wfile.buf)


def test_output_token_usage_is_summed_across_the_splice():
    store = _Store()
    sink: dict[str, int] = {}
    send, _ = _sender([_Resp(_expand_response("h")), _Resp(_text_response("ok", out_tokens=9))])
    h = _Handler()
    stream_with_expand(h, send, {"messages": []}, store, hop_by_hop=frozenset(), usage_sink=sink)
    # 7 (expanding turn) + 9 (answer) accrue; the client's terminal message_delta carries the sum
    out = bytes(h.wfile.buf).decode()
    assert '"output_tokens": 16' in out
    assert sink.get("output_tokens") == 16
    assert sink.get("input_tokens") == 100


def test_max_iters_is_bounded_on_a_model_that_always_expands():
    # A model that calls distil_expand forever must not spin the loop — it's capped.
    store = _Store()

    class _Inf:
        def __init__(self):
            self.n = 0

        def __call__(self, body):
            self.n += 1
            return _Resp(_expand_response(f"h{self.n}"))

    send = _Inf()
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, store, hop_by_hop=frozenset(), max_iters=2)
    assert st == 200
    assert send.n <= 4  # bounded: first response + at most max_iters+1 re-queries
    assert bytes(h.wfile.buf).endswith(b"0\r\n\r\n")  # stream closed cleanly


def test_requery_failure_surfaces_and_closes():
    # If the re-query (after resolving a handle) fails, the stream closes with its status.
    store = _Store()

    class _Err:
        status = 502
        headers = _Hdrs([("content-type", "application/json")])

        def read1(self, n):
            return b""

    responses = iter([_Resp(_expand_response("h")), _Err()])

    def send(body):
        return next(responses)

    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, store, hop_by_hop=frozenset())
    assert st == 502
    assert store.calls == ["h"]  # the handle was resolved before the failed re-query


def _mixed_turn(handle: str) -> bytes:
    """One assistant message carrying BOTH a client tool (Edit) and distil_expand.

    Claude Code emits parallel tool calls, so this shape is routine, not exotic.
    """
    return b"".join(
        [
            _evt(
                type="message_start",
                message={
                    "id": "msg_m",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            ),
            _evt(type="content_block_start", index=0, content_block={"type": "text", "text": ""}),
            _evt(
                type="content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": "Editing now."},
            ),
            _evt(type="content_block_stop", index=0),
            _evt(
                type="content_block_start",
                index=1,
                content_block={"type": "tool_use", "id": "tu_edit", "name": "Edit", "input": {}},
            ),
            _evt(
                type="content_block_delta",
                index=1,
                delta={
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        {"file_path": "/x.py", "old_string": "a", "new_string": "b"}
                    ),
                },
            ),
            _evt(type="content_block_stop", index=1),
            _evt(
                type="content_block_start",
                index=2,
                content_block={
                    "type": "tool_use",
                    "id": "tu_exp",
                    "name": "distil_expand",
                    "input": {},
                },
            ),
            _evt(
                type="content_block_delta",
                index=2,
                delta={"type": "input_json_delta", "partial_json": json.dumps({"handle": handle})},
            ),
            _evt(type="content_block_stop", index=2),
            _evt(
                type="message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 7}
            ),
            _evt(type="message_stop"),
        ]
    )


def _delivered(buf: bytearray) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in bytes(buf).decode().splitlines()
        if line.startswith("data: ")
    ]


def test_mixed_turn_keeps_the_clients_tool_call_executable():
    """A turn with a real tool_use alongside distil_expand must NOT be spliced.

    Splicing answered only the distil_expand block, so the client's Edit became an
    unanswered tool_use upstream (API-contract violation) AND the continuation's
    ``end_turn`` replaced the real ``tool_use`` stop_reason. Claude Code executes
    tools only on ``stop_reason == "tool_use"``, so it ran nothing, said "done",
    and left the disk untouched — the empty-diff failure.
    """
    store = _Store()
    send, bodies = _sender([_Resp(_mixed_turn("abcd1234"))])
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, store, hop_by_hop=frozenset())
    evts = _delivered(h.wfile.buf)

    assert st == 200
    assert len(bodies) == 1, "a mixed turn must not trigger a re-query"
    assert store.calls == [], "the handle must not be consumed on a turn we relay verbatim"

    tools = [
        e["content_block"].get("name")
        for e in evts
        if e.get("type") == "content_block_start" and e["content_block"].get("type") == "tool_use"
    ]
    assert tools == ["Edit"], "the client's tool must be delivered; distil_expand suppressed"

    stops = [e["delta"].get("stop_reason") for e in evts if e.get("type") == "message_delta"]
    assert stops == ["tool_use"], "stop_reason must stay tool_use or the client won't execute"
    assert evts[-1]["type"] == "message_stop"


def test_expand_only_turn_still_splices():
    """The fix must not disable expansion — an expand-ONLY turn still resolves."""
    store = _Store()
    send, bodies = _sender([_Resp(_expand_response("h1")), _Resp(_text_response("The answer"))])
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, store, hop_by_hop=frozenset())

    assert st == 200
    assert store.calls == ["h1"], "expand-only turns must still resolve the handle"
    assert len(bodies) == 2, "expand-only turns must still re-query"
    assert b"The answer" in bytes(h.wfile.buf)


def test_requery_failure_still_terminates_the_sse_message():
    """A failed re-query must not leave a truncated message.

    Without message_delta/message_stop the SDK cannot distinguish "finished" from
    "connection dropped", so it retries invisibly — wall-clock burned, no progress,
    no error the user can see.
    """
    store = _Store()

    class _Err:
        status = 502
        headers = _Hdrs([("content-type", "application/json")])

        def read1(self, n):
            return b""

    send, _ = _sender([_Resp(_expand_response("h")), _Err()])
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, store, hop_by_hop=frozenset())
    evts = _delivered(h.wfile.buf)

    assert st == 502
    assert evts[-1]["type"] == "message_stop", "the SSE message must be terminated, not truncated"


def test_max_iters_exhaustion_terminates_the_sse_message():
    """The runaway-model cap must also close with a proper terminator."""
    store = _Store()

    class _Inf:
        def __init__(self):
            self.n = 0

        def __call__(self, body):
            self.n += 1
            return _Resp(_expand_response(f"h{self.n}"))

    h = _Handler()
    st = stream_with_expand(h, _Inf(), {"messages": []}, store, hop_by_hop=frozenset(), max_iters=2)
    evts = _delivered(h.wfile.buf)
    assert st == 200
    assert evts[-1]["type"] == "message_stop"


def test_unresolvable_handle_still_terminates_the_message(monkeypatch):
    """When nothing resolves there is nothing to splice — end the message cleanly.

    resolve_expands normally returns a miss placeholder (fail-open), so this drives
    the rarer branch where it yields nothing at all.
    """
    import distil.streamexpand as se

    monkeypatch.setattr(se, "resolve_expands", lambda *a, **k: [])
    send, _ = _sender([_Resp(_expand_response("gone"))])
    h = _Handler()
    st = stream_with_expand(h, send, {"messages": []}, _Store(), hop_by_hop=frozenset())
    evts = _delivered(h.wfile.buf)
    assert st == 200
    assert evts[-1]["type"] == "message_stop", "must terminate, never truncate"

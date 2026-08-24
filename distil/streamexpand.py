"""Streaming ``distil_expand`` interception — recoverable digest WITHOUT losing TTFT.

When a request carries a recoverable digest stub, the model may call ``distil_expand``
to pull the elided detail back (see :mod:`distil.expand`). The buffered path handles
that by reading the whole response first, which turns time-to-first-token into
time-to-last-token. This module instead **speculatively streams**: it relays the
model's tokens as they arrive and only intervenes if a ``distil_expand`` call actually
appears — at which point it suppresses that (distil-internal) tool call, resolves the
handle, re-queries upstream, and **splices** the continuation into the SAME client
stream. The agent sees one coherent message and never sees distil's recovery tool;
the common case (no expand call) streams untouched.

Wire shape (Anthropic Messages SSE): a response that calls a tool ends with
``stop_reason: tool_use`` — the answer arrives in the NEXT turn. So an expanding
response is ``[thinking?][text?][distil_expand tool_use]`` then stop; we suppress the
tool_use + that turn's ``message_delta``/``message_stop``, feed the resolved handle
back, and stream the re-query's answer — re-indexing its content blocks to continue
after whatever we already relayed, and suppressing its ``message_start`` so the client
sees a single message.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable

from .expand import EXPAND_TOOL_NAME, record_signal, resolve_expands

_EVENT_SEP = re.compile(rb"\r?\n\r?\n")
_CHUNK = 8192


def _parse_event(raw: bytes) -> dict[str, Any] | None:
    """Extract the JSON ``data:`` payload of one SSE event (None if not JSON/data)."""
    data_parts: list[str] = []
    for line in raw.split(b"\n"):
        s = line.strip()
        if s.startswith(b"data:"):
            data_parts.append(s[5:].strip().decode("utf-8", "replace"))
    if not data_parts:
        return None
    payload = "".join(data_parts)
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _iter_events(read: Callable[[], bytes]) -> Any:
    """Yield parsed SSE event dicts from ``read()`` (which returns b'' at EOF).

    Robust to event frames split across read boundaries: bytes are buffered and only
    emitted on a blank-line separator.
    """
    buf = b""
    while True:
        chunk = read()
        if not chunk:
            break
        buf += chunk
        while True:
            m = _EVENT_SEP.search(buf)
            if not m:
                break
            raw, buf = buf[: m.start()], buf[m.end() :]
            evt = _parse_event(raw)
            if evt is not None:
                yield evt
    if buf.strip():
        evt = _parse_event(buf)
        if evt is not None:
            yield evt


def _frame(evt: dict[str, Any]) -> bytes:
    """Serialize an event dict back to an SSE frame (event: + data: + blank line).

    The ``event:`` line mirrors the payload ``type`` — Anthropic sends both and some
    clients key off the event line."""
    etype = evt.get("type", "message")
    return f"event: {etype}\ndata: {json.dumps(evt)}\n\n".encode()


class _Client:
    """Writes spliced SSE frames to the handler with chunked transfer framing."""

    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._h = handler
        self.next_index = 0  # monotonic client-side content-block index (across splices)
        self._started = False

    def start_once(self) -> bool:
        """True exactly once — only the first upstream message_start reaches the client."""
        if self._started:
            return False
        self._started = True
        return True

    def write(self, evt: dict[str, Any]) -> None:
        chunk = _frame(evt)
        self._h.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
        self._h.wfile.flush()

    def close(self) -> None:
        self._h.wfile.write(b"0\r\n\r\n")
        self._h.wfile.flush()


def stream_with_expand(
    handler: BaseHTTPRequestHandler,
    send: Callable[[dict[str, Any]], Any],
    body: dict[str, Any],
    store: Any,
    *,
    hop_by_hop: frozenset[str],
    extras: dict[str, str] | None = None,
    usage_sink: dict[str, int] | None = None,
    max_iters: int = 4,
    on_signal: Callable[[str, str], None] | None = record_signal,
) -> int:
    """Relay a streaming response, transparently resolving ``distil_expand`` calls.

    ``send(body) -> resp`` opens an upstream SSE stream for ``body`` and returns an
    object exposing ``.status``, ``.headers`` (items()), and ``.read1(n)`` (like an
    ``http.client``/``urllib`` response). Returns the final HTTP status.

    The first upstream response's headers are relayed to the client; on a non-2xx or a
    non-SSE first response the body is relayed unchanged (no interception).
    """
    from .streamrelay import capture_ratelimit

    first = send(body)
    status = getattr(first, "status", 200)
    hdrs = list(first.headers.items())
    # Captured before the splice/verbatim branch so both outcomes record it. Uses the
    # FIRST response deliberately: a request that resolves an expand re-queries upstream,
    # and it is the original turn's quota state we want attributed to this request.
    handler._distil_ratelimit = capture_ratelimit(first.headers)  # type: ignore[attr-defined]
    ctype = next((v for k, v in hdrs if k.lower() == "content-type"), "")
    if not (200 <= status < 300) or "text/event-stream" not in ctype.lower():
        # Not a stream we can splice — relay verbatim (chunked) and return.
        return _relay_raw(handler, first, status, hdrs, hop_by_hop, extras, usage_sink)

    handler.send_response(status)
    for k, v in hdrs:
        if k.lower() not in hop_by_hop and k.lower() != "content-length":
            handler.send_header(k, v)
    for k, v in (extras or {}).items():
        handler.send_header(k, v)
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()

    client = _Client(handler)
    usage_out = 0
    resp = first
    try:
        for _ in range(max_iters + 1):
            blocks, expand_ids, ui_input, index_map, final_delta = {}, set(), {}, {}, None
            saw_expand = False
            saw_client_tool_use = False  # a tool_use the CLIENT must execute

            stream = resp  # bound for the closure; fully consumed before any re-query
            for evt in _iter_events(lambda: stream.read1(_CHUNK)):
                etype = evt.get("type")
                if etype == "message_start":
                    if client.start_once():
                        client.write(evt)  # only the FIRST message_start reaches the client
                        if usage_sink is not None:  # input/cache usage is billed on turn 1
                            u = (evt.get("message") or {}).get("usage") or {}
                            for k in (
                                "input_tokens",
                                "cache_read_input_tokens",
                                "cache_creation_input_tokens",
                            ):
                                if k in u:
                                    usage_sink.setdefault(k, int(u[k] or 0))
                    continue
                if etype == "content_block_start":
                    ui = evt.get("index", 0)
                    cb = dict(evt.get("content_block") or {})
                    blocks[ui] = cb
                    if cb.get("type") == "tool_use" and cb.get("name") == EXPAND_TOOL_NAME:
                        expand_ids.add(ui)
                        saw_expand = True
                        ui_input[ui] = ""
                        continue  # suppress: distil-internal
                    if cb.get("type") == "tool_use":
                        # A tool only the CLIENT can run (Edit, Bash, …). Its presence
                        # makes this turn non-terminal for the agent, which changes how
                        # the turn must end — see the mixed-turn branch below.
                        saw_client_tool_use = True
                    ci = client.next_index
                    client.next_index += 1
                    index_map[ui] = ci
                    client.write({**evt, "index": ci})
                    continue
                if etype == "content_block_delta":
                    ui = evt.get("index", 0)
                    delta = evt.get("delta") or {}
                    _accumulate(blocks.get(ui), ui_input, ui, delta)
                    if ui in expand_ids:
                        continue  # suppress
                    client.write({**evt, "index": index_map.get(ui, ui)})
                    continue
                if etype == "content_block_stop":
                    ui = evt.get("index", 0)
                    if ui in expand_ids:
                        _finalize_input(blocks.get(ui), ui_input.get(ui))
                        continue  # suppress
                    _finalize_input(blocks.get(ui), ui_input.get(ui))
                    client.write({**evt, "index": index_map.get(ui, ui)})
                    continue
                if etype == "message_delta":
                    usage_out += int(((evt.get("usage") or {}).get("output_tokens")) or 0)
                    final_delta = evt  # relayed only if this turn is terminal (no expand)
                    continue
                if etype == "message_stop":
                    break  # end of this upstream message; decide below
                # ping / unknown keepalive — relay to keep the connection warm
                client.write(evt)

            if not saw_expand or saw_client_tool_use:
                # Terminal turn: emit the (usage-corrected) message_delta + message_stop.
                #
                # ``saw_client_tool_use`` makes a MIXED turn terminal too. Claude Code
                # emits parallel tool calls, so one assistant message can carry both an
                # Edit and a distil_expand. Splicing that turn was wrong twice over:
                #   * the replay answered only the distil_expand block, leaving the Edit
                #     an unanswered tool_use (an API-contract violation), and
                #   * the continuation's ``stop_reason: end_turn`` overwrote the real
                #     ``tool_use``, so the client never executed the Edit it had been
                #     handed — the agent reported success with the disk untouched.
                # Relaying verbatim keeps the client's tool call executable; the model
                # re-requests the handle on its next turn, where the turn is expand-only
                # and the loop below can serve it safely.
                if final_delta is not None:
                    client.write(_with_output_tokens(final_delta, usage_out))
                client.write({"type": "message_stop"})
                client.close()
                if usage_sink is not None:
                    usage_sink.setdefault("output_tokens", usage_out)
                return status

            # Expanding turn: resolve handles, re-query, splice the continuation.
            resp_dict = {"content": [blocks[i] for i in sorted(blocks)]}
            results = resolve_expands(resp_dict, store, on_signal=on_signal)
            if not results:  # nothing resolvable — stop cleanly rather than loop
                _close_stream(client, final_delta, usage_out, usage_sink)
                return status
            body = {
                **body,
                "messages": [
                    *(body.get("messages") or []),
                    {"role": "assistant", "content": resp_dict["content"]},
                    {"role": "user", "content": results},
                ],
            }
            resp = send(body)
            if not (200 <= getattr(resp, "status", 200) < 300):
                # Re-query failed mid-stream. Terminate the SSE message properly before
                # closing: a stream that ends without message_delta/message_stop is a
                # TRUNCATED message to the SDK, which retries it invisibly — wall-clock
                # burned with no progress and no error the user can see. Fail open to
                # what was already relayed.
                _close_stream(client, final_delta, usage_out, usage_sink)
                return getattr(resp, "status", 502)

        # max_iters exhausted — close the stream (a misbehaving model can't spin forever)
        _close_stream(client, final_delta, usage_out, usage_sink)
        return status
    except OSError:
        handler.close_connection = True
        return status


def _accumulate(
    block: dict[str, Any] | None, ui_input: dict[int, str], ui: int, delta: dict[str, Any]
) -> None:
    """Fold one content_block_delta into the reconstructed block (for the re-query)."""
    if block is None:
        return
    dtype = delta.get("type")
    if dtype == "text_delta":
        block["text"] = block.get("text", "") + (delta.get("text") or "")
    elif dtype == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + (delta.get("thinking") or "")
    elif dtype == "signature_delta":
        block["signature"] = block.get("signature", "") + (delta.get("signature") or "")
    elif dtype == "input_json_delta":
        ui_input[ui] = ui_input.get(ui, "") + (delta.get("partial_json") or "")


def _finalize_input(block: dict[str, Any] | None, buf: str | None) -> None:
    """Parse a tool_use's accumulated input_json_delta buffer into its ``input`` object."""
    if block is None or block.get("type") != "tool_use":
        return
    try:
        block["input"] = json.loads(buf) if buf and buf.strip() else block.get("input") or {}
    except (ValueError, TypeError):
        block["input"] = block.get("input") or {}


def _with_output_tokens(delta_evt: dict[str, Any], output_tokens: int) -> dict[str, Any]:
    """Return the terminal message_delta with usage.output_tokens set to the splice total."""
    usage = {**(delta_evt.get("usage") or {}), "output_tokens": output_tokens}
    return {**delta_evt, "usage": usage}


def _close_stream(
    client: "_Client",
    final_delta: dict[str, Any] | None,
    usage_out: int,
    usage_sink: dict[str, int] | None,
) -> None:
    """End the SSE message properly, then close.

    Every exit from the expand loop must go through here. An SSE message that ends
    without ``message_delta``/``message_stop`` is *truncated*, not finished: the SDK
    cannot tell it from a dropped connection, so it retries — silently burning
    wall-clock while the agent makes no progress. Emitting the terminator turns an
    invisible hang into a completed (possibly short) turn the agent can act on.
    """
    if final_delta is not None:
        client.write(_with_output_tokens(final_delta, usage_out))
    client.write({"type": "message_stop"})
    client.close()
    if usage_sink is not None:
        usage_sink.setdefault("output_tokens", usage_out)


def _relay_raw(
    handler: BaseHTTPRequestHandler,
    resp: Any,
    status: int,
    hdrs: list[tuple[str, str]],
    hop_by_hop: frozenset[str],
    extras: dict[str, str] | None,
    usage_sink: dict[str, int] | None,
) -> int:
    """Verbatim chunked relay for a first response we can't/shouldn't splice."""
    handler.send_response(status)
    length = next((v for k, v in hdrs if k.lower() == "content-length"), None)
    for k, v in hdrs:
        if k.lower() not in hop_by_hop and k.lower() != "content-length":
            handler.send_header(k, v)
    for k, v in (extras or {}).items():
        handler.send_header(k, v)
    if length is None:
        handler.send_header("Transfer-Encoding", "chunked")
    else:
        handler.send_header("Content-Length", length)
    handler.end_headers()
    try:
        while True:
            chunk = resp.read1(_CHUNK)
            if not chunk:
                break
            if length is None:
                handler.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            else:
                handler.wfile.write(chunk)
            handler.wfile.flush()
        if length is None:
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
    except OSError:
        handler.close_connection = True
    return status

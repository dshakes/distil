"""Every upstream call distil makes on the user's key is billed, recorded and netted.

Before 1.54 the per-request ledger kept only ONE call's usage per client request: the
first (streaming splice) or the last (buffered expand loops), and nothing at all for
Chat Completions or Gemini. So a request that ran a ``distil_expand`` re-query, or was
sampled for shadow replays, looked cheaper than it was, and every savings figure
leaned toward distil. Each test here fails on that code.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from distil.proxy import billed_input_equiv, build_handler, requery_input_equiv
from distil.runtime import RuntimeSavings
from distil.streamexpand import stream_with_expand
from distil.streamrelay import scan_usage
from tests.test_streamexpand import _evt, _Handler, _Resp, _sender, _Store

_LONG = "\n".join(f"line {i}: verbose tool output the digest folds away {i * 7}" for i in range(60))
_HANDLE = re.compile(r"handle=([0-9a-f]{6,})")

# opus-4-8: $5 in, $25 out, read 0.1x, write 1.25x (per Mtok)
FIRST = {"input_tokens": 1000, "cache_read_input_tokens": 5000, "output_tokens": 20}
REQUERY = {"input_tokens": 1200, "cache_read_input_tokens": 5000, "output_tokens": 30}
REQUERY_EQUIV = round((1200 * 5 + 5000 * 0.5 + 30 * 25) / 5)  # 1850 base-input tokens


# --------------------------------------------------------------------------- #
# Streaming splice (Anthropic SSE)
# --------------------------------------------------------------------------- #


def _msg(usage: dict, blocks: list[bytes], stop: str, out: int) -> bytes:
    return b"".join(
        [
            _evt(
                type="message_start",
                message={"id": "m", "role": "assistant", "content": [], "usage": usage},
            ),
            *blocks,
            _evt(type="message_delta", delta={"stop_reason": stop}, usage={"output_tokens": out}),
            _evt(type="message_stop"),
        ]
    )


def test_streaming_splice_sums_every_upstream_message():
    call = [
        _evt(
            type="content_block_start",
            index=0,
            content_block={"type": "tool_use", "id": "t", "name": "distil_expand", "input": {}},
        ),
        _evt(
            type="content_block_delta",
            index=0,
            delta={"type": "input_json_delta", "partial_json": '{"handle": "abc123"}'},
        ),
        _evt(type="content_block_stop", index=0),
    ]
    text = [
        _evt(type="content_block_start", index=0, content_block={"type": "text", "text": ""}),
        _evt(type="content_block_delta", index=0, delta={"type": "text_delta", "text": "ok"}),
        _evt(type="content_block_stop", index=0),
    ]
    in1 = {k: v for k, v in FIRST.items() if k != "output_tokens"}
    in2 = {
        **{k: v for k, v in REQUERY.items() if k != "output_tokens"},
        "cache_creation_input_tokens": 40,
    }
    send, _ = _sender(
        [_Resp(_msg(in1, call, "tool_use", 20)), _Resp(_msg(in2, text, "end_turn", 30))]
    )
    sink: dict[str, int] = {}
    stream_with_expand(
        _Handler(), send, {"messages": [{"role": "user", "content": "hi"}]}, _Store(),
        hop_by_hop=frozenset(), usage_sink=sink,
    )  # fmt: skip
    assert sink["input_tokens"] == 2200  # was 1000: the re-query's input was dropped
    assert sink["cache_read_input_tokens"] == 10_000
    assert sink["cache_creation_input_tokens"] == 40
    assert sink["output_tokens"] == 50
    assert sink["requeries"] == 1
    assert (sink["requery_input_tokens"], sink["requery_output_tokens"]) == (1200, 30)


# --------------------------------------------------------------------------- #
# Buffered expand loops, through the real proxy (Anthropic, Chat, Responses)
# --------------------------------------------------------------------------- #


def _responses(shape: str, handle: str | None) -> dict:
    """The fake upstream's reply: an expand call when *handle* is given, else a final."""
    usage = REQUERY if handle is None else FIRST
    if shape == "anthropic":
        content = (
            [
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "distil_expand",
                    "input": {"handle": handle},
                }
            ]
            if handle
            else [{"type": "text", "text": "done"}]
        )
        return {"type": "message", "role": "assistant", "content": content, "usage": usage}
    # OpenAI shapes report prompt/completion (Chat) or input/output (Responses); Chat's
    # prompt_tokens INCLUDE cached ones, so the same billed counts are spelled this way.
    cached = usage["cache_read_input_tokens"]
    if shape == "chat":
        msg: dict = {"role": "assistant", "content": None if handle else "done"}
        if handle:
            msg["tool_calls"] = [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "distil_expand",
                        "arguments": json.dumps({"handle": handle}),
                    },
                }
            ]
        return {
            "choices": [
                {"index": 0, "message": msg, "finish_reason": "tool_calls" if handle else "stop"}
            ],
            "usage": {
                "prompt_tokens": usage["input_tokens"] + cached,
                "completion_tokens": usage["output_tokens"],
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    out = (
        [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "distil_expand",
                "arguments": json.dumps({"handle": handle}),
            }
        ]
        if handle
        else [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ]
    )
    return {
        "output": out,
        "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]},
    }


def _upstream(shape: str):
    class _Up(BaseHTTPRequestHandler):
        calls: list[bytes] = []

        def log_message(self, fmt, *args):  # noqa: ARG002
            pass

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            _Up.calls.append(body)
            m = _HANDLE.search(body.decode("utf-8", "replace"))
            first = len(_Up.calls) == 1
            out = json.dumps(_responses(shape, m.group(1) if (first and m) else None)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    return _Up


def _request(shape: str) -> tuple[str, dict]:
    if shape == "anthropic":
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _LONG}],
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and now"},
        ]
        return "/v1/messages", {"model": "claude-opus-4-8", "max_tokens": 64, "messages": msgs}
    if shape == "chat":
        call = {"id": "t1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        msgs = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "t1", "content": _LONG},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and now"},
        ]
        return "/v1/chat/completions", {"model": "claude-opus-4-8", "messages": msgs}
    items = [
        {"role": "user", "content": "go"},
        {"type": "function_call", "call_id": "t1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "t1", "output": _LONG},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and now"},
    ]
    return "/v1/responses", {"model": "claude-opus-4-8", "input": items}


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _roundtrip(shape: str, handler_kwargs: dict, tmp_path: Path):
    up_cls = _upstream(shape)
    up = _serve(up_cls)
    ledger = tmp_path / "savings.jsonl"
    savings = RuntimeSavings(ledger_path=ledger)
    handler = build_handler(
        f"http://127.0.0.1:{up.server_address[1]}", savings=savings, **handler_kwargs
    )
    proxy = _serve(handler)
    try:
        path, payload = _request(shape)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            r.read()
        from distil.proxy import _drain_shadow

        _drain_shadow(handler)
    finally:
        proxy.shutdown()
        up.shutdown()
    savings.flush()
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    rec_path = Path(os.environ["DISTIL_HOME"]) / "sessions" / "s7-7.requests.jsonl"
    recs = [json.loads(line) for line in rec_path.read_text().splitlines()]
    return up_cls.calls, recs, rows


@pytest.mark.parametrize("shape", ["anthropic", "chat", "responses"])
def test_buffered_expand_records_and_nets_every_call(shape, monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_SESSION", "s7-7")
    calls, recs, rows = _roundtrip(shape, {}, tmp_path)
    assert len(calls) == 2, "the fixture must make the model expand once"
    rec = recs[-1]
    assert rec["upstream_calls"] == 2
    if shape == "responses":  # Responses reports no cache split; input is inclusive
        assert rec["usage_input_tokens"] == 2200
    else:
        assert rec["usage_input_tokens"] == 2200  # was 1200 (last call) or None (Chat)
        assert rec["usage_cache_read"] == 10_000
    assert rec["usage_output_tokens"] == 50
    assert rec["expand_requery_usage"]["output_tokens"] == 30
    # The savings ledger nets the re-query out: distil side = compressed + re-query cost.
    (row,) = rows
    equiv = requery_input_equiv(
        {"requeries": 1, **{"requery_" + k: v for k, v in rec["expand_requery_usage"].items()}},
        "claude-opus-4-8",
    )
    if shape != "responses":
        assert equiv == REQUERY_EQUIV
    compressed = rec["compressible_tokens"] - rec["tokens_saved"]
    assert row["distil_input_tokens"] == compressed + equiv
    assert row["baseline_input_tokens"] == rec["compressible_tokens"]


def test_shadow_replays_are_netted_and_attributed(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_SESSION", "s7-7")
    calls, recs, rows = _roundtrip("anthropic", {"shadow_rate": 1.0}, tmp_path)
    replays = len(calls) - 2  # 2 = the served request + its expand re-query
    assert replays >= 2
    over = [
        json.loads(line)
        for line in (Path(os.environ["DISTIL_HOME"]) / "sessions" / "s7-7.overhead.jsonl")
        .read_text()
        .splitlines()
    ]
    assert over and over[0]["kind"] == "shadow" and over[0]["calls"] == replays
    booked_extra = sum(r["distil_input_tokens"] for r in rows) - (
        recs[-1]["compressible_tokens"] - recs[-1]["tokens_saved"]
    )
    # The replays' replies are "final" answers (REQUERY usage) in this fixture.
    assert booked_extra == REQUERY_EQUIV + replays * REQUERY_EQUIV


def test_scan_usage_reads_chat_and_gemini_in_the_anthropic_convention():
    chat = json.dumps(
        {
            "usage": {
                "prompt_tokens": 900,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 600},
            }
        }
    ).encode()
    assert scan_usage(chat) == {
        "input_tokens": 300,
        "cache_read_input_tokens": 600,
        "output_tokens": 40,
    }
    gem = json.dumps(
        {
            "usageMetadata": {
                "promptTokenCount": 500,
                "candidatesTokenCount": 9,
                "cachedContentTokenCount": 100,
            }
        }
    ).encode()
    assert scan_usage(gem) == {
        "input_tokens": 400,
        "cache_read_input_tokens": 100,
        "output_tokens": 9,
    }
    anth = json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2}}).encode()
    assert scan_usage(anth) == {"input_tokens": 5, "output_tokens": 2}


def test_input_equivalents():
    assert requery_input_equiv({}, "claude-opus-4-8") == 0
    assert (
        billed_input_equiv({"input_tokens": 10, "cache_read_input_tokens": 100}, "gemini-x") == 110
    )
    assert billed_input_equiv({"output_tokens": 1}, "claude-opus-4-8") == 5
    # 1-hour cache writes bill at 2x, 5-minute at 1.25x; the split is capped at the total.
    assert billed_input_equiv({"cache_creation_input_tokens": 100}, "claude-opus-4-8") == 125
    w = {"cache_creation_input_tokens": 100, "ephemeral_1h_input_tokens": 100}
    assert billed_input_equiv(w, "claude-opus-4-8") == 200
    w["ephemeral_1h_input_tokens"] = 40
    assert billed_input_equiv(w, "claude-opus-4-8") == 155
    w["ephemeral_1h_input_tokens"] = 999
    assert billed_input_equiv(w, "claude-opus-4-8") == 200
    r = {"requery_cache_creation_input_tokens": 100, "requery_ephemeral_1h_input_tokens": 100}
    assert billed_input_equiv(r, "claude-opus-4-8", prefix="requery_") == 200


def test_zero_baseline_spend_is_flushed_not_dropped(tmp_path):
    s = RuntimeSavings(ledger_path=tmp_path / "l.jsonl")
    s.record(0, 500, model="claude-opus-4-8")
    assert s.flush()
    (row,) = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert (row["baseline_input_tokens"], row["distil_input_tokens"]) == (0, 500)

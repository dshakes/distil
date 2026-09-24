"""Contract tests: OpenAI opaque items survive distil untouched.

The OpenAI Responses API has the same class of "clients must return this byte-
identical" surface Anthropic's server-side compaction has (see
``tests/test_compaction_passthrough.py``), spread across a few item types instead
of one:

* a ``reasoning`` item's ``encrypted_content`` (stateless mode / ZDR) — the
  provider re-derives the model's reasoning from it on the next turn; editing it
  is meaningless (it is not text) and the provider is not obligated to accept
  altered opaque bytes.
* a ``compaction`` item, returned by the standalone ``POST /v1/responses/compact``
  endpoint (and inline via ``context_management: [{"type": "compaction", ...}]``
  on a regular Responses create call) — OpenAI's own docs are explicit: "do not
  prune /responses/compact output. The returned window is the canonical next
  context window, so pass it into your next /responses call as-is."
* the ``context_management`` request parameter and ``previous_response_id`` —
  request-side fields that select/anchor server state and must reach upstream
  unaltered.
* the ``/v1/responses/compact`` endpoint itself — a distinct path from
  ``/v1/responses`` that distil's adapters have never seen and must not attempt
  to parse as a compressible body.

Unlike Anthropic's ``compaction`` content block (which sits inside a Messages
``content`` list), these are top-level ``input`` *items* in the Responses API
array — dispatch in ``distil.adapters.openai._compress_response_item`` is by
``item["type"]``, so the same "digest a tool result, never an opaque item" split
applies, but the shape being asserted against differs from the sibling suite.

Two more invariants this suite pins, mirroring the Anthropic one exactly:
  * a ``function_call_output`` item next to (before or after) an opaque item must
    still digest normally — an opaque item must not accidentally halt or exempt
    what surrounds it.
  * an opaque item's own payload must never be mistaken for a digestible
    ``function_call_output`` — dispatch is by ``type``, not by duck-typing an
    ``output``/``encrypted_content`` key.

Chat Completions has no equivalent surface: OpenAI's reasoning/compaction items
are Responses-API-only (per developers.openai.com/api/docs/guides/compaction and
.../guides/reasoning) — a Chat Completions ``messages`` list carries no analogous
opaque item, so there is nothing to pin on that path beyond "still routes to the
OpenAI adapter, unaffected by this change" (already covered in
``tests/test_openai_adapter.py::TestProxyDispatch``).
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from distil.adapters.anthropic import take_census
from distil.adapters.openai import compress_responses_input, count_responses_tokens
from distil.expand import EXPAND_TOOL_NAME, run_expand_loop_responses
from distil.httpguard import is_compressible_path, is_responses_path
from distil.proxy import build_handler
from distil.streamexpand import sse_from_response

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LONG_TOOL_OUTPUT = "\n".join(
    [
        "Result from bash tool execution on the remote host:",
        "total disk usage: 48 GB across 12 partitions",
        "filesystem /dev/sda1: 32 GB used of 100 GB available",
        "filesystem /dev/sdb1: 16 GB used of 200 GB available",
        "warning: /tmp is 89% full — consider cleaning up old build artefacts",
        "warning: inode count on /var/log approaching limit (91% used)",
        "no errors detected in kernel ring buffer",
        "last boot: 2026-06-20T03:14:22Z (uptime 18h 42m)",
    ]
)  # 8 lines — above the 6-line digest threshold

# Two trailing pad items push a function_call_output out of the recency-exempt
# window (RECENCY_KEEP_TURNS = 2) so it is eligible for Tier-1 digestion.
_PAD = [
    {"type": "function_call_output", "call_id": "pad1", "output": "next"},
    {"type": "function_call_output", "call_id": "pad2", "output": "next"},
]


def _reasoning_item(**extra: Any) -> dict[str, Any]:
    """A minimal but realistic stateless-mode ``reasoning`` item."""
    return {
        "type": "reasoning",
        "id": "rs_abc123",
        "encrypted_content": "b64:opaque-provider-bytes-do-not-touch==",
        "summary": [],
        **extra,
    }


def _compaction_item(**extra: Any) -> dict[str, Any]:
    """A minimal but realistic ``compaction`` item, the shape ``/v1/responses/compact``
    returns and clients must feed back into ``input`` unchanged."""
    return {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": "b64:opaque-compacted-window-do-not-touch==",
        **extra,
    }


def _fco(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _has_handle(text: str) -> bool:
    return isinstance(text, str) and "handle=" in text


# ---------------------------------------------------------------------------
# compress_responses_input: opaque items are byte-equivalent
# ---------------------------------------------------------------------------


class TestOpaqueItemsImmutable:
    def test_reasoning_item_passthrough_identical_object_verbatim(self) -> None:
        item = _reasoning_item()
        items = [item, *_PAD]
        out, _store = compress_responses_input(items, verbatim=True)
        assert out[0] is item  # not merely equal — untouched

    def test_reasoning_item_passthrough_identical_object_digest_mode(self) -> None:
        item = _reasoning_item()
        items = [item, *_PAD]
        out, _store = compress_responses_input(items, verbatim=False)
        assert out[0] is item

    def test_compaction_item_passthrough_identical_object_digest_mode(self) -> None:
        item = _compaction_item()
        items = [item, *_PAD]
        out, _store = compress_responses_input(items, verbatim=False)
        assert out[0] is item

    def test_encrypted_content_bytes_exact_including_unicode(self) -> None:
        # encrypted_content is opaque provider bytes; anything distil does to it
        # (case-fold, whitespace-normalise, re-encode) breaks the provider's own
        # replay check on the next turn.
        blob = "AQIDsig==é☃\n\t"
        item = _compaction_item(encrypted_content=blob)
        items = [item, *_PAD]
        out, _store = compress_responses_input(items, verbatim=False)
        got = out[0]
        assert got["encrypted_content"] == blob
        assert json.dumps(got, sort_keys=True) == json.dumps(item, sort_keys=True)

    def test_not_recorded_in_restore_store(self) -> None:
        # An opaque item must never become a digest handle — it has no marker to
        # digest FROM (it is not a function_call_output) and nothing must ever
        # claim to "recover" provider-pinned bytes via distil_expand.
        item = _compaction_item()
        items = [item, *_PAD]
        _out, store = compress_responses_input(items, verbatim=False)
        assert store.handles == frozenset()

    def test_unknown_future_opaque_type_with_encrypted_content_passthrough(self) -> None:
        # Not every opaque item type is named "reasoning"/"compaction" forever — the
        # guard generalises on `encrypted_content`, so a future item type is safe by
        # construction rather than requiring an updated allowlist.
        item = {"type": "future_opaque_item", "id": "x1", "encrypted_content": "z" * 500}
        items = [item, *_PAD]
        out, _store = compress_responses_input(items, verbatim=False)
        assert out[0] is item


# ---------------------------------------------------------------------------
# Surrounding function_call_output items still compress
# ---------------------------------------------------------------------------


class TestSurroundingContentStillCompresses:
    def test_function_call_output_after_opaque_item_still_digests(self) -> None:
        items = [_reasoning_item(), _fco("c1", LONG_TOOL_OUTPUT), *_PAD]
        out, store = compress_responses_input(items, verbatim=False)
        assert out[0] is items[0]  # reasoning untouched
        assert _has_handle(out[1]["output"])
        assert store.handles

    def test_function_call_output_before_opaque_item_still_digests(self) -> None:
        items = [_fco("c1", LONG_TOOL_OUTPUT), _compaction_item(), *_PAD]
        out, _store = compress_responses_input(items, verbatim=False)
        assert _has_handle(out[0]["output"])
        assert out[1] is items[1]

    def test_function_call_output_in_later_item_still_digests(self) -> None:
        reasoning = _reasoning_item()
        items = [reasoning, _fco("c1", LONG_TOOL_OUTPUT), *_PAD]
        out, _store = compress_responses_input(items, verbatim=False)
        assert out[0] is reasoning
        assert _has_handle(out[1]["output"])


# ---------------------------------------------------------------------------
# An opaque item's own payload is never digested as if it were a function_call_output
# ---------------------------------------------------------------------------


class TestOpaqueItemsNeverDigestedAsToolResult:
    def test_large_encrypted_content_not_folded_or_digested(self) -> None:
        # Shaped to trip the digest gate a function_call_output would trip
        # (>= 6 lines of content), and dispatch must still be by `type`.
        big = "\n".join(f"compacted turn {i}: did something" for i in range(20))
        item = _compaction_item(encrypted_content=big)
        items = [item, *_PAD]
        out, store = compress_responses_input(items, verbatim=False)
        assert out[0] is item
        assert out[0]["encrypted_content"] == big
        assert store.handles == frozenset()

    def test_reasoning_item_without_output_key_never_mistaken_for_function_call_output(
        self,
    ) -> None:
        # A reasoning item has no `output` key at all — confirms dispatch never
        # falls through to the function_call_output branch by duck-typing.
        item = _reasoning_item()
        assert "output" not in item
        items = [item, *_PAD]
        out, store = compress_responses_input(items, verbatim=False)
        assert out[0] is item
        assert store.handles == frozenset()


# ---------------------------------------------------------------------------
# Census: opaque items are billed and counted (mirrors thinking/compaction on the
# Anthropic adapter — this is the equivalent gap the task asked to check for)
# ---------------------------------------------------------------------------


class TestOpaqueItemCensus:
    def test_reasoning_tokens_are_censused_not_invisible(self) -> None:
        item = _reasoning_item(encrypted_content="x" * 4000)
        items = [item, *_PAD]
        compress_responses_input(items, verbatim=False)
        census = take_census() or {}
        assert census.get("reasoning_billed", 0) > 0

    def test_compaction_tokens_are_censused_not_invisible(self) -> None:
        item = _compaction_item(encrypted_content="y" * 4000)
        items = [item, *_PAD]
        compress_responses_input(items, verbatim=False)
        census = take_census() or {}
        assert census.get("compaction_billed", 0) > 0

    def test_unknown_opaque_type_censused_under_generic_bucket(self) -> None:
        item = {"type": "future_opaque_item", "id": "x1", "encrypted_content": "z" * 4000}
        items = [item, *_PAD]
        compress_responses_input(items, verbatim=False)
        census = take_census() or {}
        assert census.get("signed_item_billed", 0) > 0


# ---------------------------------------------------------------------------
# Request-side opaque fields: context_management / previous_response_id
# ---------------------------------------------------------------------------


class _HeaderCapturingEcho(BaseHTTPRequestHandler):
    """Fake upstream: echoes the POST body back."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _run_proxy_echo() -> tuple[HTTPServer, HTTPServer, threading.Thread, threading.Thread]:
    upstream = HTTPServer(("127.0.0.1", 0), _HeaderCapturingEcho)
    up_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    up_thread.start()
    up_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    handler_cls = build_handler(up_url)
    proxy = HTTPServer(("127.0.0.1", 0), handler_cls)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    return upstream, proxy, up_thread, proxy_thread


def test_proxy_preserves_context_management_and_previous_response_id() -> None:
    upstream, proxy, _ut, _pt = _run_proxy_echo()
    try:
        context_management = [{"type": "compaction", "compact_threshold": 200_000}]
        reasoning = _reasoning_item()
        payload = {
            "model": "gpt-6-astra",
            "previous_response_id": "resp_abc123",
            "context_management": context_management,
            "input": [
                reasoning,
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": LONG_TOOL_OUTPUT,
                },
                *_PAD,
            ],
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/v1/responses",
            data=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            saved_hdr = resp.headers.get("x-distil-tokens-saved")
            echoed_body: dict[str, Any] = json.loads(resp.read())
    finally:
        proxy.shutdown()
        upstream.shutdown()

    # Real compression still happened on the surrounding function_call_output.
    assert saved_hdr is not None and int(saved_hdr) > 0
    # Request-side opaque fields reached upstream byte-identical.
    assert echoed_body["previous_response_id"] == "resp_abc123"
    assert echoed_body["context_management"] == context_management
    # The reasoning item itself reached upstream byte-identical.
    assert echoed_body["input"][0] == reasoning


# ---------------------------------------------------------------------------
# /v1/responses/compact is a distinct endpoint and must never be compressed
# ---------------------------------------------------------------------------


class TestCompactEndpointNeverCompressed:
    def test_is_responses_path_excludes_compact(self) -> None:
        # is_responses_path only matches the create endpoint — /compact is a
        # sibling path, not a query-string variant of it.
        assert is_responses_path("/v1/responses") is True
        assert is_responses_path("/v1/responses/compact") is False
        assert is_compressible_path("/v1/responses/compact") is False

    def test_proxy_forwards_compact_request_byte_identical(self) -> None:
        upstream, proxy, _ut, _pt = _run_proxy_echo()
        try:
            payload = {
                "conversation": "conv_1",
                "store": False,
                "input": [_reasoning_item(), _compaction_item()],
            }
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/v1/responses/compact",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert "x-distil-compressed" not in resp.headers
                echoed_raw = resp.read()
        finally:
            proxy.shutdown()
            upstream.shutdown()
        # Byte-identical, not merely JSON-equal: the compact endpoint is never
        # parsed or re-serialized by distil's adapters at all.
        assert echoed_raw == body


# ---------------------------------------------------------------------------
# Expand loop: opaque items survive a distil_expand re-query untouched
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._m = mapping

    def expand(self, handle: str) -> str:
        return self._m[handle]


def test_expand_requery_forwards_opaque_items_byte_identical() -> None:
    """The re-query body run_expand_loop_responses builds must carry every
    original opaque item forward unchanged — the same "clients send it back
    in later requests" contract the Anthropic streaming splice test pins."""
    reasoning = _reasoning_item()
    compaction = _compaction_item()
    store = _Store({"deadbeef": "the recovered original text"})
    body = {"input": [reasoning, compaction, _fco("c1", "short")]}
    first_response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": EXPAND_TOOL_NAME,
                "arguments": json.dumps({"handle": "deadbeef"}),
            }
        ]
    }
    posts: list[dict[str, Any]] = []

    def post(b: dict[str, Any]) -> dict[str, Any]:
        posts.append(b)
        return {"output": [{"type": "message", "role": "assistant", "content": []}]}

    run_expand_loop_responses(body, first_response, store, post)
    assert posts, "the expand call must trigger exactly one re-query"
    requery_input = posts[0]["input"]
    assert requery_input[0] is reasoning
    assert requery_input[1] is compaction


def test_sse_from_response_preserves_opaque_items_in_output_whole() -> None:
    """sse_from_response wraps the already-complete response verbatim — an opaque
    item in `output` (e.g. a reasoning item echoed back, or the automatic
    compaction item server-side compaction appends) is never touched."""
    reasoning = _reasoning_item()
    resp = {"id": "resp_1", "output": [reasoning, {"type": "message", "content": []}]}
    raw = sse_from_response("responses", resp)
    text = raw.decode()
    assert json.dumps(reasoning) in text or json.dumps(reasoning, separators=(",", ": ")) in text
    # The final response.completed frame carries the object back unmodified.
    completed_line = next(
        line for line in text.split("\n\n") if '"type": "response.completed"' in line
    )
    payload = json.loads(completed_line.split("data: ", 1)[1])
    assert payload["response"]["output"][0] == reasoning


# ---------------------------------------------------------------------------
# count_responses_tokens: reasoning/compaction are outside the compressible-zone
# baseline by the same design choice that already excludes assistant/function_call
# (documented in TestCountResponsesTokens.test_excludes_assistant_and_function_call
# in tests/test_openai_adapter.py) — pinned here so a future change to that
# baseline notices it also touches this contract.
# ---------------------------------------------------------------------------


def test_count_responses_tokens_excludes_opaque_items_by_design() -> None:
    items = [_reasoning_item(encrypted_content="x" * 5000), _compaction_item()]
    assert count_responses_tokens(items) == 0

#!/usr/bin/env python3
"""Compression proxy for the SWE-bench end-to-end eval (Phase 5 / E7).

A thin Anthropic Messages API proxy that sits between the coding agent (aider) and
``api.anthropic.com``. For every ``/v1/messages`` request it optionally rewrites the
large *context* blocks — the file contents and tool/command output the agent reads —
with a pluggable per-block compressor, then forwards the request upstream unchanged in
every other respect. Responses stream straight back.

Three eval conditions are realised by the choice of ``compressor``:

* **A. full**      — ``compressor=None``; the proxy is a transparent pass-through
  (still used so all three conditions share identical network/accounting code).
* **B. distil**    — ``compressor=trunc_500``; distil's phase-2 certifying operating
  point ``trunc@500`` (head-truncate each compressible block to its first 500
  characters), i.e. :func:`distil.conformal._truncate_level` ``(500)`` applied to the
  agent's context blocks.
* **C. llmlingua2** — ``compressor=llmlingua2``; LLMLingua-2 at its default keep-rate,
  the strongest non-distil non-truncation baseline (E5).

What counts as a *compressible context block*
--------------------------------------------
The agent's reasoning (assistant turns) and the harness/system instructions must never
be compressed — only "the file contents the agent reads" (task scope, step 3). We
therefore compress a content block iff **all** of:

* it is a ``text`` or ``tool_result`` block (not ``tool_use``, ``image``, ...);
* it lives in a **non-system, non-assistant** message (user/tool turns carry the file
  reads and command output);
* its text is at least :data:`MIN_CHARS` characters (short blocks are instructions or
  glue, and compressing them neither helps nor is in scope).

Both B and C use this *same* selector, so the only thing that differs between them is
the compressor — an apples-to-apples comparison at the block level.

Token accounting
----------------
Every request records (pre, post) input-token counts via distil's tokenizer plus the
upstream-reported ``usage`` (input/output tokens) so the driver can compute real dollar
cost per condition. Accounting is append-only JSONL, survives interruption, and never
contains prompt text.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

# distil's billing-grade-ish tokenizer (stdlib heuristic; consistent across conditions).
import sys as _sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from distil.tokenizer import DEFAULT as _tokenizer  # noqa: E402

UPSTREAM = "https://api.anthropic.com"
MIN_CHARS = 500  # blocks shorter than this are glue/instructions, not file context

# A compressor maps an original block string -> compressed string. Pure and deterministic.
Compressor = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Compressors (the three conditions)
# --------------------------------------------------------------------------- #
def trunc_500(text: str) -> str:
    """distil's phase-2 certifying operating point: head-truncate to 500 chars.

    Mirrors ``distil.conformal._truncate_level(500)`` (``b.text[:limit]``) exactly —
    the only difference is that here the "block" is an agent context block rather than
    a localization-corpus trajectory block.
    """
    return text[:500]


_LLMLINGUA = None


def _llmlingua_compressor():
    global _LLMLINGUA
    if _LLMLINGUA is None:
        # Reuse the exact benchmark adapter (rate 0.5, CPU) so E7's llmlingua-2 matches
        # E5's. Imported lazily — heavy torch + 560MB model, only condition C needs it.
        from benchmarks import llmlingua_adapter

        _LLMLINGUA = llmlingua_adapter
    return _LLMLINGUA


def llmlingua2(text: str) -> str:
    """LLMLingua-2 at the benchmark adapter's default rate (0.5). Memoised per text."""
    adapter = _llmlingua_compressor()
    return adapter.compress([text])[0]


COMPRESSORS: dict[str, Compressor | None] = {
    "full": None,
    "distil_trunc500": trunc_500,
    "llmlingua2": llmlingua2,
}


# --------------------------------------------------------------------------- #
# Block selection + rewriting
# --------------------------------------------------------------------------- #
def _compressible(role: str, block: Any) -> bool:
    """True iff this block is agent-read context we are allowed to compress."""
    if role not in ("user", "tool"):  # never system, never assistant reasoning
        return False
    if isinstance(block, str):
        return len(block) >= MIN_CHARS
    if not isinstance(block, dict):
        return False
    if block.get("type") not in ("text", "tool_result"):
        return False
    return True


def _block_text(block: Any) -> str | None:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text")
    return None


@dataclass
class CompressStats:
    requests: int = 0
    blocks_seen: int = 0
    blocks_compressed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    blocks_protected: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def compress_body(
    body: dict[str, Any],
    compressor: Compressor | None,
    stats: CompressStats,
    protect: str | None = None,
) -> dict[str, Any]:
    """Return a new request body with compressible context blocks rewritten.

    ``compressor=None`` (condition A) leaves the body byte-identical but still tallies
    block/token stats so all conditions report comparable pre-compression context size.

    ``protect`` is a substring (the SWE-bench *problem statement*) that must never be
    compressed: it is the task definition the agent is solving, not "file content the
    agent reads", so truncating it would handicap the compressed conditions for the
    wrong reason. Any block whose text contains ``protect`` is passed through verbatim
    (counted as seen+protected, never compressed) — for every condition identically.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    stats.requests += 1
    protect = protect or None

    def rewrite_text(role: str, text: str) -> str:
        stats.blocks_seen += 1
        before = len(text)
        stats.chars_before += before
        stats.tokens_before += _tokenizer.count(text)
        is_protected = protect is not None and protect in text
        if is_protected:
            stats.blocks_protected += 1
        if (
            compressor is None
            or before < MIN_CHARS
            or role not in ("user", "tool")
            or is_protected
        ):
            stats.chars_after += before
            stats.tokens_after += _tokenizer.count(text)
            return text
        out = compressor(text)
        stats.blocks_compressed += 1
        stats.chars_after += len(out)
        stats.tokens_after += _tokenizer.count(out)
        return out

    new_messages: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        role = msg.get("role", "")
        content = msg.get("content")
        if isinstance(content, str):
            new_messages.append({**msg, "content": rewrite_text(role, content)})
            continue
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content: list[Any] = []
        for block in content:
            txt = _block_text(block)
            if txt is not None and _compressible(role, block):
                new_txt = rewrite_text(role, txt)
                if isinstance(block, str):
                    new_content.append(new_txt)
                else:
                    new_content.append({**block, "text": new_txt})
            else:
                # tool_result blocks carry their payload in a nested content list.
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and _compressible(role, block)
                ):
                    new_content.append(_rewrite_tool_result(role, block, rewrite_text))
                else:
                    new_content.append(block)
        new_messages.append({**msg, "content": new_content})
    return {**body, "messages": new_messages}


def _rewrite_tool_result(
    role: str, block: dict[str, Any], rewrite_text
) -> dict[str, Any]:
    inner = block.get("content")
    if isinstance(inner, str):
        return {**block, "content": rewrite_text(role, inner)}
    if isinstance(inner, list):
        new_inner = []
        for sub in inner:
            if (
                isinstance(sub, dict)
                and sub.get("type") == "text"
                and isinstance(sub.get("text"), str)
            ):
                new_inner.append({**sub, "text": rewrite_text(role, sub["text"])})
            else:
                new_inner.append(sub)
        return {**block, "content": new_inner}
    return block


# --------------------------------------------------------------------------- #
# The proxy server
# --------------------------------------------------------------------------- #
@dataclass
class ProxyState:
    compressor: Compressor | None
    stats: CompressStats = field(default_factory=CompressStats)
    capture_path: Path | None = None  # when set, append raw request bodies (debug only)
    accounting_path: Path | None = None
    protect: str | None = (
        None  # problem statement: never compressed (task, not file content)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)


def _make_handler(state: ProxyState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence default stderr spam
            pass

        def _proxy(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            path = self.path
            body_out = raw
            if path.endswith("/v1/messages") and raw:
                try:
                    body = json.loads(raw)
                    if state.capture_path is not None:
                        with state.lock, state.capture_path.open("a") as fh:
                            fh.write(json.dumps(body) + "\n")
                    new_body = compress_body(
                        body, state.compressor, state.stats, protect=state.protect
                    )
                    body_out = json.dumps(new_body).encode()
                except (ValueError, TypeError):
                    body_out = raw  # malformed — forward untouched

            url = UPSTREAM + path
            req = urllib.request.Request(url, data=body_out, method=self.command)
            for k, v in self.headers.items():
                if k.lower() in (
                    "host",
                    "content-length",
                    "connection",
                    "accept-encoding",
                ):
                    continue
                req.add_header(k, v)
            req.add_header("Content-Length", str(len(body_out)))
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    payload = resp.read()
                    self._account(payload)
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() in (
                            "transfer-encoding",
                            "connection",
                            "content-length",
                            "content-encoding",
                        ):
                            continue
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            except urllib.error.HTTPError as e:
                payload = e.read()
                self.send_response(e.code)
                self.send_header(
                    "Content-Type", e.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (
                Exception
            ) as e:  # noqa: BLE001 — surface upstream/network errors as 502
                payload = json.dumps(
                    {"error": {"type": "proxy_error", "message": str(e)}}
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def _account(self, payload: bytes):
            try:
                data = json.loads(payload)
                usage = data.get("usage") or {}
            except (ValueError, TypeError):
                return
            with state.lock:
                state.stats.usage_input_tokens += int(usage.get("input_tokens", 0) or 0)
                state.stats.usage_output_tokens += int(
                    usage.get("output_tokens", 0) or 0
                )
                state.stats.cache_read_input_tokens += int(
                    usage.get("cache_read_input_tokens", 0) or 0
                )
                state.stats.cache_creation_input_tokens += int(
                    usage.get("cache_creation_input_tokens", 0) or 0
                )
                if state.accounting_path is not None:
                    with state.accounting_path.open("a") as fh:
                        fh.write(json.dumps({"usage": usage}) + "\n")

        do_POST = _proxy
        do_GET = _proxy

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    compressor: Compressor | None,
    capture_path: Path | None = None,
    accounting_path: Path | None = None,
    protect: str | None = None,
) -> tuple[ThreadingHTTPServer, ProxyState]:
    """Start the proxy on a background thread. Returns ``(server, state)``.

    Use ``port=0`` to bind an ephemeral port (read it from ``server.server_address[1]``).
    Call ``server.shutdown()`` to stop. ``protect`` (the problem statement) is never
    compressed.
    """
    state = ProxyState(
        compressor=compressor,
        capture_path=capture_path,
        accounting_path=accounting_path,
        protect=protect,
    )
    httpd = ThreadingHTTPServer((host, port), _make_handler(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, state


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="distil E7 compression proxy")
    ap.add_argument("--condition", choices=list(COMPRESSORS), default="full")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--capture", type=Path, default=None)
    ap.add_argument("--accounting", type=Path, default=None)
    args = ap.parse_args()
    httpd, state = serve(
        host=args.host,
        port=args.port,
        compressor=COMPRESSORS[args.condition],
        capture_path=args.capture,
        accounting_path=args.accounting,
    )
    print(
        f"distil-e7 proxy: condition={args.condition} on {args.host}:{httpd.server_address[1]}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()

"""The neutral meter: a usage-only reverse proxy that sits last, just before the provider.

Every arm's traffic crosses this proxy, so every arm is billed by the same code from the
same provider ``usage`` object — never from a tool's own estimate. The chain is::

    claude -p  ->  [arm's proxy, if any]  ->  UsageMeter  ->  provider (or mock upstream)

What it persists is content-free by construction: model id, status, the integer ``usage``
fields, byte counts and timings. Request/response bodies are parsed in memory (the request
for ``model``/``max_tokens`` to size the spend reservation, the response for ``usage``) and
never written anywhere.

Spend cap (same discipline as ``distil/mcpproxy/bench.py``'s meter): *reserve the worst
case before forwarding*, settle to the billed cost after. A request whose worst case does
not fit under the cap is refused with HTTP 402 and never reaches the provider, so the cap
is hard: realised spend can never exceed it.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from distil import pricing

#: 1-hour cache writes bill at 2x base input; ``distil.pricing`` only models the 5-minute
#: 1.25x write. Claude Code can request 1h TTLs, and ``usage.cache_creation`` splits them.
CACHE_WRITE_1H_MULT = 2.0

CANARY_USER_PREFIX = "cost-truth-canary-"  # mirrors arms.CANARY_USER_PREFIX

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",  # we must read usage out of the response, so ask for identity
}


class BudgetExceeded(RuntimeError):
    """The worst case of the next request does not fit under the hard cap."""


class UnpricedModel(ValueError):
    """A request named a model with no list price: its cost cannot be bounded."""


def _price(model: str) -> pricing.Pricing:
    p = pricing.resolve(model)
    if p is None:
        raise UnpricedModel(f"no list price for model {model!r}")
    return p


def usage_cost(model: str, usage: dict[str, Any]) -> float:
    """USD for one response's ``usage`` at list price, cache-inclusive.

    Uses the 5m/1h split in ``usage.cache_creation`` when present (1h writes are 2x),
    otherwise bills every write at the 5-minute 1.25x rate.
    """
    p = _price(model)
    writes = int(usage.get("cache_creation_input_tokens") or 0)
    split = usage.get("cache_creation") or {}
    w1h = int(split.get("ephemeral_1h_input_tokens") or 0)
    w5m = writes - w1h if split else writes
    return (
        int(usage.get("input_tokens") or 0) * p.input
        + int(usage.get("cache_read_input_tokens") or 0) * p.cache_read
        + max(w5m, 0) * p.cache_write
        + w1h * p.input * CACHE_WRITE_1H_MULT
        + int(usage.get("output_tokens") or 0) * p.output
    )


def worst_case_cost(model: str, request_bytes: int, max_tokens: int) -> float:
    """An upper bound on what one request can bill.

    Input: a token covers at least one byte, so ``request_bytes`` bounds the prompt; price
    all of it at the dearest input rate (a 1h cache write). Output: ``max_tokens`` (which
    also bounds extended thinking) at the output rate.
    """
    p = _price(model)
    return request_bytes * p.input * CACHE_WRITE_1H_MULT + max(max_tokens, 0) * p.output


class SpendMeter:
    """Thread-safe hard cap: reserve worst case, settle to actual."""

    def __init__(self, cap_usd: float) -> None:
        if cap_usd <= 0:
            raise ValueError("cap_usd must be positive")
        self.cap = cap_usd
        self.spent = 0.0
        self.reserved = 0.0
        self.refused = 0
        self._lock = threading.Lock()

    def reserve(self, worst: float) -> float:
        with self._lock:
            if self.spent + self.reserved + worst > self.cap:
                self.refused += 1
                raise BudgetExceeded(
                    f"worst case ${worst:.4f} does not fit: spent ${self.spent:.4f} + "
                    f"reserved ${self.reserved:.4f} of cap ${self.cap:.2f}"
                )
            self.reserved += worst
            return worst

    def settle(self, reservation: float, actual: float) -> None:
        with self._lock:
            self.reserved -= reservation
            self.spent += actual

    @property
    def exhausted(self) -> bool:
        return self.refused > 0


class SSEUsage:
    """Incrementally pull ``model`` and final ``usage`` out of an Anthropic SSE stream.

    ``message_start`` carries the input-side usage; each ``message_delta`` carries the
    cumulative output count (and, on newer API versions, may restate input-side fields).
    Later values overwrite earlier ones key by key.
    """

    def __init__(self) -> None:
        self.model: str | None = None
        self.usage: dict[str, Any] = {}
        self._buf = b""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == "message_start":
                msg = ev.get("message") or {}
                self.model = msg.get("model") or self.model
                self._merge(msg.get("usage"))
            elif ev.get("type") == "message_delta":
                self._merge(ev.get("usage"))

    def _merge(self, u: Any) -> None:
        if isinstance(u, dict):
            self.usage.update({k: v for k, v in u.items() if v is not None})


def content_free_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Only the integer counters — the whitelist that reaches disk."""
    out = {k: int(usage.get(k) or 0) for k in USAGE_KEYS}
    split = usage.get("cache_creation") or {}
    out["cache_creation_1h_input_tokens"] = int(split.get("ephemeral_1h_input_tokens") or 0)
    return out


@dataclass
class MeterConfig:
    upstream: str  # e.g. https://api.anthropic.com or http://127.0.0.1:NNNN (mock)
    log_path: Path  # per-run JSONL of content-free usage records
    spend: SpendMeter
    run_id: str
    timeout_s: float = 600.0
    #: bind address. Loopback by default; the live run binds where task containers can
    #: reach it (Docker Desktop forwards host.docker.internal to host loopback).
    bind: str = "127.0.0.1"
    #: preflight: a request whose ``metadata.user_id`` is the canary id is flagged
    #: ``canary: true`` (a boolean — the id itself is never written).
    canary_nonce: str | None = None


class _Handler(BaseHTTPRequestHandler):
    server: _MeterServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
        return  # the JSONL is the log; stderr noise would carry paths

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def _send_error(self, status: int, kind: str, message: str) -> None:
        body = json.dumps({"type": "error", "error": {"type": kind, "message": message}})
        raw = body.encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _proxy(self) -> None:
        cfg = self.server.cfg
        self._began = False  # response headers not yet sent (per request on keep-alive)
        self._canary = False
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        t0 = time.monotonic()
        billable = self.command == "POST" and self.path.split("?")[0].endswith("/v1/messages")
        model, reservation, stream = None, 0.0, False
        if billable:
            try:
                req = json.loads(body or b"{}")
            except ValueError:
                req = {}
            model = str(req.get("model") or "")
            stream = bool(req.get("stream"))
            meta = req.get("metadata") if isinstance(req, dict) else None
            if cfg.canary_nonce and isinstance(meta, dict):
                self._canary = meta.get("user_id") == CANARY_USER_PREFIX + cfg.canary_nonce
            try:
                worst = worst_case_cost(model, len(body), int(req.get("max_tokens") or 0))
                reservation = cfg.spend.reserve(worst)
            except UnpricedModel as e:
                self._record(model, 400, None, 0.0, t0, len(body), 0, stream, "unpriced_model")
                return self._send_error(400, "cost_truth_unpriced_model", str(e))
            except BudgetExceeded as e:
                self._record(model, 402, None, 0.0, t0, len(body), 0, stream, "budget_exceeded")
                return self._send_error(402, "cost_truth_budget_exceeded", str(e))

        actual, usage, status, resp_bytes, note = 0.0, None, 502, 0, None
        try:
            status, usage, resp_model, resp_bytes = self._forward(body)
            model = resp_model or model
            if usage is not None and model:
                actual = usage_cost(model, usage)
        except (OSError, http.client.HTTPException) as e:
            note = f"upstream_error:{type(e).__name__}"
            if not getattr(self, "_began", False):
                try:
                    self._send_error(502, "cost_truth_upstream_error", type(e).__name__)
                except OSError:
                    pass
        finally:
            if billable:
                cfg.spend.settle(reservation, actual)
        if billable or usage is not None:
            self._record(model, status, usage, actual, t0, len(body), resp_bytes, stream, note)

    def _forward(self, body: bytes) -> tuple[int, dict[str, Any] | None, str | None, int]:
        cfg = self.server.cfg
        u = urllib.parse.urlsplit(cfg.upstream)
        conn_cls = (
            http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        )
        conn = conn_cls(u.netloc, timeout=cfg.timeout_s)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        headers["accept-encoding"] = "identity"
        path = (u.path.rstrip("/") + self.path) if u.path else self.path
        try:
            conn.request(self.command, path, body=body or None, headers=headers)
            resp = conn.getresponse()
            self._began = True
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP_BY_HOP:
                    self.send_header(k, v)
            ctype = resp.getheader("content-type") or ""
            sse = SSEUsage() if "text/event-stream" in ctype else None
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            total, buf = 0, bytearray()
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if sse is not None:
                    sse.feed(chunk)
                else:
                    buf += chunk
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            conn.close()
        if sse is not None:
            return resp.status, (sse.usage or None), sse.model, total
        try:
            doc = json.loads(bytes(buf) or b"{}")
        except ValueError:
            return resp.status, None, None, total
        if not isinstance(doc, dict) or not isinstance(doc.get("usage"), dict):
            return resp.status, None, None, total
        return resp.status, doc["usage"], doc.get("model"), total

    def _record(
        self,
        model: str | None,
        status: int,
        usage: dict[str, Any] | None,
        cost: float,
        t0: float,
        req_bytes: int,
        resp_bytes: int,
        stream: bool,
        note: str | None,
    ) -> None:
        srv = self.server
        rec = {
            "run_id": srv.cfg.run_id,
            "seq": srv.next_seq(),
            "ts": round(time.time(), 3),
            "path": self.path.split("?")[0],
            "model": model,
            "status": status,
            "stream": stream,
            "usage": content_free_usage(usage) if usage else None,
            "cost_usd": round(cost, 8),
            "duration_s": round(time.monotonic() - t0, 3),
            "request_bytes": req_bytes,
            "response_bytes": resp_bytes,
            "note": note,
            "canary": self._canary,
        }
        with srv.log_lock, srv.cfg.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


class _MeterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, cfg: MeterConfig) -> None:
        super().__init__((cfg.bind, 0), _Handler)
        self.cfg = cfg
        self.log_lock = threading.Lock()
        self._seq = 0

    def next_seq(self) -> int:
        with self.log_lock:
            self._seq += 1
            return self._seq


class UsageMeter:
    """Start/stop wrapper; ``base_url`` is what the arm (or claude) points at."""

    def __init__(self, cfg: MeterConfig) -> None:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.log_path.touch()
        self._srv = _MeterServer(cfg)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._srv.server_address[1])

    @property
    def base_url(self) -> str:
        host = str(self._srv.server_address[0])
        return f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{self.port}"

    def __enter__(self) -> UsageMeter:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def read_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

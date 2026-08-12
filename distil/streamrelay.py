"""Incremental upstream→client relay for the sync HTTP servers.

The whole point of fronting an interactive agent is that tokens appear as the
model produces them. Buffering an SSE response start-to-finish turns
time-to-first-token into time-to-last-token, so this module relays the
upstream response chunk-by-chunk instead — while still returning the complete
buffered body to the caller for content-free accounting (shadow-mode decision
signatures). Chunked transfer framing is emitted when the upstream declares no
Content-Length (the SSE case); requires the handler to speak HTTP/1.1.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any

_CHUNK = 8192

# Provider-reported quota state, echoed on every response as `anthropic-ratelimit-*`.
# Worth recording because it is the ONLY first-party signal for how a request is charged
# against a plan's budget: billed token counts say what was sent, these say what it cost
# the quota. On cache-heavy traffic the two diverge sharply — a cached read is ~90% off
# the metered price, but whether a *subscription* cap discounts it the same way is not
# published, and this is the measurement that settles it for a given account.
_RATELIMIT_PREFIX = "anthropic-ratelimit-"


def capture_ratelimit(headers: Any) -> dict[str, str] | None:
    """Extract the ``anthropic-ratelimit-*`` response headers, keyed without the vendor
    prefix (``requests-remaining``, ``input-tokens-limit``, ...). Returns None when the
    upstream sent none, so the record stays absent rather than empty. Header values only
    — counters and timestamps, never content."""
    try:
        out = {
            k.lower()[len(_RATELIMIT_PREFIX) :]: v
            for k, v in headers.items()
            if k.lower().startswith(_RATELIMIT_PREFIX)
        }
    except Exception:  # noqa: BLE001 — a diagnostic must never break the relay
        return None
    return out or None


# Billed-usage capture. Works on a plain JSON response and on SSE bytes alike:
# input_tokens appears once near the start (message_start), output_tokens is
# cumulative so the LAST occurrence (final message_delta) is the total.
_USAGE_IN = re.compile(rb'"input_tokens"\s*:\s*(\d+)')
_USAGE_OUT = re.compile(rb'"output_tokens"\s*:\s*(\d+)')
# Prompt-caching: input_tokens counts ONLY the uncached tokens. The cached prefix is billed
# separately as cache_read / cache_creation. The *full* input the heuristic estimates is the
# sum of all three — without these, token accounting (and calibration) is wrong on cached traffic.
_CACHE_READ = re.compile(rb'"cache_read_input_tokens"\s*:\s*(\d+)')
_CACHE_CREATE = re.compile(rb'"cache_creation_input_tokens"\s*:\s*(\d+)')
_USAGE_SCAN_CAP = 16384  # head/tail window — usage lives at the edges of a stream


def scan_usage(blob: bytes) -> dict[str, int]:
    """Best-effort billed-token extraction from a response body (JSON or SSE).

    Returns any of ``{"input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens"}`` found — empty dict when the body carries no usage. The
    cache fields matter: on cached traffic ``input_tokens`` is only the uncached remainder, so
    the true billed input is ``input_tokens + cache_read + cache_creation``.
    """
    out: dict[str, int] = {}
    m = _USAGE_IN.search(blob)
    if m:
        out["input_tokens"] = int(m.group(1))
    for key, rx in (
        ("cache_read_input_tokens", _CACHE_READ),
        ("cache_creation_input_tokens", _CACHE_CREATE),
    ):
        cm = rx.search(blob)
        if cm:
            out[key] = int(cm.group(1))
    last = None
    for last in _USAGE_OUT.finditer(blob):  # noqa: B007 — want the final (cumulative) one
        pass
    if last is not None:
        out["output_tokens"] = int(last.group(1))
    return out


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Relay 3xx to the client instead of following it: auto-following would
    re-send the client's Authorization/x-api-key to whatever host the upstream
    names — the client's own HTTP stack must decide that."""

    def redirect_request(self, *a, **k):  # noqa: ANN002, ANN003
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _is_timeout(exc: urllib.error.URLError) -> bool:
    import socket

    return isinstance(exc.reason, (socket.timeout, TimeoutError))


def stream_upstream(
    handler: BaseHTTPRequestHandler,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    *,
    method: str = "POST",
    timeout: float,
    hop_by_hop: frozenset[str],
    extras: dict[str, str] | None = None,
    want_body: bool = False,
    usage_sink: dict[str, int] | None = None,
) -> tuple[int, bytes | None]:
    """Send the request and relay the response to *handler* incrementally.

    When ``usage_sink`` is given, the first/last ``_USAGE_SCAN_CAP`` bytes of
    the relayed stream are scanned for billed usage after relay completes and
    the result is merged into the dict — without buffering the full body.

    When ``want_body`` is set, returns the complete response body (buffered as
    it streamed) so callers can run post-hoc accounting (shadow sampling); when
    it is not, the body is relayed but never accumulated — N concurrent large
    streams would otherwise pin N full responses in memory. Always returns the
    relayed HTTP status (a synthetic 502/504 on connection failure) as the first
    element so callers can book accounting only on a confirmed 2xx; the body is
    ``None`` unless ``want_body`` was set and a response was read.
    Once streaming has begun, a mid-stream failure closes the connection —
    there is no valid way to append an error to a partially-delivered body.
    """
    req = urllib.request.Request(
        url,
        data=body,
        headers={**headers, **({"Content-Length": str(len(body))} if body else {})},
        method=method,
    )

    def _error(status: int, payload: bytes) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    try:
        resp = _OPENER.open(req, timeout=timeout)  # noqa: S310 — operator-set upstream
    except urllib.error.HTTPError as exc:
        rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
        # A 429 carries the most informative quota state of any response — capture it
        # here too, or the one request that proves the cap is the one we fail to record.
        handler._distil_ratelimit = capture_ratelimit(exc.headers)  # type: ignore[attr-defined]
        handler.send_response(exc.code)
        for k, v in exc.headers.items():
            if k.lower() not in hop_by_hop:
                handler.send_header(k, v)
        handler.send_header("Content-Length", str(len(rbody)))
        handler.end_headers()
        handler.wfile.write(rbody)
        return exc.code, None  # non-2xx relayed to client; not a bookable success
    except urllib.error.URLError as exc:
        status = 504 if _is_timeout(exc) else 502
        _error(
            status,
            json.dumps(
                {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
            ).encode(),
        )
        return status, None
    except TimeoutError:
        _error(504, b'{"error":"upstream timed out"}')
        return 504, None

    with resp:
        # Before relaying: the handler is per-request, so this reaches _emit_detail
        # without threading a return value through every streaming call site.
        handler._distil_ratelimit = capture_ratelimit(resp.headers)  # type: ignore[attr-defined]
        length = resp.headers.get("Content-Length")
        chunked = length is None
        handler.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in hop_by_hop:
                handler.send_header(k, v)
        for k, v in (extras or {}).items():
            handler.send_header(k, v)
        if chunked:
            handler.send_header("Transfer-Encoding", "chunked")
        else:
            handler.send_header("Content-Length", length)
        handler.end_headers()

        buf = bytearray()
        head = bytearray()
        tail = bytearray()
        try:
            while True:
                # read1: return as soon as ANY bytes arrive (at most one socket
                # read) — resp.read(n) would block until n bytes accumulate,
                # defeating incremental delivery on a dribbling SSE stream.
                chunk = resp.read1(_CHUNK)
                if not chunk:
                    break
                if want_body:
                    buf += chunk
                if usage_sink is not None:
                    if len(head) < _USAGE_SCAN_CAP:
                        head += chunk
                    tail += chunk
                    if len(tail) > _USAGE_SCAN_CAP:
                        del tail[: len(tail) - _USAGE_SCAN_CAP]
                if chunked:
                    handler.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                else:
                    handler.wfile.write(chunk)
                handler.wfile.flush()
            if chunked:
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
        except OSError:
            # Client disconnected or upstream stalled mid-stream: nothing valid
            # can be appended once bytes have flowed — drop the connection but
            # keep what streamed for accounting.
            handler.close_connection = True
        if usage_sink is not None:
            try:
                usage_sink.update(scan_usage(bytes(head) + b"\n" + bytes(tail)))
            except Exception:  # noqa: BLE001 — usage capture must never break the relay
                pass
        return resp.status, (bytes(buf) if want_body else None)

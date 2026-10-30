"""stream_upstream with a chunk_transform: an upstream Content-Length can't be relayed."""

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from distil.streamrelay import stream_upstream

BODY = b'event: message_start\ndata: {"n":1}\n\n'


class _Upstream(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *a: object) -> None:
        pass


class _Handler:
    """Captures what stream_upstream sends to the client."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.wfile = io.BytesIO()

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, k: str, v: str) -> None:
        self.headers[k] = v

    def end_headers(self) -> None:
        pass


def test_transform_forces_chunked_over_upstream_content_length():
    srv = HTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        h = _Handler()
        stream_upstream(
            h,
            f"http://127.0.0.1:{srv.server_port}/v1/messages",
            b"{}",
            {},
            timeout=5,
            hop_by_hop=frozenset({"content-length", "transfer-encoding", "connection"}),
            chunk_transform=lambda c: c.replace(b'"n":1', b'"n":1000'),
        )
    finally:
        srv.shutdown()
    assert "Content-Length" not in h.headers
    assert h.headers["Transfer-Encoding"] == "chunked"
    assert b'"n":1000' in h.wfile.getvalue()

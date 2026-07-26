"""Receipts — the artifact must be verifiable by someone who doesn't trust us.

A receipt says what distil did to one request. Its value depends entirely on two
properties: it cannot be altered without detection, and it cannot leak content. Both
are asserted here rather than documented.
"""

from __future__ import annotations

import json

import pytest

from distil import receipts as R


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    return tmp_path


def _mk(i: int) -> R.Receipt:
    return R.Receipt(
        ts=1000.0 + i,
        request_id=f"req{i}",
        session="s1",
        model="claude-opus-4-8",
        mode="digest",
        tokens_original=1000,
        tokens_compressed=400,
        reversible=False,
        handles=[f"aaaaaaa{i}"],
        certificate="ltt@digest",
    )


def test_clean_chain_verifies(home):
    for i in range(3):
        R.append(_mk(i))
    v = R.verify()
    assert v.ok and v.total == 3, v.statement
    assert "VERIFIED" in v.statement


def test_receipt_is_content_free(home):
    """Only declared fields ever reach disk — that is what makes the artifact shareable."""
    R.append(_mk(0))
    line = json.loads((home / "receipts.jsonl").read_text().splitlines()[0])
    assert set(line) - set(R.Receipt.FIELDS) == set(), "undeclared field written to disk"
    # nothing resembling prompt/completion text
    for key in ("text", "prompt", "completion", "content", "messages", "body"):
        assert key not in line


def test_edited_receipt_is_detected(home):
    for i in range(3):
        R.append(_mk(i))
    p = home / "receipts.jsonl"
    lines = p.read_text().splitlines()
    d = json.loads(lines[1])
    d["tokens_compressed"] = 1  # claim a far bigger saving
    lines[1] = json.dumps(d, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    v = R.verify()
    assert not v.ok and v.first_bad_index == 1
    assert "does not match its hash" in v.reason


def test_removed_receipt_is_detected(home):
    for i in range(3):
        R.append(_mk(i))
    p = home / "receipts.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")  # drop the middle one
    v = R.verify()
    assert not v.ok, "deleting a receipt must break the chain"
    assert "prev-hash" in v.reason


def test_reordered_receipts_are_detected(home):
    for i in range(3):
        R.append(_mk(i))
    p = home / "receipts.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    assert not R.verify().ok


def test_handle_order_does_not_change_the_hash():
    """Handles are a set; serialisation order must not affect verification."""
    a = R.Receipt(1.0, "r", "s", "m", "digest", 10, 5, False, handles=["bbbbbbbb", "aaaaaaaa"])
    b = R.Receipt(1.0, "r", "s", "m", "digest", 10, 5, False, handles=["aaaaaaaa", "bbbbbbbb"])
    assert a.compute_hash() == b.compute_hash()


def test_empty_chain_is_not_an_error(home):
    v = R.verify()
    assert v.ok and v.total == 0
    assert "No receipts" in v.statement


def test_malformed_line_does_not_crash_the_reader(home):
    R.append(_mk(0))
    p = home / "receipts.jsonl"
    p.write_text(p.read_text() + "{not json\n")
    assert R.verify().ok, "a corrupt trailing line must not invalidate real receipts"


def test_proxy_emits_a_verifiable_receipt_per_served_request(home, monkeypatch):
    """The wiring, not just the module: a real request through a real proxy must leave
    a receipt, and the chain must verify.

    Gated on 2xx only — deliberately NOT on `booked` (a savings ledger being attached)
    nor on being inside a `wrap` session. A compliance record that vanishes depending on
    how the proxy was launched is not a compliance record.
    """
    import json as _json
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from distil.proxy import build_handler

    class _Up(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            b = b'{"id":"m","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):  # noqa: D102
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    px = ThreadingHTTPServer(
        ("127.0.0.1", 0), build_handler(f"http://127.0.0.1:{up.server_address[1]}")
    )
    threading.Thread(target=px.serve_forever, daemon=True).start()
    try:
        payload = {
            "model": "claude-opus-4-8",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
        for _ in range(2):
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{px.server_address[1]}/v1/messages",
                    data=_json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                ),
                timeout=30,
            ).read()
    finally:
        px.shutdown()
        up.shutdown()

    got = list(R.read())
    assert len(got) == 2, f"expected one receipt per served request, got {len(got)}"
    assert R.verify().ok
    assert all(r.model == "claude-opus-4-8" for r in got)
    # content-free holds through the real path too
    on_disk = (home / "receipts.jsonl").read_text()
    assert "hello" not in on_disk

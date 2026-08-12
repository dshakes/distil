"""Genuine runtime savings — unit + a real end-to-end proxy→ledger test (no mocks)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from distil import ledger
from distil.proxy import build_handler
from distil.runtime import RuntimeSavings


def test_record_and_flush_to_ledger(tmp_path):
    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(1000, 700)
    rs.record(500, 400)
    assert rs.tokens_saved == 400
    assert rs.dollars_saved > 0
    assert rs.flush() is True
    s = ledger.summary(led)
    assert s.total_tokens_saved == 400
    assert s.by_trajectory.get("live-proxy", 0.0) > 0
    assert rs.flush() is False  # counters reset after flush


def test_accepts_str_ledger_path(tmp_path):
    led = str(tmp_path / "savings.jsonl")  # a str, not a Path
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(900, 600)
    assert rs.flush() is True  # would crash on `.parent` if not coerced
    assert ledger.summary(tmp_path / "savings.jsonl").total_tokens_saved == 300


def test_per_model_pricing(tmp_path):
    """Requests naming different models are priced at THEIR rates, not the default's."""
    from distil import pricing

    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(1_000_000, 0, model="claude-opus-4-8")  # $5.00 saved
    rs.record(1_000_000, 0, model="claude-haiku-4-5")  # $1.00 saved
    assert abs(rs.dollars_saved - 6.0) < 1e-9  # NOT 10.00 (all-Opus mispricing)
    assert rs.flush() is True
    s = ledger.summary(led)
    assert abs(s.total_dollars_saved - 6.0) < 1e-6
    # dated snapshot ids resolve to their base price
    assert pricing.resolve("claude-haiku-4-5-20251001").name == "claude-haiku-4-5"
    assert pricing.resolve("anthropic.claude-opus-4-8").name == "claude-opus-4-8"


def test_unknown_model_never_priced_at_claude_rates(tmp_path):
    """A Gemini/OpenAI upstream records genuine token savings with dollars=0 —
    never silently billed at the default Claude rate."""
    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(1_000_000, 0, model="gemini-2.0-flash")
    assert rs.dollars_saved == 0.0
    rs.flush()
    s = ledger.summary(led)
    assert s.total_tokens_saved == 1_000_000  # tokens still counted honestly
    assert s.total_dollars_saved == 0.0
    assert any(
        "unpriced" in tid
        for tid in [json.loads(line)["model"] for line in led.read_text().splitlines()]
    )


def test_maybe_flush_time_based(tmp_path):
    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(100, 50)
    # _last_flush=0 → age check fires immediately: first record is visible fast
    assert rs.maybe_flush(every=50) is True
    rs.record(100, 50)
    assert rs.maybe_flush(every=50, max_age=9999) is False  # neither condition met


def _start_echo_upstream() -> ThreadingHTTPServer:
    class Echo(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_proxy_records_genuine_savings_e2e(tmp_path):
    upstream = _start_echo_upstream()
    up_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    handler = build_handler(up_url, savings=rs, flush_every=1)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        long = "DECISION: keep\n" + "\n".join(f"verbose log line {i}" for i in range(20))
        payload = json.dumps(
            {
                "model": "claude-opus-4-8",
                "messages": [
                    {"role": "user", "content": "investigate"},
                    {"role": "user", "content": [{"type": "tool_result", "content": long}]},
                    # Two later turns keep the tool_result out of the recency-exempt
                    # window so it still digests and genuine savings are recorded.
                    {"role": "user", "content": "next"},
                    {"role": "user", "content": "next"},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert int(resp.headers.get("x-distil-tokens-saved", "0")) > 0
        # genuine savings persisted to the ledger (flush_every=1)
        s = ledger.summary(led)
        assert s.by_trajectory.get("live-proxy", 0.0) > 0
        assert s.total_tokens_saved > 0
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_render_html_uses_real_ledger_numbers(tmp_path):
    led = tmp_path / "s.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(2000, 1200)
    rs.flush()
    html = ledger.render_html(ledger.summary(led))
    assert "savings" in html.lower()
    assert "live-proxy" in html  # the real source label, not dummy data


def test_zero_savings_window_stays_in_the_denominator(tmp_path):
    """A window that saved nothing is still traffic, so it must reach the ledger.

    Dropping it kept its baseline out of the denominator as well, which reported
    savings over the traffic that compressed instead of over all traffic.
    """
    led = tmp_path / "savings.jsonl"
    rs = RuntimeSavings(model="claude-opus-4-8", ledger_path=led)
    rs.record(1000, 1000)
    rs.record(500, 500)
    assert rs.flush() is True
    rs.record(1000, 600)
    assert rs.flush() is True
    s = ledger.summary(led)
    assert s.total_tokens_saved == 400
    # 400 saved out of 2500 sent — not 400 out of the 1000 that happened to fold.
    assert s.total_baseline_tokens == 2500

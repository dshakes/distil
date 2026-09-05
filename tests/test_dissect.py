"""Tests for `distil dissect` — session picker, report math, renderers, and the
proxy/wrap logging it reads (session manifest + per-request detail records).

The proxy round-trip test mirrors tests/test_proxy.py: a stub upstream echoes
the forwarded body, the real handler runs in a thread, and the assertion is on
the *artifact* — sessions/<sid>.requests.jsonl — not on internals.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from distil import dissect as dz
from distil.cli import main
from distil.ledger import (
    append_session_request,
    session_manifest_path,
    session_requests_path,
    write_session_manifest,
)
from distil.proxy import build_handler, wrap_run

_LOG_LINES = "\n".join(
    f"[2026-07-11 12:00:{i:02d}] INFO worker-{i}: heartbeat ok, queue depth {i * 3}"
    for i in range(60)
)


_BLOB = "\n".join(f"[worker-{i}] heartbeat ok, queue depth {i * 7}, no errors" for i in range(8))


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _write_transcript(claude_dir: Path, cwd_slug: str = "-tmp-proj") -> Path:
    """A minimal Claude-Code-shaped transcript matching the seeded session
    (s200-1: requests at ts 1000/1600/1700, restore blob _BLOB)."""
    proj = claude_dir / "projects" / cwd_slug
    proj.mkdir(parents=True, exist_ok=True)
    lines = [
        {"type": "ai-title", "aiTitle": "fix the flaky tests"},
        {
            "type": "user",
            "timestamp": _iso(900),
            "cwd": "/tmp/proj",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "please run the tests"},
        },
        {
            "type": "assistant",
            "timestamp": _iso(950),
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu1", "name": "bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "timestamp": _iso(990),
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "text", "text": "$ make test\n" + _BLOB}],
                    }
                ],
            },
        },
        # a sidechain human line must NOT count as a turn
        {
            "type": "user",
            "timestamp": _iso(1200),
            "isSidechain": True,
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "subagent task prompt"},
        },
        {
            "type": "user",
            "timestamp": _iso(1500),
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": [{"type": "text", "text": "now fix the bug"}]},
        },
        {
            "type": "assistant",
            "timestamp": _iso(1550),
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu2", "name": "Read", "input": {}}],
            },
        },
        # the agent re-fetched the same content after it had been folded
        {
            "type": "user",
            "timestamp": _iso(1560),
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu2", "content": _BLOB}],
            },
        },
        {"type": "not json line skipped below"},
    ]
    path = proj / "abc-session.jsonl"
    path.write_text("\n".join(json.dumps(rec) for rec in lines) + "\nnot json\n")
    return path


_UPSTREAM_JSON = json.dumps(
    {
        "id": "msg_test",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 4321, "output_tokens": 87},
    }
).encode()

_UPSTREAM_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":'
    b'{"usage":{"input_tokens":1234,"output_tokens":1}}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":55}}\n\n'
    b"event: message_stop\ndata: {}\n\n"
)


class _EchoHandler(BaseHTTPRequestHandler):
    """Stub upstream: SSE when the request asked to stream, JSON otherwise —
    both carrying known usage figures so capture can be asserted exactly."""

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        streaming = b'"stream": true' in body or b'"stream":true' in body
        payload = _UPSTREAM_SSE if streaming else _UPSTREAM_JSON
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # quiet
        pass


@pytest.fixture()
def servers() -> Any:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    handler_cls = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield proxy.server_address[1]
    proxy.shutdown()
    upstream.shutdown()


def _post(port: int, payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req)


def _compressible_payload() -> dict[str, Any]:
    return {
        "model": "claude-test-1",
        "max_tokens": 128,
        "system": "You are a test agent." * 20,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _LOG_LINES}],
            },
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next again"},
        ],
    }


# --------------------------------------------------------------- proxy logging
class TestRequestDetailLogging:
    def test_detail_record_written_per_request(
        self, servers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SESSION", "s100-1")
        with _post(servers, _compressible_payload()) as resp:
            assert resp.status == 200
        path = session_requests_path("s100-1")
        assert path is not None and path.exists()
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(recs) == 1
        rec = recs[0]
        assert rec["model"] == "claude-test-1"
        assert rec["mode"] == "digest"
        assert rec["stream"] is False
        assert rec["status"] == 200
        assert rec["booked"] is False  # no SavingsTracker wired in this harness
        assert rec["compressible_tokens"] > 0
        assert rec["tokens_saved"] > 0
        assert rec["overhead_tokens"] > 0  # the system prompt is counted, not compressed
        assert rec["blocks"], "digested blocks should be inventoried"
        blk = rec["blocks"][0]
        assert set(blk) == {"h", "sig", "tokens"} and len(blk["h"]) == 8
        assert ":" in blk["sig"] and blk["tokens"] > 0
        assert "handle=" not in json.dumps(rec), "detail records must stay content-free"
        assert rec["client_stream"] is False
        assert rec["duration_ms"] is not None and rec["duration_ms"] >= 0
        assert rec["usage_input_tokens"] == 4321  # billed usage from the JSON response
        assert rec["usage_output_tokens"] == 87
        assert rec["expanded_handles"] == []

    def test_streamed_request_captures_sse_usage(
        self, servers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SESSION", "s101-1")
        payload = _compressible_payload() | {"stream": True}
        with _post(servers, payload) as resp:
            assert resp.status == 200
            resp.read()
        path = session_requests_path("s101-1")
        assert path is not None
        # The detail record lands after the last relayed byte — poll until a full
        # LINE is present (the file can exist momentarily before the line is flushed).
        rec = None
        for _ in range(250):
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                if lines:
                    rec = json.loads(lines[0])
                    break
            time.sleep(0.02)
        assert rec is not None, "streamed detail record was not written in time"
        assert rec["stream"] is True and rec["client_stream"] is True
        assert rec["usage_input_tokens"] == 1234  # from SSE message_start
        assert rec["usage_output_tokens"] == 55  # last (cumulative) message_delta
        assert rec["duration_ms"] is not None

    def test_no_session_no_record(
        self, servers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        with _post(servers, _compressible_payload()) as resp:
            assert resp.status == 200
        assert not list((tmp_path / "sessions").glob("*.requests.jsonl"))


class TestWrapManifest:
    def test_manifest_written_at_wrap_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        assert wrap_run(["true"], expand=True, session_delta=True, shadow_rate=0.25) == 0
        manifests = list((tmp_path / "sessions").glob("s*.json"))
        assert len(manifests) == 1
        man = json.loads(manifests[0].read_text(encoding="utf-8"))
        assert man["tool"] == "true" and man["argv"] == ["true"]
        assert man["flags"]["expand"] is True
        assert man["flags"]["session_delta"] is True
        assert man["flags"]["shadow_rate"] == 0.25
        assert man["billing"] in ("subscription", "metered", "unknown")
        assert man["started_ts"] > 0 and man["sid"] == manifests[0].stem

    def test_writers_are_fail_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        # No session -> no path -> both writers are silent no-ops.
        assert session_manifest_path() is None
        write_session_manifest({"sid": "x"})
        append_session_request({"ts": 1})


# ----------------------------------------------------------------- report math
def _seed_state(home: Path) -> None:
    """Two sessions: sA (manifest + detail + restore blobs), sB (ledger only)."""
    row = {
        "trajectory_id": "live-proxy",
        "tokenizer": "heuristic",
        "mode": "digest",
        "acct": 2,
    }
    ledger = [
        {
            **row,
            "session": "s200-1",
            "model": "m-big",
            "turns": 2,
            "ts": 1000.0,
            "baseline_input_tokens": 9000,
            "distil_input_tokens": 3000,
            "baseline_dollars": 0.9,
            "distil_dollars": 0.3,
        },
        {
            **row,
            "session": "s200-1",
            "model": "m-small",
            "turns": 1,
            "ts": 1600.0,
            "baseline_input_tokens": 1000,
            "distil_input_tokens": 900,
            "baseline_dollars": 0.1,
            "distil_dollars": 0.09,
        },
        {
            **row,
            "session": "s300-9",
            "model": "m-big",
            "turns": 1,
            "ts": 2000.0,
            "baseline_input_tokens": 500,
            "distil_input_tokens": 400,
            "baseline_dollars": 0.05,
            "distil_dollars": 0.04,
        },
        {"corrupt": "row missing session"},
    ]
    (home / "savings.jsonl").write_text("\n".join(json.dumps(r) for r in ledger) + "\nnot json\n")
    sess = home / "sessions"
    sess.mkdir()
    (sess / "s200-1.json").write_text(
        json.dumps(
            {
                "sid": "s200-1",
                "tool": "codex",
                "argv": ["codex", "--full-auto"],
                "cwd": "/tmp/proj",
                "started_ts": 990.0,
                "distil_version": "1.15.0",
                "billing": "subscription",
                "flags": {
                    "expand": True,
                    "session_delta": True,
                    "shadow_rate": 0.1,
                    "lossless_only": False,
                    "verbatim": False,
                    "shape_output": "off",
                    "upstream": "https://api.anthropic.com",
                    "env_var": "ANTHROPIC_BASE_URL",
                },
            }
        )
    )
    details = [
        {
            "ts": 1000.0,
            "model": "m-big",
            "stream": True,
            "client_stream": True,
            "status": 200,
            "booked": True,
            "duration_ms": 4000,
            "usage_input_tokens": 2100,
            "usage_output_tokens": 300,
            "expanded_handles": [],
            "mode": "digest",
            "compressible_tokens": 5000,
            "tokens_saved": 3500,
            "overhead_tokens": 700,
            "system_tokens": 100,
            "tools_tokens": 600,
            "tools": [
                {"name": "bash", "tokens": 300},
                {"name": "mcp__gmail__search", "tokens": 300},
            ],
            "delta_refs": 0,
            "delta_tokens_saved": 0,
            "prefix_msgs": 0,
            "shadow_sampled": True,
            "expanded": False,
            "output_shaping": "",
            "blocks": [
                {"h": "aaaa1111", "sig": "log:l", "tokens": 2000},
                {"h": "bbbb2222", "sig": "prose:m", "tokens": 900},
            ],
        },
        {
            "ts": 1600.0,
            "model": "m-small",
            "stream": False,
            "client_stream": True,
            "status": 200,
            "booked": True,
            "duration_ms": 9000,
            "usage_input_tokens": 1200,
            "usage_output_tokens": 150,
            "expanded_handles": ["aaaa1111"],
            "mode": "digest",
            "compressible_tokens": 2000,
            "tokens_saved": 1200,
            "overhead_tokens": 500,
            "system_tokens": 200,
            "tools_tokens": 300,
            "tools": [{"name": "bash", "tokens": 300}],
            "delta_refs": 3,
            "delta_tokens_saved": 800,
            "prefix_msgs": 4,
            "shadow_sampled": False,
            "expanded": True,
            "output_shaping": "",
            "blocks": [{"h": "aaaa1111", "sig": "log:l", "tokens": 2000}],
        },
        {
            "ts": 1700.0,
            "model": "m-small",
            "stream": False,
            "client_stream": False,
            "status": 529,
            "booked": False,
            "duration_ms": 500,
            "usage_input_tokens": None,
            "usage_output_tokens": None,
            "expanded_handles": [],
            "mode": "verbatim",
            "compressible_tokens": 0,
            "tokens_saved": 0,
            "overhead_tokens": 500,
            "system_tokens": 250,
            "tools_tokens": 250,
            "tools": [{"name": "bash", "tokens": 250}],
            "delta_refs": 0,
            "delta_tokens_saved": 0,
            "prefix_msgs": 0,
            "shadow_sampled": False,
            "expanded": False,
            "output_shaping": "",
            "blocks": [],
        },
    ]
    (sess / "s200-1.requests.jsonl").write_text("\n".join(json.dumps(r) for r in details) + "\n")
    (sess / "s200-1").write_text("1")
    (sess / "s300-9.exit").write_text("rc=0")
    restore = home / "restore"
    restore.mkdir()
    (restore / "aaaa1111").write_text(_BLOB)
    (home / "shadow.jsonl").write_text(
        json.dumps({"ts": 1500.0, "equivalent": True})
        + "\n"
        + json.dumps({"ts": 999999.0, "equivalent": False})
        + "\n"
    )


class TestDissection:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_list_sessions_newest_first_with_status(self) -> None:
        sessions = dz.list_sessions()
        assert [o.sid for o in sessions] == ["s300-9", "s200-1"]
        s200 = sessions[1]
        assert s200.tool == "codex" and s200.requests == 3
        assert s200.baseline_tokens == 10000 and s200.distil_tokens == 3900
        assert s200.status == "live"
        assert sessions[0].status == "exited"

    def test_resolve_sid(self) -> None:
        assert dz.resolve_sid("latest") == "s300-9"
        assert dz.resolve_sid("s200") == "s200-1"
        assert dz.resolve_sid("s200-1") == "s200-1"
        assert dz.resolve_sid("s") is None  # ambiguous
        assert dz.resolve_sid("nope") is None

    def test_dissection_math(self) -> None:
        d = dz.dissect("s200-1")
        assert d.baseline_tokens == 10000 and d.distil_tokens == 3900
        assert d.pct_saved == pytest.approx(61.0)
        assert d.dollars_saved == pytest.approx(0.61)
        assert d.per_model()[0] == ("m-big", 2, 9000, 3000)
        assert d.detail_available
        assert d.delta_tokens_saved == 800
        assert d.overhead_tokens_avg == 566
        assert d.verbatim_requests == 1 and d.unbooked_requests == 1
        assert d.shadow_sampled == 1 and d.expand_resolved == 1
        assert d.billing == "subscription"
        # Blocks dedup by handle across requests; folds counted per sighting.
        assert set(d.blocks) == {"aaaa1111", "bbbb2222"}
        assert d.blocks["aaaa1111"]["folds"] == 2
        assert d.blocks["aaaa1111"]["recoverable"] is True
        assert d.blocks["bbbb2222"]["recoverable"] is False
        assert d.blocks_by_kind()[0] == ("log:l", 1, 2000)
        # Shadow join is by time window: only the ts=1500 row is inside.
        assert d.shadow_window_rows == 1 and d.shadow_window_agree == 1

    def test_render_text_full(self) -> None:
        d = dz.dissect("s200-1")
        text = dz.render_text(d, color=False)
        assert "s200-1" in text and "codex" in text
        assert "61.0% saved" in text and "notional" in text
        assert "log:l" in text and "aaaa1111" in text
        assert "1/2 blocks still in restore/" in text
        assert "expand: 1 requests" in text and "shadow: 1 requests" in text
        assert "1 verbatim" in text

    def test_render_text_degrades_without_detail(self) -> None:
        d = dz.dissect("s300-9")
        text = dz.render_text(d, color=False)
        assert "manifest not recorded" in text
        assert "not recorded — per-request detail" in text
        assert "rc=0" in text

    def test_to_json_schema(self) -> None:
        payload = dz.to_json(dz.dissect("s200-1"))
        assert payload["session"] == "s200-1"
        assert payload["savings"]["pct_saved"] == 61.0
        assert payload["savings"]["dollars_notional"] is True
        assert payload["requests"]["total"] == 3
        assert payload["blocks"]["unique"] == 2 and payload["blocks"]["recoverable"] == 1
        assert payload["quality"]["shadow_sampled_requests"] == 1

    def test_render_html_self_contained(self) -> None:
        page = dz.render_html(dz.dissect("s200-1"))
        assert page.startswith("<!doctype html>")
        assert "s200-1" in page and "61.0%" in page and "notional" in page
        assert "aaaa1111" in page and "log:l" in page


class TestCli:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_picker_lists_sessions(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect"]) == 0
        out = capsys.readouterr().out
        assert "s200-1" in out and "s300-9" in out and "codex" in out

    def test_picker_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert {r["session"] for r in rows} == {"s200-1", "s300-9"}

    def test_report_and_prefix_resolution(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "s200", "--no-color"]) == 0
        assert "61.0% saved" in capsys.readouterr().out

    def test_json_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "latest", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["session"] == "s300-9"

    def test_html_export(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out_file = tmp_path / "dissect.html"
        assert main(["dissect", "s200-1", "--html", str(out_file)]) == 0
        assert out_file.read_text(encoding="utf-8").startswith("<!doctype html>")

    def test_unknown_session_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "zzz"]) == 2
        assert "no session matches" in capsys.readouterr().out


class TestInsights:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_mechanism_decomposition(self) -> None:
        d = dz.dissect("s200-1")
        assert d.tokens_saved_total == 4700
        assert d.delta_tokens_saved == 800 and d.digest_saved == 3900

    def test_overhead_tax(self) -> None:
        d = dz.dissect("s200-1")
        # Booked only: the 529 retry's 500 overhead tokens were never billed as a
        # successful request, and counting them inflated the denominator the savings
        # headline is measured against.
        assert d.overhead_tokens_total == 1200
        # Share is of what was actually SENT: 1200 overhead vs (7000 - 4700) post-
        # compression payload -> 1200/3500. Dividing by the pre-compression 7000
        # understated the tax by counting tokens distil had already removed.
        assert d.overhead_share == pytest.approx(34.29, abs=0.01)

    def test_detail_totals_reconcile_with_savings_headline(self) -> None:
        """The mechanism split must add up to the headline saving.

        These come from different files (savings.jsonl vs sessions/*.requests.jsonl);
        summing the detail over unbooked retries too made them disagree, so the report
        showed a 'savings by mechanism' total larger than the savings it decomposed."""
        d = dz.dissect("s200-1")
        assert d.digest_saved + d.delta_tokens_saved == d.tokens_saved_total
        # Both sides count booked requests only, so the unbooked 529 retry contributes
        # to neither. (This fixture's ledger totals are synthetic and independent of
        # its request rows, so the cross-file equality is asserted on real sessions.)
        assert all(r.get("booked") for r in d.booked_detail)
        assert len(d.booked_detail) == len(d.requests) - d.unbooked_requests

    def test_protected_share_explains_a_zero_percent_model(self) -> None:
        """A per-model 0.0% row must distinguish policy from a broken gate."""
        d = dz.dissect("s200-1")
        # m-big's census is all tool_result_digested -> work was done, not protected.
        assert (d.protected_share("m-big") or 0) < 50.0
        assert d.protected_share("no-such-model") is None

    def test_churn_advice_backs_off_when_the_prompt_cache_already_pays(self) -> None:
        """Don't send users after cache-delta on a session the provider already caches.

        Re-folded tokens look expensive, but when they are served from the prompt cache
        they are already discounted — recommending a dedup mechanism there overstates
        the recoverable saving by roughly the cache-read discount."""
        d = dz.dissect("s200-1")
        base = d._churn_advice()
        assert "session-delta cache absorbs" in base  # fixture is not cache-heavy

        for r in d.requests:
            r["usage_cache_read"] = 99_000
            r["usage_cache_create"] = 0
            r["usage_input_tokens"] = 1_000
        assert d.cached_input_share == pytest.approx(99.0, abs=0.01)
        advice = d._churn_advice()
        assert "prompt cache" in advice and "already discounted" in advice
        assert "session-delta cache absorbs" not in advice

    def test_legacy_rows_read_as_not_captured_not_zero(self) -> None:
        """Older records carry only the aggregate ``usage_cache_tokens``. Summing the
        split fields over those gives 0 cached against a nonzero input total, which
        renders a confident 0.0% — "the cache never hit" — for a session where it was
        never measured. Opposite diagnoses; they must not share a rendering."""
        d = dz.dissect("s200-1")
        for r in d.requests:
            r.pop("usage_cache_read", None)
            r.pop("usage_cache_create", None)
            r["usage_cache_tokens"] = 50_000  # the aggregate-only legacy shape
            r["usage_input_tokens"] = 1_000
        assert d.cached_input_share is None

        # One row carrying a split field is enough to make the number meaningful again.
        d.requests[0]["usage_cache_read"] = 0
        assert d.cached_input_share == pytest.approx(0.0)

    def test_recoverability_note_reports_evictions(self) -> None:
        """Never claim everything is recoverable when blobs have already aged out."""
        d = dz.dissect("s200-1")
        note = d._recoverability_note()
        rec = sum(1 for i in d.blocks.values() if i.get("recoverable"))
        if d.blocks and rec < len(d.blocks):
            assert "still recoverable" in note and f"{rec} of {len(d.blocks)}" in note
        else:
            assert note.startswith("Everything summarized stays recoverable")

    def test_churn(self) -> None:
        d = dz.dissect("s200-1")
        # aaaa1111 folded twice -> 2000 tokens re-digested across 1 re-folded block
        assert d.churn_tokens == 2000 and d.churned_blocks == 1

    def test_usage_and_calibration(self) -> None:
        d = dz.dissect("s200-1")
        assert d.usage_input_total == 3300 and d.usage_output_total == 450
        assert d.usage_requests == 2
        est, billed = d.calibration()
        # r1: 700 + (5000-3500) = 2200; r2: 500 + (2000-1200) = 1300
        # no learned factor yet -> identity, so the raw heuristic is reported as-is
        assert est == 3500 and billed == 3300

    def test_calibration_applies_learned_factor(self) -> None:
        """A model whose heuristic is known to run 2x low must not be reported as
        '2x off' — distil already learned the factor, so it belongs in the estimate."""
        from distil import calibration

        for _ in range(calibration.MIN_SAMPLES):
            calibration.record("m-big", 1000, 2000)
        est, billed = dz.dissect("s200-1").calibration()
        assert billed == 3300
        # 2200 x2.0 (m-big, learned) + 1300 x2.0 (m-small falls back to the pooled ratios)
        assert est == 7000

    def test_headroom_multiplier(self) -> None:
        d = dz.dissect("s200-1")
        assert d.headroom_multiplier == pytest.approx(10000 / 3900)

    def test_latency_by_path_and_forced_buffering(self) -> None:
        d = dz.dissect("s200-1")
        assert d.forced_buffered == 1
        lat = dict((k, (n, ms)) for k, n, ms in d.latency_by_path())
        assert lat["streamed"] == (1, 4000)
        assert lat["buffered (forced by expand)"] == (1, 9000)
        assert lat["buffered"] == (1, 500)

    def test_expansion_regret(self) -> None:
        d = dz.dissect("s200-1")
        assert d.expansion_regret() == [("log:l", 1, 1)]

    def test_no_anomalies_on_healthy_session(self) -> None:
        d = dz.dissect("s200-1")
        assert d.anomalies(dz.list_sessions()) == []

    def test_report_renders_insights(self) -> None:
        d = dz.dissect("s200-1")
        text = dz.render_text(d, color=False, peers=dz.list_sessions())
        assert "digest folds 3.90k (83%)" in text
        assert "34% of everything sent" in text
        assert "2.00k tokens re-digested" in text
        assert "estimate 3.50k vs billed 3.30k" in text
        assert "~2.6x" in text  # flat-rate headroom
        assert "buffered (forced by expand) 1 req @ 9.0s" in text
        assert "regret: log:l blocks pulled back 1/1" in text
        page = dz.render_html(d, peers=dz.list_sessions())
        assert "Digest folds" in page and "83% of savings" in page
        assert ">2.6\u00d7<" in page  # headroom tile
        payload = dz.to_json(d, dz.list_sessions())
        assert payload["insights"]["churn"]["tokens"] == 2000
        assert payload["insights"]["usage"]["calibration"] == {
            "estimated": 3500,
            "billed": 3300,
        }
        assert payload["insights"]["anomalies"] == []


class TestToolsAndCharts:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_tool_costs_aggregation(self) -> None:
        d = dz.dissect("s200-1")
        # bash: max 300/req, seen on 3 requests -> 900; gmail: 300 on 1 -> 300.
        assert d.tool_costs() == [("bash", 300, 900), ("mcp__gmail__search", 300, 300)]
        assert d.system_tokens_avg == 183 and d.tools_tokens_avg == 383
        assert d.system_growth() == (100, 250)

    def test_text_report_shows_tools_and_prompt_growth(self) -> None:
        text = dz.render_text(dz.dissect("s200-1"), color=False)
        assert "tool definitions (2): bash 300/req, mcp__gmail__search 300/req" in text
        assert "system prompt: 100 → 250 tokens over the session" in text

    def test_html_has_charts(self) -> None:
        page = dz.render_html(dz.dissect("s200-1"))
        assert page.count("<svg") == 3  # timeline, tools, block kinds
        # Validated categorical series, fixed order: overhead / sent / saved.
        assert "#3987e5" in page and "#199e70" in page and "#c98500" in page
        assert "overhead (system + tools)" in page  # legend, not color-alone
        assert "Tool definitions" in page and "bash" in page
        assert "data table" in page  # accessible table view of the timeline
        assert "Request composition" in page
        # Styled hover layer: JSON data-tip payloads rebuilt with textContent,
        # full-column/row hit targets, and the tooltip container + script.
        assert "request 2 · m-small" in page  # timeline column tip title
        assert "saved by distil" in page and "billed input" in page  # series rows
        assert page.count("data-tip=") >= 15  # tiles + columns + bar rows
        assert '<div id="tip" role="tooltip">' in page and "dataset.tip" in page
        assert "textContent" in page and "innerHTML" not in page  # names stay data
        # Plain-English help layer: section descriptions + tile explanations.
        assert "for what the term means" in page
        assert "Tokens distil avoided sending by replacing bulky content" in page
        assert 'class="desc"' in page
        assert "re-described to the model on every request" in page

    def test_json_has_tool_breakdown(self) -> None:
        payload = dz.to_json(dz.dissect("s200-1"))
        overhead = payload["insights"]["overhead"]
        assert overhead["tools"][0] == {
            "name": "bash",
            "tokens_per_request": 300,
            "session_tokens": 900,
        }
        assert overhead["system_growth"] == (100, 250)

    def test_charts_absent_without_detail(self) -> None:
        page = dz.render_html(dz.dissect("s300-9"))
        assert "<svg" not in page


class TestInteractivePicker:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        _seed_state(tmp_path)

    def _tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    def test_numbered_pick_renders_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        assert main(["dissect"]) == 0
        out = capsys.readouterr().out
        assert "#" in out and "s200-1" in out  # numbered list first
        assert "savings (input tokens, booked 2xx only)" in out  # then the report
        assert "codex" in out

    def test_enter_quits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        assert main(["dissect"]) == 0
        assert "savings (input tokens" not in capsys.readouterr().out

    def test_bad_pick_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "zzz")
        assert main(["dissect"]) == 2

    def test_non_tty_keeps_list_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        # capsys stdout is not a tty -> no prompt, list only (the old behavior).
        assert main(["dissect"]) == 0
        out = capsys.readouterr().out
        assert "pick one to dissect" in out
        assert "savings (input tokens" not in out


class TestAnomalies:
    def _session(self, home: Path, requests: list[dict[str, Any]], **manifest: Any) -> None:
        sess = home / "sessions"
        sess.mkdir(exist_ok=True)
        man = {
            "sid": "s900-1",
            "tool": "claude",
            "argv": ["claude"],
            "started_ts": 1.0,
            "distil_version": "1.15.0",
            "billing": "subscription",
            "flags": {
                "expand": False,
                "session_delta": False,
                "shadow_rate": 0.0,
                "lossless_only": False,
            },
        }
        man["flags"].update(manifest.pop("flags", {}))
        man.update(manifest)
        (sess / "s900-1.json").write_text(json.dumps(man))
        (sess / "s900-1.requests.jsonl").write_text(
            "\n".join(json.dumps(r) for r in requests) + "\n"
        )

    @staticmethod
    def _req(**over: Any) -> dict[str, Any]:
        base = {
            "ts": 10.0,
            "model": "m",
            "stream": True,
            "client_stream": True,
            "status": 200,
            "booked": True,
            "duration_ms": 100,
            "usage_input_tokens": None,
            "usage_output_tokens": None,
            "expanded_handles": [],
            "mode": "digest",
            "compressible_tokens": 100,
            "tokens_saved": 50,
            "overhead_tokens": 10,
            "delta_refs": 0,
            "delta_tokens_saved": 0,
            "prefix_msgs": 0,
            "shadow_sampled": False,
            "expanded": False,
            "output_shaping": "",
            "blocks": [{"h": "cccc3333", "sig": "log:m", "tokens": 40}],
        }
        base.update(over)
        return base

    def test_silent_shadow_flagged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        self._session(tmp_path, [self._req() for _ in range(12)], flags={"shadow_rate": 0.5})
        warnings = dz.dissect("s900-1").anomalies()
        assert any("shadow may be silently failing" in w for w in warnings)

    def test_expand_never_intercepted_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        # expand on, folds happening, yet every request streamed straight through.
        self._session(tmp_path, [self._req() for _ in range(12)], flags={"expand": True})
        warnings = dz.dissect("s900-1").anomalies()
        assert any("could never be intercepted" in w for w in warnings)

    def test_unrecoverable_expand_is_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F3: the saving was booked when the block was folded, but the original
        was gone when the agent asked for it. The request still succeeds, so this
        is invisible unless it is counted on its own."""
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        self._session(
            tmp_path,
            [self._req(expanded=True), self._req(expand_misses=2)],
            flags={"expand": True},
        )
        d = dz.dissect("s900-1")
        assert (d.expand_resolved, d.expand_missed) == (1, 2)
        assert any("could not be recovered" in w for w in d.anomalies())

    def test_unbooked_spike_flagged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        reqs = [self._req() for _ in range(6)] + [
            self._req(booked=False, status=529) for _ in range(4)
        ]
        self._session(tmp_path, reqs)
        warnings = dz.dissect("s900-1").anomalies()
        assert any("not booked" in w for w in warnings)

    def test_calibration_drift_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        # estimate 10+ (100-50) = 60 vs billed 20 -> ratio 3.0
        self._session(tmp_path, [self._req(usage_input_tokens=20)])
        warnings = dz.dissect("s900-1").anomalies()
        assert any("off by >50% vs billed" in w for w in warnings)

    def test_cached_prefix_counts_as_billed_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """input_tokens alone omits the cached prefix, which inflates the output share."""
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        self._session(
            tmp_path,
            [
                self._req(
                    usage_input_tokens=20,
                    usage_cache_tokens=9980,
                    usage_output_tokens=100,
                )
            ],
        )
        d = dz.dissect("s900-1")
        assert d.usage_input_total == 10_000
        assert d.calibration() == (60, 10_000)
        # 100/10100, not the 83% you get from counting uncached input only.
        assert any("(1% of billed traffic)" in h for h, _ in d.headlines())


class TestHeadlines:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_layman_story(self) -> None:
        heads = dict(dz.dissect("s200-1").headlines())
        assert "distil kept 6.10k of 10.00k input tokens off the wire (61%)" in heads
        assert (
            "mostly by summarizing large logs"
            in heads["distil kept 6.10k of 10.00k input tokens off the wire (61%)"]
        )
        out_head = "The model wrote 450 output tokens (12% of billed traffic)"
        assert out_head in heads
        # Subscription session -> the output-shaping gate is explained.
        assert "never shortened on a subscription" in heads[out_head]
        assert any("resent and re-summarized" in h for h in heads)
        assert any("spot-checked" in h for h in heads)

    def test_story_leads_the_renderers(self) -> None:
        d = dz.dissect("s200-1")
        text = dz.render_text(d, color=False)
        assert text.index("what happened") < text.index("savings (input tokens")
        page = dz.render_html(d)
        assert page.index("What happened") < page.index("Per model")
        payload = dz.to_json(d)
        assert payload["headlines"][0]["headline"].startswith("distil kept 6.10k")

    def test_no_story_without_data(self, tmp_path: Path) -> None:
        assert dz.dissect("s999-0").headlines() == []


class TestServePortal:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    @pytest.fixture()
    def portal(self) -> Any:
        server = dz.make_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}"
        server.shutdown()

    def _get(self, url: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    def test_index_lists_sessions_as_links(self, portal: str) -> None:
        status, body = self._get(portal + "/")
        assert status == 200
        assert "s200-1" in body and "s300-9" in body and "codex" in body
        assert "/session/s200-1" in body  # clickable rows

    def test_session_report_served_live(self, portal: str) -> None:
        status, body = self._get(portal + "/session/s200-1")
        assert status == 200
        assert "What happened" in body and "61.0%" in body
        assert "← sessions" in body  # back link injected before the title

    def test_prefix_and_latest_resolve(self, portal: str) -> None:
        assert self._get(portal + "/session/s200")[0] == 200
        status, body = self._get(portal + "/json/latest")
        assert status == 200
        assert json.loads(body)["session"] == "s300-9"

    def test_unknown_session_404(self, portal: str) -> None:
        assert self._get(portal + "/session/zzz")[0] == 404
        assert self._get(portal + "/nope")[0] == 404

    def test_traversal_is_not_a_session(self, portal: str) -> None:
        status, _ = self._get(portal + "/session/..%2F..%2Fetc%2Fpasswd")
        assert status == 404

    def test_index_polls_in_place_instead_of_meta_refresh(self, portal: str) -> None:
        """The picker updates itself without reloading (WCAG 2.2.1/2.2.4): no meta
        refresh, a keyed tbody to patch, and a visible control to stop the polling."""
        _status, body = self._get(portal + "/")
        assert 'http-equiv="refresh"' not in body
        assert 'id="sessions-body"' in body
        assert 'id="pause-btn"' in body and 'aria-pressed="false"' in body
        assert 'data-sid="s200-1"' in body  # rows keyed so the poll can patch them

    def test_sessions_json_backs_the_poll(self, portal: str) -> None:
        """/sessions.json is what the poll fetches — same rows as the rendered table."""
        status, body = self._get(portal + "/sessions.json")
        assert status == 200
        rows = json.loads(body)
        by_sid = {r["sid"]: r for r in rows}
        assert {"s200-1", "s300-9"} <= set(by_sid)
        row = by_sid["s200-1"]
        assert set(row) == {"sid", "tool", "started", "last", "requests", "pct", "status"}
        assert isinstance(row["pct"], float)
        # the JSON carries only session metadata — never prompt or response content
        assert "content" not in body and "messages" not in body


class TestTranscriptCorrelation:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "distil"))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        (tmp_path / "distil").mkdir()
        _seed_state(tmp_path / "distil")
        self.transcript = _write_transcript(tmp_path / "claude")

    def test_adapter_parses_the_shape(self) -> None:
        from distil.transcripts.claude_code import ClaudeCodeAdapter

        tr = ClaudeCodeAdapter().load(self.transcript)
        assert tr.label == "fix the flaky tests"
        assert [t.text for t in tr.turns] == ["please run the tests", "now fix the bug"]
        assert [c.name for c in tr.tool_calls] == ["bash", "Read"]
        assert len(tr.tool_results) == 2
        assert tr.tool_results[0].tool == "bash"  # resolved via tool_use_id
        assert tr.tool_results[1].turn == 2  # attributed to the second turn

    def test_discovery_by_window_and_cwd(self) -> None:
        from distil.transcripts import find_transcript

        tr = find_transcript("claude", (900.0, 1800.0), cwd="/tmp/proj")
        assert tr is not None and tr.path == self.transcript
        # A window that ends before the transcript's first event can't match.
        assert find_transcript("claude", (100.0, 200.0), cwd="/tmp/proj") is None

    def test_registry_is_the_extension_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proof of the contract: a new agent = one adapter + one registry entry."""
        from distil import transcripts
        from distil.transcripts.base import ToolResult, Transcript, UserTurn

        class ToyAdapter:
            name = "toyagent"

            def discover(self, window: Any, cwd: Any) -> list[Path]:
                return [Path("/toy.log")]

            def load(self, path: Path) -> Transcript:
                return Transcript(
                    agent=self.name,
                    path=path,
                    label="toy session",
                    turns=[UserTurn(index=1, ts=1.0, text="hi")],
                    tool_results=[ToolResult(ts=2.0, text="x" * 50, tool="toytool", turn=1)],
                )

        monkeypatch.setitem(transcripts.ADAPTERS, "toyagent", ToyAdapter())
        tr = transcripts.find_transcript("toyagent", (0.0, 10.0))
        assert tr is not None and tr.agent == "toyagent" and tr.label == "toy session"

    def _corr(self) -> Any:
        from distil.correlate import correlate
        from distil.transcripts import find_transcript

        d = dz.dissect("s200-1")
        tr = find_transcript("claude", (d.started, d.ended), cwd="/tmp/proj")
        assert tr is not None
        return d, correlate(d, tr)

    def test_fold_sources_named_by_content(self) -> None:
        _d, corr = self._corr()
        assert len(corr.fold_sources) == 1  # bbbb2222's blob is expired -> unnamed
        src = corr.fold_sources[0]
        assert src.handle == "aaaa1111" and src.tool == "bash" and src.turn == 1
        assert src.refetches == 2  # seen again at turn 2 -> re-fetch after fold
        assert corr.refetched and corr.refetched[0].handle == "aaaa1111"
        assert corr.unnamed_blocks == 1

    def test_unused_tools_defined_minus_invoked(self) -> None:
        _d, corr = self._corr()
        assert corr.tools_defined == 2 and corr.tools_invoked == 1
        assert corr.unused_tools == [("mcp__gmail__search", 300)]
        assert corr.unused_tokens_per_request == 300

    def test_turn_cost_attribution(self) -> None:
        _d, corr = self._corr()
        by_index = {t.index: t for t in corr.turns}
        assert by_index[1].requests == 1 and by_index[1].baseline_tokens == 5700
        assert by_index[2].requests == 2 and by_index[2].baseline_tokens == 3000
        assert corr.turns[0].index == 1  # costliest first

    def test_renderers_carry_the_correlation(self) -> None:
        d, corr = self._corr()
        text = dz.render_text(d, color=False, corr=corr)
        assert "conversation correlation (claude transcript: fix the flaky tests)" in text
        assert "bash output" in text and "unused tools (1/2 defined" in text
        assert "re-fetched after fold" in text
        assert 'turn 1: "please run the tests"' in text
        page = dz.render_html(d, corr=corr)
        assert "Conversation correlation" in page and "never invoked" in page
        assert "mcp__gmail__search" in page and "reappeared in 2 separate tool results" in page
        payload = dz.to_json(d, corr=corr)
        assert payload["correlation"]["tools"]["unused"] == [
            {"name": "mcp__gmail__search", "tokens_per_request": 300}
        ]
        assert payload["correlation"]["fold_sources"][0]["tool"] == "bash"

    def test_cli_transcript_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "s200", "--no-color", "--transcript"]) == 0
        out = capsys.readouterr().out
        assert "conversation correlation" in out

    def test_cli_no_transcript_found_is_a_note(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/nonexistent")
        assert main(["dissect", "s200", "--no-color", "--transcript"]) == 0
        out = capsys.readouterr().out
        assert "no matching agent transcript found" in out

    def test_serve_with_transcript_flag_correlates_by_default(self) -> None:
        server = dz.make_server("127.0.0.1", 0, transcript="auto")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/session/s200-1") as resp:
                page = resp.read().decode()
            assert "Conversation correlation" in page  # default on
            assert 'href="/session/s200-1?t=0"' in page  # opt-out link
            with urllib.request.urlopen(base + "/session/s200-1?t=0") as resp:
                assert "Conversation correlation" not in resp.read().decode()
        finally:
            server.shutdown()

    def test_portal_correlation_toggle(self) -> None:
        server = dz.make_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/session/s200-1") as resp:
                plain = resp.read().decode()
            assert "+ correlate with transcript" in plain
            assert "Conversation correlation" not in plain
            with urllib.request.urlopen(base + "/session/s200-1?t=1") as resp:
                correlated = resp.read().decode()
            assert "Conversation correlation" in correlated and "hide correlation" in correlated
        finally:
            server.shutdown()


class TestScanUsage:
    def test_json_body(self) -> None:
        from distil.streamrelay import scan_usage

        assert scan_usage(_UPSTREAM_JSON) == {"input_tokens": 4321, "output_tokens": 87}

    def test_sse_body_takes_last_output(self) -> None:
        from distil.streamrelay import scan_usage

        assert scan_usage(_UPSTREAM_SSE) == {"input_tokens": 1234, "output_tokens": 55}

    def test_no_usage(self) -> None:
        from distil.streamrelay import scan_usage

        assert scan_usage(b'{"error": "overloaded"}') == {}


class TestTolerantReader:
    def test_read_jsonl_skips_garbage(self, tmp_path: Path) -> None:
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\nnot json\n[1,2]\n\n{"b": 2}\n')
        assert dz._read_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_read_jsonl_missing_file(self, tmp_path: Path) -> None:
        assert dz._read_jsonl(tmp_path / "absent.jsonl") == []


class TestEligibility:
    """The census surface: why a savings number is what it is."""

    @staticmethod
    def _session(home: Path, censuses: list[dict[str, int]]) -> str:
        sess = home / "sessions"
        sess.mkdir(exist_ok=True)
        (sess / "s-elig.requests.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "ts": 1000.0 + i,
                        "model": "m",
                        "stream": True,
                        "status": 200,
                        "booked": True,
                        "mode": "digest",
                        "compressible_tokens": sum(c.values()),
                        "tokens_saved": 100,
                        "system_tokens": 10,
                        "tools_tokens": 50,
                        "census": c,
                    }
                )
                for i, c in enumerate(censuses)
            )
            + "\n"
        )
        return "s-elig"

    def test_sums_across_requests_and_ranks_by_size(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        sid = self._session(
            tmp_path,
            [{"assistant_text": 100, "tool_result_digested": 50}, {"assistant_text": 200}],
        )
        elig = dz.dissect(sid).eligibility
        assert [r for r, _t, _p in elig] == ["assistant_text", "tool_result_digested"]
        assert elig[0][1] == 300
        assert abs(sum(p for _r, _t, p in elig) - 100.0) < 0.01

    def test_absent_census_reads_as_unknown_not_as_zero(self, tmp_path, monkeypatch) -> None:
        """Sessions recorded before the census must not render as "nothing was
        eligible" — that is a different, and wrong, claim."""
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        sid = self._session(tmp_path, [{}, {}])
        d = dz.dissect(sid)
        assert d.eligibility == []
        assert "why:" not in dz.render_text(d, color=False)

    def test_a_declined_share_is_reported_even_when_most_content_is_protected(
        self, tmp_path, monkeypatch
    ) -> None:
        """The bucket that indicts the compressor must not be masked by the one that
        exonerates it. Protected content here is 63% — a naive "mostly protected, all
        is well" summary would hide a 25% refusal rate behind it."""
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        sid = self._session(
            tmp_path,
            [
                {
                    "assistant_text": 54_000,
                    "tool_result_recent": 9_000,
                    "tool_result_declined": 25_000,
                    "tool_result_digested": 12_000,
                }
            ],
        )
        text = dz.render_text(dz.dissect(sid), color=False)
        assert "digester declined" in text
        assert "compressor's to explain" in text
        assert "the design holding" not in text

    def test_protected_majority_is_explained_as_working_not_broken(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        sid = self._session(tmp_path, [{"assistant_text": 80_000, "tool_result_digested": 20_000}])
        text = dz.render_text(dz.dissect(sid), color=False)
        assert "the design holding" in text
        assert "compressor's to explain" not in text


class TestErrorReasons:
    """A non-2xx must record WHY, not only that it failed."""

    @staticmethod
    def _d(requests):
        d = dz.Dissection.__new__(dz.Dissection)
        object.__setattr__(d, "requests", requests)
        return d

    def test_error_types_are_counted_most_frequent_first(self) -> None:
        d = self._d(
            [
                {"status": 400, "error_type": "invalid_request_error"},
                {"status": 400, "error_type": "invalid_request_error"},
                {"status": 429, "error_type": "rate_limit_error"},
                {"status": 200},
            ]
        )
        assert d.error_reasons() == [("invalid_request_error", 2), ("rate_limit_error", 1)]

    def test_sessions_recorded_before_the_field_existed_report_nothing(self) -> None:
        """Old rows carry only a status — report no reasons rather than inventing one."""
        d = self._d([{"status": 400}, {"status": 200}])
        assert d.error_reasons() == []


def test_upstream_error_type_is_content_free():
    """The provider's error TYPE is kept; its message is not — it quotes content."""
    from distil.proxy import _upstream_error_type

    body = (
        b'{"error":{"type":"invalid_request_error",'
        b'"message":"prompt is too long: 231721 tokens > 200000 maximum"}}'
    )
    assert _upstream_error_type(400, body) == "invalid_request_error"
    assert _upstream_error_type(200, b"{}") is None, "a success has no error type"
    # Streaming responses are relayed frame-by-frame and never buffered.
    assert _upstream_error_type(400, b"") == "http_400"
    assert _upstream_error_type(500, b"<html>not json</html>") == "http_500"

    # The message must never reach the record.
    for probe in (b"", b'{"error":{"type":"x","message":"secret content"}}'):
        got = _upstream_error_type(400, probe) or ""
        assert "secret" not in got and "231721" not in got


class TestRetryRecovery:
    """A retried-and-succeeded failure is not a failure the user must act on."""

    @staticmethod
    def _d(requests):
        d = dz.Dissection.__new__(dz.Dissection)
        object.__setattr__(d, "requests", requests)
        return d

    def test_a_failure_followed_by_a_same_size_success_is_recovered(self) -> None:
        """The real pattern from a live session: 400 at 420,136 tokens, then an
        immediate 200 at the same size. The SDK retried; no work was lost."""
        d = self._d(
            [
                {"ts": 1, "status": 400, "compressible_tokens": 420_000, "overhead_tokens": 136},
                {"ts": 2, "status": 200, "compressible_tokens": 420_000, "overhead_tokens": 136},
            ]
        )
        assert d.retried_and_recovered() == 1

    def test_a_failure_with_no_retry_is_not_counted_as_recovered(self) -> None:
        d = self._d([{"ts": 1, "status": 400, "compressible_tokens": 1000}])
        assert d.retried_and_recovered() == 0

    def test_an_unrelated_later_success_is_not_a_retry(self) -> None:
        """A 2xx of a very different size is the next turn, not a retry of this one."""
        d = self._d(
            [
                {"ts": 1, "status": 400, "compressible_tokens": 400_000},
                {"ts": 2, "status": 200, "compressible_tokens": 5_000},
            ]
        )
        assert d.retried_and_recovered() == 0

    def test_rows_are_ordered_by_timestamp_not_file_order(self) -> None:
        d = self._d(
            [
                {"ts": 9, "status": 200, "compressible_tokens": 100_000},
                {"ts": 1, "status": 400, "compressible_tokens": 100_000},
                {"ts": 2, "status": 200, "compressible_tokens": 100_000},
            ]
        )
        assert d.retried_and_recovered() == 1

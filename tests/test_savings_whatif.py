"""`distil savings` what-if: offline replay of the user's own transcripts, both modes.

Every test runs inside conftest's HOME/DISTIL_HOME sandbox; transcripts are synthetic but
shaped like Claude Code's own JSONL (streamed assistant blocks sharing a message id, one
usage per response, tool_result user lines, non-message record types, a sidechain).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from distil import cli, ledger, whatif
from distil import savings_screen as ss

SECRET = "SECRET-LOG-LINE-7b3d"
PROMPT = "PRIVATE-PROMPT-c41a"
PROJECT = "-Users-someone-private-repo"
DAY = 86400.0


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _tool_output(i: int) -> str:
    # Varied, log-shaped output: a digest has something to take, and it is not filler.
    return "\n".join(
        f"2026-10-0{1 + j % 3}T12:{j % 60:02d}:00 INFO worker-{j % 7} {SECRET} step={i}.{j} "
        f"latency_ms={(j * 37 + i) % 997} path=/srv/app/module_{j % 13}.py ok={j % 5 != 0}"
        for j in range(150)
    )


def session_rows(start: float, turns: int, *, model: str = "claude-sonnet-4-6") -> list[Any]:
    rows: list[Any] = [
        {"type": "summary", "summary": "not a message"},
        {
            "type": "user",
            "timestamp": _iso(start),
            "uuid": "h0",
            "origin": {"kind": "human"},
            "cwd": "/Users/someone/" + PROJECT,
            "message": {"role": "user", "content": PROMPT},
        },
    ]
    for i in range(turns):
        ts = start + 10 * (i + 1)
        mid = f"msg_{start:.0f}_{i}"  # unique across sessions, as real ids are
        usage = {
            "input_tokens": 50,
            # roughly what the turns add up to: 150 log lines are ~8.2k tokens (offline estimate)
            "cache_read_input_tokens": 20_000 + 8_300 * i,
            "cache_creation_input_tokens": 8_500,
            "cache_creation": {"ephemeral_1h_input_tokens": 8_500, "ephemeral_5m_input_tokens": 0},
            "output_tokens": 300,
        }
        for block in (
            {"type": "text", "text": f"looking at {PROMPT} step {i}"},
            {"type": "tool_use", "id": f"tu{i}", "name": "Bash", "input": {"command": "make test"}},
        ):
            rows.append(
                {
                    "type": "assistant",
                    "timestamp": _iso(ts),
                    "uuid": f"a{i}{block['type']}",
                    "message": {
                        "id": mid,
                        "role": "assistant",
                        "model": model,
                        "content": [block],
                        "usage": usage,
                    },
                }
            )
        rows.append(
            {
                "type": "user",
                "timestamp": _iso(ts + 5),
                "uuid": f"r{i}",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"tu{i}", "content": _tool_output(i)}
                    ],
                },
            }
        )
    rows.append({"type": "user", "isSidechain": True, "message": {"role": "user", "content": "x"}})
    rows.append("{not json")
    return rows


def write_session(name: str, start: float, turns: int, **kw: Any) -> Path:
    root = whatif.claude_projects_root() / PROJECT
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{name}.jsonl"
    rows = session_rows(start, turns, **kw)
    p.write_text(
        "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
        encoding="utf-8",
    )
    os.utime(p, (start + 10 * turns + 5, start + 10 * turns + 5))
    return p


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    now = time.time()
    write_session("s1", now - 3600, 12)
    write_session("s2", now - 7200, 6)
    return now


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


# --------------------------------------------------------------------------- engine


def test_parse_keeps_timestamps_model_and_write_split():
    rows = session_rows(1_000_000.0, 3)
    sess = whatif.parse_transcript(r if isinstance(r, str) else json.dumps(r) for r in rows)
    assert len(sess.ts) == len(sess.messages) == 7  # prompt + 3 x (assistant, result)
    assert sess.bad_lines == 1  # "{not json"; the summary and the sidechain are not errors
    assert sess.model == "claude-sonnet-4-6"
    # usage repeats on both streamed lines of a response: counted once per response
    assert sess.cache_write == 3 * 8_500 and sess.cache_write_1h == 3 * 8_500


def test_replay_measures_both_modes(api_key):
    w = whatif.run(api_key - 7 * DAY)
    # every user message ends a request; the last one per session has no billed response
    # (interrupted), so it is replayed for cache state but not counted
    assert w.sessions == 2 and w.requests_replayed == 12 + 6
    assert w.failed_sessions == 0 and not w.stopped_early
    assert w.digest.tokens_before == w.lossless.tokens_before > 0
    # digest takes the old log-shaped tool output; lossless-only never takes more
    assert w.digest.removed > w.lossless.removed >= 0
    assert w.digest.usd_before > w.digest.usd_after
    assert w.billed_input_tokens == sum(
        50 + 20_000 + 8_300 * i + 8_500 for n in (12, 6) for i in range(n)
    )
    d = w.to_dict(2 * w.billed_input_tokens, 99)
    assert d["scale"] == pytest.approx(2.0) and d["requests_in_window"] == 99
    assert d["digest"]["input_tokens_removed"] == 2 * w.digest.removed
    assert d["digest"]["share_of_billed_input"] > 0.1
    assert w.to_dict(1, 0)["scale"] == 1.0  # never scaled down


def test_window_excludes_old_requests(api_key):
    write_session("old", api_key - 30 * DAY, 8)
    assert whatif.run(api_key - 7 * DAY).requests_replayed == 18
    assert whatif.run(None).requests_replayed == 18 + 8


def test_request_cap_is_respected(api_key):
    w = whatif.run(api_key - 7 * DAY, max_requests=5, per_session=5)
    assert 0 < w.requests_replayed <= 5


def test_deadline_stops_early_and_keeps_modes_comparable(api_key):
    w = whatif.run(api_key - 7 * DAY, deadline_s=0.0)
    assert w.stopped_early and w.requests_replayed == 0
    assert w.digest.tokens_before == w.lossless.tokens_before == 0


def test_empty_and_missing_roots():
    assert whatif.run(None).requests_replayed == 0  # no ~/.claude at all
    whatif.claude_projects_root().mkdir(parents=True)
    (whatif.claude_projects_root() / "empty.jsonl").write_text("", encoding="utf-8")
    (whatif.claude_projects_root() / "junk.jsonl").write_text("[]\nnope\n", encoding="utf-8")
    w = whatif.run(None)
    assert w.requests_replayed == 0 and w.unparseable_lines == 2


def test_adapter_failure_costs_the_session_not_the_screen(api_key, monkeypatch):
    from distil.adapters import anthropic as adapter

    def boom(*_a: Any, **_k: Any) -> Any:
        raise TypeError("bad block")

    monkeypatch.setattr(adapter, "compress_messages", boom)
    w = whatif.run(api_key - 7 * DAY)
    assert w.failed_sessions == 2 and w.requests_replayed == 0
    assert w.digest.tokens_before == 0


def test_unpriced_model_counts_tokens_not_dollars(monkeypatch):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    write_session("g", time.time() - 600, 4, model="gpt-9")
    w = whatif.run(None)
    assert w.unpriced_requests == w.requests_replayed == 4
    assert w.digest.usd_before == 0 and w.digest.tokens_before > 0


# --------------------------------------------------------------------------- the screen


def test_no_ledger_api_key_screen(api_key, capsys):
    before = _tree(Path(os.environ["HOME"]) / ".claude")
    assert cli.main(["savings", "--days", "7"]) == 0
    out = capsys.readouterr().out
    assert "what distil would have changed  (replayed 18 of" in out
    assert "lossless-only" in out and "digest" in out and "saved" in out
    assert "API-key default" in out
    assert "it can't see whether the agent would" in out and "`distil ab`" in out
    assert f"next     {ss.NEXT_STEP}" in out
    assert "flat plan" not in out
    assert _tree(Path(os.environ["HOME"]) / ".claude") == before  # read-only
    assert not any(Path(os.environ["DISTIL_HOME"]).iterdir())  # nothing written


def test_no_ledger_subscription_is_rate_limit_headroom(api_key, monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "flat plan: no per-token bill, so read this as rate-limit headroom" in out
    assert "/day" in out
    assert "← your default" in out and f"opt in: {ss.DIGEST_OPT_IN}" in out
    assert "saved" not in out.split("what distil would have changed")[1].split("next")[0]


def test_json_carries_the_whatif(api_key, capsys):
    assert cli.main(["savings", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    w = d["whatif"]
    assert d["mode"] == "transcripts" and w["requests_replayed"] == 18
    assert set(w) >= {"lossless_only", "digest", "scale", "requests_in_window", "method"}
    assert w["digest"]["share_of_billed_input"] > w["lossless_only"]["share_of_billed_input"]


def test_never_leaks_content(api_key, capsys):
    assert cli.main(["savings"]) == 0
    assert cli.main(["savings", "--json"]) == 0
    out = capsys.readouterr().out
    for needle in (SECRET, PROMPT, PROJECT, "private-repo", "make test", "worker-", "s1.jsonl"):
        assert needle not in out


def test_bad_days_is_a_clean_error(capsys):
    assert cli.main(["savings", "--days", "0"]) == 2
    assert "--days must be positive" in capsys.readouterr().err


def test_empty_transcripts_still_say_how_to_start(capsys):
    whatif.claude_projects_root().mkdir(parents=True)
    assert cli.main(["savings"]) == 0
    assert "nothing to show yet" in capsys.readouterr().out


def _ledger(now: float) -> None:
    home = Path(os.environ["DISTIL_HOME"])
    (home / "sessions").mkdir(parents=True)
    (home / "sessions" / "s.requests.jsonl").write_text(
        json.dumps(
            {
                "ts": now - 60,
                "booked": True,
                "model": "claude-sonnet-4-6",
                "usage_input_tokens": 1000,
                "usage_cache_read": 100_000,
                "usage_cache_create": 10_000,
                "usage_output_tokens": 500,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = {
        "trajectory_id": "live-proxy",
        "model": "claude-sonnet-4-6",
        "turns": 1,
        "baseline_dollars": 0.003,
        "distil_dollars": 0.0029,
        "baseline_input_tokens": 1000,
        "distil_input_tokens": 990,
        "tokenizer": "heuristic",
        "ts": now - 60,
    }
    ledger.default_path().write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_ledger_subscription_adds_one_digest_line(api_key, monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    _ledger(api_key)
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "saved    $" in out  # the ledger screen is still the ledger screen
    line = [ln for ln in out.splitlines() if ln.startswith("  what-if")]
    assert len(line) == 1 and "digest would remove" in line[0]
    assert "what distil would have changed" not in out


def test_ledger_metered_key_runs_no_replay(api_key, monkeypatch, capsys):
    _ledger(api_key)
    monkeypatch.setattr(whatif, "run", lambda *a, **k: pytest.fail("replayed on a metered key"))
    assert cli.main(["savings", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["whatif"] is None


def test_digest_line_hidden_when_not_meaningful():
    s = ss.Screen("ledger", None, 7.0)
    arm = {"input_tokens_removed": 100, "share_of_billed_input": 0.001, "usd_saved": 0.0}
    s.whatif = {"lossless_only": dict(arm), "digest": dict(arm, input_tokens_removed=150)}
    assert ss._digest_would_add(s) == []


def test_no_replayable_request_falls_back_to_the_quoted_estimate(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    p = write_session("only-usage", time.time() - 600, 2)
    rows = [r for r in p.read_text(encoding="utf-8").splitlines() if '"assistant"' in r]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")  # usage, but no request to replay
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "ESTIMATE  distil would save" in out and "what distil would have changed" not in out


def test_render_names_a_cost_increase_the_time_cap_and_unpriced_requests():
    s = ss.Screen("transcripts", 0.0, 7.0, requests=10)
    arm = {"input_tokens_removed": 1000, "share_of_billed_input": 0.05, "usd_saved": -1.5}
    s.whatif = {
        "requests_replayed": 4,
        "requests_in_window": 10,
        "stopped_early": True,
        "unpriced_requests": 2,
        "lossless_only": dict(arm, usd_saved=0.0),
        "digest": arm,
    }
    out = "\n".join(ss._render_whatif(s))
    assert "≈$1.50 MORE (cache rewrites)" in out
    assert "stopped at the time cap" in out
    assert "2 replayed requests on an unpriced model" in out


# --------------------------------------------------------------------------- digest status


def test_whatif_names_digest_status_and_the_published_state(api_key, monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert cli.main(["savings"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "digest on this machine: not enough evidence yet" in out
    assert "over its decision-change budget" in out and "maintainer's live data" in out
    assert cli.main(["savings", "--json"]) == 0
    cert = json.loads(capsys.readouterr().out)["whatif"]["digest_certification"]
    assert cert["state"] == "no-evidence"


def test_a_held_digest_is_never_recommended(api_key, monkeypatch, capsys):
    # The per-mode certification hold persisted by a proxy on this machine (ADR 0022).
    (Path(os.environ["DISTIL_HOME"]) / "cert-hold.json").write_text(
        json.dumps({"why": "digest: paired harm 5.3 pp, upper bound 8.7 pp > the 5% budget"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "held here: not recommended" in out and "held by the per-mode hold" in out
    assert f"opt in: {ss.DIGEST_OPT_IN}" not in out
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "held here: served as lossless-only" in out and "API-key default" not in out


def test_ledger_digest_line_does_not_recommend_a_held_digest():
    s = ss.Screen("ledger", None, 7.0)
    arm = {"input_tokens_removed": 100, "share_of_billed_input": 0.01, "usd_saved": 0.0}
    s.whatif = {
        "lossless_only": dict(arm),
        "digest": dict(arm, input_tokens_removed=900, share_of_billed_input=0.09),
        "digest_certification": {"state": "held", "detail": "held by the per-mode hold — x"},
    }
    text = " ".join(" ".join(ss._digest_would_add(s)).split())
    assert "digest would remove" in text and "not recommended" in text
    assert ss.DIGEST_OPT_IN not in text and "maintainer's live data" in text
    s.whatif["digest_certification"] = {"state": "certified", "detail": "certified — y"}
    assert ss.DIGEST_OPT_IN in "\n".join(ss._digest_would_add(s))

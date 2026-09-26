"""The front door: four commands in `--help`, one savings screen, one guided setup.

Every test runs inside conftest's HOME/DISTIL_HOME sandbox; nothing here reads the
developer's real ~/.claude or ~/.distil.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from distil import cli, discover, pricing
from distil import savings_screen as ss

REPO = Path(__file__).resolve().parents[1]


def _sub(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))


# --------------------------------------------------------------------------- help


def test_help_shows_exactly_the_four_commands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    listed = [ln.split()[0] for ln in out.splitlines() if ln.startswith("  ") and ln.split()]
    assert [w for w in listed if not w.startswith("-")] == ["setup", "wrap", "savings", "doctor"]
    assert "--version" in out
    assert out.rstrip().endswith("More: distil --help-all")
    # nothing else leaks in: no other command name appears as a row
    for name in _sub(cli.build_parser()).choices:
        if name not in cli.FRONT_DOOR:
            assert f"\n  {name} " not in out


def test_short_h_is_the_front_door_too(capsys):
    with pytest.raises(SystemExit):
        cli.main(["-h"])
    assert "More: distil --help-all" in capsys.readouterr().out


def test_help_all_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help-all"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for action in _sub(cli.build_parser())._choices_actions:
        assert action.dest in out, action.dest
    assert "everyday commands:" in out  # the old epilog is still there


def test_every_legacy_command_still_parses(capsys):
    """Hidden from the front door, not removed: every subcommand's own --help works."""
    names = sorted(_sub(cli.build_parser()).choices)
    assert len(names) > 40
    for name in names:
        with pytest.raises(SystemExit) as e:
            cli.build_parser().parse_args([name, "--help"])
        assert e.value.code == 0, name
    capsys.readouterr()


# --------------------------------------------------------------------------- savings: dispatch


@pytest.mark.parametrize(
    "argv",
    [
        ["savings", "--strategies"],
        ["savings", "--pricing", "claude-opus-4-8"],
        ["savings", "--record"],
        ["savings", "--tokenizer", "heuristic"],
        ["savings", "--output-tokens-per-turn", "0"],
        ["savings", "-t", "x.json"],
    ],
)
def test_legacy_savings_flags_route_to_the_strategy_pricer(argv, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_savings", lambda a: seen.append(a) or 0)
    assert cli.main(argv) == 0
    assert len(seen) == 1


def test_legacy_savings_output_unchanged(capsys):
    assert cli.main(["savings", "--strategies"]) == 0
    out = capsys.readouterr().out
    assert "model claude-opus-4-8" in out and "tokenizer=heuristic" in out
    assert "distil (cache-aware lossless)" in out and "cheaper), reversibly." in out


def test_bad_window_is_a_clean_error(capsys):
    assert cli.main(["savings", "--since", "soon"]) == 2
    assert "cannot read 'soon'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text,secs", [("7d", 7 * 86400), ("24h", 86400), ("2w", 14 * 86400), ("3", 3 * 86400)]
)
def test_parse_since(text, secs):
    assert ss.parse_since(text) == secs


def test_daily_series_is_consecutive_distinct_and_capped():
    from datetime import date, timedelta

    now = time.time()
    old = ss._day(now - 90 * 86400)
    days = ss._daily({old: ss.Day(old, 1.0)}, None, now)
    assert len(days) == ss.MAX_GRAPH_DAYS
    dates = [date.fromisoformat(d.date) for d in days]
    assert all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:]))
    assert dates[-1] == date.fromtimestamp(now)
    assert ss._daily({}, None, now) == []


def test_bar_resolution():
    assert ss.bar(0, 10) == " " * ss.GRAPH_WIDTH
    assert ss.bar(10, 10) == "█" * ss.GRAPH_WIDTH
    assert ss.bar(5, 10).rstrip() == "█" * (ss.GRAPH_WIDTH // 2)
    assert ss.bar(0.01, 10).strip() == "▏"  # a nonzero day is never drawn as empty
    assert len(ss.bar(3.3, 10)) == ss.GRAPH_WIDTH


# --------------------------------------------------------------------------- savings: ledger mode

DAY = 86400.0


def _home() -> Path:
    return Path(os.environ["DISTIL_HOME"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _req(ts: float, **kw: object) -> dict:
    row = {
        "ts": ts,
        "booked": True,
        "model": "claude-opus-4-8",
        "usage_input_tokens": 1_000_000,
        "usage_cache_read": 10_000_000,
        "usage_cache_create": 1_000_000,
        "usage_output_tokens": 100_000,
    }
    row.update(kw)
    return row


def _saved_row(ts: float, base: int, got: int) -> dict:
    return {
        "trajectory_id": "live-proxy",
        "model": "claude-opus-4-8",
        "turns": 1,
        "baseline_dollars": base * 5e-6,
        "distil_dollars": got * 5e-6,
        "baseline_input_tokens": base,
        "distil_input_tokens": got,
        "tokenizer": "heuristic",
        "ts": ts,
        "session": "s1",
        "acct": 2,
    }


@pytest.fixture
def fixture_ledger(monkeypatch):
    now = time.time()
    home = _home()
    _write_jsonl(
        home / "sessions" / "s1.requests.jsonl",
        [
            _req(now - 60),
            _req(now - DAY - 60),
            _req(now - 60, booked=False),  # a failed request is not billed
            _req(now - 60, model="gpt-9"),  # unpriced: counted, never billed at Claude rates
            _req(now - 30 * DAY),  # outside the 7-day window
        ],
    )
    _write_jsonl(
        home / "savings.jsonl",
        [
            _saved_row(now - 60, 2_000_000, 1_000_000),
            _saved_row(now - DAY - 60, 1_500_000, 1_000_000),
            _saved_row(now - 30 * DAY, 9_000_000, 0),
        ],
    )
    actions = [
        discover.Action(f"a{i}", "savings", f"finding {i}", 100 - i, None, "b", f"distil fix-{i}")
        for i in range(4)
    ]
    monkeypatch.setattr(discover, "scan", lambda **k: discover.Report(sessions=1, actions=actions))
    from distil import proof_ledger

    monkeypatch.setattr(
        proof_ledger, "_safe_proof_lines", lambda led=None: [("budget", "budget: intact")]
    )
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    return now


# per request: 1M uncached x $5 + 10M reads x $0.50 + 1M 5m-writes x $6.25 + 0.1M out x $25
REQ_USD = 5.0 + 5.0 + 6.25 + 2.5
INPUT_RATE = (5.0 + 5.0 + 6.25) / 12_000_000  # blended $/input token actually paid


def test_request_cost_prices_cache_reads_and_writes():
    t = ss.Tokens(1_000_000, 10_000_000, 1_000_000, 100_000)
    assert ss.request_cost("claude-opus-4-8", t) == pytest.approx(REQ_USD)
    # the same write at a 1-hour TTL bills 2x input, not 1.25x
    assert ss.request_cost("claude-opus-4-8", t, write_1h=1_000_000) == pytest.approx(
        REQ_USD - 6.25 + 10.0
    )
    assert ss.request_cost("claude-opus-4-8-20260101", t) == pytest.approx(REQ_USD)
    assert ss.request_cost("gpt-9", t) is None


def test_savings_screen_json_on_fixture_ledger(fixture_ledger, capsys):
    assert cli.main(["savings", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d) >= {
        "mode",
        "since",
        "days",
        "spent_usd",
        "saved_usd",
        "saved_tokens",
        "net_pct",
        "requests",
        "unpriced_requests",
        "tokens",
        "daily",
        "findings",
        "proof",
        "notional",
        "calibration_factor",
        "blended_input_usd_per_mtok",
        "estimate",
    }
    assert d["mode"] == "ledger"
    assert d["requests"] == 3 and d["unpriced_requests"] == 1
    assert d["spent_usd"] == pytest.approx(2 * REQ_USD, abs=1e-4)
    assert d["tokens"] == {
        "uncached": 3_000_000,
        "cache_read": 30_000_000,
        "cache_write": 3_000_000,
        "output": 300_000,
    }
    assert d["saved_tokens"] == 1_500_000
    saved = 1_500_000 * INPUT_RATE
    assert d["saved_usd"] == pytest.approx(saved, abs=1e-4)
    assert d["net_pct"] == pytest.approx(saved / (2 * REQ_USD + saved) * 100, abs=0.01)
    assert d["blended_input_usd_per_mtok"] == pytest.approx(INPUT_RATE * 1e6, abs=1e-4)
    assert [f["command"] for f in d["findings"]] == ["distil fix-0", "distil fix-1", "distil fix-2"]
    assert d["proof"] == "budget: intact"
    assert d["estimate"] is None and d["notional"] is False
    # the daily series: one row per calendar day in the window, zeros included
    days = {r["date"]: r for r in d["daily"]}
    assert len(d["daily"]) in (7, 8)  # 7x24h spans 8 calendar days unless it starts at midnight
    today = ss._day(fixture_ledger - 60)
    yday = ss._day(fixture_ledger - DAY - 60)
    assert days[today]["spent_usd"] == pytest.approx(REQ_USD, abs=1e-4)
    assert days[yday]["saved_usd"] == pytest.approx(500_000 * INPUT_RATE, abs=1e-4)
    assert sum(r["spent_usd"] for r in d["daily"]) == pytest.approx(2 * REQ_USD, abs=1e-3)


def test_savings_screen_text_on_fixture_ledger(fixture_ledger, capsys):
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "distil savings  ·  last 7 days" in out
    assert f"spent    ${2 * REQ_USD:,.2f}" in out
    assert "saved    $2.03  (1,500,000 tokens)" in out
    assert "(1 requests on an unpriced model, not in $)" in out
    graph = [ln for ln in out.splitlines() if ln[2:4].isdigit() and ln[4:5] == "-"]
    assert len(graph) in (7, 8)
    assert any("█" in ln for ln in graph)
    assert "    1. finding 0" in out and "       → distil fix-2" in out
    assert "finding 3" not in out  # top three only
    assert "proof    budget: intact" in out


def test_savings_all_includes_old_history(fixture_ledger, capsys):
    assert cli.main(["savings", "--all", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["since"] is None and d["requests"] == 4
    assert d["saved_tokens"] == 1_500_000 + 9_000_000


def test_ledger_without_billed_usage_falls_back_to_list_price(monkeypatch, capsys):
    now = time.time()
    _write_jsonl(_home() / "savings.jsonl", [_saved_row(now - 60, 2_000_000, 1_000_000)])
    monkeypatch.setattr(discover, "scan", lambda **k: discover.Report())
    s = ss.build(now - 7 * DAY)
    assert s.mode == "ledger" and s.spent_usd == 0.0
    assert s.saved_usd == pytest.approx(5.0)  # the ledger's own list-rate price
    assert "an upper bound (no billed usage)" in ss.render(s)


def test_advisor_failure_never_costs_the_screen(fixture_ledger, monkeypatch):
    def boom(**k: object) -> None:
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(discover, "scan", boom)
    s = ss.build(fixture_ledger - 7 * DAY)
    assert s.findings == [] and s.spent_usd > 0


# --------------------------------------------------------------------------- savings: no-install

SECRET = "SECRET-TOOL-OUTPUT-4f1c"
PROMPT = "PRIVATE-PROMPT-9a2e"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _assistant(ts: float, mid: str, block: dict) -> dict:
    return {
        "type": "assistant",
        "timestamp": _iso(ts),
        "requestId": "req_" + mid,
        "uuid": "u-" + mid + str(block.get("type")),
        "message": {
            "id": mid,
            "model": "claude-sonnet-4-6",
            "content": [block],
            "usage": {
                "input_tokens": 1_000,
                "cache_read_input_tokens": 1_000_000,
                "cache_creation_input_tokens": 100_000,
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 60_000,
                    "ephemeral_5m_input_tokens": 40_000,
                },
                "output_tokens": 10_000,
            },
        },
    }


# sonnet $3/$15: 1k x $3 + 1M reads x $0.30 + 40k x $3.75 + 60k x $6 + 10k x $15 (per Mtok)
TRANSCRIPT_REQ_USD = 0.003 + 0.30 + 0.15 + 0.36 + 0.15


@pytest.fixture
def transcripts(monkeypatch):
    home = Path(os.environ["HOME"])
    proj = home / ".claude" / "projects" / "-tmp-proj"
    now = time.time()
    rows = [
        {
            "type": "user",
            "timestamp": _iso(now - 120),
            "uuid": "h1",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": PROMPT},
        },
        # one response written as two lines (one per content block), same usage
        _assistant(now - 100, "msg_1", {"type": "text", "text": "thinking about " + PROMPT}),
        _assistant(now - 100, "msg_1", {"type": "tool_use", "id": "t1", "name": "Read"}),
        {
            "type": "user",
            "timestamp": _iso(now - 90),
            "uuid": "tr1",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": SECRET * 500},
                ],
            },
        },
        _assistant(now - DAY - 80, "msg_2", {"type": "text", "text": "done"}),
        "not json at all",
    ]
    proj.mkdir(parents=True)
    (proj / "s1.jsonl").write_text(
        "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
        encoding="utf-8",
    )
    # a subagent transcript in a nested dir is still read
    sub = proj / "s1" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a.jsonl").write_text(
        json.dumps(_assistant(now - 50, "msg_3", {"type": "text", "text": "x"})) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    return home


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


def test_no_install_mode_reads_transcripts(transcripts, capsys):
    before = _tree(transcripts / ".claude")
    assert cli.main(["savings", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["mode"] == "transcripts"
    assert d["requests"] == 3  # msg_1 counted once despite two lines
    assert d["spent_usd"] == pytest.approx(3 * TRANSCRIPT_REQ_USD, abs=1e-4)
    assert d["tokens"]["cache_write"] == 300_000
    e = d["estimate"]
    assert e["label"] == "ESTIMATE" and e["source"] == ss.EST_SOURCE
    assert e["saved_usd"] == pytest.approx(3 * TRANSCRIPT_REQ_USD * ss.EST_SAVED_SHARE, abs=1e-4)
    assert 0 < d["tool_results_share"] <= 1
    assert d["net_pct"] is None  # nothing was saved; no ratio pretends otherwise
    assert _tree(transcripts / ".claude") == before  # read-only
    assert not any(_home().iterdir())  # and nothing written under DISTIL_HOME


def test_no_install_text_never_leaks_content(transcripts, capsys):
    assert cli.main(["savings"]) == 0
    assert cli.main(["savings", "--json"]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and PROMPT not in out and "thinking about" not in out
    assert "ESTIMATE  distil would save" in out
    assert "basis: realized saving in digest mode" in out
    assert "distil wrap -- claude" in out


def test_no_install_window_excludes_old_rows(transcripts, capsys):
    assert cli.main(["savings", "--since", "1d", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["requests"] == 2


def test_no_install_notional_on_subscription(transcripts, monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "API list price — you are on a flat-rate plan" in out
    assert "lossless-only" in out


def test_nothing_anywhere_says_how_to_start(capsys):
    assert cli.main(["savings"]) == 0
    out = capsys.readouterr().out
    assert "nothing to show yet" in out and "distil setup" in out
    assert cli.main(["savings", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "empty"


def test_estimate_constants_match_the_committed_artifacts():
    live = json.loads((REPO / ss.EST_SOURCE).read_text(encoding="utf-8"))
    assert ss.EST_SAVED_SHARE == live["realized"]["saved_share_of_counterfactual_bill"]
    assert ss.EST_REQUESTS == live["requests_priced"] == live["modes"]["digest"]
    tools = json.loads((REPO / ss.TOOL_DEFS_SOURCE).read_text(encoding="utf-8"))
    assert ss.TOOL_DEFS_SHARE == tools["tool_share_of_billed_usd"]


def test_pricing_1h_write_is_2x():
    p = pricing.get("claude-sonnet-4-6")
    assert p.cache_write_1h == pytest.approx(p.input * 2.0)


# --------------------------------------------------------------------------- setup / doctor


@pytest.fixture
def onboard_env(monkeypatch):
    from distil import doctor, onboard

    env = onboard.Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="9.9.9",
        method="pipx",
        managers=["pipx"],
    )
    monkeypatch.setattr(onboard, "detect", lambda: env)
    monkeypatch.setattr(onboard, "latest_pypi_version", lambda *a, **k: None)
    calls: dict[str, list] = {"default": [], "run": [], "diagnose": 0}

    def fake_default(a: argparse.Namespace) -> int:
        calls["default"].append(a.always_on)
        return 0

    def fake_diagnose() -> list:
        calls["diagnose"] += 1
        return [doctor.Check("ledger", doctor.OK, "fine")]

    import subprocess

    monkeypatch.setattr(cli, "cmd_default", fake_default)
    monkeypatch.setattr(doctor, "diagnose", fake_diagnose)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls["run"].append(a) or None)
    return calls


def test_setup_yes_composes_onboard_and_doctor(onboard_env, capsys):
    rc = cli.main(["setup", "--yes", "--offline", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "distil onboard" in out and "distil doctor" in out
    assert out.index("distil onboard") < out.index("distil doctor")
    assert onboard_env["diagnose"] == 1
    assert onboard_env["default"] == [False]  # the alias default; --yes never means always-on
    assert onboard_env["run"] == []  # never launches the agent from setup
    settings = Path(os.environ["HOME"]) / ".claude" / "settings.json"
    assert "statusLine" in json.loads(settings.read_text(encoding="utf-8"))
    assert "next:  distil wrap -- claude" in out


def test_setup_always_on_is_explicit(onboard_env, capsys):
    assert cli.main(["setup", "--yes", "--offline", "--always-on", "--no-color"]) == 0
    assert onboard_env["default"] == [False, True]
    capsys.readouterr()


def test_setup_settings_keeps_the_original_statusline_command(onboard_env, tmp_path, capsys):
    path = tmp_path / "s.json"
    assert cli.main(["setup", "--settings", str(path)]) == 0
    out = capsys.readouterr().out
    assert "statusLine" in json.loads(path.read_text(encoding="utf-8"))
    assert "distil onboard" not in out and onboard_env["diagnose"] == 0


def test_doctor_deep_runs_both_gates(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cmd_doctor", lambda a: 0)
    monkeypatch.setattr(cli, "cmd_validate", lambda a: 1)
    monkeypatch.setattr(cli, "cmd_verify", lambda a: 0)
    assert cli.main(["doctor", "--deep"]) == 1
    out = capsys.readouterr().out
    assert "deep: distil validate" in out and "deep: distil verify" in out
    assert cli.main(["doctor", "--deep", "--json"]) == 2
    monkeypatch.setattr(cli, "cmd_doctor", lambda a: 0)
    assert cli.main(["doctor"]) == 0

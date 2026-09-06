"""Census guarantees — the promises TELEMETRY.md makes, executable.

Every test isolates state via DISTIL_HOME=tmp_path and replaces the socket
layer with a tripwire: if urlopen is reached when the rules say it must not
be, the test fails loudly.
"""

from __future__ import annotations

import json

import pytest

from distil import census


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("DISTIL_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DISTIL_CENSUS_ENDPOINT", raising=False)


def _arm_network_tripwire(monkeypatch, calls: list):
    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode()))

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(census.urllib.request, "urlopen", fake_urlopen)


def test_default_is_silent(monkeypatch):
    """No consent → no network, even with an endpoint configured."""
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    monkeypatch.setenv("DISTIL_CENSUS_ENDPOINT", "http://127.0.0.1:1/ping")
    assert census.enabled() is False
    assert census.maybe_ping() is False
    assert calls == []


def test_do_not_track_beats_stored_consent(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert census.enabled() is False
    assert census.maybe_ping() is False
    assert calls == []


def test_distil_no_telemetry_beats_stored_consent(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    monkeypatch.setenv("DISTIL_NO_TELEMETRY", "1")
    assert census.maybe_ping() is False
    assert calls == []


def test_payload_schema_frozen():
    """The census may contain EXACTLY these keys — widening it must edit this
    test and TELEMETRY.md together (that's the point)."""
    census.opt_in()
    payload = census.build_payload()
    assert set(payload) == {
        "schema",
        "install_id",
        "version",
        "os",
        "arch",
        "python",
        "runs",
        "tokens_saved",
        "dollars_saved",
        "billing",
        "by_model",
        "agents",
        "surfaces",
        "shapes",
        "equivalence",
        "modes",
        "ts",
    }
    assert payload["schema"] == 4
    assert set(payload["surfaces"]) <= {"wrap", "proxy", "gateway"}
    assert set(payload["shapes"]) <= {"anthropic", "openai-chat", "openai-responses", "gemini"}
    assert set(payload["equivalence"]) == {"pct", "shadowed"}
    assert set(payload["modes"]) <= {"interactive", "headless", "sdk"}
    assert payload["billing"] in ("subscription", "metered")
    assert isinstance(payload["by_model"], dict) and len(payload["by_model"]) <= 5
    assert all(a in ("claude", "codex", "gemini", "aider", "other") for a in payload["agents"])
    # Numbers and short platform strings only — nothing that can carry content.
    for key, value in payload.items():
        assert isinstance(value, (int, float, str, dict, list)), key
        if isinstance(value, str):
            assert len(value) < 128, key
            assert "/" not in value or key == "version", key  # no paths


def test_opt_in_sends_and_throttles(monkeypatch):
    calls: list = []
    _arm_network_tripwire(monkeypatch, calls)
    census.opt_in()
    assert census.maybe_ping() is True
    assert len(calls) == 1
    assert calls[0]["schema"] == 4
    # Second call inside 24h: throttled, no second request.
    assert census.maybe_ping() is False
    assert len(calls) == 1


def test_opt_out_deletes_install_id():
    iid = census.opt_in()
    assert census.status()["install_id"] == iid
    census.opt_out()
    st = census.status()
    assert st["install_id"] is None
    assert st["consent"] == "off"
    assert census.enabled() is False


def test_send_failure_never_raises(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("endpoint down")

    monkeypatch.setattr(census.urllib.request, "urlopen", boom)
    census.opt_in()
    assert census.maybe_ping() is True  # attempted, swallowed


def test_install_id_is_stable_and_random():
    a = census.opt_in()
    b = census.install_id()
    assert a == b and len(a) == 32
    census.opt_out()
    c = census.opt_in()
    assert c != a  # re-consent mints a fresh identity


def test_subscription_reports_dollars_and_billing(monkeypatch):
    """Subscription installs are NOT excluded: dollars are reported and the
    billing field lets the rollup bucket them as notional (never real $)."""
    census.opt_in()
    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: True)
    p = census.build_payload()
    assert p["billing"] == "subscription"
    assert p["dollars_saved"] >= 0.0  # present, bucketed server-side


def test_calibration_factor_applied(monkeypatch, tmp_path):
    """Census totals wear the same heuristic→billed correction as the proof ledger."""
    census.opt_in()
    from distil.ledger import LedgerSummary

    fake = LedgerSummary(
        runs=1,
        total_dollars_saved=10.0,
        total_tokens_saved=1000,
        by_trajectory={},
        total_baseline_tokens=2000,
        total_distil_tokens=1000,
        total_baseline_dollars=20.0,
        total_distil_dollars=10.0,
    )
    monkeypatch.setattr("distil.ledger.summary", lambda: fake)
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (0.8, 99))
    monkeypatch.setattr("distil.doctor.subscription_mode", lambda: False)
    p = census.build_payload()
    assert p["tokens_saved"] == 800  # 1000 * 0.8 — never more than billed
    assert p["dollars_saved"] == 8.0


def _fake_summary(baseline_tokens, baseline_dollars=0.0):
    from distil.ledger import LedgerSummary

    return LedgerSummary(
        runs=1,
        total_dollars_saved=0.0,
        total_tokens_saved=0,
        by_trajectory={},
        total_baseline_tokens=baseline_tokens,
        total_distil_tokens=0,
        total_baseline_dollars=baseline_dollars,
        total_distil_dollars=0.0,
    )


def test_tokens_saved_never_shrinks_on_downward_recalibration(monkeypatch):
    """THE bug the user reported: multiplying the whole lifetime cumulative by a
    drifting factor made the community counter go DOWN. Freezing each delta at
    count time means a later downward factor cannot un-count banked savings."""
    census.opt_in()
    monkeypatch.setattr("distil.ledger.summary", lambda: _fake_summary(1000))
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (1.0, 99))
    first = census.build_payload(accrue=True)["tokens_saved"]  # bank 1000 × 1.0
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (0.5, 99))
    second = census.build_payload(accrue=True)["tokens_saved"]  # raw unchanged
    assert first == 1000
    assert second >= first  # never shrinks…
    assert second == 1000  # …and no new raw ⇒ holds, does NOT restate to 500


def test_new_savings_accrue_at_the_factor_known_when_earned(monkeypatch):
    """Each period's delta wears the factor known at that time and is frozen —
    monotonic AND never more than billed."""
    census.opt_in()
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (1.0, 99))
    monkeypatch.setattr("distil.ledger.summary", lambda: _fake_summary(1000))
    census.build_payload(accrue=True)  # bank 1000 × 1.0 = 1000
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (0.5, 99))
    monkeypatch.setattr("distil.ledger.summary", lambda: _fake_summary(1500))
    assert census.build_payload(accrue=True)["tokens_saved"] == 1250  # 1000 + 500×0.5


def test_preview_does_not_advance_the_total(monkeypatch):
    """`census show` / direct build_payload() must never mutate accrual state."""
    census.opt_in()
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (1.0, 99))
    monkeypatch.setattr("distil.ledger.summary", lambda: _fake_summary(1000))
    census.build_payload()  # preview
    census.build_payload()  # preview again
    assert census.build_payload(accrue=True)["tokens_saved"] == 1000  # deltas not eaten


def test_live_heartbeat_total_is_monotonic_and_shared_with_census(monkeypatch):
    """The live counter (`_current_saved_tokens`, the heartbeat's figure) is the
    SAME monotonic accrued total the census reports — a downward factor drift
    can never make the live number the user watches shrink."""
    census.opt_in()
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (1.0, 99))
    monkeypatch.setattr("distil.ledger.summary", lambda: _fake_summary(1000))
    assert census._current_saved_tokens() == 1000  # banks 1000 × 1.0
    # Factor refines down; the live number holds, it does not un-count.
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (0.4, 99))
    assert census._current_saved_tokens() == 1000  # NOT 400
    # And the daily census reads the very same shared total.
    assert census.build_payload()["tokens_saved"] == 1000


def test_second_writer_reads_inside_the_lock(monkeypatch):
    """distil runs as several processes at once (wrap, proxy worker, gateway,
    webdash) and two of them advance this counter: the daily census and the
    heartbeat. Each used to do load -> step -> write-the-whole-file with no
    lock, so a writer holding minutes-old state wrote a SMALLER `saved` back
    and rewound `raw_seen` with it — which is how a total that can only rise
    published 1.44B, then 1.33B, then 1.16B, then 1.48B.

    The guarantee is that the second writer reads *inside* the lock, so it
    steps from the first writer's committed total, never from stale state.
    """
    import threading

    census.opt_in()
    a_inside, a_may_commit = threading.Event(), threading.Event()
    seen = {}

    def writer_a():
        with census._savings_locked(True) as st:
            census._step(st["tokens"], 5000, 1.0)
            a_inside.set()
            a_may_commit.wait(5)  # hold the lock, uncommitted

    def writer_b():
        a_inside.wait(5)  # B starts while A is still mid-write
        with census._savings_locked(True) as st:
            seen["saved"] = st["tokens"]["saved"]

    ta, tb = threading.Thread(target=writer_a), threading.Thread(target=writer_b)
    ta.start(), tb.start()
    assert a_inside.wait(5)
    a_may_commit.set()
    ta.join(5), tb.join(5)

    # Unlocked, B would have read 0 (A had not written yet) and clobbered A's
    # 5000 on the way out. Serialized, B sees the committed total.
    assert seen["saved"] == 5000
    assert census._load_savings()["tokens"]["saved"] == 5000


def test_concurrent_writers_never_lower_the_total(monkeypatch):
    """The invariant the adoption page publishes: the shared total is
    monotonic under concurrency, not just per-process."""
    import threading

    census.opt_in()
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (1.0, 99))
    readings, lock = [], threading.Lock()

    def bump(raw):
        monkeypatch.setattr("distil.ledger.summary", lambda raw=raw: _fake_summary(raw))
        val = census._current_saved_tokens()
        with lock:
            readings.append(val)

    threads = [threading.Thread(target=bump, args=(n,)) for n in (100, 900, 300, 1500, 700)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    on_disk = census._load_savings()["tokens"]["saved"]
    assert on_disk >= max(readings), "a concurrent writer rewound the shared total"
    assert on_disk > 0


class _Sum:
    def __init__(self, tokens):
        self.total_tokens_saved = tokens


@pytest.fixture
def tty(monkeypatch):
    """Claim stdout is a terminal. The invite is TTY-only so it never lands in a
    pipe or a CI log, and pytest's captured stdout is not a TTY."""
    from distil import cli

    monkeypatch.setattr(cli, "_is_tty", lambda: True)


def test_invite_offers_the_census_once_to_someone_with_savings(tty, capsys):
    """Consent was only ever offered inside `distil onboard`, in a TTY — anyone
    who went straight from `uvx`/`pipx` to `distil wrap` was never asked."""
    from distil import cli

    assert census.consent() is None
    cli._census_invite(_Sum(5_000_000))
    assert "distil census on" in capsys.readouterr().out
    # Shown once. A second `distil stats` must be silent.
    cli._census_invite(_Sum(5_000_000))
    assert capsys.readouterr().out == ""


def test_invite_never_grants_consent(tty, capsys):
    """Seeing the invite is not answering it: the user stays un-asked and can
    still opt in (or never) — the invite must not become a silent opt-in."""
    from distil import cli

    cli._census_invite(_Sum(5_000_000))
    assert "distil census on" in capsys.readouterr().out
    assert census.consent() is None
    assert not census.enabled()


def test_invite_is_silent_when_already_answered_or_disabled(tty, monkeypatch, capsys):
    from distil import cli

    census.opt_out()  # already answered
    cli._census_invite(_Sum(5_000_000))
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("DO_NOT_TRACK", "1")
    (census._home() / "census").unlink()  # back to un-asked
    cli._census_invite(_Sum(5_000_000))
    assert capsys.readouterr().out == "", "DO_NOT_TRACK must silence the invite too"


def test_invite_is_silent_below_the_savings_threshold(tty, capsys):
    from distil import cli

    cli._census_invite(_Sum(cli.INVITE_MIN_TOKENS - 1))
    assert capsys.readouterr().out == ""


def test_stats_actually_shows_the_invite(tty, monkeypatch, capsys):
    """Wiring test: the invite must reach the user through `distil stats`, not
    just exist as a function nobody calls."""
    import argparse

    from distil import cli, ledger

    summary = ledger.LedgerSummary(
        runs=3,
        total_dollars_saved=1.5,
        total_tokens_saved=5_000_000,
        by_trajectory={"live-proxy": 1.5},
        total_baseline_tokens=10_000_000,
        total_distil_tokens=5_000_000,
        total_baseline_dollars=3.0,
        total_distil_dollars=1.5,
    )
    monkeypatch.setattr(ledger, "summary", lambda *a, **k: summary)
    args = argparse.Namespace(badge=False, html=None, json=False)
    assert cli.cmd_leaderboard(args) == 0
    assert "distil census on" in capsys.readouterr().out


def test_invite_is_silent_without_a_tty(capsys):
    """Piped into a file or a CI log, `distil stats` stays clean."""
    from distil import cli

    cli._census_invite(_Sum(5_000_000))
    assert capsys.readouterr().out == ""


def _seed_ledger(tmp_path, rows):
    """Write a savings.jsonl the census helpers read via ledger.default_path()."""
    import json as _json

    (tmp_path / "savings.jsonl").write_text(
        "".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _seed_sessions(tmp_path, argvs):
    import json as _json

    sd = tmp_path / "sessions"
    sd.mkdir(parents=True, exist_ok=True)
    for i, argv in enumerate(argvs):
        (sd / f"s{i}.json").write_text(_json.dumps({"argv": argv}), encoding="utf-8")


def test_by_model_top5_and_calibrated(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr("distil.calibration.factor", lambda model=None, path=None: (2.0, 99))
    census.opt_in()
    _seed_ledger(
        tmp_path,
        [
            {"model": f"m{i}", "baseline_input_tokens": (i + 1) * 100, "distil_input_tokens": 0}
            for i in range(7)
        ],
    )
    bm = census.build_payload()["by_model"]
    assert len(bm) == 5  # top-5 only
    assert bm["m6"] == 1400  # 700 * factor 2.0 — calibrated
    assert "m0" not in bm and "m1" not in bm  # smallest two dropped


def test_agents_and_modes_from_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    _seed_sessions(
        tmp_path,
        [
            ["claude"],  # interactive
            ["claude", "-p", "hi"],  # headless
            ["/usr/bin/python", "agent.py"],  # sdk
            ["weird-tool", "--x"],  # non-agent → agents:other, mode sdk
        ],
    )
    p = census.build_payload()
    assert set(p["agents"]) == {"claude", "other"}
    assert p["modes"]["interactive"] == 1
    assert p["modes"]["headless"] == 1
    assert p["modes"]["sdk"] == 2  # python + weird-tool


def test_load_savings_resets_only_a_corrupt_channel(monkeypatch, tmp_path):
    """A partially-corrupt accrual file (one channel is not a dict) is repaired
    in place — the bad channel resets to fresh, the intact one is preserved —
    rather than throwing away all banked savings."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    census._savings_path().write_text(
        json.dumps(
            {
                "v": 1,
                "by_model": {},
                "tokens": "corrupt",  # not a dict
                "dollars": {"saved": 5.0, "raw_seen": 10},  # intact
            }
        ),
        encoding="utf-8",
    )
    st = census._load_savings()
    assert st["tokens"] == {"saved": 0.0, "raw_seen": 0}  # reset to fresh
    assert st["dollars"] == {"saved": 5.0, "raw_seen": 10}  # preserved


def test_save_savings_is_fail_open(monkeypatch, tmp_path):
    """A failed write to census-savings.json must never propagate — the census
    rides the host's exit path and can't be allowed to break it. The delta just
    re-accrues on the next send."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")  # a FILE where a dir is needed
    monkeypatch.setattr(census, "_savings_path", lambda: blocker / "census-savings.json")
    census._save_savings({"v": 1, "tokens": {}, "dollars": {}, "by_model": {}})  # must not raise
    assert not (blocker / "census-savings.json").exists()


def _paired_ledger(tmp_path, n, *, ab_changes=0, aa_changes=0):
    """A real ledger of v5 paired rows — one request, both arms."""
    from distil.shadow import ShadowLedger

    led = ShadowLedger()
    for i in range(n):
        led.record(
            i >= ab_changes,
            kind="paired",
            evidence={"aa_equal": i >= aa_changes, "mode": "digest"},
            path=tmp_path / "shadow.jsonl",
        )
    return led


def test_equivalence_from_shadow(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()

    led = _paired_ledger(tmp_path, 500)
    monkeypatch.setattr("distil.shadow.ShadowLedger.load", classmethod(lambda cls, **k: led))
    assert census.build_payload()["equivalence"] == {"pct": 100.0, "shadowed": 500}


def test_equivalence_null_below_the_reporting_floor(tmp_path, monkeypatch):
    """The feed used to publish a pct as soon as ANY A/A baseline existed (n>=10),
    which is how the adoption ring came to be fed by a sample the status line
    itself refused to render. One floor now, and below it the pct is null."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()

    led = _paired_ledger(tmp_path, 4)
    monkeypatch.setattr("distil.shadow.ShadowLedger.load", classmethod(lambda cls, **k: led))
    eq = census.build_payload()["equivalence"]
    assert eq == {"pct": None, "shadowed": 4}  # count sent, no fabricated pct


def test_equivalence_pct_is_capped_for_the_wire(tmp_path, monkeypatch):
    """The paired estimate can exceed 100% (compression agreed more often than the
    model agreed with itself). The deployed worker rejects pct > 100, so the wire
    value is capped — the uncapped number lives in `distil shadow-stats`."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()

    led = _paired_ledger(tmp_path, 60, ab_changes=0, aa_changes=10)
    monkeypatch.setattr("distil.shadow.ShadowLedger.load", classmethod(lambda cls, **k: led))
    assert led.equivalence().pct > 100.0
    assert census.build_payload()["equivalence"] == {"pct": 100.0, "shadowed": 60}


def test_helpers_fail_open_on_bad_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    # corrupt + argv-less session manifests exercise the skip branches
    sd = tmp_path / "sessions"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "bad.json").write_text("{not json", encoding="utf-8")
    (sd / "empty.json").write_text("{}", encoding="utf-8")  # no argv
    # calibration blowing up must not break the payload (identity factor)
    monkeypatch.setattr(
        "distil.calibration.factor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        "distil.doctor.subscription_mode", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    p = census.build_payload()
    assert p["agents"] == [] and p["modes"] == {}  # bad manifests skipped
    assert p["billing"] == "metered"  # _billing except → default


def test_by_model_survives_corrupt_ledger_line(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    (tmp_path / "savings.jsonl").write_text(
        '{"model":"m","baseline_input_tokens":100,"distil_input_tokens":0}\n{bad\n',
        encoding="utf-8",
    )
    assert census.build_payload()["by_model"].get("m") == 100


def test_cmd_census_handler(tmp_path, monkeypatch, capsys):
    """The `distil census on|off|status|show` CLI handler end to end."""
    import argparse
    from distil.cli import cmd_census

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr("distil.census.maybe_ping", lambda: False)  # no network in test

    assert cmd_census(argparse.Namespace(action="status")) == 0
    assert '"consent": null' in capsys.readouterr().out

    assert cmd_census(argparse.Namespace(action="on")) == 0
    assert "census: ON" in capsys.readouterr().out

    assert cmd_census(argparse.Namespace(action="show")) == 0
    out = capsys.readouterr()
    assert '"schema": 4' in out.out and "preview only" in out.err

    assert cmd_census(argparse.Namespace(action="off")) == 0
    assert "census: OFF" in capsys.readouterr().out


def test_cmd_census_on_notes_do_not_track(tmp_path, monkeypatch, capsys):
    import argparse
    from distil.cli import cmd_census

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert cmd_census(argparse.Namespace(action="on")) == 0
    assert "nothing will send" in capsys.readouterr().out


def _arm_beat_tripwire(monkeypatch, calls):
    monkeypatch.setattr(census, "_send_beat", lambda p: calls.append(p))


def test_heartbeat_sends_only_when_tokens_grew(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    calls = []
    _arm_beat_tripwire(monkeypatch, calls)
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 5000)
    # prior beat 10 min ago at 1000 tokens → grew → sends with a real rate
    (tmp_path / "heartbeat-last").write_text(
        json.dumps({"tokens": 1000, "ts": census.time.time() - 600})
    )
    assert census.maybe_heartbeat() is True
    assert len(calls) == 1
    p = calls[0]
    assert set(p) == {"v", "id", "tokens", "rate", "ts"}  # content-free
    assert p["tokens"] == 5000 and p["rate"] > 0


def test_heartbeat_liveness_when_flat(tmp_path, monkeypatch):
    # An install that ran but didn't grow its saved-token total still beats
    # (liveness) so `active` counts it — rate is 0, so no phantom projection.
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    calls = []
    _arm_beat_tripwire(monkeypatch, calls)
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 1000)
    (tmp_path / "heartbeat-last").write_text(
        json.dumps({"tokens": 1000, "ts": census.time.time() - 600})
    )
    assert census.maybe_heartbeat() is True  # liveness beat sent
    assert len(calls) == 1 and calls[0]["tokens"] == 1000 and calls[0]["rate"] == 0


def test_heartbeat_liveness_when_recalibrated_down(tmp_path, monkeypatch):
    # A downward calibration refinement (more accurate estimate) must not read as
    # inactive — still beats, rate clamped to 0 (never negative, never projects).
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    calls = []
    _arm_beat_tripwire(monkeypatch, calls)
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 900)
    (tmp_path / "heartbeat-last").write_text(
        json.dumps({"tokens": 1000, "ts": census.time.time() - 600})
    )
    assert census.maybe_heartbeat() is True
    assert len(calls) == 1 and calls[0]["tokens"] == 900 and calls[0]["rate"] == 0


def test_heartbeat_silent_when_nothing_saved(tmp_path, monkeypatch):
    # A brand-new install that has saved nothing yet is not an active saver.
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    calls = []
    _arm_beat_tripwire(monkeypatch, calls)
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 0)
    (tmp_path / "heartbeat-last").write_text(
        json.dumps({"tokens": 0, "ts": census.time.time() - 600})
    )
    assert census.maybe_heartbeat() is False  # nothing saved → nothing sent
    assert calls == []


def test_heartbeat_throttled(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    calls = []
    _arm_beat_tripwire(monkeypatch, calls)
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 9999)
    (tmp_path / "heartbeat-last").write_text(
        json.dumps({"tokens": 1, "ts": census.time.time() - 10})  # 10s ago < 5min
    )
    assert census.maybe_heartbeat() is False
    assert calls == []


def test_heartbeat_gated_by_consent_and_dnt(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    calls = []
    _arm_beat_tripwire(monkeypatch, calls)
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 5000)
    assert census.maybe_heartbeat() is False  # not opted in
    census.opt_in()
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert census.maybe_heartbeat() is False  # DNT wins
    assert calls == []


def test_heartbeat_send_failure_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 9999)
    (tmp_path / "heartbeat-last").write_text(
        json.dumps({"tokens": 1, "ts": census.time.time() - 600})
    )

    def boom(p):
        raise OSError("beat endpoint down")

    monkeypatch.setattr(census, "_send_beat", boom)
    assert census.maybe_heartbeat() is True  # attempted, error swallowed


def test_current_saved_tokens_survives_calibration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    (tmp_path / "savings.jsonl").write_text(
        json.dumps(
            {
                "trajectory_id": "t",
                "model": "m",
                "turns": 1,
                "baseline_input_tokens": 1000,
                "distil_input_tokens": 400,
                "baseline_dollars": 1.0,
                "distil_dollars": 0.4,
                "tokenizer": "heuristic",
                "ts": 1.0,
                "acct": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "distil.calibration.factor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    assert census._current_saved_tokens() == 600  # identity factor fallback


def test_send_beat_posts_to_endpoint(monkeypatch):
    """_send_beat POSTs the payload to the beat endpoint (urlopen exercised)."""
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(census.urllib.request, "urlopen", fake_urlopen)
    census._send_beat({"v": 1, "id": "a" * 32, "tokens": 5, "rate": 1.0, "ts": 1})
    assert seen["url"].endswith("/v1/beat") and seen["body"]["tokens"] == 5


def test_heartbeat_corrupt_marker_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    census.opt_in()
    (tmp_path / "heartbeat-last").write_text("{not json")  # corrupt → treated as first beat
    monkeypatch.setattr(census, "_current_saved_tokens", lambda: 100)
    calls = []
    monkeypatch.setattr(census, "_send_beat", lambda p: calls.append(p))
    # first beat with no prior baseline: last_ts=0 so throttle passes; grew from 0
    assert census.maybe_heartbeat() is True
    assert calls and calls[0]["tokens"] == 100


# ---------------------------------------------------------------------------
# Reported version — the "versions in the wild" histogram depends on this
# ---------------------------------------------------------------------------


def test_reported_version_reads_disk_not_the_running_interpreter(monkeypatch):
    """A long-lived `distil wrap` parent keeps running the code it started with
    (hot-swap replaces the worker, not the parent) and is the process that emits
    the census on exit. Reporting the imported __version__ there meant every
    wrap user's beat was pinned to whenever their session began — observed live
    as a 1.34.0 machine reporting 1.28.0 two upgrades later.
    """
    import distil
    import distil.hotswap as hotswap

    monkeypatch.setattr(hotswap, "installed_version", lambda: "9.9.9")
    monkeypatch.setattr(distil, "__version__", "1.28.0")  # the stale in-memory one
    assert census._reported_version() == "9.9.9"
    assert census.build_payload()["version"] == "9.9.9"


def test_reported_version_falls_back_when_metadata_is_absent(monkeypatch):
    """Source checkout / zipapp / vendored copy: no dist-info to discover. The
    stale-but-real version beats reporting nothing."""
    import distil
    import distil.hotswap as hotswap

    monkeypatch.setattr(distil, "__version__", "1.28.0")
    monkeypatch.setattr(hotswap, "installed_version", lambda: None)
    assert census._reported_version() == "1.28.0"

    def _boom():
        raise RuntimeError("metadata backend exploded")

    monkeypatch.setattr(hotswap, "installed_version", _boom)
    assert census._reported_version() == "1.28.0", "a probe failure must not propagate"

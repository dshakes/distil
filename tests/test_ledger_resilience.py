"""Ledger readers must survive a corrupt/partial line, and writers must lock
and back up. Round-2 P0-2 / P1-4 / P1-5."""

from __future__ import annotations

from distil import ledger


def _good(path, session="s1", tid="t1"):
    ledger.record(
        trajectory_id=tid,
        model="m",
        turns=1,
        baseline_dollars=1.0,
        distil_dollars=0.4,
        baseline_input_tokens=100,
        distil_input_tokens=40,
        session=session,
        path=path,
    )


def test_summary_skips_corrupt_lines(tmp_path):
    p = tmp_path / "savings.jsonl"
    _good(p)
    # A truncated JSON line, a valid-JSON-but-missing-keys line, and a non-object line.
    with p.open("a") as f:
        f.write('{"trajectory_id": "t", "baseline_dollars":\n')  # partial
        f.write('{"session": "x"}\n')  # missing numeric keys
        f.write("42\n")  # valid JSON, not a dict
    _good(p, tid="t2")

    s = ledger.summary(path=p)
    assert s.runs == 2  # only the two well-formed records counted
    assert s.corrupt_lines == 3
    assert s.total_dollars_saved > 0


def test_latest_session_survives_corruption(tmp_path):
    p = tmp_path / "savings.jsonl"
    with p.open("a") as f:
        f.write("not json at all\n")
    _good(p, session="live")
    sid, ts = ledger.latest_session(path=p)
    assert sid == "live"
    assert ts > 0


def test_mixed_era_accounting(tmp_path):
    p = tmp_path / "savings.jsonl"
    _good(p, tid="new")  # written with acct=2
    # Two legacy rows: one with no acct field, one with acct=1.
    with p.open("a") as f:
        f.write(
            '{"trajectory_id":"old1","model":"m","turns":1,"baseline_dollars":1.0,'
            '"distil_dollars":0.5,"baseline_input_tokens":100,"distil_input_tokens":50,'
            '"tokenizer":"heuristic","ts":1.0,"session":""}\n'
        )
        f.write(
            '{"trajectory_id":"old2","model":"m","turns":1,"baseline_dollars":1.0,'
            '"distil_dollars":0.5,"baseline_input_tokens":100,"distil_input_tokens":50,'
            '"tokenizer":"heuristic","ts":1.0,"session":"","acct":1}\n'
        )
    s = ledger.summary(path=p)
    assert s.runs == 3  # legacy rows still counted
    assert s.legacy_records == 2
    assert "pre-1.10 accounting" in ledger.render_dashboard(s, color=False)
    assert "pre-1.10 accounting" in ledger.render_html(s)


def test_no_footnote_when_all_current(tmp_path):
    p = tmp_path / "savings.jsonl"
    _good(p, tid="a")
    _good(p, tid="b")
    s = ledger.summary(path=p)
    assert s.legacy_records == 0
    assert "pre-1.10 accounting" not in ledger.render_dashboard(s, color=False)
    assert "pre-1.10 accounting" not in ledger.render_html(s)


def test_backup_created_past_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_BACKUP_GROWTH_BYTES", 200)
    p = tmp_path / "savings.jsonl"
    bak = tmp_path / "savings.jsonl.bak"
    for i in range(20):
        _good(p, tid=f"t{i}")
    assert bak.exists()
    # Backup is a prefix snapshot of the ledger, never longer than the live file.
    assert bak.stat().st_size <= p.stat().st_size


def test_latest_mode_reads_newest_skips_corruption_and_filters(tmp_path):
    """latest_mode powers the status line's mode chip — it must return the newest
    row's mode, honor a session filter, skip a corrupt line, and never crash."""
    import json

    p = tmp_path / "savings.jsonl"
    p.write_text(
        json.dumps({"session": "s1", "mode": "digest", "ts": 1}) + "\n"
        "{ corrupt not json\n"
        + json.dumps({"session": "s2", "mode": "lossless-only", "ts": 2})
        + "\n"
    )
    assert ledger.latest_mode(path=p) == "lossless-only"  # newest overall
    assert ledger.latest_mode(session="s1", path=p) == "digest"  # session-scoped, skips corrupt
    assert ledger.latest_mode(session="nope", path=p) == ""  # no match
    assert ledger.latest_mode(path=tmp_path / "absent.jsonl") == ""  # missing file never crashes


def test_mode_rates_aggregates_per_mode_and_survives_a_bad_ledger(tmp_path):
    """mode_rates answers "what has the other mode been worth on THIS machine",
    so it is quoted as a headline percentage — it must not be skewed by rows it
    cannot price, and must never crash a report on a partial write."""
    import json

    def row(mode, base, dist):
        return json.dumps(
            {"mode": mode, "baseline_input_tokens": base, "distil_input_tokens": dist}
        )

    p = tmp_path / "savings.jsonl"
    p.write_text(
        row("digest", 1000, 400)
        + "\n"
        + row("digest", 1000, 600)
        + "\n"
        + "{ truncated mid-write\n"  # a crash during append must not break the report
        + row("lossless-only", 1000, 997)
        + "\n"
        + row("digest", 0, 0)  # unpriceable: no baseline, must not count as 0% saved
        + "\n"
        + json.dumps({"baseline_input_tokens": 1000, "distil_input_tokens": 1})  # no mode
        + "\n"
    )
    rates = ledger.mode_rates(path=p)
    assert rates["digest"] == (2, 0.5), "the unpriceable row must not dilute the rate"
    assert rates["lossless-only"][0] == 1
    assert round(rates["lossless-only"][1], 3) == 0.003
    assert "" not in rates, "a row with no mode belongs to no mode"
    assert ledger.mode_rates(path=tmp_path / "absent.jsonl") == {}


def test_mode_rates_window_excludes_stale_history(tmp_path):
    """`since` is what keeps the quoted rate honest. This ledger carried digest at
    51.7% lifetime while digest over the last 7 days was 6.2% — the mix that digest
    is worth on drifts, so a lifetime rate printed next to a recent trim overstated
    the user's actual rate by ~8x and told a user to enable what they already ran."""
    import json
    import time

    now = time.time()

    def row(mode, base, dist, ts):
        return json.dumps(
            {
                "mode": mode,
                "baseline_input_tokens": base,
                "distil_input_tokens": dist,
                "ts": ts,
            }
        )

    p = tmp_path / "savings.jsonl"
    p.write_text(
        row("digest", 1000, 100, now - 60 * 86400)  # ancient: 90% saved
        + "\n"
        + row("digest", 1000, 940, now - 3600)  # recent: 6% saved
        + "\n"
        + row("digest", 1000, 500, 0)  # no usable ts → excluded by any window
        + "\n"
    )
    assert round(ledger.mode_rates(path=p)["digest"][1], 2) == 0.49, "lifetime spans all"
    windowed = ledger.mode_rates(path=p, since=now - 7 * 86400)
    assert windowed["digest"][0] == 1, "only the in-window row counts"
    assert round(windowed["digest"][1], 2) == 0.06, "the recent rate, not the flattering one"

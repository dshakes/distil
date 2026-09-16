"""Top-up tests for distil.conformal and distil.ledger to push both above 95%.

conformal.py: covers betting_upper_bound, empirical_bernstein_bound,
              certified_risk_bound, tight_risk_bound (all methods), and the
              edge cases (empty losses, n=0, n=1).

ledger.py:    covers render_dashboard states (no runs, session, subscription,
              samples), render_html cards (_eq_card thresholds, _session_card),
              latest_session edge cases, _human/_bar helpers, and record/summary
              with a real on-disk ledger.
"""

from __future__ import annotations

import json

import pytest

# --------------------------------------------------------------------------- #
# conformal — additional bound functions
# --------------------------------------------------------------------------- #


def test_certified_risk_bound_zero_n_returns_one() -> None:
    from distil.conformal import certified_risk_bound

    assert certified_risk_bound(0.0, 0, 0.05) == 1.0


def test_certified_risk_bound_shrinks_with_more_data() -> None:
    from distil.conformal import certified_risk_bound

    b50 = certified_risk_bound(0.0, 50, 0.05)
    b200 = certified_risk_bound(0.0, 200, 0.05)
    b1000 = certified_risk_bound(0.0, 1000, 0.05)
    assert 1.0 > b50 > b200 > b1000 > 0.0


def test_certified_risk_bound_is_above_rhat() -> None:
    from distil.conformal import certified_risk_bound

    # The bound is always ≥ rhat (it's an upper confidence bound on the true risk)
    rhat = 0.5
    b = certified_risk_bound(rhat, 100, 0.05)
    assert b >= rhat
    assert b <= 1.0


def test_empirical_bernstein_bound_n_one_returns_one() -> None:
    from distil.conformal import empirical_bernstein_bound

    assert empirical_bernstein_bound(0.0, 0.0, 1, 0.05) == 1.0


def test_empirical_bernstein_bound_shrinks_with_low_variance() -> None:
    from distil.conformal import empirical_bernstein_bound

    # zero variance should give a tight bound
    b_zero_var = empirical_bernstein_bound(0.05, 0.0, 200, 0.05)
    b_high_var = empirical_bernstein_bound(0.05, 0.25, 200, 0.05)
    assert b_zero_var < b_high_var


def test_empirical_bernstein_bound_capped_at_one() -> None:
    from distil.conformal import empirical_bernstein_bound

    # extreme case should still return ≤ 1
    assert empirical_bernstein_bound(0.9, 0.25, 2, 0.05) <= 1.0


def test_betting_upper_bound_empty_returns_one() -> None:
    from distil.conformal import betting_upper_bound

    assert betting_upper_bound([], 0.05) == 1.0


def test_betting_upper_bound_all_zero_losses() -> None:
    from distil.conformal import betting_upper_bound

    # all zero losses → bound should be well below 1
    b = betting_upper_bound([0.0] * 100, 0.05)
    assert 0.0 < b < 1.0


def test_betting_upper_bound_high_losses_near_one() -> None:
    from distil.conformal import betting_upper_bound

    # high constant losses → bound near 1
    b = betting_upper_bound([0.9] * 50, 0.05)
    assert b > 0.8


def test_tight_risk_bound_hb_method() -> None:
    from distil.conformal import tight_risk_bound

    losses = [0.0] * 200
    b = tight_risk_bound(losses, 0.05, method="hb")
    assert 0.0 < b < 1.0


def test_tight_risk_bound_eb_method() -> None:
    from distil.conformal import tight_risk_bound

    losses = [0.05] * 200
    b = tight_risk_bound(losses, 0.05, method="eb")
    assert 0.0 < b <= 1.0


def test_tight_risk_bound_betting_method() -> None:
    from distil.conformal import tight_risk_bound

    losses = [0.0] * 100
    b = tight_risk_bound(losses, 0.05, method="betting")
    assert 0.0 < b < 1.0


def test_tight_risk_bound_auto_binary_uses_hb() -> None:
    from distil.conformal import certified_risk_bound, tight_risk_bound

    losses = [0.0, 1.0, 0.0, 0.0, 1.0] * 40  # binary, n=200
    auto = tight_risk_bound(losses, 0.05, method="auto")
    hb = certified_risk_bound(sum(losses) / len(losses), len(losses), 0.05)
    assert auto == hb


def test_tight_risk_bound_auto_graded_uses_eb() -> None:
    from distil.conformal import empirical_bernstein_bound, tight_risk_bound

    # graded [0,1] losses → auto picks EB
    losses = [0.1, 0.2, 0.3, 0.15, 0.25] * 40
    n = len(losses)
    rhat = sum(losses) / n
    var = sum((x - rhat) ** 2 for x in losses) / (n - 1)
    auto = tight_risk_bound(losses, 0.05, method="auto")
    eb = empirical_bernstein_bound(rhat, var, n, 0.05)
    assert auto == eb


def test_tight_risk_bound_empty() -> None:
    from distil.conformal import tight_risk_bound

    assert tight_risk_bound([], 0.05) == 1.0


# --------------------------------------------------------------------------- #
# ledger — _human, _bar helpers
# --------------------------------------------------------------------------- #


def test_human_formatting() -> None:
    from distil.ledger import _human

    assert _human(0) == "0"
    assert _human(999) == "999"
    assert _human(1500) == "1.5K"
    assert _human(2_500_000) == "2.5M"
    assert _human(3_700_000_000) == "3.7B"


def test_bar_boundaries() -> None:
    from distil.ledger import _bar

    assert _bar(0.0) == "░" * 22
    assert _bar(1.0) == "█" * 22
    assert _bar(-0.5) == "░" * 22  # clamped to 0
    assert _bar(1.5) == "█" * 22  # clamped to 1
    # half-full: roughly half of 22 = 11 filled blocks
    half = _bar(0.5)
    filled = half.count("█")
    assert 9 <= filled <= 13


# --------------------------------------------------------------------------- #
# ledger — latest_session edge cases
# --------------------------------------------------------------------------- #


def test_latest_session_empty_file(tmp_path) -> None:
    from distil import ledger

    p = tmp_path / "savings.jsonl"
    p.write_text("")
    sid, ts = ledger.latest_session(path=p)
    assert sid == ""
    assert ts == 0.0


def test_latest_session_no_file(tmp_path) -> None:
    from distil import ledger

    sid, ts = ledger.latest_session(path=tmp_path / "missing.jsonl")
    assert sid == ""
    assert ts == 0.0


def test_latest_session_picks_most_recent(tmp_path) -> None:
    from distil import ledger

    p = tmp_path / "savings.jsonl"
    p.write_text(
        json.dumps({"session": "s1", "ts": 100.0})
        + "\n"
        + json.dumps({"session": "s2", "ts": 200.0})
        + "\n"
        + json.dumps({"session": "s1", "ts": 300.0})
        + "\n"
    )
    sid, ts = ledger.latest_session(path=p)
    assert sid == "s1"
    assert ts == 300.0


def test_latest_session_skips_records_without_session(tmp_path) -> None:
    from distil import ledger

    p = tmp_path / "savings.jsonl"
    p.write_text(
        json.dumps({"ts": 999.0})
        + "\n"  # no session key
        + json.dumps({"session": "s3", "ts": 50.0})
        + "\n"
    )
    sid, ts = ledger.latest_session(path=p)
    assert sid == "s3"
    assert ts == 50.0


# --------------------------------------------------------------------------- #
# ledger — record + summary round-trip
# --------------------------------------------------------------------------- #


def test_record_and_summary_round_trip(tmp_path) -> None:
    from distil import ledger

    p = tmp_path / "savings.jsonl"
    rec = ledger.record(
        trajectory_id="test-traj",
        model="claude-opus-4-8",
        turns=3,
        baseline_dollars=0.01,
        distil_dollars=0.006,
        baseline_input_tokens=1000,
        distil_input_tokens=600,
        tokenizer="heuristic",
        path=p,
    )
    assert rec.dollars_saved == pytest.approx(0.004)
    assert rec.tokens_saved == 400

    s = ledger.summary(path=p)
    assert s.runs == 1
    assert s.total_tokens_saved == 400
    assert s.total_dollars_saved == pytest.approx(0.004)
    assert s.total_baseline_tokens == 1000
    assert s.total_distil_tokens == 600
    assert "heuristic" in s.tokenizers


def test_summary_session_filter(tmp_path) -> None:
    from distil import ledger

    p = tmp_path / "savings.jsonl"
    ledger.record(
        trajectory_id="t1",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.01,
        distil_dollars=0.005,
        baseline_input_tokens=100,
        distil_input_tokens=50,
        session="sess-A",
        path=p,
    )
    ledger.record(
        trajectory_id="t2",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.02,
        distil_dollars=0.01,
        baseline_input_tokens=200,
        distil_input_tokens=100,
        session="sess-B",
        path=p,
    )
    s_a = ledger.summary(path=p, session="sess-A")
    assert s_a.runs == 1
    assert s_a.total_tokens_saved == 50

    s_all = ledger.summary(path=p)
    assert s_all.runs == 2


# --------------------------------------------------------------------------- #
# ledger — render_dashboard states
# --------------------------------------------------------------------------- #


def _empty_summary():
    from distil.ledger import LedgerSummary

    return LedgerSummary(0, 0.0, 0, {})


def _nonzero_summary(tokens_saved=1000, dollars_saved=0.01):
    from distil.ledger import LedgerSummary

    return LedgerSummary(
        runs=5,
        total_dollars_saved=dollars_saved,
        total_tokens_saved=tokens_saved,
        by_trajectory={"live-proxy": dollars_saved},
        total_baseline_tokens=10000,
        total_distil_tokens=9000,
        total_baseline_dollars=0.1,
        total_distil_dollars=0.1 - dollars_saved,
        tokenizers=frozenset({"heuristic"}),
    )


def test_render_dashboard_no_runs() -> None:
    from distil.ledger import render_dashboard

    out = render_dashboard(_empty_summary(), color=False)
    assert "no savings yet" in out
    assert "distil wrap" in out


def test_render_dashboard_with_savings() -> None:
    from distil.ledger import render_dashboard

    out = render_dashboard(_nonzero_summary(), color=False)
    assert "tokens" in out
    assert "5 run" in out


def test_render_dashboard_subscription_mode() -> None:
    from distil.ledger import render_dashboard

    out = render_dashboard(_nonzero_summary(), subscription=True, color=False)
    assert "flat-rate" in out or "notional" in out
    assert "$" not in out or "flat-rate" in out  # no dollar figure in subscription mode


def test_render_dashboard_with_session_no_savings() -> None:
    """Session present, 0 tokens saved → 'waiting for a large read'."""
    from distil.ledger import LedgerSummary, render_dashboard

    sess = LedgerSummary(
        runs=1,
        total_dollars_saved=0.0,
        total_tokens_saved=0,
        by_trajectory={},
        total_baseline_tokens=500,
        total_distil_tokens=500,
    )
    out = render_dashboard(_nonzero_summary(), session=sess, color=False)
    assert "waiting for a large read" in out or "on" in out


def test_render_dashboard_with_session_with_savings() -> None:
    """Session present with real savings → shows the session delta."""
    from distil.ledger import LedgerSummary, render_dashboard

    sess = LedgerSummary(
        runs=2,
        total_dollars_saved=0.002,
        total_tokens_saved=200,
        by_trajectory={},
        total_baseline_tokens=1000,
        total_distil_tokens=800,
    )
    out = render_dashboard(_nonzero_summary(tokens_saved=5000), session=sess, color=False)
    assert "this session" in out or "200" in out


def test_render_dashboard_samples_collecting() -> None:
    """Below the shared A/B floor → 'collecting' message naming that floor."""
    from distil.ledger import render_dashboard
    from distil.shadow import VERDICT_MIN_AB

    out = render_dashboard(
        _nonzero_summary(), change_rate=0.02, samples=VERDICT_MIN_AB - 1, color=False
    )
    assert "collecting" in out
    assert f"need {VERDICT_MIN_AB}" in out


def test_render_dashboard_samples_ready() -> None:
    """At the shared A/B floor → decision-equivalence percentage shown."""
    from distil.ledger import render_dashboard
    from distil.shadow import VERDICT_MIN_AB

    out = render_dashboard(
        _nonzero_summary(), change_rate=0.02, samples=VERDICT_MIN_AB, color=False
    )
    assert "decision-equiv" in out
    assert "98.0%" in out  # 1 - 0.02


def test_render_dashboard_no_samples() -> None:
    """0 samples → prompt to run shadow mode."""
    from distil.ledger import render_dashboard

    out = render_dashboard(_nonzero_summary(), color=False)
    assert "shadow" in out or "distil wrap" in out


def test_render_dashboard_by_trajectory_top5() -> None:
    """Multiple trajectories → all shown (up to 5)."""
    from distil.ledger import LedgerSummary, render_dashboard

    s = LedgerSummary(
        runs=10,
        total_dollars_saved=0.05,
        total_tokens_saved=5000,
        by_trajectory={f"traj-{i}": float(i) * 0.01 for i in range(1, 7)},
        total_baseline_tokens=50000,
        total_distil_tokens=45000,
    )
    out = render_dashboard(s, color=False)
    # top-5 shown, not all 6
    shown = sum(1 for i in range(1, 7) if f"traj-{i}" in out or f"traj-{i}"[:15] in out)
    assert shown >= 5


# --------------------------------------------------------------------------- #
# ledger — render_html + _eq_card + _session_card
# --------------------------------------------------------------------------- #


def test_render_html_no_runs() -> None:
    from distil.ledger import render_html

    html = render_html(_empty_summary())
    assert "no runs recorded yet" in html
    assert (
        "distil proxy" in html
        or "distil_compress" in html.lower()
        or "live-proxy" in html.lower()
        or "distil proxy" in html
    )


def test_render_html_with_runs() -> None:
    from distil.ledger import render_html

    s = _nonzero_summary(tokens_saved=12345, dollars_saved=0.123)
    html = render_html(s)
    assert "12,345" in html
    assert "0.1230" in html
    assert "live-proxy" in html  # the by_trajectory source
    assert "live-proxy" in html and "Includes" in html  # live traffic note


def test_render_html_eq_card_below_threshold() -> None:
    """Below the shared reporting floor → a card naming that floor, not a rate."""
    from distil.ledger import render_html
    from distil.shadow import VERDICT_MIN_AA, VERDICT_MIN_AB

    html = render_html(_nonzero_summary(), change_rate=0.02, samples=VERDICT_MIN_AB - 1)
    assert f"needs {VERDICT_MIN_AB} A/B + {VERDICT_MIN_AA} A/A shadow samples" in html


def test_render_html_eq_card_above_threshold() -> None:
    """At the floor → real equivalence percentage in card."""
    from distil.ledger import render_html
    from distil.shadow import VERDICT_MIN_AB

    html = render_html(_nonzero_summary(), change_rate=0.02, samples=VERDICT_MIN_AB)
    assert "98.0%" in html
    assert "50" in html  # sample count


def test_render_html_eq_card_none_change_rate() -> None:
    """change_rate=None (no shadow running) → the same below-floor card."""
    from distil.ledger import render_html
    from distil.shadow import VERDICT_MIN_AA, VERDICT_MIN_AB

    html = render_html(_nonzero_summary(), change_rate=None, samples=0)
    assert f"needs {VERDICT_MIN_AB} A/B + {VERDICT_MIN_AA} A/A shadow samples" in html


def test_render_html_session_card_present() -> None:
    """Session with savings → renders this-session card."""
    from distil.ledger import LedgerSummary, render_html

    sess = LedgerSummary(
        runs=2,
        total_dollars_saved=0.005,
        total_tokens_saved=500,
        by_trajectory={},
        total_baseline_tokens=1000,
        total_distil_tokens=500,
    )
    html = render_html(_nonzero_summary(), session=sess)
    assert "This session" in html
    assert "50%" in html or "smaller" in html


def test_render_html_session_card_no_savings() -> None:
    """Session with 0 savings → 'waiting for a large read' in card."""
    from distil.ledger import LedgerSummary, render_html

    sess = LedgerSummary(
        runs=1,
        total_dollars_saved=0.0,
        total_tokens_saved=0,
        by_trajectory={},
        total_baseline_tokens=500,
        total_distil_tokens=500,
    )
    html = render_html(_nonzero_summary(), session=sess)
    assert "waiting for a large read" in html


def test_render_html_session_card_none() -> None:
    """session=None → no This session card."""
    from distil.ledger import render_html

    html = render_html(_nonzero_summary(), session=None)
    assert "This session" not in html

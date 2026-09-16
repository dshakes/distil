"""Tests for `distil discover` — the cross-session missed-savings advisor.

Three synthetic sessions, written with the same ledger writers the proxy uses
(`ledger.record`, `write_session_manifest`, `append_session_request`) into a
DISTIL_HOME tempdir, so what is under test is the reader, not a hand-rolled
fixture shape that could drift from what the proxy actually writes:

  sA — digest mode, tool-heavy, churny, growing system prompt, drifting prefix.
  sB — lossless-only and large: the session leaving digest savings on the table.
  sC — small and clean: nothing to say about it, and too small to score.

Every detector's *negative* is asserted too. A detector that fires on everything
is an advertisement, not a finding, and the suppression rules (a cache-read share
that makes churn already-discounted; too few tools to triage; too few comparable
turns for a drift ratio) are the part that makes the report trustworthy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from distil import discover as dv
from distil.cli import main
from distil.ledger import (
    SavingsRecord,
    append_session_request,
    record,
    write_session_manifest,
)

NOW = time.time()

_TOOLS_A = [
    {"name": "mcp__heavy__one", "tokens": 4000},
    {"name": "mcp__heavy__two", "tokens": 3000},
    {"name": "mcp__heavy__three", "tokens": 2000},
    {"name": "bash", "tokens": 500},
    {"name": "grep", "tokens": 500},
]


def _manifest(sid: str, **flags: Any) -> None:
    base = {
        "expand": True,
        "session_delta": False,
        "lossless_only": False,
        "verbatim": False,
        "prefix_replay": True,
        "shadow_rate": 0.0,
        "shape_output": "off",
    }
    base.update(flags)
    write_session_manifest(
        {
            "sid": sid,
            "tool": "claude",
            "argv": ["claude"],
            "cwd": "/tmp/proj",
            "started_ts": NOW - 3600,
            "distil_version": "1.53.0",
            "billing": "metered",  # explicit: never let the host machine decide
            "flags": base,
        },
        sid,
    )


def _seed_a(home: Path, *, cache_read: int = 2000, cache_create: int = 1000) -> None:
    """Tool-heavy, churny, growing, drifting. The session with work to do."""
    _manifest("sA")
    record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=5,
        baseline_dollars=0.15,
        distil_dollars=0.05,
        baseline_input_tokens=30_000,
        distil_input_tokens=10_000,
        session="sA",
        mode="digest",
    )
    systems = [1000, 1000, 2000, 2000, 2000]
    hashes = ["p1", "p2", "p1", "p2", "p1"]
    for i, (sys_tok, phash) in enumerate(zip(systems, hashes)):
        append_session_request(
            {
                "ts": NOW - 300 + i,
                "model": "claude-opus-4-8",
                "status": 200,
                "booked": True,
                "mode": "digest",
                "stream": True,
                "client_stream": True,
                "duration_ms": 1000,
                "compressible_tokens": 6000,
                "tokens_saved": 4000,
                "overhead_tokens": sys_tok + 10_000,
                "system_tokens": sys_tok,
                "tools_tokens": 10_000,
                "tools": _TOOLS_A,
                "usage_input_tokens": 1000,
                "usage_output_tokens": 200,
                "usage_cache_tokens": cache_read + cache_create,
                "usage_cache_read": cache_read or None,
                "usage_cache_create": cache_create or None,
                "prefix_hash": phash,
                "prefix_bytes": 4096,
                "delta_tokens_saved": 0,
                "blocks": [{"h": "h-a", "sig": "log:l", "tokens": 3000}],
            },
            "sA",
        )


def _seed_single_request_session(home: Path, sid: str, prefix_hash: str, *, ts: float) -> None:
    """One session, one request, its own stable-prefix hash — the shape that
    exposes the cross-session flattening bug: no request in this session has a
    predecessor to compare a prefix against, so it must contribute zero pairs
    to `discover`'s aggregate no matter how many other sessions share the window."""
    _manifest(sid)
    record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.05,
        distil_dollars=0.02,
        baseline_input_tokens=15_000,
        distil_input_tokens=6_000,
        session=sid,
        mode="digest",
    )
    append_session_request(
        {
            "ts": ts,
            "model": "claude-opus-4-8",
            "status": 200,
            "booked": True,
            "mode": "digest",
            "compressible_tokens": 6000,
            "tokens_saved": 4000,
            "overhead_tokens": 10_000,
            "system_tokens": 1000,
            "tools_tokens": 10_000,
            "tools": [],
            "usage_input_tokens": 1000,
            "usage_output_tokens": 200,
            "usage_cache_tokens": 3000,
            "usage_cache_read": 2000,
            "usage_cache_create": 1000,
            "prefix_hash": prefix_hash,
            "prefix_bytes": 4096,
            "delta_tokens_saved": 0,
            "blocks": [],
        },
        sid,
    )


def _seed_b(home: Path) -> None:
    """Large and lossless-only: 1% saved where the digest tier was never reached."""
    _manifest("sB", lossless_only=True, expand=False)
    record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=2,
        baseline_dollars=0.50,
        distil_dollars=0.495,
        baseline_input_tokens=100_000,
        distil_input_tokens=99_000,
        session="sB",
        mode="lossless-only",
    )
    for i in range(2):
        append_session_request(
            {
                "ts": NOW - 200 + i,
                "model": "claude-opus-4-8",
                "status": 200,
                "booked": True,
                "mode": "lossless-only",
                "compressible_tokens": 20_000,
                "tokens_saved": 200,
                "overhead_tokens": 1000,
                "system_tokens": 500,
                "tools_tokens": 500,
                "tools": [{"name": "bash", "tokens": 500}],
                "blocks": [],
            },
            "sB",
        )


def _seed_c(home: Path) -> None:
    """Small, clean, and below the scoring floor — the session with no story."""
    _manifest("sC")
    record(
        trajectory_id="live-proxy",
        model="claude-opus-4-8",
        turns=1,
        baseline_dollars=0.02,
        distil_dollars=0.016,
        baseline_input_tokens=5000,
        distil_input_tokens=4000,
        session="sC",
        mode="digest",
    )
    append_session_request(
        {
            "ts": NOW - 100,
            "model": "claude-opus-4-8",
            "status": 200,
            "booked": True,
            "mode": "digest",
            "compressible_tokens": 500,
            "tokens_saved": 100,
            "overhead_tokens": 200,
            "system_tokens": 100,
            "tools_tokens": 100,
            "tools": [{"name": "bash", "tokens": 100}],
            "blocks": [],
        },
        "sC",
    )


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.delenv("DISTIL_SESSION", raising=False)
    return tmp_path


@pytest.fixture
def seeded(home: Path) -> Path:
    _seed_a(home)
    _seed_b(home)
    _seed_c(home)
    return home


def _ids(report: dv.Report) -> list[str]:
    return [a.id for a in report.actions]


def _by_id(report: dv.Report, aid: str) -> dv.Action:
    return next(a for a in report.actions if a.id == aid)


class TestDetectors:
    def test_every_detector_fires_on_the_seeded_window(self, seeded: Path) -> None:
        r = dv.scan()
        assert r.sessions == 3
        assert set(_ids(r)) == {
            "tool_overhead",
            "digest_off",
            "prefix_drift",
            "churn",
            "system_growth",
            "calibration",
        }

    def test_ranking_is_savings_first_then_descending_tokens(self, seeded: Path) -> None:
        r = dv.scan()
        # The window is exactly one day (the floor), so per-week is the total x7 and
        # the ordering below is arithmetic, not a fixture accident.
        assert _ids(r) == [
            "digest_off",  # 100k baseline x 91.5% - 1k already saved
            "tool_overhead",  # (4000+3000+2000) x 5 requests
            "churn",  # 3000 tokens x (5 folds - 1)
            "prefix_drift",  # 4k cache-write tokens on the 4 rows that actually drifted
            "system_growth",  # 1000 growth x 5/2 requests
            "calibration",  # a risk: recovers nothing, ranked last
        ]
        assert [a.kind for a in r.actions][-1] == "risk"
        savings = [a.tokens_per_week for a in r.actions if a.kind == "savings"]
        assert savings == sorted(savings, reverse=True)

    def test_tool_overhead_counts_the_three_costliest_definitions(self, seeded: Path) -> None:
        a = _by_id(dv.scan(), "tool_overhead")
        assert a.tokens_per_week == (4000 + 3000 + 2000) * 5 * 7
        assert "mcp__heavy__one" in a.title
        # The honest limit: distil never sees which tools were CALLED.
        assert "--transcript" in a.basis

    def test_tool_overhead_silent_when_there_is_nothing_to_triage(self, home: Path) -> None:
        _seed_c(home)  # one tool defined; "audit your tools" is not advice
        assert "tool_overhead" not in _ids(dv.scan())

    def test_tool_overhead_denominator_matches_dissects_own_and_never_goes_negative(
        self, home: Path
    ) -> None:
        """`sent` must be the exact same floored denominator `Dissection.overhead_share`
        uses, reused rather than re-derived. If accounting ever lets `tokens_saved`
        exceed `compressible_tokens` for a session, re-deriving `comp - saved` without
        the same `max(0, ...)` floor sends `sent` negative, which used to make the
        gate's ratio (`overhead / sent`) meaningless — sometimes suppressing a real
        finding, sometimes fabricating one."""
        _manifest("sF")
        record(
            trajectory_id="live-proxy",
            model="claude-opus-4-8",
            turns=1,
            baseline_dollars=0.05,
            distil_dollars=0.02,
            baseline_input_tokens=20_000,
            distil_input_tokens=8_000,
            session="sF",
            mode="digest",
        )
        append_session_request(
            {
                "ts": NOW - 100,
                "model": "claude-opus-4-8",
                "status": 200,
                "booked": True,
                "mode": "digest",
                "compressible_tokens": 6000,
                "tokens_saved": 20_000,  # > compressible_tokens: the bogus reading
                "overhead_tokens": 10_100,
                "system_tokens": 100,
                "tools_tokens": 10_000,
                "tools": _TOOLS_A,
                "usage_input_tokens": 500,
                "blocks": [],
            },
            "sF",
        )
        from distil import dissect as dz

        d = dz.dissect("sF")
        # Floored: 10,100 + max(0, 6,000 - 20,000) == 10,100, never negative.
        assert d.sent_tokens_total == 10_100
        assert d.overhead_share == pytest.approx(100.0)
        a = _by_id(dv.scan(), "tool_overhead")
        assert "100%" in a.title

    def test_digest_off_uses_the_published_benchmark_only_as_a_fallback(self, seeded: Path) -> None:
        a = _by_id(dv.scan(), "digest_off")
        assert a.tokens_per_week == round((100_000 * dv.BENCH_DIGEST_RATE - 1000) * 7)
        assert dv.BENCH_DIGEST_SOURCE in a.basis
        assert "not your traffic" in a.basis
        assert a.command.startswith("distil default --mode expand")

    def test_digest_off_prefers_this_machines_own_measured_rate(self, seeded: Path) -> None:
        for _ in range(60):
            record(
                trajectory_id="live-proxy",
                model="claude-opus-4-8",
                turns=1,
                baseline_dollars=0.01,
                distil_dollars=0.001,
                baseline_input_tokens=1000,
                distil_input_tokens=100,
                session="sA",
                mode="digest",
            )
        a = _by_id(dv.scan(), "digest_off")
        assert "your own traffic" in a.basis
        assert dv.BENCH_DIGEST_SOURCE not in a.basis

    def test_digest_off_discloses_injection_on_a_flat_rate_plan(self, home: Path) -> None:
        _seed_b(home)
        mp = home / "sessions" / "sB.json"
        man = json.loads(mp.read_text(encoding="utf-8"))
        man["billing"] = "subscription"
        mp.write_text(json.dumps(man), encoding="utf-8")
        r = dv.scan()
        assert "injects distil_expand" in _by_id(r, "digest_off").command
        assert "the request IS modified" in _by_id(r, "digest_off").command
        assert r.notional is True

    def test_digest_off_silent_when_every_session_reached_the_digest_tier(self, home: Path) -> None:
        _seed_a(home)
        _seed_c(home)
        assert "digest_off" not in _ids(dv.scan())

    def test_prefix_drift_prices_the_write_read_gap(self, seeded: Path) -> None:
        a = _by_id(dv.scan(), "prefix_drift")
        # 4 of sA's 5 requests actually drifted (each carries 1000 cache-write
        # tokens); the first request has no predecessor to drift from, so its
        # 1000 create tokens are a cold-start write, not a re-bill, and must NOT
        # be counted — pricing off create_tokens * drift_ratio (5000 * 100%)
        # would wrongly include it.
        assert a.tokens_per_week == 4 * 1000 * 7
        assert "4 of 4 turns" in a.title
        assert "1.15x gap" in a.basis

    def test_prefix_drift_needs_enough_comparable_turns(self, home: Path) -> None:
        """Two drifting turns is not a drift ratio, and reporting one would be a
        confident number over a sample that cannot carry it."""
        _seed_a(home)
        reqs = (home / "sessions" / "sA.requests.jsonl").read_text(encoding="utf-8").splitlines()
        (home / "sessions" / "sA.requests.jsonl").write_text("\n".join(reqs[:3]) + "\n")
        assert "prefix_drift" not in _ids(dv.scan())

    def test_prefix_drift_never_compares_across_a_session_boundary(self, home: Path) -> None:
        """Regression: five independent one-request sessions, each with its own
        distinct stable prefix, used to register as four "drift" pairs when the
        detector flattened every session's requests into one list and sorted it
        by timestamp — the session boundary itself looked like a broken cache. A
        single-request session has zero comparable pairs (a prefix is only
        "drifted" against the previous request in the SAME session), so five of
        them must contribute zero pairs, not four drifts."""
        for i, phash in enumerate(["p1", "p2", "p3", "p4", "p5"]):
            _seed_single_request_session(home, f"solo{i}", phash, ts=NOW - 500 + i * 10)
        assert "prefix_drift" not in _ids(dv.scan())

    def test_prefix_drift_fires_on_real_drift_unpolluted_by_other_sessions(
        self, home: Path
    ) -> None:
        """A real intra-session drift (sA) must report the exact same numbers
        whether or not other, unrelated single-request sessions share the window —
        proof the fix aggregates per-session rather than smearing one session's
        drift ratio across the whole window's request count."""
        _seed_a(home)
        for i, phash in enumerate(["q1", "q2", "q3", "q4", "q5"]):
            _seed_single_request_session(home, f"solo{i}", phash, ts=NOW - 500 + i * 10)
        a = _by_id(dv.scan(), "prefix_drift")
        # sA's 4 actually-drifted requests only (see the comment in
        # test_prefix_drift_prices_the_write_read_gap); the 5 solo sessions each
        # have zero pairs so contribute nothing.
        assert a.tokens_per_week == 4 * 1000 * 7
        assert "4 of 4 turns" in a.title  # sA's pairs only — the 5 solo sessions add none

    def test_prefix_drift_prices_the_drifted_rows_not_a_blended_average(self, home: Path) -> None:
        """A session where the stable turns carry LARGE cache-write tokens and the
        two drifting turns carry small ones must price off the drifted turns' own
        writes. `create_tokens_total * drift_ratio` — 20,100 total tokens x 40% —
        would claim 8,040/week; the two rows that actually re-billed wrote only
        50 tokens each."""
        _manifest("sD")
        record(
            trajectory_id="live-proxy",
            model="claude-opus-4-8",
            turns=6,
            baseline_dollars=0.5,
            distil_dollars=0.2,
            baseline_input_tokens=60_000,
            distil_input_tokens=24_000,
            session="sD",
            mode="digest",
        )
        # h1,h1 (stable, big writes) -> h2 (DRIFT, tiny write) -> h2 (stable) ->
        # h3 (DRIFT, tiny write) -> h3 (stable): 5 comparable pairs, 2 drifts.
        hashes = ["h1", "h1", "h2", "h2", "h3", "h3"]
        creates = [5000, 5000, 50, 5000, 50, 5000]
        for i, (phash, create) in enumerate(zip(hashes, creates)):
            append_session_request(
                {
                    "ts": NOW - 600 + i,
                    "model": "claude-opus-4-8",
                    "status": 200,
                    "booked": True,
                    "mode": "digest",
                    "compressible_tokens": 1000,
                    "tokens_saved": 500,
                    "overhead_tokens": 500,
                    "system_tokens": 500,
                    "tools_tokens": 0,
                    "tools": [],
                    "usage_input_tokens": 100,
                    "usage_cache_read": 0,
                    "usage_cache_create": create,
                    "prefix_hash": phash,
                    "blocks": [],
                },
                "sD",
            )
        a = _by_id(dv.scan(), "prefix_drift")
        assert a.tokens_per_week == (50 + 50) * 7

    def test_churn_excluded_when_the_provider_already_discounts_it(self, home: Path) -> None:
        """A 95%-cached session's resends are billed at the cache-read rate; counting
        them would promise a saving an order of magnitude bigger than it is."""
        _seed_a(home, cache_read=19_000, cache_create=1000)
        assert "churn" not in _ids(dv.scan())

    def test_churn_counted_when_it_is_actually_being_paid_for(self, seeded: Path) -> None:
        a = _by_id(dv.scan(), "churn")
        assert a.tokens_per_week == 3000 * 4 * 7
        assert "excluded, not assumed" in a.basis
        assert "--session-delta" in a.command

    def test_churn_ignores_sessions_whose_cache_share_was_never_measured(self, home: Path) -> None:
        """sB/sC carry no cache fields at all. Unmeasured is not cheap and not
        expensive — it is unknown, and an unknown must not enter an estimate."""
        _seed_b(home)
        _seed_c(home)
        assert "churn" not in _ids(dv.scan())

    def test_system_growth_charges_half_the_requests(self, seeded: Path) -> None:
        a = _by_id(dv.scan(), "system_growth")
        assert a.tokens_per_week == round(1000 * 5 / 2) * 7
        assert "midpoint" in a.basis

    def test_system_growth_silent_on_a_stable_system_prompt(self, home: Path) -> None:
        _seed_b(home)  # constant 500-token system prompt across both requests
        assert "system_growth" not in _ids(dv.scan())

    def test_calibration_is_a_risk_with_no_dollar_claim(self, seeded: Path) -> None:
        a = _by_id(dv.scan(), "calibration")
        assert a.kind == "risk"
        assert a.tokens_per_week == 0 and a.dollars_per_week is None
        assert "nothing to run" in a.command

    def test_calibration_silent_when_estimate_tracks_billed_usage(self, home: Path) -> None:
        _seed_a(home)
        path = home / "sessions" / "sA.requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            # raw estimate per request is overhead + (compressible - saved); make billed
            # match it so the ratio sits inside the band.
            r["usage_input_tokens"] = r["overhead_tokens"] + 2000
            r["usage_cache_tokens"] = 0
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert "calibration" not in _ids(dv.scan())


class TestBookedOnlyFiltering:
    """The proxy records a retried/failed call with ``booked: False`` — it never
    became a savings event, but the row still sits in ``<sid>.requests.jsonl``.
    Every figure below must read the exact same population dissect's own ledger
    headline is built from (`Dissection.booked_detail`), not the raw request-detail
    list, or a flaky upstream retry inflates the recommendation it never earned."""

    def _seed(self, home: Path) -> None:
        """3 booked + 2 unbooked (retried) requests on one session. The 2 unbooked
        rows carry a wildly different prefix hash and cache-create/block-fold
        count than any booked row — if they leaked into an estimate, it would be
        off by orders of magnitude, not by a rounding error."""
        _manifest("sG")
        record(
            trajectory_id="live-proxy",
            model="claude-opus-4-8",
            turns=3,
            baseline_dollars=0.1,
            distil_dollars=0.04,
            baseline_input_tokens=30_000,
            distil_input_tokens=12_000,
            session="sG",
            mode="digest",
        )
        block = [{"h": "shared", "sig": "log:l", "tokens": 1000}]
        rows = [
            # (ts_offset, booked, prefix_hash, cache_create)
            (0, True, "p1", 1000),
            (1, False, "zzzz1", 90_000),  # retried: huge drift/create if counted
            (2, True, "p1", 1000),
            (3, False, "zzzz2", 80_000),  # retried: huge drift/create if counted
            (4, True, "p2", 50),  # the one real drift, among the booked rows
        ]
        for i, booked, phash, create in rows:
            append_session_request(
                {
                    "ts": NOW - 200 + i,
                    "model": "claude-opus-4-8",
                    "status": 200 if booked else 529,
                    "booked": booked,
                    "mode": "digest",
                    "compressible_tokens": 1000,
                    "tokens_saved": 500,
                    "overhead_tokens": 500,
                    "system_tokens": 500,
                    "tools_tokens": 0,
                    "tools": [],
                    "usage_input_tokens": 100,
                    "usage_cache_read": 0,
                    "usage_cache_create": create,
                    "prefix_hash": phash,
                    "blocks": block,
                },
                "sG",
            )

    def test_request_count_is_booked_only(self, home: Path) -> None:
        self._seed(home)
        assert dv.scan().requests == 3

    def test_prefix_drift_summary_ignores_the_unbooked_rows(self, home: Path) -> None:
        self._seed(home)
        w = dv._collect(sessions=8, since_days=None)
        s = dv._prefix_summary(w)
        # Booked-only, sorted by ts: p1, p1, p2 — one stable pair then one real
        # drift (p1 -> p2, 50 create tokens). The unbooked rows' 90,000/80,000
        # create tokens and their two extra "drifts" must not appear at all.
        assert s.requests == 3
        assert s.pairs == 2
        assert s.drifts == 1
        assert s.drift_create_tokens == 50

    def test_churn_folds_only_the_booked_resends(self, home: Path) -> None:
        self._seed(home)
        from distil import dissect as dz

        d = dz.dissect("sG")
        # The shared block was folded on 3 booked requests (2 re-folds), not 5.
        assert d.churn_tokens == 1000 * (3 - 1)
        assert d.churned_blocks == 1

    def test_tool_overhead_numerator_ignores_unbooked_tool_definitions(self, home: Path) -> None:
        """The numerator (top-3 tool cost) must agree with the denominator
        (`sent_tokens_total`, already booked-only since round 3) — an unbooked
        retry's tool definitions must not be able to buy their way into the top 3
        and crowd out the tools the session is actually charged for."""
        _manifest("sH")
        record(
            trajectory_id="live-proxy",
            model="claude-opus-4-8",
            turns=3,
            baseline_dollars=0.3,
            distil_dollars=0.15,
            baseline_input_tokens=90_000,
            distil_input_tokens=30_000,
            session="sH",
            mode="digest",
        )
        for i in range(3):
            append_session_request(
                {
                    "ts": NOW - 300 + i,
                    "model": "claude-opus-4-8",
                    "status": 200,
                    "booked": True,
                    "mode": "digest",
                    "compressible_tokens": 6000,
                    "tokens_saved": 4000,
                    "overhead_tokens": 10_500,
                    "system_tokens": 500,
                    "tools_tokens": 10_000,
                    "tools": _TOOLS_A,
                    "usage_input_tokens": 1000,
                    "blocks": [],
                },
                "sH",
            )
        for i in range(2):
            # Retried/failed, unbooked: a giant tool definition that must never
            # out-rank the real top-3 the session is actually being charged for.
            append_session_request(
                {
                    "ts": NOW - 290 + i,
                    "model": "claude-opus-4-8",
                    "status": 529,
                    "booked": False,
                    "mode": "digest",
                    "compressible_tokens": 6000,
                    "tokens_saved": 4000,
                    "overhead_tokens": 10_500,
                    "system_tokens": 500,
                    "tools_tokens": 10_000,
                    "tools": [{"name": "mcp__unbooked__ghost", "tokens": 999_999}],
                    "usage_input_tokens": 1000,
                    "blocks": [],
                },
                "sH",
            )
        a = _by_id(dv.scan(), "tool_overhead")
        assert a.tokens_per_week == (4000 + 3000 + 2000) * 3 * 7
        assert "mcp__unbooked__ghost" not in a.title
        for name in ("mcp__heavy__one", "mcp__heavy__two", "mcp__heavy__three"):
            assert name in a.title


class TestTypicalVsBest:
    def test_median_is_reported_beside_the_best_and_they_differ(self, seeded: Path) -> None:
        r = dv.scan()
        # Only sA (30k) and sB (100k) clear the scoring floor; sC (5k) cannot carry a
        # percentage. The best session saved 66.7%; the typical one saved 1.0%.
        assert len(r.pcts) == 2
        assert r.best is not None and r.best[0] == "sA"
        assert r.best[1] == pytest.approx(66.67, abs=0.01)
        assert r.median_pct == pytest.approx(1.0, abs=0.01)
        assert r.p90_pct == pytest.approx(66.67, abs=0.01)

    def test_the_text_says_the_best_is_not_typical(self, seeded: Path) -> None:
        text = dv.render_text(dv.scan(), color=False)
        assert "the best session is not typical" in text
        assert "1.0% saved on the median session" in text

    def test_no_session_large_enough_to_score(self, home: Path) -> None:
        _seed_c(home)
        r = dv.scan()
        assert r.median_pct is None and r.best is None
        assert "no session in the window is large enough to score" in dv.render_text(r, color=False)


class TestWindowing:
    def test_sessions_cap_limits_the_window(self, seeded: Path) -> None:
        assert dv.scan(sessions=1).sessions == 1

    def test_since_days_excludes_older_sessions(self, seeded: Path) -> None:
        assert dv.scan(since_days=0.0001).sessions == 3  # all three are seconds old
        # Move sB's activity a week back; a 1-day window must drop it.
        led = Path(seeded / "savings.jsonl")
        rows = [json.loads(line) for line in led.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            if r.get("session") == "sB":
                r["ts"] = NOW - 7 * 86400
        led.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        reqs = seeded / "sessions" / "sB.requests.jsonl"
        old = [json.loads(line) for line in reqs.read_text(encoding="utf-8").splitlines()]
        for r in old:
            r["ts"] = NOW - 7 * 86400
        reqs.write_text("\n".join(json.dumps(r) for r in old) + "\n")
        mp = seeded / "sessions" / "sB.json"
        man = json.loads(mp.read_text(encoding="utf-8"))
        man["started_ts"] = NOW - 7 * 86400
        mp.write_text(json.dumps(man), encoding="utf-8")
        r2 = dv.scan(since_days=1.0)
        assert r2.sessions == 2
        assert "digest_off" not in _ids(r2)

    def test_rates_are_per_week_over_the_measured_window(self, seeded: Path) -> None:
        r = dv.scan()
        assert r.days == pytest.approx(1.0)  # floored: a few hours is not a week


class TestJsonSchema:
    def test_shape(self, seeded: Path) -> None:
        d = dv.scan().to_dict()
        assert set(d) == {"window", "typical", "tokens_per_week", "actions"}
        assert set(d["window"]) == {
            "sessions",
            "requests",
            "days",
            "notional_dollars",
            "calibrated",
        }
        assert set(d["typical"]) == {
            "median_pct_saved",
            "p10_pct_saved",
            "p90_pct_saved",
            "best_session",
            "best_pct_saved",
            "sessions_scored",
        }
        for a in d["actions"]:
            assert set(a) == {
                "id",
                "kind",
                "title",
                "tokens_per_week",
                "dollars_per_week",
                "basis",
                "command",
            }
            assert a["kind"] in ("savings", "risk")
        assert json.dumps(d)  # serializable, no stray objects

    def test_dollars_are_priced_through_the_ledgers_own_rate(self, seeded: Path) -> None:
        r = dv.scan()
        usd_per_token = (0.15 + 0.50 + 0.02) / (30_000 + 100_000 + 5000)
        a = _by_id(r, "tool_overhead")
        assert a.dollars_per_week == pytest.approx(a.tokens_per_week * usd_per_token)

    def test_dollars_are_none_when_the_window_has_no_priced_rows(self, home: Path) -> None:
        _seed_a(home)
        led = home / "savings.jsonl"
        rows = [json.loads(line) for line in led.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["baseline_dollars"] = 0.0
        led.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert all(a.dollars_per_week is None for a in dv.scan().actions)


class TestCli:
    def test_no_sessions_exits_zero_with_a_clear_message(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["discover"]) == 0
        out = capsys.readouterr().out
        assert "no wrap sessions recorded yet" in out
        assert "distil wrap" in out

    def test_json_output(self, seeded: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["discover", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["window"]["sessions"] == 3
        assert payload["actions"][0]["id"] == "digest_off"

    def test_no_color_output_has_no_escapes(
        self, seeded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["discover", "--no-color"]) == 0
        assert "\x1b[" not in capsys.readouterr().out

    def test_sessions_and_since_flags_reach_the_scan(
        self, seeded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["discover", "--sessions", "1", "--since", "1", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["window"]["sessions"] == 1

    def test_nothing_to_recommend_is_a_result_not_a_failure(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_c(home)
        assert main(["discover", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "nothing to recommend" in out
        assert "not a failure to look" in out


class TestWrapExitLine:
    def test_one_line_when_there_is_something_to_act_on(self, seeded: Path) -> None:
        line = dv.wrap_exit_line()
        assert line is not None
        assert line.startswith("distil discover: 6 actions could recover ~")
        assert "\n" not in line

    def test_silent_when_no_detector_fires(self, home: Path) -> None:
        _seed_c(home)
        assert dv.wrap_exit_line() is None

    def test_fail_open_on_its_own_not_just_via_its_caller(
        self, seeded: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`wrap_exit_line` must swallow a broken `scan()` itself — a future
        caller that forgets `proof_ledger`'s own try/except must not crash."""

        def boom(*, sessions: int = 8) -> dv.Report:
            raise RuntimeError("scan exploded")

        monkeypatch.setattr(dv, "scan", boom)
        assert dv.wrap_exit_line() is None

    def test_silent_on_an_empty_home(self, home: Path) -> None:
        assert dv.wrap_exit_line() is None

    def test_proof_ledger_carries_it_only_when_it_exists(self, seeded: Path) -> None:
        from distil import proof_ledger as pl

        text = pl.build_ledger_text("sA", NOW - 60)
        assert text is not None and "distil discover:" in text
        assert "run: distil discover" in text

    def test_proof_ledger_survives_a_broken_advisor(
        self, seeded: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The advisory is the least important line in the proof ledger; it must
        never be the reason the proof itself goes missing."""
        from distil import proof_ledger as pl

        def boom(**_kw: Any) -> str:
            raise RuntimeError("scan exploded")

        monkeypatch.setattr(dv, "wrap_exit_line", boom)
        text = pl.build_ledger_text("sA", NOW - 60)
        assert text is not None and "distil proof ledger" in text
        assert "distil discover:" not in text


class TestDissectReuse:
    def test_prefiltered_rows_and_skipped_shadow_give_the_same_report(self, seeded: Path) -> None:
        """discover passes `ledger_rows=`/`shadow=False` purely to avoid re-parsing;
        if that ever changed the numbers, every figure on the report would be a
        second implementation of dissect's."""
        from distil import dissect as dz

        full = dz.dissect("sA")
        rows = [r for r in dz._read_jsonl(seeded / "savings.jsonl") if r.get("session") == "sA"]
        lean = dz.dissect("sA", ledger_rows=rows, shadow=False)
        assert lean.baseline_tokens == full.baseline_tokens
        assert lean.pct_saved == full.pct_saved
        assert lean.blocks == full.blocks
        assert lean.overhead_tokens_total == full.overhead_tokens_total
        assert lean.churn_tokens == full.churn_tokens


class TestPerformance:
    """``wrap_exit_line`` runs on every ``distil wrap`` exit — it must stay cheap
    on a ledger with years of history, not just on the 3-session fixture above."""

    @staticmethod
    def _big_home(home: Path, *, old_sessions: int = 4600, cur_sessions: int = 8) -> None:
        """~37k ledger rows (~11MB) of history plus `cur_sessions` current wrap
        sessions of 150 request rows each — the scale a long-lived machine
        actually accumulates. Old rows are written directly (one bulk write)
        rather than through `record()`, whose per-call file lock would make
        building this fixture itself the slow part of the test."""
        now = time.time()
        lines = [
            json.dumps(
                SavingsRecord(
                    "live-proxy",
                    "claude-opus-4-8",
                    1,
                    0.01,
                    0.004,
                    2000,
                    800,
                    "heuristic",
                    now - 1_000_000 + i,
                    f"old{s}",
                    mode="digest",
                ).__dict__
            )
            for s in range(old_sessions)
            for i in range(8)
        ]
        (home / "savings.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for s in range(cur_sessions):
            sid = f"cur{s}"
            write_session_manifest(
                {
                    "sid": sid,
                    "tool": "claude",
                    "argv": ["claude"],
                    "cwd": "/tmp/proj",
                    "started_ts": now - 3600,
                    "distil_version": "1.53.0",
                    "billing": "metered",
                    "flags": {"expand": True, "lossless_only": False, "verbatim": False},
                },
                sid,
            )
            record(
                trajectory_id="live-proxy",
                model="claude-opus-4-8",
                turns=150,
                baseline_dollars=5.0,
                distil_dollars=2.0,
                baseline_input_tokens=300_000,
                distil_input_tokens=120_000,
                session=sid,
                mode="digest",
            )
            for i in range(150):
                append_session_request(
                    {
                        "ts": now - 3000 + i,
                        "model": "claude-opus-4-8",
                        "status": 200,
                        "booked": True,
                        "mode": "digest",
                        "compressible_tokens": 2000,
                        "tokens_saved": 1200,
                        "overhead_tokens": 2500,
                        "system_tokens": 1000,
                        "tools_tokens": 2400,
                        "tools": [{"name": "bash", "tokens": 500}],
                        "prefix_hash": "p1" if i % 3 else "p2",
                        "blocks": [{"h": f"h-{i}", "sig": "log:l", "tokens": 500}],
                    },
                    sid,
                )

    def test_ledger_parsed_once_per_scan(self, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression pin: `_collect` and `list_sessions` must share one parse of
        `savings.jsonl`. This used to be two full parses (one inside `list_sessions`,
        one to build the per-session row index) — on a large ledger that doubling
        is the dominant cost of every `distil wrap` exit."""
        self._big_home(home, old_sessions=50, cur_sessions=1)
        calls: list[Path] = []
        orig = dv._read_jsonl

        def counting(path: Path) -> list[dict[str, Any]]:
            calls.append(path)
            return orig(path)

        monkeypatch.setattr(dv, "_read_jsonl", counting)
        dv.scan(sessions=8)
        ledger_calls = [p for p in calls if p.name == "savings.jsonl"]
        assert len(ledger_calls) == 1

    def test_wrap_exit_line_bounded_at_scale(self, home: Path) -> None:
        """~37k ledger rows / ~11MB, 8 current sessions of 150 requests each (the
        scale a long-lived machine accumulates) must stay well under wrap-exit
        budget. 1s is a generous CI-noise margin over the ~0.1-0.2s measured on a
        dev laptop — this catches an accidental re-parse-everything regression,
        not a few-ms drift."""
        self._big_home(home)
        dv.wrap_exit_line(sessions=8)  # warm any filesystem cache
        t0 = time.perf_counter()
        dv.wrap_exit_line(sessions=8)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"wrap_exit_line took {elapsed:.3f}s on a 37k-row ledger"

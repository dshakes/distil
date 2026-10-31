"""Learned cache TTL: bounds, fallback, persistence, and that cold-point honours it."""

from __future__ import annotations

import json

import pytest

from distil import coldpoint, ttl_estimator as te

P, M, T = "anthropic", "claude-test", 300.0


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    te.reset()
    coldpoint.reset()
    yield tmp_path
    te.reset()
    coldpoint.reset()


def test_identity_until_enough_hits() -> None:
    te.observe(P, M, T, 900, 1000)
    te.observe(P, M, T, 900, 1000)
    assert te.bounds(P, M, T) == (900, None, 2)
    assert te.effective_ttl(P, M, T) == T  # 2 < MIN_HITS
    te.observe(P, M, T, 600, 1000)
    assert te.effective_ttl(P, M, T) == 900  # lower bound = largest hit gap


def test_misses_never_shorten() -> None:
    for _ in range(3):
        te.observe(P, M, T, 100, 1000)
    te.observe(P, M, T, 120, 0)  # miss at 120s: upper bound recorded...
    lo, hi, _ = te.bounds(P, M, T)
    assert (lo, hi) == (100, 120)
    assert te.effective_ttl(P, M, T) == T  # ...but the documented TTL still stands


def test_noise_is_ignored() -> None:
    te.observe(P, M, T, 5, 1000)  # gap too short to say anything
    te.observe(P, M, T, 900, 0, evicted=True)  # miss on a turn that rewrote the prefix
    te.observe(P, M, float("inf"), 900, 1000)
    assert te.bounds(P, M, T) == (0.0, None, 0)


def test_keys_are_separate() -> None:
    for _ in range(3):
        te.observe(P, M, T, 900, 1)
    assert te.effective_ttl(P, "other", T) == T
    assert te.effective_ttl("openai", M, T) == T
    assert te.effective_ttl(P, M, 3600.0) == 3600.0


def test_persistence_and_corrupt_file(_home) -> None:
    for _ in range(3):
        te.observe(P, M, T, 900, 1)
    assert (_home / "cache_ttl.json").exists()
    te.reset()  # a new process
    assert te.effective_ttl(P, M, T) == 900
    (_home / "cache_ttl.json").write_text("{not json")
    te.reset()
    assert te.effective_ttl(P, M, T) == T  # unreadable -> documented


def test_no_rewrite_when_nothing_changes(_home) -> None:
    for _ in range(4):
        te.observe(P, M, T, 900, 1)
    f = _home / "cache_ttl.json"
    before = f.stat().st_mtime_ns
    te.observe(P, M, T, 500, 1)
    assert f.stat().st_mtime_ns == before
    assert json.loads(f.read_text())["keys"][f"{P}|{M}|300s"][0] == 900


def _body():
    return {
        "model": M,
        "messages": [{"role": "user", "content": "hi"}],
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
    }


def _turn(key, now, read=None):
    p = coldpoint.plan(key, _body(), _body()["messages"], lambda: frozenset({"x"}), now=now)
    if read is not None:
        coldpoint.observe_usage(key, read)
    coldpoint.end(key, now=now)
    return p


def test_cold_point_learns_and_uses_it() -> None:
    t = 0.0
    _turn("k", t)
    # Three hits after 500s gaps (> 300s TTL): the entry outlived the documented TTL.
    for _ in range(3):
        t += 500
        _turn("k", t, read=5000)
    assert te.effective_ttl(P, M, T) == 500
    # 400s idle: past documented 300+60 -> would be cold, but learned 500 keeps it warm.
    t += 400
    assert _turn("k", t).reason == "warm"
    t += 500 + coldpoint.MARGIN_S + 1
    assert _turn("k", t).reason == "cold"


def test_without_evidence_cold_point_is_unchanged() -> None:
    _turn("k", 0.0)
    assert _turn("k", 400.0).reason == "cold"


def test_deferred_retry_respects_learned_ttl_and_its_miss_is_not_evidence() -> None:
    """A preserved-thinking deferral (extend -> None) followed by a retry: the retry is
    `deferred-cold`, never `cold`, while the learned TTL still holds, and the miss on the
    turn where distil finally evicts does not narrow the estimator's upper bound."""
    t = 0.0
    _turn("k", t)
    for _ in range(3):
        t += 500
        _turn("k", t, read=5000)
    assert te.effective_ttl(P, M, T) == 500
    t += 500 + coldpoint.MARGIN_S + 1
    cands = lambda: frozenset({"y"})  # noqa: E731
    p = coldpoint.plan("k", _body(), _body()["messages"], cands, now=t, extend=lambda f: None)
    coldpoint.end("k", now=t)
    assert p.reason == "deferred" and not p.fresh
    t += 400  # past documented 300+60, inside learned 500+60
    p = coldpoint.plan("k", _body(), _body()["messages"], cands, now=t, extend=lambda f: f)
    assert p.reason == "deferred-cold" and p.fresh == 1
    coldpoint.observe_usage("k", 0)  # distil's own edit: a miss proves nothing
    coldpoint.end("k", now=t)
    assert te.bounds(P, M, T)[1] is None


def test_thinking_drop_turn_miss_is_not_evidence() -> None:
    _turn("k", 0.0)
    coldpoint.plan("k", _body(), _body()["messages"], lambda: frozenset(), now=200.0)
    coldpoint.note_rewrite("k")
    coldpoint.observe_usage("k", 0)
    coldpoint.end("k", now=200.0)
    assert te.bounds(P, M, T)[1] is None
    _turn("k", 400.0, read=0)  # a clean miss still counts
    assert te.bounds(P, M, T)[1] == 200.0
    coldpoint.note_rewrite("missing")  # unknown lineage: no-op

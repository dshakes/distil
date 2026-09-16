"""GatewayState LRU cap (P2-13) and disk persistence (P1-7)."""

from __future__ import annotations

import distil.gateway as gw
from distil.gateway import GatewayState
from distil.pricing import get as pricing_get


def _state():
    return GatewayState(pricing_get("claude-opus-4-8"))


def test_persist_round_trip(tmp_path):
    p = tmp_path / "gateway_state.json"
    s = _state()
    s.record("anon-aaa", 100, 40)
    s.record("anon-aaa", 50, 20)
    s.record("acme", 200, 80)
    s.save(p)

    restored = _state()
    restored.load(p)
    snap = restored.snapshot()
    by_tenant = {t["tenant"]: t for t in snap["tenants"]}
    assert by_tenant["anon-aaa"]["requests"] == 2
    assert by_tenant["anon-aaa"]["tokens_baseline"] == 150
    assert by_tenant["acme"]["tokens_compressed"] == 80


def test_load_tolerates_missing_and_corrupt(tmp_path):
    s = _state()
    s.load(tmp_path / "does-not-exist.json")  # no crash
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"tenants": {"x": {"requests": "oops"}, "y": {"requests": 3, '
        '"tokens_baseline": 10, "tokens_compressed": 4}}}'
    )
    s.load(bad)
    tenants = {t["tenant"]: t for t in s.snapshot()["tenants"]}
    assert "x" not in tenants  # corrupt entry skipped
    assert tenants["y"]["requests"] == 3


def test_lru_eviction_bounds_the_map(monkeypatch):
    monkeypatch.setattr(gw, "_MAX_TENANTS", 3)
    s = _state()
    for i in range(5):
        s.record(f"t{i}", 10, 4)
    tenants = {t["tenant"] for t in s.snapshot()["tenants"]}
    assert len(tenants) == 3
    assert tenants == {"t2", "t3", "t4"}  # oldest two evicted


def test_load_enforces_tenant_cap(monkeypatch, tmp_path):
    """A pre-cap (or hand-edited) state file must not exceed _MAX_TENANTS in memory."""
    p = tmp_path / "gateway_state.json"
    s = _state()
    for i in range(5):
        s.record(f"t{i}", 10, 4)
    s.save(p)

    monkeypatch.setattr(gw, "_MAX_TENANTS", 3)
    s2 = _state()
    s2.load(p)
    assert len(s2.snapshot()["tenants"]) == 3


def test_save_swaps_through_the_shared_retrying_replace(tmp_path, monkeypatch):
    """`save` is an atomic writer too, so its swap owes the same Windows retry.

    Behaviour is covered once, in tests/test_filelock.py. What this call site owes is that
    it routes through the shared helper: two gateway workers persisting counters race on
    one state file, which is exactly the contention that fails a replace on Windows."""
    from pathlib import Path

    from distil import _filelock

    seen: list[str] = []
    real = _filelock.replace_retrying

    def spy(src, dst):
        seen.append(Path(dst).name)
        return real(src, dst)

    monkeypatch.setattr(_filelock, "replace_retrying", spy)
    p = tmp_path / "gateway_state.json"
    s = _state()
    s.record("acme", 200, 80)
    s.save(p)

    assert seen == ["gateway_state.json"], "the swap bypassed _filelock.replace_retrying"
    assert p.exists()

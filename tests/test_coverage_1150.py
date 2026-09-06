"""Coverage for 1.15.0 new code paths: shadow counters + diagnostic stats output,
and query-intent extraction edge shapes. Keeps the 95% ratcheting floor met."""

import argparse
import json

from distil import shadow as shadow_mod
from distil.cli import cmd_shadow_stats
from distil.compress.intent import extract_intent
from distil.shadow import ShadowCounters


# ---- ShadowCounters (content-free observability) ---------------------------------


def test_shadow_counters_roundtrip(tmp_path):
    p = tmp_path / "shadow_counters.json"
    c = ShadowCounters(p)
    c.note_seen()
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, replay_failed=True, fail_reason="401")
    c.flush_with(replay_attempted=True, recorded=True)
    c.flush_with(sig_none_skipped=True)
    data = ShadowCounters.load(p)
    assert data["requests_seen"] == 2
    assert data["sampled"] == 1
    assert data["replay_attempted"] == 2
    assert data["replay_failed"] == 1
    assert data["last_fail_reason"] == "401"
    assert data["signature_none_skipped"] == 1
    assert data["recorded"] == 1


def test_shadow_counters_load_missing(tmp_path):
    assert ShadowCounters.load(tmp_path / "nope.json") == {}


def test_shadow_counters_corrupt_file_recovers(tmp_path):
    # a corrupt counters file must not crash: load returns {}, and a subsequent
    # flush rewrites cleanly (defensive JSON branches in _write / load).
    p = tmp_path / "shadow_counters.json"
    p.write_text("{not json")
    assert ShadowCounters.load(p) == {}
    ShadowCounters(p).flush_with(recorded=True)
    assert ShadowCounters.load(p).get("recorded") == 1


# ---- cmd_shadow_stats diagnostic branches (0 recorded, attempts made) ------------


def _run_stats(tmp_path, monkeypatch, counters: dict) -> str:
    import io
    from contextlib import redirect_stdout

    monkeypatch.setattr(shadow_mod, "_state_dir", lambda: tmp_path)
    (tmp_path / ShadowCounters._FILENAME).write_text(json.dumps(counters))
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_shadow_stats(argparse.Namespace(all=False, json=False))
    return buf.getvalue()


def test_stats_all_replays_failed(tmp_path, monkeypatch):
    out = _run_stats(
        tmp_path,
        monkeypatch,
        {
            "requests_seen": 19,
            "sampled": 2,
            "replay_attempted": 2,
            "replay_failed": 2,
            "last_fail_reason": "401",
            "recorded": 0,
        },
    )
    assert "19 seen" in out and "2 failed" in out
    assert "All replays failed" in out


def test_stats_signature_none_skipped(tmp_path, monkeypatch):
    out = _run_stats(
        tmp_path,
        monkeypatch,
        {
            "requests_seen": 10,
            "sampled": 3,
            "replay_attempted": 3,
            "replay_failed": 0,
            "signature_none_skipped": 3,
            "recorded": 0,
        },
    )
    assert "seen" in out


def test_stats_json_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow_mod, "_state_dir", lambda: tmp_path)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_shadow_stats(argparse.Namespace(all=False, json=True))
    json.loads(buf.getvalue())  # valid JSON on the empty-ledger path


# ---- cmd_shadow_stats with a populated ledger (samples-present branches) ----------


def _ledger(tmp_path, n_ab, n_aa=0, *, paired=False):
    """A real ShadowLedger. Duck-typed stubs here used to drift from the class they
    imitated and hide exactly the reporting bugs these tests exist to cover."""
    led = shadow_mod.ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    if paired:
        for i in range(n_ab):
            led.record(i > 0, kind="paired", evidence={"aa_equal": True}, path=p)
        return led
    for i in range(n_ab):
        led.record(i > 0, path=p)
    for _ in range(n_aa):
        led.record(True, kind="aa", path=p)
    return led


def _run_stats_with_ledger(monkeypatch, tmp_path, led, counters=None):
    import io
    from contextlib import redirect_stdout

    monkeypatch.setattr(shadow_mod, "_state_dir", lambda: tmp_path)
    monkeypatch.setattr(shadow_mod.ShadowLedger, "load", lambda current_only=True: led)
    if counters is not None:
        (tmp_path / ShadowCounters._FILENAME).write_text(json.dumps(counters))
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_shadow_stats(argparse.Namespace(all=True, json=False))
    return buf.getvalue()


def test_stats_samples_present_paired(tmp_path, monkeypatch):
    out = _run_stats_with_ledger(
        monkeypatch,
        tmp_path,
        _ledger(tmp_path, 60, paired=True),
        counters={
            "requests_seen": 70,
            "sampled": 60,
            "replay_attempted": 60,
            "replay_failed": 1,
            "last_fail_reason": "500",
            "recorded": 60,
        },
    )
    assert "paired difference" in out
    assert "60" in out


def test_stats_samples_present_legacy_unpaired(tmp_path, monkeypatch):
    out = _run_stats_with_ledger(monkeypatch, tmp_path, _ledger(tmp_path, 60, 40))
    assert "legacy-unpaired" in out


def test_stats_collecting_still_prints_the_diagnostics(tmp_path, monkeypatch):
    """Below the floor there is no verdict — but the sampling counters must still
    print, because a thin sample is usually a replay that is failing."""
    out = _run_stats_with_ledger(
        monkeypatch,
        tmp_path,
        _ledger(tmp_path, 5),
        counters={"requests_seen": 40, "sampled": 5, "replay_attempted": 5, "replay_failed": 5},
    )
    assert "below reporting floor" in out
    assert "5 failed" in out


# ---- intent extraction edge shapes -----------------------------------------------


def test_relevant_lines_empty_intent_early_return():
    from distil.compress.intent import relevant_lines

    assert relevant_lines(["anything", "here"], frozenset()) == set()


def test_has_recoverable_stub_non_serializable_body():
    from distil.proxy import _has_recoverable_stub

    # a set is not JSON-serializable -> TypeError path -> False (never raises)
    assert _has_recoverable_stub({"messages": [{"role": "user", "content": {1, 2, 3}}]}) is False


# ---- anthropic adapter branches --------------------------------------------------


def test_restore_store_handle_collision_refused():
    from distil.adapters.anthropic import RestoreStore

    s = RestoreStore()
    assert s._record("h", "original") is True
    # same handle, different content -> refuse (never emit a stub that expands wrong)
    assert s._record("h", "different") is False


def test_apply_tier0_keeps_collapse_when_smaller():
    from distil.adapters.anthropic import _apply_tier0

    out = _apply_tier0("\n".join(["duplicate line"] * 40))
    assert out != "\n".join(["duplicate line"] * 40)  # collapse kept (fewer tokens)


def test_tool_result_active_keep_returns_tier0():
    from distil.adapters import anthropic as amod

    amod._keep_tls.fn = lambda _t: True  # force keep-byte-exact (order-independent)
    try:
        text = "\n".join(f"line {i}" for i in range(20))
        out = amod._compress_tool_result_text(text, amod.RestoreStore(), False)
        assert "handle=" not in out  # kept via tier-0, not digested behind a handle
    finally:
        amod._keep_tls.fn = None


def test_flush_with_swallows_write_errors(tmp_path, monkeypatch):
    # counter writes must never surface — a failing _write is swallowed (buffer line)
    c = ShadowCounters(tmp_path / "c.json")

    def _boom(*_a, **_k):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(c, "_write", _boom)
    c.flush_with(recorded=True)  # must not raise


def test_extract_intent_list_user_content_and_input_shapes():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "find deadlock in scheduler.rs"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "grep",
                    "input": {"patterns": ["acquire_lock", "release_lock"], "flag": None, "n": 5},
                }
            ],
        },
    ]
    terms = extract_intent(messages)
    assert "acquire_lock" in terms and "release_lock" in terms
    assert "scheduler.rs" in terms and "deadlock" in terms


def test_extract_intent_no_user_no_tooluse():
    assert extract_intent([{"role": "assistant", "content": "hi"}]) == frozenset()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    test_shadow_counters_roundtrip(d)
    test_shadow_counters_load_missing(d)
    test_extract_intent_no_user_no_tooluse()
    print("ok")

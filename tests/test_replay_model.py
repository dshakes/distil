"""``distil replay-model`` (distil/replay_model.py): offline tests.

Covers: tiered pricing, dry-run statistics, spend-cap refusal, transcript turn
parsing, and the Wilson CI used for the equivalence report. All offline — no API
calls, no real session or transcript files needed.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from distil import pricing, replay_model as rm
from distil.shadow import wilson_ci


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _record(
    model: str = "claude-opus-5-5",
    inp: int = 10_000,
    cr: int = 0,
    cw: int = 0,
    out: int = 200,
    ts: float | None = None,
) -> rm.ReplayRecord:
    return rm.ReplayRecord(
        model=model,
        input_tokens=inp,
        cache_read=cr,
        cache_write=cw,
        output_tokens=out,
        mode="digest",
        ts=ts or time.time(),
    )


def _sessions_dir(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "sessions"
    d.mkdir()
    f = d / "s12345.requests.jsonl"
    f.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return d


# --------------------------------------------------------------------------- #
# Pricing: haiku-5-5 two-tier
# --------------------------------------------------------------------------- #


def test_haiku55_is_tiered_not_in_catalog() -> None:
    p = rm._price("claude-haiku-5-5")
    assert p is not None
    assert p.name == "claude-haiku-5-5"
    assert p.input_per_mtok == pytest.approx(0.10)
    assert p.output_per_mtok == pytest.approx(0.50)
    assert p.long_tier == (100_000, 5.0)
    assert pricing.resolve("claude-haiku-5-5") is None  # one-tier callers never price it


def test_haiku55_short_tier_no_multiplier() -> None:
    rec = _record(model="claude-haiku-5-5", inp=50_000, out=100)  # total_input = 50k < 100k
    cost = rm.tiered_cost(rec, "claude-haiku-5-5")
    assert cost is not None
    p = rm._price("claude-haiku-5-5")
    assert p is not None
    expected = 50_000 * p.input + 100 * p.output
    assert cost == pytest.approx(expected)


def test_haiku55_long_tier_applies_5x() -> None:
    rec = _record(model="claude-haiku-5-5", inp=120_000, out=100)  # total > 100k
    cost = rm.tiered_cost(rec, "claude-haiku-5-5")
    assert cost is not None
    p = rm._price("claude-haiku-5-5")
    assert p is not None
    # 5x multiplier on all tokens
    expected = 5.0 * (120_000 * p.input + 100 * p.output)
    assert cost == pytest.approx(expected)


def test_haiku55_long_tier_boundary_at_100k() -> None:
    """Token count exactly AT the threshold stays at the short tier."""
    rec_at = _record(model="claude-haiku-5-5", inp=100_000, out=10)
    rec_over = _record(model="claude-haiku-5-5", inp=100_001, out=10)
    cost_at = rm.tiered_cost(rec_at, "claude-haiku-5-5")
    cost_over = rm.tiered_cost(rec_over, "claude-haiku-5-5")
    assert cost_at is not None and cost_over is not None
    # <=100k: no multiplier; >100k: 5x; so cost_over should be much larger
    assert cost_over > cost_at * 4


def test_tiered_cost_unknown_model_returns_none() -> None:
    rec = _record()
    assert rm.tiered_cost(rec, "claude-unknown-model-9-9") is None


def test_original_cost_unknown_model_returns_none() -> None:
    rec = _record(model="claude-unknown-model-9-9")
    assert rm.original_cost(rec) is None


def test_cache_fields_use_correct_multipliers() -> None:
    """Cache reads and writes at target model's cache rate, not base rate."""
    rec = _record(model="claude-haiku-5-5", inp=0, cr=50_000, cw=0, out=0)
    cost = rm.tiered_cost(rec, "claude-haiku-5-5")
    p = rm._price("claude-haiku-5-5")
    assert p is not None and cost is not None
    # total_input = cr = 50k, below threshold; no multiplier
    assert cost == pytest.approx(50_000 * p.cache_read)


# --------------------------------------------------------------------------- #
# Load records from sessions/*.requests.jsonl
# --------------------------------------------------------------------------- #


def _row(
    model: str = "claude-opus-5-5",
    inp: int = 10_000,
    cr: int = 0,
    cw: int = 0,
    out: int = 200,
    booked: bool = True,
    status: int = 200,
    ts: float | None = None,
) -> dict:
    return {
        "booked": booked,
        "model": model,
        "usage_input_tokens": inp,
        "usage_cache_read": cr,
        "usage_cache_create": cw,
        "usage_output_tokens": out,
        "status": status,
        "mode": "digest",
        "ts": ts or time.time(),
        "session": "s12345",
    }


def test_load_records_basic(tmp_path: Path) -> None:
    rows = [_row(inp=5_000 + i * 1000) for i in range(10)]
    d = _sessions_dir(tmp_path, rows)
    records = rm.load_records(200, sessions_dir=d, rng=random.Random(0))
    assert len(records) == 10
    assert all(r.model == "claude-opus-5-5" for r in records)


def test_load_records_skips_unbooked(tmp_path: Path) -> None:
    rows = [_row(booked=True)] * 3 + [_row(booked=False)] * 5
    d = _sessions_dir(tmp_path, rows)
    records = rm.load_records(200, sessions_dir=d)
    assert len(records) == 3


def test_load_records_skips_no_tokens(tmp_path: Path) -> None:
    rows = [_row(inp=0, cr=0, cw=0)]
    d = _sessions_dir(tmp_path, rows)
    records = rm.load_records(200, sessions_dir=d)
    assert len(records) == 0


def test_load_records_respects_since(tmp_path: Path) -> None:
    old_ts = time.time() - 60 * 86400  # 60 days ago
    rows = [_row(ts=old_ts)] * 3 + [_row()] * 2
    d = _sessions_dir(tmp_path, rows)
    records = rm.load_records(200, since=time.time() - 1 * 86400, sessions_dir=d)
    assert len(records) == 2


def test_load_records_reservoir_samples_to_n(tmp_path: Path) -> None:
    rows = [_row(inp=i * 100 + 1) for i in range(100)]
    d = _sessions_dir(tmp_path, rows)
    records = rm.load_records(10, sessions_dir=d, rng=random.Random(42))
    assert len(records) == 10


def test_load_records_malformed_line_is_skipped(tmp_path: Path) -> None:
    d = tmp_path / "sessions"
    d.mkdir()
    f = d / "s1.requests.jsonl"
    f.write_text(
        "not json\n" + json.dumps(_row()) + "\n{broken\n",
        encoding="utf-8",
    )
    records = rm.load_records(200, sessions_dir=d)
    assert len(records) == 1


# --------------------------------------------------------------------------- #
# Dry-run report
# --------------------------------------------------------------------------- #


def test_dry_run_basic() -> None:
    records = [_record(model="claude-opus-5-5", inp=50_000 + i * 1000) for i in range(20)]
    report = rm.dry_run(records, "claude-haiku-5-5", days=7.0)
    assert report.n_sampled == 20
    assert report.target_model == "claude-haiku-5-5"
    assert "claude-opus-5-5" in report.model_counts
    # All prompts are ~50k tokens (below 100k), so no long tier
    assert report.n_long_tier == 0
    # Haiku-5-5 base rate is much cheaper than opus-5-5
    assert report.cost_ratio < 0.5


def test_dry_run_long_tier_counted() -> None:
    short = [_record(inp=50_000) for _ in range(5)]
    long_ = [_record(inp=150_000) for _ in range(5)]
    report = rm.dry_run(short + long_, "claude-haiku-5-5")
    assert report.n_long_tier == 5


def test_dry_run_format_contains_key_fields() -> None:
    records = [_record(inp=200_000)] * 10
    report = rm.dry_run(records, "claude-haiku-5-5", days=30.0)
    text = report.format()
    assert "dry-run" in text
    assert "claude-haiku-5-5" in text
    assert "cost ratio" in text
    assert "live" in text  # hint about live path


def test_dry_run_empty_records() -> None:
    report = rm.dry_run([], "claude-haiku-5-5")
    assert report.n_sampled == 0
    assert report.estimated_total_usd == 0.0


def test_dry_run_unpriced_original_counted() -> None:
    records = [_record(model="claude-unknown-xyz-1-0")]
    report = rm.dry_run(records, "claude-haiku-5-5")
    assert report.n_unpriced_original == 1


def test_dry_run_unpriced_target_counted() -> None:
    records = [_record()]
    report = rm.dry_run(records, "claude-unknown-xyz-1-0")
    assert report.n_unpriced_target == 1


# --------------------------------------------------------------------------- #
# Wilson CI (the equivalence statistic)
# --------------------------------------------------------------------------- #


def test_wilson_ci_all_equivalent() -> None:
    lo, hi = wilson_ci(50, 50)
    assert lo > 0.90  # near 1 but interval is nonzero
    assert hi <= 1.0


def test_wilson_ci_none_equivalent() -> None:
    lo, hi = wilson_ci(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi < 0.1


def test_wilson_ci_empty() -> None:
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ci_half() -> None:
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert hi - lo < 0.2  # reasonably tight at n=100


# --------------------------------------------------------------------------- #
# Spend cap refusal (live path gate)
# --------------------------------------------------------------------------- #


def test_cmd_replay_model_live_without_spend_cap_returns_2(tmp_path: Path, capsys) -> None:
    import argparse

    args = argparse.Namespace(
        live=True,
        spend_cap=None,
        target="claude-haiku-5-5",
        n=10,
        days=30.0,
        json=False,
    )
    rc = rm.cmd_replay_model(args)
    assert rc == 2
    captured = capsys.readouterr()
    assert "--spend-cap" in captured.err


def test_cmd_replay_model_dryrun_no_records_returns_1(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run with no sessions data exits 1 with a helpful message."""
    import argparse

    monkeypatch.setattr(rm, "_home", lambda: tmp_path / "distil_empty")

    args = argparse.Namespace(
        live=False,
        spend_cap=None,
        target="claude-haiku-5-5",
        n=10,
        days=30.0,
        json=False,
    )
    rc = rm.cmd_replay_model(args)
    assert rc == 1
    assert "distil wrap" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Transcript turn parsing
# --------------------------------------------------------------------------- #


def _make_transcript(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "projects" / "test-project"
    d.mkdir(parents=True)
    f = d / "session.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path / "projects"


def _user_row(uuid: str, parent: str | None, content: str = "hi") -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-10-29T10:00:00Z",
        "message": {"role": "user", "content": content},
    }


def _asst_row(
    uuid: str,
    parent: str,
    tool_name: str | None = None,
    model: str = "claude-opus-5-5",
    inp: int = 5000,
    out: int = 100,
) -> dict:
    if tool_name:
        content = [{"type": "tool_use", "id": "t1", "name": tool_name, "input": {"cmd": "ls"}}]
    else:
        content = [{"type": "text", "text": "done"}]
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "sess1",
        "timestamp": "2026-10-29T10:01:00Z",
        "message": {
            "role": "assistant",
            "model": model,
            "content": content,
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def test_load_transcript_turns_finds_tool_use(tmp_path: Path) -> None:
    rows = [
        _user_row("u1", None, "do something"),
        _asst_row("a1", "u1", tool_name="bash"),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root, rng=random.Random(0))
    assert len(turns) == 1
    assert turns[0].recorded_decision.startswith("tool:")
    assert turns[0].model == "claude-opus-5-5"


def test_load_transcript_turns_finds_text_decision(tmp_path: Path) -> None:
    rows = [
        _user_row("u1", None, "hello"),
        _asst_row("a1", "u1", tool_name=None),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root, rng=random.Random(0))
    assert len(turns) == 1
    assert turns[0].recorded_decision == "text"


def test_load_transcript_turns_skips_no_parent(tmp_path: Path) -> None:
    """An assistant turn with no reconstructable messages is excluded."""
    rows = [
        # assistant row with no parentUuid chain into the file
        _asst_row("a1", "missing-uuid", tool_name="bash"),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)
    assert len(turns) == 0


def test_load_transcript_turns_strips_thinking_blocks(tmp_path: Path) -> None:
    """Thinking blocks in prior assistant turns are removed from messages."""
    asst_with_thinking = {
        "type": "assistant",
        "uuid": "a0",
        "parentUuid": "u0",
        "timestamp": "2026-10-29T10:00:30Z",
        "message": {
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": "let me think..."},
                {"type": "text", "text": "intermediate"},
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    rows = [
        _user_row("u0", None, "start"),
        asst_with_thinking,
        _user_row("u1", "a0", "continue"),
        _asst_row("a1", "u1", tool_name="bash"),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)
    # The reconstructed messages must not contain thinking blocks
    all_types = [
        b.get("type")
        for turn in turns
        for m in turn.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict)
    ]
    assert "thinking" not in all_types


def test_load_transcript_turns_respects_since(tmp_path: Path) -> None:
    old_ts = "2020-01-01T00:00:00Z"
    rows = [
        _user_row("u1", None),
        {
            **_asst_row("a1", "u1", tool_name="bash"),
            "timestamp": old_ts,
            "message": {
                **_asst_row("a1", "u1", tool_name="bash")["message"],
            },
        },
    ]
    # Replace the timestamp
    rows[1]["timestamp"] = old_ts
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=time.time() - 86400, transcripts_dir=root)
    assert len(turns) == 0


# --------------------------------------------------------------------------- #
# Live result formatting
# --------------------------------------------------------------------------- #


def test_live_result_format() -> None:
    r = rm.LiveResult(
        n=100,
        n_equivalent=85,
        spend_usd=0.42,
        target_model="claude-haiku-5-5",
        cost_ratio=0.15,
        equivalence_rate=0.85,
        equivalence_ci=(0.76, 0.91),
    )
    text = r.format()
    assert "85/100" in text
    assert "claude-haiku-5-5" in text
    assert "$0.42" in text


# --------------------------------------------------------------------------- #
# Pricing canary: haiku-5-5 passes self-consistency
# --------------------------------------------------------------------------- #


def test_haiku55_passes_self_consistency() -> None:
    """The tiered row must satisfy the same invariants as the rest of the table."""
    p = pricing.TIERED["claude-haiku-5-5"]
    assert p.name == "claude-haiku-5-5"
    assert p.input_per_mtok > 0
    assert p.output_per_mtok >= p.input_per_mtok
    assert (p.cache_read_mult, p.cache_write_mult, p.cache_write_1h_mult) == (
        pricing.CACHE_READ_MULT,
        pricing.CACHE_WRITE_5M_MULT,
        pricing.CACHE_WRITE_1H_MULT,
    )

"""``distil replay-model`` (distil/replay_model.py): offline tests.

Covers: tiered pricing, dry-run statistics (both arm costs), spend-cap refusal,
transcript turn parsing, the paired A/A'/B design, Wilson CI, and the minimum
A' sample size computation. All offline — no API calls, no real session or
transcript files needed.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import pytest

from distil import pricing, replay_model as rm
from distil.shadow import bootstrap_ci, wilson_ci


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
    """Cache reads at target model's cache rate, not base rate."""
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
# Minimum A' sample size
# --------------------------------------------------------------------------- #


def test_min_aa_n_default_matches_formula() -> None:
    """ceil((1.96 / (2 * 0.05))^2) = ceil(384.16) = 385."""
    n = rm._min_aa_n()
    assert n == math.ceil((1.96 / (2 * 0.05)) ** 2)
    # Wilson CI half-width at n=385, p=0.5 must be ≤ EQUIV_MARGIN
    assert rm._wilson_halfwidth(n) <= rm.EQUIV_MARGIN + 1e-9


def test_min_aa_n_wider_margin_smaller_n() -> None:
    n_tight = rm._min_aa_n(margin=0.05)
    n_loose = rm._min_aa_n(margin=0.10)
    assert n_loose < n_tight


def test_wilson_halfwidth_decreases_with_n() -> None:
    assert rm._wilson_halfwidth(100) > rm._wilson_halfwidth(400)


def test_wilson_halfwidth_zero_n_returns_one() -> None:
    assert rm._wilson_halfwidth(0) == 1.0


# --------------------------------------------------------------------------- #
# Dry-run report: both arms
# --------------------------------------------------------------------------- #


def test_dry_run_basic_arm_b_cheaper_than_original() -> None:
    records = [_record(model="claude-opus-5-5", inp=50_000 + i * 1000) for i in range(20)]
    report = rm.dry_run(records, "claude-haiku-5-5", days=7.0)
    assert report.n_sampled == 20
    assert report.target_model == "claude-haiku-5-5"
    assert "claude-opus-5-5" in report.model_counts
    assert report.n_long_tier == 0
    # Haiku-5-5 base rate is much cheaper than opus-5-5
    assert report.arm_b_usd < report.original_total_usd * 0.5


def test_dry_run_long_tier_counted() -> None:
    short = [_record(inp=50_000) for _ in range(5)]
    long_ = [_record(inp=150_000) for _ in range(5)]
    report = rm.dry_run(short + long_, "claude-haiku-5-5")
    assert report.n_long_tier == 5


def test_dry_run_n_aa_defaults_to_min_capped_at_n() -> None:
    """With only 10 records, n_aa cannot exceed 10 even if _min_aa_n() = 385."""
    records = [_record() for _ in range(10)]
    report = rm.dry_run(records, "claude-haiku-5-5")
    assert report.n_aa == 10
    assert report.n_aa_min == rm._min_aa_n()
    # CI warning is flagged (10 < 385)
    text = report.format()
    assert "WARNING" in text


def test_dry_run_n_aa_explicit_subsample() -> None:
    """Explicit n_aa < n_sampled: A' arm covers only that many records."""
    records = [_record(model="claude-opus-5-5", inp=10_000) for _ in range(50)]
    report = rm.dry_run(records, "claude-haiku-5-5", n_aa=20)
    assert report.n_aa == 20
    # A' cost ~ 20/50 of total original cost
    expected_aa_usd = report.original_total_usd * 20 / 50
    assert report.arm_aa_usd == pytest.approx(expected_aa_usd, rel=0.01)


def test_dry_run_n_aa_capped_at_n_sampled() -> None:
    """Passing n_aa > n_sampled is silently capped."""
    records = [_record() for _ in range(5)]
    report = rm.dry_run(records, "claude-haiku-5-5", n_aa=9999)
    assert report.n_aa == 5


def test_dry_run_arm_aa_cost_increases_with_subsample() -> None:
    records = [_record(model="claude-opus-5-5", inp=20_000) for _ in range(100)]
    report_small = rm.dry_run(records, "claude-haiku-5-5", n_aa=10)
    report_large = rm.dry_run(records, "claude-haiku-5-5", n_aa=50)
    assert report_large.arm_aa_usd > report_small.arm_aa_usd


def test_dry_run_format_contains_arm_labels() -> None:
    records = [_record(inp=200_000)] * 10
    report = rm.dry_run(records, "claude-haiku-5-5", days=30.0)
    text = report.format()
    assert "arm B" in text
    assert "arm A'" in text
    assert "total estimated spend" in text
    assert "--live" in text  # hint for running live
    assert "context note" in text


def test_dry_run_format_shows_both_arm_costs() -> None:
    records = [_record(model="claude-opus-5-5", inp=20_000) for _ in range(10)]
    report = rm.dry_run(records, "claude-haiku-5-5", n_aa=5)
    text = report.format()
    # Both $ amounts must appear
    assert f"${report.arm_b_usd:.4f}" in text
    assert f"${report.arm_aa_usd:.4f}" in text
    assert f"${report.arm_b_usd + report.arm_aa_usd:.4f}" in text


def test_dry_run_empty_records() -> None:
    report = rm.dry_run([], "claude-haiku-5-5")
    assert report.n_sampled == 0
    assert report.arm_b_usd == 0.0
    assert report.arm_aa_usd == 0.0
    assert report.original_total_usd == 0.0


def test_dry_run_unpriced_original_counted() -> None:
    records = [_record(model="claude-unknown-xyz-1-0")]
    report = rm.dry_run(records, "claude-haiku-5-5")
    assert report.n_unpriced_original == 1


def test_dry_run_unpriced_target_counted() -> None:
    records = [_record()]
    report = rm.dry_run(records, "claude-unknown-xyz-1-0")
    assert report.n_unpriced_target == 1


# --------------------------------------------------------------------------- #
# Wilson CI (used for the equivalence report)
# --------------------------------------------------------------------------- #


def test_wilson_ci_all_equivalent() -> None:
    lo, hi = wilson_ci(50, 50)
    assert lo > 0.90
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
        preflight=False,
        spend_cap=None,
        target="claude-haiku-5-5",
        n=10,
        n_aa=None,
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
        preflight=False,
        spend_cap=None,
        target="claude-haiku-5-5",
        n=10,
        n_aa=None,
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
    all_types = [
        b.get("type")
        for turn in turns
        for m in turn.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict)
    ]
    assert "thinking" not in all_types


def test_load_transcript_turns_respects_since(tmp_path: Path) -> None:
    rows = [
        _user_row("u1", None),
        {**_asst_row("a1", "u1", tool_name="bash"), "timestamp": "2020-01-01T00:00:00Z"},
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=time.time() - 86400, transcripts_dir=root)
    assert len(turns) == 0


# --------------------------------------------------------------------------- #
# CC-shaped fixtures: split assistant rows + interleaved tool_results
# --------------------------------------------------------------------------- #
#
# Real CC JSONL writes one row per content block of an assistant message.
# The same message.id spans multiple rows chained via parentUuid.


def _asst_row_split(
    uuid: str,
    parent: str,
    content: list[dict],
    model: str = "claude-opus-5-5",
    message_id: str = "msg_001",
    inp: int = 5000,
    out: int = 100,
) -> dict:
    """Assistant row with an explicit message.id for split-message testing."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "sess1",
        "timestamp": "2026-10-29T10:01:00Z",
        "message": {
            "id": message_id,
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


def _tool_result_row(uuid: str, parent: str, tool_use_id: str = "t1") -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-10-29T10:01:30Z",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}],
        },
    }


def test_split_assistant_rows_merged_into_one_message(tmp_path: Path) -> None:
    """Three fragments of the same message.id are merged into one API message."""
    rows = [
        _user_row("u0", None, "do it"),
        _asst_row_split("a0", "u0", [{"type": "thinking", "thinking": "..."}]),
        _asst_row_split("a1", "a0", [{"type": "text", "text": "on it"}]),
        _asst_row_split(
            "a2", "a1", [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]
        ),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root, rng=random.Random(0))
    assert len(turns) == 1
    t = turns[0]
    # Decision should come from the merged content (thinking + text + tool_use)
    # → first tool_use found → "tool:bash:..."
    assert t.recorded_decision.startswith("tool:")
    # Context: only the user message (thinking stripped → no preceding assistant message)
    assert t.messages[-1]["role"] == "user"
    # Only ONE turn sampled (a0 and a1 are non-last fragments, skipped)
    assert len(turns) == 1


def test_split_assistant_rows_only_last_fragment_sampled(tmp_path: Path) -> None:
    """Non-last fragments (a0, a1) produce no separate turns."""
    rows = [
        _user_row("u0", None, "go"),
        _asst_row_split("a0", "u0", [{"type": "thinking", "thinking": "..."}]),
        _asst_row_split("a1", "a0", [{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}]),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)
    # Only 1 turn: from a1 (the last fragment)
    assert len(turns) == 1
    assert turns[0].recorded_decision.startswith("tool:")


def test_split_tool_result_rows_merged(tmp_path: Path) -> None:
    """Two consecutive tool_result user rows are merged into one API message."""
    rows = [
        _user_row("u0", None, "run tools"),
        # Single assistant message with two parallel tool_uses
        _asst_row_split(
            "a0",
            "u0",
            [
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "read", "input": {}},
            ],
        ),
        # Two separate tool_result user rows (CC sometimes writes one per result)
        _tool_result_row("tr1", "a0", tool_use_id="t1"),
        _tool_result_row("tr2", "tr1", tool_use_id="t2"),
        # Final text response (decision row for the NEXT turn)
        _asst_row_split("a1", "tr2", [{"type": "text", "text": "done"}], message_id="msg_002"),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)
    # We expect the turn for a1 to have valid messages with merged tool_results
    text_turns = [t for t in turns if t.recorded_decision == "text"]
    assert len(text_turns) >= 1
    msg_list = text_turns[0].messages
    # The user message containing tool_results should have BOTH in one message
    tr_msgs = [
        m
        for m in msg_list
        if isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"] if isinstance(b, dict))
    ]
    assert len(tr_msgs) == 1  # merged into one
    assert len(tr_msgs[0]["content"]) == 2  # both tool_results


def test_thinking_only_fragment_skipped(tmp_path: Path) -> None:
    """A fragment with ONLY thinking contributes nothing to the context messages."""
    rows = [
        _user_row("u0", None, "think then act"),
        _asst_row_split("a0", "u0", [{"type": "thinking", "thinking": "hmm"}]),
        _asst_row_split(
            "a1", "a0", [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]
        ),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)
    assert len(turns) == 1
    # The context has no assistant messages (thinking stripped, nothing left)
    assert all(m["role"] == "user" for m in turns[0].messages)


def test_thinking_not_stripped_when_paired_with_text(tmp_path: Path) -> None:
    """Prior assistant message with thinking + text: thinking stripped, text kept.

    a0 (thinking+text, msg_000) and a1 (tool_use, msg_001) are BOTH last fragments,
    so two turns are sampled.  The turn from a1 carries a0's context with the text
    block intact but the thinking block removed.
    """
    rows = [
        _user_row("u0", None, "start"),
        # Previous turn: thinking + text (singleton for msg_000)
        _asst_row_split(
            "a0",
            "u0",
            [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "interim"}],
            message_id="msg_000",
        ),
        _user_row("u1", "a0", "continue"),
        _asst_row_split(
            "a1",
            "u1",
            [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
            message_id="msg_001",
        ),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)
    # Both a0 and a1 are last fragments → 2 turns
    assert len(turns) == 2
    # Find the tool-use turn (from a1)
    tool_turn = next(t for t in turns if t.recorded_decision.startswith("tool:"))
    # Its context must include an assistant message with text but NOT thinking
    asst_msgs = [m for m in tool_turn.messages if m["role"] == "assistant"]
    assert asst_msgs
    for m in asst_msgs:
        types = [b.get("type") for b in m["content"] if isinstance(b, dict)]
        assert "thinking" not in types
        assert "text" in types


def test_sibling_branch_tool_results_collected(tmp_path: Path) -> None:
    """CC writes tool_result for T1 as a sibling child of frag1, not in the main chain.

    Structure (Pattern A from CC JSONL):
      u0  ← user message
      a0  ← assistant frag1 [tool_use:t1], msg_id=msg_001, parent=u0
      a1  ← assistant frag2 [tool_use:t2], msg_id=msg_001, parent=a0   (same msg_id)
      tr2 ← user [tool_result:t2], parent=a1                            (in main chain)
      a2  ← assistant [text "done"],   msg_id=msg_002, parent=tr2       (decision row)

    tr1 (user [tool_result:t1], parent=a0) is a SIBLING of a1 — a child of a0 that
    is NOT on the main parentUuid chain.  The reconstruction must pick it up via the
    children_by_uuid index so that the API turn gets both results.
    """
    # tr1 is a sibling: its parentUuid=a0, but the main chain goes a0→a1→tr2→a2.
    tr1 = {
        "type": "user",
        "uuid": "tr1",
        "parentUuid": "a0",
        "timestamp": "2026-10-29T10:01:30Z",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok1"}],
        },
    }
    rows = [
        _user_row("u0", None, "run two tools"),
        _asst_row_split(
            "a0",
            "u0",
            [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
            message_id="msg_001",
        ),
        # sibling tool_result for t1 — NOT on the main parentUuid chain
        tr1,
        _asst_row_split(
            "a1",
            "a0",  # direct child of a0, same msg_id
            [{"type": "tool_use", "id": "t2", "name": "read", "input": {}}],
            message_id="msg_001",
        ),
        _tool_result_row("tr2", "a1", tool_use_id="t2"),  # in main chain
        _asst_row_split("a2", "tr2", [{"type": "text", "text": "done"}], message_id="msg_002"),
    ]
    root = _make_transcript(tmp_path, rows)
    turns = rm.load_transcript_turns(10, since=0.0, transcripts_dir=root)

    # The turn for a2 carries context with the merged assistant + merged user results.
    text_turns = [t for t in turns if t.recorded_decision == "text"]
    assert len(text_turns) >= 1
    msg_list = text_turns[0].messages

    # The assistant message in context should contain BOTH tool_uses.
    asst_msgs = [m for m in msg_list if m["role"] == "assistant"]
    assert asst_msgs
    all_asst_content = [b for m in asst_msgs for b in m.get("content", []) if isinstance(b, dict)]
    tool_use_ids = {b.get("id") for b in all_asst_content if b.get("type") == "tool_use"}
    assert tool_use_ids == {"t1", "t2"}, f"expected both tool_use ids, got {tool_use_ids}"

    # There should be EXACTLY ONE user message with tool_results, containing BOTH.
    tr_msgs = [
        m
        for m in msg_list
        if isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"] if isinstance(b, dict))
    ]
    assert len(tr_msgs) == 1, f"expected 1 tool_result user msg, got {len(tr_msgs)}"
    result_ids = {b.get("tool_use_id") for b in tr_msgs[0]["content"] if isinstance(b, dict)}
    assert result_ids == {"t1", "t2"}, f"expected both result ids, got {result_ids}"

    # validate_turn must pass (no missing-result errors).
    errs = rm.validate_turn(text_turns[0])
    assert errs == [], f"validate_turn errors: {errs}"


# --------------------------------------------------------------------------- #
# validate_turn
# --------------------------------------------------------------------------- #


def _make_turn(messages: list[dict], decision: str = "text") -> rm.TranscriptTurn:
    return rm.TranscriptTurn(
        messages=messages,
        recorded_decision=decision,
        model="claude-opus-5-5",
        input_tokens=1000,
        output_tokens=50,
        ts=time.time(),
    )


def test_validate_turn_valid_simple() -> None:
    turn = _make_turn(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "ok"},
        ]
    )
    assert rm.validate_turn(turn) == []


def test_validate_turn_valid_tool_pair() -> None:
    turn = _make_turn(
        [
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}],
            },
        ]
    )
    assert rm.validate_turn(turn) == []


def test_validate_turn_detects_ends_with_assistant() -> None:
    turn = _make_turn(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "bye"}]},
        ]
    )
    errs = rm.validate_turn(turn)
    assert any("user" in e for e in errs)


def test_validate_turn_detects_consecutive_roles() -> None:
    turn = _make_turn(
        [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": [{"type": "text", "text": "c"}]},
            {"role": "user", "content": "d"},
        ]
    )
    errs = rm.validate_turn(turn)
    assert any("consecutive" in e for e in errs)


def test_validate_turn_detects_thinking_block() -> None:
    turn = _make_turn(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "ok"},
                ],
            },
            {"role": "user", "content": "done"},
        ]
    )
    errs = rm.validate_turn(turn)
    assert any("thinking" in e for e in errs)


def test_validate_turn_detects_orphaned_tool_result() -> None:
    """Tool result that references a non-existent tool_use."""
    turn = _make_turn(
        [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t99", "content": "x"}],
            },
        ]
    )
    errs = rm.validate_turn(turn)
    assert any(
        "tool_result" in e and "orphan" in e.lower() or "not in" in e.lower() or "t99" in e
        for e in errs
    )


def test_validate_turn_detects_missing_tool_result() -> None:
    """Tool use with no following tool_result."""
    turn = _make_turn(
        [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
            },
            {"role": "user", "content": "next question"},
        ]
    )
    errs = rm.validate_turn(turn)
    assert any("tool_use" in e or "tool_result" in e for e in errs)


def test_validate_turn_empty_messages() -> None:
    turn = _make_turn([])
    errs = rm.validate_turn(turn)
    assert errs  # must have at least one error


# --------------------------------------------------------------------------- #
# Paired live result
# --------------------------------------------------------------------------- #


def test_paired_live_result_format_contains_key_fields() -> None:
    r = rm.PairedLiveResult(
        n_b=100,
        n_aa=50,
        n_eq_b=90,
        n_eq_aa=44,  # 44/50 — k must not exceed n
        diffs=[1] * 30 + [0] * 20,
        spend_b_usd=0.35,
        spend_aa_usd=4.20,
        target_model="claude-haiku-5-5",
        original_model_counts={"claude-opus-5-5": 50},
    )
    text = r.format()
    assert "90/100" in text
    assert "44/50" in text
    assert "claude-haiku-5-5" in text
    assert "$0.3500" in text
    assert "$4.2000" in text
    assert "context note" in text


def test_paired_live_result_non_inferiority_pass() -> None:
    """If diff CI lower bound ≥ −0.05 the verdict is PASS."""
    # B matches A on 95/100 turns; A' matches A on 45/50 (90%) — same rate
    # paired diffs: mostly 0 (both match) or +1 (B matches, A' doesn't)
    diffs = [1] * 5 + [0] * 45  # n=50 paired: B better on 5, tied on 45 → mean +0.10
    r = rm.PairedLiveResult(
        n_b=100,
        n_aa=50,
        n_eq_b=95,
        n_eq_aa=45,  # 45/50 (90%)
        diffs=diffs,
        spend_b_usd=0.10,
        spend_aa_usd=2.00,
        target_model="claude-haiku-5-5",
        original_model_counts={},
    )
    text = r.format()
    assert "PASS" in text


def test_paired_live_result_non_inferiority_fail() -> None:
    """If diff CI lower bound < −0.05 the verdict is FAIL."""
    # B is worse than A' on most paired turns
    diffs = [-1] * 40 + [0] * 10  # mean = -0.80
    r = rm.PairedLiveResult(
        n_b=100,
        n_aa=50,
        n_eq_b=20,
        n_eq_aa=45,  # 45/50 (90%)
        diffs=diffs,
        spend_b_usd=0.10,
        spend_aa_usd=2.00,
        target_model="claude-haiku-5-5",
        original_model_counts={},
    )
    text = r.format()
    assert "FAIL" in text


def test_paired_live_result_no_diffs_skips_paired_section() -> None:
    """When A' arm is empty (diffs=[]), the paired section is omitted."""
    r = rm.PairedLiveResult(
        n_b=10,
        n_aa=0,
        n_eq_b=8,
        n_eq_aa=0,
        diffs=[],
        spend_b_usd=0.05,
        spend_aa_usd=0.0,
        target_model="claude-haiku-5-5",
        original_model_counts={},
    )
    text = r.format()
    assert "paired diff" not in text


def test_paired_live_result_equivalence_property() -> None:
    """The .equivalence property returns a shadow.Equivalence object."""
    r = rm.PairedLiveResult(
        n_b=60,
        n_aa=30,
        n_eq_b=55,
        n_eq_aa=28,
        diffs=[0] * 30,
        spend_b_usd=0.10,
        spend_aa_usd=1.0,
        target_model="claude-haiku-5-5",
        original_model_counts={},
    )
    eq = r.equivalence
    assert eq.n_ab == 60
    assert eq.n_aa == 30
    assert eq.p_ab == pytest.approx(55 / 60, rel=1e-9)
    assert eq.p_aa == pytest.approx(28 / 30, rel=1e-9)


# --------------------------------------------------------------------------- #
# Bootstrap CI (used for the paired diff)
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_symmetric_around_zero() -> None:
    """For a zero-mean diff vector the CI should straddle 0."""
    diffs = [1, -1] * 50  # alternating, mean=0
    lo, hi = bootstrap_ci(diffs, seed=42)
    assert lo < 0 < hi


def test_bootstrap_ci_positive_mean_has_positive_lo() -> None:
    """For consistently positive diffs, the lower bound should be positive."""
    diffs = [1] * 100
    lo, hi = bootstrap_ci(diffs, seed=42)
    assert lo > 0


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


# --------------------------------------------------------------------------- #
# _segment_cost
# --------------------------------------------------------------------------- #


def test_segment_cost_single_turn_is_cache_write() -> None:
    """A single-turn segment should pay cache-write (not read) for the full context."""
    model = "claude-opus-5-5"
    ctx = 50_000
    cost_seg = rm._segment_cost([ctx], model, output_per_turn=0)

    # p.input is already dollars-per-token (not per-MTok); no / 1_000_000 needed.
    p = rm._price(model)
    assert p is not None
    w = p.input * p.cache_write_1h_mult
    expected = ctx * w
    assert cost_seg == pytest.approx(expected, rel=1e-6)


def test_segment_cost_subsequent_turns_read_prefix() -> None:
    """Turn k>0 should read the previous context at cache_read_mult, not write it."""
    model = "claude-opus-5-5"
    ctx0 = 50_000
    ctx1 = 52_000  # adds 2_000 new tokens

    cost_seg = rm._segment_cost([ctx0, ctx1], model, output_per_turn=0)

    p = rm._price(model)
    assert p is not None
    w = p.input * p.cache_write_1h_mult
    r = p.input * p.cache_read_mult
    expected_turn0 = ctx0 * w
    expected_turn1 = ctx0 * r + 2_000 * w
    assert cost_seg == pytest.approx(expected_turn0 + expected_turn1, rel=1e-6)


def test_segment_cost_cache_reads_cheaper_than_writes() -> None:
    """Multi-turn segment must be cheaper than sending each turn fresh (no caching)."""
    model = "claude-opus-5-5"
    ctx_sizes = [100_000, 102_000, 104_000]
    cost_seg = rm._segment_cost(ctx_sizes, model, output_per_turn=0)

    # Uncached cost: every turn pays full fresh-input price (p.input is $/token).
    p = rm._price(model)
    assert p is not None
    cost_uncached = sum(c * p.input for c in ctx_sizes)
    assert cost_seg < cost_uncached


def test_segment_cost_haiku_long_tier_applied() -> None:
    """Haiku 5.5 turns > 100k tokens should apply the 5x long-tier multiplier."""
    model = "claude-haiku-5-5"
    ctx = 150_000  # above 100k threshold

    cost_with_tier = rm._segment_cost([ctx], model, output_per_turn=0)

    p = rm._price(model)
    assert p is not None
    w = p.input * p.cache_write_1h_mult  # dollars per token
    base = ctx * w
    # Long tier applies 5x
    assert cost_with_tier == pytest.approx(5.0 * base, rel=1e-6)


def test_segment_cost_empty_returns_zero() -> None:
    assert rm._segment_cost([], "claude-opus-5-5") == 0.0


# --------------------------------------------------------------------------- #
# cluster_bootstrap_ci
# --------------------------------------------------------------------------- #


def test_cluster_bootstrap_ci_zero_mean_straddles_zero() -> None:
    """Zero-mean diffs should produce a CI containing 0."""
    # Mix of +1 and -1 clusters so the overall mean = 0 but bootstrap has variance.
    cluster_diffs = [[1]] * 20 + [[-1]] * 20
    rng = random.Random(42)
    lo, hi = rm.cluster_bootstrap_ci(cluster_diffs, n_boot=2_000, rng=rng)
    assert lo <= 0 <= hi


def test_cluster_bootstrap_ci_positive_mean_has_positive_lo() -> None:
    """All-positive diffs: lower bound should be positive."""
    cluster_diffs = [[1, 1, 1]] * 40
    rng = random.Random(42)
    lo, hi = rm.cluster_bootstrap_ci(cluster_diffs, n_boot=500, rng=rng)
    assert lo > 0


def test_cluster_bootstrap_ci_empty_returns_wide_interval() -> None:
    lo, hi = rm.cluster_bootstrap_ci([])
    assert lo == -1.0 and hi == 1.0


def test_cluster_bootstrap_ci_single_cluster() -> None:
    """A single cluster gives a CI anchored at the cluster mean."""
    rng = random.Random(0)
    lo, hi = rm.cluster_bootstrap_ci([[0, 0, 0]], n_boot=200, rng=rng)
    # All bootstraps sample [[0,0,0]]; mean is always 0.
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _add_cache_breakpoints
# --------------------------------------------------------------------------- #


def test_add_cache_breakpoints_tags_last_message() -> None:
    """Tags cache_control on the last block of the last message."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": "next"},
    ]
    out = rm._add_cache_breakpoints(messages, ttl="1h")
    last = out[-1]
    content = last["content"]
    assert isinstance(content, list)
    assert "cache_control" in content[-1]
    assert content[-1]["cache_control"]["ttl"] == "1h"


def test_add_cache_breakpoints_converts_string_content() -> None:
    """String content on the last message is converted to a list block."""
    messages = [{"role": "user", "content": "hello"}]
    out = rm._add_cache_breakpoints(messages)
    assert isinstance(out[0]["content"], list)
    assert "cache_control" in out[0]["content"][-1]


def test_add_cache_breakpoints_does_not_mutate_input() -> None:
    """Original messages list and dicts are not mutated."""
    msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    messages = [msg]
    rm._add_cache_breakpoints(messages, ttl="1h")
    assert "cache_control" not in messages[0]["content"][0]


def test_add_cache_breakpoints_seam_index_also_tagged() -> None:
    """seam_index message also receives a cache breakpoint."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
    ]
    out = rm._add_cache_breakpoints(messages, seam_index=0)
    # First message (seam) should be tagged
    assert "cache_control" in out[0]["content"][-1]
    # Last message should also be tagged
    assert "cache_control" in out[-1]["content"][-1]


def test_add_cache_breakpoints_empty_content_skips() -> None:
    """Messages with empty content are not touched."""
    messages = [
        {"role": "user", "content": []},
    ]
    out = rm._add_cache_breakpoints(messages)
    # No crash; content remains empty
    assert out[0]["content"] == []


# --------------------------------------------------------------------------- #
# load_transcript_segments
# --------------------------------------------------------------------------- #


def _make_jsonl_file(
    path: Path,
    session_id: str,
    n_turns: int,
    ctx_tokens: int = 10_000,
    ts: float | None = None,
) -> None:
    """Write a minimal synthetic CC-style JSONL transcript."""
    import json as _json

    _ts = ts or time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n_turns):
            # Each turn: one assistant row (decision), one user row (tool result)
            # We write simplified rows that _sample_file can parse.
            # A minimal user message (the "decision trigger")
            user_msg: dict = {
                "type": "user",
                "uuid": f"{session_id}-u{i}",
                "parentUuid": f"{session_id}-a{i - 1}" if i > 0 else None,
                "sessionId": session_id,
                "timestamp": _ts + i * 10,
                "message": {"role": "user", "content": "do something"},
            }
            # An assistant message (the decision)
            asst_msg: dict = {
                "type": "assistant",
                "uuid": f"{session_id}-a{i}",
                "parentUuid": f"{session_id}-u{i}",
                "sessionId": session_id,
                "timestamp": _ts + i * 10 + 5,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"response {i}"}],
                    "model": "claude-opus-5-5",
                    "usage": {
                        "input_tokens": ctx_tokens,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            }
            f.write(_json.dumps(user_msg) + "\n")
            f.write(_json.dumps(asst_msg) + "\n")


def test_load_transcript_segments_returns_correct_shape(tmp_path: Path) -> None:
    """Should return K segments of M turns each when data is available."""
    session_id = "sess-abc"
    jsonl = tmp_path / "proj" / "session.jsonl"
    _make_jsonl_file(jsonl, session_id, n_turns=20, ctx_tokens=5_000)

    segs, cov = rm.load_transcript_segments(
        2,
        5,
        0.0,
        transcripts_dir=tmp_path,
        rng=random.Random(0),
    )
    assert len(segs) <= 2
    for seg in segs:
        assert len(seg) == 5


def test_load_transcript_segments_coverage_stats(tmp_path: Path) -> None:
    """Coverage stats should reflect max_context_tokens filtering."""
    session_id = "sess-cov"
    jsonl = tmp_path / "proj" / "session.jsonl"
    # 20 turns at 200k tokens each (long-tier)
    _make_jsonl_file(jsonl, session_id, n_turns=20, ctx_tokens=200_000)

    _, cov = rm.load_transcript_segments(
        1,
        3,
        0.0,
        max_context_tokens=100_000,
        transcripts_dir=tmp_path,
    )
    # All turns exceed 100k → scoped_turns = 0
    assert cov["scoped_turns"] == 0
    assert cov["request_coverage_frac"] == 0.0


def test_load_transcript_segments_no_eligible_sessions(tmp_path: Path) -> None:
    """When sessions have fewer turns than turns_per_segment, return empty."""
    jsonl = tmp_path / "proj" / "short.jsonl"
    _make_jsonl_file(jsonl, "sess-short", n_turns=3)

    segs, _ = rm.load_transcript_segments(
        5,
        10,  # need 10 turns, only 3 available
        0.0,
        transcripts_dir=tmp_path,
    )
    assert segs == []


# --------------------------------------------------------------------------- #
# _dry_run_plans
# --------------------------------------------------------------------------- #


def test_dry_run_plans_returns_three_plans() -> None:
    """_dry_run_plans should return a 3-tuple of ReplayPlan objects."""
    records = [_record(inp=50_000, out=200)] * 200
    target = "claude-haiku-5-5"
    a, b, c = rm._dry_run_plans(records, target)
    assert isinstance(a, rm.ReplayPlan)
    assert isinstance(b, rm.ReplayPlan)
    assert isinstance(c, rm.ReplayPlan)


def test_dry_run_plans_a_has_effective_n_385_or_more() -> None:
    """Plan (a) must target effective_n >= 385."""
    records = [_record(inp=50_000, out=200)] * 300
    _, _, _ = rm._dry_run_plans(records, "claude-haiku-5-5")
    a, _, _ = rm._dry_run_plans(records, "claude-haiku-5-5")
    assert a.effective_n >= 385 or a.n_turns == 0  # 0 only if no records


def test_dry_run_plans_b_within_50_usd() -> None:
    """Plan (b) total cost must be ≤ $50."""
    records = [_record(inp=100_000, out=500)] * 500
    _, b, _ = rm._dry_run_plans(records, "claude-opus-5-5")
    assert b.total_usd <= 50.0 + 1e-6  # small float tolerance


def test_dry_run_plans_c_within_50_usd() -> None:
    """Plan (c) total cost must be ≤ $50."""
    records = [_record(inp=150_000, out=200)] * 300
    _, _, c = rm._dry_run_plans(records, "claude-opus-5-5")
    assert c.total_usd <= 50.0 + 1e-6


def test_dry_run_format_includes_all_three_plans() -> None:
    """DryRunReport.format() should mention plans (a), (b), (c) when plans are set."""
    records = [_record(inp=40_000, out=300)] * 100
    report = rm.dry_run(records, "claude-haiku-5-5", n_aa=30, days=30)
    text = report.format()
    assert "(a)" in text
    assert "(b)" in text
    assert "(c)" in text
    # Should mention effective n and DEFF
    assert "DEFF" in text or "effective n" in text.lower()


def test_dry_run_format_includes_coverage_note() -> None:
    """Plan (c) section should mention traffic coverage fraction."""
    records = [_record(inp=150_000, out=500)] * 100
    report = rm.dry_run(records, "claude-haiku-5-5", n_aa=30, days=30)
    text = report.format()
    # Coverage fraction should appear somewhere in the (c) block
    assert "%" in text or "coverage" in text.lower()


# --------------------------------------------------------------------------- #
# CLI wiring (offline: no provider call is ever made here)
# --------------------------------------------------------------------------- #


def _parse(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser()
    rm.register(parser.add_subparsers())
    return parser.parse_args(["replay-model", *argv])


def _turn(ok: bool = True) -> rm.TranscriptTurn:
    msgs = [{"role": "user", "content": "hi"}] if ok else []
    return rm.TranscriptTurn(msgs, "text", "claude-opus-5-5", 12_000, 300, time.time())


def test_register_parses_every_flag() -> None:
    a = _parse(
        ["--target", "m", "--n", "5", "--n-aa", "3", "--days", "2", "--preflight", "--json"]
        + ["--sessions", "4", "--turns-per-segment", "2", "--max-context-tokens", "9000"]
    )
    assert (a.target, a.n, a.n_aa, a.days, a.sessions) == ("m", 5, 3, 2.0, 4)
    assert a.preflight and a.json and not a.live and a.func is rm.cmd_replay_model
    assert (a.turns_per_segment, a.max_context_tokens, a.spend_cap) == (2, 9000, None)


@pytest.mark.parametrize("as_json", [False, True])
def test_cmd_dry_run_prints_a_report(monkeypatch, capsys, as_json) -> None:
    monkeypatch.setattr(rm, "load_records", lambda n, since, rng: [_record() for _ in range(5)])
    argv = ["--n", "5"] + (["--json"] if as_json else [])
    assert rm.cmd_replay_model(_parse(argv)) == 0
    out = capsys.readouterr().out
    if as_json:
        assert "arm_b_usd" in json.loads(out)
    else:
        assert out.strip()


@pytest.mark.parametrize("as_json", [False, True])
def test_cmd_preflight_without_a_key_skips_count_tokens(monkeypatch, capsys, as_json) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        rm, "load_transcript_turns", lambda n, since, rng: [_turn(), _turn(), _turn(False)]
    )
    argv = ["--preflight", "--n", "3"] + (["--json"] if as_json else [])
    assert rm.cmd_replay_model(_parse(argv)) == 0
    cap = capsys.readouterr()
    assert "2/3 pass" in cap.err and "skipping count_tokens" in cap.err
    assert ("arm_b_usd" in json.loads(cap.out)) if as_json else ("--live" in cap.out)


def test_cmd_preflight_with_sessions_and_nothing_valid(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cov = {"request_coverage_frac": 0.5, "token_coverage_frac": 0.25}
    monkeypatch.setattr(rm, "load_transcript_segments", lambda *a, **k: ([[_turn(False)]], cov))
    argv = ["--preflight", "--sessions", "1", "--max-context-tokens", "1000"]
    assert rm.cmd_replay_model(_parse(argv)) == 1
    cap = capsys.readouterr()
    assert "covers 50.0% of requests" in cap.out
    assert "all turns failed structural validation" in cap.err


def test_cmd_preflight_with_no_turns_returns_1(monkeypatch, capsys) -> None:
    monkeypatch.setattr(rm, "load_transcript_turns", lambda n, since, rng: [])
    assert rm.cmd_replay_model(_parse(["--preflight"])) == 1
    assert "no usable transcript turns" in capsys.readouterr().err


def test_cmd_preflight_count_tokens_gate_rejects_everything(monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(rm, "load_transcript_turns", lambda n, since, rng: [_turn()])
    monkeypatch.setattr(rm, "preflight_count_tokens", lambda t, target, api_key: ([], 0, len(t)))
    assert rm.cmd_replay_model(_parse(["--preflight"])) == 1
    err = capsys.readouterr().err
    assert "0/1 pass" in err and "0 turns passed both gates" in err

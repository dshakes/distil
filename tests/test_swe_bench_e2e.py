"""Tests for the SWE-bench Verified end-to-end eval (Phase 5 / E7).

Covered:
* deterministic seed-1729 instance sampling (order-independent, reproducible);
* the compression layer for conditions B (distil trunc@500) and C (llmlingua-2):
  the block selector compresses only agent-read context (user/tool text/tool_result
  >= MIN_CHARS) and never system or assistant reasoning, and trunc@500 matches
  distil's ``conformal._truncate_level(500)`` byte-for-byte;
* Wilson 95% CI used for reported pass@1.

These run with no network and no Docker (the heavy harness/agent paths are exercised
by the live eval, not unit tests).
"""

from __future__ import annotations

from benchmarks.swe_bench_e2e import sample as smp
from benchmarks.swe_bench_e2e.compress_proxy import (
    MIN_CHARS,
    CompressStats,
    compress_body,
    trunc_500,
)
from benchmarks.swe_bench_e2e.stats import wilson_ci


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _fake_rows(k: int) -> list[dict]:
    # ids deliberately unsorted to prove the sampler sorts before drawing.
    return [{"instance_id": f"repo__proj-{i:04d}"} for i in reversed(range(k))]


def test_sample_is_deterministic_and_order_independent():
    rows = _fake_rows(500)
    a = smp.sample_instance_ids(rows, 50, seed=1729)
    b = smp.sample_instance_ids(list(reversed(rows)), 50, seed=1729)
    assert a == b  # shuffling the input pool must not change the draw
    assert len(a) == len(set(a)) == 50


def test_sample_changes_with_seed():
    rows = _fake_rows(500)
    assert smp.sample_instance_ids(rows, 50, seed=1729) != smp.sample_instance_ids(
        rows, 50, seed=1
    )


# --------------------------------------------------------------------------- #
# trunc@500 == distil's certifying operating point
# --------------------------------------------------------------------------- #
def test_trunc500_matches_conformal_truncate_level():
    from distil.conformal import _truncate_level
    from distil.trajectory import Block, Kind, Stability

    text = "x" * 4000
    strat = _truncate_level(500)
    blk = Block(id="b0", kind=Kind.TOOL_OUTPUT, text=text, stability=Stability.VOLATILE)
    (out_block,) = strat([blk], 0)
    assert trunc_500(text) == out_block.text == text[:500]
    assert len(trunc_500(text)) == 500


def test_trunc500_noop_below_limit():
    assert trunc_500("short") == "short"


# --------------------------------------------------------------------------- #
# Block selection: only agent-read context, never reasoning/instructions
# --------------------------------------------------------------------------- #
def _big(tag: str) -> str:
    return f"{tag}:" + "A" * (MIN_CHARS + 100)


def test_compresses_user_context_not_system_or_assistant():
    big_user = _big("file")
    big_assistant = _big("reasoning")
    body = {
        "model": "claude-sonnet-4-6",
        "system": _big("instructions"),  # system must be untouched
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": big_user}]},
            {"role": "assistant", "content": [{"type": "text", "text": big_assistant}]},
        ],
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats)
    # user file content compressed to 500 chars
    assert out["messages"][0]["content"][0]["text"] == big_user[:500]
    # assistant reasoning left intact
    assert out["messages"][1]["content"][0]["text"] == big_assistant
    # system string left intact (compressor only touches the messages array)
    assert out["system"] == body["system"]
    assert stats.blocks_compressed == 1


def test_small_blocks_untouched():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "tiny instruction"}]}
        ]
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats)
    assert out["messages"][0]["content"][0]["text"] == "tiny instruction"
    assert stats.blocks_compressed == 0


def test_tool_result_string_and_list_compressed():
    payload = _big("cmd-output")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": payload},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": [{"type": "text", "text": payload}],
                    },
                ],
            },
        ],
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats)
    c = out["messages"][0]["content"]
    assert c[0]["content"] == payload[:500]
    assert c[1]["content"][0]["text"] == payload[:500]
    assert stats.blocks_compressed == 2


def test_full_condition_is_passthrough_but_counts():
    big = _big("file")
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": big}]}]}
    stats = CompressStats()
    out = compress_body(body, None, stats)
    assert out["messages"][0]["content"][0]["text"] == big  # unchanged
    assert stats.blocks_seen == 1
    assert stats.blocks_compressed == 0
    assert stats.tokens_before == stats.tokens_after  # no compression accounted


def test_problem_statement_is_protected_from_compression():
    # The problem statement is the task, not file content — it must pass through verbatim
    # even though it is a large user block, while a separate file-content block compresses.
    problem = (
        "ModelBackend.authenticate() should not query the DB when username is None. "
        * 20
    )
    file_block = _big("source-file")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": problem + " extra wrapping"}],
            },
            {"role": "user", "content": [{"type": "text", "text": file_block}]},
        ]
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats, protect=problem)
    # problem statement untouched (protected), file content truncated to 500 chars
    assert out["messages"][0]["content"][0]["text"] == problem + " extra wrapping"
    assert out["messages"][1]["content"][0]["text"] == file_block[:500]
    assert stats.blocks_protected == 1
    assert stats.blocks_compressed == 1


def test_compress_body_does_not_mutate_input():
    big = _big("file")
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": big}]}]}
    compress_body(body, trunc_500, CompressStats())
    assert body["messages"][0]["content"][0]["text"] == big  # original intact


# --------------------------------------------------------------------------- #
# Wilson CI
# --------------------------------------------------------------------------- #
def test_wilson_ci_bounds():
    import pytest

    lo, hi = wilson_ci(10, 50)
    assert 0.0 <= lo < 0.20 < hi <= 1.0
    # all-success and all-fail stay in [0,1] (upper at phat=1 is 1.0 up to float error)
    assert wilson_ci(0, 30)[0] == 0.0
    assert wilson_ci(30, 30)[1] == pytest.approx(1.0)


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 0.0)

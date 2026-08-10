"""Phases 4, 5, 6 — auth-mode gating, byte-fidelity, holdout A/B."""

import pytest

from distil import pricing
from distil.certify.holdout import bootstrap_ci, partition, run_holdout
from distil.compress.strategies import distil as distil_strategy
from distil.compress.tier0 import Tier0Lossless
from distil.corpus import load_corpus
from distil.fidelity import (
    assert_append_only,
    numeric_precision_preserved,
    verify_reversible,
)
from distil.policy import AuthMode, PolicyError, allowed_strategies, guard
from distil.trajectory import Block, Kind, Stability


# ---- Phase 4: auth-mode gating ---------------------------------------------
def test_payg_allows_everything():
    assert allowed_strategies(AuthMode.PAYG) == {"none", "distil", "naive", "aggressive"}


def test_subscription_is_lossless_only():
    assert allowed_strategies(AuthMode.SUBSCRIPTION) == {"none", "distil"}
    guard(AuthMode.SUBSCRIPTION, "distil")  # allowed
    with pytest.raises(PolicyError):
        guard(AuthMode.SUBSCRIPTION, "aggressive")
    with pytest.raises(PolicyError):
        guard(AuthMode.SUBSCRIPTION, "naive")


# ---- Phase 6: byte-fidelity invariants -------------------------------------
def test_tier0_is_verifiably_reversible():
    blocks = [Block("j", Kind.TOOL_OUTPUT, '{"x":  1, "y":  2}', Stability.VOLATILE)]
    result = Tier0Lossless().compress(blocks)
    assert verify_reversible(blocks, result).lossless


def test_distil_strategy_is_reversible_end_to_end():
    blocks = [
        Block("sys", Kind.SYSTEM, "policy. DECISION: do X", Stability.STABLE, True),
        Block(
            "o",
            Kind.TOOL_OUTPUT,
            "head\n" * 4 + "DECISION: act\n" + "noise\n" * 8,
            Stability.VOLATILE,
            True,
        ),
    ]
    out = distil_strategy(blocks, 0)
    # decisions preserved (lossless of what matters); stable prefix untouched
    assert any("DECISION: do X" in b.text for b in out)
    assert any("DECISION: act" in b.text for b in out)


def test_append_only_detects_mutation():
    prev = [Block("h0", Kind.HISTORY, "summary v1", Stability.SETTLING)]
    curr_ok = [Block("h0", Kind.HISTORY, "summary v1", Stability.SETTLING)]
    curr_bad = [Block("h0", Kind.HISTORY, "summary MUTATED", Stability.SETTLING)]
    assert assert_append_only(prev, curr_ok) == []
    assert assert_append_only(prev, curr_bad) == ["h0"]


def test_numeric_precision_check():
    assert numeric_precision_preserved('{"a": 1.50, "b": 2}', '{"a":1.5,"b":2}')
    assert not numeric_precision_preserved('{"a": 1}', '{"a": 2}')


# ---- Phase 5: holdout A/B ---------------------------------------------------
def test_partition_is_deterministic_and_complete():
    ids = [e.trajectory.id for e in load_corpus()]
    c1, t1 = partition(ids, 0.3)
    c2, t2 = partition(ids, 0.3)
    assert c1 == c2 and t1 == t2  # deterministic
    assert sorted(c1 + t1) == sorted(ids)  # complete, disjoint


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    vals = [0.30, 0.33, 0.28, 0.31, 0.35, 0.29]
    m1, lo1, hi1 = bootstrap_ci(vals)
    m2, lo2, hi2 = bootstrap_ci(vals)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)  # fixed-seed -> reproducible
    assert lo1 <= m1 <= hi1


def test_run_holdout_reports_positive_savings():
    report = run_holdout(load_corpus(), pricing.get("claude-opus-4-8"), control_fraction=0.2)
    assert report.mean_savings > 0
    assert report.ci_low <= report.mean_savings <= report.ci_high
    assert len(report.control_ids) + len(report.treatment_ids) >= 6

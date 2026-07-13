"""Phase-2 query-aware salience: features, model, gated integration, content-free flywheel,
training + certification. The end-to-end test is the money test — a promoted proximity model
pins a *semantically* relevant line that phase 1 (lexical only) would fold."""

import random

import pytest

from distil.codec.features import FEATURE_NAMES
from distil.compress import query_relevance as qr
from distil.compress.query_relevance import (
    QUERY_FEATURE_NAMES,
    QueryRelevanceModel,
    featurize_query,
)

_BASE = len(FEATURE_NAMES)
_W = len(QUERY_FEATURE_NAMES)


# --- features -----------------------------------------------------------------------


def test_query_features_width_and_neutrality():
    assert _W == _BASE + 4
    # empty query → neutral query block (no signal, maximally far from any hit)
    v = featurize_query("some line", "tool_output", frozenset())
    assert len(v) == _W and v[_BASE:] == [0.0, 0.0, 0.0, 1.0]
    # a lexical hit sets overlap + in_line + selective
    v2 = featurize_query("max_retries config", "tool_output", frozenset({"config"}))
    assert v2[_BASE] > 0 and v2[_BASE + 2] == 1.0


# --- model + persistence ------------------------------------------------------------


def test_model_relevant_lines_and_empty_query():
    w = [0.0] * _BASE + [3.0, 3.0, 0.0, 0.0]
    w[0] = -1.0
    m = QueryRelevanceModel(w)
    lines = ["alpha config value", "unrelated noise", "beta config here"]
    kept = m.relevant_lines(lines, frozenset({"config"}), "tool_output")
    assert 0 in kept and 2 in kept  # both config lines
    assert m.relevant_lines(lines, frozenset(), "tool_output") == set()  # no query → nothing


def test_model_save_load_roundtrip(tmp_path):
    w = [0.1 * i for i in range(_W)]
    p = tmp_path / "qw.json"
    QueryRelevanceModel(w).save(p)
    assert QueryRelevanceModel.load(p)._weights == w
    assert QueryRelevanceModel.load_or_none(tmp_path / "missing.json") is None


def test_bad_width_rejected():
    with pytest.raises(ValueError):
        QueryRelevanceModel([0.0] * (_W - 1))


# --- gated integration into tier1.digest --------------------------------------------


def test_digest_no_weights_is_exactly_phase1(monkeypatch):
    from distil.compress import tier1

    monkeypatch.setattr(qr, "get_model", lambda: None)
    text = "\n".join(f"line {i} content here" for i in range(20))
    out, _ = tier1.digest(text, intent=frozenset({"content"}))
    # sanity: it produced a digest and didn't crash on the None model path
    assert "handle=" in out


def test_digest_widens_keep_with_promoted_model(monkeypatch):
    from distil.compress import tier1

    # A block where the semantic answer ('max_attempts = 5') sits next to the lexical hit
    # ('retry') but shares no term with the query — phase 1 folds it, phase 2 pins it.
    lines = (
        ["preamble noise line"] * 4
        + ["retry policy section", "  max_attempts = 5", "  backoff = exp"]
        + ["trailing noise line"] * 6
    )
    text = "\n".join(lines)
    q = frozenset({"retry", "limit"})

    monkeypatch.setattr(qr, "get_model", lambda: None)
    phase1, _ = tier1.digest(text, intent=q)

    # proximity-keyed model: fire on lines near a lexical hit (low q_dist_to_hit). bias +1
    # clears threshold for the hit's neighbor (dist≈0.08) but not far lines (dist≈1).
    w = [0.0] * _BASE + [0.0, 0.0, 0.0, -4.0]
    w[0] = 1.0
    monkeypatch.setattr(qr, "get_model", lambda: QueryRelevanceModel(w))
    phase2, _ = tier1.digest(text, intent=q)

    assert "max_attempts = 5" in phase2, "phase 2 should pin the semantic answer line"
    assert "max_attempts = 5" not in phase1, "phase 1 (lexical) folds it — the gap phase 2 closes"


# --- content-free flywheel ----------------------------------------------------------


def test_flywheel_content_free_and_gated(tmp_path, monkeypatch):
    from distil import query_flywheel as fw

    p = tmp_path / "fw.jsonl"
    lines = ["retry section:", "  max_attempts = 5", "  secret_token = abc123"]
    # disabled → no-op, no file
    fw.disable()
    fw.maybe_record("h1", frozenset({"retry"}), lines, "tool_output", {1, 2}, path=p)
    assert not p.exists()
    # enabled → records numeric features only, never the source text
    monkeypatch.setattr(fw, "_enabled", True)
    monkeypatch.setattr(fw, "_sample_rate", 1.0)
    fw.maybe_record("h1", frozenset({"retry"}), lines, "tool_output", {1, 2}, path=p)
    blob = p.read_text()
    assert "max_attempts" not in blob and "secret_token" not in blob and "abc123" not in blob
    fw.disable()


def test_flywheel_join_labels(tmp_path, monkeypatch):
    from distil import query_flywheel as fw

    fwp, sig = tmp_path / "fw.jsonl", tmp_path / "sig.jsonl"
    monkeypatch.setattr(fw, "_enabled", True)
    monkeypatch.setattr(fw, "_sample_rate", 1.0)
    lines = ["a config line", "b noise", "c config line"]
    fw.maybe_record("expanded1", frozenset({"config"}), lines, "tool_output", {0, 1, 2}, path=fwp)
    fw.maybe_record("never2", frozenset({"config"}), lines, "tool_output", {0, 1, 2}, path=fwp)
    sig.write_text('{"handle": "expanded1", "recovered_chars": 30, "ts": 1.0}\n')
    labels = fw.load_labels(flywheel_path=fwp, signal_path=sig)
    assert sum(1 for _x, y in labels if y == 1.0) == 3  # expanded block's lines → positive
    assert sum(1 for _x, y in labels if y == 0.0) == 3  # never-expanded → weak negative
    fw.disable()


# --- training + certification -------------------------------------------------------


def test_training_beats_baseline_and_promotes(tmp_path):
    from distil import query_train

    rng = random.Random(0)
    samples = []
    for _ in range(600):
        x = [0.0] * _W
        x[0] = 1.0
        if rng.random() < 0.5:  # positive: semantically near a hit, NOT lexical
            x[_BASE + 3] = rng.uniform(0.0, 0.2)  # low q_dist_to_hit
            samples.append((x, 1.0))
        else:  # negative: far
            x[_BASE + 3] = rng.uniform(0.6, 1.0)
            samples.append((x, 0.0))
    rep = query_train.train_query_relevance(samples)
    assert rep["recall"] > rep["baseline_recall"], "learned recall must beat the lexical baseline"

    wp = tmp_path / "qw.json"
    r = query_train.certify_and_promote(samples, weights_path=wp, min_samples=200)
    assert r["promoted"] and QueryRelevanceModel.load_or_none(wp) is not None


def test_gate_rejects_too_few_labels(tmp_path):
    from distil import query_train

    r = query_train.certify_and_promote([([0.0] * _W, 1.0)] * 10, weights_path=tmp_path / "qw.json")
    assert not r["promoted"] and "too few" in r["reason"]

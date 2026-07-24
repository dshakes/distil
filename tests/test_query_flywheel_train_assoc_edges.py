"""Phase-2 query-aware salience — edge/error paths of query_flywheel, query_train, and
query_assoc not exercised by test_query_relevance.py / test_semantic_bridge.py: sample-rate
clamping, deterministic sampling boundaries, fail-open on malformed/unreadable logs, and the
training gate's reject branches (empty input, bad width, sub-floor recall/precision)."""

import json

import pytest

from distil.codec.features import FEATURE_NAMES
from distil.compress.query_relevance import QUERY_FEATURE_NAMES

_BASE = len(FEATURE_NAMES)
_W = len(QUERY_FEATURE_NAMES)


def _synthetic_samples(n: int = 250) -> list[tuple[list[float], float]]:
    """Deterministic (no RNG) label set: even index -> near a hit (positive), odd -> far
    (negative). Enough rows to clear the default min_samples floor."""
    samples: list[tuple[list[float], float]] = []
    for i in range(n):
        x = [0.0] * _W
        x[0] = 1.0
        if i % 2 == 0:
            x[_BASE + 3] = 0.05
            samples.append((x, 1.0))
        else:
            x[_BASE + 3] = 0.9
            samples.append((x, 0.0))
    return samples


# --- query_flywheel: enable/disable/is_enabled + path resolution --------------------


def test_enable_clamps_sample_rate_and_toggles_state():
    from distil import query_flywheel as fw

    fw.enable(2.5)  # above 1.0 -> clamped
    assert fw.is_enabled() is True
    assert fw._sample_rate == 1.0
    fw.enable(-1.0)  # below 0.0 -> clamped
    assert fw._sample_rate == 0.0
    fw.disable()
    assert fw.is_enabled() is False
    fw._sample_rate = 0.25  # restore module default so it can't leak into other tests


def test_default_path_honors_distil_home(monkeypatch, tmp_path):
    from distil import query_flywheel as fw

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    assert fw._default_path() == tmp_path / "query_flywheel.jsonl"


# --- query_flywheel: _sampled boundary + invalid-hex fail-open ----------------------


def test_sampled_rate_zero_always_false():
    from distil import query_flywheel as fw

    assert fw._sampled("ffffffff", 0.0) is False


def test_sampled_invalid_hex_handle_fails_open():
    from distil import query_flywheel as fw

    assert fw._sampled("not-hex!", 0.5) is False


def test_sampled_mid_rate_boundary():
    from distil import query_flywheel as fw

    assert fw._sampled("00000000", 0.5) is True  # bucket 0.0 < 0.5
    assert fw._sampled("ffffffff", 0.5) is False  # bucket 1.0 not < 0.5


# --- query_flywheel: maybe_record's sampled-out and no-rows no-ops ------------------


def test_maybe_record_sampled_out_writes_nothing(monkeypatch, tmp_path):
    from distil import query_flywheel as fw

    p = tmp_path / "fw.jsonl"
    monkeypatch.setattr(fw, "_enabled", True)
    monkeypatch.setattr(fw, "_sample_rate", 0.5)
    # "ffffffff" buckets to 1.0, never < 0.5 -> sampled out
    fw.maybe_record("ffffffff", frozenset({"x"}), ["a line"], "tool_output", {0}, path=p)
    assert not p.exists()


def test_maybe_record_no_rows_when_dropped_indices_out_of_range(monkeypatch, tmp_path):
    from distil import query_flywheel as fw

    p = tmp_path / "fw.jsonl"
    monkeypatch.setattr(fw, "_enabled", True)
    monkeypatch.setattr(fw, "_sample_rate", 1.0)
    fw.maybe_record("aaaa1111", frozenset({"x"}), ["a", "b"], "tool_output", {99}, path=p)
    assert not p.exists()  # every dropped index out of [0, n) -> rows stays empty -> no-op


# --- query_flywheel: load_labels fail-open + skip paths -----------------------------


def test_load_labels_missing_flywheel_file_returns_empty(tmp_path):
    from distil import query_flywheel as fw

    labels = fw.load_labels(
        flywheel_path=tmp_path / "missing.jsonl", signal_path=tmp_path / "missing_sig.jsonl"
    )
    assert labels == []


def test_load_labels_skips_malformed_flywheel_rows(tmp_path):
    from distil import query_flywheel as fw

    good_row = [0.0] * _W
    fwp = tmp_path / "fw.jsonl"
    fwp.write_text(
        "\n".join(
            [
                "",  # blank line -> skipped
                "not valid json{",  # bad json -> skipped
                json.dumps({"h": 123, "rows": [good_row]}),  # handle not a str -> skipped
                json.dumps({"h": "x", "rows": "nope"}),  # rows not a list -> skipped
                json.dumps({"h": "good", "rows": [good_row]}),  # valid, never expanded
            ]
        )
        + "\n"
    )
    labels = fw.load_labels(flywheel_path=fwp, signal_path=tmp_path / "sig.jsonl")
    assert labels == [(good_row, 0.0)]


def test_load_labels_skips_malformed_signal_rows(tmp_path):
    from distil import query_flywheel as fw

    good_row = [0.0] * _W
    fwp = tmp_path / "fw.jsonl"
    fwp.write_text(json.dumps({"h": "e1", "rows": [good_row]}) + "\n")
    sig = tmp_path / "sig.jsonl"
    sig.write_text(
        "\n".join(
            [
                "",  # blank -> skipped
                "not json{",  # bad json -> skipped
                json.dumps({"handle": "e1"}),  # valid -> marks e1 expanded
            ]
        )
        + "\n"
    )
    labels = fw.load_labels(flywheel_path=fwp, signal_path=sig)
    assert labels == [(good_row, 1.0)]


def test_load_labels_signal_read_error_fails_open(tmp_path):
    from distil import query_flywheel as fw

    good_row = [0.0] * _W
    fwp = tmp_path / "fw.jsonl"
    fwp.write_text(json.dumps({"h": "e1", "rows": [good_row]}) + "\n")
    sig_dir = tmp_path / "sig_is_a_dir.jsonl"
    sig_dir.mkdir()  # exists() True but read_text() raises IsADirectoryError (an OSError)
    labels = fw.load_labels(flywheel_path=fwp, signal_path=sig_dir)
    # signal read failed -> nothing counted as expanded -> weak negative, no crash
    assert labels == [(good_row, 0.0)]


def test_load_labels_flywheel_read_error_fails_open(tmp_path):
    from distil import query_flywheel as fw

    fw_dir = tmp_path / "fw_is_a_dir.jsonl"
    fw_dir.mkdir()
    labels = fw.load_labels(flywheel_path=fw_dir, signal_path=tmp_path / "sig.jsonl")
    assert labels == []


# --- query_train: _baseline_recall + train_query_relevance guard rails --------------


def test_baseline_recall_zero_without_positives():
    from distil import query_train

    assert query_train._baseline_recall([([0.0] * _W, 0.0)]) == 0.0


def test_train_query_relevance_rejects_empty_samples():
    from distil import query_train

    with pytest.raises(ValueError, match="empty label set"):
        query_train.train_query_relevance([])


def test_train_query_relevance_rejects_bad_width():
    from distil import query_train

    with pytest.raises(ValueError, match="width"):
        query_train.train_query_relevance([([0.0] * (_W - 1), 1.0)])


def test_train_query_relevance_rejects_empty_train_split():
    from distil import query_train

    samples = _synthetic_samples(n=10)
    # every sample buckets into the test split -> train split empty
    with pytest.raises(RuntimeError, match="Empty train split"):
        query_train.train_query_relevance(samples, test_fraction=1.0)


# --- query_train: certify_and_promote's default-samples + reject branches -----------


def test_certify_and_promote_default_samples_loads_from_flywheel(monkeypatch, tmp_path):
    from distil import query_train

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))  # no flywheel/signal files present
    report = query_train.certify_and_promote(weights_path=tmp_path / "qw.json")
    assert report["promoted"] is False
    assert report["n_labels"] == 0
    assert "too few" in report["reason"]


def test_certify_and_promote_rejects_on_recall_gain_floor(tmp_path):
    from distil import query_train

    samples = _synthetic_samples()
    # an unreachable gain floor forces the "doesn't beat baseline" branch regardless of the
    # actual trained metrics (gain is bounded in [-1, 1])
    report = query_train.certify_and_promote(
        samples, weights_path=tmp_path / "qw.json", min_samples=200, min_recall_gain=2.0
    )
    assert report["promoted"] is False
    assert "does not beat phase-1 baseline" in report["reason"]
    assert not (tmp_path / "qw.json").exists()


def test_certify_and_promote_rejects_on_precision_floor(tmp_path):
    from distil import query_train

    samples = _synthetic_samples()
    # gain floor trivially satisfied (-2.0), precision floor unreachable (2.0 > any precision)
    report = query_train.certify_and_promote(
        samples,
        weights_path=tmp_path / "qw.json",
        min_samples=200,
        min_recall_gain=-2.0,
        min_precision=2.0,
    )
    assert report["promoted"] is False
    assert "precision" in report["reason"] and "floor" in report["reason"]
    assert not (tmp_path / "qw.json").exists()


# --- query_assoc: record_cooccurrence no-op branches ---------------------------------


def test_record_cooccurrence_noop_on_empty_terms(tmp_path):
    from distil import query_assoc

    p = tmp_path / "co.jsonl"
    query_assoc.record_cooccurrence("h1", frozenset(), {"o"}, path=p)
    query_assoc.record_cooccurrence("h1", frozenset({"q"}), set(), path=p)
    assert not p.exists()


def test_record_cooccurrence_noop_when_cap_zero(tmp_path):
    from distil import query_assoc

    p = tmp_path / "co.jsonl"
    query_assoc.record_cooccurrence("h1", frozenset({"q"}), {"o"}, path=p, cap=0)
    assert not p.exists()  # hashed sets sliced to [:0] -> empty -> no-op


# --- query_assoc: build_associations fail-open + skip paths -------------------------


def test_build_associations_missing_file_returns_empty(tmp_path):
    from distil import query_assoc

    table = query_assoc.build_associations(
        cooccur_path=tmp_path / "missing.jsonl", save_to=tmp_path / "assoc.json"
    )
    assert table == {}


def test_build_associations_skips_malformed_rows(tmp_path):
    from distil import query_assoc

    cp = tmp_path / "co.jsonl"
    sp = tmp_path / "sig.jsonl"
    good = json.dumps({"h": "e0", "q": ["aa"], "o": ["bb"]})
    cp.write_text(
        "\n".join(
            [
                "",  # blank -> skipped
                "not json{",  # bad json -> skipped
                json.dumps({"h": "e1", "q": "notalist", "o": ["bb"]}),  # malformed -> skipped
                good,
                good,
                good,  # 3x co-occurrence, all in the expanded block e0
            ]
        )
        + "\n"
    )
    sp.write_text(json.dumps({"handle": "e0"}) + "\n")
    table = query_assoc.build_associations(
        cooccur_path=cp, signal_path=sp, min_count=3, min_ratio=0.6, save_to=tmp_path / "assoc.json"
    )
    assert table == {"aa": ["bb"]}


def test_build_associations_cooccur_read_error_returns_empty(tmp_path):
    from distil import query_assoc

    cp = tmp_path / "co_is_a_dir.jsonl"
    cp.mkdir()  # exists() True, read_text() raises an OSError
    table = query_assoc.build_associations(cooccur_path=cp, save_to=tmp_path / "assoc.json")
    assert table == {}


def test_build_associations_save_error_fails_open(tmp_path):
    from distil import query_assoc

    cp = tmp_path / "co.jsonl"
    cp.write_text("")  # empty -> no pairs, but the save_to write is still attempted
    bad_dest = tmp_path / "missing_dir" / "assoc.json"  # parent doesn't exist -> write raises
    table = query_assoc.build_associations(cooccur_path=cp, save_to=bad_dest)
    assert table == {}
    assert not bad_dest.exists()


# --- query_assoc: _expanded_handles skip paths + fail-open ---------------------------


def test_expanded_handles_skips_blank_and_bad_json(tmp_path):
    from distil import query_assoc

    sp = tmp_path / "sig.jsonl"
    sp.write_text(
        "\n".join(
            [
                "",  # blank -> skipped
                "not json{",  # bad json -> skipped
                json.dumps({"handle": 123}),  # not a str -> skipped
                json.dumps({"handle": "h1"}),
            ]
        )
        + "\n"
    )
    assert query_assoc._expanded_handles(sp) == {"h1"}


def test_expanded_handles_read_error_fails_open(tmp_path):
    from distil import query_assoc

    sp = tmp_path / "sig_is_a_dir.jsonl"
    sp.mkdir()
    assert query_assoc._expanded_handles(sp) == set()

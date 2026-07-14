"""Self-calibrating token counts (A + C): the factor converges to the real tokenizer, is
identity until proven, content-free, and robust. Public-facing headline numbers depend on this,
so each property is asserted directly."""

import json

import pytest

from distil import calibration


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "cal.json"


def test_identity_until_min_samples(store):
    for _ in range(calibration.MIN_SAMPLES - 1):
        calibration.record("m", 100, 130, path=store)  # true factor 1.3
    f, n = calibration.factor("m", path=store)
    assert f == 1.0 and n == calibration.MIN_SAMPLES - 1  # identity — no early skew
    calibration.record("m", 100, 130, path=store)  # crosses the threshold
    f, n = calibration.factor("m", path=store)
    assert n == calibration.MIN_SAMPLES and 1.25 <= f <= 1.35


def test_factor_converges_to_true_ratio(store):
    for i in range(60):  # billed = 1.2 * est, small noise
        est = 1000 + i
        calibration.record("claude-x", est, int(est * 1.2) + (i % 3 - 1), path=store)
    f, n = calibration.factor("claude-x", path=store)
    assert n == 60 and 1.18 <= f <= 1.22
    assert calibration.calibrate(1000, "claude-x", path=store) == round(1000 * f)


def test_content_free(store):
    calibration.record("claude-x", 400, 500, path=store)
    blob = store.read_text()
    assert "claude-x" in blob  # model id is fine
    for banned in ("prompt", "message", "text", "content"):
        assert banned not in blob  # no request text ever


def test_record_skips_nonpositive(store):
    calibration.record("m", 0, 500, path=store)
    calibration.record("m", 500, 0, path=store)
    calibration.record("m", -1, 5, path=store)
    assert not store.exists() or json.loads(store.read_text())["models"] == {}


def test_per_model_and_pooled(store):
    for _ in range(25):
        calibration.record("a", 100, 120, path=store)  # 1.2
    for _ in range(25):
        calibration.record("b", 100, 140, path=store)  # 1.4
    fa, _ = calibration.factor("a", path=store)
    fb, _ = calibration.factor("b", path=store)
    fp, np_ = calibration.factor(None, path=store)  # pooled
    assert 1.15 <= fa <= 1.25 and 1.35 <= fb <= 1.45
    assert 1.25 <= fp <= 1.35 and np_ == 50  # pooled sits between


def test_relative_ci_and_status(store):
    for _ in range(30):
        calibration.record("m", 100, 125, path=store)
    st = calibration.status("m", path=store)
    assert st["calibrated"] is True and st["samples"] == 30
    assert st["relative_ci"] is not None and st["relative_ci"] >= 0.0


def test_reservoir_bounded(store):
    for i in range(calibration._RESERVOIR + 120):
        calibration.record("m", 100, 120 + (i % 5), path=store)
    ratios = json.loads(store.read_text())["models"]["m"]["ratios"]
    assert len(ratios) == calibration._RESERVOIR  # bounded — most-recent kept
    f, n = calibration.factor("m", path=store)
    assert n == calibration._RESERVOIR + 120 and 1.15 <= f <= 1.30  # count still accumulates


def test_corrupt_store_is_safe(store):
    store.write_text("{ not json")
    assert calibration.factor("m", path=store) == (1.0, 0)  # falls back to identity
    calibration.record("m", 100, 120, path=store)  # and can recover
    assert json.loads(store.read_text())["models"]["m"]["n"] == 1

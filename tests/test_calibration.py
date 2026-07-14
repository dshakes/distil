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
    # factor's n is the count of IN-BAND ratios, bounded by the reservoir.
    assert n == calibration._RESERVOIR and 1.15 <= f <= 1.30


def test_corrupt_store_is_safe(store):
    store.write_text("{ not json")
    assert calibration.factor("m", path=store) == (1.0, 0)  # falls back to identity
    calibration.record("m", 100, 120, path=store)  # and can recover
    assert json.loads(store.read_text())["models"]["m"]["n"] == 1


def test_record_filters_implausible_pairs(store):
    # the prompt-cache miss produced ratios ~0.001 (est counted cached tokens, billed didn't).
    # That is not a tokenizer difference — it's filtered at record time, so it never enters the
    # store. Both a too-low and an absurdly-high ratio are rejected.
    for _ in range(30):
        calibration.record("m", 200_000, 200, path=store)  # ratio 0.001 — rejected
        calibration.record("m", 100, 900, path=store)  # ratio 9.0 — rejected
    assert not store.exists() or json.loads(store.read_text())["models"] == {}
    assert calibration.factor("m", path=store) == (1.0, 0)
    assert calibration.status("m", path=store)["calibrated"] is False


def test_factor_ignores_garbage_already_in_store(store):
    # Robustness: even if garbage was written by an older or concurrent producer that lacks the
    # record-time filter, factor() computes from ONLY the in-band ratios — a public headline can
    # never be poisoned by a capture miss sitting in the store.
    good = [1.03] * 22
    garbage = [0.0, 0.001, 0.0] * 40
    store.write_text(
        json.dumps(
            {
                "v": calibration._STORE_VERSION,
                "models": {
                    "m": {
                        "est_sum": 0,
                        "billed_sum": 0,
                        "n": len(good) + len(garbage),
                        "ratios": good + garbage,
                    }
                },
            }
        )
    )
    f, n = calibration.factor("m", path=store)
    assert n == 22 and 1.0 < f < 1.06  # uses the 22 good, ignores the 120 garbage
    assert calibration.status("m", path=store)["calibrated"] is True


def test_scan_usage_captures_cache_tokens():
    from distil.streamrelay import scan_usage

    blob = (
        b'{"usage":{"input_tokens":50,"cache_read_input_tokens":8000,'
        b'"cache_creation_input_tokens":1000,"output_tokens":20}}'
    )
    u = scan_usage(blob)
    # true billed input = 50 + 8000 + 1000 = 9050, not just the 50 uncached tokens
    assert u["input_tokens"] == 50
    assert u["cache_read_input_tokens"] == 8000 and u["cache_creation_input_tokens"] == 1000
    assert (
        u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"] == 9050
    )

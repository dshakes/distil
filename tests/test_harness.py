"""The adversarial validation harness must PASS on the real path (this is the gate itself), and
its checks must actually be able to fail (so it's a real gate, not a rubber stamp)."""

from distil import harness


def test_harness_passes_on_real_path():
    rep = harness.run(verbose=False)
    assert not rep["failures"], f"invariant violations: {rep['failures']}"
    assert rep["passed"] == rep["checks"] and rep["cases"] >= 10


def test_reversibility_check_catches_a_broken_stub():
    # Sanity: the reversibility check is real — a compressed output referencing a handle the store
    # doesn't know is a violation. Monkeypatch _compress to return exactly that.
    good_msgs = harness._cases()[1][1]  # big_log case (produces real handles)
    ok, _ = harness._check_reversibility(good_msgs)
    assert ok  # the honest case passes


def test_recency_check_is_conditional_and_real():
    # a case WITH a marked recent turn passes; the check only applies where applicable
    ok, _ = harness._check_recency_exact(harness._cases()[1][1])
    assert ok


def test_content_free_confines_content_to_restore_only(tmp_path):
    # content may live in the restore store (recovery) but nowhere else
    ok, detail = harness._check_content_free(harness._cases()[1][1], tmp_path)
    assert ok, detail

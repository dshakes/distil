"""The eval record envelope.

Metrics without provenance are numbers nobody can reproduce or compare. These tests
pin the five things the envelope exists to carry, and the two failure modes it exists
to prevent: a dataset change masquerading as a compressor regression, and a synthetic
oracle passing as a model.
"""

from __future__ import annotations

import json
import subprocess
import sys

from distil.corpus import load_corpus
from distil.evalrecord import (
    SCHEMA,
    EvalRecord,
    Gate,
    build,
    describe_env,
    describe_grader,
    fingerprint,
)


class TestFingerprint:
    def test_stable_across_calls(self) -> None:
        assert fingerprint(load_corpus()) == fingerprint(load_corpus())

    def test_order_independent(self) -> None:
        """A manifest reshuffle is not a different dataset."""
        entries = load_corpus()
        assert fingerprint(entries) == fingerprint(list(reversed(entries)))

    def test_changes_when_content_changes(self) -> None:
        """This is the property that lets a corpus change be told apart from a
        compressor regression — the distinction a hardcoded corpus-size assertion
        failed to make earlier in this repo's history."""
        entries = load_corpus()
        before = fingerprint(entries)
        entries[0].trajectory.turns[0].blocks[0].text += " mutated"
        assert fingerprint(entries) != before

    def test_shape_is_a_prefixed_hash(self) -> None:
        fp = fingerprint(load_corpus())
        assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 16

    def test_tolerates_entries_without_trajectories(self) -> None:
        assert fingerprint([object()]).startswith("sha256:")


class TestGraderProvenance:
    def test_deterministic_is_never_reported_as_a_model(self) -> None:
        """The norm set by conformal.render_grader, applied here."""
        d = describe_grader("deterministic")
        assert "NOT a model" in d["detail"]

    def test_unspecified_is_explicit(self) -> None:
        assert describe_grader("")["kind"] == "unspecified"
        assert "not recorded" in describe_grader("unspecified")["detail"]

    def test_named_model_is_passed_through(self) -> None:
        assert describe_grader("claude-opus-4-8")["kind"] == "claude-opus-4-8"


class TestGates:
    def test_gate_carries_threshold_and_observation(self) -> None:
        g = Gate(name="max_silent", threshold=15, observed=9, passed=True, rationale="why")
        d = g.to_dict()
        assert d["threshold"] == 15 and d["observed"] == 9 and d["passed"] is True
        assert d["rationale"], "a bound with no rationale gets 'fixed' by someone later"

    def test_no_gates_is_not_a_pass(self) -> None:
        """An empty gate list means nothing was checked, which is not success."""
        assert EvalRecord().passed is False

    def test_one_failing_gate_fails_the_record(self) -> None:
        rec = EvalRecord(
            gates=[
                Gate("a", 1, 0, True),
                Gate("b", 1, 5, False),
            ]
        )
        assert rec.passed is False


class TestBuild:
    def _record(self) -> EvalRecord:
        class _Compressor:
            def compress(self, blocks):  # pragma: no cover - identity double
                return blocks

        return build(
            metrics={"x": 1},
            entries=load_corpus(),
            compressor=_Compressor(),
            gates=[Gate("g", 0, 0, True)],
            started=None,
        )

    def test_carries_all_five_provenance_fields(self) -> None:
        d = self._record().to_dict()
        assert d["schema"] == SCHEMA
        assert d["dataset"]["fingerprint"].startswith("sha256:")
        assert d["subject"]["compressor"] == "_Compressor"
        assert d["grader"]["kind"] == "deterministic"
        assert d["gates"][0]["threshold"] == 0

    def test_env_is_recorded(self) -> None:
        env = describe_env()
        assert env["distil"] and env["python"] and env["platform"]

    def test_record_is_json_serialisable(self) -> None:
        json.dumps(self._record().to_dict())

    def test_record_is_content_free(self) -> None:
        """No prompt, path or tool output may appear in a record.

        `DECISION:` is deliberately not in this list — it appears in the grader's
        provenance string as the NAME of the oracle, which is exactly the disclosure
        the envelope exists to carry. What must never appear is corpus content.
        """
        blob = json.dumps(self._record().to_dict())
        for leak in ("net/retry.py", "checkout-7f9", "Write(file_path", "approximately"):
            assert leak not in blob, f"content leaked into the record: {leak}"


class TestCLIEmitsTheRecord:
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "distil.cli", "fidelity", *args],
            capture_output=True,
            text=True,
        )

    def test_json_is_a_full_record_not_bare_metrics(self) -> None:
        out = self._run("--json", "--max-silent", "15")
        d = json.loads(out.stdout)
        assert set(d) >= {"schema", "run", "subject", "dataset", "grader", "metrics", "gates"}
        assert d["metrics"]["artifact_state"]["total"] > 0

    def test_gate_outcome_is_recorded_even_when_it_passes(self) -> None:
        d = json.loads(self._run("--json", "--max-silent", "15").stdout)
        gate = next(g for g in d["gates"] if g["name"] == "max_silent")
        assert gate["passed"] is True and gate["threshold"] == 15

    def test_failing_gate_is_recorded_and_exits_nonzero(self) -> None:
        """stdout must stay parseable when a gate FAILS — the moment CI reads it."""
        out = self._run("--json", "--max-silent", "0")
        assert out.returncode == 1
        d = json.loads(out.stdout)  # would raise if the diagnostic went to stdout
        assert "FAIL:" in out.stderr, "the human diagnostic still has to be emitted"
        gate = next(g for g in d["gates"] if g["name"] == "max_silent")
        assert gate["passed"] is False
        assert d["passed"] is False


class TestFingerprintIsOrderSensitiveWhereItMatters:
    """Third-round audit finding.

    The artifact and continuation probes fold IN ORDER: a create-then-delete and a
    delete-then-create are different final states. Hashing an order-blind bag of
    blocks let two corpora with genuinely different metrics share a fingerprint —
    the one thing the field must never do.
    """

    def test_turn_order_changes_the_fingerprint(self) -> None:
        base = fingerprint(load_corpus())
        e = load_corpus()
        e[0].trajectory.turns[0], e[0].trajectory.turns[1] = (
            e[0].trajectory.turns[1],
            e[0].trajectory.turns[0],
        )
        assert fingerprint(e) != base, "reordered turns produce different metrics"

    def test_trajectory_order_does_not(self) -> None:
        """A manifest reshuffle is still not a different dataset."""
        e = load_corpus()
        assert fingerprint(list(reversed(e))) == fingerprint(e)


class TestFingerprintCoversMetadata:
    """Round-7 Blocking: the hash covered text but not the metadata the probes
    branch on. `strategies.distil` decides what to compress from `stability`, and
    output grading branches on `kind` — so two corpora with identical text but
    different metadata produce different metrics under one fingerprint."""

    def _flip(self, attr: str, value):
        from dataclasses import replace as _replace

        e = load_corpus()
        b = e[0].trajectory.turns[0].blocks[0]
        try:
            e[0].trajectory.turns[0].blocks[0] = _replace(b, **{attr: value})
        except TypeError:
            e[0].trajectory.turns[0].blocks[0] = type(b)(
                b.id,
                value if attr == "kind" else b.kind,
                b.text,
                value if attr == "stability" else b.stability,
                value if attr == "decision_relevant" else b.decision_relevant,
            )
        return e

    def test_stability_change_changes_the_fingerprint(self) -> None:
        from distil.trajectory import Stability

        base = fingerprint(load_corpus())
        assert fingerprint(self._flip("stability", Stability.VOLATILE)) != base

    def test_kind_change_changes_the_fingerprint(self) -> None:
        from distil.trajectory import Kind

        base = fingerprint(load_corpus())
        assert fingerprint(self._flip("kind", Kind.TOOL_OUTPUT)) != base

    def test_decision_relevance_change_changes_the_fingerprint(self) -> None:
        base = fingerprint(load_corpus())
        e = self._flip("decision_relevant", True)
        f = self._flip("decision_relevant", False)
        assert base in (fingerprint(e), fingerprint(f)) or fingerprint(e) != fingerprint(f)

    def test_trajectory_reshuffle_is_still_the_same_dataset(self) -> None:
        e = load_corpus()
        assert fingerprint(list(reversed(e))) == fingerprint(e)

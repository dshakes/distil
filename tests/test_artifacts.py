"""Artifact-state fidelity.

The tests that matter here are the ones separating `stale` from `lost`. String-recall
metrics score both as "the path is there" or "the path is gone"; the whole point of
this module is that those are different failures with different blast radii.
"""

from __future__ import annotations

from distil.artifacts import (
    Op,
    StateProbe,
    build_ledger,
    extract_ops,
    format_probe,
    measure_live,
    score,
    score_texts,
)


class TestExtraction:
    def test_structured_tool_calls(self) -> None:
        ops = dict(extract_ops('Write(file_path="src/app.py", content="x")'))
        assert ops["src/app.py"] is Op.CREATE

    def test_shell_delete(self) -> None:
        assert ("build/out.js", Op.DELETE) in extract_ops("rm -rf build/out.js")

    def test_narration(self) -> None:
        found = dict(extract_ops("I created config.yaml and then deleted old/stale.json"))
        assert found["config.yaml"] is Op.CREATE
        assert found["old/stale.json"] is Op.DELETE

    def test_diff_header_is_a_modify(self) -> None:
        assert ("pkg/mod.go", Op.MODIFY) in extract_ops("+++ b/pkg/mod.go")

    def test_path_spellings_canonicalise_to_one_artifact(self) -> None:
        led = build_ledger(["created ./src/a.py", "modified src/a.py"])
        assert list(led.state) == ["src/a.py"], "spelling variants must not split an artifact"
        assert led.state["src/a.py"] is Op.MODIFY

    def test_bare_words_are_not_paths(self) -> None:
        # No extension, no separator: not an artifact. Guards against ledger noise.
        assert extract_ops("created something and removed nothing") == []


class TestLedgerFold:
    def test_later_turn_wins(self) -> None:
        led = build_ledger(["created a.py", "modified a.py", "rm a.py"])
        assert led.state["a.py"] is Op.DELETE

    def test_delete_not_masked_by_read_in_same_turn(self) -> None:
        # Same turn asserts both. The strong op must win, or a delete disappears.
        led = build_ledger(["cat a.py && rm a.py"])
        assert led.state["a.py"] is Op.DELETE

    def test_resurrection_is_tracked(self) -> None:
        led = build_ledger(["rm a.py", "created a.py"])
        assert led.state["a.py"] is Op.CREATE, "a later create must overwrite an earlier delete"


class TestScoring:
    def test_perfect_preservation(self) -> None:
        turns = ["created a.py", "modified a.py"]
        p = score(turns, turns)
        assert (p.total, p.exact, p.stale, p.lost) == (1, 1, 0, 0)
        assert p.fidelity == 1.0

    def test_the_phantom_file(self) -> None:
        """The failure this module exists for: string kept, final state wrong."""
        original = ["created config.yaml", "deleted config.yaml"]
        compressed = ["created config.yaml"]  # the delete was compressed away
        p = score(original, compressed)
        assert p.stale == 1 and p.lost == 0 and p.exact == 0
        assert p.stale_rate == 1.0
        assert p.silent_failure_share == 1.0, "a wrong state is a silent failure"

    def test_string_recall_would_have_scored_this_perfect(self) -> None:
        """Proves the metric is not redundant with the existing artifacts dimension."""
        original = ["created config.yaml", "deleted config.yaml"]
        compressed = ["created config.yaml"]
        assert "config.yaml" in compressed[0], "string-presence recall = 100%"
        assert score(original, compressed).fidelity == 0.0, "state fidelity = 0%"

    def test_lost_is_not_stale(self) -> None:
        p = score(["created a.py"], ["nothing relevant here"])
        assert p.lost == 1 and p.stale == 0
        assert p.silent_failure_share == 0.0, "an honest gap is not a silent failure"

    def test_mixed_outcome_shares(self) -> None:
        original = ["created a.py", "deleted a.py", "created b.py", "created c.py"]
        compressed = ["created a.py", "created b.py"]  # a.py stale, b.py exact, c.py lost
        p = score(original, compressed)
        assert (p.total, p.exact, p.stale, p.lost) == (3, 1, 1, 1)
        assert p.silent_failure_share == 0.5

    def test_empty_is_vacuously_perfect_not_a_crash(self) -> None:
        p = score([], [])
        assert p.total == 0 and p.fidelity == 1.0 and p.stale_rate == 0.0

    def test_score_texts_wrapper(self) -> None:
        assert score_texts("created a.py", "created a.py").fidelity == 1.0


class TestLiveMeasurement:
    def test_reads_structured_tool_use_blocks(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "src/a.py"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "rm src/a.py"}}
                ],
            },
        ]
        compressed = [original[0]]  # the delete turn was dropped
        p = measure_live(original, compressed)
        assert p.stale == 1, "dropping the delete leaves a phantom file"

    def test_tool_results_are_read_too(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": "deleted old/x.json"}]}
        ]
        p = measure_live(msgs, msgs)
        assert p.total == 1 and p.exact == 1

    def test_malformed_payload_does_not_raise(self) -> None:
        assert measure_live(None, None).total == 0
        assert measure_live([{"content": 7}], ["not a dict"]).total == 0


class TestReporting:
    def test_probe_add_accumulates(self) -> None:
        a = StateProbe(total=2, exact=1, stale=1)
        a.add(StateProbe(total=1, exact=0, lost=1))
        assert (a.total, a.exact, a.stale, a.lost) == (3, 1, 1, 1)

    def test_format_leads_with_stale(self) -> None:
        out = format_probe(StateProbe(total=4, exact=2, stale=1, lost=1))
        assert "stale" in out and "false belief" in out
        assert out.index("stale") < out.index("lost"), "stale must be reported before lost"

    def test_format_handles_nothing_to_grade(self) -> None:
        assert "nothing to grade" in format_probe(StateProbe())

    def test_to_dict_is_json_safe_and_content_free(self) -> None:
        d = score(["created secret/path.py"], []).to_dict()
        assert set(d) == {
            "total",
            "exact",
            "stale",
            "lost",
            "fidelity",
            "stale_rate",
            "silent_failure_share",
        }
        assert "secret" not in str(d), "counts only — never a path"

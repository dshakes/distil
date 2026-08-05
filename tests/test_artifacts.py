"""Artifact-state fidelity.

The tests that matter here are the ones separating `stale` from `lost`. String-recall
metrics score both as "the path is there" or "the path is gone"; the whole point of
this module is that those are different failures with different blast radii.
"""

from __future__ import annotations

import json

from distil.artifacts import (
    _ADJUDICATED,
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


class TestAuditFindings:
    """Regressions for the three defects the cross-audit found on PR #81."""

    def test_failed_operations_are_not_recorded(self) -> None:
        """A failed tool call asserts nothing about workspace state.

        Before this, `rm missing.py` with `exit: 1` was scored as a successful DELETE,
        which made the ledger a record of ATTEMPTS. A compressor could then be
        penalised for dropping a deletion that never happened.
        """
        failed = 'Bash(command="rm missing.py")\n-> {"stdout": "rm: No such file or directory", "exit": 1}'
        assert build_ledger([failed]).state == {}

    def test_successful_operations_still_recorded(self) -> None:
        ok = 'Bash(command="rm real.py")\n-> {"stdout": "", "exit": 0}'
        assert build_ledger([ok]).state == {"real.py": Op.DELETE}

    def test_ok_false_counts_as_failure(self) -> None:
        assert build_ledger(['Write(file_path="a.py")\n-> {"ok": false}']).state == {}

    def test_extensionless_artifacts_are_visible(self) -> None:
        """Dockerfile, Makefile, LICENSE and dotfiles are among the most-edited files
        in a real repo, and every one of them was invisible to the gate."""
        for text, path, op in (
            ("rm ./Dockerfile", "Dockerfile", Op.DELETE),
            ('Edit(file_path="Makefile")', "Makefile", Op.MODIFY),
            ("created .env", ".env", Op.CREATE),
            ("deleted LICENSE", "LICENSE", Op.DELETE),
        ):
            assert (path, op) in extract_ops(text), f"{path} must be tracked"

    def test_ignorecase_does_not_turn_prose_into_artifacts(self) -> None:
        """re.IGNORECASE makes `[A-Z]` match lowercase, which turned every bare word
        into a file. The capitalisation guard is scoped with `(?-i:...)` so the
        enclosing flag cannot silently disable it."""
        assert extract_ops("created something and removed nothing") == []
        assert extract_ops("added a thing and deleted another") == []
        # ...while a genuinely capitalised name still matches under the same flag
        assert ("README", Op.DELETE) in extract_ops("DELETED README")


class TestLivePairing:
    """Second-round audit finding: a call and its result must be joined.

    Rendering `tool_use` and `tool_result` as independent strings meant the failure
    check never saw them together, so a provider-shaped failed Write was recorded as
    a successful create — the "ledger of attempts" bug, in the live path.
    """

    def _msgs(self, outcome: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Write",
                        "input": {"file_path": "a.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": outcome}],
            },
        ]

    def test_failed_call_records_nothing(self) -> None:
        assert (
            measure_live(self._msgs("Permission denied"), self._msgs("Permission denied")).total
            == 0
        )

    def test_nonzero_exit_in_a_separate_result_block(self) -> None:
        msgs = self._msgs('{"stdout": "", "exit": 1}')
        assert measure_live(msgs, msgs).total == 0

    def test_successful_call_still_records(self) -> None:
        msgs = self._msgs('{"ok": true}')
        assert measure_live(msgs, msgs).total == 1

    def test_unmatched_call_is_still_read(self) -> None:
        """No result to consult — read the call optimistically rather than lose it."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "x", "name": "Write", "input": {"file_path": "b.py"}}
                ],
            }
        ]
        assert measure_live(msgs, msgs).total == 1

    def test_orphan_result_is_still_read(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "z", "content": "deleted old/x.json"}
                ],
            }
        ]
        assert measure_live(msgs, msgs).total == 1


class TestThirdAuditRound:
    def test_live_texts_are_emitted_in_document_order(self) -> None:
        """Buffering calls and appending them after text reversed the fold.

        `Write(a.py)` then a later `deleted a.py` folded as delete-then-create, so the
        final state read EXISTS when the transcript ends with it deleted.
        """
        from distil.artifacts import _iter_tool_texts

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Write",
                        "input": {"file_path": "a.py"},
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "deleted a.py"}]},
        ]
        # The call arrives pre-adjudicated (see `_ADJUDICATED`); narration does not.
        assert [t.replace(_ADJUDICATED, "") for t in _iter_tool_texts(msgs)] == [
            'Write(file_path="a.py")',
            "deleted a.py",
        ]
        assert build_ledger(_iter_tool_texts(msgs)).state == {"a.py": Op.DELETE}

    def test_ordering_fix_did_not_break_failure_pairing(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Write",
                        "input": {"file_path": "b.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Permission denied"}
                ],
            },
        ]
        assert measure_live(msgs, msgs).total == 0

    def test_multi_dot_filenames_are_one_artifact(self) -> None:
        """`c.tar.gz` extracted as `c.tar` — truncating names and potentially
        merging two distinct files into one ledger entry."""
        for text, want in (
            ("created a/b/c.tar.gz", "a/b/c.tar.gz"),
            ("created foo.bar.py", "foo.bar.py"),
            ("created x.test.tsx", "x.test.tsx"),
            ("created api.spec.ts", "api.spec.ts"),
        ):
            assert want in [p for p, _ in extract_ops(text)], f"{text} -> expected {want}"

    def test_single_extension_still_works(self) -> None:
        assert ("src/app.py", Op.CREATE) in extract_ops("created src/app.py")


class TestProviderShapes:
    """Fourth-round Blocking finding.

    `_iter_tool_texts` only read Anthropic `content` blocks. A purely structured
    OpenAI turn produced no text at all, so the probe reported total=0, fidelity=1.0
    — a perfect score on a workspace it never looked at. Silent AND green is the
    worst failure mode this metric can have. `distil.retention` already normalises
    OpenAI shapes; this brings the ledger into line.
    """

    def test_openai_chat_completions_tool_calls(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Write", "arguments": '{"file_path": "a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command": "rm a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": '{"exit": 0}'},
        ]
        p = measure_live(msgs, msgs[:2])  # the delete turn compressed away
        assert p.total == 1, "OpenAI tool_calls must be visible to the ledger"
        assert p.stale == 1, "keeping the create and dropping the delete is stale"

    def test_openai_responses_function_call(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "call_id": "f1",
                        "name": "Write",
                        "arguments": '{"file_path": "b.py"}',
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "function_call_output", "call_id": "f1", "output": '{"ok": true}'}
                ],
            },
        ]
        assert measure_live(msgs, msgs).total == 1

    def test_openai_failed_call_is_still_rejected(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Write", "arguments": '{"file_path": "a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Permission denied"},
        ]
        assert measure_live(msgs, msgs).total == 0

    def test_anthropic_shape_is_unchanged(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Write",
                        "input": {"file_path": "c.py"},
                    }
                ],
            }
        ]
        assert measure_live(msgs, msgs).total == 1


class TestCallIsAuthoritative:
    """Fifth-round Blocking finding.

    A `role:"tool"` string result was appended as ordinary text AND joined at its
    call's position, so the same text was parsed twice. Worse, joining let the
    RESULT's narration override the CALL's operation: because the strongest op wins
    within a block, a Write whose result mentioned "deleted a.py" folded to DELETE.

    The rule: the call states WHICH operation; the result states only WHETHER it
    succeeded.
    """

    def _openai(self, name: str, args: str, result: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": result},
        ]

    def test_result_is_not_parsed_twice(self) -> None:
        from distil.artifacts import _iter_tool_texts

        msgs = self._openai("Write", '{"file_path": "a.py"}', "deleted a.py")
        texts = _iter_tool_texts(msgs)
        assert len(texts) == 1, f"result parsed more than once: {texts}"

    def test_result_narration_cannot_override_the_call(self) -> None:
        msgs = self._openai("Write", '{"file_path": "a.py"}', "deleted a.py")
        assert build_ledger(_iter_tool_texts_for(msgs)).state == {"a.py": Op.CREATE}

    def test_failing_result_still_rejects_the_operation(self) -> None:
        msgs = self._openai("Write", '{"file_path": "b.py"}', "Permission denied")
        assert build_ledger(_iter_tool_texts_for(msgs)).state == {}

    def test_orphan_result_still_describes_state(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "z", "content": "deleted orphan.json"}
                ],
            }
        ]
        assert build_ledger(_iter_tool_texts_for(msgs)).state == {"orphan.json": Op.DELETE}

    def test_anthropic_shell_delete_unaffected(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "rm c.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"exit": 0}'}],
            },
        ]
        assert build_ledger(_iter_tool_texts_for(msgs)).state == {"c.py": Op.DELETE}


def _iter_tool_texts_for(msgs: list[dict]) -> list[str]:
    from distil.artifacts import _iter_tool_texts

    return _iter_tool_texts(msgs)


class TestFailureScope:
    """Failure evidence comes from the RESULT, never from the call's own arguments.

    This has now been wrong twice, and the second time the test was the reason it
    shipped. Round 6 fixed "failure phrase inside the arguments" by scanning only the
    text after a `->` delimiter, and the regression test built that delimiter BY HAND:

        'Write(file_path="tests/test_errors.py", ...)\\n-> {"ok": true}'

    Nothing in the live path ever produced that string — successful calls were emitted
    with no delimiter at all — so the test passed against a shape that did not exist
    while real traffic stayed broken. A probe that reports a false 100% is the exact
    failure this module is for, so every case here goes through `measure_live`, the
    entry point real traffic uses.
    """

    @staticmethod
    def _live(name: str, args: dict, result: str) -> int:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": result},
        ]
        return measure_live(msgs, msgs).total

    def test_error_text_in_call_arguments_does_not_suppress_the_op(self) -> None:
        """An agent writing an error fixture is an everyday act, not a failure."""
        args = {"file_path": "tests/test_errors.py", "content": "No such file or directory"}
        assert self._live("Write", args, '{"ok": true}') == 1

    def test_written_source_containing_a_type_hint_is_not_an_outcome(self) -> None:
        """`->` is not a delimiter: every Python type hint contains one.

        Scanning after the LAST `->` read this file's own body as the tool's result,
        so a successful write of ordinary typed Python with any error handling in it
        was scored as a failed call and dropped.
        """
        src = "def run() -> int:\n    if bad:\n        sys.exit(1)\n    return 0\n"
        assert self._live("Write", {"file_path": "run.py", "content": src}, '{"ok": true}') == 1

    def test_failure_in_the_result_still_suppresses(self) -> None:
        assert self._live("Bash", {"command": "rm x.py"}, '{"exit": 1}') == 0

    def test_failure_in_the_result_suppresses_even_with_clean_arguments(self) -> None:
        assert self._live("Write", {"file_path": "b.py"}, "Permission denied") == 0

    def test_narration_failure_without_a_delimiter_still_suppresses(self) -> None:
        """Unstructured text has no call/result split, so it is still inferred."""
        assert extract_ops("rm failed: No such file or directory for y.py") == []

    def test_ordinary_extraction_unaffected(self) -> None:
        assert ("src/app.py", Op.CREATE) in extract_ops("created src/app.py")


class TestOnlyObservedActionsCount:
    """Round-7 finding: a user's REQUEST was parsed as a completed operation.

    `Please inspect: cat src/app.py` registered a READ, so compressing the request
    away reported a lost artifact when no workspace state had ever changed. Only
    the assistant's narration and actual tool calls/results report outcomes.
    """

    def test_user_request_is_not_an_operation(self) -> None:
        msgs = [{"role": "user", "content": "Please inspect: cat src/app.py"}]
        assert measure_live(msgs, [{"role": "user", "content": "x"}]).total == 0

    def test_system_prompt_naming_files_is_not_an_operation(self) -> None:
        msgs = [{"role": "system", "content": "Never edit src/generated.py directly"}]
        assert measure_live(msgs, msgs).total == 0

    def test_assistant_narration_still_counts(self) -> None:
        msgs = [{"role": "assistant", "content": "cat src/app.py"}]
        assert measure_live(msgs, msgs).total == 1

    def test_tool_result_under_a_user_role_still_counts(self) -> None:
        """Anthropic delivers tool_result inside a `user` message — that is a
        RESULT, not a request, and must not be filtered out with user prose."""
        msgs = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "z", "content": "deleted a.py"}],
            }
        ]
        assert measure_live(msgs, msgs).total == 1


class TestFailureIsAdjudicatedPerOperation:
    """One failed call must not erase the successful ones beside it.

    Rejecting a whole block on any failure marker was the safe-looking choice and it
    was wrong: coding-agent blocks routinely carry several tool calls, so a block that
    created `a.py` and failed to create `b.py` recorded NEITHER. `a.py` then never
    entered the ledger, was never graded, and a compressor could drop it for free —
    an undercount that makes the fidelity number look clean by measuring less.
    """

    def test_a_failure_does_not_erase_its_neighbour(self) -> None:
        t = 'Write(file_path="a.py") -> {"ok": true}\nWrite(file_path="b.py") -> {"ok": false}'
        assert extract_ops(t) == [("a.py", Op.CREATE)]

    def test_only_the_failing_operation_is_dropped(self) -> None:
        t = "created x.py\nrm y.py -> exit: 1\ntouch z.py"
        assert extract_ops(t) == [("x.py", Op.CREATE), ("z.py", Op.CREATE)]

    def test_a_lone_failure_still_suppresses(self) -> None:
        assert extract_ops('Bash(command="rm missing.py")\n-> {"exit": 1}') == []

    def test_a_lone_success_is_still_recorded(self) -> None:
        assert extract_ops('Bash(command="rm real.py")\n-> {"exit": 0}') == [("real.py", Op.DELETE)]

    def test_intra_line_ops_still_fold_to_the_strongest(self) -> None:
        assert build_ledger(["cat a.py && rm a.py"]).state == {"a.py": Op.DELETE}

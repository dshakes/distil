"""Row adapters for the public benchmarks.

Each adapter turns one upstream row into a `GroundTruthCase`, and each upstream has
its own idea of where the question, the answer and the compressible payload live.
The rows below are the REAL shapes, verified against live data at the time they were
written.

Two properties matter more than the field mapping:

  * a malformed row returns ``None`` rather than a half-built case. Upstreams change
    schemas without notice, and a case with an empty answer would be graded as a
    silent miss — the compressor blamed for a loader bug;
  * the compressible payload lands in ``docs``. That is the text the compressor
    actually operates on, so an adapter that puts the payload in ``question`` would
    produce a benchmark that measures nothing while still reporting a score.
"""

from __future__ import annotations

import json

import pytest

from distil import datasets


class TestAdaptersMapRealRows:
    def test_gsm8k_takes_the_answer_after_the_marker(self) -> None:
        case = datasets._gsm8k({"question": "Janet has 16 eggs?", "answer": "16-3=13\n#### 13"})
        assert case is not None
        assert case.answer == "13", "the gold is the value after ####, not the rationale"
        assert any("16-3" in text for _, text in case.docs), "rationale is the payload"

    def test_truthfulqa_uses_the_best_answer(self) -> None:
        case = datasets._truthfulqa(
            {
                "question": "What happens if you eat watermelon seeds?",
                "best_answer": "They pass through you",
                "correct_answers": ["Nothing happens", "They pass through"],
            }
        )
        assert case is not None and case.answer == "They pass through you"
        assert len(case.support) == 2

    def test_mmlu_resolves_the_index_into_the_choice_text(self) -> None:
        case = datasets._mmlu(
            {
                "question": "Degree of Q(sqrt2)?",
                "subject": "algebra",
                "choices": ["0", "4", "2"],
                "answer": 1,
            }
        )
        assert case is not None and case.answer == "4"

    def test_mmlu_rejects_an_out_of_range_index(self) -> None:
        assert datasets._mmlu({"question": "q", "choices": ["a"], "answer": 7}) is None

    def test_arc_resolves_the_answer_key_through_the_labels(self) -> None:
        case = datasets._arc(
            {
                "id": "Mercury_1",
                "question": "What happens?",
                "choices": {"text": ["density falls", "days shorten"], "label": ["A", "B"]},
                "answerKey": "B",
            }
        )
        assert case is not None and case.answer == "days shorten"

    def test_arc_rejects_a_key_absent_from_the_labels(self) -> None:
        row = {
            "question": "q",
            "choices": {"text": ["a"], "label": ["A"]},
            "answerKey": "Z",
        }
        assert datasets._arc(row) is None

    def test_humaneval_keeps_the_prompt_as_payload(self) -> None:
        case = datasets._humaneval(
            {"task_id": "HumanEval/0", "prompt": "def f(x):\n  '''doc'''\n", "entry_point": "f"}
        )
        assert case is not None and case.answer == "f"
        assert "def f" in case.docs[0][1]

    def test_msmarco_keeps_every_passage_and_marks_the_selected_one(self) -> None:
        case = datasets._msmarco(
            {
                "query": "what is a corporation?",
                "query_id": 1,
                "answers": ["A corporation is a company"],
                "passages": {
                    "passage_text": ["irrelevant one", "the gold passage", "another"],
                    "is_selected": [0, 1, 0],
                },
            }
        )
        assert case is not None
        assert len(case.docs) == 3, "all retrieved passages are the payload, distractors included"
        assert case.support == ["the gold passage"]

    def test_triviaqa_prefers_the_answer_value(self) -> None:
        case = datasets._triviaqa(
            {"question": "Who?", "question_id": "tc_2", "answer": {"value": "David Seville"}}
        )
        assert case is not None and case.answer == "David Seville"

    def test_narrativeqa_uses_the_summary_as_payload(self) -> None:
        case = datasets._narrativeqa(
            {
                "document": {"id": "d1", "summary": {"text": "A long summary.", "title": "Play"}},
                "question": {"text": "Who speaks?"},
                "answers": [{"text": "The actor"}],
            }
        )
        assert case is not None
        assert case.answer == "The actor" and case.docs[0][1] == "A long summary."

    def test_codesearchnet_grades_the_docstring_against_the_code(self) -> None:
        case = datasets._codesearchnet(
            {
                "func_name": "get_vid",
                "func_code_string": "def get_vid(url):\n  return url",
                "func_documentation_string": "Extracts video ID from URL.",
                "func_path_in_repository": "src/x.py",
            }
        )
        assert case is not None
        assert case.answer == "Extracts video ID from URL."
        assert "def get_vid" in case.docs[0][1], "the code is the payload"

    def test_bfcl_compresses_the_schema_and_requires_the_function_name(self) -> None:
        case = datasets._bfcl(
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "Find the area"}]],
                "function": [{"name": "calculate_area", "description": "d"}],
                "ground_truth": [{"calculate_area": {"base": [10]}}],
            }
        )
        assert case is not None
        assert case.question == "Find the area", "chat turns nest two lists deep"
        assert "calculate_area" in case.docs[0][1], "the tool schema is the payload"
        # Every name the gold call is built from — the function AND each argument.
        # Requiring only the function name understated the test: a schema can keep
        # `calculate_area` while losing the `base` parameter, and the call cannot be
        # formed at all.
        assert set(case.support) == {"calculate_area", "base"}


class TestAdaptersRejectMalformedRows:
    """A half-built case is worse than no case: the compressor gets blamed for a
    loader bug, as a silent miss against a gold answer that was never there."""

    @pytest.mark.parametrize(
        "adapter,row",
        [
            (datasets._gsm8k, {"question": "q", "answer": "no marker here"}),
            (datasets._gsm8k, {"question": "", "answer": "#### 3"}),
            (datasets._truthfulqa, {"question": "q", "best_answer": ""}),
            (datasets._mmlu, {"question": "q", "choices": [], "answer": 0}),
            # a non-int answer field: upstream schema drift, not a gradeable case
            (datasets._mmlu, {"question": "q", "choices": ["a"], "answer": "B"}),
            (datasets._arc, {"question": "q", "choices": {}, "answerKey": "A"}),
            (datasets._humaneval, {"prompt": "", "entry_point": "f"}),
            (datasets._msmarco, {"query": "q", "answers": [], "passages": {}}),
            (datasets._triviaqa, {"question": "q", "answer": {}}),
            (datasets._narrativeqa, {"document": {}, "question": {}, "answers": []}),
            (datasets._codesearchnet, {"func_name": "", "func_code_string": "x"}),
            (datasets._bfcl, {"question": [], "function": [], "ground_truth": None}),
        ],
    )
    def test_a_malformed_row_yields_none(self, adapter, row) -> None:
        assert adapter(row) is None


class TestNestedTextExtraction:
    """Upstreams disagree on where text lives; one helper handles every shape."""

    @pytest.mark.parametrize(
        "value,want",
        [
            ("plain", "plain"),
            ({"text": "keyed"}, "keyed"),
            ({"content": "chat"}, "chat"),
            ([[{"role": "user", "content": "two deep"}]], "two deep"),
            ([{"text": ""}, {"text": "second"}], "second"),
            ({}, ""),
            ([], ""),
            (None, ""),
            (7, ""),
        ],
    )
    def test_first_text(self, value, want) -> None:
        assert datasets._first_text(value) == want


class TestBfclTransport:
    """BFCL ships as plain JSONL files, so it needs its own fetch path.

    Questions and ground truth live in SEPARATE files keyed by `id`. A question with
    no matching answer cannot be graded, so it is dropped — scoring it would count an
    ungradeable case as a pass.
    """

    @staticmethod
    def _fake_http(monkeypatch, files: dict[str, str]):
        import io
        import urllib.request

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _open(req, timeout=None):  # noqa: ARG001
            url = req.full_url if hasattr(req, "full_url") else str(req)
            # Longest suffix first: the ground-truth URL ends with BOTH
            # "possible_answer/BFCL_v3_simple.json" and "BFCL_v3_simple.json", so a
            # naive first-match served the questions file as the answer key and every
            # row failed to join — the fake lying, not the code.
            for name in sorted(files, key=len, reverse=True):
                if url.endswith(name):
                    return _Resp(files[name].encode())
            raise OSError(f"unexpected URL {url}")

        monkeypatch.setattr(urllib.request, "urlopen", _open)

    def test_only_questions_with_a_ground_truth_are_kept(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        q = "\n".join(
            json.dumps(
                {
                    "id": f"s_{i}",
                    "question": [[{"role": "user", "content": f"q{i}"}]],
                    "function": [{"name": f"fn_{i}"}],
                }
            )
            for i in range(3)
        )
        # ground truth for two of the three
        gt = "\n".join(
            json.dumps({"id": f"s_{i}", "ground_truth": [{f"fn_{i}": {}}]}) for i in range(2)
        )
        self._fake_http(
            monkeypatch, {"BFCL_v3_simple.json": q, "possible_answer/BFCL_v3_simple.json": gt}
        )

        rows = datasets._fetch_bfcl(datasets.SPECS["bfcl"], 10)
        assert len(rows) == 2, "a question with no answer key is ungradeable, not a pass"
        assert all("ground_truth" in r for r in rows)

    def test_nothing_joining_raises_rather_than_returning_empty(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        q = json.dumps({"id": "a", "question": [[{"content": "x"}]], "function": [{"name": "f"}]})
        self._fake_http(
            monkeypatch, {"BFCL_v3_simple.json": q, "possible_answer/BFCL_v3_simple.json": ""}
        )
        with pytest.raises(datasets.DatasetUnavailable, match="ground truth"):
            datasets._fetch_bfcl(datasets.SPECS["bfcl"], 5)

    def test_a_transport_failure_is_loud(self, monkeypatch, tmp_path) -> None:
        import urllib.request

        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("no net"))
        )
        with pytest.raises(datasets.DatasetUnavailable):
            datasets._fetch_bfcl(datasets.SPECS["bfcl"], 5)

    def test_a_warm_cache_needs_no_network(self, monkeypatch, tmp_path) -> None:
        import urllib.request

        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        cache = tmp_path / "datasets"
        cache.mkdir(parents=True)
        (cache / "bfcl.jsonl").write_text(
            json.dumps(
                {
                    "id": "a",
                    "question": [[{"content": "x"}]],
                    "function": [{"name": "f"}],
                    "ground_truth": [{"f": {}}],
                }
            )
            + "\n"
        )
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: pytest.fail("cache should avoid the network")
        )
        assert len(datasets._fetch_bfcl(datasets.SPECS["bfcl"], 1)) == 1

    def test_malformed_lines_are_skipped_not_fatal(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        q = "not json\n" + json.dumps(
            {"id": "a", "question": [[{"content": "x"}]], "function": [{"name": "f"}]}
        )
        gt = json.dumps({"id": "a", "ground_truth": [{"f": {}}]})
        self._fake_http(
            monkeypatch, {"BFCL_v3_simple.json": q, "possible_answer/BFCL_v3_simple.json": gt}
        )
        assert len(datasets._fetch_bfcl(datasets.SPECS["bfcl"], 5)) == 1


class TestFetchStopsAtTheRequestedCount:
    """`want` is a budget, not a suggestion: a suite that silently over-fetches
    changes the number someone is trying to reproduce."""

    def test_bfcl_stops_at_want(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        q = "\n".join(
            json.dumps(
                {
                    "id": f"s_{i}",
                    "question": [[{"content": f"q{i}"}]],
                    "function": [{"name": f"fn_{i}"}],
                }
            )
            for i in range(10)
        )
        gt = "\n".join(
            json.dumps({"id": f"s_{i}", "ground_truth": [{f"fn_{i}": {}}]}) for i in range(10)
        )
        TestBfclTransport._fake_http(
            monkeypatch,
            {"BFCL_v3_simple.json": q, "possible_answer/BFCL_v3_simple.json": gt},
        )
        assert len(datasets._fetch_bfcl(datasets.SPECS["bfcl"], 3)) == 3


class TestCaseIdsAreStableAcrossProcesses:
    """`hash()` on a str is salted per interpreter.

    Ids built from it changed on every invocation, which quietly broke the contract
    stated at the top of `datasets.py`: sampling is deterministic so "the number you
    publish is the number a reader reproduces". Two runs of the same benchmark could
    not be diffed by case.
    """

    def test_the_same_input_gives_the_same_id(self) -> None:
        a = datasets._stable_id("gsm8k", "Janet has 16 eggs?")
        b = datasets._stable_id("gsm8k", "Janet has 16 eggs?")
        assert a == b and a.startswith("gsm8k-")

    def test_different_inputs_differ(self) -> None:
        assert datasets._stable_id("x", "one") != datasets._stable_id("x", "two")

    def test_ids_survive_a_fresh_interpreter(self) -> None:
        """The real test: a NEW process, where the hash salt is different."""
        import subprocess
        import sys

        code = "from distil import datasets; print(datasets._stable_id('t', 'seed'))"
        seen = {
            subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
            for _ in range(3)
        }
        assert len(seen) == 1, f"id drifted across processes: {seen}"

    def test_no_adapter_still_uses_a_salted_hash(self) -> None:
        """Guard the whole class, not the four instances that were found."""
        import pathlib

        src = pathlib.Path(datasets.__file__).read_text(encoding="utf-8")
        assert "abs(hash(" not in src, "a salted hash in an id breaks reproducibility"

    def test_every_adapter_produces_distinct_ids_for_distinct_cases(self) -> None:
        """A collision makes two different cases indistinguishable in a diff.

        One was shipped here: a regex meant to swap the id builder deleted the seed
        instead, so every MMLU case in a subject became `mmlu-algebra-`. The id
        looked plausible and the tests passed, because nothing asserted uniqueness.
        """
        rows = {
            datasets._mmlu: [
                {"question": f"Q{i}", "subject": "algebra", "choices": ["a", "b"], "answer": 0}
                for i in range(3)
            ],
            datasets._gsm8k: [{"question": f"Q{i}", "answer": f"r\n#### {i}"} for i in range(3)],
            datasets._truthfulqa: [
                {"question": f"Q{i}", "best_answer": "a", "correct_answers": []} for i in range(3)
            ],
            datasets._codesearchnet: [
                {
                    "func_name": "f",
                    "func_code_string": f"def f{i}(): pass",
                    "func_documentation_string": "d",
                }
                for i in range(3)
            ],
        }
        for adapter, batch in rows.items():
            ids = {c.id for c in (adapter(r) for r in batch) if c}
            assert len(ids) == len(batch), f"{adapter.__name__} collided: {ids}"

    def test_bfcl_names_are_deduplicated(self) -> None:
        """The function name arrives twice — from the schema and from the gold call.

        Counting it twice inflated support recall, and CI now gates on that number.
        """
        case = datasets._bfcl(
            {
                "id": "x",
                "question": [[{"content": "q"}]],
                "function": [{"name": "calc", "parameters": {}}],
                "ground_truth": [{"calc": {"base": [1]}}],
            }
        )
        assert case is not None
        assert case.support == sorted(set(case.support)), "duplicate golds inflate recall"
        assert case.support.count("calc") == 1

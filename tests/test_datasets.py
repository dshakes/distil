"""Public-benchmark loader — transport, cache, and honest failure.

No test here touches the network: the rows API is stubbed and the cache is redirected
to a tmp DISTIL_HOME. What is pinned is the behaviour that makes a published number
trustworthy — deterministic ordering, an atomic cache, retry only on transient
failures, and a loud raise instead of a short result on anything unexpected.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from distil import datasets as ds


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


def _hotpot_row(idx: int) -> dict[str, Any]:
    return {
        "id": f"case{idx}",
        "question": "Who directed it?",
        "answer": "Tim Burton",
        "supporting_facts": {"title": ["Ed Wood"], "sent_id": [1]},
        "context": {
            "title": ["Ed Wood", "Distractor"],
            "sentences": [
                ["Ed Wood is a film. ", "It was directed by Tim Burton. "],
                ["Unrelated filler text. "],
            ],
        },
    }


def _squad_row(idx: int, *, answerable: bool = True) -> dict[str, Any]:
    return {
        "id": f"sq{idx}",
        "title": "Normans",
        "context": "The Normans came from France in the 10th century.",
        "question": "Where did the Normans come from?",
        "answers": {"text": ["France"], "answer_start": [28]} if answerable else {"text": []},
    }


def _stub_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> list[str]:
    """Serve `rows` through a fake rows API, recording each URL requested."""
    seen: list[str] = []

    def fake_get(url: str) -> dict[str, Any]:
        seen.append(url)
        offset = int(url.split("offset=")[1].split("&")[0])
        length = int(url.split("length=")[1].split("&")[0])
        page = rows[offset : offset + length]
        return {"rows": [{"row": r} for r in page]}

    monkeypatch.setattr(ds, "_get", fake_get)
    return seen


def test_hotpotqa_resolves_gold_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    """supporting_facts is (title, sentence index); the loader must resolve it to the
    actual sentence text, so retention is checked against text we can see."""
    _stub_rows(monkeypatch, [_hotpot_row(0)])
    (case,) = ds.load("hotpotqa", 1)
    assert case.answer == "Tim Burton"
    assert case.support == ["It was directed by Tim Burton."]
    assert len(case.docs) == 2  # the distractor is kept: pruning it is distil's job
    assert case.docs[0][0] == "Ed Wood"


def test_hotpotqa_skips_unresolvable_supporting_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _hotpot_row(0)
    row["supporting_facts"] = {"title": ["Nonexistent"], "sent_id": [99]}
    _stub_rows(monkeypatch, [row])
    (case,) = ds.load("hotpotqa", 1)
    assert case.support == []  # dropped, not crashed, not fabricated


def test_squad_marks_unanswerable(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_rows(monkeypatch, [_squad_row(0), _squad_row(1, answerable=False)])
    answerable, unanswerable = ds.load("squad", 2)
    assert answerable.answerable is True and answerable.answer == "France"
    assert unanswerable.answerable is False and unanswerable.answer == ""


def test_paging_respects_the_server_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rows API rejects length > 100, so 150 cases must arrive as two pages."""
    seen = _stub_rows(monkeypatch, [_hotpot_row(i) for i in range(150)])
    cases = ds.load("hotpotqa", 150)
    assert len(cases) == 150
    assert len(seen) == 2
    assert "length=100" in seen[0] and "offset=0" in seen[0]
    assert "offset=100" in seen[1] and "length=50" in seen[1]


def test_order_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A published number must be reproducible: same n, same cases, same order."""
    _stub_rows(monkeypatch, [_hotpot_row(i) for i in range(20)])
    first = [c.id for c in ds.load("hotpotqa", 10)]
    second = [c.id for c in ds.load("hotpotqa", 10)]
    assert first == second == [f"case{i}" for i in range(10)]


def test_cache_makes_the_second_run_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_rows(monkeypatch, [_hotpot_row(i) for i in range(5)])
    ds.load("hotpotqa", 5)
    assert len(seen) == 1
    ds.load("hotpotqa", 5)  # served from cache
    assert len(seen) == 1
    assert ds.load("hotpotqa", 5, offline=True)[0].id == "case0"


def test_offline_without_enough_cache_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_rows(monkeypatch, [_hotpot_row(i) for i in range(3)])
    ds.load("hotpotqa", 3)
    with pytest.raises(ds.DatasetUnavailable, match="offline mode has 3 cached rows, need 10"):
        ds.load("hotpotqa", 10, offline=True)


def test_truncated_cache_line_is_survivable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupted write must not poison the cache: keep the intact prefix."""
    _stub_rows(monkeypatch, [_hotpot_row(i) for i in range(4)])
    ds.load("hotpotqa", 4)
    path = tmp_path / "datasets" / "hotpotqa.jsonl"
    path.write_text(path.read_text(encoding="utf-8")[:-40] + "\n{partial", encoding="utf-8")
    assert len(ds._read_cache("hotpotqa")) < 4  # prefix kept, junk line ignored


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ds.DatasetUnavailable, match="unknown dataset"):
        ds.load("nope", 1)


def test_nonpositive_n_raises() -> None:
    with pytest.raises(ds.DatasetUnavailable, match="n must be positive"):
        ds.load("squad", 0)


def test_empty_upstream_raises_not_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure mode that must never be silent: measuring zero cases."""
    monkeypatch.setattr(ds, "_get", lambda url: {"rows": []})
    with pytest.raises(ds.DatasetUnavailable, match="no rows"):
        ds.load("hotpotqa", 10)


def test_shape_change_upstream_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If every row fails to adapt, the schema moved — say so, do not report 0 cases."""
    monkeypatch.setattr(ds, "_get", lambda url: {"rows": [{"row": {"unexpected": 1}}]})
    with pytest.raises(ds.DatasetUnavailable, match="no rows survived adaptation"):
        ds.load("hotpotqa", 1)


def test_permanent_http_error_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def boom(request: Any, timeout: float = 0) -> Any:
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

    monkeypatch.setattr(ds.urllib.request, "urlopen", boom)
    with pytest.raises(ds.DatasetUnavailable, match="HTTP 404"):
        ds.load("hotpotqa", 1)
    assert calls["n"] == 1  # 404 is not retried


def test_transient_error_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    payload = json.dumps({"rows": [{"row": _hotpot_row(0)}]}).encode()

    class _Response:
        def read(self) -> bytes:
            return payload

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    def flaky(request: Any, timeout: float = 0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 503, "Unavailable", None, None)
        return _Response()

    monkeypatch.setattr(ds.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    assert ds.load("hotpotqa", 1)[0].id == "case0"
    assert calls["n"] == 2


def test_exhausted_retries_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_down(request: Any, timeout: float = 0) -> Any:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(ds.urllib.request, "urlopen", always_down)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    with pytest.raises(ds.DatasetUnavailable, match="unreachable after 3 attempts"):
        ds.load("squad", 1)


def test_non_dict_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def read(self) -> bytes:
            return b"[1, 2, 3]"

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(ds.urllib.request, "urlopen", lambda r, timeout=0: _Response())
    with pytest.raises(ds.DatasetUnavailable, match="unexpected payload"):
        ds.load("squad", 1)


def test_short_page_ends_paging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream returning fewer rows than asked means end-of-split, not an error."""
    _stub_rows(monkeypatch, [_hotpot_row(i) for i in range(7)])
    assert len(ds.load("hotpotqa", 100)) == 7


def test_blank_cache_lines_are_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_rows(monkeypatch, [_hotpot_row(0)])
    ds.load("hotpotqa", 1)
    path = tmp_path / "datasets" / "hotpotqa.jsonl"
    path.write_text("\n\n" + path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(ds._read_cache("hotpotqa")) == 1


def test_bad_sent_id_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer sentence index must be dropped, not raise."""
    row = _hotpot_row(0)
    row["supporting_facts"] = {"title": ["Ed Wood"], "sent_id": ["not-an-int"]}
    _stub_rows(monkeypatch, [row])
    assert ds.load("hotpotqa", 1)[0].support == []


def test_hotpot_row_missing_answer_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    good, bad = _hotpot_row(0), _hotpot_row(1)
    bad["answer"] = ""
    _stub_rows(monkeypatch, [good, bad])
    assert [c.id for c in ds.load("hotpotqa", 2)] == ["case0"]


def test_hotpot_mismatched_context_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    good, bad = _hotpot_row(0), _hotpot_row(1)
    bad["context"] = {"title": ["A", "B"], "sentences": [["only one"]]}
    _stub_rows(monkeypatch, [good, bad])
    assert [c.id for c in ds.load("hotpotqa", 2)] == ["case0"]


def test_squad_row_without_context_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    good, bad = _squad_row(0), _squad_row(1)
    bad["context"] = "   "
    _stub_rows(monkeypatch, [good, bad])
    assert [c.id for c in ds.load("squad", 2)] == ["sq0"]


def test_available_lists_every_spec() -> None:
    names = {name for name, _ in ds.available()}
    assert names == set(ds.SPECS)
    assert all(description for _, description in ds.available())

"""Loop / re-fetch detector: every wire shape, the digest attribution, and the keep-policy backoff."""

from __future__ import annotations

import json
from typing import Any

from distil.learn import ExpandStats, keep_predicate, signature
from distil.loops import detect, feed_stats

STUB = "head\n<< +40 lines, handle=abcd1234 >>"
BODY = "line one\nline two\nline three"


def _anthropic(rounds: list[tuple[str, dict[str, Any], str]]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for i, (name, args, result) in enumerate(rounds):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": args}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": result}],
            }
        )
    return msgs


def _chat(rounds: list[tuple[str, dict[str, Any], str]]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for i, (name, args, result) in enumerate(rounds):
        fn = {"name": name, "arguments": json.dumps(args)}
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": fn}],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})
    return msgs


def _responses(rounds: list[tuple[str, dict[str, Any], str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [{"type": "message", "role": "user", "content": "go"}]
    for i, (name, args, result) in enumerate(rounds):
        items.append(
            {
                "type": "function_call",
                "call_id": f"r{i}",
                "name": name,
                "arguments": json.dumps(args),
            }
        )
        items.append({"type": "function_call_output", "call_id": f"r{i}", "output": result})
    return items


def _gemini(rounds: list[tuple[str, dict[str, Any], str]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "go"}]}]
    for name, args, result in rounds:
        contents.append(
            {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}
        )
        contents.append(
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": name, "response": {"output": result}}}],
            }
        )
    return contents


SHAPES = [_anthropic, _chat, _responses, _gemini]


def test_identical_reread_rerun_per_shape() -> None:
    for build in SHAPES:
        rep = detect(
            build(
                [
                    ("Read", {"file_path": "a.py"}, BODY),
                    ("Read", {"file_path": "a.py", "offset": 5}, BODY),  # same file, other range
                    ("Bash", {"command": "pytest -x"}, BODY),
                    ("Bash", {"command": "pytest -x"}, BODY),
                ]
            )
        )
        # Only the newest round counts: the second pytest run, which is a rerun and identical.
        assert (rep.calls, rep.repeats, rep.rerun, rep.reread, rep.identical) == (1, 1, 1, 0, 1), (
            build
        )
        assert rep.after_digest == 0 and rep.by_tool == {"Bash": 1}


def test_reread_with_different_range_is_not_identical() -> None:
    for build in SHAPES:
        rep = detect(
            build(
                [
                    ("Read", {"file_path": "a.py"}, BODY),
                    ("Read", {"file_path": "a.py", "offset": 9}, BODY),
                ]
            )
        )
        assert (rep.repeats, rep.reread, rep.identical) == (1, 1, 0), build


def test_refetch_after_digest_is_attributed_and_hands_back_handle() -> None:
    for build in SHAPES:
        rep = detect(
            build([("Bash", {"command": "ls -R"}, STUB), ("Bash", {"command": "ls -R"}, BODY)])
        )
        assert rep.after_digest == 1 and rep.handles == ["abcd1234"], build


def test_repeat_of_undigested_result_is_not_attributed() -> None:
    rep = detect(_anthropic([("Bash", {"command": "ls"}, BODY), ("Bash", {"command": "ls"}, BODY)]))
    assert rep.repeats == 1 and rep.after_digest == 0 and rep.handles == []


def test_old_repeats_and_expand_calls_are_not_recounted() -> None:
    old = [("Bash", {"command": "ls"}, BODY), ("Bash", {"command": "ls"}, BODY)]
    rep = detect(_anthropic([*old, ("Read", {"file_path": "new.py"}, BODY)]))
    assert (rep.calls, rep.repeats) == (1, 0)
    rep = detect(
        _anthropic(
            [("Bash", {"command": "ls"}, STUB), ("distil_expand", {"handle": "abcd1234"}, BODY)]
        )
    )
    assert rep.repeats == 0


def test_no_calls_and_junk_are_harmless() -> None:
    assert detect([]).as_record() is None
    assert detect([None, "x", {"role": "user", "content": "hi"}, {"parts": [None]}]).calls == 0  # type: ignore[list-item]


def test_record_is_content_free() -> None:
    rec = detect(
        _anthropic(
            [("Read", {"file_path": "secret.py"}, STUB), ("Read", {"file_path": "secret.py"}, BODY)]
        )
    ).as_record()
    assert rec is not None
    assert "secret" not in json.dumps(rec) and "abcd1234" not in json.dumps(rec)


def test_backoff_keeps_content_type_after_repeated_refetches() -> None:
    """Re-fetches of a digested type feed ExpandStats, so learn.keep_predicate stops digesting it."""
    stats = ExpandStats()
    original = "\n".join(f'{{"k{i}": {i}}}' for i in range(60))  # json:l
    sig = signature(original)
    for _ in range(10):
        stats.record_digest(sig)
    assert not keep_predicate(stats)(original)  # nothing re-fetched yet: still digested
    rep = detect(
        _chat(
            [("Bash", {"command": "cat x.json"}, STUB), ("Bash", {"command": "cat x.json"}, BODY)]
        )
    )
    booked = 0
    for _ in range(3):  # three such rounds
        booked += feed_stats(rep, stats, lambda h: sig if h == "abcd1234" else None)
    assert booked == 3
    assert keep_predicate(stats)(original)  # 3/10 >= 0.25: kept byte-exact from now on
    assert not keep_predicate(stats)("plain prose\nonly")  # other types are untouched


def test_feed_stats_skips_unresolvable_handles() -> None:
    rep = detect(_anthropic([("Bash", {"command": "ls"}, STUB), ("Bash", {"command": "ls"}, BODY)]))
    stats = ExpandStats()
    assert feed_stats(rep, stats, lambda h: None) == 0
    assert stats.expanded == {}


def test_hook_fires_for_responses_and_gemini_with_the_forwarded_list() -> None:
    """serve_core handed on_compressed nothing for Gemini and was silent for Responses."""
    from distil import serve_core

    cases = [
        ("/v1/responses", {"model": "m", "input": _responses([("Bash", {"command": "ls"}, BODY)])}),
        (
            "/v1beta/models/g:generateContent",
            {"contents": _gemini([("Bash", {"command": "ls"}, BODY)])},
        ),
    ]
    for path, body in cases:
        seen: list[Any] = []
        serve_core.compress_or_forward(
            body,
            path,
            count=lambda m: 1,
            verbatim=False,
            expand=False,
            mode="digest",
            on_compressed=lambda kind, orig, comp, store: seen.append((kind, comp)),
        )
        assert len(seen) == 1, path
        assert detect(seen[0][1]).calls == 1, path

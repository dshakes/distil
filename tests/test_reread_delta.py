"""Re-read delta — ADR 0010.

A second read of a file drops the line run an earlier read already delivered, and points
at it. The tests here are about the three things that make that safe rather than merely
smaller: the referenced bytes are still in the conversation, the stub is reversible, and
the rendering does not drift across turns.
"""

from __future__ import annotations

import json
from typing import Any

from distil.adapters.anthropic import compress_messages
from distil.compress import rereaddelta
from distil.compress.provenance import edit_quotes, observed_view, quote_hazard

_HANDLERS = 60


def _module(marker: str = "") -> str:
    """A file of many same-shaped, individually-identifiable blocks (3 lines each)."""
    return "\n".join(
        f"def handler_{i}(request):  # {marker}\n"
        f"    payload = request.json()\n"
        f"    return {{'ok': True, 'n': {i}, 'payload': payload}}"
        for i in range(_HANDLERS)
    )


def _handler(i: int, marker: str = "") -> str:
    return (
        f"def handler_{i}(request):  # {marker}\n"
        f"    payload = request.json()\n"
        f"    return {{'ok': True, 'n': {i}, 'payload': payload}}"
    )


def _read(tid: str, path: str = "/app/handlers.py") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": "Read", "input": {"file_path": path}}],
    }


def _shell(tid: str, command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": command}}],
    }


def _result(tid: str, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def _edit(tid: str, old: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tid,
                "name": "Edit",
                "input": {
                    "file_path": "/app/handlers.py",
                    "old_string": old,
                    "new_string": old + "  # patched",
                },
            }
        ],
    }


def _session(first: str, second: str, *, reader=_read) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "refactor the handlers"},
        reader("r1"),
        _result("r1", first),
        reader("r2"),
        _result("r2", second),
    ]


def _text_of(msg: dict[str, Any]) -> str:
    block = msg["content"][0]
    content = block["content"]
    return content if isinstance(content, str) else content[0]["text"]


# --------------------------------------------------------------------------- planner


def test_a_run_shorter_than_the_minimum_is_not_worth_a_stub() -> None:
    base = [f"line {i}" for i in range(40)]
    lines = [f"other {i}" for i in range(20)] + base[:4]
    assert rereaddelta._longest_common_run(base, lines) is None
    # A block that is itself shorter than the minimum never gets that far.
    assert rereaddelta._longest_common_run(base, base[:4]) is None


def test_the_edge_margin_is_held_back_at_an_internal_cut() -> None:
    """A run that ends inside the block gives back EDGE_MARGIN lines, so a quote
    straddling the cut still occurs contiguously in the base."""
    base = [f"line {i}" for i in range(100)]
    lines = base[:60] + [f"new {i}" for i in range(20)]
    found = rereaddelta._longest_common_run(base, lines)
    assert found is not None
    assert found.start == 0, "a run reaching the block's own first line needs no margin"
    assert found.end == 60 - rereaddelta.EDGE_MARGIN


def test_both_cuts_take_a_margin_when_the_run_is_interior() -> None:
    base = [f"line {i}" for i in range(30, 90)]
    lines = [f"line {i}" for i in range(100)]
    found = rereaddelta._longest_common_run(base, lines)
    assert found is not None
    assert (found.start, found.end) == (30 + rereaddelta.EDGE_MARGIN, 90 - rereaddelta.EDGE_MARGIN)
    # The base indices track the trim, or the stub would name the wrong lines.
    assert found.base_end - found.base_start == found.end - found.start


def test_a_shell_read_may_not_serve_as_a_base() -> None:
    """Only name-keyed reads are exempt unconditionally. A shell read's exemption is
    conditional on not being superseded — and a re-read of the same path is exactly what
    supersedes it, so referencing one could point at bytes that later digest away."""
    text = "\n".join(f"line {i}" for i in range(60))
    blocks = [
        rereaddelta.ReadBlock("a", "/f.py", text, base_ok=False),
        rereaddelta.ReadBlock("b", "/f.py", text, base_ok=True),
    ]
    assert rereaddelta.plan(blocks) == {}


def test_references_never_chain() -> None:
    """An elided block is not itself a base, so every stub points at literal bytes."""
    text = "\n".join(f"line {i}" for i in range(80))
    blocks = [rereaddelta.ReadBlock(f"r{i}", "/f.py", text, base_ok=True) for i in range(3)]
    planned = rereaddelta.plan(blocks)
    assert set(planned) == {"r1", "r2"}, "both re-reads elide, and both point at r0"


# --------------------------------------------------------------------------- adapter


def test_an_offset_reread_elides_only_the_lines_already_delivered() -> None:
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    sent, _store = compress_messages(_session(head, tail))

    first, second = _text_of(sent[2]), _text_of(sent[4])
    assert first == head, "the base read must stay byte-exact"
    assert "«distil-reread" in second
    # Everything past the overlap is still there verbatim: handler_50 is well beyond it.
    assert _handler(50, "MARK") in second
    assert len(second) < len(tail)


def test_the_stub_round_trips_through_the_restore_store() -> None:
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    sent, store = compress_messages(_session(head, tail))

    second = _text_of(sent[4])
    handle = second.split("handle=", 1)[1].split("»", 1)[0]
    recovered = store.expand(handle)
    assert recovered and recovered in tail, "expand must return the elided lines byte-exact"
    # Splicing the recovery back in reproduces the original block exactly.
    stub_line = next(ln for ln in second.split("\n") if ln.startswith("«distil-reread"))
    assert second.replace(stub_line + "\n", recovered) == tail


def test_a_quote_inside_the_elided_run_still_exists_in_the_base() -> None:
    """The guarantee is about the conversation, not about one block."""
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    msgs = _session(head, tail) + [_edit("e1", _handler(22, "MARK"))]
    sent, _store = compress_messages(msgs)

    assert "distil-reread" in json.dumps(sent), "the delta did not fire — fixture is stale"
    survived, lost = quote_hazard(edit_quotes(msgs), observed_view(sent))
    assert (survived, lost) == (1, 0)


def test_a_quote_straddling_the_cut_survives_on_the_edge_margin() -> None:
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    # handler_29/handler_30 sit either side of where the run is trimmed.
    quote = _handler(29, "MARK") + "\n" + _handler(30, "MARK")
    msgs = _session(head, tail) + [_edit("e1", quote)]
    sent, _store = compress_messages(msgs)

    assert "distil-reread" in json.dumps(sent), "the delta did not fire — fixture is stale"
    survived, lost = quote_hazard(edit_quotes(msgs), observed_view(sent))
    assert (survived, lost) == (1, 0)


def test_a_shell_reread_of_a_read_tool_base_is_still_elided() -> None:
    """Bases must be name-keyed; TARGETS need not be. `Read` then `cat` is a real shape."""
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    msgs = [
        {"role": "user", "content": "refactor"},
        _read("r1"),
        _result("r1", head),
        _shell("r2", "cat /app/handlers.py"),
        _result("r2", tail),
    ]
    sent, _store = compress_messages(msgs)
    assert "«distil-reread" in _text_of(sent[4])


def test_verbatim_mode_emits_no_reference() -> None:
    """Verbatim's contract is that the model sees the content in place. A cross-block
    reference is not that, and the SDK wrapper has no expand loop to resolve it."""
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    sent, _store = compress_messages(_session(head, tail), verbatim=True)
    assert "distil-reread" not in json.dumps(sent)


def test_two_different_files_are_never_cross_referenced() -> None:
    text = _module("MARK")
    msgs = [
        {"role": "user", "content": "refactor"},
        _read("r1", "/app/a.py"),
        _result("r1", text),
        _read("r2", "/app/b.py"),
        _result("r2", text),
    ]
    sent, _store = compress_messages(msgs)
    assert "distil-reread" not in json.dumps(sent), "identical bytes, different files"


# --------------------------------------------------------------------------- cache contract


def test_the_stub_does_not_drift_as_the_conversation_grows() -> None:
    """ADR 0008 (a), for this transform: the plan is a pure function of the message
    PREFIX, so a block encodes to the same bytes on every later turn — including under the
    client shape that pins its newest turn and caches the whole history."""
    lines = _module("MARK").split("\n")
    head, tail = "\n".join(lines[:120]), "\n".join(lines[60:])
    base = _session(head, tail)

    def _turn(extra: int) -> list[dict[str, Any]]:
        msgs = [dict(m) for m in base]
        for k in range(extra):
            msgs += [
                _shell(f"b{k}", "pytest -q"),
                _result(f"b{k}", "\n".join(f"test_{i} PASSED" for i in range(30))),
            ]
        # What Claude Code does: pin the newest turn, so the whole history is cached.
        msgs[-1] = {
            **msgs[-1],
            "content": [{**msgs[-1]["content"][0], "cache_control": {"type": "ephemeral"}}],
        }
        return msgs

    def _keys(msgs: list[dict[str, Any]]) -> list[str]:
        return [json.dumps(m, separators=(",", ":"), ensure_ascii=False) for m in msgs]

    previous: tuple[list[str], list[str]] | None = None
    for extra in range(4):
        msgs = _turn(extra)
        sent, _store = compress_messages(msgs)
        current = (_keys(msgs), _keys(sent))
        if previous is not None:
            # Only messages the client re-sent byte-identical are covered by the contract:
            # moving its own marker is the client rewriting its own history (clause (d)).
            drift = [
                i
                for i in range(len(previous[0]))
                if previous[0][i] == current[0][i] and previous[1][i] != current[1][i]
            ]
            assert not drift, f"fully-cached session drifted at {drift}"
        previous = current
    assert previous is not None and "distil-reread" in "".join(previous[1]), "the delta never fired"


def test_the_validate_battery_actually_exercises_the_delta() -> None:
    """`distil validate` gained re-read cases. If none of them still produces a stub the
    invariant is being asserted over a transform that no longer runs."""
    from distil.harness import _cases

    fired = [
        name for name, msgs in _cases() if "distil-reread" in json.dumps(compress_messages(msgs)[0])
    ]
    assert fired, "no validate case reaches the re-read delta any more"


# --------------------------------------------------------------------------- Codex quotes


def _patch(path: str, context: list[str], removed: str, added: str) -> str:
    body = "\n".join(f" {line}" for line in context)
    return (
        "*** Begin Patch\n"
        f"*** Update File: {path}\n"
        "@@\n"
        f"{body}\n"
        f"-{removed}\n"
        f"+{added}\n"
        "*** End Patch\n"
    )


def test_apply_patch_pre_image_runs_through_added_lines() -> None:
    """A `+` line is not part of what the patcher matches, so the run continues across it."""
    from distil.compress.provenance import patch_quotes

    patch = (
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@ def handle():\n"
        " keep one\n"
        "-drop me\n"
        "+add me\n"
        " keep two\n"
        "*** End Patch\n"
    )
    assert patch_quotes(patch) == ["keep one\ndrop me\nkeep two"]


def test_an_added_file_hunk_has_no_pre_image() -> None:
    from distil.compress.provenance import patch_quotes

    patch = "*** Begin Patch\n*** Add File: new.py\n+import os\n+import sys\n*** End Patch\n"
    assert patch_quotes(patch) == []


def test_a_hunk_header_breaks_the_run() -> None:
    from distil.compress.provenance import patch_quotes

    patch = (
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@ first\n a\n b\n"
        "@@ second\n c\n d\n"
        "*** End Patch\n"
    )
    assert patch_quotes(patch) == ["a\nb", "c\nd"]


def test_the_responses_path_now_reports_a_quote_hazard() -> None:
    """1.51 left this as a known follow-up: the counter existed on the Messages path only,
    so Codex traffic was reported to `distil dissect` as carrying no edits at all."""
    from distil.adapters.anthropic import take_quote_hazard
    from distil.adapters.openai import compress_responses_input

    source = _module("MARK")
    lines = source.split("\n")
    # The pre-image must be a CONTIGUOUS slice of the file, or the patch would not apply
    # and a "lost" verdict would be correct rather than a bug.
    context, removed = lines[:5], lines[5]
    items: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "patch it"}],
        },
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "shell",
            "arguments": '{"command": ["cat", "/app/handlers.py"]}',
        },
        {"type": "function_call_output", "call_id": "c1", "output": source},
        {
            "type": "custom_tool_call",
            "call_id": "c2",
            "name": "apply_patch",
            "input": _patch("/app/handlers.py", context, removed, removed + "  # patched"),
        },
    ]
    compress_responses_input(items)
    assert take_quote_hazard() == {"survived": 1, "lost": 0}


def test_the_responses_counter_is_absent_when_no_edit_was_made() -> None:
    from distil.adapters.anthropic import take_quote_hazard
    from distil.adapters.openai import compress_responses_input

    items: list[dict[str, Any]] = [
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "\n".join(f"log {i}" for i in range(40)),
        },
    ]
    compress_responses_input(items)
    assert take_quote_hazard() is None


def test_the_patch_envelope_cannot_satisfy_its_own_quote() -> None:
    """`observed_view` must drop Responses model-output ITEMS, not just assistant messages —
    otherwise the patch body quotes the file to itself and the check can only pass."""
    from distil.compress.provenance import observed_view, quote_hazard, response_edit_quotes

    patch = _patch("/app/handlers.py", _module("MARK").split("\n")[:6], "a", "b")
    items = [{"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": patch}]
    survived, lost = quote_hazard(response_edit_quotes(items), observed_view(items))
    assert (survived, lost) == (0, 1)


def test_an_oversized_pair_is_skipped_rather_than_matched() -> None:
    """difflib is O(n*m); a flat product cap keeps one enormous re-read from stalling a
    request. Skipping only ever costs tokens."""
    big = [f"line {i}" for i in range(2001)]
    assert rereaddelta._longest_common_run(big, list(big)) is None
    assert rereaddelta._longest_common_run(big[:1000], big[:1000]) is not None


def test_a_run_that_survives_the_margin_by_too_little_is_dropped() -> None:
    """Both cuts take a margin, so an interior run needs > 2 * EDGE_MARGIN + MIN_RUN lines
    to be worth anything at all."""
    shared = [f"line {i}" for i in range(2 * rereaddelta.EDGE_MARGIN + 4)]
    base = [f"before {i}" for i in range(10)] + shared + [f"after {i}" for i in range(10)]
    lines = [f"other {i}" for i in range(10)] + shared + [f"tail {i}" for i in range(10)]
    assert rereaddelta._longest_common_run(base, lines) is None


def test_a_stub_costing_more_than_the_lines_it_removes_is_not_emitted() -> None:
    """distil never inflates. Eight one-character lines are cheaper than any reference."""
    from distil.adapters.anthropic import RestoreStore, _apply_reread

    text = "\n".join("x" for _ in range(40))
    elision = rereaddelta.Elision("/f.py", 0, 8, 0, 8)
    assert _apply_reread(text, elision, RestoreStore()) == text

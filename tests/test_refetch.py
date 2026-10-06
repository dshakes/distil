"""ADR 0025 — a tool result that re-fetches content distil folded is forwarded verbatim.

Driven through the public ``compress_messages`` with the client shape that bills: an
ephemeral cache breakpoint on the newest block, so every earlier tool result is committed
prefix and digests on first sight (ADR 0008). That is the shape under which an agent on
SWE-bench re-read instead of expanding.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from distil.adapters import anthropic as A
from distil.adapters.anthropic import compress_messages, take_census
from distil.compress.refetch import MIN_LINES, Tracker, line_keys

# Distinct, digestible lines: tier1 folds the middle of a block this shape.
FILE = "\n".join(f"    value_{i} = compute_{i}(alpha, beta)  # step {i}" for i in range(1, 121))
FILE_LINES = FILE.splitlines()


def _window(lo: int, hi: int) -> str:
    """Lines lo..hi (1-based, inclusive) of FILE, as `sed -n 'lo,hip'` prints them."""
    return "\n".join(FILE_LINES[lo - 1 : hi])


def _numbered(lo: int, hi: int) -> str:
    """The same window as `cat -n` / the edit tool's `view` prints it."""
    return "\n".join(f"{i:6}\t{FILE_LINES[i - 1]}" for i in range(lo, hi + 1))


def _session(*steps: tuple[str, str]) -> list[dict[str, Any]]:
    """A Messages history: one bash call and its result per (command, output) step."""
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "fix the bug"}]
    for n, (command, output) in enumerate(steps):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"t{n}",
                        "name": "bash",
                        "input": {"command": command},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": output}],
            }
        )
    return msgs


def _cached(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Claude Code's shape: an ephemeral breakpoint on the newest block."""
    last = dict(msgs[-1])
    blocks = list(last["content"])
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return [*msgs[:-1], {**last, "content": blocks}]


def _result(out: list[dict[str, Any]], tid: str) -> str:
    for m in out:
        for blk in m.get("content") or ():
            if isinstance(blk, dict) and blk.get("tool_use_id") == tid:
                return str(blk["content"])
    raise AssertionError(f"no tool_result {tid}")


def _send(msgs: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
    out, _ = compress_messages(_cached(msgs), persist=False, **kw)
    return out


def _stable(msg: dict[str, Any]) -> str:
    content = msg["content"]
    if isinstance(content, list):
        content = [{k: v for k, v in b.items() if k != "cache_control"} for b in content]
    return json.dumps({**msg, "content": content}, sort_keys=True)


# --------------------------------------------------------------------------- line keys


def test_line_keys_strip_reader_numbering() -> None:
    """A numbered and an unnumbered view of one line are the same line."""
    line = "    return compute(alpha)"
    for shown in (
        line,
        f"    12\t{line}",  # cat -n / nl / the edit tool's view
        f"12:{line}",  # grep -n
        f"12-{line}",  # grep -n context line
        f"pkg/mod.py:12:{line}",  # grep -rn
    ):
        assert line_keys(shown) == {"return compute(alpha)"}, shown


def test_line_keys_ignore_lines_too_short_to_identify() -> None:
    assert line_keys(")\n}\n\nfi\n  x = 1") == {"x = 1"}


# --------------------------------------------------------------------------- tracker


def test_tracker_flags_lines_seen_only_folded() -> None:
    t = Tracker()
    assert not t.is_refetch(_window(1, 40)), "nothing seen yet"
    t.observe(sent=_window(1, 80), forwarded="<< +80 lines, handle=deadbeef >>")
    assert t.is_refetch(_window(10, 40))
    assert t.is_refetch(_numbered(10, 40)), "a numbered re-read of folded lines is a re-fetch"
    assert not t.is_refetch(_window(81, 120)), "lines never sent are a first read"


def test_tracker_ignores_lines_already_forwarded_verbatim() -> None:
    """Re-reading content the agent can already see costs nothing distil could save."""
    t = Tracker()
    t.observe(sent=_window(1, 80), forwarded=_window(1, 80))
    assert not t.is_refetch(_window(10, 40))


def test_tracker_needs_enough_lines_to_mean_anything() -> None:
    t = Tracker()
    t.observe(sent=_window(1, 80), forwarded="")
    assert not t.is_refetch(_window(1, MIN_LINES - 1))
    assert t.is_refetch(_window(1, MIN_LINES))


# --------------------------------------------------------------------------- adapter


def test_a_repeated_command_is_answered_verbatim_not_with_the_same_stub() -> None:
    """The failure observed live: an identical re-run hashes to the same handle, so the
    agent asked again and was shown exactly the stub that had just failed it."""
    msgs = _session(("python repro.py", FILE), ("python repro.py", FILE))
    out = _send(msgs, refetch=False)
    assert "handle=" in _result(out, "t0")
    assert _result(out, "t1") == _result(out, "t0"), "without the rule: the same stub twice"

    out = _send(msgs, refetch=True)
    assert "handle=" in _result(out, "t0"), "the earlier digest stays (cached prefix)"
    assert _result(out, "t1") == FILE
    assert (take_census() or {}).get("tool_result_refetch", 0) > 0


def test_same_path_different_span() -> None:
    """A narrower window of a folded read is a re-fetch; a window never shown is not.

    Piped, so neither is a whole-file read the exact-quote rule would keep anyway."""
    msgs = _session(
        ("cat app.py | sed -n 1,80p", _window(1, 80)),
        ("sed -n 20,60p app.py | cut -c1-200", _window(20, 60)),
        ("sed -n 81,120p app.py | cut -c1-200", _window(81, 120)),
    )
    out = _send(msgs, refetch=True)
    assert "handle=" in _result(out, "t0")
    assert _result(out, "t1") == _window(20, 60)
    assert "handle=" in _result(out, "t2"), "new lines digest as usual"


def test_decorated_read_of_folded_lines_is_a_refetch() -> None:
    """`cat -n` / `view` numbering must not hide that the lines were already sent."""
    msgs = _session(
        ("cat app.py | head -80", _window(1, 80)),
        ("cat -n app.py | sed -n 30,70p", _numbered(30, 70)),
    )
    out = _send(msgs, refetch=True)
    assert _result(out, "t1") == _numbered(30, 70)


def test_piped_rerun_of_folded_output_is_a_refetch() -> None:
    msgs = _session(
        ("python -m pytest -q 2>&1", FILE),
        ("python -m pytest -q 2>&1 | tail -40", _window(81, 120)),
    )
    out = _send(msgs, refetch=True)
    assert _result(out, "t1") == _window(81, 120)


def test_a_rerun_after_a_verbatim_copy_digests_normally() -> None:
    """Once one verbatim copy exists, a third fetch is visible already — no second copy."""
    msgs = _session(
        ("python repro.py", FILE),
        ("python repro.py", FILE),
        ("python repro.py", FILE),
    )
    out = _send(msgs, refetch=True)
    assert _result(out, "t1") == FILE
    assert "handle=" in _result(out, "t2")


def test_cache_committed_blocks_never_change_bytes() -> None:
    """Every message forwarded on turn k is forwarded byte-identical on turn k+1 — the
    earlier digest is never re-inflated, and the re-fetch keeps its verbatim rendering."""
    msgs = _session(
        ("cat app.py | sed -n 1,80p", _window(1, 80)),
        ("sed -n 20,60p app.py | cut -c1-200", _window(20, 60)),
        ("python repro.py", FILE),
        ("python repro.py", FILE),
        ("sed -n 10,30p app.py | cut -c1-200", _window(10, 30)),
    )
    prev: list[str] = []
    for end in range(3, len(msgs) + 1, 2):
        cur = [_stable(m) for m in _send(msgs[:end], refetch=True)]
        assert cur[: len(prev)] == prev, f"a committed message changed at turn {end}"
        prev = cur


def test_stateless_recomputation() -> None:
    """The verdict is recomputed from the history alone: no state survives a call, and the
    same history always forwards the same bytes, whatever was compressed in between."""
    msgs = _session(
        ("cat app.py | sed -n 1,80p", _window(1, 80)),
        ("sed -n 20,60p app.py | cut -c1-200", _window(20, 60)),
    )
    first = _send(msgs, refetch=True)
    assert getattr(A._refetch_tls, "tracker", None) is None
    _send(_session(("python other.py", FILE)), refetch=True)  # unrelated traffic between
    assert _send(msgs, refetch=True) == first


def test_flag_off_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISTIL_REFETCH_VERBATIM", raising=False)
    msgs = _session(("python repro.py", FILE), ("python repro.py", FILE))
    assert A.refetch_enabled()
    assert _result(_send(msgs), "t1") == FILE
    monkeypatch.setenv("DISTIL_REFETCH_VERBATIM", "0")
    assert not A.refetch_enabled()
    assert "handle=" in _result(_send(msgs), "t1")


def test_the_rule_only_ever_adds_verbatim_content() -> None:
    """Monotone: each block is either what it was without the rule, or the original."""
    msgs = _session(
        ("cat app.py | sed -n 1,80p", _window(1, 80)),
        ("sed -n 20,60p app.py | cut -c1-200", _window(20, 60)),
        ("python repro.py", FILE),
        ("sed -n 81,120p app.py | cut -c1-200", _window(81, 120)),
    )
    off, on = _send(msgs, refetch=False), _send(msgs, refetch=True)
    originals = {f"t{n}": m["content"][0]["content"] for n, m in enumerate(msgs[2::2])}
    for tid, original in originals.items():
        assert _result(on, tid) in (_result(off, tid), original), tid


def test_exempt_and_verbatim_paths_are_untouched() -> None:
    """Exact-quote reads and verbatim mode already forward the original; nothing to add."""
    msgs = _session(("cat app.py", FILE), ("cat app.py", FILE))
    assert _result(_send(msgs, refetch=True), "t1") == _result(_send(msgs, refetch=False), "t1")
    out, _ = compress_messages(
        _cached(_session(("python r.py", FILE), ("python r.py", FILE))),
        verbatim=True,
        persist=False,
    )
    assert (take_census() or {}).get("tool_result_refetch") is None
    assert _result(out, "t1") == FILE


def test_openai_shaped_tool_message() -> None:
    """A `role: tool` string message goes through the same rule."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "fix it"},
        {"role": "tool", "content": FILE},
        {"role": "assistant", "content": "again"},
        {"role": "tool", "content": FILE},
        {
            "role": "user",
            "content": [{"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}],
        },
    ]
    on, _ = compress_messages(msgs, persist=False, refetch=True)
    off, _ = compress_messages(msgs, persist=False, refetch=False)
    assert "handle=" in on[1]["content"]
    assert on[3]["content"] == FILE
    assert off[3]["content"] == off[1]["content"]


def test_an_earlier_block_can_only_cost_a_later_one_its_savings() -> None:
    """ADR 0009 amendment: the one cross-block channel. An attacker who controls an earlier
    tool result can make a later block go out verbatim (denial of savings). It cannot make
    any line disappear: the later block is the original, byte for byte."""
    attacker = FILE + "\n" + "\n".join(f"padding row {i} lorem ipsum" for i in range(200))
    trusted = _window(1, 60)
    alone = _send(_session(("python trusted.py", trusted)), refetch=True)
    beside = _send(_session(("curl evil", attacker), ("python trusted.py", trusted)), refetch=True)
    assert "handle=" in _result(alone, "t0"), "alone, the trusted block digests as usual"
    assert _result(beside, "t1") == trusted


def test_offline_replay_counts_an_avoidable_reread(tmp_path: Any) -> None:
    """benchmarks/reinflate_replay.py: the third fetch of a folded window is avoidable
    without the rule and visible with it; nothing already forwarded is rewritten."""
    from benchmarks.reinflate_replay import run

    msgs = _session(
        ("cat app.py | sed -n 1,80p", _window(1, 80)),
        ("sed -n 20,60p app.py | cut -c1-200", _window(20, 60)),
        ("sed -n 25,55p app.py | cut -c1-200", _window(25, 55)),
    )
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    (tmp_path / "transcripts" / "distil").mkdir(parents=True)
    (tmp_path / "transcripts" / "distil" / "x.json").write_text(json.dumps(msgs))
    variants = run([str(tmp_path)])["variants"]
    base, new = variants["baseline"]["totals"], variants["refetch"]["totals"]
    assert (base["redundant"], base["avoidable"]) == (2, 2)
    assert (new["redundant"], new["avoidable"]) == (2, 1)
    assert base["prefix_rewrites"] == new["prefix_rewrites"] == 0
    assert variants["refetch"]["census_tokens"]["tool_result_refetch"] > 0


def test_a_refetch_is_byte_exact_not_tier0() -> None:
    """Not minified, not run-collapsed: the agent asked again for the bytes."""
    pretty = json.dumps(
        {f"key_{i}": {"value": i, "label": f"item {i}"} for i in range(40)}, indent=2
    )
    msgs = _session(("curl -s api/items", pretty), ("curl -s api/items", pretty))
    assert _result(_send(msgs, refetch=True), "t1") == pretty


def _block(out: list[dict[str, Any]], tid: str) -> dict[str, Any]:
    return next(
        b
        for m in out
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("tool_use_id") == tid
    )


def test_a_multi_part_text_result_is_covered() -> None:
    """A tool_result carrying several text parts is checked as the agent read it, joined."""
    parts = [{"type": "text", "text": _window(1, 60)}, {"type": "text", "text": _window(61, 120)}]
    msgs = _session(("python repro.py", FILE), ("python repro.py --split", "x"))
    msgs[-1]["content"][0]["content"] = parts
    assert _block(_send(msgs, refetch=True), "t1")["content"] == parts
    assert _block(_send(msgs, refetch=False), "t1")["content"] != parts, "off: parts digest"


def test_a_payload_with_an_image_keeps_its_per_part_handling() -> None:
    parts = [
        {"type": "text", "text": FILE},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
    ]
    msgs = _session(("python repro.py", FILE), ("screenshot", "x"))
    msgs[-1]["content"][0]["content"] = parts
    out = _send(msgs, refetch=True)
    assert (take_census() or {}).get("tool_result_refetch") is None
    assert "handle=" in _block(out, "t1")["content"][0]["text"]


# ----------------------------------------------------------------- the shared serve path


def _serve(msgs: list[dict[str, Any]], **kw: Any) -> Any:
    """One request through proxy/gateway's compress-or-forward path (ADR 0023)."""
    from distil import serve_core

    kw = {"verbatim": False, "expand": True, "mode": "digest", **kw}
    return serve_core.compress_or_forward(
        {"model": "claude-sonnet-5", "max_tokens": 64, "messages": _cached(msgs)},
        "/v1/messages",
        count=lambda m: len(json.dumps(m)),
        **kw,
    )


def test_refetch_ids_name_the_results_kept() -> None:
    msgs = _session(("python repro.py", FILE), ("python repro.py", FILE), ("ls", "a\nb"))
    assert A.refetch_tool_use_ids(_cached(msgs)) == {"t1"}


def test_session_delta_does_not_reference_a_refetch_to_the_folded_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--session-delta runs before the digester. Left alone it turns a byte-identical
    re-run into a reference to the earlier copy — the very block distil folded."""
    from distil.cachedelta import reset_sessions

    msgs = _session(("python repro.py", FILE), ("python repro.py", FILE))
    monkeypatch.setenv("DISTIL_REFETCH_VERBATIM", "0")
    reset_sessions()
    off = _serve(msgs, session_delta=True, scope="off\0").body["messages"]
    assert "distil-ref" in _result(off, "t1"), "the failure: a pointer to the stub"

    monkeypatch.delenv("DISTIL_REFETCH_VERBATIM")
    reset_sessions()
    on = _serve(msgs, session_delta=True, scope="on\0").body["messages"]
    assert "handle=" in _result(on, "t0")
    assert _result(on, "t1") == FILE


def test_session_delta_keeps_committed_blocks_byte_stable() -> None:
    from distil.cachedelta import reset_sessions

    reset_sessions()
    msgs = _session(
        ("cat app.py | sed -n 1,80p", _window(1, 80)),
        ("python repro.py", FILE),
        ("python repro.py", FILE),
        ("sed -n 10,30p app.py | cut -c1-200", _window(10, 30)),
        ("python repro.py", FILE),
    )
    prev: list[str] = []
    for end in range(3, len(msgs) + 1, 2):
        out = _serve(msgs[:end], session_delta=True, scope="stable\0").body["messages"]
        cur = [_stable(m) for m in out]
        assert cur[: len(prev)] == prev, f"a committed message changed at turn {end}"
        prev = cur


def test_held_or_lossless_serving_has_nothing_to_refetch() -> None:
    """The certification hold and a subscription both serve verbatim: nothing is folded,
    so nothing re-inflates and the census carries no re-fetch."""
    msgs = _session(("python repro.py", FILE), ("python repro.py", FILE))
    out = _serve(msgs, verbatim=True, held=True, expand=False, mode="lossless_only")
    assert _result(out.body["messages"], "t1") == FILE
    assert "handle=" not in _result(out.body["messages"], "t0")
    assert (take_census() or {}).get("tool_result_refetch") is None


def test_savings_count_the_refetch_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Booked savings are measured on the payload sent, so a re-fetch forwarded verbatim
    shrinks them by exactly what it costs — no credit for bytes that went upstream."""
    msgs = _session(("python repro.py", FILE), ("python repro.py", FILE))
    monkeypatch.setenv("DISTIL_REFETCH_VERBATIM", "0")
    off = _serve(msgs)
    monkeypatch.delenv("DISTIL_REFETCH_VERBATIM")
    on = _serve(msgs)
    assert on.before_tok == off.before_tok
    assert on.after_tok == len(json.dumps(on.body["messages"]))
    assert on.after_tok > off.after_tok


def test_offline_replay_reads_claude_code_transcripts(tmp_path: Any) -> None:
    """--claude-code: the same metric over Claude Code JSONL, counts only."""
    from benchmarks.reinflate_replay import run_claude_code

    msgs = _session(*[("python repro.py", FILE)] * 3)
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    lines = [json.dumps({"type": m["role"], "message": m}) for m in msgs]
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "s.jsonl").write_text("\n".join(lines))
    out = run_claude_code(tmp_path, sessions=5)
    assert out["sessions_replayed"] == 1
    base, new = out["variants"]["baseline"]["totals"], out["variants"]["refetch"]["totals"]
    # The first re-run cannot be helped (its source is cached as a digest); the second can.
    assert (base["redundant"], base["avoidable"], new["avoidable"]) == (2, 2, 1)
    assert "repro" not in json.dumps(out), "no command text in the aggregate"


# --------------------------------------------------------------------------- every provider
#
# The same scenario through each adapter: a folded result, then a re-fetch of its lines.
# OpenAI and Gemini cache implicitly and keep no recency window, so every result digests
# on first sight there — exactly the shape the rule exists for.


def _chat(*outputs: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "fix the bug"}]
    for n, out in enumerate(outputs):
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"t{n}",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command": "python r.py"}'},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"t{n}", "content": out})
    return msgs


def _responses(*outputs: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix"}]}
    ]
    for n, out in enumerate(outputs):
        items.append(
            {
                "type": "function_call",
                "call_id": f"t{n}",
                "name": "shell",
                "arguments": '{"command": ["python", "r.py"]}',
            }
        )
        items.append({"type": "function_call_output", "call_id": f"t{n}", "output": out})
    return items


def _gemini(*outputs: str) -> dict[str, Any]:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "fix the bug"}]}]
    for out in outputs:
        contents.append(
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "run_shell_command",
                            "args": {"command": "python r.py"},
                        }
                    }
                ],
            }
        )
        contents.append(
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "run_shell_command", "response": {"output": out}}}
                ],
            }
        )
    return {"contents": contents}


def _through(shape: str, outputs: tuple[str, ...], refetch: bool) -> list[str]:
    """The forwarded text of every tool result, in order, for *shape*."""
    from distil.adapters.gemini import compress_generate_request
    from distil.adapters.openai import compress_chat_completions, compress_responses_input

    if shape == "messages":
        out = _send(_session(*(("python r.py", o) for o in outputs)), refetch=refetch)
        return [_result(out, f"t{n}") for n in range(len(outputs))]
    if shape == "chat":
        msgs, _ = compress_chat_completions(_chat(*outputs), persist=False, refetch=refetch)
        return [m["content"] for m in msgs if m.get("role") == "tool"]
    if shape == "responses":
        items, _ = compress_responses_input(_responses(*outputs), persist=False, refetch=refetch)
        return [i["output"] for i in items if i.get("type") == "function_call_output"]
    body, _ = compress_generate_request(_gemini(*outputs), persist=False, refetch=refetch)
    return [
        p["functionResponse"]["response"]["output"]
        for c in body["contents"]
        for p in c["parts"]
        if "functionResponse" in p
    ]


_SHAPES = ("messages", "chat", "responses", "gemini")


@pytest.mark.parametrize("shape", _SHAPES)
def test_every_provider_answers_a_repeat_verbatim(shape: str) -> None:
    off = _through(shape, (FILE, FILE), refetch=False)
    assert "handle=" in off[0] and off[1] == off[0], "without the rule: the same stub twice"
    on = _through(shape, (FILE, FILE), refetch=True)
    assert "handle=" in on[0], "the earlier digest stays (cached prefix)"
    assert on[1] == FILE
    assert (take_census() or {}).get("tool_result_refetch", 0) > 0


@pytest.mark.parametrize("shape", _SHAPES)
def test_every_provider_digests_new_lines_and_stops_after_one_copy(shape: str) -> None:
    on = _through(shape, (_window(1, 80), _window(20, 60), _window(81, 120), FILE, FILE), True)
    assert on[1] == _window(20, 60), "a narrower window of folded lines is a re-fetch"
    assert "handle=" in on[2], "lines never sent digest as usual"
    assert on[4] != FILE, "one verbatim copy exists; the next fetch digests"


@pytest.mark.parametrize("shape", _SHAPES)
def test_every_provider_closes_the_tracker(shape: str) -> None:
    """Stateless: nothing survives the call, so unrelated traffic cannot change a verdict."""
    first = _through(shape, (FILE, FILE), refetch=True)
    assert getattr(A._refetch_tls, "tracker", None) is None
    assert _through(shape, (FILE, FILE), refetch=True) == first


def test_chat_cache_delta_keeps_the_refetch_out_of_the_delta() -> None:
    from distil.adapters.openai import refetch_tool_call_ids

    assert refetch_tool_call_ids(_chat(FILE, FILE)) == {"t1"}
    assert refetch_tool_call_ids(_chat(FILE)) == frozenset()

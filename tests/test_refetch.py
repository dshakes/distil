"""ADR 0022 — a tool result that re-fetches content distil folded is forwarded verbatim.

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

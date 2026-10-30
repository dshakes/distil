"""thinkguard rules, unit level (the end-to-end proof is tests/test_thinking_prefix_e2e.py)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from distil import thinkguard as tg

M = "claude-opus-5-5"


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    tg.reset()
    yield
    tg.reset()


def _t(s: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": s}


def _body(msgs: list[Any], model: str = M) -> dict[str, Any]:
    return {"model": model, "thinking": {"type": "adaptive"}, "messages": msgs}


def _sigs(body: dict[str, Any]) -> list[str]:
    return [
        b["signature"]
        for m in body["messages"]
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "thinking"
    ]


def _turns() -> list[list[Any]]:
    """Three requests of one session: each produces the assistant message the next carries."""
    u0 = {"role": "user", "content": "read the log"}
    a1 = {"role": "assistant", "content": [_t("s1"), {"type": "text", "text": "ok"}]}
    u1 = {"role": "user", "content": [{"type": "text", "text": "LOG " * 50}]}
    a2 = {"role": "assistant", "content": [_t("s2"), {"type": "text", "text": "done"}]}
    u2 = {"role": "user", "content": "next"}
    return [[u0], [u0, a1, u1], [u0, a1, u1, a2, u2]]


def _send(
    msgs: list[Any], edit: bool = False, scope: str = ""
) -> tuple[dict[str, Any], dict[str, str]]:
    client = _body(msgs)
    fwd = copy.deepcopy(client)
    if edit and len(fwd["messages"]) > 2:
        fwd["messages"][2] = {"role": "user", "content": "<<digest handle=abcd1234>>"}
    extras: dict[str, str] = {}
    return tg.guard(client, fwd, scope=scope, extras=extras), extras


def test_unchanged_prefixes_keep_every_block() -> None:
    for msgs in _turns():
        out, ex = _send(msgs)
    assert _sigs(out) == ["s1", "s2"] and not ex


def test_an_edit_before_a_block_drops_it_and_every_later_one_for_good() -> None:
    t = _turns()
    _send(t[0])
    _send(t[1])  # s1 known, nothing edited
    out, ex = _send(t[2], edit=True)  # distil now edits message 2: s2 sits after it
    assert _sigs(out) == ["s1"] and ex["x-distil-thinking-dropped"] == "1"
    out, _ = _send(t[2])  # the edit is gone again, but a removed block stays removed
    assert _sigs(out) == ["s1"]
    tg.reset()  # a restart reads the ledger back
    out, _ = _send(t[2])
    assert _sigs(out) == ["s1"]


def test_the_clients_own_edit_is_passed_through() -> None:
    t = _turns()
    for msgs in t:
        _send(msgs)
    edited = copy.deepcopy(t[2])
    edited[2] = {"role": "user", "content": "the client rewrote this"}
    out, ex = _send(edited)
    assert _sigs(out) == ["s1", "s2"] and not ex


def test_an_unknown_block_is_kept_only_when_distil_changed_nothing() -> None:
    t = _turns()
    out, _ = _send(t[2])
    assert _sigs(out) == ["s1", "s2"]
    tg.reset()
    (tg._path()).unlink()
    out, _ = _send(t[2], edit=True)
    assert _sigs(out) == ["s1"]


def test_a_gap_in_the_chain_drops_the_rest() -> None:
    t = _turns()
    for msgs in t:
        _send(msgs)
    gap = copy.deepcopy(t[2])
    gap[1]["content"] = gap[1]["content"][1:]  # the client removed s1 only: s2 is fine
    out, _ = _send(gap)
    assert _sigs(out) == ["s2"]  # removing the oldest is allowed
    third = [*t[2], {"role": "assistant", "content": [_t("s3"), {"type": "text", "text": "x"}]}]
    third.append({"role": "user", "content": "more"})
    _send([*t[2]])
    mid = copy.deepcopy(third)
    mid[3]["content"] = mid[3]["content"][1:]  # s2 out of the middle: s3 breaks
    out, _ = _send(mid)
    assert _sigs(out) == ["s1"]


def test_a_message_left_empty_gets_a_placeholder() -> None:
    t = _turns()
    _send(t[0])
    _send(t[1])
    only = copy.deepcopy(t[2])
    only[3]["content"] = [_t("s2")]
    tg.drop([_t("s2")])
    out, _ = _send(only)
    assert out["messages"][3]["content"] == [{"type": "text", "text": tg.PLACEHOLDER}]


def test_out_of_scope_is_the_same_object() -> None:
    body = _body(_turns()[2], model="claude-opus-4-6")
    assert tg.guard(body, body) is body
    assert not tg.lineage(body)
    assert tg.lineage(_body([{"role": "user", "content": "hi"}]))
    assert not tg.lineage({"model": M, "messages": [{"role": "user", "content": "hi"}]})
    assert tg.guard(body, {"model": M, "messages": "junk"}) == {"model": M, "messages": "junk"}


def test_a_failure_forwards_as_is(monkeypatch) -> None:
    monkeypatch.setattr(tg, "_guard", lambda *a: 1 / 0)
    body = _body(_turns()[2])
    assert tg.guard(body, body) is body


def test_another_process_lines_are_folded_in_and_torn_lines_wait() -> None:
    _send(_turns()[0])
    with tg._path().open("a") as f:
        f.write(json.dumps({"d": "th:feedface"}) + "\nnot json\n" + '{"d": "th:half')
    with tg._LOCK:
        tg._sync()
    assert "th:feedface" in tg._DROPPED and "th:half" not in tg._DROPPED


def test_the_ledger_compacts(monkeypatch) -> None:
    monkeypatch.setattr(tg, "_MAX_BLOCKS", 2)
    monkeypatch.setattr(tg, "_MAX_REQS", 2)
    monkeypatch.setattr(tg, "_MAX_DROPPED", 2)
    monkeypatch.setattr(tg, "_SLACK", 0)
    for i in range(20):
        tg.drop([_t(f"x{i}")])
    lines = tg._path().read_text().splitlines()
    assert len(lines) <= 5 and len(tg._DROPPED) == 2

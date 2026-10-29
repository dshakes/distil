"""Cold-point thinking compaction (ADR 0028): request-shape rules and cold/warm behaviour.

The API rules pinned here (platform.claude.com/docs/en/build-with-claude/thinking and
.../preserved-thinking, read 2026-10-29):

* the thinking of the turn the model is continuing goes back complete and unmodified —
  during tool use, and with interleaved thinking that is every assistant message since the
  last real user message;
* on preserved-thinking models only a LEADING run of thinking blocks may be removed,
  oldest first; a gap in the middle invalidates every later block;
* signatures are never altered: a block is either sent byte-identical or not at all.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from distil import coldpoint, prefixreplay
from distil.adapters import anthropic as A
from tests.test_coldpoint import _post, _settle_coldpoint, _wire, clock, proxy  # noqa: F401


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    coldpoint.reset()
    prefixreplay.reset()
    monkeypatch.delenv("DISTIL_COLD_POINT", raising=False)
    monkeypatch.setenv("DISTIL_COLD_THINKING", "1")
    yield
    coldpoint.reset()
    prefixreplay.reset()


def _think(sig: str, *, redacted: bool = False) -> dict[str, Any]:
    if redacted:
        return {"type": "redacted_thinking", "data": f"enc-{sig}"}
    return {"type": "thinking", "thinking": "", "signature": f"sig-{sig}"}


def _turn(k: int) -> list[dict[str, Any]]:
    """One completed user turn: a tool round with interleaved thinking, then an answer."""
    return [
        {"role": "user", "content": [{"type": "text", "text": f"question {k}"}]},
        {
            "role": "assistant",
            "content": [
                _think(f"{k}a"),
                {"type": "tool_use", "id": f"t{k}", "name": "Bash", "input": {"c": k}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{k}", "content": "ok"}],
        },
        {"role": "assistant", "content": [_think(f"{k}b"), {"type": "text", "text": f"done {k}"}]},
    ]


def _history(done: int, *, continuing: bool = True) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for k in range(done):
        msgs += _turn(k)
    msgs.append({"role": "user", "content": [{"type": "text", "text": f"question {done}"}]})
    if continuing:
        # The turn in progress: interleaved thinking across two tool rounds, the last
        # tool_use still waiting on the tool_result this request carries.
        msgs += [
            {"role": "assistant", "content": [_think("cur1"), _use("c1")]},
            _result("c1"),
            {"role": "assistant", "content": [_think("cur2"), _use("c2")]},
            _result("c2"),
        ]
    return msgs


def _use(tid: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": tid, "name": "Bash", "input": {}}


def _result(tid: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": "x"}],
    }


def _sigs(msgs: list[dict[str, Any]]) -> list[str]:
    return [
        b.get("signature") or b.get("data")
        for m in msgs
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") in ("thinking", "redacted_thinking")
    ]


def _drop_all(msgs: list[dict[str, Any]], model: str = "claude-opus-5-5") -> list[dict[str, Any]]:
    return A.drop_cold_thinking(msgs, A.cold_thinking_candidates(msgs, model))


# --------------------------------------------------------------------------- API rules


def test_completed_turns_drop_and_the_tool_use_continuation_is_untouched() -> None:
    msgs = _history(3)
    orig = copy.deepcopy(msgs)
    out = _drop_all(msgs)
    assert msgs == orig, "input mutated"
    # Only the turn in progress keeps thinking, every block of it, byte-identical.
    assert _sigs(out) == ["sig-cur1", "sig-cur2"]
    cur = len(orig) - 5  # the user message that opens the turn in progress
    assert out[cur:] == orig[cur:]
    # Same message count, nothing but thinking removed, block order otherwise intact.
    assert len(out) == len(orig)
    for o, m in zip(orig[:cur], out[:cur]):
        assert [b for b in o["content"] if b.get("type") != "thinking"] == m["content"]


def test_the_last_assistant_message_is_never_touched_even_without_a_user_turn() -> None:
    # A history with no real user message after the thinking (all tool results): nothing
    # is "completed", so nothing may drop.
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "content": [_think("a"), _use("x")]},
        _result("x"),
        {"role": "assistant", "content": [_think("b"), _use("y")]},
        _result("y"),
    ]
    assert A.cold_thinking_candidates(msgs, "claude-opus-5-5") == frozenset()


def test_apply_drops_exactly_the_named_blocks_wherever_they_sit() -> None:
    # The leading-run rule is a SELECTION rule (the candidates above). Applying a set is
    # positionless: a block chosen on an earlier turn stays out whatever surrounds it now,
    # because putting it back is what invalidates the chain on preserved-thinking models.
    msgs = _history(3)
    keys = A.cold_thinking_candidates(msgs, "claude-opus-5-5")
    second = A._thinking_key(_think("0b"))
    out = A.drop_cold_thinking(msgs, keys - {second})
    assert _sigs(out) == ["sig-0b", "sig-cur1", "sig-cur2"]


def test_a_thinking_only_message_ends_the_run() -> None:
    msgs = _history(2)
    msgs[1] = {"role": "assistant", "content": [_think("0a")]}  # nothing would be left
    out = _drop_all(msgs)
    assert _sigs(out) == _sigs(msgs)


def test_redacted_thinking_is_a_thinking_block_too() -> None:
    msgs = _history(2)
    msgs[1]["content"][0] = _think("0a", redacted=True)
    out = _drop_all(msgs)
    assert _sigs(out) == ["sig-cur1", "sig-cur2"]


def test_a_compaction_block_means_hands_off() -> None:
    msgs = _history(2)
    msgs[0]["content"].insert(0, {"type": "compaction", "content": "s", "signature": "c"})
    assert A.cold_thinking_candidates(msgs, "claude-opus-5-5") == frozenset()


def test_an_unsigned_or_malformed_history_drops_nothing() -> None:
    msgs = _history(2)
    del msgs[1]["content"][0]["signature"]
    assert A.cold_thinking_candidates(msgs, "claude-opus-5-5") == frozenset()
    assert A.cold_thinking_candidates([*_history(2), "junk"], "claude-opus-5-5") == frozenset()  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("model", "bills"),
    [
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-haiku-5-5", True),
        ("claude-fable-5-1", True),
        ("claude-opus-4-8", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-5", True),
        ("us.anthropic.claude-opus-5-5", True),
        ("claude-sonnet-4-5", False),  # strips prior thinking server-side
        ("claude-haiku-4-5", False),
        ("claude-3-7-sonnet-latest", False),
        ("gpt-5", False),
        ("", False),
    ],
)
def test_only_models_that_bill_prior_thinking(model: str, bills: bool) -> None:
    assert bool(A.cold_thinking_candidates(_history(2), model)) is bills


def test_off_by_default(monkeypatch) -> None:
    msgs = _history(2)
    keys = A.cold_thinking_candidates(msgs, "claude-opus-5-5")
    monkeypatch.delenv("DISTIL_COLD_THINKING")
    assert A.cold_thinking_candidates(msgs, "claude-opus-5-5") == frozenset()
    # The flag gates choosing, never applying: a set chosen while it was on stays applied
    # ("Once you remove a block, leave it out").
    assert _sigs(A.drop_cold_thinking(msgs, keys)) == ["sig-cur1", "sig-cur2"]
    assert A.drop_cold_thinking(msgs, frozenset()) is msgs


def test_compress_messages_applies_it_and_verbatim_does_not() -> None:
    msgs = _history(2)
    keys = A.cold_thinking_candidates(msgs, "claude-opus-5-5")
    out, _ = A.compress_messages(msgs, evict=keys, persist=False)
    assert _sigs(out) == ["sig-cur1", "sig-cur2"]
    out, _ = A.compress_messages(msgs, evict=keys, verbatim=True, persist=False)
    assert _sigs(out) == _sigs(msgs)


# --------------------------------------------------------------------------- cold vs warm


def _body(done: int) -> dict[str, Any]:
    msgs = _history(done, continuing=False)
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {"model": "claude-opus-5-5", "max_tokens": 64, "messages": msgs}


def test_warm_keeps_thinking_cold_drops_it_and_later_turns_stay_byte_stable(proxy, clock) -> None:  # noqa: F811
    port = proxy()
    h, fwd = _post(port, _body(2))
    assert h["x-distil-cold"] == "first-seen" and len(_sigs(fwd)) == 4
    clock[0] += 120
    h, fwd = _post(port, _body(3))
    assert h["x-distil-cold"] == "warm" and len(_sigs(fwd)) == 6, "dropped on a warm turn"

    clock[0] += 20 * 60
    h, cold = _post(port, _body(4))
    assert h["x-distil-cold"] == "cold"
    # Every completed turn's thinking drops, except the newest assistant message's: the
    # "latest assistant message" rule is kept even when its turn is over (doubt = keep).
    assert _sigs(cold) == ["sig-3b"]

    for done in (5, 6):
        clock[0] += 60
        h, fwd = _post(port, _body(done))
        assert h["x-distil-cold"] == "warm"
        assert [_wire(m) for m in fwd[: len(cold)]] == [_wire(m) for m in cold], "prefix moved"
        # Turns completed after the cold point are inside a warm prefix: kept.
        assert len(_sigs(fwd)) == 1 + 2 * (done - 4)


def test_off_forwards_thinking_untouched(proxy, clock, monkeypatch) -> None:  # noqa: F811
    monkeypatch.delenv("DISTIL_COLD_THINKING")
    port = proxy()
    _post(port, _body(2))
    clock[0] += 20 * 60
    h, cold = _post(port, _body(3))
    assert h["x-distil-cold"] == "cold" and len(_sigs(cold)) == 6


def test_a_broken_candidate_choice_fails_open(proxy, clock, monkeypatch) -> None:  # noqa: F811
    def boom(*_a: Any, **_k: Any) -> frozenset[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(A, "cold_thinking_candidates", boom)
    port = proxy()
    _post(port, _body(2))
    clock[0] += 20 * 60
    h, cold = _post(port, _body(3))
    _settle_coldpoint()
    assert h["x-distil-cold"] == "cold" and len(_sigs(cold)) == 6

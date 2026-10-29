"""Cold-point eviction on preserved-thinking models (ADR 0028 addendum).

platform.claude.com/docs/en/build-with-claude/preserved-thinking (read 2026-10-29): on
Fable 5.1, Mythos 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5 each thinking block's signature
binds the prefix it was produced after. An edit at position P fails every thinking block
after P; removing the newest blocks is allowed; a removed block must stay removed; the
latest assistant message of a tool-use continuation goes back complete and unmodified.
So a cold eviction at P drops every thinking block after P in the same set, or waits for
a turn boundary when one of them belongs to the turn in progress.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from distil import coldpoint, prefixreplay
from distil.adapters import anthropic as A
from tests.test_coldpoint import _log, _post, _settle_coldpoint, _wire, clock, proxy  # noqa: F401

PRESERVED = [
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
]
OLDER = ["claude-opus-4-5", "claude-opus-4-6", "claude-sonnet-4-5"]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    coldpoint.reset()
    prefixreplay.reset()
    monkeypatch.delenv("DISTIL_COLD_POINT", raising=False)
    monkeypatch.delenv("DISTIL_COLD_THINKING", raising=False)
    yield
    coldpoint.reset()
    prefixreplay.reset()


def _think(tag: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": f"sig-{tag}"}


def _conv(model: str, rounds: int, *, partial: bool = False) -> dict[str, Any]:
    """*rounds* completed user turns (think → Bash → result → think → answer), each ending
    in the next plain user message. *partial* adds the first half of the next turn: the
    tool-use continuation whose latest assistant message carries thinking."""
    msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    for k in range(rounds + partial):
        msgs += [
            {
                "role": "assistant",
                "content": [
                    _think(f"{k}a"),
                    {"type": "tool_use", "id": f"toolu_{k}", "name": "Bash", "input": {"k": k}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"toolu_{k}", "content": _log(f"r{k}")}
                ],
            },
        ]
        if k < rounds:
            msgs += [
                {"role": "assistant", "content": [_think(f"{k}b"), {"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": f"next {k}"}]},
            ]
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {"model": model, "max_tokens": 64, "messages": msgs}


def _sigs(msgs: list[dict[str, Any]]) -> list[str]:
    return [
        b["signature"]
        for m in msgs
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "thinking"
    ]


def _first_edit(msgs: list[dict[str, Any]]) -> int:
    return next(i for i, m in enumerate(msgs) if "distil evicted" in json.dumps(m))


def _thinking_after(msgs: list[dict[str, Any]], p: int) -> list[str]:
    return _sigs(msgs[p + 1 :])


@pytest.mark.parametrize("model", PRESERVED)
def test_cold_eviction_drops_every_thinking_block_after_the_edit(proxy, clock, model) -> None:  # noqa: F811
    port = proxy()
    h, _ = _post(port, _conv(model, 3))
    assert h["x-distil-cold"] == "first-seen"
    clock[0] += 20 * 60
    h, cold = _post(port, _conv(model, 4))
    assert h["x-distil-cold"] == "cold"
    p = _first_edit(cold)
    assert _thinking_after(cold, p) == [], "a thinking block survived after the edit"
    assert _sigs(cold) == ["sig-0a"], "thinking before the edit must stay"

    # Warm turns: the cold turn's bytes go out identically, nothing dropped comes back,
    # and the thinking produced after the cold point is kept.
    for rounds in (5, 6):
        clock[0] += 60
        h, fwd = _post(port, _conv(model, rounds))
        assert h["x-distil-cold"] == "warm"
        assert [_wire(m) for m in fwd[: len(cold)]] == [_wire(m) for m in cold], "prefix moved"
        new = [f"sig-{k}{s}" for k in range(4, rounds) for s in "ab"]
        assert _sigs(fwd) == ["sig-0a", *new]


@pytest.mark.parametrize("model", PRESERVED)
def test_a_tool_use_continuation_defers_the_eviction_to_the_turn_boundary(
    proxy,  # noqa: F811
    clock,  # noqa: F811
    model,
) -> None:
    port = proxy()
    _post(port, _conv(model, 3))
    clock[0] += 20 * 60
    body = _conv(model, 3, partial=True)
    h, fwd = _post(port, body)
    assert h["x-distil-cold"] == "deferred"
    assert "distil evicted" not in json.dumps(fwd), "edited during a tool-use continuation"
    assert fwd[-2] == body["messages"][-2], "latest assistant message touched"
    assert _sigs(fwd) == _sigs(body["messages"])

    clock[0] += 10
    h, fwd = _post(port, _conv(model, 4))  # the continuation finished: a turn boundary
    assert h["x-distil-cold"] == "deferred-cold"
    p = _first_edit(fwd)
    assert _thinking_after(fwd, p) == [] and _sigs(fwd) == ["sig-0a"]

    clock[0] += 60
    h, nxt = _post(port, _conv(model, 5))
    assert h["x-distil-cold"] == "warm"
    assert [_wire(m) for m in nxt[: len(fwd)]] == [_wire(m) for m in fwd]


@pytest.mark.parametrize("model", OLDER)
def test_older_models_evict_and_keep_their_thinking(proxy, clock, model) -> None:  # noqa: F811
    port = proxy()
    body = _conv(model, 3)
    _post(port, body)
    clock[0] += 20 * 60
    body = _conv(model, 3, partial=True)
    h, fwd = _post(port, body)
    assert h["x-distil-cold"] == "cold" and "distil evicted" in json.dumps(fwd)
    assert _sigs(fwd) == _sigs(body["messages"])


# --------------------------------------------------------------------------- unit rules


@pytest.mark.parametrize(
    ("model", "preserved"),
    [
        *((m, True) for m in PRESERVED),
        ("claude-opus-6-0", True),  # unknown future id: fail safe
        ("claude-sonnet-5", True),
        ("us.anthropic.claude-opus-5-5-v1:0", True),
        ("claude-opus-4-6", False),
        ("claude-sonnet-4-5", False),
        ("claude-3-5-sonnet-20241022", False),
        ("gpt-5", False),
        ("", False),
    ],
)
def test_preserved_thinking_predicate(model: str, preserved: bool) -> None:
    assert A.preserved_thinking(model) is preserved


def test_thinking_after_edit_rules() -> None:
    msgs = _conv("claude-opus-5-5", 3)["messages"]
    ids = frozenset({"toolu_1"})
    keys = A.thinking_after_edit(msgs, ids, "claude-opus-5-5")
    assert keys == {A._thinking_key(_think(t)) for t in ("1b", "2a", "2b")}
    assert A.thinking_after_edit(msgs, ids, "claude-opus-4-6") == frozenset()
    assert A.thinking_after_edit(msgs, frozenset({"nope"}), "claude-opus-5-5") == frozenset()
    # The turn in progress holds thinking after the edit: not now.
    cont = _conv("claude-opus-5-5", 3, partial=True)["messages"]
    assert A.thinking_after_edit(cont, ids, "claude-opus-5-5") is None
    # Shapes it does not reason about: not now.
    unsigned = json.loads(json.dumps(msgs))
    del unsigned[-2]["content"][0]["signature"]
    assert A.thinking_after_edit(unsigned, ids, "claude-opus-5-5") is None
    only = json.loads(json.dumps(msgs))
    only[-2]["content"] = [_think("x")]
    assert A.thinking_after_edit(only, ids, "claude-opus-5-5") is None
    comp = json.loads(json.dumps(msgs))
    comp[0]["content"].insert(0, {"type": "compaction", "content": "s"})
    assert A.thinking_after_edit(comp, ids, "claude-opus-5-5") is None


def test_sticky_eviction_survives_verbatim_and_exact_quote() -> None:
    """A hold (verbatim) or a later exact-quote exemption must not put an evicted block or
    a dropped thinking block back on a preserved-thinking model."""
    msgs = _conv("claude-opus-5-5", 4)["messages"]
    tail = A.thinking_after_edit(msgs, frozenset({"toolu_0"}), "claude-opus-5-5")
    assert tail
    evict = frozenset({"toolu_0"}) | tail
    sticky, _ = A.compress_messages(
        msgs, evict=evict, verbatim=True, persist=False, sticky_evict=True
    )
    assert "distil evicted" in json.dumps(sticky[2]) and _sigs(sticky) == ["sig-0a"]
    plain, _ = A.compress_messages(msgs, evict=evict, verbatim=True, persist=False)
    assert "distil evicted" not in json.dumps(plain) and len(_sigs(plain)) == len(_sigs(msgs))

    real = A.exact_quote_tool_use_ids

    def everything_exact(m: Any, **kw: Any) -> dict[str, str]:
        return {**real(m, **kw), "toolu_0": "test"}

    try:
        A.exact_quote_tool_use_ids = everything_exact  # type: ignore[assignment]
        out, _ = A.compress_messages(msgs, evict=evict, persist=False, sticky_evict=True)
        assert "distil evicted" in json.dumps(out[2])
        out, _ = A.compress_messages(msgs, evict=evict, persist=False)
        assert "distil evicted" not in json.dumps(out[2])
    finally:
        A.exact_quote_tool_use_ids = real  # type: ignore[assignment]


def test_a_failing_extend_evicts_nothing_and_does_not_defer(clock) -> None:  # noqa: F811
    body = _conv("claude-opus-5-5", 3)
    key = "k"

    def boom(_fresh: frozenset[str]) -> frozenset[str] | None:
        raise RuntimeError("boom")

    coldpoint.plan(key, body, body["messages"], lambda: frozenset({"toolu_0"}))
    coldpoint.end(key)
    clock[0] += 20 * 60
    p = coldpoint.plan(key, body, body["messages"], lambda: frozenset({"toolu_0"}), extend=boom)
    coldpoint.end(key)
    _settle_coldpoint()
    assert p.reason == "cold" and p.evict == frozenset()
    clock[0] += 10
    p = coldpoint.plan(key, body, body["messages"], lambda: frozenset({"toolu_0"}))
    assert p.reason == "warm"

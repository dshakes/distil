"""Parallel subagent fan-out must not clobber per-conversation cache state (ADR 0014/0011).

Headroom 0.42 reported a 2.5-3x net cost increase under Claude Code because its cache
tracker was keyed on the session id, which Claude Code shares across a parent and every
subagent it fans out. Interleaved conversations clobbered one shared prefix/freeze state
and rewrote prefixes that were still warm in the provider cache.

Distil keys its per-conversation state (cold-point lineages, prefix-replay lineages,
cache-delta sessions) on a *conversation fingerprint* -- model + system + tools +
first-message -- not on ``DISTIL_SESSION`` alone (``prefixreplay.lineage_key``,
``cachedelta.session_key``). These tests prove that under one shared ``DISTIL_SESSION``:

* interleaving N conversations produces byte-for-byte the same forwarded bytes each
  conversation would see if it ran alone (fan-out is neutral);
* each conversation's compressed prefix stays byte-identical turn to turn (no rewrite of
  a warm prefix -- the Headroom invariant);
* total billed cost with distil is <= without distil under an Anthropic-style prefix
  cache (hit iff the byte prefix matches the last write of the SAME conversation);
* cold-point still fires per conversation (lineages stay distinct, never collapse to the
  shared-key ``ambiguous`` state).

Deterministic and offline: no network, no API key, a driven monotonic clock.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from distil import cachedelta, coldpoint, prefixreplay
from distil.serve_core import compress_or_forward

# Same scope for every conversation: one real API key behind one proxy. Only the
# conversation fingerprint may distinguish lineages -- exactly the fan-out scenario.
_SCOPE = "acct\0"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    # Claude Code stamps ONE session id for a parent and all its subagents.
    monkeypatch.setenv("DISTIL_SESSION", "s-shared-across-subagents")
    monkeypatch.delenv("DISTIL_COLD_POINT", raising=False)
    _reset()
    yield
    _reset()


def _reset() -> None:
    coldpoint.reset()
    prefixreplay.reset()
    cachedelta.reset_sessions()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(coldpoint, "_clock", lambda: now[0])
    return now


# --------------------------------------------------------------------------- fixtures


def _log(tag: str, n: int = 40) -> str:
    return "\n".join(
        f"{tag} line {i}: status=ok elapsed={i * 7}ms worker=w{i % 5} path=/srv/app/mod_{i}.py"
        for i in range(n)
    )


def _conv(system: str, rounds: int, *, head: str) -> dict[str, Any]:
    """A Claude-Code-shaped request: a growing bash-round history, a cached system prompt,
    the breakpoint on the newest block. *system* and *head* are what make a parent and its
    subagents DIFFERENT conversations under one session id."""
    msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": head}]}]
    for k in range(rounds):
        msgs += [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"{head}_{k}",
                        "name": "Bash",
                        "input": {"command": f"make target{k}"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"{head}_{k}",
                        "content": _log(f"{head}-run{k}"),
                    }
                ],
            },
        ]
    cc = {"type": "ephemeral"}
    msgs[-1]["content"][-1]["cache_control"] = cc
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "system": [{"type": "text", "text": system, "cache_control": dict(cc)}],
        "messages": msgs,
    }


# Parent + two subagents: distinct system prompts AND distinct histories, as Claude Code
# fans out. Two subagents even share the HAIKU system prompt to stress the fingerprint --
# only their head/history differ.
_CONVS = [
    ("You are the orchestrator agent.", "fix-the-build"),
    ("You are a focused test-runner subagent on Haiku.", "run-the-unit-tests"),
    ("You are a focused test-runner subagent on Haiku.", "run-the-integration-tests"),
]


def _forward(system: str, rounds: int, head: str) -> dict[str, Any]:
    """One turn through the real proxy compression path, metered (lossy) config."""
    out = compress_or_forward(
        _conv(system, rounds, head=head),
        "/v1/messages",
        count=lambda m: sum(len(json.dumps(x, default=str)) for x in m) // 4,
        verbatim=False,
        expand=True,
        mode="digest",
        cold=True,
        scope=_SCOPE,
        session_delta=True,
        replay_scope=_SCOPE,
    )
    assert out is not None
    coldpoint.end(out.cold_key)
    return out.body


def _strip_marks(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_marks(v) for k, v in node.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_strip_marks(x) for x in node]
    return node


def _wire(msg: Any) -> str:
    """Canonical content bytes, minus the breakpoint marker (which moves every turn by
    design and is not part of the cached *content*)."""
    return json.dumps(_strip_marks(msg), separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------- an Anthropic-style prefix cache model


def _toks(wire: str) -> int:
    return max(1, len(wire) // 4)


def _bill(turns: list[list[Any]]) -> float:
    """Billed cost for one conversation's sequence of forwarded message lists under an
    Anthropic-style prefix cache: a message is a cache READ iff its bytes equal the same
    index in the previous turn AND every message before it was a read; the first mismatch
    and everything after is a cache WRITE. read=0.1x, write=1.25x (Anthropic multipliers)."""
    cost = 0.0
    prev: list[str] = []
    for msgs in turns:
        cur = [_wire(m) for m in msgs]
        broke = False
        for i, w in enumerate(cur):
            hit = (not broke) and i < len(prev) and prev[i] == w
            cost += _toks(w) * (0.1 if hit else 1.25)
            if not hit:
                broke = True
        prev = cur
    return cost


# --------------------------------------------------------------------------- tests


def test_fanout_is_neutral_and_prefix_is_byte_stable(clock) -> None:
    """3 conversations interleaved under ONE DISTIL_SESSION == each run alone, and every
    conversation's compressed prefix is byte-identical turn to turn."""
    rounds = range(2, 8)  # six growing turns per conversation

    # RUN A: each conversation alone, fresh state between them.
    alone: dict[str, list[list[Any]]] = {}
    for system, head in _CONVS:
        _reset()
        alone[head] = [_forward(system, r, head)["messages"] for r in rounds]

    # RUN B: all three interleaved round-robin, ONE shared session, state reset once.
    _reset()
    inter: dict[str, list[list[Any]]] = {head: [] for _, head in _CONVS}
    for r in rounds:
        for system, head in _CONVS:
            inter[head].append(_forward(system, r, head)["messages"])

    for _, head in _CONVS:
        for t, (a, b) in enumerate(zip(alone[head], inter[head])):
            assert [_wire(m) for m in a] == [_wire(m) for m in b], (
                f"fan-out changed {head} turn {t}: interleaving clobbered per-conversation state"
            )

        # The compressed prefix never gets rewritten (the Headroom invariant). The client
        # only appends, so every message of turn k must survive byte-identical into k+1.
        for t in range(1, len(inter[head])):
            prev, cur = inter[head][t - 1], inter[head][t]
            assert [_wire(m) for m in cur[: len(prev)]] == [_wire(m) for m in prev], (
                f"{head} turn {t} rewrote a warm prefix -- the 2-3x cost regression"
            )


def test_fanout_total_cost_not_worse_than_no_distil(clock) -> None:
    """Under the prefix-cache model, distil's total billed cost across the interleaved
    fan-out is <= forwarding the originals uncompressed."""
    rounds = range(2, 8)
    _reset()
    distil: dict[str, list[list[Any]]] = {head: [] for _, head in _CONVS}
    base: dict[str, list[list[Any]]] = {head: [] for _, head in _CONVS}
    for r in rounds:
        for system, head in _CONVS:
            distil[head].append(_forward(system, r, head)["messages"])
            base[head].append(_conv(system, r, head=head)["messages"])

    d = sum(_bill(distil[head]) for _, head in _CONVS)
    b = sum(_bill(base[head]) for _, head in _CONVS)
    assert d <= b, f"distil cost {d:.0f} > baseline {b:.0f} under fan-out"
    assert d < b, "fixture no longer exercises any saving"


def test_lineages_stay_distinct_cold_point_fires_per_conversation(clock) -> None:
    """Shared DISTIL_SESSION must NOT collapse the three conversations into one lineage:
    each still reaches its own cold point (reason 'cold'), never the shared-key 'ambiguous'.
    And once a conversation evicts, its prefix stays byte-stable."""
    seen_cold: dict[str, dict[str, Any]] = {}
    # Warm first turn for all three, interleaved.
    for system, head in _CONVS:
        _forward(system, 3, head)
    # Idle past the TTL + margin: every conversation's next turn is a genuine cold point.
    clock[0] += coldpoint.TTL_DEFAULT_S + coldpoint.MARGIN_S + 1
    for system, head in _CONVS:
        out = compress_or_forward(
            _conv(system, 4, head=head),
            "/v1/messages",
            count=lambda m: 1,
            verbatim=False,
            expand=True,
            mode="digest",
            cold=True,
            scope=_SCOPE,
            session_delta=True,
            replay_scope=_SCOPE,
        )
        coldpoint.end(out.cold_key)
        assert out.extras["x-distil-cold"] == "cold", (
            f"{head} did not reach its own cold point ({out.extras['x-distil-cold']}) -- "
            "the shared session id collapsed the lineages"
        )
        assert int(out.extras["x-distil-cold-evicted"]) > 0
        seen_cold[head] = json.loads(json.dumps(out.body))

    # A later warm turn forwards the evicted prefix byte-identically (no re-write).
    clock[0] += 10
    for system, head in _CONVS:
        later = _forward(system, 5, head)
        cold_prefix = seen_cold[head]["messages"]
        assert [_wire(m) for m in later["messages"][: len(cold_prefix)]] == [
            _wire(m) for m in cold_prefix
        ], f"{head} rewrote its evicted prefix on the next warm turn"

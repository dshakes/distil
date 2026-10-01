"""`served` — the real serving adapter as a certifiable strategy.

`distil` certifies only its volatile-tail digest; what a caching client is actually
sent comes from `adapters.anthropic.compress_messages`. These pin that `served` runs
that adapter faithfully and keeps the contracts a strategy owes the certifier.
"""

from __future__ import annotations

import re

from distil.certify.gate import certify
from distil.compress.strategies import REGISTRY, served
from distil.corpus import load_corpus
from distil.replay.expand_runner import _expand_blocks, build_restore
from distil.trajectory import Block, Kind, Stability

_HANDLE = re.compile(r"handle=([0-9a-f]{8})")


def _log(tag: str) -> str:
    return "\n".join(f"{tag} worker={i % 7} status=ok latency_ms={100 + i}" for i in range(80))


def _turn(n_outputs: int) -> list[Block]:
    """A coding turn: system + tools, the task, then n (agent message, tool output)
    pairs; the newest output is VOLATILE."""
    blocks = [
        Block("sys", Kind.SYSTEM, "You are a coding agent.", Stability.STABLE),
        Block(
            "tools",
            Kind.TOOLS,
            "- goto(line)\n- bash(command): any shell command",
            Stability.STABLE,
        ),
        Block("task", Kind.USER, "fix the flaky test", Stability.STABLE),
    ]
    for k in range(n_outputs):
        cmd = "goto 40" if k == 1 else f"python run.py --step {k}"
        blocks.append(Block(f"a{k}", Kind.HISTORY, f"next\n```\n{cmd}\n```", Stability.SETTLING))
        last = k == n_outputs - 1
        blocks.append(
            Block(
                f"o{k}",
                Kind.TOOL_OUTPUT,
                _log(f"step{k}"),
                Stability.VOLATILE if last else Stability.SETTLING,
                last,
            )
        )
    return blocks


def test_registered_for_certify_and_eval() -> None:
    from distil.cli import build_parser
    from distil.eval import frontier

    assert REGISTRY["served"] is served
    assert build_parser().parse_args(["certify", "--strategy", "served"]).strategy == "served"
    labels = [p.label for p in frontier(load_corpus(), limits=[120]).points]
    assert "served (adapter)" in labels


def test_byte_stable_for_a_given_input_and_across_turns() -> None:
    t3, t4 = _turn(3), _turn(4)
    a, b = served(t3, 3), served(t3, 3)
    assert [x.text for x in a] == [x.text for x in b]
    # an output carried by two consecutive turns is served identically on both (the
    # cache contract: the prefix a caching client re-sends never changes under it)
    by_id4 = {x.id: x.text for x in served(t4, 4)}
    for x in a:
        assert by_id4[x.id] == x.text


def test_digests_earlier_and_freshest_outputs_and_they_are_restorable() -> None:
    blocks = _turn(3)
    out = served(blocks, 3)
    restore = build_restore(blocks)
    by_id = {b.id: b for b in blocks}
    digested = [b for b in out if _HANDLE.search(b.text)]
    # a caching client (breakpoint on the newest message) gets every bash output
    # digested, the freshest included — goto is a numbered view and stays exact
    assert {b.id for b in digested} == {"o0", "o2"}
    for b in digested:
        (h,) = set(_HANDLE.findall(b.text))
        assert restore[h] == by_id[b.id].text
    back = _expand_blocks(out, [h for b in digested for h in _HANDLE.findall(b.text)], restore)
    assert [b.text for b in back] == [b.text for b in blocks]


def test_stable_prefix_untouched_and_never_bigger() -> None:
    blocks = _turn(3)
    out = served(blocks, 3)
    assert out[:2] == blocks[:2]
    assert [b.id for b in out] == [b.id for b in blocks]
    assert all(len(o.text) <= len(b.text) for o, b in zip(out, blocks))
    # agent messages are the agent's own words; the adapter never rewrites them
    assert [o.text for o in out if o.kind is Kind.HISTORY] == [
        b.text for b in blocks if b.kind is Kind.HISTORY
    ]


def test_no_command_means_a_neutral_tool_name() -> None:
    """Without a fenced command the tool is unknowable; a guessed `read` would exempt
    what serving digests. The neutral name leaves it digestible."""
    blocks = [
        Block("h", Kind.HISTORY, "[assistant] called lookup(42)", Stability.SETTLING),
        Block("o", Kind.TOOL_OUTPUT, _log("x"), Stability.VOLATILE, True),
    ]
    assert _HANDLE.search(served(blocks, 0)[1].text)


def test_certifies_on_the_corpus_with_the_deterministic_runner() -> None:
    for e in load_corpus():
        rep = certify(e.trajectory, "served")
        assert rep.match_rate == 1.0, e.trajectory.id
        assert rep.verdict == "PASS", e.trajectory.id


def test_never_touches_the_on_disk_restore_store() -> None:
    """Certifying must not evict a live proxy/MCP session's restore blobs (the store is
    capped and swept by mtime), and what is already on disk must not change its output."""
    from distil import mcp_server

    rdir = mcp_server._restore_dir()
    before = sorted(p.name for p in rdir.iterdir()) if rdir.exists() else []
    blocks = _turn(3)
    out = [b.text for b in served(blocks, 3)]
    after = sorted(p.name for p in rdir.iterdir()) if rdir.exists() else []
    assert after == before == []
    # Pre-populate the disk store with a COLLIDING original for every handle served.
    # A persisting store would decline those stubs (collision guard) and change output.
    handles = {h for t in out for h in _HANDLE.findall(t)}
    assert handles
    for h in handles:
        assert mcp_server.record_restore(h, "some other session's bytes") is True
    assert [b.text for b in served(blocks, 3)] == out


def test_compress_messages_persists_by_default() -> None:
    """The serving default is unchanged: originals still reach the disk store."""
    from distil import mcp_server
    from distil.adapters.anthropic import compress_messages

    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": _log("p"),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
    _, store = compress_messages(msgs)
    assert store.handles and all((mcp_server._restore_dir() / h).exists() for h in store.handles)
    _, mem = compress_messages(msgs, persist=False)
    assert mem.handles == store.handles

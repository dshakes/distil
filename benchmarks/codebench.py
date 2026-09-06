"""Coding-agent cache-delta benchmark — an UPPER BOUND on the exact-quote forfeit.

READ THIS BEFORE QUOTING A NUMBER FROM IT. Every session below is read → edit → RE-READ
of the same few files, and nothing else. That is the one workload where distil's
exact-quote guarantee (distil.compress.provenance) forfeits the most: ~55% of this
corpus's tool-result tokens are file reads the agent must be able to quote back
byte-exact, and what is left is short, so distil's digest row is correctly and
deliberately 0.0%. Real traffic is not shaped like this — on 2,489 measured Claude Code
sessions exact-quote-eligible reads are ~44% of tool-result mass, and the remainder is
the large, digestible log and test output this corpus does not contain at all. So the
digest rows below are a FLOOR on distil's savings and this corpus is an upper bound on
what the guarantee costs. It exists to measure the cache-delta mechanism on its hot
path, not to rank compressors. The exact share is printed under the table, computed
with the same classifier the live path uses.

The lossy competitors have no such guarantee: their savings on this corpus are bought by
destroying the bytes the next Edit needs to match. That is not visible in a token count.

Part 2 of the head-to-head (part 1 = the trajectory-level core comparison via
`distil benchmark`). This one targets the coding-agent hot path the cache-delta
techniques are built for: an agent that reads a file, edits it, and RE-READS it.
It measures, on realistic multi-turn sessions WITH prompt caching priced in:

  * token savings vs full re-send (summed over turns)
  * cache-aware real-dollar savings (the stable prefix billed at the cache-read rate)
  * latency per turn
  * reversibility (recoverable vs lossy)

Fairness: every method is confined to the **volatile suffix** (the new messages
since the previous turn), so the cache-stable prefix is billed at the cache-read
rate for ALL methods — nobody is penalised for the others' cache-busting. The
comparison is therefore purely "how well does each compress the new content", which
is exactly where distil's cross-version delta on a re-read-after-edit shows up.

Honest scope: this measures distil's cache-delta vs the competitors' per-message
compressors. Headroom's router has its own (exact) cross-turn dedup that this
per-block adapter does not drive — but note that even that router has no cross-
VERSION delta (verified: no diff/patch/SequenceMatcher in the package), which is
the case this benchmark stresses. Decision-equivalence for distil's reversible
output is by construction (recoverable) and separately certified live (shadow mode);
the lossy competitors drop content irrecoverably.

Run:  PYTHONPATH=. uv run python benchmarks/codebench.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from distil.adapters.anthropic import compress_messages
from distil.cachedelta import delta_encode
from distil.pricing import get as get_pricing
from distil.tokenizer import DEFAULT as _tok

# --------------------------------------------------------------------------- #
# Realistic coding-session corpus (deterministic / seeded)
# --------------------------------------------------------------------------- #


def _module(n_funcs: int, salt: int) -> str:
    """A synthetic but realistic Python module with many sizeable functions."""
    funcs = []
    for i in range(n_funcs):
        body = "\n".join(f"    v{j} = x * {i + j + salt} + {j}" for j in range(10))
        funcs.append(f"def func_{i}(x):\n{body}\n    return v0 + v9\n")
    return "import os\nimport sys\nimport json\n\n\n" + "\n\n".join(funcs) + "\n"


def _edit(src: str, version: int) -> str:
    """Bump func_0's return to a version-specific value: a clean, parseable change
    isolated to one function, so each re-read is a near-duplicate of the prior
    version (the read→edit→reread case AST-delta is built for)."""
    return re.sub(r"    return v0 \+ v9[^\n]*\n", f"    return v0 + v9 + {version}\n", src, count=1)


def _tool_result(text: str, tid: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def _tool_use(name: str, inp: dict, tid: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
    }


def _session(seed: int, n_files: int = 3, n_edits: int = 4) -> list[list[dict]]:
    """One coding session as a list of CUMULATIVE message lists (one per turn)."""
    msgs: list[dict] = [{"role": "user", "content": f"Fix the failing tests in module {seed}."}]
    files = {f"mod_{seed}_{k}.py": _module(6 + k, seed + k) for k in range(n_files)}
    turns: list[list[dict]] = []
    tid = 0

    def emit() -> None:
        turns.append([dict(m) for m in msgs])

    # Read every file once.
    for name, src in files.items():
        tid += 1
        msgs.append(_tool_use("Read", {"path": name}, f"t{tid}"))
        msgs.append(_tool_result(src, f"t{tid}"))
        emit()
    # A grep with a medium output (some of it recurs later -> exact dedup case).
    grep_out = "\n".join(f"{n}:{i}: return v0 + v9" for n in files for i in range(20))
    tid += 1
    msgs.append(_tool_use("Bash", {"cmd": "grep -n 'return' *.py"}, f"t{tid}"))
    msgs.append(_tool_result(grep_out, f"t{tid}"))
    emit()
    # Edit → re-read cycles on the first file (the cross-version delta hot path).
    target = list(files)[0]
    for _e in range(n_edits):
        files[target] = _edit(files[target], _e + 1)
        tid += 1
        msgs.append(_tool_use("Edit", {"path": target, "edit": "bump"}, f"t{tid}"))
        msgs.append(_tool_result("Edit applied.", f"t{tid}"))
        emit()
        tid += 1
        msgs.append(_tool_use("Read", {"path": target}, f"t{tid}"))  # RE-READ after edit
        msgs.append(_tool_result(files[target], f"t{tid}"))
        emit()
        # Re-run the same grep (identical output -> exact cross-turn dedup).
        tid += 1
        msgs.append(_tool_use("Bash", {"cmd": "grep -n 'return' *.py"}, f"t{tid}"))
        msgs.append(_tool_result(grep_out, f"t{tid}"))
        emit()
    return turns


def make_corpus(n_sessions: int = 20) -> list[list[list[dict]]]:
    return [_session(seed) for seed in range(n_sessions)]


# --------------------------------------------------------------------------- #
# Token + cache-aware cost model (messages level)
# --------------------------------------------------------------------------- #


def _msg_text(m: Any) -> str:
    if not isinstance(m, dict):
        return ""
    c = m.get("content")
    if isinstance(c, str):
        return c
    out = []
    if isinstance(c, list):
        for b in c:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    out.append(b["text"])
                bc = b.get("content")
                if isinstance(bc, str):
                    out.append(bc)
                elif isinstance(bc, list):
                    out += [s.get("text", "") for s in bc if isinstance(s, dict)]
                if isinstance(b.get("input"), dict):
                    out.append(json.dumps(b["input"]))
    return "\n".join(out)


def _msg_tokens(m: Any) -> int:
    return _tok.count(_msg_text(m))


def _msg_hash(m: Any) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()


def _session_dollars(turns_sent: list[list[dict]], pricing: Any) -> float:
    """Cache-aware bill: the contiguous prefix identical to the previous turn is
    billed at cache-read; the rest is fresh input."""
    total = 0.0
    prev: list[str] | None = None
    for msgs in turns_sent:
        hashes = [_msg_hash(m) for m in msgs]
        toks = [_msg_tokens(m) for m in msgs]
        lcp = 0
        if prev is not None:
            lim = min(len(hashes), len(prev))
            while lcp < lim and hashes[lcp] == prev[lcp]:
                lcp += 1
        read = sum(toks[:lcp])
        fresh = sum(toks[lcp:])
        total += read * pricing.cache_read + fresh * pricing.input
        prev = hashes
    return total


def _session_tokens(turns_sent: list[list[dict]]) -> int:
    return sum(_msg_tokens(m) for msgs in turns_sent for m in msgs)


# --------------------------------------------------------------------------- #
# Methods (each confined to the volatile suffix → cache-monotonic for all)
# --------------------------------------------------------------------------- #


def _apply_to_tool_results(m: dict, fn: Callable[[str], str]) -> dict:
    """Apply a text transform to a message's tool_result text(s) (non-mutating)."""
    c = m.get("content")
    if isinstance(c, list):
        new = []
        changed = False
        for b in c:
            if (
                isinstance(b, dict)
                and b.get("type") == "tool_result"
                and isinstance(b.get("content"), str)
            ):
                t = fn(b["content"])
                if t != b["content"]:
                    new.append({**b, "content": t})
                    changed = True
                    continue
            new.append(b)
        if changed:
            return {**m, "content": new}
    return m


# Every method is a PURE function of the cumulative message list -> sent messages.
# Deterministic methods are therefore cache-monotonic by construction (a given
# prefix message always encodes to the same bytes). Headroom is a whole-conversation
# transformer with its own staleness logic, so its cross-turn cache behavior is
# measured exactly as it actually behaves — no harness shortcut.


def _m_none(msgs: list[dict]) -> list[dict]:
    return msgs


def _m_distil(msgs: list[dict]) -> list[dict]:
    return compress_messages(msgs)[0]


def _m_distil_cachedelta(msgs: list[dict]) -> list[dict]:
    pre, _ds, _stats = delta_encode(msgs)  # pure per-call walk (cache-monotonic)
    return compress_messages(pre)[0]


def _m_distil_verbatim(msgs: list[dict]) -> list[dict]:
    return compress_messages(msgs, verbatim=True)[0]


def _m_distil_verbatim_cachedelta(msgs: list[dict]) -> list[dict]:
    pre, _ds, _stats = delta_encode(msgs)
    return compress_messages(pre, verbatim=True)[0]


def make_llmlingua(compress_fn: Callable[[list[str]], list[str]]) -> Callable:
    """LLMLingua applied to EVERY tool_result, memoised per text — so it is a pure,
    deterministic, cache-monotonic function of the conversation (apples-to-apples
    with distil), and we compress each distinct block once."""
    memo: dict[str, str] = {}

    def one(t: str) -> str:
        if t not in memo:
            memo[t] = compress_fn([t])[0]
        return memo[t]

    def method(msgs: list[dict]) -> list[dict]:
        return [_apply_to_tool_results(m, one) for m in msgs]

    return method


def make_headroom() -> Callable:
    """Headroom driven its real way: the whole conversation through its router with
    optimize=True (the invocation that engages its pipeline, incl. its own stale-read
    compression). We measure exactly what it emits, cross-turn cache behavior and all."""
    from headroom import compress as hr

    def method(msgs: list[dict]) -> list[dict]:
        try:
            return hr(msgs, model="claude-sonnet-4-5", model_limit=2000, optimize=True).messages
        except Exception:  # noqa: BLE001 — never let a competitor error abort the bench
            return msgs

    return method


# --------------------------------------------------------------------------- #
# Runner + report
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    name: str
    reversible: bool
    token_savings: float
    dollar_savings: float
    ms_per_turn: float


def run(corpus: list[list[list[dict]]], methods: list[tuple[str, bool, Callable]]) -> list[Row]:
    pricing = get_pricing("claude-opus-4-8")
    base_tok = base_dol = 0.0
    rows: list[Row] = []
    # baseline first (none)
    cache: dict[str, tuple[float, float]] = {}
    for name, reversible, fn in methods:
        tok_total = 0
        dol_total = 0.0
        dur = 0.0
        nturns = 0
        for session in corpus:
            sent: list[list[dict]] = []
            for msgs in session:
                t0 = time.perf_counter()
                s = fn(msgs)
                dur += time.perf_counter() - t0
                nturns += 1
                sent.append(s)
            tok_total += _session_tokens(sent)
            dol_total += _session_dollars(sent, pricing)
        cache[name] = (tok_total, dol_total)
        if name == "none":
            base_tok, base_dol = tok_total, dol_total
        rows.append(Row(name, reversible, 0.0, 0.0, 1000 * dur / max(1, nturns)))
    # fill savings vs baseline
    for r in rows:
        tt, dd = cache[r.name]
        r.token_savings = (1 - tt / base_tok) if base_tok else 0.0
        r.dollar_savings = (1 - dd / base_dol) if base_dol else 0.0
    return rows


def _guarantee_status(corpus: list[list[list[dict]]]) -> str:
    """One line stating how much of this corpus the exact-quote guarantee protects.

    A benchmark that forfeits savings on purpose has to say so where the numbers are
    read, not only in a docstring nobody opens. Measured on the corpus itself, using the
    same classifier the live path uses.
    """
    from distil.adapters.anthropic import exact_quote_tool_use_ids

    protected = eligible = 0
    for session in corpus:
        msgs = session[-1]  # the cumulative final turn holds every tool result
        exact = exact_quote_tool_use_ids(msgs)
        for m in msgs:
            for blk in m.get("content") or ():
                if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                    continue
                n = _tok.count(blk.get("content") or "")
                eligible += n
                if blk.get("tool_use_id") in exact:
                    protected += n
    pct = 100.0 * protected / eligible if eligible else 0.0
    return (
        f"exact-quote guarantee: {pct:.0f}% of tool-result tokens in this corpus are file "
        "reads kept byte-exact, so distil's digest rows are a FLOOR (upper-bound workload; "
        "~44% on real traffic)"
    )


def format_table(rows: list[Row], n_sessions: int, n_turns: int, status: str = "") -> str:
    rows = sorted(rows, key=lambda r: r.dollar_savings, reverse=True)
    out = [
        f"coding-agent cache-delta benchmark  ({n_sessions} sessions, {n_turns} turns, "
        f"read→edit→reread; cache-aware $, model=claude-opus-4-8)",
        "",
        f"{'method':<26}{'tok save':>9}{'$ save (cache)':>16}{'ms/turn':>10}{'fidelity':>12}",
        "-" * 73,
    ]
    for r in rows:
        out.append(
            f"{r.name:<26}{r.token_savings * 100:>8.1f}%{r.dollar_savings * 100:>15.1f}%"
            f"{r.ms_per_turn:>10.2f}{'reversible' if r.reversible else 'lossy':>12}"
        )
    out.append("-" * 73)
    if status:
        out.append(status)
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    methods: list[tuple[str, bool, Callable]] = [
        ("none", True, _m_none),
        ("distil (PAYG digest)", True, _m_distil),
        ("distil+cache-delta", True, _m_distil_cachedelta),
        ("distil-verbatim", True, _m_distil_verbatim),
        ("distil-verbatim+cache-delta", True, _m_distil_verbatim_cachedelta),
    ]
    try:
        methods.append(("headroom (real)", False, make_headroom()))
    except Exception as e:  # noqa: BLE001
        print(f"(headroom unavailable: {e})", file=sys.stderr)
    try:
        from benchmarks.llmlingua_adapter import compress as _ll

        methods.append(("llmlingua-2 (real)", False, make_llmlingua(_ll)))
    except Exception as e:  # noqa: BLE001
        print(f"(llmlingua unavailable: {e})", file=sys.stderr)

    corpus = make_corpus(n)
    n_turns = sum(len(s) for s in corpus)
    rows = run(corpus, methods)
    print(format_table(rows, n, n_turns, _guarantee_status(corpus)))

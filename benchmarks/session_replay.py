"""Replay local Claude Code sessions offline through distil's serving adapter.

Question answered: on *real* long agent sessions, how much of each API request does
``distil.adapters.anthropic.compress_messages`` take out, what does that save once the
provider's prompt cache is priced in, and is any safety signal (quote hazard) moving?

Method
------
1. Discover ``*.jsonl`` Claude Code transcripts under ``--root`` (sub-agent transcripts under a
   ``subagents`` directory and ``isSidechain`` lines are skipped: they are separate
   conversations). Assistant turns are streamed one content block per line; blocks sharing a
   ``message.id`` are merged, and consecutive user lines (parallel ``tool_result`` blocks) are
   merged into one user message, which is what the API request carried.
2. Every user message ends one API request: the request is ``messages[: i + 1]``.
3. ``cache_control``: Claude Code does not record it in the transcript, so unless a block
   already carries one the replay places ``{"type": "ephemeral"}`` on the last block of the
   newest message of each request, as Claude Code does. The output states which applied.
4. Each request goes through ``compress_messages(messages, persist=False)`` (default
   ``verbatim=False``, the served digest path), measured in bytes and tokens (distil's own
   ``distil.tokenizer.DEFAULT`` estimator over the JSON of each message). A second pass runs
   with the shell-search exemption (``provenance.is_shell_search``) patched off in-process.
5. Cache economics: the prefix shared with the previous request is read at 0.1x; the rest is
   written at 1.25x. For the compressed payload the read prefix ends at the first message whose
   compressed bytes changed since the previous request (a cache bust), which is measured, not
   assumed. 5-minute TTL expiry is ignored for both arms.

Privacy: transcripts are private. Nothing from them reaches the output except counts, ratios and
histograms keyed by FIXED vocabularies in this file (bucket labels, an allow-list of built-in
tool names, shell command classes, census bucket names). No path, session id, text or command
string is ever stored in the aggregate, and ``--root`` is not echoed.

Usage::

    uv run python benchmarks/session_replay.py --max-sessions 2500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

from distil.adapters import anthropic as adapter
from distil.compress import provenance

# Parsing, request reconstruction, the breakpoint rule and the request planner are the
# shipped ones: `distil savings` replays the same requests on a user's own machine.
from distil.whatif import (
    Message,
    Session,
    _Sizer,
    discover,
    parse_transcript,
    plan_requests,
    request_indices,
    with_breakpoint,
)

DEFAULT_ROOT = Path("~/.claude/projects")
OUT_DIR = Path(__file__).parent / "results" / "session-replay"

# Anthropic prompt-cache price multipliers on base input.
READ, WRITE, BASE = 0.1, 1.25, 1.0

LENGTH_BUCKETS = (("1-5", 5), ("6-20", 20), ("21-50", 50), ("51-100", 100), ("100+", 10**9))
SIZE_BUCKETS = (
    ("<10k", 10_000),
    ("10k-50k", 50_000),
    ("50k-100k", 100_000),
    ("100k-200k", 200_000),
    ("200k+", 10**12),
)
# Built-in tool names only: an MCP tool name can carry a private server name.
KNOWN_TOOLS = frozenset(
    "Bash BashOutput KillShell Read Edit MultiEdit Write NotebookEdit Grep Glob LS Task Agent "
    "WebFetch WebSearch TodoWrite ExitPlanMode".split()
)
SHELL_CLASSES: dict[str, str] = {
    **dict.fromkeys("git gh".split(), "vcs"),
    **dict.fromkeys("ls find tree du wc stat file".split(), "listing"),
    **dict.fromkeys(
        "pytest python python3 uv npm npx pnpm yarn node cargo go make mvn gradle tsc ruff mypy "
        "jest vitest".split(),
        "build_test",
    ),
    **dict.fromkeys("curl wget docker kubectl helm gcloud aws ssh".split(), "network_infra"),
}
_MARKER = re.compile(r"handle=[0-9a-f]{8}")


# --------------------------------------------------------------------------- measuring


def _result_text(blk: Message) -> str:
    c = blk.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            b["text"] for b in c if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def _tool_results(msg: Message) -> Iterator[tuple[str, str]]:
    for b in msg["content"]:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            tid = b.get("tool_use_id")
            if isinstance(tid, str):
                yield tid, _result_text(b)


def shell_class(command: str) -> str:
    """Fixed-vocabulary class of a shell command (never the command itself)."""
    if provenance.is_shell_search(command):
        return "search"
    if provenance.whole_file_read_paths(command):
        return "whole_file_read"
    for stage in command.replace("&&", ";").replace("||", ";").replace("\n", ";").split(";"):
        words = stage.split()
        if words and words[0] != "cd":
            return SHELL_CLASSES.get(os.path.basename(words[0]), "other")
    return "other"


@contextmanager
def shell_search_exemption(enabled: bool) -> Iterator[None]:
    """Counterfactual switch: with ``enabled=False``, shell-search results may digest."""
    if enabled:
        yield
    else:
        with mock.patch.object(provenance, "is_shell_search", lambda _c: False):
            yield


@dataclass
class Replayed:
    """Aggregate of one session for one arm: additive counters plus per-request records."""

    counts: Counter[str] = field(default_factory=Counter)
    # (size_bucket, bytes_b, bytes_a, tok_b, tok_a, cost_b, cost_a)
    requests: list[tuple[int, int, int, int, int, float, float]] = field(default_factory=list)


def _bucket(value: int, table: tuple[tuple[str, int], ...]) -> int:
    return next(i for i, (_, hi) in enumerate(table) if value <= hi)


def replay_session(
    sess: Session, *, exempt: bool, detail: bool, sizer: _Sizer, plan: list[tuple[int, bool]]
) -> Replayed:
    """Replay the planned requests of *sess* through the adapter; one arm."""
    out = Replayed()
    c = out.counts
    orig = [sizer.of(m) for m in sess.messages]
    prev_end = -1
    prev_keys: list[bytes] = []
    with shell_search_exemption(exempt):
        for end, recorded in plan:
            req = with_breakpoint(sess.messages[: end + 1], sess.cache_recorded)
            comp, _store = adapter.compress_messages(req, persist=False)
            hazard = adapter.take_quote_hazard()
            ex = adapter.exact_quote_tool_use_ids(req) if detail else {}
            csz = [sizer.of(m) for m in comp]
            keys = [s[0] for s in csz]
            n = end + 1
            if not recorded:
                prev_end, prev_keys = end, keys
                continue
            bb, ba = sum(s[1] for s in orig[:n]), sum(s[1] for s in csz)
            tb, ta = sum(s[2] for s in orig[:n]), sum(s[2] for s in csz)
            # Cache arms. Baseline prefix is always stable; the compressed one stops where a
            # message's bytes changed since the previous request (everything after is rewritten).
            read_b = sum(s[2] for s in orig[: prev_end + 1])
            stable = 0
            while stable <= prev_end and keys[stable] == prev_keys[stable]:
                stable += 1
            read_a = sum(s[2] for s in csz[:stable])
            cost_b = READ * read_b + WRITE * (tb - read_b)
            cost_a = READ * read_a + WRITE * (ta - read_a)
            out.requests.append((_bucket(tb, SIZE_BUCKETS), bb, ba, tb, ta, cost_b, cost_a))
            c["requests"] += 1
            c["prefix_tokens_saved"] += read_b - sum(s[2] for s in csz[: prev_end + 1])
            c["delta_tokens_saved"] += (tb - read_b) - (ta - sum(s[2] for s in csz[: prev_end + 1]))
            if prev_end >= 0:
                c["cache_requests"] += 1
                c["cache_busts"] += stable <= prev_end
            if hazard:
                c["hazard_requests"] += 1
                c["quotes_survived"] += hazard["survived"]
                c["quotes_lost"] += hazard["lost"]
                # Baseline: quotes no read ever carried byte-exact (Write-then-Edit, multi-line
                # quotes against line-numbered Read output) are lost with NO compression.
                c["quotes_lost_uncompressed"] += len(
                    provenance.missing_quotes(
                        provenance.edit_quotes(req), provenance.observed_view(req)
                    )
                )
                c["hazard_requests_with_loss"] += hazard["lost"] > 0
            _tool_stats(c, req, comp, ex, detail)
            prev_end, prev_keys = end, keys
    return out


def _tool_stats(
    c: Counter[str], req: list[Message], comp: list[Message], ex: dict[str, str], detail: bool
) -> None:
    calls = {t.id: t for t in adapter._tool_calls(req)}
    for before, after in zip(req, comp):  # same length: compress_messages is 1:1 per message
        if before["role"] != "user":
            continue
        after_text = dict(_tool_results(after))
        for tid, text in _tool_results(before):
            nbytes = len(text.encode())
            new = after_text.get(tid, text)
            digested = (
                bool(_MARKER.search(new)) and not _MARKER.search(text) and len(new) < len(text)
            )
            c["tr_n"] += 1
            c["tr_bytes"] += nbytes
            c["tr_dig_n"] += digested
            c["tr_dig_bytes"] += nbytes if digested else 0
            if not detail:
                continue
            call = calls.get(tid)
            name = call.name if call else ""
            tool = name if name in KNOWN_TOOLS else ("mcp" if name.startswith("mcp__") else "other")
            keys = [f"tool:{tool}"]
            keys.append(f"exact:{ex.get(tid, 'digestible')}")
            if tool == "Bash" and call:
                keys.append(f"bash:{shell_class(call.command)}")
            for k in keys:
                c[f"{k}:n"] += 1
                c[f"{k}:dig"] += digested
                c[f"{k}:bytes"] += nbytes
                c[f"{k}:dig_bytes"] += nbytes if digested else 0


@dataclass
class SessionResult:
    n_requests: int
    bad_lines: int
    cache_recorded: bool
    served: Replayed
    counterfactual: Replayed


def process_file(
    path: Path, max_requests: int = 100, budget_bytes: int = 600_000_000
) -> SessionResult | str:
    """Worker: one transcript -> result, or a fixed-vocabulary skip/error string."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            sess = parse_transcript(fh)
        ends = request_indices(sess.messages)
        if not ends:
            return "empty"
        sizer = _Sizer()
        plan = plan_requests(
            ends, [sizer.of(m)[1] for m in sess.messages], max_requests, budget_bytes
        )
        if plan is None:
            return "over_work_budget"
        kw: dict[str, Any] = {"sizer": sizer, "plan": plan}
        return SessionResult(
            len(ends),
            sess.bad_lines,
            sess.cache_recorded,
            replay_session(sess, exempt=True, detail=True, **kw),
            replay_session(sess, exempt=False, detail=False, **kw),
        )
    except (OSError, ValueError, KeyError, TypeError, RecursionError, IndexError) as exc:
        return f"error:{type(exc).__name__}"


# --------------------------------------------------------------------------- aggregation


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _dist(xs: list[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(xs), 4) if xs else 0.0,
        "mean": round(statistics.fmean(xs), 4) if xs else 0.0,
        "p90": round(_pct(xs, 0.9), 4),
    }


def _frac(saved: float, total: float) -> float:
    return round(saved / total, 4) if total else 0.0


def summarize(reqs: list[tuple[int, int, int, int, int, float, float]]) -> dict[str, Any]:
    """Request-size savings (fraction removed) for a group of requests."""
    bb, ba, tb, ta, cb, ca = (sum(r[i] for r in reqs) for i in range(1, 7))
    return {
        "requests": len(reqs),
        "bytes_saving": _dist([1 - r[2] / r[1] for r in reqs if r[1]]),
        "token_saving": _dist([1 - r[4] / r[3] for r in reqs if r[3]]),
        "weighted_bytes_saving": _frac(bb - ba, bb),
        "weighted_token_saving": _frac(tb - ta, tb),
        "cache_aware_cost_saving": _frac(cb - ca, cb),
        "no_cache_cost_saving": _frac(tb - ta, tb),
    }


def _rates(counts: Counter[str], prefix: str) -> dict[str, dict[str, float]]:
    names = sorted({k.split(":")[1] for k in counts if k.startswith(prefix + ":")})
    return {
        n: {
            "results": counts[f"{prefix}:{n}:n"],
            "digest_rate": _frac(counts[f"{prefix}:{n}:dig"], counts[f"{prefix}:{n}:n"]),
            "bytes_digested_share": _frac(
                counts[f"{prefix}:{n}:dig_bytes"], counts[f"{prefix}:{n}:bytes"]
            ),
        }
        for n in names
    }


def _arm(results: list[SessionResult], arm: str, *, detail: bool) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_len: list[list[tuple[int, int, int, int, int, float, float]]] = [[] for _ in LENGTH_BUCKETS]
    by_size: list[list[tuple[int, int, int, int, int, float, float]]] = [[] for _ in SIZE_BUCKETS]
    sessions_in = [0] * len(LENGTH_BUCKETS)
    allreq = []
    for r in results:
        rep: Replayed = getattr(r, arm)
        counts.update(rep.counts)
        li = _bucket(r.n_requests, LENGTH_BUCKETS)
        sessions_in[li] += 1
        by_len[li].extend(rep.requests)
        for q in rep.requests:
            by_size[q[0]].append(q)
        allreq.extend(rep.requests)
    hz = counts["quotes_survived"] + counts["quotes_lost"]
    out: dict[str, Any] = {
        "overall": summarize(allreq),
        "by_session_length": {
            lab: {"sessions": sessions_in[i], **summarize(by_len[i])}
            for i, (lab, _) in enumerate(LENGTH_BUCKETS)
        },
        "by_context_size_tokens": {
            lab: summarize(by_size[i]) for i, (lab, _) in enumerate(SIZE_BUCKETS)
        },
        "tool_results": {
            "count": counts["tr_n"],
            "bytes_digested_share": _frac(counts["tr_dig_bytes"], counts["tr_bytes"]),
            "digest_rate": _frac(counts["tr_dig_n"], counts["tr_n"]),
        },
        "savings_by_region_tokens": {
            "cached_prefix_share": _frac(
                counts["prefix_tokens_saved"],
                counts["prefix_tokens_saved"] + counts["delta_tokens_saved"],
            ),
            "new_since_last_request_share": _frac(
                counts["delta_tokens_saved"],
                counts["prefix_tokens_saved"] + counts["delta_tokens_saved"],
            ),
        },
        "cache": {
            "bust_rate": _frac(counts["cache_busts"], counts["cache_requests"]),
            "requests_considered": counts["cache_requests"],
        },
        "quote_hazard": {
            "requests_with_literal_edits": counts["hazard_requests"],
            "quotes_survived": counts["quotes_survived"],
            "quotes_lost": counts["quotes_lost"],
            "quotes_lost_without_compression": counts["quotes_lost_uncompressed"],
            "loss_rate": _frac(counts["quotes_lost"], hz),
            "requests_with_loss": counts["hazard_requests_with_loss"],
        },
    }
    if detail:
        out["by_tool"] = _rates(counts, "tool")
        out["by_exact_quote_bucket"] = _rates(counts, "exact")
        out["by_bash_command_class"] = _rates(counts, "bash")
    return out


def aggregate(results: list[SessionResult], skipped: Counter[str], found: int, seed: int) -> dict:
    from distil import __version__

    cache_rec = sum(r.cache_recorded for r in results)
    return {
        "schema": 1,
        "distil_version": __version__,
        "seed": seed,
        "method": {
            "adapter": "distil.adapters.anthropic.compress_messages(persist=False, verbatim=False)",
            "request": "every prefix ending in a user message (tool_result turns included)",
            "tokenizer": "distil.tokenizer.DEFAULT (offline heuristic) over per-message JSON",
            "cache_control": "recorded by transcript"
            if cache_rec == len(results) and results
            else "not recorded by Claude Code; placed on the newest message of each request",
            "cache_prices": {"read": READ, "write": WRITE, "base": BASE},
            "cache_model": (
                "prefix shared with the previous request is read (0.1x), the rest written "
                "(1.25x); compressed read prefix ends at the first message whose compressed "
                "bytes changed (a bust); TTL expiry ignored; input tokens only"
            ),
            "counterfactual": "provenance.is_shell_search patched to False in-process",
        },
        "sample": {
            "transcripts_found": found,
            "sessions_replayed": len(results),
            "requests_total": sum(r.n_requests for r in results),
            "requests_replayed": sum(len(r.served.requests) for r in results),
            "sessions_subsampled": sum(len(r.served.requests) < r.n_requests for r in results),
            "sessions_skipped": dict(sorted(skipped.items())),
            "unparseable_lines": sum(r.bad_lines for r in results),
            "sessions_with_recorded_cache_control": cache_rec,
        },
        "served": _arm(results, "served", detail=True),
        "without_shell_search_exemption": _arm(results, "counterfactual", detail=False),
    }


# --------------------------------------------------------------------------- report


def _pct_s(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(s: dict[str, Any]) -> str:
    m, smp = s["method"], s["sample"]
    L = [
        "# Session replay: savings on real long Claude Code sessions",
        "",
        f"distil {s['distil_version']}, seed {s['seed']}. Aggregates only; generated by "
        "`benchmarks/session_replay.py`.",
        "",
        f"- Sessions replayed: {smp['sessions_replayed']} of {smp['transcripts_found']} "
        f"transcripts found; requests: {smp['requests_replayed']} replayed of {smp['requests_total']} "
        f"({smp['sessions_subsampled']} long sessions subsampled to evenly spaced requests); "
        f"unparseable lines skipped: {smp['unparseable_lines']}.",
        "- Skipped sessions: "
        + (", ".join(f"{k} {v}" for k, v in smp["sessions_skipped"].items()) or "none")
        + ".",
        f"- cache_control: {m['cache_control']}.",
        f"- Tokenizer: {m['tokenizer']}.",
        f"- Cache model: {m['cache_model']}.",
        "",
    ]
    for arm, title in (
        ("served", "Served path (shell-search results kept verbatim)"),
        ("without_shell_search_exemption", "Counterfactual: shell-search exemption disabled"),
    ):
        a = s[arm]
        L += [f"## {title}", "", "### By session length (requests per session)", ""]
        L += _table(a["by_session_length"], "bucket", sessions=True)
        L += ["", "### By request context size (tokens before)", ""]
        L += _table(a["by_context_size_tokens"], "bucket")
        t, r, c, h = a["tool_results"], a["savings_by_region_tokens"], a["cache"], a["quote_hazard"]
        L += [
            "",
            f"- Overall: {_pct_s(a['overall']['weighted_token_saving'])} tokens, "
            f"{_pct_s(a['overall']['cache_aware_cost_saving'])} cache-aware cost.",
            f"- Tool-result bytes digested: {_pct_s(t['bytes_digested_share'])} "
            f"({t['count']} result instances summed over replayed requests, digest rate {_pct_s(t['digest_rate'])}).",
            f"- Saved tokens inside the cached prefix: {_pct_s(r['cached_prefix_share'])}; "
            f"in content new since the last request: {_pct_s(r['new_since_last_request_share'])}.",
            f"- Cache bust rate (compressed prefix changed vs previous request): "
            f"{_pct_s(c['bust_rate'])} of {c['requests_considered']} requests.",
            f"- Quote hazard: {h['quotes_lost']} of "
            f"{h['quotes_survived'] + h['quotes_lost']} edit quotes lost "
            f"({_pct_s(h['loss_rate'])}; {h['quotes_lost_without_compression']} would be lost "
            f"with no compression at all); {h['requests_with_loss']} of "
            f"{h['requests_with_literal_edits']} edit-bearing requests had a loss.",
            "",
        ]
        if arm == "served":
            for key, label in (
                ("by_tool", "tool"),
                ("by_exact_quote_bucket", "exact-quote bucket"),
                ("by_bash_command_class", "shell command class"),
            ):
                L += [
                    f"### Digest rate by {label}",
                    "",
                    f"| {label} | results | digest rate | bytes digested |",
                ]
                L += ["|---|---:|---:|---:|"]
                for k, v in a[key].items():
                    L.append(
                        f"| {k} | {v['results']:.0f} | {_pct_s(v['digest_rate'])} | "
                        f"{_pct_s(v['bytes_digested_share'])} |"
                    )
                L.append("")
    return "\n".join(L) + "\n"


def _table(groups: dict[str, Any], head: str, *, sessions: bool = False) -> list[str]:
    cols = " sessions |" if sessions else ""
    rows = [
        f"| {head} |{cols} requests | bytes median / mean / p90 | tokens median / mean / p90 "
        "| tokens (weighted) | $ cache-aware |",
        "|---|" + ("---:|" if sessions else "") + "---:|---|---|---:|---:|",
    ]
    for lab, g in groups.items():
        b, t = g["bytes_saving"], g["token_saving"]
        sess = f" {g['sessions']} |" if sessions else ""
        rows.append(
            f"| {lab} |{sess} {g['requests']} | "
            f"{_pct_s(b['median'])} / {_pct_s(b['mean'])} / {_pct_s(b['p90'])} | "
            f"{_pct_s(t['median'])} / {_pct_s(t['mean'])} / {_pct_s(t['p90'])} | "
            f"{_pct_s(g['weighted_token_saving'])} | {_pct_s(g['cache_aware_cost_saving'])} |"
        )
    return rows


# --------------------------------------------------------------------------- driver


def run(
    root: Path,
    max_sessions: int | None,
    seed: int,
    jobs: int,
    max_requests: int = 100,
    budget_bytes: int = 600_000_000,
) -> dict[str, Any]:
    files = discover(root.expanduser())
    chosen = files
    if max_sessions is not None and len(files) > max_sessions:
        chosen = sorted(random.Random(seed).sample(files, max_sessions))
    results: list[SessionResult] = []
    skipped: Counter[str] = Counter()
    fn = partial(process_file, max_requests=max_requests, budget_bytes=budget_bytes)
    if jobs > 1:
        with multiprocessing.Pool(jobs) as pool:
            outs = list(pool.imap(fn, chosen, chunksize=1))
    else:
        outs = [fn(p) for p in chosen]
    for o in outs:
        if isinstance(o, str):
            skipped[o] += 1
        else:
            results.append(o)
    return aggregate(results, skipped, len(files), seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--max-sessions", "--sample", type=int, default=None, dest="max_sessions")
    ap.add_argument(
        "--max-requests", type=int, default=100, help="per-session cap (quadratic cost)"
    )
    ap.add_argument("--budget-mb", type=int, default=600, help="per-session bytes compressed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    summary = run(
        args.root,
        args.max_sessions,
        args.seed,
        args.jobs,
        args.max_requests,
        args.budget_mb * 1_000_000,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_report(summary))
    print(
        f"replayed {summary['sample']['sessions_replayed']} sessions / "
        f"{summary['sample']['requests_replayed']} requests -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

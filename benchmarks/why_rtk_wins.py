"""Why RTK beats distil on cost for coding agents: an offline, $0 root-cause decomposition.

No API calls. Reads the outcome-harness results (per-task provider ``usage`` totals) and the
saved transcripts, reconstructs every step's prompt, and attributes the bill to where the money
actually goes. Output is aggregate-only JSON (no transcript content).

    python benchmarks/why_rtk_wins.py \
        --lite  <swebench-outcome-300-h2h dir> \
        --longh <swebench-verified-hard-max dir> \
        --tbench <cost_truth pilot dir> \
        --out benchmarks/results/why-rtk-wins/results.json [--replay]

How a step is reconstructed (the harness, benchmarks/swebench_outcome/agent.py): every request
is ``system + tools + messages[:i]`` with ONE ephemeral cache breakpoint on the newest block.
With an append-only history the provider then reads the whole previous request from cache
(0.1x) and writes only what was appended since (1.25x): the previous assistant turn plus the
new tool results. Token counts per block are not stored, so block size is estimated from its
characters with per-kind coefficients fitted (least squares) against the provider's own
per-task ``cache_write + input`` totals, and the fit is validated out-of-sample against the
provider's ``cache_read`` totals, which the append-only model predicts independently.

``--replay`` additionally re-runs distil's served transform (``compress_messages``) on raw
transcripts, step by step, exactly as the harness called it, and simulates Anthropic's prefix
cache (a read is the longest earlier request that is a block-exact prefix of this one). It
needs the distil package importable and sets DISTIL_HOME to a temp dir (never ~/.distil).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PIN, POUT = 2.0, 10.0  # claude-sonnet-5-5 / claude-sonnet-5 $/MTok (official, see harness)
W, R = 1.25, 0.10  # 5-minute cache write / read multipliers
KINDS = ("problem", "tool_result", "tool_use", "text", "thinking_sig")


# ---------------------------------------------------------------- loading


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def usd(u: dict[str, int]) -> dict[str, float]:
    return {
        "uncached_in": u["input"] * PIN / 1e6,
        "cache_write": W * u["cache_write"] * PIN / 1e6,
        "cache_read": R * u["cache_read"] * PIN / 1e6,
        "output": u["output"] * POUT / 1e6,
    }


def block_chars(m: dict[str, Any]) -> Counter[str]:
    c: Counter[str] = Counter()
    content = m["content"]
    if isinstance(content, str):
        c["problem" if m["role"] == "user" else "text"] += len(content)
        return c
    for b in content:
        t = b.get("type")
        if t == "tool_result":
            v = b.get("content", "")
            c["tool_result"] += len(v if isinstance(v, str) else json.dumps(v))
        elif t == "tool_use":
            c["tool_use"] += len(json.dumps(b.get("input", {})))
        elif t == "text":
            # The only user text here is the task statement (a list after a breakpoint).
            c["problem" if m["role"] == "user" else "text"] += len(b.get("text", ""))
        elif t == "thinking":
            c["thinking_sig"] += len(b.get("signature", "")) + len(b.get("thinking", ""))
    return c


def step_prefixes(msgs: list[dict[str, Any]]) -> list[Counter[str]]:
    """Cumulative chars by kind of the messages sent at each step (request k = msgs[:i_k])."""
    out, acc = [], Counter()
    for m in msgs:
        if m["role"] == "assistant":
            out.append(Counter(acc))
        acc += block_chars(m)
    return out


# ---------------------------------------------------------------- token model


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for the small normal equations (stdlib only)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        if abs(m[i][i]) < 1e-12:
            continue
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit(samples: list[tuple[Counter[str], float]]) -> dict[str, float]:
    """tokens = S*n + sum_k coef_k * chars_k, least squares (``n`` = feature "S", the number of
    requests the target sums over). Coefficients are clamped at 0."""
    names = ("S", *KINDS)
    # Relative least squares: each equation is scaled by its target, so a 60-step task
    # does not outweigh three hundred 4-step ones.
    xs = [[float(c[k]) / max(y, 1.0) for k in names] for c, y in samples]
    ys = [1.0 for _ in samples]
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(len(names))] for i in range(len(names))]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(len(names))]
    coef = solve(ata, aty)
    return {n: max(0.0, v) for n, v in zip(names, coef)}


def total_row(msgs: list[dict[str, Any]], u: dict[str, int]) -> list[tuple[Counter[str], float]]:
    """One cache-INVARIANT equation per task: the prompt tokens summed over every request
    (input + cache_write + cache_read) is the same whatever the provider had cached."""
    pre = step_prefixes(msgs)
    acc: Counter[str] = Counter()
    for c in pre:
        acc += c
    acc["S"] = len(pre)
    return [(acc, float(u["input"] + u["cache_write"] + u["cache_read"]))] if pre else []


def fit_rows(msgs: list[dict[str, Any]], u: dict[str, int]) -> list[tuple[Counter[str], float]]:
    """Two equations per task under the append-only model: the last request's prompt is
    everything written (+ uncached input), and the reads are the sum of all earlier prompts."""
    pre = step_prefixes(msgs)
    if not pre:
        return []
    last = Counter(pre[-1])
    last["S"] = 1
    reads: Counter[str] = Counter()
    for c in pre[:-1]:
        reads += c
    reads["S"] = len(pre) - 1
    out = [(last, float(u["cache_write"] + u["input"]))]
    if len(pre) > 1:
        out.append((reads, float(u["cache_read"])))
    return out


def tok(c: Counter[str], coef: dict[str, float], with_s: bool = True) -> float:
    return (coef["S"] if with_s else 0.0) + sum(coef[k] * c[k] for k in KINDS)


# ---------------------------------------------------------------- per-arm transcript analysis

CMD_RE = re.compile(r"^(?:cd\s+\S+\s*(?:&&|;)\s*)*(?:timeout\s+\d+\s+)?(\S+)")


def cmd_class(b: dict[str, Any]) -> str:
    if b.get("name") == "str_replace_based_edit_tool":
        return "edit:" + str(b.get("input", {}).get("command"))
    if b.get("name") != "bash":
        return str(b.get("name"))
    cmd = str(b.get("input", {}).get("command", "")).strip()
    m = CMD_RE.match(cmd)
    head = os.path.basename(m.group(1)) if m else "?"
    if head.startswith("python"):
        head = "python -c" if " -c " in cmd[:40] else ("pytest" if "pytest" in cmd else "python")
    if head == "sed" and " -n" in cmd[:12]:
        head = "sed -n"
    return head


def tool_calls(msgs: list[dict[str, Any]]) -> list[tuple[str, str, int, bool]]:
    """(class, normalized command, result chars, is_error) per tool call."""
    uses: dict[str, dict[str, Any]] = {}
    out = []
    for m in msgs:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                uses[b["id"]] = b
            elif b.get("type") == "tool_result":
                u = uses.get(b.get("tool_use_id"), {})
                v = b.get("content", "")
                v = v if isinstance(v, str) else json.dumps(v)
                norm = json.dumps(u.get("input", {}), sort_keys=True)
                out.append((cmd_class(u), norm, len(v), bool(b.get("is_error"))))
    return out


def reconstruct(msgs: list[dict[str, Any]], coef: dict[str, float]) -> dict[str, float]:
    """Append-only, perfectly cached model of one task: tokens and $ by where they come from."""
    pre = step_prefixes(msgs)
    p = [tok(c, coef) for c in pre]
    n = len(p)
    # Each token entering at step j is written once (1.25x) and re-read at every later step.
    new_by_kind: Counter[str] = Counter()
    reread_by_kind: Counter[str] = Counter()
    for j in range(n):
        delta = pre[j] - (pre[j - 1] if j else Counter())
        later = n - 1 - j
        for k in KINDS:
            t = coef[k] * delta[k]
            new_by_kind[k] += t
            reread_by_kind[k] += t * later
        if j == 0:
            new_by_kind["system+tools"] += coef["S"]
            reread_by_kind["system+tools"] += coef["S"] * later
    return {
        "pred_write": p[-1] if p else 0.0,
        "pred_read": sum(p[:-1]),
        "new": dict(new_by_kind),
        "reread": dict(reread_by_kind),
        "steps": n,
    }


# ---------------------------------------------------------------- distil replay


def replay_distil(msgs: list[dict[str, Any]], coef: dict[str, float]) -> dict[str, Any]:
    """Run the served transform at every step as the harness did; simulate the prefix cache."""
    from benchmarks.swebench_outcome.agent import with_cache_breakpoint
    from distil.adapters.anthropic import compress_messages

    sent: list[list[str]] = []  # per step: canonical json per message (block-exact prefix test)
    written: list[int] = []  # message counts of cache entries written (each request's end)
    out = {
        "steps": 0,
        "read": 0.0,
        "write": 0.0,
        "raw_tokens": 0.0,
        "sent_tokens": 0.0,
        "rewrites": 0,
        "rewrite_tokens": 0.0,
        "entry_raw": 0,
        "entry_sent": 0,
    }
    cut = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for i in cut:
        raw = msgs[:i]
        send, _store = compress_messages(with_cache_breakpoint(raw), persist=False)
        canon = [json.dumps(_strip_cc(m), sort_keys=True) for m in send]
        sz = [tok(block_chars(_strip_cc(m)), coef, with_s=False) for m in send]
        total = coef["S"] + sum(sz)
        # Longest earlier written entry that is an exact prefix of this request.
        hit = 0
        for j, prev in enumerate(sent):
            k = written[j]
            if k <= len(canon) and prev[:k] == canon[:k]:
                hit = max(hit, k)
        read = coef["S"] + sum(sz[:hit]) if hit else 0.0
        if sent:
            # Did this request change a message the previous request already sent?
            prev = sent[-1]
            changed = [j for j in range(min(len(prev), len(canon))) if prev[j] != canon[j]]
            if changed:
                out["rewrites"] += 1
                out["rewrite_tokens"] += total - read
        if raw and raw[-1]["role"] == "user" and isinstance(raw[-1]["content"], list):
            # The newest tool results, as they first enter context (compression at entry).
            out["entry_raw"] += block_chars(raw[-1])["tool_result"]
            out["entry_sent"] += block_chars(_strip_cc(send[-1]))["tool_result"]
        out["steps"] += 1
        out["read"] += read
        out["write"] += total - read
        out["raw_tokens"] += coef["S"] + sum(tok(block_chars(m), coef, with_s=False) for m in raw)
        out["sent_tokens"] += total
        sent.append(canon)
        written.append(len(canon))
    return out


def _strip_cc(m: dict[str, Any]) -> dict[str, Any]:
    c = m.get("content")
    if isinstance(c, str):  # the API treats a bare string as one text block
        return {**m, "content": [{"type": "text", "text": c}]}
    if isinstance(c, list):
        return {
            **m,
            "content": [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in c
            ],
        }
    return m


# ---------------------------------------------------------------- sections

ARMS = ("plain", "distil", "rtk", "selective", "provider-cm")
BAD = ("api_error", "env_error", "internal_error")


def decompose(rs: list[dict[str, Any]], solved: set[str] | None = None) -> dict[str, Any]:
    u = {
        k: sum(r["usage"][k] for r in rs) for k in ("input", "output", "cache_write", "cache_read")
    }
    c = usd(u)
    tot = sum(c.values())
    steps = sum(r["steps"] for r in rs)
    prompt = u["input"] + u["cache_write"] + u["cache_read"]
    out = {
        "tasks": len(rs),
        "steps": steps,
        "steps_per_task": round(steps / len(rs), 2),
        "usd": round(tot, 4),
        "usd_by_component": {k: round(v, 4) for k, v in c.items()},
        "share_by_component": {k: round(v / tot, 4) for k, v in c.items()},
        "tokens_per_step": {k: round(v / steps, 1) for k, v in u.items()},
        "prompt_tokens_total": prompt,
        "cache_hit_ratio": round(u["cache_read"] / prompt, 4),
        "write_fraction_of_prompt": round(u["cache_write"] / prompt, 4),
        "usd_per_task": round(tot / len(rs), 5),
        "usd_per_step": round(tot / steps, 6),
        "expand_calls": sum(r.get("expand_calls", 0) for r in rs),
        "stops": dict(Counter(r["stop"] for r in rs)),
    }
    if solved is not None:
        n = sum(1 for r in rs if r["instance_id"] in solved)
        out["solved"] = n
        out["usd_per_solved"] = round(tot / n, 4) if n else None
    return out


def impossible_cache(r: dict[str, Any]) -> bool:
    """A fresh append-only run writes its whole final prompt (>= every earlier prompt), so
    cache_write + input >= cache_read / (steps - 1). Below that, the provider served prompt
    this run never wrote: a cache entry left by another arm or an earlier attempt."""
    u, n = r["usage"], r["steps"]
    return n > 1 and u["cache_write"] + u["input"] < u["cache_read"] / (n - 1)


def load_transcripts(d: Path, arm: str, res: dict[tuple[str, str], dict[str, Any]]):
    for f in sorted((d / "transcripts" / arm).glob("*.json")):
        r = res.get((arm, f.stem))
        if r and r["failure_class"] not in BAD:
            yield f.stem, json.loads(f.read_text()), r


def calibrate(lite: Path, longh: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit on longh plain+rtk (clean caches: both equations) + half of the Lite raw transcripts
    (cache-invariant total only); report error on the held-out half."""
    lres = {(r["arm"], r["instance_id"]): r for r in rows(lite / "results.jsonl")}
    hres = {(r["arm"], r["instance_id"]): r for r in rows(longh / "results.jsonl")}
    hs = [(m, r) for a in ("plain", "rtk") for _, m, r in load_transcripts(longh, a, hres)]
    ls = [
        (m, r)
        for a in ("rtk", "provider-cm")
        for _, m, r in load_transcripts(lite, a, lres)
        if not (r.get("arm_stats") or {}).get("applied_calls")
    ]
    train, test = ls[::2], ls[1::2]
    coef = fit(
        [x for m, r in hs for x in fit_rows(m, r["usage"])]
        + [x for m, r in train for x in total_row(m, r["usage"])]
    )

    def err(pairs, which):
        rat = []
        for m, r in pairs:
            x, u = reconstruct(m, coef), r["usage"]
            if which == "total":
                rat.append(
                    (x["pred_write"] + x["pred_read"])
                    / (u["input"] + u["cache_write"] + u["cache_read"])
                )
            else:
                rat.append(x["pred_write"] / (u["cache_write"] + u["input"]))
        q = st.quantiles(rat, n=10)
        return {
            "n": len(rat),
            "median": round(st.median(rat), 3),
            "p10": round(q[0], 3),
            "p90": round(q[-1], 3),
        }

    return coef, {
        "coef_tokens_per_char": {k: round(v, 4) for k, v in coef.items()},
        "lite_heldout_total_prompt_ratio": err(test, "total"),
        "longh_final_prompt_ratio": err(hs, "final"),
        "longh_total_prompt_ratio": err(hs, "total"),
    }


def correct_fresh(r: dict[str, Any], x: dict[str, float]) -> dict[str, Any]:
    """Re-price a task as if it had started on a cold cache: total prompt tokens are kept
    exactly; only the write/read split moves to the append-only model's prediction."""
    u = r["usage"]
    tot = u["input"] + u["cache_write"] + u["cache_read"]
    w = min(max(x["pred_write"], u["cache_write"] + u["input"]), tot)
    return {
        "input": u["input"],
        "output": u["output"],
        "cache_write": w - u["input"],
        "cache_read": tot - w,
    }


def lite_section(lite: Path, coef: dict[str, float]) -> dict[str, Any]:
    rs = rows(lite / "results.jsonl")
    res = {(r["arm"], r["instance_id"]): r for r in rs}
    grades = {(g["arm"], g["instance_id"]): g["status"] for g in rows(lite / "grades.jsonl")}
    common = set.intersection(
        *[
            {
                r["instance_id"]
                for r in rs
                if r["arm"] == a
                and r["failure_class"] not in BAD
                and grades.get((a, r["instance_id"])) in ("resolved", "unresolved")
            }
            for a in ARMS
        ]
    )
    out: dict[str, Any] = {"common_tasks": len(common), "arms": {}}
    for a in ARMS:
        ar = [r for r in rs if r["arm"] == a and r["instance_id"] in common]
        solved = {i for (aa, i), s in grades.items() if aa == a and s == "resolved"}
        d = decompose(ar, solved)
        d["tasks_with_precached_prompt"] = sum(impossible_cache(r) for r in ar)
        out["arms"][a] = d
    # Cold-cache re-pricing of the arms that ran on 2026-10-05 with identical first requests.
    fresh: dict[str, Any] = {}
    for a in ("rtk", "provider-cm"):
        fixed, n_model = [], 0
        for i, m, r in load_transcripts(lite, a, res):
            if i not in common:
                continue
            if (r.get("arm_stats") or {}).get("applied_calls"):
                fixed.append(r)  # server-side edits: raw transcript is not what was sent
                continue
            n_model += 1
            fixed.append({**r, "usage": correct_fresh(r, reconstruct(m, coef))})
        solved = {i for (aa, i), s in grades.items() if aa == a and s == "resolved"}
        d = decompose(fixed, solved)
        d["tasks_repriced"] = n_model
        fresh[a] = d
    out["cold_cache_repriced"] = fresh
    out["paired_vs_plain"] = paired(rs, res, lite, coef, common)
    # Where the plain-equivalent money goes, from the raw same-day trajectories (provider-cm,
    # tasks with no edit applied: their requests are exactly plain's) re-priced cold.
    out["where_the_money_goes_rawtraj"] = money_split(
        [
            m
            for i, m, r in load_transcripts(lite, "provider-cm", res)
            if i in common and not (r.get("arm_stats") or {}).get("applied_calls")
        ],
        [
            r
            for i, m, r in load_transcripts(lite, "provider-cm", res)
            if i in common and not (r.get("arm_stats") or {}).get("applied_calls")
        ],
        coef,
    )
    out["tool_output"] = tool_output_compare(lite, res, common)
    out["behaviour"] = behaviour(lite, res, common)
    return out


def boot(xs: list[float], n: int = 4000) -> list[float]:
    import random

    rnd = random.Random(7)
    ms = sorted(st.mean(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return [round(ms[int(0.025 * n)], 6), round(ms[int(0.975 * n)], 6)]


def paired(rs, res, lite: Path, coef, common) -> dict[str, Any]:
    """Per-task differences vs plain on the same tasks, with 95% bootstrap CIs."""
    plain = {r["instance_id"]: r for r in rs if r["arm"] == "plain"}

    def cost(u):
        return sum(usd(u).values())

    out = {}
    trans = {
        a: {i: (m, r) for i, m, r in load_transcripts(lite, a, res)} for a in ("rtk", "provider-cm")
    }
    for a in ("distil", "rtk", "provider-cm", "selective"):
        dc, ds, dcold, dt = [], [], [], []
        for r in rs:
            i = r["instance_id"]
            if r["arm"] != a or i not in common:
                continue
            p = plain[i]
            dc.append(cost(r["usage"]) - cost(p["usage"]))
            ds.append(r["steps"] - p["steps"])
            tot = lambda u: u["input"] + u["cache_write"] + u["cache_read"]  # noqa: E731
            dt.append(tot(r["usage"]) - tot(p["usage"]))
            if a in trans and i in trans[a] and not (r.get("arm_stats") or {}).get("applied_calls"):
                m, _ = trans[a][i]
                dcold.append(cost(correct_fresh(r, reconstruct(m, coef))) - cost(p["usage"]))
        d = {
            "pairs": len(dc),
            "mean_usd_diff": round(st.mean(dc), 6),
            "ci95": boot(dc),
            "mean_step_diff": round(st.mean(ds), 3),
            "step_ci95": boot([float(x) for x in ds]),
            "mean_prompt_token_diff": round(st.mean(dt)),
            "prompt_token_ci95": boot([float(x) for x in dt]),
            "plain_mean_usd": round(
                st.mean(
                    cost(plain[r["instance_id"]]["usage"])
                    for r in rs
                    if r["arm"] == a and r["instance_id"] in common
                ),
                6,
            ),
        }
        if dcold:
            d["cold_cache_pairs"] = len(dcold)
            d["cold_cache_mean_usd_diff"] = round(st.mean(dcold), 6)
            d["cold_cache_ci95"] = boot(dcold)
        out[a] = d
    return out


def money_split(msgs_list, rs, coef) -> dict[str, Any]:
    """Cold-cache attribution: each token entering context is written once (1.25x) and re-read
    at every later step (0.1x); output priced at POUT. Shares of the reconstructed bill."""
    new, rer = Counter(), Counter()
    out_usd = sum(r["usage"]["output"] for r in rs) * POUT / 1e6
    for m in msgs_list:
        x = reconstruct(m, coef)
        new.update(x["new"])
        rer.update(x["reread"])
    groups = {
        "system+tools": ("system+tools",),
        "task statement": ("problem",),
        "tool output (results)": ("tool_result",),
        "assistant history (tool calls, text, thinking re-sent)": (
            "tool_use",
            "text",
            "thinking_sig",
        ),
    }
    rows_ = {}
    for g, ks in groups.items():
        w = sum(new[k] for k in ks) * W * PIN / 1e6
        rd = sum(rer[k] for k in ks) * R * PIN / 1e6
        rows_[g] = {"write_usd": round(w, 4), "reread_usd": round(rd, 4)}
    total = out_usd + sum(v["write_usd"] + v["reread_usd"] for v in rows_.values())
    for v in rows_.values():
        v["share"] = round((v["write_usd"] + v["reread_usd"]) / total, 4)
    rows_["output tokens (incl. thinking)"] = {
        "usd": round(out_usd, 4),
        "share": round(out_usd / total, 4),
    }
    tr_new = sum(new[k] for k in ("tool_result",))
    return {
        "tasks": len(rs),
        "reconstructed_usd": round(total, 4),
        "by_source": rows_,
        "tool_output_tokens_entered": round(tr_new),
        "tool_output_cost_per_entered_token_x_input_price": round(
            (W * tr_new + R * rer["tool_result"]) / tr_new, 3
        )
        if tr_new
        else None,
    }


def tool_output_compare(lite: Path, res, common) -> dict[str, Any]:
    """Tool-result size per call by command class: rtk (filtered) vs the same-day raw arms."""
    by: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    rew = Counter()
    for a in ("rtk", "provider-cm", "selective"):
        for i, m, r in load_transcripts(lite, a, res):
            if i not in common:
                continue
            for cls, _norm, n, _e in tool_calls(m):
                by[a][cls].append(n)
            if a == "rtk":
                rew["commands"] += r["arm_stats"]["commands"]
                rew["rewritten"] += r["arm_stats"]["rewritten"]
    classes = Counter()
    for a in by:
        for c, v in by[a].items():
            classes[c] += len(v)
    table = {}
    for c, _ in classes.most_common(14):
        table[c] = {
            a: {
                "calls": len(by[a][c]),
                "mean_chars": round(st.mean(by[a][c])) if by[a][c] else None,
                "total_chars": sum(by[a][c]),
            }
            for a in by
        }
    tot = {
        a: {
            "calls": sum(len(v) for v in by[a].values()),
            "total_chars": sum(sum(v) for v in by[a].values()),
        }
        for a in by
    }
    for a in tot:
        tot[a]["mean_chars_per_call"] = round(tot[a]["total_chars"] / tot[a]["calls"])
    return {"rtk_rewrites": dict(rew), "totals": tot, "by_class": table}


def behaviour(d: Path, res, common=None) -> dict[str, Any]:
    out = {}
    for a in ARMS:
        if not (d / "transcripts" / a).exists():
            continue
        calls = steps = dup = 0
        for i, m, r in load_transcripts(d, a, res):
            if common is not None and i not in common:
                continue
            tc = tool_calls(m)
            steps += r["steps"]
            calls += len(tc)
            seen: set[str] = set()
            for cls, norm, _n, _e in tc:
                dup += norm in seen
                seen.add(norm)
        out[a] = {
            "steps": steps,
            "tool_calls": calls,
            "tool_calls_per_step": round(calls / steps, 3) if steps else None,
            "identical_reruns": dup,
            "identical_rerun_rate": round(dup / calls, 4) if calls else None,
        }
    return out


def longh_section(longh: Path, coef: dict[str, float], replay: bool) -> dict[str, Any]:
    rs = rows(longh / "results.jsonl")
    res = {(r["arm"], r["instance_id"]): r for r in rs}
    grades = {(g["arm"], g["instance_id"]): g["status"] for g in rows(longh / "grades.jsonl")}
    arms = ("plain", "distil", "rtk")
    common = set.intersection(*[{r["instance_id"] for r in rs if r["arm"] == a} for a in arms])
    out: dict[str, Any] = {"common_tasks": len(common), "arms": {}}
    for a in arms:
        solved = {i for (aa, i), s in grades.items() if aa == a and s == "resolved"}
        d = decompose([r for r in rs if r["arm"] == a and r["instance_id"] in common], solved)
        d["tasks_with_precached_prompt"] = sum(impossible_cache(r) for r in rs if r["arm"] == a)
        out["arms"][a] = d
    out["where_the_money_goes_plain"] = money_split(
        [m for i, m, r in load_transcripts(longh, "plain", res) if i in common],
        [r for i, m, r in load_transcripts(longh, "plain", res) if i in common],
        coef,
    )
    out["tool_output"] = {}
    for a in arms:
        n = c = 0
        for i, m, r in load_transcripts(longh, a, res):
            if i in common:
                tc = tool_calls(m)
                n += len(tc)
                c += sum(x[2] for x in tc)
        out["tool_output"][a] = {"calls": n, "mean_chars_per_call": round(c / n) if n else None}
    out["behaviour"] = behaviour(longh, res, common)
    if replay:
        rep = Counter()
        for i, m, r in load_transcripts(longh, "distil", res):
            x = replay_distil(m, coef)
            for k, v in x.items():
                rep[k] += v
            rep["actual_write"] += r["usage"]["cache_write"] + r["usage"]["input"]
            rep["actual_read"] += r["usage"]["cache_read"]
        out["distil_replay"] = summarize_replay(rep)
    return out


def summarize_replay(rep: Counter) -> dict[str, Any]:
    return {
        "steps": rep["steps"],
        "raw_prompt_tokens": round(rep["raw_tokens"]),
        "sent_prompt_tokens": round(rep["sent_tokens"]),
        "prompt_reduction": round(1 - rep["sent_tokens"] / rep["raw_tokens"], 4)
        if rep["raw_tokens"]
        else None,
        "sim_write": round(rep["write"]),
        "sim_read": round(rep["read"]),
        "actual_write": round(rep["actual_write"]) if rep["actual_write"] else None,
        "actual_read": round(rep["actual_read"]) if rep["actual_read"] else None,
        "steps_that_rewrote_an_earlier_message": rep["rewrites"],
        "tokens_rewritten_by_those_steps": round(rep["rewrite_tokens"]),
        "entry_tool_chars_raw": rep["entry_raw"],
        "entry_tool_chars_sent": rep["entry_sent"],
        "entry_tool_reduction": round(1 - rep["entry_sent"] / rep["entry_raw"], 4)
        if rep["entry_raw"]
        else None,
    }


def lite_replay(lite: Path, coef: dict[str, float], limit: int) -> dict[str, Any]:
    """distil's transform on the same-day RAW plain-equivalent trajectories (provider-cm, no
    edit applied): what it would have removed and whether it would rewrite cached history."""
    res = {(r["arm"], r["instance_id"]): r for r in rows(lite / "results.jsonl")}
    rep = Counter()
    n = 0
    for i, m, r in load_transcripts(lite, "provider-cm", res):
        if (r.get("arm_stats") or {}).get("applied_calls"):
            continue
        x = replay_distil(m, coef)
        for k, v in x.items():
            rep[k] += v
        n += 1
        if n >= limit:
            break
    d = summarize_replay(rep)
    d["tasks"] = n
    return d


def tbench_section(d: Path) -> dict[str, Any]:
    """Claude Code on Terminal-Bench: per-request meter usage. A 'prefix break' is a request
    whose cache read is < 90% of the previous main-model request's whole prompt."""
    runs = rows(d / "runs.jsonl")
    bad = {(r["task"], r["seed"]) for r in runs if r["status"] == "infra_error"}
    keep = {r["run_id"]: r for r in runs if (r["task"], r["seed"]) not in bad}
    agg: dict[str, Counter] = defaultdict(Counter)
    for f in sorted((d / "meter").glob("*.jsonl")):
        if not f.name.endswith(".run.jsonl"):
            continue  # canaries are pre-flight probes, not runs
        rid = f.name[: -len(".run.jsonl")]
        if rid not in keep:
            continue
        arm = keep[rid]["arm"]
        a = agg[arm]
        a["runs"] += 1
        a["solved"] += keep[rid]["status"] == "solved"
        prev = None
        for line in f.open():
            q = json.loads(line)
            u = q.get("usage") or {}
            if not u or q.get("status") != 200:
                continue
            ui = {
                "input": u.get("input_tokens", 0),
                "output": u.get("output_tokens", 0),
                "cache_write": u.get("cache_creation_input_tokens", 0),
                "cache_read": u.get("cache_read_input_tokens", 0),
            }
            for k, v in ui.items():
                a[k] += v
            a["requests"] += 1
            if not q["model"].startswith("claude-sonnet"):
                continue
            prompt = ui["input"] + ui["cache_write"] + ui["cache_read"]
            if prev and prompt >= 0.9 * prev and ui["cache_read"] < 0.9 * prev:
                a["prefix_breaks"] += 1
                a["break_rewrite_tokens"] += prev - ui["cache_read"]
            prev = prompt
    out = {}
    for arm, a in agg.items():
        u = {k: a[k] for k in ("input", "output", "cache_write", "cache_read")}
        c = usd(u)
        tot = sum(c.values())
        out[arm] = {
            "runs": a["runs"],
            "solved": a["solved"],
            "requests": a["requests"],
            "usd_at_2_10": round(tot, 3),
            "usd_per_solved": round(tot / a["solved"], 4) if a["solved"] else None,
            "share_by_component": {k: round(v / tot, 4) for k, v in c.items()},
            "tokens_per_request": {k: round(v / a["requests"]) for k, v in u.items()},
            "cache_hit_ratio": round(
                u["cache_read"] / max(1, u["input"] + u["cache_write"] + u["cache_read"]), 4
            ),
            "prefix_breaks": a["prefix_breaks"],
            "prefix_break_extra_usd": round(a["break_rewrite_tokens"] * (W - R) * PIN / 1e6, 4),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lite", type=Path, required=True)
    ap.add_argument("--longh", type=Path, required=True)
    ap.add_argument("--tbench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay", action="store_true", help="re-run distil's transform offline")
    ap.add_argument("--replay-limit", type=int, default=10_000)
    a = ap.parse_args(argv)
    if a.replay:
        tmp = tempfile.mkdtemp(prefix="why-rtk-")
        os.environ["DISTIL_HOME"] = os.path.join(tmp, ".distil")  # never the real ~/.distil
        os.environ["HOME"] = tmp
    coef, cal = calibrate(a.lite, a.longh)
    out = {
        "prices": {"in": PIN, "out": POUT, "write_x": W, "read_x": R},
        "token_model": cal,
        "lite": lite_section(a.lite, coef),
        "longh": longh_section(a.longh, coef, a.replay),
        "tbench_pilot": tbench_section(a.tbench),
    }
    if a.replay:
        out["lite"]["distil_replay_on_raw_plain_trajectories"] = lite_replay(
            a.lite, coef, a.replay_limit
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())

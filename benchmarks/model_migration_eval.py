"""Model-migration eval for distil's live certifier.

Question it answers: if the grading model moves from the incumbent (default
claude-opus-4-8) to a newer one, does distil's compression still leave the
agent's next action unchanged -- and does the certifier itself still work?

Each case is one trajectory turn (36 curated turns from corpus/ + one turn per
benchmarks/corpus_xl entry, 100 total). Per (case, rep) it calls distil's REAL
runners -- AnthropicRunner (forced/strict decision tool) and ExpandAwareRunner
(the digest + distil_expand recovery loop) -- on five arms:

  full_a, full_b   uncompressed context, two independent samples (noise floor)
  distil           the cache-aware distil strategy
  expand           distil + the recovery loop (how distil deploys)
  trunc            truncate@160 -- positive control; SHOULD diverge

Metrics (grade dict, all 0/1): equiv_distil (headline), equiv_expand,
self_consist, decided, agree_ref (full_a vs the frozen baseline decision;
on baseline itself = self_consist), trunc_detect (control sensitivity).
An undecided side never counts as a match.

    .venv/bin/python benchmarks/model_migration_eval.py --variant baseline \
        --model claude-opus-4-8 --reps 2
    .venv/bin/python benchmarks/model_migration_eval.py --variant v1 \
        --model claude-opus-5-5 --reps 2
    --fake oracle|null|flip   offline wiring checks, no API calls
    --limit N                 first N cases (pilot)

Output: .claude/hillclimb/decision-equivalence/<variant>/{results,errors}.jsonl
+ traces/. Report: node <skill>/shared/evals/report/build-report-lite.mjs <flow>.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distil.compress.strategies import distil as distil_strategy  # noqa: E402
from distil.corpus import load_corpus  # noqa: E402
from distil.eval import _truncate  # noqa: E402
from distil.replay import prompts  # noqa: E402
from distil.replay.anthropic_runner import AnthropicRunner  # noqa: E402
from distil.replay.expand_runner import ExpandAwareRunner, build_restore  # noqa: E402
from distil.tokenizer import DEFAULT as TOK  # noqa: E402

FLOW = ROOT / ".claude/hillclimb/decision-equivalence"
HARNESS = [
    Path(__file__).resolve(),
    ROOT / "distil/replay/anthropic_runner.py",
    ROOT / "distil/replay/expand_runner.py",
    ROOT / "distil/replay/prompts.py",
    ROOT / "distil/compress/strategies.py",
]
TRUNC = _truncate(160)
NO = "<no-decision>"


# --- cases ---------------------------------------------------------------------


def load_cases() -> list[dict]:
    cases = []
    for e in load_corpus(ROOT / "corpus"):
        for t in e.trajectory.turns:
            cases.append(
                {
                    "id": f"{e.trajectory.id}-t{t.index}",
                    "turn": t,
                    "tags": [e.domain, "curated", f"turn{t.index}"],
                    "title": e.title,
                }
            )
    seen: dict[str, int] = {}
    for e in load_corpus(ROOT / "benchmarks/corpus_xl"):
        dom = e.trajectory.id.rsplit("-", 1)[0]
        k = seen.get(dom, 0)
        seen[dom] = k + 1
        turns = e.trajectory.turns
        t = turns[k % len(turns)]  # spread across history depths, deterministically
        cases.append(
            {
                "id": f"{e.trajectory.id}-t{t.index}",
                "turn": t,
                "tags": [dom, "synthetic", f"turn{t.index}"],
                "title": e.title,
            }
        )
    return cases


# --- instrumentation -------------------------------------------------------------


class Recorder:
    """Wraps an anthropic client; records every response per (case, rep)."""

    def __init__(self, inner, requested: str):
        self.inner, self.requested, self.calls = inner, requested, []
        self.messages = self

    def create(self, **kw):
        t0 = time.monotonic()
        r = self.inner.messages.create(**kw)
        served = getattr(r, "model", None)
        if served and not _same_model(served, self.requested):
            raise ServingMismatch(f"served {served} != requested {self.requested}")
        u = r.usage
        self.calls.append(
            {
                "model": served,
                "stop_reason": r.stop_reason,
                "latency_s": time.monotonic() - t0,
                "tool_choice": (kw.get("tool_choice") or {}).get("type"),
                "usage": {
                    k: getattr(u, k, 0) or 0
                    for k in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                },
            }
        )
        return r


class ServingMismatch(Exception):
    failure_class = "serving_substitution"


def _same_model(served: str, requested: str) -> bool:
    if served == requested:
        return True
    rest = served[len(requested) :] if served.startswith(requested) else None
    return rest is not None and bool(re.fullmatch(r"[-@](\d{8}|\d{4}-\d{2}-\d{2})", rest))


# --- offline fake clients (wiring checks, no spend) -------------------------------


class FakeClient:
    """oracle: constant decision (every equiv -> 1). null: no decision (-> 0).
    flip: random decision per call (equiv ~ chance)."""

    def __init__(self, mode: str, model: str):
        self.mode, self.model, self.messages = mode, model, self

    def create(self, **kw):
        if self.mode == "null":
            content = [SimpleNamespace(type="text", text="")]
        else:
            tgt = "x" if self.mode == "oracle" else str(random.random())
            if kw.get("tools"):
                content = [SimpleNamespace(type="tool_use", input={"action": "act", "target": tgt})]
            else:
                content = [
                    SimpleNamespace(type="text", text=json.dumps({"action": "act", "target": tgt}))
                ]
        return SimpleNamespace(
            model=self.model,
            content=content,
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=20,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


# --- one case ---------------------------------------------------------------------


def run_case(case: dict, client, model: str, effort: str | None = None) -> dict:
    blocks = case["turn"].blocks
    idx = case["turn"].index
    rec = Recorder(client, model)
    base = AnthropicRunner(model=model, client=rec, effort=effort)
    arms: dict[str, str] = {}
    arm_calls: dict[str, int] = {}

    def arm(name, fn):
        n0 = len(rec.calls)
        arms[name] = fn()
        arm_calls[name] = len(rec.calls) - n0

    comp = distil_strategy(blocks, idx)
    arm("full_a", lambda: base.decide(blocks))
    arm("full_b", lambda: base.decide(blocks))
    arm("distil", lambda: base.decide(comp))
    arm("expand", lambda: ExpandAwareRunner(base).decide(comp, build_restore(blocks)))
    # Same free-text decision protocol as `expand`, on the uncompressed turn (no
    # handles, so ExpandAwareRunner goes straight to the text decision prompt).
    # `expand` is graded against THIS arm, not full_a: full_a uses the forced
    # decision tool with an action enum, and that format difference alone flips
    # actions - it is not a compression effect.
    arm("full_text", lambda: ExpandAwareRunner(base).decide(blocks, {}))
    arm("trunc", lambda: base.decide(TRUNC(blocks, idx)))

    full_tok = sum(TOK.count(b.text) for b in blocks)
    comp_tok = sum(TOK.count(b.text) for b in comp)
    usage = (
        {k: sum(c["usage"][k] for c in rec.calls) for k in rec.calls[0]["usage"]}
        if rec.calls
        else {}
    )
    stops = [c["stop_reason"] for c in rec.calls]
    system, user = prompts.render(blocks)
    _, cuser = prompts.render(comp)
    transcript = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        *(
            {
                "role": "tool_call",
                "name": f"{a} ({arm_calls[a]} call{'s' * (arm_calls[a] != 1)})",
                "content": fp,
            }
            for a, fp in arms.items()
        ),
        {
            "role": "assistant",
            "content": "distil-compressed user turn (what the `distil` arm saw):\n\n" + cuser,
        },
    ]
    return {
        "arms": arms,
        "transcript": transcript,
        "calls": rec.calls,
        "model": rec.calls[0]["model"] if rec.calls else model,
        "usage": usage,
        "latency_s": round(sum(c["latency_s"] for c in rec.calls), 3),
        "stop_reason": "max_tokens"
        if "max_tokens" in stops
        else ("refusal" if "refusal" in stops else "end_turn"),
        "n_calls": len(rec.calls),
        "savings": round(1 - comp_tok / full_tok, 4) if full_tok else 0.0,
        "forced_tool_choice": base.forced_tool_choice,
    }


def _act(fp: str) -> str:
    return json.loads(fp)["action"] if fp.startswith("{") else fp


def grade(run: dict, ref_fp: str | None) -> dict:
    """distil's exact {action,target} fingerprint equality (what production
    certification uses), plus an action-only view: the target is free text, so
    exact equality also counts paraphrase ("refunds on delivered" vs "refund of
    delivered") as a decision change. The *_act metrics separate real action
    flips from that noise."""
    a = run["arms"]

    def eq(x, y, f=lambda v: v):
        return 1 if (x != NO and y != NO and f(x) == f(y)) else 0

    ref = ref_fp if ref_fp is not None else a["full_b"]
    return {
        "equiv_distil": eq(a["full_a"], a["distil"]),
        "equiv_distil_act": eq(a["full_a"], a["distil"], _act),
        "equiv_expand": eq(a["full_text"], a["expand"]),
        "equiv_expand_act": eq(a["full_text"], a["expand"], _act),
        "format_consist_act": eq(a["full_a"], a["full_text"], _act),
        "self_consist": eq(a["full_a"], a["full_b"]),
        "self_consist_act": eq(a["full_a"], a["full_b"], _act),
        "decided": 1 if a["full_a"] != NO else 0,
        "agree_ref": eq(a["full_a"], ref),
        "agree_ref_act": eq(a["full_a"], ref, _act),
        "trunc_detect": 1 if (a["full_a"] != NO and _act(a["trunc"]) != _act(a["full_a"])) else 0,
    }


def backfill_full_text(vdir: Path, client, concurrency: int) -> None:
    """Add the full_text arm to rows recorded before it existed: one extra call
    per row with the variant's own model/effort; usage and call count are added
    to the row so cost stays complete. Rewrites results.jsonl atomically."""
    cfg = json.loads((vdir / "config.json").read_text())
    cases = {c["id"]: c for c in load_cases()}
    rp = vdir / "results.jsonl"
    rows = [json.loads(ln) for ln in rp.read_text().splitlines() if ln.strip()]
    todo = [r for r in rows if "full_text" not in r["meta"]["arms"]]
    print(f"[{vdir.name}] backfilling full_text on {len(todo)} rows ({cfg['model']}, effort={cfg.get('effort')})", file=sys.stderr)

    def one(r):
        rec = Recorder(client, cfg["model"])
        base = AnthropicRunner(model=cfg["model"], client=rec, effort=cfg.get("effort"))
        fp = ExpandAwareRunner(base).decide(cases[r["prompt_id"]]["turn"].blocks, {})
        return r, fp, rec.calls

    with cf.ThreadPoolExecutor(concurrency) as ex:
        for r, fp, calls in ex.map(one, todo):
            r["meta"]["arms"]["full_text"] = fp
            for c in calls:
                for k, v in c["usage"].items():
                    r["usage"][k] = r["usage"].get(k, 0) + v
            r["tool_calls"] += len(calls)
            r["latency_s"] = round(r["latency_s"] + sum(c["latency_s"] for c in calls), 3)
    tmp = rp.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(rp)


def regrade(flow: Path) -> None:
    """Recompute every stored row's grade from its recorded arm decisions - no API calls."""
    refp = flow / "baseline/ref/decisions.json"
    ref = json.loads(refp.read_text()) if refp.exists() else {}
    for v in sorted(p for p in flow.iterdir() if p.is_dir() and re.fullmatch(r"baseline|v\d+", p.name)):
        rp = v / "results.jsonl"
        if not rp.exists():
            continue
        rows = [json.loads(ln) for ln in rp.read_text().splitlines() if ln.strip()]
        for r in rows:
            r["grade"] = grade({"arms": r["meta"]["arms"]}, None if v.name == "baseline" else ref.get(r["prompt_id"]))
        rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"regraded {len(rows)} rows in {v.name}", file=sys.stderr)


# --- harness gate ---------------------------------------------------------------


def harness_sha() -> str:
    h = hashlib.sha256()
    for p in HARNESS:
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def check_harness(approve: bool) -> None:
    sp = FLOW / "_state.json"
    st = json.loads(sp.read_text())
    sha = harness_sha()
    if st.get("harness_sha") == sha:
        return
    if approve:
        st["harness_sha"] = sha
        sp.write_text(json.dumps(st, indent=2) + "\n")
        print(f"harness approved: {sha[:12]}", file=sys.stderr)
        return
    print(
        f"harness sha {sha[:12]} not approved (recorded: {str(st.get('harness_sha'))[:12]}). "
        "Review the harness, then re-run with --approve-harness.",
        file=sys.stderr,
    )
    sys.exit(2)


# --- main -----------------------------------------------------------------------


def main() -> None:
    global FLOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--fake", choices=["oracle", "null", "flip"])
    ap.add_argument("--status", action="store_true", help="print the cross-variant status table and exit")
    ap.add_argument("--regrade", action="store_true", help="recompute grades from stored decisions (no API calls)")
    ap.add_argument("--backfill-full-text", action="store_true", help="add the full_text arm to existing rows of --variant")
    ap.add_argument("--flow", type=Path, default=FLOW)
    ap.add_argument("--approve-harness", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"baseline|v[1-9]\d*", a.variant):
        sys.exit("--variant must be 'baseline' or 'v<N>'")
    FLOW = a.flow
    if a.backfill_full_text:
        if not a.fake:
            check_harness(a.approve_harness)
        if a.fake:
            client = FakeClient(a.fake, a.model)
        else:
            import anthropic

            client = anthropic.Anthropic(max_retries=6)
        backfill_full_text(FLOW / a.variant, client, a.concurrency)
        return
    if a.regrade:
        regrade(FLOW)
    if a.status or a.regrade:
        print(status_table(FLOW))
        return
    if not a.fake:
        if not a.fake:
            check_harness(a.approve_harness)

    vdir = FLOW / a.variant
    (vdir / "traces").mkdir(parents=True, exist_ok=True)
    # One configuration per variant dir: resuming with a different model/effort
    # would silently mix two cells into one row set.
    cfg = {"model": a.model, "effort": a.effort, "reps": a.reps, "fake": a.fake}
    cfgp = vdir / "config.json"
    if cfgp.exists():
        old = json.loads(cfgp.read_text())
        if {k: old.get(k) for k in ("model", "effort", "fake")} != {k: cfg[k] for k in ("model", "effort", "fake")}:
            sys.exit(f"{vdir} was run as {old}; use a new --variant for {cfg}")
    cfgp.write_text(json.dumps(cfg, indent=2))
    sp = vdir / "summary.json"
    if not sp.exists():
        sp.write_text(json.dumps({"description": f"{a.model} @ effort={a.effort or 'default'}", "target": "code"}))
    results, errors = vdir / "results.jsonl", vdir / "errors.jsonl"
    done = set()
    if results.exists():
        for ln in results.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["prompt_id"], r["rep"]))

    ref: dict[str, str] = {}
    refp = FLOW / "baseline/ref/decisions.json"
    if a.variant != "baseline":
        if not refp.exists():
            sys.exit("run the baseline first -- agree_ref needs its frozen decisions")
        ref = json.loads(refp.read_text())

    cases = load_cases()[: a.limit] if a.limit else load_cases()
    tasks = [(c, r) for c in cases for r in range(a.reps) if (c["id"], r) not in done]
    print(
        f"[{a.variant}] {len(tasks)} of {len(cases) * a.reps} (case,rep) to run on {a.model}",
        file=sys.stderr,
    )

    if a.fake:
        client = FakeClient(a.fake, a.model)
    else:
        import anthropic

        client = anthropic.Anthropic(max_retries=6)  # SDK: jittered backoff on 429/5xx/overloaded
    lock = threading.Lock()
    ok = fail = 0

    def one(c, rep):
        run = run_case(c, client, a.model, a.effort)
        g = grade(run, ref.get(c["id"]) if a.variant != "baseline" else None)
        return run, g

    t_start = time.monotonic()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(one, c, r): (c, r, time.monotonic()) for c, r in tasks}
        for f in cf.as_completed(futs):
            c, rep, t0 = futs[f]
            try:
                run, g = f.result(timeout=max(1.0, a.timeout_s - (time.monotonic() - t0)))
                row = {
                    "prompt_id": c["id"],
                    "rep": rep,
                    "prompt": c["title"],
                    "tags": c["tags"],
                    "model": run["model"],
                    "usage": run["usage"],
                    "stop_reason": run["stop_reason"],
                    "status": "truncated" if run["stop_reason"] == "max_tokens" else "ok",
                    "latency_s": run["latency_s"],
                    "tool_calls": run["n_calls"],
                    "savings": run["savings"],
                    "grade": g,
                    "meta": {
                        "arms": run["arms"],
                        "forced_tool_choice": run["forced_tool_choice"],
                        "effort": a.effort,
                        "refusal": run["stop_reason"] == "refusal",
                    },
                }
                with lock:
                    with results.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    (vdir / "traces" / f"{c['id']}_rep{rep}.json").write_text(
                        json.dumps(run["transcript"], indent=2)
                    )
                    if a.variant == "baseline" and rep == 0:
                        refp.parent.mkdir(parents=True, exist_ok=True)
                        frozen = json.loads(refp.read_text()) if refp.exists() else {}
                        frozen.setdefault(c["id"], run["arms"]["full_a"])
                        refp.write_text(json.dumps(frozen, indent=1, sort_keys=True))
                ok += 1
            except BaseException as e:  # noqa: BLE001 -- SystemExit from AnthropicRunner._create included
                if isinstance(e, KeyboardInterrupt):
                    raise
                fail += 1
                cls = (
                    "timeout"
                    if isinstance(e, (cf.TimeoutError, TimeoutError))
                    else getattr(e, "failure_class", "harness_or_api_error")
                )
                with lock, errors.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "prompt_id": c["id"],
                                "rep": rep,
                                "failure_class": cls,
                                "error": str(e)[:500],
                            }
                        )
                        + "\n"
                    )
                print(f"  {c['id']} rep{rep} FAILED [{cls}]: {str(e)[:200]}", file=sys.stderr)

    # headline + 95% CI (Wilson) over this variant's rows, all reps
    rows = [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r["status"] == "ok"]
    for m in ("equiv_distil_act", "equiv_distil", "equiv_expand_act", "format_consist_act", "self_consist_act", "decided", "agree_ref_act", "trunc_detect"):
        n = len(rows)
        k = sum(r["grade"][m] for r in rows)
        if n:
            p, z = k / n, 1.96
            d = 1 + z * z / n
            c0 = (p + z * z / (2 * n)) / d
            hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            print(
                f"{a.variant:>9} {m:<14} {p * 100:5.1f}%  [{(c0 - hw) * 100:4.1f}, {(c0 + hw) * 100:4.1f}]  n={n}"
            )
    print(
        f"[{a.variant}] {ok} ok, {fail} failed, {time.monotonic() - t_start:.0f}s -> {results}",
        file=sys.stderr,
    )
    sys.exit(1 if fail else 0)


# --- cost + status table ----------------------------------------------------------

# $/MTok (input, output), first-party list prices. Cache write 1.25x input, read 0.1x.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0), "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0), "claude-haiku-4-5": (1.0, 5.0),
}


def cost_usd(model: str, u: dict) -> float:
    base = next((m for m in PRICES if model == m or model.startswith(m + "-") or model.startswith(m + "@")), None)
    if base is None:
        raise KeyError(f"no price for {model}; add it to PRICES")
    pin, pout = PRICES[base]
    return (u.get("input_tokens", 0) * pin + u.get("cache_creation_input_tokens", 0) * pin * 1.25
            + u.get("cache_read_input_tokens", 0) * pin * 0.1 + u.get("output_tokens", 0) * pout) / 1e6


def _paired_ci(base: dict, var: dict) -> tuple[float, float, int]:
    """Mean paired difference (var - base) over shared cases, per-case mean over reps, 95% CI."""
    ids = sorted(set(base) & set(var))
    d = [sum(var[i]) / len(var[i]) - sum(base[i]) / len(base[i]) for i in ids]
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, 1.96 * sd / math.sqrt(len(d)), len(d)


def status_table(flow: Path) -> str:
    st = json.loads((flow / "_state.json").read_text())
    test = set(st.get("test_ids") or [])
    vs = sorted([p for p in flow.iterdir() if p.is_dir() and re.fullmatch(r"baseline|v\d+", p.name)],
                key=lambda p: -1 if p.name == "baseline" else int(p.name[1:]))
    per: dict[str, dict] = {}
    lines = ["| variant | config | n | distil equiv (act / exact) | expand equiv (act / exact) | self-consist (act / exact) "
             "| format consist (act) | agree w/ ref (act) | trunc caught | Δ distil act-equiv vs base (test, 95% CI) | $/case | out tok/case "
             "| s/case | errors | spend |",
             "|" + "---|" * 15]
    for v in vs:
        rows = [json.loads(ln) for ln in (v / "results.jsonl").read_text().splitlines() if ln.strip()] \
            if (v / "results.jsonl").exists() else []
        errs = sum(1 for ln in (v / "errors.jsonl").read_text().splitlines() if ln.strip()) \
            if (v / "errors.jsonl").exists() else 0
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            continue
        cfg = json.loads((v / "config.json").read_text()) if (v / "config.json").exists() else {}
        byid: dict[str, list] = {}
        for r in ok:
            byid.setdefault(r["prompt_id"], []).append(r["grade"]["equiv_distil_act"])
        per[v.name] = {i: x for i, x in byid.items() if not test or i in test}
        costs = [cost_usd(r["model"], r["usage"]) for r in rows]

        def pct(m):
            return f"{100 * sum(r['grade'][m] for r in ok) / len(ok):.1f}%"

        if v.name == "baseline":
            delta = "—"
        else:
            m, hw, n = _paired_ci(per.get("baseline", {}), per[v.name])
            delta = f"{100 * m:+.1f} ± {100 * hw:.1f} pts (n={n})"
        lines.append(
            f"| {v.name} | {cfg.get('model', '?')} / {cfg.get('effort') or 'default'} | {len(ok)} "
            f"| {pct('equiv_distil_act')} / {pct('equiv_distil')} | {pct('equiv_expand_act')} / {pct('equiv_expand')} "
            f"| {pct('self_consist_act')} / {pct('self_consist')} | {pct('format_consist_act')} | {pct('agree_ref_act')} | {pct('trunc_detect')} | {delta} "
            f"| ${sum(costs) / len(rows):.4f} | {sum(r['usage'].get('output_tokens', 0) for r in rows) / len(rows):.0f} "
            f"| {sum(r['latency_s'] for r in rows) / len(rows):.1f} | {errs} | ${sum(costs):.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

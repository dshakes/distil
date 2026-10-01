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
]  # distil/compress/ is deliberately NOT here: it is what the compression hillclimb edits
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


# --- real cases: public τ-bench trajectories ------------------------------------------

# sierra-research/tau-bench historical_trajectories (MIT). Fetched into a gitignored
# cache by --fetch-tau-bench and verified by sha256; only the case list is committed.
TAU_URL = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/{}.json"
TAU_SHA256 = {
    "gpt-4o-airline": "e9e6c0297660c537f83d4fd9c476ce7a9a86ecd2784874b7bfc13be598e37bfa",
    "gpt-4o-retail": "df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b",
    "sonnet-35-new-airline": "fe62fcd514b855b36f156dd4c3c7748597b392b006aff739b53337a9f3ba94d1",
    "sonnet-35-new-retail": "0df526398e9d2720c32d340815cffb04fe8c4f8a61b1f4f84bf3bb558f760131",
}
TAU_DIR = ROOT / "benchmarks/.cache/tau-bench"
REAL_CASES = ROOT / "benchmarks/model_migration_cases_real.json"


def fetch_tau_bench() -> None:

    TAU_DIR.mkdir(parents=True, exist_ok=True)
    for name, sha in TAU_SHA256.items():
        p = TAU_DIR / f"{name}.json"
        if not p.exists():
            print(f"fetching {name}.json", file=sys.stderr)
            _fetch_verified(TAU_URL.format(name), p, sha)
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sha:
            sys.exit(f"{p}: sha256 {got[:12]} != pinned {sha[:12]} -- delete it and re-fetch")


def _fetch_verified(url: str, p: Path, sha: str) -> None:
    """Download to a .part file and move it into place only once its sha256 matches, so a
    corrupt or truncated download never lands at `p` (which later runs would trust)."""
    import urllib.request

    part = p.with_suffix(p.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    got = hashlib.sha256(part.read_bytes()).hexdigest()
    if got != sha:
        part.unlink()
        sys.exit(f"{url}: sha256 {got[:12]} != pinned {sha[:12]} -- download rejected")
    part.replace(p)


def _tool_menu(path: Path) -> str:
    """τ-bench trajectories don't ship tool schemas, so without this block the
    certifier free-types action names. Reconstructed from every tool the
    domain's recorded agents called (names + argument keys), plus `respond`
    for a plain reply to the user."""
    tools: dict[str, set] = {}
    for ep in json.loads(path.read_text()):
        for m in ep.get("traj") or ep.get("messages") or []:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tools.setdefault(fn.get("name", "?"), set()).update(
                    args if isinstance(args, dict) else {}
                )
    lines = [
        "AVAILABLE TOOLS (reconstructed from the tools this domain's recorded agents called; names and argument keys only):"
    ]
    lines += [f"- {n}({', '.join(sorted(a))})" for n, a in sorted(tools.items())]
    lines.append("- respond(content): reply to the user in text instead of calling a tool")
    return "\n".join(lines)


_TAU_CACHE: dict[str, tuple] = {}


def _tau(name: str):
    """(entries, gold, tools_block) for one τ-bench file, loaded once."""
    if name not in _TAU_CACHE:
        from distil.replay.realtrace import gold_actions, load_tau_bench
        from distil.trajectory import Block, Kind, Stability

        path = TAU_DIR / f"{name}.json"
        if not path.exists():
            sys.exit(f"{path} missing -- run with --fetch-tau-bench first")
        es = load_tau_bench(path)
        tools = Block(
            id="tools",
            kind=Kind.TOOLS,
            text=_tool_menu(path),
            stability=Stability.STABLE,
            decision_relevant=True,
        )
        _TAU_CACHE[name] = (es, gold_actions(es), tools)
    return _TAU_CACHE[name]


def _with_tools(turn, tools):
    from distil.trajectory import Turn

    b = list(turn.blocks)
    return Turn(index=turn.index, blocks=b[:1] + [tools] + b[1:])


def select_real_cases(total: int = 100, seed: int = 20260929, min_savings: float = 0.10) -> None:
    """Pick per_file cases per τ-bench file: one turn per episode (independence),
    only from turns where distil saves >= min_savings (elsewhere compression is a
    no-op and equivalence is trivially 1), episodes and turns drawn at random."""
    rng = random.Random(seed)
    out = []
    pools = {}
    for name in sorted(TAU_SHA256):
        es, _, tools = _tau(name)
        eligible = []
        for k, e in enumerate(es):
            turns = []
            for t in e.trajectory.turns:
                tt = _with_tools(t, tools)
                full = sum(TOK.count(b.text) for b in tt.blocks)
                comp = sum(TOK.count(b.text) for b in distil_strategy(tt.blocks, tt.index))
                if full and 1 - comp / full >= min_savings:
                    turns.append(t.index)
            if turns:
                eligible.append((k, turns))
        pools[name] = eligible
    # Equal share per file; a file with too few eligible episodes gives its
    # shortfall to the others (largest remaining pool first).
    quota = {n: min(total // len(pools), len(p)) for n, p in pools.items()}
    while sum(quota.values()) < total and any(quota[n] < len(p) for n, p in pools.items()):
        n = max(
            (n for n in pools if quota[n] < len(pools[n])), key=lambda n: len(pools[n]) - quota[n]
        )
        quota[n] += 1
    for name, eligible in pools.items():
        es = _tau(name)[0]
        for k, turns in rng.sample(eligible, quota[name]):
            out.append(
                {
                    "file": name,
                    "episode": k,
                    "traj_id": es[k].trajectory.id,
                    "turn": rng.choice(turns),
                }
            )
    REAL_CASES.write_text(
        json.dumps(
            {
                "source": "tau-bench historical_trajectories",
                "seed": seed,
                "min_savings": min_savings,
                "eligible_episodes": {n: len(p) for n, p in pools.items()},
                "quota": quota,
                "cases": out,
            },
            indent=1,
        )
    )
    print(f"wrote {len(out)} cases to {REAL_CASES}", file=sys.stderr)


def load_real_cases() -> list[dict]:
    spec = json.loads(REAL_CASES.read_text())
    cases = []
    for c in spec["cases"]:
        es, gold, tools = _tau(c["file"])
        e = es[c["episode"]]
        turn = _with_tools(next(t for t in e.trajectory.turns if t.index == c["turn"]), tools)
        g = gold.get((e.trajectory.id, c["turn"]))
        recorder, domain = c["file"].rsplit("-", 1)
        cases.append(
            {
                "id": f"{c['file']}-ep{c['episode']}-t{c['turn']}",
                "turn": turn,
                "tags": [domain, recorder, f"turn{c['turn']}"],
                "title": f"τ-bench {domain} (recorded by {recorder}) episode {c['episode']} turn {c['turn']}",
                "gold": prompts.canonical(g.action, g.target) if g else None,
            }
        )
    return cases


# --- coding cases: public SWE-agent trajectories ---------------------------------------

# 120 seeded-random GPT-4o SWE-agent runs on SWE-bench Lite (swe-bench-submissions S3,
# lite/20240728_sweagent_gpt4o/trajs/), pinned by benchmarks/swe_agent_trajs_manifest.json.
SWE_DIR = ROOT / "benchmarks/.cache/swe-agent"
SWE_MANIFEST = ROOT / "benchmarks/swe_agent_trajs_manifest.json"
SWE_URL = "https://swe-bench-submissions.s3.amazonaws.com/lite/20240728_sweagent_gpt4o/trajs/{}"
CODING_CASES = ROOT / "benchmarks/model_migration_cases_coding.json"


def fetch_swe_agent() -> None:

    SWE_DIR.mkdir(parents=True, exist_ok=True)
    for name, sha in json.loads(SWE_MANIFEST.read_text())["sha256"].items():
        p = SWE_DIR / name
        if not p.exists():
            _fetch_verified(SWE_URL.format(name), p, sha)
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            sys.exit(f"{p}: sha256 mismatch -- delete it and re-fetch")


def _swe_blocks(req: dict, obs: list[str]):
    """Blocks for one decision point, one per message so a recovery loop can restore a
    single digested observation. `obs` are the observation texts to show (original, or
    what distil's serving adapter made of them); the full and served views share this
    one formatter, so compression is the only difference between them."""
    from distil.trajectory import Block, Kind, Stability

    inst = req["inst"]
    blocks = []
    if req["system"]:
        blocks.append(Block(f"{inst}:system", Kind.SYSTEM, req["system"], Stability.STABLE, True))
    blocks.append(Block(f"{inst}:tools", Kind.TOOLS, req["menu"], Stability.STABLE, True))
    blocks.append(Block(f"{inst}:task", Kind.USER, req["task"], Stability.STABLE, True))
    last = len(req["pairs"]) - 1
    for k, ((agent, _), o) in enumerate(zip(req["pairs"], obs)):
        blocks.append(Block(f"{inst}:agent@{k}", Kind.HISTORY, agent, Stability.SETTLING))
        blocks.append(
            Block(
                f"{inst}:obs@{k}",
                Kind.TOOL_OUTPUT,
                o or "(no output)",
                Stability.VOLATILE if k == last else Stability.SETTLING,
                k == last,
            )
        )
    return blocks


def serve(req: dict):
    """What distil's serving adapter sends for this decision point: the history as the
    Anthropic tool_use/tool_result request an agent client would make, run through
    distil.adapters.anthropic.compress_messages with the client's cache breakpoint on
    the newest tool result (a caching client - every tool output is digestible, the
    freshest included). Returns (blocks, restore) where restore maps each handle in the
    served text to its original."""
    from distil.adapters.anthropic import compress_messages

    msgs = [{"role": "user", "content": req["task"]}]
    for k, (agent, o) in enumerate(req["pairs"]):
        tid = f"toolu_{k:04d}"
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": agent},
                    {
                        "type": "tool_use",
                        "id": tid,
                        "name": req["tools"][k],
                        "input": {"command": req["cmds"][k]},
                    },
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tid, "content": o or "(no output)"}
                ],
            }
        )
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    # persist=False: an eval must not write (or read back) the user's restore store.
    out, store = compress_messages(msgs, persist=False)
    served = []
    for m in out[2::2]:
        c = m["content"][-1]["content"]
        served.append(
            c
            if isinstance(c, str)
            else "".join(x.get("text", "") for x in c if isinstance(x, dict))
        )
    blocks = _swe_blocks(req, served)
    restore = {}
    for b in blocks:
        for h in re.findall(r"handle=([0-9a-f]{8})", b.text):
            try:
                restore[h] = store.expand(h)
            except Exception:  # noqa: BLE001 - an unexpandable handle stays folded
                pass
    return blocks, restore


def _swe_turns(path: Path):
    """[(turn, gold_fingerprint, req)] for every non-demo assistant step of one .traj,
    rebuilt from the trajectory's own `history` - the exact messages the agent saw, in
    order. (distil.replay.realtrace.load_swe_bench drops the system prompt and the issue
    for this format and pairs each observation with the action that produced it; this
    builds the decision point correctly.) `req` is the raw material for serve()."""
    from distil.trajectory import Turn

    d = json.loads(path.read_text())
    h = [m for m in d.get("history") or [] if not m.get("is_demo")]
    sysmsg = next((m for m in h if m.get("role") == "system"), None)
    sigs = re.findall(r"signature:\s*(.+)", sysmsg["content"]) if sysmsg else []
    names = []
    menu = ["AVAILABLE TOOLS (SWE-agent commands, from the system prompt's signatures):"]
    for sg in sigs:
        name, *args = sg.strip().split()
        names.append(name)
        menu.append(f"- {name}({' '.join(args)})")
    menu.append("- bash(command): any other shell command (python, pytest, ls, cd, grep, rm, ...)")

    def command(text):
        m = re.search(r"```\s*\n?(.*?)(?:\n|```)", text or "", re.S)
        return (m.group(1) if m else "").strip()

    msgs = [m for m in h if m.get("role") in ("user", "assistant")]
    out = []
    for j, m in enumerate(msgs):
        if m["role"] != "assistant" or j < 2 or msgs[j - 1]["role"] != "user":
            continue
        rest = msgs[1:j]  # assistant, observation, assistant, observation, ...
        pairs = [
            (rest[k]["content"], rest[k + 1]["content"])
            for k in range(0, len(rest) - 1, 2)
            if rest[k]["role"] == "assistant" and rest[k + 1]["role"] == "user"
        ]
        if not pairs:
            continue
        req = {
            "inst": path.stem,
            "system": sysmsg["content"] if sysmsg else "",
            "menu": "\n".join(menu),
            "task": msgs[0]["content"],
            "pairs": pairs,
            "cmds": [command(a) for a, _ in pairs],
        }
        # Name each call as a tool-use client would: SWE-agent's own commands (open, goto,
        # edit, ...) as tools, everything else as bash - distil's exact-quote provenance
        # keys on the tool name, so an all-"bash" request would hide it.
        req["tools"] = [
            c.split(" ", 1)[0] if c.split(" ", 1)[0] in names else "bash" for c in req["cmds"]
        ]
        line = command(m["content"])
        verb, _, rest_ = line.partition(" ")
        gold = (
            prompts.canonical(verb if verb in names else "bash", rest_ if verb in names else line)
            if line
            else None
        )
        out.append((Turn(j, _swe_blocks(req, [o for _, o in pairs])), gold, req))
    return out


def select_coding_cases(total: int = 100, seed: int = 20260929, min_savings: float = 0.10) -> None:
    """One decision point per trajectory (independence), drawn at random from turns where
    distil saves >= min_savings; trajectories themselves drawn at random."""
    rng = random.Random(seed)
    eligible = []
    for p in sorted(SWE_DIR.glob("*.traj")):
        ok = []
        for t, _, _ in _swe_turns(p):
            full = sum(TOK.count(b.text) for b in t.blocks)
            comp = sum(TOK.count(b.text) for b in distil_strategy(t.blocks, t.index))
            if full and 1 - comp / full >= min_savings:
                ok.append(t.index)
        if ok:
            eligible.append((p.name, ok))
    pick = rng.sample(eligible, min(total, len(eligible)))
    cases = [{"file": n, "turn": rng.choice(ok)} for n, ok in sorted(pick)]
    CODING_CASES.write_text(
        json.dumps(
            {
                "source": "swe-agent gpt-4o trajs (SWE-bench Lite)",
                "seed": seed,
                "min_savings": min_savings,
                "eligible_trajectories": len(eligible),
                "cases": cases,
            },
            indent=1,
        )
    )
    print(
        f"wrote {len(cases)} cases ({len(eligible)} eligible trajectories) to {CODING_CASES}",
        file=sys.stderr,
    )


def savings_report() -> dict:
    """Aggregate token savings of the certified distil strategy over EVERY turn of every
    cached SWE-agent trajectory (not just the eval cases). Model-free and free to run: the
    compression hillclimb's cost objective."""
    full = comp = served = turns = 0
    by_kind: dict[str, list[int]] = {}
    for p in sorted(SWE_DIR.glob("*.traj")):
        for t, _, req in _swe_turns(p):
            turns += 1
            served += sum(TOK.count(b.text) for b in serve(req)[0])
            c = distil_strategy(t.blocks, t.index)
            for b in t.blocks:
                by_kind.setdefault(b.kind.value, [0, 0])[0] += TOK.count(b.text)
            for b in c:
                by_kind.setdefault(b.kind.value, [0, 0])[1] += TOK.count(b.text)
            full += sum(TOK.count(b.text) for b in t.blocks)
            comp += sum(TOK.count(b.text) for b in c)
    return {
        "turns": turns,
        "full_tokens": full,
        "compressed_tokens": comp,
        "savings": round(1 - comp / full, 4),
        "served_tokens": served,
        "served_savings": round(1 - served / full, 4),
        "by_kind": {
            k: {
                "full": v[0],
                "compressed": v[1],
                "savings": round(1 - v[1] / v[0], 4) if v[0] else 0,
            }
            for k, v in sorted(by_kind.items())
        },
    }


def load_coding_cases() -> list[dict]:
    out = []
    for c in json.loads(CODING_CASES.read_text())["cases"]:
        p = SWE_DIR / c["file"]
        if not p.exists():
            sys.exit(f"{p} missing -- run with --fetch-swe-agent first")
        turn, gold, req = next((t, g, r) for t, g, r in _swe_turns(p) if t.index == c["turn"])
        repo = p.stem.split("__")[0]
        steps = turn.index // 2  # history alternates agent/observation after the task message
        out.append(
            {
                "id": f"{p.stem}-t{c['turn']}",
                "turn": turn,
                "gold": gold,
                "req": req,
                "tags": [repo, "swe-agent", f"step{steps}"],
                "title": f"SWE-bench Lite {p.stem} (SWE-agent GPT-4o) step {steps}",
            }
        )
    return out


CASE_SOURCE = "synthetic"


def cases_for(source: str) -> list[dict]:
    if source == "coding":
        return load_coding_cases()
    return load_real_cases() if source == "real" else load_cases()


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


def direct_client():
    """The first-party API, never an inherited ANTHROPIC_BASE_URL: in a distil-wrapped
    shell that points at the local distil proxy, which would compress the eval's own
    requests in flight (and relay its upstream failures as 502s)."""
    import anthropic

    base = __import__("os").environ.get("ANTHROPIC_BASE_URL")
    if base and "api.anthropic.com" not in base:
        print(
            f"note: ignoring ANTHROPIC_BASE_URL={base}; the eval calls https://api.anthropic.com directly",
            file=sys.stderr,
        )
    return anthropic.Anthropic(
        base_url="https://api.anthropic.com", max_retries=6
    )  # SDK: jittered backoff


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


ARMS = ("full_a", "full_b", "distil", "expand_structured", "trunc", "served", "served_expand")


def run_case(case: dict, client, model: str, effort: str | None = None, arms_on=ARMS) -> dict:
    blocks = case["turn"].blocks
    idx = case["turn"].index
    rec = Recorder(client, model)
    base = AnthropicRunner(model=model, client=rec, effort=effort)
    arms: dict[str, str] = {}
    arm_calls: dict[str, int] = {}

    def arm(name, fn):
        if name not in arms_on:
            return
        n0 = len(rec.calls)
        arms[name] = fn()
        arm_calls[name] = len(rec.calls) - n0

    comp = distil_strategy(blocks, idx)
    arm("full_a", lambda: base.decide(blocks))
    arm("full_b", lambda: base.decide(blocks))
    arm("distil", lambda: base.decide(comp))
    # ExpandAwareRunner commits through AnthropicRunner's decision tool
    # (structured_decision), so this arm and full_a share one decision format.
    arm("expand_structured", lambda: ExpandAwareRunner(base).decide(comp, build_restore(blocks)))
    arm("trunc", lambda: base.decide(TRUNC(blocks, idx)))
    served_savings = None
    if case.get("req") and ({"served", "served_expand"} & set(arms_on)):
        # distil's real serving adapter on the request a caching agent client would send
        sblocks, srestore = serve(case["req"])
        arm("served", lambda: base.decide(sblocks))
        arm("served_expand", lambda: ExpandAwareRunner(base).decide(sblocks, srestore))
        full_t = sum(TOK.count(b.text) for b in blocks)
        served_savings = (
            round(1 - sum(TOK.count(b.text) for b in sblocks) / full_t, 4) if full_t else 0.0
        )

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
        "gold": case.get("gold"),
        "served_savings": served_savings,
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
    g = {}
    if "served" in a:
        g["equiv_served_act"] = eq(a["full_a"], a["served"], _act)
    if "served_expand" in a:
        g["equiv_served_expand_act"] = eq(a["full_a"], a["served_expand"], _act)
    # Real traces only: the action the recorded agent actually took. Diagnostic,
    # not ground truth (another model wrote it) - never a gate.
    if run.get("gold"):
        g["agree_gold_act"] = eq(a["full_a"], run["gold"], _act)
    # Recovery-loop equivalence only where the arm ran in the structured format.
    # Rows from before that fix (free-text expand) carry no expand grade rather
    # than a number measured in a different format.
    if "expand_structured" in a:
        g["equiv_expand"] = eq(a["full_a"], a["expand_structured"])
        g["equiv_expand_act"] = eq(a["full_a"], a["expand_structured"], _act)
    return (
        g
        | {
            "equiv_distil": eq(a["full_a"], a["distil"]),
            "equiv_distil_act": eq(a["full_a"], a["distil"], _act),
            "self_consist": eq(a["full_a"], a["full_b"]),
            "self_consist_act": eq(a["full_a"], a["full_b"], _act),
            "decided": 1 if a["full_a"] != NO else 0,
            "agree_ref": eq(a["full_a"], ref),
            "agree_ref_act": eq(a["full_a"], ref, _act),
        }
        | (
            {
                "trunc_detect": 1
                if (a["full_a"] != NO and _act(a["trunc"]) != _act(a["full_a"]))
                else 0
            }
            if "trunc" in a
            else {}
        )
    )


def backfill_arm(vdir: Path, client, concurrency: int, arm: str, max_rep: int) -> None:
    """Add *arm* to rows recorded before it existed, with the variant's own
    model/effort; the arm's usage and calls are added to the row so cost stays
    complete. Rows are checkpointed (atomic rewrite) as they land and a failed
    row is logged to errors.jsonl and skipped, so a transient outage never
    loses completed calls; re-running picks up only the rows still missing."""
    if arm != "expand_structured":
        raise SystemExit(f"no backfill recipe for arm {arm}")
    cfg = json.loads((vdir / "config.json").read_text())
    cases = {c["id"]: c for c in cases_for(CASE_SOURCE)}
    rp = vdir / "results.jsonl"
    rows = [json.loads(ln) for ln in rp.read_text().splitlines() if ln.strip()]
    todo = [r for r in rows if arm not in r["meta"]["arms"] and r["rep"] < max_rep]
    print(
        f"[{vdir.name}] backfilling {arm} on {len(todo)} rows ({cfg['model']}, effort={cfg.get('effort')})",
        file=sys.stderr,
    )

    def one(r):
        rec = Recorder(client, cfg["model"])
        base = AnthropicRunner(model=cfg["model"], client=rec, effort=cfg.get("effort"))
        t = cases[r["prompt_id"]]["turn"]
        try:
            fp = ExpandAwareRunner(base).decide(
                distil_strategy(t.blocks, t.index), build_restore(t.blocks)
            )
        except BaseException as e:  # noqa: BLE001 - SystemExit from AnthropicRunner._create included
            if isinstance(e, KeyboardInterrupt):
                raise
            return r, None, rec.calls, e
        return r, fp, rec.calls, None

    def checkpoint():
        tmp = rp.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(x) + "\n" for x in rows))
        tmp.replace(rp)

    ok = fail = 0
    with cf.ThreadPoolExecutor(concurrency) as ex:
        for f in cf.as_completed([ex.submit(one, r) for r in todo]):
            r, fp, calls, err = f.result()
            if err is not None:
                fail += 1
                with (vdir / "errors.jsonl").open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "prompt_id": r["prompt_id"],
                                "rep": r["rep"],
                                "stage": f"backfill:{arm}",
                                "failure_class": getattr(
                                    err, "failure_class", "harness_or_api_error"
                                ),
                                "error": str(err)[:500],
                                "usage": {
                                    k: sum(c["usage"][k] for c in calls) for k in calls[0]["usage"]
                                }
                                if calls
                                else {},
                            }
                        )
                        + "\n"
                    )
                continue
            r["meta"]["arms"][arm] = fp
            for c in calls:
                for k, v in c["usage"].items():
                    r["usage"][k] = r["usage"].get(k, 0) + v
            r["tool_calls"] += len(calls)
            r["latency_s"] = round(r["latency_s"] + sum(c["latency_s"] for c in calls), 3)
            ok += 1
            if ok % 20 == 0:
                checkpoint()
    checkpoint()
    print(
        f"[{vdir.name}] backfill {arm}: {ok} ok, {fail} failed (re-run to retry the failed rows)",
        file=sys.stderr,
    )


def regrade(flow: Path) -> None:
    """Recompute every stored row's grade from its recorded arm decisions - no API calls."""
    refp = flow / "baseline/ref/decisions.json"
    ref = json.loads(refp.read_text()) if refp.exists() else {}
    for v in sorted(
        p for p in flow.iterdir() if p.is_dir() and re.fullmatch(r"baseline|v\d+", p.name)
    ):
        rp = v / "results.jsonl"
        if not rp.exists():
            continue
        rows = [json.loads(ln) for ln in rp.read_text().splitlines() if ln.strip()]
        for r in rows:
            r["grade"] = grade(
                {"arms": r["meta"]["arms"], "gold": r["meta"].get("gold")},
                None if v.name == "baseline" else ref.get(r["prompt_id"]),
            )
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
    # Required for runs; the fetch/select/report utility modes need neither.
    ap.add_argument("--variant")
    ap.add_argument("--model")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument(
        "--arms", default=",".join(ARMS), help="comma-separated subset of " + ",".join(ARMS)
    )
    ap.add_argument("--fake", choices=["oracle", "null", "flip"])
    ap.add_argument(
        "--cases",
        choices=["synthetic", "real", "coding"],
        default="synthetic",
        help="synthetic: corpus/ + corpus_xl (plants DECISION: markers - offline-runner annotations "
        "that leak the answer to a live model); real: public τ-bench trajectories",
    )
    ap.add_argument(
        "--fetch-tau-bench", action="store_true", help="download + verify the pinned τ-bench files"
    )
    ap.add_argument("--select-real-cases", action="store_true", help="(re)write the real case list")
    ap.add_argument(
        "--fetch-swe-agent",
        action="store_true",
        help="download + verify the pinned SWE-agent trajs",
    )
    ap.add_argument(
        "--select-coding-cases", action="store_true", help="(re)write the coding case list"
    )
    ap.add_argument(
        "--savings-report",
        help="write the all-turns SWE-agent savings report to this path (free, offline)",
    )
    ap.add_argument(
        "--status", action="store_true", help="print the cross-variant status table and exit"
    )
    ap.add_argument(
        "--regrade",
        action="store_true",
        help="recompute grades from stored decisions (no API calls)",
    )
    ap.add_argument(
        "--backfill-arm", help="add this arm to existing rows of --variant (reps < --reps)"
    )
    ap.add_argument("--flow", type=Path, default=FLOW)
    ap.add_argument("--approve-harness", action="store_true")
    a = ap.parse_args()
    global CASE_SOURCE
    CASE_SOURCE = a.cases
    if a.fetch_tau_bench:
        fetch_tau_bench()
    if a.select_real_cases:
        select_real_cases()
    if a.fetch_swe_agent:
        fetch_swe_agent()
    if a.select_coding_cases:
        select_coding_cases()
    if a.savings_report:
        rep_ = savings_report()
        Path(a.savings_report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.savings_report).write_text(json.dumps(rep_, indent=1))
        print(json.dumps({k: v for k, v in rep_.items() if k != "by_kind"}))
    if (
        a.fetch_tau_bench
        or a.select_real_cases
        or a.fetch_swe_agent
        or a.select_coding_cases
        or a.savings_report
    ):
        return
    FLOW = a.flow
    if (a.status or a.regrade) and not a.backfill_arm:
        # cross-variant, no-API modes: no --variant/--model needed
        if a.regrade:
            regrade(FLOW)
        print(status_table(FLOW))
        return
    if not a.variant or not a.model:
        ap.error(
            "--variant and --model are required (except for the fetch/select/report/status/regrade modes)"
        )
    if not re.fullmatch(r"baseline|v[1-9]\d*", a.variant):
        sys.exit("--variant must be 'baseline' or 'v<N>'")
    FLOW = a.flow
    if a.backfill_arm:
        if not a.fake:
            check_harness(a.approve_harness)
        if a.fake:
            client = FakeClient(a.fake, a.model)
        else:
            client = direct_client()
        backfill_arm(FLOW / a.variant, client, a.concurrency, a.backfill_arm, a.reps)
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
    arms_on = tuple(x for x in a.arms.split(",") if x)
    if not {"full_a", "full_b", "distil"} <= set(arms_on) or not set(arms_on) <= set(ARMS):
        sys.exit(f"--arms must include full_a,full_b,distil and be a subset of {ARMS}")
    cfg = {
        "model": a.model,
        "effort": a.effort,
        "reps": a.reps,
        "fake": a.fake,
        "arms": list(arms_on),
        "cases": a.cases,
    }
    cfgp = vdir / "config.json"
    if cfgp.exists():
        old = json.loads(cfgp.read_text())
        keys = ("model", "effort", "fake", "cases")
        if {k: old.get(k, "synthetic" if k == "cases" else None) for k in keys} != {
            k: cfg[k] for k in keys
        }:
            sys.exit(f"{vdir} was run as {old}; use a new --variant for {cfg}")
    cfgp.write_text(json.dumps(cfg, indent=2))
    sp = vdir / "summary.json"
    if not sp.exists():
        sp.write_text(
            json.dumps(
                {"description": f"{a.model} @ effort={a.effort or 'default'}", "target": "code"}
            )
        )
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

    cases = cases_for(CASE_SOURCE)
    cases = cases[: a.limit] if a.limit else cases
    tasks = [(c, r) for c in cases for r in range(a.reps) if (c["id"], r) not in done]
    print(
        f"[{a.variant}] {len(tasks)} of {len(cases) * a.reps} (case,rep) to run on {a.model}",
        file=sys.stderr,
    )

    if a.fake:
        client = FakeClient(a.fake, a.model)
    else:
        client = direct_client()
    lock = threading.Lock()
    ok = fail = 0

    def one(c, rep):
        run = run_case(c, client, a.model, a.effort, arms_on)
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
                    "served_savings": run["served_savings"],
                    "grade": g,
                    "meta": {
                        "arms": run["arms"],
                        "forced_tool_choice": run["forced_tool_choice"],
                        "gold": run["gold"],
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
    for m in (
        "equiv_distil_act",
        "equiv_distil",
        "equiv_expand_act",
        "self_consist_act",
        "decided",
        "agree_ref_act",
        "agree_gold_act",
        "trunc_detect",
    ):
        have = [r for r in rows if m in r["grade"]]
        n = len(have)
        k = sum(r["grade"][m] for r in have)
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
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def cost_usd(model: str, u: dict) -> float:
    base = next(
        (m for m in PRICES if model == m or model.startswith(m + "-") or model.startswith(m + "@")),
        None,
    )
    if base is None:
        raise KeyError(f"no price for {model}; add it to PRICES")
    pin, pout = PRICES[base]
    return (
        u.get("input_tokens", 0) * pin
        + u.get("cache_creation_input_tokens", 0) * pin * 1.25
        + u.get("cache_read_input_tokens", 0) * pin * 0.1
        + u.get("output_tokens", 0) * pout
    ) / 1e6


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
    vs = sorted(
        [p for p in flow.iterdir() if p.is_dir() and re.fullmatch(r"baseline|v\d+", p.name)],
        key=lambda p: -1 if p.name == "baseline" else int(p.name[1:]),
    )
    per: dict[str, dict] = {}
    lines = [
        "| variant | config | n | distil equiv (act / exact) | expand equiv (act / exact) | self-consist (act / exact) "
        "| agree w/ ref (act) | trunc caught | Δ distil act-equiv vs base (test, 95% CI) | $/case | out tok/case "
        "| s/case | errors | spend |",
        "|" + "---|" * 14,
    ]
    for v in vs:
        rows = (
            [json.loads(ln) for ln in (v / "results.jsonl").read_text().splitlines() if ln.strip()]
            if (v / "results.jsonl").exists()
            else []
        )
        errs = (
            sum(1 for ln in (v / "errors.jsonl").read_text().splitlines() if ln.strip())
            if (v / "errors.jsonl").exists()
            else 0
        )
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
            have = [r for r in ok if m in r["grade"]]
            if not have:
                return "—"
            s = f"{100 * sum(r['grade'][m] for r in have) / len(have):.1f}%"
            return s if len(have) == len(ok) else f"{s} (n={len(have)})"

        if v.name == "baseline":
            delta = "—"
        else:
            m, hw, n = _paired_ci(per.get("baseline", {}), per[v.name])
            delta = f"{100 * m:+.1f} ± {100 * hw:.1f} pts (n={n})"
        lines.append(
            f"| {v.name} | {cfg.get('model', '?')} / {cfg.get('effort') or 'default'} | {len(ok)} "
            f"| {pct('equiv_distil_act')} / {pct('equiv_distil')} | {pct('equiv_expand_act')} / {pct('equiv_expand')} "
            f"| {pct('self_consist_act')} / {pct('self_consist')} | {pct('agree_ref_act')} | {pct('trunc_detect')} | {delta} "
            f"| ${sum(costs) / len(rows):.4f} | {sum(r['usage'].get('output_tokens', 0) for r in rows) / len(rows):.0f} "
            f"| {sum(r['latency_s'] for r in rows) / len(rows):.1f} | {errs} | ${sum(costs):.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

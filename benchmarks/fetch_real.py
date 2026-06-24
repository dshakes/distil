#!/usr/bin/env python3
"""fetch_real.py — get real τ-bench / SWE-bench data into the proof harness.

Three sources, because that is how the data actually exists in the wild:

  tau       Convert τ-bench *result* logs (what `tau-bench` writes when you run it:
            a JSON list of episodes with a message list + a `reward`) into the
            episode shape `prove.py --dataset tau` reads. Run tau-bench yourself,
            point this at results/*.json.

  swe-traj  Validate/summarize a directory of SWE-agent `.traj` files (the harness
            adapter already reads these natively). Use this to sanity-check before a
            live grading run.

  swe-hf    Build **edit-localization** trajectories from real SWE-bench instances
            (needs `pip install datasets` + network). Each instance's gold patch gives
            the ground-truth file(s) to edit; the model must infer that target from the
            issue text amid distractor files. A legitimate decision-equivalence task
            derived from real issues+patches, no agent run required.

All outputs are validated with `distil.replay.realtrace` and reported with gold /
outcome coverage so you know the corpus is well-formed before you spend grading.

Examples
--------
  python benchmarks/fetch_real.py tau --src ~/tau-bench/results --out /data/tau.json
  python benchmarks/fetch_real.py swe-traj --src /data/swe_trajs/
  python benchmarks/fetch_real.py swe-hf --dataset princeton-nlp/SWE-bench_Lite \
      --split test --limit 200 --out /data/swe_loc.json
  # then:
  python benchmarks/prove.py --dataset tau --path /data/tau.json --runner claude-cli ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.replay import realtrace  # noqa: E402

# Real τ-bench trajectories live in the tau-bench GitHub repo (HuggingFace not
# required). 200 episodes each: message list + tool calls + reward. Reachable even
# where HF datasets-server is blocked.
TAU_SOURCES = {
    "gpt-4o-airline": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/gpt-4o-airline.json",
    "gpt-4o-retail": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/gpt-4o-retail.json",
    "sonnet-35-airline": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/sonnet-35-new-airline.json",
    "sonnet-35-retail": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/sonnet-35-new-retail.json",
}


def _resolve_src(src: str) -> str:
    """A shorthand (``tau:gpt-4o-airline``) or URL → a concrete location string."""
    if src.startswith("tau:"):
        key = src.split(":", 1)[1]
        if key not in TAU_SOURCES:
            raise SystemExit(f"unknown tau source {key!r}; choices: {', '.join(TAU_SOURCES)}")
        return TAU_SOURCES[key]
    return src


# --------------------------------------------------------------------------- #
# τ-bench result logs → episode shape
# --------------------------------------------------------------------------- #


def normalize_tau_episode(raw: dict, idx: int) -> dict | None:
    """Normalize one τ-bench result record into the harness episode shape.

    Accepts the common keys tau-bench emits: `traj`/`messages`/`trajectory` for the
    message list, `reward`/`success` for the outcome, `task_id`/`id` for the id,
    and an optional `tools`/`info.tools` schema. Returns None if it has no messages.
    """
    msgs = raw.get("messages") or raw.get("traj") or raw.get("trajectory")
    if not msgs:
        return None
    ep = {
        "id": str(raw.get("task_id") or raw.get("id") or raw.get("instance_id") or f"tau-{idx}"),
        "messages": msgs,
    }
    if "reward" in raw:
        ep["reward"] = raw["reward"]
    elif "success" in raw:
        ep["reward"] = 1 if raw["success"] else 0
    tools = raw.get("tools") or (raw.get("info") or {}).get("tools")
    if tools:
        ep["tools"] = tools
    return ep


def _iter_records(src: Path):
    files = sorted(src.glob("*.json")) + sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    for f in files:
        text = f.read_text()
        if f.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            doc = json.loads(text)
            if isinstance(doc, list):
                yield from doc
            elif isinstance(doc, dict) and "results" in doc:
                yield from doc["results"]
            else:
                yield doc


def _http_get(url: str) -> bytes:
    """Fetch a URL robustly. urllib can raise IncompleteRead on large bodies through
    some proxies, so retry, then fall back to curl."""
    import http.client
    import shutil
    import subprocess
    import tempfile

    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 (configured source)
                return r.read()
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            continue
    if shutil.which("curl"):
        with tempfile.NamedTemporaryFile(suffix=".json") as tf:
            subprocess.run(["curl", "-sSL", "--max-time", "300", url, "-o", tf.name], check=True)
            return Path(tf.name).read_bytes()
    raise SystemExit(f"could not download {url} (urllib failed and curl unavailable)")


def _records_from(src: str):
    """Yield raw records from a local file/dir OR a remote URL / `tau:<name>` shorthand."""
    loc = _resolve_src(src)
    if loc.startswith(("http://", "https://")):
        doc = json.loads(_http_get(loc).decode())
        records = (
            doc if isinstance(doc, list) else doc.get("results") or doc.get("episodes") or [doc]
        )
        yield from records
    else:
        yield from _iter_records(Path(loc))


def cmd_tau(args) -> int:
    episodes = []
    for i, rec in enumerate(_records_from(args.src)):
        ep = normalize_tau_episode(rec, i)
        if ep:
            episodes.append(ep)
    Path(args.out).write_text(json.dumps(episodes, indent=2))
    _report(realtrace.load_tau_bench(args.out), args.out)
    return 0


# --------------------------------------------------------------------------- #
# SWE-agent .traj directory → validate / summarize
# --------------------------------------------------------------------------- #


def cmd_swe_traj(args) -> int:
    entries = realtrace.load_swe_bench(args.src)
    print(f"loaded {len(entries)} SWE-agent trajectories from {args.src}")
    _report(entries, args.src)
    print("\nready: python benchmarks/prove.py --dataset swe --path", args.src, "--runner ...")
    return 0


# --------------------------------------------------------------------------- #
# SWE-bench instances (HuggingFace) → edit-localization trajectories
# --------------------------------------------------------------------------- #

_DIFF_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def parse_patch_targets(patch: str) -> list[str]:
    """Ground-truth files an instance's gold patch edits (the decision target)."""
    out = []
    for m in _DIFF_FILE.finditer(patch or ""):
        f = m.group(1).strip()
        if f and f != "/dev/null" and f not in out:
            out.append(f)
    return out


def _patch_hunk(patch: str, target: str, max_lines: int = 40) -> str:
    """A short slice of the gold patch for `target` — the load-bearing 'code search
    hit' the model should read. Truncation that drops it flips the localization."""
    lines = (patch or "").splitlines()
    keep, capturing = [], False
    for ln in lines:
        if ln.startswith("+++ b/"):
            capturing = ln.endswith(target)
        if capturing:
            keep.append(ln)
            if len(keep) >= max_lines:
                break
    return "\n".join(keep)


def build_swe_localization_episode(
    inst: dict, distractors: list[str], *, gold_seed: int | None = None
) -> dict | None:
    """One edit-localization trajectory from a SWE-bench instance.

    Context = issue text (stable) + a 'code search results' observation listing
    candidate files (the gold target's real hunk buried among distractor entries).
    Decision = {action: "edit", target: <gold file>}. The needle is the target's
    hunk; the model must map the issue to the right file. No directive.

    Position confound: by default the gold hit is appended **last** in the search
    results, which hands tail-truncation / recency baselines the needle for free
    (they keep the tail, so they keep the answer). Pass ``gold_seed`` to instead
    place the gold hit at a deterministic random position among the hits — seeded
    per-instance (``gold_seed`` × instance-id) so the corpus is reproducible and the
    placement is independent of which subsample is later drawn. The distractor set,
    the gold hunk text, and every other field are byte-identical to the unshuffled
    build; only the gold hit's index within the results changes.
    """
    patch = inst.get("patch") or inst.get("gold_patch") or ""
    targets = parse_patch_targets(patch)
    if not targets:
        return None
    target = targets[0]
    iid = str(inst.get("instance_id") or inst.get("id"))
    problem = inst.get("problem_statement") or inst.get("issue") or ""

    hits = [f"FILE {d}\n  (no obvious relation to the issue)" for d in distractors[:6]]
    gold_hit = f"FILE {target}\n{_patch_hunk(patch, target)}"
    if gold_seed is None:
        hits.append(gold_hit)  # original construction: gold last
    else:
        rng = random.Random(f"{gold_seed}:{iid}")
        hits.insert(rng.randint(0, len(hits)), gold_hit)
    obs = "code_search(issue_keywords) ->\n" + "\n\n".join(hits)

    return {
        "instance_id": iid,
        "problem_statement": problem,
        "system": (
            "You are a software engineer. Given a bug report and code-search results, "
            "decide the single file to edit. Respond with the edit action and the file path."
        ),
        "trajectory": [
            {"action": "search", "observation": "(issue received; searching repo)"},
            {"action": f"edit {target}", "observation": obs},
        ],
        "info": {"resolved": True},  # the gold patch resolves it by construction
    }


def cmd_swe_hf(args) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("need `pip install datasets` (and network) for swe-hf", file=sys.stderr)
        return 2
    ds = load_dataset(args.dataset, split=args.split)
    rows = list(ds)[: args.limit] if args.limit else list(ds)
    all_files = [t for r in rows for t in parse_patch_targets(r.get("patch", ""))]
    gold_seed = getattr(args, "shuffle_gold_seed", None)
    trajs = []
    for i, r in enumerate(rows):
        distractors = [f for f in all_files if f not in parse_patch_targets(r.get("patch", ""))]
        # rotate distractors so each instance gets a different plausible set
        ep = build_swe_localization_episode(
            r, distractors[i : i + 6] or distractors[:6], gold_seed=gold_seed
        )
        if ep:
            trajs.append(ep)
    if gold_seed is not None:
        print(f"gold-hunk position SHUFFLED (seed={gold_seed}) — recency/tail advantage removed")
    Path(args.out).write_text(json.dumps(trajs, indent=2))
    _report(realtrace.load_swe_bench(args.out), args.out)
    return 0


# --------------------------------------------------------------------------- #


def _report(entries, where) -> None:
    probs = realtrace.validate_real(entries)
    gold = realtrace.gold_actions(entries)
    n_turns = sum(len(e.trajectory.turns) for e in entries)
    labeled = sum(1 for e in entries if realtrace.success_label(e) is not None)
    print(f"\n→ {where}")
    print(f"  {len(entries)} trajectories · {n_turns} decision points")
    print(f"  gold actions: {len(gold)}   outcome-labeled trajectories: {labeled}")
    if probs:
        print(f"  STRUCTURAL PROBLEMS ({len(probs)}): " + "; ".join(probs[:5]))
    else:
        print("  structural check: clean ✔")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tau", help="download/convert tau-bench trajectories")
    p.add_argument(
        "--src",
        required=True,
        help="local file/dir, a URL, or a shorthand: " + ", ".join(f"tau:{k}" for k in TAU_SOURCES),
    )
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_tau)

    p = sub.add_parser("swe-traj", help="validate a SWE-agent .traj directory")
    p.add_argument("--src", required=True)
    p.set_defaults(fn=cmd_swe_traj)

    p = sub.add_parser("swe-hf", help="build edit-localization trajectories from SWE-bench")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--shuffle-gold-seed",
        type=int,
        default=None,
        help="place the gold hunk at a deterministic random position within the "
        "search results (seeded per-instance) instead of last — removes the "
        "recency/tail-truncation advantage. Omit for the original gold-last build.",
    )
    p.set_defaults(fn=cmd_swe_hf)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

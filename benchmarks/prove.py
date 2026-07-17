#!/usr/bin/env python3
"""prove.py — turn decision-equivalence from a slogan into a result.

Runs the credibility experiments from ``docs/PAPER_PLAN.md`` on **real agent
traces** (τ-bench / SWE-bench), graded by a **real model** — i.e. with the
circular ``DECISION:``-marker oracle removed:

  E1  Frontier .......... savings vs. decision-change rate, per ladder level.
  E2  Certification ..... THE proof. Certify at α on a calibration split, then
      coverage            measure the *realized* decision-change rate on a disjoint
                          held-out split, repeated over many random splits. Report
                          empirical P(realized risk ≤ α) — the certificate is sound
                          iff this is ≥ 1−δ. Splits are trajectory-level (no leakage).
  E3  Distribution ...... leave-one-domain-out: calibrate on all domains but one,
      shift               test on the held-out domain. The exchangeability stress
                          test reviewers will demand.

Decisions are cached on disk (per rendered-context hash, runner-namespaced) so the
expensive live-model pass is paid exactly once and the stats are reproducible.

USAGE
  # offline plumbing check (NON-EVIDENTIAL smoke runner, bundled fixtures):
  python benchmarks/prove.py --dataset fixtures --runner smoke

  # the real thing (needs ANTHROPIC_API_KEY + downloaded traces):
  python benchmarks/prove.py --dataset tau --path /data/taubench_runs.json --runner anthropic
  python benchmarks/prove.py --dataset swe --path /data/swe_trajs/ --runner anthropic --samples 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil.conformal import crc_select, default_ladder, hb_pvalue, ltt_certify  # noqa: E402
from distil.replay import realtrace  # noqa: E402
from distil.tokenizer import DEFAULT as tok  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


# --------------------------------------------------------------------------- #
# Decision cache — pay the live model once
# --------------------------------------------------------------------------- #


class DecisionCache:
    def __init__(self, runner, namespace: str):
        self.runner = runner
        self.path = CACHE_DIR / f"decisions_{namespace}.json"
        self.store: dict[str, str] = {}
        if self.path.exists():
            self.store = json.loads(self.path.read_text())
        self.hits = self.misses = 0

    @staticmethod
    def _key(blocks) -> str:
        h = hashlib.sha1()
        for b in blocks:
            h.update(f"{b.kind.value}|{b.stability.value}|{b.text}\x00".encode())
        return h.hexdigest()

    def _compose_key(self, blocks, restore, variant: str = "") -> str:
        k = self._key(blocks)
        if restore:  # expand-aware decisions depend on what's recoverable
            rk = hashlib.sha1("".join(sorted(restore)).encode()).hexdigest()[:8]
            k = f"{k}:{rk}"
        if variant:  # a distinct cache slot forcing an INDEPENDENT second draw of the
            k = f"{k}:v={variant}"  # SAME context — the A/A self-agreement control.
        return k

    def _call(self, blocks, restore) -> str:
        return (
            self.runner.decide(blocks, restore)
            if restore is not None
            else self.runner.decide(blocks)
        )

    def decide(self, blocks, restore=None, variant: str = "") -> str:
        k = self._compose_key(blocks, restore, variant)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        self.misses += 1
        self.store[k] = d = self._call(blocks, restore)
        if self.misses % 20 == 0:  # checkpoint long live runs so nothing is lost
            self.flush()
        return d

    def prefetch(self, requests, workers: int = 1) -> None:
        """Compute all uncached decisions up front, optionally concurrently. Each
        unique (blocks, restore) is graded once; results fill the cache so the matrix
        assembly is pure cache hits. Concurrency is the difference between a real
        full-corpus API run taking minutes vs. hours (runner calls run in threads;
        results are consumed serially, so the cache/flush stay single-threaded)."""
        todo: dict[str, tuple] = {}
        for req in requests:
            blocks, restore = req[0], req[1]
            variant = req[2] if len(req) > 2 else ""
            k = self._compose_key(blocks, restore, variant)
            if k not in self.store and k not in todo:
                todo[k] = (blocks, restore)
        if not todo:
            return
        items = list(todo.items())
        if workers <= 1:
            results = ((k, self._call(b, r)) for k, (b, r) in items)
        else:
            from concurrent.futures import ThreadPoolExecutor

            def work(item):
                k, (b, r) = item
                return k, self._call(b, r)

            ex = ThreadPoolExecutor(max_workers=workers)
            results = ex.map(work, items)
        done = 0
        for k, d in results:
            self.store[k] = d
            self.misses += 1
            done += 1
            if done % 20 == 0:
                self.flush()
        self.flush()

    def flush(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.store))


# --------------------------------------------------------------------------- #
# Build the per-trajectory decision/loss matrix (the one expensive pass)
# --------------------------------------------------------------------------- #


def build_matrix(
    entries,
    cache: DecisionCache,
    ladder,
    gold,
    *,
    expand: bool = False,
    baselines=None,
    workers: int = 1,
    aa_control: bool = False,
) -> dict:
    from distil.replay.expand_runner import build_restore

    # baselines are graded under the SAME runner but never with --expand (they are
    # lossy/irrecoverable — recovery would not apply), for a fair head-to-head.
    baselines = baselines or []

    # Phase 1 — plan: compute every compressed context once, gather decision requests.
    plan = []  # (tid, domain, success, [(turn_index, base_blocks, restore, gold_fp, comps)])
    requests = []
    for e in entries:
        tid = e.trajectory.id
        turns = []
        for turn in e.trajectory.turns:
            restore = build_restore(turn.blocks) if expand else None
            comps = {("level", n): strat(turn.blocks, turn.index) for n, strat in ladder}
            comps.update(
                {("baseline", n): strat(turn.blocks, turn.index) for n, strat in baselines}
            )
            requests.append((turn.blocks, restore))
            if aa_control:  # an independent second draw of the SAME base context
                requests.append((turn.blocks, restore, "aa"))
            for c in comps.values():
                requests.append((c, restore))
            g = gold.get((tid, turn.index))
            turns.append((turn.index, turn.blocks, restore, g.fingerprint if g else None, comps))
        plan.append((tid, e.domain, realtrace.success_label(e), turns))

    # Phase 2 — grade all unique decisions (optionally concurrently).
    cache.prefetch(requests, workers=workers)

    # Phase 3 — assemble from the (now warm) cache; no new model calls.
    matrix: dict[str, dict] = {}
    for tid, domain, success, turns in plan:
        rec = {"domain": domain, "success": success, "turns": []}
        for _idx, base_blocks, restore, gold_fp, comps in turns:
            base = cache.decide(base_blocks, restore)
            base_text = "\n".join(b.text for b in base_blocks)
            tr = {
                "base": base,
                "base_tok": sum(tok.count(b.text) for b in base_blocks),
                "gold": gold_fp,
                "levels": {},
                "baselines": {},
            }
            if aa_control:
                # A/A self-agreement: the model re-graded on the IDENTICAL base
                # context. A live grader disagrees with itself on ambiguous turns;
                # that floor is the noise every level's decision_change also carries
                # and cannot be attributed to compression.
                base_aa = cache.decide(base_blocks, restore, variant="aa")
                tr["aa_loss"] = 0.0 if base_aa == base else 1.0
            for (kind, name), comp in comps.items():
                dec = cache.decide(comp, restore)
                bucket = "levels" if kind == "level" else "baselines"
                tr[bucket][name] = {
                    "loss": 0.0 if dec == base else 1.0,
                    "comp_tok": sum(tok.count(b.text) for b in comp),
                    # did this level actually alter the text on this turn? a turn it
                    # leaves byte-identical is trivially loss=0 and dilutes the rate.
                    "changed": "\n".join(b.text for b in comp) != base_text,
                }
            rec["turns"].append(tr)
        matrix[tid] = rec
    return matrix


# --------------------------------------------------------------------------- #
# E1 — frontier
# --------------------------------------------------------------------------- #


def e1_frontier(matrix, ladder) -> list[dict]:
    # A/A self-agreement floor (present only when build_matrix ran with aa_control):
    # the mean rate at which the grader disagrees with ITSELF on the identical base
    # context. Every level's raw decision_change carries this same grader noise, so
    # the honest per-turn signal attributable to compression is the excess above it.
    aa = [tr["aa_loss"] for rec in matrix.values() for tr in rec["turns"] if "aa_loss" in tr]
    aa_floor = (sum(aa) / len(aa)) if aa else None

    rows = []
    for name, _ in ladder:
        losses, base_t, comp_t = [], 0, 0
        eff_losses = []  # losses only on turns this level actually changed
        for rec in matrix.values():
            for tr in rec["turns"]:
                lv = tr["levels"][name]
                losses.append(lv["loss"])
                if lv.get("changed"):
                    eff_losses.append(lv["loss"])
                base_t += tr["base_tok"]
                comp_t += tr["levels"][name]["comp_tok"]
        n, m = len(losses), len(eff_losses)
        dc = (sum(losses) / n) if n else 0.0
        row = {
            "level": name,
            "n": n,
            "decision_change": dc,
            # honest denominator: rate over turns the level actually compressed,
            # and the fraction of turns it left byte-identical (trivially safe).
            "effective_n": m,
            "decision_change_effective": (sum(eff_losses) / m) if m else 0.0,
            "trivial_frac": ((n - m) / n) if n else 0.0,
            "savings": (1.0 - comp_t / base_t) if base_t else 0.0,
        }
        if aa_floor is not None:
            row["aa_floor"] = aa_floor
            # excess divergence beyond the model's own self-disagreement (clamped
            # at 0 — a level below the floor is indistinguishable from grader noise).
            row["decision_change_vs_floor"] = max(0.0, dc - aa_floor)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Head-to-head — distil ladder vs competitor/structural baselines, same grader
# --------------------------------------------------------------------------- #


def head_to_head(matrix, ladder, *, alpha, delta) -> list[dict]:
    """One row per method (distil ladder levels + baselines), all graded identically:
    token savings, decision-change rate, and whether its risk certifies ≤ α at
    confidence 1−δ on the full data (single-shot Hoeffding–Bentkus). This is the
    comparison table reviewers expect — distil's certified levels next to LLMLingua-2,
    RECOMP-extractive, truncation, etc."""
    methods: list[tuple[str, str]] = [(n, "distil") for n, _ in ladder]
    seen = {n for n, _ in ladder}
    for rec in matrix.values():
        for n in rec["turns"][0]["baselines"] if rec["turns"] else []:
            if n not in seen:
                methods.append((n, "baseline"))
                seen.add(n)

    rows = []
    for name, kind in methods:
        losses, base_t, comp_t = [], 0, 0
        for rec in matrix.values():
            for tr in rec["turns"]:
                cell = tr["levels"].get(name) or tr["baselines"].get(name)
                if cell is None:
                    continue
                losses.append(cell["loss"])
                base_t += tr["base_tok"]
                comp_t += cell["comp_tok"]
        n = len(losses)
        rhat = (sum(losses) / n) if n else 1.0
        rows.append(
            {
                "method": name,
                "kind": kind,
                "savings": (1.0 - comp_t / base_t) if base_t else 0.0,
                "decision_change": rhat,
                "certifies": hb_pvalue(rhat, n, alpha) <= delta,
            }
        )
    rows.sort(key=lambda r: -r["savings"])
    return rows


# --------------------------------------------------------------------------- #
# E2 — certification coverage (the proof)
# --------------------------------------------------------------------------- #


def _losses_for(matrix, traj_ids, ladder):
    """level-ordered list of 0/1 loss lists, plus base/comp tokens, over traj_ids."""
    level_losses = [[] for _ in ladder]
    base_t = 0
    comp_t = [0] * len(ladder)
    for tid in traj_ids:
        for tr in matrix[tid]["turns"]:
            base_t += tr["base_tok"]
            for i, (name, _) in enumerate(ladder):
                lv = tr["levels"][name]
                level_losses[i].append(lv["loss"])
                comp_t[i] += lv["comp_tok"]
    return level_losses, base_t, comp_t


def _operating_point(level_losses, base_t, comp_t, *, alpha, delta, method):
    """Pick the certified prefix end, then the highest-SAVINGS level within it
    (matches distil.conformal.calibrate)."""
    if method == "crc":
        idx_end = crc_select(level_losses, alpha=alpha)
    else:
        idx_end, _ = ltt_certify(level_losses, alpha=alpha, delta=delta)
    if idx_end < 0:
        return -1

    def savings(i):
        return (1.0 - comp_t[i] / base_t) if base_t else 0.0

    return max(range(idx_end + 1), key=savings)


def e2_coverage(matrix, ladder, *, alpha, delta, method, reps, seed) -> dict:
    tids = list(matrix.keys())
    rng = random.Random(seed)
    hits = realized = savings = certified_any = 0
    detail = []
    for _rep in range(reps):
        order = tids[:]
        rng.shuffle(order)
        half = max(1, len(order) // 2)
        calib, test = order[:half], order[half:] or order[:half]

        cl, cb, cc = _losses_for(matrix, calib, ladder)
        idx = _operating_point(cl, cb, cc, alpha=alpha, delta=delta, method=method)
        if idx < 0:
            detail.append({"certified": None})
            continue
        certified_any += 1
        # realized risk + savings on the DISJOINT held-out test split, at the certified level
        tl, tb, tc = _losses_for(matrix, test, ladder)
        n_test = len(tl[idx])
        realized_risk = (sum(tl[idx]) / n_test) if n_test else 0.0
        test_savings = (1.0 - tc[idx] / tb) if tb else 0.0
        hit = realized_risk <= alpha
        hits += int(hit)
        realized += realized_risk
        savings += test_savings
        detail.append(
            {
                "certified": ladder[idx][0],
                "realized_risk": realized_risk,
                "savings": test_savings,
                "hit": hit,
            }
        )

    c = certified_any or 1
    return {
        "alpha": alpha,
        "delta": delta,
        "method": method,
        "reps": reps,
        "certified_frac": certified_any / reps,
        "empirical_coverage": hits / c,  # P(realized ≤ α | certified)
        "target_coverage": 1 - delta if method == "ltt" else None,
        "mean_realized_risk": realized / c,
        "mean_test_savings": savings / c,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# E3 — distribution shift (leave-one-domain-out)
# --------------------------------------------------------------------------- #


def e3_shift(matrix, ladder, *, alpha, delta, method) -> list[dict]:
    by_domain: dict[str, list[str]] = {}
    for tid, rec in matrix.items():
        by_domain.setdefault(rec["domain"], []).append(tid)
    if len(by_domain) < 2:
        return []
    out = []
    for held in by_domain:
        calib = [t for d, ts in by_domain.items() if d != held for t in ts]
        test = by_domain[held]
        cl, cb, cc = _losses_for(matrix, calib, ladder)
        idx = _operating_point(cl, cb, cc, alpha=alpha, delta=delta, method=method)
        if idx < 0:
            out.append({"held_out_domain": held, "certified": None})
            continue
        tl, tb, tc = _losses_for(matrix, test, ladder)
        n = len(tl[idx])
        out.append(
            {
                "held_out_domain": held,
                "certified": ladder[idx][0],
                "realized_risk": (sum(tl[idx]) / n) if n else 0.0,
                "savings": (1.0 - tc[idx] / tb) if tb else 0.0,
                "held_within_alpha": ((sum(tl[idx]) / n) if n else 0.0) <= alpha,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# E4 — downstream task success (does compression preserve the OUTCOME?)
# --------------------------------------------------------------------------- #


def e4_task_success(matrix, ladder, *, seed=0, boot=1000) -> dict | None:
    """Convert per-turn decision-equivalence into a downstream task metric.

    A trajectory keeps its outcome under a compression level iff EVERY decision is
    unchanged (one flip ⇒ the agent diverges and the resolution is no longer
    guaranteed — a conservative, lower-bound reading). So:

      retained success-rate(level) = (# trajectories that originally succeeded AND
                                      stay fully decision-equivalent) / N

    Reported against the uncompressed baseline success-rate, per level, with a
    bootstrap CI over trajectories. Needs trajectories that carry an outcome label
    (τ-bench reward / SWE-bench resolved); returns None if none do.
    """
    labeled = [(tid, rec) for tid, rec in matrix.items() if rec.get("success") is not None]
    if not labeled:
        return None
    n = len(labeled)
    base_success = sum(1 for _, r in labeled if r["success"]) / n
    # If every trajectory shares the same outcome label there is no real outcome
    # variance to measure — e.g. the swe-hf localization builder marks every instance
    # resolved=True by construction. Flag it so E4 is not read as a measured success rate.
    outcome_evidential = len({bool(r["success"]) for _, r in labeled}) > 1

    rng = random.Random(seed)
    rows = []
    for name, _ in ladder:

        def preserved(rec, lvl=name):  # all turns unchanged at this level
            return all(tr["levels"][lvl]["loss"] == 0.0 for tr in rec["turns"])

        retained = [1.0 if (r["success"] and preserved(r)) else 0.0 for _, r in labeled]
        base_t = sum(tr["base_tok"] for _, r in labeled for tr in r["turns"])
        comp_t = sum(tr["levels"][name]["comp_tok"] for _, r in labeled for tr in r["turns"])
        # bootstrap CI over trajectories
        means = []
        for _ in range(boot):
            sample = [retained[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        rows.append(
            {
                "level": name,
                "savings": (1.0 - comp_t / base_t) if base_t else 0.0,
                "retained_success": sum(retained) / n,
                "ci_low": means[int(0.025 * boot)],
                "ci_high": means[int(0.975 * boot)],
                "preserved_frac": sum(1 for _, r in labeled if preserved(r)) / n,
            }
        )
    return {
        "n": n,
        "baseline_success": base_success,
        "outcome_evidential": outcome_evidential,
        "levels": rows,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def gold_agreement(matrix) -> tuple[int, int]:
    matched = total = 0
    for rec in matrix.values():
        for tr in rec["turns"]:
            if tr["gold"] is not None:
                total += 1
                matched += int(tr["base"] == tr["gold"])
    return matched, total


# --------------------------------------------------------------------------- #
# Gate — CI exit-code check on E2 results
# --------------------------------------------------------------------------- #


def gate_check(cov: dict, alpha: float, delta: float) -> tuple[bool, str]:
    """Return (pass, human-readable verdict) for a CI gate over E2 coverage results.

    Gate rationale (three checks in strictness order):
    1. certified_frac > 0  — at least one calib/test split produced a certified level.
       Zero means the dataset is too small or distil's compression is fundamentally
       broken (no level passes the LTT/CRC bound). Either way the certificate is void.
    2. empirical_coverage ≥ 1−δ  — the fraction of splits where the realized
       decision-change rate stayed ≤ α. This is the LTT guarantee: P(realized ≤ α) ≥ 1−δ.
       Failure means the certificate doesn't hold out-of-sample — i.e. the compressor
       is changing more decisions on unseen data than the α budget.
    3. mean_realized_risk ≤ α  — the average realized risk across certified splits.
       A second-order check: even if *empirical_coverage* just barely passes, a large mean
       risk signals the bound is too tight for this data size. Conservative but cheap to add.

    Stability: with --samples 3 (majority vote) and disk caching, the E2 result is
    deterministic after the first live run — re-running prove.py replays from cache and
    produces identical coverage numbers. No A/A control is needed as a separate step;
    the LTT/CRC conformal framework absorbs stochasticity by design (its guarantee holds
    in expectation over the random data split, not per call).
    """
    if cov["certified_frac"] == 0.0:
        return (
            False,
            f"GATE FAIL — certified_frac=0.0: no split certified at α={alpha}; "
            "corpus too small or compression fundamentally broken",
        )
    target = 1.0 - delta
    if cov["empirical_coverage"] < target - 1e-9:
        return (
            False,
            f"GATE FAIL — empirical_coverage={cov['empirical_coverage']:.3f} < {target:.2f} "
            f"(1−δ); certificate does not hold out-of-sample",
        )
    if cov["mean_realized_risk"] > alpha + 1e-9:
        return (
            False,
            f"GATE FAIL — mean_realized_risk={cov['mean_realized_risk']:.3f} > α={alpha}; "
            "mean held-out risk exceeds budget",
        )
    return (
        True,
        f"GATE PASS — certified_frac={cov['certified_frac']:.1%}  "
        f"coverage={cov['empirical_coverage']:.1%}≥{target:.0%}  "
        f"realized_risk={cov['mean_realized_risk'] * 100:.2f}%≤{alpha * 100:.1f}%",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    # The report prints α/≤/✔/· — Windows cp1252 consoles crash on those
    # (UnicodeEncodeError killed the CI gate subprocess). Force UTF-8 with
    # replacement so the gate's exit code is always about the RESULT, never
    # the console encoding. hasattr guard: exotic/captured streams may not
    # expose reconfigure.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — cosmetics must never break the gate
                pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", choices=["tau", "swe", "fixtures"], default="fixtures")
    ap.add_argument("--path", help="trace file/dir (required for tau/swe)")
    ap.add_argument(
        "--runner",
        choices=["smoke", "anthropic", "openai", "claude-cli"],
        default="smoke",
        help="grader: smoke (offline plumbing), anthropic (API key), openai "
        "(local/vLLM/Ollama via --base-url), claude-cli (your Claude Code subscription)",
    )
    ap.add_argument("--samples", type=int, default=1, help="majority-of-k votes per decision")
    ap.add_argument(
        "--aa-control",
        action="store_true",
        help="measure the grader's A/A self-agreement floor (a second independent draw "
        "per base context) and report E1 decision-change as excess above it — the "
        "honest per-turn signal on a stochastic grader. Costs one extra draw per turn.",
    )
    ap.add_argument("--model", default="claude-opus-4-8", help="grader model id")
    ap.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="openai runner: OpenAI-compatible endpoint (vLLM/Ollama/LM Studio/OpenAI)",
    )
    ap.add_argument(
        "--api-key-env", default="OPENAI_API_KEY", help="openai runner: env var holding the key"
    )
    ap.add_argument(
        "--json-mode",
        action="store_true",
        help="openai runner: request response_format=json_object",
    )
    ap.add_argument(
        "--cli-bin", default="claude", help="claude-cli runner: path to the claude binary"
    )
    ap.add_argument(
        "--baselines",
        action="store_true",
        help="also grade competitor/structural baselines (LLMLingua-2 if installed, "
        "RECOMP-extractive, selective-context, truncation, recency-window) for a "
        "head-to-head table under the same grader",
    )
    ap.add_argument(
        "--ladder",
        choices=["full", "quick"],
        default="full",
        help="quick = 4 rungs (byte-exact, lossless, truncate@250, truncate@120) to cut paid calls",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="risk budget (decision-change rate)")
    ap.add_argument("--delta", type=float, default=0.05, help="LTT confidence 1-δ")
    ap.add_argument("--method", choices=["ltt", "crc"], default="ltt")
    ap.add_argument("--reps", type=int, default=200, help="random calib/test splits for E2")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent grader calls (use 8–16 for the API runner to finish a full "
        "real corpus in minutes; keep at 1 for smoke/local)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap #trajectories (live cost control)")
    ap.add_argument(
        "--expand",
        action="store_true",
        help="grade the reversible tier WITH the distil_expand recovery loop (the model "
        "may recover digested content before deciding) — the with-expand frontier, vs. the "
        "default conservative no-expand lower bound",
    )
    ap.add_argument("--report", help="write full JSON report here")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="CI gate mode: exit non-zero when E2 results violate thresholds "
        "(certified_frac>0, empirical_coverage≥1−δ, mean_realized_risk≤α). "
        "With disk caching the result is deterministic after the first live run.",
    )
    ap.add_argument(
        "--max-live-calls",
        type=int,
        default=None,
        metavar="N",
        help="hard ceiling on live API calls for the anthropic runner (the run fails loudly "
        "at N instead of spending silently); ignored for smoke/openai/claude-cli runners "
        "which don't count calls the same way",
    )
    args = ap.parse_args()

    # ---- load traces -------------------------------------------------------
    if args.dataset == "fixtures":
        fx = Path(__file__).resolve().parent / "fixtures"
        entries = realtrace.load_tau_bench(fx / "tau_bench_sample.json", model=args.model)
        entries += realtrace.load_swe_bench(fx / "swe_bench_sample.json", model=args.model)
    elif args.dataset == "tau":
        entries = realtrace.load_tau_bench(args.path, model=args.model)
    else:
        entries = realtrace.load_swe_bench(args.path, model=args.model)
    if args.limit and args.limit < len(entries):
        # stratify by outcome so a small sample isn't all-failures (E4 needs both),
        # then deterministically subsample the rest — avoids the first-N bias.
        labeled_ok = [e for e in entries if realtrace.success_label(e) is True]
        labeled_no = [e for e in entries if realtrace.success_label(e) is False]
        other = [e for e in entries if realtrace.success_label(e) is None]
        rng0 = random.Random(args.seed)
        for grp in (labeled_ok, labeled_no, other):
            rng0.shuffle(grp)
        picked, i = [], 0
        pools = [labeled_ok, labeled_no, other]
        while len(picked) < args.limit and any(pools):
            pool = pools[i % len(pools)]
            if pool:
                picked.append(pool.pop())
            i += 1
        entries = picked

    problems = realtrace.validate_real(entries)
    if problems:
        print("STRUCTURAL PROBLEMS:")
        for p in problems[:20]:
            print("  -", p)
    gold = realtrace.gold_actions(entries)
    n_turns = sum(len(e.trajectory.turns) for e in entries)
    print(
        f"loaded {len(entries)} trajectories · {n_turns} decision points · dataset={args.dataset}"
    )

    # ---- runner ------------------------------------------------------------
    if args.runner == "smoke":
        from distil.replay.smoke_runner import SmokeRunner

        runner = SmokeRunner()
        ns = "smoke"
        print(
            "\n" + "!" * 78 + "\n"
            "! SMOKE RUNNER — NON-EVIDENTIAL. This verifies the harness mechanics only.\n"
            "! It does NOT grade real agent behavior. For a publishable result run with a\n"
            "! real model: --runner claude-cli (your subscription) / openai (local) / anthropic.\n"
            + "!"
            * 78
        )
    elif args.runner == "anthropic":
        from distil.replay.anthropic_runner import AnthropicRunner

        runner = AnthropicRunner(
            model=args.model, samples=args.samples, max_calls=args.max_live_calls
        )
        ns = f"anthropic_{args.model}_s{args.samples}"
    elif args.runner == "openai":
        from distil.replay.openai_runner import OpenAIRunner

        runner = OpenAIRunner(
            args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            samples=args.samples,
            json_mode=args.json_mode,
        )
        ns = f"openai_{args.model.replace('/', '_')}_s{args.samples}"
    else:  # claude-cli
        from distil.replay.claude_cli_runner import ClaudeCliRunner

        cli_model = None if args.model == "claude-opus-4-8" else args.model
        runner = ClaudeCliRunner(bin=args.cli_bin, model=cli_model, samples=args.samples)
        ns = f"claudecli_{(cli_model or 'default').replace('/', '_')}_s{args.samples}"

    evidential = args.runner != "smoke"
    if evidential and args.samples == 1:
        print(
            "\nWARNING: --samples 1 with a stochastic grader inflates the decision-change rate:\n"
            "  any text change (e.g. the reversible digest) triggers a fresh sample, so grader\n"
            "  variance is counted as a 'flip'. Use --samples 3+ (majority vote) for a valid\n"
            "  measurement. Also grade traces with a model of the SAME family that produced them\n"
            "  (e.g. the sonnet-35 τ-bench logs with a Claude grader) for faithful gold-agreement.\n"
        )
    if args.expand:
        from distil.replay.expand_runner import ExpandAwareRunner

        if not hasattr(runner, "_raw"):
            raise SystemExit(f"--expand needs a runner with _raw(); '{args.runner}' has none")
        runner = ExpandAwareRunner(runner, samples=args.samples)
        ns += "+expand"
        print(
            "EXPAND-AWARE grading ON: the reversible tier is graded WITH the recovery loop\n"
            "  (with-expand frontier). Without --expand you get the conservative no-expand bound."
        )

    ladder = default_ladder()
    if args.ladder == "quick":
        keep = {"byte-exact", "lossless", "truncate@250", "truncate@120"}
        ladder = [rung for rung in ladder if rung[0] in keep]

    baselines = []
    if args.baselines:
        from benchmarks.baselines import load_baselines

        baselines = load_baselines()
        print(f"baselines: {', '.join(n for n, _ in baselines)}")

    cache = DecisionCache(runner, ns)
    matrix = build_matrix(
        entries,
        cache,
        ladder,
        gold,
        expand=args.expand,
        baselines=baselines,
        workers=args.workers,
        aa_control=args.aa_control,
    )
    cache.flush()
    print(f"decisions: {cache.hits} cached / {cache.misses} computed (workers={args.workers})")

    # ---- gold sanity (real runners: does the model reproduce real actions?) ----
    if evidential:
        m, t = gold_agreement(matrix)
        print(
            f"model↔gold next-action agreement (uncompressed): {m}/{t} = {(m / t if t else 0):.1%}"
        )
        print("  (low agreement ⇒ the grader isn't a faithful agent; fix before trusting E1/E2)")

    # ---- E1 ----------------------------------------------------------------
    print("\n=== E1 · FRONTIER (savings vs. decision-change, real grader) ===")
    print(f"{'level':<24}{'savings':>9}{'dec-change':>12}{'n':>7}{'on-changed':>12}{'trivial':>9}")
    print("-" * 73)
    f_rows = e1_frontier(matrix, ladder)
    for r in f_rows:
        line = (
            f"{r['level']:<24}{r['savings'] * 100:>8.1f}%{r['decision_change'] * 100:>11.1f}%{r['n']:>7}"
            f"{r['decision_change_effective'] * 100:>11.1f}%{r['trivial_frac'] * 100:>8.0f}%"
        )
        if "decision_change_vs_floor" in r:
            line += f"  (vs A/A floor: {r['decision_change_vs_floor'] * 100:.1f}%)"
        print(line)
    print(
        "  on-changed = decision-change over only the turns this level actually compressed\n"
        "  (effective n); trivial = % of turns left byte-identical (trivially safe, dilutes the rate)."
    )
    if f_rows and "aa_floor" in f_rows[0]:
        print(
            f"  A/A self-agreement floor: {f_rows[0]['aa_floor'] * 100:.1f}% — the grader's own "
            "run-to-run disagreement on the identical context. 'vs A/A floor' is the excess\n"
            "  decision-change attributable to compression (byte-exact reads ~0% here because "
            "it shares the base cache key — the floor is what actually quantifies grader noise)."
        )

    # ---- head-to-head vs baselines ----------------------------------------
    h2h = []
    if baselines:
        h2h = head_to_head(matrix, ladder, alpha=args.alpha, delta=args.delta)
        print(f"\n=== HEAD-TO-HEAD (same grader, α={args.alpha}, δ={args.delta}) ===")
        print(f"{'method':<22}{'kind':<10}{'savings':>9}{'dec-change':>12}{'certifies?':>12}")
        print("-" * 65)
        for r in h2h:
            print(
                f"{r['method']:<22}{r['kind']:<10}{r['savings'] * 100:>8.1f}%"
                f"{r['decision_change'] * 100:>11.1f}%{('✔ yes' if r['certifies'] else '✘ no'):>12}"
            )
        print(
            "  certifies? = decision-change rate provably ≤ α at 1−δ (Hoeffding–Bentkus, full data).\n"
            "  The honest contrast: aggressive baselines save more but flip decisions (don't certify);\n"
            "  distil's certified level is the most aggressive one that does."
        )

    # ---- E2 ----------------------------------------------------------------
    print("\n=== E2 · CERTIFICATION COVERAGE (the proof) ===")
    cov = e2_coverage(
        matrix,
        ladder,
        alpha=args.alpha,
        delta=args.delta,
        method=args.method,
        reps=args.reps,
        seed=args.seed,
    )
    tgt = cov["target_coverage"]
    print(
        f"method={args.method}  α={args.alpha}  δ={args.delta}  splits={args.reps} (trajectory-level, disjoint)"
    )
    print(f"  certified in {cov['certified_frac']:.1%} of splits")
    print(
        f"  empirical coverage  P(realized risk ≤ α) = {cov['empirical_coverage']:.1%}"
        + (f"   (target ≥ {tgt:.0%})" if tgt else "   (CRC: expected-risk control)")
    )
    print(
        f"  mean realized risk on held-out test       = {cov['mean_realized_risk'] * 100:.2f}%  (≤ α={args.alpha * 100:.1f}% ✔)"
        if cov["mean_realized_risk"] <= args.alpha
        else f"  mean realized risk on held-out test       = {cov['mean_realized_risk'] * 100:.2f}%  (> α — UNDERCOVERAGE)"
    )
    print(f"  mean certified token savings (held-out)   = {cov['mean_test_savings'] * 100:.1f}%")
    if tgt and cov["empirical_coverage"] + 1e-9 >= tgt:
        print("  VERDICT: certificate holds out-of-sample ✔  (this is the result the paper needs)")
    elif tgt:
        print("  VERDICT: UNDERCOVERAGE — investigate (too-small n, non-exchangeability, or a bug)")

    # ---- E3 ----------------------------------------------------------------
    e3 = e3_shift(matrix, ladder, alpha=args.alpha, delta=args.delta, method=args.method)
    if e3:
        print("\n=== E3 · DISTRIBUTION SHIFT (leave-one-domain-out) ===")
        print(f"{'held-out domain':<18}{'certified':<22}{'realized':>10}{'savings':>9}{'ok?':>5}")
        print("-" * 64)
        for r in e3:
            if r["certified"] is None:
                print(f"{r['held_out_domain']:<18}{'(none certified)':<22}")
                continue
            print(
                f"{r['held_out_domain']:<18}{r['certified']:<22}{r['realized_risk'] * 100:>9.1f}%"
                f"{r['savings'] * 100:>8.1f}%{'✔' if r['held_within_alpha'] else '✘':>5}"
            )
        print(
            "  (✘ under shift is EXPECTED and is itself a finding — recalibrate per the paper plan)"
        )

    # ---- E4 ----------------------------------------------------------------
    e4 = e4_task_success(matrix, ladder, seed=args.seed)
    if e4:
        print("\n=== E4 · DOWNSTREAM TASK SUCCESS (outcome preserved under compression?) ===")
        if not e4.get("outcome_evidential", True):
            print(
                "  ⚠ NON-EVIDENTIAL OUTCOME: every trajectory shares the same label (no variance —\n"
                "    e.g. swe-hf marks all instances resolved=True by construction). Read the\n"
                "    retained-decision-equivalence below, NOT the success rate, as the signal."
            )
        print(
            f"labeled trajectories: {e4['n']}  ·  baseline success-rate: {e4['baseline_success'] * 100:.1f}%"
        )
        print(f"{'level':<24}{'savings':>9}{'retained-success (95% CI)':>30}")
        print("-" * 63)
        for r in e4["levels"]:
            ci = f"{r['retained_success'] * 100:.1f}% [{r['ci_low'] * 100:.0f}–{r['ci_high'] * 100:.0f}]"
            print(f"{r['level']:<24}{r['savings'] * 100:>8.1f}%{ci:>30}")
        print(
            "  retained-success = originally-successful AND fully decision-equivalent (a flip\n"
            "  puts the outcome at risk). The safe levels hold the baseline; aggressive ones erode it."
        )

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "n_trajectories": len(entries),
                    "n_turns": n_turns,
                    "frontier": f_rows,
                    "head_to_head": h2h,
                    "coverage": cov,
                    "shift": e3,
                    "task_success": e4,
                },
                indent=2,
            )
        )
        print(f"\nreport → {args.report}")

    rc = 0
    if args.gate:
        ok, verdict = gate_check(cov, alpha=args.alpha, delta=args.delta)
        print(f"\n{verdict}")
        if not ok:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

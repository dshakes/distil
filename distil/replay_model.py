"""``distil replay-model`` — does a cheaper model make the same next decision on YOUR traffic?

Dry-run (default): samples N recorded requests from ``~/.distil/sessions/*.requests.jsonl``
and prints the estimated cost of both replay arms, with no API calls.

Live: reconstructs message context from Claude Code session transcripts
(``~/.claude/projects/**/*.jsonl``), calls the target model (B arm) for all N turns and
the original model (A' arm) for a configurable subsample, then reports the paired
A/A'/B decision-equivalence using the same estimator as ``distil shadow-stats``.
Requires ``--spend-cap`` covering both arms; refuses to run without it.

Arms
----
A  — the turn's RECORDED decision (from the transcript; no API call).
A' — the ORIGINAL model replayed on the same reconstructed context (noise-floor arm).
B  — the TARGET model replayed on the same context (the switch-candidate arm).

Paired statistic: mean(1{B==A} − 1{A'==A}) with a bootstrap 95% CI.
Non-inferiority: the lower bound of that CI is ≥ −EQUIV_MARGIN (5 pp, the referee's margin).

Context note: CC transcripts carry the uncompressed messages CC sent. When distil is active
the original call used compressed context; A' and B both use the uncompressed form. The
paired structure (same context for A' and B) preserves the A' vs B comparison; the A vs
{A', B} raw rates include a context-size difference — interpret those with that in mind.

Haiku 5.5 pricing: base $0.10/$0.50 per Mtok for prompts ≤100 k total input tokens;
5× ($0.50/$2.50) when total input > 100 k. From ``pricing.TIERED["claude-haiku-5-5"]``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_TARGET = "claude-haiku-5-5"
DEFAULT_N = 200
DEFAULT_DAYS = 30  # look-back window for session records
HAIKU_LONG_TIER_TOKENS = 100_000
#: Referee's non-inferiority margin (same as distil.referee.EQUIV_MARGIN).
EQUIV_MARGIN = 0.05
DEFAULT_TURNS_PER_SEGMENT = 10
# ICC for the PAIRED DIFFERENCE (1{B==A} − 1{A'==A}).  Differencing cancels most
# session-level variation, so within-cluster correlation is much lower than for
# the raw agreement rate.  0.10 is conservative for code-editing agents; adjust
# via --icc if you observe tighter within-session correlation.
ASSUMED_ICC_DIFF: float = 0.10


# --------------------------------------------------------------------------- #
# Minimum A' sample: smallest n such that the Wilson CI half-width ≤ EQUIV_MARGIN
# --------------------------------------------------------------------------- #


def _min_aa_n(margin: float = EQUIV_MARGIN, z: float = 1.96) -> int:
    """Smallest A' sample for Wilson CI half-width on the noise-floor proportion ≤ *margin*.

    Conservative bound: assumes p = 0.5 (maximum variance for a proportion). The formula
    is the standard sample-size for a proportion CI: n = ceil((z / (2 * margin)) ** 2).
    For EQUIV_MARGIN=0.05: n = ceil((1.96 / 0.10) ** 2) = 385.
    """
    return math.ceil((z / (2 * margin)) ** 2)


# --------------------------------------------------------------------------- #
# Request record (content-free; from sessions/*.requests.jsonl)
# --------------------------------------------------------------------------- #


@dataclass
class ReplayRecord:
    """One proxied request — metadata only, no prompt or response content."""

    model: str
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int
    mode: str
    ts: float
    session: str = ""

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read + self.cache_write


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def load_records(
    n: int,
    *,
    since: float | None = None,
    sessions_dir: Path | None = None,
    rng: random.Random | None = None,
) -> list[ReplayRecord]:
    """Reservoir-sample up to *n* usable request records from the sessions store.

    "Usable" = booked (status 200), model present, at least one input token.
    Content-free: only metadata fields are read. Fails open on malformed rows.
    """
    if since is None:
        since = time.time() - DEFAULT_DAYS * 86400
    root = sessions_dir or (_home() / "sessions")
    rng = rng or random.Random()
    reservoir: list[ReplayRecord] = []
    k = 0
    for path in sorted(Path(root).glob("*.requests.jsonl")):
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = _parse_record(line, since)
                    if row is None:
                        continue
                    k += 1
                    if len(reservoir) < n:
                        reservoir.append(row)
                    else:
                        j = rng.randint(0, k - 1)
                        if j < n:
                            reservoir[j] = row
        except OSError as exc:
            log.debug("replay-model: skipping %s: %s", path, exc)
    return reservoir


def _parse_record(line: str, since: float) -> ReplayRecord | None:
    try:
        r = json.loads(line)
    except ValueError:
        return None
    if not isinstance(r, dict) or not r.get("booked"):
        return None
    ts = float(r.get("ts") or 0)
    if ts < since:
        return None
    model = str(r.get("model") or "")
    if not model:
        return None
    inp = int(r.get("usage_input_tokens") or 0)
    cr = int(r.get("usage_cache_read") or 0)
    cw = int(r.get("usage_cache_create") or 0)
    out = int(r.get("usage_output_tokens") or 0)
    if inp + cr + cw == 0:
        return None
    return ReplayRecord(
        model=model,
        input_tokens=inp,
        cache_read=cr,
        cache_write=cw,
        output_tokens=out,
        mode=str(r.get("mode") or ""),
        ts=ts,
        session=str(r.get("session") or ""),
    )


# --------------------------------------------------------------------------- #
# Tiered cost computation
# --------------------------------------------------------------------------- #


def tiered_cost(rec: ReplayRecord, target: str) -> float | None:
    """Cost of *rec* at *target*'s rates with the long tier applied.

    Looks up TIERED first (haiku-5-5 and future per-call-tier models), then the
    single-tier CATALOG. Returns None for unknown models.
    """
    p = _price(target)
    if p is None:
        return None
    m = _tier_mult(p, rec.total_input)
    return m * (
        rec.input_tokens * p.input
        + rec.cache_read * p.cache_read
        + rec.cache_write * p.cache_write
        + rec.output_tokens * p.output
    )


def original_cost(rec: ReplayRecord) -> float | None:
    """Cost of the original request at the recorded model's list rate."""
    p = _price(rec.model)
    if p is None:
        return None
    m = _tier_mult(p, rec.total_input)
    return m * (
        rec.input_tokens * p.input
        + rec.cache_read * p.cache_read
        + rec.cache_write * p.cache_write
        + rec.output_tokens * p.output
    )


def _price(model: str) -> Any:
    """TIERED row (applied per call) before the single-tier CATALOG."""
    from . import pricing

    return pricing.TIERED.get(model) or pricing.resolve(model)


def _tier_mult(p: Any, total_input: int) -> float:
    lt = getattr(p, "long_tier", None)
    if lt is not None and total_input > lt[0]:
        return float(lt[1])
    return 1.0


# --------------------------------------------------------------------------- #
# Dry-run report: both arms, no API calls
# --------------------------------------------------------------------------- #


@dataclass
class DryRunReport:
    n_sampled: int
    target_model: str
    model_counts: dict[str, int]
    token_percentiles: dict[str, int]  # p50/p90/p95/max total_input
    n_long_tier: int  # records where total_input > HAIKU_LONG_TIER_TOKENS
    # Arm B (target model, all N turns)
    arm_b_usd: float
    arm_b_per_request: float
    # Arm A' (original model, n_aa turns)
    n_aa: int  # subsample size for A' (default = _min_aa_n(), capped at n_sampled)
    arm_aa_usd: float  # estimated A' cost
    arm_aa_per_request: float
    arm_aa_ci_halfwidth: float  # Wilson CI half-width on the noise floor at n_aa, p=0.5
    n_aa_min: int  # _min_aa_n() — n needed to reach the CI target
    # Totals
    original_total_usd: float  # arm A baseline (all N turns)
    n_unpriced_target: int
    n_unpriced_original: int
    days: float
    # Plans (a/b/c) — None until computed by dry_run()
    plans: tuple[ReplayPlan, ReplayPlan, ReplayPlan] | None = None

    def format(self) -> str:
        ratio_b = self.arm_b_usd / self.original_total_usd if self.original_total_usd else 0.0
        lines = [
            f"dry-run: {self.n_sampled} requests sampled from the last {self.days:.0f} days",
            f"  original models: {_fmt_counts(self.model_counts)}",
            f"  input tokens:    p50={self.token_percentiles['p50']:,}  "
            f"p90={self.token_percentiles['p90']:,}  "
            f"p95={self.token_percentiles['p95']:,}  "
            f"max={self.token_percentiles['max']:,}",
            f"  long-tier (>{HAIKU_LONG_TIER_TOKENS:,} tok): "
            f"{self.n_long_tier}/{self.n_sampled} "
            f"({100 * self.n_long_tier // max(self.n_sampled, 1)}%) "
            f"— billed 5x against {self.target_model}",
            "",
            "arm B  (target model, all N turns):",
            f"  model:   {self.target_model}",
            f"  n:       {self.n_sampled}",
            f"  cost:    ${self.arm_b_usd:.4f} total  (${self.arm_b_per_request:.6f}/request)",
            f"  vs original: {ratio_b:.3f}x",
            "",
            "arm A' (original model, A'/B noise-floor subsample):",
            f"  n:       {self.n_aa}  "
            f"(need {self.n_aa_min} for CI half-width ≤ {EQUIV_MARGIN * 100:.0f} pp"
            + ("; WARNING: sample below CI target" if self.n_aa < self.n_aa_min else "")
            + ")",
            f"  cost:    ${self.arm_aa_usd:.4f} total  (${self.arm_aa_per_request:.6f}/request)",
            f"  Wilson CI half-width at n={self.n_aa}: ±{100 * self.arm_aa_ci_halfwidth:.1f} pp",
            "",
            f"  total estimated spend (B + A'): ${self.arm_b_usd + self.arm_aa_usd:.4f}",
            f"  original cost baseline (all N): ${self.original_total_usd:.4f}",
        ]
        if self.n_unpriced_target or self.n_unpriced_original:
            lines.append(
                f"  note: {self.n_unpriced_target} requests unpriced at target, "
                f"{self.n_unpriced_original} unpriced at original (excluded)"
            )
        lines += [
            "",
            "context note: A' and B both use uncompressed CC transcript context.",
            "  The original model saw distil-compressed context; paired A' vs B is",
            "  valid, but raw A vs A'/B rates span a context-size difference.",
        ]
        if self.plans is not None:
            pa, pb, pc = self.plans

            def _fmt_plan(label: str, p: ReplayPlan) -> list[str]:
                rows = [
                    f"{label}",
                    f"  K={p.K} sessions × M={p.M} turns = {p.n_turns} total turns",
                    f"  cost:  B ${p.cost_b_usd:.2f}  +  A' ${p.cost_aa_usd:.2f}"
                    f"  =  ${p.total_usd:.2f}",
                    f"  effective n={p.effective_n:.0f}"
                    f"  (DEFF={1.0 + (p.M - 1) * ASSUMED_ICC_DIFF:.2f},"
                    f" ICC={ASSUMED_ICC_DIFF})",
                    f"  expected CI half-width: ±{100 * p.ci_halfwidth:.1f} pp",
                ]
                if p.note:
                    rows.append(f"  coverage: {p.note}")
                return rows

            lines += [
                "",
                "── prompt-caching plans (1h TTL, cluster bootstrap CI) ──────────────────",
                f"  assumption: ICC_diff={ASSUMED_ICC_DIFF} (paired diff cancels session"
                " correlation)",
                "  A'-agreement ~0.85–0.90 assumed (conservative Wilson CI uses p=0.5)",
                "",
            ]
            lines += _fmt_plan("(a) ±5 pp plan — minimum cost for effective n ≥ 385:", pa)
            lines += [""]
            lines += _fmt_plan("(b) best ≤$50 plan — all contexts:", pb)
            lines += [""]
            lines += _fmt_plan(
                f"(c) best ≤$50 plan — ≤{HAIKU_LONG_TIER_TOKENS:,}-token contexts:", pc
            )
            lines += [
                "",
                "  to run plan (c):",
                f"  distil replay-model --live --sessions {pc.K}"
                f" --turns-per-segment {pc.M}"
                f" --max-context-tokens {HAIKU_LONG_TIER_TOKENS}"
                f" --spend-cap {pc.total_usd:.2f}",
            ]
        else:
            lines += [
                "",
                "to run live:",
                f"  distil replay-model --live --spend-cap {self.arm_b_usd + self.arm_aa_usd:.2f}",
            ]
        return "\n".join(lines)


def _fmt_counts(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    parts = [
        f"{m} {n} ({100 * n // total}%)" for m, n in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return ", ".join(parts[:4])


def _wilson_halfwidth(n: int, z: float = 1.96) -> float:
    """Wilson CI half-width at worst case p=0.5 for the given n."""
    if n <= 0:
        return 1.0
    p = 0.5
    denom = 1.0 + z * z / n
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return half


def _cluster_ci_halfwidth(n_turns: int, M: int, icc: float = ASSUMED_ICC_DIFF) -> float:
    """Expected 95% CI half-width for the paired diff mean with a cluster design.

    DEFF = 1 + (M−1)·ICC.  Effective n = n_turns / DEFF.  Wilson formula after.
    """
    deff = 1.0 + (M - 1) * icc
    return _wilson_halfwidth(max(1, int(n_turns / max(deff, 1.0))))


# --------------------------------------------------------------------------- #
# Session-segment cost model (prompt caching)
# --------------------------------------------------------------------------- #


def _segment_cost(
    ctx_sizes: list[int],
    arm_model: str,
    output_per_turn: int = 1_000,
    *,
    ttl: str = "1h",
) -> float:
    """Estimated cost of one session segment with Anthropic prompt caching.

    First turn: cache-write the full context at the write multiplier.
    Each subsequent turn: cache-read the previous context (much cheaper)
    plus cache-write the incremental new tokens.  Output always fresh.
    Haiku 5.5 long-tier 5× applies to ALL token types when total > 100 k.
    """
    p = _price(arm_model)
    if p is None or not ctx_sizes:
        return 0.0
    w = p.input * (p.cache_write_1h_mult if ttl == "1h" else p.cache_write_mult)
    r = p.input * p.cache_read_mult
    total = 0.0
    prev = 0
    for k, ctx in enumerate(ctx_sizes):
        m = _tier_mult(p, ctx)
        if k == 0:
            total += m * (ctx * w + output_per_turn * p.output)
        else:
            delta = max(0, ctx - prev)
            total += m * (prev * r + delta * w + output_per_turn * p.output)
        prev = ctx
    return total


def cluster_bootstrap_ci(
    cluster_diffs: list[list[int]],
    *,
    n_boot: int = 2_000,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """95% bootstrap CI for mean(B≡A − A'≡A), resampling at the segment level.

    *cluster_diffs* — one inner list per session segment; each element is the
    paired difference (0 or ±1) for one turn in that segment.
    Returns (lower, upper) of a two-sided 95% CI.
    """
    if not cluster_diffs:
        return (-1.0, 1.0)
    _rng = rng or random.Random()
    K = len(cluster_diffs)
    cmeans = [sum(seg) / len(seg) for seg in cluster_diffs if seg]
    if not cmeans:
        return (-1.0, 1.0)
    boot: list[float] = sorted(
        sum(_rng.choices(cmeans, k=K)) / K
        for _ in range(n_boot)  # type: ignore[arg-type]
    )
    lo = boot[max(0, int(0.025 * n_boot))]
    hi = boot[min(len(boot) - 1, int(0.975 * n_boot))]
    return lo, hi


# --------------------------------------------------------------------------- #
# Dry-run plan optimizer
# --------------------------------------------------------------------------- #


@dataclass
class ReplayPlan:
    """One proposed K-segments × M-turns replay plan."""

    K: int  # session segments
    M: int  # turns per segment
    n_turns: int  # K * M
    cost_b_usd: float
    cost_aa_usd: float
    total_usd: float
    effective_n: float
    ci_halfwidth: float  # expected Wilson CI half-width
    note: str = ""


def _mode_model(records: list[ReplayRecord]) -> str:
    """Most common model in *records*."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r.model] = counts.get(r.model, 0) + 1
    return max(counts, key=lambda m: counts[m]) if counts else "claude-opus-5-5"


def _dry_run_plans(
    records: list[ReplayRecord],
    target: str,
    *,
    icc: float = ASSUMED_ICC_DIFF,
    budget_usd: float = 50.0,
) -> tuple[ReplayPlan, ReplayPlan, ReplayPlan]:
    """Return (plan_a, plan_b, plan_c) for the three target scenarios.

    plan_a — minimum cost that achieves effective_n ≥ 385 (≤ 5 pp CI half-width).
    plan_b — best effective_n achievable within *budget_usd* (all contexts).
    plan_c — best effective_n within *budget_usd* restricted to ≤ 100 k-token turns.

    Uses the mean context size from *records* as the representative prompt.
    Assumes:
    - A'-agreement ~0.85 (stated, not encoded in the CI formula; the CI formula
      is conservative at p=0.5).
    - ICC for the paired difference = *icc* (default ASSUMED_ICC_DIFF=0.10).
    - 1-hour cache TTL (cache_write_1h_mult × base rate for first turn of each
      segment; cache_read_mult × base rate for subsequent turns).
    """
    _empty = ReplayPlan(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.0, "no data")
    if not records:
        return _empty, _empty, _empty

    aa_model = _mode_model(records)
    all_ctx = [r.total_input for r in records]
    n = len(all_ctx)
    mean_ctx = int(sum(all_ctx) / n)
    mean_out = int(sum(r.output_tokens for r in records) / n)

    sc_ctx = [c for c in all_ctx if c <= HAIKU_LONG_TIER_TOKENS]
    mean_ctx_sc = int(sum(sc_ctx) / len(sc_ctx)) if sc_ctx else mean_ctx
    cov_count = len(sc_ctx) / n
    cov_tokens = sum(sc_ctx) / sum(all_ctx) if all_ctx else 0.0

    def _make(M: int, mean_c: int, K_override: int | None, budget: float) -> ReplayPlan:
        ctx_sizes = [mean_c] * M
        cost_b_seg = _segment_cost(ctx_sizes, target, mean_out)
        cost_aa_seg = _segment_cost(ctx_sizes, aa_model, mean_out)
        cost_seg = cost_b_seg + cost_aa_seg
        if cost_seg <= 0:
            return ReplayPlan(0, M, 0, 0.0, 0.0, 0.0, 0.0, 1.0, "unpriced")
        if K_override is not None:
            K = K_override
        else:
            K = max(0, int(budget / cost_seg))
        if K == 0:
            return ReplayPlan(0, M, 0, 0.0, 0.0, 0.0, 0.0, 1.0, "K=0")
        n_turns = K * M
        deff = 1.0 + (M - 1) * icc
        n_eff = n_turns / max(deff, 1.0)
        return ReplayPlan(
            K=K,
            M=M,
            n_turns=n_turns,
            cost_b_usd=K * cost_b_seg,
            cost_aa_usd=K * cost_aa_seg,
            total_usd=K * cost_seg,
            effective_n=n_eff,
            ci_halfwidth=_wilson_halfwidth(max(1, int(n_eff))),
        )

    Ms = [1, 5, 10, 15, 20]

    # Plan (a): minimize total cost for effective_n ≥ _min_aa_n() = 385
    n_min = _min_aa_n()
    best_a: ReplayPlan | None = None
    for M in Ms:
        deff = 1.0 + (M - 1) * icc
        K = math.ceil(n_min * deff / M)
        p = _make(M, mean_ctx, K, 0.0)
        if best_a is None or (p.K > 0 and p.total_usd < best_a.total_usd):
            best_a = p
    plan_a = best_a or _make(10, mean_ctx, 74, 0.0)

    # Plan (b): maximize effective_n subject to total ≤ budget_usd (all contexts)
    best_b: ReplayPlan | None = None
    for M in Ms:
        p = _make(M, mean_ctx, None, budget_usd)
        if best_b is None or (p.K > 0 and p.effective_n > best_b.effective_n):
            best_b = p
    plan_b = best_b or _make(10, mean_ctx, None, budget_usd)

    # Plan (c): same budget, ≤ 100 k-token contexts only
    best_c: ReplayPlan | None = None
    for M in Ms:
        p = _make(M, mean_ctx_sc, None, budget_usd)
        if best_c is None or (p.K > 0 and p.effective_n > best_c.effective_n):
            best_c = p
    base_c = best_c or _make(10, mean_ctx_sc, None, budget_usd)
    plan_c = ReplayPlan(
        K=base_c.K,
        M=base_c.M,
        n_turns=base_c.n_turns,
        cost_b_usd=base_c.cost_b_usd,
        cost_aa_usd=base_c.cost_aa_usd,
        total_usd=base_c.total_usd,
        effective_n=base_c.effective_n,
        ci_halfwidth=base_c.ci_halfwidth,
        note=(
            f"covers {100 * cov_count:.0f}% of requests "
            f"({100 * cov_tokens:.0f}% of tokens) in this window"
        ),
    )
    return plan_a, plan_b, plan_c


def dry_run(
    records: list[ReplayRecord],
    target: str,
    *,
    n_aa: int | None = None,
    days: float = float(DEFAULT_DAYS),
) -> DryRunReport:
    """Compute both arm cost estimates without any API calls.

    *n_aa* is the A' subsample size. If None, defaults to the minimum that gives
    Wilson CI half-width ≤ EQUIV_MARGIN on the noise floor, capped at len(records).
    """
    model_counts: dict[str, int] = {}
    toks: list[int] = []
    n_long = n_unp_tgt = n_unp_orig = 0
    arm_b_total = orig_total = 0.0
    per_orig: list[float] = []  # original cost per record (for A' arm estimate)

    for rec in records:
        model_counts[rec.model] = model_counts.get(rec.model, 0) + 1
        toks.append(rec.total_input)
        if rec.total_input > HAIKU_LONG_TIER_TOKENS:
            n_long += 1
        c_tgt = tiered_cost(rec, target)
        c_orig = original_cost(rec)
        if c_tgt is None:
            n_unp_tgt += 1
        else:
            arm_b_total += c_tgt
        if c_orig is None:
            n_unp_orig += 1
            per_orig.append(0.0)
        else:
            orig_total += c_orig
            per_orig.append(c_orig)

    toks.sort()
    n = len(toks)

    def pct(q: float) -> int:
        return toks[min(int(q * n), n - 1)] if n else 0

    n_aa_min = _min_aa_n()
    if n_aa is None:
        n_aa = min(n, n_aa_min)
    else:
        n_aa = min(n, n_aa)

    # A' arm cost: take the first n_aa records (ordered as they arrived; a random
    # subsample would give the same expectation, so this is a clean estimate).
    arm_aa_total = sum(per_orig[:n_aa])
    arm_aa_per = arm_aa_total / n_aa if n_aa else 0.0

    plans = _dry_run_plans(records, target)
    return DryRunReport(
        n_sampled=n,
        target_model=target,
        model_counts=model_counts,
        token_percentiles={
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "max": toks[-1] if toks else 0,
        },
        n_long_tier=n_long,
        arm_b_usd=arm_b_total,
        arm_b_per_request=arm_b_total / n if n else 0.0,
        n_aa=n_aa,
        arm_aa_usd=arm_aa_total,
        arm_aa_per_request=arm_aa_per,
        arm_aa_ci_halfwidth=_wilson_halfwidth(n_aa),
        n_aa_min=n_aa_min,
        original_total_usd=orig_total,
        n_unpriced_target=n_unp_tgt,
        n_unpriced_original=n_unp_orig,
        days=days,
        plans=plans,
    )


# --------------------------------------------------------------------------- #
# Transcript turns (live replay source)
# --------------------------------------------------------------------------- #


@dataclass
class TranscriptTurn:
    """One conversation turn: reconstructed messages context + the recorded decision."""

    messages: list[dict[str, Any]]  # sent TO the model (does not include this response)
    recorded_decision: str  # decision_signature of the MERGED original message
    model: str
    input_tokens: int  # total billed input (includes cache fields)
    output_tokens: int
    ts: float
    session_id: str = ""


def _cc_root() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "projects"


def load_transcript_turns(
    n: int,
    *,
    since: float | None = None,
    transcripts_dir: Path | None = None,
    rng: random.Random | None = None,
) -> list[TranscriptTurn]:
    """Reservoir-sample up to *n* turns from Claude Code session transcripts."""
    if since is None:
        since = time.time() - DEFAULT_DAYS * 86400
    root = transcripts_dir or _cc_root()
    rng = rng or random.Random()
    reservoir: list[TranscriptTurn] = []
    k = 0
    for jsonl_path in sorted(Path(root).glob("**/*.jsonl")):
        try:
            k = _sample_file(jsonl_path, since, reservoir, n, k, rng)
        except (OSError, ValueError):
            pass
    return reservoir


def _sample_file(
    path: Path,
    since: float,
    reservoir: list[TranscriptTurn],
    n: int,
    k_before: int,
    rng: random.Random,
    *,
    _collect_into: dict[str, list[TranscriptTurn]] | None = None,
) -> int:
    """Parse one transcript file; reservoir-sample turns into *reservoir*. Returns updated k.

    When *_collect_into* is provided the function appends every valid turn to that
    dict keyed by session_id (no reservoir sampling); *reservoir* and *n* are ignored.
    """

    try:
        with path.open(encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return k_before

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass

    by_uuid: dict[str, dict[str, Any]] = {str(r["uuid"]): r for r in rows if r.get("uuid")}

    # Build reverse index: parent_uuid → list of child rows.
    # Used by _build_messages to find sibling tool_result rows that CC writes as
    # children of intermediate assistant fragments (not on the main chain path).
    children_by_uuid: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        par = r.get("parentUuid")
        if par:
            children_by_uuid.setdefault(str(par), []).append(r)

    # Rows whose uuid is the parent of another assistant row with the SAME message.id
    # are NOT the last fragment; we skip them to avoid double-sampling.
    non_last_fragments: set[str] = set()
    for r in rows:
        par_uuid = r.get("parentUuid") or ""
        if not par_uuid or r.get("type") != "assistant":
            continue
        par = by_uuid.get(par_uuid)
        if par is None or par.get("type") != "assistant":
            continue
        r_mid = (r.get("message") or {}).get("id") or ""
        par_mid = (par.get("message") or {}).get("id") or ""
        if r_mid and r_mid == par_mid:
            non_last_fragments.add(par_uuid)

    k = k_before
    for r in rows:
        if r.get("type") != "assistant":
            continue
        uuid = r.get("uuid") or ""
        if uuid in non_last_fragments:
            continue  # only process the last fragment of each message group
        msg = r.get("message") or {}
        if not isinstance(msg, dict):
            continue
        ts = _epoch(r.get("timestamp"))
        if ts < since:
            continue
        model = str(msg.get("model") or "")
        if not model:
            continue

        result = _build_messages(r, by_uuid, children_by_uuid=children_by_uuid)
        if result is None:
            continue
        messages, rec_dec = result
        if rec_dec == "none":
            continue

        usage = msg.get("usage") or {}
        inp = int(usage.get("input_tokens") or 0)
        cr = int(usage.get("cache_read_input_tokens") or 0)
        cw = int(usage.get("cache_creation_input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        turn = TranscriptTurn(
            messages=messages,
            recorded_decision=rec_dec,
            model=model,
            input_tokens=inp + cr + cw,
            output_tokens=out,
            ts=ts,
            session_id=str(r.get("sessionId") or ""),
        )
        k += 1
        if _collect_into is not None:
            _collect_into.setdefault(turn.session_id or "", []).append(turn)
        elif len(reservoir) < n:
            reservoir.append(turn)
        else:
            j = rng.randint(0, k - 1)
            if j < n:
                reservoir[j] = turn
    return k


def load_transcript_segments(
    n_sessions: int,
    turns_per_segment: int,
    since: float,
    *,
    max_context_tokens: int | None = None,
    transcripts_dir: Path | None = None,
    rng: random.Random | None = None,
) -> tuple[list[list[TranscriptTurn]], dict[str, Any]]:
    """Return (segments, coverage) where each segment is M contiguous turns from one session.

    Only turns with at least *turns_per_segment* qualifying neighbours in their session are
    eligible.  When *max_context_tokens* is set, turns whose context_tokens exceed the limit
    are excluded from segment eligibility (but counted toward coverage totals).

    coverage dict keys: total_turns, scoped_turns, total_tokens, scoped_tokens,
    request_coverage_frac, token_coverage_frac.
    """
    _rng = rng or random.Random()
    root = transcripts_dir or _cc_root()

    all_by_session: dict[str, list[TranscriptTurn]] = {}
    for jsonl_path in sorted(Path(root).glob("**/*.jsonl")):
        try:
            _sample_file(
                jsonl_path,
                since,
                [],
                0,
                0,
                _rng,
                _collect_into=all_by_session,
            )
        except (OSError, ValueError):
            pass

    # Coverage stats
    total_turns = sum(len(v) for v in all_by_session.values())
    total_tokens = sum(t.input_tokens for turns in all_by_session.values() for t in turns)

    # Filter turns per scope
    if max_context_tokens is not None:
        scoped_by_session: dict[str, list[TranscriptTurn]] = {
            sid: [t for t in turns if t.input_tokens <= max_context_tokens]
            for sid, turns in all_by_session.items()
        }
    else:
        scoped_by_session = all_by_session

    scoped_turns = sum(len(v) for v in scoped_by_session.values())
    scoped_tokens = sum(t.input_tokens for turns in scoped_by_session.values() for t in turns)

    request_coverage_frac = scoped_turns / total_turns if total_turns else 0.0
    token_coverage_frac = scoped_tokens / total_tokens if total_tokens else 0.0

    coverage: dict[str, Any] = {
        "total_turns": total_turns,
        "scoped_turns": scoped_turns,
        "total_tokens": total_tokens,
        "scoped_tokens": scoped_tokens,
        "request_coverage_frac": request_coverage_frac,
        "token_coverage_frac": token_coverage_frac,
    }

    # Eligible sessions: those with >= turns_per_segment qualifying turns
    eligible: list[str] = [
        sid for sid, turns in scoped_by_session.items() if len(turns) >= turns_per_segment
    ]

    if not eligible:
        return [], coverage

    # Reservoir-sample n_sessions sessions
    chosen_sessions: list[str] = []
    for i, sid in enumerate(eligible):
        if i < n_sessions:
            chosen_sessions.append(sid)
        else:
            j = _rng.randint(0, i)
            if j < n_sessions:
                chosen_sessions[j] = sid

    segments: list[list[TranscriptTurn]] = []
    for sid in chosen_sessions:
        turns = scoped_by_session[sid]
        # Pick a random start offset so turns are contiguous
        max_start = len(turns) - turns_per_segment
        start = _rng.randint(0, max_start)
        segments.append(turns[start : start + turns_per_segment])

    return segments, coverage


_THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))

_MAX_CACHE_BREAKPOINTS = 4


def _add_cache_breakpoints(
    messages: list[dict[str, Any]],
    *,
    ttl: str = "1h",
    seam_index: int | None = None,
) -> list[dict[str, Any]]:
    """Return a shallow copy of *messages* with cache_control breakpoints injected.

    Tags the last content block of the last message.  When *seam_index* is given
    (the index of the last message from the previous turn) also tags that position
    so the shared prefix is re-used on turn k+1.  At most _MAX_CACHE_BREAKPOINTS
    breakpoints are placed.
    """
    cc = {"type": "ephemeral", "ttl": ttl} if ttl == "1h" else {"type": "ephemeral"}

    def _tag_last_block(msg: dict[str, Any]) -> dict[str, Any]:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            return msg
        blocks = [dict(b) for b in content]
        blocks[-1] = dict(blocks[-1])
        blocks[-1]["cache_control"] = cc
        return {**msg, "content": blocks}

    out = list(messages)
    positions: list[int] = []
    if seam_index is not None and 0 <= seam_index < len(out) - 1:
        positions.append(seam_index)
    positions.append(len(out) - 1)

    # Apply at most _MAX_CACHE_BREAKPOINTS (de-dup, keep last N)
    unique_pos = list(dict.fromkeys(positions))[-_MAX_CACHE_BREAKPOINTS:]
    for idx in unique_pos:
        out[idx] = _tag_last_block(out[idx])
    return out


def _is_pure_tool_result(content: Any) -> bool:
    """True iff content is a list of only tool_result blocks."""
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def _build_messages(
    decision_row: dict[str, Any],
    by_uuid: dict[str, dict[str, Any]],
    max_depth: int = 80,
    *,
    children_by_uuid: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], str] | None:
    """Reconstruct the messages context and the recorded decision for one turn.

    CC JSONL writes one row per content block of an assistant message (same
    ``message.id``, chained by ``parentUuid``).  This function:

    1. Walks up through same-``message.id`` assistant rows to collect ALL
       fragments of the current decision message.
    2. Merges their content blocks to compute the recorded decision via
       ``shadow.decision_signature``.
    3. Walks further up from the oldest fragment's parent to build the context
       chain.
    4. Merges consecutive same-``message.id`` assistant rows into single
       API messages; merges consecutive pure-tool_result user rows; strips ALL
       thinking/redacted_thinking blocks (no fall-back to original).
    5. Validates alternation and that the last message is a user message.

    Returns ``(messages, recorded_decision)`` or ``None`` if the turn cannot
    be reconstructed cleanly.
    """
    from .shadow import decision_signature as _dec_sig

    dec_msg_id = (decision_row.get("message") or {}).get("id") or ""

    # Step 1: walk up through same-message.id assistant rows.
    decision_fragments: list[dict[str, Any]] = [decision_row]
    cur: dict[str, Any] = decision_row
    for _ in range(max_depth):
        par_uuid = cur.get("parentUuid") or ""
        if not par_uuid:
            break
        par = by_uuid.get(par_uuid)
        if par is None:
            break
        # Only group fragments when both share the same non-empty message.id.
        par_mid = (par.get("message") or {}).get("id") or ""
        if dec_msg_id and par.get("type") == "assistant" and par_mid == dec_msg_id:
            decision_fragments.append(par)
            cur = par
        else:
            break  # cur's parent is the context_start row

    decision_fragments.reverse()  # chronological: [oldest, …, decision_row]

    # Step 2: merged decision content → recorded_decision.
    dec_content: list[dict[str, Any]] = []
    for frag in decision_fragments:
        fc = (frag.get("message") or {}).get("content") or []
        if isinstance(fc, list):
            dec_content.extend(fc)
    recorded_decision = _dec_sig({"content": dec_content})

    # Step 3: walk up from the oldest fragment's parent.
    context_start_uuid = decision_fragments[0].get("parentUuid") or ""
    if not context_start_uuid:
        return None  # No prior context at all (first turn in the session).
    context_start = by_uuid.get(context_start_uuid)
    if context_start is None:
        return None  # Parent not in this file — incomplete chain.

    chain: list[dict[str, Any]] = [context_start]
    cur = context_start
    for _ in range(max_depth):
        par_uuid = cur.get("parentUuid") or ""
        if not par_uuid:
            break
        par = by_uuid.get(par_uuid)
        if par is None:
            break
        chain.append(par)
        cur = par
    chain.reverse()  # chronological

    # Step 4: build API messages.
    api_messages: list[dict[str, Any]] = []
    i = 0
    while i < len(chain):
        row = chain[i]
        msg = row.get("message") or {}
        role = msg.get("role") or row.get("type")
        if role not in ("user", "assistant"):
            i += 1
            continue

        if role == "assistant":
            msg_id = msg.get("id") or ""
            # Merge all consecutive assistant rows with the same message.id.
            # CC writes one row per content block (same message.id, chained by
            # parentUuid).  When an assistant turn has N parallel tool_use
            # blocks, CC writes N assistant-fragment rows *and* N-1 sibling
            # user[tool_result] rows as children of the intermediate fragments
            # (not on the main parentUuid chain).  We must collect those sibling
            # results so every tool_use ends up paired with its tool_result.
            content_blocks: list[dict[str, Any]] = []
            sibling_results: list[dict[str, Any]] = []  # from off-chain branches
            fragment_uuids: list[str] = []
            j = i
            while j < len(chain):
                r2 = chain[j]
                m2 = r2.get("message") or {}
                r2_role = m2.get("role") or r2.get("type")
                if r2_role != "assistant":
                    break
                r2_mid = m2.get("id") or ""
                # Only merge rows that share the same non-empty message.id.
                # A row with no message.id is a singleton — stop after the first.
                if j > i and (not msg_id or r2_mid != msg_id):
                    break
                c = m2.get("content") or []
                if isinstance(c, list):
                    content_blocks.extend(c)
                elif isinstance(c, str):
                    content_blocks.append({"type": "text", "text": c})
                r2_uuid = r2.get("uuid") or ""
                if r2_uuid:
                    fragment_uuids.append(r2_uuid)
                j += 1

            # For each INTERMEDIATE fragment (all except the last), look for
            # sibling user[tool_result] rows that CC branched off it.  The next
            # fragment in the chain is the "direct child" — everything else that
            # is a user row with pure tool_results is a sibling result.
            next_chain_uuid = chain[j].get("uuid") if j < len(chain) else None
            for frag_uuid in fragment_uuids[:-1]:
                for child in (children_by_uuid or {}).get(frag_uuid, []):
                    child_uuid = child.get("uuid") or ""
                    if child_uuid == next_chain_uuid:
                        continue  # This is the next fragment in the main chain.
                    if child.get("type") != "user":
                        continue
                    child_content = (child.get("message") or {}).get("content") or []
                    if _is_pure_tool_result(child_content):
                        sibling_results.extend(child_content)

            # Strip ALL thinking — no fall-back to original on empty result.
            filtered = [
                b
                for b in content_blocks
                if isinstance(b, dict) and b.get("type") not in _THINKING_TYPES
            ]
            if filtered:
                api_messages.append({"role": "assistant", "content": filtered})
            # If only thinking blocks remain, skip this assistant message entirely.

            # Collect any following pure-tool_result user rows from the chain;
            # combine with sibling results into one merged user message.
            all_results: list[dict[str, Any]] = list(sibling_results)
            while j < len(chain):
                r2 = chain[j]
                m2 = r2.get("message") or {}
                r2_role = m2.get("role") or r2.get("type")
                if r2_role != "user":
                    break
                c2 = m2.get("content") or []
                if not _is_pure_tool_result(c2):
                    break
                all_results.extend(c2)  # type: ignore[arg-type]
                j += 1
            if all_results:
                api_messages.append({"role": "user", "content": all_results})
            i = j

        else:  # user
            content = msg.get("content") or []
            if isinstance(content, str):
                api_messages.append({"role": "user", "content": content})
                i += 1
            elif isinstance(content, list):
                if content:
                    api_messages.append({"role": "user", "content": content})
                i += 1
            else:
                i += 1

    # Step 5: validate.
    if not api_messages:
        return None
    if api_messages[-1]["role"] != "user":
        return None
    for idx in range(1, len(api_messages)):
        if api_messages[idx]["role"] == api_messages[idx - 1]["role"]:
            return None

    return api_messages, recorded_decision


def validate_turn(turn: TranscriptTurn) -> list[str]:
    """Structural validation of a reconstructed turn. Returns a list of errors (empty = ok).

    Checks:
    * Last message is ``user``.
    * Messages alternate ``user``/``assistant``.
    * No ``thinking`` or ``redacted_thinking`` blocks anywhere.
    * Every ``tool_use`` id in an assistant message has a matching ``tool_result``
      in the immediately following user message.
    * Every ``tool_result`` ``tool_use_id`` in a user message matches a ``tool_use``
      in the immediately preceding assistant message.
    """
    errs: list[str] = []
    msgs = turn.messages
    if not msgs:
        errs.append("empty message list")
        return errs
    if msgs[-1]["role"] != "user":
        errs.append(f"last message is {msgs[-1]['role']!r}, not 'user'")
    for i, m in enumerate(msgs):
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            errs.append(f"msg {i}: unexpected role {role!r}")
        if i > 0 and role == msgs[i - 1].get("role"):
            errs.append(f"msg {i}: consecutive {role!r} messages")
        content = m.get("content") or []
        if isinstance(content, list):
            for j, b in enumerate(content):
                if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                    errs.append(f"msg {i} block {j}: thinking block present")
            # Tool pairing.
            if role == "assistant":
                tu_ids = {
                    b["id"]
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
                }
                if tu_ids:
                    if i + 1 >= len(msgs):
                        errs.append(f"msg {i}: tool_use(s) {tu_ids} with no following user message")
                    else:
                        next_content = msgs[i + 1].get("content") or []
                        tr_ids = (
                            {
                                b.get("tool_use_id")
                                for b in next_content
                                if isinstance(b, dict) and b.get("type") == "tool_result"
                            }
                            if isinstance(next_content, list)
                            else set()
                        )
                        missing = tu_ids - tr_ids
                        if missing:
                            errs.append(
                                f"msg {i}: tool_use id(s) {missing} missing tool_result in msg {i + 1}"
                            )
            elif role == "user":
                next_content = content
                tr_ids = (
                    {
                        b.get("tool_use_id")
                        for b in next_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                    if isinstance(next_content, list)
                    else set()
                )
                if tr_ids:
                    if i == 0:
                        errs.append(f"msg {i}: tool_result(s) with no preceding assistant message")
                    else:
                        prev_content = msgs[i - 1].get("content") or []
                        tu_ids = (
                            {
                                b.get("id")
                                for b in prev_content
                                if isinstance(b, dict) and b.get("type") == "tool_use"
                            }
                            if isinstance(prev_content, list)
                            else set()
                        )
                        orphaned = tr_ids - tu_ids
                        if orphaned:
                            errs.append(
                                f"msg {i}: tool_result id(s) {orphaned} not in preceding assistant message"
                            )
    return errs


def preflight_count_tokens(
    turns: list[TranscriptTurn],
    target: str,
    *,
    api_key: str,
) -> tuple[list[TranscriptTurn], int, int]:
    """Call ``POST /v1/messages/count_tokens`` (free) for each turn.

    Returns ``(valid_turns, n_pass, n_fail)``. A 400 from count_tokens means the
    message shape would also fail a paid ``/v1/messages`` call.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")
    valid: list[TranscriptTurn] = []
    n_fail = 0
    for turn in turns:
        try:
            client.messages.count_tokens(
                model=target,
                messages=turn.messages,  # type: ignore[arg-type]
            )
            valid.append(turn)
        except anthropic.BadRequestError as exc:
            log.debug("count_tokens rejected turn: %s", exc)
            n_fail += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("count_tokens error (turn skipped): %s", exc)
            n_fail += 1
    return valid, len(valid), n_fail


def _epoch(iso: Any) -> float:
    if not iso:
        return 0.0
    try:
        import datetime

        dt = datetime.datetime.fromisoformat(str(iso).rstrip("Z"))
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except (ValueError, TypeError):
        return 0.0


# --------------------------------------------------------------------------- #
# Paired live replay
# --------------------------------------------------------------------------- #


@dataclass
class PairedLiveResult:
    """Result of a live A/A'/B paired replay.

    n_b: total B-arm turns.
    n_aa: A'-arm turns (a subset of the B turns, sharing the same context).
    Equivalence fields reuse shadow.Equivalence via _equivalence().
    """

    n_b: int
    n_aa: int
    n_eq_b: int  # B == A (recorded)
    n_eq_aa: int  # A' == A (recorded)
    diffs: list[int]  # per paired turn: 1{B==A} − 1{A'==A}
    spend_b_usd: float
    spend_aa_usd: float
    target_model: str
    original_model_counts: dict[str, int]  # for display
    segment_diffs: list[list[int]] = field(default_factory=list)  # per segment

    @property
    def equivalence(self) -> Any:
        """shadow.Equivalence via _equivalence(). Lazy to avoid import at module load."""
        from .shadow import _equivalence

        return _equivalence(self.n_b, self.n_eq_b, self.n_aa, self.n_eq_aa, self.diffs)

    def format(self) -> str:
        from .shadow import bootstrap_ci, wilson_ci

        lines = [
            f"live replay: B arm n={self.n_b}, A' arm n={self.n_aa}",
            f"  target model:  {self.target_model}",
            f"  B spend: ${self.spend_b_usd:.4f}  "
            f"A' spend: ${self.spend_aa_usd:.4f}  "
            f"total: ${self.spend_b_usd + self.spend_aa_usd:.4f}",
            "",
            "equivalence (B vs A, the raw rate):",
            f"  {self.n_eq_b}/{self.n_b} ({100 * self.n_eq_b / max(self.n_b, 1):.1f}%)",
        ]
        b_lo, b_hi = wilson_ci(self.n_eq_b, self.n_b)
        lines.append(f"  Wilson 95% CI: [{100 * b_lo:.1f}%, {100 * b_hi:.1f}%]")

        lines += [
            "",
            "noise floor (A' vs A, the model's own self-agreement):",
            f"  {self.n_eq_aa}/{self.n_aa} ({100 * self.n_eq_aa / max(self.n_aa, 1):.1f}%)",
        ]
        if self.n_aa:
            aa_lo, aa_hi = wilson_ci(self.n_eq_aa, self.n_aa)
            lines.append(f"  Wilson 95% CI: [{100 * aa_lo:.1f}%, {100 * aa_hi:.1f}%]")

        if self.diffs:
            diff_mean = sum(self.diffs) / len(self.diffs)
            diff_ci = bootstrap_ci(self.diffs)
            lines += [
                "",
                "paired diff mean(1{B==A} − 1{A'==A}):",
                f"  {diff_mean:+.3f}  bootstrap 95% CI: "
                f"[{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]  n={len(self.diffs)}",
                f"  non-inferiority (lower bound ≥ −{EQUIV_MARGIN}): "
                + (
                    "PASS"
                    if diff_ci[0] >= -EQUIV_MARGIN
                    else f"FAIL (lower bound {diff_ci[0]:+.3f} < −{EQUIV_MARGIN})"
                ),
            ]
        lines += [
            "",
            "context note: A' and B use reconstructed (uncompressed) context.",
            "  Paired A' vs B comparison is valid; raw A vs A'/B rates span a",
            "  context-size difference when distil was active.",
        ]
        return "\n".join(lines)


def run_live(
    turns: list[TranscriptTurn],
    target: str,
    spend_cap: float,
    *,
    n_aa: int | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    segments: list[list[TranscriptTurn]] | None = None,
    cache_ttl: str = "1h",
) -> PairedLiveResult:
    """Paired A/A'/B replay.

    Gates applied before any spend (no exceptions):
    1. Structural validator (``validate_turn``) — turns failing validation are
       dropped from both arms so the comparison stays paired.
    2. Free ``count_tokens`` preflight — same shape errors return 400 at $0.

    A' and B use the IDENTICAL validated turn set.

    When *segments* is provided (list-of-lists from ``load_transcript_segments``),
    cache breakpoints are injected so turns within a segment reuse the shared prefix.
    The result's ``segment_diffs`` field groups paired diffs by segment, enabling
    cluster bootstrap CI via ``cluster_bootstrap_ci``.
    """
    import anthropic

    from .shadow import decision_signature

    key: str = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""

    # Derive flat turns list (for gates and backward compat) from segments if given.
    if segments is not None:
        turns = [t for seg in segments for t in seg]

    # Gate 1: structural validation — drop invalid turns from BOTH arms equally.
    valid_set: set[int] = set()
    n_struct_fail = 0
    for idx, t in enumerate(turns):
        errs = validate_turn(t)
        if errs:
            log.debug("replay-model: structural validation failed: %s", errs)
            n_struct_fail += 1
        else:
            valid_set.add(idx)
    if n_struct_fail:
        log.info(
            "replay-model: %d/%d turns failed structural validation (dropped from both arms)",
            n_struct_fail,
            len(turns),
        )

    # Gate 2: count_tokens preflight (free; same 400 errors as a real call at $0).
    valid_list = [turns[i] for i in sorted(valid_set)]
    valid_list, n_ct_pass, n_ct_fail = preflight_count_tokens(valid_list, target, api_key=key)
    if n_ct_fail:
        log.info(
            "replay-model: %d turns rejected by count_tokens preflight (dropped from both arms)",
            n_ct_fail,
        )
    valid_set = {id(t) for t in valid_list}

    # Rebuild segments/flat turns retaining only validated entries.
    if segments is not None:
        filtered_segments: list[list[TranscriptTurn]] = [
            [t for t in seg if id(t) in valid_set] for seg in segments
        ]
        filtered_segments = [seg for seg in filtered_segments if seg]
        turns_flat: list[TranscriptTurn] = [t for seg in filtered_segments for t in seg]
    else:
        filtered_segments = []
        turns_flat = valid_list

    n_aa_actual = _min_aa_n() if n_aa is None else n_aa
    n_aa_actual = min(n_aa_actual, len(turns_flat))
    aa_indices: set[int] = set(range(n_aa_actual))  # first n_aa turns get A' call
    client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")

    # Build a mapping from turn object id → flat index.
    turn_flat_index: dict[int, int] = {id(t): i for i, t in enumerate(turns_flat)}

    # Build per-segment seam table: for each turn inside a segment, the index of
    # the last message from the *previous* turn's context (for cache placement).
    seam_table: dict[int, int | None] = {}  # id(turn) → seam_index or None
    if segments is not None:
        for seg in filtered_segments:
            for k, t in enumerate(seg):
                seam_table[id(t)] = (len(seg[k - 1].messages) - 1) if k > 0 else None
    else:
        for t in turns_flat:
            seam_table[id(t)] = None

    # Group flat turns by segment index (for cluster diff tracking).
    # flat_turn_seg_idx[i] = which segment turn i belongs to (-1 if flat mode)
    flat_turn_seg_idx: dict[int, int] = {}
    if segments is not None:
        for si, seg in enumerate(filtered_segments):
            for t in seg:
                flat_idx = turn_flat_index.get(id(t))
                if flat_idx is not None:
                    flat_turn_seg_idx[flat_idx] = si
    else:
        for i in range(len(turns_flat)):
            flat_turn_seg_idx[i] = 0  # single "segment"

    def _cached_messages(turn: TranscriptTurn) -> list[dict[str, Any]]:
        seam = seam_table.get(id(turn))
        if seam is None and not filtered_segments:
            return list(turn.messages)
        return _add_cache_breakpoints(
            list(turn.messages),
            ttl=cache_ttl,
            seam_index=seam,
        )

    n_eq_b = 0
    n_eq_aa = 0
    spend_b = 0.0
    spend_aa = 0.0
    n_b_done = 0
    diffs: list[int] = []
    # Record B decisions for the paired turns (we need them when computing A' diffs).
    b_decisions: dict[int, str] = {}

    n_segments = max(max(flat_turn_seg_idx.values(), default=0) + 1, 1)
    seg_diffs: list[list[int]] = [[] for _ in range(n_segments)]

    # --- B arm: all turns ---
    for i, turn in enumerate(turns_flat):
        est_rec = ReplayRecord(
            model=target,
            input_tokens=turn.input_tokens,
            cache_read=0,
            cache_write=0,
            output_tokens=max_tokens // 4,
            mode="",
            ts=turn.ts,
        )
        est = tiered_cost(est_rec, target) or 0.0
        if spend_b + spend_aa + est > spend_cap:
            log.info("replay-model: spend cap reached in B arm after %d turns", n_b_done)
            break
        try:
            resp_b = client.messages.create(
                model=target,
                max_tokens=max_tokens,
                messages=_cached_messages(turn),  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            log.warning("replay-model: B turn %d failed: %s", i, exc)
            continue
        usage = resp_b.usage
        actual = ReplayRecord(
            model=target,
            input_tokens=getattr(usage, "input_tokens", 0),
            cache_read=getattr(usage, "cache_read_input_tokens", 0),
            cache_write=getattr(usage, "cache_creation_input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            mode="",
            ts=turn.ts,
        )
        spend_b += tiered_cost(actual, target) or 0.0
        dec_b = decision_signature(resp_b.model_dump())
        b_decisions[i] = dec_b
        if dec_b == turn.recorded_decision:
            n_eq_b += 1
        n_b_done += 1

    # --- A' arm: subsample of turns ---
    n_aa_done = 0
    orig_model_counts: dict[str, int] = {}
    for i in sorted(aa_indices):
        if i >= len(turns_flat):
            break
        turn = turns_flat[i]
        orig_model_counts[turn.model] = orig_model_counts.get(turn.model, 0) + 1

        # Estimate cost to check cap (use original model and recorded token count)
        est_orig = ReplayRecord(
            model=turn.model,
            input_tokens=turn.input_tokens,
            cache_read=0,
            cache_write=0,
            output_tokens=max_tokens // 4,
            mode="",
            ts=turn.ts,
        )
        est_aa = original_cost(est_orig) or 0.0
        if spend_b + spend_aa + est_aa > spend_cap:
            log.info("replay-model: spend cap reached in A' arm after %d turns", n_aa_done)
            break
        try:
            resp_aa = client.messages.create(
                model=turn.model,
                max_tokens=max_tokens,
                messages=_cached_messages(turn),  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            log.warning("replay-model: A' turn %d failed: %s", i, exc)
            continue
        usage_aa = resp_aa.usage
        actual_aa = ReplayRecord(
            model=turn.model,
            input_tokens=getattr(usage_aa, "input_tokens", 0),
            cache_read=getattr(usage_aa, "cache_read_input_tokens", 0),
            cache_write=getattr(usage_aa, "cache_creation_input_tokens", 0),
            output_tokens=getattr(usage_aa, "output_tokens", 0),
            mode="",
            ts=turn.ts,
        )
        spend_aa += original_cost(actual_aa) or 0.0
        dec_aa = decision_signature(resp_aa.model_dump())
        aa_eq = int(dec_aa == turn.recorded_decision)
        if aa_eq:
            n_eq_aa += 1
        n_aa_done += 1

        # Paired diff: only for turns where both B and A' completed.
        if i in b_decisions:
            b_eq = int(b_decisions[i] == turn.recorded_decision)
            diff = b_eq - aa_eq
            diffs.append(diff)
            si = flat_turn_seg_idx.get(i, 0)
            if si < len(seg_diffs):
                seg_diffs[si].append(diff)

    return PairedLiveResult(
        n_b=n_b_done,
        n_aa=n_aa_done,
        n_eq_b=n_eq_b,
        n_eq_aa=n_eq_aa,
        diffs=diffs,
        spend_b_usd=spend_b,
        spend_aa_usd=spend_aa,
        target_model=target,
        original_model_counts=orig_model_counts,
        segment_diffs=[s for s in seg_diffs if s],
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def register(sub: Any) -> None:
    """``distil replay-model`` — one call from :mod:`distil.cli`."""
    p = sub.add_parser(
        "replay-model",
        help="does a cheaper model make the same next decision on your traffic? "
        "(dry-run: free; live: requires --spend-cap)",
        epilog=(
            "Uses a paired A/A'/B design (same as `distil audit`): B = target model, "
            "A' = original model replayed on the same turn. Paired diff mean(1{B==A} - "
            "1{A'==A}) with bootstrap CI and the referee's non-inferiority margin."
        ),
    )
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        metavar="MODEL",
        help=f"target model to evaluate (default: {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"number of turns for the B arm (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--n-aa",
        type=int,
        default=None,
        metavar="N",
        help=f"A' arm subsample size (default: {_min_aa_n()} or --n, whichever is smaller)",
    )
    p.add_argument(
        "--days",
        type=float,
        default=float(DEFAULT_DAYS),
        metavar="DAYS",
        help=f"look back this many days (default: {DEFAULT_DAYS})",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="run the live paired equivalence test (calls the provider; requires --spend-cap)",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
        help="load transcript turns, run structural validation and free count_tokens gate, "
        "then report the pass rate and updated cost estimate — no paid calls",
    )
    p.add_argument(
        "--spend-cap",
        type=float,
        metavar="USD",
        help="hard per-run spend ceiling in USD covering both arms (required for --live)",
    )
    p.add_argument(
        "--sessions",
        type=int,
        default=None,
        metavar="K",
        help="number of session segments to sample (enables caching; default: use flat --n turns)",
    )
    p.add_argument(
        "--turns-per-segment",
        type=int,
        default=DEFAULT_TURNS_PER_SEGMENT,
        metavar="M",
        help=f"contiguous turns per session segment (default: {DEFAULT_TURNS_PER_SEGMENT})",
    )
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        metavar="TOKENS",
        help="exclude turns whose reconstructed context exceeds this token count",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="machine-readable output",
    )
    p.set_defaults(func=cmd_replay_model)


def cmd_replay_model(args: Any) -> int:
    import sys

    if args.live and args.spend_cap is None:
        print(
            "distil replay-model: --live requires --spend-cap USD covering both arms "
            "(e.g. --spend-cap 15.00 for a $15 ceiling)",
            file=sys.stderr,
        )
        return 2

    since = time.time() - args.days * 86400
    rng = random.Random()
    n_aa = getattr(args, "n_aa", None)
    preflight_mode = getattr(args, "preflight", False)
    n_sessions = getattr(args, "sessions", None)
    turns_per_seg = getattr(args, "turns_per_segment", DEFAULT_TURNS_PER_SEGMENT)
    max_ctx_tok = getattr(args, "max_context_tokens", None)

    if args.live or preflight_mode:
        if n_sessions is not None:
            segs, coverage = load_transcript_segments(
                n_sessions,
                turns_per_seg,
                since,
                max_context_tokens=max_ctx_tok,
                rng=rng,
            )
            turns = [t for seg in segs for t in seg]
            if coverage.get("request_coverage_frac", 1.0) < 0.99:
                pct_req = 100.0 * coverage["request_coverage_frac"]
                pct_tok = 100.0 * coverage["token_coverage_frac"]
                print(
                    f"scope: ≤{max_ctx_tok:,} tokens covers {pct_req:.1f}% of requests "
                    f"and {pct_tok:.1f}% of token volume in the window",
                )
        else:
            segs = None
            turns = load_transcript_turns(args.n, since=since, rng=rng)
        if not turns:
            print(
                "distil replay-model: no usable transcript turns found in the last "
                f"{args.days:.0f} days under {_cc_root()}\n"
                "Run distil through Claude Code for a few sessions first.",
                file=sys.stderr,
            )
            return 1

        if preflight_mode:
            return _cmd_preflight(turns, args)

        result = run_live(
            turns,
            args.target,
            args.spend_cap,
            n_aa=n_aa,
            segments=segs,
        )
        if args.json:
            import dataclasses

            d = dataclasses.asdict(result)
            # Equivalence is a property, not a field; include it separately.
            eq = result.equivalence
            d["equivalence"] = dataclasses.asdict(eq)
            print(json.dumps(d, indent=2))
        else:
            print(result.format())
        return 0

    # Dry-run: sessions records only, no API calls.
    records = load_records(args.n, since=since, rng=rng)
    if not records:
        print(
            "distil replay-model: no usable session records found in the last "
            f"{args.days:.0f} days under {_home() / 'sessions'}\n"
            "Run distil wrap for a few sessions first, or pass --days N.",
            file=sys.stderr,
        )
        return 1
    report = dry_run(records, args.target, n_aa=n_aa, days=args.days)
    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print(report.format())
    return 0


def _cmd_preflight(turns: list[TranscriptTurn], args: Any) -> int:
    """Run the two free gates and report pass rates + updated cost estimate."""
    import sys

    n_aa = getattr(args, "n_aa", None)
    target = args.target
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Gate 1: structural validation.
    struct_ok: list[TranscriptTurn] = []
    struct_fail = 0
    for t in turns:
        errs = validate_turn(t)
        if errs:
            struct_fail += 1
            log.debug("struct fail: %s", errs)
        else:
            struct_ok.append(t)
    print(
        f"structural validation: {len(struct_ok)}/{len(turns)} pass ({struct_fail} fail)",
        file=sys.stderr,
    )

    if not struct_ok:
        print("all turns failed structural validation — nothing to preflight", file=sys.stderr)
        return 1

    # Gate 2: count_tokens (free).
    if not api_key:
        print(
            "ANTHROPIC_API_KEY not set — skipping count_tokens gate; "
            "set it via your environment to run the full preflight",
            file=sys.stderr,
        )
        ct_ok = struct_ok
        n_ct_fail = 0
    else:
        ct_ok, n_ct_pass, n_ct_fail = preflight_count_tokens(struct_ok, target, api_key=api_key)
        print(
            f"count_tokens preflight: {n_ct_pass}/{len(struct_ok)} pass ({n_ct_fail} fail)",
            file=sys.stderr,
        )

    n_valid = len(ct_ok)
    if not n_valid:
        print("0 turns passed both gates — cannot proceed to live", file=sys.stderr)
        return 1

    # Cost estimate based on valid turns only.
    # Convert TranscriptTurns to ReplayRecords (approximate — uses recorded token counts).
    valid_records = [
        ReplayRecord(
            model=t.model,
            input_tokens=t.input_tokens,
            cache_read=0,
            cache_write=0,
            output_tokens=t.output_tokens,
            mode="",
            ts=t.ts,
        )
        for t in ct_ok
    ]
    report = dry_run(valid_records, target, n_aa=n_aa, days=args.days)

    overall_pass = n_valid
    overall_total = len(turns)
    print(
        f"\npreflight summary: {overall_pass}/{overall_total} turns pass both gates "
        f"({100 * overall_pass // overall_total}%)\n",
        file=sys.stderr,
    )
    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print(report.format())
        print(
            f"\nupdated --live command (using {n_valid} validated turns):\n"
            f"  distil replay-model --live "
            f"--n {n_valid} --spend-cap {report.arm_b_usd + report.arm_aa_usd:.2f}",
        )
    return 0

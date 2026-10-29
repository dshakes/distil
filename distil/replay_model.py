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
from dataclasses import dataclass
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
    )


# --------------------------------------------------------------------------- #
# Transcript turns (live replay source)
# --------------------------------------------------------------------------- #


@dataclass
class TranscriptTurn:
    """One conversation turn: reconstructed messages context + the recorded decision."""

    messages: list[dict[str, Any]]  # sent TO the model (does not include this response)
    recorded_decision: str  # decision_signature of the original response
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
    from .shadow import decision_signature

    if since is None:
        since = time.time() - DEFAULT_DAYS * 86400
    root = transcripts_dir or _cc_root()
    rng = rng or random.Random()
    reservoir: list[TranscriptTurn] = []
    k = 0
    for jsonl_path in sorted(Path(root).glob("**/*.jsonl")):
        try:
            k = _sample_file(jsonl_path, since, reservoir, n, k, rng, decision_signature)
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
    decision_signature: Any,
) -> int:
    """Parse one transcript file; reservoir-sample turns into *reservoir*. Returns updated k."""
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

    by_uuid: dict[str, dict[str, Any]] = {str(r.get("uuid")): r for r in rows if r.get("uuid")}
    k = k_before
    for r in rows:
        if r.get("type") != "assistant":
            continue
        msg = r.get("message") or {}
        if not isinstance(msg, dict):
            continue
        ts = _epoch(r.get("timestamp"))
        if ts < since:
            continue
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        model = str(msg.get("model") or "")
        if not model:
            continue
        rec_dec = decision_signature({"content": content})
        if rec_dec == "none":
            continue
        messages = _build_messages(r, by_uuid)
        if not messages:
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
        if len(reservoir) < n:
            reservoir.append(turn)
        else:
            j = rng.randint(0, k - 1)
            if j < n:
                reservoir[j] = turn
    return k


def _build_messages(
    assistant_row: dict[str, Any],
    by_uuid: dict[str, dict[str, Any]],
    max_depth: int = 40,
) -> list[dict[str, Any]]:
    """Walk the parent chain to reconstruct the messages list for this turn."""
    chain: list[dict[str, Any]] = []
    cur = assistant_row
    for _ in range(max_depth):
        par_uuid = cur.get("parentUuid")
        if not par_uuid:
            break
        par = by_uuid.get(str(par_uuid))
        if par is None:
            break
        chain.append(par)
        cur = par
    if not chain:
        return []
    chain.reverse()
    messages: list[dict[str, Any]] = []
    for row in chain:
        msg = row.get("message") or {}
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        role = msg.get("role") or row.get("type")
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            api_content: Any = content
        elif isinstance(content, list):
            # Strip thinking blocks — cannot re-inject into a fresh call.
            filtered = [
                b
                for b in content
                if isinstance(b, dict) and b.get("type") not in ("thinking", "redacted_thinking")
            ]
            api_content = filtered or content
        else:
            continue
        if not api_content:
            continue
        messages.append({"role": role, "content": api_content})
    return messages


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
) -> PairedLiveResult:
    """Paired A/A'/B replay.

    Selects the first *n_aa* turns (or all if n_aa >= len(turns)) for the A' arm.
    All turns get a B call. B calls happen first (cheaper); A' calls only start if
    the cap allows. Stops early on either arm when the cap is exhausted.
    """
    import anthropic

    from .shadow import decision_signature

    n_aa_actual = _min_aa_n() if n_aa is None else n_aa
    n_aa_actual = min(n_aa_actual, len(turns))
    aa_indices: set[int] = set(range(n_aa_actual))  # first n_aa turns get A' call

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")

    n_eq_b = 0
    n_eq_aa = 0
    spend_b = 0.0
    spend_aa = 0.0
    n_b_done = 0
    diffs: list[int] = []
    # Record B decisions for the paired turns (we need them when computing A' diffs).
    b_decisions: dict[int, str] = {}

    # --- B arm: all turns ---
    for i, turn in enumerate(turns):
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
                messages=turn.messages,  # type: ignore[arg-type]
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
        if i >= len(turns):
            break
        turn = turns[i]
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
                messages=turn.messages,  # type: ignore[arg-type]
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
            diffs.append(b_eq - aa_eq)

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
        "--spend-cap",
        type=float,
        metavar="USD",
        help="hard per-run spend ceiling in USD covering both arms (required for --live)",
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

    if args.live:
        turns = load_transcript_turns(args.n, since=since, rng=rng)
        if not turns:
            print(
                "distil replay-model: no usable transcript turns found in the last "
                f"{args.days:.0f} days under {_cc_root()}\n"
                "Run distil through Claude Code for a few sessions first.",
                file=sys.stderr,
            )
            return 1
        result = run_live(turns, args.target, args.spend_cap, n_aa=n_aa)
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

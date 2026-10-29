"""``distil replay-model`` — does a cheaper model make the same next decision on YOUR traffic?

Dry-run (default): samples N recorded requests from ``~/.distil/sessions/*.requests.jsonl``
and prints the sample statistics plus the estimated cost of replaying them against a target
model. No API calls, no content read, no spend.

Live: reconstructs message context from Claude Code session transcripts
(``~/.claude/projects/**/*.jsonl``), calls the target model for each sampled turn, and
reports the decision-equivalence rate with a Wilson 95% CI and the $/request ratio.
Requires ``--spend-cap``; refuses to run without it.

Decision equivalence: ``decision_signature(replay_response) == recorded_decision``.
Both are ``tool:<name>:<args_hash>`` for tool calls, ``text`` for plain replies, ``none``
when the response cannot be parsed. Same signature as shadow mode; numbers are comparable.

Haiku 5.5 pricing: base $0.10/$0.50 per Mtok for prompts ≤100k total input tokens;
5× ($0.50/$2.50) when total input > 100k. See ``pricing.Pricing.long_tier``.
"""

from __future__ import annotations

import json
import logging
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
#: How many days of session history to look back by default.
DEFAULT_DAYS = 30
#: Threshold at which haiku-5-5 switches to its 5× long-prompt tier.
HAIKU_LONG_TIER_TOKENS = 100_000


# --------------------------------------------------------------------------- #
# Request record (content-free; from sessions/*.requests.jsonl)
# --------------------------------------------------------------------------- #


@dataclass
class ReplayRecord:
    """One proxied request, metadata only — no prompt or response content."""

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

    "Usable" means: booked (status 200), model present, at least one input token.
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
    if not isinstance(r, dict):
        return None
    if not r.get("booked"):
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
    """Estimated cost of replaying *rec* against *target*, applying long_tier if present.

    Returns None if *target* is unknown. Uses the RECORDED token breakdown (input,
    cache_read, cache_write, output) at the target model's rates. The long-tier
    multiplier applies to all input (uncached + cached), matching Anthropic's billing.
    """
    from . import pricing

    p = pricing.resolve(target)
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
    """Cost of the original request at the original model's list rate (for the ratio)."""
    from . import pricing

    p = pricing.resolve(rec.model)
    if p is None:
        return None
    m = _tier_mult(p, rec.total_input)
    return m * (
        rec.input_tokens * p.input
        + rec.cache_read * p.cache_read
        + rec.cache_write * p.cache_write
        + rec.output_tokens * p.output
    )


def _tier_mult(p: Any, total_input: int) -> float:
    """Long-tier multiplier: 1.0 unless pricing has a long_tier and we exceed it."""
    lt = getattr(p, "long_tier", None)
    if lt is not None and total_input > lt[0]:
        return float(lt[1])
    return 1.0


# --------------------------------------------------------------------------- #
# Dry-run: statistics without any API calls
# --------------------------------------------------------------------------- #


@dataclass
class DryRunReport:
    n_sampled: int
    target_model: str
    model_counts: dict[str, int]
    token_percentiles: dict[str, int]  # p50/p90/p95/max total_input
    n_long_tier: int  # records where total_input > HAIKU_LONG_TIER_TOKENS
    estimated_total_usd: float
    estimated_usd_per_request: float
    original_total_usd: float
    cost_ratio: float  # estimated_total / original_total (< 1 = cheaper)
    n_unpriced_target: int
    n_unpriced_original: int
    days: float

    def format(self) -> str:
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
            f"  target: {self.target_model}",
            f"  estimated cost:  ${self.estimated_total_usd:.4f} total  "
            f"(${self.estimated_usd_per_request:.6f}/request)",
            f"  original cost:   ${self.original_total_usd:.4f} total",
            f"  cost ratio:      {self.cost_ratio:.3f}x "
            f"({'cheaper' if self.cost_ratio < 1 else 'more expensive'})",
        ]
        if self.n_unpriced_target or self.n_unpriced_original:
            lines.append(
                f"  note: {self.n_unpriced_target} requests unpriced at target, "
                f"{self.n_unpriced_original} unpriced at original (excluded from ratio)"
            )
        lines += [
            "",
            "equivalence test (live path):",
            "  replays N turns against the target, reports the fraction where",
            "  target decision == original decision, with a Wilson 95% CI",
            "  (same estimator as `distil shadow-stats`)",
            "  to run live: distil replay-model --live --spend-cap <USD>",
        ]
        return "\n".join(lines)


def _fmt_counts(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    parts = [
        f"{m} {n} ({100 * n // total}%)" for m, n in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return ", ".join(parts[:4])


def dry_run(
    records: list[ReplayRecord],
    target: str,
    *,
    days: float = float(DEFAULT_DAYS),
) -> DryRunReport:
    model_counts: dict[str, int] = {}
    toks: list[int] = []
    n_long = n_unp_tgt = n_unp_orig = 0
    est_total = orig_total = 0.0

    for rec in records:
        model_counts[rec.model] = model_counts.get(rec.model, 0) + 1
        total_in = rec.total_input
        toks.append(total_in)
        if total_in > HAIKU_LONG_TIER_TOKENS:
            n_long += 1
        c_tgt = tiered_cost(rec, target)
        c_orig = original_cost(rec)
        if c_tgt is None:
            n_unp_tgt += 1
        else:
            est_total += c_tgt
        if c_orig is None:
            n_unp_orig += 1
        else:
            orig_total += c_orig

    toks.sort()
    n = len(toks)

    def pct(q: float) -> int:
        return toks[min(int(q * n), n - 1)] if n else 0

    ratio = est_total / orig_total if orig_total > 0 else 0.0
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
        estimated_total_usd=est_total,
        estimated_usd_per_request=est_total / n if n else 0.0,
        original_total_usd=orig_total,
        cost_ratio=ratio,
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
    """Reservoir-sample up to *n* turns from Claude Code session transcripts.

    A "turn" is one assistant response that produced a tool_use or text decision, with
    its reconstructed messages context. Content stays in memory; nothing is written.
    """
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
    """Parse one transcript file; reservoir-sample turns into *reservoir*.
    Returns the updated total-seen count (k)."""
    """Parse one transcript file and reservoir-sample turns into *reservoir*."""
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

        resp_obj = {"content": content}
        rec_dec = decision_signature(resp_obj)
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
    """Walk the parent chain to reconstruct the messages list for this turn.

    Returns messages sent TO the model (up to but NOT including this assistant response).
    Strips thinking blocks from prior assistant messages (same rule as deterministic_body).
    Returns [] if context cannot be reconstructed.
    """
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
            # Strip thinking blocks — cannot re-inject them into a fresh call.
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

        s = str(iso).rstrip("Z")
        dt = datetime.datetime.fromisoformat(s)
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except (ValueError, TypeError):
        return 0.0


# --------------------------------------------------------------------------- #
# Live replay
# --------------------------------------------------------------------------- #


@dataclass
class LiveResult:
    n: int
    n_equivalent: int
    spend_usd: float
    target_model: str
    cost_ratio: float  # target spend / original spend estimate
    equivalence_rate: float
    equivalence_ci: tuple[float, float]

    def format(self) -> str:
        from .shadow import wilson_ci

        lo, hi = wilson_ci(self.n_equivalent, self.n)
        lines = [
            f"live replay: {self.n} turns against {self.target_model}",
            f"  equivalent decisions: {self.n_equivalent}/{self.n} "
            f"({100 * self.equivalence_rate:.1f}%)",
            f"  Wilson 95% CI: [{100 * lo:.1f}%, {100 * hi:.1f}%]",
            f"  spend: ${self.spend_usd:.4f}  cost ratio: {self.cost_ratio:.3f}x",
        ]
        return "\n".join(lines)


def run_live(
    turns: list[TranscriptTurn],
    target: str,
    spend_cap: float,
    *,
    api_key: str | None = None,
    max_tokens: int = 4096,
) -> LiveResult:
    """Replay each turn against *target*; compare decisions; track spend.

    Stops early if accumulated cost would exceed *spend_cap*. Makes real API calls;
    requires a valid ``ANTHROPIC_API_KEY`` in the environment (or *api_key*).
    """
    import anthropic

    from .shadow import decision_signature

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")

    n_eq = 0
    spend = 0.0
    orig_spend = 0.0
    n_done = 0

    for turn in turns:
        # Reserve cost before calling to avoid overshooting the cap.
        est_rec = ReplayRecord(
            model=target,
            input_tokens=turn.input_tokens,
            cache_read=0,
            cache_write=0,
            output_tokens=max_tokens // 4,  # conservative
            mode="",
            ts=turn.ts,
        )
        est = tiered_cost(est_rec, target) or 0.0
        if spend + est > spend_cap:
            log.info("replay-model: spend cap reached at $%.4f after %d turns", spend, n_done)
            break

        try:
            resp = client.messages.create(
                model=target,
                max_tokens=max_tokens,
                messages=turn.messages,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 — individual turn failures are non-fatal
            log.warning("replay-model: turn %d failed: %s", n_done, exc)
            continue

        usage = resp.usage
        actual_rec = ReplayRecord(
            model=target,
            input_tokens=getattr(usage, "input_tokens", 0),
            cache_read=getattr(usage, "cache_read_input_tokens", 0),
            cache_write=getattr(usage, "cache_creation_input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            mode="",
            ts=turn.ts,
        )
        spend += tiered_cost(actual_rec, target) or 0.0

        orig_rec = ReplayRecord(
            model=turn.model,
            input_tokens=turn.input_tokens,
            cache_read=0,
            cache_write=0,
            output_tokens=turn.output_tokens,
            mode="",
            ts=turn.ts,
        )
        orig_spend += original_cost(orig_rec) or 0.0

        replay_dec = decision_signature(resp.model_dump())
        if replay_dec == turn.recorded_decision:
            n_eq += 1

        n_done += 1

    from .shadow import wilson_ci

    rate = n_eq / n_done if n_done else 0.0
    lo, hi = wilson_ci(n_eq, n_done)
    return LiveResult(
        n=n_done,
        n_equivalent=n_eq,
        spend_usd=spend,
        target_model=target,
        cost_ratio=spend / orig_spend if orig_spend > 0 else 0.0,
        equivalence_rate=rate,
        equivalence_ci=(lo, hi),
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
        help=f"number of recorded requests to sample (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--days",
        type=float,
        default=float(DEFAULT_DAYS),
        metavar="DAYS",
        help=f"look back this many days in the sessions history (default: {DEFAULT_DAYS})",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="run the live equivalence test (calls the provider; requires --spend-cap)",
    )
    p.add_argument(
        "--spend-cap",
        type=float,
        metavar="USD",
        help="hard per-run spend ceiling in USD (required for --live)",
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
            "distil replay-model: --live requires --spend-cap USD "
            "(e.g. --spend-cap 1.00 for a $1 ceiling)",
            file=sys.stderr,
        )
        return 2

    since = time.time() - args.days * 86400
    rng = random.Random()

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

        result = run_live(turns, args.target, args.spend_cap)
        if args.json:
            import dataclasses

            print(json.dumps(dataclasses.asdict(result), indent=2))
        else:
            print(result.format())
        return 0

    # Dry-run: sessions records only, no API calls.
    records = load_records(args.n, since=since, rng=rng)
    if not records:
        print(
            "distil replay-model: no usable session records found in the last "
            f"{args.days:.0f} days under {_home() / 'sessions'}\n"
            "Run distil wrap for a few sessions first, or pass --days N to look back further.",
            file=sys.stderr,
        )
        return 1

    report = dry_run(records, args.target, days=args.days)
    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print(report.format())
    return 0

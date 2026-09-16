"""Cross-session missed-savings advisor — ``distil discover``.

``distil dissect`` answers "what happened to *this* session?". This answers the
forward-looking question it cannot: "where am I still leaving savings on the
table, across my recent sessions, and what do I run to stop?"

Every number here is aggregated from the same on-disk sources ``dissect`` reads
(``savings.jsonl``, ``sessions/<sid>{.json,.requests.jsonl}``) through the same
:class:`~distil.dissect.Dissection` accessors — no new estimator, no new
instrumentation, nothing sent anywhere. Each detector is a small function that
returns one :class:`Action` or ``None``, and every Action carries the derivation
of its own number so a reader can reject it.

Two rules this module holds itself to:

* **Never quote a best case as a typical one.** The report prints the median and
  the p10/p90 of per-session savings *beside* the best session, because a
  headline number next to a much worse fleet median is the trust problem this
  command exists to be the opposite of.
* **Never invent a ratio.** Where an estimate needs "what would the other mode
  be worth", it uses the rate *this machine* actually measured
  (``ledger.mode_rates``); only when the machine has never run that mode does it
  fall back to a published benchmark figure, and then it says so on the line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import ledger as _ledger
from .dissect import Dissection, SessionOverview, _human, _read_jsonl, dissect, list_sessions

if TYPE_CHECKING:  # pragma: no cover — annotation only, avoids a module-level import
    from .prefix import CacheSummary

#: Fixed-overhead share at which trimming tool definitions beats compressing. Same
#: threshold ``dissect`` uses for its own "your fixed setup is N% of everything
#: sent" headline, so the two surfaces never disagree about when it matters.
OVERHEAD_SHARE_PCT = 30.0

#: Minimum tool definitions before "audit your tools" is actionable advice rather
#: than noise — with three tools there is nothing to triage.
MIN_TOOLS = 4

#: Prefix-drift ratio at which a broken cache prefix is worth reporting, and the
#: minimum comparable turns before a ratio means anything at all.
DRIFT_RATIO = 0.25
MIN_DRIFT_PAIRS = 4

#: Provider cache economics (pricing.Pricing defaults): a cache *write* bills at
#: 1.25x the base input rate and a *read* at 0.10x. A prefix that drifts pays the
#: difference on every re-billed token.
CACHE_WRITE_PREMIUM = 1.25 - 0.10

#: Above this share of billed input served from the provider's prompt cache, resent
#: content is already discounted and any dedup mechanism recovers far less than the
#: raw churn number implies. Same cut ``Dissection._churn_advice`` applies.
CACHED_SHARE_CEILING = 90.0

#: System-prompt growth worth acting on: both a relative and an absolute floor, so
#: a 40-token preamble that doubles does not outrank a real memory-file blowout.
SYSTEM_GROWTH_PCT = 25.0
SYSTEM_GROWTH_TOKENS = 300

#: Calibration ratio outside which this report's own token figures are rough. Same
#: band as ``Dissection.anomalies``.
CALIB_LOW, CALIB_HIGH = 0.67, 1.5

#: Sessions below this baseline are too small for a savings percentage to mean
#: anything; excluded from the typical/best spread. Mirrors dissect's peer cut.
MIN_BASELINE_TOKENS = 10_000

#: Fallback digest rate, used ONLY when this machine has never run digest mode and
#: so cannot supply its own. Source: BENCHMARKS.md, the messages-level codebench
#: harness (16 sessions / 256 turns of read -> edit -> re-read), "distil (PAYG
#: digest)" row: 91.5% token savings, 91.1% cache-aware dollars. It is a benchmark
#: corpus, not the reader's traffic, and every line that uses it says so.
BENCH_DIGEST_RATE = 0.915
BENCH_DIGEST_SOURCE = "BENCHMARKS.md codebench (16 sessions/256 turns), not your traffic"


@dataclass
class Action:
    """One ranked, quantified thing the user could do about their bill.

    ``tokens_per_week``/``dollars_per_week`` are the *recoverable* rate, not the
    spend: what taking the action would stop paying. ``dollars_per_week`` is
    ``None`` when the window carries no priced ledger rows — an unpriced action
    still ranks on tokens rather than silently claiming $0.00.
    """

    id: str
    kind: str  # "savings" — recovers tokens; "risk" — costs nothing, but you should know
    title: str
    tokens_per_week: int
    dollars_per_week: float | None
    basis: str  # how the number was derived, in one sentence
    command: str  # the one thing to run or change

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "tokens_per_week": self.tokens_per_week,
            "dollars_per_week": (
                None if self.dollars_per_week is None else round(self.dollars_per_week, 4)
            ),
            "basis": self.basis,
            "command": self.command,
        }


@dataclass
class Report:
    """The window, what it is typical of, and the ranked actions over it."""

    sessions: int = 0
    #: Sessions dropped from the window because they proxied nothing at all (a
    #: manifest with zero booked requests) — counted, never silently folded in.
    sessions_without_traffic: int = 0
    #: Sessions in this window that have real, priced ledger rows but predate the
    #: per-request detail file — real savings, but nothing for the detail-based
    #: detectors to read. Included in the typical/best spread, excluded from
    #: `actions`. Counted separately so "no findings" never hides "half the window
    #: could not be assessed".
    sessions_without_detail: int = 0
    #: Sessions the detail-based detectors actually read (``len(w.ds)``). Zero
    #: means no detector ran at all — distinct from "ran and found nothing" —
    #: so `render_text` never lets a ledger-only window read as an all-clear.
    detectors_assessed_sessions: int = 0
    requests: int = 0
    days: float = 0.0
    notional: bool = False  # any flat-rate session in the window -> dollars are notional
    calibrated: bool = False
    actions: list[Action] = field(default_factory=list)
    #: Per-session savings percentages, for the typical-vs-best spread.
    pcts: list[float] = field(default_factory=list)
    best: tuple[str, float] | None = None

    @property
    def median_pct(self) -> float | None:
        return _quantile(self.pcts, 0.5)

    @property
    def p10_pct(self) -> float | None:
        return _quantile(self.pcts, 0.1)

    @property
    def p90_pct(self) -> float | None:
        return _quantile(self.pcts, 0.9)

    @property
    def tokens_per_week(self) -> int:
        return sum(a.tokens_per_week for a in self.actions if a.kind == "savings")

    @property
    def assessed(self) -> bool:
        """False when the window has no booked traffic to assess — distinct from
        "assessed and found nothing wrong". True as soon as any session has
        priced ledger rows, detail or not."""
        return self.sessions > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": {
                "sessions": self.sessions,
                "sessions_without_traffic": self.sessions_without_traffic,
                "sessions_without_detail": self.sessions_without_detail,
                "detectors_assessed_sessions": self.detectors_assessed_sessions,
                "requests": self.requests,
                "days": round(self.days, 2),
                "notional_dollars": self.notional,
                "calibrated": self.calibrated,
                "assessed": self.assessed,
            },
            "typical": {
                "median_pct_saved": _round(self.median_pct),
                "p10_pct_saved": _round(self.p10_pct),
                "p90_pct_saved": _round(self.p90_pct),
                "best_session": self.best[0] if self.best else None,
                "best_pct_saved": _round(self.best[1]) if self.best else None,
                "sessions_scored": len(self.pcts),
            },
            "tokens_per_week": self.tokens_per_week,
            "actions": [a.to_dict() for a in self.actions],
        }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def _quantile(values: list[float], q: float) -> float | None:
    """Nearest-rank quantile. Returns None for an empty sample rather than 0.0 —
    "no sessions big enough to score" and "you save nothing" are opposite readings."""
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[idx]


@dataclass
class _Window:
    """Everything the detectors share: the dissections plus the window-wide totals
    each of them would otherwise recompute.

    ``ds`` carries only sessions with per-request detail — every detector reads
    it, and every one of them is detail-based (tool costs, prefix pairs, churn,
    system-prompt curve). ``ledger_only`` carries sessions with real, priced
    ledger rows but no detail file (pre-detail-format sessions): real savings,
    counted in the typical/best spread, but nothing a detector can read.
    """

    ds: list[Dissection]
    days: float
    usd_per_token: float | None
    since: float
    ledger_only: list[Dissection] = field(default_factory=list)
    sessions_without_traffic: int = 0

    @property
    def requests(self) -> int:
        return sum(len(d.booked_detail) for d in self.ds)

    def per_week(self, total: float) -> int:
        return int(round(total * 7.0 / self.days)) if self.days > 0 else 0

    def usd_per_week(self, tokens: float, *, rate_mult: float = 1.0) -> float | None:
        if self.usd_per_token is None:
            return None
        return self.per_week(tokens) * self.usd_per_token * rate_mult


def _collect(sessions: int, since_days: float | None) -> _Window:
    """Dissect the most recent *sessions* (optionally only those within
    *since_days*), reading the shared ledger exactly once.

    ``dissect()`` filters the whole ledger per session and joins shadow rows; on a
    ledger with tens of thousands of runs, doing that 20 times over is seconds of
    re-parsing for data no detector here reads. Both are passed/declined explicitly
    rather than duplicated — one implementation, still.
    """
    now = time.time()
    since = now - since_days * 86400 if since_days else 0.0
    rows = _read_jsonl(_ledger.default_path())
    overviews: list[SessionOverview] = [
        o for o in list_sessions(rows=rows, with_status=False) if not since or o.last_ts >= since
    ][:sessions]
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for rec in rows:
        sid = rec.get("session")
        if isinstance(sid, str) and sid:
            by_sid.setdefault(sid, []).append(rec)

    all_ds = [dissect(o.sid, ledger_rows=by_sid.get(o.sid, []), shadow=False) for o in overviews]
    # Three states, not two. (a) A `wrap` that started and exited without proxying
    # a single request (killed before the agent made a call, or the agent never
    # called out) has neither a ledger row nor detail — nothing here to assess,
    # and folding it into the window silently would let an all-quiet window read
    # as "all within range" rather than "nothing was observed". (b) An older
    # session, priced before the per-request detail file existed, has real
    # ledger rows but no detail — real savings, just nothing a detail-based
    # detector can read. (c) both present: the full picture.
    ds = [d for d in all_ds if d.booked_detail]
    ledger_only = [d for d in all_ds if not d.booked_detail and d.ledger_rows]
    no_traffic = len(all_ds) - len(ds) - len(ledger_only)
    starts = [d.started for d in ds + ledger_only if d.started]
    # Elapsed wall-clock since the oldest session in the window, floored at a day: a
    # rate extrapolated from a few hours would read as a week's worth of savings.
    days = max(1.0, (now - min(starts)) / 86400) if starts else 0.0

    base_tok = sum(int(r.get("baseline_input_tokens") or 0) for d in ds for r in d.ledger_rows)
    base_usd = sum(float(r.get("baseline_dollars") or 0.0) for d in ds for r in d.ledger_rows)
    # One price for every dollar on this report: the blended rate the ledger itself
    # recorded (priced through distil.pricing at record time), over the same
    # heuristic tokens the detectors count. Mixing a catalog lookup in here would
    # price some actions per the model and some per the ledger.
    usd = (base_usd / base_tok) if base_tok and base_usd else None
    return _Window(
        ds=ds,
        days=days,
        usd_per_token=usd,
        since=since,
        ledger_only=ledger_only,
        sessions_without_traffic=no_traffic,
    )


# --------------------------------------------------------------------------- detectors
# Each returns one Action or None. None means "the data does not support this
# recommendation", never "probably fine" — a detector that cannot measure stays
# quiet rather than guessing.


def _d_tool_overhead(w: _Window) -> Action | None:
    """Tool/MCP definitions are resent verbatim on every request and no compression
    touches them, so on a tool-heavy agent they are the cheapest saving available."""
    overhead = sum(d.overhead_tokens_total for d in w.ds)
    # Same denominator `Dissection.overhead_share` uses, reused rather than
    # re-derived: if tokens_saved_total ever exceeds compressible_tokens for a
    # request, re-deriving it here without the same floor would send `sent`
    # negative and the two surfaces would disagree about what "overhead" means.
    sent = sum(d.sent_tokens_total for d in w.ds)
    if not sent or 100.0 * overhead / sent < OVERHEAD_SHARE_PCT:
        return None
    per: dict[str, int] = {}
    for d in w.ds:
        for name, _per_req, total in d.tool_costs(booked_only=True):
            per[name] = per.get(name, 0) + total
    if len(per) < MIN_TOOLS:
        return None
    top = sorted(per.items(), key=lambda kv: -kv[1])[:3]
    tokens = sum(t for _n, t in top)
    if not tokens:
        return None
    names = ", ".join(n for n, _t in top)
    return Action(
        id="tool_overhead",
        kind="savings",
        title=(
            f"Your fixed setup is {100.0 * overhead / sent:.0f}% of everything sent — "
            f"the 3 costliest of {len(per)} tool definitions are {names}"
        ),
        tokens_per_week=w.per_week(tokens),
        dollars_per_week=w.usd_per_week(tokens),
        basis=(
            f"each tool definition's size x the requests that carried it, summed over "
            f"{len(w.ds)} session(s); the figure is what dropping those three would stop "
            f"resending. distil cannot see which tools your agent never CALLED — "
            f"`distil dissect <session> --transcript` joins the agent's own log and names them"
        ),
        command="disable the MCP servers / tools you do not use in your agent's config",
    )


def _d_digest_off(w: _Window) -> Action | None:
    """Sessions that never reached the digest tier (lossless-only or verbatim).

    This also covers the "expand disabled while recoverable-eligible tokens are
    high" case, and deliberately is not a second detector: distil only reaches the
    digest tier *with* a recovery tool (``proxy.build_handler`` forces verbatim when
    neither lossy mode nor ``--expand`` is available), so "digest off" and "expand
    off" are the same session, and two lines recommending one flag is noise.
    """
    off = [d for d in w.ds if _mode_of(d) in ("lossless-only", "verbatim")]
    if not off:
        return None
    base = sum(d.baseline_tokens for d in off)
    already = sum(d.baseline_tokens - d.distil_tokens for d in off)
    if not base:
        return None
    rate, source = _digest_rate(w)
    tokens = base * rate - already
    if tokens <= 0:
        return None
    subscription = any(d.billing == "subscription" for d in off)
    note = (
        " On a flat-rate plan this is an informed opt-in: it injects distil_expand, so "
        "nothing is irreversibly lost, but the request IS modified, which the "
        "subscription-safe default never does."
        if subscription
        else ""
    )
    return Action(
        id="digest_off",
        kind="savings",
        title=(
            f"{len(off)} of {len(w.ds)} sessions never reached the digest tier "
            f"(lossless-only/verbatim) and saved {100.0 * already / base:.1f}%"
        ),
        tokens_per_week=w.per_week(tokens),
        dollars_per_week=w.usd_per_week(tokens),
        basis=(
            f"{_human(base)} baseline tokens in those sessions x a {rate * 100:.1f}% digest "
            f"rate ({source}), minus the {_human(already)} they already saved"
        ),
        command="distil default --mode expand" + note,
    )


def _mode_of(d: Dissection) -> str:
    """The compression mode a session actually ran in: the manifest flags when the
    manifest actually recorded them, else the mode the ledger rows were booked
    under (pre-manifest sessions, or an older/minimal manifest that has no `flags`
    key at all). A manifest present but silent on `flags` is not evidence of
    "digest" — it is no evidence at all, and defaulting to digest there is exactly
    what let a lossless/verbatim legacy session dodge `digest_off`. "unknown" when
    neither source can tell, so a caller skips the session instead of guessing."""
    manifest = d.manifest or {}
    if "flags" in manifest:
        flags = manifest["flags"] or {}
        if flags.get("verbatim"):
            return "verbatim"
        if flags.get("lossless_only"):
            return "lossless-only"
        return "digest"
    modes = {str(r.get("mode") or "") for r in d.ledger_rows}
    for candidate in ("digest", "lossless-only", "verbatim"):
        if candidate in modes:
            return candidate
    return "unknown"


def _digest_rate(w: _Window) -> tuple[float, str]:
    """What digest mode is worth, preferring the rate THIS machine measured.

    A lifetime rate answers a different question than a recent one — what digest
    yields depends on the content mix, and that drifts — so the window's own rate
    wins, a lifetime rate is labelled as history, and the published benchmark is
    only reached by a machine that has never run digest at all.
    """
    scope = " in this window" if w.since else ""
    windowed = _ledger.mode_rates(since=w.since or None).get("digest")
    if windowed and windowed[0] >= 50:
        return windowed[1], f"your own traffic, {windowed[0]:,} runs{scope}"
    lifetime = _ledger.mode_rates().get("digest")
    if lifetime and lifetime[0] >= 50:
        return lifetime[1], f"your own traffic, {lifetime[0]:,} runs lifetime — not this window"
    return BENCH_DIGEST_RATE, BENCH_DIGEST_SOURCE


def _prefix_summary(w: _Window) -> CacheSummary:
    """One :class:`~distil.prefix.CacheSummary`, folded from a per-session summary
    each — never one list of every session's requests flattened and re-sorted by
    timestamp, which would compare the last request of one session against the
    first of the next and count the session boundary itself as "drift"."""
    from . import prefix as _prefix

    total = _prefix.CacheSummary()
    for d in w.ds:
        records = sorted(d.booked_detail, key=lambda r: float(r.get("ts") or 0))
        s = _prefix.summarise(records)
        total.requests += s.requests
        total.read_tokens += s.read_tokens
        total.create_tokens += s.create_tokens
        total.uncached_tokens += s.uncached_tokens
        total.drifts += s.drifts
        total.pairs += s.pairs
        total.drift_create_tokens += s.drift_create_tokens
        total.reported = total.reported or s.reported
        total.legacy_cache_tokens += s.legacy_cache_tokens
        total.legacy_rows += s.legacy_rows
    return total


def _d_prefix_drift(w: _Window) -> Action | None:
    """A prefix that changes between turns re-bills the whole cached span.

    Priced off the provider's own usage fields rather than our estimate: cache
    *creation* tokens are the ones that were re-billed, and the drift ratio is the
    share of turns that caused it. ``drift_create_tokens`` is the sum of cache-write
    tokens on the rows that actually drifted — not ``create_tokens * drift_ratio``,
    which would price every write at the average rate even when the drifting turns
    and the stable turns carry very different prefix sizes.
    """
    s = _prefix_summary(w)
    replay_off = any(
        ((d.manifest or {}).get("flags") or {}).get("prefix_replay") is False for d in w.ds
    )
    if s.pairs < MIN_DRIFT_PAIRS or (s.drift_ratio < DRIFT_RATIO and not replay_off):
        return None
    tokens = s.drift_create_tokens
    if tokens <= 0:
        return None
    why = (
        "prefix replay was switched off for at least one session; "
        if replay_off
        else "prefix replay is on by default; "
    )
    return Action(
        id="prefix_drift",
        kind="savings",
        title=(
            f"{s.drifts:,} of {s.pairs:,} turns changed the cached prefix "
            f"({s.drift_ratio:.0%}) — each one re-bills the whole span"
        ),
        tokens_per_week=w.per_week(tokens),
        dollars_per_week=w.usd_per_week(tokens, rate_mult=CACHE_WRITE_PREMIUM),
        basis=(
            f"{_human(s.drift_create_tokens)} tokens the provider billed as cache WRITES "
            f"on the {s.drifts:,} turn(s) whose stable-prefix hash actually changed "
            f"(of {_human(s.create_tokens)} written in total, {s.drift_ratio:.0%} of "
            f"{s.pairs:,} comparable turns); priced at the {CACHE_WRITE_PREMIUM:.2f}x gap "
            f"between a cache write (1.25x input) and the cache read (0.10x) it could "
            f"have been"
        ),
        command=(
            why + "run `distil cache` to see where the prefix broke, then "
            "https://dshakes.github.io/distil/cache-contract.html"
        ),
    )


def _d_churn(w: _Window) -> Action | None:
    """Content the client resends, re-folded every time — but only counted where it
    is actually being paid for.

    A well-cached session already has this content billed at the cache-read rate,
    so recommending a dedup mechanism against it sends the user after a saving an
    order of magnitude smaller than the churn number implies. Sessions whose cache
    share was never measured are excluded rather than assumed cheap OR expensive.
    """
    paying = [
        d
        for d in w.ds
        if d.cached_input_share is not None and d.cached_input_share < CACHED_SHARE_CEILING
    ]
    tokens = sum(d.churn_tokens for d in paying)
    saved = sum(d.tokens_saved_total for d in paying)
    if not tokens or not saved or tokens < 0.25 * saved:
        return None
    blocks = sum(d.churned_blocks for d in paying)
    share = sum(d.cached_input_share or 0.0 for d in paying) / len(paying)
    return Action(
        id="churn",
        kind="savings",
        title=(
            f"{_human(tokens)} tokens were resent and re-summarized across "
            f"{blocks} block(s) the client kept sending again"
        ),
        tokens_per_week=w.per_week(tokens),
        dollars_per_week=w.usd_per_week(tokens),
        basis=(
            f"each block's size x (folds - 1), over the {len(paying)} session(s) whose "
            f"measured cache-read share was under {CACHED_SHARE_CEILING:.0f}% "
            f"(mean {share:.0f}%) — where the provider is NOT already discounting the "
            f"resend. Sessions with no cache measurement are excluded, not assumed"
        ),
        command=(
            "distil wrap --session-delta -- <agent>  (cache-delta absorbs exactly this; "
            "it recovers much less on a well-cached session, which is why the number "
            "above excludes those)"
        ),
    )


def _d_system_growth(w: _Window) -> Action | None:
    """Memory files and context injections grow the system prompt, and every token
    of that growth is resent on every remaining request of the session."""
    grown = 0.0
    sessions = 0
    first_t = last_t = 0
    for d in w.ds:
        g = d.system_growth(booked_only=True)
        if not g:
            continue
        first, last = g
        delta = last - first
        if delta < SYSTEM_GROWTH_TOKENS or not first or 100.0 * delta / first < SYSTEM_GROWTH_PCT:
            continue
        sessions += 1
        first_t, last_t = first, last
        # The grown tokens are paid on every request after the growth. Charging half
        # the session's requests is the midpoint of "grew immediately" and "grew at
        # the end" — stated, because no record says WHEN it grew.
        grown += delta * len(d.booked_detail) / 2.0
    if not sessions or grown <= 0:
        return None
    return Action(
        id="system_growth",
        kind="savings",
        title=(
            f"The system prompt grew during {sessions} session(s) "
            f"(last one: {_human(first_t)} -> {_human(last_t)} tokens)"
        ),
        tokens_per_week=w.per_week(grown),
        dollars_per_week=w.usd_per_week(grown),
        basis=(
            "growth (last system prompt - first) x half the session's requests — the "
            "midpoint, since no record says at which request it grew. Nothing compresses "
            "the system prompt; it is resent verbatim every turn"
        ),
        command=(
            "trim the memory/context files your agent injects (CLAUDE.md, AGENTS.md, "
            "auto-loaded rules); `distil dissect <session>` shows the per-session curve"
        ),
    )


def _d_calibration(w: _Window) -> Action | None:
    """When distil's own token estimate and the provider's billed usage disagree by
    more than half, every percentage on this report is rough — including this one's."""
    est = billed = 0
    for d in w.ds:
        cal = d.calibration()
        if cal is None:
            continue
        est += cal[0]
        billed += cal[1]
    if not billed or not est:
        return None
    ratio = est / billed
    if CALIB_LOW <= ratio <= CALIB_HIGH:
        return None
    from . import calibration as _calib

    _f, n = _calib.factor()
    return Action(
        id="calibration",
        kind="risk",
        title=(
            f"Token estimates are off by >50% vs billed usage "
            f"({_human(est)} est vs {_human(billed)} billed, x{ratio:.2f})"
        ),
        tokens_per_week=0,
        dollars_per_week=None,
        basis=(
            f"calibrated estimate vs the API's own usage fields, summed over the window. "
            f"The correction is learned automatically from billed usage and needs "
            f"{_calib.MIN_SAMPLES} samples ({n} so far)"
        ),
        command=(
            "nothing to run — treat the percentages above as rough until calibration "
            "converges; `distil doctor` reports the sample count"
        ),
    )


DETECTORS = (
    _d_tool_overhead,
    _d_digest_off,
    _d_prefix_drift,
    _d_churn,
    _d_system_growth,
    _d_calibration,
)


def scan(*, sessions: int = 20, since_days: float | None = None) -> Report:
    """Aggregate recent sessions and rank what could still be recovered."""
    w = _collect(sessions, since_days)
    if not w.ds and not w.ledger_only:
        return Report(sessions_without_traffic=w.sessions_without_traffic)
    # The typical/best spread only needs ledger totals (pct_saved, baseline_tokens),
    # so a detail-less session still has something to say there — the detectors
    # below are the part that needs per-request rows, and only ``w.ds`` has those.
    scoreable = w.ds + w.ledger_only
    scored = [(d.sid, d.pct_saved) for d in scoreable if d.baseline_tokens >= MIN_BASELINE_TOKENS]
    actions = [a for a in (fn(w) for fn in DETECTORS) if a is not None]
    # Savings first, biggest first; risks last (they recover nothing, but a report
    # that buries "your numbers may be wrong" under them is worse than one that does not).
    actions.sort(key=lambda a: (a.kind != "savings", -a.tokens_per_week))
    from .calibration import MIN_SAMPLES, factor

    _f, n = factor()
    return Report(
        sessions=len(scoreable),
        sessions_without_traffic=w.sessions_without_traffic,
        sessions_without_detail=len(w.ledger_only),
        detectors_assessed_sessions=len(w.ds),
        requests=w.requests,
        days=w.days,
        notional=any(d.billing == "subscription" for d in scoreable),
        calibrated=n >= MIN_SAMPLES,
        actions=actions,
        pcts=[p for _s, p in scored],
        best=max(scored, key=lambda t: t[1]) if scored else None,
    )


# --------------------------------------------------------------------------- render
EMPTY = (
    "no wrap sessions recorded yet — nothing to discover.\n"
    "  `distil discover` reads the per-session request ledger, which only a wrap "
    "session writes.\n"
    "  Run your agent under `distil wrap -- <cmd>` (or `distil default --always-on`), "
    "then re-run."
)


def render_text(r: Report, *, color: bool = True) -> str:
    def c(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    if not r.sessions:
        if r.sessions_without_traffic:
            # Sessions exist (manifests were written) but not one proxied a
            # request — an empty window, not a clean one. Must never fall through
            # to "nothing to recommend", which reads as a successful all-clear.
            return (
                f"no proxied traffic in the last {r.sessions_without_traffic} "
                "session(s) — nothing to assess yet (run your agent through "
                "`distil wrap` first)."
            )
        return EMPTY
    out = [c("1", "distil discover — where your remaining savings are")]
    out.append(
        f"  window   {r.sessions} session(s), {r.requests:,} requests, "
        f"{r.days:.1f} days  (rates below are per week)"
    )
    med, p10, p90 = r.median_pct, r.p10_pct, r.p90_pct
    if med is None:
        out.append(
            "  typical  no session in the window is large enough to score "
            f"(need {MIN_BASELINE_TOKENS:,}+ baseline tokens)"
        )
    else:
        best = f", best {r.best[1]:.1f}% ({r.best[0]})" if r.best else ""
        out.append(
            f"  typical  {med:.1f}% saved on the median session "
            f"(p10 {p10:.1f}%, p90 {p90:.1f}%){best}"
        )
        out.append(
            c("2", "           the median is what to expect; the best session is not typical")
        )
    if not r.calibrated:
        out.append(c("2", "  note     token counts are uncalibrated estimates (see distil doctor)"))
    if r.sessions_without_detail:
        out.append(
            c(
                "2",
                f"  note     {r.sessions_without_detail} older session(s) lack per-request "
                "detail — savings counted, actions not assessed for them",
            )
        )

    out.append("")
    if not r.actions:
        if r.detectors_assessed_sessions:
            out.append(c("1", "nothing to recommend"))
            out.append(
                "  No detector fired on this window: your fixed overhead, cache prefix, "
                "re-fold churn\n  and system prompt are all within range. That is the "
                "result, not a failure to look."
            )
        else:
            # No detector ran at all — every session in the window is ledger-only.
            # Falling through to "nothing to recommend" here would claim an
            # all-clear for a window nothing actually checked.
            out.append(c("1", "no actions assessed"))
            out.append(
                f"  actions need per-request detail; none of these {r.sessions_without_detail} "
                "session(s) carries it\n  (written by distil >= 1.15)."
            )
        return "\n".join(out)

    money = ""
    if r.tokens_per_week:
        money = f" — up to {_human(r.tokens_per_week)} tokens/week"
    out.append(c("1", f"{len(r.actions)} action(s){money}"))
    notional = " notional" if r.notional else ""
    for i, a in enumerate(r.actions, 1):
        rate = f"{_human(a.tokens_per_week)} tokens/week" if a.kind == "savings" else "no $ effect"
        if a.kind == "savings" and a.dollars_per_week is not None:
            rate += f", ${a.dollars_per_week:,.2f}/week{notional}"
        out.append("")
        out.append(f"  {i}. {a.title}")
        out.append(c("32" if a.kind == "savings" else "33", f"     recovers  {rate}"))
        out.append(c("2", f"     how       {a.basis}"))
        out.append(f"     do        {a.command}")
    out.append("")
    if r.notional:
        out.append(c("2", "dollars are notional on a flat-rate plan — the bill does not move; the"))
        out.append(c("2", "context budget does (the same window goes further)."))
    out.append(c("2", "sources: savings.jsonl, sessions/<sid>{.json,.requests.jsonl} — the same"))
    out.append(c("2", "content-free records `distil dissect` reads. Nothing leaves this machine."))
    return "\n".join(out)


def wrap_exit_line(*, sessions: int = 8) -> str | None:
    """One line for the wrap-exit proof ledger, or None when there is nothing to say.

    Deliberately a smaller window than the command's default: this runs on every
    session exit, and it must not turn a clean exit into a second of ledger parsing.
    Fail-open in its own right (not just via its caller): this must never be the
    reason a wrap exit raises, and it must never print "0 actions" as a finding.
    """
    try:
        r = scan(sessions=sessions)
    except Exception:  # noqa: BLE001 — an advisory must never break a wrap exit
        return None
    if not r.actions:
        return None
    n = len(r.actions)
    if r.tokens_per_week:
        return (
            f"distil discover: {n} action{'s' if n != 1 else ''} could recover "
            f"~{_human(r.tokens_per_week)} tokens/week"
        )
    return f"distil discover: {n} thing{'s' if n != 1 else ''} worth a look"

"""``abtest_summary`` and ``distil ab`` — the causal, model-aware answer.

Headline. Mean list-price cost per randomised session, distil ÷ holdout, over every
randomised session in the current era that has ended — failed ones at their actual cost
(usually $0), none dropped for being cheap. Interval: :func:`stats.robust_ratio`, an
empirical-Bernstein betting sequence per arm on cost capped at :data:`stats.COST_CAP`,
anytime-valid within the current era with no asymptotics. It is the ONE primary claim,
at α = ``conformal.BUDGET_DELTA``; everything else is secondary and not
multiplicity-adjusted: the efficient estimate (strata-pooled, CUPED, normal-mixture,
asymptotic), the mediators (turns/tasks per session, cost per task/turn — outcomes
distil can change), the per-cell table and the rollout DiD. A sliding ``--window`` is a
fixed-n look and says so.

Cells. A session's stratum is ``(model, client)``: the model of its agent loop, fixed
before treatment acts (:func:`outcomes.stratum_model`), and the client at major.minor.
Its GROUP is the same pair without versions; a newer version SUPERSEDES the older one,
so the headline compares within the new model only and the rollout is reported as a
difference-in-differences. Inside a stratum, eras are cut at a holdout-rate change and
at a CUSUM alarm on the HOLDOUT arm's log cost per session (a silent change behind an
unchanged id) — the one series distil cannot move.

Scope. Every number is about this machine's own sessions. Nothing here is pooled across
installs; a cross-install estimate would have to cluster by install.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import DISTIL, HOLDOUT, disclosure, holdout_rate
from . import stats as st
from .outcomes import SessionOutcome, harvest, load


@dataclass
class Effect:
    """One contrast, ready to print. Relative values: −0.2 = distil 20% lower."""

    rel: float
    lo: float
    hi: float
    evalue: float | None  # None for the robust headline (a CS, not an e-process summary)
    significant: bool
    n_distil: int
    n_holdout: int
    variance_reduction: float = 0.0  # CUPED; 0 when not applied
    method: str = "mixture-SPRT (asymptotic)"

    @classmethod
    def robust(cls, r: st.RobustRatio | None) -> Effect | None:
        if r is None:
            return None
        return cls(
            rel=r.rel,
            lo=r.lo,
            hi=r.hi,
            evalue=None,
            significant=r.significant,
            n_distil=r.n1,
            n_holdout=r.n0,
            method=f"empirical-Bernstein CS, cost capped at ${r.cap:g}/session",
        )

    @classmethod
    def of(cls, c: st.Contrast | None, alpha: float) -> Effect | None:
        if c is None:
            return None
        lo, hi = c.interval(alpha)
        le = c.log_evalue()
        return cls(
            rel=c.rel,
            lo=lo,
            hi=hi,
            evalue=math.exp(min(le, 700.0)),
            significant=le >= math.log(1.0 / alpha),
            n_distil=c.n1,
            n_holdout=c.n0,
            variance_reduction=c.variance_reduction,
        )


@dataclass
class Cell:
    model: str
    client: str
    group: str
    era: int
    reason: str  # why this era began
    start: float
    end: float
    n_distil: int
    n_holdout: int
    cost_per_session: dict[str, float | None]  # arm -> mean $
    cost: Effect | None  # per session, asymptotic (per-cell; the headline pools robustly)
    turns: Effect | None  # turns per session (mediator)
    cost_per_task: Effect | None  # mediator
    current: bool  # contributes to the headline


@dataclass
class Event:
    ts: float
    group: str
    kind: str  # "model" | "shift" | "rate"
    text: str


@dataclass
class DiD:
    group: str
    before: str
    after: str
    effect: Effect


@dataclass
class ABSummary:
    status: str  # "ok" | "insufficient" | "no-data" | "disabled"
    message: str  # the one line a savings screen prints
    holdout_rate: float
    disclosure: str
    window_days: float | None
    alpha: float
    n_sessions: int = 0
    n_distil: int = 0
    n_holdout: int = 0
    n_open: int = 0
    n_unpriced: int = 0
    n_expanded: int = 0  # distil-arm requests with an expand re-query (cost caveat)
    notional: bool = False  # any subscription session: $ are list-price, not billed
    need_holdout: int | None = None
    need_sessions: int | None = None  # need_holdout at the current rate, all arms
    cap: float = st.COST_CAP
    cost: Effect | None = None  # THE headline: mean cost per session, robust CS
    efficient: Effect | None = None  # same estimand, strata-pooled + CUPED, asymptotic
    mediators: dict[str, Effect | None] = field(default_factory=dict)
    bootstrap: tuple[float, float] | None = None
    bootstrap_disagrees: bool = False  # verdicts differ: said so in the output
    balance: dict[str, dict[str, int]] = field(default_factory=dict)  # arm -> counts
    holdout_spend: float = 0.0
    holdout_cost: tuple[float, float, float] | None = None  # extra $ (est, lo, hi)
    model_change: str | None = None
    cells: list[Cell] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    did: list[DiD] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Strata, groups, eras
# --------------------------------------------------------------------------- #


def family(model: str) -> str:
    """``claude-opus-4-8-20260101`` → ``claude-opus``; ``gpt-4o-2024-08-06`` → ``gpt-4o``.

    ponytail: drops purely numeric tokens, so ``gemini-2.5-pro`` stays its own family
    and a 2.5 → 3 move reads as a new group, not a version bump. A family table
    would fix it; nobody routes Gemini through a wrap A/B yet.
    """
    m = model.strip()
    if m.startswith("anthropic."):
        m = m[len("anthropic.") :]
    m = m.split("@", 1)[0]
    return "-".join(t for t in m.split("-") if not t.isdigit()) or m


def _client_family(client: str) -> str:
    return client.split("/", 1)[0]


def _log_cost(o: SessionOutcome) -> float | None:
    return math.log(o.cost) if o.cost and o.cost > 0 else None


#: CUSUM threshold for the holdout-arm detector (see split_eras). 12, not 8: at a 5%
#: holdout every false era throws away scarce data, and h=8 false-alarmed on 12% of
#: 1,000-point stationary series (h=12: 0%). A missed shift costs only a regime average.
HOLDOUT_H = 12.0


def split_eras(sessions: list[SessionOutcome]) -> list[tuple[list[SessionOutcome], str, str]]:
    """One stratum's sessions (time-ordered) → eras ``(members, kind, reason)``.

    A rate change opens an era at the first session drawn under the new rate. Within
    a rate segment, a CUSUM alarm on log cost per task opens an era at the first session
    AFTER the alarm; sessions between the estimated change and the alarm are a
    transition and belong to no era — they are the ones the detector selected for being
    extreme (see :func:`stats.cusum_changepoints`).

    The detector reads the HOLDOUT arm only, at ``h = 12``: distil cannot move that
    series, so the detector never mistakes a distil upgrade for a model change and its
    own selection never lands on the treated arm. Measured at the shipped 5% holdout
    (N=4,000, log-sd 1.0, 100 runs; ``benchmarks/abtest_montecarlo.py``), the choice
    barely moves the headline — under a silent +40% model change every option is within
    noise of no detector — so it is made on that principle, not on a measured gain. At a
    small holdout it rarely fires; a missed shift costs only a regime average, which
    randomisation keeps unbiased. Explicit model-id changes are the common case and are
    handled by stratification, not by this detector.
    """
    segments: list[tuple[list[SessionOutcome], str, str]] = []
    for o in sessions:
        if segments and segments[-1][0][-1].rate == o.rate:
            segments[-1][0].append(o)
        else:
            kind, why = (
                ("rate", f"holdout rate changed to {o.rate * 100:g}%") if segments else ("", "")
            )
            segments.append(([o], kind, why))
    eras: list[tuple[list[SessionOutcome], str, str]] = []
    for members, kind, why in segments:
        pts = [(o, _log_cost(o)) for o in members if o.arm == HOLDOUT]
        obs = [(o, y) for o, y in pts if y is not None]
        # (transition start, alarm session start) per detected shift, in time order
        spans = [
            (obs[c][0].start, obs[a][0].start)
            for c, a in st.cusum_changepoints([y for _, y in obs], h=HOLDOUT_H)
        ]
        cur: list[SessionOutcome] = []
        b = 0
        for o in members:
            while b < len(spans) and o.start > spans[b][1]:
                if cur:
                    eras.append((cur, kind, why))
                cur, kind = [], "shift"
                why = "shift in the holdout arm's cost per session (CUSUM)"
                b += 1
            if b < len(spans) and spans[b][0] <= o.start <= spans[b][1]:
                continue  # transition: selected by the detector, estimated in no era
            cur.append(o)
        if cur:
            eras.append((cur, kind, why))
    return eras


# --------------------------------------------------------------------------- #
# The summary
# --------------------------------------------------------------------------- #


def _covariates(sessions: list[SessionOutcome]) -> dict[str, float]:
    """sid → log mean cost per session of the SAME workspace's sessions that ended
    before it started. ponytail: O(n · sessions-per-workspace); fine into the tens of
    thousands."""
    by_ws: dict[str, list[SessionOutcome]] = {}
    for o in sessions:
        if o.ws and o.cost:
            by_ws.setdefault(o.ws, []).append(o)
    out: dict[str, float] = {}
    for o in sessions:
        prior = [p.cost or 0.0 for p in by_ws.get(o.ws, ()) if p.end < o.start]
        if prior and sum(prior) > 0:
            out[o.sid] = math.log(sum(prior) / len(prior))
    return out


#: metric -> (numerator, denominator) per session. "session" is the headline's estimand;
#: the rest are mediators — outcomes distil can change.
_METRICS: dict[str, tuple[Callable[[SessionOutcome], float], Callable[[SessionOutcome], float]]] = {
    "session": (lambda o: o.cost or 0.0, lambda o: 1.0),
    "turns_per_session": (lambda o: float(o.turns), lambda o: 1.0),
    "tasks_per_session": (lambda o: float(o.tasks), lambda o: 1.0),
    "cost_per_task": (lambda o: o.cost or 0.0, lambda o: float(o.tasks)),
    "cost_per_turn": (lambda o: o.cost or 0.0, lambda o: float(max(1, o.turns))),
}
MEDIATORS = ("turns_per_session", "tasks_per_session", "cost_per_task", "cost_per_turn")


def _units(members: list[SessionOutcome], metric: str, cov: dict[str, float]) -> list[st.Unit]:
    num, den = _METRICS[metric]
    x = cov if metric == "session" else {}
    return [st.Unit(num(o), den(o), o.arm == DISTIL, x.get(o.sid)) for o in members]


def _mean_cost(members: list[SessionOutcome], arm: str) -> float | None:
    xs = [o.cost or 0.0 for o in members if o.arm == arm]
    return sum(xs) / len(xs) if xs else None


def _cap() -> float:
    import os

    try:
        v = float(os.environ.get("DISTIL_AB_COST_CAP", "") or st.COST_CAP)
        return v if v > 0 else st.COST_CAP
    except ValueError:
        return st.COST_CAP


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "?"


def summarize(
    outcomes: list[SessionOutcome],
    *,
    rate: float | None = None,
    window: float | None = None,
    now: float | None = None,
    alpha: float | None = None,
) -> ABSummary:
    """Pure function of the outcome rows — everything ``distil ab`` prints."""
    a = st.default_alpha() if alpha is None else alpha
    r = holdout_rate() if rate is None else rate
    now = time.time() if now is None else now
    rows = [o for o in outcomes if o.arm in (DISTIL, HOLDOUT)]
    if window is not None:
        rows = [o for o in rows if o.start >= now - window * 86400]
    s = ABSummary(
        status="no-data",
        message="",
        holdout_rate=r,
        disclosure=disclosure(r),
        window_days=window,
        alpha=a,
        n_sessions=len(rows),
        n_open=sum(1 for o in rows if not o.complete),
        n_unpriced=sum(1 for o in rows if o.complete and o.cost is None),
    )
    use = sorted((o for o in rows if o.complete and o.cost is not None), key=lambda o: o.start)
    s.n_distil = sum(1 for o in use if o.arm == DISTIL)
    s.n_holdout = sum(1 for o in use if o.arm == HOLDOUT)
    s.n_expanded = sum(o.expanded for o in use if o.arm == DISTIL)
    s.notional = any(o.billing == "subscription" for o in use)
    s.cap = _cap()
    # Balance check: how sessions ended, per arm. A distil-caused failure mode shows here
    # before it shows anywhere else; unpriced and open sessions are counted, not hidden.
    for arm in (DISTIL, HOLDOUT):
        mine = [o for o in rows if o.arm == arm]
        b: dict[str, int] = {}
        for o in mine:
            b[o.ended] = b.get(o.ended, 0) + 1
        b["unpriced"] = sum(1 for o in mine if o.complete and o.cost is None)
        b["capped"] = sum(1 for o in use if o.arm == arm and (o.cost or 0) > s.cap)
        b["pre_1_54_expand_rows"] = sum(o.expanded for o in mine)
        s.balance[arm] = b
    cov = _covariates(use)

    # strata → groups; a newer model/client version supersedes the older one
    strata: dict[tuple[str, str], list[SessionOutcome]] = {}
    for o in use:
        strata.setdefault((o.model, o.client), []).append(o)
    groups: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(strata, key=lambda k: strata[k][0].start):
        g = f"{family(key[0])} | {_client_family(key[1])}"
        if groups.get(g):
            prev = groups[g][-1]
            what = "model" if prev[0] != key[0] else "client"
            old, new = (prev[0], key[0]) if what == "model" else (prev[1], key[1])
            s.events.append(Event(strata[key][0].start, g, what, f"{what} changed {old} → {new}"))
        groups.setdefault(g, []).append(key)

    current_parts: dict[str, list[st.Contrast]] = {m: [] for m in _METRICS}
    current_cells: list[list[st.Unit]] = []
    head_members: list[SessionOutcome] = []
    for g, keys in groups.items():
        # per version: (first era's contrast, last era's contrast), for the rollout DiD
        ends: list[tuple[str, st.Contrast | None, st.Contrast | None]] = []
        for ki, key in enumerate(keys):
            firsts: list[st.Contrast | None] = []
            eras = split_eras(strata[key])
            for ei, (members, kind, why) in enumerate(eras):
                if kind in ("shift", "rate"):
                    s.events.append(Event(members[0].start, g, kind, why))
                n1 = sum(1 for o in members if o.arm == DISTIL)
                n0 = len(members) - n1
                enough = n1 >= st.MIN_ARM_CELL and n0 >= st.MIN_ARM_CELL
                con = {
                    m: st.ratio_contrast(_units(members, m, cov)) if enough else None
                    for m in _METRICS
                }
                current = ki == len(keys) - 1 and ei == len(eras) - 1
                reason = why or ("first seen" if ki == 0 else "new version")
                s.cells.append(
                    Cell(
                        model=key[0],
                        client=key[1],
                        group=g,
                        era=ei + 1,
                        reason=reason,
                        start=members[0].start,
                        end=max(o.end for o in members),
                        n_distil=n1,
                        n_holdout=n0,
                        cost_per_session={
                            DISTIL: _mean_cost(members, DISTIL),
                            HOLDOUT: _mean_cost(members, HOLDOUT),
                        },
                        cost=Effect.of(con["session"], a),
                        turns=Effect.of(con["turns_per_session"], a),
                        cost_per_task=Effect.of(con["cost_per_task"], a),
                        current=current,
                    )
                )
                firsts.append(con["session"])
                if current:
                    # The headline pools every current cell, however small: the robust
                    # estimate is a randomised comparison over their union.
                    head_members.extend(members)
                    for m in _METRICS:
                        if con[m] is not None:
                            current_parts[m].append(con[m])  # type: ignore[arg-type]
                    current_cells.append(_units(members, "session", cov))
            ends.append(
                (
                    f"{key[0]} | {key[1]}",
                    firsts[0] if firsts else None,
                    firsts[-1] if firsts else None,
                )
            )
        # A rollout is a version change: the old version's LAST era against the new
        # one's FIRST, the two cells adjacent in time. Not across CUSUM eras: those
        # are not rollouts, and a closed one carries the detector's selection bias.
        for (b_name, _, before), (a_name, after, _) in zip(ends, ends[1:]):
            if before is not None and after is not None:
                eff = Effect.of(st.difference(after, before), a)
                if eff is not None:
                    s.did.append(DiD(g, b_name, a_name, eff))
    s.events.sort(key=lambda e: e.ts)
    # The line a savings screen prints: the latest version change if there is one
    # (that is what readers mean by "the model changed"), else the latest shift.
    changes = [e for e in s.events if e.kind in ("model", "client")] or [
        e for e in s.events if e.kind == "shift"
    ]
    if changes:
        e = changes[-1]
        what = "behaviour" if e.kind == "shift" else e.kind
        within = "era" if e.kind == "shift" else e.kind
        s.model_change = (
            f"{what} changed on {_day(e.ts)} ({e.group}): comparing within the new {within} only"
        )

    if r <= 0 and not use:
        s.status = "disabled"
        s.message = disclosure(r)
        return s
    if not use:
        s.message = "no randomised sessions yet — the A/B starts with the next session"
        return s
    d_costs = [o.cost or 0.0 for o in head_members if o.arm == DISTIL]
    h_costs = [o.cost or 0.0 for o in head_members if o.arm == HOLDOUT]
    capped = [min(c, s.cap) for c in d_costs + h_costs]
    if len(capped) >= 2 and sum(capped) > 0:
        mu = sum(capped) / len(capped)
        cv = math.sqrt(sum((c - mu) ** 2 for c in capped) / (len(capped) - 1)) / mu
    else:
        cv = 1.5  # the maintainer's own sessions measured ~1.55
    s.need_holdout = max(st.MIN_HOLDOUT, st.robust_needed(cv, r or 0.05))
    s.need_sessions = int(s.need_holdout / (r or 0.05))
    headline = st.robust_ratio(d_costs, h_costs, cap=s.cap, alpha=a)
    upfront = (
        f"Individual results take months to become conclusive: need ≈{s.need_sessions:,} "
        f"sessions (≈{s.need_holdout:,} held out) to resolve a "
        f"{st.TARGET_EFFECT * 100:.0f}% effect on this machine"
    )
    s.message = upfront
    if headline is None or len(h_costs) < st.MIN_HOLDOUT:
        s.status = "insufficient"
        s.message = (
            f"{upfront}; so far {len(d_costs)} distil / {len(h_costs)} holdout sessions in "
            "the current model"
        ) + (f" · {s.model_change}" if s.model_change else "")
        return s
    s.status = "ok"
    head = Effect.robust(headline)
    assert head is not None
    s.cost = head
    pooled = {m: st.pool(v) for m, v in current_parts.items()}
    s.efficient = Effect.of(pooled["session"], a)
    s.mediators = {m: Effect.of(pooled[m], a) for m in MEDIATORS}
    s.bootstrap = st.bootstrap_pooled(current_cells, alpha=a)
    if s.bootstrap is not None:
        boot_sig = s.bootstrap[1] < 0 or s.bootstrap[0] > 0
        s.bootstrap_disagrees = boot_sig != head.significant
    s.holdout_spend = sum(h_costs)
    h = s.holdout_spend
    hi = head.hi if math.isfinite(head.hi) else math.inf
    s.holdout_cost = (-head.rel * h, -hi * h, -head.lo * h)
    verdict = "" if head.significant else " (not yet distinguishable from 0)"
    s.message = (
        f"distil vs holdout: {head.rel * 100:+.1f}% mean cost per session "
        f"({(1 - a) * 100:.0f}% anytime CI {_pct(head.lo)} … {_pct(head.hi)})"
        f"{verdict}, n={headline.n1}/{headline.n0} on this machine"
        + (f" · {s.model_change}" if s.model_change else "")
        + ("" if head.significant else f" · {upfront}")
    )
    return s


def abtest_summary(window: float | None = None) -> ABSummary:
    """Fold whatever is on disk, then summarise. ``window`` is days (None = all time)."""
    harvest()
    return summarize(list(load().values()), window=window)


# --------------------------------------------------------------------------- #
# Rendering + CLI
# --------------------------------------------------------------------------- #


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return "+∞" if x == math.inf else f"{x * 100:+.1f}%"


def _eff(e: Effect | None, alpha: float, *, label: str = "anytime CI") -> str:
    if e is None:
        return "—"
    tail = "significant" if e.significant else "not significant"
    ev = "" if e.evalue is None else f"; e-value {e.evalue:.3g}"
    return (
        f"{_pct(e.rel)}  ({(1 - alpha) * 100:.0f}% {label} {_pct(e.lo)} … {_pct(e.hi)}{ev}; {tail})"
    )


def _money(x: float | None) -> str:
    if x is None:
        return "—"
    if x == math.inf:
        return "∞"
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.4f}" if abs(x) < 1 else f"{sign}${abs(x):,.2f}"


_MED_LABEL = {
    "turns_per_session": "turns per session",
    "tasks_per_session": "tasks per session",
    "cost_per_task": "cost per task",
    "cost_per_turn": "cost per turn",
}


def render(s: ABSummary) -> str:
    L = ["distil ab — does distil lower what a session costs? (randomised holdout)"]
    L.append(f"  {s.disclosure}")
    L.append("  Scope: this machine's own sessions only.")
    if s.window_days is None:
        win = "all time (the anytime guarantee holds within the current era)"
    else:
        win = f"last {s.window_days:g} days (a sliding window is a fixed-n look, not anytime-valid)"
    L.append(
        f"  window: {win} · {s.n_sessions} sessions (ended+priced: {s.n_distil} distil, "
        f"{s.n_holdout} holdout) · {s.n_open} open · {s.n_unpriced} unpriced"
    )
    L.append("")
    if s.status != "ok":
        L.append(f"  {s.message}")
    else:
        assert s.cost is not None
        L.append(f"  mean cost per session  {_eff(s.cost, s.alpha)}")
        L.append(
            f"    the one primary claim: {s.cost.method}; anytime-valid within this era, "
            "no normality assumed"
        )
        if not s.cost.significant and s.need_sessions:
            L.append(
                f"    Individual results take months to become conclusive: need ≈"
                f"{s.need_sessions:,} sessions ({s.need_holdout:,} held out)."
            )
        if s.bootstrap:
            L.append(
                f"  bootstrap cross-check (fixed-n, strata-pooled): "
                f"{_pct(s.bootstrap[0])} … {_pct(s.bootstrap[1])}"
            )
            if s.bootstrap_disagrees:
                L.append(
                    "    ⚠ the bootstrap and the anytime interval DISAGREE on whether this is "
                    "distinguishable from 0 — trust the anytime interval; a fixed-n "
                    "interval re-checked over time overstates certainty, and heavy tails "
                    "make it worse"
                )
        L.append(
            f"  efficient estimate (strata-pooled, CUPED, asymptotic)  "
            f"{_eff(s.efficient, s.alpha, label='CI')}"
        )
        if s.efficient and s.efficient.variance_reduction:
            L.append(f"    CUPED variance −{s.efficient.variance_reduction * 100:.0f}%")
        L.append("  mediators — outcomes distil can change, so never the headline:")
        for m in MEDIATORS:
            L.append(f"    {_MED_LABEL[m]:<18} {_eff(s.mediators.get(m), s.alpha, label='CI')}")
        L.append(
            "    (secondary claims are asymptotic and not multiplicity-adjusted; only the "
            "headline spends the error budget)"
        )
        if s.holdout_cost:
            est, lo, hi = s.holdout_cost
            more = "more" if est >= 0 else "less"
            L.append(
                f"  cost of the holdout: those {s.cost.n_holdout} sessions cost "
                f"{_money(s.holdout_spend)}; compressed, ≈{_money(abs(est))} {more} "
                f"(range {_money(lo)} … {_money(hi)})"
            )
        if s.model_change:
            L.append(f"  {s.model_change}")
    if s.balance:
        L.append("")
        L.append("  balance check — how sessions ended, per arm:")
        for arm, b in s.balance.items():
            parts = ", ".join(f"{k} {v}" for k, v in sorted(b.items()) if v)
            L.append(f"    {arm:<8} {parts or '—'}")
        if any(b.get("capped") for b in s.balance.values()):
            L.append(f"    capped = sessions above the ${s.cap:g} cap, counted at the cap")
    if s.notional:
        L.append("  $ are list price; subscription sessions are not billed per token (notional)")
    if s.n_expanded:
        L.append(
            f"  caveat: {s.n_expanded} distil-arm requests recorded before 1.54 ran a "
            "distil_expand re-query whose own usage was not kept — those sessions' cost is "
            "a lower bound"
        )
    if s.cells:
        L += [
            "",
            "  strata (model | client, era)                  n dist  n hold  $/sess dist  "
            "$/sess hold  Δ $/session  Δ turns/sess  Δ $/task",
        ]
        for c in s.cells:
            name = f"{c.model} | {c.client} #{c.era}{'*' if c.current else ''}"
            L.append(
                f"  {name[:45]:<45} {c.n_distil:>6}  {c.n_holdout:>6}  "
                f"{_money(c.cost_per_session.get(DISTIL)):>11}  "
                f"{_money(c.cost_per_session.get(HOLDOUT)):>11}"
                f"  {_pct(c.cost.rel if c.cost else None):>11}  "
                f"{_pct(c.turns.rel if c.turns else None):>12}  "
                f"{_pct(c.cost_per_task.rel if c.cost_per_task else None):>8}"
            )
        L.append(
            "  * = current era, pooled into the headline · Δ columns are per-cell, "
            "CUPED-adjusted and asymptotic"
        )
    if s.events:
        L += ["", "  eras / change points"]
        L += [f"    {_day(e.ts)}  {e.group}: {e.text}" for e in s.events]
    if s.did:
        L += [
            "",
            "  difference-in-differences across a rollout ((distil−holdout)_after − _before; "
            "asymptotic)",
        ]
        L += [
            f"    {d.group}: {d.before} → {d.after}: {_eff(d.effect, s.alpha, label='CI')}"
            for d in s.did
        ]
    return "\n".join(L)


def _clean(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def cmd_ab(args: argparse.Namespace) -> int:
    if args.holdout_rate is not None:
        from . import parse_rate, set_holdout_rate

        try:
            set_holdout_rate(parse_rate(args.holdout_rate))
        except (ValueError, OSError) as exc:
            print(f"distil ab: {exc}", file=sys.stderr)
            return 2
        print(disclosure())
        return 0
    s = abtest_summary(args.window)
    if args.json:
        print(json.dumps(_clean(s.to_json()), indent=2, sort_keys=True))
    else:
        print(render(s))
    return 0


def register(sub: Any) -> None:
    """``distil ab`` — one call from :mod:`distil.cli`."""
    p = sub.add_parser(
        "ab",
        help="causal task-level A/B: distil vs a randomised uncompressed holdout, per model",
    )
    p.add_argument(
        "--window", type=float, metavar="DAYS", help="only sessions started in the last N days"
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--holdout-rate",
        metavar="R",
        help="set the holdout fraction for new sessions (0 disables; e.g. 0.05 or 5%%) and exit",
    )
    p.set_defaults(func=cmd_ab)

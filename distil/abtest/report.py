"""``abtest_summary`` and ``distil ab`` — the causal, model-aware answer.

Cells. A session's stratum is ``(model, client)`` — the model that carried most of its
cost (a dated snapshot or a version bump is a different model) and the client name at
major.minor. Its GROUP is the same pair with versions stripped (``claude-opus |
claude-cli``); a newer version appearing in a group SUPERSEDES the older one, so the
headline compares within the new model only and the old version's last era is
compared with the new version's first as a difference-in-differences. Inside a
stratum, eras are cut by (a) a change in the holdout rate — the arms' mix changes —
and (b) a CUSUM alarm on log cost per task, read across both arms (a silent change
behind an unchanged model id). No estimate ever mixes two cells.

Why randomisation, not the change points, is what protects the estimate: both arms
are sampled concurrently, so a model change that moves turns per task moves them in
both and cancels in the contrast even inside one cell. Eras exist so the number
describes the CURRENT regime rather than an average over regimes.
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
    evalue: float
    significant: bool
    n_distil: int
    n_holdout: int
    variance_reduction: float = 0.0  # CUPED; 0 when not applied

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
    cost_per_task: dict[str, float | None]  # arm -> $ (ratio of sums)
    cost: Effect | None
    turns: Effect | None
    cost_per_turn: Effect | None
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
    cost: Effect | None = None
    turns: Effect | None = None
    cost_per_turn: Effect | None = None
    bootstrap: tuple[float, float] | None = None
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


def _log_cpt(o: SessionOutcome) -> float | None:
    c = o.cost_per_task
    return math.log(c) if c and c > 0 else None


def split_eras(sessions: list[SessionOutcome]) -> list[tuple[list[SessionOutcome], str, str]]:
    """One stratum's sessions (time-ordered) → eras ``(members, kind, reason)``.

    A rate change opens an era at the first session drawn under the new rate. Within
    a rate segment, a CUSUM alarm on log cost per task opens an era at the first session
    AFTER the alarm; sessions between the estimated change and the alarm are a
    transition and belong to no era — they are the ones the detector selected for being
    extreme (see :func:`stats.cusum_changepoints`).

    The detector reads BOTH arms, arm-blind — not the holdout arm alone. Any change
    detector selects the data it cuts on; reading one arm puts all of that selection on
    one side of the contrast. Measured at h=8 on 150 stationary replications (true effect
    −20%, 1,500 sessions, 50/50; ``benchmarks/abtest_montecarlo.py``): eras closed by a
    holdout-only alarm were biased +6.7pp, arm-blind +1.3pp; the current era was
    unbiased either way (−0.2pp / −0.3pp). At a 5%
    holdout the arm-blind series is also 20x denser, so a real shift is caught in tens
    of sessions rather than hundreds. The price: a change in distil's own effect (an
    upgrade) can open an era too, which is why it is labelled a shift, not a model change.
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
        pts = [(o, _log_cpt(o)) for o in members]
        obs = [(o, y) for o, y in pts if y is not None]
        # (transition start, alarm session start) per detected shift, in time order
        spans = [
            (obs[c][0].start, obs[a][0].start)
            for c, a in st.cusum_changepoints([y for _, y in obs])
        ]
        cur: list[SessionOutcome] = []
        b = 0
        for o in members:
            while b < len(spans) and o.start > spans[b][1]:
                if cur:
                    eras.append((cur, kind, why))
                cur, kind = [], "shift"
                why = "shift in cost per task, both arms (CUSUM)"
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
    """sid → log mean cost/task of the SAME workspace's sessions that ended before it
    started. ponytail: O(n · sessions-per-workspace); fine into the tens of thousands."""
    by_ws: dict[str, list[SessionOutcome]] = {}
    for o in sessions:
        if o.ws and o.cost_per_task:
            by_ws.setdefault(o.ws, []).append(o)
    out: dict[str, float] = {}
    for o in sessions:
        prior = [
            p.cost_per_task for p in by_ws.get(o.ws, ()) if p.end < o.start and p.cost_per_task
        ]
        if prior:
            out[o.sid] = math.log(sum(prior) / len(prior))  # type: ignore[arg-type]
    return out


def _units(members: list[SessionOutcome], metric: str, cov: dict[str, float]) -> list[st.Unit]:
    num: Callable[[SessionOutcome], float]
    den: Callable[[SessionOutcome], float]
    if metric == "cost":
        num, den = (lambda o: o.cost or 0.0), (lambda o: float(o.tasks))
    elif metric == "turns":
        num, den = (lambda o: float(o.turns)), (lambda o: float(o.tasks))
    else:  # cost per turn
        num, den = (lambda o: o.cost or 0.0), (lambda o: float(max(1, o.turns)))
    return [st.Unit(num(o), den(o), o.arm == DISTIL, cov.get(o.sid)) for o in members]


def _cpt(members: list[SessionOutcome], arm: str) -> float | None:
    xs = [o for o in members if o.arm == arm]
    tasks = sum(o.tasks for o in xs)
    return sum(o.cost or 0.0 for o in xs) / tasks if xs and tasks else None


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

    current_parts: dict[str, list[st.Contrast | None]] = {"cost": [], "turns": [], "cpt": []}
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
                    for m in ("cost", "turns", "cpt")
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
                        cost_per_task={
                            DISTIL: _cpt(members, DISTIL),
                            HOLDOUT: _cpt(members, HOLDOUT),
                        },
                        cost=Effect.of(con["cost"], a),
                        turns=Effect.of(con["turns"], a),
                        cost_per_turn=Effect.of(con["cpt"], a),
                        current=current and con["cost"] is not None,
                    )
                )
                firsts.append(con["cost"])
                if current and con["cost"] is not None:
                    for m in current_parts:
                        current_parts[m].append(con[m])
                    current_cells.append(_units(members, "cost", cov))
                    head_members.extend(members)
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

    pooled = {m: st.pool([c for c in v if c is not None]) for m, v in current_parts.items()}
    n0_head = pooled["cost"].n0 if pooled["cost"] else 0
    if r <= 0 and not use:
        s.status = "disabled"
        s.message = disclosure(r)
        return s
    if not use:
        s.message = "no randomised sessions yet — the A/B starts with the next `distil wrap`"
        return s
    # The per-session spread for the projection: from the data when there is some.
    lin = [c.var_raw * (c.n1 * c.n0) / (c.n1 + c.n0) for c in current_parts["cost"] if c]
    unit_var = sum(lin) / len(lin) if lin else 1.0
    if pooled["cost"] is None or n0_head < st.MIN_HOLDOUT:
        s.status = "insufficient"
        s.need_holdout = max(st.MIN_HOLDOUT, st.holdout_needed(unit_var, r or 0.05))
        s.message = (
            f"not enough data yet (n={s.n_distil} distil / {s.n_holdout} holdout; "
            f"need ≈{s.need_holdout} holdout sessions in the current model)"
        ) + (f" · {s.model_change}" if s.model_change else "")
        return s
    s.status = "ok"
    s.cost = Effect.of(pooled["cost"], a)
    s.turns = Effect.of(pooled["turns"], a)
    s.cost_per_turn = Effect.of(pooled["cpt"], a)
    s.bootstrap = st.bootstrap_pooled(current_cells, alpha=a)
    s.holdout_spend = sum(o.cost or 0.0 for o in head_members if o.arm == HOLDOUT)
    assert s.cost is not None
    h = s.holdout_spend
    s.holdout_cost = (-s.cost.rel * h, -s.cost.hi * h, -s.cost.lo * h)
    verdict = "" if s.cost.significant else " (not yet distinguishable from 0)"
    s.message = (
        f"distil vs holdout: {s.cost.rel * 100:+.1f}% cost per task "
        f"({(1 - a) * 100:.0f}% anytime CI {s.cost.lo * 100:+.1f}% … {s.cost.hi * 100:+.1f}%)"
        f"{verdict}, n={pooled['cost'].n1}/{n0_head}"
        + (f" · {s.model_change}" if s.model_change else "")
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
    return "—" if x is None else f"{x * 100:+.1f}%"


def _eff(e: Effect | None, alpha: float) -> str:
    if e is None:
        return "—"
    tail = "significant" if e.significant else "not significant"
    return (
        f"{_pct(e.rel)}  ({(1 - alpha) * 100:.0f}% anytime CI {_pct(e.lo)} … {_pct(e.hi)}; "
        f"e-value {e.evalue:.3g}; {tail})"
    )


def _money(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.4f}" if abs(x) < 1 else f"{sign}${abs(x):,.2f}"


def render(s: ABSummary) -> str:
    L = ["distil ab — task-level A/B (session-randomised holdout)", f"  {s.disclosure}"]
    win = "all time" if s.window_days is None else f"last {s.window_days:g} days"
    L.append(
        f"  window: {win} · {s.n_sessions} sessions (complete+priced: {s.n_distil} distil, "
        f"{s.n_holdout} holdout) · {s.n_open} open · {s.n_unpriced} unpriced"
    )
    L.append("")
    if s.status != "ok":
        L.append(f"  {s.message}")
    else:
        L.append(f"  cost per task   {_eff(s.cost, s.alpha)}")
        L.append(f"  turns per task  {_eff(s.turns, s.alpha)}")
        L.append(f"  cost per turn   {_eff(s.cost_per_turn, s.alpha)}")
        if s.bootstrap:
            L.append(
                f"  bootstrap cross-check (fixed-n {(1 - s.alpha) * 100:.0f}%): "
                f"{_pct(s.bootstrap[0])} … {_pct(s.bootstrap[1])}"
            )
        if s.cost and s.cost.variance_reduction:
            L.append(
                f"  CUPED (workspace pre-period cost/task): variance "
                f"−{s.cost.variance_reduction * 100:.0f}%"
            )
        if s.holdout_cost:
            est, lo, hi = s.holdout_cost
            more = "more" if est >= 0 else "less"
            L.append(
                f"  cost of the holdout: those {s.cost.n_holdout if s.cost else 0} sessions cost "
                f"{_money(s.holdout_spend)}; compressed, ≈{_money(abs(est))} {more} "
                f"(range {_money(lo)} … {_money(hi)})"
            )
        if s.model_change:
            L.append(f"  {s.model_change}")
    if s.notional:
        L.append("  $ are list price; subscription sessions are not billed per token (notional)")
    if s.n_expanded:
        L.append(
            f"  caveat: {s.n_expanded} distil-arm requests ran a distil_expand re-query "
            "whose own input tokens are not metered — the distil arm's cost is a lower bound"
        )
    if s.cells:
        L += [
            "",
            "  strata (model | client, era)                  n dist  n hold   $/task dist  "
            "$/task hold  Δ cost/task  Δ turns/task  Δ $/turn",
        ]
        for c in s.cells:
            name = f"{c.model} | {c.client} #{c.era}{'*' if c.current else ''}"
            L.append(
                f"  {name[:45]:<45} {c.n_distil:>6}  {c.n_holdout:>6}  "
                f"{_money(c.cost_per_task.get(DISTIL)):>12}  {_money(c.cost_per_task.get(HOLDOUT)):>11}"
                f"  {_pct(c.cost.rel if c.cost else None):>11}  "
                f"{_pct(c.turns.rel if c.turns else None):>12}  "
                f"{_pct(c.cost_per_turn.rel if c.cost_per_turn else None):>8}"
            )
        L.append("  * = current era, in the headline (inverse-variance pooled)")
    if s.events:
        L += ["", "  eras / change points"]
        L += [f"    {_day(e.ts)}  {e.group}: {e.text}" for e in s.events]
    if s.did:
        L += ["", "  difference-in-differences across a change ((distil−holdout)_after − _before)"]
        L += [f"    {d.group}: {d.before} → {d.after}: {_eff(d.effect, s.alpha)}" for d in s.did]
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

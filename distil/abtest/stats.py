"""The statistics behind ``distil ab`` — small, stdlib-only, and each piece standard.

Headline (since the 2026-09-25 statistical review). The primary estimand is the MEAN
LIST-PRICE COST PER RANDOMISED SESSION, distil ÷ holdout, over every randomised session
that has ended (failed ones at their actual cost, zero-cost ones at 0). Its interval is
:func:`robust_ratio`: one empirical-Bernstein betting confidence sequence per arm
(Waudby-Smith & Ramdas, JRSS-B 2023, "PrPl-EB") on per-session cost WINSORISED at a cap
fixed in advance (:data:`COST_CAP`), each at α/2, combined into an interval on the
ratio. It is anytime-valid with no asymptotics: bounded data is all it assumes. The
normal-mixture sequence below was validated on light tails, but at a 5% holdout with
session log-sd 1.5 it rejected a true null 8.3% of the time against a 5% budget (10.7%
at log-sd 2.0), and real sessions sit at log-sd ~1.9. It remains for the SECONDARY
estimates (strata-pooled with CUPED, the mediators, the rollout DiD), labelled
asymptotic.

Mediators. Cost per task, turns per session, tasks per session and cost per turn have
denominators distil can move (compression can change how often a user re-asks). They are
reported beside the headline, never as it: +10% cost per session with +25% tasks reads
as −12% cost per task.

Effect (secondary). For a ratio metric (cost per task = Σcost/Σtasks, turns per task, cost per
turn) the per-arm estimate is ``log(C̄/T̄)`` over SESSIONS, the randomisation unit, and
the effect is the difference of the two logs: ``exp(Δ) − 1`` is the relative change in
the arm's mean cost per task. Its variance is the delta method on the per-session
linearisation ``L_i = C_i/C̄ − T_i/T̄`` (Deng, Knoblich & Lu, KDD 2018).

CUPED (Deng, Xu, Kohavi & Walker, WSDM 2013). ``L`` is regressed on a pre-period
covariate ``X`` — the log of the workspace's mean cost per task over sessions that
ENDED BEFORE this one started. Those sessions' own arms were drawn independently of
this one's, so ``X`` is independent of this session's assignment and the adjustment
cannot bias the contrast; it only removes the between-project spread. Sessions with
no prior history get ``X`` imputed at the mean (their term vanishes). Below
:data:`CUPED_MIN` covariate-bearing sessions the adjustment is skipped.

Anytime validity. Every surface re-reads the estimate whenever it likes, so a
fixed-n interval would inflate the error rate with each look. The interval is the
normal-mixture confidence sequence / mixture SPRT (Robbins 1970; Johari, Koomen,
Pekelis & Walsh, "Always valid inference", OR 2022): with ``V`` the variance of
``Δ̂`` and mixing variance ``τ²`` the e-value is
``Λ = √(V/(V+τ²)) · exp(τ²Δ̂² / (2V(V+τ²)))`` and the ``1−α`` sequence is
``Δ̂ ± √(V(V+τ²)/τ² · log((V+τ²)/(Vα²)))``. By Ville's inequality
``P(∃t: Λ_t ≥ 1/α) ≤ α`` under the null, however often it is checked. The guarantee
is asymptotic (plug-in variance, CLT on session costs) — which is why there is a
data floor, and why the Monte Carlo in the tests checks it at realistic n.
``α`` is :data:`distil.conformal.BUDGET_DELTA`, the one failure budget every
statistical claim in distil shares.

Pooling. Strata (model × client) and eras are never mixed inside one estimate; the
headline is the inverse-variance weighted mean of the per-cell contrasts (post-
stratification). Weighting by precision targets a precision-weighted average effect,
not a traffic-weighted one — stated in the report.

Change points. Two-sided self-starting CUSUM (Page 1954; Hawkins 1987) on log cost per
task of every session in the stratum, arm-blind (why not the holdout arm alone: see
:func:`distil.abtest.report.split_eras`): each point standardised by the running in-era mean/sd,
``k = 0.5``, ``h = 12``. ``h = 5`` (the textbook default, ARL₀ ≈ 465) false-alarmed on
~28% of 150-point stationary series, as that ARL predicts; ``h = 8`` still cut two false
eras into a simulated 7,500-session history and starved the headline of its current
era. The arm-blind series is dense (every session, not 5% of them), so detection delay
is cheap to buy: at ``h = 12`` the measured self-starting false-alarm rate is ~1% by
3,000 points, a 1σ shift is flagged in a median 22 points, and a 0.5σ shift is caught
within 500 points about half the time. A missed shift costs nothing but a mixed-regime
average: randomisation keeps the contrast unbiased either way. The current era starts
after the alarm (data the decision never saw); closed eras carry a small selection
bias, so no difference-in-differences is taken across a CUSUM era.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

#: Mixing scale of the mSPRT on the log-ratio scale. The CS is tightest for effects
#: near τ; 0.2 (≈ ±20%) is the scale distil's real effects live at. Any fixed τ is
#: valid — it is chosen before the data, never tuned on it.
TAU = 0.2
#: Covariate-bearing sessions needed before CUPED is applied.
CUPED_MIN = 10
#: Per-arm sessions a (stratum, era) cell needs before it is estimated at all.
MIN_ARM_CELL = 5
#: Holdout sessions (summed over contributing cells) before a headline is printed.
MIN_HOLDOUT = 20
#: Effect size the "need ≈ N" projection is sized to resolve.
TARGET_EFFECT = 0.10
#: Per-session cost cap (USD) for the robust headline, fixed in advance and disclosed.
#: The maintainer's own wrap sessions have a median around $80 and log-sd ~1.9; $500 caps
#: the extreme tail without touching a typical session. Winsorising changes the estimand
#: to the capped mean — the report prints how many sessions the cap touched per arm.
#: Changing it is a change of estimand; ``DISTIL_AB_COST_CAP`` overrides it.
COST_CAP = 500.0


def default_alpha() -> float:
    from .. import conformal

    return float(conformal.BUDGET_DELTA)


@dataclass(frozen=True)
class Unit:
    """One session: ``num/den`` is its contribution to the ratio metric."""

    num: float
    den: float
    treated: bool
    x: float | None = None  # CUPED covariate


@dataclass(frozen=True)
class Contrast:
    """``Δ = log(metric_distil) − log(metric_holdout)`` and its variance."""

    delta: float
    var: float
    n1: int = 0
    n0: int = 0
    var_raw: float = 0.0  # the variance CUPED started from (== var when not applied)
    cuped: bool = False

    @property
    def rel(self) -> float:
        return math.expm1(self.delta)

    def interval(self, alpha: float | None = None, tau: float = TAU) -> tuple[float, float]:
        """The anytime-valid ``1−α`` interval, on the relative scale."""
        a = default_alpha() if alpha is None else alpha
        h = cs_halfwidth(self.var, a, tau)
        return math.expm1(self.delta - h), math.expm1(self.delta + h)

    def log_evalue(self, tau: float = TAU) -> float:
        return msprt_log_evalue(self.delta, self.var, tau)

    def significant(self, alpha: float | None = None, tau: float = TAU) -> bool:
        a = default_alpha() if alpha is None else alpha
        return self.log_evalue(tau) >= math.log(1.0 / a)

    @property
    def variance_reduction(self) -> float:
        return 1.0 - self.var / self.var_raw if self.cuped and self.var_raw > 0 else 0.0


def _var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _arm(units: Sequence[Unit]) -> tuple[float, list[float]] | None:
    n = len(units)
    if n < 2:
        return None
    nb = sum(u.num for u in units) / n
    db = sum(u.den for u in units) / n
    if nb <= 0 or db <= 0:
        return None
    return math.log(nb / db), [u.num / nb - u.den / db for u in units]


def ratio_contrast(units: Sequence[Unit], *, cuped: bool = True) -> Contrast | None:
    """Treated-vs-control contrast of a ratio metric, CUPED-adjusted when possible."""
    t = [u for u in units if u.treated]
    c = [u for u in units if not u.treated]
    at, ac = _arm(t), _arm(c)
    if at is None or ac is None:
        return None
    (mt, lt), (mc, lc) = at, ac
    raw = _var(lt) / len(t) + _var(lc) / len(c)
    delta, var, used = mt - mc, raw, False
    xs = [u.x for u in units if u.x is not None]
    if cuped and len(xs) >= CUPED_MIN and _var(xs) > 0:
        xbar = sum(xs) / len(xs)
        xt = [(u.x - xbar) if u.x is not None else 0.0 for u in t]
        xc = [(u.x - xbar) if u.x is not None else 0.0 for u in c]
        # Pooled slope; L is centred within each arm by construction.
        num = sum(a * b for a, b in zip(lt, xt)) + sum(a * b for a, b in zip(lc, xc))
        den = sum(b * b for b in xt) + sum(b * b for b in xc)
        theta = num / den if den > 0 else 0.0
        adj_t = mt - theta * (sum(xt) / len(t))
        adj_c = mc - theta * (sum(xc) / len(c))
        v = _var([a - theta * b for a, b in zip(lt, xt)]) / len(t) + _var(
            [a - theta * b for a, b in zip(lc, xc)]
        ) / len(c)
        if v > 0:
            delta, var, used = adj_t - adj_c, v, True
    if var <= 0:
        return None
    return Contrast(delta, var, len(t), len(c), raw, used)


def msprt_log_evalue(delta: float, var: float, tau: float = TAU) -> float:
    """log of the normal-mixture likelihood ratio (the mSPRT e-value)."""
    t2 = tau * tau
    return 0.5 * math.log(var / (var + t2)) + t2 * delta * delta / (2.0 * var * (var + t2))


def cs_halfwidth(var: float, alpha: float, tau: float = TAU) -> float:
    """Half-width of the normal-mixture confidence sequence at variance ``var``."""
    t2 = tau * tau
    return math.sqrt(var * (var + t2) / t2 * math.log((var + t2) / (var * alpha * alpha)))


def pool(contrasts: Sequence[Contrast]) -> Contrast | None:
    """Inverse-variance (post-stratified) pooling of independent cell contrasts."""
    cs = [c for c in contrasts if c.var > 0]
    if not cs:
        return None
    w = [1.0 / c.var for c in cs]
    sw = sum(w)
    raw_w = sum(1.0 / c.var_raw for c in cs if c.var_raw > 0)
    return Contrast(
        delta=sum(wi * c.delta for wi, c in zip(w, cs)) / sw,
        var=1.0 / sw,
        n1=sum(c.n1 for c in cs),
        n0=sum(c.n0 for c in cs),
        var_raw=1.0 / raw_w if raw_w > 0 else 1.0 / sw,
        cuped=any(c.cuped for c in cs),
    )


def difference(after: Contrast, before: Contrast) -> Contrast:
    """Difference-in-differences: ``Δ_after − Δ_before`` (independent samples)."""
    return Contrast(
        after.delta - before.delta,
        after.var + before.var,
        after.n1 + before.n1,
        after.n0 + before.n0,
        after.var_raw + before.var_raw,
        after.cuped or before.cuped,
    )


def bootstrap_pooled(
    cells: Sequence[Sequence[Unit]],
    *,
    iters: int = 400,
    alpha: float | None = None,
    seed: int = 1234,
) -> tuple[float, float] | None:
    """Percentile bootstrap of the pooled log-ratio (no CUPED), resampling sessions
    within arm within cell, weights re-derived per replicate. Relative scale.

    A fixed-n cross-check, NOT anytime-valid: it should roughly agree with the
    sequence (which is wider by design); a large disagreement flags heavy tails the
    normal approximation is not handling.
    """
    a = default_alpha() if alpha is None else alpha
    rng = random.Random(seed)
    arms = [([u for u in cell if u.treated], [u for u in cell if not u.treated]) for cell in cells]
    arms = [(t, c) for t, c in arms if len(t) >= 2 and len(c) >= 2]
    if not arms:
        return None
    stats: list[float] = []
    for _ in range(iters):
        parts = []
        for t, c in arms:
            con = ratio_contrast(
                [t[rng.randrange(len(t))] for _ in t] + [c[rng.randrange(len(c))] for _ in c],
                cuped=False,
            )
            if con is not None:
                parts.append(con)
        pooled = pool(parts)
        if pooled is not None:
            stats.append(pooled.delta)
    if len(stats) < 20:
        return None
    stats.sort()
    lo = stats[int((a / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - a / 2) * len(stats)))]
    return math.expm1(lo), math.expm1(hi)


def cusum_changepoints(
    ys: Sequence[float], *, burn: int = 10, k: float = 0.5, h: float = 12.0, clip: float = 4.0
) -> list[tuple[int, int]]:
    """``(change, alarm)`` index pairs: two-sided SELF-STARTING CUSUM (Hawkins 1987),
    restarted after each alarm.

    ``change`` is the estimated start of the shift, ``alarm`` the point that tripped it.
    A caller that estimates on the new regime must start it AFTER ``alarm``, not at
    ``change``: the points in ``[change, alarm]`` are the ones that were selected for
    being extreme, and keeping them biased the post-change contrast by −2.4pp (3.4 SE)
    in this module's Monte Carlo. After the alarm, data never informed the decision.

    Each point is standardised by the mean and sd of the in-era points BEFORE it
    (``z = (y − m)/s · √(n/(n+1))``), so no reference is frozen from a short burn-in —
    a frozen 10-point median/MAD reference false-alarmed on 60% of stationary series in
    this module's own test. ``z`` is clipped at ±``clip`` so one heavy-tailed session
    cannot trip it alone. ``change`` is the first observation after the triggering
    statistic was last zero (the CUSUM change-time estimate). Monitoring restarts, with
    a fresh reference, after the alarm.
    """
    cuts: list[tuple[int, int]] = []
    start = 0
    while start + burn < len(ys):
        n, mean, m2 = 0, 0.0, 0.0
        sp = sn = 0.0
        zp = zn = start + burn
        found: tuple[int, int] | None = None
        for i in range(start, len(ys)):
            y = ys[i]
            if n >= burn:
                sd = math.sqrt(m2 / (n - 1)) if n > 1 else 0.0
                z = (y - mean) / max(sd, 1e-6) * math.sqrt(n / (n + 1))
                z = max(-clip, min(clip, z))
                sp, sn = max(0.0, sp + z - k), max(0.0, sn - z - k)
                if sp == 0.0:
                    zp = i + 1
                if sn == 0.0:
                    zn = i + 1
                if sp > h or sn > h:
                    found = (zp if sp > h else zn, i)
                    break
            n += 1  # Welford: the reference absorbs only in-control points
            d = y - mean
            mean += d / n
            m2 += d * (y - mean)
        if found is None:
            break
        cuts.append(found)
        start = found[1] + 1
    return cuts


def eb_cs(xs: Sequence[float], alpha: float) -> tuple[float, float]:
    """Running-intersection empirical-Bernstein CS for the mean of ``xs`` ⊂ [0, 1].

    PrPl-EB (Waudby-Smith & Ramdas 2023, Thm 2): predictable λ_t, capital-free closed
    form. Valid uniformly over time at level ``alpha`` (two-sided), so the running
    intersection over every prefix is valid too — and never widens. Processed in
    arrival order; ``(0, 1)`` for an empty sequence.
    """
    lo, hi = 0.0, 1.0
    log2a = math.log(2.0 / alpha)
    s = sum_l = sum_lx = sum_vpsi = 0.0
    sq = 0.25
    mu_prev, sig2_prev = 0.5, 0.25
    for t, x in enumerate(xs, start=1):
        lam = min(math.sqrt(2.0 * log2a / (sig2_prev * t * math.log(1.0 + t))), 0.5)
        v = 4.0 * (x - mu_prev) ** 2
        psi = (-math.log(1.0 - lam) - lam) / 4.0
        sum_l += lam
        sum_lx += lam * x
        sum_vpsi += v * psi
        centre = sum_lx / sum_l
        half = (log2a + sum_vpsi) / sum_l
        lo, hi = max(lo, centre - half), min(hi, centre + half)
        s += x
        mu_prev = (0.5 + s) / (t + 1)
        sq += (x - mu_prev) ** 2
        sig2_prev = sq / (t + 1)
    return lo, min(hi, 1.0) if hi >= lo else lo


@dataclass(frozen=True)
class RobustRatio:
    """Headline: capped mean cost per session, distil ÷ holdout, with its anytime CS."""

    rel: float  # point estimate of mean_distil/mean_holdout − 1 (capped means)
    lo: float
    hi: float  # may be +inf when the holdout interval reaches 0
    n1: int
    n0: int
    mean1: float
    mean0: float
    capped1: int  # sessions the cap touched
    capped0: int
    cap: float

    @property
    def significant(self) -> bool:
        return self.hi < 0.0 or self.lo > 0.0


def robust_ratio(
    distil_costs: Sequence[float],
    holdout_costs: Sequence[float],
    *,
    cap: float = COST_CAP,
    alpha: float | None = None,
) -> RobustRatio | None:
    """The shipped headline method. Costs in arrival order, one per session."""
    a = default_alpha() if alpha is None else alpha
    if len(distil_costs) < 2 or len(holdout_costs) < 2 or cap <= 0:
        return None
    x1 = [min(max(c, 0.0), cap) / cap for c in distil_costs]
    x0 = [min(max(c, 0.0), cap) / cap for c in holdout_costs]
    m1, m0 = sum(x1) / len(x1), sum(x0) / len(x0)
    if m0 <= 0:
        return None
    lo1, hi1 = eb_cs(x1, a / 2)
    lo0, hi0 = eb_cs(x0, a / 2)
    return RobustRatio(
        rel=m1 / m0 - 1.0,
        lo=lo1 / hi0 - 1.0 if hi0 > 0 else -1.0,
        hi=hi1 / lo0 - 1.0 if lo0 > 0 else math.inf,
        n1=len(x1),
        n0=len(x0),
        mean1=m1 * cap,
        mean0=m0 * cap,
        capped1=sum(1 for c in distil_costs if c > cap),
        capped0=sum(1 for c in holdout_costs if c > cap),
        cap=cap,
    )


def robust_needed(
    cv: float, rate: float, *, target: float = TARGET_EFFECT, alpha: float | None = None
) -> int:
    """Holdout sessions for :func:`robust_ratio`'s interval to resolve ``target``.

    Planning approximation, not a guarantee: each arm's EB half-width is ≈
    ``cv·√(2·log(4/α)/n)`` relative to its mean (its asymptotic width), and the ratio's
    log-interval is about the sum of the two. ``cv`` is the per-session coefficient of
    variation of capped cost.
    """
    a = default_alpha() if alpha is None else alpha
    p = min(max(rate, 1e-3), 0.5)
    goal = math.log1p(target)
    k = cv * math.sqrt(2.0 * math.log(4.0 / a))
    n = 2
    while n < 10_000_000:
        n1 = n * (1 - p) / p
        if k / math.sqrt(n) + k / math.sqrt(n1) <= goal:
            break
        n = int(n * 1.25) + 1
    return n


def holdout_needed(
    unit_var: float,
    rate: float,
    *,
    target: float = TARGET_EFFECT,
    alpha: float | None = None,
    tau: float = TAU,
) -> int:
    """Holdout sessions for the sequence's half-width to drop below ``log(1+target)``.

    ``unit_var`` is the per-session variance of the linearised metric (≈ CV² of cost
    per task). With ``n1 = n0(1−p)/p``: ``V ≈ unit_var/(n0(1−p))``.
    """
    a = default_alpha() if alpha is None else alpha
    p = min(max(rate, 1e-3), 0.5)
    goal = math.log1p(target)
    n = 2
    while n < 10_000_000 and cs_halfwidth(unit_var / (n * (1 - p)), a, tau) > goal:
        n = int(n * 1.25) + 1
    return n

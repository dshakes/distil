"""Anytime-valid drift detection — fire when live decision-change exceeds the certified budget.

The Decision-Equivalence Risk Certificate (:mod:`distil.conformal`) holds under
*exchangeability*: it is valid for traffic that looks like the calibration set. The standing
GA risk is silent drift — a new model, a prompt change, or a workload shift pushes the true
decision-change rate above the budget α the operating point was certified at, and nothing
notices until quality has already degraded.

This module closes that gap with a sequential test that may be checked **after every turn**
without a multiplicity penalty. It runs a betting e-process for the null ``H0: risk ≤ α``:
capital ``K_t = ∏ (1 + λ_i (X_i − α))`` (predictable stakes ``λ_i``, the same tuning as
:func:`distil.conformal.betting_upper_bound`). Under ``H0`` the capital is a non-negative
supermartingale with ``K_0 = 1``, so by Ville's inequality ``P(∃t: K_t ≥ 1/δ) ≤ δ`` — the
false-alarm probability is at most ``δ`` *no matter how often you peek*. When capital crosses
``1/δ`` the monitor trips: the live risk has exceeded the budget with confidence ``1−δ``, and
the operating point should be recalibrated (:func:`distil.calibrate.calibrate_operating_point`)
or the gate should fall back to full context.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DriftMonitor:
    """A running, anytime-valid alarm for ``risk > alpha``.

    Feed it one per-turn loss at a time (``1.0`` iff the compressed decision diverged from the
    full-context decision, or any value in ``[0, 1]`` for a graded loss). ``tripped`` becomes
    ``True`` once the live risk is certified to exceed ``alpha`` at confidence ``1 − delta``.
    """

    alpha: float
    delta: float = 0.05
    c: float = 0.5

    capital: float = 1.0
    n: int = 0
    tripped: bool = False
    _run_sum: float = 0.0
    _run_sq: float = 0.0
    _sig2_prev: float = 0.25  # σ̂²_0

    def update(self, loss: float) -> bool:
        """Record one turn's loss; return whether the monitor is now tripped."""
        if self.tripped:
            return True
        self.n += 1
        ld = math.log(1.0 / self.delta)
        lam = min(self.c, math.sqrt(2.0 * ld / (self._sig2_prev * self.n)))  # predictable
        self.capital *= 1.0 + lam * (loss - self.alpha)
        if self.capital < 0.0:
            self.capital = 0.0
        # advance the running shrinkage variance for the next bet
        self._run_sum += loss
        mu = (0.5 + self._run_sum) / (1 + self.n)
        self._run_sq += (loss - mu) ** 2
        self._sig2_prev = (0.25 + self._run_sq) / (1 + self.n)
        if self.capital >= 1.0 / self.delta:
            self.tripped = True
        return self.tripped

    def observe(self, losses) -> bool:
        """Feed a batch of losses; return whether tripped after the batch."""
        for x in losses:
            if self.update(x):
                break
        return self.tripped

    @property
    def evalue(self) -> float:
        """Accumulated capital (an e-value); ``>= 1/delta`` means tripped."""
        return self.capital

    def status(self) -> dict:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "n": self.n,
            "evalue": round(self.capital, 4),
            "threshold": round(1.0 / self.delta, 2),
            "tripped": self.tripped,
            "action": "recalibrate or fall back to full context" if self.tripped else "ok",
        }


# --------------------------------------------------------------------------- #
# Live wiring — the same e-process, fed by shadow mode's paired rows
# --------------------------------------------------------------------------- #

#: The decision-change budget every certificate in this codebase is written against
#: — the ``--alpha`` default of ``distil conformal`` / ``certify`` /
#: ``certify-trajectories``, and the "≤5% at 95% confidence" the site publishes.
#: Named here so the live alarm and the offline certificate cannot drift apart;
#: ``tests/test_drift_live.py`` pins it to those CLI defaults.
BUDGET_ALPHA = 0.05

#: Failure probability of the alarm (Ville): at most this often under the null,
#: no matter how many times it is checked. Matches ``--delta``'s default.
BUDGET_DELTA = 0.05


def paired_loss(diff: int) -> float:
    """Map shadow's paired per-request difference into the ``[0, 1]`` loss ``drift`` wants.

    ``ShadowLedger`` stores ``d = 1{A==B} − 1{A==A'}`` per request (see
    :class:`distil.shadow.Equivalence`): ``−1`` iff compression changed the decision on a
    request where the model agreed with *itself*, ``+1`` for the reverse, ``0`` when both
    arms agreed or both disagreed. ``−d`` is therefore the compression-attributable harm,
    with the model's own sampling noise netted out request by request — but it lives in
    ``{−1, 0, 1}`` and the e-process needs ``[0, 1]``.

    The map is affine: ``x = (1 − d) / 2``. Then ``E[x] = (1 + harm) / 2``, so testing
    ``harm ≤ α`` is exactly testing ``E[x] ≤ (1 + α) / 2`` — which is why
    :func:`live_monitor` instantiates the monitor at that shifted budget rather than at α.
    Affine, not clipped, on purpose: ``max(0, −d)`` would count every request where the
    compressed arm happened to disagree while the self arm happened to agree, and on real
    (hot-sampled) traffic with ~56% A/A self-agreement that runs ~26% — it would print
    BREACHED against a 5% budget on traffic the paired estimator scores at 2.5%.
    """
    return (1.0 - float(diff)) / 2.0


def _state_path() -> Path:
    import os

    home = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return home / "drift.json"


_FIELDS = ("alpha", "delta", "capital", "n", "tripped", "_run_sum", "_run_sq", "_sig2_prev")


@dataclass
class LiveDrift:
    """The e-process above, persisted across sessions and fed from ``shadow.jsonl``.

    Anytime-validity is a property of the ORDER losses arrive in, so the state file
    carries a ``consumed`` count: rows are folded in file order, exactly once, and a row
    already counted is never re-bet. ``tripped`` is sticky by construction (the monitor
    refuses further updates), which is the reset semantics ``drift.py`` already has — the
    breach stays visible until the state is cleared by ``distil reset --shadow``, which
    archives the evidence the alarm was computed from.
    """

    monitor: DriftMonitor
    consumed: int = 0
    tripped_at: int = 0

    @classmethod
    def load(cls, path: Path | None = None) -> LiveDrift:
        p = path or _state_path()
        mon = DriftMonitor(alpha=(1.0 + BUDGET_ALPHA) / 2.0, delta=BUDGET_DELTA)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return cls(mon)
        if not isinstance(raw, dict):
            return cls(mon)
        # A budget change invalidates the accumulated capital — it was bet against a
        # different null. Start over rather than carry a number that means nothing.
        if raw.get("alpha") != mon.alpha or raw.get("delta") != mon.delta:
            return cls(mon)
        for f in _FIELDS:
            if f in raw:
                try:
                    setattr(mon, f, type(getattr(mon, f))(raw[f]))
                except (TypeError, ValueError):
                    return cls(DriftMonitor(alpha=mon.alpha, delta=mon.delta))
        return cls(mon, int(raw.get("consumed") or 0), int(raw.get("tripped_at") or 0))

    def save(self, path: Path | None = None) -> None:
        """Persist; best-effort, like every other content-free store here."""
        p = path or _state_path()
        payload = {f: getattr(self.monitor, f) for f in _FIELDS}
        payload["consumed"] = self.consumed
        payload["tripped_at"] = self.tripped_at
        try:
            from . import _filelock

            p.parent.mkdir(parents=True, exist_ok=True)
            with _filelock.locked(p):
                p.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

    def advance(self, diffs: list[int]) -> LiveDrift:
        """Fold every paired difference not yet counted. ``diffs`` is the full history."""
        for d in diffs[self.consumed :]:
            self.consumed += 1
            was = self.monitor.tripped
            if self.monitor.update(paired_loss(d)) and not was:
                self.tripped_at = self.consumed
        return self

    def line(self) -> str | None:
        """One sentence, or None below the shared reporting floor.

        Same floor discipline as every other surface: an e-value off a handful of
        samples is a number wearing a verdict, so below it we say how far along we are
        and claim nothing.
        """
        from .shadow import VERDICT_MIN_AB

        m = self.monitor
        if self.consumed < VERDICT_MIN_AB:
            return f"not enough samples yet ({self.consumed}/{VERDICT_MIN_AB})"
        if m.tripped:
            return (
                f"BREACHED at sample {self.tripped_at} "
                f"(e-value {m.evalue:.1f} ≥ {1.0 / m.delta:.0f}, n={self.consumed}) — "
                f"recalibrate: distil calibrate"
            )
        return f"intact (e-value {m.evalue:.2f}, n={self.consumed})"


def live_monitor(diffs: list[int], *, path: Path | None = None) -> LiveDrift:
    """Load the persisted monitor, fold in the new paired rows, persist, return it."""
    state = LiveDrift.load(path).advance(diffs)
    state.save(path)
    return state

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

import hashlib
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


def _chain(prev: str, diffs: list[int], sig: int) -> str:
    """Extend the prefix fingerprint by ``diffs``. ``prev`` is the fingerprint so far.

    ``consumed`` on its own is a count, and a count is only meaningful against the stream
    it counted. Fingerprinting the prefix makes that binding checkable: a truncation, an
    archival, a rewrite, a reordering or a signature bump that filters rows out all change
    it, where a length alone would miss every case that keeps the file the same size or
    larger. Content-free — the inputs are the ``{-1, 0, 1}`` decision differences already
    stored in the state file's own counters, never text.

    Chained rather than computed over the whole prefix, so folding one new row costs one
    hash of one integer instead of re-hashing every row ever seen. The saved fingerprint
    is therefore an accumulator, and the staleness check re-derives it the same way.
    """
    h = prev or hashlib.sha256(b"distil.drift/2:%d" % sig).hexdigest()[:16]
    for d in diffs:
        h = hashlib.sha256(b"%s:%d" % (h.encode(), d)).hexdigest()[:16]
    return h


@dataclass
class LiveDrift:
    """The e-process above, persisted across sessions and fed from ``shadow.jsonl``.

    Anytime-validity is a property of the ORDER losses arrive in, so the state file
    carries a ``consumed`` count: rows are folded in file order, exactly once, and a row
    already counted is never re-bet. ``tripped`` is sticky by construction (the monitor
    refuses further updates), which is the reset semantics ``drift.py`` already has — the
    breach stays visible until the state is cleared by ``distil reset --shadow``, which
    archives the evidence the alarm was computed from.

    **The count is bound to the stream it counted.** A bare index silently desynchronises:
    archive or truncate ``shadow.jsonl`` outside ``distil reset --shadow``, or bump
    ``SIG_VERSION`` so older rows are filtered out, and ``consumed`` exceeds the rows that
    now exist — ``diffs[consumed:]`` is empty, the line keeps reporting a stale ``n``, and
    every fresh sample is ignored until the new file outgrows the old count. So the state
    also carries ``sig`` and ``stream``, a fingerprint of the prefix actually folded. When
    either fails to match the rows in front of it, the e-process is **rebuilt from zero
    over the current stream** rather than carrying capital that was bet on evidence no
    longer on disk. That preserves anytime-validity: the rebuilt process folds the current
    stream in order from ``K_0 = 1``, so Ville's inequality applies to it exactly as
    before. What a rebuild cannot carry is validity *across* the replacement — and it
    should not, because the evidence the old capital summarised is gone. A state file
    written before this provenance existed has no fingerprint to check and rebuilds once,
    for the same reason.

    A caller who can replace ``shadow.jsonl`` at will can therefore clear a breach. That
    is the same capability ``distil reset --shadow`` already exposes deliberately; the
    alarm is an instrument for the operator, not a control on them.
    """

    monitor: DriftMonitor
    consumed: int = 0
    tripped_at: int = 0
    stream: str = ""  # fingerprint of the prefix already folded ("" = unknown provenance)
    sig: int = 0  # shadow SIG_VERSION that prefix was read under (0 = unknown)
    rebuilt: bool = False  # this fold restarted the e-process; not persisted

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
        try:
            consumed = int(raw.get("consumed") or 0)
            tripped_at = int(raw.get("tripped_at") or 0)
            sig = int(raw.get("sig") or 0)
            if consumed < 0 or tripped_at < 0:
                raise ValueError("negative counter")
        except (TypeError, ValueError):
            # Well-formed JSON with a corrupt field is still a corrupt state file: start
            # over rather than raise out of a verdict line that is meant to be fail-open.
            return cls(DriftMonitor(alpha=mon.alpha, delta=mon.delta))
        return cls(mon, consumed, tripped_at, str(raw.get("stream") or ""), sig)

    def save(self, path: Path | None = None) -> None:
        """Persist; best-effort, like every other content-free store here.

        Written to a temp file and renamed, never in place. ``tripped`` is sticky and
        capital accumulates across sessions, so a write torn by a crash, a full disk or a
        Ctrl-C would not corrupt the alarm noisily — it would silently reset it to a fresh
        monitor on the next load, which reads as "intact" over evidence that said BREACHED.
        A rename is atomic, so a reader sees either the old state or the new one.
        """
        import contextlib

        p = path or _state_path()
        payload: dict[str, object] = {f: getattr(self.monitor, f) for f in _FIELDS}
        payload["consumed"] = self.consumed
        payload["tripped_at"] = self.tripped_at
        payload["stream"] = self.stream
        payload["sig"] = self.sig
        tmp = p.with_name(p.name + ".tmp")
        try:
            from . import _filelock, atrest

            p.parent.mkdir(parents=True, exist_ok=True)
            with _filelock.locked(p):
                with open(tmp, "w", encoding="utf-8", opener=atrest.owner_only) as fh:
                    fh.write(json.dumps(payload))
                    fh.flush()
                _filelock.replace_retrying(tmp, p)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()

    def _stale(self, diffs: list[int], sig: int) -> bool:
        """Is the persisted prefix still the prefix of the stream in front of us?

        Re-derives the accumulator over the prefix. That is O(consumed) hashes of one
        integer each — no JSON, no allocation per row — against a single shared parse of
        the ledger that the caller has already paid for.
        """
        if not self.consumed:
            return False
        if len(diffs) < self.consumed or self.sig != sig:
            return True
        return self.stream != _chain("", diffs[: self.consumed], self.sig)

    def advance(self, diffs: list[int]) -> LiveDrift:
        """Fold every paired difference not yet counted. ``diffs`` is the full history.

        Rebuilds from zero first when the stream no longer matches what was folded — see
        the class docstring for why that keeps the e-process valid rather than breaking it.
        """
        from .shadow import SIG_VERSION

        if self._stale(diffs, SIG_VERSION):
            self.monitor = DriftMonitor(alpha=self.monitor.alpha, delta=self.monitor.delta)
            self.consumed = 0
            self.tripped_at = 0
            self.rebuilt = True
        if self.rebuilt or not self.stream:
            self.stream = _chain("", [], SIG_VERSION)
        fresh = diffs[self.consumed :]
        for d in fresh:
            self.consumed += 1
            was = self.monitor.tripped
            if self.monitor.update(paired_loss(d)) and not was:
                self.tripped_at = self.consumed
        self.sig = SIG_VERSION
        self.stream = _chain(self.stream, fresh, SIG_VERSION)
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
        # A rebuild makes n drop and can clear a breach. Saying so is the difference
        # between a verdict that explains itself and one that quietly went away.
        note = " — restarted: the shadow stream was replaced" if self.rebuilt else ""
        if m.tripped:
            return (
                f"BREACHED at sample {self.tripped_at} "
                f"(e-value {m.evalue:.1f} ≥ {1.0 / m.delta:.0f}, n={self.consumed}) — "
                f"recalibrate: distil calibrate{note}"
            )
        return f"intact (e-value {m.evalue:.2f}, n={self.consumed}){note}"


def live_monitor(diffs: list[int], *, path: Path | None = None) -> LiveDrift:
    """Load the persisted monitor, fold in the new paired rows, persist, return it."""
    state = LiveDrift.load(path).advance(diffs)
    state.save(path)
    return state

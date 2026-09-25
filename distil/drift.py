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
the operating point should be recalibrated (:func:`distil.calibrate.calibrate_operating_point`).

The trip acts: :class:`DriftGuard` holds the proxy at lossless-only from the next request,
persists that across restarts, and writes a receipt — see "The guard" below. The budget
itself lives in :mod:`distil.conformal` (``BUDGET_ALPHA``), shared with the certificate.

There is exactly ONE live e-process: the capital in ``drift.json``. Proxies are its only
writers, each folding the paired verdict it just produced under a file lock (:func:`fold`),
so every row is bet exactly once, in one global order, no matter how many proxies run or
how often they restart. Every reporting surface (wrap exit, ``distil stats``, the status
line) only reads it.
"""

from __future__ import annotations

import hashlib
import json
import math
import contextlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import conformal as _budget

log = logging.getLogger(__name__)


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

# The budget this alarm bets against is NOT defined here: it is
# ``conformal.BUDGET_ALPHA`` / ``BUDGET_DELTA``, the one pair every certificate and
# verdict reads, looked up at call time so the alarm cannot drift from the certificate.


def _null_mean() -> float:
    """The shifted budget the monitor runs at — see :func:`paired_loss` for why."""
    return (1.0 + _budget.BUDGET_ALPHA) / 2.0


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
    :class:`LiveDrift` instantiates the monitor at that shifted budget rather than at α.
    Affine, not clipped, on purpose: ``max(0, −d)`` would count every request where the
    compressed arm happened to disagree while the self arm happened to agree, and on real
    (hot-sampled) traffic with ~56% A/A self-agreement that runs ~26% — it would print
    BREACHED against a 5% budget on traffic the paired estimator scores at 2.5%.
    """
    return (1.0 - float(diff)) / 2.0


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def _state_path() -> Path:
    return _home() / "drift.json"


_FIELDS = ("alpha", "delta", "capital", "n", "tripped", "_run_sum", "_run_sq", "_sig2_prev")

#: drift.json schema. 2 = the proxy-folded e-process. Anything older was the exit-time
#: fold of shadow.jsonl and is rebuilt once, from that file, on the next proxy start.
_SCHEMA = 2

#: The one command that releases a hold. Printed by every surface that reports one.
RELEASE_CMD = "distil reset --drift-guard"


@dataclass
class LiveDrift:
    """The persisted e-process: the monitor, plus when (sample, wall clock) it tripped.

    ``tripped`` is sticky by construction (the monitor refuses further updates), and it is
    the hold: while it is set, every proxy serves lossless-only. The other way to be held
    is ``corrupt`` — an existing state file that cannot be read — or ``quarantined``, the
    persisted record that a writer found one and moved it aside. A breach may be what the
    unreadable file held, so failing safe is the only answer that cannot lose one. Both
    are cleared only by :func:`release` — ``distil reset --drift-guard`` (or ``--shadow``,
    which also archives the evidence).
    """

    monitor: DriftMonitor
    tripped_at: int = 0  # sample number the trip happened at
    tripped_ts: float = 0.0
    schema: int = _SCHEMA
    corrupt: bool = False  # the file exists and could not be read; never persisted
    quarantined: str = ""  # name the unreadable file was moved to; persisted until release

    @property
    def held(self) -> bool:
        return self.monitor.tripped or self.corrupt or bool(self.quarantined)

    @classmethod
    def fresh(cls) -> LiveDrift:
        return cls(DriftMonitor(alpha=_null_mean(), delta=_budget.BUDGET_DELTA))

    @classmethod
    def load(cls, path: Path | None = None) -> LiveDrift:
        """Read-only. Missing or bet against another budget → a fresh monitor. An existing
        file that cannot be read or parsed → a fresh monitor marked ``corrupt`` (held)."""
        p = path or _state_path()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (FileNotFoundError, NotADirectoryError):
            return cls.fresh()
        except (OSError, ValueError, TypeError):
            return cls._corrupt()
        if not isinstance(raw, dict):
            return cls._corrupt()
        state = cls.fresh()
        mon = state.monitor
        # A budget change invalidates the accumulated capital — it was bet against a
        # different null. Start over rather than carry a number that means nothing.
        if raw.get("alpha") != mon.alpha or raw.get("delta") != mon.delta:
            return state
        try:
            for f in _FIELDS:
                if f in raw:
                    setattr(mon, f, type(getattr(mon, f))(raw[f]))
            state.tripped_at = int(raw.get("tripped_at") or 0)
            state.tripped_ts = float(raw.get("tripped_ts") or 0.0)
            state.schema = int(raw.get("v") or 1)
            state.quarantined = str(raw.get("quarantined") or "")
            if mon.n < 0 or state.tripped_at < 0:
                raise ValueError("negative counter")
        except (TypeError, ValueError):
            # Well-formed JSON with a corrupt field is still a corrupt state file.
            return cls._corrupt()
        return state

    @classmethod
    def _corrupt(cls) -> LiveDrift:
        state = cls.fresh()
        state.corrupt = True
        return state

    def advance(self, diffs: list[int]) -> bool:
        """Fold paired differences in order. Returns True iff this call tripped it."""
        was = self.monitor.tripped
        for d in diffs:
            if self.monitor.update(paired_loss(d)):
                break
        if self.monitor.tripped and not was:
            self.tripped_at, self.tripped_ts = self.monitor.n, time.time()
            return True
        return False

    def _write(self, p: Path) -> bool:
        """Atomic, durable replace; the CALLER holds the lock. False if it failed.

        fsync before the rename, then the rename is atomic: a reader sees either the old
        state or the new one, and a crash cannot leave a zero-length file behind the name.
        """
        if self.corrupt and not self.quarantined:
            # The unreadable file on disk is the only copy of whatever it recorded, and it
            # is what keeps every reader held. Never replace it with an un-held state.
            return False
        payload: dict[str, object] = {f: getattr(self.monitor, f) for f in _FIELDS}
        payload.update(tripped_at=self.tripped_at, tripped_ts=self.tripped_ts, v=_SCHEMA)
        if self.quarantined:
            payload["quarantined"] = self.quarantined
        tmp = p.with_name(p.name + ".tmp")
        try:
            from . import _filelock, atrest

            with open(tmp, "w", encoding="utf-8", opener=atrest.owner_only) as fh:
                fh.write(json.dumps(payload))
                fh.flush()
                os.fsync(fh.fileno())
            _filelock.replace_retrying(tmp, p)
            return True
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            return False

    def line(self, bound: float | None = None) -> str:
        """One sentence. ``bound`` is the (1−δ) risk bound printed beside it.

        Same floor discipline as every other surface: an e-value off a handful of
        samples is a number wearing a verdict, so below it we say how far along we are
        and claim nothing. A trip is the exception — it is an anytime-valid statement at
        any n, and it is holding the proxy, which the reader must be told.

        "intact" is only printed when the bound beside it is inside the budget. The
        e-process not having tripped means "no breach proven", which is a weaker claim
        than "within budget".
        """
        from .shadow import VERDICT_MIN_AB

        m = self.monitor
        if not m.tripped and (self.corrupt or self.quarantined):
            where = f" (moved aside as {self.quarantined})" if self.quarantined else ""
            return (
                f"HELD — the drift state file was unreadable{where}, so a breach it may "
                f"have recorded cannot be ruled out{held_note()}"
            )
        if m.tripped:
            when = (
                time.strftime(" %Y-%m-%d %H:%M", time.localtime(self.tripped_ts))
                if self.tripped_ts
                else ""
            )
            return (
                f"BREACHED at sample {self.tripped_at}{when} "
                f"(e-value {m.evalue:.1f} ≥ {1.0 / m.delta:.0f}, n={m.n}){held_note()}"
            )
        if m.n < VERDICT_MIN_AB:
            return f"not enough samples yet ({m.n}/{VERDICT_MIN_AB})"
        stat = f"e-value {m.evalue:.2f}, n={m.n}"
        if bound is not None and not _budget.within_budget(bound):
            return (
                f"unproven ({stat}) — no breach detected, but the bound is above the "
                f"{_budget.budget_pct()} budget"
            )
        within = "" if bound is None else f" — bound within the {_budget.budget_pct()} budget"
        return f"intact ({stat}){within}"


# --------------------------------------------------------------------------- #
# The one writer
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _locked(p: Path) -> Iterator[None]:
    from . import _filelock

    with contextlib.suppress(OSError):
        p.parent.mkdir(parents=True, exist_ok=True)
    with _filelock.locked(p):
        yield


def fold(diffs: list[int], *, path: Path | None = None) -> LiveDrift:
    """Fold ``diffs`` into THE e-process: load, bet, save, all under one file lock.

    Loading inside the lock is what makes this one e-process rather than one per process:
    a proxy that restarted, a hot-swap worker and a second wrap all continue the same
    capital, so no row is ever bet twice and no restart re-branches from stale capital.
    The trip's receipt is written inside the same critical section, so two processes
    crossing the threshold together produce exactly one — as long as the advisory lock is
    obtainable. ``_filelock.locked`` fails open to no lock (unlockable filesystem), and
    then two processes crossing together can each write a trip receipt; the hold itself
    is unaffected. Archives (``.reset-*``, ``.corrupt-*``) accumulate at human pace — one
    per release or corruption — and are never pruned, because they are evidence.
    """
    p = path or _state_path()
    with _locked(p):
        state = _load_for_write(p)
        tripped_now = state.advance(diffs)
        state._write(p)
        if tripped_now:
            _trip_receipt(state)
    return state


def _load_for_write(p: Path) -> LiveDrift:
    """:meth:`LiveDrift.load`, for a writer holding the lock: an unreadable file is COPIED
    aside (it is evidence) and the caller then writes a HELD state over it that records
    where the copy went.

    Copy, not move: until the held replacement is durably in place, ``drift.json`` is
    still the unreadable file, and an unreadable file reads as held. A move would leave a
    window — and, if the replacement write failed, a permanent gap — in which the state
    is MISSING, which every watcher reads as released. If the copy fails, ``quarantined``
    stays empty and :meth:`LiveDrift._write` refuses to overwrite the only copy.
    The name carries nanoseconds, and an existing archive is never overwritten: a clash
    takes the next ``-N`` suffix. Nanoseconds alone are not unique — Windows' clock ticks
    every ~15.6 ms, so two quarantines in one tick got the same name, the link failed and
    the copy fallback hit the first archive (a hard link to the same file) and gave up.
    """
    state = LiveDrift.load(p)
    if state.corrupt:
        base = f"{p.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        try:
            for n in range(100):
                dest = p.with_name(base if n == 0 else f"{base}-{n}")
                try:
                    _copy_aside(p, dest)
                except FileExistsError:
                    continue
                state.quarantined = dest.name
                log.warning("distil drift: unreadable %s copied to %s; holding", p.name, dest.name)
                break
            else:
                raise FileExistsError(f"{base}: every archive name is taken")
        except OSError:
            log.warning("distil drift: %s is unreadable and could not be copied; holding", p)
    return state


def _copy_aside(src: Path, dest: Path) -> None:
    """Hard-link *src* to *dest*, else copy it; ``FileExistsError`` if *dest* exists.

    The copy is created exclusively (``xb``), so neither branch can overwrite an archive.
    """
    try:
        os.link(src, dest)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # no hard links here (FAT, some network shares): copy instead
    with open(src, "rb") as fi, open(dest, "xb") as fo:
        shutil.copyfileobj(fi, fo)
    shutil.copystat(src, dest)


def _bootstrap(path: Path | None = None) -> LiveDrift:
    """Proxy start: carry existing shadow evidence into the e-process exactly once.

    Folds ``shadow.jsonl`` (file order, from ``K_0 = 1``) in two cases, and only these:

    * an existing pre-schema-2 ``drift.json`` — the old exit-time fold, migrated; and
    * a MISSING ``drift.json`` with no release archive (``drift.json.reset-*``) beside
      it — a first-ever state, i.e. a fresh install or first upgrade. Starting at zero
      there would discard real harm evidence the machine already collected.

    A missing file WITH a release archive means the user released before (and the fresh
    state was deleted, or its write failed): that is a fresh e-process, never a re-fold,
    or the rows that were just released would re-trip the hold. A quarantined
    ``drift.json.corrupt-*`` is not a release. If the user deletes both ``drift.json``
    and every release archive, nothing distinguishes that machine from a first install,
    so it re-bootstraps from ``shadow.jsonl`` — acceptable: the archive is the release's
    only durable trace, and ``distil reset --shadow`` archives the ledger too.

    An unreadable file is quarantined and held (:func:`_load_for_write`).
    """
    p = path or _state_path()
    with _locked(p):
        existed = p.exists()
        current = _load_for_write(p)
        if existed:
            migrate = not current.held and current.schema < _SCHEMA
        else:
            migrate = not any(p.parent.glob(p.name + ".reset-*"))
        if not migrate:
            if current.quarantined:
                current._write(p)
            return current
        from .shadow import ShadowLedger

        state = LiveDrift.fresh()
        tripped_now = state.advance(list(ShadowLedger.load(current_only=True).paired_diffs))
        state._write(p)
        if tripped_now:
            _trip_receipt(state)
    return state


def release(stamp: str, path: Path | None = None) -> bool:
    """Archive the e-process (and its hold) and start a fresh one. True if one existed.

    Archived, not deleted — the trip is evidence, and its receipt stays on the chain.
    Raises ``OSError`` when either step fails, so the command can say it did not work
    rather than claim a release it did not make.
    """
    p = path or _state_path()
    with _locked(p):
        existed = p.exists()
        if existed:
            p.rename(p.with_name(p.name + f".reset-{stamp}"))
        if not LiveDrift.fresh()._write(p):
            raise OSError(f"could not write a fresh drift state to {p}")
    return existed


def _trip_receipt(state: LiveDrift) -> None:
    """One hash-chained receipt naming the trip, so the moment compression was held sits
    in the same third-party-verifiable artifact as every request it affected.

    An event row, not a request: counts are zero, no handles, and ``reversible`` is
    False because no compression happened that could be reversed.
    """
    try:
        from . import receipts

        m = state.monitor
        receipts.append(
            receipts.Receipt(
                ts=state.tripped_ts,
                request_id=hashlib.sha256(f"drift-trip:{state.tripped_ts}".encode()).hexdigest()[
                    :16
                ],
                session=str(os.environ.get("DISTIL_SESSION") or ""),
                model="-",
                mode="drift-trip",
                tokens_original=0,
                tokens_compressed=0,
                reversible=False,
                certificate=(
                    f"drift e-value {m.evalue:.4f} >= {1.0 / m.delta:.0f} at n={m.n}; "
                    f"budget {_budget.BUDGET_ALPHA}; lossless-only until {RELEASE_CMD}"
                ),
            )
        )
    except Exception:  # noqa: BLE001 — the receipt is a record of the trip, not the trip
        log.debug("drift-trip receipt failed", exc_info=True)


# --------------------------------------------------------------------------- #
# The guard — a breach that acts
# --------------------------------------------------------------------------- #
#
# Scope is GLOBAL (drift.json under DISTIL_HOME), not per-session. The e-process and the
# budget are already global, and the breach is a statement about this machine's operating
# point, not about one wrap. A per-session hold would let the very next `distil wrap`
# resume lossy compression right after a certified breach. The multi-tenant gateway is
# deliberately NOT held by it — see docs/adr/0016-drift-guard-scope.md.

#: Opt-out, same shape as DISTIL_NO_LEDGER: the alarm still trips and still prints,
#: but the proxy keeps compressing.
GUARD_OPT_OUT = "DISTIL_NO_DRIFT_GUARD"


def guard_disabled() -> bool:
    return os.environ.get(GUARD_OPT_OUT) == "1"


def held_note() -> str:
    """The clause every surface appends to a trip — including the exact release command."""
    if guard_disabled():
        return f" — guard off ({GUARD_OPT_OUT}=1): compression was NOT held"
    return (
        " — compression held at lossless-only; recalibrate (distil calibrate), "
        f"then release: {RELEASE_CMD}"
    )


def held_now() -> bool:
    """Is compression being held right now? One small file read; for the status line."""
    return not guard_disabled() and LiveDrift.load().held


_WATCHERS: list[DriftGuard] = []
_WATCHERS_LOCK = threading.Lock()


def stop_watchers() -> None:
    """Stop every watcher thread this process started (tests, embedded servers)."""
    with _WATCHERS_LOCK:
        for g in _WATCHERS:
            g.stop()
        _WATCHERS.clear()


@dataclass
class DriftGuard:
    """A proxy's view of the one e-process. One attribute read per request.

    ``held`` is refreshed three ways, none on the request path: by :meth:`observe` (this
    proxy's own shadow verdicts, in the shadow thread), and by a daemon watcher that
    stats ``drift.json`` every :attr:`POLL_S` seconds and re-reads it only when it
    changed — so a long-lived proxy notices another process's trip, and a release, without
    a restart.
    """

    held: bool = False
    disabled: bool = False
    _seen: tuple[int, int] | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    #: Watcher interval. Coarse on purpose: a trip already took hundreds of samples.
    POLL_S = 30.0

    @classmethod
    def start(cls, *, watch: bool = True, write: bool = True) -> DriftGuard:
        """Never raises: a guard that cannot load starts un-held.

        ``write=False, watch=False`` is the read-only guard for diagnostics (``distil
        doctor``'s self-test): it reads the hold and nothing else — no migration write, no
        quarantine, no watcher thread.
        """
        g = cls(disabled=guard_disabled())
        try:
            g.held = (_bootstrap() if write else LiveDrift.load()).held
            if write:
                g.refresh()
        except Exception:  # noqa: BLE001 — the alarm must never stop the proxy starting
            log.debug("drift guard start failed", exc_info=True)
        if watch:
            with _WATCHERS_LOCK:
                _WATCHERS.append(g)
            threading.Thread(target=g._watch, name="distil-drift-guard", daemon=True).start()
        return g

    @property
    def engaged(self) -> bool:
        """Serve lossless-only? The hot-path check — no I/O, no lock."""
        return self.held and not self.disabled

    def refresh(self) -> None:
        """Re-read the state iff the file changed since the last look. Never raises."""
        try:
            try:
                st = _state_path().stat()
                key: tuple[int, int] | None = (st.st_mtime_ns, st.st_size)
            except FileNotFoundError:
                key = None
            if key == self._seen:
                return
            self._seen = key
            self._set(LiveDrift.load().held)
        except Exception:  # noqa: BLE001 — the alarm must never break the proxy
            log.debug("drift guard refresh failed", exc_info=True)

    def observe(self, diff: int) -> None:
        """Fold this proxy's paired shadow difference into the one e-process.

        Fail-open: an exception here is logged and swallowed. The request this verdict
        came from was served long ago; the alarm must never be why the next one is not.
        """
        try:
            self._set(fold([diff]).held)
        except Exception:  # noqa: BLE001 — the alarm must never break the proxy
            log.debug("drift guard observe failed", exc_info=True)

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(self.POLL_S):
            self.refresh()

    def _set(self, tripped: bool) -> None:
        if tripped != self.held:
            log.warning(
                "distil drift guard: %s",
                f"tripped — serving lossless-only until {RELEASE_CMD}"
                if tripped
                else "released — lossy compression resumes",
            )
        self.held = tripped

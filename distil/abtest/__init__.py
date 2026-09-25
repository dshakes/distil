"""Task-level A/B: does distil lower what a task costs, measured causally?

Per-request "tokens saved" cannot answer that across a model change: a new model
changes turns per task, cost per turn and cost per task on its own, and a
before/after comparison books all of it to distil. The only comparison a model
change cannot contaminate is a *concurrent randomised* one — so a small, disclosed
fraction of new wrap sessions (default 5%) is held out: forwarded byte-for-byte,
no compression, no shaping, no cold-point, no prefix replay. Both arms see the same
models at the same time, so a model change moves both and cancels in the contrast.

This module is the assignment. The unit is the SESSION (``DISTIL_SESSION``): the
arm is ``HMAC-SHA256(install secret, session id) < rate``, deterministic, so a
nested or resumed wrap that inherits the id keeps its arm, and nobody who does not
hold the secret can predict or steer it. The secret never leaves this machine and
is not the census id (``distil census off`` must not re-randomise anyone).

Outcomes, statistics and the report live in :mod:`.outcomes`, :mod:`.stats` and
:mod:`.report`. Everything is content-free: ids, counts, dollars.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .report import ABSummary

log = logging.getLogger(__name__)

#: The disclosed default. Small enough that the cost is a rounding error on a year of
#: use (see "cost of the holdout" in ``distil ab``), large enough to resolve a ~10%
#: effect within a few hundred sessions.
DEFAULT_RATE = 0.05
#: A holdout above one half would compress LESS traffic than it holds out.
MAX_RATE = 0.5
RATE_ENV = "DISTIL_HOLDOUT_RATE"

DISTIL = "distil"
HOLDOUT = "holdout"


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def config_path() -> Path:
    return _home() / "ab.json"


def _read_config() -> dict[str, object]:
    try:
        raw = json.loads(config_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("distil ab: unreadable %s (%s); using defaults", config_path(), exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_config(cfg: dict[str, object]) -> None:
    from .. import atrest

    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    atrest.write_owner_only(p, (json.dumps(cfg, sort_keys=True) + "\n").encode())


def parse_rate(raw: object) -> float:
    """A holdout rate from config or env. Raises ``ValueError`` on anything else.

    Accepts ``0.05``, ``"0.05"``, ``"5%"`` and the spellings of off. Out of range is an
    error, not a clamp: a user who typed ``DISTIL_HOLDOUT_RATE=5`` meant 5%, not 50%.
    """
    if isinstance(raw, bool):
        raise ValueError(f"not a rate: {raw!r}")
    s = str(raw).strip().lower()
    if s in ("off", "no", "false", "none", "disable", "disabled"):
        return 0.0
    val = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
    if not 0.0 <= val <= MAX_RATE:
        raise ValueError(f"holdout rate {val:g} outside [0, {MAX_RATE:g}]")
    return val


def holdout_rate() -> float:
    """The rate in force: ``DISTIL_HOLDOUT_RATE`` > ``ab.json`` > :data:`DEFAULT_RATE`.

    An env value that does not parse DISABLES the holdout (with a warning) rather than
    falling back to the default: someone who set the variable was trying to change the
    holdout, and the only reading that cannot go against an opt-out is "off".
    """
    env = os.environ.get(RATE_ENV)
    if env is not None and env.strip():
        try:
            return parse_rate(env)
        except ValueError as exc:
            log.warning(
                "distil ab: %s=%r not understood (%s); holdout disabled", RATE_ENV, env, exc
            )
            return 0.0
    cfg = _read_config().get("holdout_rate")
    if cfg is not None:
        try:
            return parse_rate(cfg)
        except ValueError as exc:
            log.warning("distil ab: holdout_rate in %s: %s; holdout disabled", config_path(), exc)
            return 0.0
    return DEFAULT_RATE


def set_holdout_rate(rate: float) -> None:
    """Persist the rate (``distil ab --holdout-rate``). Keeps the secret."""
    cfg = _read_config()
    cfg["holdout_rate"] = parse_rate(rate)
    _write_config(cfg)


def _secret() -> bytes:
    """The per-install assignment key, created 0600 on first use.

    Two wraps starting together on a fresh install can both create one; the loser's
    session is assigned under a key that is then overwritten. That session's arm is
    still in its manifest (the manifest, not a recomputation, is what a resumed wrap
    reads), so nothing is misattributed. ponytail: last-writer-wins, O_EXCL if it matters.
    """
    cfg = _read_config()
    key = cfg.get("secret")
    if isinstance(key, str) and len(key) >= 32:
        return key.encode()
    key = secrets.token_hex(32)
    cfg["secret"] = key
    try:
        _write_config(cfg)
    except OSError as exc:
        log.warning("distil ab: cannot persist the assignment key (%s)", exc)
    return key.encode()


def keyed_hash(value: str) -> str:
    """Content-free, install-keyed fingerprint (used for the workspace covariate key)."""
    return hmac.new(_secret(), value.encode(), hashlib.sha256).hexdigest()[:16]


def bucket(sid: str) -> float:
    """The session's uniform draw in ``[0, 1)``."""
    h = hmac.new(_secret(), sid.encode(), hashlib.sha256).hexdigest()[:13]
    return int(h, 16) / float(1 << 52)


@dataclass(frozen=True)
class Assignment:
    arm: str  # "distil" | "holdout" | "" (not randomised: disabled, or no session)
    rate: float

    def manifest(self) -> dict[str, object]:
        return {"arm": self.arm, "rate": self.rate}


def assign(sid: str | None, previous: dict[str, object] | None = None) -> Assignment:
    """Assign session *sid*. ``previous`` is the manifest already on disk for it, if any.

    A session that was already assigned keeps its arm — including if the rate has
    changed since, because re-drawing mid-session would put one session in both arms.
    """
    ab = previous.get("ab") if previous else None
    if isinstance(ab, dict):
        if ab.get("arm") in (DISTIL, HOLDOUT, ""):
            try:
                return Assignment(str(ab["arm"]), float(ab.get("rate") or 0.0))
            except (TypeError, ValueError):
                pass
    rate = holdout_rate()
    if not sid or rate <= 0.0:
        return Assignment("", rate)
    return Assignment(HOLDOUT if bucket(sid) < rate else DISTIL, rate)


def disclosure(rate: float | None = None) -> str:
    """The one sentence every surface that mentions the holdout prints."""
    r = holdout_rate() if rate is None else rate
    if r <= 0:
        return (
            "A/B holdout: off. distil cannot measure its own effect on task cost "
            f"(re-enable: distil ab --holdout-rate {DEFAULT_RATE:g})."
        )
    return (
        f"A/B holdout: {r * 100:g}% of new wrap sessions run uncompressed so distil can "
        "measure what it saves per task, causally (distil ab). "
        f"Opt out: distil ab --holdout-rate 0, or {RATE_ENV}=0."
    )


def abtest_summary(window: float | None = None) -> ABSummary:
    """Lazy re-export of :func:`distil.abtest.report.abtest_summary`."""
    from .report import abtest_summary as _summary

    return _summary(window)


__all__ = [
    "DEFAULT_RATE",
    "DISTIL",
    "HOLDOUT",
    "RATE_ENV",
    "Assignment",
    "abtest_summary",
    "assign",
    "bucket",
    "disclosure",
    "holdout_rate",
    "keyed_hash",
    "parse_rate",
    "set_holdout_rate",
]

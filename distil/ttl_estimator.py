"""Learned per-(provider, model, ttl class) prompt-cache lifetime, for cold-point (ADR 0014).

Evidence from usage after an idle gap *g* (nothing in flight, one clean conversation):

* cache_read > 0 -> the entry was alive after *g*: ``lo = max(lo, g)`` (a LOWER bound).
* cache_read == 0 on a turn that evicted nothing new -> ``hi = min(hi, g)`` (UPPER bound).

Decision rule: ``effective = max(documented, lo)`` once >= ``MIN_HITS`` hits are on file,
else the documented TTL. The upper bound is recorded but never used to SHORTEN the TTL: a
miss can also be an eviction, a mutated prefix or a 5m/1h breakpoint mix, and acting on it
would bust a warm cache, while ignoring it only skips a recompaction. Gaps are measured from
when distil last finished forwarding, so they under-state idleness (also the safe side).
State is content-free numbers under ``$DISTIL_HOME/cache_ttl.json``; fail-open.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("distil.ttl_estimator")

MIN_HITS = 3  # identity (documented TTL) until this many hits are on file
MIN_GAP_S = 30.0  # shorter gaps say nothing about a TTL measured in minutes
MAX_TTL_S = 86400.0  # nothing documented lives longer; a bigger "bound" is noise
_MAX_KEYS = 256

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"path": None, "data": {}}  # data: {key: [lo, hi|None, hits]}


def _path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "cache_ttl.json"


def tclass(ttl_s: float) -> str:
    return f"{int(ttl_s)}s"


def _key(provider: str, model: str, ttl_s: float) -> str:
    return f"{provider}|{model}|{tclass(ttl_s)}"


def _data() -> dict[str, list[Any]]:
    """Per-process copy of the file, re-read when DISTIL_HOME changes. Caller holds _LOCK."""
    p = _path()
    if _STATE["path"] != str(p):
        data: dict[str, list[Any]] = {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            for k, v in raw.get("keys", {}).items():
                lo, hi, hits = v
                if isinstance(k, str) and float(lo) >= 0 and int(hits) >= 0:
                    data[k] = [
                        min(float(lo), MAX_TTL_S),
                        None if hi is None else float(hi),
                        int(hits),
                    ]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError):
            log.debug("cache-ttl state unreadable; starting empty", exc_info=True)
        _STATE.update(path=str(p), data=data)
    return _STATE["data"]


def bounds(provider: str, model: str, ttl_s: float) -> tuple[float, float | None, int]:
    """``(lo, hi, hits)`` observed so far; ``(0, None, 0)`` when nothing is known."""
    with _LOCK:
        lo, hi, hits = _data().get(_key(provider, model, ttl_s), [0.0, None, 0])
        return lo, hi, hits


def effective_ttl(provider: str, model: str, documented_s: float) -> float:
    """The TTL cold-point should wait out: never below *documented_s*, only extended by
    evidence (see module docstring). inf and non-positive pass through unchanged."""
    if not 0 < documented_s < MAX_TTL_S:
        return documented_s
    lo, _hi, hits = bounds(provider, model, documented_s)
    return max(documented_s, lo) if hits >= MIN_HITS else documented_s


def observe(
    provider: str, model: str, ttl_s: float, gap_s: float, cache_read: int, *, evicted: bool = False
) -> None:
    """Record one post-idle-gap usage observation. *evicted*: this turn rewrote the prefix
    itself, so a miss proves nothing. Fail-open: never raises."""
    if gap_s < MIN_GAP_S or not 0 < ttl_s < MAX_TTL_S:
        return
    if cache_read <= 0 and evicted:
        return
    try:
        with _LOCK:
            data = _data()
            k = _key(provider, model, ttl_s)
            lo, hi, hits = data.get(k, [0.0, None, 0])
            old = (lo, hi, min(hits, MIN_HITS))
            if cache_read > 0:
                lo, hits = max(lo, min(gap_s, MAX_TTL_S)), hits + 1
            else:
                hi = gap_s if hi is None else min(hi, gap_s)
            if (lo, hi, min(hits, MIN_HITS)) == old:
                return  # nothing new; no write
            data[k] = [lo, hi, hits]
            while len(data) > _MAX_KEYS:
                data.pop(next(iter(data)))
            _write(data)
    except Exception:  # noqa: BLE001 — learning is an optimisation; never break a request
        log.debug("cache-ttl observation dropped", exc_info=True)


def _write(data: dict[str, list[Any]]) -> None:
    from . import _filelock
    from .config_wrap import _atomic_write_secure

    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _filelock.locked(p):
        # Merge with other workers (hot-swap): widen lo, narrow hi, keep more hits.
        try:
            disk = json.loads(p.read_text(encoding="utf-8")).get("keys", {})
        except (OSError, ValueError, AttributeError):
            disk = {}
        for k, v in disk.items():
            try:
                lo, hi, hits = float(v[0]), v[1], int(v[2])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            m = data.setdefault(k, [lo, hi, hits])
            m[0] = max(m[0], lo)
            m[1] = hi if m[1] is None else (m[1] if hi is None else min(m[1], hi))
            m[2] = max(m[2], hits)
        blob = json.dumps({"version": 1, "keys": data}, separators=(",", ":")).encode()
        _atomic_write_secure(p, blob)


def reset() -> None:
    """Drop the in-memory copy (tests)."""
    with _LOCK:
        _STATE.update(path=None, data={})

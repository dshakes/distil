"""Content-free integration-surface counters — which door traffic came through.

Answers "are SDKs / headless clients actually used?" for the opt-in census
(schema 3) without touching content or identity: two small count dicts,

    surfaces: wrap | proxy | gateway   (how distil was launched)
    shapes:   anthropic | openai-chat | openai-responses | gemini
              (which API wire format the request used)

The proxy bumps an in-memory counter per compressible request and persists a
merged snapshot to ``~/.distil/surfaces.json`` every ``FLUSH_EVERY`` bumps
(plus a merge-on-read), under an exclusive flock so wrap + worker + gateway
processes never clobber each other. Everything is fail-open: a counting
problem must never affect a request.

The surface label comes from ``DISTIL_SURFACE``, set by the launching CLI
command (wrap sets it before spawning the proxy; the hot-swap worker inherits
it). Unset — e.g. an embedded/library use — counts as "proxy".
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

SURFACES = frozenset({"wrap", "proxy", "gateway"})
SHAPES = frozenset({"anthropic", "openai-chat", "openai-responses", "gemini"})
FLUSH_EVERY = 20

_PATH_TO_SHAPE = {
    "/v1/messages": "anthropic",
    "/v1/chat/completions": "openai-chat",
    "/v1/responses": "openai-responses",
}

_lock = threading.Lock()
_pending: dict[str, dict[str, int]] = {"surfaces": {}, "shapes": {}}
_since_flush = 0


def store_path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "surfaces.json"


def shape_for_path(path: str) -> str:
    """Map a compressible request path to its API shape ("gemini" fallback —
    the proxy only calls this for paths it already decided to compress)."""
    return _PATH_TO_SHAPE.get(path, "gemini")


def current_surface() -> str:
    s = os.environ.get("DISTIL_SURFACE", "")
    return s if s in SURFACES else "proxy"


def bump(path: str) -> None:
    """Count one compressible request. Cheap (in-memory), throttled persist."""
    global _since_flush
    try:
        shape = shape_for_path(path)
        surface = current_surface()
        with _lock:
            _pending["shapes"][shape] = _pending["shapes"].get(shape, 0) + 1
            _pending["surfaces"][surface] = _pending["surfaces"].get(surface, 0) + 1
            _since_flush += 1
            if _since_flush >= FLUSH_EVERY:
                _flush_locked()
    except Exception:  # noqa: BLE001 — counters must never affect a request
        pass


def flush() -> None:
    """Persist pending counts now (session teardown)."""
    try:
        with _lock:
            _flush_locked()
    except Exception:  # noqa: BLE001
        pass


def _flush_locked() -> None:
    """Merge pending into the on-disk snapshot under an exclusive flock.
    Caller holds ``_lock``."""
    global _since_flush
    if not (_pending["shapes"] or _pending["surfaces"]):
        return
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # no flock (e.g. Windows): lossy merge beats no data
        f.seek(0)
        try:
            disk = json.loads(f.read() or "{}")
        except json.JSONDecodeError:
            disk = {}
        for group in ("surfaces", "shapes"):
            merged = disk.get(group) or {}
            for k, v in _pending[group].items():
                merged[k] = int(merged.get(k, 0)) + v
            disk[group] = merged
        f.seek(0)
        f.truncate()
        json.dump(disk, f, sort_keys=True)
    _pending["surfaces"] = {}
    _pending["shapes"] = {}
    _since_flush = 0


def snapshot() -> dict:
    """Lifetime counts for the census: disk + pending, allowlisted keys only."""
    try:
        with _lock:
            try:
                disk = json.loads(store_path().read_text(encoding="utf-8") or "{}")
            except (OSError, json.JSONDecodeError):
                disk = {}
            out: dict[str, dict[str, int]] = {}
            for group, allow in (("surfaces", SURFACES), ("shapes", SHAPES)):
                merged = {}
                for src in (disk.get(group) or {}, _pending[group]):
                    for k, v in src.items():
                        if k in allow and isinstance(v, (int, float)) and v >= 0:
                            merged[k] = merged.get(k, 0) + int(v)
                out[group] = merged
            return out
    except Exception:  # noqa: BLE001
        return {"surfaces": {}, "shapes": {}}

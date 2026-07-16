"""Gateway key management: issue, list, revoke, and verify bearer keys.

Keys are ``dsk-`` prefixed URL-safe tokens shown ONCE at issue time; only their
sha256 hash is persisted to disk.  The key file is chmod 0600 and written
atomically (tmp + rename), following the same pattern as ledger.py and
gateway_state.py.

Why a separate file from gateway_state.json: the key store has a CLI-driven
write pattern (issue/revoke) and a per-request read pattern (O(1) hash lookup).
Separating them avoids write-contention with the accounting checkpoint and lets
permissions be tightened independently.

Shared-deployment note
----------------------
``record_restore`` (called inside ``compress_messages``) persists digest handles
to ``~/.distil/restore/`` — a gateway-wide (not per-tenant) disk store.  The
gateway has no expand HTTP endpoint, so no tenant can retrieve another tenant's
original content through the gateway API.  Access is still possible via the
``distil expand`` CLI or the MCP server if an attacker can read the restore
directory (which is chmod 0600, owner-only) — that is a local-access problem,
not a gateway-key problem.  Documented in THREAT_MODEL.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl as _fcntl  # POSIX advisory locking; absent on Windows

    _HAVE_FCNTL = True
except ImportError:
    _HAVE_FCNTL = False

_KEY_PREFIX = "dsk-"
_KEYS_FILE = "gateway_keys.json"
# Reload from disk at most once per window (picks up CLI-issued keys while gateway runs).
_CACHE_TTL_SECS = 2.0


def _keys_path() -> Path:
    """Canonical location, honoring DISTIL_HOME."""
    home = os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))
    return Path(home) / _KEYS_FILE


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class KeyRecord:
    """One issued key.  Never contains the raw token — only the tenant label and limits."""

    id: str  # short operator-visible reference, e.g. ``gk_a1b2c3d4``
    tenant: str  # tenant label for accounting
    created: float  # unix timestamp
    revoked: float | None  # unix timestamp if revoked, else None
    rpm: int | None  # per-minute request cap (None = unlimited)
    daily_tokens: int | None  # per-UTC-day token quota (None = unlimited)

    @property
    def is_active(self) -> bool:
        return self.revoked is None


class GatewayKeyStore:
    """Thread-safe key store backed by a chmod-0600 JSON file.

    Lookup hashes the raw key on every call and checks an in-memory cache that
    is refreshed from disk whenever the file mtime changes or ``_CACHE_TTL_SECS``
    has elapsed — this lets the CLI issue/revoke keys while the gateway is
    running without a restart.

    The in-memory cache maps ``sha256(raw_key) -> KeyRecord``.  The raw key
    never touches the cache.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _keys_path()
        import threading

        self._lock = threading.Lock()
        # hash -> record; refreshed lazily from disk
        self._cache: dict[str, KeyRecord] = {}
        self._mtime: float = 0.0
        self._last_load: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_locked(self) -> None:
        """Reload from disk if stale.  Must be called under ``self._lock``."""
        now = time.monotonic()
        if now - self._last_load < _CACHE_TTL_SECS:
            return
        self._last_load = now
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        loaded: dict[str, KeyRecord] = {}
        for h, rec in (data.get("keys") or {}).items():
            try:
                loaded[h] = KeyRecord(
                    id=str(rec["id"]),
                    tenant=str(rec["tenant"]),
                    created=float(rec["created"]),
                    revoked=float(rec["revoked"]) if rec.get("revoked") is not None else None,
                    rpm=int(rec["rpm"]) if rec.get("rpm") is not None else None,
                    daily_tokens=(
                        int(rec["daily_tokens"]) if rec.get("daily_tokens") is not None else None
                    ),
                )
            except (KeyError, TypeError, ValueError):
                continue  # skip corrupt entries
        self._cache = loaded

    def _save_locked(self, cache: dict[str, KeyRecord]) -> None:
        """Atomically persist ``cache`` to disk.  Must be called under ``self._lock``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "keys": {
                h: {
                    "id": r.id,
                    "tenant": r.tenant,
                    "created": r.created,
                    "revoked": r.revoked,
                    "rpm": r.rpm,
                    "daily_tokens": r.daily_tokens,
                }
                for h, r in cache.items()
            },
        }
        tmp = self._path.with_name(self._path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            if _HAVE_FCNTL:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
            f.write(json.dumps(data, indent=2))
            f.flush()
        os.replace(tmp, self._path)
        self._path.chmod(0o600)
        # Update the in-memory state so the next lookup doesn't re-read immediately.
        self._cache = cache
        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            pass
        self._last_load = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def issue(
        self,
        tenant: str,
        rpm: int | None = None,
        daily_tokens: int | None = None,
    ) -> tuple[str, KeyRecord]:
        """Generate and store a new key.

        Returns ``(raw_key, record)``.  ``raw_key`` is the plaintext token —
        it is shown once to the operator and never stored.
        """
        raw = _KEY_PREFIX + secrets.token_urlsafe(32)
        h = _hash_key(raw)
        rec = KeyRecord(
            id="gk_" + secrets.token_hex(4),
            tenant=tenant,
            created=time.time(),
            revoked=None,
            rpm=rpm,
            daily_tokens=daily_tokens,
        )
        with self._lock:
            self._load_locked()
            cache = dict(self._cache)
            cache[h] = rec
            self._save_locked(cache)
        return raw, rec

    def list_keys(self) -> list[KeyRecord]:
        """Return all key records (no hashes, no raw tokens)."""
        with self._lock:
            self._last_load = 0.0  # force a fresh read
            self._load_locked()
            return list(self._cache.values())

    def revoke(self, key_id: str) -> KeyRecord | None:
        """Mark a key revoked by its ``id`` field.  Returns the updated record or None."""
        with self._lock:
            self._last_load = 0.0
            self._load_locked()
            cache = dict(self._cache)
            for h, rec in cache.items():
                if rec.id == key_id:
                    cache[h] = KeyRecord(
                        id=rec.id,
                        tenant=rec.tenant,
                        created=rec.created,
                        revoked=time.time(),
                        rpm=rec.rpm,
                        daily_tokens=rec.daily_tokens,
                    )
                    self._save_locked(cache)
                    return cache[h]
        return None

    def lookup(self, raw_key: str) -> KeyRecord | None:
        """Validate *raw_key* and return its record, or None if invalid/revoked.

        The raw key is never logged or stored — only its hash is compared.
        """
        if not raw_key.startswith(_KEY_PREFIX):
            return None
        h = _hash_key(raw_key)
        with self._lock:
            self._load_locked()
            rec = self._cache.get(h)
        if rec is None or not rec.is_active:
            return None
        return rec

    def has_active_keys(self) -> bool:
        """True if at least one non-revoked key exists."""
        with self._lock:
            self._load_locked()
            return any(r.is_active for r in self._cache.values())

    def has_any_keys(self) -> bool:
        """True if the key file has ANY records at all (active or revoked).

        Used for the auth-gate decision: once a key has been issued, inbound
        auth stays enforced even if all keys are later revoked.  Revoking all
        keys leaves no valid credential, so every request will 401 — but the
        gate does NOT silently reopen (revoking-all is not equivalent to
        "no keys were ever issued").
        """
        with self._lock:
            self._load_locked()
            return bool(self._cache)

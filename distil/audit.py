"""Append-only audit log for gateway security events.

Enterprise deployments have to answer "who used which key, and what was refused?"
long after the fact — SOC2 CC7.2 and ISO 27001 A.12.4 both require it, and it is
the first thing a security questionnaire asks about a shared gateway.

**Content-free by construction.** An audit record never contains prompt text,
completion text, tool output, or a raw key. It holds the event, the key *id* and
tenant label an operator already sees in ``distil gateway keys list``, and the
outcome. The point is accountability, not surveillance: an audit trail that
captured request bodies would be a bigger liability than the gap it closes.

Failures are swallowed. An unwritable audit file must never take the gateway
down — the same fail-open contract as the savings ledger. Callers that need a
hard guarantee should check :func:`audit_path` at startup.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FCNTL = False

#: Events an operator can filter on. Kept as plain strings so a log stays readable
#: without this module, and so adding one never breaks an existing consumer.
AUTH_OK = "auth.ok"
AUTH_FAIL = "auth.fail"
RATE_LIMITED = "rate.limited"
KEY_ISSUED = "key.issued"
KEY_REVOKED = "key.revoked"


def audit_path() -> Path:
    """Location of the audit log (``$DISTIL_HOME/audit.jsonl``)."""
    home = os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))
    return Path(home) / "audit.jsonl"


def record(event: str, **fields: Any) -> None:
    """Append one audit event.  Best-effort: never raises, never blocks a request.

    ``fields`` should carry only identifiers and outcomes (key_id, tenant, reason,
    status, remote) — never request or response content.
    """
    rec: dict[str, Any] = {"ts": time.time(), "event": event}
    rec.update(fields)
    try:
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # 0600 at creation: the trail names tenants and key ids, so it is not for
        # every local user to read. Opened O_APPEND so concurrent writers cannot
        # overwrite each other's offsets.
        # 0600 at creation, O_APPEND so concurrent writers never clobber offsets.
        with os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8"
        ) as fh:
            if _HAVE_FCNTL:
                # The gateway serves requests concurrently and a record can exceed
                # the atomic-append size; lock so lines never interleave.
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError):
        pass  # ponytail: an audit write must never break the request it describes


def read_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read audit events oldest-first, skipping any corrupt line.

    A truncated final line (killed mid-write) is normal for an append-only log and
    must not make the whole trail unreadable.
    """
    path = audit_path()
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out[-limit:] if limit else out

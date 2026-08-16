"""Subscription quota telemetry — read-only, fail-open, never logged.

On a flat-rate Claude subscription there is no per-token bill, so distil's dollar
figures are notional (see :func:`distil.doctor.subscription_mode`). The currency that
*is* real is the rate-limit window: every token spent is quota that isn't available
for the next task. This module reads that number so subscription savings can be
**measured** rather than asserted.

Source
------
``GET https://api.anthropic.com/api/oauth/usage``, authenticated with the OAuth
access token Claude Code already stores for the logged-in user. The response carries
per-window ``utilization`` — the same numbers the ``/usage`` command shows.

This is an **undocumented internal endpoint**. It is read-only and touches the user's
own account, but it can change shape or disappear without notice, so every failure
mode here resolves to "unavailable" and never to a fabricated number: a wrong quota
reading is worse than no quota reading, because it would silently corrupt the
before/after comparison this module exists to enable.

Scope note: reading your own usage is not model access. Anthropic's consumer terms
restrict *accessing the Services* through automated means (generating completions on
subscription credentials); a read of your own rate-limit meter is the same
information the client already shows you, which is why distil reads it and still
refuses to route subscription traffic through the digest tier.

Secrets
-------
The token is read into a local and used once. It is never logged, never written to a
receipt or ledger, never included in an exception message, and never returned to a
caller. :func:`snapshot` returns counters only.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_BETA_HEADER = "oauth-2025-04-20"
_TIMEOUT_S = 5.0

# The windows the endpoint reports. Order matters for display: the 5-hour window is
# what a user hits mid-session, the 7-day ones are what they hit mid-week.
_WINDOWS = ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet")


@dataclass(frozen=True)
class Window:
    """One rate-limit window.

    The endpoint reports consumption as ``utilization`` — a **percentage float**
    (85.0 means 85% of the window is gone), not a token count. Verified live
    2026-08-16. Dollar fields are present in the payload but are ``null`` on at least
    some plans, so they are surfaced as optional and never relied upon.
    """

    name: str
    utilization_pct: float
    resets_at: str | None = None
    used_dollars: float | None = None
    limit_dollars: float | None = None

    @property
    def remaining_pct(self) -> float:
        return max(0.0, 100.0 - self.utilization_pct)


@dataclass(frozen=True)
class Snapshot:
    """One poll. ``available`` is False when the quota could not be read."""

    windows: tuple[Window, ...] = ()
    available: bool = False
    reason: str = "not polled"

    def window(self, name: str) -> Window | None:
        for w in self.windows:
            if w.name == name:
                return w
        return None


def _token() -> str | None:
    """Resolve the OAuth token, or None. Never logs or returns it to a caller.

    Order mirrors what Claude Code itself does, most explicit first. On macOS the
    credentials usually live in the login keychain rather than on disk, so the file
    check alone reports "no token" on a machine that is in fact logged in.
    """
    env = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env:
        return env

    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    path = Path(config_dir) / ".credentials.json"
    try:
        if path.is_file():
            blob = json.loads(path.read_text(encoding="utf-8"))
            tok = (blob.get("claudeAiOauth") or {}).get("accessToken")
            if isinstance(tok, str) and tok:
                return tok
    except (OSError, ValueError):
        pass  # unreadable or malformed -> fall through, never raise

    if os.uname().sysname == "Darwin":
        try:
            out = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                blob = json.loads(out.stdout)
                tok = (blob.get("claudeAiOauth") or {}).get("accessToken")
                if isinstance(tok, str) and tok:
                    return tok
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    return None


def _coerce_window(name: str, raw: Any) -> Window | None:
    """Pull a window out of the payload, tolerating shape drift.

    Windows the account does not have are ``null`` in the payload (e.g.
    ``seven_day_opus`` on a plan without it), so a missing window is normal and is
    dropped rather than reported as zero usage.
    """
    if not isinstance(raw, dict):
        return None
    util = raw.get("utilization")
    if not isinstance(util, (int, float)):
        return None

    def _money(key: str) -> float | None:
        v = raw.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    resets = raw.get("resets_at") or raw.get("resetsAt")
    return Window(
        name=name,
        utilization_pct=float(util),
        resets_at=resets if isinstance(resets, str) else None,
        used_dollars=_money("used_dollars"),
        limit_dollars=_money("limit_dollars"),
    )


def snapshot(token: str | None = None) -> Snapshot:
    """Poll the usage endpoint. Never raises; never returns the token.

    Returns a Snapshot with ``available=False`` and a short ``reason`` whenever the
    quota cannot be read, so a caller can distinguish "no quota data" from "zero
    usage" — conflating those would invert every before/after comparison.
    """
    tok = token or _token()
    if not tok:
        return Snapshot(reason="no oauth token (not logged in, or metered API key only)")

    req = urllib.request.Request(  # noqa: S310 - fixed https URL, not caller-controlled
        _USAGE_URL,
        headers={
            "Authorization": f"Bearer {tok}",
            "anthropic-beta": _BETA_HEADER,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            if resp.status != 200:
                return Snapshot(reason=f"http {resp.status}")
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Deliberately does not include the response body: it is an authenticated
        # endpoint and the body could echo request context.
        return Snapshot(reason=f"http {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError):
        return Snapshot(reason="unreachable")
    except ValueError:
        return Snapshot(reason="malformed response")

    if not isinstance(payload, dict):
        return Snapshot(reason="unexpected payload")

    windows = tuple(
        w for name in _WINDOWS if (w := _coerce_window(name, payload.get(name))) is not None
    )
    if not windows:
        return Snapshot(reason="no readable windows (endpoint shape changed?)")
    return Snapshot(windows=windows, available=True, reason="ok")


def delta(before: Snapshot, after: Snapshot, window: str = "five_hour") -> float | None:
    """Percentage points of the window consumed between two polls.

    This is the measurement the hook A/B depends on: poll, run a workload, poll
    again, and compare the deltas across arms. Returns ``None`` rather than 0.0 when
    either poll is unusable, so missing data can never read as "saved everything".

    Units are percentage points of the window, because that is what the endpoint
    reports (see :class:`Window`). A window that resets between the two polls yields
    a negative delta; callers comparing arms should discard those runs rather than
    treat them as savings — check ``resets_at`` when precision matters.
    """
    if not (before.available and after.available):
        return None
    b, a = before.window(window), after.window(window)
    if b is None or a is None:
        return None
    return a.utilization_pct - b.utilization_pct

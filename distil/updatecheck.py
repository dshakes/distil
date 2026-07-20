"""Once-a-day update notice on long-lived commands (wrap/proxy).

The gh/Homebrew pattern: check PyPI for a newer version at most every 24h and
print one stderr line if you're behind. This talks ONLY to pypi.org (the same
lookup `distil onboard` already does) — it is not telemetry, sends nothing
about you, and is disclosed in TELEMETRY.md. Opt out: DISTIL_NO_UPDATE_CHECK=1.

Runs in a daemon thread so a slow PyPI can never delay proxy/wrap startup.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

MIN_INTERVAL_S = 24 * 3600


def _last_path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "update-last"


def _due() -> bool:
    if os.environ.get("DISTIL_NO_UPDATE_CHECK", "") not in ("", "0"):
        return False
    try:
        last = float(_last_path().read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        last = 0.0
    return time.time() - last >= MIN_INTERVAL_S


def _check() -> None:
    try:
        _last_path().parent.mkdir(parents=True, exist_ok=True)
        _last_path().write_text(str(int(time.time())), encoding="utf-8")
        from . import __version__
        from .onboard import is_outdated, latest_pypi_version

        latest = latest_pypi_version()
        if latest and is_outdated(__version__, latest):
            print(
                f"  distil {latest} is available (you run {__version__}) — `distil upgrade`",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001 — a version notice must never break the host command
        pass


def maybe_notify() -> bool:
    """Kick off the throttled background check. Returns True if one started."""
    if not _due():
        return False
    threading.Thread(target=_check, daemon=True, name="distil-updatecheck").start()
    return True

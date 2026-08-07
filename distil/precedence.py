"""Detect settings files that silently outrank ``distil wrap``'s environment.

The outage this exists to prevent
--------------------------------
``distil wrap`` starts a proxy on an ephemeral port and exports
``ANTHROPIC_BASE_URL`` for the child agent. But Claude Code reads its own
``settings.json``, and **a base URL there outranks the process environment**. So
if ``distil default --always-on`` (or anything else) has pinned a base URL in
settings, the wrap's export is dead on arrival:

* Best case, the pinned port is a *live* distil — the wrap's own proxy receives
  zero requests, silently, and its savings never move.
* Worst case, the pinned port is **dead** — every session on the machine fails
  with ``API Error: Unable to connect to API (ConnectionRefused)``, an error that
  names the provider rather than distil. The user has no reason to suspect us.

Both were observed in the field. The second is a total outage of the agent, and
it persists across reboots because the settings file is durable while the daemon
that was supposed to answer is not. Detecting this at wrap start is the only
point where we can speak *before* the failure instead of after it.

This module is deliberately dependency-free and read-only: it never edits a
settings file. It answers one question — "will the environment I am about to set
actually be honoured?" — and hands the caller the exact command to fix it.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

__all__ = ["Conflict", "check_claude_settings", "settings_candidates"]

# Highest precedence last: Claude Code resolves later files over earlier ones, so
# the LAST hit is the value the agent will actually use.
_USER_SETTINGS = ("~/.claude/settings.json", "~/.claude/settings.local.json")
_PROJECT_SETTINGS = (".claude/settings.json", ".claude/settings.local.json")


@dataclass(frozen=True)
class Conflict:
    """A settings-file base URL that will override the wrap's environment."""

    path: Path
    env_var: str
    value: str
    reachable: bool

    @property
    def fatal(self) -> bool:
        """True when following this setting cannot possibly work.

        An unreachable pinned URL is not a warning — it is a guaranteed outage
        for every session started on this machine.
        """
        return not self.reachable

    def message(self) -> str:
        where = str(self.path).replace(str(Path.home()), "~")
        if self.fatal:
            return (
                f"{self.env_var}={self.value} in {where} points at a port that is "
                f"NOT accepting connections, and that file outranks the environment "
                f"`distil wrap` sets. Every session will fail with a connection "
                f"error that blames the API.\n"
                f"    Fix (pick one):\n"
                f"      distil default --always-on     # start the service it expects\n"
                f"      distil offboard                # remove the pin, use wrap only"
            )
        return (
            f"{self.env_var}={self.value} in {where} outranks the environment "
            f"`distil wrap` sets, so THIS wrap's proxy will receive no traffic "
            f"(the already-running one on {self.value} handles it instead).\n"
            f"    Running `distil wrap` on top of always-on is redundant — use one."
        )


def settings_candidates(cwd: Path | None = None) -> list[Path]:
    """Settings files that can pin a base URL, in ascending precedence order."""
    cwd = cwd or Path.cwd()
    out = [Path(p).expanduser() for p in _USER_SETTINGS]
    out += [cwd / p for p in _PROJECT_SETTINGS]
    return out


def _reachable(url: str, timeout: float = 0.35) -> bool:
    """Can we open a TCP connection to *url*'s host:port?

    A bare connect, not an HTTP request: we are asking "is anything listening",
    and an auth rejection or a 404 both mean yes. Non-loopback hosts are assumed
    reachable — probing an arbitrary remote at every wrap start would add latency
    and, on a corporate network, look like scanning.
    """
    try:
        u = urlparse(url)
        host, port = u.hostname or "", u.port or (443 if u.scheme == "https" else 80)
    except (ValueError, TypeError):
        return True  # unparseable: not our call to block on
    if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return True
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_claude_settings(
    env_var: str, wrap_url: str = "", *, cwd: Path | None = None
) -> Conflict | None:
    """Return the settings-file entry that will override *env_var*, if any.

    ``wrap_url`` is the URL this wrap is about to export; a settings entry that
    already names it is not a conflict. Returns ``None`` when the wrap's
    environment will be honoured.
    """
    if os.environ.get("DISTIL_IGNORE_SETTINGS_PRECEDENCE"):
        return None
    winner: Conflict | None = None
    for path in settings_candidates(cwd):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        env = data.get("env")
        if not isinstance(env, dict):
            continue
        value = env.get(env_var)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if wrap_url and value.rstrip("/") == wrap_url.rstrip("/"):
            continue  # points at us — no conflict
        # Later files win, so keep overwriting: the last hit is the effective one.
        winner = Conflict(path=path, env_var=env_var, value=value, reachable=_reachable(value))
    return winner

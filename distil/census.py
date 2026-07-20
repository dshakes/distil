"""Opt-in adoption census — content-free, transparent, revocable.

Answers "how many active installs, which versions, how much is the community
saving" with the smallest defensible signal: one JSON object, at most once per
24h, containing ONLY numbers and platform strings (schema below, frozen by
test). It is OFF until the user explicitly opts in (`distil census on`, or
saying yes at `distil onboard`), and two kill switches beat a stored opt-in
unconditionally: the cross-tool `DO_NOT_TRACK=1` convention and
`DISTIL_NO_TELEMETRY=1`.

Design rules (see docs/adr/0002-adoption-telemetry-opt-in-census.md):
- Content never appears here — the payload is built from `ledger.summary()`
  totals (numbers) and platform metadata, nothing else.
- `distil census show` prints the exact payload; nothing is sent.
- Fail-open: a send gets 1.5s and swallows every failure; it must never slow
  or break the proxy/wrap exit path it rides on.
- Revocable: `distil census off` deletes the anonymous install id.

The ingest endpoint activates in a later phase; until it is deployed, sends
fail silently (opt-in state and UX are unaffected).
"""

from __future__ import annotations

import json
import os
import platform
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_ENDPOINT = "https://distil-census.vercel.app/v1/ping"
SEND_TIMEOUT_S = 1.5
MIN_INTERVAL_S = 24 * 3600

# The frozen payload contract — tests/test_census.py refuses any new key, so
# widening the census is a deliberate, reviewed act (and a TELEMETRY.md edit).
# Schema 2 added the usage dimensions: billing, by_model, agents.
PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "install_id",
        "version",
        "os",
        "arch",
        "python",
        "runs",
        "tokens_saved",
        "dollars_saved",
        "billing",
        "by_model",
        "agents",
        "ts",
    }
)

# Agent names travel ONLY through this allowlist — anything else becomes
# "other", so an exotic argv can never leak into the census.
AGENT_ALLOWLIST = frozenset({"claude", "codex", "gemini", "aider"})
MAX_MODELS = 5


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def _consent_path() -> Path:
    return _home() / "census"


def _id_path() -> Path:
    return _home() / "install-id"


def _last_path() -> Path:
    return _home() / "census-last"


def hard_disabled() -> bool:
    """Kill switches that beat a stored opt-in: DO_NOT_TRACK (cross-tool
    convention, consoledonottrack.com) and DISTIL_NO_TELEMETRY."""
    return os.environ.get("DO_NOT_TRACK", "") not in ("", "0") or os.environ.get(
        "DISTIL_NO_TELEMETRY", ""
    ) not in ("", "0")


def consent() -> str | None:
    """Stored decision: "on", "off", or None (never asked)."""
    try:
        text = _consent_path().read_text(encoding="utf-8").strip()
        return text if text in ("on", "off") else None
    except OSError:
        return None


def enabled() -> bool:
    return not hard_disabled() and consent() == "on"


def opt_in() -> str:
    """Record consent and mint the anonymous install id. Returns the id."""
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    _consent_path().write_text("on", encoding="utf-8")
    return install_id()


def opt_out() -> None:
    """Record refusal and delete the identifier — revocation, not just a stop."""
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    _consent_path().write_text("off", encoding="utf-8")
    for p in (_id_path(), _last_path()):
        try:
            p.unlink()
        except OSError:
            pass


def install_id() -> str:
    """The anonymous id: a random UUID with no derivation from the machine
    (no MAC, no hostname, no username) — deleting the file IS revocation."""
    p = _id_path()
    try:
        existing = p.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    fresh = uuid.uuid4().hex
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(fresh, encoding="utf-8")
    return fresh


def build_payload() -> dict:
    """The census, exactly as it would be sent. Numbers and platform strings
    only — never content, paths, hashes-of-content, or key material.

    Anti-bloat rules (the community totals must survive scrutiny):
    - token/dollar counts get the same calibration factor the proof ledger
      applies (heuristic counts → billed-count estimate; identity until enough
      samples), so the census never reports more than the provider would bill;
    - a flat-rate subscription saves REAL tokens but only NOTIONAL dollars —
      those report dollars_saved=0 rather than inflating the community $ sum.
    """
    from . import __version__, ledger

    s = ledger.summary()
    try:
        from .calibration import factor as _calib_factor

        f, _ = _calib_factor()
    except Exception:  # noqa: BLE001 — calibration is a correction, never a blocker
        f = 1.0
    tokens = max(0, s.total_baseline_tokens - s.total_distil_tokens)
    dollars = max(0.0, s.total_baseline_dollars - s.total_distil_dollars)
    try:
        from .doctor import subscription_mode

        if subscription_mode():
            dollars = 0.0  # flat-rate: dollar savings are notional, don't report them
    except Exception:  # noqa: BLE001 — detection failure must not block the census
        pass
    payload = {
        "schema": 2,
        "install_id": install_id(),
        "version": __version__,
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "runs": s.runs,
        "tokens_saved": round(tokens * f),
        "dollars_saved": round(dollars * f, 4),
        "billing": _billing(),
        "by_model": _by_model(f),
        "agents": _agents(),
        "ts": int(time.time()),
    }
    assert set(payload) == PAYLOAD_KEYS  # contract, enforced in prod too
    return payload


def _billing() -> str:
    try:
        from .doctor import subscription_mode

        return "subscription" if subscription_mode() else "metered"
    except Exception:  # noqa: BLE001
        return "metered"


def _by_model(f: float) -> dict:
    """Top models by calibrated tokens saved — model ids only (claude-*, gpt-*…),
    keys length-capped, at most MAX_MODELS entries."""
    from . import ledger

    sums: dict[str, int] = {}
    try:
        for line in ledger.default_path().read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = str(r.get("model", ""))[:64]
            saved = int(r.get("baseline_input_tokens", 0)) - int(r.get("distil_input_tokens", 0))
            if model and saved > 0:
                sums[model] = sums.get(model, 0) + saved
    except OSError:
        return {}
    top = sorted(sums.items(), key=lambda kv: -kv[1])[:MAX_MODELS]
    return {m: round(v * f) for m, v in top}


def _agents() -> list:
    """Distinct wrapped agents seen in session manifests — allowlisted names
    only; anything unrecognized collapses to "other"."""
    seen: set[str] = set()
    try:
        for mf in (_home() / "sessions").glob("*.json"):
            try:
                argv = json.loads(mf.read_text(encoding="utf-8")).get("argv") or []
            except (OSError, json.JSONDecodeError):
                continue
            if not argv:
                continue
            name = os.path.basename(str(argv[0]))
            seen.add(name if name in AGENT_ALLOWLIST else "other")
    except OSError:
        pass
    return sorted(seen)


def status() -> dict:
    """Machine-readable state for `distil census status`."""
    return {
        "consent": consent(),
        "hard_disabled": hard_disabled(),
        "enabled": enabled(),
        "install_id": _id_path().read_text(encoding="utf-8").strip()
        if _id_path().exists()
        else None,
        "endpoint": os.environ.get("DISTIL_CENSUS_ENDPOINT", DEFAULT_ENDPOINT),
    }


def _send(payload: dict) -> None:
    endpoint = os.environ.get("DISTIL_CENSUS_ENDPOINT", DEFAULT_ENDPOINT)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S):
        pass


def maybe_ping() -> bool:
    """Send the census if (and only if) opted in and >24h since the last try.

    Fail-open by contract: rides the proxy/wrap exit path, so it swallows every
    failure and must never block beyond the socket timeout. Returns True only
    when a send was attempted.
    """
    try:
        if not enabled():
            return False
        now = time.time()
        try:
            last = float(_last_path().read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            last = 0.0
        if now - last < MIN_INTERVAL_S:
            return False
        # Throttle on attempt, not success — a dead endpoint must not turn
        # every session exit into a connect-timeout wait.
        _last_path().parent.mkdir(parents=True, exist_ok=True)
        _last_path().write_text(str(int(now)), encoding="utf-8")
        _send(build_payload())
        return True
    except Exception:  # noqa: BLE001 — census must never affect the host path
        return True

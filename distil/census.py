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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_ENDPOINT = "https://distil-census.vercel.app/v1/ping"
DEFAULT_BEAT_ENDPOINT = "https://distil-census.vercel.app/v1/beat"
SEND_TIMEOUT_S = 1.5
MIN_INTERVAL_S = 24 * 3600
# Heartbeat: the near-real-time community signal. A tiny content-free
# {v, id, tokens, rate, ts} sent at most every HEARTBEAT_INTERVAL_S, and ONLY
# when tokens actually grew since the last beat (idle machines send nothing).
# Same opt-in + DO_NOT_TRACK gates as the daily census; feeds the live counter.
HEARTBEAT_INTERVAL_S = 300
BEAT_SCHEMA = 1

# The frozen payload contract — tests/test_census.py refuses any new key, so
# widening the census is a deliberate, reviewed act (and a TELEMETRY.md edit).
# Schema 2 added the usage dimensions: billing, by_model, agents.
# Schema 3 added the integration dimensions: surfaces, shapes.
# Schema 4 added the trust dimension: equivalence (the proof that compression
# didn't change the agent's decision — distil's core claim, as a number) and
# modes (session kind: interactive / headless / sdk).
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
        "surfaces",
        "shapes",
        "equivalence",
        "modes",
        "ts",
    }
)

MODES = frozenset({"interactive", "headless", "sdk"})

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


def _ask_count_path() -> Path:
    return _home() / "census-asked"


# A prompt nobody answers must not become a prompt nobody escapes. Ctrl-C is
# deliberately not recorded as a decline (see `onboard`), which means an
# interrupted prompt returns next session — fine once or twice, a nag by the
# fourth time. After this many unanswered asks, stop asking forever; `distil
# census on` is still there for anyone who changes their mind.
MAX_UNANSWERED_ASKS = 3


def maybe_ask_consent(*, out=None, inp=None) -> bool | None:
    """Ask for census consent ONCE, at the end of a session that actually saved
    something. Returns True (opted in), False (declined), or None (not asked).

    Placed at wrap exit because that is the only moment the question is fair:
    the user has run the thing, there is a real number on screen, and "share
    this?" is concrete rather than hypothetical. Until now the sole consent
    surface was `distil onboard` — so anyone who did `pipx install distil-llm`
    and went straight to `distil wrap`, which is the natural path, was never
    asked and could never be counted. The census did not undercount by accident;
    it undercounted by construction.

    Every guard the onboard prompt established is preserved, because consent
    code earns nothing by being clever:

      * asked once — a stored "on" or "off" ends it permanently;
      * DO_NOT_TRACK / DISTIL_NO_TELEMETRY win outright;
      * both streams must be a TTY, so pipes, CI and headless runs never prompt
        and therefore never enrol;
      * Ctrl-C / EOF is NOT an answer — recording "off" there would burn the one
        chance to ask on a keystroke that meant "not now";
      * and nothing here may raise: this runs on the wrap teardown path, where an
        exception would change a user's exit code over telemetry.

    The saved-something condition is new and deliberate. Asking a user who got
    no value to share their numbers is both a worse question and a worse
    experience than not asking at all.
    """
    import sys

    out = out or sys.stdout
    inp = inp or sys.stdin
    try:
        if hard_disabled() or consent() is not None:
            return None
        if not (
            getattr(inp, "isatty", lambda: False)() and getattr(out, "isatty", lambda: False)()
        ):
            return None
        if _current_saved_tokens() <= 0:
            return None  # nothing worth sharing yet — ask when there is

        asked = 0
        try:
            asked = int(_ask_count_path().read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            asked = 0
        if asked >= MAX_UNANSWERED_ASKS:
            return None

        out.write(
            "\n  Share an anonymous, content-free census? Numbers only — version, OS,\n"
            "  tokens and $ saved. Never prompts or responses. Max one JSON per day.\n"
            "  Preview exactly what would be sent:  distil census show\n"
        )
        out.flush()
        try:
            answer = input("  Share the anonymous census? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Not an answer. Count it so an unanswerable terminal cannot nag
            # forever, but never treat the silence as a decision.
            try:
                _ask_count_path().parent.mkdir(parents=True, exist_ok=True)
                _ask_count_path().write_text(str(asked + 1), encoding="utf-8")
            except OSError:
                pass
            out.write("\n  no answer recorded — enable anytime: distil census on\n")
            return None

        if answer in ("y", "yes"):
            opt_in()
            out.write("  census on — TELEMETRY.md documents exactly what is sent\n")
            return True
        opt_out()
        out.write("  census off — re-enable anytime: distil census on\n")
        return False
    except Exception:  # noqa: BLE001 — consent must never affect a user's exit code
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
    for p in (_id_path(), _last_path(), _savings_path(), _beat_last_path()):
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


# ---- monotonic count-time savings accrual --------------------------------
# The community counters — the daily census (tokens_saved / dollars_saved /
# by_model) AND the near-real-time heartbeat total — must be MONOTONIC and must
# never exceed what the provider would bill. Multiplying the whole LIFETIME
# cumulative by the CURRENT factor (the old code) fails both: as the factor
# drifts down toward a better estimate, the reported total shrinks — the live
# counter un-counts tokens already saved. We instead calibrate each census
# DELTA at count time and freeze it: the running total advances by
# Δraw × factor-known-now and is never restated when the factor later moves.
# That is monotonic by construction, and honest — every increment is valued at
# the best estimate available when it was earned, exactly like invoicing each
# period as it closes. State is persisted so it survives across sessions and is
# wiped with ~/.distil (and on opt-out), so revocation stays real.


def _savings_path() -> Path:
    return _home() / "census-savings.json"


def _factor(model: str | None = None) -> float:
    """The calibration factor as a bare float, fail-open to identity (1.0).
    Calibration is a correction, never a blocker."""
    try:
        from .calibration import factor as _cf

        return _cf(model)[0]
    except Exception:  # noqa: BLE001
        return 1.0


def _fresh_savings() -> dict:
    return {
        "v": 1,
        "tokens": {"saved": 0.0, "raw_seen": 0},
        "dollars": {"saved": 0.0, "raw_seen": 0.0},
        "by_model": {},
    }


def _parse_savings(text: str) -> dict:
    """Accrual state from raw JSON, falling back to a fresh zeroed state."""
    fresh = _fresh_savings()
    try:
        d = json.loads(text)
        if isinstance(d, dict) and d.get("v") == 1 and isinstance(d.get("by_model"), dict):
            for k in ("tokens", "dollars"):
                if not isinstance(d.get(k), dict):
                    d[k] = fresh[k]
            return d
    except (json.JSONDecodeError, ValueError):
        pass
    return fresh


def _invite_path() -> Path:
    return _home() / "census-invite"


def invite_seen() -> bool:
    """Whether the one-time census invite has already been shown."""
    return _invite_path().exists()


def mark_invite_seen() -> None:
    """Record that the invite was shown, so it is shown exactly once.

    Deliberately separate from `consent`: seeing the invite is not answering it.
    A user who ignores it stays un-asked (`consent() is None`) and can still opt
    in later, but is never shown the same nudge twice.
    """
    try:
        p = _invite_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1", encoding="utf-8")
    except OSError:
        pass  # fail-open: worst case the invite prints once more


def accrued_tokens() -> int | None:
    """The count-time accrued token total, or None if nothing has accrued yet.

    This is the figure the census and the heartbeat report: each delta banked at
    the factor known when it was earned, so it never restates history. Local
    surfaces ("your savings") read it through here so the number a user sees in
    `distil dashboard` is the same one the community total is built from, rather
    than lifetime×current-factor — which silently shrinks as calibration refines.

    Read-only: never creates or advances state, so it is safe to call regardless
    of consent (and returns None once `census off` has deleted the state).
    """
    saved = float(_load_savings().get("tokens", {}).get("saved") or 0.0)
    return round(saved) if saved > 0 else None


def _load_savings() -> dict:
    """Persisted accrual state. Each channel remembers the raw total already
    banked (`raw_seen`) and the frozen calibrated cumulative (`saved`, a float
    so per-period rounding never compounds)."""
    try:
        return _parse_savings(_savings_path().read_text(encoding="utf-8"))
    except OSError:
        return _fresh_savings()


@contextmanager
def _savings_locked(persist: bool) -> Iterator[dict]:
    """Yield the accrual state for a read-modify-write under an exclusive flock.

    `saved` is monotonic *per step*, but the counter is advanced from two
    places (the daily census and the near-real-time heartbeat) and distil runs
    as several processes at once — wrap, proxy worker, gateway, webdash. Each
    used to do load → step → write-the-whole-file with no lock, so the last
    writer clobbered the others: a process holding state from minutes ago wrote
    a *smaller* `saved` back, and rewound `raw_seen` with it so an
    already-banked delta got counted a second time. That is how a total that
    can only rise published 1.44B, then 1.33B, then 1.16B, then 1.48B.

    Reading inside the lock is what fixes it: every writer sees the newest
    state before it steps. Mirrors `surfaces._flush_locked`.
    """
    if not persist:  # previews never mutate, so they never need the lock
        yield _load_savings()
        return
    try:
        p = _savings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "a+", encoding="utf-8")
    except OSError:
        yield _load_savings()  # fail-open: accrue in memory, lose only the write
        return
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # no flock (e.g. Windows): degrades to the old last-write-wins
        fh.seek(0)
        st = _parse_savings(fh.read())
        yield st
        try:
            fh.seek(0)
            fh.truncate()
            json.dump(st, fh)
        except OSError:
            pass  # a lost write just re-accrues the same delta next time
    finally:
        fh.close()


def _save_savings(st: dict) -> None:
    try:
        p = _savings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st), encoding="utf-8")
    except OSError:
        pass  # fail-open: a lost write just re-accrues the same delta next time


def _step(ch: dict, raw_now: float, f: float) -> float:
    """Advance one monotonic channel: bank max(0, Δraw)·f at count time and
    freeze it. `saved` only ever rises. Returns the new cumulative saved."""
    if raw_now - ch.get("raw_seen", 0) > 0:
        ch["saved"] = ch.get("saved", 0.0) + (raw_now - ch["raw_seen"]) * f
    # ponytail: a shrinking raw (ledger truncation/reset) rebases without
    # subtracting — census-savings.json is wiped together with ~/.distil, so a
    # real reset re-accrues from zero and never double-counts.
    ch["raw_seen"] = raw_now
    return ch.get("saved", 0.0)


def _accrue(
    raw_tokens: int,
    raw_dollars: float,
    raw_by_model: dict,
    *,
    persist: bool,
) -> dict:
    """Fold this census's raw savings into the frozen cumulative and return the
    emit-ready numbers. Persists only when `persist` — previews never mutate."""
    with _savings_locked(persist) as st:
        tokens_saved = _step(st["tokens"], raw_tokens, _factor())
        dollars_saved = _step(st["dollars"], raw_dollars, _factor())
        bm = st["by_model"]
        for model, raw in raw_by_model.items():
            ch = bm.setdefault(model, {"saved": 0.0, "raw_seen": 0})
            _step(ch, raw, _factor(model))  # per-model factor preserves the model mix
    top = sorted(bm.items(), key=lambda kv: -kv[1].get("saved", 0.0))[:MAX_MODELS]
    return {
        "tokens": round(tokens_saved),
        "dollars": round(dollars_saved, 4),
        "by_model": {m: round(c["saved"]) for m, c in top if c.get("saved", 0.0) > 0},
    }


def _reported_version() -> str:
    """The distil version to report as installed — read from DISK, not from this
    interpreter's already-imported ``__version__``.

    These differ, and the difference silently corrupted the version histogram.
    ``distil wrap`` is the flagship surface and is deliberately long-lived: the
    hot-swap replaces the proxy *worker* on upgrade so the agent session never
    restarts, which means the wrap parent keeps running the code it started
    with — for days. It is also the process that emits the census on session
    exit. So every wrap user's beat reported whatever version was installed when
    their session began, and kept reporting it for the life of that session. The
    "versions in the wild" chart was therefore a picture of when people last
    restarted, not of what they have installed. Observed live: a machine running
    1.34.0 beat ``1.28.0`` two upgrades later.

    ``importlib.metadata`` re-discovers the dist-info on each call, so a
    pipx/pip upgrade under a running process is visible immediately — that is
    precisely what ``hotswap.installed_version()`` was built for, so this reuses
    it rather than re-deriving it.

    Falls back to the interpreter's own version when the package metadata is not
    discoverable (a source checkout, a zipapp, a vendored copy). That fallback is
    the old, slightly-stale answer, which is strictly better than reporting
    nothing and is never wrong about *which* distil is running.
    """
    from . import __version__

    try:
        from .hotswap import installed_version

        on_disk = installed_version()
    except Exception:  # noqa: BLE001 — telemetry must never break the caller
        on_disk = None
    return on_disk or __version__


def build_payload(*, accrue: bool = False) -> dict:
    """The census, exactly as it would be sent. Numbers and platform strings
    only — never content, paths, hashes-of-content, or key material.

    Anti-bloat rules (the community totals must survive scrutiny):
    - token/dollar/by-model counts are a MONOTONIC cumulative built by
      calibrating each census DELTA at count time and freezing it, so the total
      never goes down on a downward recalibration and never reports more than
      the provider would bill (see the accrual note above);
    - dollars are always reported, but the `billing` field lets the rollup
      bucket them honestly: metered dollars are real savings, subscription
      dollars are the NOTIONAL API-rate value (shown, labeled, never mixed
      into the real-$ total).

    `accrue` gates the ONE side effect: the running total is advanced and
    persisted only on a real send (`maybe_ping`). `distil census show` and any
    direct preview call leave `~/.distil/census-savings.json` untouched.
    """
    from . import ledger

    s = ledger.summary()
    raw_tokens = max(0, s.total_baseline_tokens - s.total_distil_tokens)
    raw_dollars = max(0.0, s.total_baseline_dollars - s.total_distil_dollars)
    acc = _accrue(raw_tokens, raw_dollars, _raw_by_model(), persist=accrue)
    from . import surfaces as _surfaces

    integ = _surfaces.snapshot()
    payload = {
        "schema": 4,
        "install_id": install_id(),
        "version": _reported_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "runs": s.runs,
        "tokens_saved": acc["tokens"],
        "dollars_saved": acc["dollars"],
        "billing": _billing(),
        "by_model": acc["by_model"],
        "agents": _agents(),
        "surfaces": integ["surfaces"],
        "shapes": integ["shapes"],
        "equivalence": _equivalence(),
        "modes": _modes(),
        "ts": int(time.time()),
    }
    assert set(payload) == PAYLOAD_KEYS  # contract, enforced in prod too
    return payload


def _modes() -> dict:
    """Session kinds, content-free: classify each wrap session by the SHAPE of
    its command, never its content —
      interactive : an agent binary (claude/codex/…) with no print flag
      headless    : an agent binary invoked with -p / --print (e.g. `claude -p`)
      sdk         : a non-agent argv[0] (python/node/…) — Agent SDK or custom driver
    Only the presence of allowlisted flags is inspected; the prompt/args are
    never read or stored."""
    counts: dict[str, int] = {}
    try:
        for mf in (_home() / "sessions").glob("*.json"):
            try:
                argv = json.loads(mf.read_text(encoding="utf-8")).get("argv") or []
            except (OSError, json.JSONDecodeError):
                continue
            if not argv:
                continue
            name = os.path.basename(str(argv[0]))
            if name not in AGENT_ALLOWLIST:
                mode = "sdk"
            elif any(a in ("-p", "--print") for a in argv):
                mode = "headless"
            else:
                mode = "interactive"
            counts[mode] = counts.get(mode, 0) + 1
    except OSError:
        pass
    return counts


def _equivalence() -> dict:
    """Live decision-equivalence: {pct, shadowed}. ``pct`` is the noise-adjusted
    equivalence percentage (compression didn't change the agent's next action),
    but ONLY when an A/A self-agreement baseline exists — otherwise the number
    conflates sampling nondeterminism with real harm, so we send ``pct: None``
    and just the shadowed count. Content-free (booleans behind the scenes)."""
    try:
        from .shadow import ShadowLedger

        led = ShadowLedger.load(current_only=True)
        shadowed = int(led.samples)
        if shadowed > 0 and led.aa_agreement() is not None:
            pct = round((1.0 - led.adjusted_rate()) * 100.0, 2)
        else:
            pct = None
        return {"pct": pct, "shadowed": shadowed}
    except Exception:  # noqa: BLE001 — trust metric is best-effort, never blocks
        return {"pct": None, "shadowed": 0}


def _billing() -> str:
    try:
        from .doctor import subscription_mode

        return "subscription" if subscription_mode() else "metered"
    except Exception:  # noqa: BLE001
        return "metered"


def _raw_by_model() -> dict:
    """Lifetime raw (uncalibrated) tokens saved per model — model ids only
    (claude-*, gpt-*…), keys length-capped. Calibration (per-model factor) and
    the top-N cut happen in `_accrue`, so each model's cumulative is monotonic."""
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
    return sums


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
        _send(build_payload(accrue=True))  # the only census-send site that advances the total
        return True
    except Exception:  # noqa: BLE001 — census must never affect the host path
        return True


def _beat_last_path() -> Path:
    return _home() / "heartbeat-last"


def _current_saved_tokens() -> int:
    """The live-counter figure — the SAME monotonic accrued total the daily
    census reports, advanced (and persisted) here so the near-real-time beat
    stays fresh between censuses. Shares `census-savings.json`'s tokens channel,
    so a downward recalibration can never make the live counter shrink."""
    from . import ledger

    s = ledger.summary()
    raw_tokens = max(0, s.total_baseline_tokens - s.total_distil_tokens)
    # the heartbeat is a real send; advance the shared total under the same
    # lock the census uses, or the two writers clobber each other.
    with _savings_locked(True) as st:
        val = _step(st["tokens"], raw_tokens, _factor())
    return round(val)


def maybe_heartbeat() -> bool:
    """Near-real-time community pulse: at most one tiny content-free beat every
    HEARTBEAT_INTERVAL_S, from any install that has saved tokens. Same opt-in +
    kill-switch gates as the census; fail-open. Returns True iff a beat was sent.

    Liveness, not growth: a beat fires each interval so ``active`` counts installs
    that are *running distil*, not only those whose saved-token total happens to be
    climbing right now — a downward calibration refinement (the estimate getting
    more accurate) must not make a live install read as inactive. ``rate`` still
    reflects real growth (0 when flat or refined down), so the odometer never
    projects phantom tokens; only a genuinely new install with 0 saved tokens
    stays silent (nothing to report yet)."""
    try:
        if not enabled():
            return False
        now = time.time()
        try:
            prev = json.loads(_beat_last_path().read_text(encoding="utf-8"))
            last_ts = float(prev.get("ts", 0))
            last_tokens = int(prev.get("tokens", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            last_ts, last_tokens = 0.0, 0
        if now - last_ts < HEARTBEAT_INTERVAL_S:
            return False
        tokens = _current_saved_tokens()
        # Advance the local marker every interval so the next rate reflects THIS
        # interval's real pace.
        _beat_last_path().parent.mkdir(parents=True, exist_ok=True)
        rate = (tokens - last_tokens) / (now - last_ts) if tokens > last_tokens and last_ts else 0.0
        _beat_last_path().write_text(
            json.dumps({"tokens": tokens, "ts": int(now)}), encoding="utf-8"
        )
        if tokens <= 0:
            return False  # never saved anything yet → not an active saver
        _send_beat(
            {
                "v": BEAT_SCHEMA,
                "id": install_id(),
                "tokens": tokens,
                "rate": round(rate, 2),
                "ts": int(now),
            }
        )
        return True
    except Exception:  # noqa: BLE001 — heartbeat must never affect the host path
        return True


def _send_beat(payload: dict) -> None:
    endpoint = os.environ.get("DISTIL_BEAT_ENDPOINT", DEFAULT_BEAT_ENDPOINT)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S):
        pass

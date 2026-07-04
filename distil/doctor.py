"""``distil doctor`` — diagnose a distil setup end-to-end.

Answers the questions that otherwise turn into a dead-end: is the proxy machinery
healthy, is traffic being recorded, is outcome-validation running, and (for Claude
Code) is the status line wired and is this a flat-rate subscription. Every check is
cheap, never hangs, and degrades gracefully — a check that errors reports ``fail``
with the reason rather than taking the whole command down.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Status glyphs (ANSI applied by the renderer in cli.py, not here).
OK, WARN, INFO, FAIL = "ok", "warn", "info", "fail"


@dataclass
class Check:
    name: str
    status: str  # OK | WARN | INFO | FAIL
    detail: str
    hint: str = ""


def subscription_mode() -> bool:
    """True when the dollar figures are notional (flat-rate, no per-token bill).

    Explicit ``DISTIL_SUBSCRIPTION`` wins (on/off). Otherwise auto-detect a Claude
    Pro/Max OAuth login (``~/.claude.json`` has ``oauthAccount``) with no API key
    set — the common case where distil's measured dollars don't map to a real bill.
    """
    env = os.environ.get("DISTIL_SUBSCRIPTION", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if os.environ.get("ANTHROPIC_API_KEY"):
        return False  # a metered key means the dollars are real
    return _claude_oauth_present()


def _claude_oauth_present() -> bool:
    # A content-free presence check: scan for the key string only (never parse or
    # read any token). The file is local and the user's own; we extract nothing.
    p = Path.home() / ".claude.json"
    try:
        return '"oauthAccount"' in p.read_text(encoding="utf-8")
    except OSError:
        return False


def _check_version() -> Check:
    import sys

    from . import __version__

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 9)  # must match pyproject requires-python
    return Check(
        "distil",
        OK if ok else FAIL,
        f"{__version__} (Python {py})",
        "" if ok else "distil needs Python 3.9+",
    )


def _find_all_distil() -> list[str]:
    """Every `distil` on PATH, in resolution order (like `which -a`)."""
    import os

    seen: list[str] = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(d, "distil")
        if cand not in seen and os.path.isfile(cand) and os.access(cand, os.X_OK):
            seen.append(cand)
    return seen


def _check_shadowed_install() -> Check:
    """The single most confusing distil failure: two installs (brew + pipx),
    where an upgrade lands on a copy the shell never reaches, so `doctor` keeps
    reporting the old version. Catch it explicitly."""
    paths = _find_all_distil()
    if len(paths) <= 1:
        return Check("install", OK, f"one distil on PATH{f' ({paths[0]})' if paths else ''}")
    active = paths[0]

    def _mgr(p: str) -> str:
        return (
            "homebrew"
            if "/Cellar/" in p or "/homebrew/" in p or p.startswith("/usr/local/")
            else "pipx"
            if "/pipx/" in p or "/.local/bin/" in p
            else "uv"
            if "/uv/" in p
            else "pip/other"
        )

    others = ", ".join(f"{p} ({_mgr(p)})" for p in paths[1:])
    return Check(
        "install",
        WARN,
        f"{len(paths)} distil installs — ACTIVE: {active} ({_mgr(active)}); shadowed: {others}",
        "an upgrade to a shadowed copy won't take effect. Keep one: upgrade the "
        f"active install ({_mgr(active)}), or remove it so another wins.",
    )


def _check_ledger() -> Check:
    from . import ledger

    try:
        s = ledger.summary()
    except Exception as exc:  # noqa: BLE001
        return Check("savings ledger", FAIL, f"could not read ledger — {exc}")
    if s.runs == 0:
        return Check(
            "savings ledger",
            INFO,
            "no runs recorded yet",
            "route an agent through distil:  distil wrap -- claude",
        )
    live = "live-proxy" in s.by_trajectory
    saved_tok = (
        f"{s.total_tokens_saved / 1e6:.1f}M"
        if s.total_tokens_saved >= 1e6
        else str(s.total_tokens_saved)
    )
    detail = f"{s.runs} runs, {saved_tok} tokens saved"
    if not subscription_mode():
        detail += f" / ${s.total_dollars_saved:,.2f}"
    detail += " (genuine live traffic)" if live else " (corpus only — no live proxy yet)"
    return Check("savings ledger", OK, detail)


def _check_session() -> Check:
    """Explain what the status line's 'session' segment is showing — especially
    the 'watching' state, so a user doesn't have to wonder why ▼ is empty."""
    import time

    from . import ledger

    try:
        sid, last_ts = ledger.latest_session()
        if not sid or (time.time() - last_ts) > 4 * 3600:
            return Check(
                "this session", INFO, "no recent session — start one: distil wrap -- claude"
            )
        s = ledger.summary(session=sid)
    except Exception as exc:  # noqa: BLE001
        return Check("this session", INFO, f"session slice unavailable — {exc}")
    if not s.runs or not s.total_baseline_tokens:
        return Check("this session", INFO, "no traffic recorded yet")
    reqs = f"{s.runs} request{'s' if s.runs != 1 else ''}"
    if s.total_tokens_saved > 0:
        pct = (1 - s.total_distil_tokens / s.total_baseline_tokens) * 100
        return Check(
            "this session",
            OK,
            f"▼{s.total_tokens_saved:,} tokens saved ({pct:.0f}% smaller) over {reqs}",
        )
    seen = (
        f"{s.total_baseline_tokens / 1000:.1f}K tokens"
        if s.total_baseline_tokens >= 1000
        else f"{s.total_baseline_tokens} token{'s' if s.total_baseline_tokens != 1 else ''}"
    )
    return Check(
        "this session",
        INFO,
        f"watching — {reqs}, {seen} seen, 0 saved yet",
        "normal early in a session: savings come from LARGE tool output (file reads, "
        "logs). A small request that's mostly the system prompt has nothing to trim — "
        "▼ climbs once your agent reads big content.",
    )


def _check_live_routing() -> Check:
    """Catch the silent failure that reads as 'watching forever': a `distil
    wrap`/`proxy` process is running, but the agent's traffic isn't reaching it
    (terminal opened before the alias was sourced, or a raw agent), so nothing
    gets recorded and savings never move."""
    import re
    import subprocess
    import time

    from . import ledger

    try:
        out = subprocess.run(["ps", "axww"], capture_output=True, text=True, timeout=3)
    except Exception:  # noqa: BLE001 — no `ps` (e.g. Windows): can't tell, skip
        return Check("live routing", INFO, "process check not available on this platform")
    running = bool(re.search(r"distil\s+(wrap|proxy|gateway)\b", out.stdout))
    if not running:
        return Check("live routing", INFO, "no distil wrap/proxy running (nothing to route)")
    try:
        _sid, last_ts = ledger.latest_session()
        age_min = (time.time() - last_ts) / 60 if last_ts else 1e9
    except Exception:  # noqa: BLE001
        age_min = 1e9
    if age_min <= 5:
        return Check(
            "live routing", OK, f"wrapped agent live · traffic recorded {age_min:.0f}m ago"
        )
    return Check(
        "live routing",
        WARN,
        f"a distil wrap/proxy is running but NO traffic recorded in {age_min:.0f} min",
        "your agent is probably bypassing distil. In its terminal run "
        "`echo $ANTHROPIC_BASE_URL` — empty means that shell was opened before the "
        "alias; open a fresh terminal, or launch it with `distil wrap -- <agent>`.",
    )


def _check_shadow() -> Check:
    try:
        from .shadow import ShadowLedger

        led = ShadowLedger.load()
    except Exception as exc:  # noqa: BLE001
        return Check("shadow validation", FAIL, f"could not read shadow ledger — {exc}")
    if led.samples == 0:
        return Check(
            "shadow validation",
            WARN,
            "not running — no decision-equivalence samples",
            "start it in one command:  distil wrap --shadow 0.1 -- claude",
        )
    smp = f"{led.samples} sample{'s' if led.samples != 1 else ''}"
    if led.samples < 25:
        # Same 25-sample floor the status line uses — a rate over a handful is noise.
        return Check(
            "shadow validation",
            INFO,
            f"collecting — {smp} (need 25 for a decision-equivalence rate)",
        )
    eq = 100 * (1 - led.rate())
    return Check("shadow validation", OK, f"{eq:.1f}% decision-equivalence over {smp}")


def _check_proxy_selftest() -> Check:
    """End-to-end proxy round-trip with an in-process fake upstream — no network.

    Proves the proxy machinery (forwarding, compression, response relay) works on
    this machine, which is the thing users can't otherwise verify."""
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from .proxy import build_handler

    class _Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(n)
            body = json.dumps(
                {"id": "msg_selftest", "content": [{"type": "text", "text": "ok"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:  # silence
            pass

    up = px = None
    try:
        up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        up_url = f"http://127.0.0.1:{up.server_address[1]}"

        px = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(up_url))
        threading.Thread(target=px.serve_forever, daemon=True).start()
        px_url = f"http://127.0.0.1:{px.server_address[1]}"

        payload = json.dumps(
            {
                "model": "claude-3-5-haiku",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello from doctor"}],
            }
        ).encode()
        req = urllib.request.Request(
            px_url + "/v1/messages", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status == 200 and b"msg_selftest" in resp.read()
        if ok:
            return Check("proxy self-test", OK, "request routed through distil and back (local)")
        return Check("proxy self-test", FAIL, "round-trip returned an unexpected response")
    except Exception as exc:  # noqa: BLE001
        return Check("proxy self-test", FAIL, f"round-trip failed — {exc}")
    finally:
        for srv in (px, up):
            if srv is not None:
                srv.shutdown()


def _check_anthropic_extra() -> Check:
    import importlib.util

    has = importlib.util.find_spec("anthropic") is not None
    if has:
        return Check(
            "anthropic extra", OK, "installed (live grading / billing-grade tokenizer available)"
        )
    return Check(
        "anthropic extra",
        INFO,
        "not installed",
        "only needed for --runner/--tokenizer anthropic:  pipx inject distil-llm anthropic",
    )


def _check_api_key() -> Check:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return Check("ANTHROPIC_API_KEY", OK, "set (live Anthropic paths available)")
    return Check(
        "ANTHROPIC_API_KEY",
        INFO,
        "not set",
        "offline compression/cert needs no key; set it only for --runner/--tokenizer anthropic",
    )


def _check_claude_code() -> list[Check]:
    """Claude Code-specific checks: status-line wiring + subscription detection."""
    out: list[Check] = []
    settings = Path.home() / ".claude" / "settings.json"
    try:
        data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    sl = (data.get("statusLine") or {}).get("command", "") if isinstance(data, dict) else ""
    if "distil" in sl or "statusline.sh" in sl:
        out.append(Check("status line", OK, "wired into ~/.claude/settings.json"))
    else:
        out.append(
            Check(
                "status line",
                INFO,
                "not wired (or uses another script)",
                "see the distil plugin README to add the savings status line",
            )
        )
    if subscription_mode():
        out.append(
            Check(
                "billing mode",
                INFO,
                "flat-rate subscription detected — dollar figures are notional",
                "set DISTIL_SUBSCRIPTION=1 to show tokens-only everywhere",
            )
        )
    return out


def _check_pricing_catalog() -> Check:
    """Warn when the ledger contains models the pricing catalog can't price —
    their dollar savings are recorded as $0 (honest floor), which understates
    the headline number until the catalog is updated."""
    from . import ledger, pricing

    path = ledger.default_path()
    unknown: set[str] = set()
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            model = json.loads(line).get("model", "")
            base = model.replace(" (unpriced)", "")
            if pricing.resolve(base) is None:
                unknown.add(base)
    except OSError:
        return Check("pricing", OK, f"catalog covers {len(pricing.CATALOG)} model ids")
    if unknown:
        return Check(
            "pricing",
            WARN,
            f"ledger contains unpriced models: {', '.join(sorted(unknown)[:4])}",
            "their savings show as $0 — update distil (or the catalog) to price them",
        )
    return Check("pricing", OK, f"all recorded models priced ({len(pricing.CATALOG)} in catalog)")


def _check_mode() -> Check:
    """Surface the compression mode of the managed always-on proxy service, if
    one is installed — because a `verbatim` service silently caps savings near
    zero (Tier-1 digest off), which otherwise looks like a broken statusline."""
    import platform

    home = Path.home()
    sysname = platform.system()
    if sysname == "Darwin":
        svc = home / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
    elif sysname == "Linux":
        svc = home / ".config" / "systemd" / "user" / "distil-proxy.service"
    else:
        svc = None
    if svc is None or not svc.exists():
        return Check(
            "compression mode",
            INFO,
            "no always-on service — mode is set per `distil wrap`/`proxy` run",
            "digest (default) and lossless-only compress; verbatim only trims "
            "whitespace/JSON (near-zero savings on code/prose)",
        )
    try:
        text = svc.read_text(encoding="utf-8")
    except OSError:
        return Check("compression mode", INFO, "service present (mode unreadable)")
    mode = (
        "verbatim"
        if "--verbatim" in text
        else ("lossless-only" if "--lossless-only" in text else "digest")
    )
    if mode == "verbatim":
        return Check(
            "compression mode",
            WARN,
            "always-on proxy runs in VERBATIM mode — the reversible digest is off, "
            "so savings are near-zero by design (this is why ▼ can read 0)",
            "switch to reversible savings:  distil default --mode lossless-only "
            "(subscription-safe) or --mode expand (metered)",
        )
    return Check("compression mode", OK, f"always-on proxy: {mode} (reversible digest active)")


def _check_tokenizer_grade() -> Check:
    """Say out loud which tokenizer produced the numbers users see daily."""
    from . import ledger

    try:
        s = ledger.summary()
    except Exception:  # noqa: BLE001
        return Check("tokenizer", INFO, "no ledger yet — counts will use the heuristic tokenizer")
    if s.runs and s.tokenizers <= {"heuristic"}:
        return Check(
            "tokenizer",
            INFO,
            "savings measured with the heuristic tokenizer (≈, directionally accurate)",
            "billing-grade counts: distil savings --tokenizer anthropic (needs API key)",
        )
    return Check("tokenizer", OK, f"tokenizers in ledger: {', '.join(sorted(s.tokenizers)) or '—'}")


def diagnose() -> list[Check]:
    """Run every check; each is isolated so one failure can't abort the rest."""
    checks: list[Check] = []
    for fn in (
        _check_version,
        _check_shadowed_install,
        _check_ledger,
        _check_session,
        _check_live_routing,
        _check_shadow,
        _check_proxy_selftest,
        _check_anthropic_extra,
        _check_api_key,
        _check_pricing_catalog,
        _check_tokenizer_grade,
        _check_mode,
    ):
        try:
            checks.append(fn())
        except Exception as exc:  # noqa: BLE001 — a check must never crash doctor
            checks.append(Check(fn.__name__.replace("_check_", ""), FAIL, f"check errored — {exc}"))
    try:
        checks.extend(_check_claude_code())
    except Exception:  # noqa: BLE001
        pass
    return checks

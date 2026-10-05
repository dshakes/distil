"""``distil audit`` — distil as referee: did a compressor help or hurt, on YOUR traffic?

Any compressor can claim a saving. What none of them report is the two numbers a user
actually pays in: dollars, and the agent's decisions. ``distil audit`` measures both for
a compressor the user names, on a sampled fraction of their own requests, with the same
paired A/A'/B design the shadow already uses for distil itself (``distil/shadow.py``,
ADR 0001; the estimator is :meth:`shadow.ShadowLedger.equivalence`):

  * **A, A'** — the request replayed twice exactly as the client sent it. Their
    agreement is the model's own noise floor.
  * **B** — the same request as the audited compressor shapes it.

Per request the row stores ``1{A==B} - 1{A==A'}`` and the provider's billed tokens for
each arm, so the report can say "changes decisions beyond the model's own noise" or
"costs more than it saves" — or that it cannot tell yet, and how many more samples it
needs. Decision of record: ``docs/adr/0022-distil-as-referee.md``.

What is faithful, per compressor (and so what is implemented):

  * ``distil`` — B is the body distil actually served.
  * ``anthropic-context-editing`` — B is the same request plus the documented
    ``context_management`` parameter (``clear_tool_uses_20250919``, documented defaults,
    beta ``context-management-2025-06-27``). The provider applies its own edit; whether it
    fired is read from ``context_management.applied_edits``. A request that already
    carries ``context_management`` is audited the other way round: A drops it.
  * ``openai-compaction`` — Responses API only: B adds
    ``context_management=[{"type": "compaction", "compact_threshold": ...}]``; firing is
    the ``compaction`` output item.
  * ``headroom`` — B goes through the user's own running Headroom proxy (``--via``, a
    loopback URL); A and A' go straight to the provider. The real package does the
    shaping; distil never imitates it.
  * ``rtk`` — **not auditable per request.** RTK rewrites shell commands inside the agent
    before they run; by the time a request reaches distil the un-rewritten output never
    existed, so there is no faithful A. It is measured offline instead (the scoreboard).

Opt-in, disclosed, capped: off until ``distil audit --compressor X`` is confirmed; a hard
per-day dollar cap (default $1) gates every sample before it starts; the ledger holds
hashes, token counts and dollars, never content; replays go only where the request was
already going (or to a loopback proxy the user named).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

log = logging.getLogger(__name__)

#: compressor -> how B is produced (shown by ``distil audit`` and in the report).
COMPRESSORS: dict[str, str] = {
    "distil": "the body distil served",
    "anthropic-context-editing": "the request plus context_management clear_tool_uses_20250919",
    "openai-compaction": "the Responses request plus context_management compaction",
    "headroom": "the request sent through your running Headroom proxy (--via)",
    "rtk": "not auditable per request",
}
NOT_AUDITABLE: dict[str, str] = {
    "rtk": (
        "RTK rewrites shell commands inside the agent before they run, so the request distil "
        "sees already carries RTK's output and the un-rewritten output never existed: there is "
        "no faithful A to compare against. RTK is measured offline instead, by the real pinned "
        "binary on SWE-bench Lite and Terminal-Bench: see docs/scoreboard.html."
    ),
}

DEFAULT_RATE = 0.02
DEFAULT_CAP_USD = 1.0
#: Non-inferiority margin on the paired decision difference: the same 5 points as the
#: SWE-bench outcome eval (benchmarks/swebench_outcome/report.MARGIN) and BUDGET_DELTA.
EQUIV_MARGIN = 0.05
#: The documented defaults (trigger 100k input tokens, keep 3 tool uses): audit the
#: feature as a user who turns it on would get it.
CM_EDIT: dict[str, Any] = {"type": "clear_tool_uses_20250919"}
#: OpenAI's own documented example threshold; the parameter has no documented default.
COMPACT_THRESHOLD = 200_000
#: Reservation inputs. ~3.5 bytes/token is on the token-rich side for JSON transcripts;
#: output is capped because Claude Code asks for 32k it rarely uses. Input is reserved at
#: the BASE rate: reserving every byte as a 1-hour cache write (2x) would refuse every
#: ~100k-token request at the default cap, so the audit would measure nothing.
BYTES_PER_TOKEN = 3.5
OUT_TOKENS_EST = 4096


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def config_path() -> Path:
    return _home() / "audit.json"


def ledger_path() -> Path:
    return _home() / "audit.jsonl"


def budget_path() -> Path:
    return _home() / "audit_budget.json"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class AuditConfig:
    compressor: str
    rate: float = DEFAULT_RATE
    cap_usd: float = DEFAULT_CAP_USD
    via: str | None = None
    since: float = 0.0

    def save(self) -> None:
        p = config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls) -> AuditConfig | None:
        try:
            d = json.loads(config_path().read_text(encoding="utf-8"))
            cfg = cls(
                compressor=str(d["compressor"]),
                rate=float(d.get("rate", DEFAULT_RATE)),
                cap_usd=float(d.get("cap_usd", DEFAULT_CAP_USD)),
                via=d.get("via") or None,
                since=float(d.get("since", 0.0)),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return cfg if validate(cfg) is None else None


def loopback_url(url: str) -> bool:
    """``--via`` gets the user's API key and their content, so it may only be local."""
    try:
        u = urlparse(url)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and u.hostname in ("127.0.0.1", "localhost", "::1")


def validate(cfg: AuditConfig) -> str | None:
    """The reason *cfg* cannot run, or None."""
    if cfg.compressor in NOT_AUDITABLE:
        return NOT_AUDITABLE[cfg.compressor]
    if cfg.compressor not in COMPRESSORS:
        return f"unknown compressor {cfg.compressor!r}"
    if not (0 < cfg.rate <= 1):
        return "--rate must be in (0, 1]"
    if not (cfg.cap_usd > 0 and math.isfinite(cfg.cap_usd)):
        return "--cap-usd must be a positive number of dollars"
    if cfg.compressor == "headroom":
        if not cfg.via:
            return "headroom is audited through your own running Headroom proxy: pass --via URL"
        if not loopback_url(cfg.via):
            return "--via must be a loopback http(s) URL (it receives your API key and content)"
    elif cfg.via:
        return "--via applies only to --compressor headroom"
    return None


# --------------------------------------------------------------------------- #
# Shaping B
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shaped:
    a: bytes
    b: bytes
    b_headers: dict[str, str]
    via: str | None
    model: str | None


def _with_beta(headers: dict[str, str], beta: str) -> dict[str, str]:
    out = dict(headers)
    key = next((k for k in out if k.lower() == "anthropic-beta"), "anthropic-beta")
    have = [b.strip() for b in out.get(key, "").split(",") if b.strip()]
    if beta not in have:
        have.append(beta)
    out[key] = ",".join(have)
    return out


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj).encode("utf-8")


def shape(
    compressor: str,
    path: str,
    raw: bytes,
    served: bytes,
    headers: dict[str, str],
    via: str | None = None,
) -> Shaped | None:
    """A and B replay bodies for one request, or None when this compressor does not apply
    to it (wrong endpoint, not JSON). Both sides go through the shadow's own replay
    preparation (temperature pinned where allowed, prior thinking stripped)."""
    from .httpguard import is_messages_path, is_responses_path
    from .shadow import deterministic_body

    ra = deterministic_body(raw)
    if ra is None:
        return None
    if compressor == "distil":
        rb = deterministic_body(served)
        return None if rb is None else Shaped(ra.body, rb.body, dict(headers), None, ra.model)
    if compressor == "headroom":
        return Shaped(ra.body, ra.body, dict(headers), via, ra.model)
    obj = json.loads(ra.body)
    if compressor == "anthropic-context-editing":
        if not is_messages_path(path):
            return None
        from .certify.provider_compaction import CONTEXT_MGMT_BETA

        param: Any = {"edits": [dict(CM_EDIT)]}
        hdrs = _with_beta(headers, CONTEXT_MGMT_BETA)
    elif compressor == "openai-compaction":
        if not is_responses_path(path):
            return None
        param = [{"type": "compaction", "compact_threshold": COMPACT_THRESHOLD}]
        hdrs = dict(headers)
    else:
        return None
    if "context_management" in obj:
        # Already on upstream of distil: audit what the user runs against it turned off.
        a_obj = {k: v for k, v in obj.items() if k != "context_management"}
        b_obj = obj
    else:
        a_obj, b_obj = obj, {**obj, "context_management": param}
    return Shaped(_dumps(a_obj), _dumps(b_obj), hdrs, None, ra.model)


_APPLIED = re.compile(rb'"applied_edits"\s*:\s*\[\s*\{')
_COMPACTION = re.compile(rb'"type"\s*:\s*"compaction"')


def fired(compressor: str, sh: Shaped, b_response: bytes) -> bool | None:
    """Did the compressor actually change B? None = cannot tell (counted as fired)."""
    if compressor == "distil":
        return sh.a != sh.b
    if compressor == "anthropic-context-editing":
        return bool(_APPLIED.search(b_response))
    if compressor == "openai-compaction":
        return bool(_COMPACTION.search(b_response))
    return None


def signature(raw: bytes) -> str:
    """The shadow's decision signature, plus the Responses API (``output`` items, or the
    ``response.completed`` event of a stream), which the shadow does not read."""
    from .shadow import _sse_payloads, decision_signature_from_body

    sig = decision_signature_from_body(raw)
    if sig != "none":
        return sig
    from .certify.provider_compaction import _responses_signature

    text = raw.decode("utf-8", "replace").strip()
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        for ev in _sse_payloads(text):
            if isinstance(ev, dict) and ev.get("type") == "response.completed":
                out = (ev.get("response") or {}).get("output")
                return _responses_signature(out) if isinstance(out, list) else "none"
        return "none"
    if isinstance(obj, dict) and isinstance(obj.get("output"), list):
        return _responses_signature(obj["output"])
    return "none"


# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #


def estimate_usd(obj: dict[str, Any], body: bytes, model: str | None) -> float | None:
    """Reserved cost of ONE replay arm: every input token at the uncached base rate, plus
    capped output. A cache write (1.25x-2x) can push one arm past it; that is the
    overshoot :class:`Budget` documents. None for an unpriceable model: the audit then
    refuses to spend rather than spend blind."""
    from . import pricing

    p = pricing.resolve(model)
    if p is None:
        return None
    try:
        asked = int(obj.get("max_tokens") or obj.get("max_output_tokens") or OUT_TOKENS_EST)
    except (TypeError, ValueError):
        asked = OUT_TOKENS_EST
    return (len(body) / BYTES_PER_TOKEN) * p.input + min(asked, OUT_TOKENS_EST) * p.output


def billed_usd(usage: dict[str, int], model: str | None) -> float:
    """What one replay actually cost, cache tiers included (the provider's own usage)."""
    from . import pricing
    from .proxy import billed_input_equiv

    p = pricing.resolve(model)
    return 0.0 if p is None else billed_input_equiv(usage, model) * p.input


def list_usd(usage: dict[str, int], model: str | None) -> float | None:
    """One arm at UNCACHED list price: every input token at the base rate. The arms hit
    the prefix cache differently by construction (replay order, and an edit that busts
    the prefix), so a cache-priced comparison would measure the replay order. Same rule
    as :func:`shadow.cost_delta`."""
    from . import pricing

    p = pricing.resolve(model)
    if p is None:
        return None
    return total_input(usage) * p.input + int(usage.get("output_tokens", 0)) * p.output


def total_input(usage: dict[str, int]) -> int:
    return sum(
        int(usage.get(k, 0) or 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )


# --------------------------------------------------------------------------- #
# The daily cap
# --------------------------------------------------------------------------- #


class Budget:
    """Per-day dollar cap. A sample RESERVES its estimate before its first replay and
    settles to the provider-reported cost afterwards, under a file lock, so concurrent
    sessions cannot jointly overshoot. No sample starts unless today's spend plus every
    open reservation plus its own estimate fits under the cap. A sample whose real cost
    exceeds its estimate (cache writes, output past the estimate's cap) overshoots by that
    difference; every later sample then sees the overspend and is refused."""

    def __init__(self, cap_usd: float, path: Path | None = None) -> None:
        self.cap = cap_usd
        self.path = path or budget_path()

    @staticmethod
    def _day(now: float | None = None) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(time.time() if now is None else now))

    def _update(self, fn: Callable[[dict[str, Any]], Any], now: float | None = None) -> Any:
        from . import _filelock

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _filelock.locked(self.path):
            day = self._day(now)
            fresh = {"day": day, "spent": 0.0, "reserved": 0.0, "samples": 0, "skipped_cap": 0}
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(d, dict):
                    raise ValueError("budget file is not an object")
                if d.get("day") == day:
                    float(d["spent"]) + float(d["reserved"])  # malformed counters fail closed
            except FileNotFoundError:
                d = fresh
            except (OSError, ValueError, KeyError, TypeError):
                # Fail closed: a truncated or unreadable file must not reset today's
                # spend to zero and re-open the cap. Treat today as exhausted; the next
                # day starts clean.
                d = {**fresh, "spent": self.cap, "unreadable": True}
            if d.get("day") != day:
                d = fresh
            out = fn(d)
            # Atomic: tmp + fsync + replace, inside the lock (the lock is a sibling
            # ``.lock`` file, so replacing the data file does not drop it). A crash
            # mid-write leaves the previous file, never a truncated one.
            tmp = self.path.with_name(self.path.name + ".tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(d, sort_keys=True) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                _filelock.replace_retrying(tmp, self.path)
            except OSError:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            return out

    def reserve(self, est: float, now: float | None = None) -> bool:
        def go(d: dict[str, Any]) -> bool:
            if d["spent"] + d["reserved"] + est > self.cap:
                d["skipped_cap"] += 1
                return False
            d["reserved"] += est
            return True

        return bool(self._update(go, now))

    def settle(self, est: float, actual: float, now: float | None = None) -> None:
        def go(d: dict[str, Any]) -> None:
            d["reserved"] = max(0.0, d["reserved"] - est)
            d["spent"] += actual
            d["samples"] += 1

        self._update(go, now)

    def today(self, now: float | None = None) -> dict[str, Any]:
        return dict(self._update(lambda d: d, now))


# --------------------------------------------------------------------------- #
# The sampler (request path) and the replay (background thread)
# --------------------------------------------------------------------------- #

Post = Callable[[str, bytes, dict[str, str]], tuple[int, dict[str, str], bytes]]


def post_via(base: str, path: str, body: bytes, headers: dict[str, str]) -> tuple[int, dict, bytes]:
    """POST through the user's own loopback proxy (``--via``)."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=body,
        headers={**headers, "Content-Length": str(len(body))},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 — loopback only
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read() if exc.fp else b""


@dataclass
class Auditor:
    cfg: AuditConfig
    rng: random.Random = field(default_factory=random.Random)
    via_post: Callable[..., tuple[int, dict, bytes]] = post_via

    @classmethod
    def load(cls) -> Auditor | None:
        """The opted-in auditor, or None (the default: no config file). Fail-open."""
        try:
            if not config_path().exists():
                return None
            cfg = AuditConfig.load()
        except Exception:  # noqa: BLE001 — the audit must never break the proxy
            return None
        return None if cfg is None else cls(cfg)

    def should_sample(self) -> bool:
        """Request path: one random draw, no I/O."""
        return self.rng.random() < self.cfg.rate

    def run(
        self,
        path: str,
        raw: bytes,
        served: bytes,
        headers: dict[str, str],
        post: Post,
        book: Callable[..., None] | None = None,
        conv: str = "",
        ledger: Path | None = None,
    ) -> dict[str, Any] | None:
        """One audited sample: reserve, replay A/A'/B in random order, settle, record.
        Returns the row written, or None when nothing was recorded."""
        from .shadow import ShadowLedger
        from .streamrelay import scan_usage

        sh = shape(self.cfg.compressor, path, raw, served, headers, self.cfg.via)
        if sh is None:
            return None
        per_arm = estimate_usd(json.loads(sh.a), sh.a, sh.model)
        if per_arm is None:
            return None
        est = 3 * per_arm
        budget = Budget(self.cfg.cap_usd)
        if not budget.reserve(est):
            return None
        arms: list[tuple[str, bytes, dict[str, str], str | None]] = [
            ("a", sh.a, headers, None),
            ("aa", sh.a, headers, None),
            ("b", sh.b, sh.b_headers, sh.via),
        ]
        self.rng.shuffle(arms)
        sigs: dict[str, str] = {}
        usage: dict[str, dict[str, int]] = {}
        b_resp = b""
        ok = True
        try:
            for name, body, hdrs, via in arms:
                st, _h, rbody = (
                    self.via_post(via, path, body, hdrs) if via else post(path, body, hdrs)
                )
                ok = ok and 200 <= st < 300
                if not ok:
                    break  # the sample is lost either way; do not pay for the rest
                sigs[name] = signature(rbody)
                usage[name] = scan_usage(rbody[:16384] + b"\n" + rbody[-16384:])
                if name == "b":
                    b_resp = rbody
        finally:
            budget.settle(est, sum(billed_usd(u, sh.model) for u in usage.values()))
            if book is not None:
                book("audit", sh.model, list(usage.values()), conv)
        if not ok or "none" in sigs.values():
            return None
        ua, ub = usage["a"], usage["b"]
        ev: dict[str, Any] = {
            "compressor": self.cfg.compressor,
            "aa_equal": sigs["a"] == sigs["aa"],
            "fired": fired(self.cfg.compressor, sh, b_resp),
            "model": sh.model,
            "sig_a": sigs["a"],
            "sig_aa": sigs["aa"],
            "sig_b": sigs["b"],
            "in_a": total_input(ua),
            "in_b": total_input(ub),
            "out_a": int(ua.get("output_tokens", 0)),
            "out_b": int(ub.get("output_tokens", 0)),
            "usd_a": list_usd(ua, sh.model),
            "usd_b": list_usd(ub, sh.model),
        }
        led = ledger or ledger_path()
        ShadowLedger().record(sigs["a"] == sigs["b"], kind="paired", evidence=ev, path=led)
        return {"equivalent": sigs["a"] == sigs["b"], **ev}

    def thread(
        self,
        path: str,
        raw: bytes,
        served: bytes,
        headers: dict[str, str],
        post: Post,
        book: Callable[..., None] | None = None,
        conv: str = "",
    ) -> threading.Thread:
        def target() -> None:
            try:
                self.run(path, raw, served, dict(headers), post, book, conv)
            except Exception:  # noqa: BLE001 — never surfaces into a request
                log.debug("audit sample failed", exc_info=True)

        return threading.Thread(target=target, daemon=True)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _rows(path: Path | None = None) -> list[dict[str, Any]]:
    from .shadow import SIG_VERSION, _rows

    return [
        r
        for r in _rows(path or ledger_path())
        if r.get("sig") == SIG_VERSION and r.get("kind") == "paired" and r.get("compressor")
    ]


def _need_for(values: list[float], offset: float = 0.0) -> int | None:
    """Samples needed for a 95% interval on mean(values)+offset to exclude 0 at the
    observed spread; None when the effect is zero (no n resolves it)."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n + offset
    sd = math.sqrt(sum((v - mean + offset) ** 2 for v in values) / (n - 1))
    if mean == 0:
        return None
    return max(0, math.ceil((1.96 * sd / abs(mean)) ** 2) - n)


@dataclass
class CompressorReport:
    compressor: str
    audited: int
    fired: int
    decisions: str  # the shadow's one honest sentence for the fired rows
    diff: float | None
    diff_ci: tuple[float, float] | None
    saved_usd_mean: float | None  # per request, A - B at uncached list price; + = saved
    saved_usd_ci: tuple[float, float] | None
    saved_usd_total: float | None
    priced: int
    verdict: str  # "helped" | "hurt" | "can't tell yet"
    why: str
    need: int | None


def summarize(compressor: str, rows: list[dict[str, Any]]) -> CompressorReport:
    """Pure function of the ledger rows for one compressor."""
    from .shadow import VERDICT_MIN_AB, ShadowLedger, bootstrap_ci

    led = ShadowLedger()
    for r in rows:
        if r.get("fired") is not False:  # decisions are judged where it changed something
            led._ingest(r)
    eq = led.equivalence()
    saved = [
        float(r["usd_a"]) - float(r["usd_b"])
        for r in rows
        if isinstance(r.get("usd_a"), (int, float)) and isinstance(r.get("usd_b"), (int, float))
    ]
    s_mean = sum(saved) / len(saved) if saved else None
    s_ci = bootstrap_ci(saved) if saved else None
    rep = CompressorReport(
        compressor=compressor,
        audited=len(rows),
        fired=eq.n_paired,
        decisions=eq.line(),
        diff=eq.diff,
        diff_ci=eq.diff_ci,
        saved_usd_mean=s_mean,
        saved_usd_ci=s_ci,
        saved_usd_total=sum(saved) if saved else None,
        priced=len(saved),
        verdict="can't tell yet",
        why="",
        need=None,
    )
    if eq.below_floor or s_ci is None or eq.diff_ci is None:
        rep.need = max(0, VERDICT_MIN_AB - eq.n_paired)
        rep.why = f"{eq.shortfall or 'no priced rows'}: need {rep.need} more audited requests"
        if compressor != "distil" and len(rows) > eq.n_paired:
            rep.why += f" where {compressor} fired"
        return rep
    d_lo, d_hi = eq.diff_ci
    s_lo, s_hi = s_ci
    if d_hi < 0:
        rep.verdict = "hurt"
        rep.why = "changes the agent's next action more often than the model disagrees with itself"
    elif s_hi < 0:
        rep.verdict = "hurt"
        rep.why = "costs more per request than it saves"
    elif s_lo > 0 and d_lo >= -EQUIV_MARGIN:
        rep.verdict = "helped"
        rep.why = (
            "saves money per request, and decisions stay within "
            f"{EQUIV_MARGIN * 100:.0f} points of the model's own self-agreement"
        )
    else:
        diffs = [
            float(int(r["equivalent"]) - int(r["aa_equal"]))
            for r in rows
            if r.get("fired") is not False
        ]
        needs = [x for x in (_need_for(saved), _need_for(diffs, EQUIV_MARGIN)) if x is not None]
        rep.need = max(needs) if needs else None
        rep.why = (
            "the intervals still straddle the line"
            + (f": need ≈{rep.need} more audited requests" if rep.need else "")
            if needs
            else "no measurable difference at this n"
        )
    return rep


def report(path: Path | None = None) -> list[CompressorReport]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in _rows(path):
        by.setdefault(str(r["compressor"]), []).append(r)
    return [summarize(c, by[c]) for c in sorted(by)]


def _usd(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{'+' if x >= 0 else '−'}${abs(x):,.4f}"


def render(reports: list[CompressorReport], budget: dict[str, Any], cap: float | None) -> str:
    L = ["distil audit — referee report (this machine's own traffic)"]
    cap_s = f" of ${cap:.2f} cap" if cap is not None else ""
    L.append(
        f"  today: ${budget.get('spent', 0.0):.4f}{cap_s} spent on {budget.get('samples', 0)} "
        f"samples; {budget.get('skipped_cap', 0)} skipped at the cap"
    )
    if not reports:
        L += ["", "  nothing audited yet. Turn it on: distil audit --compressor <name>"]
    for r in reports:
        need = f" (need {r.need} more)" if r.verdict == "can't tell yet" and r.need else ""
        L += ["", f"  {r.compressor}: {r.verdict.upper()}{need} — {r.why}"]
        L.append(f"    audited {r.audited} requests; changed the request on {r.fired}")
        L.append(f"    decisions (1{{A=B}} − 1{{A=A'}}, paired): {r.decisions}")
        ci = (
            f" (95% CI {_usd(r.saved_usd_ci[0])} … {_usd(r.saved_usd_ci[1])})"
            if r.saved_usd_ci
            else ""
        )
        L.append(
            f"    $ saved per request (A − B, uncached list price): {_usd(r.saved_usd_mean)}{ci}; "
            f"total {_usd(r.saved_usd_total)} over {r.priced} priced"
        )
    L += [
        "",
        "  turns: not measurable here — a replay is one request. Extra turns, failed attempts",
        "  and cost per solved task are measured by `distil ab` (your sessions) and the",
        "  scoreboard (docs/scoreboard.html). Cache effects are not priced per request.",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def disclosure(cfg: AuditConfig) -> str:
    where = (
        f"through your Headroom proxy at {cfg.via}"
        if cfg.compressor == "headroom"
        else "to the provider the request was already going to"
    )
    return (
        f"distil audit --compressor {cfg.compressor} (opt-in)\n"
        f"  For {cfg.rate * 100:g}% of requests, distil re-sends the request three more times on "
        "YOUR key, in the background:\n"
        "    A, A' — exactly as your client sent it (the model's own noise floor)\n"
        f"    B     — as {cfg.compressor} shapes it: {COMPRESSORS[cfg.compressor]}, {where}\n"
        "  and compares the agent's next action and the provider's billed tokens.\n"
        f"  Cost: capped at ${cfg.cap_usd:.2f}/day. A sample starts only if its estimate fits under "
        "what is\n  left today; one sample can overshoot by its own estimate error (cache writes).\n"
        "  At the default cap a ~100k-token Sonnet request is about one sample a day.\n"
        "  Content never leaves this machine except to where it was already going; the audit "
        "ledger\n  (~/.distil/audit.jsonl) holds hashes, token counts and dollars only.\n"
        "  Takes effect when the proxy next starts. Off: distil audit off. Results: distil audit "
        "report"
    )


def cmd_audit(args: argparse.Namespace) -> int:
    if args.compressor:
        if args.action != "status":
            print("distil audit: --compressor turns the audit on; drop the action", file=sys.stderr)
            return 2
        new = AuditConfig(args.compressor, args.rate, args.cap_usd, args.via, time.time())
        why = validate(new)
        if why:
            print(f"distil audit: {why}", file=sys.stderr)
            return 2
        print(disclosure(new))
        if not args.yes:
            if not sys.stdin.isatty():
                print("distil audit: not turned on — re-run with --yes to confirm", file=sys.stderr)
                return 2
            if input("Turn it on? [y/N] ").strip().lower() not in ("y", "yes"):
                print("not turned on")
                return 1
        new.save()
        print(f"audit on: {new.compressor}")
        return 0
    cfg = AuditConfig.load()
    if args.action == "off":
        try:
            config_path().unlink()
        except FileNotFoundError:
            pass
        print("audit off (the ledger is kept: distil audit report)")
        return 0
    cap = cfg.cap_usd if cfg else None
    budget = Budget(cap or DEFAULT_CAP_USD).today()
    if args.action == "report":
        reps = report()
        if args.json:
            print(
                json.dumps({"budget": budget, "compressors": [asdict(r) for r in reps]}, indent=2)
            )
        else:
            print(render(reps, budget, cap))
        return 0
    if cfg is None:
        print(
            "distil audit: off. Turn it on with: distil audit --compressor {"
            + ",".join(c for c in COMPRESSORS if c not in NOT_AUDITABLE)
            + "}"
        )
    else:
        print(disclosure(cfg))
        print(
            f"  today: ${budget['spent']:.4f} of ${cfg.cap_usd:.2f} on {budget['samples']} samples"
        )
    return 0


def register(sub: Any) -> None:
    """``distil audit`` — one call from :mod:`distil.cli`."""
    p = sub.add_parser(
        "audit",
        help="referee any compressor on your own traffic: did it help or hurt, in $ and "
        "decisions (opt-in, capped)",
    )
    p.add_argument("action", nargs="?", default="status", choices=["status", "report", "off"])
    p.add_argument("--compressor", choices=list(COMPRESSORS), help="turn the audit on for this one")
    p.add_argument(
        "--rate", type=float, default=DEFAULT_RATE, help="sampled fraction (default 0.02)"
    )
    p.add_argument(
        "--cap-usd", type=float, default=DEFAULT_CAP_USD, help="hard per-day $ cap (default 1)"
    )
    p.add_argument("--via", metavar="URL", help="headroom: your running Headroom proxy (loopback)")
    p.add_argument("--yes", action="store_true", help="confirm without the prompt")
    p.add_argument("--json", action="store_true", help="report: machine-readable output")
    p.set_defaults(func=cmd_audit)

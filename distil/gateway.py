"""Managed multi-tenant gateway with per-tenant savings accounting and a live dashboard.

Drop-in extension of the distil proxy: adds per-tenant token/dollar accounting,
a JSON stats endpoint (/distil/stats), and a self-contained dark HTML dashboard
(/distil/dashboard).  All other paths are handled identically to proxy.py.

Usage
-----
::

    from distil.gateway import serve_gateway
    serve_gateway(host="127.0.0.1", port=8789, upstream="https://api.anthropic.com")

Or from the module::

    python -m distil.gateway
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ._log import log
from .adapters.anthropic import compress_messages
from .adapters.gemini import compress_generate_request
from .adapters.gemini import count_tokens as _gemini_count
from .adapters.gemini import is_gemini_path
from .authz import AuthzError as _AuthzError
from .authz import identity_from_claims as _identity_from_claims
from .authz import oidc_config_from_env as _oidc_config_from_env
from .authz import verify_jwt as _verify_jwt
from . import audit as _audit
from .gateway_keys import GatewayKeyStore, KeyRecord  # noqa: F401
from .httpguard import parse_content_length, safe_forward_path
from .pricing import Pricing, get as pricing_get
from .proxy import (
    _OPENER,
    _UPSTREAM_TIMEOUT,
    QuietHTTPServer,
    _install_sigterm_flush,
    _warn_if_version_skew,
)
from .tokenizer import DEFAULT as _tokenizer


class _OidcRejected(Exception):
    """Raised after a 401 has already been sent for a rejected OIDC token.

    Lets the auth path unwind without a second response being written — the 401
    is already on the wire by the time this propagates.
    """


# Safe tenant label: bounded length, no markup / control characters.
_TENANT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# ---------------------------------------------------------------------------
# Paths that carry a ``messages`` payload worth compressing
# ---------------------------------------------------------------------------

_COMPRESSIBLE_PATHS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"})

# Hop-by-hop headers must never be forwarded.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailers",
        "upgrade",
    }
)


# ---------------------------------------------------------------------------
# Per-tenant stats
# ---------------------------------------------------------------------------


@dataclass
class TenantStats:
    requests: int = 0
    tokens_baseline: int = 0
    tokens_compressed: int = 0
    # Requests refused by a per-tenant quota (RPM or daily tokens). Counted
    # separately from `requests`, which only ever counts work actually done:
    # folding the two would make a throttled tenant look busy instead of blocked.
    rejected_quota: int = 0

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_baseline - self.tokens_compressed)

    def dollars_saved(self, price: Pricing) -> float:
        """USD saved, based on input token pricing."""
        return self.tokens_saved * price.input

    def pct_saved(self) -> float:
        if self.tokens_baseline == 0:
            return 0.0
        return self.tokens_saved / self.tokens_baseline * 100.0


# ---------------------------------------------------------------------------
# Thread-safe gateway state
# ---------------------------------------------------------------------------


# Cap the per-tenant map so a key-spraying client can't grow it without bound;
# least-recently-active tenants are evicted first. Generous for real fleets.
_MAX_TENANTS = 50_000
_CHECKPOINT_SECS = 30.0  # max staleness of persisted tenant accounting on a hard crash


def _state_path() -> Path:
    """Where per-tenant accounting is persisted across restarts (honours DISTIL_HOME)."""
    home = os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))
    return Path(home) / "gateway_state.json"


# ---------------------------------------------------------------------------
# In-memory rate limiter (RPM + per-UTC-day token quota)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Per-tenant RPM + daily-token quota enforcer.

    Both windows are in-memory only — a gateway restart resets them.  This is
    accepted behaviour: quotas protect against runaway clients, not against a
    deliberate restart-to-reset.

    # ponytail: global lock, per-tenant locks if throughput matters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # tenant -> (count_this_window, window_start_monotonic)
        self._rpm: dict[str, tuple[int, float]] = {}
        # tenant -> (tokens_today, utc_day_int)
        self._daily: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _today() -> int:
        return int(time.time() // 86400)

    def check_rpm(self, tenant: str, limit: int) -> bool:
        """Return True (allowed) or False (rate-limited).  Limit 0 = unlimited."""
        if not limit:
            return True
        with self._lock:
            now = time.monotonic()
            count, t0 = self._rpm.get(tenant, (0, now))
            if now - t0 >= 60.0:
                count, t0 = 0, now
            if count >= limit:
                return False
            self._rpm[tenant] = (count + 1, t0)
        return True

    def check_daily_tokens(self, tenant: str, limit: int, tokens: int) -> bool:
        """Return True if adding *tokens* stays within *limit* for today.

        Also records the usage — call once per request, after RPM passes.
        Limit 0 = unlimited.
        """
        if not limit:
            return True
        day = self._today()
        with self._lock:
            used, d = self._daily.get(tenant, (0, day))
            if d != day:
                used = 0
            if used + tokens > limit:
                return False
            self._daily[tenant] = (used + tokens, day)
        return True


class GatewayState:
    """Thread-safe, LRU-bounded map of tenant_id -> TenantStats, persistable to disk."""

    def __init__(self, price: Pricing) -> None:
        self._lock = threading.Lock()
        self._tenants: OrderedDict[str, TenantStats] = OrderedDict()
        self._price = price
        self._last_save = time.monotonic()

    def record_quota_rejection(self, tenant: str) -> None:
        """Count one request refused by a per-tenant quota.

        Exported as ``distil_requests_rejected_total`` so an operator can see a
        throttled tenant without reading logs — quota enforcement that is invisible
        is indistinguishable from an outage from the client's side.
        """
        with self._lock:
            s = self._tenants.get(tenant)
            if s is None:
                s = TenantStats()
                self._tenants[tenant] = s
                if len(self._tenants) > _MAX_TENANTS:
                    self._tenants.popitem(last=False)
            else:
                self._tenants.move_to_end(tenant)
            s.rejected_quota += 1

    def record(self, tenant: str, baseline_tokens: int, compressed_tokens: int) -> None:
        """Accumulate one request's worth of token counts for *tenant* (LRU-bounded)."""
        with self._lock:
            s = self._tenants.get(tenant)
            if s is None:
                s = TenantStats()
                self._tenants[tenant] = s
                if len(self._tenants) > _MAX_TENANTS:
                    self._tenants.popitem(last=False)  # evict least-recently-active
            else:
                self._tenants.move_to_end(tenant)  # mark active for LRU
            s.requests += 1
            s.tokens_baseline += baseline_tokens
            s.tokens_compressed += compressed_tokens
        # Crash-safety checkpoint: the shutdown-path save() never runs on
        # kill -9/OOM, which would zero same-day tenant accounting. Throttled
        # so it costs at most one atomic write per interval; claiming the
        # timestamp before saving keeps concurrent racers to one write-ish.
        now = time.monotonic()
        if now - self._last_save >= _CHECKPOINT_SECS:
            self._last_save = now
            self.save()

    def save(self, path: Path | None = None) -> None:
        """Atomically persist per-tenant counters so a restart/SIGTERM doesn't zero them."""
        path = path or _state_path()
        with self._lock:
            data = {
                "tenants": {
                    tid: {
                        "requests": s.requests,
                        "tokens_baseline": s.tokens_baseline,
                        "tokens_compressed": s.tokens_compressed,
                        "rejected_quota": s.rejected_quota,
                    }
                    for tid, s in self._tenants.items()
                }
            }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # best-effort — never let a failed save crash shutdown

    def load(self, path: Path | None = None) -> None:
        """Restore per-tenant counters written by :meth:`save` (best-effort)."""
        path = path or _state_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        tenants = data.get("tenants", {}) if isinstance(data, dict) else {}
        with self._lock:
            for tid, d in tenants.items():
                try:
                    s = TenantStats()
                    s.requests = int(d["requests"])
                    s.tokens_baseline = int(d["tokens_baseline"])
                    s.tokens_compressed = int(d["tokens_compressed"])
                    # Older state files predate this field — default, don't discard.
                    s.rejected_quota = int(d.get("rejected_quota", 0))
                except (KeyError, TypeError, ValueError):
                    continue  # skip a corrupt tenant entry
                self._tenants[tid] = s
            # A pre-cap (or hand-edited) state file may exceed the ceiling;
            # enforce the invariant at load, not just on record().
            while len(self._tenants) > _MAX_TENANTS:
                self._tenants.popitem(last=False)

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable dict with per-tenant stats and totals."""
        with self._lock:
            tenants: list[dict[str, Any]] = []
            tot_req = 0
            tot_baseline = 0
            tot_compressed = 0
            tot_rejected = 0
            for tid, s in sorted(
                self._tenants.items(), key=lambda kv: kv[1].tokens_saved, reverse=True
            ):
                dollars = s.dollars_saved(self._price)
                tenants.append(
                    {
                        "tenant": tid,
                        "requests": s.requests,
                        "tokens_baseline": s.tokens_baseline,
                        "tokens_compressed": s.tokens_compressed,
                        "tokens_saved": s.tokens_saved,
                        "dollars_saved": round(dollars, 6),
                        "pct_saved": round(s.pct_saved(), 2),
                        "rejected_quota": s.rejected_quota,
                    }
                )
                tot_req += s.requests
                tot_baseline += s.tokens_baseline
                tot_compressed += s.tokens_compressed
                tot_rejected += s.rejected_quota

            tot_saved = max(0, tot_baseline - tot_compressed)
            tot_dollars = tot_saved * self._price.input
            tot_pct = (tot_saved / tot_baseline * 100.0) if tot_baseline else 0.0

            totals = {
                "requests": tot_req,
                "tokens_baseline": tot_baseline,
                "tokens_compressed": tot_compressed,
                "tokens_saved": tot_saved,
                "dollars_saved": round(tot_dollars, 6),
                "pct_saved": round(tot_pct, 2),
                "rejected_quota": tot_rejected,
            }
            return {"tenants": tenants, "totals": totals}


# ---------------------------------------------------------------------------
# Tenant identification — no raw key ever stored or logged
# ---------------------------------------------------------------------------


def tenant_of(headers: Any, *, trust_tenant_header: bool = False) -> str:
    """Derive a tenant identifier from request headers.

    Tenant identity comes from the AUTHENTICATED credential (a stable
    ``anon-<sha256(key)[:8]>`` id), never from a client-writable header: any
    caller could otherwise send ``x-distil-tenant: acme-corp`` and book its
    traffic under another tenant's accounting line. The explicit header is
    honored only when the operator opts in (``trust_tenant_header=True`` /
    ``--trust-tenant-header``) for deployments where an upstream gateway they
    control sets it.
    """
    if trust_tenant_header:
        explicit = headers.get("x-distil-tenant")
        if explicit:
            label = explicit.strip()
            # Bounded, safe labels only; anything else falls through to the
            # credential-derived id rather than entering accounting/dashboard.
            if _TENANT_RE.match(label):
                return label

    for header in ("x-api-key", "authorization"):
        val = headers.get(header)
        if val:
            h = hashlib.sha256(val.encode()).hexdigest()[:8]
            return f"anon-{h}"

    return "default"


# ---------------------------------------------------------------------------
# Token-saving estimator (mirrors proxy.py)
# ---------------------------------------------------------------------------


def _count_tokens(msgs: list[dict[str, Any]]) -> int:
    total = 0
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _tokenizer.count(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "content"):
                    val = block.get(key)
                    if isinstance(val, str):
                        total += _tokenizer.count(val)
                    elif isinstance(val, list):
                        for sub in val:
                            if isinstance(sub, dict):
                                sv = sub.get("text", "")
                                if isinstance(sv, str):
                                    total += _tokenizer.count(sv)
    return total


# ---------------------------------------------------------------------------
# Dashboard HTML generator
# ---------------------------------------------------------------------------


def _dashboard_html(snap: dict[str, Any]) -> str:
    tenants = snap["tenants"]
    totals = snap["totals"]

    rows = ""
    for t in tenants:
        rows += (
            f"<tr>"
            f"<td>{html.escape(str(t['tenant']))}</td>"
            f"<td>{t['requests']}</td>"
            f"<td>{t['tokens_saved']:,}</td>"
            f"<td>${t['dollars_saved']:.4f}</td>"
            f"<td>{t['pct_saved']:.1f}%</td>"
            f"</tr>\n"
        )

    if not rows:
        rows = '<tr><td colspan="5" class="empty">No requests recorded yet.</td></tr>\n'

    totals_row = (
        f"<tr class='total-row'>"
        f"<th scope='row'><strong>TOTAL</strong></th>"
        f"<td><strong>{totals['requests']}</strong></td>"
        f"<td><strong>{totals['tokens_saved']:,}</strong></td>"
        f"<td><strong>${totals['dollars_saved']:.4f}</strong></td>"
        f"<td><strong>{totals['pct_saved']:.1f}%</strong></td>"
        f"</tr>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>distil gateway — live dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #06070a;
    color: #e7e9ee;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    padding: 2rem;
    min-height: 100vh;
  }}
  h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, #8b7bff, #5ad1c9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .subtitle {{
    color: #8b93a3;
    font-size: 0.85rem;
    margin-bottom: 2rem;
  }}
  .headline-cards {{
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }}
  .card {{
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    min-width: 160px;
    flex: 1;
  }}
  .card-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b93a3;
    margin-bottom: 0.35rem;
  }}
  .card-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #8b7bff;
  }}
  .card-value.teal {{ color: #5ad1c9; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  thead th {{
    text-align: left;
    padding: 0.7rem 1rem;
    border-bottom: 2px solid #1e2130;
    color: #8b93a3;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
  }}
  tbody tr {{
    border-bottom: 1px solid #13161f;
    transition: background 0.12s;
  }}
  tbody tr:hover, tfoot tr:hover {{ background: #0d1020; }}
  tbody td {{
    padding: 0.75rem 1rem;
    font-variant-numeric: tabular-nums;
  }}
  tbody td:first-child {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: #8b7bff;
    font-size: 0.82rem;
  }}
  .total-row td, .total-row th {{
    border-top: 2px solid #1e2130;
    color: #5ad1c9;
    padding: 0.75rem 1rem;
    font-variant-numeric: tabular-nums;
    text-align: left;
    font-weight: 400;
  }}
  .total-row td:first-child, .total-row th:first-child {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
  }}
  caption.sr-only {{
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
  }}
  .empty {{
    color: #4b5563;
    text-align: center;
    padding: 2rem !important;
    font-style: italic;
  }}
  .table-wrap {{
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 10px;
    overflow: hidden;
  }}
  .refresh-note {{
    color: #8b93a3;
    font-size: 0.72rem;
    text-align: right;
    margin-top: 0.75rem;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 0.6rem;
  }}
  .pause-btn {{
    background: #0f1117;
    border: 1px solid #1e2130;
    color: #c7c9d1;
    font-size: 0.72rem;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    cursor: pointer;
  }}
  .pause-btn:hover {{ background: #171a24; }}
  .pause-btn:focus-visible {{ outline: 2px solid #8b7bff; outline-offset: 2px; }}
</style>
</head>
<body>
<h1>distil gateway</h1>
<p class="subtitle">Per-tenant token compression leaderboard &mdash; refreshes every 5 s</p>

<div class="headline-cards">
  <div class="card">
    <div class="card-label">Total Requests</div>
    <div class="card-value" id="c-requests">{totals["requests"]}</div>
  </div>
  <div class="card">
    <div class="card-label">Tokens Saved</div>
    <div class="card-value teal" id="c-tokens">{totals["tokens_saved"]:,}</div>
  </div>
  <div class="card">
    <div class="card-label">Dollars Saved</div>
    <div class="card-value" id="c-dollars">${totals["dollars_saved"]:.4f}</div>
  </div>
  <div class="card">
    <div class="card-label">Compression Rate</div>
    <div class="card-value teal" id="c-pct">{totals["pct_saved"]:.1f}%</div>
  </div>
</div>

<div class="table-wrap">
<table>
  <caption class="sr-only">Per-tenant token compression leaderboard</caption>
  <thead>
    <tr>
      <th scope="col">Tenant</th>
      <th scope="col">Requests</th>
      <th scope="col">Tokens Saved</th>
      <th scope="col">$ Saved</th>
      <th scope="col">% Saved</th>
    </tr>
  </thead>
  <tbody id="tenant-rows">
    {rows}
  </tbody>
  <tfoot>
    {totals_row}
  </tfoot>
</table>
</div>
<p class="refresh-note">
  <span id="refresh-note">Updates every 5 s &bull; distil gateway</span>
  <button type="button" class="pause-btn" id="pause-btn" aria-pressed="false">Pause</button>
</p>
<script>
(function() {{
  var POLL_MS = 5000;
  var tbody = document.getElementById("tenant-rows");
  var pauseBtn = document.getElementById("pause-btn");
  var note = document.getElementById("refresh-note");
  var timer = null, paused = false;

  function esc(s) {{
    return String(s).replace(/[&<>"']/g, function(c) {{
      return ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}})[c];
    }});
  }}

  function rowsHtml(tenants, totals) {{
    var out = "";
    if (!tenants.length) {{
      out += '<tr><td colspan="5" class="empty">No requests recorded yet.</td></tr>';
    }} else {{
      tenants.forEach(function(t) {{
        out += "<tr>" +
          "<td>" + esc(t.tenant) + "</td>" +
          "<td>" + t.requests + "</td>" +
          "<td>" + t.tokens_saved.toLocaleString("en-US") + "</td>" +
          "<td>$" + t.dollars_saved.toFixed(4) + "</td>" +
          "<td>" + t.pct_saved.toFixed(1) + "%</td>" +
          "</tr>";
      }});
    }}
    out += "<tr class='total-row'>" +
      "<td><strong>TOTAL</strong></td>" +
      "<td><strong>" + totals.requests + "</strong></td>" +
      "<td><strong>" + totals.tokens_saved.toLocaleString("en-US") + "</strong></td>" +
      "<td><strong>$" + totals.dollars_saved.toFixed(4) + "</strong></td>" +
      "<td><strong>" + totals.pct_saved.toFixed(1) + "%</strong></td>" +
      "</tr>";
    return out;
  }}

  function renderCards(totals) {{
    document.getElementById("c-requests").textContent = totals.requests;
    document.getElementById("c-tokens").textContent = totals.tokens_saved.toLocaleString("en-US");
    document.getElementById("c-dollars").textContent = "$" + totals.dollars_saved.toFixed(4);
    document.getElementById("c-pct").textContent = totals.pct_saved.toFixed(1) + "%";
  }}

  function poll() {{
    fetch("/distil/stats", {{cache: "no-store"}}).then(function(r) {{ return r.json(); }}).then(function(d) {{
      renderCards(d.totals);
      tbody.innerHTML = rowsHtml(d.tenants, d.totals);
    }}).catch(function() {{}});
  }}

  function schedule() {{ timer = setInterval(poll, POLL_MS); }}

  pauseBtn.addEventListener("click", function() {{
    paused = !paused;
    pauseBtn.textContent = paused ? "Resume" : "Pause";
    pauseBtn.setAttribute("aria-pressed", paused ? "true" : "false");
    note.textContent = paused ? "Updates paused • distil gateway" : "Updates every 5 s • distil gateway";
    if (paused) {{
      clearInterval(timer);
    }} else {{
      poll();
      schedule();
    }}
  }});

  schedule();
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def build_gateway_handler(
    upstream: str,
    state: GatewayState,
    price: Pricing,
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    admin_token: str | None = None,
    loopback: bool = True,
    trust_tenant_header: bool = False,
    key_store: GatewayKeyStore | None = None,
    require_keys: bool = False,
    rate_limiter: _RateLimiter | None = None,
    default_rpm: int = 0,
    default_daily_tokens: int = 0,
) -> type[BaseHTTPRequestHandler]:
    """Return a BaseHTTPRequestHandler subclass for the multi-tenant gateway.

    Parameters
    ----------
    upstream:
        Base URL of the real LLM API, e.g. ``"https://api.anthropic.com"``.
    state:
        Shared ``GatewayState`` instance updated on every compressible request.
    price:
        ``Pricing`` used for dollar calculations in stats / dashboard.
    lossless_only:
        Policy mode (no tool injection). The reversible digest still runs.
    verbatim:
        When *True*, skip the Tier-1 digest (Tier-0 only) — interactive-safe.
    admin_token:
        When set, ``/distil/stats`` and ``/distil/dashboard`` require
        ``Authorization: Bearer <token>``. When unset AND the server is bound
        to a non-loopback interface, those routes are refused (403): per-tenant
        usage metadata must not be readable by anyone on the network.
    loopback:
        Whether the server is bound to a loopback interface (set by
        ``serve_gateway`` from the bind host).
    trust_tenant_header:
        Honor the client-supplied ``x-distil-tenant`` header for accounting.
        Off by default — tenant identity comes from the credential hash.
    key_store:
        Optional ``GatewayKeyStore``.  When supplied (and when *require_keys*
        is True or active keys exist), inbound requests must present a
        ``dsk-`` key via ``Authorization: Bearer dsk-…`` or ``x-distil-key``.
        Missing or revoked keys → 401.  The distil key is stripped before
        forwarding; the provider credential passes through unchanged.
        When *key_store* is None and *require_keys* is False (the default),
        behaviour is identical to today's: tenant derives from the credential
        hash, no auth gate.
    require_keys:
        Force key auth even when no keys have been issued yet (useful to lock
        down a gateway before distributing its first key).
    rate_limiter:
        Optional ``_RateLimiter`` instance.  If None and limits are configured,
        one is created automatically.
    default_rpm:
        Gateway-wide requests-per-minute cap per tenant (0 = unlimited).
        Per-key overrides win when a key record carries a non-None ``rpm``.
    default_daily_tokens:
        Gateway-wide per-tenant daily input-token cap (0 = unlimited).
        Per-key overrides win when a key record carries a non-None
        ``daily_tokens``.
    """

    _upstream = upstream.rstrip("/")

    # lossless-only implies Tier-0-only: without an injected expand tool the agent
    # cannot recover a Tier-1 digest stub, so a stub there would be irreversibly
    # lossy. Fold it into verbatim (the flag that already disables Tier-1 digests).
    from .policy import AuthMode, may_compress_lossy

    # Route the lossy-allowed decision through policy (single source of truth):
    # subscription / OAuth sessions are lossless-only, forcing Tier-0-only (verbatim).
    _auth_mode = AuthMode.SUBSCRIPTION if lossless_only else AuthMode.PAYG
    verbatim = verbatim or not may_compress_lossy(_auth_mode)

    # Eager-load the streaming relay the handler otherwise imports lazily per
    # request, so a gateway upgraded in place never loads a post-upgrade .py
    # mid-serve against the running interpreter (version skew). Warmed here at
    # server setup; the per-request `from .streamrelay import ...` is then a
    # module-cache hit, and CLI cold start stays cheap.
    from .streamrelay import stream_upstream as _stream_upstream  # noqa: F401

    from . import __version__ as _running_version

    # Version-skew guard: warn once if the gateway was upgraded on disk while this
    # long-lived process keeps running its old in-memory code (see proxy.py).
    _version_state: dict[str, Any] = {"running": _running_version}

    # Key-auth is active when the operator has either issued at least one key
    # (auto-detect) or explicitly set --require-keys.  Once active it never
    # silently falls back to the credential-hash path — auth failures are closed.
    _key_store = key_store
    _rl = rate_limiter if rate_limiter is not None else _RateLimiter()

    class _GatewayHandler(BaseHTTPRequestHandler):
        # HTTP/1.1 so streamed responses can use chunked transfer framing.
        protocol_version = "HTTP/1.1"

        # ----------------------------------------------------------------
        # Silence request logs
        # ----------------------------------------------------------------

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

        # ----------------------------------------------------------------
        # Gateway key auth helpers
        # ----------------------------------------------------------------

        def _auth_required(self) -> bool:
            """True when key auth should be enforced for this request.

            Uses ``has_any_keys()`` (not just active keys) so that revoking
            all keys keeps the gate locked instead of silently reopening it.
            Revoking leaves no valid credential → every request 401s, which is
            the right outcome for a depleted key set.
            """
            return require_keys or (_key_store is not None and _key_store.has_any_keys())

        def _extract_distil_key(self) -> tuple[str | None, str | None]:
            """Pull the dsk- key and the header name it came from.

            Returns ``(raw_key, header_name)`` or ``(None, None)``.
            Checks ``Authorization: Bearer dsk-…`` first, then ``x-distil-key``.
            The header_name is used to strip the distil key before forwarding.
            """
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer dsk-"):
                return auth[len("Bearer ") :], "authorization"
            xdk = self.headers.get("x-distil-key", "")
            if xdk.startswith("dsk-"):
                return xdk, "x-distil-key"
            return None, None

        def _identity_from_oidc(self):
            """Verify an ``Authorization: Bearer <jwt>`` against configured OIDC.

            Returns an Identity, or None when OIDC is not configured or no bearer
            was presented. A bearer that IS presented but fails verification is a
            hard 401 — never a fall-through to "unauthenticated but allowed".
            """
            cfg = _oidc_config_from_env()
            if not cfg.get("issuer"):
                return None  # OIDC disabled
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth.startswith("Bearer dsk-"):
                return None
            token = auth[len("Bearer ") :].strip()
            try:
                claims = _verify_jwt(
                    token,
                    secret=cfg["secret"],
                    public_key_pem=cfg["public_key_pem"],
                    issuer=cfg["issuer"],
                    audience=cfg["audience"],
                )
            except _AuthzError as exc:
                self._reject(401, f"OIDC token rejected: {exc}")
                raise _OidcRejected from exc
            return _identity_from_claims(
                claims,
                role_claim=cfg["role_claim"],
                tenant_claim=cfg["tenant_claim"],
            )

        def _check_inbound_auth(self) -> tuple[str | None, str | None] | None:
            """Validate the gateway key when auth is required.

            Returns ``(tenant_override, strip_header)`` on success, or ``None``
            if auth is off.  On failure, sends the 401 and returns the sentinel
            ``("", "")`` — callers check ``result is None`` vs ``result[0] == ""``.
            """
            # Reset per-request key state (handler instances serve sequential
            # keep-alive requests — a stale per-key limit must not carry over).
            self._key_daily_tokens = None

            if not self._auth_required():
                return None  # auth off — no key needed

            if _key_store is None:
                # require_keys=True but no key store was supplied (e.g. in tests
                # that set require_keys without issuing keys yet).
                self._reject(
                    401,
                    "gateway key required but no key store is configured "
                    "(distil gateway keys issue --tenant <name>)",
                )
                return ("", "")

            raw_key, strip_hdr = self._extract_distil_key()
            if raw_key is None:
                # No dsk- key. If OIDC is configured, a verified bearer JWT is an
                # equally valid credential — additive, so enabling OIDC can never
                # lock out a deployment that already runs on issued keys.
                try:
                    ident = self._identity_from_oidc()
                except _OidcRejected:
                    # 401 already written by the extractor; unwind cleanly rather
                    # than letting the handler emit a second response.
                    return ("", "")
                if ident is not None:
                    try:
                        ident.require("operator")  # proxying is an operator action
                    except _AuthzError as exc:
                        self._reject(403, str(exc))
                        return ("", "")
                    if not _rl.check_rpm(ident.tenant, default_rpm):
                        state.record_quota_rejection(ident.tenant)
                        body = json.dumps({"error": "rate limit exceeded"}).encode()
                        self._relay(
                            429,
                            {"Content-Type": "application/json"},
                            body,
                            {"Retry-After": "60"},
                        )
                        return ("", "")
                    # Strip the bearer so the upstream never sees our IdP token.
                    return (ident.tenant, "authorization")
                self._reject(
                    401,
                    "gateway key required (Authorization: Bearer dsk-… or x-distil-key header)",
                )
                return ("", "")

            rec = _key_store.lookup(raw_key)
            if rec is None:
                # Content-free: the presented key is never logged, only the fact of
                # a rejection and where it came from.
                _audit.record(
                    _audit.AUTH_FAIL,
                    reason="invalid or revoked gateway key",
                    remote=self.client_address[0] if self.client_address else None,
                )
                self._reject(401, "invalid or revoked gateway key")
                return ("", "")

            # RPM gate: per-key limit wins over gateway default.
            rpm_limit = rec.rpm if rec.rpm is not None else default_rpm
            if not _rl.check_rpm(rec.tenant, rpm_limit):
                state.record_quota_rejection(rec.tenant)
                _audit.record(
                    _audit.RATE_LIMITED,
                    key_id=rec.id,
                    tenant=rec.tenant,
                    limit_rpm=rpm_limit,
                    remote=self.client_address[0] if self.client_address else None,
                )
                body = json.dumps({"error": "rate limit exceeded"}).encode()
                self._relay(429, {"Content-Type": "application/json"}, body, {"Retry-After": "60"})
                return ("", "")

            # Per-key daily quota override, resolved HERE (the only place with the
            # KeyRecord) and read by _handle_compressible — it wins over the
            # gateway default, same precedence as the per-key rpm above.
            self._key_daily_tokens = rec.daily_tokens

            _audit.record(
                _audit.AUTH_OK,
                key_id=rec.id,
                tenant=rec.tenant,
                remote=self.client_address[0] if self.client_address else None,
            )
            return rec.tenant, strip_hdr

        # ----------------------------------------------------------------
        # HTTP verb dispatch
        # ----------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/distil/health":
                # Liveness probe: unauthenticated by design (leaks nothing but
                # "up"), answers locally, never touches the billed upstream.
                payload = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path == "/distil/stats":
                if self._admin_authorized():
                    self._handle_stats()
            elif self.path == "/distil/metrics":
                # Behind the SAME admin gate as /distil/stats: the series are
                # labelled by tenant, so an open /metrics would publish the
                # tenant list to anyone who can reach the port.
                if self._admin_authorized():
                    self._handle_metrics()
            elif self.path == "/distil/dashboard":
                if self._admin_authorized():
                    self._handle_dashboard()
            else:
                auth_result = self._check_inbound_auth()
                if auth_result is not None and auth_result[0] == "":
                    return  # 401/429 already sent
                self._passthrough(strip_header=auth_result[1] if auth_result else None)

        def _admin_authorized(self) -> bool:
            """Gate the management endpoints. Open on loopback with no token
            configured (local single-operator use); everything else needs the
            bearer token — replies with the error itself when unauthorized."""
            if admin_token:
                supplied = self.headers.get("Authorization", "")
                if hmac.compare_digest(supplied, f"Bearer {admin_token}"):
                    return True
                self._reject(401, "invalid or missing admin token")
                return False
            if loopback:
                return True
            self._reject(
                403,
                "management endpoints are disabled on non-loopback binds "
                "unless --admin-token is set",
            )
            return False

        def do_POST(self) -> None:  # noqa: N802
            _warn_if_version_skew(_version_state)
            auth_result = self._check_inbound_auth()
            if auth_result is not None and auth_result[0] == "":
                return  # 401/429 already sent
            tenant_override = auth_result[0] if auth_result else None
            strip_hdr = auth_result[1] if auth_result else None
            # Strip query string for path matching
            path = self.path.split("?", 1)[0]
            if path in _COMPRESSIBLE_PATHS or is_gemini_path(path):
                self._handle_compressible(tenant_override=tenant_override, strip_header=strip_hdr)
            else:
                self._passthrough(strip_header=strip_hdr)

        def do_PUT(self) -> None:  # noqa: N802
            auth_result = self._check_inbound_auth()
            if auth_result is not None and auth_result[0] == "":
                return
            self._passthrough(strip_header=auth_result[1] if auth_result else None)

        def do_DELETE(self) -> None:  # noqa: N802
            auth_result = self._check_inbound_auth()
            if auth_result is not None and auth_result[0] == "":
                return
            self._passthrough(strip_header=auth_result[1] if auth_result else None)

        def do_PATCH(self) -> None:  # noqa: N802
            auth_result = self._check_inbound_auth()
            if auth_result is not None and auth_result[0] == "":
                return
            self._passthrough(strip_header=auth_result[1] if auth_result else None)

        def do_HEAD(self) -> None:  # noqa: N802
            auth_result = self._check_inbound_auth()
            if auth_result is not None and auth_result[0] == "":
                return
            self._passthrough(strip_header=auth_result[1] if auth_result else None)

        def do_OPTIONS(self) -> None:  # noqa: N802
            auth_result = self._check_inbound_auth()
            if auth_result is not None and auth_result[0] == "":
                return
            self._passthrough(strip_header=auth_result[1] if auth_result else None)

        # ----------------------------------------------------------------
        # distil management endpoints
        # ----------------------------------------------------------------

        def _handle_stats(self) -> None:
            snap = state.snapshot()
            body = json.dumps(snap, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_metrics(self) -> None:
            """Prometheus scrape target. Never raises: a failed scrape must not
            take down the proxy path that shares this server. A scraper that
            gets a 500 retries in 15s; an exception escaping here would drop the
            connection mid-response and leave a half-written body on the wire."""
            from . import metrics as _metrics

            try:
                from . import __version__ as _v
            except Exception:  # noqa: BLE001
                _v = ""
            try:
                body = _metrics.render(state.snapshot(), version=_v).encode()
            except Exception:  # noqa: BLE001 — observability must never break the proxy
                self._reject(500, "metrics render failed")
                return
            self.send_response(200)
            self.send_header("Content-Type", _metrics.CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_dashboard(self) -> None:
            snap = state.snapshot()
            body = _dashboard_html(snap).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ----------------------------------------------------------------
        # Compression path
        # ----------------------------------------------------------------

        def _handle_compressible(
            self,
            *,
            tenant_override: str | None = None,
            strip_header: str | None = None,
        ) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                self._reject(413, "request body too large or malformed Content-Length")
                return
            headers = self._client_headers(strip_header)
            # Use key-derived tenant when auth is active; fall back to the
            # credential-hash path for the no-auth (single-user localhost) case.
            tenant = tenant_override or tenant_of(
                self.headers, trust_tenant_header=trust_tenant_header
            )

            # RPM without key auth: when keys are off, the auth path never ran, so
            # enforce --tenant-rpm here against the credential-derived tenant —
            # otherwise the startup banner advertises a limit nothing enforces.
            # (With key auth on, _check_inbound_auth already charged this request.)
            if tenant_override is None and default_rpm and not _rl.check_rpm(tenant, default_rpm):
                state.record_quota_rejection(tenant)
                body_err = json.dumps({"error": "rate limit exceeded"}).encode()
                self._relay(
                    429, {"Content-Type": "application/json"}, body_err, {"Retry-After": "60"}
                )
                return

            try:
                body: dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON — forward as-is
                status, rhdrs, rbody = self._post_upstream(self.path, raw, headers)
                self._relay(status, rhdrs, rbody)
                return

            # Echo the tenant label back only when it's an operator-trusted
            # explicit label — an anon-<hash> is a stable credential-derived
            # correlator that shouldn't ride response headers.
            extras: dict[str, str] = {}
            # Tenant accounting is booked only after a confirmed 2xx (P0-1).
            _pending_tenant_record: tuple[str, int, int] | None = None
            if not tenant.startswith("anon-"):
                extras["x-distil-tenant"] = tenant

            if "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                try:
                    compressed, _store = compress_messages(original, verbatim=verbatim)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug("compress_messages failed; forwarding uncompressed", exc_info=True)
                    compressed = original

                baseline_tokens = _count_tokens(original)
                compressed_tokens = _count_tokens(compressed)
                tokens_saved = max(0, baseline_tokens - compressed_tokens)

                # Daily token quota: checked against the baseline (pre-compression)
                # token count so the quota reflects actual input volume. The per-key
                # override (resolved in _check_inbound_auth, the only place holding
                # the KeyRecord) wins over the gateway default.
                _key_daily = getattr(self, "_key_daily_tokens", None)
                daily_limit = _key_daily if _key_daily is not None else default_daily_tokens
                if daily_limit and not _rl.check_daily_tokens(tenant, daily_limit, baseline_tokens):
                    state.record_quota_rejection(tenant)
                    body_err = json.dumps({"error": "daily token quota exceeded"}).encode()
                    self._relay(
                        429,
                        {"Content-Type": "application/json"},
                        body_err,
                        {"Retry-After": "60"},
                    )
                    return

                _pending_tenant_record = (tenant, baseline_tokens, compressed_tokens)

                body = {**body, "messages": compressed}
                extras["x-distil-tokens-saved"] = str(tokens_saved)

            elif "contents" in body and isinstance(body["contents"], list):
                # Gemini generateContent shape. Same per-key-over-default quota
                # precedence as the messages branch above.
                baseline_tokens = _gemini_count(body)
                _key_daily = getattr(self, "_key_daily_tokens", None)
                _daily_limit = _key_daily if _key_daily is not None else default_daily_tokens
                if _daily_limit and not _rl.check_daily_tokens(
                    tenant, _daily_limit, baseline_tokens
                ):
                    state.record_quota_rejection(tenant)
                    body_err = json.dumps({"error": "daily token quota exceeded"}).encode()
                    self._relay(
                        429,
                        {"Content-Type": "application/json"},
                        body_err,
                        {"Retry-After": "60"},
                    )
                    return
                try:
                    body, _store = compress_generate_request(body, verbatim=verbatim)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    log.debug("gemini compression failed; forwarding uncompressed", exc_info=True)
                compressed_tokens = _gemini_count(body)
                tokens_saved = max(0, baseline_tokens - compressed_tokens)
                _pending_tenant_record = (tenant, baseline_tokens, compressed_tokens)
                extras["x-distil-tokens-saved"] = str(tokens_saved)

            new_raw = json.dumps(body).encode()
            # Streamed requests relay incrementally — TTFT preserved per tenant.
            if bool(body.get("stream")) or ":streamGenerateContent" in self.path:
                from .streamrelay import stream_upstream

                status_s, _ = stream_upstream(
                    self,
                    _upstream + self.path,
                    new_raw,
                    headers,
                    timeout=_UPSTREAM_TIMEOUT,
                    hop_by_hop=_HOP_BY_HOP,
                    extras=extras,
                )
                # Book tenant accounting only after a fully-relayed 2xx (P0-1).
                if _pending_tenant_record is not None and 200 <= status_s < 300:
                    state.record(*_pending_tenant_record)
                return
            status, rhdrs, rbody = self._post_upstream(self.path, new_raw, headers)
            # Book tenant accounting only after a confirmed 2xx (P0-1): failed or
            # SDK-retried upstream calls must not inflate per-tenant savings.
            if _pending_tenant_record is not None and 200 <= status < 300:
                state.record(*_pending_tenant_record)
            self._relay(status, rhdrs, rbody, extras=extras)

        # ----------------------------------------------------------------
        # Transparent passthrough (unchanged body, any verb)
        # ----------------------------------------------------------------

        def _passthrough(self, *, strip_header: str | None = None) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                self._reject(413, "request body too large or malformed Content-Length")
                return
            headers = self._client_headers(strip_header)
            url = _upstream + self.path
            req = urllib.request.Request(
                url,
                data=raw or None,
                headers={**headers, **({"Content-Length": str(len(raw))} if raw else {})},
                method=self.command,
            )
            try:
                with _OPENER.open(req, timeout=_UPSTREAM_TIMEOUT) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    self._relay(resp.status, rhdrs, rbody)
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                self._relay(exc.code, rhdrs, rbody)
            except urllib.error.URLError as exc:
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
                ).encode()
                self._relay(502, {"Content-Type": "application/json"}, rbody)
            except TimeoutError:
                self._relay(
                    504, {"Content-Type": "application/json"}, b'{"error":"upstream timed out"}'
                )

        # ----------------------------------------------------------------
        # Shared helpers
        # ----------------------------------------------------------------

        def _read_body(self) -> bytes | None:
            length = parse_content_length(self.headers.get("Content-Length"))
            if length is None:
                return None
            return self.rfile.read(length) if length else b""

        def _reject(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self._relay(code, {"Content-Type": "application/json"}, body)

        def _client_headers(self, also_strip: str | None = None) -> dict[str, str]:
            """Client headers with hop-by-hop stripped.

            When *also_strip* is supplied (lowercase header name), that header
            is removed too — used to strip the distil gateway key before
            forwarding so the provider never sees it.
            """
            skip = _HOP_BY_HOP
            if also_strip:
                skip = skip | {also_strip.lower()}
            # Defense in depth: a distil gateway key must NEVER reach the provider,
            # whichever carrier it rode in on — a client sending BOTH an
            # `Authorization: Bearer dsk-…` AND an `x-distil-key` would otherwise
            # leak the second carrier upstream (only one name arrives via
            # *also_strip*). dsk- keys are only meaningful to this gateway, so
            # stripping unconditionally can't break provider auth.
            skip = skip | {"x-distil-key"}
            out = {k: v for k, v in self.headers.items() if k.lower() not in skip}
            for k in list(out):
                if k.lower() == "authorization" and out[k].startswith("Bearer dsk-"):
                    del out[k]
            return out

        def _relay(
            self,
            status: int,
            resp_headers: dict[str, str],
            resp_body: bytes,
            extras: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in _HOP_BY_HOP:
                    self.send_header(k, v)
            if extras:
                for k, v in extras.items():
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def _post_upstream(
            self,
            path: str,
            body: bytes,
            headers: dict[str, str],
        ) -> tuple[int, dict[str, str], bytes]:
            if safe_forward_path(path) is None:
                return (
                    400,
                    {"Content-Type": "application/json"},
                    b'{"error":"invalid request path"}',
                )
            url = _upstream + path
            req = urllib.request.Request(
                url,
                data=body,
                headers={**headers, "Content-Length": str(len(body))},
                method="POST",
            )
            try:
                with _OPENER.open(req, timeout=_UPSTREAM_TIMEOUT) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    return resp.status, rhdrs, rbody
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                return exc.code, rhdrs, rbody
            except urllib.error.URLError as exc:
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
                ).encode()
                return 502, {"Content-Type": "application/json"}, rbody
            except TimeoutError:
                return 504, {"Content-Type": "application/json"}, b'{"error":"upstream timed out"}'

    return _GatewayHandler


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------


def serve_gateway(
    host: str = "127.0.0.1",
    port: int = 8789,
    upstream: str = "https://api.anthropic.com",
    *,
    pricing_model: str = "claude-opus-4-8",
    lossless_only: bool = False,
    verbatim: bool = False,
    admin_token: str | None = None,
    trust_tenant_header: bool = False,
    require_keys: bool = False,
    tenant_rpm: int = 0,
    tenant_daily_tokens: int = 0,
) -> None:
    """Run a blocking ThreadingHTTPServer gateway.

    Parameters
    ----------
    host:           Interface to bind on.
    port:           Port to listen on.
    upstream:       Real LLM API base URL (no trailing slash).
    pricing_model:  Model key from ``distil.pricing.CATALOG`` for dollar accounting.
    lossless_only:  Policy mode (no tool injection); the reversible digest still runs.
    verbatim:       When *True*, skip the Tier-1 digest (Tier-0 only) — interactive-safe.
    admin_token:    Bearer token required for /distil/stats and /distil/dashboard.
                    Mandatory for those routes on non-loopback binds.
    trust_tenant_header:
                    Honor the client-supplied x-distil-tenant header (off by
                    default; tenant identity comes from the credential hash).
    require_keys:   Require a dsk- gateway key on every inbound request, even
                    if no keys have been issued yet.  Off by default; auth
                    activates automatically when keys exist.
    tenant_rpm:     Gateway-wide requests-per-minute cap per tenant (0 = unlimited).
    tenant_daily_tokens:
                    Gateway-wide per-tenant daily input-token cap (0 = unlimited).
    """
    price = pricing_get(pricing_model)
    state = GatewayState(price)
    state.load()  # restore per-tenant accounting from a previous run
    loopback = host in ("127.0.0.1", "::1", "localhost")
    key_store = GatewayKeyStore()
    handler = build_gateway_handler(
        upstream,
        state,
        price,
        lossless_only=lossless_only,
        verbatim=verbatim,
        admin_token=admin_token or os.environ.get("DISTIL_GATEWAY_TOKEN") or None,
        loopback=loopback,
        trust_tenant_header=trust_tenant_header,
        key_store=key_store,
        require_keys=require_keys,
        default_rpm=tenant_rpm,
        default_daily_tokens=tenant_daily_tokens,
    )
    server = QuietHTTPServer((host, port), handler)
    print(f"distil gateway listening on http://{host}:{port}")
    print(f"  dashboard: http://{host}:{port}/distil/dashboard")
    print(f"  metrics:   http://{host}:{port}/distil/metrics  (Prometheus)")
    auth_active = require_keys or key_store.has_active_keys()
    if auth_active:
        print("  key auth: enabled (dsk- bearer keys required)")
    if tenant_rpm:
        print(f"  rate limit: {tenant_rpm} req/min per tenant")
    if tenant_daily_tokens:
        print(f"  daily quota: {tenant_daily_tokens:,} tokens/day per tenant")
    if not loopback and not (admin_token or os.environ.get("DISTIL_GATEWAY_TOKEN")):
        print("  ! non-loopback bind without --admin-token: /distil/* routes are disabled")
    print(f"  → upstream: {upstream}")
    # Turn SIGTERM (systemd stop / kill) into KeyboardInterrupt so the finally
    # block persists tenant accounting instead of a kill zeroing it.
    _install_sigterm_flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.save()  # persist tenant accounting across restarts
        server.server_close()


if __name__ == "__main__":
    serve_gateway()

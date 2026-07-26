"""Federated, verifiable savings telemetry.

Each opt-in instance contributes a SIGNED, CONTENT-FREE savings aggregate with
its certification verdict attached.  A public leaderboard aggregates ONLY verified
submissions, so every number on the board is tamper-evident.

Signing uses HMAC-SHA256 (symmetric: both sides share a per-instance key).  The
natural upgrade — if you want a leaderboard anyone can verify without sharing keys —
is ed25519: the instance keeps the signing key, publishes the verify key, and the
aggregator uses the public key only.  That upgrade is a drop-in swap at the ``sign``
/ ``verify`` boundary; nothing else in this module changes.
"""

from __future__ import annotations

import hashlib
import hmac
import html as _html
import json
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SavingsAggregate:
    """Numbers-only snapshot of what one instance saved.  No prompt/response content."""

    instance_id: str
    tokens_saved: int
    dollars_saved: float
    runs: int
    certified: bool
    ts: float


# ---------------------------------------------------------------------------
# Canonical representation (deterministic, key-order-stable)
# ---------------------------------------------------------------------------


def _canonical(agg: SavingsAggregate) -> str:
    """Return a deterministic JSON string of the aggregate fields (sorted keys).

    The signature is computed over this string, so the representation must be
    stable across Python versions and dict orderings.
    """
    d = asdict(agg)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Sign / verify
# ---------------------------------------------------------------------------


def sign(agg: SavingsAggregate, key: str) -> dict:
    """Return a dict of all aggregate fields plus an HMAC-SHA256 ``sig`` field.

    The HMAC is computed over ``_canonical(agg)`` encoded as UTF-8.
    """
    canonical = _canonical(agg)
    sig = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return {**asdict(agg), "sig": sig}


def verify(signed: dict, key: str) -> bool:
    """Return True iff the ``sig`` in *signed* matches a fresh HMAC over the fields.

    Uses ``hmac.compare_digest`` for constant-time comparison (tamper-evident).
    The ``sig`` field is stripped before recomputing the canonical form.
    """
    signed = dict(signed)  # shallow copy — don't mutate caller's dict
    expected_sig = signed.pop("sig", None)
    if expected_sig is None:
        return False
    # Reconstruct the aggregate from the remaining fields so we use the same
    # canonical path as sign().
    try:
        agg = SavingsAggregate(**signed)
    except TypeError:
        return False
    canonical = _canonical(agg)
    fresh_sig = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_sig, fresh_sig)


# ---------------------------------------------------------------------------
# Submission persistence
# ---------------------------------------------------------------------------


def submit(signed: dict, dir: str) -> str:
    """Append *signed* as one JSONL line to ``<dir>/submissions.jsonl``.

    Returns the absolute path of the file written.
    """
    path = Path(dir) / "submissions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signed) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


@dataclass
class Leaderboard:
    """Aggregated view over verified submissions only.

    ``verified``  — per-instance dicts sorted by ``tokens_saved`` descending.
    ``totals``    — dict with keys: tokens_saved, dollars_saved, runs, instances.
                    Only ``certified`` submissions count toward these totals.
    ``rejected``  — count of submissions that failed signature verification.
    """

    verified: list[dict]
    totals: dict
    rejected: int


def build_leaderboard(dir: str, keys: dict[str, str]) -> Leaderboard:
    """Read submissions, verify each with its per-instance key, aggregate.

    *keys* maps instance_id → shared HMAC key.  Submissions whose instance_id
    has no entry in *keys*, or whose signature does not verify, are dropped and
    counted in ``rejected``.

    For each instance the LATEST (highest ``ts``) verified submission wins.
    Only submissions with ``certified == True`` contribute to headline totals.
    """
    path = Path(dir) / "submissions.jsonl"
    rejected = 0
    # latest verified record per instance_id
    best: dict[str, dict] = {}

    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue

            iid = record.get("instance_id")
            key = keys.get(iid) if iid else None
            if key is None or not verify(record, key):
                rejected += 1
                continue

            prev = best.get(iid)
            if prev is None or record.get("ts", 0) > prev.get("ts", 0):
                best[iid] = record

    verified = sorted(best.values(), key=lambda r: r.get("tokens_saved", 0), reverse=True)

    # Totals: only certified submissions
    total_tokens = sum(r["tokens_saved"] for r in verified if r.get("certified"))
    total_dollars = sum(r["dollars_saved"] for r in verified if r.get("certified"))
    total_runs = sum(r["runs"] for r in verified if r.get("certified"))
    instances = len(verified)

    totals = {
        "tokens_saved": total_tokens,
        "dollars_saved": total_dollars,
        "runs": total_runs,
        "instances": instances,
    }
    return Leaderboard(verified=verified, totals=totals, rejected=rejected)


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------


def render_leaderboard_html(lb: Leaderboard) -> str:
    """Return a self-contained dark HTML page showing per-instance verified savings."""

    def _fmt_dollars(v: float) -> str:
        return f"${v:,.4f}"

    def _fmt_tokens(v: int) -> str:
        return f"{v:,}"

    rows_html = ""
    for i, r in enumerate(lb.verified, 1):
        certified_cell = (
            '<td class="cert yes">&#10004; certified</td>'
            if r.get("certified")
            else '<td class="cert no">not certified</td>'
        )
        rows_html += (
            f"<tr>"
            f'<td class="rank">{i}</td>'
            f'<td class="iid">{_html.escape(str(r.get("instance_id", "")))}</td>'
            f'<td class="num">{_fmt_tokens(r.get("tokens_saved", 0))}</td>'
            f'<td class="num">{_fmt_dollars(r.get("dollars_saved", 0.0))}</td>'
            f'<td class="num">{r.get("runs", 0)}</td>'
            f"{certified_cell}"
            f"</tr>\n"
        )

    t = lb.totals
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Verifiable savings</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#06070b;color:#f2f3f7;font-family:Inter,system-ui,sans-serif;padding:2rem}}
  h1{{font-size:1.75rem;font-weight:700;color:#8b7bff;margin-bottom:.25rem}}
  .sub{{color:#5ad1c9;font-size:.85rem;margin-bottom:2rem}}
  .totals{{display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:2rem}}
  .card{{background:#0d0f17;border:1px solid #1e2030;border-radius:.5rem;padding:1rem 1.5rem}}
  .card .label{{font-size:.75rem;color:#8b8ea8;text-transform:uppercase;letter-spacing:.06em}}
  .card .value{{font-size:1.4rem;font-weight:600;color:#8b7bff;margin-top:.15rem}}
  table{{width:100%;border-collapse:collapse;font-size:.9rem}}
  th{{text-align:left;padding:.5rem .75rem;border-bottom:2px solid #1e2030;color:#8b8ea8;
      font-size:.75rem;text-transform:uppercase;letter-spacing:.06em}}
  td{{padding:.55rem .75rem;border-bottom:1px solid #1e2030}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#0d0f17}}
  .rank{{color:#8b8ea8;width:2.5rem}}
  .iid{{font-family:monospace;color:#5ad1c9}}
  .num{{text-align:right;font-variant-numeric:tabular-nums}}
  .cert.yes{{color:#5ad1c9;font-weight:600}}
  .cert.no{{color:#8b8ea8}}
  .badge{{display:inline-block;background:#1a1c2e;border:1px solid #8b7bff;
          color:#8b7bff;font-size:.7rem;padding:.1rem .4rem;border-radius:.25rem;
          margin-left:.5rem;vertical-align:middle}}
  .rejected{{font-size:.8rem;color:#8b8ea8;margin-top:1rem}}
</style>
</head>
<body>
<h1>Verifiable savings <span class="badge">HMAC-SHA256</span></h1>
<p class="sub">Signed, content-free aggregates — every number is tamper-evident</p>
<div class="totals">
  <div class="card"><div class="label">Tokens saved</div>
    <div class="value">{_fmt_tokens(t.get("tokens_saved", 0))}</div></div>
  <div class="card"><div class="label">Dollars saved</div>
    <div class="value">{_fmt_dollars(t.get("dollars_saved", 0.0))}</div></div>
  <div class="card"><div class="label">Runs</div>
    <div class="value">{t.get("runs", 0):,}</div></div>
  <div class="card"><div class="label">Instances</div>
    <div class="value">{t.get("instances", 0):,}</div></div>
</div>
<table>
<thead><tr>
  <th>#</th><th>Instance</th><th style="text-align:right">Tokens saved</th>
  <th style="text-align:right">Dollars saved</th><th style="text-align:right">Runs</th>
  <th>Certified</th>
</tr></thead>
<tbody>
{rows_html}</tbody>
</table>
<p class="rejected">Rejected (tampered / unverified): {lb.rejected}</p>
</body>
</html>"""
    return html

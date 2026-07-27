#!/usr/bin/env python3
"""Roll census + passive snapshots into public aggregates and badge JSONs.

Reads (from the metrics branch checkout):
    data/census.jsonl    — opted-in pings (schema validated on ingest)
    data/adoption.jsonl  — nightly passive registry snapshots

Writes:
    data/aggregates.json — the whole adoption picture, one document:
                           installs (census + per-channel), actives 7/30d,
                           versions in the wild, community tokens/$ saved
    data/badges/*.json   — shields.io "endpoint" schema, so README badges and
                           the docs site update the night data changes

Rules: latest census per install_id wins (a machine is one instance, not one
per ping); rows failing validation or the hard ceilings are dropped, so a
hostile ping that somehow reached the file still can't skew the totals.

Usage: python3 scripts/census_rollup.py <metrics-dir>
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from census_validate import validate  # noqa: E402 — CI runs this file directly


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # one corrupt line never kills the rollup
    return rows


def _humanize(n: float) -> str:
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= cut:
            return f"{n / cut:.1f}{suffix}"
    return str(int(n))


def rollup(metrics_dir: Path, now: float | None = None) -> dict:
    now = now or time.time()
    census = [r for r in _read_jsonl(metrics_dir / "data" / "census.jsonl") if validate(r) is None]
    adoption = _read_jsonl(metrics_dir / "data" / "adoption.jsonl")

    # Per-install merge: scalar fields (tokens, ts, version…) take the NEWEST
    # ping, but schema-dependent dimensions (equivalence, modes, usage maps)
    # carry forward the latest ping that actually CARRIES them — so a mixed
    # fleet (a newer ping from an older client that omits a field) never blanks
    # a known-good value. Without this, a v1.23 (schema-3) ping arriving after a
    # v1.24 (schema-4) ping would erase that install's decision-equivalence.
    _CARRY = ("equivalence", "by_model", "agents", "surfaces", "shapes", "modes", "billing")

    def _nonempty(v: object) -> bool:
        if v is None:
            return False
        if isinstance(v, dict) and "shadowed" in v:  # equivalence: real iff shadowed>0
            return int(v.get("shadowed") or 0) > 0
        return bool(v)

    latest: dict[str, dict] = {}
    for row in census:
        iid = row["install_id"]
        cur = latest.get(iid)
        if cur is None or row["ts"] >= cur["ts"]:
            merged = dict(row)
            if cur is not None:  # carry forward any dim the newer row lacks
                for k in _CARRY:
                    if not _nonempty(merged.get(k)) and _nonempty(cur.get(k)):
                        merged[k] = cur[k]
            latest[iid] = merged
        else:  # an older row can still supply a dim the newer one is missing
            for k in _CARRY:
                if not _nonempty(cur.get(k)) and _nonempty(row.get(k)):
                    cur[k] = row[k]

    def active(days: int) -> int:
        return sum(1 for r in latest.values() if now - r["ts"] <= days * 86400)

    def contributing(days: int) -> int:
        """Installs that are active AND have actually saved something.

        Consenting is not contributing: a machine can turn the census on and
        never run anything, and two of those look identical in `active_30d`.
        The page says "N machines saving" — that claim needs this number, not
        the consent count, or one idle opt-in inflates the community story.
        """
        return sum(
            1
            for r in latest.values()
            if now - r["ts"] <= days * 86400 and int(r.get("tokens_saved") or 0) > 0
        )

    # MEASURED community savings rate: for each install with >=2 pings, the
    # token delta over the time delta between its two most recent censuses
    # (dropping resets where tokens went backwards), summed across installs.
    # This is measured from real pings, not estimated — it's what lets the
    # page tick a live projection between daily censuses honestly.
    per_install: dict[str, list[dict]] = {}
    for row in census:
        per_install.setdefault(row["install_id"], []).append(row)
    rate_per_sec = 0.0
    for rs in per_install.values():
        rs.sort(key=lambda r: r["ts"])
        if len(rs) >= 2:
            a, b = rs[-2], rs[-1]
            dt = b["ts"] - a["ts"]
            dtok = int(b["tokens_saved"]) - int(a["tokens_saved"])
            if dt > 0 and dtok >= 0:
                rate_per_sec += dtok / dt

    # Community-total token trajectory over time (real points, for a sparkline)
    # AND the headline community total (`tokens`, below) — both are a faithful
    # Σ latest-per-install tokens_saved, walked over time: each install contributes
    # its most-recent value seen so far. The client is monotonic AT SOURCE, so this
    # only moves backward when an install is genuinely re-baselined (local state
    # wiped) — which it SHOULD. The old code banked upward per-install deltas across
    # the whole stream; that pinned the counter to a stale per-install high water
    # that survived even after that install re-baselined lower, minting a community
    # total no install could reproduce (the 1.4B ghost). Σ-of-latest can't ghost:
    # it always equals what the live worker reports and what a reinstall recomputes.
    history: list[dict] = []
    seen: dict[str, int] = {}  # per-install latest tokens_saved seen so far in time
    for row in sorted(census, key=lambda r: r["ts"]):
        seen[row["install_id"]] = int(row["tokens_saved"])
        history.append({"ts": row["ts"], "tokens": sum(seen.values())})
    history = history[-60:]

    as_of_ts = max((int(r["ts"]) for r in latest.values()), default=int(now))
    total_runs = sum(int(r.get("runs", 0)) for r in latest.values())

    # Decision-equivalence (schema 4): the trust number — compression provably
    # didn't change the agent's next action. Aggregate as a shadowed-count
    # weighted mean across installs that reported a real (non-null) pct.
    eq_shadowed = 0
    eq_weighted = 0.0
    eq_weight = 0
    for r in latest.values():
        eq = r.get("equivalence") or {}
        sh = int(eq.get("shadowed") or 0)
        eq_shadowed += sh
        pct = eq.get("pct")
        if pct is not None and sh > 0:
            eq_weighted += float(pct) * sh
            eq_weight += sh
    equivalence = {
        "pct": round(eq_weighted / eq_weight, 2) if eq_weight else None,
        "shadowed": eq_shadowed,
    }

    versions = Counter(r["version"] for r in latest.values())
    tokens = sum(int(r["tokens_saved"]) for r in latest.values())  # Σ latest-per-install (faithful)
    # Dollars are bucketed by billing: metered = real savings; subscription =
    # notional API-rate value (shown and labeled, never mixed into real $).
    # Schema-1 rows carry no billing → conservatively counted as notional.
    dollars_real = round(
        sum(float(r["dollars_saved"]) for r in latest.values() if r.get("billing") == "metered"),
        2,
    )
    dollars_notional = round(
        sum(float(r["dollars_saved"]) for r in latest.values() if r.get("billing") != "metered"),
        2,
    )

    # Usage dimensions (schema-2+ rows; schema-1 rows simply don't contribute).
    by_model: Counter = Counter()
    billing: Counter = Counter()
    agents: Counter = Counter()
    surfaces: Counter = Counter()
    shapes: Counter = Counter()
    modes: Counter = Counter()
    for r in latest.values():
        for m, v in (r.get("by_model") or {}).items():
            by_model[m] += int(v)
        if r.get("billing"):
            billing[r["billing"]] += 1
        for a in r.get("agents") or []:
            agents[a] += 1
        # Schema-3 integration attribution: request counts per surface/shape.
        for k, v in (r.get("surfaces") or {}).items():
            surfaces[k] += int(v)
        for k, v in (r.get("shapes") or {}).items():
            shapes[k] += int(v)
        # Schema-4 session kinds: interactive / headless / sdk.
        for k, v in (r.get("modes") or {}).items():
            modes[k] += int(v)

    # The newest passive row that actually carries pypi data (a partially
    # degraded night must not blank the dashboard).
    last_passive = adoption[-1] if adoption else {}
    pypi: dict = {}
    for row in reversed(adoption):
        cand = row.get("pypi_downloads", {})
        if isinstance(cand, dict) and "month" in cand:
            pypi = cand
            break
    return {
        "generated_ts": int(now),
        "installs": {
            "census_total": len(latest),
            "active_7d": active(7),
            "active_30d": active(30),
            # Machines that actually saved something, not just consented. The
            # savings tiles must never count an idle opt-in as a contributor.
            "contributing_7d": contributing(7),
            "contributing_30d": contributing(30),
            "by_version": dict(versions.most_common()),
        },
        "savings": {
            "tokens": tokens,
            "dollars": dollars_real,
            "dollars_notional": dollars_notional,
            "instances": len(latest),
            "contributing": sum(1 for r in latest.values() if int(r.get("tokens_saved") or 0) > 0),
            # Live-projection inputs — all measured, none estimated:
            "as_of_ts": as_of_ts,  # the token total is exact as of this ts
            "rate_per_sec": round(rate_per_sec, 2),  # measured Δtokens/Δt
            "total_runs": total_runs,
            "avg_per_run": round(tokens / total_runs) if total_runs else 0,
            "history": history,  # community total tokens over time (sparkline)
        },
        "equivalence": equivalence,  # {pct, shadowed} — the trust number
        "usage": {
            "by_model": dict(by_model.most_common()),
            "billing": dict(billing.most_common()),
            "agents": dict(agents.most_common()),
            "surfaces": dict(surfaces.most_common()),
            "shapes": dict(shapes.most_common()),
            "modes": dict(modes.most_common()),
        },
        "channels": {
            "pypi_downloads_month": pypi.get("month"),
            "pypi_downloads_week": pypi.get("week"),
            "github_stars": last_passive.get("github", {}).get("stars"),
            "clones_uniques_14d": last_passive.get("clones", {}).get("uniques_14d"),
            "docker_pulls": last_passive.get("docker", {}).get("pulls"),
        },
        # Bot-filtered detail for the adoption page: downloads with no reported
        # OS are scanners/crawlers, so "real" = pip on an actual machine.
        "pypi_detail": {
            "real_os_30d": pypi.get("real_os_30d"),
            "real_os_7d": pypi.get("real_os_7d"),
            "bots_30d": pypi.get("bots_30d"),
            "by_system_30d": pypi.get("by_system_30d", {}),
            "by_python_30d": pypi.get("by_python_30d", {}),
        },
    }


def badges(agg: dict) -> dict[str, dict]:
    """shields.io endpoint-schema documents, one per badge."""

    def badge(label: str, message: str, color: str = "6e56cf") -> dict:
        # cacheSeconds: ask shields to re-poll every 5 min (its minimum) so the
        # badges track the metrics branch near-real-time.
        return {
            "schemaVersion": 1,
            "label": label,
            "message": message,
            "color": color,
            "cacheSeconds": 300,
        }

    pypi_month = agg["channels"]["pypi_downloads_month"]
    real30 = agg.get("pypi_detail", {}).get("real_os_30d")
    eq = agg.get("equivalence") or {}
    eq_pct = eq.get("pct")
    return {
        "savings-tokens": badge(
            "community tokens saved", _humanize(agg["savings"]["tokens"]) or "0", "5ad19a"
        ),
        "savings-dollars": badge(
            "community $ saved · metered", f"${_humanize(agg['savings']['dollars'])}"
        ),
        "active-installs": badge("active installs (30d)", str(agg["installs"]["active_30d"])),
        "equivalence": badge(
            "decision-equivalence",
            (
                f"{eq_pct:g}% · {_humanize(eq.get('shadowed') or 0)} shadowed"
                if eq_pct is not None
                else "n/a"
            ),
            "5ad19a",
        ),
        "downloads-month": badge(
            "pypi downloads/month", _humanize(pypi_month) if pypi_month else "n/a"
        ),
        "downloads-real": badge(
            "pypi installs/mo · bot-filtered", _humanize(real30) if real30 else "n/a", "5ad19a"
        ),
    }


def main() -> int:
    metrics_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    agg = rollup(metrics_dir)
    out = metrics_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregates.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    badge_dir = out / "badges"
    badge_dir.mkdir(exist_ok=True)
    for name, doc in badges(agg).items():
        (badge_dir / f"{name}.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(json.dumps(agg["savings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

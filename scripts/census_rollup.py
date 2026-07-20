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

    latest: dict[str, dict] = {}
    for row in census:
        cur = latest.get(row["install_id"])
        if cur is None or row["ts"] >= cur["ts"]:
            latest[row["install_id"]] = row

    def active(days: int) -> int:
        return sum(1 for r in latest.values() if now - r["ts"] <= days * 86400)

    versions = Counter(r["version"] for r in latest.values())
    tokens = sum(int(r["tokens_saved"]) for r in latest.values())
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

    # Usage dimensions (schema-2 rows; schema-1 rows simply don't contribute).
    by_model: Counter = Counter()
    billing: Counter = Counter()
    agents: Counter = Counter()
    for r in latest.values():
        for m, v in (r.get("by_model") or {}).items():
            by_model[m] += int(v)
        if r.get("billing"):
            billing[r["billing"]] += 1
        for a in r.get("agents") or []:
            agents[a] += 1

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
            "by_version": dict(versions.most_common()),
        },
        "savings": {
            "tokens": tokens,
            "dollars": dollars_real,
            "dollars_notional": dollars_notional,
            "instances": len(latest),
        },
        "usage": {
            "by_model": dict(by_model.most_common()),
            "billing": dict(billing.most_common()),
            "agents": dict(agents.most_common()),
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
    return {
        "savings-tokens": badge(
            "community tokens saved", _humanize(agg["savings"]["tokens"]) or "0"
        ),
        "savings-dollars": badge(
            "community $ saved · metered", f"${_humanize(agg['savings']['dollars'])}"
        ),
        "active-installs": badge("active installs (30d)", str(agg["installs"]["active_30d"])),
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

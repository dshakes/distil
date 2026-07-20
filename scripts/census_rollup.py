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
    dollars = round(sum(float(r["dollars_saved"]) for r in latest.values()), 2)

    last_passive = adoption[-1] if adoption else {}
    pypi = last_passive.get("pypi_downloads", {})
    return {
        "generated_ts": int(now),
        "installs": {
            "census_total": len(latest),
            "active_7d": active(7),
            "active_30d": active(30),
            "by_version": dict(versions.most_common()),
        },
        "savings": {"tokens": tokens, "dollars": dollars, "instances": len(latest)},
        "channels": {
            "pypi_downloads_month": pypi.get("month"),
            "pypi_downloads_week": pypi.get("week"),
            "github_stars": last_passive.get("github", {}).get("stars"),
            "clones_uniques_14d": last_passive.get("clones", {}).get("uniques_14d"),
            "docker_pulls": last_passive.get("docker", {}).get("pulls"),
        },
    }


def badges(agg: dict) -> dict[str, dict]:
    """shields.io endpoint-schema documents, one per badge."""

    def badge(label: str, message: str, color: str = "6e56cf") -> dict:
        return {"schemaVersion": 1, "label": label, "message": message, "color": color}

    pypi_month = agg["channels"]["pypi_downloads_month"]
    return {
        "savings-tokens": badge(
            "community tokens saved", _humanize(agg["savings"]["tokens"]) or "0"
        ),
        "savings-dollars": badge("community $ saved", f"${_humanize(agg['savings']['dollars'])}"),
        "active-installs": badge("active installs (30d)", str(agg["installs"]["active_30d"])),
        "downloads-month": badge(
            "pypi downloads/month", _humanize(pypi_month) if pypi_month else "n/a"
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

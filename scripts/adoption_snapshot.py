#!/usr/bin/env python3
"""Passive adoption snapshot — registry-side numbers, zero user telemetry.

Pulls the adoption signals that exist WITHOUT any code running on a user's
machine (see specs/adoption-telemetry.md, phase 0) and prints ONE JSON line:

    pypistats.org  — downloads last day/week/month
    GitHub API     — stars, forks
    GitHub traffic — clones + unique cloners, views (14-day rolling window,
                     which is exactly why this runs on a nightly cron: the
                     history evaporates unless somebody snapshots it)
    Docker Hub     — cumulative pulls (absent until an image is published)

Each source degrades independently: a failure records {"error": "..."} for
that source and never kills the row. Stdlib only; GITHUB_TOKEN (with repo
push access, e.g. the Actions token) enables the traffic endpoints.

Usage:  python scripts/adoption_snapshot.py >> adoption.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

REPO = os.environ.get("DISTIL_REPO", "dshakes/distil")
PYPI_PACKAGE = os.environ.get("DISTIL_PYPI", "distil-llm")
DOCKER_IMAGE = os.environ.get("DISTIL_DOCKER", "")  # e.g. "dshakes/distil"


def _get(url: str, headers: dict | None = None) -> dict:
    # Real UA (pypistats blocks the default urllib one) + 3 tries with backoff
    # (shared Actions runner IPs hit 429s).
    hdrs = {"User-Agent": "distil-adoption-snapshot (github.com/dshakes/distil)"}
    hdrs.update(headers or {})
    last: Exception = RuntimeError("unreachable")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 — retried, then surfaced per-source
            last = exc
            time.sleep(3 * (attempt + 1))
    raise last


def _source(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — per-source degradation is the contract
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def snapshot() -> dict:
    gh_headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        gh_headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    def pypi() -> dict:
        data = _get(f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/recent")["data"]
        return {"day": data["last_day"], "week": data["last_week"], "month": data["last_month"]}

    def github() -> dict:
        d = _get(f"https://api.github.com/repos/{REPO}", gh_headers)
        return {"stars": d["stargazers_count"], "forks": d["forks_count"]}

    def clones() -> dict:
        d = _get(f"https://api.github.com/repos/{REPO}/traffic/clones", gh_headers)
        return {"count_14d": d["count"], "uniques_14d": d["uniques"]}

    def views() -> dict:
        d = _get(f"https://api.github.com/repos/{REPO}/traffic/views", gh_headers)
        return {"count_14d": d["count"], "uniques_14d": d["uniques"]}

    def docker() -> dict:
        if not DOCKER_IMAGE:
            return {"skipped": "no image configured"}
        d = _get(f"https://hub.docker.com/v2/repositories/{DOCKER_IMAGE}/")
        return {"pulls": d["pull_count"], "stars": d["star_count"]}

    return {
        "ts": int(time.time()),
        "date": time.strftime("%Y-%m-%d"),
        "pypi_downloads": _source(pypi),
        "github": _source(github),
        "clones": _source(clones),
        "views": _source(views),
        "docker": _source(docker),
    }


if __name__ == "__main__":
    row = snapshot()
    json.dump(row, sys.stdout, sort_keys=True)
    print()
    # Non-zero only when EVERY source failed — one dead API must not fail the cron.
    sources = [v for k, v in row.items() if isinstance(v, dict)]
    if all("error" in s for s in sources):
        sys.exit(1)

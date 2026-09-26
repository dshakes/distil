#!/usr/bin/env python3
"""External-only adoption report — stars, forks, traffic, external issues/PRs,
PyPI downloads, releases. Zero census, zero maintainer activity, zero user data.

Deliberately excludes what ``docs/adoption.html`` already shows: census totals
(opt-in, one machine dominates it — see the community-total note in
specs/adoption-telemetry.md) and anything the maintainer (``dshakes``) or a
collaborator did. This answers a different question than that dashboard: did
anyone OUTSIDE the project show up this week, and does the registry agree.

Sources, each degrading independently (a dead API records an ``error`` on its
own section, never kills the report):

    gh api repos/{repo}                        stars, forks
    gh api repos/{repo}/releases                release count
    gh api repos/{repo}/stargazers (star+json)  star timestamps -> delta since
    gh api repos/{repo}/traffic/views|clones    14-day rolling views/clones
    gh api repos/{repo}/traffic/popular/referrers
    gh api repos/{repo}/issues?state=all        issues+PRs opened, filtered to
                                                 non-maintainer non-bot authors
    gh api repos/{repo}/collaborators           who counts as "maintainer"
    pypistats.org /system                       downloads by OS, without mirrors
                                                 (the endpoint excludes mirror
                                                 traffic by default)

Traffic and collaborators need `gh` authenticated with at least read access to
the repo (a fine-grained PAT with Administration: read for traffic, same as
adoption-stats.yml's optional TRAFFIC_TOKEN) — locally this is just `gh auth
login`. Everything else works with an unauthenticated `gh`.

Usage:
    python3 scripts/ops/adoption_report.py [--repo dshakes/distil]
        [--pypi-package distil-llm] [--since 2026-09-11] [--json out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adoption_snapshot import _get as _http_get  # noqa: E402 — reuse retry/backoff/UA

DEFAULT_REPO = "dshakes/distil"
DEFAULT_PYPI_PACKAGE = "distil-llm"

# Exceptions a single source is allowed to fail with; anything else is a bug and
# should surface, not get swallowed under a generic "source failed" message.
_SOURCE_ERRORS = (
    subprocess.CalledProcessError,
    subprocess.TimeoutExpired,
    OSError,
    json.JSONDecodeError,
    KeyError,
    ValueError,
)


def _fmt_err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def _gh_api(args: list[str]) -> str:
    result = subprocess.run(
        ["gh", "api", *args], capture_output=True, text=True, timeout=30, check=True
    )
    return result.stdout


def _gh_api_lines(args: list[str]) -> list[str]:
    """Lines from a paginated ``--jq`` call — raw jq output, one value per line."""
    return [line for line in _gh_api(args).splitlines() if line.strip()]


def _gh_api_json_lines(args: list[str]) -> list[dict]:
    return [json.loads(line) for line in _gh_api_lines(args)]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GithubCore:
    stars: int | None = None
    stars_delta_since: int | None = None
    forks: int | None = None
    releases_total: int | None = None
    releases_since: int | None = None
    error: str | None = None


@dataclasses.dataclass
class Traffic:
    views_count_14d: int | None = None
    views_uniques_14d: int | None = None
    clones_count_14d: int | None = None
    clones_uniques_14d: int | None = None
    top_referrers: list[dict] | None = None  # [{referrer, count, uniques}], top 5
    error: str | None = None


@dataclasses.dataclass
class ExternalActivity:
    issues_opened: int | None = None
    prs_opened: int | None = None
    contributors: list[str] | None = None  # unique non-maintainer, non-bot logins
    error: str | None = None


@dataclasses.dataclass
class PyPIDownloads:
    last_day: int | None = None
    last_week: int | None = None
    last_month: int | None = None
    by_os_30d: dict[str, int] | None = None
    human_proxy_30d: int | None = None  # Darwin + Windows
    ci_proxy_30d: int | None = None  # Linux
    error: str | None = None


@dataclasses.dataclass
class AdoptionReport:
    repo: str
    pypi_package: str
    since: str
    generated_at: str
    github: GithubCore
    traffic: Traffic
    external_activity: ExternalActivity
    pypi: PyPIDownloads

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Fetchers — one per section, each catching only the source-failure exceptions.
# ---------------------------------------------------------------------------


def fetch_github_core(repo: str, since: date) -> GithubCore:
    try:
        core = json.loads(_gh_api([f"repos/{repo}"]))
        tags = _gh_api_lines([f"repos/{repo}/releases", "--paginate", "--jq", ".[].tag_name"])
        published = _gh_api_json_lines(
            [
                f"repos/{repo}/releases",
                "--paginate",
                "--jq",
                ".[] | {published_at: .published_at}",
            ]
        )
        since_iso = since.isoformat()
        releases_since = sum(
            1 for r in published if r.get("published_at") and r["published_at"][:10] >= since_iso
        )
        star_lines = _gh_api_lines(
            [
                "-H",
                "Accept: application/vnd.github.star+json",
                f"repos/{repo}/stargazers",
                "--paginate",
                "--jq",
                ".[].starred_at",
            ]
        )
        stars_delta = sum(1 for ts in star_lines if ts[:10] >= since_iso)
        return GithubCore(
            stars=core["stargazers_count"],
            stars_delta_since=stars_delta,
            forks=core["forks_count"],
            releases_total=len(tags),
            releases_since=releases_since,
        )
    except _SOURCE_ERRORS as exc:
        return GithubCore(error=_fmt_err(exc))


def fetch_traffic(repo: str) -> Traffic:
    try:
        views = json.loads(_gh_api([f"repos/{repo}/traffic/views"]))
        clones = json.loads(_gh_api([f"repos/{repo}/traffic/clones"]))
        referrers = json.loads(_gh_api([f"repos/{repo}/traffic/popular/referrers"]))
        return Traffic(
            views_count_14d=views["count"],
            views_uniques_14d=views["uniques"],
            clones_count_14d=clones["count"],
            clones_uniques_14d=clones["uniques"],
            top_referrers=sorted(referrers, key=lambda r: -r["count"])[:5],
        )
    except _SOURCE_ERRORS as exc:
        return Traffic(error=_fmt_err(exc))


def fetch_external_activity(repo: str, since: date) -> ExternalActivity:
    try:
        maintainers = {
            login.lower()
            for login in _gh_api_lines([f"repos/{repo}/collaborators", "--jq", ".[].login"])
        }
        since_dt = datetime.combine(since, datetime.min.time(), tzinfo=timezone.utc)
        rows = _gh_api_json_lines(
            [
                f"repos/{repo}/issues?state=all&since={since_dt.isoformat().replace('+00:00', 'Z')}"
                "&per_page=100",
                "--paginate",
                "--jq",
                ".[] | {number, is_pr: (.pull_request != null), login: .user.login, "
                "type: .user.type, created_at}",
            ]
        )
        external = [
            r
            for r in rows
            if r["created_at"] >= since_dt.isoformat().replace("+00:00", "Z")
            and r["type"] != "Bot"
            and not r["login"].endswith("[bot]")
            and r["login"].lower() not in maintainers
        ]
        return ExternalActivity(
            issues_opened=sum(1 for r in external if not r["is_pr"]),
            prs_opened=sum(1 for r in external if r["is_pr"]),
            contributors=sorted({r["login"] for r in external}),
        )
    except _SOURCE_ERRORS as exc:
        return ExternalActivity(error=_fmt_err(exc))


def fetch_pypi_downloads(package: str) -> PyPIDownloads:
    try:
        recent = _http_get(f"https://pypistats.org/api/packages/{package}/recent")["data"]
        cut30 = (date.today() - timedelta(days=30)).isoformat()
        by_os: dict[str, int] = {}
        for row in _http_get(f"https://pypistats.org/api/packages/{package}/system")["data"]:
            if row["date"] >= cut30:
                # pypistats reports unclassified downloads (no OS in the UA — scanners,
                # scripted mirrors, curl) as the literal string "null", not JSON null.
                cat = "unknown_or_bot" if row["category"] in (None, "null") else row["category"]
                by_os[cat] = by_os.get(cat, 0) + row["downloads"]
        return PyPIDownloads(
            last_day=recent["last_day"],
            last_week=recent["last_week"],
            last_month=recent["last_month"],
            by_os_30d=by_os,
            human_proxy_30d=by_os.get("Darwin", 0) + by_os.get("Windows", 0),
            ci_proxy_30d=by_os.get("Linux", 0),
        )
    except _SOURCE_ERRORS as exc:
        return PyPIDownloads(error=_fmt_err(exc))


def build_report(
    repo: str = DEFAULT_REPO, pypi_package: str = DEFAULT_PYPI_PACKAGE, since: date | None = None
) -> AdoptionReport:
    since = since or (date.today() - timedelta(days=14))
    return AdoptionReport(
        repo=repo,
        pypi_package=pypi_package,
        since=since.isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        github=fetch_github_core(repo, since),
        traffic=fetch_traffic(repo),
        external_activity=fetch_external_activity(repo, since),
        pypi=fetch_pypi_downloads(pypi_package),
    )


# ---------------------------------------------------------------------------
# Rendering — facts only, no interpretation. The "what moved and why" and the
# top-3 actions are the judgment call the weekly routine's prompt makes, not
# this script (see ops/routines/weekly-adoption-report.md).
# ---------------------------------------------------------------------------


def _line(label: str, value: object) -> str:
    return f"- **{label}:** {value}"


def render_markdown(report: AdoptionReport) -> str:
    lines = [
        f"# Adoption report — {report.repo}",
        "",
        f"Since **{report.since}** · generated {report.generated_at}",
        "",
        "External signals only: no census totals, no maintainer activity.",
        "",
        "## GitHub",
    ]
    g = report.github
    if g.error:
        lines.append(f"- error: {g.error}")
    else:
        lines += [
            _line("Stars", f"{g.stars} ({g.stars_delta_since:+d} since {report.since})"),
            _line("Forks", g.forks),
            _line("Releases", f"{g.releases_total} total, {g.releases_since} since {report.since}"),
        ]

    lines += ["", "## Traffic (14-day rolling)"]
    t = report.traffic
    if t.error:
        lines.append(f"- error: {t.error}")
    else:
        lines += [
            _line("Views", f"{t.views_count_14d} ({t.views_uniques_14d} unique)"),
            _line("Clones", f"{t.clones_count_14d} ({t.clones_uniques_14d} unique)"),
        ]
        if t.top_referrers:
            lines.append("- Top referrers:")
            for r in t.top_referrers:
                lines.append(f"  - {r['referrer']}: {r['count']} ({r['uniques']} unique)")

    lines += ["", "## External issues/PRs"]
    e = report.external_activity
    if e.error:
        lines.append(f"- error: {e.error}")
    else:
        lines += [
            _line("Issues opened", e.issues_opened),
            _line("PRs opened", e.prs_opened),
            _line("Contributors", ", ".join(e.contributors) if e.contributors else "none"),
        ]

    lines += ["", "## PyPI downloads (without mirrors)"]
    p = report.pypi
    if p.error:
        lines.append(f"- error: {p.error}")
    else:
        lines += [
            _line("Last day / week / month", f"{p.last_day} / {p.last_week} / {p.last_month}"),
            _line("Human proxy 30d (Darwin+Windows)", p.human_proxy_30d),
            _line("CI proxy 30d (Linux)", p.ci_proxy_30d),
            _line("By OS 30d", p.by_os_30d),
        ]

    lines.append("")
    return "\n".join(lines)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {value!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--pypi-package", default=DEFAULT_PYPI_PACKAGE)
    parser.add_argument(
        "--since", type=_parse_date, default=None, help="YYYY-MM-DD, default 14 days ago"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write JSON here")
    args = parser.parse_args(argv)

    report = build_report(repo=args.repo, pypi_package=args.pypi_package, since=args.since)
    print(render_markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

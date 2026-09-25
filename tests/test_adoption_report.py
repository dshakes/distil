"""scripts/ops/adoption_report.py — no network: every gh/pypistats call is
monkeypatched to a recorded fixture, so the suite runs offline and fast."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "ops"))
import adoption_report as ar  # noqa: E402

REPO = "dshakes/distil"
SINCE = date(2026, 9, 11)


# ---------------------------------------------------------------------------
# gh api fixtures — recorded shapes, not live calls.
# ---------------------------------------------------------------------------


def _gh_fixture(args: list[str]) -> str:
    endpoint = args[0] if not args[0].startswith("-H") else args[2]
    if endpoint == f"repos/{REPO}":
        return json.dumps({"stargazers_count": 18, "forks_count": 3})
    if endpoint == f"repos/{REPO}/releases" and args[-1] == ".[].tag_name":
        return "v1.54.0\nv1.53.0rc1\nv1.52.0\n"
    if endpoint == f"repos/{REPO}/releases":
        return (
            json.dumps({"published_at": "2026-09-25T15:54:51Z"})
            + "\n"
            + json.dumps({"published_at": "2026-09-08T16:18:46Z"})
            + "\n"
        )
    if endpoint == f"repos/{REPO}/stargazers":
        return "2026-09-24T15:20:10Z\n2026-08-13T21:05:24Z\n"
    if endpoint == f"repos/{REPO}/traffic/views":
        return json.dumps({"count": 65, "uniques": 36})
    if endpoint == f"repos/{REPO}/traffic/clones":
        return json.dumps({"count": 1934, "uniques": 309})
    if endpoint == f"repos/{REPO}/traffic/popular/referrers":
        return json.dumps([{"referrer": "github.com", "count": 16, "uniques": 8}])
    if endpoint == f"repos/{REPO}/collaborators":
        return "dshakes\nbhavik2383\n"
    if endpoint.startswith(f"repos/{REPO}/issues"):
        rows = [
            {
                "number": 1,
                "is_pr": False,
                "login": "dshakes",
                "type": "User",
                "created_at": "2026-09-24T20:58:32Z",
            },
            {
                "number": 2,
                "is_pr": True,
                "login": "someuser",
                "type": "User",
                "created_at": "2026-09-15T10:00:00Z",
            },
            {
                "number": 3,
                "is_pr": False,
                "login": "dependabot[bot]",
                "type": "Bot",
                "created_at": "2026-09-16T10:00:00Z",
            },
            {
                "number": 4,
                "is_pr": False,
                "login": "olduser",
                "type": "User",
                "created_at": "2026-08-01T00:00:00Z",  # before `since` — excluded
            },
        ]
        return "\n".join(json.dumps(r) for r in rows) + "\n"
    raise AssertionError(f"unexpected gh api call: {args}")


def _http_fixture(url: str, headers: dict | None = None) -> dict:
    if url.endswith("/recent"):
        return {"data": {"last_day": 7, "last_week": 337, "last_month": 1699}}
    if url.endswith("/system"):
        return {
            "data": [
                {"category": "Darwin", "date": "2026-09-20", "downloads": 27},
                {"category": "Windows", "date": "2026-09-20", "downloads": 6},
                {"category": "Linux", "date": "2026-09-20", "downloads": 87},
                {"category": "null", "date": "2026-09-20", "downloads": 1579},
            ]
        }
    raise AssertionError(f"unexpected http call: {url}")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(ar, "_gh_api", lambda args: _gh_fixture(args))
    monkeypatch.setattr(ar, "_http_get", lambda url, headers=None: _http_fixture(url, headers))


# ---------------------------------------------------------------------------


def test_build_report_all_sections_populate_from_fixtures():
    report = ar.build_report(repo=REPO, since=SINCE)
    assert report.github.error is None
    assert report.github.stars == 18
    assert report.github.stars_delta_since == 1  # only the 09-24 star is >= since
    assert report.github.releases_total == 3
    assert report.github.releases_since == 1  # only the 09-25 publish is >= since

    assert report.traffic.error is None
    assert report.traffic.views_count_14d == 65
    assert report.traffic.top_referrers == [{"referrer": "github.com", "count": 16, "uniques": 8}]

    assert report.pypi.error is None
    assert report.pypi.human_proxy_30d == 33
    assert report.pypi.ci_proxy_30d == 87
    assert report.pypi.by_os_30d["unknown_or_bot"] == 1579
    assert "null" not in report.pypi.by_os_30d


def test_external_activity_excludes_maintainers_bots_and_pre_since():
    report = ar.build_report(repo=REPO, since=SINCE)
    e = report.external_activity
    assert e.error is None
    # dshakes = maintainer, dependabot[bot] = bot, olduser = before `since`.
    # Only someuser's PR survives the filter.
    assert e.issues_opened == 0
    assert e.prs_opened == 1
    assert e.contributors == ["someuser"]


def test_source_failure_is_isolated_and_reported_as_error(monkeypatch):
    def _boom(args):
        raise subprocess.CalledProcessError(1, ["gh", "api", *args], stderr="rate limited")

    monkeypatch.setattr(ar, "_gh_api", _boom)
    report = ar.build_report(repo=REPO, since=SINCE)

    assert report.github.error is not None
    assert report.traffic.error is not None
    assert report.external_activity.error is not None
    # pypi is untouched by the gh failure — sources are independent.
    assert report.pypi.error is None

    # And rendering must not blow up on a partially-failed report.
    md = ar.render_markdown(report)
    assert "error:" in md


def test_pypi_failure_is_isolated(monkeypatch):
    def _boom(url, headers=None):
        raise ValueError("bad json")

    monkeypatch.setattr(ar, "_http_get", _boom)
    report = ar.build_report(repo=REPO, since=SINCE)
    assert report.pypi.error is not None
    assert report.github.error is None


def test_render_markdown_is_content_free_and_has_all_sections():
    report = ar.build_report(repo=REPO, since=SINCE)
    md = ar.render_markdown(report)
    for heading in ("## GitHub", "## Traffic", "## External issues/PRs", "## PyPI downloads"):
        assert heading in md
    # No maintainer/bot names leak into the rendered contributor line.
    assert "dshakes" not in md.split("## External issues/PRs")[1].split("## PyPI")[0]


def test_to_dict_round_trips_through_json():
    report = ar.build_report(repo=REPO, since=SINCE)
    blob = json.dumps(report.to_dict())
    restored = json.loads(blob)
    assert restored["github"]["stars"] == 18
    assert restored["since"] == "2026-09-11"


def test_parse_date_rejects_garbage():
    with pytest.raises(SystemExit):
        ar.main(["--since", "not-a-date"])


def test_main_defaults_since_to_14_days_ago(capsys):
    rc = ar.main(["--repo", REPO])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"# Adoption report — {REPO}" in out

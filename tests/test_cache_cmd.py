"""`distil cache` — the CLI over the session request ledger.

The case worth guarding is the empty one. A cache report over zero rows can print a
perfectly plausible "0 drifts" and exit 0, which reads as a clean bill of health for a
session nobody measured. It has to fail instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _run(home, *args: str):
    """Inherit the real environment and override only what the test controls.

    An earlier version built a minimal env from scratch — `{"PATH": "/usr/bin:/bin",
    "HOME": ...}`. That is POSIX-shaped in two ways at once: Windows resolves the home
    directory from USERPROFILE rather than HOME, and that PATH names directories that do
    not exist there. The CLI could not start, so every assertion in this file compared
    its expected string against `RuntimeError: Could not determine home directory.`

    `DISTIL_HOME` is the only isolation these tests actually need — conftest already
    points it at a fresh directory, and the fixture writes the ledger under it.
    """
    env = {**os.environ, "DISTIL_HOME": str(home / ".distil")}
    return subprocess.run(
        [sys.executable, "-m", "distil.cli", "cache", *args],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / ".distil"))
    (tmp_path / ".distil" / "sessions").mkdir(parents=True)
    return tmp_path


def _write(home, sid: str, rows: list[dict]) -> None:
    p = home / ".distil" / "sessions" / f"{sid}.requests.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class TestCacheCommand:
    def test_no_data_fails_rather_than_reporting_clean(self, home) -> None:
        r = _run(home)
        assert r.returncode == 1, "an empty report exiting 0 reads as 'cache is fine'"
        assert "no proxied requests" in r.stderr

    def test_a_held_prefix_reports_no_drift(self, home) -> None:
        _write(
            home,
            "s1",
            [
                {"ts": 1, "prefix_hash": "aa", "usage_cache_read": 800},
                {"ts": 2, "prefix_hash": "aa", "usage_cache_read": 800},
            ],
        )
        r = _run(home)
        assert r.returncode == 0, r.stderr
        assert "none across 1 turns" in r.stdout

    def test_drift_is_named_with_its_cause(self, home) -> None:
        _write(
            home,
            "s1",
            [
                {"ts": 1, "prefix_hash": "aa", "usage_cache_create": 800},
                {"ts": 2, "prefix_hash": "bb", "usage_cache_create": 800},
            ],
        )
        out = _run(home).stdout
        assert "1 of 1 turns changed" in out
        assert "upstream" in out, "distil cannot rewrite a stable block — say whose fault it is"

    def test_rows_are_ordered_by_time_not_by_file(self, home) -> None:
        """Two sessions interleave in wall-clock time. Folding them file-by-file would
        compare the last request of one against the first of the other and invent a
        drift that never happened."""
        _write(home, "s1", [{"ts": 1, "prefix_hash": "aa"}, {"ts": 3, "prefix_hash": "aa"}])
        _write(home, "s2", [{"ts": 2, "prefix_hash": "aa"}, {"ts": 4, "prefix_hash": "aa"}])
        r = _run(home, "--json")
        assert json.loads(r.stdout)["drifts"] == 0

    def test_json_is_machine_readable(self, home) -> None:
        _write(home, "s1", [{"ts": 1, "prefix_hash": "aa", "usage_cache_read": 8}])
        data = json.loads(_run(home, "--json").stdout)
        assert data["requests"] == 1 and data["hit_ratio"] == 1.0

    def test_an_unknown_session_fails_loudly(self, home) -> None:
        r = _run(home, "--session", "nope")
        assert r.returncode == 1 and "no request ledger" in r.stderr


class TestInProcess:
    """The same paths driven directly.

    The subprocess tests above are the honest end-to-end check — they prove the
    parser, the exit code and the ledger path all line up as a user sees them. But a
    subprocess runs outside coverage, so on their own they leave `cmd_cache` reading
    as dead code and a regression in it would not register.
    """

    def _args(self, **kw):
        import argparse

        return argparse.Namespace(**{"session": None, "sessions": 5, "json": False, **kw})

    def test_reports_and_returns_zero(self, home, capsys) -> None:
        from distil.cli import cmd_cache

        _write(home, "s1", [{"ts": 1, "prefix_hash": "aa", "usage_cache_read": 900}])
        assert cmd_cache(self._args()) == 0
        assert "requests" in capsys.readouterr().out

    def test_json_path(self, home, capsys) -> None:
        from distil.cli import cmd_cache

        _write(home, "s1", [{"ts": 1, "prefix_hash": "aa", "usage_cache_create": 900}])
        assert cmd_cache(self._args(json=True)) == 0
        assert json.loads(capsys.readouterr().out)["create_tokens"] == 900

    def test_named_session_path(self, home, capsys) -> None:
        from distil.cli import cmd_cache

        _write(home, "s7", [{"ts": 1, "prefix_hash": "aa"}])
        assert cmd_cache(self._args(session="s7")) == 0

    def test_missing_session_and_empty_both_return_one(self, home) -> None:
        from distil.cli import cmd_cache

        assert cmd_cache(self._args(session="ghost")) == 1
        assert cmd_cache(self._args()) == 1

    def test_sessions_limit_is_honoured(self, home, capsys) -> None:
        """`--sessions 1` must fold in exactly one file, or the default silently
        widens the window a reader thinks they narrowed."""
        from distil.cli import cmd_cache

        for i, ts in enumerate((1, 2, 3), start=1):
            _write(home, f"s{i}", [{"ts": ts, "prefix_hash": "aa"}])
        assert cmd_cache(self._args(sessions=1, json=True)) == 0
        assert json.loads(capsys.readouterr().out)["requests"] == 1

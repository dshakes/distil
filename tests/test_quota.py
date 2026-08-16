"""Subscription quota telemetry: fail-open, never fabricate, never leak the token.

The whole point of this module is to make subscription savings *measurable*, so its
most dangerous failure is not an exception — it's returning a plausible number when
the real one is unknown. A fabricated zero would read as "saved everything" in a
before/after comparison. These tests pin the distinction between "no data" and "zero
usage" hard.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

from distil.quota import Snapshot, Window, delta, snapshot

# The real payload shape, verified live 2026-08-16: `utilization` is a PERCENTAGE
# FLOAT, not used/limit integers, and windows the account lacks are null.
LIVE_PAYLOAD = {
    "five_hour": {
        "utilization": 85.0,
        "resets_at": "2026-08-16T08:50:00.594137+00:00",
        "limit_dollars": None,
        "used_dollars": None,
        "remaining_dollars": None,
    },
    "seven_day": {"utilization": 41.0, "resets_at": "2026-08-16T07:59:59.185448+00:00"},
    "seven_day_opus": None,
    "seven_day_sonnet": None,
}


class _Resp:
    def __init__(self, payload, status=200):
        self._b = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _with_response(payload, status=200):
    return patch("urllib.request.urlopen", return_value=_Resp(payload, status))


class TestParsing:
    def test_parses_live_payload_shape(self):
        with patch("distil.quota._token", return_value="tok"), _with_response(LIVE_PAYLOAD):
            s = snapshot()
        assert s.available
        five = s.window("five_hour")
        assert five is not None
        assert five.utilization_pct == 85.0
        assert five.remaining_pct == 15.0

    def test_null_windows_are_dropped_not_zeroed(self):
        """A plan without seven_day_opus must report it absent, not at 0% used."""
        with patch("distil.quota._token", return_value="tok"), _with_response(LIVE_PAYLOAD):
            s = snapshot()
        assert s.window("seven_day_opus") is None
        assert {w.name for w in s.windows} == {"five_hour", "seven_day"}

    def test_dollar_fields_optional(self):
        with patch("distil.quota._token", return_value="tok"), _with_response(LIVE_PAYLOAD):
            s = snapshot()
        assert s.window("five_hour").used_dollars is None


class TestFailOpen:
    """Every failure resolves to unavailable — never to a number."""

    def test_no_token(self):
        with patch("distil.quota._token", return_value=None):
            s = snapshot()
        assert not s.available and "token" in s.reason

    def test_http_error(self):
        err = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        with (
            patch("distil.quota._token", return_value="tok"),
            patch("urllib.request.urlopen", side_effect=err),
        ):
            s = snapshot()
        assert not s.available and "403" in s.reason
        assert s.windows == ()

    def test_unreachable(self):
        with (
            patch("distil.quota._token", return_value="tok"),
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")),
        ):
            s = snapshot()
        assert not s.available and s.reason == "unreachable"

    def test_malformed_json(self):
        class Bad(_Resp):
            def read(self):
                return b"<html>not json"

        with (
            patch("distil.quota._token", return_value="tok"),
            patch("urllib.request.urlopen", return_value=Bad({})),
        ):
            s = snapshot()
        assert not s.available and s.reason == "malformed response"

    def test_shape_drift_is_detected(self):
        """If the undocumented endpoint changes shape, say so — don't invent data."""
        with (
            patch("distil.quota._token", return_value="tok"),
            _with_response({"five_hour": {"pct_used": 85}}),
        ):
            s = snapshot()
        assert not s.available and "shape" in s.reason

    def test_non_dict_payload(self):
        with patch("distil.quota._token", return_value="tok"), _with_response([1, 2, 3]):
            s = snapshot()
        assert not s.available


class TestDelta:
    def test_measures_consumption(self):
        a = Snapshot(windows=(Window("five_hour", 40.0),), available=True)
        b = Snapshot(windows=(Window("five_hour", 47.5),), available=True)
        assert delta(a, b) == 7.5

    def test_unavailable_poll_is_none_not_zero(self):
        """None means 'unknown'. Returning 0.0 would read as 'consumed nothing'."""
        ok = Snapshot(windows=(Window("five_hour", 10.0),), available=True)
        bad = Snapshot(reason="unreachable")
        assert delta(bad, ok) is None
        assert delta(ok, bad) is None

    def test_missing_window_is_none(self):
        a = Snapshot(windows=(Window("five_hour", 10.0),), available=True)
        b = Snapshot(windows=(Window("five_hour", 20.0),), available=True)
        assert delta(a, b, window="seven_day_opus") is None

    def test_window_reset_shows_negative(self):
        """A reset mid-measurement yields a negative delta; callers discard those
        runs rather than bank them as savings."""
        a = Snapshot(windows=(Window("five_hour", 90.0),), available=True)
        b = Snapshot(windows=(Window("five_hour", 2.0),), available=True)
        assert delta(a, b) < 0


class TestSecrets:
    def test_token_never_appears_in_output(self):
        secret = "oat-fake-token-for-tests"  # not a real credential shape
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with (
            patch("distil.quota._token", return_value=secret),
            patch("urllib.request.urlopen", side_effect=err),
        ):
            s = snapshot()
        blob = f"{s!r} {s.reason}"
        assert secret not in blob
        assert "401" in blob  # the status is fine to surface; the credential is not

    def test_utilization_zero_division_guard(self):
        assert Window("x", 0.0).remaining_pct == 100.0


class TestTokenResolution:
    """Token discovery order, without ever touching the real keychain or a real key."""

    def test_env_var_wins(self, monkeypatch):
        from distil.quota import _token

        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "from-env")
        assert _token() == "from-env"

    def test_credentials_file(self, monkeypatch, tmp_path):
        from distil.quota import _token

        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        (tmp_path / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "from-file"}})
        )
        assert _token() == "from-file"

    def test_malformed_credentials_file_falls_through(self, monkeypatch, tmp_path):
        """An unreadable credentials file must not raise — it means 'no token'."""
        from distil.quota import _token

        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        (tmp_path / ".credentials.json").write_text("{not json")
        with patch("subprocess.run", side_effect=OSError("no keychain")):
            assert _token() is None

    def test_missing_everything_is_none(self, monkeypatch, tmp_path):
        from distil.quota import _token

        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        with patch("subprocess.run", side_effect=OSError("no keychain")):
            assert _token() is None

    def test_keychain_path(self, monkeypatch, tmp_path):
        """macOS stores credentials in the keychain, not on disk — a file-only
        lookup reports 'not logged in' on a machine that is in fact logged in."""
        import subprocess as sp

        from distil.quota import _token

        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        done = sp.CompletedProcess(
            [], 0, stdout=json.dumps({"claudeAiOauth": {"accessToken": "from-keychain"}})
        )
        with (
            patch("os.uname", return_value=type("U", (), {"sysname": "Darwin"})()),
            patch("subprocess.run", return_value=done),
        ):
            assert _token() == "from-keychain"

    def test_explicit_token_skips_resolution(self):
        with _with_response(LIVE_PAYLOAD):
            s = snapshot(token="explicit")
        assert s.available

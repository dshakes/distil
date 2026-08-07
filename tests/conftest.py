"""Test env hygiene: no test may touch the developer's real ~/.distil.

Developers run this suite from terminals that are themselves under
`distil wrap` (dogfooding), so the inherited env carries a LIVE
DISTIL_SESSION and the default DISTIL_HOME. Without isolation, wrap/proxy
tests would write ledger rows and session markers into the real store —
monkeypatch mutates os.environ, so subprocess-spawning tests inherit the
sandbox too.
"""

from __future__ import annotations

import pytest

from distil import setup as _setup

#: The unpatched enumerator, captured before the autouse sandbox below replaces it.
#: Reach it through the ``real_claude_settings_files`` fixture to test enumeration
#: itself; everything else must see the sandbox.
_REAL_CLAUDE_SETTINGS_FILES = _setup.claude_settings_files


@pytest.fixture(autouse=True)
def _distil_home_sandbox(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path_factory.mktemp("distil-home")))
    monkeypatch.delenv("DISTIL_SESSION", raising=False)
    # `distil default --undo` now sweeps EVERY settings file Claude Code merges and
    # deletes any loopback ANTHROPIC_BASE_URL it finds. That reach is the whole point
    # of the fix — and it is precisely why no test may keep it: a suite run from this
    # repo would otherwise rewrite the developer's own ~/.claude settings, which is the
    # accident that bricked Claude Code once already. Point it at an empty sandbox.
    sandbox = tmp_path_factory.mktemp("claude-settings")
    monkeypatch.setattr(
        _setup, "claude_settings_files", lambda cwd=None: [sandbox / "settings.json"]
    )


@pytest.fixture
def real_claude_settings_files():
    """The genuine enumerator, for the tests that assert on precedence order."""
    return _REAL_CLAUDE_SETTINGS_FILES

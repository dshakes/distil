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


@pytest.fixture(autouse=True)
def _restore_sigpipe_disposition():
    """Keep SIGPIPE ignored for the pytest process, whatever a test did to it.

    `distil.cli.main()` sets SIGPIPE to SIG_DFL — correct for a CLI filter, so
    `distil stats | head` dies quietly instead of spewing BrokenPipeError. But
    the disposition is PROCESS-GLOBAL, and eight test modules call `main()`
    in-process. After the first one, any broken pipe anywhere in the suite kills
    pytest outright with exit 141 instead of raising BrokenPipeError.

    That is how it presented in CI: a green suite locally, and a run that simply
    stopped at ~41% on one Python version with exit 141 and no failing test to
    point at. Ordering decided whether it happened, so it looked like flake.

    Restoring after every test makes the suite immune regardless of execution
    order. Only the main thread may set a handler, and Windows has no SIGPIPE —
    both are skipped rather than raising.
    """
    import signal

    sigpipe = getattr(signal, "SIGPIPE", None)
    if sigpipe is None:  # Windows
        yield
        return
    try:
        previous = signal.getsignal(sigpipe)
    except (ValueError, OSError):  # pragma: no cover - not the main thread
        yield
        return
    try:
        yield
    finally:
        try:
            if signal.getsignal(sigpipe) != previous:
                signal.signal(sigpipe, previous)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            pass

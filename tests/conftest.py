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


#: Captured before the autouse sandbox replaces them; reach them via the
#: ``real_service_api`` fixture when the function under test IS one of these.
_REAL_SERVICE_API = {
    "service_spec": _setup.service_spec,
    "service_unload_cmd": _setup.service_unload_cmd,
    "service_reload": _setup.service_reload,
}


@pytest.fixture(autouse=True)
def _launchd_service_sandbox(monkeypatch, tmp_path_factory):
    """No test may stop, delete, or reload the developer's REAL proxy service.

    ``cmd_offboard`` and ``cmd_default --undo`` call ``service_spec(8788, ...)``,
    which returns the genuine ``~/Library/LaunchAgents/com.distil.proxy.plist``,
    then run ``launchctl unload`` on it and ``path.unlink()`` it. Any test that
    exercised those paths without remembering to patch ``service_spec`` therefore
    tore down the always-on proxy of whoever ran the suite — and because the
    machine's ``ANTHROPIC_BASE_URL`` still pointed at the now-dead port, every
    Claude Code session on that machine started failing with ConnectionRefused.

    That is not hypothetical. It happened repeatedly in one evening on a
    maintainer's machine, and cost hours of misdiagnosis: the job came back
    *unregistered* rather than crashed, so ``proxy.err`` was empty, there was no
    crash report, and ``KeepAlive`` could not help a job that was no longer
    loaded. Five tests were missing the patch; the next one added would have been
    a coin flip.

    Remembering to patch per-test is the wrong mechanism — it fails open, and it
    failed open five times. This closes it by construction: the plist path is
    redirected into a tmp dir and the destructive commands become inert, so a
    forgotten patch costs nothing.
    """
    sandbox = tmp_path_factory.mktemp("launch-agents")

    def _sandboxed_spec(port, mode):
        path, content, load = _REAL_SERVICE_API["service_spec"](port, mode)
        if path is None:
            return path, content, load
        # Same filename so content assertions and basename checks still hold.
        return sandbox / path.name, content, "true  # sandboxed: no real launchctl"

    monkeypatch.setattr(_setup, "service_spec", _sandboxed_spec)
    monkeypatch.setattr(_setup, "service_unload_cmd", lambda: "true  # sandboxed")
    monkeypatch.setattr(
        _setup, "service_reload", lambda port: (True, "sandboxed: no real launchctl")
    )
    # `distil wrap` refuses to start when a settings file pins a base URL it does
    # not own. On a maintainer's machine that pin is always-on distil, so ~13 wrap
    # tests failed for an environmental reason with nothing wrong in the code.
    #
    # Do NOT fix that by setting DISTIL_IGNORE_SETTINGS_PRECEDENCE — that switches
    # the feature OFF, so the tests covering it silently stop testing anything.
    # Isolate the INPUT instead: point the user-level candidates at an empty
    # sandbox so the guard runs for real against files no developer owns. Project
    # -scoped candidates are cwd-relative and already land under tmp_path.
    from distil import precedence as _precedence

    monkeypatch.setattr(
        _precedence, "_USER_SETTINGS", (str(sandbox / "settings.json"),), raising=False
    )


@pytest.fixture
def real_service_api():
    """The genuine service functions, for tests that assert on them directly.

    Requesting this fixture is an explicit statement that the test drives these
    through mocked subprocess calls and will not reach a real supervisor.
    """
    for name, fn in _REAL_SERVICE_API.items():
        setattr(_setup, name, fn)
    return _REAL_SERVICE_API


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

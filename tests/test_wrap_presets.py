"""Per-agent wrap preset tests.

Covers: preset selection per command name, --env-var override wins, unknown command
falls back to the ANTHROPIC_BASE_URL default, and the injected variable is visible
in the child process's environment.
"""

from __future__ import annotations

import argparse
import sys

import pytest


def _ns(**kw: object) -> argparse.Namespace:
    """Minimal namespace for cmd_wrap; kw overrides the defaults."""
    defaults: dict = dict(
        command=[],
        host="127.0.0.1",
        upstream=None,  # None = not given on CLI → preset or default applies
        env_var=None,  # None = not given on CLI → preset or default applies
        lossless_only=False,
        verbatim=False,
        expand=False,
        shape_output="off",
        no_record=True,
        pricing="claude-opus-4-8",
        session_delta=False,
        shadow=0.0,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _mock_wrap_run(monkeypatch):
    """Patch proxy.wrap_run to capture the kwargs it was called with."""
    captured: dict = {}

    def fake(cmd, *, env_var, upstream, **kw):
        captured["env_var"] = env_var
        captured["upstream"] = upstream
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr("distil.proxy.wrap_run", fake)
    return captured


# ---------------------------------------------------------------------------
# Preset selection
# ---------------------------------------------------------------------------


def test_preset_claude(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["claude", "-p", "hello"]))
    assert rc == 0
    assert captured["env_var"] == "ANTHROPIC_BASE_URL"
    assert captured["upstream"] == "https://api.anthropic.com"
    out = capsys.readouterr().out
    assert "Claude Code" in out and "ANTHROPIC_BASE_URL" in out


def test_preset_codex(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["codex"]))
    assert rc == 0
    assert captured["env_var"] == "OPENAI_BASE_URL"
    assert captured["upstream"] == "https://api.openai.com"
    out = capsys.readouterr().out
    assert "Codex CLI" in out and "OPENAI_BASE_URL" in out


def test_preset_gemini(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["gemini"]))
    assert rc == 0
    assert captured["env_var"] == "GOOGLE_GEMINI_BASE_URL"
    assert captured["upstream"] == "https://generativelanguage.googleapis.com"
    out = capsys.readouterr().out
    assert "Gemini CLI" in out and "GOOGLE_GEMINI_BASE_URL" in out


def test_preset_aider(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["aider", "--model", "gpt-4o"]))
    assert rc == 0
    assert captured["env_var"] == "OPENAI_BASE_URL"
    assert captured["upstream"] == "https://api.openai.com"
    out = capsys.readouterr().out
    assert "aider" in out and "OPENAI_BASE_URL" in out


# ---------------------------------------------------------------------------
# --env-var override always wins
# ---------------------------------------------------------------------------


def test_env_var_override_wins_over_preset(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["claude"], env_var="MY_CUSTOM_VAR"))
    assert rc == 0
    assert captured["env_var"] == "MY_CUSTOM_VAR"
    # No preset message when explicitly overriding
    out = capsys.readouterr().out
    assert "MY_CUSTOM_VAR" not in out or "preset" not in out  # preset line suppressed


def test_upstream_override_wins_over_preset(monkeypatch):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["codex"], upstream="https://my-proxy.example.com"))
    assert rc == 0
    assert captured["upstream"] == "https://my-proxy.example.com"
    # env_var preset still fires (only upstream was overridden)
    assert captured["env_var"] == "OPENAI_BASE_URL"


# ---------------------------------------------------------------------------
# Unknown command falls back to existing default
# ---------------------------------------------------------------------------


def test_unknown_command_fallback(monkeypatch):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["my-custom-agent", "--flag"]))
    assert rc == 0
    assert captured["env_var"] == "ANTHROPIC_BASE_URL"
    assert captured["upstream"] == "https://api.anthropic.com"


def test_unknown_command_explicit_env_var(monkeypatch):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["my-custom-agent"], env_var="OPENAI_BASE_URL"))
    assert rc == 0
    assert captured["env_var"] == "OPENAI_BASE_URL"


# ---------------------------------------------------------------------------
# Injected env var is visible to the child process (real subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="subprocess env injection test")
def test_preset_env_visible_to_child(tmp_path, monkeypatch):
    """The preset env var injected by wrap_run actually lands in the child's env.

    We drive this through a real proxy.wrap_run call (not via cmd_wrap) so it
    is an integration check of the injection path, not just the preset selector.
    OPENAI_BASE_URL is used here to confirm a non-default var works end to end."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from distil import proxy

    class Echo(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):  # noqa: ANN002
            pass

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    up_url = f"http://127.0.0.1:{server.server_address[1]}"

    child = (
        "import os, sys\n"
        "v = os.environ.get('OPENAI_BASE_URL', '')\n"
        "sys.exit(0 if v.startswith('http') else 1)\n"
    )
    try:
        code = proxy.wrap_run(
            [sys.executable, "-c", child],
            upstream=up_url,
            env_var="OPENAI_BASE_URL",
            record=False,
        )
        assert code == 0, "child did not see OPENAI_BASE_URL in its environment"
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Bypass tripwire (post-run "no requests flowed" warning)
# ---------------------------------------------------------------------------


def _run_tripwire(tmp_path, monkeypatch, capsys, marker: str | None) -> str:
    """Run cmd_wrap for a preset agent with the session traffic marker preset
    to `marker` (None = no marker file) and return captured stderr."""
    from distil.cli import cmd_wrap

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("DISTIL_SESSION", "s123-1")
    if marker is not None:
        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        (sessions / "s123-1").write_text(marker, encoding="utf-8")
    _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["claude", "-p", "hi"]))
    assert rc == 0
    return capsys.readouterr().err


def test_tripwire_warns_on_bypass(tmp_path, monkeypatch, capsys):
    """Marker still "0" after the session → the agent bypassed the proxy → warn."""
    err = _run_tripwire(tmp_path, monkeypatch, capsys, "0")
    assert "no requests flowed" in err


def test_tripwire_silent_when_traffic_flowed(tmp_path, monkeypatch, capsys):
    """Marker "1" (traffic reached the proxy) → no warning, even with an empty
    savings ledger — a short zero-savings session must not cry wolf."""
    err = _run_tripwire(tmp_path, monkeypatch, capsys, "1")
    assert "no requests flowed" not in err


def test_tripwire_silent_without_marker(tmp_path, monkeypatch, capsys):
    """No marker file (wrap_run never created one) → can't tell → stay silent."""
    err = _run_tripwire(tmp_path, monkeypatch, capsys, None)
    assert "no requests flowed" not in err

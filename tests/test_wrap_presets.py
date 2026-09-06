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

    def fake(cmd, *, env_var, upstream, extra_env=None, **kw):
        captured["env_var"] = env_var
        captured["upstream"] = upstream
        captured["cmd"] = cmd
        captured["extra_env"] = extra_env
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
    # LiteLLM's name, not the OpenAI SDK's — aider never reads OPENAI_BASE_URL
    # (aider.chat/docs/llms/openai-compat.html).
    assert captured["env_var"] == "OPENAI_API_BASE"
    assert captured["upstream"] == "https://api.openai.com"
    out = capsys.readouterr().out
    assert "aider" in out and "OPENAI_API_BASE" in out


def test_preset_grok(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["grok"]))
    assert rc == 0
    assert captured["env_var"] == "GROK_MODELS_BASE_URL"
    # The /v1 belongs to the base URL here — unlike the OpenAI SDK, which appends
    # it. Dropping it sends every request to a 404 that reads like a distil bug.
    assert captured["upstream"] == "https://api.x.ai/v1"
    out = capsys.readouterr().out
    assert "Grok CLI" in out and "GROK_MODELS_BASE_URL" in out


def test_preset_copilot(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["copilot"]))
    assert rc == 0
    assert captured["env_var"] == "COPILOT_PROVIDER_BASE_URL"
    assert captured["upstream"] == "https://api.anthropic.com"
    out = capsys.readouterr().out
    assert "GitHub Copilot CLI" in out and "COPILOT_PROVIDER_BASE_URL" in out


def test_preset_kimi(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["kimi"]))
    assert rc == 0
    assert captured["env_var"] == "KIMI_BASE_URL"
    assert captured["upstream"] == "https://api.moonshot.ai/v1"
    out = capsys.readouterr().out
    assert "Kimi CLI" in out and "KIMI_BASE_URL" in out


def test_preset_openhands(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["openhands", "--override-with-envs"]))
    assert rc == 0
    assert captured["env_var"] == "LLM_BASE_URL"
    out = capsys.readouterr().out
    assert "OpenHands" in out and "LLM_BASE_URL" in out


def test_openhands_without_its_flag_is_warned_at_wrap_time(monkeypatch, capsys):
    """The preset alone does not route openhands, and the failure is silent.

    Selecting LLM_BASE_URL is necessary but not sufficient: without
    --override-with-envs the agent reads ~/.openhands/settings.json and never looks
    at the environment. The wrap still runs — it is the user's command — but it has
    to say so, or the only symptom is savings that never arrive.
    """
    from distil.cli import cmd_wrap

    _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["openhands", "--headless"]))
    assert rc == 0
    assert "--override-with-envs" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --env-var override always wins
# ---------------------------------------------------------------------------


def test_preset_extra_env_reaches_wrap_run(monkeypatch):
    """cmd_wrap must thread a preset's extra_env dict through to wrap_run unchanged."""
    from distil.cli import cmd_wrap
    from distil.onboard import AGENT_PRESETS

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["goose"]))
    assert rc == 0
    assert captured["extra_env"] == AGENT_PRESETS["goose"][3]
    assert captured["extra_env"] == {"ANTHROPIC_HOST": "$BASE"}


def test_explicit_env_var_suppresses_preset_extra_env(monkeypatch):
    """Overriding --env-var opts out of the whole preset shape, extra_env included."""
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["goose"], env_var="MY_CUSTOM_VAR"))
    assert rc == 0
    assert captured["extra_env"] == {}


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


# ---------------------------------------------------------------------------
# extra_env resolution ($BASE mirror, $VARNAME passthrough, literal, masking)
# ---------------------------------------------------------------------------


def test_extra_env_resolution_shapes(tmp_path, monkeypatch):
    """$BASE mirrors the proxy URL, $VARNAME passes an existing var through
    (skipped when unset/empty), and anything else is set literally."""
    from distil import proxy

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("MY_PASSTHROUGH", "carried-value")
    monkeypatch.delenv("MY_EMPTY_PASSTHROUGH", raising=False)
    result_path = tmp_path / "result.json"

    child = (
        "import json, os\n"
        f"open({str(result_path)!r}, 'w').write(json.dumps({{\n"
        "    'primary': os.environ.get('MY_BASE_VAR', ''),\n"
        "    'mirror': os.environ.get('MIRROR_VAR', ''),\n"
        "    'literal': os.environ.get('LITERAL_VAR', ''),\n"
        "    'passthrough': os.environ.get('PASS_VAR', ''),\n"
        "    'empty_absent': 'EMPTY_VAR' not in os.environ,\n"
        "}))\n"
    )
    code = proxy.wrap_run(
        [sys.executable, "-c", child],
        upstream="https://example.invalid",
        env_var="MY_BASE_VAR",
        record=False,
        extra_env={
            "MIRROR_VAR": "$BASE",
            "LITERAL_VAR": "literal-value",
            "PASS_VAR": "$MY_PASSTHROUGH",
            "EMPTY_VAR": "$MY_EMPTY_PASSTHROUGH",
        },
    )
    assert code == 0
    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["mirror"] == result["primary"] != ""
    assert result["literal"] == "literal-value"
    assert result["passthrough"] == "carried-value"
    assert result["empty_absent"] is True, "an unset source var must not be invented"


def test_extra_env_never_clobbers_a_user_override(tmp_path, monkeypatch):
    """A value the user already exported outranks a preset's extra_env literal."""
    from distil import proxy

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setenv("LITERAL_VAR", "user-set-value")
    result_path = tmp_path / "result.json"

    child = (
        "import json, os\n"
        f"open({str(result_path)!r}, 'w').write(json.dumps("
        "{'literal': os.environ.get('LITERAL_VAR', '')}))\n"
    )
    code = proxy.wrap_run(
        [sys.executable, "-c", child],
        upstream="https://example.invalid",
        env_var="MY_BASE_VAR",
        record=False,
        extra_env={"LITERAL_VAR": "preset-value"},
    )
    assert code == 0
    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["literal"] == "user-set-value"


def test_extra_env_masks_key_named_values_on_print(tmp_path, monkeypatch, capsys):
    """A var whose name contains KEY must never appear in plain text on stdout."""
    from distil import proxy

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))

    proxy.wrap_run(
        [sys.executable, "-c", "pass"],
        upstream="https://example.invalid",
        env_var="MY_BASE_VAR",
        record=False,
        extra_env={"SOME_PROVIDER_API_KEY": "super-secret-value"},
    )
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "••••••" in out


def test_a_new_preset_actually_carries_a_request_to_the_provider(tmp_path, monkeypatch):
    """The routing proof: a child using the injected variable reaches the far side.

    Selector tests assert distil picked a variable name. That is not the claim
    `wrap` makes to a user — the claim is that their agent's traffic now goes
    through distil. This drives the whole chain for a newly-added preset: wrap
    injects GROK_MODELS_BASE_URL -> the child reads it -> the proxy accepts the
    request -> the upstream records the hit. If any link is wrong the upstream stays
    untouched and this fails, which is the difference between "we wrote a config"
    and "a request provably arrived".
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from distil import proxy
    from distil.onboard import AGENT_PRESETS

    # Read the variable from the preset table rather than restating it, so a wrong
    # preset fails HERE instead of only in the selector test. Hardcoding the name
    # would make this look like an end-to-end proof while testing a constant.
    grok_env = AGENT_PRESETS["grok"][0]
    seen: list[str] = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.path)
            body = json.dumps(
                {"id": "m", "content": [{"type": "text", "text": "ok"}], "model": "m"}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa: ANN002
            pass

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    up_url = f"http://127.0.0.1:{server.server_address[1]}"

    child = (
        "import json, os, urllib.request, sys\n"
        f"base = os.environ[{grok_env!r}].rstrip('/')\n"
        "body = json.dumps({'model':'m','max_tokens':4,"
        "'messages':[{'role':'user','content':'hi'}]}).encode()\n"
        "req = urllib.request.Request(base + '/v1/messages', data=body,\n"
        "    headers={'content-type':'application/json'})\n"
        "urllib.request.urlopen(req, timeout=10).read()\n"
        "sys.exit(0)\n"
    )
    try:
        code = proxy.wrap_run(
            [sys.executable, "-c", child],
            upstream=up_url,
            env_var=grok_env,
            record=False,
        )
    finally:
        server.shutdown()

    assert code == 0, f"the child could not reach the proxy through {grok_env}"
    assert seen, "the request never arrived upstream — the wrap routed nothing"

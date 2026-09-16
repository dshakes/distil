"""The target catalogue: one source of truth for what `distil wrap` reaches.

Covers the Mistral Vibe preset (the first whose variable takes a JSON document
rather than a URL, proven against a real child process), `distil wrap --list`
in both shapes, and the invariant that keeps this honest: every preset carries
the primary source its contract was read from, and the README/docs tables are
generated from the same data rather than restated by hand.

The failure this file exists to prevent is specific. A preset built on a
guessed variable makes `wrap` report success, start a proxy, and route zero
traffic — the user sees "distil is on" and a savings counter frozen at zero,
with nothing saying which of the two lied.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from distil import config_wrap, targets
from distil.onboard import AGENT_ENV_TEMPLATES, AGENT_META, AGENT_PRESETS

from tests.test_wrap_presets import _mock_wrap_run, _ns

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Every preset carries its source, and the catalogue joins cleanly.
# ---------------------------------------------------------------------------


def test_every_env_preset_has_doc_metadata():
    """A preset without a cited source is exactly how a guessed variable ships:
    nothing else in the suite would notice."""
    assert AGENT_META.keys() == AGENT_PRESETS.keys()


def test_every_cited_source_is_a_url_and_a_real_date():
    for target in targets.catalog():
        assert target.doc_url.startswith("https://"), target
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", target.verified), target
        assert target.label and target.shape and target.knob, target


def test_catalog_keys_are_unique_across_both_registries():
    keys = [t.key for t in targets.catalog()]
    assert len(keys) == len(set(keys)), f"duplicate target key: {keys}"


def test_an_unreachable_target_never_shadows_a_real_preset():
    """`code` and `cursor` must stay warnings; `vibe` and `cn` must stay
    presets. An alias colliding with a preset key would make `wrap` print "this
    routes nothing" over a wrap that routes fine."""
    wrappable = set(AGENT_PRESETS) | set(config_wrap.CONFIG_PRESETS)
    for target in targets.UNREACHABLE:
        assert target.key not in wrappable, target.key
        assert not (set(target.aliases) & wrappable), target.aliases


def test_env_templates_only_exist_for_presets_that_have_one():
    assert set(AGENT_ENV_TEMPLATES) <= set(AGENT_PRESETS)
    for cmd, template in AGENT_ENV_TEMPLATES.items():
        assert "$BASE" in template, f"{cmd}: a template without $BASE routes nothing"


# ---------------------------------------------------------------------------
# Mistral Vibe: VIBE_PROVIDERS takes a JSON provider array, not a URL.
# ---------------------------------------------------------------------------


def test_preset_vibe(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    captured = _mock_wrap_run(monkeypatch)
    rc = cmd_wrap(_ns(command=["vibe"]))
    assert rc == 0
    assert captured["env_var"] == "VIBE_PROVIDERS"
    assert captured["upstream"] == "https://api.mistral.ai"
    out = capsys.readouterr().out
    assert "Mistral Vibe" in out and "VIBE_PROVIDERS" in out


def test_vibe_passes_its_json_template_to_wrap_run(monkeypatch):
    """The value is a document, not a URL. Exporting the bare URL here would
    leave Vibe reading its own config and talking straight to Mistral, while
    `wrap` reported success."""
    from distil.cli import cmd_wrap

    captured: dict = {}

    def fake(cmd, *, env_var, upstream, env_value_template=None, **kw):
        captured["template"] = env_value_template
        return 0

    monkeypatch.setattr("distil.proxy.wrap_run", fake)
    assert cmd_wrap(_ns(command=["vibe"])) == 0
    rendered = json.loads(captured["template"].replace("$BASE", "http://127.0.0.1:1234"))
    assert rendered[0]["name"] == "mistral", "must merge onto the default provider by name"
    assert rendered[0]["api_base"] == "http://127.0.0.1:1234/v1"
    assert rendered[0]["api_key_env_var"] == "MISTRAL_API_KEY", "never inline a credential"


def test_explicit_env_var_drops_the_template(monkeypatch):
    """`--env-var` means the user picked a variable that takes a URL — rendering
    Vibe's JSON into it would be nonsense."""
    from distil.cli import cmd_wrap

    captured: dict = {}

    def fake(cmd, *, env_var, env_value_template=None, **kw):
        captured["env_var"] = env_var
        captured["template"] = env_value_template
        return 0

    monkeypatch.setattr("distil.proxy.wrap_run", fake)
    assert cmd_wrap(_ns(command=["vibe"], env_var="OPENAI_BASE_URL")) == 0
    assert captured["env_var"] == "OPENAI_BASE_URL"
    assert captured["template"] is None


def _run_child(tmp_path, monkeypatch, source: str, **kw) -> int:
    """Drive a real ``proxy.wrap_run`` against a throwaway upstream and return
    the exit code of a child that inspects its own environment. The injection
    path is the thing under test, so nothing about it is mocked."""
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
    try:
        return proxy.wrap_run(
            [sys.executable, "-c", source],
            upstream=f"http://127.0.0.1:{server.server_address[1]}",
            record=False,
            **kw,
        )
    finally:
        server.shutdown()


@pytest.mark.skipif(sys.platform == "win32", reason="subprocess env injection test")
def test_vibe_child_sees_valid_json_pointing_at_the_proxy(tmp_path, monkeypatch):
    """End to end through a real wrap_run and a real child process: the child's
    VIBE_PROVIDERS must parse as JSON whose api_base is this wrap's proxy — and
    nothing may be written to the Vibe config directory, because this preset's
    whole point is that it touches no file (so there is nothing to restore)."""
    home = tmp_path / "home"
    (home / ".vibe").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    code = _run_child(
        tmp_path,
        monkeypatch,
        "import json, os, sys\n"
        "p = json.loads(os.environ['VIBE_PROVIDERS'])\n"
        "sys.exit(0 if p[0]['api_base'].startswith('http://127.0.0.1') "
        "and p[0]['api_base'].endswith('/v1') else 1)\n",
        env_var="VIBE_PROVIDERS",
        env_value_template=AGENT_ENV_TEMPLATES["vibe"],
    )
    assert code == 0, "child did not see a usable VIBE_PROVIDERS document"
    assert list((home / ".vibe").iterdir()) == [], "the env preset must touch no file"


@pytest.mark.skipif(sys.platform == "win32", reason="subprocess env injection test")
def test_a_plain_preset_still_gets_the_bare_url(tmp_path, monkeypatch):
    """The template is opt-in per preset: everything else must keep exporting
    the URL itself."""
    code = _run_child(
        tmp_path,
        monkeypatch,
        "import os, sys\n"
        "sys.exit(0 if os.environ['ANTHROPIC_BASE_URL'].startswith('http://127.0.0.1') else 1)\n",
    )
    assert code == 0


@pytest.mark.skipif(sys.platform == "win32", reason="subprocess env injection test")
def test_extra_env_interpolates_an_embedded_base(tmp_path, monkeypatch):
    """`$BASE` inside a longer literal is substituted, not treated as a
    passthrough variable name — the same rule the primary value follows."""
    monkeypatch.delenv("SOME_JSON", raising=False)
    code = _run_child(
        tmp_path,
        monkeypatch,
        "import json, os, sys\n"
        "sys.exit(0 if json.loads(os.environ['SOME_JSON'])['url']"
        ".startswith('http://127.0.0.1') else 1)\n",
        extra_env={"SOME_JSON": '{"url": "$BASE/v1"}'},
    )
    assert code == 0


# ---------------------------------------------------------------------------
# `distil wrap --list`
# ---------------------------------------------------------------------------


def test_wrap_list_names_every_target(capsys):
    from distil.cli import cmd_wrap

    assert cmd_wrap(_ns(list=True, json=False)) == 0
    out = capsys.readouterr().out
    for target in targets.catalog():
        assert target.label in out, target.label
    assert "Warp" in out, "the ones wrap cannot reach are the point of the list"


def test_wrap_list_json_is_machine_readable(capsys):
    from distil.cli import cmd_wrap

    assert cmd_wrap(_ns(list=True, json=True)) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == len(targets.catalog())
    by_cmd = {r["command"]: r for r in rows}
    assert by_cmd["vibe"]["mechanism"] == "env" and by_cmd["vibe"]["wrappable"]
    assert by_cmd["cn"]["mechanism"] == "config"
    assert not by_cmd["warp"]["wrappable"]
    assert by_cmd["warp"]["doc_url"].startswith("https://docs.warp.dev/")


def test_wrap_list_runs_before_anything_else(monkeypatch, capsys):
    """--list must not start an update check, set DISTIL_SURFACE, or need a
    command — it is a question, not a session."""
    from distil.cli import cmd_wrap

    monkeypatch.delenv("DISTIL_SURFACE", raising=False)

    def boom() -> None:
        raise AssertionError("--list must not reach the wrap path")

    monkeypatch.setattr("distil.updatecheck.maybe_notify", boom)
    assert cmd_wrap(_ns(list=True, json=False)) == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------
# The warning for a target `wrap` cannot reach cites its source.
# ---------------------------------------------------------------------------


def test_warning_for_an_unreachable_target_cites_the_doc(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    _mock_wrap_run(monkeypatch)
    assert cmd_wrap(_ns(command=["warp"])) == 0
    err = capsys.readouterr().err
    assert "Warp" in err and "route NOTHING" in err
    assert "docs.warp.dev" in err, "a claim about someone else's tool must cite its source"


def test_warning_fires_on_an_alias_too(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    _mock_wrap_run(monkeypatch)
    assert cmd_wrap(_ns(command=["cursor-agent"])) == 0
    assert "Cursor CLI" in capsys.readouterr().err


def test_no_warning_for_a_real_preset(monkeypatch, capsys):
    from distil.cli import cmd_wrap

    _mock_wrap_run(monkeypatch)
    assert cmd_wrap(_ns(command=["vibe"])) == 0
    assert "route NOTHING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The docs are generated, so they cannot drift from the registries.
# ---------------------------------------------------------------------------


def test_docs_tables_match_the_code():
    """Fails when a preset is added (or a verification date refreshed) without
    re-running the generator. Fix: python3 scripts/build_agent_tables.py"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_agent_tables

    stale = [
        path.name
        for path, wanted in build_agent_tables.render(ROOT).items()
        if path.read_text(encoding="utf-8") != wanted
    ]
    assert not stale, f"stale: {stale} — run python3 scripts/build_agent_tables.py"


def test_generator_refuses_a_document_missing_its_markers():
    """A silent no-op here would mean a table that stopped being generated and
    started drifting again, with the --check still green."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_agent_tables

    with pytest.raises(SystemExit):
        build_agent_tables.replace_region("no markers here", "agent-presets-bullet", "x")

"""`distil default --always-on` keeps Claude Code's MCP tool search on — and undo only
takes back what distil added.

Claude Code switches tool search off behind any non-first-party ANTHROPIC_BASE_URL, and
the always-on pin is exactly that (ADR 0013). So the pin is wired with
``ENABLE_TOOL_SEARCH=true`` beside it — only where the key is absent — and every way
out (``--undo``, ``offboard``, the ``sh`` escape hatch) removes it only from files
distil recorded adding it to, and only while it still holds distil's value.

Every test runs against a tmp HOME and DISTIL_HOME; nothing here touches the real
~/.claude or ~/.distil.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

import distil.setup as setup_mod
from distil import cli, onboard
from distil.setup import (
    TOOL_SEARCH_VAR,
    escape_hatch_spec,
    settings_added_path,
    tool_search_added_to,
    unwire_tool_search,
    wire_tool_search,
)


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("DISTIL_HOME", str(h / ".distil"))
    return h


def _settings(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _env(p: Path) -> dict:
    return json.loads(p.read_text()).get("env", {})


class TestWire:
    def test_fresh_write_adds_the_key_and_records_it(self, home: Path) -> None:
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {"KEEP": "1"}, "model": "opus"}))
        assert wire_tool_search(sp)[0] == "ok"
        assert _env(sp) == {"KEEP": "1", TOOL_SEARCH_VAR: "true"}
        assert json.loads(sp.read_text())["model"] == "opus"
        assert tool_search_added_to() == [str(sp.absolute())]

    def test_missing_file_is_created(self, home: Path) -> None:
        sp = _settings(home)
        assert wire_tool_search(sp)[0] == "ok"
        assert _env(sp) == {TOOL_SEARCH_VAR: "true"}

    @pytest.mark.parametrize("value", ["true", "false", "auto:5"])
    def test_a_user_value_is_never_touched_or_recorded(self, home: Path, value: str) -> None:
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {TOOL_SEARCH_VAR: value}}))
        before = sp.read_text()
        assert wire_tool_search(sp)[0] == "user"
        assert sp.read_text() == before
        assert tool_search_added_to() == []
        # ...so no undo path may remove it either
        assert unwire_tool_search(sp)[0] == "absent"
        assert sp.read_text() == before

    def test_rerun_is_idempotent(self, home: Path) -> None:
        sp = _settings(home)
        wire_tool_search(sp)
        first = sp.read_text()
        assert wire_tool_search(sp)[0] == "exists"
        assert sp.read_text() == first
        assert tool_search_added_to() == [str(sp.absolute())]  # recorded once

    @pytest.mark.parametrize(
        "body", ["{not json", "[1, 2]", json.dumps({"env": "ENABLE_TOOL_SEARCH=1"})]
    )
    def test_malformed_settings_are_never_clobbered(self, home: Path, body: str) -> None:
        sp = _settings(home)
        sp.write_text(body)
        assert wire_tool_search(sp)[0] == "error"
        assert sp.read_text() == body
        assert not settings_added_path().exists()


class TestUnwire:
    def test_removes_only_the_key_distil_added(self, home: Path) -> None:
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {"KEEP": "1"}}))
        wire_tool_search(sp)
        assert unwire_tool_search(sp)[0] == "ok"
        assert _env(sp) == {"KEEP": "1"}
        assert tool_search_added_to() == []
        assert unwire_tool_search(sp)[0] == "absent"  # second undo: nothing to do

    def test_empty_env_block_is_dropped(self, home: Path) -> None:
        sp = _settings(home)
        wire_tool_search(sp)
        unwire_tool_search(sp)
        assert "env" not in json.loads(sp.read_text())

    def test_a_value_the_user_changed_since_is_kept_and_forgotten(self, home: Path) -> None:
        sp = _settings(home)
        wire_tool_search(sp)
        sp.write_text(json.dumps({"env": {TOOL_SEARCH_VAR: "false"}}))
        assert unwire_tool_search(sp)[0] == "user"
        assert _env(sp) == {TOOL_SEARCH_VAR: "false"}
        assert tool_search_added_to() == []  # theirs now; a later undo must not touch it

    def test_malformed_settings_at_undo_are_left_alone(self, home: Path) -> None:
        sp = _settings(home)
        wire_tool_search(sp)
        sp.write_text("{broken")
        assert unwire_tool_search(sp)[0] == "error"
        assert sp.read_text() == "{broken"
        assert tool_search_added_to() == [str(sp.absolute())]  # still ours to clean later


def _default_ns(**kw: object) -> argparse.Namespace:
    base: dict = dict(
        undo=False,
        always_on=True,
        rc=None,
        port=34121,
        agent="claude",
        mode="lossless-only",
        no_start=False,
        force=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _always_on_machine(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        onboard, "detect", lambda: onboard.Env(os_name="Darwin", agents=[("claude", "Claude Code")])
    )
    rc_file = home / ".zshrc"
    rc_file.write_text("")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", rc_file))
    monkeypatch.setattr(
        setup_mod, "service_spec", lambda *a, **k: (home / "s.plist", "content", "load-cmd")
    )
    monkeypatch.setattr(setup_mod, "socket_unit_spec", lambda *a, **k: (None, None))
    monkeypatch.setattr(setup_mod, "service_reload", lambda port: (True, "registered"))
    monkeypatch.setattr(setup_mod, "probe_routing", lambda *a, **k: (True, "routes"))
    monkeypatch.setattr(setup_mod, "service_unload_cmd", lambda: "")
    monkeypatch.setattr(setup_mod, "claude_settings_files", lambda cwd=None: [_settings(home)])
    return rc_file


class TestAlwaysOn:
    def test_wires_beside_the_pin_and_undo_takes_back_only_its_own(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _always_on_machine(home, monkeypatch)
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {"KEEP": "1"}}))
        assert cli.cmd_default(_default_ns()) == 0
        env = _env(sp)
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:34121"
        assert env[TOOL_SEARCH_VAR] == "true"
        assert cli.cmd_default(_default_ns()) == 0  # re-run: idempotent
        assert _env(sp) == env
        assert cli.cmd_default(_default_ns(undo=True)) == 0
        assert _env(sp) == {"KEEP": "1"}

    def test_a_user_false_survives_wiring_and_undo(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _always_on_machine(home, monkeypatch)
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {TOOL_SEARCH_VAR: "false"}}))
        assert cli.cmd_default(_default_ns()) == 0
        assert _env(sp)[TOOL_SEARCH_VAR] == "false"
        assert cli.cmd_default(_default_ns(undo=True)) == 0
        assert _env(sp) == {TOOL_SEARCH_VAR: "false"}

    def test_offboard_removes_only_the_distil_added_key(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _always_on_machine(home, monkeypatch)
        monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {"KEEP": "1"}}))
        cli.cmd_default(_default_ns())
        # the pin removed by hand first: the key must not be stranded
        env = _env(sp)
        del env["ANTHROPIC_BASE_URL"]
        sp.write_text(json.dumps({"env": env}))
        rc = cli.cmd_offboard(argparse.Namespace(purge=False, yes=True, no_interactive=False))
        assert rc == 0
        assert _env(sp) == {"KEEP": "1"}


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX sh escape hatch")
class TestEscapeHatch:
    def _run(self, home: Path) -> subprocess.CompletedProcess[str]:
        path, content = escape_hatch_spec(8788)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return subprocess.run(
            ["sh", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
        )

    def test_removes_the_distil_added_key_with_the_pin(self, home: Path) -> None:
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}))
        wire_tool_search(sp)
        r = self._run(home)
        assert r.returncode == 0, r.stderr
        assert "env" not in json.loads(sp.read_text())
        assert tool_search_added_to() == []

    def test_removes_the_distil_added_key_even_without_a_pin(self, home: Path) -> None:
        sp = _settings(home)
        sp.write_text(json.dumps({"env": {"KEEP": "1"}}))
        wire_tool_search(sp)
        r = self._run(home)
        assert r.returncode == 0, r.stderr
        assert _env(sp) == {"KEEP": "1"}
        assert "removed ENABLE_TOOL_SEARCH" in r.stdout

    @pytest.mark.parametrize("value", ["true", "false"])
    def test_never_touches_a_user_value(self, home: Path, value: str) -> None:
        sp = _settings(home)
        sp.write_text(
            json.dumps(
                {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788", TOOL_SEARCH_VAR: value}}
            )
        )
        r = self._run(home)
        assert r.returncode == 0, r.stderr
        assert _env(sp) == {TOOL_SEARCH_VAR: value}

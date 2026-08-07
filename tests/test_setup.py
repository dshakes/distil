"""`distil setup` — status-line wiring must be idempotent, conflict-safe, and
preserve other settings."""

from __future__ import annotations

import json

from distil.setup import (
    alias_body,
    detect_shell,
    env_body,
    remove_managed,
    service_spec,
    unwire_settings_env,
    wire_settings_env,
    wire_statusline,
    write_managed,
)


def test_wire_fresh(tmp_path) -> None:
    p = tmp_path / "settings.json"
    status, _ = wire_statusline(p)
    assert status == "ok"
    assert "distil" in json.loads(p.read_text())["statusLine"]["command"]


def test_wire_idempotent(tmp_path) -> None:
    p = tmp_path / "settings.json"
    wire_statusline(p)
    status, _ = wire_statusline(p)
    assert status == "exists"


def test_wire_conflict_needs_force_and_does_not_clobber(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"statusLine": {"type": "command", "command": "mine.sh"}}))
    status, _ = wire_statusline(p)
    assert status == "conflict"
    assert "mine.sh" in p.read_text()  # untouched
    status, _ = wire_statusline(p, force=True)
    assert status == "ok"
    assert (tmp_path / "settings.json.bak").exists()  # backed up
    assert "distil" in json.loads(p.read_text())["statusLine"]["command"]


def test_wire_preserves_other_settings(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "opus", "env": {"X": "1"}}))
    wire_statusline(p)
    data = json.loads(p.read_text())
    assert data["model"] == "opus" and data["env"]["X"] == "1"
    assert "distil" in data["statusLine"]["command"]


def test_wire_rejects_non_object(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("[1, 2, 3]")
    status, _ = wire_statusline(p)
    assert status == "error"


# ── settings.json env wiring — reaches IDE-launched Claude Code (VSCode, ────
# Cursor's Claude Code extension) that a shell rc export never touches ──────


def test_wire_settings_env_fresh(tmp_path) -> None:
    p = tmp_path / "settings.json"
    status, _ = wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "ok"
    assert json.loads(p.read_text())["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"


def test_wire_settings_env_idempotent(tmp_path) -> None:
    p = tmp_path / "settings.json"
    wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    status, _ = wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "exists"


def test_wire_settings_env_conflict_needs_force_and_does_not_clobber(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://mine.example"}}))
    status, _ = wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "conflict"
    assert "mine.example" in p.read_text()  # untouched
    status, _ = wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788", force=True)
    assert status == "ok"
    assert (tmp_path / "settings.json.bak").exists()
    assert json.loads(p.read_text())["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"


def test_wire_settings_env_preserves_other_settings_and_env_keys(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "opus", "env": {"OTHER_VAR": "1"}}))
    wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    data = json.loads(p.read_text())
    assert data["model"] == "opus"
    assert data["env"]["OTHER_VAR"] == "1"
    assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"


def test_wire_settings_env_rejects_non_object(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("[1, 2, 3]")
    status, _ = wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "error"


def test_wire_settings_env_bad_json_is_error(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    status, _ = wire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "error"


def test_unwire_settings_env_removes_only_distils(tmp_path) -> None:
    p = tmp_path / "settings.json"
    # foreign value is preserved
    p.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://mine.example"}}))
    status, _ = unwire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "foreign"
    assert "mine.example" in p.read_text()

    # distil's value is removed, other env keys + settings preserved, backup made
    p.write_text(
        json.dumps(
            {"model": "opus", "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788", "X": "1"}}
        )
    )
    status, _ = unwire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert status == "ok"
    data = json.loads(p.read_text())
    assert "ANTHROPIC_BASE_URL" not in data["env"]
    assert data["env"]["X"] == "1"
    assert data["model"] == "opus"
    assert (tmp_path / "settings.json.bak").exists()

    # idempotent: nothing left to remove
    assert unwire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")[0] == "absent"


def test_unwire_settings_env_drops_empty_env_block(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}))
    unwire_settings_env(p, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")
    assert "env" not in json.loads(p.read_text())


def test_unwire_settings_env_absent_and_bad(tmp_path) -> None:
    assert unwire_settings_env(tmp_path / "nope.json", "ANTHROPIC_BASE_URL", "x")[0] == "absent"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert unwire_settings_env(bad, "ANTHROPIC_BASE_URL", "x")[0] == "error"


# ── distil default: managed-block + shell detection (reliable across machines) ──


def test_managed_block_add_idempotent_update_remove(tmp_path) -> None:
    rc = tmp_path / ".zshrc"
    rc.write_text("# my rc\nexport FOO=1\n")

    st, _ = write_managed(rc, "alias claude='distil wrap -- claude'")
    assert st == "ok"
    assert "alias claude=" in rc.read_text()
    assert "export FOO=1" in rc.read_text()  # original preserved

    st, _ = write_managed(rc, "alias claude='distil wrap -- claude'")
    assert st == "exists"  # identical → no churn

    st, _ = write_managed(rc, "alias claude='distil wrap --expand -- claude'")
    assert st == "updated"  # body changed → single block replaced, not duplicated
    assert rc.read_text().count("alias claude=") == 1
    assert (tmp_path / ".zshrc.bak").exists()  # backed up before change

    st, _ = remove_managed(rc)
    assert st == "ok"
    assert "distil" not in rc.read_text()
    assert "export FOO=1" in rc.read_text()  # untouched


def test_remove_managed_absent_is_safe(tmp_path) -> None:
    rc = tmp_path / ".bashrc"
    assert remove_managed(rc)[0] == "absent"  # no file
    rc.write_text("plain\n")
    assert remove_managed(rc)[0] == "absent"  # file but no block
    assert rc.read_text() == "plain\n"


def test_detect_shell_explicit_shell_wins_over_existing_rc(tmp_path, monkeypatch) -> None:
    # Both a .zshrc and a .bashrc exist; $SHELL must decide, not file-existence.
    home = tmp_path
    (home / ".zshrc").write_text("")
    (home / ".bashrc").write_text("")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("distil.setup._is_windows", lambda: False)
    monkeypatch.setenv("SHELL", "/bin/bash")
    name, rc = detect_shell()
    assert name == "bash" and rc.name == ".bashrc"
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    assert detect_shell()[0] == "fish"


def test_detect_shell_fallback_when_shell_unset(tmp_path, monkeypatch) -> None:
    home = tmp_path
    (home / ".zshrc").write_text("")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("distil.setup._is_windows", lambda: False)
    monkeypatch.setenv("SHELL", "")
    assert detect_shell() == ("zsh", home / ".zshrc")


def test_shell_specific_bodies() -> None:
    assert alias_body("claude", "lossless-only", shell="zsh") == (
        "alias claude='distil wrap --lossless-only -- claude'"
    )
    # PowerShell needs a function (forwards @args), not an alias.
    assert "function claude" in alias_body("claude", "expand", shell="powershell")
    # fish uses `set -gx`, posix `export`, powershell `$env:`.
    assert env_body(8788, shell="fish") == "set -gx ANTHROPIC_BASE_URL http://127.0.0.1:8788"
    assert env_body(8788, shell="zsh") == "export ANTHROPIC_BASE_URL=http://127.0.0.1:8788"
    assert env_body(8788, shell="powershell").startswith("$env:ANTHROPIC_BASE_URL")


def test_service_spec_per_platform(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    path, content, load = service_spec(8788, "lossless-only")
    assert path.name == "com.distil.proxy.plist"
    assert "--lossless-only" in content and "8788" in content
    assert "launchctl" in load

    monkeypatch.setattr("platform.system", lambda: "Linux")
    path, content, load = service_spec(9000, "expand")
    assert path.name == "distil-proxy.service"
    assert "ExecStart=" in content and "--expand" in content and "9000" in content
    assert "systemctl" in load

    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert service_spec(8788, "expand") == (None, None, None)  # unsupported → graceful


# ── distil default: cmd-level glue (install / undo / always-on), all sandboxed ──


def _default_args(tmp_path, **over):
    import argparse

    base = dict(
        rc=str(tmp_path / ".zshrc"),
        agent="claude",
        mode="lossless-only",
        port=8788,
        undo=False,
        always_on=False,
        no_start=True,
        force=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_default_installs_alias_and_undoes(tmp_path, capsys) -> None:
    from distil import cli

    rc = tmp_path / ".zshrc"
    rc.write_text("# keep me\n")
    from distil import setup

    assert cli.cmd_default(_default_args(tmp_path)) == 0
    # Expect the platform-appropriate managed block (alias on POSIX, function on Windows).
    assert setup.alias_body("claude", "lossless-only") in rc.read_text()
    assert "# keep me" in rc.read_text()

    assert cli.cmd_default(_default_args(tmp_path, undo=True)) == 0
    assert "distil" not in rc.read_text()
    assert "# keep me" in rc.read_text()


def _stub_preflight(monkeypatch, *, routed: bool = True) -> None:
    """`distil default --always-on` refuses to wire anything until the proxy proves it
    serves /v1/messages. No test starts a real one, so state the verdict explicitly —
    a test that silently skipped the preflight would be asserting the old, broken order."""
    from distil import setup

    monkeypatch.setattr(
        setup,
        "probe_routing",
        lambda host, port, **kw: (routed, f"stub: routed={routed} on {host}:{port}"),
    )


def test_cmd_default_always_on_writes_service_and_env(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli, setup

    svc = tmp_path / "svc.plist"
    monkeypatch.setattr(
        setup, "service_spec", lambda port, mode: (svc, f"PLIST {port} --{mode}", "true")
    )
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup, "default_settings_path", lambda: settings)
    monkeypatch.setattr(setup, "claude_settings_files", lambda cwd=None: [settings])
    _stub_preflight(monkeypatch)
    # Record shell-outs so we can assert install is silent but undo stops the service.
    import subprocess

    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    rc = tmp_path / ".zshrc"
    assert cli.cmd_default(_default_args(tmp_path, always_on=True, mode="lossless-only")) == 0
    assert svc.exists() and "8788" in svc.read_text()
    assert setup.env_body(8788) in rc.read_text()
    assert calls == []  # --no-start never shells out
    # settings.json is wired too — reaches IDE-launched Claude Code (VSCode,
    # Cursor's Claude Code extension) that never sources the rc file above.
    assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"

    # undo stops the running service (one shell-out) and cleans up rc + file + settings.json
    assert cli.cmd_default(_default_args(tmp_path, undo=True)) == 0
    if setup.service_unload_cmd() is not None:
        assert calls  # unload was invoked
    else:
        assert calls == []  # no service manager on this platform (e.g. Windows)
    assert not svc.exists()
    assert "ANTHROPIC_BASE_URL" not in rc.read_text()
    assert "ANTHROPIC_BASE_URL" not in settings.read_text()


def test_cmd_default_always_on_non_claude_agent_skips_settings_json(tmp_path, monkeypatch) -> None:
    """Only Claude Code reads ~/.claude/settings.json — wiring it for another
    wrapped agent (e.g. codex) would be a no-op at best, confusing at worst."""
    from distil import cli, setup

    svc = tmp_path / "svc.plist"
    monkeypatch.setattr(
        setup, "service_spec", lambda port, mode: (svc, f"PLIST {port} --{mode}", "true")
    )
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup, "default_settings_path", lambda: settings)
    _stub_preflight(monkeypatch)
    assert (
        cli.cmd_default(_default_args(tmp_path, always_on=True, agent="codex", no_start=True)) == 0
    )
    assert not settings.exists()


def test_cmd_default_always_on_settings_env_conflict_needs_force(
    tmp_path, monkeypatch, capsys
) -> None:
    from distil import cli, setup

    monkeypatch.setattr(
        setup, "service_spec", lambda port, mode: (tmp_path / "svc.plist", "PLIST", "true")
    )
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://mine.example"}}))
    monkeypatch.setattr(setup, "default_settings_path", lambda: settings)
    _stub_preflight(monkeypatch)
    assert cli.cmd_default(_default_args(tmp_path, always_on=True)) == 0
    assert "mine.example" in settings.read_text()  # left untouched without --force
    out = capsys.readouterr().out
    assert "--force" in out

    assert cli.cmd_default(_default_args(tmp_path, always_on=True, force=True)) == 0
    assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"


def test_cmd_default_always_on_unsupported_platform(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli, setup

    monkeypatch.setattr(setup, "service_spec", lambda port, mode: (None, None, None))
    rc = cli.cmd_default(_default_args(tmp_path, always_on=True))
    assert rc == 1  # graceful, non-zero
    assert "isn't supported" in capsys.readouterr().out


# ── distil offboard: unwire status line + remove footprint ──


def test_unwire_statusline_removes_only_distils(tmp_path) -> None:
    import json as _json

    from distil.setup import unwire_statusline, wire_statusline

    sp = tmp_path / "settings.json"
    # foreign status line is preserved
    sp.write_text(
        _json.dumps({"statusLine": {"type": "command", "command": "mine.sh"}, "model": "opus"})
    )
    assert unwire_statusline(sp)[0] == "foreign"
    assert "mine.sh" in sp.read_text()

    # distil status line is removed, other settings preserved, backup made
    wire_statusline(sp, force=True)
    assert "distil" in sp.read_text()
    st, _ = unwire_statusline(sp)
    assert st == "ok"
    data = _json.loads(sp.read_text())
    assert "statusLine" not in data and data["model"] == "opus"
    assert (tmp_path / "settings.json.bak").exists()

    # idempotent: nothing left to remove
    assert unwire_statusline(sp)[0] == "absent"


def test_unwire_statusline_absent_and_bad(tmp_path) -> None:
    from distil.setup import unwire_statusline

    assert unwire_statusline(tmp_path / "nope.json")[0] == "absent"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert unwire_statusline(bad)[0] == "error"


def test_service_unload_cmd_per_platform(monkeypatch) -> None:
    from distil.setup import service_unload_cmd

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert "launchctl unload" in service_unload_cmd()
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert "systemctl --user disable" in service_unload_cmd()
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert service_unload_cmd() is None


def _offboard_args(**over):
    import argparse

    base = dict(purge=False, yes=True, no_interactive=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_offboard_removes_footprint_keeps_data(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli, setup

    rc = tmp_path / ".zshrc"
    setup.write_managed(rc, setup.alias_body("claude", "lossless-only", shell="zsh"))
    settings = tmp_path / "settings.json"
    setup.wire_statusline(settings)
    data = tmp_path / ".distil"
    data.mkdir()
    (data / "savings.jsonl").write_text("{}\n")

    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", rc))
    monkeypatch.setattr(setup, "default_settings_path", lambda: settings)
    monkeypatch.setattr(setup, "service_spec", lambda p, m: (None, None, None))
    monkeypatch.setenv("DISTIL_HOME", str(data))

    assert cli.cmd_offboard(_offboard_args(purge=False)) == 0
    assert "distil" not in rc.read_text()  # alias gone
    assert "statusLine" not in settings.read_text()  # unwired
    assert data.exists()  # ledger kept without --purge
    out = capsys.readouterr().out
    assert "uninstall" in out  # tells you how to finish


def test_cmd_offboard_purge_deletes_data(tmp_path, monkeypatch) -> None:
    from distil import cli, setup

    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setattr(setup, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(setup, "service_spec", lambda p, m: (None, None, None))
    data = tmp_path / ".distil"
    data.mkdir()
    (data / "savings.jsonl").write_text("{}\n")
    monkeypatch.setenv("DISTIL_HOME", str(data))
    assert cli.cmd_offboard(_offboard_args(purge=True)) == 0
    assert not data.exists()  # --purge removed it


def test_cmd_offboard_non_interactive_is_safe(tmp_path, monkeypatch, capsys) -> None:
    # Not interactive and no --yes → must NOT delete anything.
    from distil import cli, setup

    rc = tmp_path / ".zshrc"
    setup.write_managed(rc, setup.alias_body("claude", "expand", shell="zsh"))
    monkeypatch.setattr(setup, "detect_shell", lambda: ("zsh", rc))
    monkeypatch.setattr(setup, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(setup, "service_spec", lambda p, m: (None, None, None))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.cmd_offboard(_offboard_args(yes=False, no_interactive=True)) == 0
    assert "distil" in rc.read_text()  # untouched — safe

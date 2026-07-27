"""`distil onboard` — detection + a guide tailored to the environment."""

from __future__ import annotations

from distil import onboard
from distil.onboard import Env


def test_detect_returns_env_shape() -> None:
    env = onboard.detect()
    assert env.os_name
    assert isinstance(env.managers, list)
    assert isinstance(env.agents, list)


def test_best_install_prefers_pipx_then_uv_then_fallback() -> None:
    assert onboard.best_install_command(["pipx", "uv"]) == "pipx install distil-llm"
    assert "uv tool install" in onboard.best_install_command(["uv"])
    assert "pipx install distil-llm" in onboard.best_install_command([])  # fallback bootstraps pipx


def test_guide_subscription_uses_lossless() -> None:
    env = Env(
        os_name="Darwin", agents=[("claude", "Claude Code")], subscription=True, has_anthropic=True
    )
    steps = onboard.next_steps(env)
    cmds = [cmd for _, cmd, _ in steps]
    notes = [note for _, _, note in steps]
    assert any("--lossless-only -- claude" in c for c in cmds)
    assert not any("--expand -- claude" in c for c in cmds)
    # Tradeoff must be explicit: subscription users save context+latency but NOT tokens/cost
    lossless_notes = [
        n
        for n in notes
        if "lossless" in n.lower() or "flat-rate" in n.lower() or "does not" in n.lower()
    ]
    assert lossless_notes, "subscription step must have a note explaining the lossless tradeoff"
    # the note must be honest on BOTH axes: lossless mode DOES cut tokens, but not a flat-rate bill
    joined = " ".join(lossless_notes).lower()
    assert "lossless" in joined and "token" in joined, "note must state lossless token savings"
    assert "bill" in joined or "cost" in joined or "rate" in joined, (
        "note must clarify the flat-rate bill is unchanged"
    )


def test_guide_metered_uses_expand_and_primary_agent() -> None:
    env = Env(os_name="Darwin", agents=[("codex", "Codex")], subscription=False, has_anthropic=True)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("--expand -- codex" in c for c in cmds)  # routes the detected primary agent


def test_guide_no_agent_prompts_install() -> None:
    env = Env(os_name="Windows", agents=[], subscription=False)
    titles = [t for t, _, _ in onboard.next_steps(env)]
    assert any("Install a coding agent" in t for t in titles)


def test_guide_offers_anthropic_extra_when_missing() -> None:
    env = Env(os_name="Darwin", agents=[("claude", "Claude Code")], has_anthropic=False)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("pipx inject distil-llm anthropic" in c for c in cmds)


def test_guide_always_includes_shadow_and_doctor() -> None:
    env = Env(os_name="Linux", agents=[("claude", "Claude Code")], has_anthropic=True)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("--shadow 0.1" in c for c in cmds)
    assert any(c.strip() == "distil doctor" for c in cmds)


def test_cmd_onboard_wires_statusline_and_prints_guide(tmp_path, monkeypatch, capsys):
    import argparse
    import json

    import distil.cli as cli
    import distil.setup as setup_mod

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: settings)
    rc = cli.cmd_onboard(
        argparse.Namespace(
            dry_run=False,
            force=False,
            no_color=True,
            json=False,
            offline=True,
            upgrade=False,
            yes=False,
            no_interactive=True,
        )
    )
    assert rc == 0
    assert "distil" in json.loads(settings.read_text())["statusLine"]["command"]
    out = capsys.readouterr().out
    assert "Next steps" in out and "distil doctor" in out


def test_cmd_onboard_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    import argparse

    import distil.cli as cli
    import distil.setup as setup_mod

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: settings)
    rc = cli.cmd_onboard(
        argparse.Namespace(
            dry_run=True,
            force=False,
            no_color=True,
            json=False,
            offline=True,
            upgrade=False,
            yes=False,
            no_interactive=True,
        )
    )
    assert rc == 0
    assert not settings.exists()  # dry-run wrote nothing
    assert "Next steps" in capsys.readouterr().out


def test_is_outdated_semantics() -> None:
    assert onboard.is_outdated("1.2.0", "1.3.0") is True
    assert onboard.is_outdated("1.3.0", "1.3.0") is False
    assert onboard.is_outdated("1.4.0.dev0", "1.3.0") is False  # dev build is ahead of release
    assert (
        onboard.is_outdated("1.3.0.dev0", "1.3.0") is True
    )  # pre-release of the release → upgrade
    assert onboard.is_outdated("1.3.0", None) is False  # offline / check failed
    # A pre-release compared to ITSELF is not outdated. Same base + "ours is a
    # pre-release" used to short-circuit to True, so anyone running an rc that
    # PyPI also reported as latest was nagged to upgrade to what they had.
    assert onboard.is_outdated("1.34.0rc1", "1.34.0rc1") is False
    assert onboard.is_outdated("1.34.0rc1", "1.34.0") is True  # …but the GA is newer
    assert onboard.is_outdated("1.34.0rc1", "1.33.1") is False  # …and rc leads last stable


def test_upgrade_command_per_method() -> None:
    assert onboard.upgrade_command("pipx") == "pipx upgrade distil-llm"
    assert "uv tool upgrade" in onboard.upgrade_command("uv")
    assert "pip install --upgrade" in onboard.upgrade_command("pip")


def test_report_is_agent_ready() -> None:
    env = Env(
        os_name="Darwin",
        agents=[("claude", "Claude Code")],
        installed_version="1.2.0",
        method="pipx",
        subscription=True,
    )
    r = onboard.report(env, "1.3.0")
    assert r["upgrade_available"] is True
    assert r["upgrade_command"] == "pipx upgrade distil-llm"
    assert r["primary_agent"] == "claude"
    assert r["billing"] == "subscription"
    assert r["next_steps"] and all(
        {"title", "command", "note"} <= s.keys() for s in r["next_steps"]
    )


def test_cmd_onboard_json_is_pure(tmp_path, monkeypatch, capsys) -> None:
    import argparse
    import json

    import distil.cli as cli
    import distil.onboard as ob
    import distil.setup as setup_mod

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: settings)
    monkeypatch.setattr(ob, "latest_pypi_version", lambda *a, **k: "9.9.9")  # no real network
    rc = cli.cmd_onboard(
        argparse.Namespace(
            json=True, offline=False, dry_run=False, force=False, upgrade=False, no_color=True
        )
    )
    assert rc == 0
    assert not settings.exists()  # --json takes no actions
    data = json.loads(capsys.readouterr().out)
    assert data["upgrade_available"] is True  # installed < 9.9.9


def test_cmd_onboard_yes_runs_first_step(tmp_path, monkeypatch, capsys) -> None:
    import argparse
    import subprocess

    import distil.cli as cli
    import distil.onboard as ob
    import distil.setup as setup_mod

    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "s.json")
    monkeypatch.setattr(
        ob,
        "detect",
        lambda: ob.Env(
            os_name="Darwin",
            agents=[("claude", "Claude Code")],
            installed_version="1.4.0",
            method="pipx",
        ),
    )
    monkeypatch.setattr(ob, "latest_pypi_version", lambda *a, **k: None)  # up to date
    ran = {}

    class _R:
        returncode = 0

    def fake_run(cmd, *a, **k):
        ran["cmd"] = cmd
        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = cli.cmd_onboard(
        argparse.Namespace(
            json=False,
            offline=False,
            dry_run=False,
            force=False,
            upgrade=False,
            no_color=True,
            yes=True,
            no_interactive=False,
        )
    )
    assert rc == 0
    assert "wrap" in " ".join(ran["cmd"])  # --yes launched the route command (step 1)


def test_uninstall_command_per_method() -> None:
    from distil.onboard import uninstall_command

    assert uninstall_command("pipx") == "pipx uninstall distil-llm"
    assert uninstall_command("uv") == "uv tool uninstall distil-llm"
    assert "uninstall" in uninstall_command("pip")
    assert uninstall_command("uvx").startswith("#")  # ephemeral — nothing to remove


def test_cmd_onboard_ephemeral_offers_permanent_install(tmp_path, monkeypatch, capsys) -> None:
    """Run via uvx (method='uvx') → onboard must offer to install distil permanently,
    since nothing is on PATH. All side-effects sandboxed to tmp_path."""
    import argparse
    import subprocess

    from distil import cli, onboard
    from distil import setup as setup_mod

    env = onboard.Env(
        os_name="Darwin",
        managers=["pipx", "uv"],
        agents=[("claude", "Claude Code")],
        subscription=True,
        installed_version="1.5.0",
        method="uvx",
    )
    calls: list = []
    monkeypatch.setattr(onboard, "detect", lambda: env)
    monkeypatch.setattr(onboard, "latest_pypi_version", lambda *a, **k: None)
    monkeypatch.setattr(setup_mod, "default_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(setup_mod, "detect_shell", lambda: ("zsh", tmp_path / ".zshrc"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a[0]) or _rc0())

    cli.cmd_onboard(
        argparse.Namespace(
            dry_run=False,
            force=False,
            no_color=True,
            json=False,
            offline=True,
            upgrade=False,
            yes=True,
            no_interactive=False,
        )
    )
    assert any("pipx install distil-llm" in str(c) for c in calls)  # bootstrapped itself
    assert "running ephemerally" in capsys.readouterr().out


def _rc0():
    import argparse

    return argparse.Namespace(returncode=0)

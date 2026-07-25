"""`distil version` + `distil upgrade` + installer detection (single source)."""

from __future__ import annotations

import argparse

from distil import __version__, onboard
from distil.cli import cmd_upgrade, cmd_version


def test_version_prints(capsys):
    assert cmd_version(argparse.Namespace()) == 0
    assert __version__ in capsys.readouterr().out


def test_install_method_detects_homebrew(monkeypatch):
    # brew wrapper on PATH → homebrew, and the command must be brew (not bare pip)
    monkeypatch.setattr(onboard.shutil, "which", lambda _n: "/usr/local/bin/distil")
    monkeypatch.setattr(onboard.Path, "resolve", lambda self: self)
    assert onboard.install_method() == "homebrew"
    assert onboard.upgrade_command("homebrew") == "brew upgrade distil"
    assert "brew uninstall" in onboard.uninstall_command("homebrew")


def test_command_maps_never_bare_pip_for_managed_installers():
    # brew/pipx/uv commands must be exact (no pip); pip cases carry the venv
    # caveat so a user never hits PEP 668 unguided
    for m in ("homebrew", "pipx", "uv"):
        assert "pip " not in onboard.upgrade_command(m)
        assert "pip " not in onboard.uninstall_command(m)
    assert "venv" in onboard.upgrade_command("pip")
    assert "venv" in onboard.uninstall_command("pip")


def test_upgrade_dry_run_does_not_run(monkeypatch, capsys):
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    ran = {"x": False}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.__setitem__("x", True))
    rc = cmd_upgrade(argparse.Namespace(dry_run=True))
    assert rc == 0 and not ran["x"]
    assert "pipx upgrade distil-llm" in capsys.readouterr().out


def test_upgrade_uvx_is_noop(monkeypatch, capsys):
    monkeypatch.setattr(onboard, "install_method", lambda: "uvx")
    assert cmd_upgrade(argparse.Namespace(dry_run=False)) == 0
    assert "nothing to upgrade" in capsys.readouterr().out.lower()


def test_bad_file_gives_clean_error_not_traceback(capsys):
    """Missing input file → clean message + exit 2, never a raw traceback."""
    from distil.cli import main

    rc = main(["compress", "--trajectory", "/nonexistent-xyz.json"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Traceback" not in err
    assert "distil compress:" in err and "nonexistent-xyz.json" in err


def test_malformed_json_gives_clean_error(tmp_path, capsys):
    from distil.cli import main

    bad = tmp_path / "bad.jsonl"
    bad.write_text("this is not json\n")
    rc = main(["certify-trajectories", str(bad)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Traceback" not in err and "not valid JSON" in err


def test_help_epilog_lists_only_real_commands():
    """Every command named in the grouped --help epilog must be a real subparser
    (regression guard: the epilog once advertised expand/sweep/gate/corpus/adaptive)."""
    import re

    from distil.cli import _HELP_EPILOG, build_parser

    real = set(build_parser()._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
    # words in the epilog that look like command names (lowercase, hyphens)
    named = set(re.findall(r"\b[a-z][a-z-]{2,}\b", _HELP_EPILOG))
    prose = {
        "commands",
        "everyday",
        "guided",
        "install",
        "health",
        "check",
        "wiring",
        "route",
        "agent",
        "traffic",
        "through",
        "compression",
        "your",
        "genuine",
        "savings",
        "live",
        "equivalence",
        "show",
        "version",
        "update",
        "distil",
        "place",
        "analysis",
        "tuning",
        "research",
        "internals",
        "each",
        "flags",
        "leaderboard",
        "and",
        "help",
        "health-check",
        "shows",
        "command",
    }
    phantom = {w for w in named if w not in real and w not in prose}
    assert not phantom, f"help epilog names non-existent commands: {phantom}"

"""Uninstalling distil must never break the machine it was installed on.

Written after a real outage. `distil default` wired two things that both named
`distil` unconditionally — a shell alias (`alias claude='distil wrap -- claude'`) and
an `ANTHROPIC_BASE_URL` pin in `~/.claude/settings.json` — and then the package was
uninstalled. The alias survived, so the user's own agent stopped existing::

    $ claude
    zsh: command not found: distil

and `distil offboard`, the one command that removes it, had been uninstalled in the
same breath. The machine was stranded with no tool left on it able to help.

These tests assert the two properties that prevent a repeat, and they assert them by
EXECUTION, not by string comparison: a generated shell body that merely *contains*
the right substrings is not evidence that a shell runs it correctly. Real `sh`/`zsh`
runs the function; real `sh` runs the escape hatch against a real (sandboxed) HOME.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from distil.setup import alias_body, escape_hatch_spec, service_spec, write_managed

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX shell behaviour; Windows uses the PowerShell body"
)

SHELLS = [sh for sh in ("sh", "bash", "zsh") if shutil.which(sh)]


def _fake_bin(tmp_path, name: str, label: str, code: int):
    """A stand-in executable that reports its own name and argv, then exits *code*."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(
        '#!/bin/sh\nprintf "%s argc=%s" ' + f'"{label}" "$#"\n'
        'for a in "$@"; do printf " <%s>" "$a"; done\necho\n'
        f"exit {code}\n"
    )
    p.chmod(0o755)
    return p


def _run(shell: str, body: str, cmd: str, tmp_path, home=None):
    """Run *body* then *cmd* in a pristine env where only tmp_path/bin is on PATH."""
    return subprocess.run(
        [shell, "-c", f"{body}\n{cmd}"],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "HOME": str(home or tmp_path),
        },
    )


@pytest.mark.skipif(not SHELLS, reason="no POSIX shell available")
@pytest.mark.parametrize("shell", SHELLS)
def test_wiring_routes_through_distil_when_it_is_installed(shell, tmp_path) -> None:
    """The happy path still works — the fallback must not cost us the feature."""
    _fake_bin(tmp_path, "claude", "REAL-CLAUDE", 7)
    _fake_bin(tmp_path, "distil", "DISTIL", 3)
    r = _run(
        shell,
        alias_body("claude", "expand", shell="zsh"),
        "claude --continue 'two words'",
        tmp_path,
    )
    assert "DISTIL" in r.stdout, r.stdout
    assert "<wrap> <--expand> <--> <claude> <--continue> <two words>" in r.stdout
    assert r.returncode == 3, "the agent's exit code must propagate through the wrapper"


@pytest.mark.skipif(not SHELLS, reason="no POSIX shell available")
@pytest.mark.parametrize("shell", SHELLS)
def test_wiring_survives_uninstall(shell, tmp_path) -> None:
    """THE regression test: distil gone → the real agent runs, unchanged.

    Fails loudly against the old `alias claude='distil wrap -- claude'`, which produces
    "command not found: distil" and exit 127 the moment the package is removed.
    """
    _fake_bin(tmp_path, "claude", "REAL-CLAUDE", 7)  # note: no `distil` on PATH at all
    r = _run(
        shell,
        alias_body("claude", "expand", shell="zsh"),
        "claude --continue 'two words'",
        tmp_path,
    )
    assert "REAL-CLAUDE argc=2" in r.stdout, f"agent did not run: {r.stdout!r} {r.stderr!r}"
    assert "<--continue> <two words>" in r.stdout, "quoted arguments must survive as one word"
    assert "not found" not in r.stderr.lower(), r.stderr
    assert r.returncode == 7, "the agent's own exit code, not the shell's 127"


@pytest.mark.skipif(not SHELLS, reason="no POSIX shell available")
@pytest.mark.parametrize("shell", SHELLS)
def test_wiring_does_not_recurse(shell, tmp_path) -> None:
    """A function named `claude` whose fallback calls `claude` would loop forever.

    `command` bypasses shell functions, which is what stops it — the test proves the
    process terminates rather than trusting that reading.
    """
    _fake_bin(tmp_path, "claude", "REAL-CLAUDE", 0)
    r = _run(shell, alias_body("claude", "expand", shell="zsh"), "claude", tmp_path)
    assert r.returncode == 0
    assert r.stdout.count("REAL-CLAUDE") == 1, "ran more than once → recursion"


# ── the escape hatch: cleanup that outlives the package ───────────────────────


def _machine(tmp_path, *, base_url="http://127.0.0.1:8788", end_marker=True):
    """A sandboxed HOME wired exactly the way `distil default --always-on` wires one."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    block = "# >>> distil (managed) — route your agent through distil >>>\nexport X=1\n"
    if end_marker:
        block += "# <<< distil (managed) <<<\n"
    (home / ".zshrc").write_text(f"export FOO=1\n{block}export BAR=2\n")
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": base_url, "KEEP": "1"}, "model": "opus"})
    )
    (home / "Library" / "LaunchAgents" / "com.distil.proxy.plist").write_text("<plist/>")
    return home


def _run_hatch(home, tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(home / ".distil"))
    path, content = escape_hatch_spec(8788)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    # `env` deliberately omits PATH entries that could contain a real distil: the whole
    # point is that this script works on a machine where distil no longer exists.
    return subprocess.run(
        ["sh", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
    )


def test_escape_hatch_is_valid_shell() -> None:
    """Generated by f-string with brace-escaping — a syntax slip would only surface on
    a user's machine, at the exact moment they have no other way out."""
    _, content = escape_hatch_spec(8788)
    r = subprocess.run(["sh", "-n", "-"], input=content, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_escape_hatch_unwires_a_machine_without_distil(tmp_path, monkeypatch) -> None:
    home = _machine(tmp_path)
    r = _run_hatch(home, tmp_path, monkeypatch)
    assert r.returncode == 0, r.stderr

    rc = (home / ".zshrc").read_text()
    assert "distil" not in rc
    assert "export FOO=1" in rc and "export BAR=2" in rc, "user's own rc content destroyed"
    assert (home / ".zshrc.distil-bak").exists(), "must back up before editing"

    settings = json.loads((home / ".claude" / "settings.json").read_text())
    assert "ANTHROPIC_BASE_URL" not in settings["env"], "the pin that strands the machine"
    assert settings["env"]["KEEP"] == "1" and settings["model"] == "opus", "clobbered user settings"

    assert not (home / "Library" / "LaunchAgents" / "com.distil.proxy.plist").exists()


def test_escape_hatch_keeps_a_base_url_that_is_not_ours(tmp_path, monkeypatch) -> None:
    """A corporate gateway is not distil's to delete, even when run by mistake."""
    home = _machine(tmp_path, base_url="https://gateway.corp.example.com")
    r = _run_hatch(home, tmp_path, monkeypatch)
    assert r.returncode == 0
    settings = json.loads((home / ".claude" / "settings.json").read_text())
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "https://gateway.corp.example.com"
    assert "kept" in r.stdout


def test_escape_hatch_refuses_a_block_with_no_end_marker(tmp_path, monkeypatch) -> None:
    """`sed '/start/,/end/d'` deletes to EOF when the end marker is missing.

    On a file whose block was half-removed by hand that would silently take everything
    after it — so the script must decline and say why, not "helpfully" truncate.
    """
    home = _machine(tmp_path, end_marker=False)
    r = _run_hatch(home, tmp_path, monkeypatch)
    assert r.returncode == 0
    rc = (home / ".zshrc").read_text()
    assert "export BAR=2" in rc, "content after the unterminated block was destroyed"
    assert "no end marker" in r.stdout


def test_escape_hatch_is_idempotent_and_safe_on_a_clean_machine(tmp_path, monkeypatch) -> None:
    home = _machine(tmp_path)
    assert _run_hatch(home, tmp_path, monkeypatch).returncode == 0
    before = (home / ".zshrc").read_text()
    second = _run_hatch(home, tmp_path, monkeypatch)
    assert second.returncode == 0
    assert (home / ".zshrc").read_text() == before
    assert "already clean" in second.stdout


def test_escape_hatch_tolerates_malformed_settings_json(tmp_path, monkeypatch) -> None:
    """Half-written JSON must not abort the run — the rc and service still need removing."""
    home = _machine(tmp_path)
    (home / ".claude" / "settings.json").write_text("{not json")
    r = _run_hatch(home, tmp_path, monkeypatch)
    assert r.returncode == 0, r.stderr
    assert "distil" not in (home / ".zshrc").read_text(), "one bad file stopped the whole cleanup"


@pytest.mark.skipif(not SHELLS, reason="no POSIX shell available")
@pytest.mark.parametrize("sh", SHELLS)
def test_wiring_survives_a_preexisting_alias(sh, tmp_path) -> None:
    """An existing `alias claude=...` must not make the rc file fail to parse.

    bash and zsh expand an alias on the word before `()` while READING a function
    definition, so `claude() { ... }` after `alias claude=...` is a syntax error —
    and the shell abandons the rest of the rc file, taking the user's PATH and
    prompt with it. Earlier distil versions installed exactly such an alias, so
    every upgrading user hits this. String-matching the body cannot catch it;
    only running a real shell can.
    """
    _fake_bin(tmp_path, "claude", "REAL-CLAUDE", 0)
    _fake_bin(tmp_path, "distil", "DISTIL", 0)
    # Sourcing a file is what makes this bite: that is how an rc file is read, and
    # alias expansion happens as each line is parsed.
    rc = tmp_path / "rc"
    rc.write_text(
        "alias claude='echo STALE_ALIAS'\n"
        + alias_body("claude", "expand", shell="zsh")
        + "\nRC_COMPLETED=yes\n"
    )
    r = _run(sh, "", f'. "{rc}"; echo "rc_done=$RC_COMPLETED"; claude', tmp_path)
    assert "rc_done=yes" in r.stdout, (
        f"the rc file aborted early — a pre-existing alias broke parsing.\n"
        f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    )
    assert "DISTIL" in r.stdout, r.stdout
    assert "STALE_ALIAS" not in r.stdout


@pytest.mark.skipif(not SHELLS, reason="no POSIX shell available")
@pytest.mark.parametrize("shell", SHELLS)
def test_a_hyphenated_agent_never_breaks_the_rc_file(shell, tmp_path) -> None:
    """`claude-code() {` is a SYNTAX ERROR in dash, and that aborts ~/.profile.

    POSIX function names are [A-Za-z_][A-Za-z0-9_]*. bash and zsh accept `-`
    anyway, but dash is /bin/sh on Debian/Ubuntu and reads ~/.profile, so wiring a
    hyphenated agent there would take out everything the profile sets after our
    block — PATH included. Degrading to "no wrapper for this agent" is acceptable;
    breaking the user's shell is not.
    """
    _fake_bin(tmp_path, "claude-code", "REAL-AGENT", 5)
    rc = tmp_path / "rc"
    rc.write_text(alias_body("claude-code", "expand", shell="zsh") + "\nRC_COMPLETED=yes\n")
    r = _run(shell, "", f'. "{rc}"; echo "rc_done=$RC_COMPLETED"; claude-code x', tmp_path)
    assert "rc_done=yes" in r.stdout, (
        f"the rc file aborted on a hyphenated agent name.\nstderr={r.stderr!r}"
    )
    # The agent itself must still run, wrapper or not.
    assert "REAL-AGENT" in r.stdout, r.stdout
    assert r.returncode == 5, "the agent's own exit code must survive"


def test_escape_hatch_cleans_the_rc_file_that_was_actually_wired(tmp_path, monkeypatch) -> None:
    """`--rc /some/path` and $ZDOTDIR both put the block outside the hardcoded list.

    A hatch that scans only `$HOME/.zshrc` and friends misses the one file it was
    written to clean — on exactly the machine that needed it, since a user with a
    custom rc path is the user whose setup the defaults did not fit.
    """
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    custom = tmp_path / "dotfiles" / "shell rc"  # a space, to prove the quoting
    custom.parent.mkdir(parents=True)
    custom.write_text(
        "export MINE=1\n"
        "# >>> distil (managed) — route your agent through distil >>>\n"
        "export ANTHROPIC_BASE_URL=http://127.0.0.1:8788\n"
        "# <<< distil (managed) <<<\n"
        "export ALSO_MINE=2\n"
    )
    monkeypatch.setenv("DISTIL_HOME", str(home / ".distil"))
    path, content = escape_hatch_spec(8788, custom)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    r = subprocess.run(
        ["sh", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
    )
    assert r.returncode == 0, r.stderr
    left = custom.read_text()
    assert "distil" not in left, f"the wired rc was not cleaned:\n{left}"
    assert "export MINE=1" in left and "export ALSO_MINE=2" in left


def test_a_truncated_managed_block_is_reported_not_silently_kept(tmp_path) -> None:
    """Start marker with no end marker must be an error, not a cheerful ✓.

    `--undo` and `offboard` both print a tick straight off this status, so
    returning "ok" while removing nothing tells the user the machine is clean
    when the wiring is still live. Deleting to end-of-file instead would take
    every line they added after our block with it — so it refuses and says so.
    """
    rc = tmp_path / ".zshrc"
    from distil.setup import _MARK_START, remove_managed

    rc.write_text(f"export MINE=1\n{_MARK_START}\nexport ANTHROPIC_BASE_URL=x\n")
    status, msg = remove_managed(rc)
    assert status == "error", f"a truncated block reported {status!r}: {msg}"
    assert "end marker" in msg
    assert "export MINE=1" in rc.read_text(), "the user's own lines must survive"
    assert _MARK_START in rc.read_text(), "nothing was removed, so nothing may be claimed"


def test_managed_write_does_not_destroy_a_symlinked_rc(tmp_path) -> None:
    """Dotfile repos symlink ~/.zshrc; os.replace onto the link would break it.

    Replacing the symlink with a regular file silently detaches the rc from the
    repo that manages it — the user's own edits stop applying and ours disappear
    at the next stow/chezmoi run.
    """
    real = tmp_path / "dotfiles" / "zshrc"
    real.parent.mkdir()
    real.write_text("# managed by my dotfiles repo\n")
    link = tmp_path / ".zshrc"
    link.symlink_to(real)

    write_managed(link, alias_body("claude", "lossless-only", shell="zsh"))

    assert link.is_symlink(), "the symlink was replaced by a regular file"
    assert link.resolve() == real.resolve()
    assert "# managed by my dotfiles repo" in real.read_text()
    assert "distil" in real.read_text(), "the block must land in the real file"


# ── the service must be able to say why it died ───────────────────────────────


def test_service_logs_its_own_failure(monkeypatch, tmp_path) -> None:
    """The outage was undiagnosable: launchd discarded the proxy's stdout and stderr,
    so nothing on the machine recorded why it stopped — no `log show` entry, no crash
    report. A mandatory service must leave evidence."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    _, content, _ = service_spec(8788, "expand")
    assert f"<key>StandardOutPath</key><string>{tmp_path / 'proxy.log'}</string>" in content
    assert f"<key>StandardErrorPath</key><string>{tmp_path / 'proxy.err'}</string>" in content
    # 1s, not the 10s a crash-loop guard would want. Once launchd holds the socket
    # (see test_activation.py) a client's connection is already queued and waiting
    # during a restart, so every second of throttle is latency on a live request
    # rather than protection. The crash-loop concern is answered by the logs above.
    assert "<key>ThrottleInterval</key><integer>1</integer>" in content

    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, unit, _ = service_spec(8788, "expand")
    assert "RestartSec=10" in unit  # journald already captures output on this platform


def test_service_plist_is_well_formed_xml(monkeypatch, tmp_path) -> None:
    """A malformed plist is rejected by launchd at load time, and the failure surfaces
    much later as "the proxy isn't running" with no obvious cause."""
    import plistlib

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    _, content, _ = service_spec(8788, "expand")
    parsed = plistlib.loads(content.encode())
    assert parsed["Label"] == "com.distil.proxy"
    assert parsed["ProgramArguments"][1:] == ["proxy", "--expand", "--port", "8788"]
    assert parsed["KeepAlive"] is True and parsed["ThrottleInterval"] == 1
    assert os.path.isabs(parsed["StandardErrorPath"])

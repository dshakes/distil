"""``distil setup`` — wire the distil status line into Claude Code settings.

Replaces a manual ``settings.json`` edit. Idempotent, never clobbers an existing
status line without ``--force`` (and backs it up when it does), and preserves every
other setting.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_COMMAND = "distil statusline"

# An agent name is spliced verbatim into a shell alias/function written to the
# user's rc file, so it must be a bare command token — never arbitrary shell.
_AGENT_RE = re.compile(r"[\w.+-]+")


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (tmp + os.replace) so an interrupted
    write can't leave a half-written shell rc that breaks the next shell start.

    Symlinks are resolved first. Dotfiles are very often a git repo symlinked
    into ``$HOME``, and ``os.replace`` onto a symlink *destroys the link*,
    silently detaching the user's rc file from the repo that manages it: their
    edits stop taking effect and ours vanish on the next `stow`/`chezmoi` run.
    Writing through the link keeps the file the user actually version-controls.
    """
    target = path.resolve() if path.is_symlink() else path
    tmp = target.with_name(target.name + ".distil.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def default_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def claude_settings_files(cwd: Path | None = None) -> list[Path]:
    """Every settings file Claude Code merges, highest precedence first.

    ``doctor`` and ``--undo`` used to read only ``~/.claude/settings.json``. Claude
    Code also reads *project*-scoped settings, and those **override** the user's —
    so a dead ``ANTHROPIC_BASE_URL`` in a repo's ``.claude/settings.local.json`` is
    invisible to a check that reads the home file, and survives every uninstall.
    That is exactly how one stale port outlived ``distil default --undo`` and went
    on killing every session started in that directory after distil was gone.

    Includes non-existent paths: callers test for presence, and an undo that only
    visits files it already knows about is the bug this exists to close.
    """
    home = Path.home()
    paths = [
        Path("/etc/claude-code/managed-settings.json")
        if _is_windows() or not Path("/Library/Application Support").is_dir()
        else Path("/Library/Application Support/ClaudeCode/managed-settings.json")
    ]
    start = (cwd or Path.cwd()).resolve()
    for d in (start, *start.parents):
        paths += [d / ".claude" / "settings.local.json", d / ".claude" / "settings.json"]
        if d == home:
            break
    paths += [home / ".claude" / "settings.local.json", home / ".claude" / "settings.json"]
    return list(dict.fromkeys(paths))  # dedupe, first (highest-precedence) wins


def loopback_base_url(settings_path: Path) -> str | None:
    """The loopback ``ANTHROPIC_BASE_URL`` in *settings_path*, or None.

    Read-only, so a caller can ask "is there anything here?" before prompting.
    ``offboard`` used to prompt off ``settings_path.exists()`` alone, which asked
    about files holding nothing and — worse — asked about nothing at all when the
    home file was absent, skipping every other file in the process.
    """
    from urllib.parse import urlparse

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    env = data.get("env") if isinstance(data, dict) else None
    val = env.get("ANTHROPIC_BASE_URL") if isinstance(env, dict) else None
    if not val or urlparse(str(val)).hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    return str(val)


def unwire_base_url(settings_path: Path) -> tuple[str, str]:
    """Remove a **loopback** ``ANTHROPIC_BASE_URL`` from *settings_path*.

    The port-exact :func:`unwire_settings_env` is not enough for undo: wire on one
    port, undo without repeating ``--port``, and the entry is judged "foreign" and
    left behind — an uninstall that reports success and leaves the machine broken.
    Any ``127.0.0.1``/``localhost`` URL is distil's to clean up; a real remote host
    is someone else's gateway and is left exactly as it is.

    Returns ``(status, message)``: ``ok`` | ``absent`` | ``foreign`` | ``error``.
    """
    from urllib.parse import urlparse

    if not settings_path.exists():
        return ("absent", f"no settings file at {settings_path}")
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return ("error", f"{settings_path} is not valid JSON ({exc})")
    if not isinstance(data, dict):
        return ("error", f"{settings_path} is not a JSON object")

    env = data.get("env")
    existing = env.get("ANTHROPIC_BASE_URL") if isinstance(env, dict) else None
    if not existing:
        return ("absent", f"no ANTHROPIC_BASE_URL in {settings_path}")
    if urlparse(str(existing)).hostname not in ("127.0.0.1", "localhost", "::1"):
        return ("foreign", f"ANTHROPIC_BASE_URL is {existing!r} (not loopback) — left as-is")

    settings_path.with_name(settings_path.name + ".bak").write_text(
        settings_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert isinstance(env, dict)  # a truthy `existing` came out of that dict
    del env["ANTHROPIC_BASE_URL"]
    if not env:
        del data["env"]
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ("ok", f"removed ANTHROPIC_BASE_URL ({existing}) from {settings_path}")


def wire_statusline(
    settings_path: Path, *, command: str = DEFAULT_COMMAND, force: bool = False
) -> tuple[str, str]:
    """Wire the distil status line into ``settings_path``.

    Returns ``(status, message)`` where status is one of:
    ``ok`` (wired), ``exists`` (already distil), ``conflict`` (another line set,
    needs ``--force``), ``error`` (unreadable / not an object)."""
    data: object = {}
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return ("error", f"{settings_path} is not valid JSON ({exc}) — fix it or edit by hand")
    if not isinstance(data, dict):
        return ("error", f"{settings_path} is not a JSON object")

    sl = data.get("statusLine")
    existing = sl.get("command", "") if isinstance(sl, dict) else ""
    if "distil" in (existing or ""):
        return ("exists", "distil status line already wired")
    if existing and not force:
        return (
            "conflict",
            f"a status line is already set ({existing!r}); "
            "re-run with --force to replace it (it'll be backed up first)",
        )
    if existing:  # force: back up the current settings before replacing
        settings_path.with_name(settings_path.name + ".bak").write_text(
            settings_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    data["statusLine"] = {"type": "command", "command": command, "padding": 0}
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ("ok", f"wired the distil status line into {settings_path}")


def wire_settings_env(
    settings_path: Path, env_var: str, value: str, *, force: bool = False
) -> tuple[str, str]:
    """Wire ``{env_var: value}`` into ``settings_path``'s ``env`` block.

    Claude Code reads its own ``settings.json`` on every launch, regardless of
    how the ``claude`` binary was started — unlike a shell alias or an
    ``export`` in an rc file, this also reaches IDE-launched sessions (VSCode's
    Claude Code extension, or that same extension installed inside Cursor),
    since those exec the binary directly and never source an interactive
    shell's rc file. Idempotent; preserves every other key including other
    ``env`` entries; backs up before replacing a conflicting value.

    Returns ``(status, message)``: ``ok`` | ``exists`` | ``conflict`` | ``error``.
    """
    data: object = {}
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return ("error", f"{settings_path} is not valid JSON ({exc}) — fix it or edit by hand")
    if not isinstance(data, dict):
        return ("error", f"{settings_path} is not a JSON object")

    env = data.get("env")
    existing = env.get(env_var) if isinstance(env, dict) else None
    if existing == value:
        return ("exists", f"distil's {env_var} already wired")
    if existing and not force:
        return (
            "conflict",
            f"{env_var} is already set to {existing!r} in {settings_path}; "
            "re-run with --force to replace it (it'll be backed up first)",
        )
    if existing:  # force: back up the current settings before replacing
        settings_path.with_name(settings_path.name + ".bak").write_text(
            settings_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    data["env"] = {**env, env_var: value} if isinstance(env, dict) else {env_var: value}
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ("ok", f"wired {env_var} into {settings_path}")


def unwire_settings_env(settings_path: Path, env_var: str, value: str) -> tuple[str, str]:
    """Remove distil's ``env_var`` from ``settings_path`` (inverse of
    :func:`wire_settings_env`). Only touches the entry if it still holds the
    exact value distil set — a foreign value (the user repointed it, or set it
    to something else) is left untouched. Backs up before changing. Returns
    ``(status, message)``: ``ok`` | ``absent`` | ``foreign`` | ``error``."""
    if not settings_path.exists():
        return ("absent", f"no settings file at {settings_path}")
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return ("error", f"{settings_path} is not valid JSON ({exc})")
    if not isinstance(data, dict):
        return ("error", f"{settings_path} is not a JSON object")

    env = data.get("env")
    existing = env.get(env_var) if isinstance(env, dict) else None
    if existing != value:
        if existing is None:
            return ("absent", f"no distil {env_var} to remove")
        return ("foreign", f"{env_var} is {existing!r}, not distil's — left as-is")

    settings_path.with_name(settings_path.name + ".bak").write_text(
        settings_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert isinstance(env, dict)  # existing == value (a real str) ⇒ env is the dict we read it from
    del env[env_var]
    if not env:
        del data["env"]
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ("ok", f"removed distil's {env_var} from {settings_path}")


def unwire_statusline(settings_path: Path) -> tuple[str, str]:
    """Remove the distil status line from ``settings_path`` (the inverse of
    :func:`wire_statusline`). Only touches a status line that is distil's — a
    foreign one is left untouched. Backs up before changing. Returns ``(status,
    message)``: ``ok`` | ``absent`` | ``foreign`` | ``error``."""
    if not settings_path.exists():
        return ("absent", f"no settings file at {settings_path}")
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return ("error", f"{settings_path} is not valid JSON ({exc})")
    if not isinstance(data, dict):
        return ("error", f"{settings_path} is not a JSON object")

    sl = data.get("statusLine")
    cmd = sl.get("command", "") if isinstance(sl, dict) else ""
    if "distil" not in (cmd or ""):
        if sl is None:
            return ("absent", "no distil status line to remove")
        return ("foreign", f"status line is {cmd!r}, not distil — left as-is")

    settings_path.with_name(settings_path.name + ".bak").write_text(
        settings_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    del data["statusLine"]
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ("ok", f"removed the distil status line from {settings_path}")


# ── distil default: route an agent through distil by default ─────────────────
# Two strategies, both via a single marked block we can add/replace/remove:
#   A (alias):     wrap the agent command — no daemon, no single point of failure.
#   B (always-on): a persistent ANTHROPIC_BASE_URL + a managed proxy service —
#                  universal (every SDK), but the proxy must stay up.

_MARK_START = "# >>> distil (managed) — route your agent through distil >>>"
_MARK_END = "# <<< distil (managed) <<<"


def _is_windows() -> bool:
    import platform

    return platform.system() == "Windows"


def detect_shell() -> tuple[str, Path]:
    """(shell_name, rc_path) — the file an *interactive* shell actually sources.

    Each machine differs, so this is explicit about conventions and reported back
    to the user rather than applied blind: zsh→.zshrc, fish→config.fish, bash→.bashrc
    (interactive; .bash_profile only if that's the one present), PowerShell→$PROFILE,
    otherwise →.profile."""
    import os

    home = Path.home()
    if _is_windows():
        prof = os.environ.get("PROFILE")
        return (
            "powershell",
            Path(prof)
            if prof
            else home / "Documents" / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1",
        )
    name = os.path.basename(os.environ.get("SHELL", "")).lower()
    fish_rc = home / ".config" / "fish" / "config.fish"

    def _bash_rc() -> Path:
        if (home / ".bashrc").exists():
            return home / ".bashrc"
        if (home / ".bash_profile").exists():
            return home / ".bash_profile"
        return home / ".bashrc"

    # An explicit $SHELL is authoritative — it beats file-existence heuristics.
    if "fish" in name:
        return ("fish", fish_rc)
    if "zsh" in name:
        return ("zsh", home / ".zshrc")
    if "bash" in name:
        return ("bash", _bash_rc())
    # $SHELL unset/unknown: fall back to whichever rc actually exists.
    if fish_rc.exists():
        return ("fish", fish_rc)
    if (home / ".zshrc").exists():
        return ("zsh", home / ".zshrc")
    if (home / ".bashrc").exists() or (home / ".bash_profile").exists():
        return ("bash", _bash_rc())
    return (name or "sh", home / ".profile")


def default_shell_rc() -> Path:
    """Back-compat: just the rc path from :func:`detect_shell`."""
    return detect_shell()[1]


def alias_body(agent: str, mode: str, *, shell: str | None = None) -> str:
    """Strategy A — wrap the agent command on demand, falling back to the real one.

    A **function**, not an alias, and the difference is the whole point. The alias
    this replaced (``alias claude='distil wrap -- claude'``) named ``distil``
    unconditionally, so the moment the package was uninstalled the user's own agent
    stopped existing::

        $ claude
        zsh: command not found: distil

    — and the tool that removes the alias (``distil offboard``) had just been deleted
    too, so nothing on the machine could undo it. Uninstalling a compression proxy
    must never take the agent down with it: if ``distil`` is gone, run the real binary
    and say nothing. ``command`` bypasses this function, so there is no recursion.
    """
    if not _AGENT_RE.fullmatch(agent or ""):
        raise ValueError(
            f"invalid agent name {agent!r}: expected only letters, digits, '.', '+', '-', '_' "
            "(it is spliced into a shell function body, so arbitrary strings are refused)"
        )
    sh = shell if shell is not None else ("powershell" if _is_windows() else "")
    if sh == "powershell":
        return (
            f"function {agent} {{ if (Get-Command distil -ErrorAction SilentlyContinue) "
            f"{{ distil wrap --{mode} -- {agent} @args }} else "
            f"{{ & (Get-Command {agent} -CommandType Application | "
            "Select-Object -First 1).Source @args } }"
        )
    if sh == "fish":
        return (
            f"function {agent}; if command -q distil; distil wrap --{mode} -- {agent} $argv; "
            f"else; command {agent} $argv; end; end"
        )
    # `unalias` first, on its own line, and it is load-bearing. bash and zsh
    # alias-expand a word BEFORE the `()` when reading a function definition, so
    # if the user already has `alias claude=...` (very common, and what earlier
    # versions of distil itself installed), the next line parses as garbage and
    # the shell aborts the rest of the rc file — taking their PATH, prompt, and
    # every other tool down with it. Expansion happens as each line is read, so
    # an unalias on the preceding line is in effect by the time this one parses.
    # `|| true` keeps `set -e` rc files alive when there was no alias to remove.
    definition = (
        f"{agent}() {{ if command -v distil >/dev/null 2>&1; "
        f'then distil wrap --{mode} -- {agent} "$@"; '
        f'else command {agent} "$@"; fi; }}'
    )
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", agent):
        # POSIX allows only [A-Za-z_][A-Za-z0-9_]* in a function name. bash and zsh
        # accept `-` and `.` anyway, but dash (which is /bin/sh on Debian/Ubuntu,
        # and reads ~/.profile) treats `claude-code() {` as a SYNTAX ERROR — and a
        # syntax error aborts the rest of the file, so wiring an agent whose binary
        # happens to be hyphenated would silently take out the user's PATH and
        # everything else their profile sets.
        #
        # The fix is a capability PROBE in a subshell, not `eval ... || true`:
        # `eval` is a POSIX *special builtin*, and a syntax error in a special
        # builtin makes a non-interactive shell exit immediately — the `|| true`
        # never runs. Verified: dash and macOS `sh` abandoned the file on the spot.
        # Inside `( )` the failure is confined to the subshell and reaches us as an
        # exit status, so the real definition only runs on a shell that has just
        # demonstrated it can parse the name. Worst case is "no wrapper for this
        # agent" — the agent still works, since it is on PATH either way.
        assert "'" not in definition  # guaranteed by _AGENT_RE / the mode check
        definition = (
            f"if (eval '_distil_probe-() {{ :; }}') 2>/dev/null; then eval '{definition}'; fi"
        )
    return f"unalias {agent} 2>/dev/null || true\n{definition}"


def env_body(port: int, *, shell: str | None = None) -> str:
    """Strategy B — point every SDK at the always-on proxy, but only if distil exists.

    The guard is not decoration. An unconditional ``export ANTHROPIC_BASE_URL`` outlives
    the package that honours it, and a base URL pointing at a port nothing listens on
    fails every SDK in the shell with a connection error that reads like a provider
    outage. ``command -v`` is a builtin — no measurable shell-startup cost — and it
    turns "uninstalled distil" from a broken machine into a no-op.

    It deliberately does NOT probe the port: a shell rc file is the wrong place to open
    a socket, and a running distil with a stopped proxy is what ``distil wrap``'s
    precedence guard and ``distil doctor`` exist to catch.
    """
    sh = shell if shell is not None else ("powershell" if _is_windows() else "")
    if sh == "powershell":
        return (
            "if (Get-Command distil -ErrorAction SilentlyContinue) "
            f'{{ $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:{port}" }}'
        )
    if sh == "fish":
        return f"if command -q distil; set -gx ANTHROPIC_BASE_URL http://127.0.0.1:{port}; end"
    return f"command -v distil >/dev/null 2>&1 && export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}"


def write_managed(rc: Path, body: str) -> tuple[str, str]:
    """Install/update the single managed block in ``rc`` (backs up before changing).

    Returns (status, message): ``ok`` | ``updated`` | ``exists``."""
    import re

    block = f"{_MARK_START}\n{body}\n{_MARK_END}\n"
    text = rc.read_text(encoding="utf-8") if rc.exists() else ""
    if _MARK_START in text:
        new = re.sub(
            re.escape(_MARK_START) + r".*?" + re.escape(_MARK_END) + r"\n?", block, text, flags=re.S
        )
        if new == text:
            return ("exists", f"already configured in {rc}")
        rc.with_name(rc.name + ".bak").write_text(text, encoding="utf-8")
        _atomic_write(rc, new)
        return ("updated", f"updated the distil default in {rc}")
    rc.parent.mkdir(parents=True, exist_ok=True)
    if rc.exists():
        rc.with_name(rc.name + ".bak").write_text(text, encoding="utf-8")
    sep = "" if (not text or text.endswith("\n")) else "\n"
    _atomic_write(rc, text + sep + "\n" + block)
    return ("ok", f"configured the distil default in {rc}")


def remove_managed(rc: Path) -> tuple[str, str]:
    """Remove the managed block from ``rc`` (idempotent)."""
    import re

    if not rc.exists():
        return ("absent", f"nothing to remove ({rc} doesn't exist)")
    text = rc.read_text(encoding="utf-8")
    if _MARK_START not in text:
        return ("absent", f"distil default not found in {rc}")
    new = re.sub(
        r"\n?" + re.escape(_MARK_START) + r".*?" + re.escape(_MARK_END) + r"\n?",
        "\n",
        text,
        flags=re.S,
    )
    if new == text:
        # The start marker is present but the substitution matched nothing, so the
        # end marker is missing — a block truncated by a hand-edit or a crashed
        # write. Reporting "removed" here would be a lie with consequences: undo
        # and offboard both print ✓ off this result, the user believes the machine
        # is clean, and the wiring keeps running. Deleting to EOF instead would be
        # worse — everything the user added after our block would go with it.
        return (
            "error",
            f"found distil's start marker in {rc} but no end marker — the block looks "
            f"truncated, so it was left alone rather than guessing where it ends. "
            f"Remove the lines from '{_MARK_START}' to the end of distil's block by hand.",
        )
    rc.with_name(rc.name + ".bak").write_text(text, encoding="utf-8")
    _atomic_write(rc, new)
    return ("ok", f"removed the distil default from {rc}")


def log_dir() -> Path:
    """Where the always-on service writes stdout/stderr. Honours ``DISTIL_HOME``."""
    import os

    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def escape_hatch_spec(port: int, rc: "Path | None" = None) -> tuple[Path, str]:
    """A standalone un-wiring script: (path, content). Needs nothing but ``sh``.

    ``distil offboard`` is the supported way out, and it is *part of the package* —
    which makes it useless in the one situation that matters most. Uninstall first
    (the obvious order: remove the tool, then clean up) and the ``ANTHROPIC_BASE_URL``
    pin in Claude Code's ``settings.json`` stays behind with nothing left on the
    machine able to remove it. Static JSON cannot self-heal the way the shell wiring
    now does, so every session keeps failing with a connection error that names the
    provider, and the documented fix (``distil offboard``) is a command that no longer
    exists.

    So the way out gets written to disk at wiring time, in plain ``sh``, and outlives
    the package by construction. Deleting distil can no longer strand the machine.
    """
    hatch = log_dir() / "uninstall.sh"
    # The rc file `distil default` actually wrote to, quoted for sh. The hardcoded
    # list below covers the common layouts, but it cannot know about `--rc
    # /some/path` or a $ZDOTDIR-relocated zsh config — and a hatch that misses the
    # one file it was supposed to clean is a hatch that does not work on precisely
    # the machine that needed it.
    import shlex as _shlex

    extra_rc = f" {_shlex.quote(str(rc))}" if rc is not None else ""
    # Only ever removes wiring that points at loopback: a user whose ANTHROPIC_BASE_URL
    # is a corporate gateway keeps it, even if they run this by mistake.
    content = f"""#!/bin/sh
# distil escape hatch — removes distil's machine wiring WITHOUT needing distil.
#
# Written by `distil default --always-on`. Safe to run twice, safe to run after the
# package is uninstalled — that is the entire point of it existing. Prefer
# `distil offboard` while distil is still installed; use this when it isn't.
#
# Removes: the managed shell block, the always-on proxy service, and any loopback
# ANTHROPIC_BASE_URL pinned in a Claude Code settings file. Leaves your savings data.
set -u
echo "distil escape hatch — removing machine wiring"
changed=0

# 1 · managed block in shell rc files
for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" \\
          "$HOME/.zprofile" "$HOME/.config/fish/config.fish" \\
          "${{ZDOTDIR:-$HOME}}/.zshrc" "${{ZDOTDIR:-$HOME}}/.zprofile"{extra_rc}; do
    [ -f "$rc" ] || continue
    grep -q '# >>> distil (managed)' "$rc" 2>/dev/null || continue
    # BOTH markers, or nothing. `sed '/start/,/end/d'` deletes through end-of-file when
    # the end marker is missing, so a half-deleted block would take the whole rest of
    # the user's rc file with it — silently, and only for someone whose file was already
    # damaged. Refuse and say so instead.
    if ! grep -q '# <<< distil (managed)' "$rc" 2>/dev/null; then
        echo "  ! $rc has a distil start marker but no end marker — left alone (edit by hand)"
        continue
    fi
    cp "$rc" "$rc.distil-bak" || continue
    sed '/# >>> distil (managed)/,/# <<< distil (managed)/d' "$rc" > "$rc.distil-tmp" \\
        && mv "$rc.distil-tmp" "$rc" \\
        && {{ echo "  removed distil block from $rc  (backup: $rc.distil-bak)"; changed=1; }}
done

# 2 · always-on proxy service
plist="$HOME/Library/LaunchAgents/com.distil.proxy.plist"
if [ -f "$plist" ]; then
    launchctl unload "$plist" 2>/dev/null
    rm -f "$plist" && {{ echo "  removed proxy service $plist"; changed=1; }}
fi
unit="$HOME/.config/systemd/user/distil-proxy.service"
sock="$HOME/.config/systemd/user/distil-proxy.socket"
# The socket unit OWNS the listening port. Leaving it behind keeps 8788 bound
# after an uninstall and keeps systemd trying to start a service that is gone.
if [ -f "$unit" ] || [ -f "$sock" ]; then
    systemctl --user disable --now distil-proxy.socket distil-proxy.service 2>/dev/null
    rm -f "$unit" && {{ echo "  removed proxy service $unit"; changed=1; }}
    rm -f "$sock" && {{ echo "  removed proxy socket $sock"; changed=1; }}
    systemctl --user daemon-reload 2>/dev/null
fi

# 3 · loopback ANTHROPIC_BASE_URL in Claude Code settings (JSON — needs python3).
# Extra settings paths may be passed as arguments, for project-scoped files this
# script could not know about when it was written.
for s in "$HOME/.claude/settings.json" "$HOME/.claude/settings.local.json" "$@"; do
    [ -f "$s" ] || continue
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$s" <<'PY'
import json, sys
from urllib.parse import urlparse
p = sys.argv[1]
try:
    with open(p) as fh:
        d = json.load(fh)
except Exception as exc:
    print(f"  ! could not read {{p}}: {{exc}}")
    sys.exit(0)
env = d.get("env") if isinstance(d, dict) else None
url = env.get("ANTHROPIC_BASE_URL") if isinstance(env, dict) else None
if not url or not isinstance(url, str):
    sys.exit(0)
# Parse the host; do NOT substring-match. "127.0.0.1" appearing anywhere in a URL
# is not evidence that it IS loopback: a corporate gateway at
# https://gw.corp.example.com/pools/127.0.0.1 would be silently deleted by a
# substring test, and http://127.1:8788 — a valid loopback shorthand — would be
# kept. This mirrors loopback_base_url() in distil/setup.py.
try:
    host = urlparse(url).hostname or ""
except ValueError:
    host = ""
if host not in ("127.0.0.1", "127.1", "localhost", "::1", "0.0.0.0"):
    print(f"  · kept ANTHROPIC_BASE_URL={{url}} in {{p}} (not a local proxy — not ours)")
    sys.exit(0)
del env["ANTHROPIC_BASE_URL"]
if not env:
    d.pop("env", None)
try:
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\\n")
except OSError as exc:
    print(f"  ! could not write {{p}}: {{exc}} — remove ANTHROPIC_BASE_URL by hand")
    sys.exit(0)
print(f"  removed ANTHROPIC_BASE_URL={{url}} from {{p}}")
sys.exit(10)  # 10 = "I changed something", so the summary below cannot overclaim
PY
        [ $? -eq 10 ] && changed=1
    else
        echo "  ! python3 not found — remove the ANTHROPIC_BASE_URL line from $s by hand"
    fi
done

# 4 · anything left pointing at us, so a silent miss is impossible
if grep -rl 'ANTHROPIC_BASE_URL' "$HOME/.claude" 2>/dev/null | grep -q .; then
    echo ""
    echo "  Still referencing ANTHROPIC_BASE_URL (check these by hand):"
    grep -rln 'ANTHROPIC_BASE_URL' "$HOME/.claude" 2>/dev/null | sed 's/^/    /'
fi

[ "$changed" -eq 1 ] || echo "  nothing to remove — this machine is already clean"
echo ""
echo "Done. Open a NEW terminal (this shell still has the old settings loaded)."
echo "Your savings data is untouched in {log_dir()}"
echo "Port {port} is no longer pinned by distil."
"""
    return hatch, content


def service_spec(port: int, mode: str) -> tuple[Path | None, str | None, str | None]:
    """Always-on proxy service for this platform: (path, file_content, load_command).

    Returns (None, None, None) on an unsupported platform."""
    import platform
    import shutil
    from xml.sax.saxutils import escape as _xml

    home = Path.home()
    distil = shutil.which("distil") or "distil"
    # `mode` is constrained by argparse `choices=` and `port` by `type=int` on the
    # CLI path, but service_spec is a public function and the values below are
    # spliced straight into XML. A home directory containing `&` (legal on macOS
    # and Linux) is enough to emit a plist launchd cannot parse — and launchd's
    # only symptom for that is refusing to load, i.e. "the proxy just isn't
    # running", with no error anywhere the user would look.
    if not isinstance(port, int) or not (0 < port < 65536):
        raise ValueError(f"invalid port {port!r}: expected an integer in 1..65535")
    if not _AGENT_RE.fullmatch(mode or ""):
        raise ValueError(f"invalid mode {mode!r}: expected a bare flag name like 'lossless-only'")
    sysname = platform.system()
    if sysname == "Darwin":
        path = home / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
        # Where a crash goes. Without these two keys launchd discards the proxy's
        # stdout and stderr entirely: when the service died, `log show` had no record,
        # DiagnosticReports had no crash file, and the only symptom the user ever saw
        # was every session failing with what looked like a provider outage. A service
        # that is mandatory for the machine's API traffic must be able to say why it
        # stopped. ThrottleInterval is the companion guard — KeepAlive alone will
        # respawn a crash-on-boot proxy as fast as it can die, burning CPU and filling
        # the log with the same traceback; 10s makes a crash-loop legible instead.
        logdir = log_dir()
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>\n'
            "  <key>Label</key><string>com.distil.proxy</string>\n"
            "  <key>ProgramArguments</key><array>\n"
            f"    <string>{_xml(distil)}</string><string>proxy</string>"
            f"<string>--{_xml(mode)}</string><string>--port</string><string>{port}</string>\n"
            "  </array>\n"
            "  <key>RunAtLoad</key><true/>\n"
            "  <key>KeepAlive</key><true/>\n"
            # launchd creates and holds this socket, and hands the descriptor to
            # each launch of the job. That is what makes a crash survivable: the
            # listener belongs to launchd, so connections queue in the kernel
            # backlog while the proxy restarts instead of being refused. Without
            # it, KeepAlive still restarts the proxy — but every request in the
            # gap fails, and with a pinned ANTHROPIC_BASE_URL there is nothing to
            # fall back to. See distil/activation.py.
            "  <key>Sockets</key><dict>\n"
            "    <key>Listeners</key><dict>\n"
            "      <key>SockNodeName</key><string>127.0.0.1</string>\n"
            f"      <key>SockServiceName</key><string>{port}</string>\n"
            "      <key>SockType</key><string>stream</string>\n"
            "      <key>SockFamily</key><string>IPv4</string>\n"
            "    </dict>\n"
            "  </dict>\n"
            # 1s, not the 10s a crash-loop guard would want: with the socket held
            # by launchd a client's connection is already queued and waiting, so
            # every second here is latency the user feels on a live request.
            "  <key>ThrottleInterval</key><integer>1</integer>\n"
            # Raises jetsam priority. The proxy is a small, mostly-idle process,
            # which is exactly what macOS kills first under memory pressure — and
            # on a machine also running a VM and a browser, that happens often.
            "  <key>ProcessType</key><string>Interactive</string>\n"
            f"  <key>StandardOutPath</key><string>{_xml(str(logdir / 'proxy.log'))}</string>\n"
            f"  <key>StandardErrorPath</key><string>{_xml(str(logdir / 'proxy.err'))}</string>\n"
            "</dict></plist>\n"
        )
        # NOT a command any more. `cmd_default` calls `service_reload()`, which
        # waits for the job record to clear and verifies registration; this third
        # element survives only as the "this platform has a service" flag callers
        # test for truthiness. Returning the old `unload; load` string here left
        # the exact command that caused the outage sitting in a public API, one
        # `subprocess.run(service_spec(...)[2])` away from being reintroduced.
        return path, content, "service_reload"
    if sysname == "Linux":
        path = home / ".config" / "systemd" / "user" / "distil-proxy.service"
        # systemd already routes stdout/stderr to the journal, so unlike launchd there
        # is a record by default (`journalctl --user -u distil-proxy`). RestartSec is
        # the same crash-loop guard as ThrottleInterval above.
        #
        # Requires=/After= the socket unit is what makes the fault tolerance real
        # here rather than launchd-only: systemd creates the listening socket, hands
        # it down as LISTEN_FDS (see distil/activation.py), and holds it across a
        # restart of this service — so a crash queues connections instead of
        # refusing them. Without the socket unit the proxy self-binds and the gap
        # is exactly as wide as it always was.
        content = (
            "[Unit]\nDescription=distil compression proxy\nAfter=network-online.target\n"
            "Requires=distil-proxy.socket\nAfter=distil-proxy.socket\n\n"
            f"[Service]\nExecStart={distil} proxy --{mode} --port {port}\n"
            "Restart=always\nRestartSec=1\n\n"
            "[Install]\nWantedBy=default.target\n"
        )
        # See the Darwin branch: a marker, not a command. service_reload() owns the
        # ordering, which matters here — the old service must be stopped before the
        # socket unit can bind.
        return path, content, "service_reload"
    return (None, None, None)


def socket_unit_spec(port: int) -> tuple[Path | None, str | None]:
    """The systemd ``.socket`` unit that owns the listening socket, or (None, None).

    launchd expresses this inside the job's own plist (the ``Sockets`` key), but
    systemd needs a separate unit — which is why the first cut of this change
    shipped ``_from_systemd()`` and a README claiming socket activation on both
    platforms while Linux quietly still self-bound. The claim and the code have to
    match: without this unit nothing ever sets ``LISTEN_FDS``, and a Linux restart
    keeps the refused-connection gap the whole change exists to close.
    """
    import platform

    if platform.system() != "Linux":
        return (None, None)
    if not isinstance(port, int) or not (0 < port < 65536):
        raise ValueError(f"invalid port {port!r}: expected an integer in 1..65535")
    path = Path.home() / ".config" / "systemd" / "user" / "distil-proxy.socket"
    content = (
        "[Unit]\nDescription=distil compression proxy socket\n\n"
        f"[Socket]\nListenStream=127.0.0.1:{port}\n"
        # The service is what accepts; systemd only holds the listener.
        "Accept=no\n\n"
        "[Install]\nWantedBy=sockets.target\n"
    )
    return path, content


def probe_routing(host: str, port: int, *, deadline: float = 10.0) -> tuple[bool, str]:
    """Does the proxy at *host:port* actually SERVE ``/v1/messages``, or merely listen?

    A listening socket proves nothing. A proxy pointed at the wrong upstream accepts
    the connection and hands back that upstream's 404 for every request — and because
    Claude Code skips model-name validation whenever ``ANTHROPIC_BASE_URL`` is set
    (behind a gateway the provider defines the model names), the user sees "there's an
    issue with the selected model" in every session on the machine. The error names the
    wrong thing, so the search starts in the wrong place.

    Credentials are deliberately omitted: an HTTP 401 is *proof* the request reached a
    real messages handler, which is the only thing under test. Anything but a 404 means
    routing works. We retry on connection errors only — a 404 is a definitive answer,
    not a race — so a service that needs a second to bind still passes.
    """
    import time
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "model": "claude-sonnet-5",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "."}],
        }
    ).encode()
    url = f"http://{host}:{port}/v1/messages"
    last = "no response"
    end = time.monotonic() + deadline
    while True:
        req = urllib.request.Request(  # noqa: S310 — loopback only, built from our own port
            url,
            data=payload,
            headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:  # noqa: S310
                code, headers = resp.status, resp.headers
            break
        except urllib.error.HTTPError as exc:  # 401/400 are answers, and good ones
            code, headers = exc.code, exc.headers
            break
        except OSError as exc:  # refused / unbound / timed out — may still be starting
            last = str(exc)
            if time.monotonic() >= end:
                return False, f"no response from {host}:{port} — {last}"
            time.sleep(0.4)
    if code == 404:
        return False, (
            f"{host}:{port} is listening but answered 404 for /v1/messages — it is not "
            "routing (a wrong --upstream does exactly this)"
        )
    # Not required to pass: the headers ride on the compression path, and a proxy can be
    # correctly routing before it has anything to compress. Reported when present because
    # it upgrades "something answers" to "distil answers".
    stamped = any(str(k).lower().startswith("x-distil-") for k in (headers or {}))
    who = "distil" if stamped else "a proxy"
    return True, f"{who} on {host}:{port} routes /v1/messages (HTTP {code})"


def service_is_running(port: int) -> tuple[bool, str]:
    """Is the always-on service REGISTERED with the supervisor and running?

    Distinct from "something answers on the port", and the distinction is the
    whole bug this function exists for. ``launchctl unload`` is asynchronous: the
    old proxy can still be holding the port when the replacement job starts, so
    the job dies on EADDRINUSE and launchd discards it — while the orphaned old
    process keeps answering. A port probe passes. The machine is nonetheless left
    with *no registered job*, so when the orphan finally exits nothing restarts
    it and every session fails until a human re-runs the install. Observed in the
    field ~14 times in one day, each time with an empty proxy.err, because the
    replacement never got far enough to log anything.
    """
    import platform
    import subprocess

    sysname = platform.system()
    if sysname == "Darwin":
        try:
            out = subprocess.run(
                ["launchctl", "print", f"gui/{os.getuid()}/com.distil.proxy"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"could not query launchd: {exc}"
        if out.returncode != 0:
            return False, "no launchd job registered for com.distil.proxy"
        for line in out.stdout.splitlines():
            if line.strip().startswith("pid = "):
                return True, f"launchd job running (pid {line.strip().split('= ', 1)[1]})"
        return False, "launchd job is registered but has no running process"
    if sysname == "Linux":
        try:
            out = subprocess.run(
                ["systemctl", "--user", "is-active", "distil-proxy.service"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"could not query systemd: {exc}"
        state = out.stdout.strip() or "unknown"
        return state == "active", f"systemd unit is {state}"
    return False, f"no service supervisor known for {sysname}"


def _port_free(port: int, *, deadline: float = 8.0) -> bool:
    """Wait for *port* to stop accepting connections, up to *deadline* seconds.

    The missing step in the old reload. Bootstrapping while the previous process
    still holds the port is what makes the new job die on EADDRINUSE.
    """
    import socket as _socket
    import time

    # Ask by BINDING, not by connecting. A connect probe answers "is something
    # accepting", which is a different question and a misleading one: each probe
    # occupies a slot in the listener's backlog, so against a proxy that is hung
    # (listening but never calling accept) the second probe times out and the port
    # is reported free. We would then bootstrap onto a port that is still held —
    # EADDRINUSE, job discarded, exactly the failure this function exists to stop.
    # A bind attempt is precisely what launchd is about to do.
    end = time.monotonic() + deadline
    while True:
        with _socket.socket() as probe:
            probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return True
            except OSError:
                pass
        if time.monotonic() >= end:
            return False
        time.sleep(0.25)


def _run(cmd: list[str], *, timeout: float = 30.0):
    """Run a supervisor command, or return None if it could not be run at all.

    `service_is_running` already guarded its subprocess calls; `service_reload`
    did not, so a `launchctl`/`systemctl` that hung past its timeout or could not
    be executed surfaced as a raw traceback out of `distil default` — a stack
    trace where the user needed a sentence. Callers treat None as "could not talk
    to the supervisor" and say so.
    """
    import subprocess

    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def service_reload(port: int) -> tuple[bool, str]:
    """Stop, restart, and PROVE the always-on service came back.

    Replaces ``launchctl unload; launchctl load``, which reported the exit code
    of the legacy ``load`` — a value that is 0 even when the job failed to spawn.
    This uses the modern bootout/bootstrap API, waits for the port to actually be
    released between the two, and confirms a running process afterwards. It
    returns False rather than leaving the caller to infer success from a probe
    that a leftover process can satisfy.
    """
    import platform
    import time

    sysname = platform.system()
    if sysname == "Darwin":
        path = Path.home() / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
        domain = f"gui/{os.getuid()}"
        _run(["launchctl", "bootout", f"{domain}/com.distil.proxy"], timeout=20)
        # bootout is asynchronous in *two* ways, and both have to be waited out.
        # The job lingers in a `SIGTERMed` state after the call returns, and
        # bootstrapping into a domain that still holds a record for the label
        # fails with the famously unhelpful "Bootstrap failed: 5: Input/output
        # error". Waiting only for the port (the first version of this) was not
        # enough: the port frees the instant the process dies, while the job
        # record outlives it. Poll for the record to actually be gone.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if not service_is_running(port)[0]:
                probe = _run(["launchctl", "print", f"{domain}/com.distil.proxy"], timeout=10)
                if probe is None or probe.returncode != 0:  # no record at all — safe to bootstrap
                    break
            time.sleep(0.25)
        # Let the port settle, but do NOT abort on it. Since 1.42.0 the plist
        # declares Sockets, so launchd owns the listener and the SIGTERMed child
        # that inherited the descriptor can still hold it right here — a held
        # port is the design, not evidence of a foreign process. The Linux branch
        # below documents the same trap for the socket unit.
        #
        # The old fatal check fired on that ordinary case, and — this is the part
        # that made it an outage rather than an error — it fired AFTER bootout.
        # The job was already unregistered, so returning here left the machine
        # with no proxy and nothing to restart it, while cmd_default printed
        # "your existing setup is untouched". Observed on a maintainer's machine
        # switching modes with `distil default --always-on`.
        #
        # If the port really is stolen, bootstrap still runs and the
        # service_is_running() poll at the end reports it — with the job
        # REGISTERED, so KeepAlive keeps retrying instead of stranding the box.
        _port_free(port)
        # Even after the record clears, launchd can transiently refuse. Retrying a
        # bootstrap is safe (it is not partially applied) and turns a spurious EIO
        # into a non-event instead of a machine with no registered job.
        boot = None
        for attempt in range(8):
            boot = _run(["launchctl", "bootstrap", domain, str(path)], timeout=30)
            if boot is None:
                return False, "could not talk to launchd (launchctl bootstrap did not run)"
            if boot.returncode == 0:
                break
            if service_is_running(port)[0]:
                break  # someone else won the race; a running proxy is the goal
            time.sleep(1.0 if attempt < 4 else 2.0)
        if boot is not None and boot.returncode != 0 and not service_is_running(port)[0]:
            return False, (
                f"launchctl bootstrap failed after retries "
                f"({boot.stderr.strip() or boot.returncode}) — the service is NOT "
                f"registered, so nothing will restart it"
            )
    elif sysname == "Linux":
        _run(["systemctl", "--user", "daemon-reload"], timeout=30)
        # ORDER MATTERS, and getting it wrong breaks every upgrade.
        #
        # Before 1.42 the service self-bound the port; there was no socket unit. On
        # an upgrade that old service is still RUNNING and still owns 127.0.0.1:PORT,
        # so `enable --now distil-proxy.socket` cannot bind and fails — after
        # cmd_default has already overwritten the unit files. The user is left with
        # new units, a failure message, and the old proxy still holding the port.
        #
        # So: stop the service first (freeing the port), start the SOCKET (systemd
        # takes ownership of the listener), then start the service, which now
        # receives the descriptor instead of binding for itself.
        # BOTH units, and the socket especially. On an install that is already
        # socket-activated the socket unit holds the port BY DESIGN and keeps
        # holding it after the service stops — that is the whole feature. Stopping
        # only the service and then demanding a free port therefore fails on every
        # normal re-run of `distil default --always-on`, which is the common case,
        # while fixing the rarer upgrade-from-self-binding one. Stop both.
        stopped = _run(
            ["systemctl", "--user", "stop", "distil-proxy.socket", "distil-proxy.service"],
            timeout=60,
        )
        if stopped is None:
            return False, "could not talk to systemd (systemctl stop did not run)"
        if not _port_free(port):
            return False, (
                f"port {port} is still held after stopping distil-proxy.socket and .service — "
                f"something else is listening there. Find it with: "
                f"lsof -nP -iTCP:{port} -sTCP:LISTEN"
            )
        res = _run(["systemctl", "--user", "enable", "--now", "distil-proxy.socket"], timeout=60)
        if res is None:
            return False, "could not talk to systemd (systemctl enable --now socket)"
        if res.returncode != 0:
            return False, (
                f"could not enable distil-proxy.socket: {res.stderr.strip() or res.returncode}"
            )
        # The socket unit is what survives a restart of the service, so a machine
        # where it did not come up still has the gap this change exists to remove.
        sock = _run(["systemctl", "--user", "is-active", "distil-proxy.socket"], timeout=30)
        if sock is None or sock.stdout.strip() != "active":
            state = "unknown" if sock is None else (sock.stdout.strip() or "inactive")
            return False, (
                f"distil-proxy.socket is {state} — systemd is not holding the listener, "
                f"so a restart would refuse connections"
            )
        res = _run(["systemctl", "--user", "enable", "--now", "distil-proxy.service"], timeout=60)
        if res is None:
            return False, "could not talk to systemd (systemctl enable --now service)"
        if res.returncode != 0:
            return False, (
                f"distil-proxy.service did not start: {res.stderr.strip() or res.returncode}"
            )
    else:
        return False, f"no service supervisor known for {sysname}"

    # Registration is not instantaneous; poll rather than sleep-and-hope.
    end = time.monotonic() + 10.0
    detail = "service did not come up"
    while time.monotonic() < end:
        ok, detail = service_is_running(port)
        if ok:
            return True, detail
        time.sleep(0.25)
    # Both branches have registered/enabled the job by the time we get here, so
    # the supervisor keeps retrying it. A foreign process owning the port is the
    # likeliest reason it cannot come up — this is where that diagnosis belongs,
    # not in a precondition that aborts before anything has been started.
    if not _port_free(port, deadline=0.5):
        detail += (
            f" — and port {port} is held by another process. "
            f"Find it with: lsof -nP -iTCP:{port} -sTCP:LISTEN"
        )
    return False, detail


def service_unload_cmd() -> str | None:
    """Command to stop/unload the always-on proxy service on this platform, or None.

    Used by both ``distil default --undo`` and ``distil offboard`` so the running
    service is stopped — not just its definition file removed."""
    import platform

    sysname = platform.system()
    if sysname == "Darwin":
        path = Path.home() / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
        return f"launchctl unload '{path}' 2>/dev/null"
    if sysname == "Linux":
        return "systemctl --user disable --now distil-proxy.socket distil-proxy.service 2>/dev/null"
    return None

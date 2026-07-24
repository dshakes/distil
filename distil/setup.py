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
    write can't leave a half-written shell rc that breaks the next shell start."""
    tmp = path.with_name(path.name + ".distil.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


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
    """Strategy A — wrap the agent command on demand (fish/posix share `alias`)."""
    if not _AGENT_RE.fullmatch(agent or ""):
        raise ValueError(
            f"invalid agent name {agent!r}: expected only letters, digits, '.', '+', '-', '_' "
            "(it is spliced into a shell alias, so arbitrary strings are refused)"
        )
    sh = shell if shell is not None else ("powershell" if _is_windows() else "")
    if sh == "powershell":
        return f"function {agent} {{ distil wrap --{mode} -- {agent} @args }}"
    return f"alias {agent}='distil wrap --{mode} -- {agent}'"


def env_body(port: int, *, shell: str | None = None) -> str:
    """Strategy B — point every SDK at the always-on proxy (shell-specific syntax)."""
    sh = shell if shell is not None else ("powershell" if _is_windows() else "")
    if sh == "powershell":
        return f'$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:{port}"'
    if sh == "fish":
        return f"set -gx ANTHROPIC_BASE_URL http://127.0.0.1:{port}"
    return f"export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}"


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
    rc.with_name(rc.name + ".bak").write_text(text, encoding="utf-8")
    _atomic_write(rc, new)
    return ("ok", f"removed the distil default from {rc}")


def service_spec(port: int, mode: str) -> tuple[Path | None, str | None, str | None]:
    """Always-on proxy service for this platform: (path, file_content, load_command).

    Returns (None, None, None) on an unsupported platform."""
    import platform
    import shutil

    home = Path.home()
    distil = shutil.which("distil") or "distil"
    sysname = platform.system()
    if sysname == "Darwin":
        path = home / "Library" / "LaunchAgents" / "com.distil.proxy.plist"
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>\n'
            "  <key>Label</key><string>com.distil.proxy</string>\n"
            "  <key>ProgramArguments</key><array>\n"
            f"    <string>{distil}</string><string>proxy</string>"
            f"<string>--{mode}</string><string>--port</string><string>{port}</string>\n"
            "  </array>\n"
            "  <key>RunAtLoad</key><true/>\n"
            "  <key>KeepAlive</key><true/>\n"
            "</dict></plist>\n"
        )
        load = f"launchctl unload '{path}' 2>/dev/null; launchctl load '{path}'"
        return path, content, load
    if sysname == "Linux":
        path = home / ".config" / "systemd" / "user" / "distil-proxy.service"
        content = (
            "[Unit]\nDescription=distil compression proxy\nAfter=network-online.target\n\n"
            f"[Service]\nExecStart={distil} proxy --{mode} --port {port}\nRestart=always\n\n"
            "[Install]\nWantedBy=default.target\n"
        )
        load = (
            "systemctl --user daemon-reload && systemctl --user enable --now distil-proxy.service"
        )
        return path, content, load
    return (None, None, None)


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
        return "systemctl --user disable --now distil-proxy.service 2>/dev/null"
    return None

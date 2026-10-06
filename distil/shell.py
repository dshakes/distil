"""``distil sh`` — shape a shell command's output at the source (ADR 0026).

The proxy digest elides a tool result *after* the agent has read it; the agent then
re-runs the command to see what was elided. This module works one step earlier: it runs
the command, applies one small deterministic filter keyed on what the command is, and
prints the result, so the compact form is the only form that ever enters the context.

Two tiers, chosen by the same billing policy as the post-tool hook
(``hook.tier_decision``):

* **lossless** — always: ANSI escapes stripped and carriage-return overwrites resolved
  (the visible text is unchanged), exact repeated lines collapsed to ``<<xN>>``, and
  ``git status`` advice lines — fixed text git prints about how to use git, never about
  the repository — dropped.
* **elide** — a metered key, or an explicit ``--digest`` opt-in on a subscription: the
  lines a test runner prints for each *passing* test are dropped, and the full output is
  kept behind a ``distil expand`` handle printed in the compact output. A line the
  shared keep policy marks as an error, failure, warning or result summary is never
  dropped, whatever the filter thinks.

Fail-open: ``DISTIL_SH_OFF=1``, an unparseable argv, or any error while shaping prints
the command's own output verbatim. The exit code is always the command's own.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any

#: Stamped into every receipt and marker, so a shadow or A/B comparison can attribute a
#: change in behaviour to a change in the filter set. Bump on ANY change to what a
#: filter keeps or drops.
FILTERS_VERSION = "sh-v1"

# ----------------------------------------------------------------------------------
# Which commands have a filter
# ----------------------------------------------------------------------------------

_PY = re.compile(r"^(?:.*/)?python(?:\d+(?:\.\d+)?)?$")
_ASSIGN = re.compile(r"^[A-Za-z_]\w*=")


def classify(argv: list[str]) -> str | None:
    """The filter for *argv* (leading ``VAR=value`` assignments skipped), or None.

    Search and listing commands (``grep``, ``rg``, ``find``, ``ls``, ``tree``) are
    deliberately absent: their output is the agent's map of the code, and 1.56.3 measured
    that eliding it costs extra steps (ADR 0026).
    """
    i = 0
    while i < len(argv) and _ASSIGN.match(argv[i]):
        i += 1
    a = argv[i:]
    if not a:
        return None
    head = os.path.basename(a[0])
    if a[:2] == ["git", "status"]:
        return "git-status"
    if head in ("pytest", "py.test") or (_PY.match(a[0]) and a[1:3] == ["-m", "pytest"]):
        return "pytest"
    if _PY.match(a[0]) and a[1:3] == ["-m", "unittest"]:
        return "unittest"
    if head == "runtests.py" or (_PY.match(a[0]) and a[1:2] and a[1].endswith("runtests.py")):
        return "unittest"  # Django's runner prints unittest's `... ok` lines
    if a[:2] == ["cargo", "test"]:
        return "cargo-test"
    if a[:2] == ["go", "test"]:
        return "go-test"
    if head in ("npm", "yarn", "pnpm") and (a[1:2] == ["test"] or a[1:3] == ["run", "test"]):
        return "js-test"
    if head in ("jest", "vitest") or (head == "npx" and a[1:2] in (["jest"], ["vitest"])):
        return "js-test"
    return None


# One simple command, optionally after `cd DIR &&`, optionally followed by `2>&1` and a
# pipe into head/tail. Anything else (`;`, `&&` chains, redirects, `$(…)`, backticks,
# quoted metacharacters) is left alone: the rewrite only ever PREFIXES the command, so
# the shell parses exactly what the agent wrote.
_SHAPE = re.compile(
    r"^(?P<cd>\s*cd\s+[^;&|<>`$()\n]+?\s*&&\s*)?"
    r"(?P<body>[^;&|<>`\n]+?)"
    r"(?P<redir>\s+2>&1)?"
    r"(?P<pipe>\s*\|\s*(?:tail|head)(?:\s+[-+\w]+)*)?\s*$"
)


def plan(command: str) -> tuple[str, str, str, str] | None:
    """``(prefix, body, suffix, filter)`` when *command* can be shaped, else None."""
    if "distil sh" in command or "$(" in command:
        return None
    m = _SHAPE.match(command)
    if not m:
        return None
    try:
        argv = shlex.split(m["body"])
    except ValueError:
        return None
    kind = classify(argv)
    if kind is None:
        return None
    return m["cd"] or "", m["body"].strip(), (m["redir"] or "") + (m["pipe"] or ""), kind


def rewrite(command: str, *, digest: bool = False) -> str | None:
    """*command* with ``distil sh --`` inserted before the shaped stage, or None."""
    p = plan(command)
    if p is None:
        return None
    prefix, body, suffix, _ = p
    return f"{prefix}distil sh {'--digest ' if digest else ''}-- {body}{suffix}"


# ----------------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def visible(text: str) -> str:
    """What a terminal would show: ANSI escapes removed, ``\\r`` overwrites resolved."""
    text = _ANSI.sub("", text).replace("\r\n", "\n")
    if "\r" not in text:
        return text
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))


# git's advice text, e.g. `  (use "git add <file>..." to update what will be committed)`.
_GIT_ADVICE = re.compile(r'^\s*\((?:use "git |commit or discard |fix conflicts and run ")[^\n]*\)$')

#: Lines a runner prints for a test that PASSED, per filter. A match is only a
#: candidate: `_protected` vetoes it.
_PASS: dict[str, re.Pattern[str]] = {
    "pytest": re.compile(
        r"^\S.*::\S.* PASSED(?: +\[ *\d+%\])?$"  # -v
        r"|^\S+\.py \.+ *(?:\[ *\d+%\])?$"  # default: a file whose tests all passed
    ),
    "unittest": re.compile(r"^\S.* \.\.\. ok$|^\.+$"),
    "cargo-test": re.compile(r"^test \S+ \.\.\. ok$"),
    "go-test": re.compile(r"^=== (?:RUN|PAUSE|CONT|NAME) |^ *--- PASS: "),
    "js-test": re.compile(r"^ *[✓✔√] "),
}

_MIN_ELIDED = 5


_COUNTS = re.compile(r"\b\d+ +(?:passed|passing|failed|failing|skipped|errors?|errored)\b", re.I)


def _protected(line: str) -> bool:
    """Never dropped: anything the shared keep policy's generic net pins (errors,
    exceptions, failures, warnings, panics, URLs), any FAIL/ERROR token, and any count
    summary. (The LOG kind's per-test ``--- PASS:`` pin is deliberately not inherited:
    a passing test's verdict line is exactly what the elide tier removes.)"""
    from .compress.keep_policy import ContentKind, must_keep

    return (
        must_keep(line, ContentKind.GENERIC)
        or "FAIL" in line
        or "ERROR" in line
        or _COUNTS.search(line) is not None
    )


def lossless(text: str, kind: str) -> str:
    from .compress.tier0 import collapse_runs

    text = visible(text)
    if kind == "git-status":
        text = "\n".join(ln for ln in text.split("\n") if not _GIT_ADVICE.match(ln))
    return collapse_runs(text)


def elide(text: str, kind: str) -> tuple[str, int]:
    """Drop passing-test lines. ``(text, lines dropped)``."""
    pat = _PASS.get(kind)
    if pat is None:
        return text, 0
    kept: list[str] = []
    dropped = 0
    for ln in text.split("\n"):
        if pat.match(ln) and not _protected(ln):
            dropped += 1
        else:
            kept.append(ln)
    return "\n".join(kept), dropped


def shape(
    raw: str, kind: str, *, lossy: bool, save: Callable[[str], str | None]
) -> tuple[str, str]:
    """``(output, tier_applied)``. *save* stores the raw output and returns a handle (or
    None when it could not be stored — then nothing lossy is emitted)."""
    out = lossless(raw, kind)
    tier = "lossless"
    if lossy:
        cut, n = elide(out, kind)
        if n >= _MIN_ELIDED:
            h = save(raw)
            if h is not None:
                end = "" if cut.endswith("\n") else "\n"
                out = (
                    f"{cut}{end}[distil sh {FILTERS_VERSION}: {n} passing-test lines elided; "
                    f"failures, errors and the summary are verbatim above. "
                    f"Full output: `distil expand {h}`]\n"
                )
                tier = "elide"
    if len(out) >= len(raw):
        return raw, "none"
    return out, tier


def _save(raw: str) -> str | None:
    from .compress.tier1 import _handle
    from .hook import _persist

    h = _handle(raw)
    return h if _persist(h, raw) else None


# ----------------------------------------------------------------------------------
# `distil sh -- <command>`
# ----------------------------------------------------------------------------------


def _exec_untouched(argv: list[str]) -> int:
    """Fail-open: become the original command, so nothing distil does can change it."""
    if sys.platform == "win32":
        # Windows has no exec: os.execvp spawns and exits 0, hiding the command's code.
        try:
            return subprocess.run(argv).returncode
        except OSError as exc:
            sys.stderr.write(f"distil sh: {argv[0]}: {exc.strerror or exc}\n")
            return 127
    try:
        os.execvp(argv[0], argv)
    except OSError as exc:
        sys.stderr.write(f"distil sh: {argv[0]}: {exc.strerror or exc}\n")
        return 127
    return 0  # pragma: no cover - execvp does not return


def main(argv: list[str], *, digest: bool = False) -> int:
    """Run *argv*, print its shaped output, exit with its exit code."""
    if not argv:
        sys.stderr.write("usage: distil sh [--digest] -- <command> [args...]\n")
        return 2
    kind = classify(argv)
    if os.environ.get("DISTIL_SH_OFF") == "1" or kind is None:
        return _exec_untouched(argv)
    env = dict(os.environ)
    while argv and _ASSIGN.match(argv[0]):
        k, _, v = argv.pop(0).partition("=")
        env[k] = v
    if not argv:
        return 0
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except OSError:
        return _exec_untouched(argv)

    def _forward(signum: int, _frame: Any) -> None:  # a timeout kill must reach the child
        proc.send_signal(signum)

    prev = signal.signal(signal.SIGTERM, _forward)
    try:
        data, _ = proc.communicate()
    finally:
        signal.signal(signal.SIGTERM, prev)
    raw = data.decode("utf-8", errors="replace")
    out, tier = raw, "none"
    try:
        from .hook import tier_decision

        lossy = tier_decision("digest" if digest else "auto")[0]
        out, tier = shape(raw, kind, lossy=lossy, save=_save)
        if tier != "none":
            from .hook import record_receipt

            record_receipt(
                f"sh:{kind}", len(raw), len(out), client="sh", filters=FILTERS_VERSION, tier=tier
            )
    except Exception:
        out = raw  # shaping is optional; the command's own output is not
    sys.stdout.write(out)
    sys.stdout.flush()
    return proc.returncode if proc.returncode >= 0 else 128 - proc.returncode  # signal -N -> 128+N


# ----------------------------------------------------------------------------------
# Claude Code PreToolUse: rewrite a Bash command to run under `distil sh`
# ----------------------------------------------------------------------------------


def _rules(cwd: str | None) -> dict[str, list[str]]:
    """``allow``/``ask``/``deny`` Bash rules from every settings file Claude Code merges."""
    import json
    from pathlib import Path

    from .hook import config_path
    from .setup import claude_settings_files

    out: dict[str, list[str]] = {"allow": [], "ask": [], "deny": []}
    files = claude_settings_files(Path(cwd) if cwd else None) + [config_path("claude")]
    for f in dict.fromkeys(files):
        try:
            perms = json.loads(f.read_text(encoding="utf-8")).get("permissions") or {}
        except (OSError, ValueError, AttributeError):
            continue
        if not isinstance(perms, dict):
            continue
        for key in out:
            for r in perms.get(key) or []:
                if isinstance(r, str) and (r == "Bash" or r.startswith("Bash(")):
                    out[key].append(r[5:-1] if r.endswith(")") else "")
    return out


def _allows(pattern: str, cmd: str) -> bool:
    """Exact match or ``prefix:*`` / ``prefix *`` — the forms we can match the way Claude
    Code does. Any other wildcard is not matched (the call then prompts as usual)."""
    if not pattern:  # bare `Bash`: every command
        return True
    for tail in (":*", " *"):
        if pattern.endswith(tail):
            stem = pattern[: -len(tail)]
            return "*" not in stem and (cmd == stem or cmd.startswith(stem + " "))
    return "*" not in pattern and cmd == pattern


def _touches(pattern: str, cmd: str) -> bool:
    """Could a deny/ask rule apply to *cmd*? Over-matches on purpose: a hit only means
    distil leaves the command alone and Claude Code's own rule decides."""
    stem = re.split(r"[*:]", pattern, maxsplit=1)[0].strip()
    return not stem or cmd.startswith(stem) or stem.startswith(cmd)


def pre_tool_use(event: dict[str, Any], *, digest: bool = False) -> dict[str, Any] | None:
    """The PreToolUse envelope rewriting a Bash command, or None to leave it alone.

    Claude Code evaluates permission rules against the REWRITTEN input, so the rewrite
    must not widen what the user allowed: a command that any deny/ask rule could touch
    is never rewritten, and ``allow`` is emitted only when one of the user's own allow
    rules matches the ORIGINAL command. Otherwise no decision is returned and the
    normal permission flow runs on the rewritten command.
    """
    import shutil

    if event.get("tool_name") != "Bash" or os.environ.get("DISTIL_SH_OFF") == "1":
        return None
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("command"), str):
        return None
    cmd = tool_input["command"]
    new = rewrite(cmd, digest=digest)
    if new is None or shutil.which("distil") is None:
        return None
    rules = _rules(event.get("cwd") if isinstance(event.get("cwd"), str) else None)
    stage = cmd.strip()
    p = plan(cmd)
    assert p is not None  # rewrite() returned a command, so plan() matched
    prefix, body, suffix, _ = p
    if any(_touches(r, c) for r in rules["deny"] + rules["ask"] for c in (stage, body)):
        return None
    out: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "updatedInput": {**tool_input, "command": new},
    }
    # A `cd` or a pipe is a second subcommand Claude Code would check on its own; then
    # no decision is mirrored and its normal flow judges the rewritten command.
    single = not prefix and "|" not in suffix
    if single and any(_allows(r, body) for r in rules["allow"]):
        out["permissionDecision"] = "allow"
        out["permissionDecisionReason"] = "distil sh: an allow rule matches the original command"
    return {"hookSpecificOutput": out}

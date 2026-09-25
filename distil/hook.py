"""Post-tool hooks — compress tool output before it enters an agent's history.

Why this exists
---------------
A proxy reaches every agent that lets you set a base URL. Some don't, and on a
flat-rate subscription distil keeps the proxy lossless-only anyway. A post-tool
hook is a different mechanism: a documented, first-party extension point where the
client hands us a tool result and uses what we hand back instead. No proxy, no
OAuth interception, no credential bridging.

What it does
------------
Tier-0 (JSON minification, exact-run collapse) always; then, for verbose
line-oriented output, the same Tier-1 decision-aware digest the proxy uses. A
digest is only emitted after the full original is persisted in the RestoreStore
and read back byte-exact, so every elided line is recoverable with
``distil expand <handle>`` (shell) or the ``distil_expand`` MCP tool. Reject-if-
bigger by tokens. Results the agent must quote byte-exact are never touched: the
file-read tool names in ``compress.provenance`` and shell whole-file reads
(``cat app.py``, ``sed -n 1,80p app.py``) — an Edit lifted from a digest would not
apply.

Append-only by construction: a hook sees one result, once, when it is produced, and
cannot rewrite history, so it cannot bust the prompt cache the way a sliding
recency window can.

Per-client contract (each verified against the client's own docs; dates in
``CLIENTS``)
----------------------------------------------------------------------------
* **Claude Code** ``PostToolUse`` → ``hookSpecificOutput.updatedToolOutput``. The
  replacement must match the tool's output shape or it is ignored *silently*, so
  unknown keys are carried through and ``--selftest`` proves the adapters offline.
  A failed command (non-empty stderr, interrupted) is never touched.
* **Cursor** ``postToolUse`` → ``updated_mcp_tool_output``, which the docs scope to
  **MCP tools only**. Shell output cannot be replaced (``afterShellExecution`` is
  observe-only), so Cursor gets MCP compression and nothing else.
* **Gemini CLI** ``AfterTool`` → ``decision: "deny"`` + ``reason``: "This text
  replaces the tool result sent back to the model." A result carrying ``error`` is
  never touched.
* **Codex CLI** ``PostToolUse`` → ``decision: "block"`` + ``reason``: Codex
  "replaces the tool result with that feedback". (``updatedMCPToolOutput`` is
  documented as parsed-but-unsupported and marks the hook failed, so it is never
  emitted.) Codex asks you to trust a new hook in ``/hooks`` before it runs.

Gemini and Codex deliver the replacement through their block/deny channel, so it
opens with one line saying the output was compacted, not refused.

Fail-safe posture: any exception, unknown tool, unreadable payload, or transform that
does not reduce tokens emits ``{}``, which leaves the original untouched. Doing
nothing is always a correct outcome here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Below this, compression cannot save enough to be worth any risk. Tool results are
# long-tailed: the savings live in test logs, file dumps and tracebacks, not in the
# one-line results that dominate by count.
_MIN_CHARS = 2048
# Same floor as Tier1Reversible: shorter text has no droppable middle.
_MIN_DIGEST_LINES = 6

#: First line of a replacement delivered through a block/deny channel (Gemini, Codex),
#: so the model reads it as output rather than as a refusal.
_BLOCK_NOTE = "[distil: tool output compacted, not an error]\n"


def _tier0(text: str) -> str:
    """Lossless Tier-0: JSON minification, then run collapse, reject-if-bigger.

    Mirrors ``distil.adapters.anthropic._apply_tier0`` — the same transforms the
    subscription-safe proxy path uses, so the hook and the proxy cannot disagree
    about what "lossless" means. Run-collapse is rejected when it does not reduce
    *tokens*, because collapsing near-free whitespace into a ``<<xN>>`` marker can
    cost more than it saves.
    """
    from .compress.tier0 import collapse_runs, minify_json
    from .tokenizer import resolve

    tok = resolve("heuristic")
    mj = minify_json(text)
    base = mj if mj is not None else text
    collapsed = collapse_runs(base)
    if collapsed != base and tok.count(collapsed) <= tok.count(base):
        return collapsed
    return base


def _persist(handle: str, original: str) -> bool:
    """Record *original* in the RestoreStore and prove it reads back byte-exact.

    ``record_restore`` is best-effort (a failed write still returns True, because the
    proxy also keeps an in-memory copy). A hook process has no memory that outlives
    it, so the disk copy is the ONLY recovery path — verify it before emitting a stub.
    """
    from .mcp_server import load_restore, record_restore

    return record_restore(handle, original) and load_restore(handle) == original


def _compress_plain(text: str) -> str:
    """Tier-0, then the Tier-1 digest when it is smaller and recoverable."""
    from .compress.tier0 import minify_json
    from .tokenizer import resolve

    best = _tier0(text)
    # JSON is structured: its lossless minified form stays parseable, a line digest
    # of it would not. The proxy routes it to the structured folds for the same reason.
    if minify_json(text) is not None or text.count("\n") + 1 < _MIN_DIGEST_LINES:
        return best
    from .compress.tier1 import _handle, digest

    digested, changed = digest(text)
    if not changed:
        return best
    h = _handle(text)
    stub = f"{digested}\n<< full output: `distil expand {h}` >>"
    tok = resolve("heuristic")
    if tok.count(stub) < tok.count(best) and _persist(h, text):
        return stub
    return best


def compress_text(text: str) -> str | None:
    """A smaller replacement for one tool-output string, or None to leave it alone.

    ``<distil:keep>…</distil:keep>`` spans pass through byte-exact (see
    ``compress.keeptags``). Reject-if-bigger by tokens.
    """
    if not isinstance(text, str) or len(text) < _MIN_CHARS:
        return None
    from .compress.keeptags import apply as _keep_apply
    from .tokenizer import resolve

    out = _keep_apply(text, _compress_plain)
    if out == text:
        return None
    tok = resolve("heuristic")
    return out if tok.count(out) <= tok.count(text) else None


def is_exact_quote(tool_name: str, tool_input: Any = None) -> bool:
    """Must this result stay byte-exact? Reuses the proxy's provenance rule.

    Name-keyed readers (``Read``, ``read_file``, ``grep``, …) and ``distil_expand``
    itself, matched as a bare name or as the tail of a namespaced MCP name
    (``mcp__fs__read_file``, ``mcp_fs_read_file``, ``MCP:read_file``), plus shell
    whole-file reads decided from the command line.
    """
    from .compress.provenance import (
        EXACT_QUOTE_TOOLS,
        EXPAND_TOOL_NAME,
        command_text,
        whole_file_read_paths,
    )

    name = tool_name.lower()
    if name.startswith("mcp:"):
        name = name[4:]
    for exact in (*EXACT_QUOTE_TOOLS, EXPAND_TOOL_NAME):
        if name == exact or name.endswith("_" + exact):
            return True
    return bool(whole_file_read_paths(command_text(tool_input)))


def _compress_bash(out: dict[str, Any]) -> dict[str, Any] | None:
    """Compress a Claude Code ``Bash`` result: ``{stdout, stderr, interrupted, isImage}``.

    Only ``stdout`` is touched, and only on a clean run. A non-empty ``stderr`` or an
    interrupt means something went wrong and every byte may matter to the diagnosis.
    """
    if out.get("interrupted") or out.get("isImage"):
        return None
    if out.get("stderr"):
        return None
    stdout = out.get("stdout")
    if not isinstance(stdout, str):
        return None
    shrunk = compress_text(stdout)
    if shrunk is None:
        return None
    # Preserve every key we were given; replace exactly one value. Reconstructing the
    # dict from known keys would silently drop fields added by future versions.
    return {**out, "stdout": shrunk}


def _compress_mcp(out: Any) -> Any | None:
    """Compress MCP tool output: a bare string or the common content-block list.

    Anything else is left alone. An error result is never touched.
    """
    if isinstance(out, str):
        return compress_text(out)

    if not isinstance(out, dict) or out.get("isError"):
        return None
    blocks = out.get("content")
    if not isinstance(blocks, list):
        return None

    new_blocks, changed = [], False
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
            shrunk = compress_text(b["text"])
            if shrunk is not None:
                new_blocks.append({**b, "text": shrunk})
                changed = True
                continue
        new_blocks.append(b)  # non-text blocks (images, resources) pass through
    return {**out, "content": new_blocks} if changed else None


def compress_tool_output(tool_name: str, tool_output: Any, tool_input: Any = None) -> Any | None:
    """Claude Code: a compressed replacement, or None to leave the original alone.

    Pure and importable so the behaviour can be tested without a live session —
    which matters because the failure mode in production is *silence*.
    """
    if is_exact_quote(tool_name, tool_input):
        return None
    if tool_name == "Bash":
        return _compress_bash(tool_output) if isinstance(tool_output, dict) else None
    if tool_name.startswith("mcp__"):
        return _compress_mcp(tool_output)
    # Read/Grep/Glob and other built-ins have undocumented shapes. Guessing costs
    # nothing when wrong (the original is used) but yields no savings either, so we
    # decline until each shape is probed live.
    return None


#: Off during ``--selftest``: a verification run must not book synthetic savings
#: into the user's real receipts.
_RECEIPTS = True


def _receipt_path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "hook-receipts.jsonl"


def record_receipt(
    tool_name: str, before: int, after: int, *, path: Path | None = None, client: str = "claude"
) -> None:
    """Append one content-free line: what this hook actually did.

    Numbers only: client, tool name, character counts before/after, timestamp. Never
    content. Best-effort and silent on failure: bookkeeping must never break a tool
    result.
    """
    if not _RECEIPTS and path is None:
        return
    try:
        p = path or _receipt_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "client": client,
                        "tool": tool_name,
                        "chars_before": before,
                        "chars_after": after,
                        "chars_saved": max(0, before - after),
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
    except (OSError, ValueError):
        # OSError: unwritable path. ValueError: an embedded NUL in DISTIL_HOME, which
        # raises before any I/O. Losing a receipt is cheap, losing the compression it
        # describes is not.
        pass


def _measure(value: Any) -> int:
    """Character size of a tool payload, whatever shape it takes."""
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------------
# Per-client adapters: event dict -> (tool, before, replacement, stdout JSON) or None
# --------------------------------------------------------------------------------

#: Cursor's built-in tool types (docs: "Shell, Read, Write, Grep, Delete, Task").
#: ``updated_mcp_tool_output`` applies to MCP tools only, so these are never touched.
_CURSOR_BUILTINS = frozenset({"Shell", "Read", "Write", "Grep", "Delete", "Task"})


def _claude(event: dict[str, Any]) -> tuple[str, Any, Any, dict[str, Any]] | None:
    tool_name = event.get("tool_name")
    tool_output = event.get("tool_response", event.get("tool_output"))
    if not isinstance(tool_name, str) or tool_output is None:
        return None
    replacement = compress_tool_output(tool_name, tool_output, event.get("tool_input"))
    if replacement is None:
        return None
    envelope = {
        "hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": replacement}
    }
    return tool_name, tool_output, replacement, envelope


def _cursor(event: dict[str, Any]) -> tuple[str, Any, Any, dict[str, Any]] | None:
    tool_name = event.get("tool_name")
    raw = event.get("tool_output")
    if not isinstance(tool_name, str) or tool_name in _CURSOR_BUILTINS:
        return None
    if is_exact_quote(tool_name, event.get("tool_input")):
        return None
    # Documented as a JSON-stringified result payload, not raw text.
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    replacement = _compress_mcp(payload)
    if not isinstance(replacement, dict):  # the field is documented as an object
        return None
    return tool_name, payload, replacement, {"updated_mcp_tool_output": replacement}


def _gemini(event: dict[str, Any]) -> tuple[str, Any, Any, dict[str, Any]] | None:
    tool_name = event.get("tool_name")
    resp = event.get("tool_response")
    if not isinstance(tool_name, str) or not isinstance(resp, dict) or resp.get("error"):
        return None
    if tool_name != "run_shell_command" and not tool_name.startswith("mcp_"):
        return None
    if is_exact_quote(tool_name, event.get("tool_input")):
        return None
    text = resp.get("llmContent")
    if not isinstance(text, str):
        return None
    shrunk = compress_text(text)
    if shrunk is None:
        return None
    reason = _BLOCK_NOTE + shrunk
    return tool_name, text, reason, {"decision": "deny", "reason": reason}


def _codex_text(tool_name: str, resp: Any) -> str | None:
    if tool_name == "Bash":
        return resp if isinstance(resp, str) else None
    if not tool_name.startswith("mcp__") or not isinstance(resp, dict) or resp.get("isError"):
        return None
    blocks = resp.get("content")
    if not isinstance(blocks, list) or not blocks:
        return None
    # The replacement is one string, so only an all-text result can be carried over.
    if not all(isinstance(b, dict) and b.get("type") == "text" for b in blocks):
        return None
    texts = [b.get("text") for b in blocks]
    return "\n".join(texts) if all(isinstance(t, str) for t in texts) else None  # type: ignore[arg-type]


def _codex(event: dict[str, Any]) -> tuple[str, Any, Any, dict[str, Any]] | None:
    tool_name = event.get("tool_name")
    if not isinstance(tool_name, str) or is_exact_quote(tool_name, event.get("tool_input")):
        return None
    text = _codex_text(tool_name, event.get("tool_response"))
    if text is None:
        return None
    shrunk = compress_text(text)
    if shrunk is None:
        return None
    reason = _BLOCK_NOTE + shrunk
    return tool_name, text, reason, {"decision": "block", "reason": reason}


_ADAPTERS = {"claude": _claude, "cursor": _cursor, "gemini": _gemini, "codex": _codex}


def run(stdin_text: str, client: str = "claude") -> str:
    """Map one hook invocation to its stdout. Never raises."""
    try:
        event = json.loads(stdin_text)
        if not isinstance(event, dict):
            return "{}"
        result = _ADAPTERS[client](event)
        if result is None:
            return "{}"
        tool_name, before, after, envelope = result
        record_receipt(tool_name, _measure(before), _measure(after), client=client)
        return json.dumps(envelope)
    except Exception:
        # A hook that crashes must not break the user's session, and a hook that
        # emits garbage must not corrupt a tool result. Both resolve to "do nothing".
        return "{}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return _selftest()
    client = "claude"
    if "--client" in argv:
        i = argv.index("--client")
        client = argv[i + 1] if i + 1 < len(argv) else ""
    if client not in _ADAPTERS:
        sys.stdout.write("{}")  # an unknown client must still be a no-op, never an error
        return 0
    sys.stdout.write(run(sys.stdin.read(), client))
    return 0


def _selftest() -> int:
    """Prove the adapters offline, since a live mismatch is silent.

    Checks the properties that actually matter: we shrink what we should, we emit the
    exact documented envelope per client, and we refuse everything that carries risk.
    """
    global _RECEIPTS
    _RECEIPTS = False
    try:
        return _selftest_checks()
    finally:
        _RECEIPTS = True


def _selftest_checks() -> int:
    big_log = "ERROR connection refused\n" * 400
    checks: list[tuple[str, bool]] = []

    # Bash: compresses clean large stdout, preserving the full shape.
    out = compress_tool_output(
        "Bash", {"stdout": big_log, "stderr": "", "interrupted": False, "isImage": False}
    )
    checks.append(("bash compresses large clean stdout", out is not None))
    if out is not None:
        checks.append(("bash actually shrank", len(out["stdout"]) < len(big_log)))

    # Live Claude Code sends a FIFTH field the docs' four-field example omits:
    # `noOutputExpected`. Verified 2026-08-16 against a real session, where the
    # replacement was accepted only because unknown keys were carried through.
    live_shape = {
        "stdout": big_log,
        "stderr": "",
        "interrupted": False,
        "isImage": False,
        "noOutputExpected": False,
    }
    live_out = compress_tool_output("Bash", live_shape)
    checks.append(("bash handles undocumented live shape", live_out is not None))
    if live_out is not None:
        checks.append(("bash preserves unknown keys", set(live_out) == set(live_shape)))
        checks.append(("bash preserves unknown values", live_out["noOutputExpected"] is False))

    # Refusals — each of these must return None.
    refusals = [
        (
            "stderr present",
            {"stdout": big_log, "stderr": "boom", "interrupted": False, "isImage": False},
        ),
        ("interrupted", {"stdout": big_log, "stderr": "", "interrupted": True, "isImage": False}),
        ("isImage", {"stdout": big_log, "stderr": "", "interrupted": False, "isImage": True}),
        (
            "below size floor",
            {"stdout": "tiny", "stderr": "", "interrupted": False, "isImage": False},
        ),
    ]
    for label, payload in refusals:
        checks.append((f"bash refuses: {label}", compress_tool_output("Bash", payload) is None))
    checks.append(
        (
            "bash refuses an exact-quote read (cat)",
            compress_tool_output(
                "Bash",
                {"stdout": big_log, "stderr": "", "interrupted": False},
                {"command": "cat app.log"},
            )
            is None,
        )
    )

    checks.append(("unknown tool refused", compress_tool_output("Read", {"file": "x"}) is None))
    checks.append(
        (
            "mcp error result refused",
            compress_tool_output(
                "mcp__x__y", {"isError": True, "content": [{"type": "text", "text": big_log}]}
            )
            is None,
        )
    )
    mcp = compress_tool_output("mcp__x__y", {"content": [{"type": "text", "text": big_log}]})
    checks.append(("mcp text block compressed", mcp is not None))

    # Envelopes: the exact documented field names, or the whole thing is a silent no-op.
    bash_event = {
        "tool_name": "Bash",
        "tool_response": {"stdout": big_log, "stderr": "", "interrupted": False, "isImage": False},
    }
    hso = json.loads(run(json.dumps(bash_event))).get("hookSpecificOutput", {})
    checks.append(("claude envelope hookEventName", hso.get("hookEventName") == "PostToolUse"))
    checks.append(("claude envelope updatedToolOutput", "updatedToolOutput" in hso))

    mcp_payload = {"content": [{"type": "text", "text": big_log}]}
    cur = json.loads(
        run(
            json.dumps({"tool_name": "MCP:query", "tool_output": json.dumps(mcp_payload)}),
            "cursor",
        )
    )
    checks.append(("cursor envelope updated_mcp_tool_output", "updated_mcp_tool_output" in cur))
    checks.append(
        (
            "cursor refuses Shell (not replaceable)",
            run(json.dumps({"tool_name": "Shell", "tool_output": big_log}), "cursor") == "{}",
        )
    )
    gem = json.loads(
        run(
            json.dumps(
                {"tool_name": "run_shell_command", "tool_response": {"llmContent": big_log}}
            ),
            "gemini",
        )
    )
    checks.append(
        ("gemini envelope deny+reason", gem.get("decision") == "deny" and "reason" in gem)
    )
    cdx = json.loads(run(json.dumps({"tool_name": "Bash", "tool_response": big_log}), "codex"))
    checks.append(
        ("codex envelope block+reason", cdx.get("decision") == "block" and "reason" in cdx)
    )

    # Robustness: malformed input must degrade to a no-op, never a traceback.
    checks.append(("malformed json -> no-op", run("not json") == "{}"))
    checks.append(("empty event -> no-op", run("{}") == "{}"))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------------------
# Install / uninstall / status
#
# The hook is worthless until it is wired into the client's config, and a hand-written
# config is exactly where a silent typo costs a user every byte of savings without
# ever raising an error. So distil writes it: atomically (0600 temp, fsync, mode and
# owner kept, symlinks written through — ``setup._atomic_write``), idempotently, and
# without touching hooks it did not create. What distil created (the file, the event
# list) is recorded only AFTER the write succeeds, so undo removes only that.
# --------------------------------------------------------------------------------

_MARKER = "distil.hook"


@dataclass(frozen=True)
class HookClient:
    """One client whose documented post-tool hook can REPLACE tool output."""

    key: str
    label: str
    event: str
    scope: str  # what gets compressed there, in one line
    doc_url: str
    verified: str  # YYYY-MM-DD the contract was read from doc_url


CLIENTS: dict[str, HookClient] = {
    c.key: c
    for c in (
        HookClient(
            "claude",
            "Claude Code",
            "PostToolUse",
            "Bash + MCP tool output",
            "https://code.claude.com/docs/en/hooks",
            "2026-09-25",
        ),
        HookClient(
            "cursor",
            "Cursor",
            "postToolUse",
            "MCP tool output only (Cursor's output override is MCP-only)",
            "https://cursor.com/docs/agent/hooks",
            "2026-09-25",
        ),
        HookClient(
            "gemini",
            "Gemini CLI",
            "AfterTool",
            "run_shell_command + MCP tool output",
            "https://github.com/google-gemini/gemini-cli/blob/main/docs/hooks/reference.md",
            "2026-09-25",
        ),
        HookClient(
            "codex",
            "Codex CLI",
            "PostToolUse",
            "Bash + all-text MCP tool output",
            "https://developers.openai.com/codex/hooks",
            "2026-09-25",
        ),
    )
}

#: Clients checked and found to have NO post-tool hook that can replace output.
#: key -> (label, what was found, source, date). Documented, never faked.
UNSUPPORTED: dict[str, tuple[str, str, str, str]] = {
    "windsurf": (
        "Windsurf (Cascade)",
        "post_run_command / post_mcp_tool_use are observe-only: only pre-hooks can act "
        "(block via exit 2), no output field replaces a result, and post_run_command "
        "does not even receive the command's output",
        "https://docs.windsurf.com/windsurf/cascade/hooks",
        "2026-09-25",
    ),
}


def config_path(client: str) -> Path:
    """The user-level config file *client* reads its hooks from."""
    home = Path.home()
    if client == "claude":
        return Path(os.environ.get("CLAUDE_CONFIG_DIR") or str(home / ".claude")) / "settings.json"
    if client == "cursor":
        return home / ".cursor" / "hooks.json"
    if client == "gemini":
        return home / ".gemini" / "settings.json"
    if client == "codex":
        return Path(os.environ.get("CODEX_HOME") or str(home / ".codex")) / "hooks.json"
    raise KeyError(client)


def _settings_path() -> Path:  # 1.x name, kept for callers/tests
    return config_path("claude")


def _hook_command(client: str = "claude") -> str:
    """The exact command the client will run.

    Uses the *current* interpreter rather than a bare ``distil``: a pipx install puts
    distil on PATH but an editor may not inherit that PATH, and the failure would be
    silent (hook not found -> original output used, no error surfaced). Claude Code's
    command carries no ``--client`` so entries written by earlier releases still match.
    """
    exe = f'"{sys.executable}"' if " " in sys.executable else sys.executable
    cmd = f"{exe} -m distil.hook"
    return cmd if client == "claude" else f"{cmd} --client {client}"


def _entry(client: str = "claude") -> dict[str, Any]:
    cmd = _hook_command(client)
    if client == "cursor":
        # Flat entry; the matcher is a regex over the tool type, MCP tools as MCP:<name>.
        return {"command": cmd, "matcher": "MCP:"}
    hook: dict[str, Any] = {"type": "command", "command": cmd}
    if client == "gemini":
        return {"matcher": "run_shell_command|mcp_.*", "hooks": [{**hook, "name": "distil"}]}
    if client == "codex":
        hook["statusMessage"] = "distil: compacting tool output"
    return {"matcher": "Bash|mcp__.*", "hooks": [hook]}


def _scaffold(client: str) -> dict[str, Any]:
    """Top-level keys a brand-new config file needs (Cursor's schema is versioned)."""
    return {"version": 1} if client == "cursor" else {}


def _ours(entry: Any) -> bool:
    try:
        return _MARKER in json.dumps(entry)
    except (TypeError, ValueError):
        return False


# -- ownership record: what distil created, committed only after a successful write


def _owned_path() -> Path:
    from .setup import log_dir

    return log_dir() / "hooks-added.json"


def _read_owned() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_owned_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _write_owned(data: dict[str, dict[str, Any]]) -> None:
    from .setup import _atomic_write

    p = _owned_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, json.dumps(data, indent=2) + "\n")


def _load(path: Path) -> tuple[dict[str, Any] | None, str]:
    from .setup import _load_settings

    return _load_settings(path)


def install_hook(client: str = "claude") -> int:
    from .setup import _write_settings

    meta = CLIENTS[client]
    path = config_path(client)
    existed = path.exists()
    data, why = _load(path)
    if data is None:
        print(f"distil: cannot read {path}: {why}")
        print("  fix the file (or move it aside) and re-run; refusing to overwrite it.")
        return 1

    hooks = data.get("hooks", {})
    event_list = hooks.get(meta.event, []) if isinstance(hooks, dict) else None
    if not isinstance(hooks, dict) or not isinstance(event_list, list):
        print(f"distil: {path} has an unexpected hooks.{meta.event} shape; not touching it.")
        return 1
    created = [
        name
        for name, absent in (
            ("file", not existed),
            ("hooks", "hooks" not in data),
            ("event", meta.event not in hooks),
        )
        if absent
    ]

    # Idempotent: replace our own entry, never duplicate it, never touch anyone else's.
    kept = [e for e in event_list if not _ours(e)]
    kept.append(_entry(client))
    new = {**_scaffold(client), **data} if not existed else data
    new["hooks"] = {**hooks, meta.event: kept}
    try:
        _write_settings(path, new)
    except OSError as exc:  # nothing written, so nothing may be recorded as ours
        print(f"distil: could not write {path} ({exc}) — left untouched")
        return 1

    key = os.path.abspath(path)
    owned = _read_owned()
    prior = owned.get(key, {}).get("created", [])
    # A re-install must not forget that an earlier install created the file.
    owned[key] = {"client": client, "created": sorted(set(prior) | set(created))}
    try:
        _write_owned(owned)
    except OSError as exc:
        print(f"distil: hook installed, but the ownership record was not updated ({exc})")

    print(f"distil: {meta.label} hook installed in {path}")
    print(f"  command: {_hook_command(client)}")
    print(f"  scope:   {meta.scope}, results >= 2 KB; digests recoverable via `distil expand`")
    print(f"  verify:  distil hook --selftest     (contract: {meta.doc_url})")
    if client == "codex":
        print("\n  Codex runs a new hook only after you trust it: open /hooks in Codex.")
    else:
        print(f"\n  Restart {meta.label} for it to take effect.")
    return 0


def uninstall_hook(client: str = "claude") -> int:
    from .setup import _write_settings

    meta = CLIENTS[client]
    path = config_path(client)
    if not path.is_file():
        print(f"distil: nothing to remove ({path} does not exist)")
        return 0
    data, why = _load(path)
    if data is None:
        print(f"distil: cannot read {path}: {why}")
        return 1

    hooks = data.get("hooks")
    event_list = hooks.get(meta.event) if isinstance(hooks, dict) else None
    if not isinstance(event_list, list):
        print("distil: no distil hook found")
        return 0
    kept = [e for e in event_list if not _ours(e)]
    if len(kept) == len(event_list):
        print("distil: no distil hook found")
        return 0

    key = os.path.abspath(path)
    owned = _read_owned()
    created = set(owned.get(key, {}).get("created", []))
    assert isinstance(hooks, dict)
    hooks[meta.event] = kept
    # Remove only containers distil itself created, and only once they are empty.
    if not kept and "event" in created:
        del hooks[meta.event]
    if not hooks and "hooks" in created:
        del data["hooks"]
    try:
        if "file" in created and data == _scaffold(client):
            path.unlink()
        else:
            _write_settings(path, data)
    except OSError as exc:
        print(f"distil: could not write {path} ({exc}) — left untouched")
        return 1
    if key in owned:
        del owned[key]
        try:
            _write_owned(owned)
        except OSError:
            pass  # a stale record only means a later undo keeps empty containers
    print(f"distil: {meta.label} hook removed from {path}")
    return 0


def hook_status(client: str) -> tuple[bool, Path]:
    """Is distil's hook present in *client*'s config? ``(installed, path)``."""
    path = config_path(client)
    data, _ = _load(path) if path.is_file() else (None, "")
    hooks = data.get("hooks") if isinstance(data, dict) else None
    entries = hooks.get(CLIENTS[client].event) if isinstance(hooks, dict) else None
    return (isinstance(entries, list) and any(_ours(e) for e in entries)), path


def detected_clients() -> list[str]:
    """Clients whose config directory exists on this machine (Claude Code always)."""
    found = ["claude"]
    for key in ("cursor", "gemini", "codex"):
        if config_path(key).parent.is_dir():
            found.append(key)
    return found


def print_status() -> int:
    for key, meta in CLIENTS.items():
        installed, path = hook_status(key)
        mark = "✓ installed" if installed else "· not installed"
        print(f"  {meta.label:<12} {mark:<16} {path}")
        print(f"  {'':<12} scope: {meta.scope}")
    for label, why, url, date in UNSUPPORTED.values():
        print(f"  {label:<12} ✗ unsupported: {why} ({url}, checked {date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

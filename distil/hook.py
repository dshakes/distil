"""Claude Code ``PostToolUse`` hook — lossless tool-output compression.

Why this exists
---------------
On a flat-rate subscription there is no per-token bill to cut, but there *is* a
rate-limit window: tokens spent on a 40 KB test log are tokens unavailable for the
next task. distil's proxy cannot help there — Anthropic's consumer terms restrict
automated access on subscription credentials, so the proxy stays lossless-only and
distil does not route subscription traffic through the digest tier.

A ``PostToolUse`` hook is a different mechanism entirely: a documented, first-party
extension point. Claude Code hands us a tool result, we hand back a smaller one via
``hookSpecificOutput.updatedToolOutput``, and it reaches the model instead of the
original. No proxy, no OAuth interception, no credential bridging.

Append-only by construction
---------------------------
distil measured its proxy digest *doubling* cost on a live A/B: the sliding recency
window rewrote the previous turn's last tool_result each turn, one message before the
cache breakpoint, collapsing the whole cached prefix (median cache_read 233.8k ->
24.7k). A hook cannot do that. It sees one result, once, at the moment it is
produced, and no hook event can rewrite history. Compression here is therefore
append-only *because the platform makes it so* — the fix direction that investigation
identified, enforced rather than merely intended.

That same property is why this path is lossless-only: with no way to rewrite history,
a digest stub could never be expanded back, so a recall tool would be useless even if
we wanted one. Policy and capability agree.

The contract (verified against the official docs, 2026-08)
----------------------------------------------------------
Return ``{"hookSpecificOutput": {"hookEventName": "PostToolUse",
"updatedToolOutput": <value>}}`` on stdout, exit 0.

    "The replacement value must match the tool's output shape... For built-in tools,
    a value that doesn't match the tool's output schema is ignored and the original
    output is used. MCP tool output is passed through without schema validation.
    Stripping error details that Claude needs can cause it to proceed on a false
    assumption."

Two consequences drive every design choice below:

1. **A schema near-miss fails silently.** No error, no warning — the original is used.
   So this hook emits nothing at all unless it is confident of the shape, and ships
   ``--selftest`` to prove the adapters offline.
2. **Error text is load-bearing.** We never compress a failed tool's output and never
   touch a non-empty ``stderr``. Anthropic's warning is precisely the failure distil's
   decision-equivalence contract exists to prevent; we do not commit it ourselves.

Fail-safe posture: any exception, unknown tool, unreadable payload, or transform that
does not strictly reduce tokens results in an empty object on stdout, which leaves the
original untouched. Doing nothing is always a correct outcome here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Below this, compression cannot save enough to be worth any risk. Tool results are
# long-tailed: the savings live in test logs, file dumps and tracebacks, not in the
# one-line results that dominate by count.
_MIN_CHARS = 2048


def _tier0(text: str) -> str:
    """Lossless Tier-0: JSON minification, then run collapse, reject-if-bigger.

    Mirrors ``distil.adapters.anthropic._apply_tier0`` — the same transforms the
    subscription-safe proxy path uses, so the hook and the proxy cannot disagree
    about what "lossless" means. Run-collapse is rejected when it does not reduce
    *tokens* (what the window is metered in), because collapsing near-free
    whitespace into a ``<<xN>>`` marker can cost more than it saves.
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


def _compress_bash(out: dict[str, Any]) -> dict[str, Any] | None:
    """Compress a ``Bash`` result: ``{stdout, stderr, interrupted, isImage}``.

    Only ``stdout`` is touched, and only on a clean run. A non-empty ``stderr`` or an
    interrupt means something went wrong and every byte may matter to the diagnosis.
    """
    if out.get("interrupted") or out.get("isImage"):
        return None
    if out.get("stderr"):
        return None
    stdout = out.get("stdout")
    if not isinstance(stdout, str) or len(stdout) < _MIN_CHARS:
        return None

    shrunk = _tier0(stdout)
    if shrunk == stdout:
        return None
    # Preserve every key we were given; replace exactly one value. Reconstructing the
    # dict from known keys would silently drop fields added by future versions.
    return {**out, "stdout": shrunk}


def _compress_mcp(out: Any) -> Any | None:
    """Compress MCP tool output, which the docs pass through without schema checks.

    MCP servers vary wildly, so this handles only the common content-block list and
    leaves anything else alone. An error result is never touched.
    """
    if isinstance(out, str):
        return _tier0(out) if len(out) >= _MIN_CHARS and _tier0(out) != out else None

    if not isinstance(out, dict) or out.get("isError"):
        return None
    blocks = out.get("content")
    if not isinstance(blocks, list):
        return None

    new_blocks, changed = [], False
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
            text = b["text"]
            if len(text) >= _MIN_CHARS:
                shrunk = _tier0(text)
                if shrunk != text:
                    new_blocks.append({**b, "text": shrunk})
                    changed = True
                    continue
        new_blocks.append(b)  # non-text blocks (images, resources) pass through
    return {**out, "content": new_blocks} if changed else None


def compress_tool_output(tool_name: str, tool_output: Any) -> Any | None:
    """Return a compressed replacement, or None to leave the original alone.

    Pure and importable so the behaviour can be tested without a live session —
    which matters because the failure mode in production is *silence*.
    """
    if tool_name == "Bash":
        return _compress_bash(tool_output) if isinstance(tool_output, dict) else None
    if tool_name.startswith("mcp__"):
        return _compress_mcp(tool_output)
    # Read/Grep/Glob and other built-ins have undocumented shapes. Guessing costs
    # nothing when wrong (the original is used) but yields no savings either, so we
    # decline until each shape is probed live.
    return None


def run(stdin_text: str) -> str:
    """Map one hook invocation to its stdout. Never raises."""
    try:
        event = json.loads(stdin_text)
        tool_name = event.get("tool_name")
        tool_output = event.get("tool_response", event.get("tool_output"))
        if not isinstance(tool_name, str) or tool_output is None:
            return "{}"

        replacement = compress_tool_output(tool_name, tool_output)
        if replacement is None:
            return "{}"

        return json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "updatedToolOutput": replacement,
                }
            }
        )
    except Exception:
        # A hook that crashes must not break the user's session, and a hook that
        # emits garbage must not corrupt a tool result. Both resolve to "do nothing".
        return "{}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return _selftest()
    sys.stdout.write(run(sys.stdin.read()))
    return 0


def _selftest() -> int:
    """Prove the adapters offline, since a live mismatch is silent.

    Checks the properties that actually matter: we shrink what we should, we emit the
    exact documented envelope, and we refuse everything that carries risk.
    """
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
    # replacement was accepted only because unknown keys were carried through. Since
    # an unrecognised shape is rejected *silently*, the adapter must preserve keys it
    # does not know about rather than rebuild the dict from the documented four.
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

    # Envelope: the exact documented field names, or the whole thing is a silent no-op.
    emitted = json.loads(
        run(
            json.dumps(
                {
                    "tool_name": "Bash",
                    "tool_response": {
                        "stdout": big_log,
                        "stderr": "",
                        "interrupted": False,
                        "isImage": False,
                    },
                }
            )
        )
    )
    hso = emitted.get("hookSpecificOutput", {})
    checks.append(("envelope hookEventName", hso.get("hookEventName") == "PostToolUse"))
    checks.append(("envelope updatedToolOutput", "updatedToolOutput" in hso))

    # Robustness: malformed input must degrade to a no-op, never a traceback.
    checks.append(("malformed json -> no-op", run("not json") == "{}"))
    checks.append(("empty event -> no-op", run("{}") == "{}"))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------------
# Install / uninstall
#
# The hook is worthless until it is wired into settings.json, and a hand-written
# config is exactly where a silent typo costs a user every byte of savings without
# ever raising an error. So distil writes it, idempotently, and refuses to clobber
# hooks it did not create.
# --------------------------------------------------------------------------------

_MARKER = "distil.hook"


def _settings_path() -> Path:
    cfg = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(cfg) / "settings.json"


def _hook_command() -> str:
    """The exact command Claude Code will run.

    Uses the *current* interpreter rather than a bare ``distil``: a pipx install puts
    distil on PATH but Claude Code may not inherit that PATH, and the failure would
    be silent (hook not found -> original output used, no error surfaced).
    """
    return f"{sys.executable} -m distil.hook"


def _entry() -> dict[str, Any]:
    return {
        "matcher": "Bash|mcp__.*",
        "hooks": [{"type": "command", "command": _hook_command()}],
    }


def install_hook() -> int:
    path = _settings_path()
    try:
        settings = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError) as exc:
        print(f"distil: cannot read {path}: {exc}")
        print("  fix the file (or move it aside) and re-run; refusing to overwrite it.")
        return 1

    hooks = settings.setdefault("hooks", {})
    post = hooks.setdefault("PostToolUse", [])
    if not isinstance(post, list):
        print(f"distil: {path} has an unexpected PostToolUse shape; not touching it.")
        return 1

    # Idempotent: replace our own entry, never duplicate it, never touch anyone else's.
    kept = [
        e
        for e in post
        if not any(_MARKER in str(h.get("command", "")) for h in (e.get("hooks") or []))
    ]
    kept.append(_entry())
    hooks["PostToolUse"] = kept

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"distil: hook installed in {path}")
    print(f"  command: {_hook_command()}")
    print("  scope:   Bash + MCP tool results >= 2 KB, lossless only")
    print("  verify:  distil hook --selftest")
    print("\n  Restart Claude Code for it to take effect.")
    return 0


def uninstall_hook() -> int:
    path = _settings_path()
    if not path.is_file():
        print(f"distil: nothing to remove ({path} does not exist)")
        return 0
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"distil: cannot read {path}: {exc}")
        return 1

    post = (settings.get("hooks") or {}).get("PostToolUse")
    if not isinstance(post, list):
        print("distil: no distil hook found")
        return 0
    kept = [
        e
        for e in post
        if not any(_MARKER in str(h.get("command", "")) for h in (e.get("hooks") or []))
    ]
    if len(kept) == len(post):
        print("distil: no distil hook found")
        return 0
    settings["hooks"]["PostToolUse"] = kept
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"distil: hook removed from {path}")
    return 0

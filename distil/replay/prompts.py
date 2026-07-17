"""Shared decision-prompt rendering + fingerprint parsing for all live runners.

Every runner (Anthropic API, OpenAI-compatible/vLLM, the `claude -p` CLI) must put
the *same* context in front of the model and extract the *same* canonical
``{action,target}`` fingerprint, so their verdicts are comparable. That logic lives
here, once.
"""

from __future__ import annotations

import json
import re

from ..trajectory import Block, Kind, Stability

INSTRUCTION = (
    "You are choosing the single next action an autonomous agent should take given the "
    "context above. Respond with ONLY a compact JSON object and nothing else:\n"
    '{"action": "<tool or operation name>", "target": "<the primary argument>"}'
)


def render(blocks: list[Block]) -> tuple[str, str]:
    """Split blocks into (system, user) text exactly like the Anthropic runner does:
    the stable system/tool schema becomes the system prompt; everything else (history,
    fresh observations) becomes the user turn."""
    system_parts = [
        b.text
        for b in blocks
        if b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS)
    ]
    rest = [
        b
        for b in blocks
        if not (b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS))
    ]
    user = "\n\n".join(f"[{b.kind.value}] {b.text}" for b in rest)
    system = "\n\n".join(system_parts) or "You are an autonomous agent."
    return system, user


def decision_prompt(blocks: list[Block]) -> tuple[str, str]:
    """(system, user+instruction) for text/JSON runners that don't use a forced tool."""
    system, user = render(blocks)
    return system, f"{user}\n\n{INSTRUCTION}"


# The expand-or-decide protocol: lets the grader recover digested content before
# committing — a runner-agnostic simulation of the distil_expand recovery loop.
EXPAND_INSTRUCTION = (
    "Some context was digested to save tokens and appears as a marker like "
    "'<< +N lines, handle=XXXXXXXX >>' (or a «…» marker). You may recover the full "
    "original content of any digested block before deciding. Respond with ONLY one "
    "compact JSON object:\n"
    '  {"expand": ["<handle>", ...]}   to recover digested content you need, OR\n'
    '  {"action": "<tool/op>", "target": "<primary argument>"}   to commit to the next action.\n'
    "Expand only what you actually need; commit as soon as you can."
)


def expand_prompt(blocks: list[Block]) -> tuple[str, str]:
    system, user = render(blocks)
    return system, f"{user}\n\n{EXPAND_INSTRUCTION}"


_EXPAND = re.compile(r'\{[^{}]*"expand"[^{}]*\}', re.DOTALL)
_HANDLE = re.compile(r"\b([0-9a-f]{8})\b")


def parse_expand(text: str) -> list[str] | None:
    """Return the list of handles the model asked to expand, or None if it committed
    to a decision instead (or emitted nothing parseable as an expand request)."""
    if not text:
        return None
    m = _EXPAND.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        # fall back to scraping handle-looking tokens out of the expand object text
        return _HANDLE.findall(m.group(0)) or None
    exp = obj.get("expand") if isinstance(obj, dict) else None
    if not exp:
        return None
    return [str(h) for h in exp]


_OBJ = re.compile(r'\{[^{}]*"action"[^{}]*\}', re.DOTALL)


def _norm_action(a: str) -> str:
    """Canonicalize a tool/action name so paraphrase ≠ decision change:
    ``search_flights`` == ``SearchFlights`` == ``search flights``. Lowercase, drop
    non-alphanumerics. (Targets are free-form; see paper §7 on fingerprint
    granularity — a structured/forced-tool grader avoids this entirely.)"""
    return re.sub(r"[^a-z0-9]", "", str(a).lower())


def canonical(action: str, target: str) -> str:
    """The canonical decision fingerprint shared by the grader and the gold label, so
    they are comparable: normalized action + case/space-folded target."""
    return json.dumps(
        {"action": _norm_action(action), "target": str(target).strip().lower()},
        sort_keys=True,
        separators=(",", ":"),
    )


# A single forced "record_decision" tool, used by the structured graders (Anthropic
# forced tool, OpenAI forced function). A structured schema makes the model fill an
# `action` and a `target` field rather than emit prose, which removes the paraphrase
# noise that free-text grading suffers (§ fingerprint granularity). The expand loop
# still uses the free-text `_raw` protocol so it works across all backends.
DECISION_TOOL_NAME = "record_decision"
DECISION_PARAMS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "The tool/operation to invoke next."},
        "target": {"type": "string", "description": "The primary argument or target."},
    },
    "required": ["action", "target"],
    "additionalProperties": False,
}
DECISION_TOOL_DESC = "Record the single next action the agent will take given the context."

# Tool declarations in context look like "- tool_name(args): description".
_TOOL_DECL = re.compile(r"^\s*-\s*([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def available_actions(blocks) -> list[str]:
    """Tool names declared in the context's TOOLS blocks, in order, deduped.

    Used to constrain the decision tool's ``action`` field to the actions the
    agent could actually take. Without the constraint the grader model free-
    types action names and invents paraphrases ("summarize_and_share" vs
    "summarize_findings_and_share_chart" — observed live), which registers as
    a decision change when the decision is identical. Constraining to the real
    tool menu removes that noise WITHOUT loosening fingerprint equality — a
    real agent picks from its real tools too. Returns [] when no declarations
    parse (the schema then stays a free string — fail-open)."""
    from ..trajectory import Kind

    names: list[str] = []
    for b in blocks:
        if getattr(b, "kind", None) is Kind.TOOLS:
            names.extend(_TOOL_DECL.findall(b.text))
    return list(dict.fromkeys(names))


def fingerprint_from_args(args) -> str:
    """Canonical fingerprint from a tool/function call's parsed arguments (dict or
    JSON string)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return parse_fingerprint(args)
    if isinstance(args, dict) and "action" in args:
        return canonical(args.get("action", ""), args.get("target", ""))
    return "<no-decision>"


def parse_fingerprint(text: str) -> str:
    """Extract a canonical ``{"action":..,"target":..}`` fingerprint from free model
    text (handles ```json fences, surrounding prose, key reordering). The action is
    normalized so surface paraphrase of the same tool is not counted as a decision
    change. Returns ``"<no-decision>"`` if nothing parseable is found."""
    if not text:
        return "<no-decision>"
    candidates = [text.strip()]
    m = _OBJ.search(text)
    if m:
        candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "action" in obj:
            return canonical(obj.get("action", ""), obj.get("target", ""))
    return "<no-decision>"

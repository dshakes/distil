"""Output compression — reduce the cost of the model's *generated* tokens.

Input compression (the rest of distil) shrinks what the model reads. This module
addresses the other side of the bill — output tokens cost ~4-5× input — with two
real, complementary mechanisms:

1. **Lossless output-on-re-entry digest** (`digest_output_blocks`). A long model
   answer becomes *history* on the next turn, where it is re-sent every step.
   We digest those large assistant/history blocks decision-aware and reversibly
   (same Tier-1 machinery as tool outputs), so a verbose past answer stops
   costing full price as context. Lossless, offline-measurable.

2. **Generation-side shaping** (`shape_request`). A verbosity-control directive
   is injected as a `role:"system"` operator message (provider-ToS-safe, not
   user-spoofable) so the model *emits* fewer tokens. This is lossy by nature —
   the wording changes — so it is **gated** (PAYG only, never on subscription/
   OAuth) and **measured**: `measure_output_savings` reports the token reduction
   *and* the rate at which the underlying answer is preserved, with a bootstrap
   CI. We never claim a reduction without checking the answer survived.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .certify.holdout import bootstrap_ci
from .compress.tier1 import Tier1Reversible
from .tokenizer import DEFAULT, Tokenizer
from .trajectory import Block, Kind

# --- generation-side shaping ------------------------------------------------

OUTPUT_DIRECTIVES: dict[str, str] = {
    "off": "",
    "light": (
        "Answer directly and concisely. Omit preamble, restating the question, "
        "and closing recaps. Do not narrate routine steps."
    ),
    "aggressive": (
        "Be maximally concise. Lead with the answer or decision in one line. "
        "Reference identifiers and prior context instead of reproducing them. "
        "No preamble, no recaps, no narration of routine actions, no apologies."
    ),
}


def shape_request(
    body: dict, *, level: str = "light", allow: bool = True, shape: str = "auto"
) -> dict:
    """Return a copy of an Anthropic/OpenAI/Gemini request body with a verbosity-control
    directive injected the way each provider actually accepts it. No-op when
    `level == "off"` or `allow` is False (auth-mode gating). Never mutates input.

    The Anthropic Messages API rejects ``role:"system"`` entries inside
    ``messages`` (400) — there the directive goes into the top-level ``system``
    field. The OpenAI chat shape takes an appended system message. Gemini uses
    ``systemInstruction: {"parts": [{"text": "..."}]}``. ``shape`` is
    ``"anthropic"``/``"openai"``/``"gemini"`` when the caller knows the request
    path, or ``"auto"`` to sniff (``contents`` key → Gemini; top-level ``system``
    or an Anthropic model id → Anthropic; otherwise OpenAI).
    """
    if level == "off" or not allow:
        return body
    directive = OUTPUT_DIRECTIVES.get(level)
    if not directive:
        raise ValueError(f"unknown output level {level!r}; choose {sorted(OUTPUT_DIRECTIVES)}")
    # OpenAI Responses API: directive goes into top-level ``instructions`` (the system prompt).
    if shape == "responses":
        instr = body.get("instructions")
        new_instr = (instr + "\n\n" + directive) if isinstance(instr, str) and instr else directive
        return {**body, "instructions": new_instr}
    if shape == "gemini":
        gemini, anthropic = True, False
    elif shape == "auto":
        # Gemini bodies have ``contents`` and no ``messages`` or ``system``.
        gemini = "contents" in body and "messages" not in body and "system" not in body
        anthropic = not gemini and (
            "system" in body or str(body.get("model", "")).startswith("claude")
        )
    else:
        gemini = False
        anthropic = shape == "anthropic"
    if gemini:
        # Inject into systemInstruction; create it if absent.
        sys_instr = body.get("systemInstruction")
        if isinstance(sys_instr, dict):
            existing = list(sys_instr.get("parts") or [])
            return {
                **body,
                "systemInstruction": {**sys_instr, "parts": [*existing, {"text": directive}]},
            }
        return {**body, "systemInstruction": {"parts": [{"text": directive}]}}
    if anthropic:
        sys_prompt = body.get("system")
        if isinstance(sys_prompt, list):
            new_system: object = [*sys_prompt, {"type": "text", "text": directive}]
        elif isinstance(sys_prompt, str) and sys_prompt:
            new_system = sys_prompt + "\n\n" + directive
        else:
            new_system = directive
        return {**body, "system": new_system}
    messages = list(body.get("messages", []))
    messages.append({"role": "system", "content": directive})
    return {**body, "messages": messages}


# --- lossless output re-entry digest ----------------------------------------

_DIGESTIBLE_OUTPUT = {Kind.HISTORY}  # assistant answers settle into history


def digest_output_blocks(blocks: list[Block], *, min_lines: int = 6) -> tuple[list[Block], dict]:
    """Reversibly digest large assistant/history blocks so verbose past outputs
    stop costing full price when they re-enter context. Returns (blocks, restore)."""
    digester = Tier1Reversible(min_lines=min_lines)
    out: list[Block] = []
    restore: dict = {}
    for b in blocks:
        if b.kind in _DIGESTIBLE_OUTPUT and b.text.count("\n") + 1 >= min_lines:
            # Tier1 only digests TOOL_OUTPUT/RETRIEVED by kind; re-tag for the digest.
            proxy = Block(b.id, Kind.TOOL_OUTPUT, b.text, b.stability, b.decision_relevant)
            res = digester.compress([proxy])
            restore.update(res.restore)
            out.append(Block(b.id, b.kind, res.blocks[0].text, b.stability, b.decision_relevant))
        else:
            out.append(b)
    return out, restore


# --- A/B measurement (the evaluation) ---------------------------------------

_ANSWER_RE = re.compile(
    r"(?:DECISION|ANSWER|RESULT)\s*:\s*(.+?)(?=\.\s|\.$|$)", re.IGNORECASE | re.MULTILINE
)


def answer_fingerprint(text: str) -> str:
    """Extract the decision/answer content used to test that shaping preserved
    the substance. Prefers explicit DECISION/ANSWER lines; else a normalized
    whitespace/case-folded form of the whole text."""
    hits = [m.group(1).strip().lower() for m in _ANSWER_RE.finditer(text)]
    if hits:
        return " | ".join(sorted(hits))
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class OutputSavingsReport:
    n: int
    mean_reduction: float
    ci_low: float
    ci_high: float
    answer_match_rate: float

    @property
    def summary(self) -> str:
        return (
            f"output tokens cut {self.mean_reduction * 100:.1f}% "
            f"(95% CI {self.ci_low * 100:.1f}–{self.ci_high * 100:.1f}%), "
            f"answer preserved {self.answer_match_rate * 100:.1f}% of the time, n={self.n}"
        )


def measure_output_savings(
    pairs: list[tuple[str, str]], *, tok: Tokenizer = DEFAULT
) -> OutputSavingsReport:
    """Given (baseline_output, shaped_output) pairs, report the token reduction
    with a bootstrap CI and the rate at which the answer fingerprint is preserved.
    The answer-match rate is the quality gate: a reduction that drops answers is
    not a saving."""
    if not pairs:
        raise ValueError("need at least one (baseline, shaped) pair")
    reductions: list[float] = []
    matches = 0
    for baseline, shaped in pairs:
        b = tok.count(baseline)
        s = tok.count(shaped)
        reductions.append((1.0 - s / b) if b else 0.0)
        if answer_fingerprint(baseline) == answer_fingerprint(shaped):
            matches += 1
    mean, lo, hi = bootstrap_ci(reductions)
    return OutputSavingsReport(len(pairs), mean, lo, hi, matches / len(pairs))

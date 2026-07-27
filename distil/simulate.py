"""Dry-run: what distil would do to a request, and why — without calling a model.

Two questions a user actually asks before switching compression on:

  1. "How much would this save me?"  — answerable by running the real pipeline
     locally, which costs nothing because no model is in the path.
  2. "What would it touch, and what would it leave alone?" — the one that decides
     whether they trust it.

The second is the reason this module exists rather than a bare ratio. distil
already computes *why* a block is protected — the recency exemption, the
assistant's own words, a decision-bearing line pinned by the keep policy — and a
preview that shows the ratio while hiding that reasoning is asking for the same
trust every other compressor asks for. Here the protected set is reported as
first-class output, per block, with the rule that protected it.

Waste attribution answers the follow-up ("why is my saving only 12%?"): it splits
the input into the recognisable shapes that inflate a prompt — pretty-printed
JSON, whitespace runs, verbatim repetition — so the answer is a number and a
cause rather than a shrug.

Nothing here is an estimate of model behaviour. It reports what the pipeline
does, measured, on the caller's own payload.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .compress.keep_policy import classify, must_keep
from .compress.recency import RECENCY_KEEP_TURNS
from .compress.tier0 import collapse_runs, minify_json
from .tokenizer import DEFAULT as _tok

# A run of 3+ blank lines / long indentation is padding, not structure.
_WS_RUN = re.compile(r"[ \t]{4,}|\n{3,}")


def _count(text: str) -> int:
    return _tok.count(text)


def _block_text(block: Any) -> str:
    """The billable text of one content block, whatever shape it arrives in."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    for key in ("text", "content"):
        val = block.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return "\n".join(_block_text(sub) for sub in val)
    # A tool_use block's billable content is its serialized arguments. Without
    # this it read as empty and was skipped entirely, so the preview never showed
    # that tool_use is protected — the omission a caller would notice only after
    # trusting the report.
    if block.get("type") == "tool_use":
        args = block.get("input")
        if args is not None:
            try:
                return json.dumps(args, sort_keys=True)
            except (TypeError, ValueError):
                return str(args)
        return str(block.get("name") or "")
    return ""


def waste_signals(text: str) -> dict[str, dict[str, int]]:
    """Bloat attributable to each recognisable shape, in BOTH chars and tokens.

    Both, because they disagree and the disagreement is the point. distil's
    default offline tokenizer is whitespace-insensitive: a pretty-printed JSON
    array and its minified form count IDENTICALLY (measured: 5,882 chars vs
    3,721 chars, 2,874 tokens either way). Reporting only tokens would print
    `json_bloat: 0` over an obviously bloated payload; reporting only chars would
    imply a saving the tokenizer will not actually give you.

    So: `chars` is what the shape costs on the wire, `tokens` is what THIS
    tokenizer will charge for it. Run with a billing-grade tokenizer to see the
    number your provider would bill.

    Deliberately overlapping: a pretty-printed array is both json_bloat and
    whitespace, and no honest split exists — so they are separate diagnostics,
    never summed into one "waste" figure.
    """
    out = {
        "json_bloat": {"chars": 0, "tokens": 0},
        "whitespace": {"chars": 0, "tokens": 0},
        "repetition": {"chars": 0, "tokens": 0},
    }

    def _record(name: str, reduced: str) -> None:
        if reduced == text or len(reduced) >= len(text):
            return
        out[name] = {
            "chars": len(text) - len(reduced),
            "tokens": max(0, _count(text) - _count(reduced)),
        }

    minified = minify_json(text)
    if minified is not None:
        _record("json_bloat", minified)
    _record("whitespace", _WS_RUN.sub(" ", text))
    _record("repetition", collapse_runs(text))
    return out


def _protection(role: str, btype: str, is_recent: bool, text: str) -> str:
    """Why this block would be left byte-exact, or "" if it is compressible.

    The wording is the *rule*, not a restatement of the outcome — a user who
    disagrees with a protection should be able to find the mechanism from this
    string alone.
    """
    if btype == "tool_use":
        return "tool_use — arguments are executed verbatim, never rewritten"
    if btype == "image":
        return "image — vision blocks are only ever de-duplicated, never altered"
    if role == "assistant":
        return "assistant turn — the model's own words are never rewritten"
    if is_recent:
        return (
            f"recency — within the last {RECENCY_KEEP_TURNS} tool-bearing turns, "
            "which stay byte-exact so the agent never reasons blind on fresh output"
        )
    kind = classify(text)
    pinned = [ln for ln in text.splitlines() if must_keep(ln, kind)]
    if pinned and len(pinned) == len(text.splitlines()):
        return f"every line is decision-bearing ({kind.value}) — nothing is droppable"
    return ""


def _recent_indices(messages: list[dict[str, Any]], k: int) -> set[int]:
    idxs = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") in ("user", "tool")
    ]
    return set(idxs[-k:]) if k > 0 else set()


def simulate(messages: list[dict[str, Any]], *, verbatim: bool = False) -> dict[str, Any]:
    """Run the real compression pipeline and report what it did.

    Returns totals, a per-block breakdown (including the protected set and the
    rule that protected each one), and waste attribution. No network, no model,
    no side effects — safe to run across a sample of production traffic to size
    the saving before switching anything on.
    """
    from .adapters.anthropic import compress_messages  # local: avoids an import cycle

    before_json = json.dumps(messages)
    compressed, store = compress_messages(messages, verbatim=verbatim)
    after_json = json.dumps(compressed)

    recent = _recent_indices(messages, RECENCY_KEEP_TURNS)
    blocks: list[dict[str, Any]] = []
    totals = {k: {"chars": 0, "tokens": 0} for k in ("json_bloat", "whitespace", "repetition")}

    for idx, (src, dst) in enumerate(zip(messages, compressed)):
        if not isinstance(src, dict):
            continue
        role = src.get("role", "")
        s_content = src.get("content")
        d_content = dst.get("content") if isinstance(dst, dict) else None
        s_list = s_content if isinstance(s_content, list) else [s_content]
        d_list = d_content if isinstance(d_content, list) else [d_content]

        for j, s_blk in enumerate(s_list):
            text = _block_text(s_blk)
            if not text:
                continue
            d_blk = d_list[j] if j < len(d_list) else None
            after_text = _block_text(d_blk)
            btype = s_blk.get("type", "text") if isinstance(s_blk, dict) else "text"
            b_tok, a_tok = _count(text), _count(after_text)

            sig = waste_signals(text)
            for k in totals:
                totals[k]["chars"] += sig[k]["chars"]
                totals[k]["tokens"] += sig[k]["tokens"]

            blocks.append(
                {
                    "turn": idx,
                    "role": role,
                    "type": btype,
                    "tokens_before": b_tok,
                    "tokens_after": a_tok,
                    "saved": max(0, b_tok - a_tok),
                    "kind": classify(text).value if btype == "tool_result" else "",
                    "protected_by": _protection(role, btype, idx in recent, text),
                    "waste": sig,
                }
            )

    before, after = _count(before_json), _count(after_json)
    protected = [b for b in blocks if b["protected_by"]]
    return {
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": max(0, before - after),
        "savings_percent": round((before - after) / before * 100, 1) if before else 0.0,
        # Every digest is one distil_expand away; this is the count of blocks the
        # agent could pull back in full, which is what makes the saving reversible
        # rather than a deletion.
        "recoverable_blocks": len(store.handles),
        "blocks": blocks,
        "protected_blocks": len(protected),
        "waste_signals": totals,
        "verbatim": verbatim,
    }


def format_report(sim: dict[str, Any], *, width: int = 76) -> str:
    """Human-readable rendering of :func:`simulate`, for the CLI."""
    L: list[str] = []
    pct = sim["savings_percent"]
    L.append("distil simulate — what would happen, no model called")
    L.append("")
    L.append(
        f"  {sim['tokens_before']:,} tokens in  ->  {sim['tokens_after']:,} out"
        f"   ({pct}% saved, {sim['recoverable_blocks']} block(s) recoverable)"
    )
    if sim["verbatim"]:
        L.append("  mode: verbatim — in-context-lossless transforms only, no digests")
    L.append("")

    ws = sim["waste_signals"]
    if any(v["chars"] for v in ws.values()):
        L.append("  where the waste is (diagnostics; they overlap, so they are not summed)")
        for name, val in sorted(ws.items(), key=lambda kv: -kv[1]["chars"]):
            if val["chars"]:
                L.append(f"    {name:<12} {val['chars']:>8,} chars   {val['tokens']:>7,} tokens")
        if not any(v["tokens"] for v in ws.values()):
            L.append("")
            L.append("    chars > 0 but tokens = 0: this tokenizer is whitespace-insensitive,")
            L.append("    so it will not bill you for that shape. Re-run with a billing-grade")
            L.append("    tokenizer to see what your provider would actually charge.")
        L.append("")

    movers = [b for b in sim["blocks"] if b["saved"] > 0]
    if movers:
        L.append("  compressed")
        for b in sorted(movers, key=lambda b: -b["saved"])[:8]:
            kind = f" [{b['kind']}]" if b["kind"] else ""
            L.append(
                f"    turn {b['turn']:<3} {b['role']:<9} {b['type']:<12}{kind}"
                f"  {b['tokens_before']:,} -> {b['tokens_after']:,}"
            )
        L.append("")

    protected = [b for b in sim["blocks"] if b["protected_by"]]
    if protected:
        L.append("  left byte-exact, and why")
        seen: set[str] = set()
        for b in protected:
            why = b["protected_by"]
            if why in seen:
                continue
            seen.add(why)
            head = f"    turn {b['turn']:<3} {b['type']:<12} "
            # Wrap rather than slice: these strings explain a RULE, and a reason
            # cut mid-sentence ("…which stay byte-exact so the") teaches nothing.
            words, line = why.split(), head
            for w in words:
                if len(line) + len(w) + 1 > width and line.strip() != head.strip():
                    L.append(line)
                    line = " " * len(head) + w
                else:
                    line = (line + " " + w) if line != head else line + w
            if line.strip():
                L.append(line)
        if len(protected) > len(seen):
            L.append(f"    … {len(protected) - len(seen)} more block(s) under the same rules")
    return "\n".join(L)

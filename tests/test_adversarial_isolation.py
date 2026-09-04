"""Trusted/untrusted budget isolation, and the COMA-class battery as a unit gate.

Background: arXiv 2510.22963 (ASE 2026) shows an attacker who controls untrusted input can
perturb it so the *compressor* discards task-critical content. The paper's validated mitigation
is isolating trusted from untrusted content into separate compression budgets.

distil has that isolation **by construction** rather than by policy: there is no global keep
budget anywhere in the pipeline. ``tier1.digest`` decides head/tail plus must-keep lines from a
single block's own text, and ``adapters.anthropic.compress_messages`` walks blocks one at a
time. Nothing is shared, so nothing can be starved. That is a strong claim about an absence,
and an absence is exactly what a later refactor can quietly fill in — a top-N-lines-per-request
cap would look like a reasonable optimisation and would silently create the shared budget the
paper attacks. These tests are what makes that regression loud.
"""

from __future__ import annotations

import json

import pytest

from distil import harness
from distil.adapters.anthropic import compress_messages

DECISION = "DECISION: roll back to build 4417 - canary failed"
# 4000 lines x ~220 chars: far larger than any plausible per-request budget.
ATTACKER = "\n".join(f"attacker filler line {i} " + "y" * 200 for i in range(4000))
TRUSTED = "\n".join([f"step {i} completed" for i in range(40)] + [DECISION])


def _convo(blocks: list[str]) -> list[dict[str, object]]:
    """A session carrying `blocks` as tool results, plus two filler turns so none of the
    blocks under test fall inside the recency carve-out (which would exempt them from
    compression entirely and make the assertion vacuous)."""
    msgs: list[dict[str, object]] = [{"role": "user", "content": "go"}]
    for i, b in enumerate(blocks):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": b}],
            }
        )
    for j in range(2):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"f{j}", "name": "bash", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"f{j}", "content": "ok"}],
            }
        )
    return msgs


def _block_text(messages: list[dict[str, object]], idx: int) -> str:
    out, _store = compress_messages(messages)
    return out[idx]["content"][0]["content"]  # type: ignore[index]


def test_a_huge_untrusted_block_cannot_change_a_trusted_one() -> None:
    """The isolation property, stated as an equality rather than a threshold.

    The trusted block must compress to *exactly* the same bytes whether or not a 4000-line
    attacker-controlled block sits beside it, in either order. An equality is the right form
    here: any shared budget, however generous, would make the two outputs differ.
    """
    alone = _block_text(_convo([TRUSTED]), 2)
    attacker_first = _block_text(_convo([ATTACKER, TRUSTED]), 4)
    attacker_second = _block_text(_convo([TRUSTED, ATTACKER]), 2)

    assert alone == attacker_first, "a preceding untrusted block changed the trusted block"
    assert alone == attacker_second, "a following untrusted block changed the trusted block"
    assert DECISION in alone, "the fixture stopped exercising the property — no DECISION kept"


def test_untrusted_bulk_does_not_evict_the_trusted_decision() -> None:
    """The same property read the way it actually matters: the load-bearing line is still
    there in what we forward, with an attacker block three orders of magnitude larger."""
    out, _store = compress_messages(_convo([ATTACKER, TRUSTED]))
    assert DECISION in json.dumps(out)


def test_one_block_cannot_starve_another_within_a_single_message() -> None:
    """Two tool_result blocks in ONE user message — the shape where a per-message budget
    would be most tempting to implement, and would break isolation."""
    msgs = [
        {"role": "go", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a", "name": "bash", "input": {}},
                {"type": "tool_use", "id": "b", "name": "bash", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": ATTACKER},
                {"type": "tool_result", "tool_use_id": "b", "content": TRUSTED},
            ],
        },
        {"role": "user", "content": "next"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c", "name": "bash", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "ok"}]},
    ]
    out, _store = compress_messages(msgs)
    assert DECISION in json.dumps(out)


@pytest.mark.parametrize(
    ("name", "messages", "needle"),
    harness._adversarial_cases(),
    ids=[c[0] for c in harness._adversarial_cases()],
)
def test_load_bearing_line_is_never_silently_lost(
    name: str, messages: list[dict[str, object]], needle: str
) -> None:
    """Every COMA-class case: the genuine line survives, or it is recoverable through a
    handle distil issued *and* its stub declares the elision. Never silently gone."""
    ok, detail = harness._check_load_bearing(messages, needle)
    assert ok, f"{name}: {detail}"


def test_dedup_baiting_is_the_case_that_needs_reversibility() -> None:
    """A finding, pinned so it cannot regress unnoticed in either direction.

    Shape-based noise dedup normalises digits away, so 300 attacker lines that differ from the
    real error only in a shard number share its shape and collapse with it. The keep policy
    does lose that line — this is the one case in the battery where it does — and only
    reversibility saves it. If a future change makes it survive outright, that is an
    improvement, but it should be a deliberate one, so this asserts the current shape rather
    than a bound.
    """
    cases = {c[0]: c for c in harness._adversarial_cases()}
    _name, messages, needle = cases["dedup_baiting"]
    compressed, store = harness._compress(messages)
    blob = json.dumps(compressed)

    assert json.dumps(needle)[1:-1] not in blob, (
        "dedup baiting no longer evicts the real error line — good, but update this test "
        "and ADR 0009, which document it as the residual risk"
    )
    recovered = [h for h in store.handles if needle in store.expand(h)]
    assert recovered, "the evicted line is not recoverable from any handle — silent loss"
    assert harness._STUB_RE.search(blob), "no stub declares the elision to the model"


def test_a_forged_handle_is_never_vouched_for() -> None:
    """A tool result can print anything, including distil's own stub syntax. distil must not
    adopt a handle it did not issue: the store is keyed by content address, so a fabricated
    handle simply does not resolve."""
    cases = {c[0]: c for c in harness._adversarial_cases()}
    _name, messages, _needle = cases["handle_forging"]
    _compressed, store = harness._compress(messages)

    assert "deadbeef" not in set(store.handles)
    with pytest.raises(KeyError):
        store.expand("deadbeef")


def test_decoy_flooding_costs_savings_not_correctness() -> None:
    """The honest characterisation of the decoy attack: an attacker who floods a block with
    fake DECISION lines cannot evict the real one (verdict lines are exempt from dedup and
    always kept), but they CAN drive that block's savings to zero. Denial-of-savings is a
    real outcome and is stated as such rather than reported as a clean pass."""
    cases = {c[0]: c for c in harness._adversarial_cases()}
    _name, messages, needle = cases["decoy_verdict_flood"]
    compressed, _store = harness._compress(messages)

    assert json.dumps(needle)[1:-1] in json.dumps(compressed), "the real verdict was evicted"
    savings = 1.0 - len(json.dumps(compressed)) / len(json.dumps(messages))
    assert savings < 0.05, (
        "decoy flooding now compresses — the keep policy changed; re-check that the real "
        "verdict still survives for the right reason"
    )

"""Generate corpus/agent-worklog.json — a trajectory that exercises the state probes.

Why this has to exist: the artifact-state, continuation and overclaim probes ship
default-on, but every existing corpus trajectory carries logs, JSON, prose or HTML.
Running them over the whole corpus finds **4 file operations and 0 stated
obligations**, so all three report a perfect score against almost nothing. That is
the same failure the HTML transform had before `corpus/web-research.json` existed:
a probe certified by no evidence, reporting green.

This trajectory is a multi-turn coding agent that:

  * creates, edits, reads and finally DELETES files — so the artifact ledger has real
    state transitions, including the create-then-delete pair whose loss produces the
    phantom-file failure `distil.artifacts` exists to catch;
  * keeps a running plan with `[x]` / `[ ]` items that move between turns — so
    `distil.continuation` has pending work whose disappearance is measurable;
  * hedges some of its findings ("approximately 4200 ms", "may be a race") — so
    `distil.overclaim` has qualified claims to strip.

The DECISION marker each turn keeps it compatible with the existing bench oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are an autonomous coding agent working in a Python repository. You keep a "
    "running plan and update it as you go.\n"
    "Operating rules:\n"
    "- Re-read a file before editing it; never edit from memory.\n"
    "- Keep the plan current: mark items [x] as they complete.\n"
    "- State uncertainty explicitly rather than guessing.\n"
    "DECISION: when the plan has no remaining [ ] items, stop and report."
)

TOOLS = (
    "AVAILABLE TOOLS (json schema, abbreviated):\n"
    '- Read(file_path: string) -> {"content": string}\n'
    '- Write(file_path: string, content: string) -> {"ok": bool}\n'
    '- Edit(file_path: string, old: string, new: string) -> {"ok": bool}\n'
    '- Bash(command: string) -> {"stdout": string, "exit": int}\n'
    "DECISION: prefer Edit over Write when the file already exists."
)

PLANS = [
    "PLAN\n- [ ] add a retry wrapper in net/retry.py\n"
    "- [ ] wire it into net/client.py\n- [ ] delete the dead net/legacy_retry.py\n"
    "- [ ] add a regression test",
    "PLAN\n- [x] add a retry wrapper in net/retry.py\n"
    "- [ ] wire it into net/client.py\n- [ ] delete the dead net/legacy_retry.py\n"
    "- [ ] add a regression test",
    "PLAN\n- [x] add a retry wrapper in net/retry.py\n"
    "- [x] wire it into net/client.py\n- [ ] delete the dead net/legacy_retry.py\n"
    "- [ ] add a regression test",
    "PLAN\n- [x] add a retry wrapper in net/retry.py\n"
    "- [x] wire it into net/client.py\n- [x] delete the dead net/legacy_retry.py\n"
    "- [ ] add a regression test",
]

OBSERVATIONS = [
    'Write(file_path="net/retry.py", content="def with_retry(fn, attempts=3): ...")\n'
    '-> {"ok": true}\n'
    "Created net/retry.py with an exponential backoff helper. Measured a baseline of "
    "approximately 4200 ms for the failing call path; the variance suggests this may "
    "be a race rather than a fixed timeout.\n"
    "DECISION: the wrapper exists, so mark it done and move to wiring.",
    'Read(file_path="net/client.py")\n-> {"content": "class Client: def fetch(self): ..."}\n'
    'Edit(file_path="net/client.py", old="def fetch", new="@with_retry\\n    def fetch")\n'
    '-> {"ok": true}\n'
    'Write(file_path="net/scratch_bench.py", content="# throwaway timing harness")\n'
    '-> {"ok": true}\n'
    "Modified net/client.py to route fetch through the wrapper, and created "
    "net/scratch_bench.py as a throwaway timing harness. Roughly 12 call sites "
    "were affected; at least 3 of them are in tests.\n"
    "DECISION: wiring is complete, so mark it done and remove the dead module.",
    'Bash(command="rm net/legacy_retry.py")\n-> {"stdout": "", "exit": 0}\n'
    "Deleted net/legacy_retry.py — nothing imports it any more, confirmed with grep. "
    "The old module reportedly handled a case the new one does not, but I could not "
    "reproduce it.\n"
    "DECISION: the dead module is gone, so mark it done and write the test.",
    'Write(file_path="tests/test_retry.py", content="def test_retries(): ...")\n'
    '-> {"ok": true}\n'
    'Bash(command="rm net/scratch_bench.py")\n-> {"stdout": "", "exit": 0}\n'
    "Created tests/test_retry.py covering the backoff path, and deleted "
    "net/scratch_bench.py now that the timing work is done. The suite runs in about "
    "1800 ms locally.\n"
    "DECISION: every plan item is complete, so stop and report.",
]

NOISE = [
    "RETRIEVED (speculative context, similarity=0.59): 'Choosing a Logging Library' — "
    "a comparison of structured logging backends, sink fan-out and sampling. Nothing "
    "about retries or the client under change.",
    "RETRIEVED (speculative context, similarity=0.55): 'Kubernetes Pod Disruption "
    "Budgets Explained' — voluntary vs involuntary disruption and rollout safety. "
    "Unrelated to this module.",
    "RETRIEVED (speculative context, similarity=0.61): 'A Short History of Semantic "
    "Versioning' — pre-release identifiers and range syntax. Not applicable here.",
    "RETRIEVED (speculative context, similarity=0.57): 'Designing Office Wifi' — AP "
    "placement and channel planning. Included on a spurious overlap with 'client'.",
]

QUESTIONS = [
    "Start on the retry work.",
    "Good — now wire it in.",
    "Clean up the dead module.",
    "Finish with a test.",
]

# A past assistant answer, as it actually re-enters context on the next turn: long.
# These were one-line summaries, which made them unrepresentative of the thing output
# digestion exists to compress AND left the output surface untestable — the digest only
# touches HISTORY blocks of >= 6 lines, every block here was 5, so `distil fidelity`
# reported a scored output surface that had never digested anything. The probe was
# grading untouched text against itself at a guaranteed 100%.
#
# Each entry carries what the three output-surface probes need: a file operation, a
# hedged claim, and a plan item. Anything less and the surface is exercised but not
# actually measured on the properties it reports.
HISTORY = [
    "assistant: created the retry wrapper in net/retry.py; it caps attempts at approximately 3 for now.\n"
    "\n"
    "What it does:\n"
    "  - wraps the transport call in a bounded retry with jittered backoff\n"
    "  - caps total attempts at 3, which should hold for the p99 case\n"
    "  - leaves the timeout budget alone; that is roughly 4200 ms today\n"
    "\n"
    "Baseline latency looks approximately unchanged in the smoke run, though I only\n"
    "sampled a handful of requests, so treat that as indicative and not measured.\n"
    "\n"
    "- [ ] wire the wrapper into the client fetch path",
    "assistant: wired the wrapper into the client's fetch path in net/client.py; this likely covers 2 of the call sites.\n"
    "\n"
    "Changes:\n"
    "  - fetch() now calls through the retry wrapper instead of the raw transport\n"
    "  - retry budget is shared per request, not per hop\n"
    "  - error classification is unchanged, so callers see the same exceptions\n"
    "\n"
    "The pagination path in myapp/pagination.py probably needs the same treatment,\n"
    "but I have not confirmed whether it goes through fetch() at all.\n"
    "\n"
    "- [x] wire the wrapper into the client fetch path\n"
    "- [ ] remove the legacy retry module",
    "assistant: removed the legacy retry module at net/legacy_retry.py after finding at most 1 stale reference.\n"
    "\n"
    "Before deleting I checked for importers:\n"
    "  - grepped the tree for legacy_retry and found no live references\n"
    "  - the only hit was a comment in net/client.py, now stale\n"
    "  - at most one caller could have been missed if it imports dynamically\n"
    "\n"
    "I also deleted net/scratch_bench.py, which was a throwaway harness I added\n"
    "earlier in this session and is no longer useful.\n"
    "\n"
    "- [x] remove the legacy retry module\n"
    "- [ ] add a regression test for the retry path",
]


def build() -> dict:
    turns = []
    for i in range(4):
        blocks = [
            {
                "id": "system",
                "kind": "system",
                "stability": "stable",
                "decision_relevant": True,
                "text": SYSTEM,
            },
            {
                "id": "tools",
                "kind": "tools",
                "stability": "stable",
                "decision_relevant": True,
                "text": TOOLS,
            },
        ]
        for h in range(i):
            blocks.append(
                {
                    "id": f"hist-{h}",
                    "kind": "history",
                    "stability": "settling",
                    "decision_relevant": False,
                    "text": HISTORY[h],
                }
            )
        blocks.append(
            {
                "id": f"plan-{i}",
                "kind": "history",
                "stability": "volatile",
                "decision_relevant": True,
                "text": PLANS[i],
            }
        )
        blocks.append(
            {
                "id": f"obs-{i}",
                "kind": "tool_output",
                "stability": "volatile",
                "decision_relevant": True,
                "text": OBSERVATIONS[i],
            }
        )
        blocks.append(
            {
                "id": f"doc-{i}",
                "kind": "retrieved",
                "stability": "volatile",
                "decision_relevant": False,
                "text": NOISE[i],
            }
        )
        blocks.append(
            {
                "id": f"user-{i}",
                "kind": "user",
                "stability": "volatile",
                "decision_relevant": False,
                "text": QUESTIONS[i],
            }
        )
        turns.append({"index": i, "blocks": blocks})

    return {
        "id": "agent-worklog",
        "model": MODEL,
        "_note": (
            "A 4-turn coding agent that creates, edits, reads and deletes files while "
            "maintaining a running [x]/[ ] plan and hedging some of its findings. It "
            "exists because the artifact-state, continuation and overclaim probes ship "
            "default-on while the rest of the corpus contains 4 file operations and 0 "
            "stated obligations between them — so all three reported a perfect score "
            "against almost no evidence. In particular the create-then-delete of "
            "net/legacy_retry.py is the phantom-file case: if compression keeps the "
            "creation and drops the deletion, string recall stays 100% while artifact "
            "state fidelity goes to 0."
        ),
        "turns": turns,
    }


if __name__ == "__main__":
    out = Path("corpus/agent-worklog.json")
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

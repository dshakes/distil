"""Strands Agents — compress an agent's messages in-process.

    python examples/python_strands.py

Strands content is a list of blocks: {"text": ...} for prose and
{"toolResult": {...}} for tool output. Both shapes are handled natively, and the
block wrapper (including sibling keys like cache markers) is preserved.

distil never imports strands — the integration is duck-typed — so this runs with
strands NOT installed.
"""

from __future__ import annotations

from distil.integrations.strands import compress_messages

BIG = "\n".join(f"test_module_{i}.py::test_case PASSED" for i in range(400))

messages = [
    {"role": "user", "content": [{"text": "did the suite pass?"}]},
    {
        "role": "user",
        "content": [{"toolResult": {"status": "success", "content": [{"text": BIG}]}}],
    },
]

out = compress_messages(messages)
inner = out[1]["content"][0]["toolResult"]["content"][0]["text"]
print(f"tool result: {len(BIG):,} -> {len(inner):,} chars")
print(f"status preserved: {out[1]['content'][0]['toolResult']['status']!r}")

# Or hook it into the agent loop so it happens on every model call:
#
#     from distil.integrations.strands import compressing_hook
#     agent = Agent(model=..., hooks=[compressing_hook()])

# User PROSE is Tier-0 lossless only — never a digest stub. Putting the human's
# own words behind a handle would widen what the certificate covers.
assert out[0]["content"][0]["text"] == "did the suite pass?"
print("user prose left byte-exact (lossless tier), as designed")

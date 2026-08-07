"""Embed distil in your own agent — no proxy, no daemon.

    python examples/python_library.py

The proxy (`distil wrap` / `distil proxy`) is still the zero-config path, and it
is the one that reaches the reversible DIGEST tier. Use this module's approach
when you are building the agent, you already hold the message list, and you want
compression without a process to keep alive.
"""

from __future__ import annotations

from distil import compress_messages, expand_handle

# A tool result of the shape that actually costs money: long, repetitive, and
# re-sent on every turn of the loop.
TOOL_OUTPUT = (
    "\n".join(
        f"2026-08-07T12:00:{i % 60:02d}Z INFO  worker.pool  heartbeat ok seq={i} latency_ms=12"
        for i in range(400)
    )
    + "\nFATAL  worker.pool  pool exhausted after 400 heartbeats\n"
)

messages = [
    {"role": "system", "content": "You are a site reliability engineer."},
    {"role": "user", "content": "Why did the worker pool die?"},
    {"role": "tool", "content": TOOL_OUTPUT},
]

result = compress_messages(messages)

print(f"tokens : {result.tokens_before:,} -> {result.tokens_after:,}")
print(f"saved  : {result.tokens_saved:,} ({result.saved_pct:.1f}% smaller)")
print(f"handles: {result.handles}")

# The input is never mutated, and unchanged messages come back by identity.
assert messages[2]["content"] == TOOL_OUTPUT
assert result.messages[0] is messages[0]

# Now send `result.messages` to your model exactly as you would have sent
# `messages`:
#
#     client.messages.create(model=..., messages=result.messages)

# Reversibility is the contract. Every handle resolves to the original bytes —
# from this process, from another process, and after a restart, because the
# restore store is on disk and shared with the proxy and the MCP server.
if result.handles:
    original = expand_handle(result.handles[0])
    assert original is not None
    assert original in TOOL_OUTPUT
    print(f"\nexpand_handle({result.handles[0]!r}) recovered {len(original):,} bytes, byte-exact")

# On a flat-rate subscription, or any time you want zero digest stubs, ask for
# lossless-only. Fewer savings, nothing behind a handle.
lossless = compress_messages(messages, verbatim=True)
print(f"\nverbatim=True: {lossless.saved_pct:.1f}% smaller, {len(lossless.handles)} handles")

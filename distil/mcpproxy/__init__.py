"""``distil mcp`` — a transparent MCP proxy that compresses tool definitions and results.

Sits between any MCP client and one or more MCP servers (stdio). Tool definitions are
compressed at an explicit, named level; tool results are optionally digested with the
same recoverable digest distil's LLM proxy uses. See ``docs/adr/0017-mcp-compressor.md``
for the design and ``docs/research/mcp-compressor-protocol.md`` for how each level's
accuracy is certified.

Modules
-------
* ``levels``  — pure transforms: L0 lossless canonicalisation, L1 extractive summary,
  L2 lazy surface, L3 adaptive pinning, R result compression.
* ``proxy``   — JSON-RPC session, stdio backends, the stdio serve loop.
* ``events``  — content-free local event log + per-server learned state.
* ``install`` — rewrite a client's MCP config to route through the proxy, exact undo.
* ``watch``   — ``distil mcp watch`` terminal view and the webdash ``/mcp`` page.
* ``bench``   — the pre-registered tool-use accuracy harness (dry-run by default).
"""

from __future__ import annotations

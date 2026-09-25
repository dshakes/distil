# 0013 — Unused connectors are Claude Code's to defer, and distil stops switching that off

- **Status:** accepted
- **Date:** 2026-09-24
- **Relates to:** `distil/onboard.py` (`AGENT_PRESETS["claude"]`), `distil/proxy.py`
  (`_mcp_record`, `tools_deferred`), `distil/discover.py` (`unused_connectors`),
  `benchmarks/lazy_tools_model.py`, ADR 0012 (tool definitions are left alone)
- **Artifact:** `benchmarks/results/2026-09-24/lazy_tools_model.json`

## Context

ADR 0012 measured tool definitions at 35.6% of billed dollars on the maintainer's
traffic, 98.6% of it served as cache reads, and ruled out rewriting definitions in
place: the lossless ceiling is about 1.1% of the bill, and any byte drift turns
0.1x reads into 1.25x writes. What was left open is the larger, lossy lever that
atlassian-labs/mcp-compressor takes: do not send a definition the model does not
need.

**What mcp-compressor actually does.** It is an MCP *server-side* proxy. It connects
to the backend servers and exposes, per server, `<server>_get_tool_schema` and
`<server>_invoke_tool` (plus `<server>_list_tools` at its `max` level), with the tool
listing folded into the wrapper's description at four levels (`low` = name, args,
full description; `medium`, the default = first sentence; `high` = name and args;
`max` = name only). The model asks for a schema, then calls `invoke_tool`, which the
compressor forwards to the backend. What is lost: the model sees every tool through
one generic `invoke_tool` signature, so it can no longer be steered by per-tool
schemas without the extra lookup, and every tool it uses costs at least one extra
round-trip. It can do this because it *is* the MCP server; it executes the call.

distil is not the MCP server. It sits between the agent and the model API, and the
client executes tools. A distil version therefore has to hide definitions and
re-inject them on request (a `distil_tools` meta-tool, recovered in-proxy the way
`distil_expand` is). And because the tools array is the first thing in the prompt,
every unlock rewrites the whole cached prefix.

**The finding that decided this.** Anthropic's API has native deferred tool loading
(`defer_loading: true` plus a tool-search tool; the API excludes deferred tools from
the prefix and expands a `tool_reference` inline, so the cache is untouched), and
Claude Code uses it for MCP tools by default. But Claude Code's documentation says it
**turns tool search off when `ANTHROPIC_BASE_URL` points to a non-first-party host,
since most proxies don't forward `tool_reference` blocks**. `distil wrap` and
`distil default --always-on` both set exactly that. The ~390k-token single-connector
arrays in ADR 0012's data are not something Claude Code does to its users. distil
caused them.

Confirmed on the data. On the three large sessions, the smallest billed input
(425,825 tokens) is about the recorded tools (388,772) plus system plus messages. The
definitions were billed, not deferred.

## Measurement

`benchmarks/lazy_tools_model.py` on 14,253 priced requests (2026-09-16 to 09-24,
$2,617.60 list price). MCP-definition tokens are charged against where they sit in the
prompt (cache read first, then write, then uncached), scaled to the provider's billed
input. Discovery events come from the agent's own transcripts, using tool names only.
Each distinct MCP tool per transcript counts as one event (42 over 7,158 assistant
turns). Only transcript lines timestamped inside the
window are counted. Server names in the artifact are anonymized (`server_NN`).
To reproduce: `--window 2026-09-16T03:40:40+00:00 2026-09-24T20:34:27+00:00`.

| | share of bill |
|---|---|
| MCP definitions billed (low = named top-24 per request, high = + unnamed remainder) | 7.6% – 24.5% |
| **native tool search**, net of name index + one extra round-trip per discovery | **7.3% – 24.2%** |
| distil lazy-tools, net of the same + one full-prefix cache rewrite per unlock | 5.0% – 22.0% |

Prefix rewrites cost distil lazy-tools $59.30 over the window, against $5.42 for the
extra round-trips the native mechanism pays. If each MCP *call* were an unlock rather
than each distinct tool (7.64x more events), the rewrites alone reach about $453. That
is more than the low-bound saving, so lazy-tools goes net negative, while native stays
positive.

## Decision

1. **Do not build distil lazy-tools.** Taken alone, it clears the 5% bar this work set
   (5.0% at the low bound), but only just: under the per-call sensitivity it goes
   negative. More to the point, it is redundant. On the same traffic, the native
   mechanism beats it by about 2.3 points of the bill at the low bound. It has no prefix-rewrite
   exposure, it is the path the model was trained on, and it needs no tool injection.
   That last point matters because injection is what the subscription-safe default
   exists to avoid.
2. **Stop disabling the native one.** The `claude` wrap preset sets
   `ENABLE_TOOL_SEARCH=true` in the child environment, through `setdefault`, so a
   user's own `ENABLE_TOOL_SEARCH=false` still wins. This is Claude Code's own
   first-party default, not a distil transform, so it applies on subscription traffic
   too. `tests/test_tool_search_passthrough.py` pins that `defer_loading`,
   `tool_reference` blocks and the beta header reach the provider byte-identical with
   the digest path active.
3. **Account for it honestly.** A `defer_loading: true` definition is not billed, so
   the request record no longer counts it as overhead (`tools_deferred` counts it
   instead). Otherwise dissect and discover would report 390k tokens of overhead that
   nobody paid, and the calibrator would learn a factor from an estimate the bill
   never saw.
4. **Say which connectors are dead weight.** The request record gains `mcp_servers`
   (definition tokens per `mcp__<server>`) and `mcp_called` (servers the
   conversation's own tool_use history has called). Both are names only, never
   schemas or arguments. `distil discover` reports a server sent on 20 or more
   requests and never called anywhere in the window, priced at its cache position.

5. **The always-on install does the same.** `distil default --always-on` adds
   `ENABLE_TOOL_SEARCH=true` beside its `ANTHROPIC_BASE_URL` pin in Claude Code's
   user settings, but only where the key is absent. It records the file in
   `settings-added.json` in the distil home. `--undo`, `offboard` and the `sh`
   escape hatch remove it only from recorded files, and only while it still holds
   `true`. Ownership is recorded only after the settings write succeeds, and forgotten
   only after the key is gone. A key the user deleted is never re-added. The pin and
   the key are one atomic read-modify-write.

## Reopen when

- A client with **no native deferral** is behind distil, with its MCP definitions at
  5% or more of the bill after its own tool search (the non-Claude-Code agents in
  `AGENT_PRESETS`, or the Agent SDK without `defer_loading`). Rerun the model on that
  traffic. The distil design would still need the rewrite cost above to clear the bar.
- A live session shows Claude Code with `ENABLE_TOOL_SEARCH=true` failing through
  distil. The passthrough test is a stub, and **no live run backs decision 2 yet**
  (this change was made without a model call). Revert the preset first, then
  investigate.

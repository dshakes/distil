# 0012 — Tool definitions are left alone

- **Status:** accepted
- **Date:** 2026-09-24
- **Relates to:** `benchmarks/tool_schema_share.py`, `benchmarks/results/2026-09-24/tool_schema_share.json`, ADR 0008 (the cache contract), `distil discover`

## Context

Coding agents send their full tool definitions (built-ins plus every connected MCP
server) on every request, and distil has never touched the request `tools` array.
Some tools do compress them, and Atlassian's mcp-compressor rewrites MCP tool schemas.
The question was whether distil should too.

A past audit invented three gaps from synthetic data, so this one was measured before
anything was built. There are two inputs:

1. **What tool definitions cost.** `distil wrap` already records, per request, the
   heuristic token count of every tool definition (`tools_tokens`) next to the
   provider's own `input` / `cache_read` / `cache_creation` usage. The script puts each
   request's tool share on the billed scale. Because tools come first in the cache
   prefix (tools, then system, then messages), it assigns tool tokens to cache reads
   before cache writes, and then prices every request with `distil.pricing`.
2. **How much of a tool array can be removed losslessly.** One real Claude Code
   request's `tools` array was captured with a local sink that stores only the
   `tools` key and never forwards anything. It has 173 tools: 29 built-ins plus 12 MCP
   servers. Only two transforms are provably lossless for the model:
   - dropping the `$schema` dialect URI;
   - dropping a property `title` that restates the property's name.

   `additionalProperties: false` is not one of them, because strict tool use requires
   it. Descriptions, enums, defaults and bounds all change what the model does, so
   they are not candidates either.

## Measurement

Values are from `benchmarks/results/2026-09-24/tool_schema_share.json`. The run covered
12,722 successful requests on one machine, about 3.10B billed input tokens and about
$2,441 of list-price spend.

| | |
|---|---|
| Tool definitions, share of billed input tokens (aggregate) | 39.6% |
| Tool definitions, share per request (median) | 8.8% |
| Tool tokens served from cache reads (billed at 0.1x) | 98.6% |
| Tool tokens written to cache (1.25x) / uncached | 1.4% / 0.0% |
| Tool definitions, share of billed dollars | 35.6% |
| Lossless reduction available on a real tools array | 3.2% |
| **Expected saving of a perfect, byte-stable lossless compactor** | **1.1% of the bill** |

The gap between 39.6% (aggregate) and 8.8% (median) is real. A few sessions had one
claude.ai connector that sent tool arrays of about 390k tokens on every turn, and those
sessions account for most of the tool tokens. In the other sessions, tool definitions
are about 10% of input.

## Decision

distil does not rewrite tool definitions. The request `tools` array is forwarded
byte-for-byte as the client sent it. The only exception is distil's own
`distil_expand` tool, which is appended when recoverable digest is on.

## Why

- **The lossless ceiling is too small.** Tool definitions are a large share of the
  bill, but the part that can be removed without changing what the model can call is
  about 3%. That puts the whole feature at about 1.1% of billed cost, even when it is
  perfect. It is below the 2% bar a default-on transform has to clear.
- **The downside is much larger than the upside.** Tools lead the cache prefix, and 98.6%
  of tool tokens are cache reads. One non-deterministic byte in a rewrite turns every
  cache read into a cache write at 12.5x the price. Distil has measured this failure
  before: the digest recency bug doubled cost (ADR 0008). A rewrite could save at most 1.1% of
  the bill, and a mistake in it could cost many times that. Leaving the array alone
  has no such risk.
- **The real lever is choosing tools, not compressing them.** The large numbers come
  from connectors the session never calls. Dropping or deferring those is a behaviour
  change that only the user can make. `distil discover` already ranks tool/MCP
  definitions by what they cost per week and points at the setting to change.

## Consequences

- There is no `--compact-tools` flag and no default-on schema transform. That leaves
  nothing to opt out of and no way for it to break the cache.
- Re-run `benchmarks/tool_schema_share.py --tools <captured.json>` if the inputs
  change. The first input is tool arrays whose lossless headroom (`$schema`, repeated
  titles) is large enough that the product clears 2%. The second is a provider that
  stops re-rendering tool schemas, which would make request-side whitespace billable.
  If the result clears 2%, revisit this ADR with that artifact.

## Limits of this measurement

- It covers one machine and one maintainer's sessions. Token counts are distil's
  heuristic scaled to the provider's billed totals. In the high-tool-share sessions,
  billed tokens were close to the heuristic estimate (ratio about 1.0), so the tools
  are billed and not deferred.
- The lossless headroom comes from one captured tools array, not from the connector
  that dominates the aggregate. For the decision to flip, the headroom on that
  connector would need to be about 1.8x what was measured.
- The capture file is not committed because it contains third-party tool text. The
  artifact keeps only the counts.

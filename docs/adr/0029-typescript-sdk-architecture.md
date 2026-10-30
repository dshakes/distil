# ADR 0029 — TypeScript SDK architecture: pure-TS port over WASM or proxy client

- Status: Accepted
- Date: 2026-10-29
- Deciders: distil maintainers

## Context

JS/TS agent builders want an in-process `compress(messages, {provider})` call —
no daemon, no proxy, no network — with typed message shapes for Anthropic and
OpenAI APIs. The existing `distil-llm` npm package ships this in plain CommonJS
JavaScript (`packaging/npm/lib/tier0.js`, `compress.js`) but has no native
TypeScript source, no strict types, and no provider-discriminated API.

Three architectures were evaluated:

### (a) Compile `rust/distil-core` to WASM and wrap it

`distil-core` exposes `minify_json`, `collapse_runs`, and `count_tokens` via
PyO3 bindings (`pyo3 = { version = "0.29" }`). There are no `wasm-bindgen`
dependencies in `Cargo.toml` and no WASM build target. Compiling to WASM would
require:

- replacing or adding `wasm-bindgen` + `wasm-pack` toolchain
- re-wiring the build pipeline for both Python and WASM outputs
- shipping a WASM binary with every npm install (~100 kB for a compression core
  that three hundred lines of TS can also cover)

Crucially, `distil-core` only covers **three helper primitives** — not the
full tier-0 pipeline (`applyTier0`, the reject-if-bigger guard, the message
traversal, the provider-specific content-block handling). WASM would not
eliminate the TS layer; it would only replace the inner loops. The maintenance
surface doubles with no gain in correctness — the Python test already enforces
byte-identity of the JS port on every CI run.

**Rejected.**

### (b) Port the lossless transforms to TypeScript — chosen

The JS port in `packaging/npm/lib/tier0.js` is already proven byte-identical
to the Python engine by `tests/test_ts_conformance.py`. The TypeScript SDK
(`sdk/typescript`) is a strict-typed restatement of the same algorithm: same
control flow, same edge-case handling, same reject-if-bigger guard. No new
semantics are introduced.

One source of truth for compression semantics is the Python engine. The JS/TS
port is the *verified approximation* — byte-identical on the certified corpus,
closed by the transitive proof:

```
Python _apply_tier0
  ≡ (tests/test_ts_conformance.py)
    packaging/npm/lib/tier0.js applyTier0
      ≡ (sdk/typescript golden-parity vitest suite)
        sdk/typescript/src/tier0.ts applyTier0
```

The TS SDK adds over the existing npm package:

- Native TypeScript source (strict mode, `noUncheckedIndexedAccess`,
  `exactOptionalPropertyTypes`)
- Provider-discriminated `compress(messages, { provider })` API for Anthropic
  and OpenAI message shapes — including content-block arrays, not just string
  content
- Typed `CompressReceipt` (content-free: token counts only, no content, no
  handles)
- ESM + CJS dual output via `tsc`

**Accepted.**

### (c) Thin client to a local distil proxy

Requires a running daemon, a network round-trip, and a separate authentication
story. Not in-process.

**Rejected.**

## Decision

Build `sdk/typescript` as a pure TypeScript port of the tier-0 lossless
transforms, verified byte-identical to the Python engine via the existing
conformance infrastructure.

## Parity coverage and honest gaps

**Covered:**

| Transform | TS SDK | Byte-identical to Python |
|-----------|--------|--------------------------|
| `minifyJson` | ✓ | ✓ (safe subset; declines integral floats / integer-like keys) |
| `collapseRuns` | ✓ | ✓ |
| `countTokens` (banker's rounding) | ✓ | ✓ |
| `applyTier0` (pipeline + reject-if-bigger) | ✓ | ✓ |
| `compress()` — Anthropic message shapes | ✓ | ✓ (routes through applyTier0) |
| `compress()` — OpenAI message shapes | ✓ | ✓ (routes through applyTier0) |

**Not covered (in-process):**

| Feature | Why not in this pass |
|---------|----------------------|
| Tier-1 reversible digest | Mints restore handles that must live in ONE store shared with the proxy and the MCP server. A second implementation would be a second thing to certify. Use the proxy for digest-tier savings. |
| Structured fold / template fold | Complex multi-pattern compaction in `compress/structured.py`; not a primitive; shares the same "one certifier" argument as Tier-1. |
| HTML extraction | `compress/htmlx.py`; not a primitive; proxy path. |
| Code skeleton | `distil/skeleton.py`; same. |
| Cache-control placement | Anthropic-specific; proxy path. |
| Query-aware salience | Requires a certified model weight; not a stateless transform. |

## Consequences

- JS/TS agent builders can call `compress(messages, { provider })` with zero
  runtime dependencies, no daemon, and full strict-type coverage.
- The in-process lossless tier carries the same decision-equivalence guarantee
  as the proxy for the lossless tier (same algorithm, same byte output,
  same conformance gate).
- The digest tier and all complex compaction strategies remain proxy-only,
  keeping one certifiable implementation.
- The `sdk/typescript` package lives alongside `packaging/npm` and is the
  TypeScript-first entry point; `packaging/npm` remains the lightweight
  CommonJS launcher for teams that don't use TypeScript.

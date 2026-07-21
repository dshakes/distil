# Competitive plan — close the Headroom gaps, press our moat

Grounded in a full audit of Headroom's source (2,122 files, Rust+Python) and
its live site/positioning (2026-07). Headroom = same wedge (reversible
context-compression proxy) but ~60.8k★ vs our 5★, with npm + a shipped HF
model + ~190 releases. Its real weakness: **no per-request correctness proof**
(offline benchmarks only), a **4.8% fleet-median** compression behind a 60–95%
headline, and a **mandatory Rust core** (missing `_core` → hard exit).

## Strategy: close distribution/capability gaps WITHOUT giving up our moat

Our moat is per-request **decision-equivalence** + **honest, calibrated
numbers** + **pure-Python low-friction install**. Every feature we copy must be
re-implemented in that frame — certified + reversible + honest — not cloned.

## Phase 1 — buildable now, highest leverage

- **1A. npm `distil` (TS proxy client + `npx distil`)** — Headroom ships a thin
  TS client (not a reimpl); we do the same: point any JS/TS SDK's baseURL at the
  proxy, and `npx distil wrap -- <cmd>`. Closes the biggest adoption gap; low
  effort. Package under `packaging/npm/`.
- **1B. Structured JSON / tool-output compressor** — the biggest real token sink
  in agent traffic. A reversible structured-JSON digest (row/field dedup,
  schema-factoring) gated by our existing decision-equivalence + expand-tool
  recovery → SmartCrusher's savings WITH the per-request proof they lack.
- **1C. Cache economics, first-class** — we already have `--session-delta` +
  cache-prefix headers; make the provider-discount math (Anthropic 0.1× cached
  reads) a default-on, documented "cache mode" story, not a buried flag.
- **1D. Positioning** — rewrite `docs/compare.html` with the REAL Headroom facts
  (4.8% median vs 60–95% headline; README accuracy claims unbacked by their
  methodology page; no runtime proof; Rust mandatory), leading with our moat;
  add `docs/llms-full.txt`; expand + market the framework matrix.

## Phase 2 — bigger, later

- tree-sitter code-aware compressor (9 langs) — matches CodeCompressor.
- revive the inert learned-salience artifacts into a shipped, certified model
  (our answer to Kompress-v2), content-free.
- subscription-OAuth passthrough for more providers (Copilot/Codex/Kimi).
- community: Discord, release cadence, Trendshift.

## Non-negotiables

- Never ship a lossy compressor without the reversible expand path + the
  decision-equivalence gate. Their edge to us is offline-only correctness; we
  must not trade our runtime proof for their raw ratios.
- Keep pure-Python install working (no mandatory native extension).
- Every published number stays calibrated + honest (no 60–95%-headline-vs-4.8%-
  reality gap).

# Changelog

All notable changes to Distil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.25.1] — 2026-06-25

Version bump only; same content as the v0.25.0 release notes — fixes the PyPI
publish that failed on a duplicate filename in v0.25.0 (the package version was
still `0.24.0`, so the wheel/sdist collided with an already-uploaded
distribution). The v0.25.0 tag and GitHub Release are intentionally left in
place. See the [v0.25.0 release](https://github.com/dshakes/distil/releases/tag/v0.25.0)
for the substantive change (Phase 5 / E7 SWE-bench Verified end-to-end eval).

## [0.24.0] — Ecosystem hooks + on-motto gap-closing

New surface area for agent frameworks and observability — every addition kept
under the decision-equivalence certificate, with the platform scope-creep
deliberately declined.

- **LangGraph hook** (`distil/integrations/langgraph.py`) — a drop-in
  `pre_model_hook()` that compresses graph state right before the model node, plus
  a `compress_state()` helper for manual use inside any node. Duck-typed (never
  imports langgraph/langchain); returns only the updated message list so every
  other state field is untouched. Joins the existing LiteLLM + LangChain hooks.
  Example: `examples/python_langgraph.py`.
- **Cache-prefix observability** — the proxy now emits
  `x-distil-cache-prefix-msgs: <n>` under `--session-delta`, exposing exactly how
  many leading messages stayed byte-identical vs the previous turn (the
  prompt-cache-read region). The verifiable benefit of a prefix-freeze router,
  content-free — distil is cache-monotonic by construction, so the prefix is real,
  not rewritten.
- **Pluggable salience scorer seam** — `salient_tokens(..., scorer=…)` accepts an
  optional callable (a semantic / NER / embedding model) whose spans are unioned
  into the model-free signals. Off by default (runtime stays model-free,
  zero-dep); a bad scorer can never break compression (guarded), and whatever it
  returns is still judged by the same certificate — the seam adds *coverage*,
  never an unverified guarantee.
- **Docs:** README now documents the framework hooks and a "Deliberately *not* a
  platform" section — why memory/knowledge-graph, hosted semantic cache, and
  editor-auth are out of scope (they can't be put under the certificate), and what
  we adopted instead because it survives the gate.

## [0.23.2] — Mobile docs, animated architecture diagram, distribution fix

- **Fixed a broken Homebrew distribution.** Both formulas (repo + tap) had frozen
  their `url`/`version` at v0.21.0 while the `sha256` advanced — so `brew install`
  failed on a sha mismatch. Root cause (a version-specific regex in the update step)
  fixed with a version-agnostic pattern; both formulas now consistent at the current
  release and verified against the published tarball.
- **Mobile-responsive docs.** Wide benchmark tables now scroll horizontally instead
  of overflowing; landing stats collapse to one column, CTAs stack, padding/typography
  scale down — across the docs site and the landing page.
- **New animated architecture diagram** (`docs/assets/architecture.svg`) — a realistic
  depiction of the pipeline (agent → compress/cache-pin/forward → provider), the
  transparent recovery loop, and the quality-contract band (certificate · shadow ·
  flywheel), with flowing-data animation. Shown on README, Concepts, Architecture.
- **Vocabulary consistency:** `distil bench` now reports savings as "reversibly"
  (the strategy uses the Tier-1 reversible digest), matching the v0.23.1 terminology.

## [0.23.1] — Honest vocabulary: "reversible" vs "lossless"

A precision pass on terminology so no claim can be read as an overclaim:
- **The default Tier-1 digest is now described as "reversible"** (byte-recoverable on
  demand), not "lossless". "Lossless" is reserved for the **byte-in-context** tier
  (Tier-0 / `--verbatim`), where the model sees content unchanged. "Lossy" stays for
  the irrecoverable competitors. All three Distil tiers remain certified
  decision-equivalent. Updated the README headline + prose, the benchmark method
  label, and added an explicit three-tier definition to the README and Concepts page.
- **`--safe`** added as a clearer alias for **`--lossless-only`** (the
  policy/subscription-safe mode: no lossy shaping, no tool injection — the reversible
  digest still runs); `--verbatim` remains the byte-in-context switch. Internal
  strategy/ladder identifiers are unchanged (no behavior change).

## [0.23.0] — GA polish: grounded docs, genuine head-to-head, recipes

A go-live pass: every customer-facing claim audited against code, every benchmark
number re-measured genuinely, and the docs made white-glove.

- **Genuine, apples-to-apples benchmarks.** Fixed the local competitor adapters so
  they actually engage: Headroom is now driven its real whole-conversation way
  (`optimize=True`) instead of no-op'ing on a per-block user message; LLMLingua-2 is
  applied to every tool result and memoised (pure function). Corrected the v0.22.0
  coding-agent competitor numbers (which were harness artifacts) to the real ones —
  **LLMLingua-2 56.8% tok / 57.2% $ / 274 ms / lossy**, **Headroom 22.4% tok /
  −16.8% $ (busts the prompt cache by default) / 5.3 ms / lossy**. distil leads on
  cache-aware dollars (91.1%) and is the only reversible method.
- **Docs claims audit (no fake, all real grounded).** Removed fabricated claims (a
  non-existent "proxy detects subscription keys" path; `ingest --format` auto-detect;
  `output-savings --mode/--runner` flags; invented `corpus.validate()` invariants),
  corrected CLI flag tables, fixed `compress_messages(verbatim=)` signature, the
  salience module path, and the 8 proxy response headers; clarified that the live
  83.2%/53.1%/35.3% run needs an API key and is not offline-reproducible.
- **Diagrams** (YC-style, on-brand): `cache-delta.svg` and `ast-delta.svg`, embedded
  in the techniques and benchmark pages.
- **White-glove "Use it on your workflow" recipes** in the README — coding and
  non-coding use cases, a see-it/prove-it table, and a config rule-of-thumb.

## [0.22.0] — Coding-agent benchmark + two correctness fixes it found

Building the messages-level coding-agent benchmark (`benchmarks/codebench.py`:
read→edit→reread sessions, cache-aware dollars vs the real headroom + llmlingua
packages) surfaced two real bugs, both now fixed:

- **Cache-delta is now cache-monotonic by construction.** `cachedelta.delta_encode`
  was rewritten as a *pure per-call walk* over the cumulative messages (message *i*
  is deduped only against messages 0..i-1). Previously the stable prefix was passed
  through as originals while the prior turn had emitted those messages as markers, so
  a re-read flipped marker→original on entering the prefix and **busted the prompt
  cache**. The pure-walk encoding emits identical bytes for the cached prefix every
  turn. (`session` is now optional — the cumulative conversation is the memory.)
- **Tier-0 never inflates tokens.** `collapse_runs` could turn a run of near-free
  blank lines into a `<<x N>>` marker that costs *more* tokens; the adapter had only
  a char-based guard. `_apply_tier0` now keeps the collapse only when it reduces the
  **token** count. (Fixes verbatim mode showing negative savings on whitespace-heavy
  content.)
- Net effect on the coding benchmark: verbatim+cache-delta went from −1.4% to a real
  **+43.8% cache-aware savings (reversible)**; plain verbatim from −3.5% to 0.0%.
  The PAYG digest remains the dominant lever (~91%). 3 regression tests (498 total).

## [0.21.0] — Edit-equivalence (decision-equivalence, made precise for code)

- **Edit-equivalence**: the decision signature now AST-normalizes code-bearing tool
  inputs (e.g. an `Edit`/`Write` `new_str`). For coding agents the decision *is* the
  edit, so two responses that make the agent write the same code with trivially
  different whitespace or comments now count as **equivalent**, while a real logic
  change still differs. This stops shadow-mode over-reporting drift and lets the
  certificate claim safe savings it previously, conservatively, could not.
- Implemented model-free with the stdlib `ast` (`_normalize_decision` → `ast.dump`),
  applied through shared signature builders so the JSON, streamed (SSE), and
  chunk-array paths all stay consistent. Non-code strings and non-Python pass
  through untouched. 5 tests (495 total). ruff clean, verify + bench PASS.

## [0.20.0] — AST-structural delta (the deepest cache-delta layer)

- **AST-structural delta** (`astdelta.py`, stdlib `ast`, model-free): for Python,
  cross-version delta now diffs by *parsed structure*. Each top-level definition is
  fingerprinted with `ast.dump` (attributes off) — invariant to whitespace,
  comments, and import order. A reformat-only re-read is recognised as "no
  definition changed" and referenced; only definitions whose AST actually changed
  are sent in full. Textual diff explodes on reformatting; the structural delta
  isolates exactly what changed.
- Wired as the preferred near-duplicate path in `cachedelta.py` (the `--session-delta`
  feature); non-Python or unparseable (mid-edit) source falls back to the textual
  unified diff, so it never fails a request. Decision-equivalent (unchanged defs are
  still in cached context) and reversible (`distil_expand` recovers the full file).
- 8 tests. Full suite 490 passed, ruff clean, verify + bench PASS.

## [0.19.0] — Cache-delta context coding (cross-version delta)

The coding-agent moat. The hot path is read → edit → **re-read**, and the re-read
file is a *near-duplicate* (one hunk changed), so exact-duplicate dedup misses it
and re-sends the whole file. Cache-delta coding (`cachedelta.py`,
`distil proxy --session-delta`, opt-in) sends only the diff:

- **Cross-version delta** — a re-read-after-edit is replaced by a reference to the
  prior version + a unified diff of what changed; exact re-sends become a compact
  back-reference. Both confined to the **volatile suffix** — the stable cached
  prefix is never mutated (*cache-monotonicity*), so prompt-cache hits survive.
- **Decision-equivalent + reversible**: prior-version (still in cached context) +
  diff carries the same information for the next action; the full current version is
  kept locally and recovered byte-exact via `distil_expand`. Shadow mode measures it.
- Wired into `distil proxy` / `distil wrap` (messages format) behind `--session-delta`;
  emits `x-distil-cache-refs` / `-delta` / `-tokens-saved` headers. End-to-end a
  re-read-after-edit saved ~85% of the re-read (902 of 1063 tokens) vs re-sending whole.
- 10 tests. Full suite 482 passed, ruff clean, verify + bench PASS.

## [0.18.0] — Streaming-aware shadow mode (Claude Code / Codex / Gemini)

- **Shadow-mode now works on streaming sessions.** Real agent sessions (Claude Code,
  Codex, the Gemini CLI) stream their responses over SSE, which the previous shadow
  comparison couldn't parse — so it silently recorded nothing. `shadow.py` now
  reconstructs the decision from a streamed body: `decision_signature_from_body`
  reads a non-streaming JSON body directly and rebuilds a streamed (SSE or
  chunk-array) one via `_decision_from_chunks`, accumulating the first tool call
  across chunks for all three providers (Anthropic `input_json_delta`, OpenAI
  `tool_calls` argument deltas, Gemini `functionCall`). A streamed response yields
  the same signature as its non-streamed equivalent, so comparisons are valid.
- The proxy shadow path now compares raw bodies via `decision_signature_from_body`,
  so `distil proxy --shadow` measures live decision-equivalence on streaming traffic.
  Verified end-to-end on an SSE tool-call response.

## [0.17.0] — Decouple compression aggression from auth (`--verbatim`)

Resolves an overload introduced in 0.16.0. `--lossless-only` had been redefined to
mean "Tier-0 only," which **contradicted `policy.py`** (where the reversible digest
*is* the lossless strategy that subscription sessions use) and silently de-tuned
autonomous agents on subscription/OAuth from ~70%+ down to ~10%.

- **`--lossless-only` restored** to its policy meaning: lossless *strategies* only
  (no lossy output-shaping) + no tool injection. The reversible, certificate-backed
  Tier-1 digest **still runs** — consistent with `policy.py` and the project's
  definition of "lossless" (reversible + decision-equivalent).
- **New `--verbatim` flag** (proxy / `wrap` / gateway): skips the Tier-1 digest
  entirely (Tier-0 only) so the model sees content un-stubbed. The right mode for
  interactive (human-in-the-loop) sessions or out-of-distribution traffic. Lower
  savings, byte-in-context fidelity.
- Adapter/integration kwargs renamed to match: `compress_messages(..., verbatim=)`,
  `compress_generate_request(..., verbatim=)`; LiteLLM `distil_verbatim`; LangChain
  `compress_messages(..., verbatim=)`. Docs reconciled across CLI / adapters /
  integrations / faq / deploy-security.

## [0.16.0] — Ecosystem hooks: MCP server + LiteLLM/LangChain

- **MCP server** (`mcp_server.py`, `distil mcp`): a zero-dependency, stdlib-only
  Model Context Protocol server over stdio JSON-RPC 2.0. Exposes `distil_compress`
  (reversible digest + handle, original kept in a local on-disk store),
  `distil_expand` (recover by handle), and `distil_savings`. Wire it into any MCP
  client (Claude Desktop, IDEs, agents). The message handler is a pure function and
  is unit-tested without real stdio; the loop is verified end-to-end.
- **In-process framework hooks** (`integrations/`): LiteLLM (`compress`/`completion`/
  `acompletion`) and LangChain (`compress_messages`, duck-typed over message objects
  *and* dicts) compress requests before they leave the process — same reversible
  compression as the proxy, no sidecar required. Both lazy-import their framework, so
  distil stays zero-runtime-deps.

## [0.15.0] — Claude Code plugin + status line

- **`distil statusline`** (new CLI command): renders a compact one-line savings
  summary from the local ledger (tokens, dollars, runs, and live decision-
  equivalence when shadow-mode has samples). Reads the optional Claude Code status-
  line JSON on stdin for the model name; never raises.
- **Claude Code plugin** (`plugins/distil/` + `.claude-plugin/marketplace.json`):
  installable via `/plugin marketplace add dshakes/distil`. Ships a `/distil`
  command (savings report + setup help) and a `statusline.sh` that calls
  `distil statusline`. Honest scope: a plugin cannot reroute a running session or
  set the main status line from its manifest, so the README documents the one-line
  `settings.json` addition; traffic is compressed via `distil wrap` / `distil proxy`.

## [0.14.0] — Google Gemini adapter + true lossless-only

- **Gemini adapter** (`adapters/gemini.py`): the proxy, async proxy, and gateway now
  compress Google's `generateContent` request shape (`contents` / `parts` /
  `functionResponse`) — a third first-class provider alongside Anthropic and the
  OpenAI-compatible family. `text` parts get Tier-0 lossless transforms; large
  `functionResponse` string values get the Tier-1 *reversible* digest (recoverable
  via the local store); `functionCall`, `inlineData`, `fileData`, and model-authored
  text pass through untouched. Path-detected (`:generateContent` /
  `:streamGenerateContent`), so just `--upstream https://generativelanguage.googleapis.com`.
  Shadow-mode live decision-equivalence works for Gemini too. (Expand-tool injection,
  output shaping, and Gemini context caching remain messages-format-only for now.)
- **`--lossless-only` is now genuinely lossless-in-context** (GA correctness fix). It
  previously still applied the Tier-1 digest, replacing tool output the model could not
  recover (tool injection is disallowed on subscription/OAuth) with a stub — despite
  the "safe for subscription" label. It now applies only Tier-0 transforms in this
  mode, so the model sees semantically identical content. The aggressive,
  certificate-backed reversible digest remains the default (PAYG) behavior.

## [0.13.0] — Shadow-mode live decision-equivalence

- **Shadow mode** (`shadow.py`, `distil proxy --shadow RATE`, `distil shadow-stats`):
  samples a fraction of live requests, runs each one **both compressed and
  uncompressed** in a background thread (never blocking the client), and records a
  **content-free live decision-change rate** on real traffic. The continuous online
  counterpart to the offline certificate — decision-equivalence becomes observable
  in production. Decision = the agent's next `tool_use`/`tool_call`; equivalence
  iff that action matches.
- README: a "See it working — real-time savings & live equivalence" section
  (per-request headers, gateway dashboard, genuine-savings ledger, shadow mode,
  and one-env-var org-wide enforcement).

## [0.12.1] — GA hardening

Pre-GA security + correctness pass (no behavior change to the happy path):

- **Request-path safety** (`httpguard.py`, applied across `proxy`, `aproxy`, `gateway`):
  upstream-path validation (blocks `@`/`//`/`..` host-injection SSRF), defensive
  `Content-Length` parsing, an 8 MiB body cap, and a bounded async connector.
- **Crash-resistance**: `compress_messages` and `ingest` no longer raise on
  malformed-but-valid JSON (missing/non-string `text`, non-dict messages, bad
  JSONL lines) — they pass such input through untouched; the compress call in
  every proxy is additionally guarded so compression can never break a request.
- **Gateway**: tenant labels are sanitized to a safe charset (no injection into
  accounting or the dashboard) and all HTML renderers (`gateway`, `telemetry`,
  `ledger`) escape interpolated values (stored-XSS fix).
- **Correctness**: `salience.protect()` now falls back to the byte-exact original
  (never the stripped block) so a salient line is never silently dropped, and uses
  exact line membership; `structured.fold` leaves null-bearing records byte-exact
  (no null-vs-missing ambiguity); the Rust hot-path pins JSON key order to match
  the Python backend.

## [0.12.0]

The Decision-Equivalence Risk Certificate (conformal risk control, `distil conformal`),
salience protection (model-free frontier shifter), and the live head-to-head vs. the
real LLMLingua-2 / Headroom packages. See `BENCHMARKS.md`.

## [0.9.0 – 0.11.0]

Recoverable compression (`distil_expand`), the self-improving learning flywheel
(`distil learn`), and the conformal certificate foundations.

## [0.2.0]

Both sides of the bill, the proof pack, and the leapfrog tracks.

### Added
- **Output compression** — gated generation-side verbosity shaping + lossless
  output-on-re-entry digest + an A/B harness (answer-preservation gate);
  `distil output-savings`, `distil proxy --shape-output`.
- **Certified compression frontier** — `eval.py`, `distil eval`: savings-vs-
  decision-equivalence curve where every point carries its certification verdict.
- **Self-distilling keep-model** — `online.py`, `distil online`: learns from
  causal labels from your own traffic, retrains, promotes only if non-inferior.
- **Verifiable federated telemetry** — `telemetry.py`,
  `distil federated-leaderboard`: HMAC-signed, content-free savings + verdict.
- **Async high-concurrency proxy** — `aproxy.py`, `distil proxy --async` (`[async]`).
- **Rust hot-path core** — `rust/distil-core` (PyO3), `distil/native.py` with a
  pure-Python parity fallback (transparent acceleration when built).
- **Managed gateway** — `gateway.py`, `distil gateway` with a live per-tenant dashboard.
- **Real-trace ingestion** — `ingest.py`, `distil ingest` (Anthropic + OpenAI shapes).
- **Performance benchmark** — `perf.py`, `distil perf` (p50/p95).
- **Transformer keep-model** — ONNX adapter + training pipeline (`distil train-transformer`);
  verified demo checkpoint on the release.
- OpenAI `role:"tool"` messages now get the decision-aware reversible digest.

## [0.1.0]

The first end-to-end cut: compression with a quality contract.

### Added
- **Cache-aware cost engine** (`compress/cache_aware.py`) — prices a multi-turn
  agent loop and proves naive recompression busts the prompt cache.
- **Risk-graded compression** — Tier-0 provably-lossless transforms, Tier-1
  reversible digest with retrieval handles, cache stabilization (schema
  canonicalization + volatile-field extraction), reject-if-bigger invariant.
- **Causal / counterfactual pruning** (`replay/ablation.py`) — discovers
  context that never changes a decision.
- **Quality contract** — TOST non-inferiority gate (`certify/`), decision-equivalence.
- **Multi-domain trajectory corpus** (7 domains) + `distil bench` CI gate.
- **Auth-mode gating** (`policy.py`) — lossless-only on subscription/OAuth.
- **Holdout A/B** (`certify/holdout.py`) — savings with a bootstrap 95% CI.
- **Byte-fidelity gate** (`fidelity.py`) — reversibility + append-only, `distil verify`.
- **Phase-7 building blocks** — BM25 partial retrieval (`retrieval.py`), delta /
  append-only context (`delta.py`), keep-model codec (`codec/`), gist tool-schema
  caching (`gist.py`).
- **Runtime adapter** (`adapters/anthropic.py`) — compress an Anthropic Messages
  request with no caller code change.
- **Billing-grade path** — Anthropic `count_tokens` tokenizer and live
  `AgentRunner` (opt-in `distil[live]`).
- **Distributables** — PyPI wheel/sdist, Docker image, single-file `distil.pyz`,
  CI + release workflows.

# Provider parity: every mechanism, every wire format

Inventory of what distil does on each of the four request shapes it serves, before and
after the cross-provider parity pass (branch `feat/cross-provider-parity`, base 1.57.0 +
ADR 0025). The point is to make whatever compression **policy** wins ship on all four
shapes at once. This pass changes mechanisms, not policy: what gets digested, and when,
is the same as before on every shape.

Shapes: **Anth** = Anthropic Messages (`/v1/messages`), **Chat** = OpenAI Chat Completions
(`/v1/chat/completions`, Azure forms included), **Resp** = OpenAI Responses
(`/v1/responses`, what Codex speaks), **Gem** = Gemini `(stream)generateContent`.

Legend: **yes** supported · **partial** · **no** missing · **n/a** cannot exist for that
provider (reason given). `before → after` where this pass changed a cell. Line numbers
are for this branch.

## Request path

The proxy, gateway and async proxy all route by path (`httpguard.is_compressible_path`,
`adapters.gemini.is_gemini_path`). The proxy and gateway then share one serve path,
`serve_core.compress_or_forward` (`distil/serve_core.py:244`), which picks the adapter by
body shape.

| # | Capability | Anth | Chat | Resp | Gem |
|---|---|---|---|---|---|
| 1 | Tool-result digest adapter | yes `compress_messages` (`adapters/anthropic.py:1137`) | yes `compress_chat_completions` (`adapters/openai.py:380`) | yes `compress_responses_input` (`openai.py:784`) | yes `compress_generate_request` (`adapters/gemini.py:357`) |
| 2 | Exact-quote exemption | yes | yes | yes | yes (positional pairing, `gemini.py:186`) |
| 3 | Edit-quote guard (measure + widen) | yes `_guard_quotes` (`anthropic.py:1080`) | no → yes `_guard_quotes_with` (`openai.py:731`) + `provenance.chat_edit_quotes` | yes (same helper) | no → yes (`provenance.gemini_edit_quotes`; Gemini CLI's `replace` tool is an edit; `role:"model"` excluded from the observed view) |
| 4 | Census (eligibility buckets) | yes | yes | yes | partial (images only) → yes (text, exact-quote, tool output) |
| 5 | Re-fetch verbatim (ADR 0025) | yes (`anthropic.py:172`) | no → yes (`openai.py`, `_compress_openai_message`) | no → yes (`_compress_response_item`) | no → yes (`functionResponse` string leaves, `gemini.py:_json_text`) |
| 6 | Cold-point eviction (ADR 0014) | yes (`cache_control` TTL) | no → yes, `cold_candidates_chat` (`openai.py:226`), TTL `coldpoint.openai_request_ttl` (`coldpoint.py:324`; GPT-5.6+, recognised by its cache-write price row, and unpriced models are never cold) | no → yes, `cold_candidates_responses` (`openai.py:582`) | n/a — implicit caching documents no lifetime, so expiry is never certain |
| 7 | Cache-delta (`--session-delta`) | yes | partial (list-shaped tool messages skipped; re-fetch not kept out) → yes (`cachedelta._rewrite_tool_texts`, `refetch_tool_call_ids` `openai.py:484`) | no (unchanged) | no (unchanged) |
| 8 | Expand-tool injection | yes | yes | yes | yes |
| 9 | Expand loop, buffered | yes | yes | partial (re-query dropped the reasoning item) → yes (`expand.py:340`) | yes |
| 10 | Expand loop, streaming | yes, TTFT-preserving splice (`streamexpand.py:115`) | partial: buffered + re-encoded SSE (TTFT lost) — unchanged | partial: re-encoded stream had only `created`/`completed` (Codex saw an empty turn) → events per item (`streamexpand.py:400`); TTFT still lost | partial: SSE frames even without `alt=sse` → JSON array when the client asked for one; TTFT still lost |
| 11 | Prefix replay (ADR 0011) | yes | yes | yes | yes; `?key=` credential now scopes the lineage (`prefixreplay.query_credentials`, `prefixreplay.py:291`) |
| 12 | Output shaping | yes | yes | yes | yes in the proxy; async proxy no → yes (`aproxy.py`) |
| 13 | Streaming relay | yes | yes | yes | yes |
| 14 | Per-mode certification hold (ADR 0022) | yes | yes | partial (no shadow rows, see 16) → yes | yes |
| 15 | Usage scan (`scan_usage`, `streamrelay.py:102`) | yes | yes buffered; streamed only if the client sets `stream_options.include_usage` (unchanged); `cache_write_tokens` now split out | **no**: `input_tokens` read as Anthropic's exclusive count, cache read never booked → yes (`_split_openai_input`, `streamrelay.py:88`) | partial (`thoughtsTokenCount` dropped from output) → yes |
| 16 | Shadow sampling | yes | yes | no (every replay "none", dropped, still billed) → yes (`shadow.py:468`) | partial (temperature never pinned) → yes (pins an existing `generationConfig.temperature`, same never-inject rule as Anthropic) |
| 17 | Calibration estimate | yes | yes | partial (`instructions` not in overhead) → yes | partial (`systemInstruction` not in overhead) → yes; tool names found for Chat/Gemini |
| 18 | Subscription policy | yes (`policy.session_auth_mode`) | yes, process-wide | yes, process-wide | yes, process-wide |

Row 18, why no per-provider detection was added: the flat-rate logins of the other two
agents do not reach a compressible path. Codex with a ChatGPT login talks to the
`chatgpt.com/backend-api/codex/responses` backend and Gemini CLI with Google OAuth to the
Code Assist `v1internal:*` endpoints; neither matches `_RESPONSES_RE` / `_GENERATE_RE`, so
both pass through byte-for-byte. A Claude OAuth login on the machine makes the whole proxy
lossless-only, metered OpenAI/Gemini traffic included — the safe direction, left as is.

## Accounting

| # | Capability | Anth | OpenAI (Chat, Resp) | Gem |
|---|---|---|---|---|
| 19 | Price rows + cached-input discount | yes (`pricing.py`) | no (`gpt-5.2` deliberately `UNPRICED`) → yes: rows checked against developers.openai.com/api/docs/pricing 2026-10-06, read = cached/input, GPT-5.6+/6.x write 1.25x, no write surcharge before (`pricing._auto`, `pricing.py:92`) | no → yes: ai.google.dev/gemini-api/docs/pricing 2026-10-06, read = context-caching price, no write charge |
| 20 | `resolve()` id spellings | yes (`anthropic.`, `@`, snapshots) | no LiteLLM `openai/` → yes | no `models/`, `gemini/` → yes |
| 21 | Savings / receipts / census $ | yes | tokens booked, $0 → priced | tokens booked, $0 → priced |
| 22 | `distil audit` referee | yes | no in practice (unpriced, so never sampled) → yes | no (unpriced; model is in the URL) → yes (`referee.shape` falls back to `_model_from_path`) |
| 23 | What-if transcript replay | yes (Claude Code) | Codex CLI: no → yes (`transcripts/codex.py`, `whatif._compress_other` `whatif.py:342`) | Gemini CLI: no → yes (`transcripts/gemini_cli.py`) |
| 24 | `dissect` transcript correlation | yes | no → yes (registry `transcripts/__init__.py:16`) | no → yes |
| 25 | Post-tool hooks (`distil/hook.py`) | yes (Claude Code) | yes (Codex) | yes (Gemini CLI) |
| 26 | LiteLLM proxy hook | yes (`anthropic_messages`) | Chat yes; Responses no → yes (`litellm_hook.py:106`) | via Chat shape |
| 27 | LiteLLM in-process wrapper | — | Chat-shaped messages went through the Anthropic adapter, digest ignored the subscription policy → Chat adapter + the hook's policy gate (`integrations/litellm.py:28`) | via Chat shape |
| 28 | Transcript-mode spend screen (`savings_screen.transcripts_screen`) | yes | no (unchanged) | no (unchanged) |

## Still open, and why

- **TTFT-preserving streaming splice for Chat, Responses and Gemini** (row 10). The
  buffered fallback is now shape-correct for every client, but a streamed turn that
  carries a digest stub is answered all at once. The upgrade is a per-format splice in
  `streamexpand.py` (hold tool-call frames, resolve `distil_expand`, splice the
  re-query's frames with re-indexed outputs); three wire formats, each its own parser.
- **Cache-delta on Responses and Gemini** (row 7). Opt-in, mutually exclusive with
  cold-point, and its rewriter is shaped around `messages`; not ported.
- **Chat streamed usage** (row 15). Getting it needs `stream_options.include_usage`
  injected, which adds a final chunk with an empty `choices` the client did not ask for;
  clients that index `choices[0]` on every chunk break. Not injected.
- **LiteLLM billed usage** (row 26): the hook has no post-call callback, so no
  calibration pairs from LiteLLM traffic.
- **Long-context price tiers** (row 19): OpenAI >272K and Gemini Pro >200K prompts bill
  ~2x; one tier per row, so those prompts are under-priced.
- **Transcript-mode spend** (row 28) and the empty-screen hint name Claude Code only;
  what-if itself scans all three agents' roots.
- **Vertex / Bedrock path forms** are not routed, so every row above is "passthrough"
  there.
- **Other framework integrations** (`agno`, `autogen`, `langchain`, `strands`) call the
  Anthropic adapter; whether their messages are Chat-shaped was not audited here.

## How parity is tested

Same scenario, every adapter, equivalent assertions:

- `tests/test_refetch.py` — `_SHAPES` parametrisation: a repeat answered verbatim, new
  lines digested, one copy only, tracker closed — Messages, Chat, Responses, Gemini.
- `tests/test_coldpoint.py` — documented-TTL table; Chat and Responses evict only after
  the retention bound, never the newest turns, byte-stable after.
- `tests/test_exact_quote_adapters.py` — the quote guard measures Messages, Chat, Gemini.
- `tests/test_expand_usage.py` — Responses usage (JSON and SSE), Chat cache writes,
  Gemini thinking; buffered expand nets the re-query on Anth/Chat/Resp alike; overhead.
- `tests/test_provider_parity.py` — Responses item events, Gemini JSON-array stream.
- `tests/test_shadow_provider_parity.py`, `tests/test_whatif_codex_gemini.py`,
  `tests/test_pricing_canary.py`, `tests/test_cachedelta.py`, `tests/test_cache_contract.py`,
  `tests/test_integrations_litellm_hook.py`, `tests/test_aproxy_cov.py`, `tests/test_expand.py`.

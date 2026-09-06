# 0008 — The cache contract

- **Status:** accepted
- **Date:** 2026-09-04
- **Relates to:** `distil/compress/recency.py`, `distil/adapters/*.py`, `tests/test_cache_contract.py`

## Context

Prompt caching is the largest single term in distil's cost model, and it is
all-or-nothing. A provider cache entry covers a *prefix*: rewrite one byte at or before
the boundary and the entry for the entire prefix is thrown away, so the next turn pays
the full write rate instead of the ~0.1x read rate.

distil has already been on the wrong side of this once. A recency carve-out counted back
from the *end* of the message list slides forward as a conversation grows, so every block
was protected while fresh and digested one turn later — rewriting content the client had
by then committed to its cached prefix. Measured against the real API that produced zero
cache reads on every turn and **2x the cost of compressing nothing at all**. The whole
prefix was re-written at 1.25x instead of re-read at 0.1x. `compress/recency.py` carries
that history in its docstring; the fix was to anchor the carve-out to the client's own
`cache_control` breakpoint rather than to the end of the list.

The residual problem is that this failure is *silent*. Every request still succeeds. The
savings report still shows tokens removed. Nothing raises, nothing 5xxs, and the only
symptom is the bill. A property that can regress invisibly and cost double needs to be
written down as a contract and enforced by a test, not left as a comment in one module
that three adapters happen to respect.

## Decision

distil states the following contract about the bytes it forwards, and enforces each
clause in `tests/test_cache_contract.py` against all three provider shapes.

### (a) Prefix stability

For every message the client re-sends **byte-identical** to the previous turn, distil
forwards it **byte-identical**, at every index at or before the provider's cache
boundary.

The boundary is per-provider:

- **Anthropic** — the last `cache_control` marker the client placed
  (`compress.recency.cached_prefix_end`). Anthropic caches only what the client marks.
- **OpenAI, Gemini** — the last index. Both cache prefixes implicitly and commit
  everything they are sent, so the boundary is the whole request and the adapters give
  them no recency carve-out at all (`_recent_chat_verbatim_indices` and its siblings
  return the empty set, by design and not by oversight).

The binding form of the clause uses a **high-water mark**, not the current turn's
boundary: once the provider has cached through index *i*, rewriting *i* on any later
turn invalidates the entry, even if this turn's marker sits earlier.

### (b) Compression touches only the volatile suffix

Anything distil does change lies strictly after that boundary. Two mechanisms enforce it
today and both are load-bearing: the recency carve-out anchors to the breakpoint, and
query-aware salience is scoped to blocks after it, so that changing the question does not
re-choose which lines survive inside an already-cached block. The second is easy to lose
— intent terms change every turn by design, and letting them reach cached content
reproduces the same bust from a second direction, invisibly to any per-request test.

### (c) Digest determinism

A digest emitted for a block at turn N is byte-identical when the same block is forwarded
at turn N+1. This holds because handles are **content-addressed**: `tier1._handle` and
`adapters.anthropic._handle` are both `sha256(text)[:8]`, with no per-request nonce,
timestamp, or counter. This was verified in code rather than assumed, and it is asserted
directly — a random handle would rewrite every digest stub on every turn and bust the
cache on its own while every other clause still passed.

### (d) What is explicitly **not** guaranteed

- **A client that rewrites its own history gets no promise.** If the client edits,
  re-orders, re-serialises, or moves its own `cache_control` marker, the input is not the
  same input and the contract does not apply. distil cannot make an unstable client
  stable.
- **Provider TTL.** Cache entries expire on the provider's schedule. Byte-stability is
  necessary for a hit, not sufficient.
- **A client that sends no cache marker at all.** With no marker there is no Anthropic
  prefix to protect, so the plain last-*k* recency window applies and a block does go
  verbatim on one turn and digested on the next. This is intended: nothing is being
  invalidated. It is bounded rather than unbounded, and the test asserts the bound —
  that churn must occur strictly *after* the boundary, and must still occur at all, so a
  silently-dead carve-out fails too.

## Consequences

- The contract is now falsifiable. `tests/test_cache_contract.py` replays 6-turn
  synthetic sessions through `compress_messages`, `compress_chat_completions`,
  `compress_responses_input`, and `compress_generate_request` — the same public functions
  the proxy calls — and fails on any same-input byte drift at or before the boundary.
- Measured at adoption time, the contract **holds on all three providers**, including the
  realistic Claude Code shape where the client pins its newest turn and the entire
  history is cached. In that configuration there is no uncached tail and nothing moves at
  all. No bug was found; the test codifies a property that was true and undefended.
- `distil dissect` now reports a **cache-read share** line per session
  (`cache_read_input_tokens / total billed input`), so the contract can be checked
  against what the provider actually did rather than only against what distil intended.
  The fields were already in the ledger (`usage_cache_read`, `usage_cache_create`); only
  the reporting was missing. It renders `None`, never `0.0`, when the records predate
  those fields — "we did not measure this" and "the cache never hit" are opposite
  diagnoses and must not share a rendering.
- This constrains future transforms. Any new compressor that wants to consider
  cross-block or whole-conversation state must either confine itself to the volatile
  suffix or accept that it will fail this gate.

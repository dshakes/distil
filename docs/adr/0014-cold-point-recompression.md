# 0014 — Cold-point recompression

- **Status:** proposed (on by default in the rc; GA gated on the soak and live A/B below)
- **Date:** 2026-09-24
- **Relates to:** `distil/coldpoint.py`, `distil/adapters/anthropic.py` (`cold_candidates`, `compress_messages(evict=…)`), `distil/proxy.py`, `tests/test_coldpoint.py`, ADR 0008, ADR 0011
- **Amends:** ADR 0008 — adds clause (g): on a turn where the provider cache has certainly expired, distil may change bytes that were cached before, and from then on forwards the new form byte-identical

## Context

The live-savings research (`docs/research/live-savings-gap.md` on branch
`research/live-savings-gap`, artifact `benchmarks/results/2026-09-24/live_savings_decomposition.json`)
ranked the levers left on digest-mode Claude Code traffic. The largest was this one:
tool_results carried in the cached prefix are billed on every later turn at the read rate,
and nothing can shrink them after first sight without busting the cache. The August
incident (a sliding recency window rewriting one block per turn) showed what that costs:
every turn pays the write rate on the whole prefix.

The research left one exception open. Anthropic's cache entry lives for a fixed TTL after
the last request that touched it: 5 minutes by default, 1 hour when the client marks
`cache_control: {"type": "ephemeral", "ttl": "1h"}`. A turn that arrives after the entry
has expired pays the write rate on the whole prefix **whatever distil forwards**. Changing
old bytes on that turn costs nothing extra, and the smaller form is what gets cached and
re-read for the rest of the session.

## Decision

On a **cold turn**, distil replaces older tool_results with a recoverable stub and keeps
forwarding that stub on every later turn in the lineage.

1. **State** (`distil/coldpoint.py`). Keyed by `account_scope` + `prefixreplay.lineage_key`
   (model, system, tools, head message, `DISTIL_SESSION`). `account_scope` hashes the static
   API-key headers (`x-api-key`, `api-key`, `x-goog-api-key`) and deliberately NOT
   `Authorization`: Claude Code's OAuth bearer refreshes mid-session, and keying on it (as
   replay's `credential_scope` does) forked the lineage and un-evicted a warm prefix at
   every refresh. It holds a sleep-counting monotonic timestamp (`CLOCK_MONOTONIC` on darwin,
   which keeps counting through sleep per clock_gettime(3), unlike Python's
   `time.monotonic` = `mach_absolute_time`; `CLOCK_BOOTTIME` on Linux), the message count and a 16-hex hash of
   the last message, the longest TTL the lineage has asked for, an in-flight counter, the
   evicted `tool_use_id` set and an `ambiguous` flag. The state holds no content. An LRU caps
   it at 256 lineages, and every request does one O(1) lookup.
2. **Decide once.** `plan()` runs before compression. A new eviction is decided only when
   every one of these holds:
   - the lineage is known. First-seen, a restarted proxy, a hot-swapped worker or an
     LRU-dropped lineage does nothing;
   - nothing of the lineage is in flight;
   - the lineage is not ambiguous;
   - the TTL is parseable (`5m`, `1h` or absent);
   - `now − last > ttl + MARGIN_S` (60 s, one named constant in `coldpoint.py`).

   The candidates come from `adapters.anthropic.cold_candidates`.
3. **Apply every turn.** `compress_messages(evict=…)` renders each evicted id as
   `<<distil evicted older tool output (N lines); distil_expand handle=H recovers it>>`.
   The stub is a pure function of the block's content, so it has the same bytes on every
   turn. The original is recorded through the existing `RestoreStore._record`, which writes
   to memory and to the disk restore store. `distil_expand` and the expand loop recover it
   with no new store. The evicted set only grows. On re-application the stub is still
   reject-if-bigger (stub tokens < block tokens), a pure function of the block's text, so a
   client that rewrote an old result into something short gets it back verbatim, stably.
4. **Persist the set.** Whenever a lineage's set grows, `{lineage key: ids, oldest first}` is
   merged into `$DISTIL_HOME/coldpoint.json` under a file lock and written mkstemp (0600 at
   creation) → fsync → `os.replace`. Bounds: 512 lineages (LRU) and the 2048 most recent ids
   per lineage (`_MAX_PERSISTED_IDS`; a session past that loses its oldest ids from the file
   only, and pays one rewrite for them after a restart). A first-seen lineage loads its set
   and re-applies it, deciding nothing new. The file is content-free: hashed keys and the
   provider's random `tool_use_id`s.

   **What it covers, precisely.** The key includes `DISTIL_SESSION` (via
   `prefixreplay.lineage_key`), so persistence covers a hot-swap (every upgrade), a worker
   restart and an in-memory LRU drop **within the same wrap session**. It does NOT cover a
   fresh `distil wrap`, or `claude --resume` in a new terminal: the new session id makes a
   new lineage, which is first-seen with an empty set, so a resumed conversation whose
   stubbed prefix is still warm pays one rewrite, exactly as before persistence. The keying
   is deliberately left alone (it is replay's, and ADR 0011 scopes lineages per session).

   **Cost on the request path.** The process keeps a parsed copy keyed on the file's
   (path, `st_mtime_ns`, size). A first-seen lineage — title generation, subagents, quota
   pings, an LRU-dropped lineage — costs one `stat`, and the file is re-parsed only when
   another process changed it. A read that fails (a Windows sharing violation mid-replace, a
   torn or corrupt file) is treated as "no change" and keeps the cached copy, so a transient
   error can neither un-evict nor let the next write erase other lineages' sets. The file is
   written only on a cold turn that evicted something new. Any error falls open to
   in-memory behaviour.
5. **Recovery.** Eviction runs only where the expand tool is injected (`expand and not
   verbatim`), so a stub is always recoverable in the conversation. This is the same
   gate that already governs every Tier-1 stub.

### What is never evicted

`cold_candidates` applies every rule the digest applies, plus some stricter ones:

- the last `RECENCY_KEEP_TURNS` tool-bearing turns, **counted from the end** and not
  anchored to the cache breakpoint, because the agent reasons over them to pick its next
  action;
- exact-quote results (`exact_quote_tool_use_ids`: file reads an `Edit`/`MultiEdit`
  quotes back, shell reads, `distil_expand` results). This is the one Edit-safety keep
  policy, and it is reused here rather than copied. It also wins at **apply** time: if an
  Edit later comes to depend on an evicted block, the block goes back to verbatim.
  Correctness beats one cache write;
- any block containing an `old_string` that an Edit in the history already quotes;
- learned-keep content (the outcome and expand keep predicates);
- any handle the model has expanded in this process under the same account scope
  (`note_expanded(scope, handle)`);
- non-single-text results (images, multi-part), and anything under 128 tokens.

## The cache-safety argument

The invariant is that **a byte may change only on a turn where the provider cache is
already lost, and it is then stable for the rest of the lineage.**

- **Certain expiry, from distil's own observation.** `last` is taken from a monotonic clock
  in the proxy process. It is touched when the request is planned and again when it
  **finishes** (the `finally` in `do_POST`, which runs after the stream has been relayed
  and the expand loop has finished). Shadow replays call `begin`/`end`, so the compressed
  arm keeps the lineage in flight. So every path by which distil refreshes the provider's
  entry after the forward either counts as in flight or moves `last` forward. The 60 s
  margin covers the network gap between the two clocks.
- **Merging only makes distil evict less.** If two conversations share a lineage key,
  `last` is the later touch of the two, so the gap distil measures is never longer than
  either one's own gap.
- **Ambiguity is detected and is permanent (for the history shapes it can see).** Each request has to extend the previous one:
  at least as many messages, and the previous last message canonically unchanged
  (`prefixreplay.canonical`, which ignores the moving `cache_control`). Parallel
  subagents that share a key, a fork, a rewind or a client that rewrites its history
  all fail that test. The lineage then goes `ambiguous` for good and no new eviction is
  decided on it. The research found that 38% of lineage pairs could not be grouped
  cleanly, and this is the conservative answer to that. An already-evicted set keeps
  being applied, because stability beats novelty. A fork that inherited un-evicted
  history pays at most one rewrite.
- **What the extension test does NOT see: rewrites of OLDER messages.** It hashes only the
  previous last message. A client that rewrites earlier history in place — Claude Code's
  microcompact clearing old tool results, say — passes it. Consequence: the client has
  already busted its own cache at that message (contract clause d), so distil's timing is
  unaffected; an evicted id whose content the client replaced is re-rendered from the NEW
  content (still recoverable, still reject-if-bigger, and stable from then on). Distil
  never fills a cleared block back in, and never reads another conversation's timing from
  it. The one blind spot is two conversations that differ only before their shared last
  message, which no real client produces.
- **Byte stability after the cold point.** The stubs are deterministic and the set is
  re-applied every turn. Prefix replay (ADR 0011) therefore sees the evicted form as
  canonically equal on the next turn and holds it. `tests/test_coldpoint.py::
  test_byte_stable_after_the_cold_point` drives a real proxy through
  first-seen → warm → 20-minute gap → four warm turns. It asserts that every message
  forwarded on the cold turn is forwarded byte-identical (markers aside) on each later
  turn, and that blocks which age past the window while the cache is warm are never
  evicted.
- **Keyed by `tool_use_id`, not by content handle.** Identical output from a later call
  therefore stays unevicted: it is never rewritten when it ages, because it was never
  in the set.

### Known costs, accepted

- **State loss no longer un-evicts** (the set is persisted, above). What is still lost on a
  restart is the timing: a restarted lineage is first-seen and decides nothing new until
  distil has observed a full TTL of its own.
- **Exact-quote wins, and that un-evicts.** If an Edit later quotes a block that was
  evicted, `exact_quote_tool_use_ids` now names it and the apply step sends it verbatim
  again. That rewrites a prefix that may be warm: one cache write, accepted, because the
  agent's edit applying is worth more than the read.
- **Bearer-token callers of one proxy are not told apart.** Two different OAuth users of
  one proxy sending the same conversation head share a lineage. Interleaved, they fail the
  extension check and go ambiguous (no new eviction); the worst case is a missed saving or
  one rewrite, never another caller's bytes — the state holds no content and every stub is
  rendered from the request's own. Tenant isolation is the gateway's job, and the gateway
  does not run this yet.
- **A block evicted and later expanded stays evicted.** The model can expand it again.
  Taking it back would bust a warm prefix.
- **Cache refreshes distil cannot see.** Another client, or another key in the same
  organisation, sending the byte-identical prefix directly to the provider keeps the entry
  warm without distil seeing it. The worst case is one extra cache write on the cold
  turn. It never produces a wrong answer.
- **Decision-equivalence is not certified.** An evicted block is a full elision, not a
  digest. The expand path makes it recoverable. It is not proven harmless, and that is
  what the rollout below has to measure.

## Scope

- **Anthropic Messages only**, through the threaded proxy (`distil proxy` / `distil wrap`).
- **Not the async proxy.** It is verbatim-only and has no expand loop, so it has no
  recoverable stubs.
- **Not the gateway yet.** TODO: scope the state per tenant, the same way replay is
  scoped, then wire `plan`/`end` into `gateway.py`.
- **Not `--session-delta`.** Its references would be evicted into stubs of stubs.
- **OpenAI Responses / Chat Completions: TODO, deliberately not implemented.** OpenAI
  caches prefixes automatically and gives no client TTL. Its documented eviction is
  "typically 5–10 minutes of inactivity, up to an hour off-peak", which is not a bound
  distil can call *certain*. Extended retention (`prompt_cache_retention`) differs again.
  The mechanism would be the same, keyed on a TTL the provider guarantees. That guarantee
  does not exist today.

## Controls and observability

- On by default wherever the recoverable digest is already on. Opt out with
  `--no-cold-point` (`distil proxy`, `distil wrap`, recorded in the session manifest and
  printed by `dissect` as `no-cold-point`) or `DISTIL_COLD_POINT=0`, which serves
  launch-agent installs whose argv is pinned.
- Lossless-only and verbatim are untouched. So is a subscription session without
  `--expand`: the same policy as every other stub (`policy.may_compress_lossy` plus the
  explicit `--expand` opt-in).
- The drift guard hook is `plan(held=True)`. A held lineage keeps applying its set and
  decides nothing new. Wiring DriftGuard in is one argument.
- **Fail-open.** An exception in planning compresses exactly as before. An exception in
  candidate selection evicts nothing new.
- **Accounting.** There is one path, not a parallel one. Evicted tokens reduce the payload,
  so they land in `x-distil-tokens-saved` → the savings ledger → `dissect` / `distil
  savings`. They are also censused under `tool_result_evicted`, which `dissect` labels in
  its eligibility line. Their handles are in the receipt, restorability is checked, and
  the mode is recorded as `digest`, which it is. Per request, `x-distil-cold` (the
  decision: `first-seen | warm | cold | ambiguous | inflight | unknown-ttl | held`) and
  `x-distil-cold-evicted` (the lineage's set size) go on the response. The session ledger
  records them as `cold` / `cold_evicted`, and records nothing when the feature is off.

## Rollout

1. **rc soak (≥3 days)** on the maintainer's live traffic, with two explicit promotion
   gates read from the session ledger:
   - **Reason distribution.** Report the count of every `cold` reason (`first-seen`,
     `warm`, `cold`, `ambiguous`, `inflight`, `unknown-ttl`, `held`) per session. If
     `ambiguous` dominates on single-conversation sessions, the extension test is too strict
     for Claude Code's real history shape; understand it before GA.
   - **TTL exactness.** On every row with `cold == "cold"`, `usage_cache_read` must not
     exceed the static prefix (`system_tokens + tools_tokens`, calibrated). A cold turn that
     read history from cache means the entry was still alive — distil's clock and the
     provider's disagree — and `MARGIN_S` must be raised before GA. Zero violations is the
     gate.
2. **Live A/B is the real gate.** It has not been run. Use two arms over the same idle-gap
   workload (≥5 min between turns), `--no-cold-point` against the default. Compare
   `distil cache` read/write totals and billed dollars, and count expand round trips per
   evicted block with `expanded_handles`. The shadow harness supplies the
   decision-equivalence verdict on sampled cold turns. Promote only if dollars fall,
   cache reads after the cold point do not, and shadow equivalence holds at the existing
   floor.
3. If either fails, flip the default to off (a one-line change in `build_handler`) and
   keep the flag.

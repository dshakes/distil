# 0011 — Forwarded-bytes prefix replay

- **Status:** accepted
- **Date:** 2026-09-06
- **Relates to:** `distil/prefixreplay.py`, `distil/proxy.py`, `tests/test_cache_contract.py`, ADR 0008

## Context

ADR 0008 states distil's half of the cache contract: **when the client re-sends a
message byte-identical, distil forwards it byte-identical.** Clause (d) then says what
happens when the client does not — *"a client that rewrites its own history gets no
promise. distil cannot make an unstable client stable."*

That clause was written as an edge case. It is the common case.

Agentic clients rewrite their history on every single turn, without changing a token the
model reads:

- the `cache_control` breakpoint **advances** to the newest block (Claude Code's shape,
  and the one that actually bills);
- an SDK shim stamps positional **`index`** fields onto blocks that had none, renumbered
  as the conversation grows;
- **string content becomes a single text block**, or back again, as a request round-trips
  through a client's own types. The Responses API spells the same block `input_text`.

distil forwards what it receives, so each of those reaches the provider as changed bytes,
and the provider misses on a prefix it is already holding. The whole span is then re-billed
at the write rate instead of read at ~0.1x. This fails exactly the way ADR 0008 warned
about: every request succeeds, nothing raises, and the only symptom is the bill.

Measured offline against distil's own adapters, before this change
(`benchmarks/prefix_replay_stability.py`, 8 turns, share of re-sent messages forwarded
byte-identical to the previous turn):

| provider shape | `index` stamped | string/block sugar |
|---|---|---|
| anthropic/messages | 0.0% | 0.0% |
| openai/chat-completions | 14.6% | 0.0% |
| openai/responses | 0.0% | 0.0% |
| gemini/generateContent | 0.0% | n/a — no second spelling |

Zero. Not degraded: gone. A client doing nothing but renumbering an `index` field loses
the entire prefix, on every provider, on every turn.

The marker-advance case is **already** free, and that is worth recording as a result
rather than a gap: anchoring the recency carve-out to the client's breakpoint (ADR 0008)
neutralised it, and the same table reads 100% for it before and after.

## Decision

distil remembers, per conversation lineage, the previous turn's `(original, forwarded)`
pair. On each request it walks the new originals against the old ones with a **canonical**
comparison and, for the longest canonically-equal leading prefix, forwards **the bytes it
forwarded last turn** rather than the bytes it would produce now.

On by default (`--no-prefix-replay` opts out), per the house rule that intelligence is the
default and flags are opt-outs.

### The comparison key

Canonicalisation ignores `cache_control` and `index`, normalises the spellings of a single
text block (bare string, `text`, `input_text`, `output_text`), and normalises JSON key
order. It ignores nothing else. `citations`, `annotations` and their relatives are **not**
in the set: they may carry meaning, and the cost of leaving them out is a missed hit, not a
wrong prefix. Tool inputs (`input`, `arguments`, `args`) are **opaque** — compared as they
arrive, never structurally rewritten, because a tool input may itself contain a key called
`content` and applying the message-level sugar to it would declare two different tool calls
equal.

The comparison key and the forwarded bytes are separate things and never touched to each
other: the key is `sort_keys` JSON, the wire is not.

### The guard: replay restores bytes, never decisions

This is the clause that makes the mechanism safe, and it is not the obvious design.

distil's compressor is **not** a pure function of one message. The exact-quote guarantee
(`compress.provenance`) keeps a tool result verbatim because of an `Edit` that arrives
*later* in the list, so a block legitimately digested at turn N can need to be verbatim at
turn N+1. Overlaying turn N's stub there would break the agent's next edit to buy a cache
hit — trading correctness for money, which is the one trade distil does not make.

So an item is replayed only when the previous turn's forwarded form and this turn's
forwarded form are **themselves canonically equal**. Replay is then a pure byte
restoration, and *"prefix replay never changes semantic content"* is not a claim about the
implementation but its loop condition. `distil validate` asserts it as an invariant anyway,
because a loop condition is exactly the kind of thing a later optimisation deletes.

### Scope and stopping

- **Lineage** = leading system-run + model + tools + the conversation's (canonicalised)
  head, per session. Change any of them and the provider holds a different entry, so
  replay never crosses that boundary.
- **Stop at the first divergence, and stay stopped.** A cached prefix is a byte prefix;
  an item restored after a break repairs nothing and only risks pairing old bytes with a
  history the client has since edited. A divergence inside what the provider already
  cached is the client's own rewrite and is forwarded exactly as it arrived — ADR 0008
  clause (d), unchanged.
- **Breakpoints follow the client.** The replayed bytes carry the client's *current*
  `cache_control` placement, block index for block index. The marker delimits the cached
  span and is not part of it; `prefix._flatten` has taken that position since 1.41. A
  block whose marker did **not** move is returned untouched rather than rebuilt — rebuilding
  moves `cache_control` to the end of the key order, and JSON key order is part of the bytes
  the provider hashes, so the naive version busts the prefix the first time it replays a
  block whose marker was not already last. Caught by the per-server tests, not by the
  adapter-level ones.
- **Fail-open.** Any exception forwards exactly what the compressor produced.

### Prior art

Headroom's `PrefixCacheTracker.overlay_cached_prefix` does the same thing — replay
previously-forwarded bytes while a canonical comparison holds, with the same
comparison-key/forwarded-bytes separation, the same opaque treatment of tool inputs,
breakpoint re-placement at the client's positions, and conversation-lineage scoping. It is
the right mechanism and there is no point pretending otherwise. What distil adds is the
guard above (Headroom classifies a miss rather than repairing it, and has no
exact-quote exemption to collide with) and the falsifiable form: the property is a gate in
`tests/test_cache_contract.py` and an invariant in `distil validate`, on all four request
shapes, with a control arm.

### State

In-memory, bounded (16 lineages, LRU, 8 MB of retained history each at most), and
**deliberately not persisted**. It holds message bytes, and distil has exactly one place
content is allowed to rest on disk — the TTL'd, owner-only restore store. A hot-swap
therefore starts a lineage cold and rebuilds it from the next turn, at a cost of one cache
write. That is the correct trade for not opening a second content-at-rest surface.

## Consequences

- Measured offline after the change, the same table reads **100.0%** on every rewrite ×
  provider pairing that applies, and the un-rewritten control reads 100.0% before and
  after — replay repairs rewrites and does nothing when there is nothing to repair. Added
  latency is ~0.13 ms per request at 8 turns.
- This is distil's forwarding, not the provider's bill. It cannot be otherwise offline.
  Live confirmation is the **cache-read share** in `distil dissect`, alongside the new
  per-session `prefix replay: N forwarded as previously sent, M compressed fresh; R client
  rewrites repaired` line, after release and a soak.
- `restored` is reported separately from `hits` on purpose. Hits with zero restored is the
  healthy steady state (the client re-sent the prefix byte-identical and there was nothing
  to fix), and folding the two together would make a dead mechanism indistinguishable from
  a working one — the same failure mode ADR 0008 called out for `None` vs `0.0%`.
- **All three servers, not one.** The threaded proxy, the async proxy (`--async`) and the
  multi-tenant gateway apply replay at the same point — the final forwarded body, after
  every transform, immediately before serialization — through one shared `prefixreplay.apply`,
  so the fail-open exists once instead of three times. The first draft shipped it on the
  threaded proxy only and had `--async` *announce* its absence, which is the 1.46.0 mistake
  in a new costume: managed installs run `distil proxy`, and a default-on feature missing
  from the server they happen to use is a feature they do not have. A server that
  re-serialises every body (the async proxy, the gateway) still benefits: its output is
  deterministic given the same items, so replaying the items is what makes its prefix
  stable.
- **The gateway scopes the lineage by tenant.** A cached prefix belongs to one credential
  at the provider. The lineage key is content-derived, so without the tenant prefix two
  tenants posting the same conversation would be "the same lineage" and one tenant's
  forwarded bytes could land in another's request. Asserted directly.
- A client that sends non-compact JSON pays one extra cache write the first time replay
  fires, because `_serialize_if_changed` forwards the client's original bytes when nothing
  changed and compact bytes when something did. From the next turn on both sides are
  compact and stable. Not worth per-item byte plumbing to remove.
- This constrains future transforms in a new way. Anything whose output for message *i*
  depends on messages after *i* will find replay declining to hold it — correctly. If that
  ever becomes the common case, the guard is where to look, not the symptom.

# 0022 — A re-fetch of folded content is forwarded verbatim

- **Status:** accepted (on by default; offline-measured, live outcome not yet run)
- **Date:** 2026-10-04
- **Relates to:** `distil/compress/refetch.py`, `distil/adapters/anthropic.py` (`compress_messages`, `refetch_enabled`), `benchmarks/reinflate_replay.py`, `benchmarks/results/reinflate-replay/`, `tests/test_refetch.py`, ADR 0008, ADR 0009, ADR 0010, ADR 0021
- **Amends:** ADR 0009 — adds one cross-block dependency, monotone toward verbatim

## Context

On 300 SWE-bench Lite tasks served through distil, the agent called `distil_expand` 13
times; after the shell-search fix (1.56.3), 5 times. When an earlier tool result had been
digested and the agent needed what was folded, it ran the command again or re-read the
file. That costs a step each time, and the served arm is −2 pts against plain on resolved
tasks (inconclusive).

Two facts about the client shape that bills make this worse than it sounds. A caching
client puts its breakpoint on the newest message, so every tool result is committed prefix
on the request that first carries it, and is digested on first sight (ADR 0008, ADR 0021).
And the digest of a block is a pure function of its text. So a byte-identical re-run hashes
to the same handle and comes back as **the same stub**: the agent asked again and was shown
exactly what had just failed it. Chains of three and four reads of one window are in the
transcripts.

Offline, over the 300 post-fix transcripts (below), 61 of the agent's 894 tool calls
(`distil_expand` excluded; results of at least three comparable lines) returned content that
was already in an earlier tool result. 25 of them could see it there; **36 could not,
because distil had forwarded that earlier result as a digest.** Those 36 are the re-reads
distil can be blamed for — an upper bound, since an agent re-reads for its own reasons too.

## Options

Every option was replayed on the same 300 transcripts, request by request, with the
harness's cache breakpoint on the newest block (`benchmarks/reinflate_replay.py`). The
agent's actions are fixed, so this measures what it would have seen, not what it would
have done.

| option | avoidable re-reads (of 61) | request-size savings | cache-aware savings | prefix rewrites |
|---|---|---|---|---|
| current (1.56.4) | 36 | 11.90% | 9.34% | 0 |
| **(a) re-fetch verbatim — chosen** | **31** | **11.21%** | **8.87%** | **0** |
| (a′) … and once tripped, stop digesting for the session | 29 | 9.37% | 7.73% | 0 |
| (b) likely-needed: every sequence whose stages are all reads/searches stays verbatim | 22 | 7.83% | 6.68% | 0 |
| (b′) likely-needed: any command with a reader stage stays verbatim | 8 | 3.48% | 3.37% | 0 |
| (c) re-inflate the earlier block when a re-read arrives | — | — | — | every fire |

**(a) Re-fetch verbatim.** When a new tool result is mostly lines (≥ 80%, line-number
prefixes stripped) that an earlier tool result carried *as the client sent it* but that
distil did *not* forward verbatim, the new result is forwarded verbatim. The earlier digest
is left exactly as it was. Matching is on content, not command: the commands in a chain vary
freely (`cat f | sed -n 1,80p`, then `sed -n 55,100p f`, then a `python` heredoc printing
the same lines, then `… | cut -c1-110`), the lines do not.

**(c) Re-inflate the earlier block** is what "automatic re-inflation" first suggests, and it
is the one option the cache contract rules out. The earlier block is in the provider's
cached prefix; flipping it from digest to verbatim rewrites the prefix from that point on
at the 1.25x write rate, which ADR 0008 measured at 2x the cost of compressing nothing. It
would also fire *after* the re-read it was meant to prevent, so it saves nothing on that
step. Rejected without needing a number.

**(b), (b′) Likely-needed retention.** Most avoidable re-reads trace back to a source the
exact-quote classifier does not call a read: a compound or piped command with a read stage
in it (`cat f | sed -n 1,80p; grep …`). Part of that is a plain gap: a sequence is
classified by its **last** stage, so `sed -n 55,90p a.py; grep -n x b.py` is neither a
read (last stage is `grep`) nor a search (first command is `sed`), and is digested whole,
while `grep …; sed -n …` is kept. (b) closes only that order dependence; (b′) keeps any
command with a reader stage. Both avoid more re-reads than (a) does — but they forfeit 2.7 points of cache-aware
savings for 14 re-reads (b), or 6 points for 28 (b′), on traffic where the post-fix
savings are only 9.3% to begin with. Per re-read made visible, (a) costs 0.09 points; (b)
and (b′) cost 0.19 and 0.21. (b′) leaves distil saving almost nothing on coding agents.
More importantly, (b) is a change to the *exact-quote* rule — what an `Edit` may quote
back — and deserves to be judged on quote safety in its own record, not smuggled in as a
re-read heuristic. Not adopted here; flagged as the follow-up with the larger ceiling.

**(a′) Session-sticky** trips (a) into "digest nothing new" once a re-fetch is seen, the way
the quote guard widens on a miss. Two more re-reads for 1.1 more points. Not worth it.

**Command-keyed repeat detection** (same normalised command ⇒ verbatim) was not carried
forward: it misses every chain above whose command changed, and content matching already
catches an identical repeat.

## Decision

Adopt (a), **on by default**, with `DISTIL_REFETCH_VERBATIM=0` to turn it off. New census
bucket `tool_result_refetch` carries its cost.

### Placement

In `compress_messages`, after the exact-quote exemption, the recency carve-out and
cold-point eviction, and immediately before the digester — the one point in the walk where
the rule can only ever stop a digest. A re-fetched block gets what a recency-exempt block
gets: Tier-0 lossless transforms and `<distil:keep>` handling, no stub.

### The cache argument

The verdict for block *i* is a function of the blocks before it — their text as sent and
their bytes as forwarded — and the forwarded bytes are themselves functions of their own
prefixes. So block *i* encodes to the same bytes on every turn that carries it, the same
construction as the re-read delta (ADR 0010). The new block is the only one whose rendering
the rule chooses, and it has never been sent, so there is no cached rendering to break. The
replay checks this directly: across every request of every variant, **0** messages already
forwarded changed bytes on a later request. `tests/test_refetch.py` asserts it turn by turn.

### Stateless

No session state. A `Tracker` is opened per compression pass (the quote guard's widened
retry gets its own) and fed every tool result in conversation order; it is discarded when
the call returns. The history is the state, as everywhere else in the adapter.

### Safety

Monotone: the rule can only turn a digest into the original, never the reverse, so no line
the agent would have seen can be lost by it and the exact-quote guarantee is untouched.

It is, however, a dependency **between blocks**, which ADR 0009 said any such transform must
argue for. The argument: an adversary who controls an earlier tool result can at most make a
later block go out verbatim — a denial of savings on that block, bounded by its own size,
of the kind ADR 0009 already accepts for decoy flooding. They cannot make a line of any
block disappear, and the trusted block's bytes, when they differ, are its original bytes.
`test_an_earlier_block_can_only_cost_a_later_one_its_savings` pins exactly that.

## Consequences

- **The effect is real and small, and is stated as such.** 36 → 31 avoidable re-reads on
  300 tasks (−14%), for 0.69 points of request-size savings and 0.47 points of cache-aware
  savings. On the pre-fix transcripts (`benchmarks/results/swebench-outcome-300/`, replayed
  with today's adapter) 58 → 55. It does not prevent the *first* re-read of a folded block —
  nothing that respects the cache contract can, short of not digesting it — only the second
  and later ones, and it ends the "same stub again" loop on the read the agent just issued.
- The offline replay cannot say whether those re-reads would have been skipped, or whether
  task success moves. That needs the live run: the distil arm only, against the committed
  plain rows (command in `benchmarks/results/reinflate-replay/README.md`).
- ADR 0021's served strategy runs the real `compress_messages`, so it now serves re-fetches
  verbatim too. Its description "each block's served bytes depend only on its own text"
  becomes "on its own text and the blocks before it" — still prefix-deterministic, which is
  the property ADR 0008 needs.
- Scope: the Anthropic Messages adapter, including `role: "tool"` string messages that pass
  through it. The OpenAI Responses and Gemini walkers do not have the rule yet; the measured
  traffic is Anthropic.
- Cost per request is two set unions over the tool-result lines already in memory.
- Cold-point eviction (ADR 0014) still wins over the rule: an evicted block is a stub, as
  before. A re-fetch of it is then forwarded verbatim, which is the behaviour wanted.

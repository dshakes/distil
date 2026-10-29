# 0028 — Drop prior-turn thinking at the cold point (opt-in until certified)

- **Status:** accepted (shipped OFF by default)
- **Date:** 2026-10-29
- **Relates to:** ADR 0014 (cold-point recompression), ADR 0022 (certification hold),
  `distil/adapters/anthropic.py` (`cold_thinking_candidates`, `drop_cold_thinking`),
  `tests/test_cold_thinking.py`

## Context

Claude Opus 4.5 and every model numbered 4.6 or higher keep earlier turns' thinking blocks
in context and bill them as input; older models strip them server-side ("Thinking block
preservation by model", platform.claude.com/docs/en/build-with-claude/thinking). With the
default `display: "omitted"` on 4.7+, the blocks arrive with empty text, so the tokens
are billed but invisible to distil's local counter.

Removing them while the prompt cache is warm rewrites the cached prefix (ADR 0014's
measured 2x cost). On a cold turn the prefix is re-written anyway, so it is the one free
moment.

## Decision

1. Thinking blocks ride ADR 0014's evicted set as `th:<sha256(signature)[:16]>` keys,
   so they inherit everything that set already guarantees: decided only on a cold turn
   distil observed itself, re-applied byte-identically on every later turn, persisted
   across restarts, no fresh decision while the drift guard holds, kill switch
   `DISTIL_COLD_POINT=0`.
2. API rules, enforced at render time on every turn (any doubt drops nothing):
   - only a **leading run** of thinking blocks is removed, oldest first. Preserved
     thinking (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5) chains each block to the
     previous one, and removing from the middle invalidates every later block;
   - never the turn in progress: everything from the last user message with no
     `tool_result` onward. Its thinking must go back "complete and unmodified" during tool
     use, interleaved blocks included. Never the last assistant message either;
   - whole blocks only. A signature is never altered, an assistant message is never
     left empty, and nothing happens if a `compaction` block is present;
   - only on models that bill prior thinking, which is where there is anything to save.
3. **Off by default** (`DISTIL_COLD_THINKING=1` opts in). This is lossy for the model's
   own reasoning, and no shadow or certify evidence covers it yet. The certification
   rule (ADR 0022) needs at least `VERDICT_MIN_AB` paired rows of evidence before a mode
   counts as inside the budget. Once opted in, cold turns sampled by shadow are booked
   under the serving mode, so the per-mode hold watches them.

## Consequences

- `x-distil-cold-evicted` counts thinking keys together with tool ids.
- On `display: "omitted"` traffic, `tokens-saved` under-reports this transform, because
  the dropped text is empty locally while the billed reasoning is not. The provider's
  `usage` is the only true measure.
- To turn it on by default, collect ≥50 paired shadow rows on opted-in cold turns and
  require the paired-harm upper bound to sit inside `BUDGET_ALPHA`.

## Addendum (2026-10-29) — an eviction is a prefix edit on preserved-thinking models

On Fable 5.1, Mythos 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5 each thinking block's
signature binds the prefix it was produced after; an edit at position P fails that block
and every later one (a 400 on accounts past the enforcement cutoff and always on Mythos
5.1; billed garbage input otherwise). ADR 0014's default-on eviction is such an edit. So,
on those models and any unknown `claude-<family>-5+` id (`preserved_thinking`, fail safe):

* a cold eviction at P also drops every thinking block after P (`thinking_after_edit`) —
  "remove the newest", which the API allows. Those keys join the evicted set, persist with
  it, and are re-applied byte-identically every later turn;
* when a block after P belongs to the turn in progress (tool-use continuation), or the
  history has a shape this does not reason about, the eviction is deferred: reason
  `deferred`, then applied on the next turn boundary as `deferred-cold`. That turn's cache
  may still be warm — one rewrite, accepted, since the alternative is never evicting in
  an agent loop that went cold mid-tool;
* the set is sticky (`sticky_evict`): neither a drift-guard hold (verbatim) nor a later
  exact-quote exemption puts an evicted block or a dropped thinking block back;
* `drop_cold_thinking` applies its set wherever the blocks sit, independent of
  `DISTIL_COLD_THINKING` (the flag gates choosing, never applying).

`distil validate` carries the `thinking-prefix` invariant for it.

## Addendum 2 (2026-10-29) — every other prefix edit: freeze, verify, drop

The cold point was one of several ways distil rewrote bytes before a signed thinking block.
The audit found nine more on these models: recency exit and re-read elision (when the
client's breakpoint is absent or lags), query-aware intent re-digesting uncached blocks,
the exact-quote guard widening, output shaping appended to `system` and toggled by the
drift-guard hold, the hold itself flipping every digest, a fail-open turn forwarded
uncompressed, prefix replay lost on restart or skipped above 8 MB, and the expand loop's
hidden `assistant[thinking, distil_expand] + user[tool_result]` round. The docs add that
the prefix is `system`, `tools` and every content block before the thinking block (thinking
blocks themselves excluded, chained instead), with JSON formatting and key order ignored,
and that a proxy should not hide the 400.

Decision, for Anthropic Messages on a preserved-thinking model (`thinkguard.lineage`:
thinking enabled or thinking blocks in the history; nothing changes otherwise — other
models and thinking-off requests go out byte-identical to before):

1. **Freeze rendering at first emission.** Prefix replay (ADR 0011) restores decisions,
   not only bytes, on these lineages (`freeze=True`): an unchanged client item goes out as
   it went out last turn even when distil would now decide differently. Only items the
   previous turn did not have get fresh decisions. This covers recency exit, intent,
   quote-guard widening, the hold flip and the fail-open turn in one place. The one
   exception is a cold turn (`Plan.fresh > 0`): its eviction is the sanctioned edit and
   drops the thinking after it itself (addendum 1).
2. **`system` is fixed per lineage.** Output shaping is off for the whole lineage rather
   than frozen: it is the only transform that touches `system`, freezing it would need
   per-lineage state, and the doc's sanctioned alternative (a mid-conversation system
   message) is a different feature. The `tools` array was already stable (distil_expand is
   injected on every request while expand is on).
3. **Verify, then drop — never 400.** `distil/thinkguard.py` runs last on the final body
   (proxy, gateway and async proxy). It records, per forwarded request, the client-side and
   forwarded prefix hash at the end of the history; a signature first seen at message *k*
   was produced by the request whose client prefix was messages[:k], which gives its
   production hash. Each block is checked against it (and against its recorded
   predecessor, the chain rule). From the first block whose prefix distil changed, every
   thinking block is removed — the allowed "remove the newest" — and the removal is
   persisted, so it is never re-added; an assistant message left empty gets a fixed
   `[thinking omitted]` text block. A block whose CLIENT-side prefix changed is the
   client's own edit and is passed through untouched. A signature with no record is kept
   only when distil changed nothing before it. `x-distil-thinking-dropped` and an INFO log
   line report every drop (the doc's "don't hide the 400"). State is content-free (hashes)
   in an append-only `$DISTIL_HOME/thinkguard.jsonl`, followed by offset so a hot-swap peer
   sees it, compacted past twice the in-memory caps. This is the backstop for restart,
   hot-swap, the 8 MB replay cap and anything future code forgets.
4. **Expand: option (i), drop after the hidden round.** The continuation's thinking is
   signed over a prefix that contains the hidden round, so the expand loops (buffered and
   streamed) hand its signatures to `thinkguard.drop`; they are removed from every later
   request, and thinking produced afterwards is valid. Chosen over (ii) passing
   distil_expand through to the client, which cannot execute it, and (iii) disabling
   expand with thinking, which makes every stub on these sessions unrecoverable. Re-
   inserting the hidden round on later turns would keep that one block, but it holds
   content in memory, breaks replay's index alignment, and still needs this drop when the
   memory is gone; its cost is one message's thinking and one cache restart per expand,
   which is rare.

Consequences and what is still not guaranteed:

* Frozen lineages are not re-digested after first emission, and the newest tool result is
  always in the recency window when first sent, so Tier-1 history savings on these
  sessions come only from cold-point eviction (or after a restart/cap, when the guard has
  already dropped the thinking a re-render would break). This is the price of keeping
  thinking valid and the cache warm.
* The tool-use-continuation rule ("send that assistant turn back with its thinking intact")
  can be broken by a drop when the failing block is in the turn in progress (a streamed
  expand continuation inside a tool loop). The doc treats it as quality guidance, not a
  400; it is the only drop distil makes inside a turn.
* `cache_control` is treated as not part of the bound prefix (Claude Code moves it every
  turn and preserved thinking works there); a key the API binds that distil normalises
  away would be missed.
* A signature with no record whose prefix distil changed is dropped, so a session that
  first reaches distil mid-way loses its earlier thinking once (it would 400 otherwise),
  and so does one whose records aged out of the caps (50k blocks, 5k requests).
* distil does not set `thinking.block_binding.prefix_mismatch_behavior` (it needs a beta
  header, works only with adaptive thinking on Sonnet/Haiku 5.5, and would choose for the
  caller); it strips the failing blocks itself instead.
* The OpenAI shapes and Gemini are untouched: no signed-prefix binding there.

## Addendum 3 (2026-10-29) — decide at first sight, then freeze

Correction to addendum 2's first consequence. "The newest tool result is always in the
recency window when first sent" holds only for a client whose `cache_control` breakpoint is
missing or lags. The recency window is anchored to the breakpoint (`compress.recency`), so a
client that marks its newest block — Claude Code, the harness's `_cc_last` — never had a
recency exemption: its newest result was already digested the request it first appeared in,
on main and on the freeze branch alike. Measured on the Headroom 0.42 h2h session set (30
real Claude Code sessions, 12 requests each, Sonnet 5.5 cache pricing, `benchmarks/
headroom_h2h.py` sessions arm, offline): main and the freeze branch are identical on that
shape (−29.7% $/session vs plain). The ~0 savings are real for breakpoint-less and lagging
clients: there the freeze kept every result verbatim (−0.0%), and main, which re-digested a
result as it left the window, rewrote its prefix every turn (+181% vs plain, and a 400 on
these models).

Decision, same scope as addendum 2 plus thinking **on in this request**, a lossy mode, the
`distil_expand` tool offered and prefix replay on (the freezer) — `thinkguard.first_sight`
and `serve_core`:

1. **No recency window** (`compress_messages(first_sight=True)`): every tool result,
   the newest included, gets its compression decision in the request where it first
   appears; prefix replay's freeze then forwards that rendering for the rest of the
   lineage. It is the same digest as everywhere else: Tier-1 with errors/verdicts and the
   densest identifier lines kept in view (commit cd44582, cherry-picked), reject-if-not-
   ≥5%-smaller in tokens, fail-open, exact-quote and re-fetch exemptions unchanged, full
   original behind `distil_expand`. Query-aware salience applies where it always did — to
   content the client has not marked cacheable — with that turn's intent, then frozen.
2. **Expand stays 400-free.** An expand of a first-sight digest is the hidden round of
   addendum 2 (4): its continuation's thinking is dropped by `thinkguard`, the rest of the
   history is untouched. The e2e stub (`tests/test_thinking_prefix_e2e.py`) runs it with the
   freeze on, buffered and streamed, expanding the result it was just shown.
3. **Gate: default on, watched as its own mode.** Shadow rows of these requests are booked
   as `digest-first-sight` (`x-distil-first-sight: 1` on the request), so the per-mode
   certification hold (ADR 0022) and the drift e-process judge first sight on its own
   evidence, not on digest's. A hold serves new results verbatim (Tier-0) and leaves frozen
   ones alone, so it never breaks thinking. Kill switch `DISTIL_FIRST_SIGHT=0`.

Why default on without paired shadow evidence of its own (it has none yet):

* on marker-on-newest clients — most of this traffic — it changes nothing: the newest was
  already digested at first sight, default on, under digest's shadow evidence. The only new
  thing there is the mode label;
* on the clients it does change, the alternatives are 0% (freeze) or +181% and 400s (main);
* what the model sees at first sight is better than what Claude Code traffic has been
  shown since the recency anchor shipped: visible recall of the newest results 95.5%
  (uncached, so query-aware) vs 57.4% on main for marker-on-newest; true recall 100%.

The risk, stated plainly: **the model now sees a digest of the result it is about to act
on.** 4.5% of the facts in a breakpoint-less client's newest result (24.5% for a marker-on-
newest client, which was 42.6% before cd44582) are out of view until it calls
`distil_expand`. Each such expand costs one extra upstream round, the continuation's
thinking and its share of cache. Neither the expand rate nor task success under first sight
is measured — offline recall is not decision-equivalence, and E7 (SWE-bench) showed lossy
compression can cost task success that a certificate did not predict. The live gate is the
2% default shadow sample; at `VERDICT_MIN_AB` paired rows the mode is inside the budget or
held. To revisit: if that verdict breaches, or the expand rate on first-sight lineages
exceeds the savings it buys, turn the default off.

Measured (paired bootstrap over the 30 sessions, $ difference as a share of plain):

| client breakpoint | main | freeze (fix) | first sight | Headroom 0.42 proxy default |
|---|---|---|---|---|
| on newest (Claude Code) | −29.7% | −29.7% | −25.9% | −0.2% |
| none / lagging | +181.4% | −0.0% | −8.5% | −0.2% |

First sight − freeze: +3.8 pp [+2.9, +4.8] on newest (all of it cd44582's recall trade, not
first sight), −8.5 pp [−11.4, −5.9] without a breakpoint. First sight − Headroom: −25.8 pp
[−31.1, −20.7] and −8.3 pp [−11.2, −5.8]. Headroom is run with its own harness breakpoint
in both rows. Session $ is the policysim Anthropic cache model on o200k token counts, 5m TTL,
10 s between requests; thinking blocks are stripped from the sessions, so this prices the
text the model reads, not billed reasoning.

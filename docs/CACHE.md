# Cache behaviour

Provider prompt caching is the largest single lever on an agent's bill, and it is
all-or-nothing: the stable prefix must be **byte-identical** between turns or the
whole cached span is re-billed at full price. One changed character at the front —
a timestamp, a session id, a reordered tool list — costs the entire prefix.

| provider | cached input | how it is claimed |
|---|---|---|
| Anthropic | ~90% cheaper | explicit `cache_control` breakpoint |
| OpenAI | ~50% cheaper | automatic, needs a ≥1024-token byte-identical prefix |
| Google | ~75% cheaper | explicit `CachedContent`, ≥32k tokens |

## What distil does

**It holds the cached span byte-stable.** Whatever the provider has cached must
reach the wire in the same form on every later turn, so the compressor is
*append-only* over the cached prefix: it may compress newly-arrived content, never
content already committed.

This page previously claimed the property held "by construction", and it did not.
Until 1.45 the recency carve-out kept the freshest tool outputs verbatim using a
window counted back from the end of the message list. That window slid forward as
the conversation grew, so each block was protected while fresh and digested one
turn later — rewriting a message the provider had by then cached. Measured against
the live API it cost **every cache read on every turn**, and compression that
halved token volume still doubled the bill, because the prefix was re-written at
1.25× instead of re-read at 0.1×.

Reversibility and decision-equivalence did not catch it: both are properties of a
single request, and this was a property of the *sequence*. What catches it now is
an append-only invariant test on each adapter — compress a growing conversation and
assert no earlier turn's bytes ever change. Recency is anchored to the client's
`cache_control` breakpoint (Anthropic) or dropped entirely for providers that cache
implicitly and commit everything they are sent (OpenAI, Gemini).

**It marks the boundary.** `adapters.anthropic.place_cache_control` places the
`cache_control` breakpoint at the end of the stable prefix, so the provider can
actually reuse what stayed identical. A stable prefix nobody marked is a discount
nobody claims.

**It sends re-reads as deltas.** `distil/cachedelta.py` (`--session-delta`) keeps a
per-session map of what has already been sent. When a file is re-read after an edit,
the second send is a diff against the first rather than a fresh copy — cross-turn
dedupe plus cross-version delta. This is a different lever from the provider cache
and composes with it: the cache makes repeats *cheap*, the delta stops them being
*sent*.

**It reports drift it did not cause.** `distil cache` folds the per-request ledger
into one picture, and deliberately mixes two different kinds of number: reads and
writes come from the provider's own `usage` — ground truth about money — while drift
is our diagnosis of *why*, from a content-free hash of the stable blocks we sent.

Three live requests, the third with a session id prepended to the system prompt:

```
requests        3
cache reads     31,614 tokens  (billed at a discount)
cache writes    15,819 tokens  (billed at a surcharge)
uncached        40 tokens
hit ratio       66.6% of cacheable tokens were reads
prefix drift    1 of 2 turns changed the stable prefix (50%)
                Each one re-bills the whole prefix. distil holds the cached
                span byte-stable (pinned by an append-only test), so look
                upstream: a timestamp or session id in the system prompt, or
                a tool list whose order varies. Before 1.45 distil itself
                rewrote the prefix every turn — upgrade before hunting.
```

The two halves are derived independently and agree: the turn our hash flagged is the
turn the provider re-billed 15,819 tokens to re-create. Turns 1 and 2 read the cache
even though `messages` grew between them, which is the property that matters — a
conversation is *supposed* to grow, and a diagnostic that called that drift would fire
on every healthy turn and be switched off within a day.

Worth stating plainly, since it is the lesson of the 1.45 bug: this diagnostic only
watches the *stable prefix hash*. It says nothing about whether the compressor is
rewriting content inside that prefix between turns, which is what actually happened.
The provider's own `cache_read` going to zero is the signal that catches that, and it
is why `distil dissect` now reports reads and writes next to the savings number.

Counts and a hash escape; prompt text never does.

## What distil deliberately does not do

**A response cache.** Hash the request, serve a stored response, skip inference
entirely. distil has no equivalent, and that is a decision rather than a gap:

- **Agent contexts grow monotonically.** Every turn appends. Two requests being
  byte-identical is rare outside retries and replays, so the hit rate is low exactly
  where the bill is high.
- **The correctness surface is sharp.** The key must include the system prompt, the
  tool set, sampling config and output shape, or two requests with identical
  `messages` collide and the second caller is served the first's answer. That is a
  wrong-answer bug, not a slow one.
- **It is the wrong layer.** A compression proxy narrows what a request costs. Deciding
  that a request need not run at all is an application's call, with the application's
  knowledge of staleness — and it can be built above distil without distil owning the
  risk.

Worth noting: the implementations we compared ship this **off by default** too.

## Cold prefixes

Byte-faithful forwarding only pays while the cache is warm. Once a session idles past
the provider TTL (Anthropic 5 min, extended on hit), resending the prefix verbatim
buys nothing — which makes that the safe moment to rewrite it, since there is no warm
cache left to destroy. distil's serving strategy is conservative here and does not
recompact cold prefixes today; the diagnostic above is what would tell you whether it
is worth it for your traffic.

## Checking your own setup

```bash
distil cache                       # hit ratio + prefix drift over recent sessions
distil cache --session <sid>       # one session
distil cache --json                # machine-readable, for a CI check
distil doctor                      # what the proxy is actually doing
distil stats                       # savings, and the last-7-days rate beside lifetime
distil proxy --session-delta       # cross-turn dedupe + cross-version deltas
```

`distil cache` reads the per-session request ledger, which only a wrap session writes —
with no proxied requests it exits non-zero rather than printing a reassuring zero.

Two things it will tell you that are easy to misread:

- **"not reported by the provider"** means the response carried no cache usage at all.
  Usually the prefix is under the model's minimum (Anthropic 1024–2048 tokens depending
  on model), or the client never sent a `cache_control` breakpoint. It does not mean a
  miss.
- **"no comparable pairs"** means fewer than two requests carried a prefix hash — rows
  written before 1.41 do not. That is not the same as no drift, and it says so.

If `distil stats` shows near-zero recent compression, the cause is usually mode
rather than cache: a subscription session defaults to lossless-only, which is Tier-0
only. `distil default --mode expand` turns the recoverable digest on permanently.

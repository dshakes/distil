# Changelog

All notable changes to Distil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [1.50.2] — a failure that healed itself was still reported as a failure

distil recorded **5,397 non-2xx requests across 18,455** and discarded the reason
every time. Only the status survived, so the one question a broken session raises —
*why?* — had no answer in distil's own logs, and `dissect` guessed on the user's
behalf: *"upstream errors or rate limiting."*

That guess spans two opposite diagnoses. "Your conversation outgrew the context
window" means nothing is wrong with distil; "distil sent a malformed body" means
everything is. A tool whose pitch is proof rather than trust should not be unable to
tell those apart, and this one wasn't: the missing field caused a real misdiagnosis of
a live session, where a size correlation looked causal and was not.

Two things now get recorded:

- **The provider's error `type`** (`invalid_request_error`, `rate_limit_error`, …) —
  a short enum. The human message is deliberately NOT stored: it quotes request
  content (`"prompt is too long: 231721 tokens > 200000"`). Streaming responses are
  relayed frame-by-frame and never buffered, so those record `http_<status>`.
- **Whether a retry recovered it.** A non-2xx immediately followed by a same-size 2xx
  is the SDK doing its job. On the session that prompted this, **100% of the 400s were
  retried and succeeded** — a ~30% "failure rate" in which nothing was ever lost.
  Reporting that as failure sends people hunting a bug that already healed.

Sessions recorded before this release report no reasons rather than an invented one.

## [1.50.1] — the tokens distil could not see

Extended thinking carries its payload under `thinking`, not `text`, so distil counted
it as **zero** — a 1,000-character block scored 0 tokens. On Claude 4.6+ prior-turn
thinking is re-sent as input and billed every turn, which means real billed tokens sat
in neither the before nor the after side of every percentage distil reports.

The blocks are still never rewritten, and that is not timidity: the provider pins each
block by its `signature` and re-expands the original server-side, so editing the text
achieves nothing and risks the signature being rejected on replay. But a cost distil
cannot reduce is exactly the cost it should not hide. Thinking now appears in the
token baseline and in the eligibility census as `thinking_billed`.

## [1.50.0] — the model that could never learn, and the store nobody could see

Two capabilities that were present in the code and unreachable in practice.

**Query-aware salience can finally train.** The shipped `query_weights.json` never
existed on any install, and the reason was structural rather than a cold start: label
collection ran only under `--expand`, and the labels themselves were expand events. So
the flywheel could only learn from the one configuration an independent benchmark
tells users not to adopt — and never at all on a subscription, where the digest tier
is off and no expand can occur.

Shadow mode already produces a better label. An A/B verdict says whether compressing
*this request* changed the agent's next action — the property the certificate is
about, not a proxy for it — and shadow runs by default at 2% on every configuration.
Training now treats a shadow decision-change as a positive alongside an expand, joined
on the request digest both sides already compute. A/A rows are excluded: those re-run
the same compressed request twice, so a disagreement there is provider
nondeterminism, and learning from it would teach the model to keep content because
the sampler was noisy.

**`distil memory` — the cross-agent recall store, made visible.** A handle minted
while compressing for Claude Code was always expandable from Codex, Gemini, or any MCP
client reading the same `DISTIL_HOME`: the store is machine-wide, encrypted at rest,
and needs no database. Nothing surfaced it, so nobody knew, and nobody could tell when
it was full. `distil memory` reports what is stored, how old it is, and warns within
10% of the cap — the point where the oldest handles start being evicted and a stub in
an agent's context quietly stops being expandable. `--clear` empties it.

Deliberately NOT built: an embedding-backed semantic memory. The comparable feature
elsewhere costs a vector database, a graph store and a sentence-transformers model.
distil ships zero runtime dependencies and installs anywhere on day one, and the
recall property users actually need — *get me back the exact bytes that were folded* —
is served by a keyed store rather than a similarity search over paraphrases.

## [1.49.0] — the agent said "done" and wrote nothing

**Fix this one.** An independent benchmark ran 75 agent sessions (Claude Code, Opus 5,
ground truth taken from the API's own usage fields rather than any tool's dashboard)
and found `distil wrap --expand` completing **6 of 15 coding tasks against bare Claude
Code's 13** — seven runs ended with an **empty diff**: the agent ran 18–51 turns,
wrote nothing to disk, and reported success. Zero empty diffs occurred across the
other 60 runs. On the long refactor it failed all three times, and the repository's
1,371 existing tests still passed, because nothing had been modified.

Four defects, all now fixed and each pinned by a regression test verified to fail
without its fix.

**The agent never ran its own tool call.** Claude Code emits parallel tool calls, so
one assistant message routinely carries both an `Edit` and a `distil_expand`. The
expand loop assumed `distil_expand` was the only tool call in the turn and spliced it:
the replay answered *only* the expand block — leaving the client's `tool_use`
unanswered upstream, an API-contract violation — and the continuation's
`stop_reason: end_turn` replaced the turn's real `tool_use`. Claude Code executes
tools only on `stop_reason == "tool_use"`, so it delivered the `Edit`, ran nothing,
printed the continuation's "all done", and stopped. A turn carrying a client tool call
is now terminal: relayed verbatim, stop_reason intact, no re-query. Guarded on all
four paths — streaming, Messages, Responses, Gemini.

**A failed recovery truncated the stream.** When the re-query failed, the connection
closed with no `message_delta`/`message_stop`. A terminator-less SSE message is
*truncated*, not finished, so the SDK retries it invisibly — wall-clock burned with no
progress and no error the user can see (the benchmark's 808-second, zero-write
signature). Every exit now emits a terminator.

**A file read could be digested before it was edited.** Recency is positional, so a
read from three turns ago was digestible — but the agent must still reproduce that
text character-for-character in an `Edit(old_string=…)`. Once the read is a digest no
exact match exists, so the edit silently fails or is never attempted. The exemption is
now keyed on *provenance* — results answering Read/Grep/Glob and their MCP equivalents
stay byte-exact at any age. Logs and test output are unaffected and still digest.

**Compression that saved nothing still cost cache.** On a subscription (lossless-only
by design, 0.0% savings correctly reported) distil still wrote **1.56× baseline
cache-creation tokens**, 2.52× on a short session. Two causes, both fixed: the
`distil_expand` tool was injected only once a handle existed, so the tools array —
which Anthropic caches *ahead of* the system prompt — changed shape the turn
compression first fired; and an unmodified body was re-serialized rather than
forwarded byte-for-byte. Injection is now session-sticky, and unchanged bodies
forward their original bytes.

`distil cache` also stops sending users to the wrong place: it blamed prefix drift on
"a tool list whose order varies" upstream, while distil's own tool list was the one
varying. It now names distil's own causes first.

**Hook mode can finally account for itself.** The same benchmark found hook mode the
strongest distil arm — 12/15 tasks, 0.94× baseline cache writes, the highest cache-hit
rate measured — and the only one that wrote no ledger at all, so its effect was
visible only in the provider's billing. It now appends a content-free receipt per
compressed result, readable with `distil hook --stats`.

- `distil wrap -- cursor` (and Copilot, Cline, Continue, Windsurf, Zed) now says these
  are editor extensions with no process to wrap and no base-URL variable they read,
  and names the proxy path that does work — instead of setting a variable nothing
  reads and reporting success.
- A missed expand handle is no longer logged as a successful recovery. The placeholder
  was being passed to the learning signal as content, so a *failed* recovery trained
  the keep-model that the block it could not return was safe to drop. `distil dissect`
  gains `expand_missed`, so `expand_resolved` finally has a denominator.

## [1.48.1] — a page that answers "which one am I?" before the jargon does

**Docs and one nudge.** Nothing new is callable and no behaviour changed, so this is
a patch: if you are tracking releases for capability, skip it.

distil has seven ways to run and the names are all jargon — wrap, hook, proxy, MCP,
library, gateway, statusline. A newcomer had to learn our vocabulary before they could
tell which one applied to them. `docs/which-mode.html` inverts that: two plain
questions (how do you pay, what are you running) and the answer falls out, with an
animated decision tree and honest savings ranges per surface.

The page leads with the thing most compression tools bury: **the mode barely matters,
what your tools print matters enormously** — JSON 28–33%, repeated log lines up to
99%, prose ~0%, and 0.00% on distil's own corpus.

- `distil onboard` now closes with a pointer to that page. It already detects your OS,
  agents and billing, but it could never mention MCP, the library, or the gateway,
  because those cannot be detected — which is how the MCP server (the only thing that
  works in Claude Desktop) stayed invisible to most installs.
- The provider-compaction paper gains an affiliation and its arXiv category
  (`cs.SE` primary, `cs.LG` cross-list), plus the submission step that is easy to get
  wrong: upload the `.tex` **and** its generated macro fragment, or every number
  renders as `--`.
- A nav-markup guard, after review caught two links sharing one `<li>` — a defect that
  had already shipped across 35 pages in the previous release's nav.

## [1.48.0] — the saving on a subscription is the window, not the bill

**A reader was right and our copy was wrong.** distil's README said a flat-rate
Pro/Max plan gets "context and latency, not the bill". But fewer tokens per turn means
more turns before the rate-limit window closes, and on a flat-rate plan that window
*is* the currency. The copy is fixed and this release makes the saving real.

**The proxy can't help there, by design.** Anthropic's consumer terms (§3, item 7)
restrict automated access on subscription credentials, so distil deliberately runs
`--lossless-only` on a subscription and measures **0.27%**. Your account isn't worth a
few percent.

**So use the door that's open.** `distil hook --install` wires a Claude Code
`PostToolUse` hook: the agent compresses its own tool output, in its own process,
through a documented first-party extension point. No proxy, no credentials touched.
Because a hook sees each result once and cannot rewrite history, compression is
append-only *by construction* — the fix direction our own cache-busting investigation
identified, enforced by the platform rather than by our discipline.

Measured on a paired live A/B, both arms answering correctly: tool_result **−38.6%**,
`cache_creation` **−67.4%**, cost-weighted **−68.3%**, decision-equivalence **5/5**.
Critically `cache_read` did *not* collapse — the proxy digest's failure mode does not
reproduce here.

**And the number that doesn't flatter us:** on distil's own eval corpus the hook saves
**0.00%**. Savings are shape-dependent — verbose JSON 28–33%, duplicated log runs up to
99%, prose and unique-line output ~0% — and our corpus contains none of the winning
shapes. Published because quoting only the favourable fixtures is the overclaim we
criticise in others.

- `distil hook --install / --selftest / --uninstall` — idempotent, preserves foreign
  hooks, refuses to clobber an unreadable `settings.json`
- `distil quota` — the rate-limit windows, read-only, fails open to "unavailable"
  rather than a fabricated zero
- `docs/subscription.html` with an animated diagram and a troubleshooting table
- The provider-compaction paper (`docs/paper/provider_compaction.tex`), arXiv-ready,
  every number generated from run artifacts by a script that refuses to emit LaTeX
  unless each report's protocol hash matches its pre-registration

Credit where it's due: Headroom shipped subscription quota telemetry against this
endpoint before we did. They built the instrument for the subscription user's real
currency while our copy still said that currency didn't count.

Windows note: three defects in this work were Windows-only (`os.uname()` in the code,
then in the test that proved the fix, then pytest IDs blowing the 32,767-character
environment-variable cap). All fixed, and the last one is now fenced by a guard
verified to actually fail.

## [1.47.0] — the key file was world-readable, and the audit trail wasn't watching

**Two things an enterprise security review would have caught before you did.** The
master key that decrypts every restore blob was written at the process umask —
measured `0o644` — and only chmod'd to `0600` afterwards. Any local user reading it
in that window decrypts everything. The gateway key file had the same shape: a
polling thread caught its temp file at `0o644` during a 400-key issue loop. Both now
create `0600` at open time, so the window does not exist. Re-probed after the fix:
1,855 samples, only `0o600` ever observed.

**The gateway had no audit log at all** — `grep -c audit` over `gateway.py` and
`authz.py` returned `0` and `0`. There was no record of who authenticated, which
tenant used which key, or what was refused. That is the first thing a security
questionnaire asks about a shared gateway, and SOC 2 CC7.2 / ISO 27001 A.12.4 both
require it. There is now an append-only, flock-guarded, `0600` JSONL trail of
`auth.ok` / `auth.fail` / `rate.limited` / `key.issued` / `key.revoked`, read with
`distil gateway audit` (`--json` for SIEM ingestion).

It covers every refusal path, not just the convenient ones: no credential presented,
no key store configured, invalid or revoked key, OIDC token rejection, OIDC RBAC
denial, OIDC success, both RPM paths, and both daily-token quota paths. The first cut
missed most of those — a deployment on OIDC got an empty audit log no matter how much
traffic it refused — and that gap was caught in review before shipping.

Content-free by construction: identifiers and outcomes, never prompt text, completion
text, tool output, or the raw `dsk-` token. Writes fail open, because bookkeeping must
never drop a customer's request.

**Keys can now expire.** `distil gateway keys issue --tenant acme --expires-in-days 90`
gives a key a bounded lifetime, enforced through the same `is_active` chokepoint that
enforces revocation — so no code path can honour an expired key by checking only
`revoked`. Keys issued without an expiry still never expire and pre-expiry key files
load untouched, so nothing changes for existing deployments. `keys list` reports
`expired` separately from `revoked`: an operator debugging a sudden 401 should not go
hunting for a revocation that never happened.

**Compression got faster without changing what it produces.** `_xor_stream` XOR'd
byte-by-byte in a Python loop; `must_keep` ran three to four regexes on every line,
where one pattern cost more than all the others combined. Vectorising the first and
putting literal prefilters in front of the second takes a 3.2 MB context from 1014 ms
to 719 ms per request, with savings percentages unchanged (95.8 / 92.3 / 81.9 / 48.3
at 60 / 30 / 12 / 4 turns). The encryption output is byte-identical — verified on 627
cases including leading-zero and partial-block torture — and the keep decisions are
identical across a 400,000-case differential test.

**Minified JSON was leaving most of its savings on the floor.** A single-line payload
returned before it ever reached the columnar folder that already existed for it: every
REST tool call, every `curl | jq -c`. Minified JSON goes from 10.5% to 57.5%, nested
records to 91.2%. The fold stays in-context lossless — all 2,500 field values across
500 records appear literally in the output — declines when values contain tabs or
newlines so no column can shift, and leaves the recency carve-out byte-identical.

Also fixed: `os.write` is permitted to write fewer bytes than it is handed, and the
first cut of the key-permission fix discarded that return value. A short write left a
truncated key, the next load minted a fresh one, and everything encrypted with the
first became unreadable. `Path.write_bytes` loops; the hand-rolled version did not.

Known limitation, documented in `atrest.py` where someone will hit it: concurrent
*first touch* of a brand-new key store can still leave two creators on different keys.
Five attempts to close it each passed on Linux and macOS and broke Windows a new way;
it wants a Windows runner in the loop rather than a sixth blind fix.

## [1.46.0] — every managed install was running blind

**If you installed distil the managed way, your proxy has never sampled a single
decision-equivalence check.** `distil wrap` defaults `--shadow 0.02` and
`--retention 0.05`; `distil proxy` defaulted both to `0.0`. The `com.distil.proxy`
launch agent runs **proxy**. So the two quality loops that make distil's central
claim checkable — shadow's live decision-change rate and the fact-retention meter —
were off on every managed install since the daemon shipped. A 93-minute, 767-request
session sampled 0. The same work under `wrap` would have sampled ~15.

Both defaults now match `wrap`, so existing installs are fixed by upgrading; no
reinstall. `--shadow 0` / `--retention 0` still opt out. Retention costs nothing
either way (in-process scan, counts only); shadow spends ~2% extra tokens on sampled
requests, which is the price of having evidence at all.

### `distil dissect` reported three different denominators for the same session

Reading one report end to end, four numbers disagreed:

- The request-detail lines summed **all** requests while the savings headline counts
  **booked** (2xx, non-retry) ones. 153 unbooked retries contributed 9.85M overhead
  tokens and 3.75M of "savings" that were never billed.
- `overhead_share` divided by the **pre**-compression payload — measuring the fixed
  tax against tokens distil had already removed. It read 28% where the session's own
  numbers implied 42%.

Both now use the booked population and the post-compression denominator. "Savings by
mechanism" reconciles with the headline it decomposes: a 3.75M discrepancy became
72k (0.34%), which is heuristic-vs-ledger tokenizer rounding. Same defect shape as
1.44.0's, which fixed it in `savings` and did not carry dissect along.

### "Everything summarized stays recoverable" was false while it printed

The restore store has a 500-blob LRU cap, not just a TTL. A single session folded 704
blocks and evicted 204 of them **mid-session** — including its most re-used fold, one
referenced by 272 requests — while the report promised full recoverability. It now
reports the measured count. The cap is configurable (`DISTIL_RESTORE_CAP`, default
raised 500 → 5000). A cap of 0 now means "no count cap" instead of evicting
everything, which is what `[:-0]` did.

### A flat 0.0% per-model row is now explained, not left looking broken

One model showed exactly 0.0% across 152 requests. That is policy, not a failure: the
traffic was 100% user-role string content with no tools — subagent calls — which
routes to Tier-0 lossless by design. The row now says so.

### Papers and published claims, audited against the code

- Both papers compile with **zero undefined references**. Fixed a missing
  `\bibitem`, a dangling cross-reference, and a figure quoting a number no artifact
  produces.
- The headline macro fallback silently rendered a **different operating point** as
  the headline (0.1% certified savings) when the generated macros were absent. It now
  warns at build time.
- `docs/paper/generated/headtohead_orig.tex` was untracked, so the built PDF rendered
  numbers absent from version control.
- The paper described cache-monotonicity as holding by construction. That property was
  **false in shipped code** until 1.45.0 fixed it. The passage now describes the
  breakpoint anchoring that actually ships, and reports the measured failure.
- The NeurIPS variant was missing the live-validation negative result — the section
  where a real grader failed the live margin twice, reproducibly. Ported.
- The security whitepaper said OIDC and role-based access control were "not
  implemented"; `distil/authz.py` implements both. Corrected to the honest gap
  (SAML, SCIM).
- Docs said the proxy needs Python 3.11+; the floor is 3.9.
- The site advertised 25.3% aggregate corpus savings over "8 domains" (listing 7).
  `distil bench` reports **48.4% over 9 domains**. We were underselling by ~22pp
  against our own free, offline gate.

## [1.45.1] — the explainer command was the one reporting uncalibrated numbers

`distil dissect` compared the raw heuristic token estimate against billed usage, so
its "off by >50%" tripwire fired on essentially every calibrated session. Calibration
is applied on every other surface; dissect is the surface whose entire job is
explaining the numbers, and it was the one quoting them uncalibrated.

## [1.45.0] — the compressor was rewriting the cache it was supposed to protect

**On cached agent traffic, distil cost about twice what sending nothing compressed would
have.** Provider prompt caching discounts a repeated prefix by ~10x, but only while it is
byte-identical. distil kept the freshest tool outputs verbatim so an agent could see its
latest observations exactly — using a window counted back from the end of the message list.
That window slides. Every block was protected while fresh and digested one turn later,
which rewrote a message the client had by then committed to its cached prefix. One changed
block, and the whole prefix is re-billed at the 1.25x write rate instead of re-read at 0.1x.

Measured against the live API, 10 turns, per-arm and per-run salt so no arm reads another's
cache entry:

    arm                     cache_create   cache_read        $   vs control
    control (no proxy)            29,446      146,572   0.0517
    lossless_only                 29,450      146,608   0.0517       +0.0%
    digest                        85,876            0   0.1076     +108.1%

Zero cache reads, on every turn. Compression halved token volume and still doubled the
bill. `--session-delta` did not help. This is the exact failure mode distil has always
attributed to naive compressors.

`docs/CACHE.md` claimed the property held "by construction", and two shipped CLI
diagnostics told anyone debugging drift that the cause must be upstream in their own
prompt assembly. Those are retracted.

The rule now is **exempt only content the provider will not have cached**. Anthropic caches
what the client marks, so recency is anchored to the last `cache_control` breakpoint — with
no marker there is no prefix to invalidate and the previous last-k window is kept. OpenAI
and Gemini cache implicitly and commit everything they are sent, so they digest on first
sight instead. There were four copies of the sliding window; all four now route through one
rule.

A second cause was found by looking for other per-request inputs: **query-aware salience**
was re-deciding how already-cached blocks compress, because intent terms come from the
newest user turn. Asking a different follow-up over the same history rewrote cached message
#0 with no recency window involved. Intent is now scoped to uncached content — a block's
first rendering still gets it, later requests reuse those bytes. On OpenAI and Gemini that
means query-aware salience no longer applies at all; those providers cache everything they
receive, so there is no uncached region for it to work in. That is a real reduction in what
the 1.16.0 feature does, taken deliberately against a prefix rewrite on every question.

After both fixes, same harness: **-60.3% vs control**, cache reads intact.

Pinned by append-only invariant tests on all three adapters — compress a growing
conversation and assert no earlier turn's bytes ever change. The existing gates could not
have caught this: reversibility and decision-equivalence are properties of a single
request, and this was a property of the sequence.

Digesting the freshest observation is what the certified strategy already does, so serving
moves toward certification rather than away from it. It is still a live behaviour change for
OpenAI and Gemini users and deserves a shadow run before it is treated as settled.

Also in this release: `lossless_only` measured **+0.0%** against no proxy at all — it forces
Tier-0-only, and `serve()`'s docstring claimed the opposite, which hid why live savings were
flat. Corrected. `--expand` lifts the force.

## [1.44.0] — the savings number could not explain itself

**A low percentage was indistinguishable from a broken compressor.** `distil stats` would
report 0.4% and stop there, and from the outside that reading has four completely different
causes: a request that was mostly the model's own prose (which distil never rewrites, by
policy), mostly recency-exempt tool output (a carve-out that shrinks as a conversation
grows), mostly fragments below the digest threshold, or a digester that was handed real
work and declined it. Three of those are the design holding. One is a defect. The saved-token
count is identical in all four, so the only way to tell them apart was to reason from the
outside about what the traffic probably contained — and that reasoning is wrong often enough
to send an investigation in the wrong direction for hours.

Every block is now attributed to the gate that actually claimed it, and the counts ride on
the per-request record. `distil dissect` sums them and prints the answer next to the number
that prompted the question:

    why: the model's own words (never rewritten) 54%, digester declined 25%,
         digested 12%, freshest tool output (kept byte-exact) 9%

A declined share of 5% or more is called out as the compressor's to explain. Otherwise a
protected majority is named as the design holding rather than a fault. Declined wins the tie
deliberately: a session can be 63% policy-protected *and* refusing a quarter of what it was
handed, and summarising that as "mostly protected, all is well" would file a real defect
under an exoneration.

The counts are taken at the decision points themselves, not by a second pass that re-derives
the rules — a parallel implementation is free to drift from the compressor it describes, and
a report that quietly disagrees with behaviour is worse than no report. The first draft of
this predicted the branch order and got it wrong: HTML extraction runs *before* the
minimum-lines gate, because minified markup is one enormous line, so every browser fetch
would have been filed as "too short to digest".

Sessions recorded before this ships render no `why:` line at all, rather than showing zeros.
"No data" and "nothing was eligible" are different claims and only one of them is true.

**Billed tokens never said what a request cost your plan.** Provider responses carry
`anthropic-ratelimit-*` headers — the only first-party signal for how a request draws down a
quota, as opposed to how many tokens it contained. The two diverge sharply on cached traffic,
where a read is billed at roughly a tenth of the metered price, and whether a subscription cap
discounts it the same way is not published anywhere. Those headers are now recorded per
request, including on the 429 that carries the most informative quota state of any response.
Captured at all three upstream reads, because they are genuinely separate code paths and the
`--expand` splice path is the one real agent traffic uses.

**`distil stats` quoted a digest rate from a window it did not belong to.** The rate beside
the trim was computed over all recorded history while the trim itself covered the recent
window, so a mode enabled today was described with a number earned by months of traffic under
a different one. Both figures now come from the same window.

## [1.43.0] — the safe default could not say what it cost, and the reload could strand you

**`distil default --always-on` could take macOS down.** The reload boots the job out,
replaces the plist, and then demanded a free port before bootstrapping. Since 1.42.0 the
plist declares `Sockets`, so launchd owns the listener and the SIGTERMed child that
inherited the descriptor can still be holding the port at exactly that moment — a held
port there is the design, not a foreign process. The check fired on that ordinary case,
and it fired *after* the bootout: the job was already unregistered, so the machine was
left with no proxy and nothing to restart it, while the command printed "your existing
setup is untouched". It was reproduced by nothing more exotic than switching modes.
Linux got this fix for the upgrade path in 1.42.1; macOS never got the equivalent. The
port settle is now a wait, not a verdict — if the port really is stolen, the bootstrap
still runs and the failure is reported with the job **registered**, so `KeepAlive` keeps
retrying instead of stranding the machine. And a failed reload no longer claims nothing
changed; it says the previous service is stopped and gives the command back.

**The status line called a fully-routed session a bypass.** With an always-on install,
the settings-file `ANTHROPIC_BASE_URL` pin outranks the env var `distil wrap` injects, so
the agent talks to the always-on proxy — which stamps ledger rows with its own session id
and cannot know the agent's. The wrap session's traffic marker therefore never flipped,
and after the grace period the line read `⚠ wrapped, agent bypassing proxy` for the life
of a session whose traffic was flowing through distil the entire time, with nothing the
user could do to clear it. The marker only ever proved "this wrap's own proxy saw
nothing"; the warning claimed "no distil proxy is seeing this traffic". The evidence now
matches the claim.

**A subscription default that could not say what it costs.** A flat-rate session runs
lossless-only so no digest is left unrecoverable and nothing is injected — deliberate,
and unchanged. But on the machine that prompted this it meant 8,319 runs at 0.27% while
that same ledger held 2,353 digest runs at 52.27%, and all distil ever said was that
near-zero "usually means lossless-only". No figure, no sample size. That line printed on
every one of those runs and was read past every time. It now quotes the rate you earned
on your own traffic, and states both halves of the trade — the safe default does not
touch the request, and `--expand` does — because you cannot weigh a choice shown only
one side of.

**You could opt into tool injection with no notice.** The disclosure was gated on the
`--lossless-only --expand` spelling only. The route the docs actually hand a subscription
user, `distil default --mode expand`, leaves that flag False, so the notice never fired —
and injection into a first-party session is precisely what the safe default promises not
to do. It is now keyed on real billing and fires on every route into it.

`policy.may_inject_tools()` is gone. It stated that injection is PAYG-only, a rule distil
does not implement, and **nothing ever called it** while a passing test asserted it — so
the drift between what distil said and what distil did was invisible in CI. An unenforced
policy with a green test is worse than no policy. [ADR 0006](docs/adr/0006-the-subscription-boundary.md)
records what the boundary actually is: consent, not recoverability. distil does not modify
a first-party session unasked. Recoverability is still why a digest is *safe* once opted
into; it is not why the default exists.

**Two more agents, and one needed more than a preset.** `distil wrap` routes **Grok
CLI** (`GROK_BASE_URL`; the `/v1` belongs to the base URL there, unlike the OpenAI SDK
which appends it) and **OpenHands**. OpenHands reads `LLM_BASE_URL` *only* when run with
`--override-with-envs` — without it the agent reads its own settings file and ignores the
environment entirely, so a preset alone would print success while routing nothing. `wrap`
checks argv and says so before the session starts. The IDE extensions (cursor, copilot,
cline, continue, windsurf) remain deliberately absent — no argv to wrap and no published
env contract — and a test now pins that.

**Oversized images can be downscaled, behind a recovery handle.** Duplicate elision cannot
touch a *first* occurrence, so a 2400x1200 screenshot still cost its full ~1,600 input
tokens every turn. [ADR 0007](docs/adr/0007-downscaling-behind-a-recovery-handle.md) permits
downscaling in exactly one form: the original goes into the RestoreStore first, and the
smaller image is emitted **as a pair** with a note stating what changed and the handle that
returns the untouched original. A downscaled image emitted alone would be silent loss and
is not allowed. Ships **off**, behind its own certificate, with **no bundled certificate** —
`vision`'s shipped result certifies a provably identical image, while this is a claim about
whether losing detail changes decisions on *your* screenshots. Needs the optional `[image]`
extra; inert without it.

**Networks that inspect TLS no longer make distil look broken.** Behind a corporate MITM
appliance every outbound connection is re-signed by an internal CA. `curl` and the browser
read the OS trust store; Python does not — so distil failed certificate verification while
every other tool on the machine worked. The upstream opener now honours `DISTIL_CA_BUNDLE`
/ `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` / `SSL_CERT_FILE`, a verification failure returns
the sentence that fixes it rather than a raw OpenSSL string, and `distil doctor` names the
bundle actually in effect. A path that does not exist is ignored rather than fatal. There is
no option to disable verification. See [`docs/ENTERPRISE.md`](docs/ENTERPRISE.md) §3b.

**`distil dissect` was reporting almost nothing on always-on installs, and the cause was one
unexported variable.** The savings recorder mints a session id when `distil wrap` did not set
one and put it on ledger rows — but every session-scoped *path* resolves that id from the
environment, and an always-on proxy runs under launchd or systemd, neither of which inherits
a shell environment. So per-request records, the session manifest and the liveness marker
were all silently dropped while the ledger looked healthy. `dissect` reported `blocks=0` on
sessions worth over a million input tokens: the analysis was never broken, its input was
never recorded. This was the third surface of that one cause — the status line calling
fully-routed sessions "bypassing", and ledger rows attributed to the proxy rather than the
wrap, were the first two.

**`distil learn --write` puts what a session measured where the agent will read it.** That
analysis has always lived in a report a human reads once; the agent that produced the cost
never saw it, so it repeated itself next session. This writes a managed block into
`CLAUDE.local.md` keyed to *your* project: which content type dominated, how much of the
bulky content it was, and the move that reduces it. It writes only to a `*.local.md` file
unless forced (a tracked `CLAUDE.md` is reviewed by your teammates), preserves everything
outside its markers byte-for-byte — including line endings and non-UTF-8 bytes — and
declines entirely when no single content type reached half the measured volume. A file of
plausible advice is one the agent learns to skim.

## [1.42.1] — the upgrade could not claim the port it had just been given

**`distil default --always-on` was broken on Linux in 1.42.0**, in both directions,
and macOS was unaffected — which is why it shipped.

Upgrading from 1.41.x: before 1.42 the systemd service self-bound the port and there
was no socket unit. On upgrade that old service is still *running* and still owns
`127.0.0.1:PORT`, so `enable --now distil-proxy.socket` cannot bind and fails — after
the unit files have already been overwritten. The user is left with new units, an
error, and the old proxy still holding the port.

Re-running on an install that already works: the socket unit holds the port **by
design** and keeps holding it after the service stops — that is the entire feature. A
reload that stops only the service and then demands a free port therefore aborts every
ordinary re-run with "something else is listening", pointing the user at a culprit that
is distil itself.

Both are one fault: ordering. The reload now stops **both** units, waits for the port,
starts the socket so systemd owns the listener, then starts the service, which receives
the descriptor instead of binding for itself. No single `enable --now` can express that
sequence, which is why the command string `service_spec` used to return could never
have been right — it is now a marker, and the last copy of the racy
`launchctl unload; launchctl load` is gone from the codebase.

Three more faults in the same area:

- **Extra descriptors were leaked *and* hung.** A dual-stack socket unit hands down more
  than one fd; distil wrapped the first and left the rest open. The leak was the lesser
  problem — connections arriving on an unaccepted listener queue in the kernel forever,
  so a client **hangs** instead of failing. A hang is worse than a refusal: nothing
  surfaces an error to act on. Extras are now closed on both platforms.
- **`service_reload` could hand the user a traceback.** Its sibling guarded its
  subprocess calls; it did not, so a `launchctl`/`systemctl` that hung past its timeout
  printed a stack trace where a sentence belonged.
- **Socket cleanup was gated on the `.service` file existing.** A socket unit can
  outlive its service — a partial uninstall, a hand-deleted unit, an interrupted upgrade
  — and it is the unit that owns the port. `--undo` and `offboard` now clean up when
  *either* is present.

Also: `libc.free` is given explicit `argtypes`. ctypes' default conversion happens to
pass a 64-bit pointer correctly; "happens to" is not a contract.

Every fix has a regression test verified to fail without it. One of those tests is a
correction in itself: the first version stubbed `_port_free` to `True`, mocking away the
exact check under test, which is how the re-run break got past a green suite. It now
drives the real function against a genuinely held socket.

## [1.42.0] — the outage was the test suite

**distil's own test suite tore down the developer's live proxy, and that is what took
a machine's API traffic out roughly fifteen times in one evening.** `cmd_offboard` and
`distil default --undo` call `service_spec(8788, ...)`, which returns the *real*
`~/Library/LaunchAgents/com.distil.proxy.plist` — then `launchctl unload` it and
`unlink()` it. Five tests never patched `service_spec`. Every full-suite run silently
uninstalled the always-on service while `ANTHROPIC_BASE_URL` stayed pinned at the
now-dead port, so every session afterwards failed with `ConnectionRefused`, an error
that names the *provider*.

It was hard to see because the job came back **unregistered** rather than crashed:
`proxy.err` was empty, there was no crash report, and `KeepAlive` cannot restart a job
that is no longer loaded. Remembering to patch per-test fails open, and it failed open
five times — so `conftest` now redirects the plist into a tmp dir and makes the
destructive commands inert *by construction*.

**`launchctl unload; launchctl load` was a coin flip.** `unload` is asynchronous, so the
replacement job often died on `EADDRINUSE` and launchd discarded it — while the legacy
`load` still exited 0 and the orphaned old process kept answering just long enough for
the routing probe to pass. `distil default` printed ✓ and wired the pin onto a machine
with *no registered job*; nothing restarted it when the orphan exited. `service_reload()`
uses the modern bootout/bootstrap API, waits for the job *record* to clear (the port
frees the instant the process dies, while the record lingers in `SIGTERMed` — and
bootstrapping into that returns `Bootstrap failed: 5: Input/output error`), retries
transient EIO, and verifies a running pid. A failed start now wires nothing.

**A restart is no longer a refusal.** launchd and systemd own the listening socket
(`Sockets` in the plist; a real `distil-proxy.socket` unit on Linux), so connections
queue in the kernel backlog while the proxy restarts instead of being refused. Verified
on hardware: `kill -9` against the running proxy four times returned HTTP 401 every
time, `connect=0.26ms`, zero refusals — where the same machine had previously logged a
37-second window of `ECONNREFUSED`.

Nine more faults in the machine-wiring path, each with a regression test verified to
fail without its fix:

- A pre-existing `alias claude=...` made the rc file **unparseable**. bash and zsh
  alias-expand the word before `()` while *reading* a function definition, so the
  managed block was a syntax error that abandoned the rest of the user's rc — PATH
  included. An earlier distil installed exactly such an alias.
- `os.replace` **destroyed a symlinked rc**, detaching dotfile repos (stow, chezmoi)
  from the file they manage.
- A hyphenated agent name (`claude-code`) is a syntax error in dash, which is `/bin/sh`
  on Debian and reads `~/.profile`. A subshell capability probe now gates the
  definition. `eval … || true` does *not* work: `eval` is a POSIX special builtin, and
  a syntax error in one exits the shell before the `||` is reached.
- The plist spliced paths and mode into XML **unescaped**; a `&` in `$HOME` produced a
  plist launchd silently refuses to load.
- The escape hatch matched loopback by **substring**, so a corporate gateway at
  `…/pools/127.0.0.1` would have been deleted and `http://127.1:8788` kept.
- The escape hatch scanned only hardcoded `$HOME` rc paths, missing `$ZDOTDIR` and any
  `--rc` path — the machines least served by the defaults.
- `remove_managed` returned `ok` when the end marker was missing, so undo and offboard
  printed ✓ while the wiring stayed live.
- `_port_free` probed by **connecting**, which fills the listener backlog: against a
  hung proxy it reported a held port as free, causing the exact `EADDRINUSE` it exists
  to prevent. It now probes by binding, which is what the supervisor does.
- Nothing removed the systemd **socket unit** on uninstall. An orphaned enabled
  `.socket` keeps the port bound after distil is gone and keeps systemd starting a
  service that no longer exists.

Also: the subscription notice drops from seven lines to two (it prints on every bare
wrap, and a wall of text is skipped exactly by the readers it exists for), and
`CITATION.cff` gains the drift guard it never had — it was the one version location no
test pinned, and it had already fallen behind to 1.41.1.

**`distil shadow-stats` now reports the running build's replay-failure rate, not a
lifetime average** (#96). The counters accumulated for the life of the install, so
failures from an already-fixed bug stayed in the displayed rate forever: one install
read 42% lifetime while the rate since the fix was 5.3%. That does two things, and the
second is the one that matters — a fixed bug makes the sampler look permanently broken,
and the next *real* regression has to clear that stale noise floor before anyone
notices. Counters are now bucketed by version (lifetime totals kept, as context and as
a trend), and `last_fail_reason` becomes a `fail_reasons` histogram, so a mixed run says
`1 failed (429×1)` instead of reporting whichever failure happened to land last.

**Socket activation on Linux is now proven rather than asserted.** The launchd half was
verified on hardware from the start; the systemd half was covered only by tests against
mocked units. Two Linux-gated tests close that: an end-to-end check that does what
systemd does (bind, hand the listener down as fd 3, exec, then SIGKILL the worker and
demand the next request still be served), and `systemd-analyze verify` on the generated
units — because asserting on our own generated strings cannot catch a directive that is
misspelled, deprecated, or illegal in the section it was placed in. Both run on the
Linux CI matrix.

**A security test that had never once executed now does.** `test_authz.py`'s RS256
malformed-key assertion guards itself with `importorskip("cryptography")`, and the CI
gate never installed it — so it skipped silently in every run since it was written. The
same shape as the outage above: a green result that was green because the thing wasn't
running.

**What is and isn't verified**, since this release changes machine wiring on two
platforms. macOS is proven on hardware: `kill -9` against the running proxy four times
returned HTTP 401 every time, and `distil default --always-on` went from roughly a coin
flip to 5/5 deterministic. On Linux the `LISTEN_FDS` handoff is proven end to end and
the generated units are accepted by systemd's own parser, both on every CI run. What is
still *not* covered by a test is systemd actually starting the units and handing the
descriptor over on a live host — `systemd-analyze verify` proves they are well-formed,
not that a running systemd wires them. Windows is unaffected by the wiring paths and is
only verified not to break.

## [1.41.2] — the escape hatch that wasn't, and a price nobody was charged

**`distil default --always-on` took `launchctl load` returning 0 as proof the proxy
worked.** It means *the job was accepted*, nothing more. On one real machine the proxy
bound its port and 404'd every request — a `--upstream` pointed at a server that does
not serve `/v1/messages` — and because Claude Code skips model-name validation whenever
`ANTHROPIC_BASE_URL` is set, every session on that machine failed as "there's an issue
with the selected model". The message sends you hunting the model. The fault was the
base URL, and distil had written it.

`probe_routing` now POSTs `/v1/messages` and demands anything but a 404 — a 401 is
*proof*, because it means the request reached a real messages handler. `--always-on`
starts the service, proves the route, and only then writes the base URL; a failed
preflight wires nothing and exits 1. `doctor` probes the same way: "the port is
listening" was the old bar, and it was already true the entire time everything was
broken.

**The second defect was worse, because it was the way out.** Both removal paths —
`default --undo` and `offboard`, the command someone runs when they have had enough —
looked only in `~/.claude/settings.json` and matched the entry against a port: `--undo`
against the current `--port`, `offboard` against a hardcoded `:8788`. Claude Code also
merges *project* settings, and those take **precedence** over the home file. `offboard`
carried a third defect of its own: it prompted only if the home file existed, so on a
machine without one it asked nothing and cleaned nothing, in any file — then reported
success and printed the uninstall command.

All of it was true of the same machine at once. A dead `127.0.0.1:8788` in a repo's
`.claude/settings.local.json` survived uninstalling distil **entirely**, and went on
killing every session started in that directory.

`claude_settings_files()` enumerates every file Claude Code merges, in precedence
order. `unwire_base_url()` cleans by shape — any loopback URL is ours — while leaving a
real remote gateway (LiteLLM, Bedrock, a corporate egress) untouched even under
`--yes`. `loopback_base_url()` is the read-only probe the prompt was missing, so
`offboard` asks only where something is really there and names the value it found.
`doctor` diagnoses the winning entry and reports the ones it shadows as shadowed,
because fixing a loser changes nothing and sends you in circles.

**`claude-opus-5` was not in the pricing catalog.** The catalog knew `claude-fable-5`
and the Opus 4.x line but not the model actually serving traffic, so savings on it
resolved to no price and rendered as $0 — under-reporting real work without ever
failing. Now $5.00 / $25.00 per MTok. `claude-mythos-5` is deliberately absent: same
$10/$50 as fable, but Project Glasswing only, so it would be a guess at a model most
callers cannot reach.

**`tests/conftest.py` sandboxes the new reach.** `--undo` and `offboard` can now
rewrite Claude Code settings machine-wide, which is precisely why no test may keep it:
a suite run from this repo would otherwise rewrite the developer's own `~/.claude` —
the accident that started all of this.

---

**rc2 — the same failure, one layer up.** rc1 fixed `--always-on` writing a base URL it
had not proved. Three outages on one machine in a single day showed the other half: a
base URL in `~/.claude/settings.json` **outranks** the environment `distil wrap` sets.
So when the always-on service was not running, wrap dutifully started a proxy, exported
its port, and Claude Code ignored it and dialled the dead one. Every session failed with
`API Error: Unable to connect (ConnectionRefused)` — an error naming the *provider*.
`distil wrap` now refuses to start into that configuration and names the fix, and the
end-of-session warning no longer guesses "agent update?" when it can check.

**Distil is embeddable.** `from distil import compress_messages, expand_handle`. The
package previously exported only `__version__`, so nothing could depend on it without
shelling out to the CLI. Not named `compress`/`expand`: `distil.compress` and
`distil.expand` are modules, and Python binds a submodule onto its parent on import, so
those names would resolve to the function or the module depending on unrelated import
order. A test enumerates submodules and fails on any future collision.

**`▼413 · 0% smaller` is gone.** A real token count beside a rounded zero reads as
broken, and it is the likeliest first-run uninstall trigger for traffic that genuinely
compresses to very little. Sub-1% now renders `<1% smaller` with the reason, across all
four surfaces. `doctor` also stopped reporting a routing proxy as bypassed: it inferred
traffic from the savings ledger, which skips zero-saving windows by design.

**A real TypeScript library.** The npm package was a CLI bridge. `compress(messages)`
now runs natively in Node and is **byte-identical** to the Python engine, enforced by a
92-case conformance suite. The digest tier is deliberately not ported — it mints handles
into a store shared with the proxy and MCP server, and it is the tier the certificate
measures. Where JS cannot reproduce Python's bytes (integral floats, integer-like keys)
the port declines rather than emitting uncertified output.

**Enterprise surface.** OIDC + RBAC on the gateway (three ordered roles, stdlib-only
JWS; RS256 refused rather than unverified when the extra is absent), a Helm chart with
secure defaults asserted by test, Prometheus alert rules and a Grafana dashboard whose
every expression is CI-checked against the metrics distil actually emits,
`distil_requests_rejected_total` so quota enforcement is visible, `SECURITY.md`, and a
security whitepaper that states the gaps — no SOC 2, no SLA, no SAML — rather than
omitting them.

**Reach and honesty.** `opencode` and `qwen` wrap presets; Agno and Strands
integrations; `docs/IDE-AGENTS.md` documents the proxy route for Cursor/Cline/Continue
and says plainly that Copilot is not redirectable. The README now leads with what distil
does and moves the proof below the fold.

**Fixed while getting here:** `npm test` ran `node --test test/`, which on Node 22
resolves `test` as a module and dies — the package's tests were not running; CI
hand-listed one npm test file so new ones never executed; and `cli.main()` left SIGPIPE
at `SIG_DFL` process-wide, so any later broken pipe killed pytest with exit 141 and no
failing test to point at.

## [1.41.1] — the diagrams, and the one nobody could turn off

Documentation and accessibility only. No compressor, proxy or CLI behaviour changes.

**Six diagrams were stills.** Each now carries the motion its own argument needs
rather than a uniform fade: `ast-delta`'s three "same AST → unchanged" verdicts land
one at a time, because the claim is cumulative — a reader has to watch reformatting
fail to count, three times, before "1 / 3 defs" means anything. `io` sends a packet
down each leg, because the point is that *both* directions are billed and a reader who
only sees the outbound arrow half-remembers that. `vision`'s duplicate tiles dim as
they are elided, so it reads as "these two, and only these two". `observability`
reveals its four scopes left to right, because the claim is that each is *wider* than
the last. `install` is a menu rather than an argument, so its cards simply deal in.
`banner` is deliberately restrained — it is the README hero, and the only thing that
moves is the mark's three bars, which already encode compression by narrowing.

`logo`, `logo-lockup` and `og` stay static. A favicon that moves is a bug, and every
social platform renders an OpenGraph card as a still — animating it risks the frame a
scraper happens to capture being a half-drawn one.

**`hero-terminal.svg` had twelve animations and no `prefers-reduced-motion` guard.**
It had been on the landing page since the redesign, through an entire accessibility
series, because nothing checked. It is also the awkward case: it plays once and
freezes rather than looping, so the usual `animate { display: none }` would have left
it permanently blank — every group starts at opacity 0, the typed lines are revealed
by zero-width clip rects, and the result box is drawn by a dash offset. The guard
asserts the end state of each. A hero that renders as an empty terminal for anyone
with reduced motion enabled is worse than one that animates.

**Four diagrams had no accessible name at all** — `cache-aware`, `cross-sdk`,
`domains`, `head-to-head`. `<img alt>` names them on a page, but an SVG is also a
document people open directly (GitHub renders `docs/assets/*.svg` as a page) and there
`alt` does not exist. Titles and descriptions now travel with the files.

**`tests/test_svg_assets.py`** makes all of it permanent: well-formed XML, a
reduced-motion guard on anything that moves, `keyTimes` that start at 0 and end at 1
and are non-decreasing, `values`/`keyTimes` counts that agree, and an accessible name.
Both failure modes were verified to fail the test rather than assumed to. Static and
unnamed assets are allowlisted with their reason, so the next one has to be argued for.

## [1.41.0] — what the cache bought, and what a name is worth

Three things of the same shape: a claim distil could make but not show, and a number
that was true for the wrong reason.

### `distil cache` — the observability half of cache alignment

distil already *prevented* prefix drift by construction. `strategies.distil` compresses
the volatile tail and leaves every stable block byte-identical, so compression cannot be
the cause of a cache miss. What was missing was any way to **show** it — and any way to
name the case distil cannot fix: a caller whose own system prompt carries a timestamp.

The report deliberately mixes two kinds of number. Cache reads and writes come from the
provider's own `usage` — ground truth about money. Prefix drift is distil's diagnosis of
why, from a content-free hash of the stable blocks it sent. It never prints one without
the other, because a diagnosis with no measurement behind it is a guess.

```
requests        3
cache reads     31,614 tokens  (billed at a discount)
cache writes    15,819 tokens  (billed at a surcharge)
hit ratio       66.6% of cacheable tokens were reads
prefix drift    1 of 2 turns changed the stable prefix (50%)
```

Verified live on both the streaming and non-streaming paths: the prefix held across a
turn append, then a session id prepended to the system prompt forced a re-create. The
turn the hash flagged is the turn the provider re-billed — derived independently,
agreeing. Only `system`, `tools` and `tool_choice` are hashed: a conversation growing is
not drift, and a warning that fires on every healthy turn is one people switch off.

The proxy now records `usage_cache_read` and `usage_cache_create` separately. Summed,
they cannot tell a working cache from a thrashing one — a write is a surcharge, a read a
discount, and a prefix that drifts every turn writes forever and never reads.

With no proxied requests, `distil cache` exits non-zero rather than printing a
reassuring zero. A cache report over no data reads as a clean bill of health for a
session nobody measured.

New: [docs/CACHE.md](docs/CACHE.md), including the one cache feature deliberately **not**
shipped — a response cache — and the three reasons why.

### BFCL was being scored as prose

Its golds are bare code names; the generic matcher anchors on non-word boundaries. On
the n=25 sample that credited **11 of 85 golds by accident** — `'a'` matching inside
`"tool-schemas"`.

Names are now matched as **identifiers**: a quoted JSON token, tolerating one level of
escaping. The escaping is the part that makes it work — a tool schema is JSON nested
inside a JSON payload, so its quotes arrive as `\"base\"`. An earlier attempt requiring
a bare `"base"` read **2.9%** on a payload where every name was plainly present; that
number measured the matcher, not the compressor. This removes the "upper bound" caveat
that shipped with 1.40.0.

Support recall is now **100.0%** — and the report says what that means. **Visible recall
is 0%**: the schema sits behind a restore handle, one `distil_expand` away. At matched
savings (89.3% vs 90.1%) truncation keeps **0 of 70** names; distil keeps all 70, none of
them on screen. The table prints `visible → true support: bfcl 0%→100%` rather than the
flattering figure alone, because a reader who assumes the model can *see* a schema it
must first expand has been misled by numbers that are each individually correct.

Fifteen golds BFCL genuinely names `a`, `b`, `c` are excluded **and counted**. A
one-letter token occurs in almost any English text, so it can be neither credited nor
failed honestly — the same treatment as an abstractive answer. A recall computed against
a quietly reduced denominator is an inflated result.

### `make gate` was weaker than CI

CI ran `distil validate`; the Makefile did not. A local green gate promised a push it
could not deliver — the one thing that target exists to prevent. Added, along with a test
that compares the two by invoked subcommand and fails naming the missing one.

### Also

- Three tests fetched BFCL **live** to assert on `--min-recall` gate logic. Seven CI
  runners hitting the same endpoint produced an HTTP 429, and the assertion degraded into
  comparing its expected string against a rate-limit message. A red gate meaning "a third
  party throttled us" trains people to re-run until green. They run offline now.
- `eval-stack.svg` still said five gates and omitted `distil suite`. Rebuilt animated,
  six gates lighting in run order.
- `phantom-file.svg` and `cache-delta.svg` animated — the deletion is struck through and
  dimmed, and the two re-read bars race at true relative scale. Both resolve to their end
  state under `prefers-reduced-motion`.
- `cli.html` never got a `suite` section from 1.40.0. Added, with `cache`.
- `evals.html` was marking Corpus as the active nav page.

## [1.40.1] — the numbers, corrected

Bug fixes and honest reporting. No compressor, proxy or CLI behaviour changes; the
1.40.0 eval suite is unchanged.

**`distil stats` crashed on a legacy Windows console.** The savings line contains
`→`, and Python's default `errors="strict"` turns that into a `UnicodeEncodeError`
mid-render on a cp1252 terminal: exit 1, a traceback, output truncated. It only fired
once a ledger had baseline tokens, so it had been invisible since the line was
written. Streams now degrade with `errors="replace"` instead of failing — a
reporting tool must never fail on the report.

**A lifetime figure was being read as the current rate.** Validating the published
adoption numbers against this repo's own ledger found both correct and both
misleading the same way: `−20.4%` lifetime against `−0.4%` over the last 7 days, two
orders of magnitude apart, with only the lifetime number shown. `distil stats` now
prints the recent window whenever it disagrees, names the cause, and names the
remedy. A window that comes out LARGER than baseline reads `+10.0% LARGER` rather
than the previous `−-10.0%`, and gets advice for overhead rather than for
lossless-only.

**The subscription default never said what it costs.** A subscription session runs
lossless-only so no digest is left unrecoverable — correct, and Tier-0 only, which
measures ~0-2% against the 30-60% the recoverable digest reaches. The notice now
leads with that cost and offers the persistent opt-in (`distil default --mode
expand`) rather than only the per-run flag. **The default itself is unchanged.**

**Offline artifact parsing.** A tool call's own ARGUMENTS no longer condemn it — a
successful `Write(file_path="a.py", content="No such file or directory")` was
recording nothing. The call boundary is parsed (paren depth, quote-aware) rather
than guessed from a `->` delimiter, and only a closed set of recognised invocation
heads gets that immunity, so an exception repr or a result wrapper is still read as
evidence. Ops inside one call share its outcome, so a failed
`Bash(command="rm a.py && touch b.py")` records neither. The live path was never
affected: it knows the real call/result boundary.

Also: `--max-silent` diagnostics name every component of their own total, the eval
record's `subject` in the docs matches what the CLI emits, and an output surface
that changed blocks but endangered no facts says so.

## [1.40.0] — what recall cannot see

Four state probes, both priced surfaces, and every result as a reproducible record.

`distil retention` answers "is the fact still there". `distil fidelity` answers the
questions that survive a yes:

- **artifact state** — is the file's final STATE right, or only its name present?
  `stale` (present, wrong) is reported apart from `lost` (absent), because a
  compressor that drops a whole file history is safer than one that preserves half
  of it: the first leaves a gap the agent can see, the second leaves a false belief
  it acts on.
- **overclaim** — did the value keep its uncertainty? `"approximately 4200 ms"` →
  `"4200 ms"` is a distortion every recall metric scores perfect. Reversing a bound
  (`at least 3` → `at most 3`) is reported separately as `inverted`.
- **continuation** — does the agent still know what is left to do?
- **propagation** — does a loss at turn *k* show up as a changed decision at *k+n*?

Both halves of the bill are graded: what the model READS, and what its own past
answers cost when they re-enter as history.

`--json` now emits an **eval record** rather than bare metrics — schema version,
dataset fingerprint, subject identity, grader provenance, and gates carrying
threshold, observation and outcome. A number nobody can reproduce or compare is a
number nobody should act on. The schema ships at `schemas/eval-record.schema.json`
and is validated against real output.

**Measured on the shipped reversible tier:** artifact state 100% (7/7), hedge
fidelity 94.7% (162/171) with 9 genuine overclaims, pending-work recall 100%,
output surface 6 blocks digested with 4 facts removed and 0 silent failures. CI
gates at `--max-silent 15` — the *measured* band, not zero, because gating at zero
would assert a property the compressor does not have.

The rule these probes enforce is that **no gate may pass without evidence**, and
the cross-audit found it broken in six places — each one green, each one measuring
nothing: an empty gate list, a fresh shadow ledger, an unscored output surface, a
propagation profile with no events, a threshold accepted without running, and a
probe with a zero denominator. All six now fail rather than certify. Three sweeps
close the classes underneath: every alternative of every alternation must fire, no
gate may certify an evidence-free input, and every `distil …` command printed in
the docs must actually parse.

Also in this release: the SDLC auto-fix loop closes end-to-end (#74–#80). A
`GITHUB_TOKEN` push raises no workflow events, which had silently broken three
chains.

Upgrading is safe and additive — no behaviour of the compressor, proxy or CLI
changes. `distil fidelity` is new; every existing command keeps its output shape.

## [1.39.1] — the release path, hardened and then actually run

**The installed package is unchanged.** Every commit since 1.39.0 touches
`.github/workflows/` or `packaging/homebrew/distil.rb`, neither of which ships in the
wheel — `distil/` and the bundled corpus are byte-identical to 1.39.0. Nothing about
compression, the proxy, or the CLI is different, and there is no reason to upgrade for
behaviour. It is cut so the hardened publish path executes against a version that is not
yet on PyPI, npm, or the tap, rather than waiting to find out during a release that
matters.

### Fixed — publish jobs that reported success while doing nothing

- **A missing `NPM_TOKEN` now fails the release** instead of exiting 0 behind a warning.
  That silent skip is how npm sat on 1.25.0 while PyPI reached 1.39.0 — fourteen
  releases, each green, publishing nothing, while the README carried an npm badge and an
  `npm i distil-llm` instruction aimed at the stale build. The registry now holds exactly
  `1.25.0` and `1.39.0`, which is the evidence.
- **A missing `TAP_TOKEN` now fails the release** for the same reason, behind an even
  quieter `::notice::`. It worked only because the token happened to be set.
- Both check idempotency **before** demanding a credential, so re-cutting a release with
  nothing to publish neither fails nor needs one. For the tap that comparison covers the
  url, the `sha256` **and** the version: a re-cut tag keeps the version and changes the
  hash, and matching on version alone would have skipped repairing a formula whose
  `brew install` fails its checksum.
- A tarball fetch failure now says *"does the tag exist?"* instead of a bare `exit 56` —
  the guard existed but was unreachable under `set -euo pipefail`.

### Fixed — auditors that never saw the fix

- **Both cross-audits now run on `synchronize`.** They triggered on open only, so an
  auditor reviewed the commit that *contained* a bug and never the correction, unless
  someone applied the `agent:audit` label by hand. Bot pushes are skipped only on
  `synchronize` (the fix loop's rounds); PRs *opened* by Renovate, Copilot and release
  bots still audit, because a blanket bot exclusion silently withdraws coverage from
  dependency and generated-code changes.
- **Concurrency is job-level, not workflow-level.** A workflow-level group is joined by
  the run before the job's `if` is evaluated, so an event the job will skip still takes a
  slot — cancelling the audit in flight, and evicting a queued one, since GitHub keeps a
  single pending run per group. The PR ends with no verdict, which reads exactly like a
  clean one. A skipped job never joins the group at all.

## [1.39.0] — certify the HTML transform, and a headline one fixture cannot swing

### Documentation

- **`langchain-distil` is now named in this repo.** LangChain merged
  [langchain-ai/docs#4943](https://github.com/langchain-ai/docs/pull/4943), listing the package in its
  community middleware integrations with `docs_url` pointing here — and the README,
  `docs/langchain.html` and `docs/integrations.html` mentioned it exactly zero times. A reader
  clicking through from LangChain landed on a page that never named the package they had just
  been told to install. All three now carry install and usage for the real entry points
  (`compress_messages`, `pre_model_hook`, `as_runnable`), and three contract tests pin the
  symbols the published wrapper imports, so renaming one can no longer break it silently.
- **The corpus is eight domains, and the site said seven** — 42 stale claims across 16 files,
  caused by this release's own `web-research` trajectory. `domains.svg` gained the row (89.8%,
  PASS, 428 pruned), the viewBox to fit it, and the animation stagger the other seven have.

### Added

- **An HTML trajectory in the certification corpus** (`corpus/web-research.json`). 1.38.0
  shipped the HTML transform default-on in the serving path while every corpus
  trajectory carried logs, JSON or prose — so `distil bench` never routed a byte through
  `compress/htmlx.py` and the transform was certified by nothing. ADR 0003/0004 set the
  precedent that a new content type is certified before it goes default-on; this pays
  that debt.

  A 4-turn web-research agent whose tool results are real HTML documents: chrome the
  extractor must strip (script, style, nav, cookie banner, aside, footer) wrapped around
  an `<article>` carrying the DECISION the oracle grades. It certifies at **89.8%
  savings**, and it has teeth — an extractor patched to swallow `<article>` drops
  `match_rate` to 0.0 and fails the gate.

### Changed

- **The retention headline is now a per-domain macro average, so one fixture cannot
  swing it.** `distil retention` reports **21.4%** — the mean across all 8 domains, each
  counted once — and prints the fact-weighted figure beside it rather than as the
  headline.

  The reason is the previous entry. Adding a single HTML trajectory moved the
  fact-weighted number from 9.8% to 62.6% with nothing about the compressor changing:
  `web-research` alone contributes 666 of the 1083 facts at 4.4% visible, because HTML is
  dense in `href` URLs that extraction drops as navigation. A number that a fixture can
  move that far is measuring the corpus, not the compressor.

  Both are still reported, because the two diverging is itself a signal that the corpus
  is unbalanced — and the report gained per-domain rows, which is the read that actually
  shows the spread (0.0% on `coding`, 95.6% on `web-research`). The JSON payload leads
  with `macro` and labels `micro` with why it moves. `--max-lost` is unaffected: a lost
  fact is a count, not a ratio.

## [1.38.0] — recall, and a number a stranger can check

Every quality gate distil shipped until now was graded on **our** corpus against
**our** oracle. The statistics were rigorous; the external validity was zero — a
reader had nothing they already trusted to check us against. And none of it answered
the question users actually ask, which is not "did the next action survive?" but
"is the number I needed still there?"

### Added

- **`distil retention` — fact-level recall.** Keyed numerics, artifacts
  (paths/URLs/hashes/UUIDs) and error lines are classified `retained` /
  `recoverable` / `lost`. `recoverable` is **verified, not assumed**: the handle must
  appear in that block's own compressed text *and* the fact must appear in that
  handle's restore bytes. On the corpus: **100% true recall, 90.2% visible, 0 lost** —
  so being reversible rather than lossy is worth **9.8% recall**, the first time that
  property has had a number instead of an argument. Now a per-commit CI gate
  (`--max-lost 0`); it is zero-cost, needs no API key and no network.
- **`distil retention --dataset {hotpotqa,squad}` — public ground truth.** Graded
  against answer keys written by someone else, next to a truncation baseline tuned to
  reproduce distil's own token savings on the same case. HotpotQA (n=100, 14.3%
  savings): **100% answer recall and 100% gold-sentence recall, vs 91.6% / 82.7% for
  truncation at 14.1% savings.** Rows load from the HuggingFace datasets-server REST
  API as plain JSON, so the core stays **stdlib-only** (`dependencies = []` holds) and
  cache under `$DISTIL_HOME/datasets` for offline reproduction. Runs nightly, not
  per-commit: a required check gated on a third-party host would make main's health
  hostage to their uptime.
- **`distil retention --live` — recall on your own traffic, content-free.** A sampled
  meter (`--retention RATE`; 0.05 by default under `wrap`, 0 under a bare `proxy`)
  scores retention **in-process** and persists **counts only** — three integers per
  dimension, with deliberately no field that could carry content. There is no
  plaintext session recorder to secure, which is the whole point. Bounded to 64 KiB
  per request, fail-open, and unlike `--shadow` it costs **no extra tokens** because
  there is no second upstream call.
- **Reversible HTML extraction — the web-fetch blind spot.** Building the retention
  harness exposed a capability gap it could then measure: distil compressed **0.0%** of
  an HTML tool result. The cause was structural, not a missing heuristic — minified
  markup arrives as one enormous line, so Tier-1's line-folding had nothing to fold and
  the JSON/record folds do not recognise markup. Any agent with a fetch or browser tool
  was paying full price for `<script>`, `<style>`, nav and footer chrome.

  `distil/compress/htmlx.py` extracts the content with stdlib `html.parser` and keeps
  the exact original behind an expand handle. Measured on real pages:

  | page | before | after | saved | facts lost |
  |---|---|---|---|---|
  | Wikipedia article | 281,093 tok | 14,260 tok | **94.9%** | **0** |
  | Python docs page | 32,322 tok | 4,229 tok | **86.9%** | 0 |

  Deliberately **recall-biased**: only tags that cannot hold article content are
  dropped, plus four unambiguous chrome landmarks (`nav`/`footer`/`aside`/`form`).
  `<header>` is kept because it usually wraps the `<h1>`, and `img` alt text is kept
  because it is often a figure's only description. Unlike a lossy extractor the
  heuristic is **recoverable** — `distil retention` measures 100% true recall with 0
  lost on the pages above, so a bad call costs one `distil_expand`, not the content.
  Skipped in `verbatim` mode (no expand tool to recover with) and on recency-exempt
  blocks (the agent's latest output stays byte-exact).

  Documented tradeoff: on raw HTML most probe-able "artifacts" are `href` URLs, which
  extraction drops as navigation. They stay recoverable, but visible recall for that
  dimension is low by design — now a measured number instead of an unknown.
### Fixed after cross-audit

All three findings from the PR's independent audit reproduced, so all three are fixed:

- **Unclosed chrome tags swallowed the article.** Chrome skipping keyed off a matching
  close tag and real HTML often never sends one, so content *before* an unclosed
  `<aside>` was emitted while the article after it was dropped. An `<article>`/`<main>`
  landmark now ends any active skip, plus a skipped-data budget as a backstop for
  `<div>`-built chrome. `script`/`style` stay exempt — their payload is legitimately huge.
- **Synchronous disk I/O in the request path.** `RestoreStore.expand()` falls back to a
  disk read for unknown handles, and the meter called it for every `handle=` match — so
  tool output merely *containing* that string could turn one sampled request into a
  series of file reads. Matched handles are now intersected with the store's in-memory
  set.
- **Short answers false-matched in recovered bytes.** `"12"` matched inside
  `file-12.csv`. Recoverability now requires token boundaries, and values under four
  characters require whitespace boundaries.

That last fix then failed the corpus gate, which is how it proved its worth: it exposed
that `_NUMERIC_RE` had been extracting values ending **mid-token** — `invoices=88` clipped
from `invoices=88ms`, junk like `T09:14` from inside a timestamp. Extraction now takes the
whole token including its unit. The probe set is smaller and cleaner (417 facts, not 503),
which moves the corpus figures to **90.2% visible / 100% true / 0 lost** and the measured
value of reversibility from 12.3% to **9.8%**. The earlier number was inflated by junk
probes; this one is honest.

- [ADR 0005](docs/adr/0005-external-validity-and-fact-level-recall.md) and a new
  §5 in [docs/EVALUATION.md](docs/EVALUATION.md).

### Honesty notes

Three defects found while building this, each of which had been reporting numbers in
distil's favour:

- HotpotQA's comparison questions answer "yes"/"no", which is never a span in the
  passage. Grading them reported the dataset's answer *format* as 10% compression
  loss. Answers are now graded **only when present in the uncompressed context**.
- SQuAD v2's unanswerable questions (55% of the split) were being scored as retained —
  a free win on a majority of cases. They are excluded and counted separately.
- A recall number from compression that barely engaged is arithmetic, not evidence, so
  `--min-recall` now **fails** below 1% savings. This is not hypothetical: distil's
  compressors correctly decline to touch short prose, so `--shape prose` yields 0%
  savings and a vacuous 100% recall. The default `--shape json` reflects how a
  retrieval tool actually returns documents.

Worth knowing: on SQuAD at 84.8% savings, true recall is 100% but **visible recall is
0%** — every gold answer sits behind a handle. Lossless, but one round trip per
answer. The report says so instead of printing only the reassuring number.

## [1.37.0] — you will be asked, once, at the moment it makes sense

**The census undercounted because nobody was asked.** `distil onboard` was the
only place that ever requested consent, so anyone who ran `pipx install
distil-llm` and went straight to `distil wrap` — the natural path, and the one
the README showed most prominently until this week — was never asked and could
never be counted. The live numbers made the shape of it obvious: **2 opted-in
installs against 11,102 monthly downloads and 374 unique cloners.** That is not a
counter that is broken; it is a question that was never put.

**You will now be asked once, at the end of a session that actually saved
something.** That is the only moment the question is fair: you have run the
thing, there is a real number on screen, and "share this?" is concrete rather
than hypothetical. The prompt sits before the census send, so a first yes counts
the session you were looking at rather than one a day later.

Every guard the original prompt established is kept, and one is new. Asked once —
a stored yes or no ends it permanently. `DO_NOT_TRACK` and `DISTIL_NO_TELEMETRY`
win outright. Both streams must be a terminal, so pipes, CI and headless runs
never prompt and therefore never enrol. **Ctrl-C is not an answer** — recording a
decline there would burn the one chance to ask on a keystroke that meant "not
now" — though unanswered prompts stop after three, because a prompt nobody
answers must not become a prompt nobody escapes. The new one: **nothing is asked
if the session saved nothing.** Asking someone who got no value to share their
numbers is a worse question and a worse experience than staying quiet. And none
of it can raise: this runs on the wrap teardown path, where an exception would
change your exit code over telemetry.

**What this does not do is make you identifiable.** `install_id` is a random
UUID, and `distil census off` deletes it — so a machine that toggles consent
mints a new one. There is no way to tell one person's second laptop from two
different people, and there will not be: the fix for a small opted-in set is a
larger opted-in set, not fingerprinting.

**Docs.** A light theme (page chrome only — diagrams and code blocks stay dark,
because SVGs loaded through `<img>` cannot inherit page CSS and half-recolouring
them would look broken rather than deliberate; contrast measured at 7.24:1 body
and 17.5:1 headings). Per-page **Copy as Markdown**, because these docs are read
by agents as often as by people. And eight per-framework pages — Anthropic SDK,
OpenAI SDK, LiteLLM, LangChain, Vercel AI SDK, Agno, Strands, CrewAI — each
carrying the caveat that actually bites, such as CrewAI resolving `planning_llm`
separately and silently bypassing the proxy.

## [1.36.0] — the image transform is certified, and now on by default

1.35.0 shipped vision compression **merged but inert**: the gate demanded a
certificate and none could exist, because the certification path was text-only
end to end. `Block.text` is a `str`, so a trajectory could not hold an image, and
every runner graded a decision from text. This closes that.

**Certified, live.** `Block` now carries optional `media`; the Anthropic runner
renders it as real provider image blocks; `vision` is a registered strategy; and
`corpus/vision-ci-dashboard.json` is a decision-bearing vision trajectory with
real, CRC-valid PNGs whose context accumulates, so byte-identical screenshots
genuinely pile up. Against `claude-opus-4-8`: **100% decision-equivalence, A/A
self-agreement floor 100%, TOST p<0.0001, VERDICT PASS.**

**The first live run failed, and that is the point.** Turn 3 diverged — baseline
`promote_release`, compressed `open_failing_build`. The cause was not the
compression: the runner was hoisting every image to the front of the turn,
severing each screenshot from the caption identifying it. An A/B whose two arms
differ in prompt *shape* measures the shape. Interleaving fixed it. No offline
deterministic oracle could have surfaced that, which is exactly why this domain
is certified live and is deliberately excluded from `distil bench`.

**What changes for you on upgrade.** The maintainer's certificate ships in the
package, so vision de-duplication is **enabled by default**. An agent that sends
repeated identical screenshots — UI automation, dashboard polling — will start
seeing duplicates elided, and its reported savings will rise accordingly. Only
byte-identical payloads are ever elided, the first occurrence is untouched, url
sources are never treated as duplicates, and every elision is recoverable through
`distil_expand`. `DISTIL_VISION=0` disables it outright.

**What you are inheriting, precisely.** The shipped certificate states its own
scope: the model, the corpus, the verdict, and the exact command to reproduce it.
It does **not** certify your traffic. Run `distil certify --strategy vision
--runner anthropic` against your own captured trajectory to certify your
workload — and if that run FAILS, your result wins: a local failing verdict is
never overridden by the shipped pass. The gate reads three sources in order
(`DISTIL_VISION` → local → shipped) and every one of them must parse and carry an
explicit passing verdict. Inheriting a certificate is not skipping the gate.

## [1.35.0] — see what it does before you trust it

Eight merged PRs. The theme is the same one twice: **a number that claims to be
current, and a preview that claims to be complete.** Both were wrong, in public.

**Three adoption numbers were stale, not false.** Reported as "the page shows a
previous version when I have the latest" and it turned out to be three separate
bugs of one shape. `distil wrap` is deliberately long-lived — hot-swap replaces
the proxy *worker* on upgrade so your session never restarts, which means the
wrap parent keeps running the code it started with, for days, and it is the
process that emits the census. Every wrap user's beat reported whatever was
installed when their session began; a machine running 1.34.0 was observed beating
`1.28.0`, two upgrades later. `by_version` also counted every install ever seen
under a chart that says "in the wild", so a machine that pinged once and vanished
sat there forever. And the live hero read "2 machines saving now" directly above
its own sentence "summed from 1 machine that actually reported savings" — the
endpoint's `active` field is a *consent* count, and the page fell back to it.
All three fixed; the version now reads from disk, the histogram is 30d-scoped,
and "saving now" means saving.

**`distil simulate` — a dry run that says what it will NOT touch.** Running the
real pipeline locally with no model in the path was already possible. What was
missing is the question that actually decides adoption: *what would you leave
alone, and why?* distil already computes that — the recency exemption, the
assistant's own words, tool_use arguments, lines the keep policy pins as
decision-bearing — so the protected set is now first-class output, per block,
naming the rule. Two guarantees are reported separately, because conflating them
made the first version lie: `byte-exact` (not touched at all) and `lossless-only`
(bytes may change, meaning preserved, nothing moved behind a handle). Recency is
the second kind, not the first — it exempts a block from the Tier-1 digest, not
from Tier-0.

**Prometheus and OpenTelemetry.** `GET /distil/metrics` serves the standard text
exposition, stdlib-only, behind the same admin gate as `/distil/stats` because the
series are labelled by tenant. OTel counters mirror it, recorded at the same
instrumentation point as the span attributes and *before* the span guard, so they
survive tracing being sampled off. This corrects two published claims: the README
and Deploy & Security both said distil has no metrics endpoint and cited that as a
differentiator. It has one; the gate is the differentiator, and it is tested.

**Images become a certifiable content type (ADR 0003).** The prevailing technique
here downscales, which is lossy by construction — the model sees a different image
and nobody can say whether the answer changed. Instead, the second and later
appearances of a *byte-identical* image become a recoverable reference; the first
is untouched. It ships **disabled**: `vision.enabled()` parses the certificate and
requires an explicit passing verdict, so with none present the adapter is
byte-for-byte what it was before. Reversibility is proven; decision-equivalence is
not yet, and ADR 0004 records exactly what blocks it.

**A savings figure that was wrong in public.** The proxy's estimator never counted
image blocks, so an image was ~0 tokens on the "before" side. Any agent that sends
images had an understated baseline. Now counted at pixel-area cost — which means
**reported savings and census figures will shift for image traffic.** That is a
correction, not an improvement.

**Docs search.** 22 pages with no way to search them. ⌘K, static index, no service
and no dependency. A test fails if the committed index goes stale, because a
generated artifact that is committed rots the first time someone adds a page — and
rots silently, since search simply never returns it.

Also: Agno, Strands, Cursor and CrewAI added to the integration matrix, each with
the caveat that actually bites (Cursor's override covers its agent panel but not
tab-completion; CrewAI resolves `planning_llm` separately, so leaving it unset
bypasses the proxy with no error). ADR 0004 stack-ranks what is genuinely left and
records three gaps the earlier audit had *understated* — we had more than we
thought. And the Gemini cross-audit workflow, dead on every PR, now runs from the
base commit with a fail-closed tool allowlist rather than trusting the PR's tree.

## [1.34.0] — say whose savings those are

The counter fix in 1.33.1 stopped the community total moving backwards. This is
the other half of the same problem: what that total actually *means*.

**Consenting is not contributing.** The adoption page counted every machine that
had turned the census on as a machine that was saving. On the real data that
made "2 machines saving now" out of one machine with 1.6B tokens and one that
had reported `tokens_saved: 0` twice and never run anything. The rollup now
emits `contributing` (installs that have actually reported savings) alongside
`instances`, and the page uses it: the headline reads "from 1 machine", the
install tile reads "1 of 2 have reported savings · 1 opted in but idle", and
below five contributors the hero carries an explicit small-sample notice —
*this is one real ledger, not a community aggregate*. Aggregates written before
the field existed make **no** claim rather than falling back to the consent
count, which is the overstatement this removes.

**Your savings, one number.** `distil dashboard --web` computed savings as
lifetime-raw × *current* calibration factor, the exact method `census.py`
documents as wrong: it restates every token already earned whenever calibration
refines, so the number on screen can drop without a token being un-saved. It
also disagreed with the census on the same machine — 1,522,590,876 against
1,491,058,879. Both now read the count-time accrued total through
`census.accrued_tokens()`, so your dashboard, your census payload, and the
community rollup are the same figure.

**Most users are never asked.** Consent is offered in exactly one place —
inside `distil onboard`, at a TTY. Anyone who went from `uvx`/`pipx` straight to
`distil wrap` was never asked at all, which is most of them. `distil stats` now
mentions the census once, only to someone with real savings to contribute, only
at a terminal, never twice, and silent under `DO_NOT_TRACK`. It does not send
anything and does not grant consent — ignoring it leaves you un-asked.

## [1.33.1] — the zipapp can name itself

`distil.pyz` — a release asset offered as the install path for anyone PyPI is
blocked for — reported its version as `0+source` rather than the version it was
built from. It has done so since at least 1.31.0.

The archive carries no `dist-info`, so `importlib.metadata` misses; and the
pyproject fallback in `distil/__init__.py` calls `read_text()` on a path inside
the zip, which a zipapp cannot open. Both paths failed silently to a literal.
`build_pyz.sh` now stamps `distil/_version.py` from `pyproject.toml` at build
time — `zipimport` can import a module even though it cannot read a file — and
the fallback chain tries the stamp before the unreachable pyproject read.

Nothing functional changed: the archive always executed correctly, it just could
not answer `--version`. Customer-facing regardless, since that string is what a
bug report quotes.

**Why it survived three releases**, which is the more useful finding: no test ever
ran the `.pyz`. And running it naively still would not have caught this — under
any interpreter with `distil-llm` installed, `importlib.metadata` answers
correctly and the zipapp path never executes, so the artifact looks green in
exactly the environment a developer tests it in. `tests/test_packaging_smoke.py`
now builds the archive and runs it in a bare `venv --without-pip`, where the
package is genuinely absent. The test was verified to fail (`zipapp reports
'distil 0+source', pyproject says '1.33.0'`) with the stamping step removed.

Also folds in the in-repo Homebrew formula sync, so the tag and `main` no longer
diverge — `v1.33.0` was tagged one commit before it.

### Census: the community total could run backwards, and Ctrl-C burned the consent ask

`saved` is monotonic per step, but it was advanced from two places (the daily
census and the near-real-time heartbeat) across several concurrent processes —
wrap, proxy worker, gateway, webdash — each doing load → step → write-whole-file
with no lock. The last writer clobbered the rest: a process holding minutes-old
state wrote a smaller `saved` back and rewound `raw_seen` with it, so an
already-banked delta could be counted twice. That is how a total which can only
rise published 1.44B, then 1.33B, then 1.16B, then 1.48B. `_savings_locked()`
now holds an exclusive `flock` across the whole read-modify-write, so every
writer steps from the newest state.

Separately, `except (EOFError, KeyboardInterrupt)` around the consent prompt fell
through to `opt_out()`. A user who pressed Ctrl-C — or whose stdin was a pipe —
was permanently recorded as having declined and was never asked again. No answer
now leaves consent unset, so the question can be asked another time.

## [1.33.0] — the reports are readable now

A WCAG 2.2 pass over the docs site and **every HTML surface distil generates** —
the gateway dashboard, the live web dashboard, the savings ledger, the technique
leaderboard, the benchmark report, and the `dissect` portal. Contributed as eight
focused PRs by [@pjdoland](https://github.com/pjdoland) ([#34](https://github.com/dshakes/distil/pull/34)–[#41](https://github.com/dshakes/distil/pull/41)).

Some of this is plain bug-fixing that happened to surface through an accessibility
lens. Two docs pages wired their mobile navigation button to a function that was
defined nowhere, so the menu threw a `ReferenceError` for everyone; nine code blocks
had a copy button pointing at a missing `copyCode`. (The report that the *working*
copy control captured the word "copy" rather than the snippet did not reproduce
under an intercepted `clipboard.writeText` — the nine dead buttons were real and
exactly counted; that secondary claim was not.) The leaderboard's
"not certified" state was an em-dash at **1.75:1** contrast — a verification marker
you effectively could not read, in a project whose entire pitch is that you should
verify rather than trust. Muted text across the reports sat at 3.28:1; it is now
5.45–6.37:1, comfortably past AA.

The interaction changes are real UX wins, not just conformance. The gateway
dashboard and the sessions portal used `<meta http-equiv="refresh">`, so they
reloaded wholesale every 5 s and 15 s — destroying scroll position, text selection,
and keyboard focus while you were reading them. Both now poll JSON and patch rows
in place, with a visible Pause control (WCAG 2.2.1/2.2.4). Charts in the dissect
report carry accessible names and a `data table` fallback with the same numbers.
Tooltips are focusable, dismissible with Escape, and announced. Tables have real
headers, captions, and scope; the docs sidebar is a labeled `<nav>` with list
semantics and `aria-current`.

No compression behavior changed: `make gate` (corpus non-inferiority + byte
fidelity) and the full suite pass unchanged.

### Provider compaction, experiment 2 — the *default* clearing policy

1.32.0 measured Anthropic context editing at `keep=0` and found 92.5% of agent
decisions changed. The obvious rebuttal is that nobody runs `keep=0`. So this
release runs the shipped default, `keep=3`, on a 7-round corpus whose two
decision-bearing tool results sit at the head and whose four routine rounds
(inventory, shipping scans, comms, promotions) sit at the tail — the ordinary
shape of a long agent trajectory, where the load-bearing facts are gathered
early and then buried.

Retention did not help. Clearing changed **95.0%** and **100%** of decisions
across two independent executions of the pre-registered protocol (both published;
selecting one after the fact is the practice this harness exists to refuse).
OpenAI's summarizing compaction changed **20.0%** on the same corpus — a ~5×
gap, consistent with the ~7× at `keep=0`.

The rate is not the finding. The *failure mode inverted*: at `keep=0`, 37 of 40
flips were tool→text — the agent lost its facts and stopped acting. At the
default `keep=3`, stalls nearly vanish and 29–31 of 40 flips become
tool→**wrong** tool. The three surviving routine results are enough to keep the
agent confidently acting while the records it needed are gone. Keeping the most
*recent* tool uses cannot protect a decision that depends on the most *relevant*
ones; it converts a visible stall into a silent wrong write. Neither feature
certifies decision-safe at α=0.1.

`certify-provider` runs are now **resumable**. A 360-call run takes ~40 minutes
and OpenAI's compaction path returns 500s in bursts; two full runs were lost to
it before each finished case was made durable. Every completed case is appended
to `cases.jsonl` stamped with the protocol hash, so a re-invocation replays what
it already bought and a changed parameter orphans the ledger instead of blending
two experiments. A resume reuses the original `protocol.json`, so
"pre-registered before the first call" stays literally true across the restart,
and `calls_made` reports the whole experiment's cost rather than the last
attempt's.

## [1.32.0] — certify the provider's own context manipulation

`distil certify-provider`: a pre-registered, A/A-controlled, budget-capped live A/B
that measures whether **the provider's own context manipulation** changes an agent's
next decision — Anthropic context editing (`clear_tool_uses`) and OpenAI server-side
compaction (`--provider openai`). Vendors will not publish decision-equivalence for
their own features; a third party on the wire can.

The design is the shadow/certificate machinery pointed at a new A/B: same multi-turn
tool transcript, manipulation ON vs OFF, plus a second baseline arm for the sampling
noise floor. Firing is ground-truthed per request (`applied_edits` / the `compaction`
output item); cases where the manipulation never fired are excluded from the sample.
The protocol (n, α, δ, votes, trigger, model) is written to disk before the first
API call, live calls are hard-capped, and transient upstream failures retry with
bounded backoff instead of burning an unattended run.

First pre-registered certificates (n=40, majority-of-3, worst-case configs on a
synthetic decision-bearing corpus, `benchmarks/results/provider-compaction/`):
Anthropic clearing changed the agent's decision on **92.5%** of fired cases —
37/40 flips were tool→text, the agent stops acting rather than acting differently —
while OpenAI compaction changed **12.5%**. Neither certifies decision-safe at α=0.1.
Honest scope: aggressive triggers on transcripts built so tool results are
decision-bearing; the default-config number on long real traffic is future work.

Also: the `live` extra now includes the OpenAI SDK.

## [1.31.1] — prove the attestation gate

No runtime change. This release exists to run the corrected attestation check in CI,
because a verification step that has never executed is not a verification step.

1.31.0's release job failed on a **false negative**: the gate read
`/pypi/<pkg>/<ver>/json`, which PyPI does not populate with attestation data. Every
release back to 1.19.0 does carry a bundle, on the `/integrity/<pkg>/<ver>/<file>/provenance`
endpoint. The fix landed in `95fd3f5` but not in the `v1.31.0` tag, so re-running that job
replayed the same bug — a rerun could never have gone green.

This tag carries the corrected gate. If the release job passes, the check works against a
live publish; if it fails, the check is still wrong and we find out now rather than on a
release that matters.

## [1.31.0] — evidence that checks itself

Four things that all failed the same way: a claim nothing verified.

### PEP 740 attestations — now enforced by the release job
`README.md` claimed releases carried Sigstore attestations. **They did** — every release back
to 1.19.0 carries an attestation bundle. What was missing was any check, so the claim was
merely *unverified* rather than false.

The release job now queries PyPI for the version it just published and **fails** if no
attestation bundle is present, so the claim cannot drift.

> **Correction.** An earlier draft of this entry — and commit `b3eb4a7` — stated that every
> release published with `attestations=none`. That was wrong. It read `/pypi/<pkg>/<ver>/json`,
> which PyPI does not populate with an attestations field; the data lives on the `/integrity/
> <pkg>/<ver>/<file>/provenance` endpoint. The first run of the new gate failed 1.31.0 for the
> same reason — a false negative in the check itself. Both the gate and the claim are corrected
> here. The irony is not lost: a verification step that reported a false failure is the exact
> defect class this release is about.

### Certificates name the oracle that graded them
A `Certificate` recorded α, δ, n, savings and a guarantee — but not what produced the
losses. A run graded by the synthetic `DECISION:` string-match oracle was byte-identical to
one graded by `claude-opus-4-8`. `Certificate.grader` is stamped from the runner, and the
synthetic oracle can never read as a model: `Graded by: deterministic (synthetic DECISION:
oracle — NOT a model)`.

### Per-request receipts
`distil receipts` — a hash-chained, content-free record of what happened to each request:
counts, mode, handles issued, whether they still resolve, and the certificate that authorised
the mode. Edits, deletions and reorders are all detected. `Receipt.FIELDS` is the exhaustive
persisted set and a test fails if anything outside it reaches disk; no prompt or completion
text is ever written. Emitted for every 2xx the proxy serves — not only inside a `wrap`
session, and not only when a savings ledger happens to be attached.

### Fixed: digest reported `changed=True` on byte-identical output
`tier1.digest()` returned `True` unconditionally. Where every line is must-keep — a test log
whose verdict policy pins each `PASS` line — nothing is dropped, no marker is emitted, and
the output is identical to the input. Callers believed it anyway:
`RestoreStore._record` persisted ~19KB of plaintext per block to `~/.distil/restore` for
content that was never digested and can never need recovery (three such entries on a single
request), and the MCP server handed back a handle for text it had not compressed. `changed`
now means the output actually differs.

Found while chasing a receipt that read `27764->27764, saved=0` beside three issued handles.
The savings header itself was honest — the verdict keep policy was correctly retaining every
line — so **no reported savings number was ever overstated**.

### Also
- Packaging gate extended to Docker (`ENTRYPOINT`/`CMD` resolve to real targets, plus a CI
  job that builds the image and runs it), the Homebrew formula (self-consistent url/version/
  sha comment), and the Claude Code plugin — which caught `plugin.json` sitting at 1.8.6,
  22 releases stale.
- `release.sh` now aborts a tag when `pyproject`, `CITATION.cff`, `plugin.json` and
  `server.json` disagree on version.
- `server.json` description brought under the registry's 100-character cap.

## [1.30.0] — the official MCP registry entry actually launches

distil has been listed in the official MCP registry since 2026-07-17 with a launch spec
that could never have worked:

```
$ uvx distil-llm mcp        # what the registry entry resolves to
An executable named `distil-llm` is not provided by package `distil-llm`.
```

The distribution is `distil-llm` but its console scripts were `distil` and `distil-mcp`,
and `uvx <pkg>` runs the executable *named after the package*. Anyone who discovered
distil through the registry — the highest-traffic MCP discovery surface — and used its
declared spec got a failure.

### `uvx distil-llm` now starts the MCP server
Adds a `distil-llm` console script pointing at `distil.mcp_server:serve`. Because
`serve()` ignores argv, the already-published `uvx distil-llm mcp` form works too — so
this repairs the live registry entry for existing clients without waiting for anything
to be republished. `distil`, `distil-mcp`, and `uvx --from distil-llm distil-mcp` are all
unaffected; every form was verified against a built wheel.

Ships `server.json` at the repo root so the registry entry has a committed manifest to
publish from, instead of existing only as server-side state.

### Fixed
- `scripts/release.sh` rewrote `url`, `sha256` and `version` in the Homebrew formula but
  not the comment naming which tag that sha belonged to — it sat at `v1.11.2` while the
  hash moved 18 releases on. A comment that confidently names the wrong tag is worse than
  no comment: anyone recomputing the hash would have checked it against the v1.11.2
  tarball and concluded the formula was corrupt. It now moves in the same `sed`.

## [1.29.1] — nothing can pin a worker thread forever

Two unbounded waits, at opposite ends of the same pipe. Both let one stuck peer
outlive a SIGTERM'd worker; both are now finite, for the reason the upstream socket
was already finite.

### The drain has a deadline
`server_close()` joins the non-daemon handler threads — that join is *how* in-flight
streams finish draining — but it had no bound, so a single wedged handler pinned the
whole worker. Caught as an intermittent macOS CI hang, and captured in a stack rather
than inferred:

```
handler thread : distil/proxy.py _post_upstream -> socket readinto
main thread    : socketserver.py server_close -> join -> threading join
```

`server_close()` now runs on a helper thread joined with `_DRAIN_BUDGET_S`
(`DISTIL_DRAIN_BUDGET_S`, default 300s — well under the supervisor's 15-minute
SIGKILL cap). Bounding the join alone is not enough: returning from `main()` re-joins
those same non-daemon threads at interpreter shutdown, so past the budget — shadow and
savings already flushed — the worker exits directly. Only the hot-swap worker could
ever hit this; `QuietHTTPServer` inherits `daemon_threads = True`, so the in-thread
proxy's join is a no-op.

### Client sockets have a timeout
`_DistilHandler` inherited `StreamRequestHandler.timeout = None`, so accepted client
sockets had no timeout at all: a peer that connects and goes silent, or stops reading
while the response fills the socket buffer, parked its handler thread for the life of
the process. The upstream socket has carried a finite timeout since it was written,
with a comment saying it is finite precisely so a wedged upstream "can never pin a
worker thread forever" — the client half now gets the same bargain via
`_CLIENT_TIMEOUT` (`DISTIL_CLIENT_TIMEOUT`, default 600s, generous because HTTP/1.1
keep-alive means an idle agent between turns is sitting in exactly that read).
`socket.timeout` joins `handle_error`'s quiet list, since a stalled peer now surfaces
there on write the way a vanished one already did.

Measured, both directions, same scenario — connect, then say nothing:
`timeout=None` still held the connection open past 8s; `timeout=2.0` closed it at 2.0s.

### Fixed
- Test teardown called `upstream.shutdown()` without `server_close()`, which stops the
  accept loop but leaves the listener open — a worker connecting afterwards completed
  its handshake into the backlog and waited out the full 600s upstream timeout for a
  reply nobody would send. "Upstream is gone" was a black hole rather than
  `ECONNREFUSED`. New `_stop_upstream()` helper, applied at all 12 sites.
- CI now gates on `ruff format --check` (whole tree, pinned `ruff@0.15.10`). The
  formatter had silently drifted on 16 files because only `ruff check` was gated; that
  drift is also fixed here, verified AST-identical so nothing changed but layout.

### Gates
1832 tests · ruff · format · mypy · bench · verify · validate — all green, and both
fixes carry a regression test that fails without them
(`test_drain_is_bounded_when_a_handler_cannot_finish`,
`test_stalled_client_cannot_pin_a_handler_thread`).

## [1.29.0] — the MCP server tells agents what it actually does

Distil's MCP server worked but under-described itself: three tools with one-line
descriptions, no annotations, and no install instructions anywhere. An agent had to infer
whether `distil_expand` was safe to retry, and a human had to guess the client config.

### MCP tools are annotated and self-describing
All three tools now carry MCP `annotations` — `title`, `readOnlyHint`, `destructiveHint`,
`idempotentHint`, `openWorldHint` — so a client knows `distil_expand` is a safe, repeatable,
offline read without parsing prose for it. Descriptions state what was previously implicit:
the JSON return shape, the literal error string on an unknown handle, that the store is
local and encrypted, that handles age out after `DISTIL_RESTORE_TTL_DAYS`, and when *not*
to call each tool. `distil_expand` also declares `pattern: ^[0-9a-f]{8}$`, matching the
handle regex the server already enforced.

Glama's tool-definition rubric scored `distil_expand` 3.7/5 ("no annotations provided, so
description bears full burden"); since the server-level score is 60% mean + 40% *minimum*,
the weakest tool set the ceiling.

### Install instructions that fit on one screen
README gains an MCP section and `docs/integrations.html` is rewritten: the one-line
`claude mcp add distil -- distil mcp`, the shared `mcpServers` JSON for Claude Desktop /
Cursor / VS Code, a `uvx` no-install variant, a client-free `tools/list` verify command, and
a per-tool "when your agent reaches for it" table. Both state the thing users get wrong —
this is the **recall** path, not the savings path: `distil wrap` compresses traffic, the MCP
server exists so any agent can expand a handle it finds in context.

### Fixed
- **`CITATION.cff` was stale at 1.27.0 while `pyproject.toml` shipped 1.28.0.** The v1.28.0
  tag was cut without `scripts/release.sh`, whose consistency check (`release.sh:94`) exists
  precisely to catch this. Both are now 1.29.0 — cite the version you actually installed.
- `glama.json` added at the repo root so the registry listing stays claimed if the repo ever
  moves to an organization.

## [1.28.0] — recoverable compression everywhere, and it streams

Closes the "expand-injection gap." Lossy Tier-1 digest could leave stubs the agent
couldn't recover, and the recoverable path lost streaming. Both are fixed: recoverable
digest is now the default on every path **and** keeps time-to-first-token.

### Streaming `distil_expand` interception — recover WITHOUT losing TTFT (new)
Recoverable digest injects a `distil_expand` tool so the model can pull an elided block
back on demand. Handling that used to require **buffering the whole response** (the expand
loop needs the complete turn), turning time-to-first-token into time-to-last-token on any
session that carried a digested stub. `distil/streamexpand.py` now **speculatively
streams**: it relays tokens as they arrive and only intervenes if a `distil_expand` call
actually appears — suppressing that internal call, resolving the handle, re-querying, and
**splicing** the continuation into the same client stream (re-indexed, one coherent
message). Most turns never call expand and stream untouched; the rare expanding turn still
streams its answer. The agent never sees distil's recovery tool.

### Recoverable by default on every metered path
`make_app` now turns the expand loop ON wherever lossy digest will actually run (any
metered/PAYG session that didn't force `--verbatim`). Previously a bare `distil
proxy`/`wrap` without `--expand`, the async proxy, or a direct `make_app` caller could run
Tier-1 digest with **no** recovery tool — leaving irreversibly-lossy stubs, the exact harm
the subscription force-verbatim already prevents, silently un-guarded on PAYG. With
streamexpand there is no TTFT reason to leave that gap open. `--verbatim` still wins;
subscription stays lossless-only (no digest).

### The async proxy stops emitting unrecoverable stubs
`distil.aproxy` injects no expand tool and runs no expand loop, so any Tier-1 stub it
created could never be pulled back. It now folds all lossy digest into verbatim — the async
path stays Tier-0 (reversible lossless transforms like the #24 columnar fold still apply,
so real savings remain) and never leaves an irrecoverable stub. (Recoverable Tier-1 on the
async path awaits a streaming expand loop of its own.)

### Gates
- Full suite green (**1828 tests**; +8 for streamexpand — pass-through, interception+splice,
  block re-indexing, SSE frames split across read boundaries, usage summing, max-iters bound,
  re-query failure — plus an end-to-end streamed intercept through the real proxy). Coverage
  ≥95%; pinned ruff + mypy clean; `distil verify` / `bench` / `validate` unaffected.

## [1.27.1] — the shadow gate actually runs; the compression mode stops flipping

Two bug fixes in the request path's *measurement* and *policy* layers. Neither changes
what the model is sent on a normal request — but both were silently degrading guarantees
distil advertises.

### The decision-equivalence shadow gate was recording almost nothing
- **The bug:** `shadow_counters` showed **295/323 replays failing with HTTP 400**
  (`signature_none_skipped: 295`, `last_fail_reason: "400"`), so the "compression provably
  didn't change the agent's next action" number was computed from ~28 samples, not the
  live stream — the safety net was effectively off.
- **Root cause** (`distil/shadow.py`, `force_deterministic`): the temp-0 replay pinned
  `temperature = 0` unconditionally, but two API constraints reject that on the models
  Claude Code runs — (1) extended **thinking** requires `temperature` unset/1 (400
  otherwise), and Claude Code enables thinking by default; (2) **Opus 4.7+ removed
  `temperature`/`top_p`/`top_k` entirely** (any value 400s), so the client omits it.
  Injecting `temperature: 0` therefore 400'd ~every sampled request.
- **The fix:** only pin an **existing** temperature, and never when thinking is on;
  otherwise replay the request exactly as sent (already API-valid). Greedy determinism is
  kept where the knob still exists; elsewhere the existing A/A baseline (`aa_agreement` /
  `adjusted_rate`) absorbs the residual sampling noise. `force_deterministic` is the *only*
  code in the request path that touches sampling params, and only the shadow worker calls
  it — so this never affected live requests, only the background measurement.

### The compression mode flipped digest↔lossless-only between launches
- **The bug:** the same machine sometimes ran the aggressive **digest** compressor and
  sometimes near-passthrough **lossless-only** — visible as wildly inconsistent per-request
  savings — depending only on whether `ANTHROPIC_API_KEY` happened to be exported when the
  proxy started.
- **Root cause** (`distil/doctor.py`, `subscription_mode`): it classified any environment
  with `ANTHROPIC_API_KEY` set as metered → digest, even for a Claude Pro/Max user whose
  Claude Code traffic authenticates with the **OAuth** token, not the key. A volatile env
  var was the deciding signal.
- **The fix:** the stable OAuth-login signal (`~/.claude.json` has `oauthAccount`) now wins
  — an OAuth login classifies as subscription (lossless-only) even with a key in the env. It
  fails safe: misreading subscription traffic as metered would apply lossy digest to it (the
  exact harm the mode gate prevents), while the reverse only leaves savings on the table. A
  bare key with no OAuth login is still metered; `DISTIL_SUBSCRIPTION=0` forces metered under
  an OAuth login.

### Also shipped (deploy-tier — not in the wheel)
- The community live counter was showing a **1.44B ghost** and a phantom **"874.9M/day"**
  from a single idle machine. Fixed by making every downstream layer faithful transport of
  the client's current emit: the worker heartbeat store is now last-write-wins by ts, the
  rollup community total is Σ latest-per-install (not a peak-banking ratchet), and the
  adoption page drops the max-anchor and the `rate×86400` projection (`packaging/census-worker`,
  `scripts/census_rollup.py`, `docs/adoption.html`).

### Gates
- Full suite green; added tests for both fixes (shadow: thinking replays left valid, no
  temperature injected when absent; subscription: OAuth wins over a stray key, override still
  honored). Pinned ruff + mypy clean; `distil verify` / `bench` / `validate` unaffected.

## [1.27.0] — the coverage gate enforces the floor it advertises

Bug fix + test-debt paydown (issue #32). The `coverage` CI job reported **success while its own log printed** `FAIL … Total coverage: 94.90%`. Root cause: `[tool.coverage.report]` set no `precision`, and coverage.py defaults it to `0` — which rounds the number used for the `--cov-fail-under` decision, not just the printed report. `94.90%` rounded to `95`, so `95 >= 95` passed a floor the suite was actually under. For a project whose whole pitch is *certified gates*, a gate that lies is worse than a red build.

- **`precision = 2`** in `[tool.coverage.report]` (`pyproject.toml`): the fail_under comparison now uses the real two-decimal figure. (Diagnosed in #32 by @dshakes.)
- **Real coverage lifted honestly above the floor**, not by lowering the bar. Meaningful tests for genuinely-untested branches: the phase-2 learned-salience trio `query_flywheel` / `query_train` / `query_assoc` went **80.6% → 100%** (deterministic-sampling boundaries, both certify-gate rejection floors, every fail-open / skip / malformed-row path), and `census.py`'s accrual persistence gained corrupt-channel-repair + write-fail-open tests. Total coverage now **95.48%**, enforced honestly.

### Gates
- Full suite green (1818 tests); coverage ≥95% enforced at 2-decimal precision; ruff + mypy clean; `distil verify` / `bench` / `validate` unaffected and PASS.

## [1.26.0] — the community counter is monotonic; `distil default --always-on`

### The live counter never un-counts again — count-time delta calibration
- **The bug:** the community "tokens saved" odometer visibly **shrank**. Root cause: every census (and every heartbeat) multiplied the entire **lifetime** cumulative by the **current** calibration factor (`round(lifetime × f)` in `build_payload`, `_by_model`, and `_current_saved_tokens`). Calibration is bidirectional — as an install gathered more real `usage.*` samples and the factor drifted *down* toward a better estimate, the reported total dropped. A single active machine recalibrating downward dragged the public counter backward: tokens already saved got un-counted.
- **The fix** (`distil/census.py`): the total is now a **monotonic cumulative built from count-time-calibrated deltas** — each census banks `Δraw × factor-known-now` and **freezes it**; a later factor move never restates a past increment. Monotonic by construction, and still honest: every increment is valued at the best estimate available when it was earned (exactly like invoicing each period as it closes), so the census never reports more than the provider would bill. State persists in `~/.distil/census-savings.json`, is wiped with `~/.distil` and on `census off`, and is advanced **only on a real send** — `distil census show` and every preview leave it untouched. The daily census and the near-real-time heartbeat now read the **same** shared total, so the live number and the board agree and neither can shrink.
- **Aggregator belt-and-suspenders** (`scripts/census_rollup.py`): the community total and sparkline are rebuilt from cumulative **positive** per-install deltas (mirroring the existing `rate_per_sec` `dtok >= 0` primitive), so even the residual pre-fix rows already in `census.jsonl` can't pull the public number down during the upgrade transition. For a fixed (monotonic) client this telescopes to its latest value — nothing is lost.

### `distil default --always-on` — persist the base URL for every Claude Code launch (thanks @tolgatuncoglu!)
- Contributed by **Tolga Tuncoglu** ([#31](https://github.com/dshakes/distil/pull/31)): `distil default --always-on` writes `ANTHROPIC_BASE_URL` into `~/.claude/settings.json` so Claude Code routes through distil on every launch without a wrapper (`distil/setup.py`, `distil/cli.py`).
- **Safety guard** added on merge (`distil/doctor.py`): a stale or dead `ANTHROPIC_BASE_URL` in `settings.json` now **fails loud** in `distil doctor` instead of silently breaking every Claude Code session with a connection refused. `distil default` writes are isolated from the developer's real `~/.claude/settings.json` under test.

### Gates
- Full suite green; `distil bench` / `verify` / `validate` PASS (byte-reversible, decision-equivalent, 60/60 adversarial). Coverage ≥95%, ruff + mypy clean.

## [1.25.1] — live counter: hot-swap sessions now pulse

Bug fix. The near-real-time community counter read **"no machines active"** even while installs were running distil. Root cause: the in-session liveness heartbeat was wired only into the legacy in-thread `serve()` path (`_start_heartbeat_timer`). Production now runs the hot-swap **supervisor + worker** architecture — the worker subprocess never touches census, and the supervisor (the one process that outlives every worker swap) sent no beat. A long-lived `distil wrap` session therefore emitted **zero** heartbeats until exit, so `/v1/live` reported `active: 0` the whole time it ran.

Fix: the supervisor's `_watch()` poll loop (already ticking every 30 s) now pulses `census.maybe_heartbeat()` beside its crash breadcrumb (`distil/hotswap.py`). census self-throttles to ≤1/5 min and only sends when opted-in with saved tokens, so the added call is a cheap no-op most ticks. Liveness-only, honesty preserved: `rate` stays 0 when tokens are flat or refined downward — the odometer never projects phantom growth; only `active` reflects the running install. Fail-open (a census fault never touches the serving path). Verified live: `active` 0→1 through the new path, `rate` held at 0 on a downward calibration refinement.

Note: the supervisor cannot hot-reload itself (that is *why* workers are separate subprocesses), so a session already running when you upgrade picks this up on its **next** `distil wrap`, not mid-session.

## [1.25.0] — the compression frontier: nested JSON, constant-column collapse, multi-language code

Three new reversible, gate-certified techniques that close the compression gap with lossy "smart crushers" — every one keeps distil's per-request decision-equivalence proof and pure-Python install.

### Nested-record JSON fold — the shape agents actually traffic in
- **`fold_records`** (`distil/compress/structured.py`): the strict columnar `fold` only folded flat scalar records; real tool output (API responses, search hits, DB rows) nests dicts/lists and fell through to the generic digest, saving far less. `fold_records` extends the columnar fold to nested records — non-scalar cells render as compact JSON, the header marks which columns are JSON-encoded — for **42% fewer tokens** on nested tool output. Reversible (byte-exact original one `expand(handle)` away), DECISION-marker-safe, reject-if-not-smaller. Wired into the tier1 path and the anthropic lossless path.

### Constant-column collapse — Parquet-style entropy coding
- A nested column repeating one value in every row (`status:"active"`, a region, a flag) is hoisted into a single `«=name<TAB>value` directive and dropped from the body — the same constant-encoding a columnar database applies to a low-entropy column. Takes the nested fold from ~42% to **~62% fewer tokens** on enum-heavy output. The shared value is stated once, verbatim, so the view stays maximally readable; dictionary-indexing *varying* columns is deliberately declined (the integer→value indirection is a decision-equivalence risk a constant hoist doesn't carry).

### Multi-language code compressor — zero native dependency
- **`generic_code_skeleton`** (`distil/skeleton.py`): distil's Python-only `ast` skeleton is joined by a language-agnostic brace-block skeleton for **JS/TS/Go/Rust/Java/C/C++/Swift/Kotlin** — keep every signature and brace, elide the pure-body runs between them, driven by a string- and comment-aware brace-depth scanner (braces inside strings, `//`, `#`, `/* */` don't count; unbalanced/mid-edit source bails intact — save less, never corrupt). **~44% on a TypeScript file** with every signature still visible. A competitor's *CodeCompressor* capability delivered with **zero native deps** — a tree-sitter grammar would be a mandatory native extension; distil stays pure-Python.
- Now wired into the **live** tier1 path (was only in the offline conformal/gate path) and into `smart_digest`, on the **active-recovery path only** — an elided body needs the `distil_expand` handle, so it never runs on the flat-rate lossless path where the folds stay information-complete.

### Live counter — liveness beat so `active` reflects real usage
- **`maybe_heartbeat`** (`distil/census.py`) now beats every interval from any install that has saved tokens — not only when the saved-token total is *climbing*. A downward calibration refinement (the estimate getting more accurate, as happened at 1.24→1.25) was making a live install read as **inactive**; now `active` counts installs that are *running distil*. `rate` still reflects genuine growth (0 when flat or refined down), so the odometer never projects phantom tokens — an idle community shows an exact, unmoving total by design.

### Gates
- `distil bench` / `verify` / `validate` all PASS on every change (byte-reversible, decision-equivalent with recovery, 60/60 adversarial). Coverage ≥95%, ruff + mypy clean.

## [1.24.0] — census schema 4: the trust number, session modes, next-gen live board

### The metric that drives belief — decision-equivalence, as a community number
- **`equivalence` {pct, shadowed}** (`distil/census.py`): the noise-adjusted live decision-equivalence from the shadow ledger — *compression provably didn't change the agent's next action*, distil's core claim. `pct` is `None` until an A/A self-agreement baseline exists (never a verdict on sampling nondeterminism). The rollup aggregates it as a shadowed-count-weighted mean and the adoption page renders it as a glowing trust ring.
- **`modes` {interactive, headless, sdk}**: session kind from the *shape* of the wrapped command — an agent binary (interactive) vs `-p`/`--print` (headless) vs a non-agent argv0 driving the Agent SDK (sdk). Flag presence only; the prompt/args are never read. Answers "Claude Code TUI vs `claude -p` vs Agent SDK" content-free.
- **API keys remain untracked, by design** — `billing` is the only cohort split; keys are never persisted, hashed, or derived from.

### Real derived metrics + next-gen live adoption page
- Rollup now emits **measured** `savings.rate_per_sec` (Δtokens/Δt between consecutive censuses, resets dropped), `as_of_ts`, `total_runs`, `avg_per_run`, and a `history[]` community-total time series — all measured, none estimated.
- `docs/adoption.html` rebuilt: a **live odometer** ticking at the measured savings rate (labeled projection, exact anchor, snaps on each new census), the decision-equivalence trust ring, a savings-rate readout, a tokens sparkline, a session-mode panel, and auto-refresh every 45s. Audit trail intact — every number traces to `census.jsonl`.
- Both validators accept schemas 1–4 (all-or-nothing keys, prior-schema rules still enforced). Live-verified end to end: real payload `equivalence {100%, 582}`, `modes {interactive: 28, sdk: 1}`; the worker rejects out-of-range pct and unknown modes.
- The rollup **carries forward per-install dimensions** so a mixed-version fleet can't blank them — a newer lower-schema ping (a v1.23 client, no `equivalence`) arriving after a v1.24 ping no longer erases that install's trust number.

### Near-real-time community counter — opt-in heartbeat + edge aggregate
- **Heartbeat** (`distil/census.py`): the daily census stays the exact auditable archive; a tiny content-free `{v, id, tokens, rate, ts}` beat — at most every 5 min and only when saved tokens grew (idle machines send nothing) — drives the live counter. Same opt-in + `DO_NOT_TRACK` gates, fail-open, sent from the wrap/proxy exit and a lightweight in-session timer.
- **Worker** (`packaging/census-worker/`): `/v1/beat` validates + upserts latest-per-install into Upstash Redis (no history, no IPs); `/v1/live` sums it on read (exact total, no drift) and reports active installs + their combined rate. Both degrade gracefully without Upstash.
- **Adoption page** projects the odometer forward at the **active-only** rate, bounded — it ticks while the community works and goes **static the instant everyone idles**, never inventing growth; anchors to `max(live, census)` so it never regresses below the archive and falls back to the census exact total when the live store is empty.

### Real-time LOCAL dashboard + npm/JS-TS bridge
- **`distil dashboard --web`** (`distil/webdash.py`): a localhost page fed by your own live ledger — the odometer rolls up the instant a real request books more saved tokens. Content-free, local-only, zero-dep.
- **npm `distil-llm`** (`packaging/npm/`): the JS/TS bridge — `npx distil-llm wrap -- <agent>` resolves a Python runner (uvx/pipx/pip) so JS/TS devs use distil without touching pip; `distilBaseURL()` helpers point any SDK at the proxy. Closes the biggest distribution gap.

## [1.23.0] — census schema 3: integration attribution (SDK / headless surfaces)

### Are SDKs & headless clients actually used? — now answered, content-free
- **Integration-surface + API-shape counters** (`distil/surfaces.py`): the proxy counts each compressible request by the door it came through (`wrap` / `proxy` / `gateway`, from `DISTIL_SURFACE` set by the launching CLI command) and by API wire format (`anthropic` / `openai-chat` / `openai-responses` / `gemini`, from the request path). In-memory, flock-merged snapshot in `~/.distil/surfaces.json`, flushed in `serve()`'s teardown before the census reads it. No key-, token-, or identity-derived data — allowlisted keys only, fail-open by contract.
- **Census schema 3** adds `surfaces` and `shapes` (request-count maps). Both validators (worker JS + CI Python) accept schemas 1–3 with all-or-nothing keys and prior-schema rules still enforced; the rollup publishes `usage.surfaces` / `usage.shapes`; the adoption page gains an **Integration surface · API shape** panel. Deliberately NOT tracked: API keys — never persisted, hashed, or derived from (the `billing` field is the closest cohort split, by design).
- Live-verified end to end: standalone proxy records both shapes, a genuine wrapped `claude -p` records `{wrap:1, anthropic:1}`, the v3 census flows client → worker → CI → metrics branch → rollup → live panel; hostile keys (`surfaces:{botnet:1}`) rejected at the worker.

## [1.22.0] — census schema 2: usage dimensions, honest downloads, live adoption dashboard

### Census (schema 2 — TELEMETRY.md updated in lockstep with the frozen-schema tests)
- **Usage dimensions**: `by_model` (calibrated tokens saved per model id, top 5), `billing` (`subscription`|`metered`), `agents` (allowlist-only — `claude`/`codex`/`gemini`/`aider`, anything else collapses to `"other"` so an exotic argv can never leak). Both validators (worker JS + CI Python) accept schema 1 and 2, all-or-nothing keys, live-verified accept/reject including skew and injection attempts.
- **Honest dollars, nobody excluded**: totals are calibration-corrected (never more than billed), and the rollup buckets dollars by billing — metered = real community $, subscription = notional API-rate value, published separately and labeled. Validated on a real ledger: 1.02B tokens / 5,743 runs rolled up with subscription $ correctly bucketed notional.

### Adoption surfaces
- **Bot-filtered downloads**: no-OS PyPI downloads are scanners/crawlers; the snapshot now records the real-vs-bot split plus per-OS and per-Python breakdowns (real-OS ≈ 1.9k/mo vs ≈ 10k bot traffic at ship time). New `downloads-real` badge replaces the raw count in the README.
- **Live adoption dashboard** (`docs/adoption.html`): stat tiles with count-up, real-vs-bot split band, OS/Python/version/model bars, billing+agent chips, animated pipeline architecture diagram with live install count; linked from every sidebar, the landing strip, and the README badges. Near-real-time: census ingest re-rolls aggregates+badges on every ping; badges re-poll in 5 min.
- **Ops fix**: a Vercel git-integration connected to this repo was clobbering the census worker's production deployment with function-less repo-root builds (the `/v1/ping` 404 outage); a root `vercel.json` disables git deployments — the worker deploys explicitly from `packaging/census-worker/` (which gains a `package.json` the zero-config build needs).

## [1.21.0] — adoption picture: opt-in census, passive registry pipeline, live badges

### Adoption & community savings (ADR 0002, TELEMETRY.md)
- **Opt-in content-free census** (`distil census on|off|status|show`) — the answer to "how many active installs, which versions, how much is the community saving" that keeps "nothing leaves your machine" honest: OFF until explicit consent (`--yes` never consents; onboard asks exactly once), random install id deleted on revoke, `DO_NOT_TRACK`/`DISTIL_NO_TELEMETRY` beat stored consent, ≤1 numbers-only JSON per 24h from the proxy-exit flush, fail-open. Schema frozen by test — widening it must edit `TELEMETRY.md` and the test together.
- **Auditable ingest pipeline, live** — zero-dep worker at `distil-census.vercel.app` (strict validation, 1 KB cap, numeric skew ceilings, stores nothing, no IPs) → `repository_dispatch` → CI re-validates (defense in depth) → appends to the public `metrics` branch → nightly rollup dedupes latest-per-install-id into `aggregates.json` + shields badges. The datastore is a git branch anyone can read.
- **Passive registry snapshot** (`scripts/adoption_snapshot.py`, nightly) — PyPI downloads, GitHub stars/clones/views (the 14-day rolling traffic window becomes history), Docker pulls; per-source degradation; real UA + retry for shared-runner IPs; optional `TRAFFIC_TOKEN` (fine-grained PAT, Administration: read) for the traffic API.
- **Live displays** — README badges (PyPI downloads, community tokens saved, active installs 30d) and a live adoption strip on the docs site, both fed by the metrics branch.

### UX
- **Once-a-day update notice** on `distil wrap`/`distil proxy` (`distil/updatecheck.py`): background PyPI check, one stderr line when behind, `DISTIL_NO_UPDATE_CHECK=1` opts out, disclosed in TELEMETRY.md.

## [1.20.2] — bypass tripwire trusts the traffic marker; headless-agent examples

### Fixed
- **False "no requests flowed through distil" warning on short wrap sessions.** The post-run bypass tripwire checked the savings ledger, which books no rows for a session that saves 0 tokens — so a quick `distil wrap -- claude -p "…"` warned that the agent "may have stopped honoring ANTHROPIC_BASE_URL" even when its traffic demonstrably flowed through the proxy. The tripwire now reads the session traffic marker (written `0` at wrap start, flipped to `1` by the first proxied request — the signal built for bypass detection); it still fires on genuine bypass and stays silent when no marker exists (`distil/cli.py`, tests in `tests/test_wrap_presets.py`). Verified live: headless `claude -p` and the Claude Agent SDK both route through wrap with no false warning, and Claude Code 2.1.215 honors `ANTHROPIC_BASE_URL` on both API-key and OAuth auth (captured `HEAD /` preflight + `POST /v1/messages?beta=true`).

### Docs & examples
- **Headless-agent coverage, all live-verified:** new `examples/python_claude_agent_sdk.py` (Claude Agent SDK → bundled CLI → `ANTHROPIC_BASE_URL`) and `examples/js_anthropic.ts` (Anthropic TypeScript SDK `baseURL`); a "Headless agents (Agent SDK, `claude -p`, CI)" section in `examples/README.md`; headless + TS rows in the README integration table and `docs/integrations.html`; the previously unlisted Gemini example added to the examples table.

## [1.20.1] — proxy worker survives broken client writes

### Fixed
- **`proxy worker died (exit=-13)` on long sessions.** `main()` restores SIGPIPE to `SIG_DFL` (correct for CLI filters piped to `head`), but the proxy worker and the in-thread proxy are long-lived network servers: a write to a client/upstream socket that had hung up killed the whole worker with signal 13 instead of raising a catchable `BrokenPipeError`, dropping in-flight state on each `respawning`. Both serving paths now reinstate the server-safe `SIG_IGN` (`distil/hotswap.py`, `distil/proxy.py`), next to where they already override SIGINT for the same "server, not a filter" reason — a broken client now aborts only that one request. Regression test asserts a worker survives 30 abrupt RST disconnects and still serves (`tests/test_hotswap.py`), verified to fail without the fix.

## [1.20.0] — proof surfaces: live gate, OpenAI/Gemini parity, gateway keys, encrypt-at-rest

### Certification & honesty
- **Nightly live gate** (`.github/workflows/live-cert.yml`) — re-certifies the whole corpus against a real model on a nightly cron: ONE pooled TOST over every turn (`distil certify -t corpus/`), majority-of-3 sampling (`--samples`), and an A/A self-agreement control on by default (`--no-aa-control` opts out) so a turn where the model disagrees with *itself* cannot indict compression — the shadow v2→v3 lesson applied to the certify gate. Budget-capped with a hard `--max-live-calls` ceiling so an unattended run can never spend silently; `--margin 0.10` is the live regression margin, distinct from the offline 0.02 proof margin. Live-validated before shipping (claude-haiku-4-5, 28 pooled turns: mean diff −0.036, p=0.0415, PASS). Per-commit gates remain the synthetic offline oracle — the A/A control is a provable no-op there; both layers are labeled precisely.

### Adapters
- **First-class OpenAI adapter** (`distil/adapters/openai.py`) — Chat Completions and Responses API shapes, recency carve-out, and the same Tier-0/1 machinery as the Anthropic adapter — including expand-tool injection and output shaping for the Responses shape (bounded loop, same PAYG gating as the messages path) and query-aware intent from `input_text` + `function_call` args. Live-unverified against a real OpenAI endpoint (offline shape coverage only).
- **Gemini parity** (`distil/adapters/gemini.py`) — recency carve-out, query-aware intent from `functionCall` args, output shaping via `shape="gemini"`, and expand-tool injection as a `functionDeclarations` entry with a bounded `functionCall`→`functionResponse` loop. `cachedContent` needs no guard: cached turns live server-side and never appear in `contents`. Live-unverified against a real Gemini endpoint (offline shape coverage only).

### UX
- **Per-agent wrap presets** (`distil/onboard.py:AGENT_PRESETS`) — `distil wrap -- claude|codex|gemini|aider` auto-selects the correct env var (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL`) and upstream; cursor-agent omitted (env var undocumented). Explicit `--env-var`/`--upstream` always win. Prints `preset: <label> detected → <VAR>`.
- **Proof Ledger** (`distil/proof_ledger.py`) — end-of-session printout on `distil wrap` exit: calibrated tokens/cost, shadow verdict with honest suppression labels, restorability. Silent on zero requests. Opt-out: `DISTIL_NO_LEDGER=1`.
- **Zero-traffic tripwire** — `distil wrap` warns when a known-agent session ends with 0 proxied requests (upstream env-var contract broken, likely agent update). Tests pinned in `tests/test_upstream_contracts.py`.

### Security
- **Encrypt digest originals at rest** (`distil/atrest.py`) — HMAC-SHA256-CTR + encrypt-then-MAC, `DSTL1` magic header, `restore.key` at `chmod 0600`. Protects against backup/sync leakage and cross-user reads on shared filesystems. Does not protect against same-UID attackers (documented in `THREAT_MODEL.md`). Legacy plaintext files still load transparently. Opt-out: `DISTIL_NO_ENCRYPT_AT_REST=1`.

### Gateway
- **Gateway keys** (`distil/gateway_keys.py`, `distil gateway keys issue|list|revoke`) — issued `dsk-` keys hashed at rest (SHA-256); auth fails closed; upstream never sees the distil key (both header carriers stripped unconditionally). `--require-keys` forces key auth even before any keys are issued. `--tenant-rpm` and `--tenant-daily-tokens` enforce per-tenant rate and quota limits — including with key auth off (credential-derived tenants) and per-key overrides set at issue time; every advertised control was verified enforcing under independent security review, which also judged the at-rest construction sound. Default path unchanged (anonymous hash-based tenant IDs still work without keys).

### Benchmarks
- **OTel session correlation** (`distil/otel.py`) — `distil.session.id` span attribute on every proxied request, enabling per-session trace correlation in any OTel backend.
- **Referee scorecard** (`benchmarks/scorecard.py`) — grades any compressor on distil's five invariants; used by the head-to-head harness and linkable from `docs/benchmarks.html`.

## [1.19.0] — `distil validate`: adversarial real-path gate

### Added
- **`distil validate`** — a validation harness that drives the real compressor against a battery
  of diverse and hostile inputs (huge / unicode / deeply-nested / malformed tool output,
  marker-injection that mimics distil's own `<< handle >>` stubs, secret-looking strings) and
  asserts the load-bearing guarantees on every one: **reversibility** (every handle recovers its
  exact bytes), **reject-if-bigger**, **recency-exact** (latest tool output byte-identical),
  **fail-open** (no input makes the compressor raise), and **content-free** (content reaches only
  the local restore store, never a telemetry file — proven with a unique marker). Runs in CI on
  every push alongside `bench` and `verify`; exits non-zero on any violation.

This is the adversarial layer between "green unit suite" and "GA-ready" — it exists because a
passing suite kept coexisting with real-traffic bugs in code paths the corpus never exercised.

## [1.18.2] — calibration robustness (capture-miss immunity)

### Fixed
- **Calibration is now immune to token-usage capture misses.** On real Claude Code traffic some
  requests logged the new (uncached) input but not the cached prefix, giving `billed ≪ est` and a
  ~0.001 ratio that polluted the store. Now `record()` filters any pair outside the plausible
  band `[0.5, 3.0]`, and `factor()` computes the **median of only in-band ratios** — so it is
  robust to garbage already written by an older or concurrent producer. A public headline can
  never be multiplied by a capture artifact. Measured correction on real traffic is ~1.03–1.05
  (the heuristic is accurate because the large cached system prompt dominates and is well
  estimated) — smaller than the 15–20% feared.

## [1.18.1] — calibration: prompt-cache correctness + sanity bounds

### Fixed
- **Calibration now accounts for prompt caching.** `usage.input_tokens` counts only the *uncached*
  tokens; the cached prefix is billed under `cache_read`/`cache_creation`. Comparing the heuristic's
  full-request estimate to `input_tokens` alone made the factor collapse (~0.001 on real cached
  traffic). `scan_usage` now captures the cache fields and calibration uses the full billed input
  (`input + cache_read + cache_creation`). `dissect.calibration()` had the same bug — also fixed.
- **Sanity-bounded factor.** A correction outside `[0.5, 3.0]` is a data problem, not a tokenizer
  difference — the factor falls back to identity, so bad data can never poison a public headline.

## [1.18.0] — self-calibrating token counts (billing-grade, no network)

The offline heuristic is ~15–20% off the real BPE (40%+ on dense code — measured). But distil is
a proxy: it sees the provider's real `usage.*` on every response. It now **learns the correction
from that pairing** and reports token counts that converge to the real tokenizer — with no
per-string network call. The compression **percentage was always exact** (numerator and
denominator use the same estimator); this fixes the *absolute* counts the leaderboard shows.

### Added
- **Self-calibrating token counts** (`distil/calibration.py`, mechanisms A + C + D). A per-model
  store of `(heuristic estimate, billed)` pairs — **integers only, never text** — recorded by the
  proxy from `usage.input_tokens`. `factor()` returns the correction (aggregate ratio blended with
  the per-request median, robust to outliers) and is **identity (1.0) until 20 observations**, so
  an uncalibrated install reports exactly today's numbers — no regression, no early skew. The
  leaderboard (text + HTML) applies it to the absolute totals and prints "calibrated to your billed
  usage (N requests, ±X%)"; the percentage is unchanged (scale-invariant).
- **`--tokenizer subword`** (mechanism B) — a length-aware offline BPE approximation, still
  zero-dependency: it charges longer identifiers more (as BPE does) and adds a surcharge for
  multi-byte characters. Measured closer to the real count than the flat heuristic (33% vs 41%
  error on a code sample); a better *base* for calibration to correct the rest of.

Validated: 13 unit tests (convergence, identity-until-proven, content-free, per-model + pooled,
CI, bounded reservoir, corrupt-store-safe, subword properties); a live Anthropic `count_tokens`
comparison; mypy clean; full gate + `distil bench` + `distil verify` PASS.

## [1.17.0] — semantic bridge (always-on query↔answer matching, zero-dep)

Phase 2 (1.16.0) pinned semantically-relevant lines only near a lexical hit and only once a
model was trained. This closes that gap: an **always-on, zero-dependency semantic bridge** lets
a query term match an answer term that shares no spelling — with no model and no embeddings.

### Added
- **Semantic bridge** (`compress/lexicon.py`) — four composable, pure-Python mechanisms unioned
  into the query-relevant keep set, all **additive** (they only ever widen keeps, so a wrong
  match wastes a little compression, never drops an answer):
  - ① a suffix-stripping **stemmer** + a curated **technical synonym map** (retry↔attempt,
    limit↔max/cap/threshold, timeout↔deadline/ttl…), with compound-identifier splitting so
    `max_attempts` → {max, attempts} bridges to {limit, retry};
  - ③ **char-trigram** Jaccard for typos / near-morphology (config↔configs);
  - ④ optional **distributional vectors** — pure-Python cosine over a bundled table, *no
    torch/numpy at runtime*; inert (no-op) until a table is provided, so the zero-dependency
    posture is preserved.
- **Flywheel-learned associations** (`query_assoc.py`) — the moat: distil learns *your*
  vocabulary (tenant↔org, max_tries↔retry) from real expands, by expand-conditioned
  co-occurrence. **Content-free** — only **hashed** term pairs are stored, joined to the existing
  content-free expand log. Rebuilt by `distil query-relevance`.

The bridge is on by default in the digest (phase 1 + bridge); the phase-2 learned model adds
proximity on top. Validated: 13 unit tests (each mechanism + additive-safety + content-free);
a `claude -p`-confirmed answer (timeout→`deadline_ms`) recovered by the bridge that lexical
misses; `distil bench` (verdict-retention) + `distil verify` (byte-fidelity) PASS.

## [1.16.0] — query-aware salience, phase 2 (learned semantic relevance)

Phase 1 (1.15.0) pins tool-output lines that **lexically** match the agent's intent — a grep
hit, a config key, a SHA. Phase 2 adds the **semantic** case a fixed lexical rule can't reach:
ask "what's the retry limit?" and the answer line `max_attempts = 5` shares no token with the
query, so phase 1 folds it (recoverable, but a round-trip). Phase 2 pins it inline.

### Added
- **Learned query-relevance scorer** (`distil/compress/query_relevance.py`) — an embedding-free,
  stdlib-only logistic model over query-conditioned features (lexical overlap, selectivity,
  proximity to a lexical hit). It's layered **additively** over phase 1: its kept lines are
  unioned in, so it can only ever *widen* the keep set — reversibility and the decision-
  equivalence certificate are untouched. Gated on promoted weights: with none, behavior is
  **exactly phase 1**, so shipping it is behavior-neutral.
- **Content-free expand flywheel** (`distil/query_flywheel.py`) — the model's training labels
  come from distil's own traffic: which digested block the agent expanded, paired with the query
  live at digest time. Records only **numeric feature vectors + the expand outcome** — never a
  raw prompt, response, tool result, or query term. Off by default; the live proxy enables it
  under `--expand`. Sampled and fail-open.
- **Train + certify** (`distil/query_train.py`, `distil query-relevance`) — trains on the flywheel
  labels (reusing the keep-model's logistic trainer, generalized to any feature width) and
  promotes weights only if held-out recall **beats the phase-1 lexical baseline** at a precision
  floor. Additive-only makes it decision-safe by construction; the gate guards compression waste.

Validated: recall 1.0 vs a 0.0 lexical baseline on the semantic case; a `claude -p`-confirmed
answer line that phase 1 folds is recovered by phase 2; `distil bench` + `distil verify` PASS.

## [1.15.4] — GA hardening: structured-audit fixes

A three-front structured audit (proxy request path, compression correctness, policy/telemetry
privacy). The content-free telemetry guarantee, the opt-in transcript correlation, and the #28
fix were all independently verified clean. Six real findings fixed, each with a proof test; the
async silent-drop fix from the prior batch is included here.

### Fixed
- **[HIGH] Subscription-safe default now applies to direct `distil wrap`/`distil proxy`.** Only
  the managed `distil default` install auto-selected lossless-only before; a bare
  `distil wrap -- claude` on a subscription ran the Tier-1 digest with no expand tool — leaving
  irreversibly-lossy stubs in a subscription session. Now a detected subscription with no explicit
  mode flag defaults to lossless-only. An explicit `--expand`/`--verbatim`/`--lossless-only`
  always wins.
- **[HIGH] Recency exemption was silently bypassed for the standard tool_result *list* shape.**
  The list-content path dropped the `is_recent` flag (the string and OpenAI paths passed it), so
  the agent's most-recent output — which the real Anthropic SDK always sends as a list — got
  folded instead of kept byte-exact. A regression introduced with the 1.15.1 columnar fold.
- **[MED] `--lossless-only --shape-output X` no longer falsely claims to shape.** It printed
  "output shaping: X" at startup while the handler correctly suppressed shaping on lossless-only.
  Now it warns the request is suppressed (sync + async proxies).
- **[MED] `distil proxy --async` no longer silently drops `--expand`/`--session-delta`/`--shadow`**
  (from the prior batch) — it names the ignored flags and points at the standard proxy.
- **[LOW] Disk restore now has the in-memory collision guard** — a 32-bit handle collision across
  sessions can no longer clobber and expand to the wrong bytes.
- **[LOW] `fold` bails when a JSON key contains `,`** (the column-header delimiter) instead of
  emitting a mis-keyed table.

## [1.15.3] — honor explicit --expand on a subscription (#28)

### Fixed
- **`--expand` is no longer silently disabled on a subscription** (#28, thanks @pliablepixels).
  Subscription sessions run lossless-only, which forces verbatim and turns the Tier-1 digest
  off — so `--expand` did nothing there. The verbatim force exists because an *unrecoverable*
  stub is irreversibly lossy; but `--expand` injects `distil_expand`, which makes every stub
  recoverable, so that hazard doesn't apply. An explicit `--expand` now lifts the force even on
  a subscription: maximum recoverable compression, nothing irreversibly lost, with a one-time
  startup notice. **The default is unchanged — no `--expand` still means lossless-only.**
  Genuinely-lossy output shaping stays metered-only (it rewrites the response, which expand
  can't recover).

## [1.15.2] — CI portability + docs

Maintenance release. No shipped-code behaviour change from 1.15.1 — the package is byte-identical
apart from the version; this cuts a clean tag over the test/docs fixes below.

### Fixed
- **CI green on Windows and under load** — two portability bugs in the adopted dissect tests
  (both passed on Linux/macOS): a report file was read with the platform default codec
  (`UnicodeDecodeError` on Windows cp1252 for the `«` fold marker), and the streamed detail
  record was read before its line flushed (`IndexError` under CI timing). Test-only.

### Docs
- The landing page, getting-started FAQ, and techniques page now depict the full 1.15 line —
  content-type keep policy, query-aware salience, columnar fold, and `distil dissect`. The
  "will it save me money?" answer no longer under-sells subscriptions: lossless mode **does**
  cut tokens per turn (headroom + rate-limit room), it just doesn't lower a flat-rate bill.

## [1.15.1] — distil dissect + lossless columnar fold for subscriptions

Driven by the same independent power-user (@pliablepixels, #24 / #26 / PR #27). Both a big
new observability feature and a real subscription-savings gap addressed.

### Added
- **`distil dissect`** (#26, PR #27, thanks @pliablepixels) — a per-session deep-dive report:
  savings by model and by mechanism (digest vs cache-delta), the digest inventory (blocks by
  kind, largest folds, re-fold churn, restore recoverability), billed usage captured from API
  responses with a heuristic-calibration figure, latency by path (the `--expand` buffering tax
  as a measured number), quality loops, and a **"worth your attention" anomaly list that
  auto-detects the #25 signatures** so that class of silent failure can't hide again. Optional,
  strictly opt-in transcript correlation names tools/files/prompts; everything else stays
  content-free. New content-free session logging on the proxy is fail-open and off the request
  path. `distil dissect [session] [--html|--json|--serve]`.
- **Lossless columnar fold on the subscription/lossless path** (#24) — a JSON array of flat
  records (extremely common tool output) now folds to a compact, self-describing table (all
  rows inline, no recovery handle to invite an unavailable `distil_expand`), ~70–79% lossless
  and ToS-safe. Recent tool_results stay byte-exact (never fold) so the agent's latest output
  is unchanged. Inherits fold's decision-equivalence certification.

### Fixed
- **Subscription onboard clarity** (#24) — the note now states plainly that lossless mode *does*
  cut tokens (JSON minify + run collapse + columnar fold), it just doesn't lower a flat-rate bill.
- **Adopted with review fixes over the contributor's branch**: `argv` persisted as `command[:1]`
  only (no credential-in-flag leak to the session manifest), and `fcntl.flock` on the per-request
  JSONL append (concurrent-write safety).

## [1.15.0] — query-aware salience, content-type keep policy, expand reliability

The digest gets genuinely content- and intent-aware, and two reliability bugs on the
subscription path are fixed. Several items began as issues/a PR from an external tester
(@pliablepixels, #22–#25) and are credited on the commits.

### Added
- **Query-aware salience** — distil is a proxy, so at compress time it holds the agent's
  intent (its `tool_use` arguments + latest ask) in the *same request* as the output being
  compressed. Lines that match a *discriminating* intent term are now additively pinned, so
  the one line the agent is looking for survives even in arbitrary output where no fixed rule
  knows which line matters (a grep hit, a config value, a SHA). No post-hoc compressor has
  that query/output pairing. Strictly additive: the keep set only widens, so reversibility is
  untouched and the certificate can only hold or improve. A non-discriminating term (one
  matching most lines) is dropped, preserving compression. `distil/compress/intent.py`;
  spec in `specs/query-aware-salience.md`. Example: a 4,152-token log with the answer buried
  in 600 neutral lines → ~163 tokens with the answer kept.
- **Per-content-type keep policy** (#23, thanks @pliablepixels) — a new
  `distil/compress/keep_policy.py` classifies a block (log / traceback / diff / generic) and
  keeps each kind's load-bearing lines: a log's pass/fail verdict, a traceback's stack frames,
  a diff's hunk headers — on top of the generic error/DECISION net. Supersedes the 1.14.x
  inline verdict rules with a cleaner, extensible module (the "per-content-type codec" the
  tier-1 docstring long promised); the outcome-aware dedup layer rides on top unchanged.
- **Shadow observability** (#25, thanks @pliablepixels) — shadow sampling now keeps
  content-free counters (requests seen / sampled / replays attempted / failed + last reason /
  recorded). `distil shadow-stats` explains a `0 recorded` result ("19 seen, 2 sampled, 2
  replays failed (last: 401)") instead of a silent "no samples yet", so a failing replay path
  (e.g. OAuth rejecting proxy replays) is visible instead of indistinguishable from bad luck.

### Fixed
- **`distil_expand` tool_use escaping on the streamed path** (#25, thanks @pliablepixels) —
  the expand gate keyed on handles created *this* request, but `RestoreStore` persists to disk,
  so a streamed turn that digested nothing new yet referenced an older stub emitted a
  `distil_expand` call with no tool injected and no expand loop — and it escaped to the client
  as "No such tool available". The gate now keys on any recoverable handle in the outgoing
  conversation, injecting the tool and buffering to resolve the call server-side.

### Changed
- **Subscription onboarding clarity** (#24, thanks @pliablepixels) — when a flat-rate
  subscription is detected, `distil onboard` now states plainly that lossless mode trims
  context + latency but does **not** reduce token count or cost. (The suggested lossy default
  for subscriptions was declined: it is not provider-ToS-safe and `--session-delta` needs the
  `distil_expand` tool that lossless mode intentionally withholds.)

## [1.14.1] — outcome-aware routing + verdict gate

### Added
- **Outcome-aware routing (tier-1)** — the first content-type profile. When the
  log's own verdict says GREEN (tests passed / build succeeded, nothing failed),
  ERROR/WARN stdout is by definition noise the SUT logged on purpose: dedup
  tightens to one sample per shape. Red or unknown outcome keeps the cautious 2;
  explicit `max_repeats` still overrides; everything folded stays recoverable.
  Power-user log shape: 19.9k tokens → ~93 with verdict + error signal in front
  of the agent.
- **Verdict-retention self-check in `distil bench`** — the gate digests a
  canonical green and red test log and requires the verdict line to survive both
  (the comparison's own substring check). A digest change that compresses away
  the answer flips CI red instead of shipping silently.

## [1.14.0] — verdict-aware digest: keep the answer, fold the noise

Driven by an independent power-user comparison on a live repo (4 tools, 3 tasks):
distil tied for best on bug-fact retention and was the only tool with *measured*
byte-exact reversibility — but on a 22,971-token passing test log it folded the
one line that mattered (`1955 passed`) into a handle while keeping repeated
ERROR/WARN stdout. Both halves of that inversion are fixed.

### Added
- **Verdict preservation (tier-1)** — `_SUMMARY_RE` in the digest keep-net pins
  command *result* lines verbatim: vitest/jest/pytest/mocha counts, cargo
  `test result:`, go `ok`/`PASS`/`--- FAIL:`, gradle/maven `BUILD SUCCESSFUL|FAILED`,
  and `exit code N`. A green run's verdict (which carries no error keyword) can no
  longer be compressed away.
- **Verdict preservation (salience)** — `SalienceKeepModel` scores the same verdict
  lines at the 1.0 never-drop floor (they previously scored 0.3–0.6 and dropped
  while ERROR noise scored 0.95). Single source of truth: tier-1's `_SUMMARY_RE`.
- **Error-noise dedup (tier-1)** — near-identical error/warn repeats (same line
  *shape* after normalizing digits/hex) keep their first 2 occurrences as signal;
  the rest fold behind the existing handle markers, fully recoverable. `DECISION:`
  and verdict lines are exempt.

### Changed
- On the comparison's log shape (passing suite, looped on-purpose ERROR/WARN
  stdout, verdict near the tail): 19.9k tokens → ~131 tokens **with** the verdict
  and first-occurrence error signal in front of the agent — previously 90%
  reduction but the answer required a second round-trip to recover.

Everything is additive to the keep-rules and reversible; no wire, config, or
API changes.

## [1.13.0] — trustworthy shadow gate: deterministic decision-equivalence + mode visibility

Promoted to GA on live validation: **100% decision-equivalence over 116 sampled
production requests** (0 decision changes), with a temperature-0 A/A self-agreement
baseline of **31/31** confirming the result is compression fidelity, not sampling
noise. (Prerelease train: rc1–rc7.)

### Added

- **Status-line mode chip** — the compression mode is now visible at a glance:
  `⬢ digest` · `◇ lossless` · `▪ verbatim`, rendered right after `distil`. Read from
  the ledger's most recent row via `ledger.latest_mode()`.
- **Docs** — plain-English + technical mode explanations (digest / expand /
  lossless-only / verbatim), the billing→mode auto-default (`distil default`/onboard),
  corrected shadow-gate mechanics (50 A/B + 30 A/A), `--shadow 1.0` high-fidelity
  validation, and a validated decision-equivalence callout, across README and the docs site.

### Fixed

- **Shadow decision-equivalence is now measured deterministically.** The A/A
  self-agreement baseline was reading ~38% — not because compression changed the
  agent's decision, but because the replay ran at the agent's live sampling
  temperature, so the model disagreed with *itself* on identical input. Both the
  served and replay sides of every shadow sample are now re-issued at
  `temperature 0` (`shadow.force_deterministic`), never reusing the live hot
  response. A/A collapses toward ~100%, so the A/B rate becomes a real compression
  signal instead of noise. Signature methodology bumps to **v3**; v2 samples are
  scoped out (never averaged with v3), so the gate restarts on a clean baseline.
  This GA ships with the v3 gate passed (see the validation note above).
- As a side effect, the streaming path no longer buffers the upstream response
  body for shadow (it re-issues its own calls), removing that per-request copy.

## [1.13.0] — 1.13.0rc6 — seamless hot-swap: upgrades apply to live sessions, no restart

### Added

- Savings ledger rows are stamped with the compression `mode`
  (verbatim / lossless-only / digest), so "why was ▼ low on this session?" is
  answerable directly from `savings.jsonl` instead of by inference — a
  lossless-only row saving ~0% is subscription safety working as designed; a
  digest row saving ~0% is genuinely low-redundancy content. Optional; pre-1.13
  rows read as unknown mode.

### Fixed

- `distil stats` / leaderboard decision-equivalence now scopes to the current
  signature version and reports the noise-adjusted rate (like the status line),
  instead of the stale, un-adjusted v1 number.

- Shadow decision-equivalence is trustworthy again (decision-signature **v2**, see
  `docs/adr/0001-shadow-decision-signature-v2.md`). The v1 signature hashed tool
  arguments verbatim (normalizing only Python code), so wording jitter — `ls -la`
  vs `ls  -la`, re-serialized JSON — read as a *changed decision*, inflating the
  measured divergence for both compressed and self-replay traffic (a live ledger
  showed only 72.7% A/A self-agreement). v2 canonicalizes formatting whitespace on
  all arguments without merging genuinely different tokens. The signature algorithm
  is now versioned (`SIG_VERSION`); ledger rows are stamped with `sig`+build and the
  verdict scopes to the current algorithm (`shadow-stats --all` reads every row), so
  old-version rows can no longer drag a live verdict. The status-line verdict now
  requires robust evidence (≥50 A/B, ≥30 A/A samples) before showing ✓/✗, warming as
  `de baseline N/30` otherwise. Alarm thresholds are unchanged — the sample gate, not
  a looser threshold, stops false alarms, so real degradation still trips it.

- Hot-swap supervisor no longer cries wolf when a worker dies during a
  non-atomic reinstall. `pip`/`uv --force-reinstall` deletes the package files
  before rewriting them; a worker spawned in that ~1s window dies importing
  half-gone code, and the supervisor logged a scary `WARNING proxy worker died;
  respawning` and tight-looped. It already self-heals once the install
  completes — now, when `installed_version()` is momentarily unreadable, it logs
  at INFO and waits `_UPGRADE_SETTLE_S` before respawning. (Surfaced by
  reinstalling a shared pipx venv under live `distil wrap` sessions.)

- Shadow health no longer shows a red `✗` degraded verdict before the A/A noise
  baseline exists. `adjusted_rate()` silently falls back to the raw, un-adjusted
  rate when there is no baseline, so the status line was painting sampling
  nondeterminism as compression harm (e.g. a scary `✗de 36.0%` over 25 samples
  with a 3/10 baseline). The verdict glyph now gates on `aa_agreement()` and
  shows a neutral `de baseline N/10` while warming; `shadow-stats --json` nulls
  its `adjusted_*` fields until the baseline lands instead of labelling raw
  values "adjusted". Display-only — never affected routing, compression, or
  savings.

### Added

- **Seamless proxy hot-swap** (POSIX, on by default): `distil wrap` now runs the
  proxy as a supervised subprocess on a wrap-owned listener FD. When
  `pipx upgrade` (or pip) puts a new version on disk, the wrap spawns a fresh
  worker — new code, same socket, same port — health-checks it, then drains the
  old one: in-flight requests (including long LLM streams) finish on the old
  worker while new requests land on the new one. The agent session never
  restarts and its `ANTHROPIC_BASE_URL` never changes.
  - Zero request-path overhead: supervision is out-of-band; the upgrade poll is
    one metadata read every 30 s in a daemon thread.
  - Fail-safe twice over: a worker that doesn't report ready is discarded and
    the old one keeps serving; a supervisor that can't start falls back to the
    historical in-thread proxy. The feature can never cost a session.
  - A worker that *dies* mid-session (crash/OOM) is respawned automatically —
    the same self-heal contract the in-thread accept loop had.
  - Manual trigger: `kill -USR1 <wrap pid>`. Opt out: `DISTIL_HOT_SWAP=0`.
  - Windows keeps the in-thread proxy and the existing skew warning (FD
    inheritance is POSIX-only — same accepted platform split as file locking).
- `distil proxy-worker` (internal) — the supervised worker entry point.
- `distil upgrade` now says which sessions hot-swap on their own instead of
  telling you to restart everything.

## [1.12.0] — soaking as 1.12.0rc4 since 2026-07-06 — statusline honesty round 3: "✓ on" means traffic actually flows

First release through the new rc + soak pipeline (runtime code → rc first).

### Added (post-rc4, headed for rc5)

- **OpenTelemetry GenAI spans** (opt-in): `pip install 'distil-llm[otel]'` emits
  `gen_ai.*` semantic-convention spans per proxied request with
  `distil.tokens.original/compressed`, `distil.compression.ratio`, and
  `distil.shadow.sampled` attributes. Strict no-op without the extra; an OTel
  failure can never break the request path. Core stays zero-dependency.
- **Supply chain**: CycloneDX SBOM attached to GitHub releases; weekly OpenSSF
  Scorecard on `main`; PEP 740 Sigstore attestations confirmed active.
- **docs/EVALUATION.md**: the evaluation methodology — why compression ratio
  without a task-success delta is meaningless, the E7 negative result, the A/A
  nondeterminism baseline, and what the trajectory certificate does/doesn't prove.

### Fixed (post-rc4, headed for rc5)

- **CI red on `main` since 1.11.2 — test-side, not product**: the wrap signal
  tests synchronized on fixed sleeps and lost the race on loaded CI runners
  (SIGTERM landing before the handler installs kills the wrap raw). Children
  now write a readiness marker after arming; tests wait on it.
- **`shadow.jsonl` appends now flocked** like `ledger.json` — shadow is
  on-by-default since rc3 and rc4 rows exceed the size where bare appends are
  atomic; concurrent sessions can no longer tear rows.
- `distil doctor` no longer silently drops a crashed Claude Code check —
  it reports FAIL like every other check.

### Fixed
- **"✓ on" no longer trusts env vars alone — a wrapped agent that bypasses the proxy is
  called out.** Found live: a claude.ai-subscription (OAuth) Claude Code session keeps
  `DISTIL_SESSION` and the loopback `ANTHROPIC_BASE_URL` in its env yet sends model calls
  straight to api.anthropic.com (verified with lsof: direct TLS to the provider, zero
  ledger rows, proxy healthy). The 1.11.1 honesty fix checked routing *setup*; this one
  checks routing *reality*. `distil wrap` now writes a per-session traffic marker
  (`~/.distil/sessions/<sid>`, "0" at start), the proxy's first proxied request flips it
  to "1", and a marker still at "0" after a 3-minute grace shows
  **"⚠ wrapped, agent bypassing proxy"** (minimal mode: "⚠ bypassed") instead of "✓ on".
  Markers are single-writer (no locking), best-effort (never block the wrap), swept
  after 7 days, and standalone `distil proxy` never fabricates one.

### Added
- **A/A noise baseline makes the de number interpretable (rc4).** Soak found raw
  compressed-vs-full agreement at 47% — alarming until you notice the comparator's bar
  is "same tool with identical normalized arguments" under live sampling: the model
  disagrees with *itself* on identical requests (a Bash command worded two ways is a
  "changed decision" with zero compression involvement). A third of shadow samples now
  replay the SAME compressed request twice, measuring self-agreement; the statusline
  de rate is reported relative to that baseline, and `distil shadow-stats` shows the
  full decomposition (raw / self-agreement / adjusted). Shadow rows also carry
  content-free evidence now — request digest + both decision signatures — so any
  divergence is diagnosable instead of a bare `false`.
- **Shadow decision-equivalence sampling is on by default (rc3): 2% of wrap requests.**
  It was opt-in (`--shadow`, default 0) — so nobody ran it, the statusline's `de 1/25`
  counter sat frozen for a week implying live measurement, and the launch gate's
  "✓de ≥ 99% at n ≥ 25" evidence could never accrue. Per the intelligence-is-the-default
  rule the flag is now an opt-out: `--shadow 0` disables, `--shadow 0.1` collects faster.
  Cost is explicit: a sampled request is re-run uncompressed for comparison, so the
  default adds ~2% tokens.
- **Wrap-signal breadcrumbs (rc3).** Tonight's quit produced no `.exit` file — because
  the killer took out the wrap *with* the child (process-group kill; terminal-tab SIGHUP),
  so the child-exit path never ran. The wrap's SIGTERM/SIGHUP handler now appends
  "wrap received SIGNAME" to the `.exit` file before dying — the only trace a group kill
  leaves. Both lines together read as a story: `wrap received SIGTERM; child exit 143`.
- **Child-exit breadcrumb (rc2).** Soak day 1 hit a recurring silent agent quit — no
  crash report, no error in the transcript, no way to tell an OOM abort from a clean
  exit after the fact. The wrap is the only witness, so it now records how the child
  ended (`~/.distil/sessions/<sid>.exit`: "exit code N" / "signal NAME" + timestamp);
  `scripts/soak-report.sh` prints it per session.
- **Terminal private-mode reset on wrap exit (rc2).** A crashed TUI leaves xterm modes
  on that `tcsetattr` can't undo — mouse reporting (the `65;76;9M` junk on click),
  bracketed paste, the alternate screen, a hidden cursor. The wrap's restore now resets
  them explicitly; all idempotent on clean exits.
- Test-env hygiene: `tests/conftest.py` sandboxes `DISTIL_HOME` and strips the inherited
  `DISTIL_SESSION` for every test — dogfooding developers run the suite from wrapped
  terminals, and no test may touch the real `~/.distil`.

## [1.11.4] — 2026-07-05 — Release hardening: chaos suite in CI, rc + soak policy, launch gate

No runtime code changed in this release — it hardens the process that ships the runtime.
The 1.10.0→1.11.3 day (six releases, each fixing the previous, all correct in review and
wrong under real use) showed the release gate was blind to signal/lifecycle chaos and had
no bake time. Both gaps close here. (Per the new soak policy this release is soak-exempt:
tests/CI/docs/tooling only.)

### Added
- **Chaos suite in CI** (`tests/test_chaos.py`): the ad-hoc harnesses used to verify the
  1.11.3 Ctrl+C fix are now permanent, bounded tests that run on every push —
  a ~400-signal sustained SIGINT hammer against `distil wrap` with a live child (pins the
  1.11.3 immune-parent property; the 1.11.2 structure fails this), and a
  crash-the-accept-loop test proving the wrap proxy self-heals and keeps answering on the
  same port (the 1.11.0 self-heal path was previously untested).
- **rc + soak release policy** (RELEASING.md): any release that changes runtime behavior
  ships as `X.Y.ZrcN` first and bakes ≥ 3 days on real traffic before the final. rc tags
  are fully wired: GitHub release marked prerelease, PyPI gets the rc (pip ignores
  prereleases unless `--pre`), Homebrew and the Docker image skip rcs, `release.sh`
  detects rc versions and adjusts its preflight (changelog entry lives under the final).
- **Launch gate** (docs/GA_READINESS.md): a binary, evidence-based checklist separating
  engineering GA from the marketing launch — 14 quiet days at head, external beta, live
  decision-equivalence at n ≥ 25 from multiple users, human fresh-install walkthrough on
  all three OSes, claims re-audit at the launch commit.

### Fixed
- **Statusline `de` honesty (rc3, honesty gap #3).** A sub-25 sample count now shows
  `de n/25` only while the shadow ledger was fed within the last 24h; otherwise
  `de idle` — a frozen counter must not read as live measurement.
- **Signal-handler breadcrumb wrote an empty file (rc3).** `_signal_breadcrumb` used
  `time.strftime` but `proxy.py` only imported `time` inside `wrap_run` — the NameError
  was swallowed by the handler's best-effort except, leaving a created-but-empty `.exit`.
  Caught by the new SIGHUP test; `time` is now a module-level import.
- **`release.yml` would have served an rc to everyone.** A `v*rc*` tag previously bumped
  the Homebrew tap and pushed the Docker image as `latest` — both now final-only, and the
  GitHub release for an rc is marked prerelease.

## [1.11.3] — 2026-07-05 — Ctrl+C fix, take two: the wrap parent is now immune to SIGINT entirely

### Fixed
- **Rapid Ctrl+C could still kill a wrapped session (escape path in the 1.11.2 fix).** 1.11.2 caught the Ctrl+C `KeyboardInterrupt` only while blocked in `proc.wait()`; a press landing while the parent was executing the except clause itself (users mash Ctrl+C — Claude Code literally prompts "press ctrl-c again", and a held key auto-repeats) escaped the loop, tore the proxy down under the live agent, and killed the session on its next API call. Reproduced under a SIGINT hammer: the 1.11.2 structure died after ~1.7k signals with the agent still alive; the new one survived 1.7M. The wrap parent now installs a no-op SIGINT handler for its lifetime — immune to any number and timing of presses. A Python-level handler (unlike `SIG_IGN`) resets to default across `exec`, so the agent still receives its Ctrl+C normally (verified empirically). SIGTERM keeps terminate-child + flush-savings + exit semantics.
- `scripts/release.sh`: dropped the stale `distil/__init__.py` version-literal check — the literal was removed when the version became single-sourced from `pyproject.toml`, so the check could only fail.

## [1.11.2] — 2026-07-05 — Ctrl+C no longer kills wrapped sessions; fresh-install statusline honesty

### Fixed
- **Ctrl+C no longer tears down the proxy under a live agent.** A terminal Ctrl+C is delivered to the whole foreground process group; agents like Claude Code survive the first press (it cancels the turn, not the app), but `distil wrap` treated it as shutdown — exiting and leaving the agent pointed at a dead port, so the session died on its next API call. The wrap now keeps waiting through SIGINT (the child owns that signal); SIGTERM keeps its terminate-child + flush-savings + exit semantics.
- **Statusline honesty for fresh installs.** A routed session (`distil wrap` env present) with an empty ledger was told to run `distil wrap -- <agent>` — the exact state every new user hits first. It now shows "✓ on · no savings yet"; the wrap hint remains only for genuinely unrouted shells.
- **Alias-mode verify hint fixed.** `distil default` told users to check `echo $ANTHROPIC_BASE_URL`, which is empty by design in alias mode (the URL is injected only into the wrapped agent's env). It now says `type <agent>` (should show the distil wrap alias); the env-var check applies to `--always-on` only.

## [1.11.1] — 2026-07-05 — Statusline honesty, pre-1.10 warning in terminal, `distil reset`

### Added
- **`distil reset`** — archives the savings ledger to `savings.jsonl.reset-<utc>` (non-destructive, auditable) and starts fresh on post-1.10 accounting; `--shadow` also resets decision-equivalence stats. For ledgers dominated by pre-1.10 records whose savings may be overstated.

### Fixed
- **Statusline honesty: "✓ on" now means routed.** The idle segment said "✓ on" even in a session whose requests went straight to the provider (no `distil wrap`, no loopback base URL). Unrouted sessions now show "off — session not routed".
- **Pre-1.10 overstatement warning reaches the terminal.** `distil stats` text output now prints the legacy-accounting footnote (was HTML-only, despite the 1.10 changelog claim), with a pointer to `distil reset`.
- Windows: `distil default --undo` test no longer assumes a service manager exists (none is wired on Windows).

## [1.11.0] — 2026-07-05 — Ops-ready: debuggable fail-open, crash-safe accounting, health probes; claims audit

### Added
- **`GET /distil/health`** on all three entry points (sync proxy, async proxy, gateway): unauthenticated liveness probe for load balancers / k8s readiness checks. Answers locally — never touches the billed upstream.
- **Debug escape hatch for fail-open paths.** `DISTIL_DEBUG=1` (or `DISTIL_LOG_LEVEL=<level>`) logs every swallowed compression/learning/shadow exception to stderr with a traceback, via a `distil` logger that never touches the root logger. Silent-by-default is unchanged.
- **Restore-store TTL.** Digest originals in `~/.distil/restore/` now age out after `DISTIL_RESTORE_TTL_DAYS` (default 14, `0` disables), on top of the existing 500-file count cap — plaintext agent content no longer sits on disk indefinitely under low traffic.
- Windows CI job (`windows-latest`, 3.12) — the classifier says OS Independent; now the fcntl/termios/SIGPIPE guards are actually exercised.

### Fixed
- **Gateway accounting is crash-safe.** Per-tenant counters now checkpoint to disk at most every 30 s during traffic (atomic replace), not only in the graceful-shutdown path — a `kill -9`/OOM loses ≤ 30 s of accounting instead of everything since startup.
- **MCP store race.** Two concurrent `distil_compress` tool calls could interleave load/save and silently drop one handle (later `distil_expand` on it failed). The read-modify-write now runs under an advisory lock, same pattern as the savings ledger.
- **`distil wrap` proxy self-heals.** If the embedded proxy's accept loop ever crashes, it logs and restarts instead of leaving the wrapped agent with connection-refused for the rest of the session.

### Changed
- **Claims tightened to what the artifacts back** (audit follow-up): "only Distil *certifies* the reversible tier" (Headroom ships an uncertified retrieve — the old wording overclaimed "offers"); the ~1,000× speed multiple now carries its no-ML-model-vs-transformer-inference framing at first mention; LLMLingua-2's SWE-bench row notes only 16/500 runs completed; the Rust core is labeled build-from-source (published wheels run the pure-Python engine).

## [1.10.1] — 2026-07-05 — Review follow-ups

### Fixed
- **No more 0-savings ledger rows.** A flush window that saved nothing (typical under `--lossless-only`) no longer writes a ledger row — session ledgers stay signal-only. Request accounting elsewhere is unchanged.
- **Dedup markers are expand-recoverable.** The `«repeat of earlier tool output …»` marker now carries the `handle=` form that `distil expand` keys on, resolving to the byte-exact original.

### Changed
- **One statusline label for decision-equivalence.** `de 12/25` while collecting → `✓de 99.5% (n)` at 25+ samples (was `eq` for the rate form).

## [1.10.0] — 2026-07-05 — Production hardening: truly lossless, honest accounting, lifecycle fixes

### Fixed

- **`--lossless-only` is now truly lossless (no Tier-1 stubs).** Previously a Tier-1 reversible digest stub could appear in a lossless-only session — but without an injected expand tool the agent could never recover it, making the stub effectively irreversible. The flag now folds directly into verbatim (Tier-0-only) at all three proxy entry points (`aproxy`, `proxy`, `gateway`). No separate `--verbatim` flag needed.
- **Recoverable digests everywhere.** All four digest forms (Tier-1, columnar, template, skeleton) now emit `handle=` markers backed by the RestoreStore. Originals persist to `~/.distil/restore/` (respecting `DISTIL_HOME`) and survive proxy restarts — `distil expand <handle>` works across sessions.
- **Honest savings accounting (numbers dip — that means they became correct).** Records are booked only after a confirmed 2xx response; failed or retried requests are no longer counted. New ledger rows carry `acct:2`; mixed-era ledgers print a footnote: "(includes N records from pre-1.10 accounting — savings may be overstated)". The cache simulator is now write-once-then-read, eliminating a double-counting path.
- **Terminal corruption fixed.** `distil wrap` now saves and restores the terminal state (`termios`) on exit, so a wrapped agent that dies mid-output no longer leaves the terminal in raw mode.
- **Upgrade version-skew warning.** `distil upgrade` now detects running proxy/wrap/gateway processes and warns to restart them — a live proxy loaded pre-upgrade modules can hit a version-skew crash on lazy import mid-request.
- **Decision-equivalence in statusline.** The `✓/⚠/✗ eq <rate>% (n)` display now appears only at ≥ 25 shadow samples; below that threshold it is suppressed everywhere (statusline, leaderboard, doctor, dashboard) — a rate over a handful of samples is noise, not a guarantee.
- **Ledger resilience.** Corrupt lines are tolerated (skipped with a warning rather than crashing the whole read), cross-process writes use advisory file locking, and a backup is kept on each write cycle.
- **Gateway persistence and tenant cap enforced at state load.** Per-tenant accounting persists across restarts; the tenant cap is checked at state-load time, not only at request time.

### Added

- `tests/test_live_certified_equivalence.py` — pins the live proxy's compression decisions to the certified strategy, making any drift a visible, reviewed change. The one documented intentional delta: a recency carve-out keeps the last few tool-result turns verbatim so the agent always sees its freshest output byte-exact.

## [1.8.1] — 2026-07-04 — Believe-it UX + honest ▼0

### Fixed
- **Statusline session view**: shows THIS session first (`▼75.0K −62% $0.31`),
  lifetime as one `Σ` figure; theme-proof 256-color palette + ✓/⚠/✗ health
  glyphs (basic magenta rendered unreadable on dark themes); a session with
  traffic but nothing trimmed yet reads `watching · N seen`, not `▼0 −0%`.
- **▼0 self-explains**: every compressed response carries `x-distil-mode` +
  `x-distil-compressible-tokens`; `distil doctor` warns when the always-on
  proxy runs in verbatim (which caps savings near zero by design).
- Ledger records carry a session id; proxy no longer writes zero-baseline records.

### Added
- Landing hero: two-door router + a real terminal proof card; benchmark chart
  in the hero; site-wide editorial layering (both audiences, less prose).
- `distil stats --badge` (shareable measured-savings badge); LAUNCH.md.

### Docs / process
- E14 propagated to ALL paper artifacts (main.pdf, NeurIPS variant, PAPER.md);
  paper-build now rebuilds + commits PDFs on push to main so they can't drift.

## [1.8.2] — 2026-07-04 — GA polish: no papercuts

### Fixed
- **No raw tracebacks on bad input.** A missing/malformed input file across
  8 commands dumped a Python stack trace; one guard at the dispatch point now
  prints a clean `distil <cmd>: <error>` and exits 2.
- **`--help` no longer lists commands that don't exist** (expand/sweep/gate/
  corpus/adaptive were phantom); a regression test fails if any return.
- **One installer-detection source of truth** (`onboard.install_method`):
  `upgrade`, `offboard`, and `doctor` all use it, so upgrade/uninstall hints
  are always the runnable command (brew/pipx/uv/pip-with-venv-caveat) — no
  more bare `pip` that PEP 668 blocks.
- **`distil doctor` detects shadowed installs** (two `distil` on PATH) and
  **verbatim mode** — the two traps that made "▼0" or "upgrade didn't take".

### Added
- `distil version` (the word people type) and `distil upgrade` (installer-aware).
- World-class README hero (runnable terminal proof block); figures in PAPER.md;
  Homebrew tap auto-bumps on release; GHCR image + PDFs auto-rebuilt on push.

## [1.8.3] — 2026-07-04 — Latest & greatest: statusline, plain-English docs, self-service

### Added
- **Redesigned status line** — rich by default (`distil · session ▼7.8K · 4% smaller · $0.31 · total ▼27.0M · ✓eq 99%`), the session number pops in bold green; `DISTIL_STATUSLINE=minimal` for crowded composite lines. Clear session/total labels, `N% smaller` (no misleading `−`), cohesive teal/green palette.
- **`distil version`** and **`distil upgrade`** (auto-detects brew/pipx/uv/pip).
- Landing page: a plain-English "How it works" section for non-technical readers.

### Fixed
- `distil doctor` flags shadowed installs (two `distil` on PATH) and verbatim mode.
- `distil offboard` prints the uninstall command that actually works per installer (no bare `pip` that PEP 668 blocks).
- No raw tracebacks on bad input; `--help` no longer lists commands that don't exist.
- One installer-detection source of truth (`onboard.install_method`).

### Docs
- Lean README (~40% less prose) + live/clickable badges; 18-page site polish; every link verified; PAPER.md figures; honest banner.

## [1.8.4] — 2026-07-04 — Statusline polish + landing/docs GA audit

### Changed
- **Status line** fully colored (cohesive teal→green, no gray): session number pops bold green, trim rate mid-teal, total muted teal. `N% smaller` (not a misleading `−N%`).
- **Version single-sourced** — `__init__` reads pyproject instead of a hardcoded literal (no drift, no merge-back conflicts).

### Fixed (docs, proactive audit)
- Landing page: `Python 3.11+`→`3.9+` (factual); heading hierarchy; two "How it works"→ one is "Under the hood"; proof section now cites E14 (42.0% vs 39.2%); plain-English section linked from nav + hero; smart quotes.
- benchmark.html cites E14; getting-started smart quotes + stale version example.

## [1.8.5] — 2026-07-04 — Statusline clarity + self-diagnosing doctor

### Fixed
- **Statusline no longer flickers across terminals.** `distil default` spawns a proxy+session per terminal; the live ▼ now aggregates a 15-minute activity WINDOW across ALL sessions instead of one flickering "latest session".
- **Zero-savings state is unmistakable:** `✓ on · waiting for a large read` (bright green, clearly active) instead of a dim, easily-misread "watching".
- **`distil doctor` self-diagnoses the two traps:** `live routing` warns when a wrap/proxy is running but no traffic is recorded (agent bypassing distil); `this session` explains the watching state.

## [1.8.6] — 2026-07-04 — GA presentation + full-surface audit

Rendered every user-facing surface and fixed everything found — the engine was
already proven solid (an evidence-based runtime audit came back clean).

### Fixed — presentation & consistency
- **Status line**: ONE pattern in every state (`distil · <live> · total ▼<lifetime>`);
  live = 15-min window across ALL terminals (no session flicker); zero-savings
  reads `✓ on · waiting for a large read` (never a broken-looking `▼0 −0%`);
  all-teal palette, no gray.
- **No tracebacks on bad input** anywhere — added `NotADirectoryError` to the
  dispatch guard (a `--corpus` pointing at a file leaked a traceback on 6
  commands); `perf --iterations 0` and `holdout --control-fraction` out of range
  now give clean errors; `ingest` no longer silently 'succeeds' on garbage.
- **decision-equivalence suppressed below 25 samples EVERYWHERE** (status line,
  leaderboard, doctor, dashboard, shadow-stats) — no 100% guarantee off n=1.
- Dollars 2dp (or notional on a subscription); correct singular/plural
  (`1 request`/`1 sample`/`1 matched trajectory`); `online` shows `87.3%` not
  16 digits; certify `p=<0.0001` not `p=0`.
- **`distil default` now says: RESTART your agent** — the #1 onboarding trap
  (an agent started before the alias bypasses distil → savings stay at zero).
  `distil doctor` also flags this (`live routing`) and explains the `watching`
  state (`this session`).

### Docs
- Statusline state table (saving / watching / idle) in README + Integrations;
  proof-first hero everywhere (dropped the unmeasured "in half"); technique
  numbering aligned CLI↔site.

## [1.9.1] — 2026-07-04 — Quiet client disconnects

### Fixed
- **No more traceback spam on client disconnects**: agents (Claude Code
  especially) reset/abandon connections constantly — cancelled streams,
  retries, statusline polls — and every one dumped a full
  `ConnectionResetError: [Errno 54]` stack trace into the terminal running
  `distil wrap`/`proxy`/`gateway`. All three servers now run on a
  `QuietHTTPServer` that silently drops `ConnectionResetError` /
  `BrokenPipeError` / `ConnectionAbortedError`; real errors still print.

## [1.9.0] — 2026-07-04 — Per-session savings + hardened CI

### Added
- **True per-session status line** (the headline UX): each terminal now shows
  ITS OWN session's savings (`distil · ▼30.0K · 60% smaller`), while `total`
  stays lifetime across all sessions. `distil wrap` stamps a `DISTIL_SESSION`
  id inherited by both the proxy (which tags every ledger record) and the agent
  → the status line it spawns — so attribution is exact, with no cross-terminal
  bleed. A fresh terminal reads `✓ on` until it compresses something. The
  `distil dashboard` mirrors the same per-session view.

### Changed
- **Sharper positioning everywhere** (README, docs site, social image): dropped
  the "statistical fidelity certificate" jargon → *"Every other compressor asks
  you to trust it won't break your agent. Distil is the only one that proves it
  won't."* The E14 result reframed as a win — *"compressed context didn't just
  match the full context — it beat it: 42.0% vs 39.2%."*

### Quality
- **95% test coverage** (was a 78% floor), 1140+ tests: the CLI, status line,
  doctor, ledger, the network layer (proxy / gateway / streamrelay / async
  proxy), and the statistical-certificate paths are all exercised. Genuinely
  external code (the torch training loop, live-model proof-harness runners) is
  documented-and-omitted, not hidden.
- Fixed a Python-3.9-only flaky proxy-timeout test and a `$HOME`-dependent test
  that failed in a clean CI environment; the coverage floor now ratchets at 95%.

## [1.8.0] — 2026-07-04 — GA: compression that beats full context, certified

### Headline result (E14, SWE-bench Verified n=500, official harness)
- The v1.7 **surprise-preserving digest resolves 42.0% of tasks vs full
  context's 39.2%** (+2.8pp, paired CI [−0.6, +6.2]pp — statistically
  non-inferior with the point estimate above full) and +5.2pp over the E8
  head-digest gate. The shipped trajectory certificate certifies it
  (α=0.10, observed degradation 6.2%). Mechanism confirmed end-to-end:
  keeping a traceback's tail preserves the anomaly the next action needs.
  Paper §E14; `docs/compare.html`.

### Added
- **GA container image**: `ghcr.io/dshakes/distil` (amd64+arm64), published
  on release tags. Multi-stage, non-root, gate-verified.
- **Session-first statusline**: this session leads (`▼75.0K −62% $0.31`),
  lifetime collapses to `Σ27.0M`; compact composite-friendly grammar;
  theme-proof 256-color palette with ✓/⚠/✗ health glyphs; equivalence shown
  only at 25+ shadow samples (a rate over a handful of samples is noise).
- **`distil stats --badge`** — shareable shields.io badge of your measured
  savings; ledger records carry a session id (`ledger.summary(session=)`).
- **Decision-equivalence + session cards** on the HTML savings page and a
  session row in the TUI dashboard.
- **Claude Code plugin 1.8**: `/distil-certify` and `/distil-badge` commands;
  full command table on the Integrations page.
- **E14 benchmark condition** (`distil_gated_surprise`) + committed results,
  paper section, and macro generator.
- `docs/compare.html` (honest head-to-head), LiteLLM Proxy recipe,
  compliance-teams section, THREAT_MODEL.md, LAUNCH.md.

### Fixed
- Homebrew tap served 0.24.0 (pre-GA) — bumped to current and verified.
- `ledger.default_path()` honors `DISTIL_HOME`; forward path never follows
  redirects; identity encoding on compressible requests; 411 on chunked
  bodies; gateway stops echoing the anon tenant hash; mypy-clean package
  with typecheck + coverage floor in CI.



## [1.7.0] — 2026-07-03 — The trajectory-level certificate, true streaming, trust-critical savings fixes

### Added
- **Trajectory-level risk certificate** (`distil certify-trajectories`,
  `distil.certify.trajectory_risk`): certify the invariant that actually
  transfers to task success — a distribution-free Conformal Risk Control /
  Learn-Then-Test bound on **end-to-end task degradation** over matched
  full-context/compressed runs, with stated exchangeability assumptions,
  small-sample refusal, and an anytime-valid drift monitor that flags when the
  certificate needs recalibration. This is the corrected certificate target:
  per-step next-action equivalence provably overpredicts multi-step success
  (our E7 experiment; arXiv 2412.17483).
- **Outcome-guided compression policy** (`distil.compress.guideline`):
  ACON-style learning from trajectory outcomes — content classes whose
  digestion co-occurs with end-to-end regressions get protected byte-exact.
  Never-regressing by construction (only makes compression more conservative);
  content-free signatures only; always on in the proxy.
- **Surprise-preserving retention**: a fourth salience signal — error lines,
  failures, anomalies, and unified-diff changes are over-retained (the "lost
  if surprise" failure mode of lossy compressors), plus file-path protection.
- **True streaming pass-through** in all three servers (proxy, async proxy,
  gateway): SSE responses relay chunk-by-chunk, preserving time-to-first-token
  (previously every response was buffered start-to-finish). Shadow-mode
  decision-equivalence accounting tees off the streamed bytes.
- **`--json` output** on `doctor`, `leaderboard`/`stats`, and `shadow-stats`;
  a `stats` alias for `leaderboard`; grouped `distil --help`.
- **Doctor checks** for pricing-catalog drift (unpriced models in the ledger)
  and tokenizer grade (heuristic vs billing-grade counts).

### Fixed
- **Savings were priced at one fixed model.** The proxy now accounts each
  request under the model it names (mixed Opus/Haiku sessions are no longer
  all priced at the Opus rate), the pricing catalog covers current model ids
  (dated/Bedrock/Vertex shapes resolve too), and unknown upstreams (e.g.
  Gemini) record token savings with dollars=0 rather than being silently
  billed at Claude rates. The async proxy now records savings at all.
- **One-liner `def f(): pass` functions vanished from code skeletons**,
  leaving orphaned `...` where the signature should be.
- **`--shape-output` broke against the Anthropic API** (injected
  `role:"system"` into `messages`, which `/v1/messages` rejects); the
  directive now goes into the top-level `system` field on Anthropic bodies.
- **Upstream calls had no timeout** — a wedged upstream pinned a worker
  thread forever; now a finite (env-tunable) timeout maps to a 504.
- **Savings flushed only every 50 requests and were dropped on `kill`** —
  now every 10 requests or 30 s, plus a SIGTERM handler that flushes (and
  forwards the signal to the wrapped agent).
- **Gateway tenant identity trusted a client header** — accounting identity
  now derives from the credential hash; `x-distil-tenant` is honored only
  under `--trust-tenant-header`. `/distil/stats` and `/distil/dashboard`
  require `--admin-token` (Bearer) and are refused on non-loopback binds
  without one. The MCP handle store is bounded and chmod 0600.
- Concurrency race in expand-mode learning stats (intermittent 500s), sparse
  record arrays no longer fold ambiguously, delta replay order is a declared
  field with a loud error on mismatched turns, salience re-injection keeps
  indentation, `online` warns when reporting train-set metrics.

### Changed
- **Status line is now glanceable**: shows the percent trimmed next to the token
  figure (`1.2M→0.5M tok −58%`), a single `$X.XX saved` delta instead of two
  dollar figures, and colors decision-equivalence by health (green ≥99%,
  yellow ≥95%, red below) so a fidelity regression is visible at a glance.
- **`distil stats`** now prints the orig→compressed token totals with the percent
  trimmed and the live decision-equivalence (with shadow sample count) alongside
  the dollar totals.

## [1.6.2] — 2026-06-30 — Consistent version reporting

### Fixed
- **`distil --version` (and `distil doctor`) reported `1.6.0` on the 1.6.1 release.**
  The version lived in two places and only `pyproject` was bumped, so the published
  wheel's `__version__` lagged. Now single-sourced: `distil.__version__` reads the
  installed distribution metadata (`importlib.metadata`), so the CLI can never drift
  from the published package again. 1.6.2 carries the same Python 3.9+ fix as 1.6.1.

## [1.6.1] — 2026-06-30 — Installable on Python 3.9+ (fixes "from versions: none")

### Fixed
- **`pipx install distil-llm` / `pip install distil-llm` failed with `Could not find
  a version that satisfies the requirement distil-llm (from versions: none)` on stock
  macOS.** Root cause: `requires-python` was `>=3.11`, but macOS ships Python 3.9 as
  the system `python3`, so pip filtered out every release and reported that misleading
  message. The package is stdlib-only and uses `from __future__ import annotations`,
  so it already imports and passes the `distil bench` gate on 3.9/3.10 — the floor was
  simply set too high.
- **Lowered the floor to Python 3.9** (`requires-python = ">=3.9"`, classifiers added)
  and aligned `distil doctor`'s version check. CI now runs the full suite + gate on
  3.9–3.13 so the support claim stays true. Docs/troubleshooting updated.
  > Reaches users once **1.6.1 is published to PyPI** (the live 1.6.0 still pins
  > `>=3.11`). Publish by pushing a `v1.6.1` tag.

## [1.6.0] — 2026-06-30 — Onboard ensures everything

### Added
- **`distil onboard` now ensures you have everything — including a permanent
  install.** When run ephemerally (e.g. `uvx --from distil-llm distil onboard`),
  distil isn't on PATH, so onboard detects that and **offers to install distil
  permanently first** (pipx/uv/brew, per your machine) before wiring the status
  line and routing your agent. Makes `uvx --from distil-llm distil onboard` a true
  one-command setup. Intelligent by default — no flag to opt in.

## [1.5.0] — 2026-06-30 — Clean teardown

### Added
- **`distil offboard` — remove distil's footprint, the inverse of `onboard`.**
  Undoes the shell default (alias/env block), stops + removes the always-on proxy
  service, and unwires the status line from Claude Code settings — asking before
  each (non-interactive without `--yes` removes nothing). Your savings ledger is
  **kept** unless you pass `--purge`. It can't uninstall the running package
  itself, so it prints the exact uninstall command for how distil was installed
  (pipx/uv/pip). `distil default --undo` now also **stops** a running proxy service
  (launchctl/systemctl), not just deletes its definition file.

## [1.4.0] — 2026-06-30 — Make distil the default

### Added
- **`distil default` — make distil the default for your agent, no per-session
  `distil wrap`.** Writes a single managed (marked, backed-up, idempotent) block
  to the shell rc that distil actually detects for *this* machine — zsh (`.zshrc`),
  bash (`.bashrc`/`.bash_profile`), fish (`config.fish`), or PowerShell (`$PROFILE`)
  — using the right syntax for each (alias / function / `export` / `set -gx`). An
  explicit `$SHELL` wins over file-existence guesses, and the command **reports
  what it detected** rather than acting blind. `--always-on` installs a persistent
  proxy service (launchd / systemd) + `ANTHROPIC_BASE_URL` so *every* SDK routes
  through distil (with an honest single-point-of-failure caveat); `--undo` removes
  whichever is installed. `distil onboard` now offers it interactively.
- **`distil onboard` is now upgrade-aware and agent-ready.** It checks PyPI
  (offline-safe) and, if a newer release exists, shows the exact upgrade command
  for your install method (pipx/uv/pip) — `--upgrade` runs it. New
  **`distil onboard --json`** emits the full environment + version status +
  recommendations as structured data so an agent can reason over it.
- **Intelligent `/distil-onboard` skill** — rather than a static installer, the
  Claude Code command now senses via `--json`, assesses *your* situation
  (upgrade, which agent, billing reality, gaps), and guides you through setup +
  validation conversationally, asking and adapting rather than ticking boxes.

## [1.3.0] — 2026-06-30 — One-command onboarding

### Added
- **`distil onboard`** — one command that detects your environment (OS, package
  managers, agent CLIs, install method, the `anthropic` extra, Claude Code +
  subscription), wires the savings status line, and prints a **next-steps guide
  tailored to what it found** — how to route the detected agent (subscription-safe
  vs metered), validate outcomes with shadow mode, watch savings, run the gate,
  and re-verify with `distil doctor`. `--dry-run` changes nothing. Cross-platform
  (macOS / Windows).

## [1.2.0] — 2026-06-30 — Setup & diagnostics UX

Friction-killers for getting distil running and trusting it.

### Added
- **`distil doctor`** — one command diagnoses a setup end-to-end: distil/Python
  version, savings ledger (subscription-aware), shadow-validation status, an
  **in-process proxy round-trip self-test** (proves the proxy machinery works with
  no network), the optional `anthropic` extra + API key, and Claude Code
  status-line wiring + subscription detection. Exits non-zero on any failure.
- **`distil setup`** — wire the savings status line into Claude Code's
  `settings.json` in one command: idempotent, never clobbers an existing line
  without `--force` (backs it up), preserves all other settings.
- **Subscription auto-detect** — the status line and dashboard now drop the
  notional dollar figure automatically on a Claude OAuth subscription (no more
  manual `DISTIL_SUBSCRIPTION=1`; the env var still overrides).
- **Status line** shows the shadow **sample count** next to `eq%`
  (`eq 99.5% (1.2k)`) so the confidence is visible.
- **Dashboard** gains a **live recent-decisions strip** under decision-equivalence
  (▰ same next action · ▱ changed), refreshing with the panel.
- Verified **multi-provider shadow** — decision discrimination tested for
  Anthropic / OpenAI / Gemini response shapes.

## [1.1.0] — 2026-06-30 — Hardening + live-validation UX

Post-GA hardening of the 1.0 line, validated end-to-end across every command.
Zero-dependency stdlib core; **665 tests**.

### Fixed
- **Status line `BrokenPipeError`** — on Python 3.13+, when the status-line
  consumer (e.g. Claude Code) read the line and closed the pipe, the interpreter's
  shutdown flush faulted with a traceback. The `statusline` path now flushes under
  guard and exits cleanly; verified 0/40 on the real binary.
- **Shadow-mode dropped samples** — each sampled decision ran in a daemon thread
  that was killed on proxy teardown (quick runs / last turn), so
  `distil wrap --shadow` could show 0 samples despite live traffic. In-flight
  comparison threads are now drained (bounded) on shutdown.
- **Raw tracebacks → actionable messages** — `--tokenizer/--runner anthropic`
  (missing `anthropic` extra or API key) and `distil ingest --input <bad-path>`
  now fail with a clear, single-line message instead of a Python traceback.
- **Claude Code plugin manifest** — `repository` must be a string URL (was an
  object), which blocked installation.

### Added
- **`distil dashboard`** — a live, zero-dependency terminal TUI: alternate-screen
  framed panel with Unicode bars for token-trim and decision-equivalence,
  original → compressed tokens/cost, and per-trajectory bars.
- **`distil wrap --shadow RATE`** — one-command live decision-equivalence: wraps
  the agent, starts the proxy, sets the base URL, and shadow-samples — no second
  terminal, no manual env var.
- **Status line** now shows **original → compressed** tokens and cost, surfaces
  live decision-equivalence (`eq N%`) when shadow has samples, and drops the
  notional dollar figure on flat-rate subscriptions (`DISTIL_SUBSCRIPTION=1`).
- **Plugin commands** — `/distil-stats`, `/distil-shadow`, `/distil-dashboard`
  alongside `/distil`.
- **Docs** — README and the docs site document `--shadow` outcome validation, the
  dashboard, subscription mode, and the one-command shadow flow.

## [1.0.0] — 2026-06-29 — General Availability

**1.0 / GA.** The compression engine, the proxy/SDK integrations, and the
decision-equivalence certificate machinery are production-grade, API-stable, and
covered by **658 tests** with a zero-dependency stdlib core. This release folds in the
cross-model, cost-frontier, and continuous-assurance work that landed after 0.28.0 and
declares a stable public surface.

**What "1.0 / GA" means (and what it doesn't).** It is a commitment to a stable API and
to the contract that protects you — *certify decision-equivalence, or fall back to full
context; never silently lossy*. It is **not** a claim that aggressive compression is safe
on every agent untuned: E7/E11 show the opposite, which is precisely why the operating
point is **auto-calibrated per deployment and fail-safe**. Honest scope, unchanged: the
guarantee is distribution-free and finite-sample, **conditional on exchangeability** with
your calibration distribution. See [`docs/GA_READINESS.md`](docs/GA_READINESS.md) for the
full ledger of what is closed and what remains empirical breadth.

### Added — cross-model generality (E11)
- **Validated across 5 models / 3 vendors.** The long-horizon harness (30-turn ReAct,
  SWE-bench Verified) now reports gpt-4o-mini and gpt-4.1 (OpenAI), Sonnet 4.6 (Anthropic),
  Haiku 4.5 (Anthropic, n=500), and DeepSeek-V3 (n=200). **gate@12 shows no statistically
  significant degradation on any of the five models.** The two well-powered runs (Haiku
  n=500, DeepSeek n=200) confirm non-inferiority; the three n=50 runs are directionally
  consistent with wide CIs (honestly marked as not powered).
- **Corrected finding.** An earlier reading of DeepSeek alone ("aggressiveness must scale
  with model capability") is **refuted** by the wider sweep: harm appears only as the
  product of *realized compression × the agent's reliance on aged-out context* — a
  workload×model interaction, not raw capability. A fixed `gate_recent` cannot predict it,
  which is why you must calibrate on outcomes per deployment.
- **OpenAI 429 handling** — retry on TPM rate-limits with backoff + `Retry-After`.

### Added — auto-calibration, productionized (closes the headline GA risk)
- `distil calibrate` selects the most aggressive working-set size whose task-success loss
  is non-inferior to full context (paired McNemar), and **fails safe to full context** if
  none certifies — the operating-point analogue of the certificate. Reproduces the manual
  E11 choice automatically (selects gate@12, rejects gate@6 on DeepSeek). `distil/calibrate.py`,
  `tests/test_calibrate.py`. The relevance gate is now a shippable library primitive
  (`distil/gate.py`: `working_set_indices`, `gate_fraction`), not benchmark-only.

### Added — cost frontier under the motto (E12)
- **Cache-monotone gate** (`gate.py:monotone_gate`) — deterministic append-only digests so
  the digested prefix is byte-stable and prompt-cache/KV reuse captures it.
- **Graded gate** (`gate.py:graded_gate`) — per-distance compression tiers, certified with
  the tighter empirical-Bernstein (Maurer–Pontil) bound (`conformal.py`).
- **Speculative expansion** (`speculative.py`) and **constrained-bandit operating-point
  search** (`calibrate.py:bandit_select_operating_point`) — fail-safe, shipped + tested.
  All levers cut cost *inside* the certified envelope; they never trade the guarantee for
  dollars.

### Added — continuous assurance under drift (E13)
- **Anytime-valid drift monitor** (`drift.py:DriftMonitor`) — a betting e-process for
  `H0: risk ≤ α` (Waudby-Smith & Ramdas 2023) you may check after *every* turn with
  false-alarm probability ≤ δ regardless of how often you peek (Ville's inequality). Trips
  when live decision-change exceeds the certified budget → recalibrate or fall back.
- **Cross-family grader ensemble** (`ensemble.py:EnsembleGrader`) — conservative "any-change"
  aggregation keeps measured risk an upper bound even if one grader family is unfaithful.
- **Anytime-valid certificate** for graded losses (`conformal.py:betting_upper_bound`).

### Changed
- Package version reconciled to **1.0.0** (`pyproject.toml`, `distil/__init__.py`,
  `CITATION.cff`); PyPI classifier → **Production/Stable**.
- Docs/site test counts corrected to 658; the landing page's E11 narrative updated to the
  corrected (5-model) finding.

## [0.28.0] — 2026-06-26

E10: trajectory-level decision-equivalence certificate — the first distribution-free,
out-of-sample-proven guarantee at the whole-run level for agent context compression.

- **E10 trajectory-level certificate.** Lifts the per-turn E2 certificate to the
  full trajectory (task) level using the same Learn-Then-Test / Hoeffding–Bentkus
  engine (`distil.conformal.certified_risk_bound`), inverted to a (1−δ) upper
  confidence bound on per-trajectory 0/1 loss. Two loss functions on the full
  500-instance SWE-bench Verified set (δ=0.05):
  - **Divergence** (outcome ≠ full context): empirical 14.4%, certified ≤ **18.0%**.
  - **Harm** (full resolved the task, gated did not): empirical 8.4%, certified
    ≤ **11.4%** — about 1 in 9 solvable tasks, certified.
  - Plain-language: "With 95% confidence, the relevance-gated compressor changes
    a run's outcome on ≤18.0% of exchangeable tasks and costs a solvable task on
    ≤11.4%."
- **Out-of-sample proof.** Over 1000 random calibration/test splits, the bound β
  is certified on the calibration half and checked on the disjoint test half.
  Realized coverage: **95.4%** (divergence) and **96.7%** (harm) — both at or
  above the 95% target. The bound holds on held-out data, not merely asserted on
  training data.
- **Honest reporting: ungated reversible tier.** The ungated tier (condition D, E8)
  also certifies: divergence ≤23.2%, out-of-sample coverage 93.9% — marginally
  below the 95% target. Reported without softening.
- **Honest scope.** The guarantee is exchangeability-conditional: valid for traffic
  exchangeable with the calibration distribution (SWE-bench Verified, this agent +
  model). Changing the agent, model, or task distribution requires re-certification.
- **Why it matters.** E2 guaranteed a per-turn proxy. E7/E8 showed that proxy
  doesn't naively transfer to task success under aggressive compression. E9
  quantified the composition gap. E10 closes it: the first trajectory-level,
  distribution-free decision-equivalence certificate for agent context compression
  (to our knowledge).
- **Reproducible.** `benchmarks/trajectory_certificate.py`; numbers trace to
  `docs/paper/results/swe_e2e_longhorizon/trajectory_certificate.json`.
- **Docs updated:** `docs/research.html` (E10 section with results table and OOS
  proof), `docs/index.html` (honest-scope headline line), `docs/concepts.html`
  (certificate callout).

## [0.27.0] — 2026-06-26

Final E8 long-horizon results: 6-condition frontier including Headroom
competitor, skeleton digest, sticky expansion, digest-mode-per-tier ablation,
and the E9 trajectory-composition certificate bound.

- **E8 long-horizon SWE-bench Verified — final 6-condition frontier.** A
  custom multi-turn ReAct coding agent (read / search / edit_file / run_tests,
  up to 30 turns, `claude-haiku-4-5`, temp 0) run end-to-end on the full
  500-instance SWE-bench Verified set, scored by the official `swebench`
  harness (hidden tests, per-instance Docker). Runs average ~27 turns. Six
  conditions, same agent, compressor differs (ordered by pass@1, Wilson 95%
  CI, resolved/500):
  - **A (full context):** 196/500 — 39.2% [35.0, 43.5]
  - **E (distil reversible, relevance-gated):** 184/500 — **36.8%** [32.7, 41.1]
  - **F (Headroom, lossy competitor):** 163/500 — 32.6% [28.6, 36.8]
  - **D (distil reversible + skeleton digest, ungated):** 162/500 — 32.4% [28.4, 36.6]
  - **B (distil `trunc@500`, aggressive lossy):** 28/500 — 5.6% [3.9, 8.0]
  - **C (LLMLingua-2, lossy competitor):** 12/500 — 2.4% [1.4, 4.2]
  - Total API spend across all six conditions: $571.15
- **Key results (paired McNemar, same 500 instances).**
  - Gate (E) vs full context (A): −2.4 pp, 95% CI [−5.7, +0.9], McNemar
    p=0.19. Non-inferior at a 6 pp margin (borderline at strict 5 pp). This
    is a non-inferiority result, not equivalence. The gate is the **only
    condition statistically non-inferior to full context**.
  - Gate (E) vs Headroom (F): +4.2 pp, McNemar p=0.035. Statistically
    significant. Distil is not cheapest — Headroom is cheaper — but beats
    Headroom on task success with significance.
  - Gate (E) vs LLMLingua-2 (C): 174 gate wins vs 2 LLMLingua-2 wins,
    McNemar p<0.001. E and C remove nearly identical context fractions (53%
    vs 52%), isolating *what* is kept as the deciding factor.
  - Lossy truncation (B) vs full: p<0.001.
- **Honest headline.** On the axis that defines the field — certified
  decision-equivalence plus real task success — distil leads. It does **not**
  claim cost-domination. Headroom is cheaper. The claim is: the only certified
  and reversible compressor, with the highest task-success of any compressor
  tested, and the only one statistically non-inferior to full context.
- **New technique: content-aware skeleton digest** (`distil/compress/skeleton.py`).
  For the active-recovery (ungated) tier, large source files are digested to a
  navigable skeleton: every `import`/`class`/`def` signature retained, traceback
  tails kept, bodies elided. Deterministic and stdlib-only (no model, no network
  — auditable and secure). Byte-exact reversible via content handle. Lifted
  ungated pass@1 from 28.8% to 32.4% (condition D).
- **New technique: sticky expansion** (`distil/expand.py`). Once the agent
  recovers a block via `distil_expand`, that block stays full for the rest of
  the session (handles are deterministic). Eliminates re-expansion thrash on
  repeatedly-accessed files. Never-regressing by construction.
- **Honest ablation: digest mode per tier.** Applying the skeleton digest to
  the *relevance-gated* (passive) tier regressed pass@1 from 36.8% to 5.6%,
  matching lossy truncation. A navigable digest makes the agent over-trust the
  summary and stop re-reading. Skeleton digest is correct for the
  active-recovery tier; head-truncation is correct for the passive tier. This
  finding is published as-is.
- **E9 trajectory-composition certificate bound.** The per-turn certificate
  extends to multi-turn trajectories. Across ~27-turn runs, only ~1.8 turns are
  outcome-determining, so the naive composition bound (which becomes vacuous at
  ~27 turns) overstates risk. The formal per-trajectory bound remains an open
  problem; reversibility is the operative safety guarantee for the
  active-recovery tier.
- **Docs updated:** `docs/research.html` (6-condition table, Headroom row,
  skeleton/sticky sections, honest-ablation note, certificate scope), plus
  `docs/index.html`, `docs/concepts.html`, `docs/benchmark.html`,
  `docs/techniques.html` (skeleton digest and sticky expansion sections).
- Numbers trace to
  `docs/paper/results/swe_e2e_longhorizon/swe_bench_verified_longhorizon.json`.

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

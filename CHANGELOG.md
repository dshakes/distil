# Changelog

All notable changes to Distil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

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

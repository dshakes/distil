# Beta program — distil 1.20.0rc1 (CLOSED, kept as a record)

> **This program is over and the commands below are stale.** It ran before GA;
> distil is now generally available and well past 1.20.0. Installing the pinned
> `1.20.0rc1` here would give you a release candidate many versions behind, missing
> every fix since — including the cache-anchoring fix that stops compression from
> rewriting your cached prefix.
>
> **To install distil today:** `pipx install distil-llm && distil onboard`
> (or `uvx --from distil-llm distil onboard`). See
> [getting-started](https://dshakes.github.io/distil/getting-started.html).
>
> This page is retained only as a record of what the beta asked for and found.

**Goal (as written at the time):** ≥10 external users running real work through
distil for ~a week, with decision-equivalence data from at least 3. That evidence
gated the GA launch.

---

## What you get

Distil is a localhost proxy that compresses the context your coding agent
re-sends every turn and proves its decisions don't change. On a metered API key
it saves real dollars — fewer tokens billed per turn. On a flat-rate
subscription it trims context size and latency, not the monthly charge (and it
has no effect on what the subscription costs).

> **Will it save money?** Only on **metered billing** (API key) — fewer tokens,
> fewer dollars. On a flat-rate **subscription** it trims context + latency, not
> the bill. The biggest wins are on long, many-turn sessions; short sessions are
> typically ~7%.

All compression is reversible — originals are kept locally and can be restored.
The proxy is localhost-only; nothing leaves your machine.

---

## Install and route

**Install (rc pin required — pip hides pre-releases by default):**

```bash
pipx install 'distil-llm==1.20.0rc1'
```

Or with `uvx` (no install, just run):

```bash
uvx --from 'distil-llm==1.20.0rc1' distil onboard
```

**Route your agent:**

```bash
distil wrap -- claude       # Claude Code
distil wrap -- codex        # Codex (sets OPENAI_BASE_URL)
distil wrap -- gemini       # Gemini CLI (sets GOOGLE_GEMINI_BASE_URL)
distil wrap -- aider        # aider (sets OPENAI_BASE_URL)
```

Each recognized agent auto-selects the right env var and upstream — no extra
flags. `distil onboard` runs a guided setup that detects your agent and billing
mode and prints the exact command for your situation.

---

## What to watch

```bash
distil leaderboard          # cumulative tokens + $ saved (from the local ledger)
distil dashboard            # live terminal TUI — token-trim and decision-equiv bars
distil shadow-stats         # live decision-equivalence rate on your real traffic
```

At session end, `distil wrap` prints a proof ledger: tokens saved, equivalence
rate, and restorability. No configuration required.

---

## The ask

This is the point of the beta:

**a. Run your normal work through distil for about a week.** Just `distil wrap
-- claude` (or your agent) instead of the bare agent. The status line and session
summary show what it did.

**b. Enable shadow validation on a slice of requests:**

```bash
distil wrap --shadow 0.1 -- claude
```

Shadow mode runs 10% of requests both compressed and uncompressed in the
background, then compares the agent's chosen next action. After ~25+ shadowed
requests:

```bash
distil shadow-stats
```

Share the output of that command. **What it shares: counts and rates only —
never prompt, response, or tool content.** Specifically: total shadowed requests,
decision-change count, raw agreement rate, A/A self-agreement rate (how often the
model agrees with itself on identical requests), and the noise-adjusted
equivalence rate. Decision signatures in the ledger are 12-character hex hashes
of normalized tool inputs (`sha256[:12]`), not the inputs themselves. No request
content is stored or shared.

**c. Report anything odd.** Breakage, weird compression, latency spikes, or
unexpected behavior → [GitHub Issues](https://github.com/dshakes/distil/issues).
Note your agent, OS, and whether you're on a metered API key or subscription.

---

## Safety and trust

- **Localhost-only** — the proxy binds `127.0.0.1` and forwards only to the
  configured upstream. No SSRF.
- **Fail-open** — if compression fails for any reason, the original request is
  forwarded unchanged.
- **Reversible** — `distil verify` checks that every compression is byte-exactly
  reversible on the bundled corpus. `distil validate` runs 12 adversarial inputs
  against 5 invariants (reversibility, reject-if-bigger, recency-exactness,
  fail-open, content-free telemetry).
- **Encrypted at rest** — digest originals in `~/.distil/restore/` are encrypted
  with HMAC-SHA256-CTR.
- **No secret logging** — request bodies and credentials are never logged.
- Threat model, full scope, and out-of-scope assumptions: [`THREAT_MODEL.md`](../THREAT_MODEL.md)

Attestation verify (supply chain):
```bash
uvx pypi-attestations verify pypi \
  --repository https://github.com/dshakes/distil \
  pypi:distil_llm-1.20.0rc1-py3-none-any.whl
```

---

## Known rough edges

- **OpenAI and Gemini adapters** — integrated and pass `distil bench`, but
  live decision-equivalence evidence from external user traffic is not yet at the
  ≥3-user bar. The Claude adapter is the validated path.
- **Windows** — the proxy works, but hot-swap (live upgrade without restarting
  the session) is not available. The wrap keeps the historical in-thread proxy
  and warns on version skew. Upgrades apply on the next session.
- **Subscription dollar figures** — the leaderboard shows dollar savings for
  metered sessions; on flat-rate subscriptions those numbers are notional and
  auto-hidden unless `DISTIL_SUBSCRIPTION=0` is set.
- **Subscription-agent interception is undocumented upstream.** Claude Code
  OAuth traffic has honored `ANTHROPIC_BASE_URL` in testing (6.1M tokens saved
  across two OAuth sessions, 2026-07-06), but Anthropic documents no such
  contract and some endpoints are hardcoded. A Claude Code update could break
  interception for subscription users. The status line shows `⚠ wrapped, agent
  bypassing proxy` if routing stops working.

---

## Feedback

Open a [GitHub Issue](https://github.com/dshakes/distil/issues) for bugs or
unexpected behavior, and share your `distil shadow-stats` output when you have
≥25 samples. That data is what closes the beta gate.

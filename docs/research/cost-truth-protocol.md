# Pre-registered protocol — the cost truth of agent context compressors

*Status: **DRAFT for review**, 2026-09-25. Not yet frozen. No live run has happened and no
result exists; this document contains no outcome numbers. It is frozen by a signed git tag
(`cost-truth-protocol-v1`) after the competitor-review window (§13). ADR 0018.*

Harness: `benchmarks/cost_truth/` (`python -m benchmarks.cost_truth {plan,dry-run,analyze,estimate,power}`).
Tests: `tests/test_cost_truth.py`. Cost estimate: `benchmarks/results/cost_truth/cost_estimate.json`.

## 0. Conflict of interest

This study is designed and run by the authors of **distil**, one of the three tools under
test. We have a financial and reputational interest in distil looking good. Every design
choice below that could tilt the result is either (a) made in the direction that is *worse*
for distil where there is a choice, or (b) handed to a mechanism we do not control. The
mitigations are listed in §13; the reader should treat any distil-favourable result with the
scepticism this paragraph invites, and check it against the raw logs we publish.

## 1. Question

> On a real, cache-inclusive provider bill, does routing a coding agent through a context
> compressor reduce the **dollars spent per task solved**, without reducing the **rate at
> which tasks are solved**?

The question is about money, not tokens. Almost all agent input is billed as prompt-cache
reads at 0.1x (distil's own live data: cache reads were 72% of $ and 97.6% of input tokens,
`docs/research/live-savings-gap.md`). A compressor that removes 50% of *tokens* can move the
bill by far less, by nothing, or upwards — if it busts the cache (distil measured exactly
this on itself: a sliding recency window doubled cost, 2026-08) or makes the agent take more
turns (Quesma, below).

**Hypotheses, per compressor arm A (rtk, headroom, distil) vs control C:**

* H1 (cost): the ratio R_A of $/solved (§8.1) is < 1.
* H2 (quality): the success rate of A is non-inferior to C at margin 5 pp (§8.2).
* A tool is reported as **"saves money"** iff H1 and H2 are both supported. Otherwise the
  report says "not shown" and gives the estimates and intervals; it never says "no effect".

## 2. Prior art and what this study changes

Quesma, *"RTK reports huge token savings, but our cost benchmarks disagree"* (2026-09-11;
1,740 Terminal-Bench 2.1 attempts, Claude Code + Fable 5.0 and OpenCode + DeepSeek V4 Pro,
RTK 0.45.0, 5 attempts per task per arm, ~$1,500). Their findings shape this design:

| Quesma finding | Consequence here |
|---|---|
| `rtk gain` claimed 89% (349M tokens) while billed cost moved −5%/+5% | Each tool's own claim is recorded and set against the billed truth on the *same* runs (§12). |
| `rtk gain` = bytes removed / 4, including a 120M-token credit for a `head -1` of a file the agent never read in full | Claims are compared in their own unit; we never convert a claim to $ and call it a saving. |
| One task (`winning-avg-corewars`) produced almost all of Fable's savings | Primary inference resamples *tasks* (cluster bootstrap); task-weighted mean log-ratio and a leave-one-task-out table are pre-registered secondaries. |
| Cost per solve and per-task averages told different stories | Both are pre-registered, one of them is primary. |
| Extra turns erased per-turn savings; turns ↔ cost | Turns are a secondary outcome; $ already includes them. |
| A tool bug (`rtk find` loop, 339 errors, 9x cost) | Tool bugs count against the tool (intention-to-treat, §10); we do not swap versions mid-study. |
| Removed 4 security tasks after refusals | Exclusions are fixed at freeze from the pilot, never after seeing arm outcomes (§3). |

What Quesma did not do, and we add: a neutral edge meter that bills every arm from the
provider's `usage` object at the same point in the chain (§6); pre-registration with a power
calculation; explicit prompt-cache TTL handling (§5.3); a hard spend cap; a non-inferiority
test on success instead of eyeballing a 1–2 pp gap; and three compressors, not one.

## 3. Task suite: Terminal-Bench 2.1 via Harbor

**Choice:** Terminal-Bench 2.1 (Apache-2.0), run with the Harbor harness (Apache-2.0,
`harbor==0.23.0`), all tasks in the dataset minus exclusions fixed at freeze. Expected ~89
tasks.

| Candidate | Multi-turn tool use | Automatic pass/fail | Headless | Licence | Verdict |
|---|---|---|---|---|---|
| **Terminal-Bench 2.1** | heavy, terminal-first | per-task test script | yes (Harbor + Docker) | Apache-2.0 | **chosen** |
| SWE-bench Verified subset | yes, but Read/Edit-heavy | yes (FAIL_TO_PASS) | yes, heavier infra | MIT (dataset) | replication candidate |
| distil trajectory corpus | recorded, not live | **no** | n/a | ours | rejected |

Reasons:

1. **Automatic, externally-owned pass/fail.** Each task ships its own verifier. Neither we
   nor the competitors wrote it.
2. **Real multi-turn tool use** with Bash output as the dominant tool — the surface RTK
   targets and the one digest/compression tools act on. This is RTK's *best* case, not
   distil's: we deliberately chose the suite most favourable to the competitor that only
   compresses terminal output.
3. **Direct comparability** with Quesma's 1,740-run result on the same suite and version:
   our control arm is a partial replication of theirs.
4. **Success rate is high enough** (Quesma: 84% with Claude Code) that $/solved is well
   defined per task. Terminal-Bench 3.0/4.0 are rejected for the same reason Quesma gives:
   too few passes to price a solve.
5. The distil corpus is rejected on two independent grounds: no pass/fail, and we built it.

SWE-bench Verified is the pre-registered **replication suite** if the primary shows any
effect (a separate protocol amendment, not part of this study's claims).

**Exclusions** (decided once, at freeze, from the pilot's *control* arm only): tasks whose
container fails to build, and tasks where the control model refuses. The list is committed
with the frozen protocol. No task is excluded after the confirmatory run starts.

## 4. Agent, models, arms

* **Agent:** Claude Code `2.1.282`, installed by Harbor's own `claude-code` installer and
  run headless exactly as Harbor's stock agent runs it (`--verbose
  --output-format=stream-json --permission-mode=bypassPermissions --model <m> --print`,
  instruction on stdin), through the custom agent `benchmarks/cost_truth/harbor_agent.py`.
  Auto-update and non-essential traffic disabled; no MCP servers or CLAUDE.md beyond what
  an arm's own documented setup adds (Amendment 1).
* **Models:** `claude-sonnet-5` (primary), `claude-haiku-4-5` (replication). Every request's
  model is priced by what the response says it was (Claude Code's own Haiku side-calls are
  billed too, and counted in $, but not in "turns").
* **Arms** (exact commands: `python -m benchmarks.cost_truth plan`; data: `arms.py`). Each tool
  runs at its **default** configuration for the documented Claude Code integration, pinned:

| Arm | Version | Integration (documented) | Its upstream |
|---|---|---|---|
| control | — | `claude -p` direct | meter |
| rtk | 0.50.0 (musl binary, sha256-pinned) | `rtk init -g --auto-patch` → Claude Code Bash hook | meter (no proxy) |
| headroom | headroom-ai[all] 0.38.0, Python 3.13 (hash-locked, 171 pkgs) | `headroom wrap claude -- …` | meter (`ANTHROPIC_TARGET_API_URL`) |
| distil | distil-llm 1.54.0 **from PyPI**, Python 3.12 (hash-locked) | `distil wrap --upstream … -- claude …` | meter (`--upstream`) |

Distil is tested as a user would install it: the published wheel, never the authors'
working tree. If a newer GA of any tool ships before freeze, the frozen versions are the
latest GA on the freeze date — the same rule for all three.

**Where each piece runs:** every tool inside the task container, installed and launched as
its own docs say — see Amendment 1 (which replaces the host-side shim design of the first
draft).

## 5. Design

### 5.1 Pairing

Unit = **block** = (task, seed). Every block is run once in every arm, same model, same
Claude Code version, same task timeout. "Seed" is a replicate index (Claude Code exposes no
sampling seed); pairing controls the task, not the sampling noise, which is why each task
gets k seeds.

### 5.2 Randomisation

Blocks are shuffled; the arm order within each block is an independent random permutation
(`build_schedule`, seeded; the seed is committed at freeze). The schedule is the dispatch
*priority*; §5.3 can delay a run but never reorders arms within a block by outcome.

### 5.3 Prompt-cache TTL

Anthropic's cache is per organisation and keyed on exact prefix. Two runs of the same task
share the prefix (tools + system + instruction) until the first sampled turn diverges, so a
run started within the TTL of another arm's run of the *same task* inherits a warm cache.

1. **Spacing:** a task is not dispatched until `MIN_GAP_S` = 360 s (5-min TTL + 60 s) after
   its previous run ended (`next_eligible`). If the pilot shows any 1-hour cache writes
   (`cache_creation_1h_input_tokens > 0`), the gap becomes 3,900 s before freeze.
2. **Measurement:** each run records `first_request_cache_read` — cache reads on its first
   main-model request, the only place cross-run inheritance can show.
3. **Sensitivity (pre-registered):** the primary ratio is recomputed with those reads
   repriced as cold writes. A conclusion that flips under this repricing is reported as
   cache-order-sensitive.
4. The shared tools/system prefix is warm across *different* tasks for every arm, as it is
   for real users; arms that rewrite tools/system warm their own prefix the same way.
   Randomised order makes any residual symmetric in expectation.

### 5.4 Isolation

Each run gets a fresh directory with fresh `HOME`, `XDG_*`, `CLAUDE_CONFIG_DIR`,
`DISTIL_HOME`, `TMPDIR`; the env is built from an allowlist (`PATH`, locale, `TERM`, `TZ`),
so no base URL, key or tool config leaks from the operator shell (`isolated_env`, which also
refuses a non-empty run dir). The API key is injected explicitly. Each run gets its own meter
and (for proxy arms) its own proxy process on a fresh port; both are killed and the run dir
deleted after the claim is read. Harbor gives each trial a fresh container.

### 5.5 Concurrency

Up to 8 runs in flight, subject to §5.3. Concurrency is the same for all arms because it is
applied to the interleaved schedule, not per arm.

## 6. Metering — one meter for every arm

### 6.1 The neutral edge meter

`meter.UsageMeter` is a reverse proxy that is the **last hop before the provider** in every
arm (`claude → [arm proxy] → meter → api.anthropic.com`). It forwards bytes unchanged
(streamed, chunk by chunk), and records per request only: model, HTTP status, the integer
`usage` fields (`input_tokens`, `output_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens`, and the 1h split), byte counts, timings. Bodies are parsed in
memory and never written (tested: no prompt text, no key, no response text reaches the log).

Cost = list price from `distil.pricing` (cache read 0.1x, 5-min write 1.25x) plus 1-hour
writes at 2.0x, which `distil.pricing` does not model; the meter reads the split from
`usage.cache_creation`. Unpriced models are refused (HTTP 400) rather than billed at $0.
Upstream errors bill $0 (the provider does not bill them) and are recorded.

Why not the tools' own numbers, or Claude Code's transcript? Tools' numbers are the claims
under test. Claude Code's transcript usage is what the arm's proxy *returned*, which a proxy
could in principle rewrite; the edge meter sees the provider's own response. Claude Code's
transcript usage is kept as a **cross-check**: a run whose transcript total differs from the
meter by > 1% is flagged and listed (not excluded).

### 6.2 Hard spend cap

Same discipline as `distil/mcpproxy/bench.py`: before forwarding, the meter reserves the
request's worst case (every request byte as a 1h cache write — a token covers ≥ 1 byte — plus
`max_tokens` of output); after the response, it settles to the billed cost. A request whose
worst case does not fit is refused with HTTP 402 and never reaches the provider, so realised
spend cannot exceed the cap. One cap per phase (§9), shared across all concurrent runs. The
first refusal stops dispatch; the in-flight run is `budget_stop` (§10).

### 6.3 Preflight (live only; $ < 1)

For each arm, at the start of every phase: the **canary chain proof** of Amendment 1 §D —
one tiny request launched by the arm's own integration must reach the meter, and for proxy
arms must have crossed the tool to get there; RTK must also show its hook and rewriter. Any
failure aborts the phase before a task runs. Each run then records the tool's own claim
(`claim.json`); a run whose meter saw no billed traffic is `arm_crash` (§10). The live
runner refuses to start while any arm spec in `arms.py` is unverified. `preflight.json` is
published.

## 7. Sample size and power

The analysis unit is the task (clustered bootstrap), so n is the number of tasks, fixed by
the suite (~89); the free parameter is k seeds per task.

**Cost.** Normal approximation on the per-task paired difference of mean log cost over k
seeds: Var = 2σ_w²/k + σ_b², where σ_w is the SD of log cost across attempts of one task in
one arm and σ_b is task-by-arm heterogeneity. Two-sided α = 0.05/3 (Bonferroni, three
comparisons), power 0.8 (`analysis.mde_ratio`, `n_tasks_for_mde`; `python -m
benchmarks.cost_truth power`):

| σ_w | σ_b | detectable $/solved ratio at 89 tasks × 5 seeds | tasks needed for a 10% cut |
|---|---|---|---|
| 0.4 | 0.1 | 0.911 | 70 |
| 0.4 | 0.2 | 0.895 | 99 |
| 0.6 | 0.1 | 0.874 | 146 |
| 0.6 | 0.2 | 0.863 | 174 |
| 0.8 | 0.2 | 0.830 | 280 |

**Success.** With 445 paired attempts, discordance 0.10–0.15 and design effect 1.0–1.5, the
smallest NI margin provable at power 0.8 is **4.5–6.7 pp**. The 2 pp `CERT_MARGIN` from
`distil.conformal` would need ~2,300 paired attempts per comparison (≈ 5x this study). The
pre-registered co-primary margin is therefore `BUDGET_DELTA` = **5 pp** (also from
`distil.conformal`); the 2 pp test is reported as a secondary and is expected to be
inconclusive. This is a deliberate, stated deviation from "NI at CERT_MARGIN" in the brief.

**Decision.** Primary: Sonnet 5, **89 tasks × 5 seeds × 4 arms = 1,780 runs** (Quesma used
5). Replication: Haiku 4.5, 89 × 3 × 4 = 1,068 runs. Honest expectation: the study can
detect a ~13% $/solved cut if σ_w ≈ 0.6; it cannot distinguish a 5% effect from zero. The
report will say that in the first paragraph.

**Pilot** (10 tasks × 2 seeds × 4 arms, Sonnet 5, 80 runs): estimates σ_w, σ_b, discordance,
the per-attempt token profile (§9) and the 1h-cache question (§5.3). Pilot data are never
used in the confirmatory analysis. If the pilot's σ_w implies an MDE worse than 0.85 at
k = 5, k is raised to at most 8 **if the budget allows**, recorded in a dated amendment
before the first confirmatory run. The pilot does not look at arm differences.

## 8. Analysis (one look, at the end)

Only complete blocks enter (every arm has a counted status, §10). α = 0.05 family-wise,
0.05/3 per comparison. Bootstrap B = 10,000, resampling tasks with replacement and carrying
every seed and arm of a task together; seed committed at freeze. Code: `analysis.analyze`.

### 8.1 Primary — $ per solved task

R_A = (Σ$ over all of A's attempts / #A solved) / (Σ$ over all of C's attempts / #C solved).
All spend counts, including failed and crashed attempts (a failure is money spent for no
solve). Report the point estimate and the two-sided 98.33% percentile CI. **Cheaper** iff
the upper bound < 1; **dearer** iff the lower bound > 1; else inconclusive. A bootstrap
replicate with zero solves in A is R = ∞ (so an arm that is cheap because it never solves is
dearer, not cheaper — tested).

### 8.2 Co-primary — task success

D_A = success rate A − C over paired attempts. Non-inferior iff the one-sided (α/3) lower
bootstrap bound > −5 pp.

### 8.3 Secondaries (reported, never used to rescue a primary)

* TOST equivalence at ±5 pp; attempt-level McNemar NI at 2 pp (`distil.certify.stats`).
* Task-weighted mean of log(mean cost A / mean cost C) with CI (Quesma's second measure).
* Leave-one-task-out R_A range (the "one task did it" check).
* Warm-start-repriced R_A (§5.3).
* Tokens by class (uncached, cache read, cache write 5m/1h, output), per attempt.
* Turns (main-model requests), wall time, crash rate, timeouts.
* Total $ (not per solve) ratio.
* The claims table (§12).
* Replication model: the same table; a primary conclusion is reported as "replicated" only if
  the Haiku point estimate is on the same side of 1.

## 9. Cost estimate and budget

`benchmarks/results/cost_truth/cost_estimate.json` (`python -m benchmarks.cost_truth
estimate`). Assumed attempt profile (an **assumption**, replaced by pilot medians before
freeze): 30 turns, 22k-token shared prefix, 1.8k new tokens and 450 output tokens per turn →
1.443M cache-read, 76k cache-write, 13.5k output tokens. That is ~$0.92 per Sonnet 5 attempt,
~1.8x Quesma's observed per-attempt spend scaled to Sonnet 5 prices, so it errs high.

| Phase | Model | Runs | $ with cache | $ if nothing cached | Hard cap (×1.25) |
|---|---|---|---|---|---|
| pilot | sonnet-5 | 80 | 73.65 | 380.78 | **92.07** |
| primary | sonnet-5 | 1,780 | 1,638.79 | 8,472.39 | **2,048.49** |
| replication | haiku-4-5 | 1,068 | 327.76 | 1,694.48 | **409.70** |
| **total** | | 2,928 | | | **2,550.26** |

Per arm: $409.70 (primary) and $81.94 (replication) with cache. The "no cache" column shows
why token counts mislead: the same tokens cost 5.2x more if none are cache reads. An arm that
busts the cache can therefore cost up to ~2x the control; that would exhaust the cap, which
stops the study (reported as truncated) rather than extending the budget.

## 10. Outcome of a run — what counts as what

| Status | Definition | Treatment |
|---|---|---|
| solved | Harbor verifier passes | counted |
| failed | verifier fails, or the agent exits without solving | counted, $ counted |
| timeout | the task's own Harbor timeout | counted as failed, $ counted |
| arm_crash | the arm's tool errors, its proxy dies or returns its own 5xx, its hook loops, or the meter saw **no** billed traffic (broken chain) | **counted as failed against that arm**, $ counted (intention-to-treat) |
| infra_error | container build fails, Harbor fails before the agent starts, or the provider is unavailable for the whole run (meter shows only upstream 5xx/529) | block excluded from **all** arms; re-run once at the end of the schedule; if it fails again, excluded and listed |
| budget_stop | the meter refused a request under the cap | block excluded; study truncated |

Distinguishing arm_crash from infra_error uses the meter log and the arm's own process exit
status only, never the task outcome.

## 11. Stopping rules

* **No efficacy looks.** No interim analysis of arm differences. The analysis script runs
  once, on the complete data, after the last run.
* **Budget:** the hard cap (§6.2). Truncation is reported with the number of complete blocks
  and the realised MDE.
* **Operational (arm-level) stop:** if an arm crashes in 20 consecutive runs, dispatch of
  that arm pauses and its maintainer is contacted. The arm resumes unchanged (same version)
  within 7 days or is reported as "could not be run" with the logs. No version swap.
* **Provider incident:** a provider-side outage pauses the whole schedule; affected blocks
  are infra_error.

## 12. Each tool's own claim vs the billed truth

After each run the harness executes the arm's documented savings report against that run's
isolated state (`rtk gain`; headroom's savings report; `distil stats --json`) and stores the
parsed numbers content-free with the run. The report puts, per arm and over the same paired
blocks: the tool's claimed savings (in its own unit — e.g. RTK's bytes/4 "tokens"), the
billed input-token difference vs control, and the billed $ difference vs control; and the
ratio claim ÷ billed tokens. Distil's claim is treated exactly like the others; a distil
overclaim is called out by name.

## 13. Conflict-of-interest mitigations

1. **Pinned competitor versions** at their latest GA, with hashes, frozen for the whole study.
2. **Each tool's documented default config** for its Claude Code integration. Before freeze
   we invite each maintainer to recommend a *publicly documented* config; any such
   recommendation is adopted as-is and recorded.
3. **Competitor review:** the frozen-candidate protocol, `arms.py` and the preflight outputs
   are sent to the RTK and Headroom maintainers (GitHub issue + email) with a **14-day**
   review window before freeze. Their comments and our responses are published in an appendix.
4. **The neutral meter** bills every arm the same way from the provider's own usage; no tool's
   estimate is used for $.
5. **Suite chosen for the competitor's best case** (terminal-heavy), and prior art we did not
   produce (Quesma) as an external replication anchor for the control arm.
6. **Distil gets no advantage we control:** published wheel only; no distil-specific env,
   flags or tuning; the harness's own dry-run fixtures assign distil the *worst* synthetic
   profile.
7. **Publication regardless of outcome:** per-run records and per-request meter logs
   (content-free: model, status, usage integers, bytes, timings), the schedule, the manifest,
   the analysis JSON, and this protocol — including if distil loses.
8. **Frozen before data:** protocol, code, schedule seed and bootstrap seed are tagged before
   the pilot; amendments are dated commits made before the confirmatory run.

## 14. Deviations

Any deviation after freeze is a dated commit to this file saying what changed, why, and
whether it was made before or after any arm outcome was visible. The report lists all of them.

## 15. Threats to validity (for the reviewer)

* **Host-side proxies vs in-container agent** (§4): the shim is a reproduction of `wrap`, not
  `wrap` itself. *Superseded by Amendment 1: tools now run their real `wrap` in-container.*
* **Network path:** proxy arms add a loopback hop the control lacks. Latency is a secondary
  outcome; it does not affect $.
* **Cache coupling across concurrent runs** of *different* tasks (shared tools/system prefix)
  is realistic and symmetric, but it makes absolute $ depend on concurrency; ratios are the
  reported quantity.
* **One suite, one agent.** Results do not transfer to other agents or to IDE-style use without
  the replication suite.
* **σ_w is guessed** until the pilot. If it is large, the study will be underpowered and will
  say so rather than add runs after looking.

## Amendment 1 — 2026-09-25 (before any data; no live run has happened)

Made after verifying every arm against its primary source. Evidence per arm is in
`benchmarks/cost_truth/arms.py` (`Arm.evidence`) and printed by `python -m
benchmarks.cost_truth plan`. Wheels and tarballs were downloaded, hashed, unpacked and
read; no competitor code was executed on the host.

**A. All tools run in the task container, as documented.** The first draft ran proxy tools
on the host behind a `claude` shim. Reading `headroom wrap claude` (0.38.0) showed that its
default setup also registers two MCP servers at user scope (Headroom's retrieve tool, and
Serena via `uvx`), writes `.claude/settings.local.json` in the working directory, and sets
environment for the child — none of which a host-side shim can reproduce. So every arm now
runs its real integration inside the container:

* RTK 0.50.0: musl binary to `/usr/local/bin/rtk`, then `rtk init -g --auto-patch`.
  `--auto-patch` is required, not a tuning choice: without a TTY `rtk init` defaults the
  settings.json patch to *no* and installs no hook (`src/hooks/init.rs`).
* Headroom 0.38.0: its documented install, `headroom-ai[all]` on Python 3.13, via a pinned
  `uv` 0.12.19 and a hash-locked closure (171 packages incl. torch); `uv`/`uvx` on PATH so
  its default Serena registration happens as it does for a `uv tool install` user; Serena's
  run-time resolution pinned with `UV_EXCLUDE_NEWER=2026-09-25`.
* distil 1.54.0: the published wheel on Python 3.12, hash-locked.
* Installs run from a read-only, sha256-verified host mount (`/opt/cost-truth/host`) plus a
  shared uv cache; install time is not part of any outcome. x86_64 containers only.

**B. `ENABLE_TOOL_SEARCH=true` in every arm.** Both Headroom (`cli/wrap.py`, issue #746) and
distil (`onboard.py`) document that Claude Code turns tool-search deferral off behind any
non-first-party `ANTHROPIC_BASE_URL`, and both wraps set it back on. The meter makes every
arm non-first-party, so without this the control and RTK arms would load every tool schema
eagerly — an artifact of the meter that the proxy arms would be credited for undoing.
Setting it identically in all arms restores first-party behaviour; both tools keep an
existing value.

**C. Headroom routes through the edge meter — no fallback needed.** `headroom wrap` starts
its proxy with `os.environ.copy()`; the proxy resolves its Anthropic upstream from
`ANTHROPIC_TARGET_API_URL` (`proxy/server.py`, `providers/registry.py`). The canary (§D)
proves it per run set.

**D. Canary preflight replaces the wrap capture.** Before any task, per arm: a `claude`
shim first on PATH is launched *by the arm's own integration* and sends one request
(`max_tokens` 8, a nonce in `metadata.user_id`) to the base URL it was handed. Pass iff the
meter saw the nonce (flagged as a boolean, never logged) AND, for proxy arms, the URL
handed to Claude Code was not the meter (so the request crossed the tool); for RTK also the
hook entry in `$HOME/.claude/settings.json` and a working `rtk rewrite`. Any failure aborts
the phase. Expected cost ≈ $0.001.

**E. Harbor differences, identical across arms.** The custom agent keeps Harbor's stock
Claude Code install and flags but (i) does not relocate `CLAUDE_CONFIG_DIR` (RTK writes its
hook to `$HOME/.claude`, which Harbor's relocation would silently disable); (ii) does not
remap every model alias to the main model (Harbor does this behind a custom base URL;
real users' Haiku side-calls stay Haiku); (iii) passes the arm as an agent kwarg, not agent
env, so the arm's name is not in the environment the model's Bash tool can read.

**F. distil re-pinned to 1.54.0** (released 2026-09-25), the latest GA, per §4's rule.

## Running the pilot

One command, from the repo root, with `ANTHROPIC_API_KEY` exported and Docker running:

```
uv run python -m benchmarks.cost_truth live --phase pilot --i-approve-spend 92.07
```

It refuses unless the approval equals the pilot cap to the cent, then: verifies and mounts
the pinned artifacts, downloads `terminal-bench@2.1` with Harbor 0.23.0 if absent, samples
10 tasks with the committed seed, runs the four canaries, then 10 tasks × 2 seeds × 4 arms
(80 runs, ≤ 4 concurrent, 360 s same-task spacing) under one $92.07 hard cap. It writes
content-free results to `benchmarks/results/cost_truth/pilot-<ts>/` (runs, per-request meter
logs, preflight, manifest) and `pilot_summary.json` (σ_w, discordance, token-profile
medians, 1h-cache flag — pooled over arms, no arm comparison). Transcripts stay in the
gitignored `benchmarks/results/scratch/cost_truth/trials/`. The whole path runs offline
against the mock with `--mock`.

<p align="center">
  <img src="docs/assets/banner.svg" alt="Distil — compression with a quality contract" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/dshakes/distil/actions/workflows/ci.yml"><img src="https://github.com/dshakes/distil/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://pypi.org/project/distil-llm/"><img src="https://img.shields.io/pypi/v/distil-llm?color=5ad1c9&label=pypi" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/distil-llm/"><img src="https://img.shields.io/pypi/pyversions/distil-llm?color=5ad1c9" alt="Python versions"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/distil-llm?color=8b7bff" alt="license"/></a>
  <a href="#-what-we-wont-pretend"><img src="https://img.shields.io/badge/runtime%20deps-0-5ad19a" alt="zero runtime deps"/></a>
  <a href="https://dshakes.github.io/distil/architecture.html"><img src="https://img.shields.io/badge/typed-py.typed%20%C2%B7%20mypy%20clean-8b7bff" alt="typed"/></a>
</p>

<h2 align="center">Cut your agent's token bill in half.<br/>Prove its decisions don't change.</h2>

<p align="center"><b>The only context compressor with a statistical fidelity certificate.</b><br/>Compressed context <b>solved more</b> than full context — <b>42.0% vs 39.2%</b> on 500 SWE-bench Verified tasks.</p>

```console
$ uvx --from distil-llm distil bench     # ~10s, no API key
GATE: PASS — every trajectory certified non-inferior; aggressive rejected.

$ distil wrap -- claude                  # route Claude Code, zero config
distil · session ▼75.0K · 62% smaller · $0.31 · total ▼27.0M · ✓eq 99.5%
```

<table align="center"><tr>
<td align="center"><b>⚡ Get the savings</b><br/><sub>2 min, no config</sub><br/><br/><code>pipx install distil-llm</code><br/><code>distil onboard</code></td>
<td align="center"><b>🔬 See the proof</b><br/><sub>real harness</sub><br/><br/><a href="#-the-proof"><b>benchmark ↓</b></a> · <a href="docs/PAPER.md">paper</a><br/><a href="https://dshakes.github.io/distil/compare.html">vs the others</a></td>
</tr></table>

<p align="center"><sub>Honest scope: +2.8pp is a point estimate (CI −0.6..+6.2pp — <b>non-inferiority certified, superiority not yet</b>). <a href="#-the-proof">Details, incl. what doesn't transfer →</a></sub></p>

<p align="center">
  <a href="#-use-it-now">Use it</a> ·
  <a href="#-works-with-every-sdk">Integrations</a> ·
  <a href="#-install-your-way">Install</a> ·
  <a href="https://dshakes.github.io/distil/compare.html">vs the others</a> ·
  <a href="https://dshakes.github.io/distil/getting-started.html"><b>Full Docs →</b></a>
</p>

---

<h3 align="center">Proof first — not a pitch 📊</h3>

<p align="center"><img src="docs/assets/head-to-head.svg" alt="Distil vs LLMLingua-2 vs Headroom — token savings, decision-change rate, latency" width="100%"/></p>

<table align="center">
<tr><th>On a real 500-instance long-horizon agent<br/><sub>(SWE-bench Verified, official harness)</sub></th><th>task success</th><th>tied with full context?</th><th>reversible&nbsp;+&nbsp;certified?</th></tr>
<tr><td><b>Distil</b> (gated + surprise digest, v1.7)</td><td align="center"><b>42.0%</b></td><td align="center">✅ <b>+2.8pp over full</b> <sub>(CI −0.6..+6.2)</sub></td><td align="center">✅</td></tr>
<tr><td><b>Distil</b> (relevance-gated, E8)</td><td align="center"><b>36.8%</b></td><td align="center">✅</td><td align="center">✅</td></tr>
<tr><td>Headroom <sub>(lossy)</sub></td><td align="center">32.6%</td><td align="center">❌ −6.6pp</td><td align="center">❌</td></tr>
<tr><td>LLMLingua-2 <sub>(lossy)</sub></td><td align="center">2.4%</td><td align="center">❌ −36.8pp</td><td align="center">❌</td></tr>
<tr><td>no compression <sub>(full)</sub></td><td align="center">39.2%</td><td align="center">—</td><td align="center">—</td></tr>
</table>

<p align="center"><b>Distil is the only compressor statistically tied with full context — and its v1.7 surprise-preserving digest lands <i>above</i> full context (42.0% vs 39.2%, paired non-inferiority certified)</b> while every lossy tool craters. And on the live head-to-head above (graded by <code>claude-opus-4-8</code>), it certifies <b>83.2% savings at a 0% decision-change rate</b>, ~1,000× faster than the nearest tool. <a href="#-the-proof">Full breakdown ↓</a></p>

---

## 🚀 Use it now

**One command sets you up and tells you what to do next:**

```bash
pipx install distil-llm
distil onboard      # detects your agent + billing, wires the status line, prints a guided tour
```

It detects your environment (Claude Code · Codex · Gemini CLI; metered vs subscription) and hands you the exact commands. Or wrap your agent directly — **no config, no code change:**

```bash
# Claude Code on a metered API key — saves real $$:
distil wrap --expand -- claude

# Claude Code on a Pro/Max subscription — flat-rate, ToS-safe (trims context, not $):
distil wrap --lossless-only -- claude

# Codex, Gemini CLI, or any agent — same pattern:
distil wrap --expand -- codex
```

<details>
<summary><b>Make it the default</b> — never type <code>distil wrap</code> again</summary>

**Tired of typing `distil wrap` every time?** Make it the default — once:

```bash
distil default            # adds a managed shell alias so `claude` always routes through distil
distil default --undo     # remove it anytime (backed up before any change)
```

It detects your shell (zsh / bash / fish / PowerShell) and billing mode, writes the
right line to the rc file your shell actually reads, and **tells you what it detected**.
Want every SDK covered (not just the agent you type)? `distil default --always-on`
runs a persistent proxy service — powerful, but it's a daemon you keep alive.


</details>

Then watch genuine savings from **your** traffic — measured, not estimated:

```bash
distil leaderboard          # cumulative tokens + $ saved, from the local ledger
distil dashboard            # live terminal TUI — token-trim + decision-equiv bars, Ctrl-C to exit
```

**Validate it on your traffic.** `--shadow` runs a fraction of requests twice (compressed **and** full) and compares the agent's chosen next action:

```bash
distil wrap --shadow 0.1 -- claude   # wrap + shadow 10% of requests
distil shadow-stats                  # live decision-equivalence rate
```

Honest scope: that's next-action equivalence — a **proxy**, not task success ([E7](#-the-proof) shows it doesn't fully transfer under aggressive *lossy* compression). Distil fails safe to full context.

> **Will it save money?** Only on **metered** billing (API key) — fewer tokens, fewer dollars. On a flat-rate **subscription** it trims context + latency, not the bill. Coding agents: short sessions ~7%, big wins on **long, many-turn** sessions the model never re-reads.

---

## 💡 Why Distil is different

You don't need byte-equivalence — you need **decision-equivalence**: your agent taking the *same actions* with compressed context. That's measurable and certifiable.

- **Certified, not estimated** — a strategy ships only if a non-inferiority test passes; can't certify → full context.
- **Certified end-to-end, too** — `distil certify-trajectories` bounds how many solvable tasks compression can cost (no other compressor certifies either level).
- **Reversible, not lossy** — digests behind a handle, keeps the original, hands the agent a `distil_expand` tool. Compress fearlessly.
- **Compounds on outcomes** — expansions and matched failures teach the policy what to protect (signatures only, never content) — always *more* conservative.
- **Streams like it isn't there** — SSE relays chunk-by-chunk; TTFT preserved.

> **Fidelity tiers:** lossless (`--verbatim`) · reversible (byte-recoverable on demand — default) · lossy (every other tool). Only Distil offers and certifies the reversible tier.

---

## ⚡ Prove the numbers yourself — no API key

Don't take the table above on faith. `distil bench` re-certifies savings *and* decision-equivalence on a bundled 7-domain corpus, offline, in seconds — the same gate that runs in CI:

```bash
uvx --from distil-llm distil bench   # certify savings + quality across 7 domains, in seconds
```

```
domain            trajectory                $ saved   distil   aggr  pruned
---------------------------------------------------------------------------
ops/sre           sre-disk-incident           33.1%     PASS   FAIL     615
coding            coding-bugfix               28.7%     PASS   FAIL     736
support           support-refund              32.6%     PASS   FAIL     765
research          research-synthesis          25.7%     PASS   FAIL     809
data-analysis     data-analysis-sql           18.1%     PASS   FAIL     965
devops            devops-rollback             25.0%     PASS   FAIL     857
finance           finance-reconcile           29.1%     PASS   FAIL    1014
---------------------------------------------------------------------------
aggregate: distil cuts $0.14212 -> $0.10402 (26.8% cheaper) reversibly; 5761 tokens causally prunable.
GATE: PASS — every trajectory certified non-inferior; aggressive rejected on all.
```

<p align="center"><img src="docs/assets/domains.svg" alt="measured across 7 domains" width="100%"/></p>

> **Why trust the number?** Token-savings numbers are easy to fake — measure quality at *low* compression, advertise savings at *high* compression. Distil refuses that: accuracy and compression are measured on the **same** trajectories, and a strategy that can't pass non-inferiority doesn't ship.
> ```
> distil certify --strategy distil       # VERDICT: PASS  (100% decision-equivalence)
> distil certify --strategy aggressive   # VERDICT: FAIL  (mean diff −1.0, blocked)
> ```

`distil eval` plots the **certified compression frontier** — a savings-vs-quality curve where every point carries its certification verdict, locating the cliff past which lossy compression drops decisions. The artifact no competitor publishes: [benchmark.html](https://dshakes.github.io/distil/benchmark.html).

---

## 📊 The proof

Three results, all reproducible, all published with caveats:

- **Live head-to-head** vs real `llmlingua` / `headroom-ai` (graded by `claude-opus-4-8`): **83.2% savings at 0% decision-change**, ~1,000× faster. → [benchmark](https://dshakes.github.io/distil/benchmark.html)
- **E7 (SWE-bench Verified):** aggressive *lossy* compression **craters** task success (52% → 16%) — a per-step certificate doesn't transfer to multi-turn. The **reversible** tier survives (56% vs 52%). We publish it because it's true. → [E7](https://dshakes.github.io/distil/research.html#e7)
- **E8–E14 (500-instance agent):** the reversible tier is the **only compressor non-inferior to full context**, generalizes across 5 models / 3 vendors, and the newest digest lands *above* full (42.0% vs 39.2%). → [E8–E14](https://dshakes.github.io/distil/research.html#e8)

Full methodology, McNemar tests, per-instance data: [`docs/PAPER.md`](docs/PAPER.md) · [PDF](docs/paper/main.pdf).

---

## 📡 See it working

Measured on **your** traffic, never estimated, nothing leaves your machine:

- **Per request:** `x-distil-*` response headers (`tokens-saved`, `mode`, `compressible-tokens`, `expanded`).
- **Per machine:** `distil leaderboard` (`--html` for a page).
- **Shadow mode:** `distil proxy --shadow 0.05` reports the live decision-change rate — streaming-aware.
- **Org-wide:** `distil proxy` sidecar + set `ANTHROPIC_BASE_URL` once; every client routes through it.

Dashboard, status-line plugin, federated leaderboard: [Deploy & observability](https://dshakes.github.io/distil/deploy-security.html).

## 🔌 Works with every SDK

One proxy. Point any `base_url`-honoring client at it — **Python, TypeScript, any language** — and get cache-aware **reversible** compression with **no code change**.

<p align="center"><img src="docs/assets/cross-sdk.svg" alt="one proxy, every SDK" width="100%"/></p>

```bash
distil proxy --upstream https://api.anthropic.com   # localhost:8788
```

| SDK / framework | Change | Example |
|---|---|---|
| Anthropic SDK (Py/TS) | `base_url="http://127.0.0.1:8788"` | [`examples/python_anthropic.py`](examples/python_anthropic.py) |
| OpenAI SDK | `base_url="http://127.0.0.1:8788/v1"` | [`examples/python_openai.py`](examples/python_openai.py) |
| Vercel AI SDK | `createAnthropic({ baseURL: '…:8788' })` | [`examples/js_vercel_ai_sdk.ts`](examples/js_vercel_ai_sdk.ts) |
| LangChain (py/js) | `anthropicApiUrl` / base URL | [`examples/js_langchain.ts`](examples/js_langchain.ts) |
| LiteLLM | `api_base="http://127.0.0.1:8788"` | [`examples/python_litellm.py`](examples/python_litellm.py) |
| Google Gemini | `--upstream https://generativelanguage.googleapis.com` | [`examples/python_gemini.py`](examples/python_gemini.py) |

Prefer in-process? Wrap the client directly — still no call-site change:

```python
from distil.adapters.anthropic import wrap
client = wrap(anthropic.Anthropic())   # compresses the request, keeps the cache warm
```

**Framework hooks (no proxy, no network hop)** — for agent frameworks that own the message list, compress it where it lives:

| Framework | Hook | Example |
|---|---|---|
| LiteLLM | `distil.integrations.litellm.compress(kwargs)` | [`examples/python_litellm.py`](examples/python_litellm.py) |
| LangChain | `distil.integrations.langchain.compress_messages(msgs)` | — |
| LangGraph | `pre_model_hook=pre_model_hook()` (compresses graph state before the model node) | [`examples/python_langgraph.py`](examples/python_langgraph.py) |

---

## 📦 Install your way

**New here?** `pipx install distil-llm`, then `distil onboard` — it sets you up and guides you (see [Use it now](#-use-it-now)). Want to see it prove itself first instead? `distil bench` runs the certified gate in ~10s, no API key. The matrix below is for picking an *install format* — everything in it is an alternative, not a requirement.

<details>
<summary><b>Install gotchas & troubleshooting</b> (package name, old-Python errors, stale mirrors)</summary>

> ⚠️ **The one gotcha — the name.** The PyPI package is **`distil-llm`** but the command is **`distil`** (the bare name was taken). So `pipx install distil-llm` → run `distil …`. `pip install distil` installs something else.

> 🔧 **Seeing `Could not find a version that satisfies the requirement distil-llm (from versions: none)`?** The package **is** on PyPI — that error means your `pip`/`pipx` is on a Python older than the package's floor, so pip filters every release out. **Distil now supports Python 3.9+** (the version macOS ships), so a current install just works; if you still hit this on a very old Python, let **uv provision one for you**: `uvx --python 3.12 --from distil-llm distil bench` (or `uv tool install --python 3.12 distil-llm`). Check yours with `python3 --version`.

> 🔧 **Got an *old* version (e.g. `0.25.1`) instead of the latest?** Public PyPI always serves the newest (`pip index versions distil-llm` lists them). If you got an older one, your `pip`/`pipx` is **not resolving against public PyPI** — almost always a **stale internal mirror** (Artifactory / CodeArtifact / Nexus that hasn't synced the latest yet — common right after a release) or a **`<1.0` version pin** in a constraints file / `pip.conf`. Diagnose and fix:
> ```bash
> pip index versions distil-llm     # stops at an old version? → your index/mirror is stale
> pip config list ; env | grep -i pip   # look for an index-url or PIP_CONSTRAINT pin
> # unblock now — force public PyPI:
> pipx install --pip-args="--index-url https://pypi.org/simple/" distil-llm
> # (or, if you must use the mirror, ask your platform team to sync distil-llm; it exists upstream)
> ```


</details>

<p align="center"><img src="docs/assets/install.svg" alt="install options" width="100%"/></p>

| Format | Command | Prereq |
|---|---|---|
| **Zero install** | `uvx --from distil-llm distil bench` | [uv](https://docs.astral.sh/uv/) — **auto-provisions Python 3.9+** |
| **Isolated CLI** | `pipx install distil-llm` → `distil bench` | Python **3.9+** (else `pipx install --python python3.12 distil-llm`) |
| **Homebrew** | `brew install dshakes/tap/distil` | Homebrew |
| **Docker** | `docker run ghcr.io/dshakes/distil:latest bench` (or `docker build -t distil .`) | Docker |
| **Single file** | `make pyz` → `python dist/distil.pyz bench` | Python 3.9+ |
| **In a venv** | `pip install distil-llm` (inside an active virtualenv) | Python 3.9+ |

> The import package and CLI are `distil`; the PyPI distribution is `distil-llm` (the bare name was taken — so `uvx`/`pip` must reference `distil-llm`, not `distil`). Distil is a CLI: install it **isolated** (pipx/uv/brew/Docker), because modern macOS/Linux block system-wide `pip install` ([PEP 668](https://peps.python.org/pep-0668/)). **Node / any language:** point your SDK's `base_url` at `distil proxy`, or use `distil wrap -- <agent>` — no Distil-specific package needed.

---

## 🧰 Cheat-sheet

Basics are in [Use it now](#-use-it-now) and [Works with every SDK](#-works-with-every-sdk). Beyond that:

| Goal | Command |
|---|---|
| **Set up + a guided tour (start here)** | `distil onboard` |
| Make distil the default (no per-session `wrap`) | `distil default` · undo: `distil default --undo` |
| Remove distil's footprint (before uninstalling) | `distil offboard` · also clear data: `distil offboard --purge` |
| Diagnose your setup (ledger, shadow, proxy self-test, wiring) | `distil doctor` |
| Wire the savings status line into Claude Code | `distil setup` (compact segment: `DISTIL_STATUSLINE=minimal`) |
| Watch genuine savings accumulate | `distil leaderboard` · live TUI: `distil dashboard` |
| Live decision-equivalence on real traffic | `distil wrap --shadow 0.1 -- claude` → `distil shadow-stats` |
| Certify on *your* domain | `distil ingest --input prod.jsonl --out ./mycorpus` → `distil conformal --corpus ./mycorpus` |
| Recover digested detail from any agent (MCP) | `distil mcp` |
| Self-improving keep policy | `distil learn` / `distil online` |

> **Status line:** rich by default — `distil · session ▼7.8K · 4% smaller · total ▼27.0M · ✓eq 99%` (this run + lifetime + health). Sharing the line with git/cwd/model? `DISTIL_STATUSLINE=minimal` gives a two-fact segment: `distil ▼7.8K · 27M total`. On a flat-rate **Claude subscription** dollars are notional and auto-hidden (override: `DISTIL_SUBSCRIPTION=0/1`).

Rule of thumb: **subscription/interactive → `--lossless-only` (+`--verbatim`)** · **PAYG/autonomous → default digest (+`--expand`)** · **coding re-reads → add `--session-delta`**.

---

## 🧠 How it works

<p align="center"><img src="docs/assets/architecture.svg" alt="architecture — pipeline and the quality-contract loop" width="100%"/></p>

Two techniques carry most of the win — they target where the money actually is in an agent loop, not where it looks like it is.

### ① Cache-aware compression — the dominant lever

You re-send the growing context every step. With prompt caching a cache **read is ~10× cheaper** than fresh input, so the real cost is cache **misses**, not context **size**. Distil keeps the prefix byte-stable (schema canonicalization + lifting volatile fields like timestamps/UUIDs out of the prefix) and compresses only the volatile tail.

<p align="center"><img src="docs/assets/cache-aware.svg" alt="cache-aware savings" width="100%"/></p>

> Naive recompression sends **fewer tokens yet costs more than not compressing at all**, because it rewrites the cached prefix every turn. Distil doesn't — that's the whole game most tools miss.

### ② Causal / counterfactual pruning — the discovery engine

The eval isn't a ruler bolted on the side; it's a *discovery engine*. Remove a context block, replay, did any decision change? Blocks that never change a decision are **provably free to drop**.

```bash
distil prune
# doc-0   PRUNE (causally inert)     # speculative retrieval, never cited
# obs-0   keep (changed a decision)  # carries the decision-driving signal
```

---

## 🎓 The certificate (DERC)

The gate answers *"is this strategy non-inferior on my corpus?"*. The **Decision-Equivalence Risk Certificate** answers the operational one: *"for a risk budget I choose (say ≤5% decision-change), how hard can I compress with a guarantee that holds on my real traffic?"*

```bash
distil conformal --corpus ./mycorpus --alpha 0.05 --delta 0.05
# ✔ CERTIFIED 'lossless' → 57.4% savings; decision-change ≤ 5.0% at 95% confidence (Learn-Then-Test)
```

It's **conformal risk control** (Learn-Then-Test / CRC — distribution-free, finite-sample), not a heuristic threshold. The one load-bearing caveat: the guarantee requires **exchangeability** (calibration traffic ≈ live traffic) and is **marginal** over that distribution — recalibrate on drift. Full theory + citations: [Concepts](https://dshakes.github.io/distil/concepts.html) · [`docs/PAPER.md`](docs/PAPER.md).

### 🏔 The trajectory-level certificate

DERC certifies the *step*; this certifies the *task*. Our E7 experiment — and the 2024–26 agent-compression literature — shows per-step fidelity can pass while end-to-end success collapses, so distil also certifies the level users actually feel: run your eval suite twice (full context vs compressed), feed the matched outcomes in, and get a distribution-free bound on **how many solvable tasks compression may cost you**:

```bash
distil certify-trajectories outcomes.jsonl --alpha 0.05 --delta 0.05
# each line: {"task_id": "...", "full_success": true, "compressed_success": true}
# → With confidence 95%, compression degrades at most 5.0% of tasks the full
#   context would have solved (observed 0.5% over 200 matched trajectories).
```

It refuses to certify on small samples, states its exchangeability assumptions in the certificate itself, and ships an anytime-valid **drift monitor** (`trajectory_risk.drift_monitor`) that tells you when live traffic has shifted enough that the certificate is stale. Matched failures also feed the **outcome-guided policy** (`distil.compress.guideline`): content classes that break tasks when digested get protected byte-exact, automatically.

## 🧩 What's inside

40+ shipped capabilities, all real (no stubs): the cache-aware cost engine, causal pruning, the TOST gate + conformal certificate, the proxy + Anthropic/OpenAI/Gemini adapters, an MCP server, LiteLLM/LangChain/LangGraph hooks, learned keep-models, output compression, and a Rust hot-path core — with **zero runtime dependencies** in the core.

Full module-by-module map: [Architecture](https://dshakes.github.io/distil/architecture.html) · [Techniques](https://dshakes.github.io/distil/techniques.html) · [CLI reference](https://dshakes.github.io/distil/cli.html).

## 🔒 Security & deployment

- **Localhost-only by default** — the proxy binds `127.0.0.1` and forwards only to the single configured upstream (no SSRF).
- **No secret/body logging** — request bodies and credentials are never logged.
- **Auth-mode gating** — `--lossless-only` keeps subscription/OAuth sessions to lossless strategies and never injects tools (provider-ToS-safe); the reversible, certified digest still runs. Add `--verbatim` to skip the digest entirely (Tier-0 only) for interactive sessions.
- **Stateless** — nothing is persisted; ZDR-compatible.

See [Deploy & security](https://dshakes.github.io/distil/deploy-security.html) for topologies (local sidecar, container sidecar, shared gateway) and the threat model.

---

## ✅ What we won't pretend

- **Default tokenizer is an offline heuristic** — ratios robust, dollars approximate. `--tokenizer anthropic` for billing-grade counts.
- **Default runner is a deterministic stand-in** (offline gate with ground truth). Non-circular eval grades **real agent traces with a real model** — [proof harness](#-reproducible-evaluation--the-paper).
- **Credible grading, enforced:** majority-vote (single samples let grader noise look like a decision change), a same-family grader, and grading the reversible tier *with* its `distil_expand` recovery loop.
- **No fabricated weights** — the keep-model is a real logistic classifier (96.4%); the transformer ships a demo checkpoint you retrain on your traces.

### Deliberately *not* a platform

Distil is a **compression engine with a correctness gate**, not a context suite. We declined what can't go under the certificate:

| Adjacent feature | Our stance |
|---|---|
| Persistent memory / knowledge graph | **Out of scope** — a lossy store is the opposite of byte-reversible. |
| Hosted semantic cache | **Out of scope** — we make the *provider's* prompt cache pay off, not a second lossy one. |
| Editor/Copilot auth | **Out of scope** — Distil sits on the wire or in-process; never brokers credentials. |

What we *did* adopt (it survives the gate): a pluggable salience scorer to *protect* entities, cache-prefix observability, and framework hooks.

---

## 🎯 Both sides of the bill

Distil compresses **input/context** (comprehensive) **and output** — generation-side verbosity shaping (PAYG, measured with `distil output-savings`) plus a reversible output-on-re-entry digest, so verbose past answers stop costing full price as history. Details: [Output & I/O](https://dshakes.github.io/distil/output.html).

## 🔬 Reproducible evaluation & the paper

Every number reproduces from the bundled corpus (`distil bench`, no key). The non-circular proof harness grades **real agent traces with a real model** (τ-bench / SWE-bench): [`benchmarks/PROVE.md`](benchmarks/PROVE.md). Compiled paper, LaTeX source, and all committed results: [`docs/PAPER.md`](docs/PAPER.md) · [`docs/paper/`](docs/paper/) · [paper PDF](docs/paper/main.pdf).

<h3 align="center">Stop paying to re-send context your agent never reads.</h3>

<p align="center">
<code>pipx install distil-llm && distil bench</code><br/>
<sub>certified savings across 7 domains in ~10 seconds — zero API key, zero runtime deps</sub>
</p>

<p align="center">
<a href="https://dshakes.github.io/distil/getting-started.html"><b>Get started →</b></a> ·
<a href="#-works-with-every-sdk">Wire it into your SDK</a> ·
<a href="docs/PAPER.md">Read the proof</a> ·
<a href="https://pypi.org/project/distil-llm/">PyPI</a>
</p>

---

## ⭐ If distil saved you tokens

A star is how the next engineer finds provable savings instead of a lossy guess — and
`distil stats --badge` gives you a shareable badge of **your own measured number** to
show alongside it. That badge + this repo are the whole marketing department.

## 🤝 Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one rule that matters: **a new compression strategy must pass `make gate`** (non-inferior on every domain, byte-reversible). No green gate, no merge. That's the whole philosophy in one sentence.

## License

[Apache-2.0](LICENSE) · *“Same potency, less volume.”*

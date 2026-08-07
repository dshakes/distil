# GA-readiness

An honest, current ledger of what is production-grade and what still gates a "drop it on any
agent and trust it unattended" general-availability claim. Updated as items close. The bar:
distil should never *silently* ship a lossy operating point — when it cannot certify safety,
it falls back to full context.

## Status: 1.0 GA — the engine, integrations, and certificate machinery are production-grade and API-stable

**What "1.0 / GA" means here (and what it does not).** The compression engine, the proxy/SDK
integrations, and the decision-equivalence certificate machinery are production-grade,
API-stable, and covered by 700+ tests with a zero-dependency stdlib core. The guarantee is
**honestly scoped**: decision-equivalence is certified *distribution-free and finite-sample*,
**conditional on exchangeability** with your calibration distribution — and when distil cannot
certify safety it **falls back to full context**, never silently shipping a lossy operating
point. 1.0 is a commitment to a stable surface and a never-silently-lossy contract; it is **not**
a claim that aggressive compression is safe on every agent untuned (E7/E11 show the opposite —
that is exactly why calibration is mandatory and fail-safe).

The headline GA risk surfaced by E11 (the safe operating point is capability-dependent, and a
hand-tuned constant can silently lose 31 pp on a stronger model) is **closed** by
auto-calibration with a fail-safe default. The next two largest items — **drift detection** and
the **single-grader** caveat — are now closed too (anytime-valid monitor + conservative grader
ensemble; see "Recently closed"). What remains is not missing machinery but **empirical breadth
that requires live runs/compute we have not spent**: validating the shipped speculative,
multi-grader, and RL-policy paths end-to-end, and broadening task-success beyond SWE-bench
Verified + τ-bench to more domains and models. The design is domain-agnostic; closing these is
*running it at scale*, not building more. We mark them honestly rather than claim them — and we
ship 1.0 because the contract that protects you (certify-or-fall-back) is itself complete.

## Launch gate (marketing / "grand release")

Engineering GA (above) and *telling the world* are different bars. 1.10.0→1.11.3 shipped
six same-day releases, each fixing the previous — all found internally, which is the good
news, and the reason to launch only when the checklist below is green. Every item is
binary and checkable; opinions don't count, evidence does.

- [x] **Chaos suite in CI** — the bug classes that shipped (signal races, accept-loop
  death, crash-lost accounting) are pinned by tests that run on every push
  (`tests/test_chaos.py`, `tests/test_wrap.py`, `tests/test_gateway_cov.py`).
- [x] **rc + soak release policy in force** — runtime-behavior releases bake as an rc
  for ≥ 3 days of real use before the final (RELEASING.md).
- [ ] **14 consecutive days at head with zero P0/P1** — a same-day fix release resets
  the clock. **Reset 2026-08-07:** the settings-precedence outage above is a P1 found
  on real traffic (three occurrences in one day). Per RELEASING.md this means rc2 and
  a restarted soak clock — 1.41.2 final cannot be cut on the original 2026-08-10 date.
- [ ] **External beta** — ≥ 10 real users wrapping real agents for ≥ 1 week, with a
  feedback channel; no open P0/P1 from the cohort.
- [ ] **Live decision-equivalence evidence** — shadow-mode ✓de ≥ 99% at n ≥ 25 on the
  maintainer's own daily traffic *and* from at least 3 beta users.
- [ ] **Fresh-install first-run verified on all three OSes** — pipx/brew install →
  `distil wrap` → first savings visible, walked end-to-end on macOS, Linux, Windows
  (CI covers the test matrix; this item is the *human* walkthrough).
- [ ] **Claims re-audit at launch commit** — every README/site/docs claim re-checked
  against a committed artifact (last full audit: 1.11.0). E7's scope limit stays in
  the copy: compression aggressiveness is calibrated per agent, never marketed as
  universally safe untuned.

### Risk register (external dependencies, tracked not gated)

- **Subscription-agent interception is undocumented upstream.** Claude Code
  subscription (OAuth) traffic honors `ANTHROPIC_BASE_URL` in practice today (verified
  live 2026-07-06: two OAuth sessions, 6.1M tokens saved), but Anthropic documents no
  such contract, some OAuth endpoints are hardcoded (anthropics/claude-code #23022,
  #48011), and per-session bypass has been observed once (frequency unmeasured until
  soak markers accumulate — `scripts/soak-report.sh`). A Claude Code update could break
  interception for subscription users; we can detect it (statusline ⚠ + marker rate)
  but not prevent it. Mitigation: beta cohort reports auth type + marker data.

## Closed

| Item | Evidence |
|---|---|
| **Relevance gate is a shippable library primitive** (not benchmark-only) | `distil/gate.py` (`working_set_indices`, `gate_fraction`); contract verified against the benchmark proxy in `tests/test_gate.py` |
| **Operating point is auto-calibrated to agent capability** | `distil/calibrate.py` selects the most aggressive `gate_recent` still non-inferior to full; `distil calibrate` CLI; validated on real E11 data (selects gate@12, rejects gate@6) in `tests/test_calibrate.py` |
| **Fail-safe default (never silently lossy)** | If no operating point certifies non-inferior, calibration returns `fail_safe` → caller keeps full context (`test_fail_safe_when_nothing_certifies`, `test_real_e11_strict_margin_fails_safe`) |
| **Cross-model generality demonstrated** | E11: non-inferiority transfers to DeepSeek-V3 (different vendor, far stronger) at a capability-appropriate point |
| **Engineering maturity** | v1.0.0, 700+ tests, full CI (ci/pages/paper-build/release), zero-dependency stdlib core, packaged (`distil` entrypoint) |
| **Per-turn + trajectory certificates, validated out-of-sample** | E2 (coverage 96.6–100%), E10 (trajectory, coverage 95.4/96.7%) |
| **Observability: OTel GenAI spans (opt-in `[otel]` extra)** | `distil/otel.py` emits `gen_ai.*` semconv spans + `distil.tokens.original/compressed`, `distil.compression.ratio`, `distil.shadow.sampled`; no-op boolean guard without the extra; `tests/test_otel.py`. **Correction (2026-08-07):** a `/distil/metrics` endpoint now exists, so "structurally absent" no longer holds — the leak class is instead closed by an *admin gate* (open on loopback; `--admin-token` required and refused without one on any non-loopback bind), tested directly for 403-unauthenticated, 401-wrong-token, and label injection |
| **Supply chain: attestations + SBOM + Scorecard** | PEP 740 Sigstore attestations active (trusted publishing, `gh-action-pypi-publish@release/v1` ≥ v1.11); CycloneDX SBOM attached to GitHub releases (`release.yml`); weekly OpenSSF Scorecard (`scorecard.yml`) |
| **Shared-state writes locked across concurrent sessions** | `shadow.jsonl` append now flocked like `ledger.json`/`mcp_store.json`; cross-process torn-row hammer test in `tests/test_shadow.py` |
| **CI signal tests deflaked (red-main root cause)** | readiness-marker sync replaces fixed sleeps in the wrap signal tests — red CI on `main` since 1.11.2 was test-side races on loaded runners, not product regressions |
| **Upgrades apply to live sessions (hot-swap)** | wrap supervises the proxy as a subprocess on a shared listener FD; upgrade → new worker, same port, old worker drains in-flight streams; health-checked with rollback and dead-worker respawn (`distil/hotswap.py`, `tests/test_hotswap.py`: e2e handover, mid-stream drain, rollback — 20/20 deterministic) |

## Open (tracked GA items)

| Item | Why it gates full GA | Mitigation today |
|---|---|---|
| **Validation breadth** | Task-success is validated on SWE-bench Verified coding agents across **5 models / 3 vendors** (E8 n=500 Haiku; E11 n=200 DeepSeek-V3, n=50 Sonnet 4.6, n=50 gpt-4.1, n=50 gpt-4o-mini) plus τ-bench (proxy). OpenAI runs are now DONE; note gpt-4.1 gate@6 is partial (account credit exhausted mid-run, 32/50 instances) and all three n=50 OpenAI/Sonnet points have wide CIs (directional, not powered). Still single-domain (coding); broad multi-domain production traffic is not yet covered. | The certificate machinery is domain-agnostic; broadening is data + a new-domain harness, not redesign. |
| **Calibration data requirement** | Auto-calibration needs a small paired full-vs-candidate run on representative traffic before the gate can ship aggressively. | Fail-safe means the *absence* of calibration data degrades to full context (correct, not lossy) rather than to a guessed operating point. |

### Closed 2026-08-07 (adoption + enterprise pass)

| Item | How it closed |
|---|---|
| **Embeddable as a dependency** | `from distil import compress_messages, expand_handle` (`distil/api.py`). Previously the package exported only `__version__`, so no framework could depend on distil without shelling out to the CLI — the structural reason third-party integrations did not exist. |
| **Settings-precedence outage (P1, found in the field)** | A base URL pinned in an agent's `settings.json` outranks the environment `distil wrap` exports. With the always-on service down, every session failed with a connection error naming the *provider*. `distil wrap` now refuses to start into that configuration and names the fix (`distil/precedence.py`). |
| **First-run "0% smaller"** | A real token count beside a rounded `0%` reads as broken. Sub-1% now renders `<1% smaller` across all four surfaces (status line, terminal dashboard, HTML card, browser dashboard), with the reason attached. |
| **`doctor` false "bypassed" warn** | It inferred traffic from the savings ledger, which skips zero-saving windows by design. Receipts are written per request regardless of savings and are now the signal. |
| **SSO / RBAC for the gateway** | OIDC bearer tokens alongside `dsk-` keys, three ordered roles, stdlib-only JWS verification; RS256 refused rather than unverified when the `[oidc]` extra is absent. Adversarial suite + live-gateway e2e. |
| **Kubernetes deployability** | Helm chart (`packaging/helm/distil-gateway`) with secure defaults asserted by test, plus alert rules and a Grafana dashboard whose every PromQL expression is CI-checked against the emitted metric set. |
| **Security policy + review pack** | `SECURITY.md` (was missing; Scorecard checks for it) and `docs/SECURITY-WHITEPAPER.md`, which states the gaps — no SOC 2, no SLA, no SAML, no cross-tenant memory isolation — rather than omitting them. |
| **IDE agents** | `docs/IDE-AGENTS.md`: proxy + base-URL setting for Cursor/Cline/Continue/Windsurf, and an explicit "not supportable" for Copilot instead of silence. |

### Recently closed (were open)

| Item | How it closed |
|---|---|
| **Drift detection / auto-recalibration** | `distil/drift.py:DriftMonitor` — an *anytime-valid* sequential alarm (betting e-process for `H0: risk ≤ α`) that may be checked after every turn with false-alarm probability ≤ δ *no matter how often you peek* (Ville's inequality). Trips when live decision-change exceeds the certified budget → signal to recalibrate or fall back to full context. Validated: bounded false alarms under peeking + high detection power (`tests/test_drift.py`). |
| **Single grader family** | `distil/ensemble.py:EnsembleGrader` — grade with multiple model families, default **"any"-change** aggregation, which is conservative (can only *raise* measured risk), so the certificate stays valid even if one grader family is unfaithful. Aggregation logic shipped + tested (`tests/test_ensemble.py`); multi-family *validation* still needs a live multi-API run. |
| **Anytime-valid / tighter certificate** | `distil/conformal.py:betting_upper_bound` — the hedged-capital betting confidence sequence (Waudby-Smith & Ramdas, JRSSB 2023): variance-adaptive and valid simultaneously at every `t`. Coverage + anytime property Monte-Carlo–validated (`tests/test_conformal_bounds.py`). Honest tradeoff: for one-shot binary losses Bentkus is already near-optimal, so betting is *comparable* there; its edge is continuous monitoring and graded-loss adaptivity. |

## Cost frontier under the motto (advanced techniques)

"Best in class" holds on the motto's axis (certified decision-equivalence + task success), not
on raw cost — an uncertified lossy method can always be cheaper because it is allowed to change
decisions. Within the certified envelope, these techniques cut cost without spending the
certificate. Status is honest about shipped-and-validated vs. framework vs. research.

| # | Technique | Status | Where |
|---|---|---|---|
| 1 | **Cache-monotone gate** — deterministic, append-only digests so the digested prefix is byte-stable and prompt-cache/KV reuse captures it | **Shipped + tested** | `distil/gate.py:monotone_gate`; `tests/test_cost_frontier.py` |
| 2 | **Graded gate** — per-distance compression tiers (crush the far periphery harder), certified with the tighter empirical-Bernstein bound | **Shipped + tested** | `distil/gate.py:graded_gate`; `distil/conformal.py:tight_risk_bound` |
| 3 | **Tighter conformal (empirical-Bernstein)** — certifies more savings at the same confidence on *graded* losses; coverage-validated by Monte-Carlo | **Shipped + coverage-tested** | `distil/conformal.py:empirical_bernstein_bound`; `tests/test_conformal_bounds.py` |
| 4 | **Speculative expansion** — pay for full context only when a certified divergence trigger fires; controller + certified miss-rate | **Framework shipped + tested; needs a live calibration run for end-to-end savings** | `distil/speculative.py` |
| 5 | **Constrained-bandit operating-point search** — online successive-elimination under the NI constraint, fail-safe | **Shipped + tested**; full constrained-RL keep-policy is **research** (needs training data) | `distil/calibrate.py:bandit_select_operating_point` |

Honest cost caveat baked into the design and tests: on content that is *already fully
cacheable*, caching alone can be cheaper than any compression (compressing rewrites cached
bytes as fresh). The cache-monotone gate's win is over a cache-*hostile* gate; the gate's
primary payoff stays accuracy (E8/E11). The certificate's tightening (#3) and the
graded/speculative/bandit machinery cut cost *inside* the certified envelope — they never
trade the guarantee for dollars.

## How to calibrate before shipping the gate

```bash
# 1. Run your agent on a small calibration set under full context and 2–3 candidate
#    working-set sizes (gate_recent), producing swebench-style score JSONs.
# 2. Let distil pick the most aggressive safe operating point (fail-safe to full):
distil calibrate \
  --baseline scores/full.json \
  --candidate gate@6=scores/gate6.json:6 \
  --candidate gate@12=scores/gate12.json:12 \
  --margin 0.05 \
  --json calibration_certificate.json
# 3. Deploy with DISTIL_E7_GATE_RECENT set to the selected value (or keep full context if
#    the certificate is fail-safe).
```

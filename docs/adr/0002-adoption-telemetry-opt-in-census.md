# ADR 0002 — Adoption measurement: passive channels + opt-in content-free census

- Status: Accepted
- Date: 2026-07-19
- Deciders: distil maintainers

## Context

We need an adoption picture: how many installs (by version), whether they are
active, and community-wide tokens/$ saved. Today distil has **zero egress to a
distil-controlled host** — the only non-LLM outbound request in the codebase is
the PyPI version check inside `distil onboard` (skippable with `--offline`).
The "federated leaderboard" (`distil/telemetry.py`) signs content-free
aggregates but writes them to a **local file**; nothing is ever sent.

Two public claims are load-bearing and must survive this feature:

- `README.md`: "Measured on **your** traffic, never estimated, **nothing leaves
  your machine**"
- `distil/ledger.py`: community aggregation "is a deliberate **OPT-IN** … this
  module never sends anything"

Industry survey (2026-07): opt-out phone-home is the market norm (Next.js,
Homebrew, gh, Continue, VS Code, .NET) but every one of them has drawn
privacy backlash, and Continue shipped a bug where opt-out still sent. aider
(opt-in + publishes its own event log) and LiteLLM (no telemetry at all) are
the privacy-respecting poles. Headroom — our closest competitor — publishes
fleet-wide savings ("1.4B tokens saved across 249 instances") from an
opt-out-documented beacon. Passive registry channels (pypistats, pepy,
Homebrew analytics, Docker Hub, GitHub traffic) count installs with **zero
client code**.

## Decision

Three layers, in order of preference:

1. **Passive first.** Install/version/geo adoption comes from registry-side
   channels (PyPI stats, pepy per-version, GitHub traffic, Docker pulls),
   snapshotted nightly by CI into a `metrics` branch — no byte of telemetry
   code runs on a user's machine. This is a differentiator, not a compromise:
   *we don't need to phone home to know we're used.*

2. **Census is OPT-IN, content-free by construction, and transparent.** The one
   thing passive channels cannot see — active usage and community savings — is
   a periodic ping (`distil/census.py`) that sends ONLY: anonymous install id
   (random UUID, revocable by deleting a file), distil version, OS/arch/python,
   and the same numeric `SavingsAggregate` shape the federated leaderboard
   already signs (tokens/dollars saved, runs — numbers, never content). It is
   **off until the user says yes** (`distil census on`, or accepting the
   onboard prompt). `DO_NOT_TRACK=1` and `DISTIL_NO_TELEMETRY=1` win over a
   stored opt-in, unconditionally. `distil census show` prints the exact
   payload that would be sent; TELEMETRY.md publishes the full schema.
   Transport is fail-open (1.5s budget, never blocks, never changes exit
   codes) and fires at most once per 24h, from the wrap/proxy exit flush.

3. **Community savings board rides the same consent.** Opted-in censuses
   aggregate server-side into a public JSON the docs site renders — the
   Headroom-style fleet number, earned the opt-in way.

The market-norm alternative (opt-out) was rejected: for a tool whose brand is
"content never leaves your machine," a single opt-out incident is existential,
and the README claim would become false the day the feature ships.

## Consequences

- README/THREAT_MODEL gain one honest qualifier: nothing leaves your machine
  *except an opt-in census you can read before sending* (TELEMETRY.md).
- A distil-controlled ingest endpoint becomes part of the surface
  (`DISTIL_CENSUS_ENDPOINT` override; server code lives in-repo so the
  deployment is reviewable). Until it is deployed, census stays dark —
  consent UX exists, sends fail silently.
- The anonymous install id is the first persisted identifier
  (`~/.distil/install-id`); it is minted only after consent, and `distil
  census off` deletes it.
- Version detail beyond PyPI's own stats (e.g. per-version splits) comes from
  pepy/BigQuery, not from users' machines.

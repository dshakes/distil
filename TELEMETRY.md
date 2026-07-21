# Telemetry

**Default: distil sends nothing.** No usage analytics, no crash reporting, no
phone-home. The only network traffic a default install ever produces is your
own LLM API requests to the upstream you configured, plus one PyPI
version-lookup during `distil onboard` (skippable with `--offline`).

Adoption numbers (installs, versions) come from **registry-side channels** —
PyPI download stats, GitHub traffic — that involve zero code on your machine.

## The opt-in census

One thing can't be measured registry-side: community-wide tokens/$ saved,
because savings are computed locally. For that there is an **opt-in census**:

```sh
distil census show     # print the EXACT payload — nothing is sent
distil census on       # consent (mints a random install id)
distil census off      # revoke — deletes the install id
distil census          # status
```

It is **off until you say yes** — either `distil census on` or answering the
one-time question in `distil onboard` (declining is recorded; you are never
asked again; `--yes` does *not* consent for you).

### Exactly what is sent

At most **one JSON object per 24 hours**, fired from the proxy/wrap shutdown
path, fail-open (a dead endpoint costs ≤1.5s once per day, and errors are
swallowed). The schema is frozen by a test
(`tests/test_census.py::test_payload_schema_frozen`) — widening it requires a
reviewed change to this file and that test:

| field | example | note |
|---|---|---|
| `schema` | `1` | payload version |
| `install_id` | `"3f2a…"` | random UUID — no MAC/hostname/username derivation; deleted by `census off` |
| `version` | `"1.20.2"` | distil version |
| `os` / `arch` / `python` | `"Darwin"` / `"arm64"` / `"3.12.13"` | platform triple |
| `runs` | `412` | ledger run count |
| `tokens_saved` | `183_220` | lifetime baseline − distil input tokens, **calibration-corrected** (the same heuristic→billed factor the proof ledger applies) |
| `dollars_saved` | `12.41` | same, in dollars (0 for unpriced models). The rollup buckets it by `billing`: **metered → real $ saved; subscription → notional API-rate value**, published separately and labeled — included, never mixed |
| `billing` | `"metered"` | `"subscription"` or `"metered"` — nothing else valid |
| `by_model` | `{"claude-opus-4-8": 91_000}` | calibrated tokens saved per model id, **top 5 only**, key length-capped |
| `agents` | `["claude"]` | wrapped agents seen, **allowlist-only** (`claude`/`codex`/`gemini`/`aider`) — anything else becomes `"other"`, so an exotic command line can never leak |
| `surfaces` | `{"wrap": 5100, "proxy": 320}` | request counts per integration door — `wrap` / `proxy` / `gateway` (allowlist-only keys) |
| `shapes` | `{"anthropic": 5000, "openai-chat": 400}` | request counts per API wire format — `anthropic` / `openai-chat` / `openai-responses` / `gemini` — the "are SDKs used?" answer, with no key- or identity-derived data |
| `ts` | `1784500000` | send time |

Never sent, opted in or not: prompt/response content, message digests, file
paths, hostnames, usernames, API keys, model inputs of any kind. The census is
built from `ledger.summary()` totals — the same numbers-only ledger described
in [THREAT_MODEL.md](THREAT_MODEL.md).

### Kill switches (beat a stored opt-in, always)

- `DO_NOT_TRACK=1` — the [console DNT convention](https://consoledonottrack.com)
- `DISTIL_NO_TELEMETRY=1`
- `DISTIL_CENSUS_ENDPOINT=<url>` — point the census at your own collector
  (useful for org-internal fleets)

## Where the aggregate goes — a fully auditable pipeline

Every stage of the pipeline is public code in this repo:

1. **Ingest** ([`packaging/census-worker/`](packaging/census-worker/)): a
   stateless function that validates the schema (2 KB cap, strict allowlist,
   hard numeric ceilings) and forwards to a GitHub `repository_dispatch`. It
   stores nothing — no database, no IP addresses.
2. **Append** ([`.github/workflows/census-ingest.yml`](.github/workflows/census-ingest.yml)):
   CI **re-validates** the payload (defense in depth) and appends it to
   `data/census.jsonl` on the public
   [`metrics` branch](https://github.com/dshakes/distil/tree/metrics) — the
   datastore is a git branch anyone can read.
3. **Rollup** ([`scripts/census_rollup.py`](scripts/census_rollup.py), nightly):
   dedupes to the latest census per install id, drops out-of-ceiling rows, and
   publishes `data/aggregates.json` + shields.io badge JSONs — the numbers
   behind the README badges and the site's live adoption strip.

Passive registry stats (PyPI downloads, GitHub traffic, Docker pulls) join the
same rollup via [`scripts/adoption_snapshot.py`](scripts/adoption_snapshot.py)
— those never involve your machine at all.

## The heartbeat (opt-in, for the near-real-time community counter)

The daily census is the exact archive; the **heartbeat** is what lets the live
community counter tick in near-real-time. It rides the *same* opt-in +
`DO_NOT_TRACK`/`DISTIL_NO_TELEMETRY` gates — if you haven't opted in, none of
this fires.

- **What:** a tiny content-free JSON — `{v:1, install_id, tokens_saved, rate, ts}`
  — numbers and the same anonymous id, nothing else.
- **When:** at most once every **5 minutes**, and **only when your saved-token
  total actually grew** since the last beat. An idle machine sends nothing.
  Sent from the proxy/wrap exit and a lightweight in-session timer, fail-open
  (≤1.5s, swallowed). Opt out of just this with `DISTIL_BEAT_ENDPOINT=` unset
  is not required — `DO_NOT_TRACK=1` disables it like everything else.
- **Where it goes:** an edge key-value store (Upstash Redis) that keeps only the
  *latest per install* (`tokens`, `last-seen`, `rate`) — no history, no IPs. The
  `/v1/live` aggregate sums it on read (exact community total) and reports how
  many installs are **active right now** plus their combined rate. The docs
  page projects the counter forward at that **active-only** rate, bounded, so it
  ticks while people work and **goes static the instant everyone idles** — it
  never invents growth. The number is exact at every pulse.

Not stored anywhere: content, prompts, keys, per-request detail, or history —
the heartbeat overwrites the previous value for your install id.

## The update check (not telemetry, disclosed anyway)

`distil wrap` and `distil proxy` check **pypi.org** for a newer version at
most once per 24h, in a background thread, and print one line if you're
behind. Nothing about you is sent — it is the same public PyPI metadata
lookup `distil onboard` performs. Opt out: `DISTIL_NO_UPDATE_CHECK=1`.

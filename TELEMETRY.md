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
| `tokens_saved` | `183_220` | lifetime baseline − distil input tokens |
| `dollars_saved` | `12.41` | same, in dollars (0 for unpriced models) |
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

## Where the aggregate goes

Opted-in censuses aggregate into public totals — installs by version, active
instances, community tokens/$ saved — published on the
[distil site](https://dshakes.github.io/distil/). The ingest service is in
this repo (reviewable), stores no IP addresses, and keeps only the latest
census per install id.

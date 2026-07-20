# Census ingest worker

The receiving end of the opt-in adoption census (`distil census on`,
[TELEMETRY.md](../../TELEMETRY.md)). A single zero-dependency Vercel function:

```
POST /v1/ping  →  validate (lib/validate.js)  →  repository_dispatch "census"
                                                  → .github/workflows/census-ingest.yml
                                                  → metrics branch data/census.jsonl
                                                  → nightly rollup → aggregates + badges
```

The worker **stores nothing** — no database, no IPs. GitHub's metrics branch
is the datastore, so ingest, history, and rollup are all publicly auditable.

## Deploy (one time, ~3 minutes)

1. Create a **fine-grained PAT** (github.com → Settings → Developer settings)
   scoped to the **single repo** `dshakes/distil` with **Contents: Read and
   write** only. That is the entire blast radius of a leaked token.
2. ```sh
   cd packaging/census-worker
   vercel deploy --prod
   vercel env add GITHUB_DISPATCH_TOKEN   # paste the PAT (Production)
   vercel deploy --prod                   # redeploy with the env var
   ```
3. Point the DNS name at it: CNAME `census.distil.dev` → the Vercel
   deployment (or skip DNS and ship a release that sets
   `DEFAULT_ENDPOINT` in `distil/census.py` to the `*.vercel.app` URL).
4. Recommended: enable Vercel Firewall rate limiting on `/v1/ping`
   (the in-function bucket is best-effort per instance only).

Until this is deployed, opted-in clients fail open — one swallowed 1.5s
attempt per day, nothing else.

## Test

```sh
node --test packaging/census-worker/test/validate.test.mjs
```

## Verify after deploy

```sh
distil census show                                     # the payload
distil census show | curl -sS -X POST --json @- \
  https://census.distil.dev/v1/ping -o /dev/null -w '%{http_code}\n'   # → 202
```

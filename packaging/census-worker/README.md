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
3. The client default (`distil/census.py`) already points at the live
   production domain `https://distil-census.vercel.app/v1/ping`. To move to a
   custom domain later (e.g. `census.distil.dev`), CNAME it to the deployment
   and bump `DEFAULT_ENDPOINT` — old clients keep working via the .vercel.app
   domain either way.
4. Recommended: enable Vercel Firewall rate limiting on `/v1/ping`
   (the in-function bucket is best-effort per instance only).
   Note: the team-scoped deployment aliases sit behind Vercel SSO (401);
   only the public production domain above serves anonymous pings — that
   asymmetry is fine and expected.

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
  https://distil-census.vercel.app/v1/ping -o /dev/null -w '%{http_code}\n'   # → 202
```

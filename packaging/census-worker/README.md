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

It also serves the **near-real-time community counter** (optional — needs an
Upstash Redis store):

```
POST /v1/beat  →  validate (lib/beat_validate.js)  →  Upstash HSET latest-per-install
GET  /v1/live  →  Upstash HGETALL → { total, active, rate, installs }  (CORS-open)
```

`/v1/beat` and `/v1/live` degrade gracefully when Upstash isn't configured
(`/v1/beat` → 503, `/v1/live` → `{available:false}`), and the docs page falls
back to the exact daily-census total. So the census works with zero extra setup;
the live counter lights up once you add Upstash.

## Enable the live counter (optional, ~3 minutes)

1. Vercel dashboard → your `distil-census` project → **Storage** → **Marketplace**
   → add **Upstash Redis** (free tier is plenty). Vercel injects
   `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` as env vars.
2. Redeploy (`vercel deploy --prod` or trigger from the dashboard) so the
   function picks up the vars.
3. Verify: `curl https://distil-census.vercel.app/v1/live` → `{"available":true,…}`.
   Opted-in clients (v1.24+) then beat automatically as they use distil, and the
   adoption page's odometer ticks in near-real-time.

Blast radius: the Upstash store holds only `{install_id → tokens, ts, rate}` —
anonymous ids and numbers, no content, no IPs, overwritten each beat.

## Deploy (one time, ~3 minutes)

1. Create a **fine-grained PAT** (github.com → Settings → Developer settings)
   scoped to the **single repo** `dshakes/distil` with **Contents: Read and
   write** — that is the entire blast radius of a leaked token. Tip: also
   grant **Administration: Read** and save the same PAT as the repo Actions
   secret `TRAFFIC_TOKEN`, which lights up the clones/views columns in the
   adoption snapshot (the built-in Actions token can't read the traffic API).
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

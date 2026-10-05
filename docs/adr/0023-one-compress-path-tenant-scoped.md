# 0023 — One compress-or-forward path for proxy and gateway, scoped per tenant

- **Status:** accepted
- **Date:** 2026-10-04
- **Relates to:** `distil/serve_core.py` (`compress_or_forward`, `TenantHandles`), `distil/proxy.py`, `distil/gateway.py`, ADR 0011 (prefix replay), ADR 0014 (cold-point), ADR 0016 (drift guard scope), `tests/test_serve_core.py`

## Context

The threaded proxy (`distil wrap`, `distil proxy`) and the multi-tenant gateway each carried
their own copy of the request path: parse, pick the adapter by shape, compress, fail open,
count, inject `distil_expand`, replay the prefix. The copies had drifted far enough that a
gateway tenant got a different product from a local user:

- cold-point recompression (ADR 0014) and cache-delta coding existed only in the proxy;
- the gateway had no expand loop, so it was forced Tier-0-only (`verbatim = True`) — no
  recoverable digest at all on PAYG, not even as an opt-in;
- the gateway's token counter was an older copy that missed image and thinking blocks, so
  tenant savings and daily quotas were computed on a smaller baseline than the proxy's;
- the gateway re-encoded every body with `json.dumps` defaults, while prefix replay models
  the compact encoding the proxy sends — so replay on the gateway reasoned about bytes it
  never put on the wire, and an unmodified body still rewrote the cached prefix;
- the gateway did not request identity encoding, which an expand loop needs to read the
  response.

## Decision

1. **One function.** `serve_core.compress_or_forward(body, path, *, count, verbatim,
   expand, mode, keep, session_delta, cold, scope, held, shape_output, replay_scope,
   on_compressed, admit) -> Outbound | None` owns Tier-1/Tier-0 compression, the Chat
   Completions / Messages / Responses / Gemini dispatch, cache-delta, cold-point planning,
   expand-tool injection, output shaping and prefix replay, each fail-open. The proxy and
   the gateway both call it. The response side they share — buffering a non-Anthropic
   stream for the expand loop, loop dispatch by shape, SSE re-emission, the streaming
   opener — is in the same module. Everything a caller does differently is a parameter.
2. **The gateway can run the proxy's compression — as an operator opt-in.** The default
   gateway stays Tier-0 only, with no tool injection, as before this ADR. With
   `distil gateway --digest` a PAYG gateway runs the recoverable digest with
   `distil_expand` injected and answered (buffered loops for every shape, the streaming
   splice for Anthropic Messages), cold-point on by default there, `--session-delta` and
   `--no-cold-point` available. Subscription / `--lossless-only` / `--verbatim` stay
   Tier-0-only with no tool injection whatever the flag, exactly as on the proxy.
   **The gateway digest is unguarded: no per-mode certification on the gateway yet.**
   The per-mode certification hold (ADR 0022), shadow and drift guard that switch a
   local proxy's digest off while it is over its decision-change budget are proxy-only
   (next section), and on the maintainer's live data digest is currently over that
   budget. A default-on gateway digest would contradict ADR 0022, so it is opt-in and the
   gateway prints an `UNGUARDED` line at start when it is on.
3. **Per-tenant scope for every piece of cross-request state.** The gateway passes
   `scope = tenant + "\0"`, which keys cold-point lineages and expanded-handle
   exclusions, cache-delta sessions, and prefix replay (already tenant-scoped). The proxy
   passes its account hash, as before.
4. **A tenant restore boundary.** Digest originals live in one content-addressed restore
   store per gateway, so a bare store answers any 8-hex handle a model names — including
   one a tenant typed into its own history. `TenantHandles` records which handles were
   issued to which tenant; the gateway hands its expand loops a view that resolves only
   those and returns the miss placeholder for anything else.

## Kept different, deliberately

- **No learned state on the gateway.** The proxy passes its learned keep predicate (outcome
  and expand flywheel) and records learning/retention signals; the gateway passes none.
  Those models are machine-local evidence about one user's workload; applying them across
  tenants is the same mistake ADR 0016 avoided for the drift guard.
- **No drift guard, shadow, per-mode certification hold, A/B holdout, receipts or session
  ledger on the gateway** — all unchanged from before; this ADR unifies the compression
  path, not the observability. That is why the digest is `--digest`, not the default.
- **No output shaping on the gateway** (it never had it).
- **`aproxy.py` is not routed through the shared path.** It is verbatim-only by design
  (no expand loop) and its headers and accounting are its own.

## Consequences

- With `--digest`, the gateway's forwarded bytes for a request sequence are identical to
  the proxy's (`test_cold_point_parity`, `test_cache_delta_parity`), so a fix to
  compression lands in both servers or in neither. Without it (the default) the gateway
  forwards Tier-0 output, as it did before this ADR (`test_payg_never_emits_an_unrecoverable_stub`).
- Lifting the opt-in needs per-tenant certification evidence on the gateway (a tenant-
  scoped shadow and hold); ADR 0016's reason for not sharing the machine-global one stands.
- Gateway responses carry the proxy's `x-distil-mode`, `x-distil-compressible-tokens`,
  `x-distil-cold*` and `x-distil-cache-*` headers.
- Tenant accounting and quotas use the proxy's counter, which counts images and thinking:
  baselines go up for tenants that send them.
- Isolation is exactly as strong as tenant identity. With the default (credential hash) or
  dsk-/OIDC keys it holds; under the operator opt-in `--trust-tenant-header` a client that
  claims another tenant's label gets that tenant's grants, as it already got its accounting.
- The grant table is in memory and LRU-bounded (65,536 handles). After a gateway restart
  an older stub's expand returns the miss placeholder — fail-safe, but a regression in
  recoverability against the proxy, which falls back to the on-disk store. Persist grants
  per hashed tenant if that becomes a problem.

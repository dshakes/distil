# Threat model

What distil trusts, what it protects, and what it deliberately does not defend
against. Scope: the local proxy (`distil proxy` / `wrap`), the async proxy, the
multi-tenant gateway, and the MCP server. Audited 2026-07 (serving-path
security review); update this file when a trust boundary moves.

## Assets

1. **Customer API keys** — pass through every request.
2. **Conversation content** — full prompts and responses transit the process;
   originals of digested blocks are retained for reversibility.
3. **Usage metadata** — per-tenant token/dollar volumes (gateway), savings
   ledger, shadow equivalence samples.

## Trust boundaries

| Component | Trusts | Does NOT trust |
|---|---|---|
| proxy / wrap | the local user (same UID), the operator-configured upstream URL | request bodies (parsed defensively; compression failure fails open to the original), upstream response content |
| gateway | the operator (flags/env), upstream URL | callers: tenant identity derives from the credential hash; `x-distil-tenant` honored only under `--trust-tenant-header`; `/distil/*` requires `--admin-token` off loopback |
| MCP server | the local MCP client (stdio, same UID) | tool arguments (validated) |

## Guarantees (enforced in code, tested)

- **Keys are never persisted or logged.** Forwarded only to the configured
  upstream. Ledger/shadow/learn/telemetry files carry counts, hashes, and
  booleans — never content, never credentials.
- **No auto-redirect on the forward path.** A 3xx from the upstream is relayed
  to the client, never followed — credentials are never re-sent to a host the
  operator didn't configure.
- **TLS verification is stock** (urllib/aiohttp defaults). No verify-off
  escape hatch exists.
- **SSRF/path guards** on every forward (`httpguard.safe_forward_path`):
  userinfo, scheme injection, traversal, and control characters are rejected;
  request bodies are size-capped.
- **Fail open to fidelity, closed on safety.** A compression error forwards
  the original request unchanged; a guard rejection returns an error — content
  is never silently altered by a failure path.
- **Bounded state.** Restore stores and session maps are capped; the MCP
  handle store is FIFO-bounded, chmod 0600, and encrypted at rest (DSTL1
  construction — see the *Restore store* section below).

## Content at rest

### Restore store (encrypted since 1.20.0)

Digest originals (tool outputs and other compressible agent content) are
persisted to two locations for cross-process and cross-restart reversibility:

- `~/.distil/restore/<handle>` — one file per 8-hex handle, written by the
  proxy adapter on every digest. FIFO-capped at 500 files, 14-day TTL.
- `~/.distil/mcp_store.json` — the MCP server's handle→original map,
  FIFO-capped at 512 entries.

Both are **encrypted at rest** using a per-install 256-bit master key at
`~/.distil/restore.key` (chmod 0600, created on first use). The construction
is HMAC-SHA256-CTR + encrypt-then-MAC (see `distil/atrest.py` module docstring
for the full specification). All files are also chmod 0600.

**What this protects against:**
- Cloud backup / sync leakage: iCloud Drive, Dropbox, Time Machine, and
  similar services that snapshot `~` will copy ciphertexts, not content.
- Cross-user filesystem reads on multi-user NAS or world-readable backup
  snapshots: the data files are unreadable without `restore.key`.

**What this does NOT protect against:**
- A local attacker with the same UID. They can read both `restore.key` and
  the data files and trivially decrypt. distil is not a privilege boundary.
- Physical access to the running process (the master key is in memory).
- An operator running a truly ephemeral zero-data-retention deployment should
  point `DISTIL_HOME` at a ramdisk (`tmpfs`) — encryption adds no value there
  and can be disabled with `DISTIL_NO_ENCRYPT_AT_REST=1`.

**Upgrade / downgrade:** `load_restore` and `_load_store` detect the `DSTL1`
magic header and fall back to plaintext JSON / UTF-8 for legacy files written
before 1.20.0. Authentication failure (tampered or wrong-key file) is treated
as missing — the same fail-open behaviour as an expired TTL — and logged at
debug. Legacy files are rewritten encrypted on the next `record_restore` call
for the same handle.

**Key rotation:** deleting `restore.key` causes a new key to be generated;
all existing encrypted files become unreadable (treated as missing). This is
the correct behaviour for a fresh credential rotation — handles that can no
longer expand degrade exactly like expired ones.

### Other persisted state

- The savings ledger / shadow ledger / learn stats: numbers only, no content.

## Out of scope (explicitly not defended)

- **A hostile local user on the same machine.** The proxies bind loopback by
  default and trust the local UID; distil is not a privilege boundary.
- **A malicious operator-configured upstream.** Whoever controls `--upstream`
  sees the traffic — that is the point of a proxy. Point it only at providers
  you trust.
- **Malicious model output.** Distil relays responses byte-faithfully; agent-
  side prompt-injection defense belongs to the agent harness.
- **Network eavesdropping between distil and the client.** Local loopback
  traffic is unencrypted; bind non-loopback only behind your own TLS/network
  controls (and set `--admin-token`).

## Shared-gateway deployment (distil gateway with --require-keys)

### What key auth does

When at least one `dsk-` key has been issued (or `--require-keys` is set),
every inbound request must carry the key as `Authorization: Bearer dsk-…` or
`x-distil-key: dsk-…`.  Missing or revoked keys → 401.  Rate limits
(`--tenant-rpm`, `--tenant-daily-tokens`) → 429 with `Retry-After: 60`.

Keys are stored **hashed** (sha256) in `~/.distil/gateway_keys.json`
(chmod 0600).  The raw token is shown once at `distil gateway keys issue` time
and never persisted.  The gateway strips the distil key before forwarding so
the provider never sees it; the provider credential (`x-api-key` /
`Authorization: Bearer <provider-key>`) passes through unchanged.

### What key auth does NOT protect

- **Tenants share one process.**  There is no memory isolation between tenants;
  a bug in the gateway could allow one tenant's in-flight state to leak to
  another.  Run separate gateway processes for tenants with strict isolation
  requirements.

- **Restore store is per-gateway, not per-tenant.**  `compress_messages` writes
  digest originals to `~/.distil/restore/` (a directory shared across all
  tenants of a single gateway process).  The gateway has no expand HTTP
  endpoint, so no tenant can retrieve another tenant's original content through
  the gateway API.  However, a local user with access to `~/.distil/restore/`
  (which is chmod 0600, owner-only) could expand any handle via the `distil
  expand` CLI or MCP server.  In a multi-tenant shared deployment, run distil
  as a dedicated OS user so that directory is not readable by tenant processes.

- **Provider credentials pass through.**  Whoever controls `--upstream` sees
  every request including all provider credentials.  Point the gateway only at
  providers you trust.

- **No TLS between gateway and client.**  Bind non-loopback only behind your
  own TLS/network controls.

- **In-memory rate counters reset on restart.**  Daily token quotas are tracked
  in memory only; a gateway restart resets them.  This is a quota-gaming
  ceiling, not a billing guarantee.

## Residual risks (known, accepted, ranked)

1. Gateway per-tenant metadata is visible to anyone holding the admin token —
   scope the token like a credential.
2. Restore store is shared across all tenants of a single gateway process; see
   the shared-gateway section above.
3. `DeltaSession` keys on the first message hash; agents with identical fixed
   first turns share a session's content-free prefix stats (cosmetic only —
   verified no content crosses sessions).
4. Upstream error strings are relayed (truncated to 200 chars) to the local
   client for debuggability.

## Reporting

Security reports: open a GitHub security advisory (preferred) or a private
issue. Do not post exploits in public issues.

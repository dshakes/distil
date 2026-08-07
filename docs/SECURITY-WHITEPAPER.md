# Distil — security & data handling

For security reviewers, platform teams, and procurement. Written to be answerable
against evidence rather than assertion: every claim below names the file, command,
or test that backs it, so you can verify rather than trust.

Scope: distil `1.41.x`, the Apache-2.0 open-source distribution.
Companion documents: [`THREAT_MODEL.md`](../THREAT_MODEL.md) (adversary model and
explicit non-goals), [`TELEMETRY.md`](../TELEMETRY.md) (frozen telemetry schema).

---

## 1. What distil is, in one paragraph

Distil sits between your agent and your model provider and reduces the number of
input tokens sent, either as a local proxy, a shared gateway, or an in-process
library call. It does not store your prompts in a service, does not call any
distil-operated endpoint, and requires no account. The compression is
**reversible**: elided spans are replaced by a short handle, and the original
bytes stay on the machine that compressed them.

## 2. Data flow and residency

**Nothing leaves your infrastructure.** Distil forwards to exactly one configured
upstream — your model provider — and to nothing else.

| Data | Where it lives | Leaves the machine? |
|---|---|---|
| Prompt / message content | In flight only; forwarded to your configured upstream | Only to your provider |
| Digest originals (restore store) | `${DISTIL_HOME:-~/.distil}/restore/`, encrypted, `0600` | No |
| Savings ledger | `${DISTIL_HOME}/savings.jsonl` — token **counts** only | No |
| Receipts (audit chain) | `${DISTIL_HOME}/receipts.jsonl` — counts, mode, handles | No |
| Community census | Opt-in only, numbers only, off by default | Only if explicitly enabled |

The proxy binds `127.0.0.1` by default and forwards only to the single configured
upstream, so it cannot be used as an SSRF pivot. Egress in the shipped Helm chart
is restricted to DNS and 443 by NetworkPolicy
(`packaging/helm/distil-gateway/templates/networkpolicy.yaml`).

**Verify:** `distil proxy --help` (bind address), and read
[`TELEMETRY.md`](../TELEMETRY.md) for the frozen census schema. Preview exactly
what a census payload would contain, before consenting, with `distil census show`.

## 3. Encryption at rest

Digest originals are encrypted with **HMAC-SHA256-CTR, encrypt-then-MAC** (`DSTL1`
header), with a 32-byte key at `${DISTIL_HOME}/restore.key`, `chmod 0600`, created
on first use (`distil/atrest.py`).

This protects against backup/sync leakage and cross-user reads on a shared
filesystem. It explicitly does **not** protect against a same-UID attacker who can
read both the data and the key — that is stated as out of scope in
[`THREAT_MODEL.md`](../THREAT_MODEL.md) rather than papered over.

Handles age out after `DISTIL_RESTORE_TTL_DAYS` (default 14). Opt out of
encryption with `DISTIL_NO_ENCRYPT_AT_REST=1` (not recommended).

## 4. Secrets

- Request bodies and credentials are **never logged**.
- Provider credentials are forwarded, not parsed, stored, or inspected.
- The gateway's own admin token is passed by environment, not argv, so it does not
  appear in `ps` or a pod spec dump
  (`packaging/helm/distil-gateway/templates/deployment.yaml`).
- The adversarial gate drives the compressor against secret-looking inputs and
  asserts they are never emitted in telemetry.

**Verify:** `distil validate` — the adversarial suite (huge, unicode, nested,
malformed, marker-injection, secret-looking inputs), asserting reversibility,
fail-open, and content-free telemetry on every one.

## 5. Multi-tenancy (gateway)

The gateway issues `dsk-` bearer keys and derives the tenant **from the
credential**, never from a client-supplied header. `--trust-tenant-header` exists
for trusted-network accounting and is off by default; with it on, any client can
bill another tenant's quota, which is why it is opt-in.

Per-tenant RPM and daily-token quotas are enforced in-process. Rejections are
counted and exported as `distil_requests_rejected_total`, labelled by tenant —
quota enforcement that cannot be observed is indistinguishable from an outage from
the client's side.

**Known limitation, stated plainly:** tenants share one process. There is no
memory isolation between tenants, and the restore store is per-gateway rather than
per-tenant. For hard isolation, run one gateway per trust boundary. This is
surfaced in `distil gateway --help`, not buried here.

## 6. Metrics and admin endpoints

`/distil/metrics` is labelled by tenant, so it sits behind the same admin gate as
`/distil/stats`: open on loopback, and on any non-loopback bind it **requires**
`--admin-token` and refuses to start without one.

This is deliberate — an unauthenticated, tenant-labelled `/metrics` is precisely
the [LiteLLM disclosure](https://github.com/BerriAI/litellm/issues/13644) class of
bug. It is tested directly: 403 unauthenticated, 401 on a wrong token, plus
label-injection and no-secrets-in-exposition tests (`tests/test_gateway.py`).

## 7. Audit trail

Every request writes a **hash-chained receipt**: counts, mode, handles issued, and
whether they resolved. Never content. Each receipt carries the previous receipt's
hash, so removing or editing an entry breaks the chain.

```bash
distil receipts            # verify the chain; non-zero exit if broken
distil receipts --export   # newline-delimited JSON for your SIEM
```

The verification needs nothing but the file — no server, no key escrow. Retention
is yours to set; distil never prunes receipts.

## 8. Supply chain

- **PEP 740 Sigstore attestations** via PyPI Trusted Publishing on every release.
  The release job **fails** if PyPI does not report an attestation bundle for the
  version it just published, so this claim cannot silently drift.
- **CycloneDX SBOM** attached to every GitHub release.
- **OpenSSF Scorecard** runs weekly on `main`.
- **Zero runtime dependencies** in the core — the entire attack surface is the
  Python standard library plus distil's own code. Optional extras (`[otel]`,
  `[live]`) are opt-in.

**Verify independently:**

```bash
curl -s https://pypi.org/integrity/distil-llm/<version>/<filename>/provenance
uvx pypi-attestations verify pypi \
  --repository https://github.com/dshakes/distil \
  pypi:distil_llm-<version>-py3-none-any.whl
```

(Use the *integrity* API — `/pypi/<pkg>/<ver>/json` carries no attestations field
and reading it there reports a false negative.)

## 9. Availability and failure behaviour

Distil is in the request path, so its failure modes are your failure modes. The
governing rule is **fail open**: if compression raises, the original request is
forwarded uncompressed rather than failing. `DISTIL_DEBUG=1` surfaces everything
the fail-open path swallows.

`GET /distil/health` is an unauthenticated liveness probe on every entry point and
never touches the billed upstream. Gateway accounting checkpoints to disk every
30s, so a hard crash loses at most 30s of counters, not the ledger.

**One failure mode worth your attention.** A base URL pinned in an agent's own
settings file outranks the environment `distil wrap` sets. If that pinned port
stops answering, every session fails with a connection error naming the *provider*.
`distil wrap` now refuses to start into that configuration and names the fix
(`distil/precedence.py`, `tests/test_precedence.py`). Reviewers should note this
as the class of risk any in-path proxy carries.

## 10. Identity and access

**Present:** bearer-key authentication with per-key tenant, quota, and revocation
(`distil gateway keys issue|list|revoke`).

**Not present, and honestly so:** SAML, OIDC, SCIM, and role-based access control
are **not implemented**. If your review requires SSO for the gateway's admin
surface today, distil does not meet that bar. The current recommendation is to
place the gateway behind your existing authenticating proxy or service mesh and
treat the admin token as an infrastructure secret.

## 11. Compliance status

Stated plainly so nobody discovers it late:

| | Status |
|---|---|
| SOC 2 Type II | **Not held.** No audit in progress. |
| ISO 27001 | Not held. |
| DPA / MSA / SLA | Not offered — Apache-2.0 open source, no commercial entity behind it today. |
| Support | Best-effort via GitHub issues. No response-time commitment. |
| Data processing | Distil operates no service and processes no customer data off-machine, so there is no processor relationship to paper. |

For most security reviews the last row is the important one: because distil runs
entirely inside your boundary and calls no distil-operated endpoint, it is
typically assessed as a **software dependency** rather than a subprocessor.

## 12. Reporting a vulnerability

See [`SECURITY.md`](../SECURITY.md). Please do not open a public issue for a
suspected vulnerability.

# Enterprise deployment & commercial status

What distil can support today, what it cannot, and what closing the gap requires.
Written to be usable in a procurement conversation without a sales call —
including the parts that will disqualify it for some buyers.

Companion: [`SECURITY-WHITEPAPER.md`](SECURITY-WHITEPAPER.md) (security review),
[`GA_READINESS.md`](GA_READINESS.md) (engineering status).

---

## 1. Deployment topologies

| Topology | When | How |
|---|---|---|
| **Local sidecar** | Individual developers | `distil wrap -- <agent>`, or `distil default --always-on` |
| **Container sidecar** | One proxy per app pod | `ghcr.io/dshakes/distil` (multi-arch, amd64 + arm64) |
| **Shared gateway** | Team or org, per-tenant accounting and quotas | Helm chart below |

### Kubernetes

```bash
helm install distil packaging/helm/distil-gateway \
  --set adminToken.existingSecret=distil-admin \
  --set gateway.upstream=https://api.anthropic.com \
  --set providerSecret.existingSecret=anthropic \
  --set providerSecret.keys.ANTHROPIC_API_KEY=api-key \
  --set serviceMonitor.enabled=true \
  --set prometheusRule.enabled=true
```

Defaults are chosen to be safe rather than convenient, and a test asserts they stay
that way (`tests/test_packaging_assets.py`):

- `requireKeys: true` — a gateway reachable on a cluster network is never open
- `trustTenantHeader: false` — otherwise any client can bill another tenant's quota
- non-root, read-only root filesystem, all capabilities dropped, seccomp
  `RuntimeDefault`
- `maxUnavailable: 0` on rollout — the gateway is in the request path
- PodDisruptionBudget and topology spread across nodes
- optional NetworkPolicy restricting egress to DNS and 443 only

**Read `persistence.enabled` before going to production.** With it off, the restore
store is an `emptyDir`, so digest handles do **not** survive pod replacement. That
is fine for lossless/verbatim modes, and wrong for `--expand`, where the agent may
try to expand a handle after a restart.

## 2. Identity and access

| | Status |
|---|---|
| Issued bearer keys (`dsk-`), per-key tenant / quota / revocation | ✅ |
| OIDC bearer tokens (HS256; RS256 via the `[oidc]` extra) | ✅ |
| Roles: `viewer` < `operator` < `admin` | ✅ |
| SAML | ❌ Not implemented |
| SCIM provisioning | ❌ Not implemented |
| Per-tenant memory isolation within one gateway process | ❌ By design — run one gateway per trust boundary |

```bash
export DISTIL_OIDC_ISSUER=https://login.example.com
export DISTIL_OIDC_AUDIENCE=distil-gateway
export DISTIL_OIDC_PUBLIC_KEY="$(cat idp-public.pem)"   # RS256, needs [oidc]
export DISTIL_OIDC_ROLE_CLAIM=roles                     # scalar or array
export DISTIL_OIDC_TENANT_CLAIM=org_id
```

OIDC is **additive**: `dsk-` keys keep working, and with `DISTIL_OIDC_ISSUER` unset
a JWT does not authenticate at all. Enabling it cannot lock out a running
deployment, and leaving it off cannot silently open one.

## 3. Observability

- **Prometheus** — `GET /distil/metrics`, labelled by tenant, behind the admin
  gate. Alert rules ship in the chart; a Grafana dashboard ships in
  [`packaging/grafana`](../packaging/grafana/distil-gateway-dashboard.json).
- **OpenTelemetry** — GenAI semantic-convention spans plus distil's own
  attributes, via the `[otel]` extra.
- **Audit** — hash-chained receipts, `distil receipts` to verify, `--export` for a
  SIEM.

One alert deserves explanation before it pages someone: `DistilCompressionIneffective`
fires at under 2% savings over six hours. That is often **not** a fault — traffic
with large stable prefixes and small tool outputs genuinely has little to fold.
It is alertable because it is indistinguishable from a misconfigured mode, and
that distinction needs a human.

## 3b. Networks that inspect TLS

If your network re-signs outbound TLS with an internal CA — Zscaler, Netskope, a
corporate MITM appliance — point distil at the CA bundle:

```bash
export REQUESTS_CA_BUNDLE=/path/to/corp-ca.pem   # or SSL_CERT_FILE, CURL_CA_BUNDLE,
                                                 # or DISTIL_CA_BUNDLE (checked first)
distil doctor            # "TLS trust" names the bundle actually in effect
```

distil honours the variables your environment has almost certainly already
exported for `curl` and `requests`, so on most managed machines this is already
configured and nothing needs doing.

**Why it is needed at all, and why it looks like a distil bug.** `curl` and your
browser read the OS trust store; Python does not. So the corporate root that makes
everything else on the machine work is invisible to distil, and the *only* symptom
is distil failing to reach the provider while every other tool succeeds. The 502 it
returns now carries the explanation and the fix rather than a raw OpenSSL string.

A bundle path that does not exist is **ignored**, not fatal — a stale export in
someone's shell profile must not stop the proxy from starting — and `distil doctor`
warns when it finds one, because silently ignored is the worst outcome for a user
who believes they configured it.

There is no option to disable certificate verification, deliberately. distil sees
every prompt on the machine; "it works with verification off" is how that becomes
trusting anything on the wire.

## 4. Commercial status — read this before planning around distil

| | Status |
|---|---|
| License | Apache-2.0 |
| Commercial entity | **None today.** No company to contract with. |
| Paid support / SLA | **Not offered.** |
| DPA / MSA | **Not offered** — and see below on whether you need one. |
| SOC 2 Type II | **Not held.** No audit in progress. |
| Support channel | GitHub issues, best effort |
| Roadmap commitments | None contractual |

**Whether you need a DPA at all.** Distil operates no hosted service. It runs
inside your boundary, calls no distil-operated endpoint, and requires no account.
Most reviews therefore classify it as a **software dependency** rather than a
subprocessor, which removes the usual data-processing paperwork. Confirm with your
own counsel — but the technical fact underneath is verifiable: the proxy forwards
to exactly one configured upstream and nothing else.

**What this means practically.** If your procurement requires a signed SLA, a SOC 2
report, or a vendor to escalate to at 3am, distil does not meet that bar today, and
no amount of engineering in this repository changes it — those are company
artifacts, not code. The honest options are:

1. Adopt it as an open-source dependency under your own operational ownership
   (the intended path today).
2. Sponsor the gap: a design-partner arrangement that funds SOC 2 and a support
   commitment. There is no program to point you at yet; open an issue.

## 5. What would close the remaining gaps

Ordered by what actually blocks deals, not by effort:

| Gap | What closing it requires | Code or company? |
|---|---|---|
| Support SLA | A named owner and a response-time commitment | Company |
| SOC 2 Type II | 3–6 months of evidence collection plus an auditor | Company |
| DPA / MSA | A legal entity | Company |
| SAML | Implementation on top of the existing role model | Code |
| SCIM | Implementation | Code |
| Per-tenant isolation | Process-per-tenant, or a rearchitecture | Code |
| Multi-region / HA guidance | Load testing and a documented envelope | Code |

The identity work already done (`distil/authz.py`) was the load-bearing part: roles,
verification, and tenant mapping exist, so SAML would be a second credential type
against the same model rather than a new authorization system.

## 6. Evaluating distil in an enterprise

A defensible evaluation, in order:

```bash
# 1. Prove the quality claim on OUR corpus, offline, no key, ~10s
uvx --from distil-llm distil bench

# 2. Prove it on YOUR traffic — decision-equivalence on live requests
distil wrap --shadow 0.1 -- <your agent>
distil shadow-stats

# 3. Certify YOUR risk budget
distil ingest --input prod.jsonl --out ./mycorpus
distil conformal --corpus ./mycorpus --alpha 0.05 --delta 0.05

# 4. Verify the supply chain independently
uvx pypi-attestations verify pypi \
  --repository https://github.com/dshakes/distil \
  pypi:distil_llm-<version>-py3-none-any.whl
```

Step 2 is the one that matters and the one most evaluations skip. Distil's central
claim is not "it compresses" — every competitor compresses — it is "your agent
still makes the same decisions". `--shadow` measures that on **your** workload, and
a result there outranks anything in this repository, **including a bad one**.

# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a suspected vulnerability.**

Report privately through GitHub's [private vulnerability
reporting](https://github.com/dshakes/distil/security/advisories/new) on this
repository. That channel is monitored and keeps the report non-public until a fix
is available.

Include what you need to make it actionable:

- the version (`distil --version`) and how it was installed
- what an attacker gains, and what access they need to start
- a reproduction — ideally a failing command or a short script
- whether you have published or plan to publish anything about it

### What to expect

This is an open-source project maintained without a commercial support
organisation behind it, so the honest commitment is effort, not a contractual
clock:

| | |
|---|---|
| Acknowledgement | Best effort, typically within a few days |
| Assessment and severity | After acknowledgement, shared with you |
| Fix and release | Prioritised over feature work; release cadence is fast |
| Credit | Offered in the advisory and CHANGELOG unless you decline |

If you need a guaranteed response time, distil does not offer one today. See
[`docs/SECURITY-WHITEPAPER.md`](docs/SECURITY-WHITEPAPER.md) § 11.

## Supported versions

Fixes land on the latest release. There is no long-term-support branch, and
backports to older minors are not provided — upgrading is the supported path, and
releases are frequent and small for exactly that reason.

## Scope

Distil runs entirely inside your own boundary: it operates no hosted service, and
compression is local. Reports in scope include, for example:

- content leaking out of the machine (telemetry, logs, the census payload)
- the proxy or gateway being usable as an SSRF pivot beyond its single configured
  upstream
- authentication or tenant-isolation failures in the gateway
- a digest that is not byte-reversible, or a handle resolving to another block's
  content
- secrets appearing in logs, metrics, receipts, or error output
- supply-chain issues in the published artifacts

### Explicitly out of scope

These are documented design boundaries, not oversights — see
[`THREAT_MODEL.md`](THREAT_MODEL.md):

- **A same-UID local attacker.** Encryption at rest protects against backup/sync
  leakage and cross-user reads on shared filesystems. Someone who can already read
  both `${DISTIL_HOME}/restore/` and `restore.key` as your user is out of scope.
- **Cross-tenant memory isolation within one gateway process.** Tenants share a
  process by design; run one gateway per trust boundary where that matters. This
  is stated in `distil gateway --help`.
- **Compression quality.** A digest that loses information you wanted is a bug,
  and an important one — but file it as a normal issue with a reproduction, not as
  a vulnerability.

## Hardening this project applies to itself

- Zero runtime dependencies in the core — the attack surface is the Python
  standard library plus this repository.
- Releases carry [PEP 740](https://peps.python.org/pep-0740/) Sigstore
  attestations via PyPI Trusted Publishing; the release job fails if PyPI does not
  report an attestation bundle for the version it just published.
- A CycloneDX SBOM is attached to every GitHub release.
- [OpenSSF Scorecard](https://github.com/ossf/scorecard) runs weekly on `main`.
- CI runs an adversarial gate (`distil validate`) that drives the compressor
  against hostile inputs — including secret-looking ones — and asserts
  reversibility, fail-open behaviour, and content-free telemetry.

Verify a published artifact yourself:

```bash
uvx pypi-attestations verify pypi \
  --repository https://github.com/dshakes/distil \
  pypi:distil_llm-<version>-py3-none-any.whl
```

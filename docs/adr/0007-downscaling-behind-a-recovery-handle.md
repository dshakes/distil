# 0007 — Image downscaling, behind a recovery handle

- **Status:** accepted
- **Date:** 2026-08-10
- **Amends:** ADR 0003 (content-type coverage and the certificate boundary)

## Context

`distil/compress/vision.py` states the position this ADR revisits:

> The state of the art here is **downscaling** or dropping the provider's detail
> level. That is lossy by construction: the model sees a different image and nobody
> can say whether the answer changed. distil's contract forbids it.

That reasoning is sound on its own terms, and the module built the alternative:
byte-identical duplicate elision, certified, and now enabled by default.

It leaves the original gap open. A first-occurrence screenshot is not a duplicate,
so elision never touches it, and a 2400x1200 UI capture costs its full ~1,600 input
tokens on every turn it stays in context. For an agent that screenshots, that is the
dominant remaining cost and elision cannot reach it.

## Decision

Downscaling is permitted **only** in the form that carries a recovery handle:

1. The original `source` object goes into the `RestoreStore` before the image is
   altered, so `distil_expand` returns the untouched bytes.
2. The downscaled image is emitted **as a pair** with a text note stating that it was
   downscaled, the dimensions before and after, and the handle. A downscaled image
   emitted alone is silent loss and is not permitted.
3. It ships **disabled**, behind its own certificate (`--strategy vision-downscale`),
   and **no certificate is bundled** — unlike `vision`, whose shipped certificate
   covers a provably-identical image.
4. Recency still outranks it: the freshest turns are never altered.
5. It requires an optional codec (`distil[image]`). Without it the transform is inert,
   so the stdlib-only guarantee holds for anyone who does not opt in.

### Why this is not a reversal of the contract

distil's contract is not "never alter what the model sees" — the Tier-1 text digest
alters it on every request. The contract is that nothing is *irrecoverably* lost:
the replacement says what it replaced and how to get it back. Points 1 and 2 put
downscaling in that same category. What ADR 0003 forbids, and this ADR continues to
forbid, is downscaling with no way back.

## Consequences

- The largest remaining vision cost becomes addressable, for users who certify it.
- **A known weakness, stated plainly:** an elided text span is *visibly* absent — the
  model reads a marker where content used to be. A downscaled image is not. It looks
  like an image, and a model that does not read the accompanying note may never
  realise detail is missing and so never expand. This asymmetry cannot be engineered
  away from inside the transform, and it is the reason this ships off, uncertified,
  with no inherited certificate, rather than following `vision` into the default path.
- Certifying it is a claim about *your* screenshots. Whether losing pixel detail
  changes a decision depends on what is in them — a maintainer's corpus cannot answer
  that for someone else, which is why the shipped-certificate mechanism that makes
  `vision` usable out of the box is deliberately not extended here.
- If a future certification run over a broad screenshot corpus shows non-inferiority
  robustly, bundling a scoped certificate becomes arguable. It is not arguable today,
  on zero evidence.

# 0006 — The subscription boundary: what distil does to a flat-rate session

- **Status:** accepted
- **Date:** 2026-08-10
- **Supersedes:** nothing. Records a boundary that existed in code but was never written down, and which two parts of the codebase described differently.

## Context

A flat-rate (Claude Pro/Max OAuth) session defaults to `lossless-only`. Two places in the tree explained that default, and they did not agree.

`distil/onboard.py` told the user the subscription path is:

> flat-rate plan: real lossless token savings … **ToS-safe (no lossy digest, no tool injection)**

`distil/proxy.py` justified the same default on recoverability:

> lossless-only is a hard safety boundary: with no injected expand tool the agent can never recover a Tier-1 digest stub, so a stub there is irreversibly lossy.

These lead to opposite conclusions the moment `--expand` enters the picture. Under the recoverability reading, `--expand` injects `distil_expand`, every stub becomes recoverable, the stated harm disappears, and the default should simply flip — the same reasoning already applied to PAYG, which defaults to recoverable-digest for exactly this reason. Under the ToS reading, injection is the thing being avoided, and `--expand` is precisely what you must not do by default.

A third artifact voted for the ToS reading and enforced nothing: `policy.may_inject_tools(mode)` returned `mode is AuthMode.PAYG`, was asserted by `test_policy_fidelity_holdout.py`, and **was never called by any production code**. A subscription session passing `--expand` got injection with nothing consulting that function.

The stakes are not small. Measured on a maintainer's machine over its own ledger:

| mode | runs | smaller | tokens saved |
|---|---|---|---|
| digest | 2,353 | 52.27% | 644,922,653 |
| lossless-only | 8,319 | 0.27% | 9,601,508 |

~190×, at 100% decision-equivalence over 847 shadowed requests. The safe default is not conservative at the margin; it is most of the product, off.

## Decision

**The boundary is consent, not recoverability.** Stated positively:

1. **The default for a flat-rate session injects nothing and loses nothing.** `lossless-only` stays the default. Tier-0 transforms only.
2. **`--expand` is an explicit opt-in that DOES modify the request.** It is permitted on a flat-rate session — an injected `distil_expand` is what makes a Tier-1 digest reversible — and it is the user's call to make, not distil's.
3. **Every route into injection must say so, once, on stderr.** Both spellings (`--lossless-only --expand` and `distil default --mode expand`) cross the same line and both must disclose it.
4. **Output shaping stays PAYG-only.** It rewrites the *response*, which `--expand` does not make recoverable. This is the one thing an opt-in does not buy.

Recoverability remains true and remains the reason a digest is *safe* once opted into. It is not the reason for the default. The reason for the default is that distil does not modify a first-party session unasked.

## Consequences

- `onboard.py`'s "no tool injection" claim stays accurate, because it describes the default path, which still injects nothing.
- The disclosure gap is closed. Previously the notice fired only on `lossless_only and expand`, so the documented route (`distil default --mode expand`) opted the user into injection **silently**. It is now keyed on `expand and (lossless_only or subscription_mode())`.
- `may_inject_tools()` and its assertions are **deleted**. It stated a rule distil does not implement, and its passing test made the drift invisible in CI. An unenforced policy with a green test is worse than no policy. Do not reintroduce it without a call site.
- Flipping the default to `expand` remains a live proposal, and this ADR is what it has to argue against: not "is it recoverable" (it is) but "may distil modify a flat-rate session without being asked". That is a product decision, and it is unresolved.

## If the default is ever flipped, one trap

`proxy.py` derives `_auth_mode` from the `lossless_only` **flag**, not from real billing:

```python
_auth_mode = AuthMode.SUBSCRIPTION if lossless_only else AuthMode.PAYG
```

Simply defaulting `lossless_only=False` on a subscription would therefore make the session look like PAYG, and `--shape-output` would take effect on a flat-rate session — violating decision 4 above as a side effect of an unrelated change. Derive `_auth_mode` from `subscription_mode()` first.

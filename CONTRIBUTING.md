# Contributing to Distil

Thanks for helping build compression you can trust. Distil has exactly one
non-negotiable rule, and it *is* the project's thesis:

> **A change that touches compression must pass `make gate`.**
> Non-inferior on every domain in the corpus, and byte-reversible. No green gate, no merge.

## Dev loop

```bash
git clone https://github.com/dshakes/distil && cd distil
make test     # full test suite (stdlib-only; uv handles the env)
make gate     # tests + corpus non-inferiority gate + byte-fidelity gate
make lint     # ruff (enforced in CI — a PR won't merge with lint errors)
```

No runtime dependencies are allowed in the core. Optional features go behind an
extra (e.g. `distil[live]`) and must import lazily so the core still runs with
zero deps and no API key.

## Adding a compression strategy

1. Implement it in `distil/compress/strategies.py` (or a Tier module).
2. Run `distil certify --strategy <name>` — it must certify **non-inferior**.
3. Run `distil bench` — it must pass on **every** domain.
4. If it's lossy, gate it in `distil/policy.py` (lossless-only on subscription).

## Adding a domain trajectory

Drop a JSON file in `corpus/`, add it to `corpus/manifest.json`, and make sure
`distil bench` stays green. The invariants `distil/corpus.py::validate` enforces
(cacheable prefix, decision-driven tool output, prunable noise) are what keep the
savings/ablation/certification signals real rather than artifacts.

## Style

Python 3.11+, full type hints, `from __future__ import annotations`, ruff
(line-length 100). Match the surrounding code; keep comments earning their place.

## Cutting a release

See [`RELEASING.md`](RELEASING.md) — push a `v*` tag, CI publishes to PyPI via Trusted
Publishing (no token anywhere). `./scripts/release.sh` drives it end to end.

## Credit

Landed a PR? Add yourself to [`CONTRIBUTORS.md`](CONTRIBUTORS.md) in the same PR —
one line, newest last. Every merged contribution earns a spot.

## Good first issues

Issues labelled [`good first issue`](https://github.com/dshakes/distil/labels/good%20first%20issue)
are scoped so the gate can tell you whether you got it right. If none are open,
these are always welcome and always in scope:

| | Why it is a good first change |
|---|---|
| **A new `wrap` preset** | One line in `distil/onboard.py:AGENT_PRESETS` plus the doc comment. The bar is a **published** env-var contract — a guessed variable ships a wrap that reports success and routes nothing, which is worse than no preset. Cursor/Copilot/Cline are excluded for exactly this reason; see `docs/IDE-AGENTS.md`. |
| **A framework integration** | Copy `distil/integrations/agno.py`. The rule: duck-typed, never import the framework, so distil stays zero-dependency and a framework release cannot break us. |
| **A corpus domain** | `distil bench` grades per domain. A new one with real (content-free) traffic makes every certificate broader. |
| **A failing case for `distil validate`** | An input that breaks reversibility, inflates output, or leaks into telemetry. A reproduction is a contribution even without a fix. |
| **Docs that were wrong** | Especially a claim that is no longer true. `tests/test_packaging_assets.py` pins the checkable ones; the rest need a human who noticed. |

Not sure if something is in scope? Open the issue first and ask — that is cheaper
than a PR neither of us wants to reject.

## Conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The short version: be kind, be
rigorous, and report results faithfully — if a check failed, say so with the
output.

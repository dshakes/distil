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

## Conduct

Be kind, be rigorous, report results faithfully (if a check failed, say so with
the output). That's it.

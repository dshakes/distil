"""Distil — compression with a quality contract.

Thesis: in an agentic runtime you don't need byte-equivalence, you need
*decision-equivalence* — the agent takes the same actions and produces the same
final outputs whether or not the context was compressed. That is measurable,
certifiable, and compatible with aggressive compression.

This package demonstrates the two highest-leverage techniques end-to-end:
  1. Cache-aware compression (distil.compress.cache_aware) — the dominant cost
     in a multi-turn agent loop is cache *misses*, not context *size*. Keep the
     prefix byte-stable, compress only the volatile tail. Priced in real dollars.
  4. Causal / counterfactual pruning (distil.replay.ablation) — the eval engine
     is not a ruler, it is a discovery engine: it finds context that never
     changes a decision and is therefore free to drop. Certified by a TOST
     non-inferiority gate (distil.certify.stats).
"""

# Single-sourced from the installed distribution's metadata (pyproject `version`),
# so `distil --version` can never drift from the published package. The literal
# fallback is only used when running from source/zipapp with no installed metadata.
from importlib.metadata import PackageNotFoundError, version as _pkg_version  # noqa: E402

try:
    __version__ = _pkg_version("distil-llm")
except PackageNotFoundError:  # source checkout / zipapp without dist-info
    # Read pyproject directly rather than hardcode a literal — a duplicated
    # version string drifts from the real one AND conflicts on every release
    # merge-back (a conflicted pyproject then bricks `uv run`). Single source.
    try:
        import pathlib
        import re

        _pp = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        _m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', _pp)
        __version__ = _m.group(1) if _m else "0+source"
    except Exception:  # noqa: BLE001 — version must never break import
        __version__ = "0+source"

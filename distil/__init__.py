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

__version__ = "0.25.1"

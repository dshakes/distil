"""Counterfactual cost simulator for context-compression policies (research harness, $0).

Replays real agent trajectories request by request through a *policy* (what bytes would
have been sent) and an exact provider prompt-cache cost model (what those bytes would have
been billed), so policies can be compared offline before any paid run.

Lives in ``benchmarks/`` rather than ``distil/`` on purpose: it reads benchmark artefacts,
fits a token model to one run's billed usage, and makes no product claim. If its
recommendation survives a live run, the *policy* graduates into ``distil/`` (ADR 0027);
the simulator stays a research tool.

Modules: ``trajectory`` (loaders), ``costmodel`` (Anthropic / OpenAI / Gemini cache
pricing), ``policies`` (plain, distil, rtk-like, entry-only, ...), ``calibrate`` (fit +
held-out error vs billed usage), ``run`` (evaluation, re-run penalty, search, Pareto).
"""

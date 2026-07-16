"""Recency carve-out — the shared constant, and the statement of how the certificate
transfers to serving.

An agent must see its freshest tool outputs byte-exact to choose its next action (and
the in-context path may not be able to expand a stub there), so the live adapter never
digests tool outputs in the last ``RECENCY_KEEP_TURNS`` tool-bearing turns. Digesting
only OLDER turns is also strictly cache-safe: the cached prefix never contains recent
message history, so exempting recent turns only ever *reduces* what we rewrite.

The certified ``distil`` strategy deliberately does NOT apply this carve-out: it digests
every volatile block, including the freshest observation. That makes certification
*harsher* than serving, which is the safe transfer direction — if decisions survive with
the freshest output digested, they survive a-fortiori when serving keeps it verbatim.
The invariant that serving's digest-set is a SUBSET of certification's digest-set is
pinned by ``tests/test_live_certified_equivalence.py``; the constant lives here so both
sides of that invariant are reviewed in one place.
"""

from __future__ import annotations

RECENCY_KEEP_TURNS = 2

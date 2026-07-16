"""Live (adapter) vs certified (strategy) equivalence — conservative transfer, pinned (F9).

The empirical decision-equivalence certificate is computed over the block-level
``distil`` strategy in :mod:`distil.compress.strategies`, while live serving compresses
message dicts through :func:`distil.adapters.anthropic.compress_messages`. They are two
code paths with ONE deliberate difference, and this test pins both the agreement and the
direction of the difference:

  * The certified strategy digests EVERY volatile tool output — including the freshest
    one. Certification is deliberately *harsher* than serving.
  * The live adapter additionally keeps the last ``RECENCY_KEEP_TURNS`` tool-bearing
    turns byte-exact (an agent must see its freshest tool output verbatim to choose its
    next action, and the in-context path may not be able to expand a stub there).

So the SUBSET INVARIANT holds: the set of blocks serving digests is a strict subset of
the set certification digests. Non-inferiority proven under the harsher compression
transfers a-fortiori to the strictly gentler served path. The shared constant lives in
:mod:`distil.compress.recency` so the rule is reviewed in one place.

Both paths anchor recovery on the same handle: ``sha256(original_text)[:8]`` — a
certificate about the digest's recoverability names the same object on both paths.

If the digest algorithm or the recency rule changes on either side, this test breaks —
which is the point: divergence must be a visible, reviewed change, never silent drift
between "what we certify" and "what we ship".
"""

from __future__ import annotations

import hashlib
import re

from distil.adapters.anthropic import _RECENCY_KEEP_TURNS, compress_messages
from distil.compress.recency import RECENCY_KEEP_TURNS
from distil.compress.strategies import REGISTRY
from distil.trajectory import Block, Kind, Stability

_HANDLE = re.compile(r"handle=([0-9a-f]{8})")


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _tool_output(name: str) -> str:
    # 12 keyword-free filler lines: well above the 6-line digest threshold, not JSON
    # (so it never folds) and not templated (so template-mining never fires) — the
    # plain reversible-digest path, with a droppable middle guaranteed.
    return "\n".join([f"{name}: detail line {i} of routine output" for i in range(12)])


def _tool_result_message(text: str) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "content": text}]}


def _handles_in(text: str) -> set[str]:
    return set(_HANDLE.findall(text))


def test_adapter_uses_the_shared_recency_constant() -> None:
    """The adapter's carve-out is the one reviewed constant in compress.recency."""
    assert _RECENCY_KEEP_TURNS == RECENCY_KEEP_TURNS


def test_served_digests_are_a_subset_of_certified_digests() -> None:
    originals = [_tool_output(n) for n in ("alpha", "beta", "gamma", "delta")]
    assert len(originals) > RECENCY_KEEP_TURNS  # need older blocks AND a recency tail

    # --- certified path: the block-level `distil` strategy the certificate is proven on
    blocks = [
        Block(id=f"b{i}", kind=Kind.TOOL_OUTPUT, text=t, stability=Stability.VOLATILE)
        for i, t in enumerate(originals)
    ]
    certified_out = REGISTRY["distil"](blocks, 0)
    cert_by_orig = {originals[i]: b.text for i, b in enumerate(certified_out)}
    cert_digested = {o for o, t in cert_by_orig.items() if t != o}

    # --- live path: the message-dict adapter used in real serving
    messages = [_tool_result_message(t) for t in originals]
    new_messages, store = compress_messages(messages)
    live_by_orig = {originals[i]: m["content"][0]["content"] for i, m in enumerate(new_messages)}
    live_digested = {o for o, t in live_by_orig.items() if t != o}

    # 1) Certification is the harsher compression: it digests every volatile tool output.
    assert cert_digested == set(originals)

    # 2) SUBSET INVARIANT (the safe transfer direction): serving digests exactly the
    #    certified set MINUS the recency tail — never anything certification didn't.
    recency_tail = set(originals[-RECENCY_KEEP_TURNS:])
    assert live_digested == cert_digested - recency_tail
    assert live_digested < cert_digested  # strict subset, by construction
    for o in recency_tail:
        assert live_by_orig[o] == o  # verbatim-recency semantics: byte-exact

    # 3) Wherever BOTH paths digest, they emit the SAME recovery handle == sha256(orig)[:8],
    #    and the live store resolves it back to the byte-exact original. Same anchor ->
    #    a certificate about the digest's recoverability names the same object live.
    for o in live_digested:
        want = _sha8(o)
        assert _handles_in(live_by_orig[o]) == {want}
        assert _handles_in(cert_by_orig[o]) == {want}
        assert store.expand(want) == o

"""claude-opus-5 must price. It shipped unpriced, so `distil doctor` reported
"ledger contains unpriced models: claude-opus-5" and every run on the model
showed $0 saved — silently under-reporting the headline number on live traffic.
"""

from __future__ import annotations

from distil.pricing import CATALOG, resolve


def test_opus_5_is_priced() -> None:
    p = CATALOG["claude-opus-5"]
    assert (p.input_per_mtok, p.output_per_mtok) == (5.0, 25.0)


def test_opus_5_resolves_in_the_id_shapes_real_traffic_uses() -> None:
    for wire_id in (
        "claude-opus-5",
        "anthropic.claude-opus-5",  # Bedrock prefix
        "claude-opus-5@20260101",  # Vertex version separator
        "claude-opus-5-20260101",  # dated snapshot → longest-prefix match
    ):
        assert resolve(wire_id) is not None, wire_id
        assert resolve(wire_id).name == "claude-opus-5", wire_id

    # The guard that must not regress: an unknown upstream is never billed at
    # Claude rates just because the catalog got a new entry.
    assert resolve("gemini-3-pro") is None

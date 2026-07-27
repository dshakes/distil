"""The corpus gate, as tests — every trajectory in every domain must hold the
invariants that make the savings/ablation/certification signals real.

This is the safety gate: a strategy that can't pass non-inferiority across the
whole corpus does not ship.
"""

import pytest

from distil import pricing
from distil.certify.gate import certify
from distil.compress.cache_aware import simulate
from distil.corpus import CORPUS_DIR, load_corpus, validate
from distil.fidelity import verify_reversible
from distil.replay.ablation import discover

ENTRIES = load_corpus()
IDS = [e.trajectory.id for e in ENTRIES]


def test_corpus_is_multidomain():
    assert len(ENTRIES) >= 6
    assert len({e.domain for e in ENTRIES}) >= 6  # genuinely cross-domain, not just coding


def test_manifest_covers_every_corpus_file():
    """Every corpus file must be accounted for — as an offline-gate trajectory or
    explicitly as live-only. An unlisted file is dead weight nothing exercises.

    `live_only` exists for domains the DeterministicRunner structurally cannot
    grade. The vision domain's decision is carried by the IMAGE; that runner reads
    DECISION: markers out of text, and corpus.validate() demands one on a
    decision-relevant volatile block. Meeting that here would mean writing the
    screenshot's content into the text, making the image decorative and any
    resulting certificate a claim about text. Those domains are certified against
    a live vision runner instead (see corpus/manifest.json for the reasoning).
    """
    import json

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    listed = {e.file for e in ENTRIES} | {t["file"] for t in manifest.get("live_only", [])}
    on_disk = {p.name for p in CORPUS_DIR.glob("*.json") if p.name != "manifest.json"}
    assert on_disk == listed, f"orphan/missing corpus files: {on_disk ^ listed}"

    for entry in manifest.get("live_only", []):
        assert entry.get("why_not_in_the_offline_gate"), (
            f"{entry['file']} is live-only but does not say why — without the reason "
            "someone will 'fix' it back into the offline gate and certify text"
        )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_structurally_valid(entry):
    assert validate(entry.trajectory) == []


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_distil_certified_non_inferior(entry):
    report = certify(entry.trajectory, "distil")
    assert report.match_rate == 1.0
    assert report.verdict == "PASS"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_gate_rejects_aggressive(entry):
    report = certify(entry.trajectory, "aggressive")
    assert report.verdict == "FAIL"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_ablation_finds_prunable(entry):
    assert discover(entry.trajectory).tokens_freed > 0


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_distil_is_cheaper_losslessly(entry):
    price = pricing.get(entry.trajectory.model)
    base = simulate(entry.trajectory, price, strategy="none", caching=False)
    dist = simulate(entry.trajectory, price, strategy="distil", caching=True)
    assert dist.total_dollars < base.total_dollars
    assert dist.cache_hit_tokens > 0


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_distil_is_byte_reversible(entry):
    # every turn's distil output must be reconstructable from the local restore table
    for turn in entry.trajectory.turns:
        from distil.compress.tier0 import Tier0Lossless
        from distil.compress.tier1 import Tier1Reversible
        from distil.trajectory import Stability

        volatile = [b for b in turn.blocks if b.stability is Stability.VOLATILE]
        r1 = Tier1Reversible().compress(volatile)
        r0 = Tier0Lossless().compress(r1.blocks)
        merged_restore = {**r1.restore, **r0.restore}
        from distil.compress.base import CompressResult

        merged = CompressResult(r0.blocks, merged_restore)
        report = verify_reversible(volatile, merged)
        assert report.lossless, f"{entry.file}: irrecoverable {report.irrecoverable}"

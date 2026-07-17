"""Model-free salience protection — keep the needle, compress the haystack."""

from __future__ import annotations

from distil.compress.salience import (
    protect,
    reference_index,
    salient_lines,
    salient_tokens,
)
from distil.trajectory import Block, Kind, Stability


def _vol(bid, text, dr=False):
    return Block(bid, Kind.TOOL_OUTPUT, text, Stability.VOLATILE, dr)


def test_pattern_signal_catches_identifier_shapes():
    t = "payment PAY-12345 failed; sha 9af2bc1; host 10.0.3.7; v2.1.0; a@b.com"
    toks = salient_tokens(t)
    assert "PAY-12345" in toks
    assert "9af2bc1" in toks  # mixed hex hash
    assert "10.0.3.7" in toks
    assert "v2.1.0" in toks
    assert "a@b.com" in toks


def test_pattern_signal_catches_urls():
    """Live-grading find (2026-07-16): a digested URL made the reference model
    fall back from fetchurl to a websearch detour — URLs are decision-relevant
    artifacts exactly like SHAs and paths."""
    t = "monthly report at https://www.ncei.noaa.gov/access/monitoring/monthly-report/global/202313 per NOAA"
    toks = salient_tokens(t)
    assert any(t.startswith("https://www.ncei.noaa.gov") for t in toks)


def test_entropy_signal_catches_novel_ids_no_regex_anticipates():
    # a random-looking key no fixed pattern would match
    toks = salient_tokens("token=x7Qm9Zk2Lp4Rt8Wv the rest is ordinary prose words here")
    assert any("x7Qm9Zk2Lp4Rt8Wv" == t for t in toks)
    # ordinary words are NOT salient
    assert "ordinary" not in toks and "prose" not in toks


def test_reference_signal_protects_cross_block_anchors():
    blocks = [
        _vol("a", "tool schema: rotate_logs(node) acts on NODE7 targets"),
        _vol("b", "observation: NODE7 disk at 95 percent"),
    ]
    ref = reference_index(blocks)
    # NODE7 appears in two blocks -> anchor; protected even without a pattern match
    assert salient_tokens(blocks[1].text, ref_index=ref) & {"NODE7"}


def test_salient_lines_keep_the_decision_unit_together():
    text = "verbose preamble line\nDECISION DIRECTIVE: notify_customer(PAY-77001)\ntrailing noise"
    lines = salient_lines(text)
    assert any("notify_customer(PAY-77001)" in ln for ln in lines)  # verb + target together
    assert all("verbose preamble" not in ln for ln in lines)


def test_protect_makes_truncation_preserve_the_directive():
    # a long block whose load-bearing directive sits past a tight truncation limit
    pre = "diagnostic context for the audit trail only; " * 12
    block = _vol(
        "obs", f"get_status() -> {pre}\nDECISION DIRECTIVE: block_card(CARD-90887)", dr=True
    )

    def truncate200(blocks, turn):
        return [b.copy_with(b.text[:200]) for b in blocks]

    # plain truncation drops the directive
    plain = truncate200([block], 0)[0]
    assert "CARD-90887" not in plain.text

    # protected truncation re-injects the salient line -> target survives
    out = protect(truncate200)([block], 0)[0]
    assert "CARD-90887" in out.text
    assert "block_card" in out.text  # the action verb came along (line-level)
    # and it never exceeds the original (reject-if-bigger)
    assert len(out.text) <= len(block.text)


def test_protect_is_noop_when_nothing_salient_dropped():
    block = _vol("x", "just ordinary prose with no identifiers at all here")

    def truncate5(blocks, turn):
        return [b.copy_with(b.text[:5]) for b in blocks]

    out = protect(truncate5)([block], 0)[0]
    assert out.text == "just "  # nothing salient to protect, truncation stands


def test_protect_noop_on_lossless_identity():
    block = _vol("x", "DECISION DIRECTIVE: refund_order(ORD-5)")

    def identity(blocks, turn):
        return list(blocks)

    out = protect(identity)([block], 0)[0]
    assert out.text == block.text  # unchanged input -> unchanged output


def test_salient_tokens_pluggable_scorer():
    from distil.compress.salience import salient_tokens

    base = salient_tokens("the deployment looks fine and normal today")
    # A scorer (stand-in for a semantic/NER model) contributes extra spans.
    with_scorer = salient_tokens(
        "the deployment looks fine and normal today",
        scorer=lambda t: ["deployment", "normal"],
    )
    assert {"deployment", "normal"} <= with_scorer
    assert base <= with_scorer  # seam only ADDS coverage


def test_salient_tokens_bad_scorer_never_crashes():
    from distil.compress.salience import salient_tokens

    def boom(_t):
        raise RuntimeError("model down")

    # A broken scorer must not break compression — falls back to model-free signals.
    out = salient_tokens("rotate node N7 now", scorer=boom)
    assert isinstance(out, set)

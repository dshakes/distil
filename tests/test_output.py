"""Output compression — shaping (gated + adaptive), lossless re-entry digest, A/B."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from distil.certify.gate import CERT_MARGIN
from distil.output import (
    AUTO_LEVEL,
    SHAPE_EVIDENCE_DAYS,
    answer_fingerprint,
    digest_output_blocks,
    measure_output_savings,
    resolve_shape_output,
    shape_request,
)
from distil.shadow import SIG_VERSION, VERDICT_MIN_AA, VERDICT_MIN_AB, lever
from distil.trajectory import Block, Kind, Stability

PAIRS_FILE = Path(__file__).resolve().parent.parent / "corpus" / "output_pairs.jsonl"


def _ledger(
    tmp_path: Path,
    *,
    n_ab: int = VERDICT_MIN_AB,
    equivalent: bool = True,
    out_delta: int = -80,
    alternating: bool = False,
    n_cost: int | None = None,
    shape: str | None = "off",
    age_days: float = 0.0,
) -> Path:
    """A shadow.jsonl of synthetic PAIRED rows — no network.

    Written to disk and read back through ``ShadowLedger.load`` so the lever filter
    and the estimator under test are the shipped ones. ``alternating`` flips the sign
    of the per-row output delta so the bootstrap interval straddles zero. ``shape``
    is the shaping lever the rows were measured under; ``None`` writes pre-tag rows.
    """
    n_cost = n_ab if n_cost is None else n_cost
    base: dict = {"sig": SIG_VERSION, "mode": "digest", "ts": time.time() - age_days * 86400}
    if shape is not None:
        base["levers"] = {"compression": "digest", "shape": shape}
    rows = []
    for i in range(n_ab):
        rec = {**base, "kind": "paired", "equivalent": equivalent, "aa_equal": True}
        if i < n_cost:
            delta = -out_delta if (alternating and i % 2) else out_delta
            rec |= {
                "model": "claude-opus-4-8",
                "in_a": 1000,
                "in_b": 500,
                "out_a": 400,
                "out_b": 400 + delta,
            }
        rows.append(rec)
    # Top the A/A arm up to its own floor without touching the A/B pool.
    rows += [{**base, "kind": "aa", "equivalent": True}] * max(0, VERDICT_MIN_AA - n_ab)
    path = tmp_path / "shadow.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


# --- adaptive mode ----------------------------------------------------------
def test_auto_turns_shaping_on_when_the_referee_says_so(tmp_path):
    d = resolve_shape_output("auto", lossy_ok=True, path=_ledger(tmp_path))
    assert d.level == AUTO_LEVEL and d.on
    assert "decision-equivalence" in d.reason and "shorter" in d.reason


def test_auto_cannot_be_kept_on_by_its_own_shaped_rows(tmp_path):
    """The self-reinforcing gate. Once shaping is on, the shadow B arm carries the
    directive, so "replies shorter" is shaping measuring itself. Rows that say
    exactly what would flip the gate on — but were measured WITH shaping — must
    not count, whatever they show."""
    shaped = _ledger(tmp_path, shape=AUTO_LEVEL)
    d = resolve_shape_output("auto", lossy_ok=True, path=shaped)
    assert d.level == "off" and "below reporting floor" in d.reason
    assert "n=0 A/B" in d.reason  # nothing counted, not merely too little


def test_auto_ignores_rows_that_predate_the_lever_tag(tmp_path):
    # An untagged row's shaping state is unknown — never assumed off.
    d = resolve_shape_output("auto", lossy_ok=True, path=_ledger(tmp_path, shape=None))
    assert d.level == "off" and "n=0 A/B" in d.reason


def test_auto_decides_from_the_unshaped_rows_in_a_mixed_ledger(tmp_path):
    # Shaped rows showing a big "saving" alongside unshaped rows showing none: the
    # verdict is the unshaped one.
    _ledger(tmp_path, shape=AUTO_LEVEL, out_delta=-300)
    path = _ledger(tmp_path, alternating=True)
    d = resolve_shape_output("auto", lossy_ok=True, path=path)
    assert d.level == "off" and "does not exclude zero" in d.reason


def test_shaped_rows_showing_harm_turn_auto_off(tmp_path):
    """The reverse direction. Unshaped rows justify shaping; the shaped rows — the
    only ones that measure the directive's own effect — show every decision
    changed. Shaping's own harm must turn it off, or auto could never undo itself."""
    _ledger(tmp_path)  # unshaped: would turn shaping on alone
    path = _ledger(tmp_path, shape=AUTO_LEVEL, equivalent=False)
    d = resolve_shape_output("auto", lossy_ok=True, path=path)
    assert d.level == "off"
    assert "with shaping on" in d.reason and "certified budget" in d.reason


def test_thin_shaped_evidence_does_not_block_the_first_on(tmp_path):
    # Below the floor, shaped rows are no evidence either way.
    _ledger(tmp_path)
    path = _ledger(tmp_path, shape=AUTO_LEVEL, equivalent=False, n_ab=5)
    assert resolve_shape_output("auto", lossy_ok=True, path=path).level == AUTO_LEVEL


def test_auto_evidence_expires(tmp_path):
    """An "on" decision cannot rest forever on the past: rows older than the
    window do not count, so stale evidence reads as no evidence."""
    stale = _ledger(tmp_path, age_days=SHAPE_EVIDENCE_DAYS + 1)
    d = resolve_shape_output("auto", lossy_ok=True, path=stale)
    assert d.level == "off" and "n=0 A/B" in d.reason
    # And stale shaped harm does not veto fresh evidence either.
    _ledger(tmp_path, shape=AUTO_LEVEL, equivalent=False, age_days=SHAPE_EVIDENCE_DAYS + 1)
    fresh = _ledger(tmp_path)
    assert resolve_shape_output("auto", lossy_ok=True, path=fresh).level == AUTO_LEVEL


def test_auto_is_off_below_the_reporting_floor(tmp_path):
    path = _ledger(tmp_path, n_ab=VERDICT_MIN_AB - 1)
    d = resolve_shape_output("auto", lossy_ok=True, path=path)
    assert d.level == "off"
    assert "below reporting floor" in d.reason


def test_auto_is_off_when_the_output_interval_includes_zero(tmp_path):
    # Half the samples shorten, half lengthen: the bootstrap interval straddles 0,
    # so there is no evidence the reply side pays on this traffic.
    d = resolve_shape_output("auto", lossy_ok=True, path=_ledger(tmp_path, alternating=True))
    assert d.level == "off"
    assert "does not exclude zero" in d.reason


def test_auto_is_off_when_harm_exceeds_the_certified_budget(tmp_path):
    # Every paired row a decision CHANGE: diff = -1, far outside ±CERT_MARGIN.
    d = resolve_shape_output("auto", lossy_ok=True, path=_ledger(tmp_path, equivalent=False))
    assert d.level == "off"
    assert "certified budget" in d.reason and f"{CERT_MARGIN * 100:.0f}pp" in d.reason


def test_auto_is_off_without_enough_priced_samples(tmp_path):
    d = resolve_shape_output("auto", lossy_ok=True, path=_ledger(tmp_path, n_cost=3))
    assert d.level == "off" and "priced shadow sample" in d.reason


def test_subscription_is_off_whatever_was_asked_for(tmp_path):
    path = _ledger(tmp_path)
    for requested in ("auto", "light", "aggressive"):
        d = resolve_shape_output(requested, lossy_ok=False, path=path)
        assert d.level == "off"
        assert "flat-rate" in d.reason
        assert d.requested == requested  # the ask is recorded, not silently rewritten


def test_explicit_levels_override_the_evidence(tmp_path):
    # Evidence that would say "off" (below floor) must not veto an explicit ask.
    thin = _ledger(tmp_path, n_ab=1)
    for requested in ("light", "aggressive"):
        d = resolve_shape_output(requested, lossy_ok=True, path=thin)
        assert d.level == requested and d.reason == "explicitly requested"
    assert resolve_shape_output("off", lossy_ok=True, path=thin).level == "off"


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown shape-output mode"):
        resolve_shape_output("sometimes", lossy_ok=True)


def test_lever_reads_the_tag_and_treats_untagged_as_unknown():
    assert lever({"levers": {"shape": "off", "compression": "digest"}}, "shape") == "off"
    assert lever({"levers": {"compression": "digest"}}, "shape") is None
    assert lever({}, "shape") is None
    assert lever({"levers": "garbage"}, "shape") is None


# --- generation-side shaping ------------------------------------------------
def test_shape_request_anthropic_uses_top_level_system():
    # The Anthropic Messages API 400s on role:"system" inside messages —
    # a Claude body must get the directive via the top-level system field.
    body = {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    out = shape_request(body, level="light", allow=True)
    assert "concise" in out["system"].lower()
    assert all(m["role"] != "system" for m in out["messages"])
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # input not mutated


def test_shape_request_anthropic_appends_to_existing_system():
    body = {"model": "claude-opus-4-8", "system": "You are a bot.", "messages": []}
    out = shape_request(body, level="light", allow=True)
    assert out["system"].startswith("You are a bot.")
    assert "concise" in out["system"].lower()
    # list-form system prompts get a text block appended
    body2 = {"system": [{"type": "text", "text": "core"}], "messages": []}
    out2 = shape_request(body2, level="light", allow=True, shape="anthropic")
    assert out2["system"][0] == {"type": "text", "text": "core"}
    assert "concise" in out2["system"][1]["text"].lower()


def test_shape_request_openai_appends_system_message():
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    out = shape_request(body, level="light", allow=True, shape="openai")
    assert out["messages"][-1]["role"] == "system"
    assert "concise" in out["messages"][-1]["content"].lower()


def test_shape_request_noop_when_off_or_disallowed():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert shape_request(body, level="off", allow=True) is body
    assert shape_request(body, level="aggressive", allow=False) is body  # auth-gated


def test_shape_request_unknown_level_raises():
    with pytest.raises(ValueError):
        shape_request({"messages": []}, level="ludicrous", allow=True)


# --- lossless re-entry digest -----------------------------------------------
def test_digest_output_blocks_is_reversible():
    long_answer = "DECISION: ship it\n" + "\n".join(f"reasoning step {i}" for i in range(20))
    blocks = [Block("h0", Kind.HISTORY, long_answer, Stability.SETTLING)]
    out, restore = digest_output_blocks(blocks)
    assert len(out[0].text) < len(long_answer)  # compressed
    assert "DECISION: ship it" in out[0].text  # decision preserved
    assert long_answer in restore.values()  # original recoverable
    assert out[0].kind is Kind.HISTORY  # kind preserved


def test_digest_output_blocks_skips_short():
    blocks = [Block("h0", Kind.HISTORY, "short answer", Stability.SETTLING)]
    out, restore = digest_output_blocks(blocks)
    assert out[0].text == "short answer" and not restore


# --- A/B measurement (the evaluation) ---------------------------------------
def test_answer_fingerprint_extracts_decision():
    a = "lots of preamble. DECISION: roll back to rev 6. trailing recap."
    b = "DECISION: roll back to rev 6."
    assert answer_fingerprint(a) == answer_fingerprint(b)


def test_measure_output_savings_on_real_fixture():
    pairs = [
        (d["baseline"], d["shaped"])
        for d in (json.loads(line) for line in PAIRS_FILE.read_text().splitlines() if line.strip())
    ]
    report = measure_output_savings(pairs)
    assert report.mean_reduction > 0.4  # verbose -> concise is a big cut
    assert report.answer_match_rate == 1.0  # every answer preserved (the gate)
    assert report.ci_low <= report.mean_reduction <= report.ci_high


def test_measure_flags_dropped_answer():
    # a "compression" that drops the decision must NOT count as a clean saving
    pairs = [("DECISION: do X. blah blah blah blah", "here is a terse but wrong summary")]
    report = measure_output_savings(pairs)
    assert report.answer_match_rate < 1.0


# --- wiring: the decision reaches every surface that reports it -------------
def test_build_handler_resolves_auto_and_exposes_the_decision():
    """`auto` must never survive into the request path — shape_request only knows
    concrete levels and raises on anything else."""
    from distil.proxy import build_handler

    handler = build_handler("https://api.anthropic.com", shape_output="auto")
    decision = handler.shape_decision
    assert decision.requested == "auto"
    assert decision.level in ("off", AUTO_LEVEL)
    # Sandboxed DISTIL_HOME → empty shadow ledger → no evidence → off, with a reason.
    assert decision.level == "off" and "below reporting floor" in decision.reason


def test_build_handler_keeps_the_subscription_boundary():
    from distil.proxy import build_handler

    handler = build_handler(
        "https://api.anthropic.com", lossless_only=True, shape_output="aggressive"
    )
    assert handler.shape_decision.level == "off"
    assert "flat-rate" in handler.shape_decision.reason


def test_worker_config_carries_the_mode_across_a_hot_swap(monkeypatch):
    from distil.hotswap import _CONFIG_ENV, WorkerConfig

    cfg = WorkerConfig(upstream="https://api.anthropic.com", shape_output="auto")
    monkeypatch.setenv(_CONFIG_ENV, cfg.to_env())
    assert WorkerConfig.from_env().shape_output == "auto"
    # And the resolved level a wrap session decided on survives the same trip, so a
    # restarted worker cannot re-decide from a ledger that moved mid-session.
    monkeypatch.setenv(_CONFIG_ENV, WorkerConfig(upstream="u", shape_output=AUTO_LEVEL).to_env())
    assert WorkerConfig.from_env().shape_output == AUTO_LEVEL


def test_dissect_flags_line_marks_an_automatic_decision():
    from distil.dissect import _flags_line

    auto_on = {"flags": {"shape_output": "light", "shape_requested": "auto"}}
    assert "shape_output=light(auto)" in _flags_line(auto_on)
    asked = {"flags": {"shape_output": "light", "shape_requested": "light"}}
    assert "shape_output=light" in _flags_line(asked) and "(auto)" not in _flags_line(asked)

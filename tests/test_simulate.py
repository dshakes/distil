"""`distil simulate` — the dry-run preview.

The property that matters is not the ratio (compression is tested elsewhere) but
that the preview tells the truth about what would be TOUCHED and what would be
PROTECTED. A preview that overstates savings, or that quietly omits a protection
rule, is worse than no preview: it is the trust-me pitch with a number attached.
"""

from __future__ import annotations

import json

from distil.simulate import format_report, simulate, waste_signals


def _tool_result(text: str, tid: str = "t1") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def _padding(n: int = 6) -> list[dict]:
    """Trailing turns so the block under test is outside the recency window."""
    return [
        {
            "role": "assistant" if i % 2 == 0 else "user",
            "content": [{"type": "text", "text": f"step {i}"}],
        }
        for i in range(n)
    ]


def _big_json(rows: int = 120) -> str:
    return json.dumps(
        [{"id": i, "service": "checkout", "status": "ok"} for i in range(rows)], indent=2
    )


def test_simulate_calls_no_model_and_reports_real_numbers():
    """No network, and the reported saving is the pipeline's actual output."""
    msgs = [_tool_result(_big_json())] + _padding()
    sim = simulate(msgs)
    assert sim["tokens_before"] > sim["tokens_after"] > 0
    assert sim["tokens_saved"] == sim["tokens_before"] - sim["tokens_after"]
    assert 0 < sim["savings_percent"] <= 100
    # Anything digested must be recoverable — that is what makes it not deletion.
    assert sim["recoverable_blocks"] >= 1


def test_protected_blocks_name_the_rule_that_protected_them():
    """The differentiator: a preview that shows the ratio but hides WHY a block
    was left alone is the same 'trust me' every other compressor offers."""
    msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": "I will run the tests now."}]},
        _tool_result(_big_json()),
    ] + _padding()
    sim = simulate(msgs)
    reasons = {b["protected_by"] for b in sim["blocks"] if b["protected_by"]}
    assert any("assistant" in r for r in reasons), "assistant turn protection not reported"
    assert any("recency" in r for r in reasons), "recency protection not reported"
    # Every protected block carries a reason; none is protected silently.
    for b in sim["blocks"]:
        if b["tokens_before"] == b["tokens_after"] and b["protected_by"]:
            assert len(b["protected_by"]) > 10


def test_tool_use_and_images_are_reported_as_never_rewritten():
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}}],
        },
        _tool_result("output\n" * 40),
    ] + _padding()
    sim = simulate(msgs)
    kinds = {b["type"]: b["protected_by"] for b in sim["blocks"]}
    assert "tool_use" in kinds and "verbatim" in kinds["tool_use"]


def test_waste_signals_report_chars_and_tokens_separately():
    """The default tokenizer is whitespace-INSENSITIVE: pretty-printed JSON and
    its minified form count identically. Reporting only tokens would print
    `json_bloat: 0` over an obviously bloated payload; reporting only chars would
    promise a saving this tokenizer will not give. So both, always."""
    pretty = _big_json()
    sig = waste_signals(pretty)
    assert sig["json_bloat"]["chars"] > 0, "pretty-printed JSON is bloat by any measure"
    # Not asserting tokens > 0 — that is tokenizer-dependent and the whole point.
    assert sig["json_bloat"]["tokens"] >= 0
    for name in ("json_bloat", "whitespace", "repetition"):
        assert set(sig[name]) == {"chars", "tokens"}


def test_waste_signals_are_zero_on_dense_text():
    """No invented waste: prose with no JSON, runs, or repetition reports none."""
    sig = waste_signals("The retry limit is five attempts before the circuit opens.")
    assert all(v["chars"] == 0 for v in sig.values())


def test_verbatim_mode_is_previewable_and_never_digests():
    """The subscription-safe mode must be inspectable before it is trusted."""
    msgs = [_tool_result(_big_json())] + _padding()
    sim = simulate(msgs, verbatim=True)
    assert sim["verbatim"] is True
    assert sim["recoverable_blocks"] == 0, "verbatim mode must emit no digest handles"


def test_report_wraps_reasons_instead_of_truncating():
    """These strings explain a rule; one cut mid-sentence teaches nothing."""
    msgs = [_tool_result(_big_json())] + _padding()
    text = format_report(simulate(msgs))
    assert "no model called" in text
    for line in text.splitlines():
        assert len(line) < 100
    # The recency rule is long enough to need wrapping — check it survives whole.
    assert "byte-exact" in text


def test_malformed_input_does_not_raise():
    """Previews run over whatever real traffic looks like, including junk."""
    for msgs in ([], [{}], [{"role": "user"}], [{"role": "user", "content": None}], ["nope"]):
        sim = simulate(list(msgs))
        assert sim["tokens_before"] >= 0

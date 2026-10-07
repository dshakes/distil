"""The append-only cache model behind benchmarks/why_rtk_wins.py, on a hand-built transcript."""

from benchmarks.why_rtk_wins import impossible_cache, reconstruct, step_prefixes

COEF = {
    "S": 100.0,
    "problem": 1.0,
    "tool_result": 1.0,
    "tool_use": 0.0,
    "text": 0.0,
    "thinking_sig": 0.0,
}


def _transcript():
    def call(i):
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}],
        }

    def result(i, n):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * n}],
        }

    return [
        {"role": "user", "content": "p" * 10},
        call(1),
        result(1, 20),
        call(2),
        result(2, 30),
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


def test_append_only_reconstruction():
    m = _transcript()
    assert len(step_prefixes(m)) == 3  # one request per assistant turn
    x = reconstruct(m, COEF)
    # prompts: 110, 130, 160 -> the last is written, the earlier ones are read
    assert x["pred_write"] == 160
    assert x["pred_read"] == 110 + 130
    # the 20-char result enters at step 2 and is re-read once (step 3); the 30-char one never
    assert x["new"]["tool_result"] == 50 and x["reread"]["tool_result"] == 20


def test_flags_precached_prompt():
    u = {"input": 0, "cache_write": 160, "cache_read": 240, "output": 0}
    assert not impossible_cache({"usage": u, "steps": 3})
    # a run that wrote less than its average read prompt must have been served a cache it never wrote
    assert impossible_cache({"usage": {**u, "cache_write": 50}, "steps": 3})

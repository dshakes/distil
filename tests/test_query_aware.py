"""Query-aware salience: distil keeps the tool-output line the agent asked about.

distil is a proxy, so it holds the agent's intent (its tool_use args + latest ask) in
the same request as the output being compressed. This exercises the whole live path:
extract_intent(messages) -> compress_messages -> digest(intent=...), and the additive
+ selectivity invariants the spec promises. See distil/compress/intent.py and tier1.digest.
"""

from distil.adapters.anthropic import compress_messages
from distil.compress.intent import extract_intent, relevant_lines, terms_of
from distil.compress.tier1 import digest


def _big_log(needle_line: str) -> str:
    lines = ["build started", ""]
    lines += [f"compiling module_{i} ok" for i in range(60)]  # neutral -> folds
    lines.append(needle_line)  # the one line the agent is looking for, buried
    lines += [f"linking artifact_{i}" for i in range(60)]
    return "\n".join(lines)


def test_grep_hit_survives_via_tool_use_intent():
    # agent asked a tool about MAX_RETRIES; the answer line is buried in a big log
    needle = "config/app.py:  MAX_RETRIES = 5"
    messages = [
        {"role": "user", "content": "what is the retry limit?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    # not grep: shell search output is now kept verbatim, so it would never fold
                    "input": {"command": "python scripts/show_config.py MAX_RETRIES config/"},
                }
            ],
        },
        # the tool_result must be OLDER than the recency window (last 2 turns are kept
        # verbatim), so it actually gets digested — that is where query-aware salience acts.
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _big_log(needle)}],
        },
        {"role": "assistant", "content": "Let me also check the timeout."},
        {"role": "user", "content": "sure, go ahead"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ]
    out, _store = compress_messages(messages)
    blob = str(out)
    assert "handle=" in blob, "neutral middle must still fold — compression preserved"
    assert "MAX_RETRIES = 5" in blob, "query-relevant needle was folded away"


def test_intent_off_still_folds_needle():
    # without intent, the needle is just neutral text and folds (baseline contrast)
    out, _ = digest(_big_log("config/app.py:  SOMEFLAG = 5"))
    assert "handle=" in out


def test_selectivity_guard_ignores_non_discriminating_term():
    # a term that appears on (almost) every line is not a needle -> keeps nothing extra
    lines = [f"compiling module_{i} ok" for i in range(50)]
    idx = relevant_lines(lines, frozenset({"compiling"}))
    assert idx == set(), "a term matching most lines must be dropped by the selectivity guard"


def test_additive_property_never_drops_a_base_keep():
    log = _big_log('File "x.py", line 5, in f')  # a traceback frame (base keep)
    base, _ = digest(log)
    widened, _ = digest(log, intent=frozenset({"artifact_3"}))
    for ln in base.splitlines():
        if not ln.startswith("<< +"):
            assert ln in widened, f"query intent dropped a previously-kept line: {ln!r}"


def test_extract_intent_pulls_tooluse_and_user_terms():
    messages = [
        {"role": "user", "content": "check the deadlock in scheduler.rs"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t",
                    "name": "grep",
                    "input": {"pattern": "acquire_lock"},
                }
            ],
        },
    ]
    terms = extract_intent(messages)
    assert "acquire_lock" in terms
    assert "scheduler.rs" in terms
    assert "deadlock" in terms
    assert "the" not in terms  # stopword


def test_terms_of_drops_short_and_stopwords():
    assert terms_of("the MAX_RETRIES ok") == {"max_retries"}


if __name__ == "__main__":
    test_grep_hit_survives_via_tool_use_intent()
    test_intent_off_still_folds_needle()
    test_selectivity_guard_ignores_non_discriminating_term()
    test_additive_property_never_drops_a_base_keep()
    test_extract_intent_pulls_tooluse_and_user_terms()
    test_terms_of_drops_short_and_stopwords()
    print("ok — query-aware salience keeps the needle, additively, on the live path")

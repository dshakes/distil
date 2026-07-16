"""Gemini generateContent adapter — path detection, compression, recovery, tokens.

Locks in Phase 2+3: distil compresses Google's generateContent shape
(contents/parts/functionResponse) reversibly, with recency carve-out, query-aware
intent, and output shaping — parity with the Anthropic/OpenAI adapters.
"""

from __future__ import annotations

from distil.adapters.gemini import (
    _extract_gemini_intent,
    _recent_gemini_verbatim_indices,
    compress_generate_request,
    count_tokens,
    is_gemini_path,
)
from distil.compress.recency import RECENCY_KEEP_TURNS
from distil.output import shape_request
from distil.shadow import decision_signature

# A tool output big enough to trigger the Tier-1 reversible digest (>= 6 lines).
_BIG_OUTPUT = "\n".join(f"row {i}: value_{i} status=ok detail=lorem ipsum dolor" for i in range(40))


def _req() -> dict:
    return {
        "contents": [
            {"role": "user", "parts": [{"text": "List the failing rows."}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "query_db", "args": {"q": "SELECT *"}}}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "query_db",
                            "response": {"output": _BIG_OUTPUT},
                        }
                    }
                ],
            },
        ],
    }


# --- path detection -------------------------------------------------------- #


def test_is_gemini_path_matches_generate_and_stream():
    assert is_gemini_path("/v1beta/models/gemini-1.5-pro:generateContent")
    assert is_gemini_path("/v1beta/models/gemini-2.0-flash:streamGenerateContent")
    assert is_gemini_path("/v1/models/gemini-1.5-pro:generateContent?key=abc123")
    assert is_gemini_path("/v1beta/models/gemini-1.5-pro:generateContent?alt=sse")


def test_is_gemini_path_rejects_other_paths():
    assert not is_gemini_path("/v1/messages")
    assert not is_gemini_path("/v1/chat/completions")
    assert not is_gemini_path("/v1beta/models")
    assert not is_gemini_path("/v1beta/models/gemini-1.5-pro:countTokens")


# --- compression + recovery ------------------------------------------------ #


def test_function_response_is_digested_and_recoverable():
    # Use the multi-turn fixture: _req() has only 2 user turns so both land inside
    # the RECENCY_KEEP_TURNS=2 window and are kept verbatim (correct behaviour).
    # The multi-turn fixture has older turns that are outside the window and DO get
    # digested — that is what this test covers.
    body = _multi_turn_req()
    new_body, store = compress_generate_request(body)

    # At least one functionResponse was replaced by a shorter digest...
    assert store.handles, "expected at least one digest handle from older turns"
    recovered = {store.expand(h) for h in store.handles}
    assert _BIG_OUTPUT in recovered  # ...and is byte-exactly recoverable.


def test_input_is_not_mutated():
    body = _req()
    before = body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"]
    compress_generate_request(body)
    assert body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"] == before


def test_model_text_is_never_rewritten():
    body = {
        "contents": [
            {"role": "model", "parts": [{"text": '{"a":  1,   "b": 2}'}]},
        ]
    }
    new_body, _ = compress_generate_request(body)
    # Model-authored text passes through byte-exact even though it is minifiable JSON.
    assert new_body["contents"][0]["parts"][0]["text"] == '{"a":  1,   "b": 2}'


def test_function_call_passed_through():
    body = _req()
    new_body, _ = compress_generate_request(body)
    assert new_body["contents"][1]["parts"][0]["functionCall"] == {
        "name": "query_db",
        "args": {"q": "SELECT *"},
    }


def test_no_op_returns_same_body_object():
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    new_body, store = compress_generate_request(body)
    assert new_body is body  # nothing compressible -> identity, no copy
    assert not store.handles


# --- token accounting ------------------------------------------------------ #


def test_count_tokens_drops_after_compression():
    # Use the multi-turn fixture so there are turns outside the recency window
    # that actually get compressed (see comment in test_function_response_is_digested…).
    body = _multi_turn_req()
    before = count_tokens(body)
    new_body, _ = compress_generate_request(body)
    after = count_tokens(new_body)
    assert before > 0
    assert after < before


# --- robustness (malformed but valid JSON must never crash) ---------------- #


def test_malformed_shapes_do_not_crash():
    for body in [
        {"contents": "not a list"},
        {"contents": [None, 1, "x"]},
        {"contents": [{"role": "user"}]},  # no parts
        {"contents": [{"role": "user", "parts": "nope"}]},
        {"contents": [{"role": "user", "parts": [None, {"text": 5}, {}]}]},
        {"contents": [{"role": "user", "parts": [{"functionResponse": {}}]}]},
        {},
    ]:
        new_body, store = compress_generate_request(body)
        assert isinstance(new_body, dict)
        assert count_tokens(body) >= 0


# --- recency carve-out ----------------------------------------------------- #


def _multi_turn_req() -> dict:
    """Three rounds: user→model→user (tool result)×3. Last RECENCY_KEEP_TURNS user
    turns must stay verbatim; earlier ones are eligible for Tier-1 digest."""
    turns = []
    for i in range(3):
        turns.append({"role": "user", "parts": [{"text": f"round {i}"}]})
        turns.append(
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "fetch", "args": {"id": i}}}],
            }
        )
        turns.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "fetch",
                            "response": {"output": _BIG_OUTPUT},
                        }
                    }
                ],
            }
        )
    return {"contents": turns}


def test_recency_indices_selects_last_k_user_turns():
    contents = _multi_turn_req()["contents"]
    # role:"user" turns are at indices 0, 2, 4, 6, 8 — last K of those are recent.
    user_idxs = [i for i, c in enumerate(contents) if c.get("role") == "user"]
    recent = _recent_gemini_verbatim_indices(contents, RECENCY_KEEP_TURNS)
    assert recent == set(user_idxs[-RECENCY_KEEP_TURNS:])


def test_recent_turns_stay_verbatim():
    """The last RECENCY_KEEP_TURNS user turns must not be digested."""
    body = _multi_turn_req()
    new_body, store = compress_generate_request(body)
    contents = new_body["contents"]
    # Find the last RECENCY_KEEP_TURNS user turns and verify their functionResponse
    # output is byte-exact (no digest stub).
    user_turn_indices = [i for i, c in enumerate(contents) if c.get("role") == "user"]
    recent_indices = set(user_turn_indices[-RECENCY_KEEP_TURNS:])
    for idx in recent_indices:
        turn = contents[idx]
        for part in turn["parts"]:
            if "functionResponse" in part:
                output = part["functionResponse"]["response"]["output"]
                assert output == _BIG_OUTPUT, (
                    f"recent turn {idx} was digested but must stay verbatim"
                )


def test_older_turns_are_compressed():
    """Turns outside the recency window must be digested (savings come from them)."""
    body = _multi_turn_req()
    new_body, store = compress_generate_request(body)
    # At least one handle was recorded — something was digested.
    assert store.handles, "expected at least one older turn to be digested"


# --- query-aware intent ---------------------------------------------------- #


def test_extract_gemini_intent_from_user_text():
    contents = [
        {"role": "user", "parts": [{"text": "find the authentication_token for user xyz"}]},
    ]
    terms = _extract_gemini_intent(contents)
    assert "authentication_token" in terms
    assert "xyz" in terms


def test_extract_gemini_intent_from_function_call():
    contents = [
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "grep_files", "args": {"pattern": "api_key"}}}],
        },
    ]
    terms = _extract_gemini_intent(contents)
    assert "grep_files" in terms
    assert "api_key" in terms


def test_extract_gemini_intent_empty_on_no_user_text():
    # A body with only model turns and no user text yields empty or only function terms.
    contents = [
        {"role": "model", "parts": [{"text": "I will help."}]},
    ]
    terms = _extract_gemini_intent(contents)
    # No functionCall, no user text — may be empty or small; must not raise.
    assert isinstance(terms, frozenset)


def test_extract_gemini_intent_degrades_gracefully_on_bad_shapes():
    for contents in [
        [],
        [None, 42, "oops"],
        [{"role": "user"}],  # no parts
        [{"role": "user", "parts": None}],
    ]:
        result = _extract_gemini_intent(contents)
        assert isinstance(result, frozenset)


# --- output shaping -------------------------------------------------------- #


def test_shape_request_gemini_creates_system_instruction():
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    out = shape_request(body, level="light", allow=True, shape="gemini")
    assert "systemInstruction" in out
    assert out["systemInstruction"]["parts"][0]["text"]
    assert "concise" in out["systemInstruction"]["parts"][0]["text"].lower()
    assert out["contents"] is body["contents"]  # input not mutated


def test_shape_request_gemini_appends_to_existing_system_instruction():
    body = {
        "contents": [],
        "systemInstruction": {"parts": [{"text": "You are a bot."}]},
    }
    out = shape_request(body, level="light", allow=True, shape="gemini")
    parts = out["systemInstruction"]["parts"]
    assert parts[0]["text"] == "You are a bot."
    assert "concise" in parts[1]["text"].lower()


def test_shape_request_gemini_auto_detection():
    # Auto mode: body with ``contents`` and no ``messages``/``system`` → Gemini path.
    body = {"contents": [{"role": "user", "parts": [{"text": "go"}]}]}
    out = shape_request(body, level="light", allow=True, shape="auto")
    assert "systemInstruction" in out
    assert "messages" not in out


def test_shape_request_gemini_noop_when_off_or_disallowed():
    body = {"contents": []}
    assert shape_request(body, level="off", allow=True, shape="gemini") is body
    assert shape_request(body, level="light", allow=False, shape="gemini") is body


# --- round-trip (compress then expand restores original) ------------------- #


def test_round_trip_compress_expand():
    # Multi-turn: older functionResponse blocks get digested; expand recovers them exactly.
    body = _multi_turn_req()
    _, store = compress_generate_request(body)
    assert store.handles, "older turns must produce at least one digest handle"
    for h in store.handles:
        assert store.expand(h) == _BIG_OUTPUT


# --- shadow decision extraction for Gemini --------------------------------- #


def test_decision_signature_gemini_function_call():
    resp = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "f", "args": {"x": 1}}}],
                }
            }
        ]
    }
    sig = decision_signature(resp)
    assert sig.startswith("tool:")
    # Same call -> same signature; different args -> different signature.
    assert sig == decision_signature(resp)
    resp2 = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "f", "args": {"x": 2}}}],
                }
            }
        ]
    }
    assert decision_signature(resp2) != sig


def test_decision_signature_gemini_text_and_none():
    text_resp = {"candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}}]}
    assert decision_signature(text_resp) == "text"
    assert decision_signature({"candidates": []}) == "none"
    assert decision_signature({}) == "none"

"""Transparent agent-callable expansion — the recoverable-compression loop."""

from __future__ import annotations

import json

from distil.expand import (
    EXPAND_TOOL_NAME,
    inject_expand_tool,
    resolve_expands,
    run_expand_loop,
)


class _Store:
    def __init__(self, mapping):
        self._m = mapping

    def expand(self, handle):
        return self._m[handle]


def test_inject_expand_tool_adds_it_once():
    body = inject_expand_tool({"messages": []})
    assert any(t["name"] == EXPAND_TOOL_NAME for t in body["tools"])
    # idempotent — never double-injected
    again = inject_expand_tool(body)
    assert sum(t["name"] == EXPAND_TOOL_NAME for t in again["tools"]) == 1
    # preserves an existing tool
    withtool = inject_expand_tool({"tools": [{"name": "get_logs"}], "messages": []})
    names = {t["name"] for t in withtool["tools"]}
    assert names == {"get_logs", EXPAND_TOOL_NAME}


def test_resolve_expands_returns_tool_results_and_signals():
    store = _Store({"a1b2c3d4": "the full original log with the load-bearing line"})
    resp = {
        "content": [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": EXPAND_TOOL_NAME,
                "input": {"handle": "a1b2c3d4"},
            }
        ]
    }
    signals = []
    out = resolve_expands(resp, store, on_signal=lambda h, t: signals.append((h, len(t))))
    assert out == [
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "the full original log with the load-bearing line",
        }
    ]
    assert signals == [("a1b2c3d4", len("the full original log with the load-bearing line"))]


def test_resolve_expands_none_when_no_expand_call():
    resp = {"content": [{"type": "text", "text": "done"}]}
    assert resolve_expands(resp, _Store({})) is None


def test_run_expand_loop_recovers_then_returns_final(tmp_path):
    store = _Store({"deadbeef": "ERROR root cause: tenant_id missing on line 42"})
    calls = {"n": 0}

    def post(body):
        # the model first asks to expand; after it receives the detail, it answers.
        calls["n"] += 1
        last = body["messages"][-1]
        if (
            last["role"] == "user"
            and isinstance(last["content"], list)
            and last["content"][0].get("type") == "tool_result"
        ):
            assert "tenant_id missing" in last["content"][0]["content"]  # got the recovered detail
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "fix line 42"}]}
        raise AssertionError("unexpected call shape")

    first = {
        "content": [
            {
                "type": "tool_use",
                "id": "tu_9",
                "name": EXPAND_TOOL_NAME,
                "input": {"handle": "deadbeef"},
            }
        ]
    }
    body = {"messages": [{"role": "user", "content": "investigate"}]}
    final = run_expand_loop(body, first, store, post, on_signal=None)
    assert final["content"][0]["text"] == "fix line 42"
    assert calls["n"] == 1  # exactly one recovery round-trip, invisible to the agent


def test_run_expand_loop_is_bounded(tmp_path):
    # a model that ALWAYS asks to expand must not spin forever
    store = _Store({"h": "x"})
    expand_resp = {
        "content": [
            {"type": "tool_use", "id": "t", "name": EXPAND_TOOL_NAME, "input": {"handle": "h"}}
        ]
    }
    n = {"c": 0}

    def post(body):
        n["c"] += 1
        return expand_resp

    final = run_expand_loop({"messages": []}, expand_resp, store, post, max_iters=3, on_signal=None)
    assert n["c"] == 3  # capped
    assert final is expand_resp


def test_inject_expand_tool_gemini_idempotent():
    from distil.expand import EXPAND_TOOL_NAME, inject_expand_tool_gemini

    body = inject_expand_tool_gemini({"contents": []})
    tools = body["tools"]
    names = [fd["name"] for t in tools for fd in t.get("functionDeclarations", [])]
    assert EXPAND_TOOL_NAME in names
    # idempotent — second inject doesn't duplicate
    again = inject_expand_tool_gemini(body)
    names2 = [fd["name"] for t in again["tools"] for fd in t.get("functionDeclarations", [])]
    assert names2.count(EXPAND_TOOL_NAME) == 1
    # preserves an existing functionDeclaration group
    existing = {"functionDeclarations": [{"name": "search_web", "parameters": {}}]}
    with_tool = inject_expand_tool_gemini({"contents": [], "tools": [existing]})
    all_names = [fd["name"] for t in with_tool["tools"] for fd in t.get("functionDeclarations", [])]
    assert "search_web" in all_names and EXPAND_TOOL_NAME in all_names


def test_run_expand_loop_gemini_recovers():
    from distil.expand import EXPAND_TOOL_NAME, run_expand_loop_gemini

    store = _Store({"deadbeef": "root cause: missing tenant_id on line 42"})
    calls = {"n": 0}

    def post(body):
        calls["n"] += 1
        last_user = body["contents"][-1]
        # After recovery, the functionResponse carries the original text.
        assert last_user["role"] == "user"
        got = last_user["parts"][0]["functionResponse"]["response"]["content"]
        assert "tenant_id" in got
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": "fixed"}]}}]}

    first = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": EXPAND_TOOL_NAME, "args": {"handle": "deadbeef"}}}
                    ],
                }
            }
        ]
    }
    body = {"contents": [{"role": "user", "parts": [{"text": "debug"}]}]}
    final = run_expand_loop_gemini(body, first, store, post, on_signal=None)
    assert final["candidates"][0]["content"]["parts"][0]["text"] == "fixed"
    assert calls["n"] == 1


def test_run_expand_loop_gemini_bounded():
    from distil.expand import EXPAND_TOOL_NAME, run_expand_loop_gemini

    store = _Store({"h": "x"})
    spin = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": EXPAND_TOOL_NAME, "args": {"handle": "h"}}}
                    ],
                }
            }
        ]
    }
    n = {"c": 0}

    def post(body):
        n["c"] += 1
        return spin

    run_expand_loop_gemini({"contents": []}, spin, store, post, max_iters=3, on_signal=None)
    assert n["c"] == 3


def test_signal_log_is_content_free(tmp_path):
    from distil.expand import record_signal

    p = tmp_path / "sig.jsonl"
    record_signal("abc12345", "x" * 1234, path=p)
    rec = json.loads(p.read_text().splitlines()[0])
    assert rec["handle"] == "abc12345" and rec["recovered_chars"] == 1234
    assert "content" not in rec and "text" not in rec  # numbers only, never content


def test_proxy_expand_loop_end_to_end(tmp_path, monkeypatch):
    """Full path: proxy digests a tool_result → fake model asks to expand the handle
    → proxy resolves it from the local store and re-queries → agent gets the final
    answer, never seeing the recovery round-trip."""
    import re
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from distil.proxy import build_handler

    # Isolate the learning store so accumulated real signals (or this test's own
    # writes) can't flip the keep-policy to byte-exact and starve the expand path.
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))

    class _Upstream(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            already_expanded = any(
                m.get("role") == "assistant"
                and isinstance(m.get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("name") == "distil_expand" for b in m["content"]
                )
                for m in body["messages"]
            )
            if already_expanded:  # second call: the model now has the recovered detail
                recovered = body["messages"][-1]["content"][0]["content"]
                assert "load-bearing line 27" in recovered  # the elided middle came back
                resp = {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "done with detail"}],
                }
            else:  # first call: ask to expand the handle we see in the digested body
                m = re.search(r"handle=([0-9a-f]{8})", json.dumps(body))
                resp = {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu1",
                            "name": "distil_expand",
                            "input": {"handle": m.group(1) if m else "00000000"},
                        }
                    ],
                }
            out = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):  # noqa: ANN002
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    handler = build_handler(f"http://127.0.0.1:{up.server_address[1]}", expand=True)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        big = "\n".join(
            ("load-bearing line 27" if i == 27 else f"verbose log line {i}") for i in range(40)
        )
        payload = json.dumps(
            {
                "model": "claude-opus-4-8",
                "messages": [
                    {"role": "user", "content": "investigate"},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t", "content": big}],
                    },
                    # Later turns keep the tool_result out of the recency-exempt
                    # window so it still digests and the expand loop has a handle.
                    {"role": "user", "content": "next"},
                    {"role": "user", "content": "next"},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.headers.get("x-distil-expanded") == "1"  # recovery happened, transparently
            final = json.loads(r.read())
        assert final["content"][0]["text"] == "done with detail"  # agent got the resolved answer
    finally:
        proxy.shutdown()
        up.shutdown()


def test_unrecoverable_handle_is_not_logged_as_a_recovery(tmp_path):
    """F3, post-compression access failure: the store cannot return a handle it
    was asked for. Fail-open still hands the agent a placeholder, but nothing
    downstream may book that as a successful expansion — the trainers key
    positives off a bare .get("handle"), so a miss written under that key would
    teach the keep-model that the block it lost was safe to drop."""
    import json

    from distil.expand import is_miss, record_signal, resolve_expands

    resp = {
        "content": [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": EXPAND_TOOL_NAME,
                "input": {"handle": "deadbeef"},
            }
        ]
    }
    out = resolve_expands(resp, _Store({}))  # handle evicted / aged out
    assert out is not None
    text = out[0]["content"]
    assert is_miss(text), "the agent must be told recovery failed, not handed silence"

    sig = tmp_path / "expand-signals.jsonl"
    record_signal("deadbeef", text, path=sig)
    record_signal("a1b2c3d4", "the real original", path=sig)
    recs = [json.loads(line) for line in sig.read_text().splitlines()]

    # what the flywheel and assoc trainers actually read
    positives = {r.get("handle") for r in recs if isinstance(r.get("handle"), str)}
    assert positives == {"a1b2c3d4"}, "a failed recovery must not be a positive label"
    misses = [r for r in recs if r.get("miss")]
    assert [r["miss"] for r in misses] == ["deadbeef"]
    assert "recovered_chars" not in misses[0], "nothing was recovered"


def test_mixed_turn_is_not_intercepted_messages_api():
    """A turn carrying a client tool_use alongside distil_expand is handed back as-is.

    Resolving only the expand block would replay the assistant message with the
    client's tool_use unanswered, and the continuation's stop_reason would mask the
    real ``tool_use`` — so the agent never runs the tool it asked for.
    """
    from distil.expand import has_client_tool_use, run_expand_loop

    store = _Store({"deadbeef": "the original"})
    mixed = {
        "content": [
            {"type": "text", "text": "Editing now."},
            {"type": "tool_use", "id": "tu_edit", "name": "Edit", "input": {"file_path": "/x"}},
            {
                "type": "tool_use",
                "id": "tu_exp",
                "name": EXPAND_TOOL_NAME,
                "input": {"handle": "deadbeef"},
            },
        ],
        "stop_reason": "tool_use",
    }
    assert has_client_tool_use(mixed) is True

    posts = []

    def post(body):
        posts.append(body)
        return {"content": [{"type": "text", "text": "should never run"}]}

    out = run_expand_loop({"messages": []}, mixed, store, post)
    assert out is mixed, "the mixed turn must be returned untouched"
    assert posts == [], "a mixed turn must not trigger a re-query"
    assert out["stop_reason"] == "tool_use", "the client needs tool_use to execute"


def test_expand_only_turn_still_resolves_messages_api():
    """The mixed-turn guard must not disable ordinary expansion."""
    from distil.expand import has_client_tool_use, run_expand_loop

    store = _Store({"deadbeef": "RECOVERED"})
    expand_only = {
        "content": [
            {
                "type": "tool_use",
                "id": "tu_exp",
                "name": EXPAND_TOOL_NAME,
                "input": {"handle": "deadbeef"},
            }
        ]
    }
    assert has_client_tool_use(expand_only) is False

    final = {"content": [{"type": "text", "text": "done"}]}
    posts = []

    def post(body):
        posts.append(body)
        return final

    out = run_expand_loop({"messages": []}, expand_only, store, post)
    assert out is final
    assert len(posts) == 1, "expand-only turns must still re-query"
    recovered = posts[0]["messages"][-1]["content"][0]["content"]
    assert recovered == "RECOVERED"


def test_mixed_turn_is_not_intercepted_responses_api():
    """Same guard on the OpenAI Responses shape."""
    from distil.expand import run_expand_loop_responses

    store = _Store({"deadbeef": "the original"})
    mixed = {
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "apply_patch", "arguments": "{}"},
            {
                "type": "function_call",
                "call_id": "c2",
                "name": EXPAND_TOOL_NAME,
                "arguments": json.dumps({"handle": "deadbeef"}),
            },
        ]
    }
    posts = []

    def post(body):
        posts.append(body)
        return {"output": []}

    out = run_expand_loop_responses({"input": []}, mixed, store, post)
    assert out is mixed
    assert posts == []


def test_mixed_turn_is_not_intercepted_gemini():
    """Same guard on the Gemini contents/parts shape."""
    from distil.expand import run_expand_loop_gemini

    store = _Store({"deadbeef": "the original"})
    mixed = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "write_file", "args": {"path": "/x"}}},
                        {
                            "functionCall": {
                                "name": EXPAND_TOOL_NAME,
                                "args": {"handle": "deadbeef"},
                            }
                        },
                    ]
                }
            }
        ]
    }
    posts = []

    def post(body):
        posts.append(body)
        return {"candidates": []}

    out = run_expand_loop_gemini({"contents": []}, mixed, store, post)
    assert out is mixed
    assert posts == []

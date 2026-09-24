"""Cold-point recompression (ADR 0014).

The whole feature is one invariant: a byte may change only on a turn where the provider
cache is already lost, and after that the new form is byte-stable for the rest of the
lineage. The proxy tests below drive a real threaded proxy through a session with a
simulated idle gap and assert on the bytes the upstream actually received.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from distil import coldpoint, prefixreplay
from distil.adapters.anthropic import cold_candidates, compress_messages, evicted_stub

_STUB = re.compile(
    r"<<distil evicted older tool output \(\d+ lines\); distil_expand handle=([0-9a-f]{8}) recovers it>>"
)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    coldpoint.reset()
    prefixreplay.reset()
    monkeypatch.delenv("DISTIL_COLD_POINT", raising=False)
    yield
    coldpoint.reset()
    prefixreplay.reset()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(coldpoint, "_clock", lambda: now[0])
    return now


# --------------------------------------------------------------------------- fixtures


def _log(tag: str, n: int = 40) -> str:
    return "\n".join(
        f"{tag} line {i}: status=ok elapsed={i * 7}ms worker=w{i % 5} path=/srv/app/mod_{i}.py"
        for i in range(n)
    )


def _use(tid: str, name: str, inp: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
    }


def _res(tid: str, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def _conv(rounds: int, *, ttl: str | None = None) -> dict[str, Any]:
    """A Claude-Code-shaped session: one bash round per turn, breakpoint on the newest
    block (which moves every turn), a cached system prompt."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "fix the build"}]}
    ]
    for k in range(rounds):
        msgs += [
            _use(f"toolu_{k}", "Bash", {"command": f"make target{k}"}),
            _res(f"toolu_{k}", _log(f"run{k}")),
        ]
    cc: dict[str, Any] = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    msgs[-1]["content"][-1]["cache_control"] = cc
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a coding agent.", "cache_control": dict(cc)}],
        "messages": msgs,
    }


def _strip_marks(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_marks(v) for k, v in node.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_strip_marks(x) for x in node]
    return node


def _wire(msg: Any) -> str:
    return json.dumps(_strip_marks(msg), separators=(",", ":"), ensure_ascii=False)


def _plan(key: str, body: dict[str, Any], cands: frozenset[str] = frozenset({"x"}), **kw: Any):
    p = coldpoint.plan(key, body, body["messages"], lambda: cands, **kw)
    coldpoint.end(key)
    return p


# --------------------------------------------------------------------------- state machine


def test_first_seen_warm_and_cold(clock) -> None:
    body = _conv(3)
    assert _plan("k", body).reason == "first-seen"
    clock[0] += 200
    body = _conv(4)
    assert _plan("k", body).evict == frozenset(), "warm turn evicted"
    clock[0] += 300 + coldpoint.MARGIN_S + 1
    p = _plan("k", _conv(5))
    assert (p.reason, p.evict, p.fresh) == ("cold", frozenset({"x"}), 1)
    # Once evicted, always evicted — on warm turns too, with nothing new decided.
    clock[0] += 5
    p = _plan("k", _conv(6), frozenset({"x", "y"}))
    assert (p.reason, p.evict, p.fresh) == ("warm", frozenset({"x"}), 0)


def test_the_margin_is_respected(clock) -> None:
    _plan("k", _conv(3))
    clock[0] += 300 + coldpoint.MARGIN_S  # exactly TTL + margin: not certain yet
    assert _plan("k", _conv(4)).reason == "warm"


def test_a_one_hour_ttl_is_honoured(clock) -> None:
    _plan("k", _conv(3, ttl="1h"))
    clock[0] += 30 * 60
    assert _plan("k", _conv(4, ttl="1h")).reason == "warm"
    clock[0] += 3600 + coldpoint.MARGIN_S + 1
    assert _plan("k", _conv(5, ttl="1h")).reason == "cold"


def test_a_one_hour_ttl_seen_once_outlives_the_request_that_asked(clock) -> None:
    """The 1h entry written earlier is still alive when a later turn asks for 5m."""
    _plan("k", _conv(3, ttl="1h"))
    clock[0] += 10 * 60
    assert _plan("k", _conv(4)).reason == "warm"


def test_an_unknown_ttl_never_evicts(clock) -> None:
    _plan("k", _conv(3, ttl="30m"))
    clock[0] += 10 * 3600
    assert _plan("k", _conv(4)).reason == "unknown-ttl"
    assert coldpoint.request_ttl({"system": [{"cache_control": {"ttl": ["1h"]}}]}) == float("inf")


def test_a_request_in_flight_keeps_the_lineage_warm(clock) -> None:
    """A long stream or an expand re-query refreshes the cache after the forward."""
    coldpoint.plan("k", _conv(3), _conv(3)["messages"], lambda: frozenset({"x"}))
    clock[0] += 3600
    assert _plan("k", _conv(4)).reason == "inflight"
    coldpoint.end("k")
    # ...and `end` itself is a touch: the gap restarts from when the request finished.
    clock[0] += 100
    assert _plan("k", _conv(5)).reason == "warm"


def test_a_shadow_replay_holds_the_lineage_in_flight(clock) -> None:
    _plan("k", _conv(3))
    coldpoint.begin("k")
    clock[0] += 3600
    assert _plan("k", _conv(4)).reason == "inflight"


def test_two_conversations_under_one_key_never_evict(clock) -> None:
    """Parallel subagents (or a fork) sharing a lineage key: the second history does not
    extend the first, so the lineage goes ambiguous for good."""
    a, b = _conv(4), _conv(4)
    b["messages"][-1]["content"][0]["content"] = _log("other-branch")
    _plan("k", a)
    _plan("k", b)
    clock[0] += 3600
    assert _plan("k", _conv(5)).reason == "ambiguous"
    clock[0] += 3600
    assert _plan("k", _conv(6)).reason == "ambiguous", "ambiguity must be sticky"


def test_a_rewound_history_is_ambiguous(clock) -> None:
    _plan("k", _conv(5))
    clock[0] += 3600
    assert _plan("k", _conv(3)).reason == "ambiguous"


def test_ambiguity_keeps_what_was_already_evicted(clock) -> None:
    """Stability beats novelty: a set that is on the wire stays on the wire."""
    _plan("k", _conv(3))
    clock[0] += 3600
    assert _plan("k", _conv(4)).evict == {"x"}
    assert _plan("k", _conv(2)).evict == {"x"}


def test_held_applies_but_decides_nothing(clock) -> None:
    _plan("k", _conv(3))
    clock[0] += 3600
    p = _plan("k", _conv(4), held=True)
    assert (p.reason, p.evict) == ("held", frozenset())


def test_a_failing_candidate_choice_evicts_nothing(clock) -> None:
    _plan("k", _conv(3))
    clock[0] += 3600

    def boom() -> frozenset[str]:
        raise RuntimeError("tokenizer blew up")

    p = coldpoint.plan("k", _conv(4), _conv(4)["messages"], boom)
    assert (p.reason, p.evict, p.fresh) == ("cold", frozenset(), 0)


def test_state_is_bounded(clock, monkeypatch) -> None:
    monkeypatch.setattr(coldpoint, "_MAX_LINEAGES", 4)
    for i in range(10):
        _plan(f"k{i}", _conv(2))
    assert len(coldpoint._STATES) == 4
    monkeypatch.setattr(coldpoint, "_MAX_EXPANDED", 2)
    for h in ("a", "b", "c"):
        coldpoint.note_expanded(h)
    assert coldpoint.expanded() == {"b", "c"}


# --------------------------------------------------------------------------- the adapter


def test_candidates_respect_every_keep_policy() -> None:
    read = "\n".join(f"def f{i}(): return {i}  # body" for i in range(60))
    quoted = _log("quoted")
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        _use("bash_old", "Bash", {"command": "make"}),
        _res("bash_old", _log("old")),
        _use("read_1", "Read", {"file_path": "/app/f.py"}),
        _res("read_1", read),
        _use(
            "edit_1",
            "Edit",
            {"file_path": "/app/f.py", "old_string": read.splitlines()[3], "new_string": "x"},
        ),
        _res("edit_1", "ok"),
        _use("short", "Bash", {"command": "echo"}),
        _res("short", "one line"),
        _use("learned", "Bash", {"command": "cat keep"}),
        _res("learned", "KEEPME\n" + _log("learned")),
        _use("expanded", "Bash", {"command": "make x"}),
        _res("expanded", _log("expanded")),
        _use("quoted", "Bash", {"command": "grep -r x"}),
        _res("quoted", quoted),
        _use(
            "edit_2",
            "Edit",
            {"file_path": "/app/g.py", "old_string": quoted.splitlines()[5], "new_string": "y"},
        ),
        _res("edit_2", "ok"),
        _use("recent_1", "Bash", {"command": "make a"}),
        _res("recent_1", _log("recent1")),
    ]
    from distil.adapters.anthropic import _handle

    got = cold_candidates(
        msgs,
        keep=lambda t: t.startswith("KEEPME"),
        exclude_handles=frozenset({_handle(_log("expanded"))}),
    )
    assert got == {"bash_old"}, got


def test_the_stub_is_a_pure_function_of_the_content() -> None:
    text = _log("x")
    assert evicted_stub(text, "abcd1234") == evicted_stub(text, "abcd1234")
    assert _STUB.fullmatch(evicted_stub(text, "abcd1234"))


def test_evicted_blocks_are_recoverable_through_the_restore_store() -> None:
    from distil.mcp_server import load_restore

    msgs = _conv(4)["messages"]
    out, store = compress_messages(msgs, evict=frozenset({"toolu_0"}))
    stub = out[2]["content"][0]["content"]
    m = _STUB.fullmatch(stub)
    assert m, stub
    assert store.expand(m.group(1)) == _log("run0")
    assert load_restore(m.group(1)) == _log("run0"), "not on disk: lost on restart"


def test_an_exact_quote_result_is_never_evicted_even_if_asked() -> None:
    """If an Edit comes to depend on an evicted block later, the exemption wins."""
    read = "\n".join(f"line {i} of the file" for i in range(60))
    msgs = [
        {"role": "user", "content": "go"},
        _use("r", "Read", {"file_path": "/a.py"}),
        _res("r", read),
        _use(
            "e",
            "Edit",
            {"file_path": "/a.py", "old_string": "line 7 of the file", "new_string": "z"},
        ),
        _res("e", "ok"),
        {"role": "user", "content": "next"},
        {"role": "user", "content": "and next"},
    ]
    out, _ = compress_messages(msgs, evict=frozenset({"r"}))
    assert out[2]["content"][0]["content"] == read


def test_verbatim_and_recent_ignore_evict() -> None:
    msgs = _conv(3)["messages"]
    out, _ = compress_messages(msgs, verbatim=True, evict=frozenset({"toolu_0"}))
    assert "distil evicted" not in json.dumps(out)
    # No cache_control → the plain last-k window: the newest result is recent.
    plain = copy.deepcopy(msgs)
    plain[-1]["content"][-1].pop("cache_control")
    out, _ = compress_messages(plain, evict=frozenset({"toolu_2"}))
    assert "distil evicted" not in json.dumps(out[-1])


def test_eviction_is_censused() -> None:
    from distil.adapters.anthropic import take_census

    compress_messages(_conv(3)["messages"], evict=frozenset({"toolu_0"}))
    assert (take_census() or {}).get("tool_result_evicted", 0) > 0


# --------------------------------------------------------------------------- through the proxy


class _Upstream(BaseHTTPRequestHandler):
    seen: list[bytes] = []

    def do_POST(self):  # noqa: N802
        type(self).seen.append(self.rfile.read(int(self.headers.get("content-length", 0))))
        payload = json.dumps(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-test",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def proxy():
    _Upstream.seen = []
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    servers = [up]

    def make(**kw: Any) -> int:
        from distil.proxy import build_handler

        px = ThreadingHTTPServer(
            ("127.0.0.1", 0), build_handler(f"http://127.0.0.1:{up.server_address[1]}", **kw)
        )
        threading.Thread(target=px.serve_forever, daemon=True).start()
        servers.append(px)
        return px.server_address[1]

    yield make
    for s in servers:
        s.shutdown()


def _post(port: int, body: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-test"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
        headers = {k.lower(): v for k, v in r.headers.items()}
    return headers, json.loads(_Upstream.seen[-1])["messages"]


def test_byte_stable_after_the_cold_point(proxy, clock) -> None:
    """The acceptance test. Turns before the gap evict nothing; the cold turn evicts the
    older results; every later turn forwards the cold turn's prefix byte-for-byte — and
    never evicts anything newer, however old it gets while the cache stays warm."""
    port = proxy()
    h, _ = _post(port, _conv(3))
    assert h["x-distil-cold"] == "first-seen"
    clock[0] += 120
    h, warm = _post(port, _conv(4))
    assert h["x-distil-cold"] == "warm"
    assert "distil evicted" not in json.dumps(warm), "evicted on a warm turn"

    clock[0] += 20 * 60
    h, cold = _post(port, _conv(5))
    assert (h["x-distil-cold"], h["x-distil-cold-evicted"]) == ("cold", "3")
    evicted = [i for i, m in enumerate(cold) if "distil evicted" in json.dumps(m)]
    assert evicted == [2, 4, 6], "wrong blocks evicted (recent turns must stay)"

    later = []
    for rounds in range(6, 10):
        clock[0] += 60
        h, fwd = _post(port, _conv(rounds))
        assert h["x-distil-cold"] == "warm"
        later.append(fwd)
    for fwd in later:
        drift = [i for i in range(len(cold)) if _wire(fwd[i]) != _wire(cold[i])]
        assert not drift, f"prefix after the cold point moved at {drift}"
    for fwd in later:
        assert [i for i, m in enumerate(fwd) if "distil evicted" in json.dumps(m)] == [2, 4, 6]


def test_the_cold_turn_is_recoverable_and_booked(proxy, clock, monkeypatch, tmp_path) -> None:
    from distil import ledger
    from distil.mcp_server import load_restore

    monkeypatch.setenv("DISTIL_SESSION", "s-cold")
    port = proxy()
    _post(port, _conv(3))
    clock[0] += 3600
    h, cold = _post(port, _conv(4))
    handles = _STUB.findall(json.dumps(cold).replace("\\n", "\n"))
    assert handles
    assert {load_restore(x) for x in handles} == {_log(f"run{k}") for k in range(len(handles))}
    assert int(h["x-distil-tokens-saved"]) > 0

    path = ledger.session_requests_path()
    assert path is not None
    import time as _t

    for _ in range(50):
        if path.exists() and len(path.read_text().splitlines()) >= 2:
            break
        _t.sleep(0.05)
    rec = json.loads(path.read_text().splitlines()[-1])
    assert rec["cold"] == "cold" and rec["cold_evicted"] == len(handles)
    assert rec["census"].get("tool_result_evicted", 0) > 0
    assert rec["mode"] == "digest"


def test_the_streaming_path_evicts_too(proxy, clock) -> None:
    port = proxy()
    _post(port, _conv(3))
    clock[0] += 3600
    body = _conv(4)
    body["stream"] = True
    h, fwd = _post(port, body)
    assert h.get("x-distil-cold", "cold") == "cold"
    assert "distil evicted" in json.dumps(fwd)


@pytest.mark.parametrize(
    "kw,env",
    [
        ({"cold_point": False}, None),
        ({}, "0"),
        ({"lossless_only": True}, None),
        ({"verbatim": True}, None),
        ({"session_delta": True}, None),
    ],
    ids=["flag", "env", "lossless-only", "verbatim", "session-delta"],
)
def test_off_means_untouched(proxy, clock, monkeypatch, kw, env) -> None:
    if env is not None:
        monkeypatch.setenv("DISTIL_COLD_POINT", env)
    port = proxy(**kw)
    _post(port, _conv(3))
    clock[0] += 3600
    h, fwd = _post(port, _conv(4))
    assert "x-distil-cold" not in h
    assert "distil evicted" not in json.dumps(fwd)


def test_a_restarted_proxy_does_nothing(proxy, clock) -> None:
    """No state = no knowledge of when the provider last saw the prefix."""
    port = proxy()
    _post(port, _conv(3))
    coldpoint.reset()  # what a hot-swap or restart looks like from inside
    clock[0] += 3600
    h, fwd = _post(port, _conv(4))
    assert h["x-distil-cold"] == "first-seen"
    assert "distil evicted" not in json.dumps(fwd)


def test_a_broken_planner_forwards_as_usual(proxy, clock, monkeypatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("state corrupted")

    monkeypatch.setattr(coldpoint, "plan", boom)
    port = proxy()
    h, fwd = _post(port, _conv(3))
    assert "x-distil-cold" not in h
    assert h["x-distil-compressed"] == "1"
    assert "distil evicted" not in json.dumps(fwd)


def test_two_credentials_never_share_a_lineage(proxy, clock) -> None:
    port = proxy()
    _post(port, _conv(3))
    clock[0] += 3600
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(_conv(4)).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-other"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.headers["x-distil-cold"] == "first-seen"

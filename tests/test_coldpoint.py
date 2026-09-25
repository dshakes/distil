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
        coldpoint.note_expanded("acct\0", h)
    assert coldpoint.expanded("acct\0") == {"b", "c"}
    assert coldpoint.expanded("other\0") == frozenset(), "expansions leaked across scopes"


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


# --------------------------------------------------------------------------- review round


def test_the_clock_counts_sleep(monkeypatch) -> None:
    """Behaviour, not configuration: whatever clock is picked must carry a sleep through.
    The sleep is simulated by the injected `gettime`, never by patching the process's
    real clock."""
    import time

    # The constant is only an id handed to the injected gettime; Windows has none, so
    # supply one (raising=False) rather than depend on the host interpreter.
    monkeypatch.setattr(time, "CLOCK_MONOTONIC", getattr(time, "CLOCK_MONOTONIC", 1), raising=False)
    wall = [100.0]
    clk = coldpoint._pick_clock("darwin", lambda cid: wall[0])
    before = clk()
    wall[0] += 3600  # the lid was closed for an hour
    assert clk() - before == 3600
    # ...and so a lineage idle across it is cold.
    assert clk() - before > coldpoint.TTL_DEFAULT_S + coldpoint.MARGIN_S

    def unsupported(cid: int) -> float:
        raise OSError("clock not supported")

    assert coldpoint._pick_clock("darwin", unsupported) is time.monotonic
    assert coldpoint._pick_clock("darwin", None) is time.monotonic

    real = coldpoint._pick_clock()
    a = real()
    time.sleep(0.01)
    assert real() > a, "the picked clock does not advance"


def test_reapplying_never_inflates() -> None:
    """A set can outlive the content it was chosen for; a stub is sent only if smaller."""
    msgs = _conv(3)["messages"]
    msgs[2]["content"][0]["content"] = "ok"
    out, _ = compress_messages(msgs, evict=frozenset({"toolu_0"}))
    assert out[2]["content"][0]["content"] == "ok"


def test_the_evicted_set_survives_a_restart(proxy, clock) -> None:
    """A hot-swap (every upgrade) must not un-evict a warm stubbed prefix."""
    import os
    import stat

    port = proxy()
    _post(port, _conv(3))
    clock[0] += 3600
    _, cold = _post(port, _conv(4))
    assert "distil evicted" in json.dumps(cold)
    path = coldpoint._persist_path()
    assert path.exists()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "run0" not in path.read_text(), "content reached the state file"

    coldpoint.reset()  # the new worker: empty memory, same disk
    prefixreplay.reset()
    clock[0] += 30
    h, warm = _post(port, _conv(5))
    assert h["x-distil-cold"] == "first-seen"
    drift = [i for i in range(len(cold)) if _wire(warm[i]) != _wire(cold[i])]
    assert not drift, f"restart un-evicted the prefix at {drift}"


def test_a_token_refresh_does_not_fork_the_lineage(proxy, clock) -> None:
    """Claude Code's OAuth bearer refreshes mid-session; the lineage must not notice."""

    def post(token: str, body: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
        return hdrs, json.loads(_Upstream.seen[-1])["messages"]

    port = proxy()
    post("oauth-1", _conv(3))
    clock[0] += 3600
    _, cold = post("oauth-1", _conv(4))
    clock[0] += 30
    h, warm = post("oauth-2-refreshed", _conv(5))
    assert h["x-distil-cold"] == "warm"
    assert not [i for i in range(len(cold)) if _wire(warm[i]) != _wire(cold[i])]


def test_persistence_is_bounded_and_fails_open(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(coldpoint, "_MAX_PERSISTED", 3)
    for i in range(5):
        coldpoint._persist(f"k{i}", frozenset({f"t{i}"}))
    coldpoint._persist("k2", frozenset({"u"}))  # merges, and refreshes its LRU slot
    data = coldpoint._parse(coldpoint._persist_path())
    assert data is not None
    assert list(data) == ["k3", "k4", "k2"] and set(data["k2"]) == {"t2", "u"}

    # A file that cannot be parsed is "no information", not "no state": the cached copy
    # stays, so the next write cannot erase every other lineage's set.
    coldpoint._persist_path().write_text("{not json")
    assert coldpoint._load("k2") == {"t2", "u"}
    coldpoint._persist("k9", frozenset({"z"}))
    healed = coldpoint._parse(coldpoint._persist_path())
    assert healed is not None and set(healed["k2"]) == {"t2", "u"}, "a bad read erased state"

    coldpoint._persist_path().write_text('{"lineages": {"k": ["a", 3], "bad": 1}}')
    assert coldpoint._load("k") == {"a"}

    # An unwritable home costs the optimisation, never the request.
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path / "file-not-dir"))
    (tmp_path / "file-not-dir").write_text("x")
    coldpoint._persist("k", frozenset({"a"}))
    assert coldpoint._load("k") == frozenset()


def test_first_seen_lineages_parse_the_file_once(monkeypatch, tmp_path) -> None:
    """Title-gen, subagents and quota pings are all first-seen lineages: each costs a
    stat, and the file is parsed again only when it actually changed."""
    import os

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    coldpoint._persist("k", frozenset({"a"}))
    coldpoint.reset()  # a fresh process: nothing cached
    parses = []
    real = coldpoint._parse
    monkeypatch.setattr(coldpoint, "_parse", lambda p: parses.append(p) or real(p))

    for i in range(50):
        coldpoint._load(f"side-request-{i}")
    assert coldpoint._load("k") == {"a"}
    assert len(parses) == 1, f"{len(parses)} parses for 51 first-seen lineages"

    # Another process (the old worker during a hot-swap) writes: re-parse, once.
    path = coldpoint._persist_path()
    path.write_text('{"version":1,"lineages":{"k":["a","b"],"other":["c"]}}')
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert coldpoint._load("k") == {"a", "b"}
    assert coldpoint._load("other") == {"c"}
    assert len(parses) == 2

    # Our own write refreshes the cache without a parse.
    coldpoint._persist("k", frozenset({"d"}))
    assert coldpoint._load("k") == {"a", "b", "d"}
    assert len(parses) == 2


def test_ids_per_lineage_are_capped_most_recent_kept(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(coldpoint, "_MAX_PERSISTED_IDS", 4)
    coldpoint._persist("k", frozenset({"a1", "a2", "a3"}))
    coldpoint._persist("k", frozenset({"a1", "a2", "a3", "b1", "b2"}))
    data = coldpoint._parse(coldpoint._persist_path())
    assert data is not None and data["k"] == ("a2", "a3", "b1", "b2")
    mtime = coldpoint._persist_path().stat().st_mtime_ns
    coldpoint._persist("k", frozenset({"b2"}))  # nothing new: no write at all
    assert coldpoint._persist_path().stat().st_mtime_ns == mtime


def test_a_persisted_set_is_reapplied_but_nothing_new_is_decided(clock) -> None:
    coldpoint._persist("k", frozenset({"old"}))
    p = _plan("k", _conv(3), frozenset({"new"}))
    assert (p.reason, p.evict) == ("first-seen", frozenset({"old"}))


def test_account_scope_ignores_the_bearer_token() -> None:
    assert coldpoint.account_scope({"Authorization": "Bearer a"}) == ""
    assert coldpoint.account_scope({"Authorization": "Bearer b"}) == ""
    one = coldpoint.account_scope({"x-api-key": "k1"})
    assert one and one != coldpoint.account_scope({"x-api-key": "k2"})


def test_ttl_is_read_from_every_marker_position() -> None:
    """A message-level marker counts as much as a block-level one, and a malformed
    message is skipped rather than raising on the request path."""
    body = {
        "messages": [
            "not-a-message",
            {"role": "user", "content": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]
    }
    assert coldpoint.request_ttl(body) == 3600.0
    assert coldpoint.request_ttl({"messages": ["junk"]}) == coldpoint.TTL_DEFAULT_S


def test_a_file_that_vanishes_before_the_read_is_empty_not_unreadable(tmp_path) -> None:
    """stat → replace → read can race: a missing file is genuinely no state (empty),
    unlike a read error, which is no information (None, keep the cache)."""
    assert coldpoint._parse(tmp_path / "gone.json") == {}
    (tmp_path / "dir.json").mkdir()
    assert coldpoint._parse(tmp_path / "dir.json") is None


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_an_interpreter_without_clock_constants_falls_back(monkeypatch, platform) -> None:
    """Windows' `time` has no CLOCK_* constants: the pick is time.monotonic, not a raise."""
    import time

    monkeypatch.delattr(time, "CLOCK_MONOTONIC", raising=False)
    monkeypatch.delattr(time, "CLOCK_BOOTTIME", raising=False)
    assert coldpoint._pick_clock(platform, lambda cid: 42.0) is time.monotonic


def test_linux_picks_a_boot_clock_when_the_platform_has_one() -> None:
    import time

    picked = coldpoint._pick_clock("linux", lambda cid: 42.0)
    if hasattr(time, "CLOCK_BOOTTIME"):
        assert picked() == 42.0
    else:  # no CLOCK_BOOTTIME on this interpreter: the plain monotonic clock, not a guess
        assert picked is time.monotonic

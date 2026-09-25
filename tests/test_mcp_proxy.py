"""distil mcp proxy: MCP protocol conformance, lazy unlock flow, relays, fail-open, stdio."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
from pathlib import Path

import pytest

from distil import cli
from distil.mcpproxy import events, fakeserver, levels, proxy

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"},
    },
}


class FakeBackend:
    """An in-process MCP server backed by a vendored fixture."""

    def __init__(
        self,
        name: str,
        fixture: str | None = None,
        *,
        caps=None,
        instructions=None,
        init_error=False,
    ):
        self.name = name
        self.fixture = fakeserver.load_fixture(fixture or name)
        self.caps = caps if caps is not None else {"tools": {}}
        self.instructions = instructions
        self.init_error = init_error
        self.on_message = lambda msg: None
        self.requests: list[tuple[str, object, object]] = []
        self.notes: list[tuple[str, object]] = []
        self.sent: list[dict] = []
        self.closed = False

    def request(self, method, params=None, *, id=None):
        self.requests.append((method, params, id))
        if method == "initialize":
            if self.init_error:
                return {"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": "nope"}}
            res = {
                "protocolVersion": "2025-06-18",
                "capabilities": self.caps,
                "serverInfo": {"name": self.name},
            }
            if self.instructions:
                res["instructions"] = self.instructions
            return {"jsonrpc": "2.0", "id": id, "result": res}
        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {"resources": [{"uri": f"{self.name}://r"}]},
            }
        return fakeserver.handle(
            self.fixture,
            {
                "jsonrpc": "2.0",
                "id": id if id is not None else "x",
                "method": method,
                "params": params,
            },
        )

    def notify(self, method, params=None):
        self.notes.append((method, params))

    def send(self, msg):
        self.sent.append(msg)

    def close(self):
        self.closed = True


def make(*backends, level="L0", results=True, tmp_path=None, **kw):
    store: dict[str, str] = {}
    px = proxy.Proxy(
        {b.name: b for b in backends},
        level=level,
        results=results,
        session="s1",
        state_root=tmp_path,
        log=events.EventLog("s1", (tmp_path / "events.jsonl") if tmp_path else None),
        record=lambda h, t: store.setdefault(h, t) == t,
        expand=store.get,
        **kw,
    )
    emitted: list[dict] = []
    px.emit = emitted.append
    return px, store, emitted


def call(px, name, args=None, msg_id=7):
    return px.handle(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args or {}},
        }
    )


def tools(px):
    return px.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})[0]["result"]["tools"]


# ---------------------------------------------------------------- initialize / list


def test_initialize_advertises_list_changed_and_merges(tmp_path):
    a = FakeBackend("git", caps={"tools": {}, "logging": {}}, instructions="use git")
    b = FakeBackend("time", caps={"tools": {}, "prompts": {}})
    px, _, _ = make(a, b, tmp_path=tmp_path)
    (resp,) = px.handle(INIT)
    res = resp["result"]
    assert resp["id"] == 1
    assert res["capabilities"]["tools"] == {"listChanged": True}
    assert {"logging", "prompts"} <= set(res["capabilities"])
    assert res["serverInfo"]["name"] == "distil-mcp[git,time]"
    assert res["instructions"] == "use git"
    assert res["protocolVersion"] == "2025-06-18"


def test_initialize_error_is_relayed_for_a_single_server(tmp_path):
    px, _, _ = make(FakeBackend("git", init_error=True), tmp_path=tmp_path)
    (resp,) = px.handle(INIT)
    assert resp["error"]["message"] == "nope" and resp["id"] == 1


def test_a_server_that_fails_initialize_is_dropped_from_many(tmp_path):
    px, _, _ = make(FakeBackend("git", init_error=True), FakeBackend("time"), tmp_path=tmp_path)
    (resp,) = px.handle(INIT)
    assert "result" in resp and list(px.backends) == ["time"]
    px2, _, _ = make(
        FakeBackend("git", init_error=True), FakeBackend("time", init_error=True), tmp_path=tmp_path
    )
    assert "error" in px2.handle(INIT)[0]


def test_l0_list_is_the_canonical_list(tmp_path):
    px, _, _ = make(FakeBackend("git"), results=False, tmp_path=tmp_path)
    listed = tools(px)
    assert listed == [levels.canonical_tool(t) for t in fakeserver.load_fixture("git")["tools"]]


def test_list_is_paginated_from_the_backend(tmp_path):
    b = FakeBackend("git")
    pages = {None: (b.fixture["tools"][:5], "c1"), "c1": (b.fixture["tools"][5:], None)}

    def request(method, params=None, *, id=None):
        if method == "tools/list":
            chunk, nxt = pages[(params or {}).get("cursor")]
            res = {"tools": chunk, **({"nextCursor": nxt} if nxt else {})}
            return {"jsonrpc": "2.0", "id": id, "result": res}
        return FakeBackend.request(b, method, params, id=id)

    b.request = request
    px, _, _ = make(b, results=False, tmp_path=tmp_path)
    assert len(tools(px)) == len(b.fixture["tools"])


def test_tool_name_collisions_get_a_server_prefix(tmp_path):
    px, _, _ = make(
        FakeBackend("a", "time"), FakeBackend("b", "time"), results=False, tmp_path=tmp_path
    )
    names = [t["name"] for t in tools(px)]
    assert names == ["get_current_time", "convert_time", "b_get_current_time", "b_convert_time"]
    out = call(px, "b_convert_time", {"time": "1"})
    assert "convert_time ok" in out[0]["result"]["content"][0]["text"]


# ---------------------------------------------------------------- tools/call + lazy flow


def test_call_routes_with_the_client_id_and_counts_usage(tmp_path):
    b = FakeBackend("git")
    px, _, _ = make(b, tmp_path=tmp_path)
    (resp,) = call(px, "git_status", {"repo_path": "/r"}, msg_id=42)
    assert resp["id"] == 42 and b.requests[-1] == (
        "tools/call",
        {"name": "git_status", "arguments": {"repo_path": "/r"}},
        42,
    )
    assert events.ServerState("git", tmp_path).usage() == {"git_status": 1}


def test_lazy_unlock_surfaces_the_real_tool(tmp_path):
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=tmp_path)
    assert [t["name"] for t in tools(px)] == [
        "git_get_tool_schema",
        "git_invoke_tool",
        "git_expand",
    ]
    out = call(px, "git_get_tool_schema", {"tool_name": "git_log"})
    assert "git_log(repo_path" in out[0]["result"]["content"][0]["text"]
    assert out[1] == {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
    listed = tools(px)
    assert listed[-1]["name"] == "git_log"  # appended, real name, real schema
    real = next(t for t in fakeserver.load_fixture("git")["tools"] if t["name"] == "git_log")
    assert set(listed[-1]["inputSchema"]["properties"]) == set(real["inputSchema"]["properties"])
    # a second fetch of the same tool changes nothing
    assert len(call(px, "git_get_tool_schema", {"tool_name": "git_log"})) == 1
    # the unlocked set persisted for this session
    assert events.ServerState("git", tmp_path).unlocked("s1") == ["git_log"]
    assert px.list_changes == 1


def test_invoke_fallback_calls_and_unlocks(tmp_path):
    b = FakeBackend("git")
    px, _, _ = make(b, level="L2", results=False, tmp_path=tmp_path)
    out = call(px, "git_invoke_tool", {"tool_name": "git_status", "arguments": {"repo_path": "/r"}})
    assert "git_status ok" in out[0]["result"]["content"][0]["text"]
    assert out[1]["method"] == "notifications/tools/list_changed"
    assert b.requests[-1][1] == {"name": "git_status", "arguments": {"repo_path": "/r"}}


def test_calling_a_hidden_real_tool_directly_works_and_unlocks(tmp_path):
    px, _, _ = make(FakeBackend("git"), level="L2", results=False, tmp_path=tmp_path)
    out = call(px, "git_status", {"repo_path": "/r"})
    assert "git_status ok" in out[0]["result"]["content"][0]["text"] and len(out) == 2


def test_schema_for_an_unknown_tool_is_a_tool_error_with_suggestions(tmp_path):
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=tmp_path)
    (resp,) = call(px, "git_get_tool_schema", {"tool_name": "git_stauts"})
    assert (
        resp["result"]["isError"] is True and "git_status" in resp["result"]["content"][0]["text"]
    )
    (resp,) = call(px, "git_invoke_tool", {"tool_name": 5})
    assert resp["result"]["isError"] is True


def test_results_are_digested_and_expandable(tmp_path):
    px, store, _ = make(FakeBackend("git"), tmp_path=tmp_path)
    (resp,) = call(px, "git_log", {"repo_path": "/r"})
    text = resp["result"]["content"][0]["text"]
    handle = re.search(r"handle=([0-9a-f]{8})", text).group(1)
    assert "ERROR: entry-0200 failed checksum" in text
    (exp,) = call(px, "git_expand", {"handle": handle})
    assert exp["result"]["content"][0]["text"] == store[handle]
    assert exp["result"]["isError"] is False
    (bad,) = call(px, "git_expand", {"handle": "00000000"})
    assert bad["result"]["isError"] is True


def test_backend_tool_errors_pass_through_untouched(tmp_path):
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=tmp_path)
    tools(px)
    px.surfaces["git"].by_name["ghost"] = {
        "name": "ghost"
    }  # proxy thinks it exists; backend does not
    resp = call(px, "ghost")[0]
    assert resp["error"]["code"] == -32602


def test_unknown_tool_single_server_is_the_servers_call(tmp_path):
    px, _, _ = make(FakeBackend("git"), tmp_path=tmp_path)
    (resp,) = call(px, "definitely_not_a_tool")
    assert resp["error"]["message"].startswith("unknown tool")


def test_unknown_tool_many_servers_is_invalid_params(tmp_path):
    px, _, _ = make(FakeBackend("git"), FakeBackend("time"), tmp_path=tmp_path)
    (resp,) = call(px, "nope")
    assert resp["error"]["code"] == -32602


# ---------------------------------------------------------------- relays


def test_ping_and_relayed_methods(tmp_path):
    b = FakeBackend("git")
    px, _, _ = make(b, tmp_path=tmp_path)
    assert px.handle({"jsonrpc": "2.0", "id": 3, "method": "ping"}) == [
        {"jsonrpc": "2.0", "id": 3, "result": {}}
    ]
    (resp,) = px.handle({"jsonrpc": "2.0", "id": 4, "method": "prompts/list"})
    assert resp["id"] == 4 and resp["error"]["code"] == -32601  # the backend's own answer


def test_list_methods_merge_across_capable_servers(tmp_path):
    a = FakeBackend("git", caps={"tools": {}, "resources": {}})
    b = FakeBackend("time", caps={"tools": {}, "resources": {}})
    c = FakeBackend("fetch", caps={"tools": {}})
    px, _, _ = make(a, b, c, tmp_path=tmp_path)
    px.handle(INIT)
    (resp,) = px.handle({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
    assert [r["uri"] for r in resp["result"]["resources"]] == ["git://r", "time://r"]
    (resp,) = px.handle({"jsonrpc": "2.0", "id": 6, "method": "completion/complete"})
    assert resp["error"]["code"] == -32601  # nobody advertises completions


def test_client_notifications_reach_every_server(tmp_path):
    a, b = FakeBackend("git"), FakeBackend("time")
    px, _, _ = make(a, b, tmp_path=tmp_path)
    assert px.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) == []
    px.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 7}})
    assert (
        a.notes
        == b.notes
        == [("notifications/initialized", None), ("notifications/cancelled", {"requestId": 7})]
    )


def test_server_to_client_requests_round_trip_with_remapped_ids(tmp_path):
    b = FakeBackend("git")
    px, _, emitted = make(b, tmp_path=tmp_path)
    b.on_message({"jsonrpc": "2.0", "id": 99, "method": "roots/list"})
    (req,) = emitted
    assert req["method"] == "roots/list" and req["id"] != 99
    px.handle({"jsonrpc": "2.0", "id": req["id"], "result": {"roots": []}})
    assert b.sent == [{"jsonrpc": "2.0", "id": 99, "result": {"roots": []}}]
    px.handle({"jsonrpc": "2.0", "id": "stray", "result": {}})  # unknown response: dropped
    assert len(b.sent) == 1


def test_backend_list_changed_refreshes_but_keeps_unlocks(tmp_path):
    b = FakeBackend("git")
    px, _, emitted = make(b, level="L2", tmp_path=tmp_path)
    call(px, "git_get_tool_schema", {"tool_name": "git_log"})
    b.on_message({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    assert emitted[-1]["method"] == "notifications/tools/list_changed" and "git" not in px.surfaces
    assert tools(px)[-1]["name"] == "git_log"
    b.on_message({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info"}})
    assert emitted[-1]["method"] == "notifications/message"


# ---------------------------------------------------------------- fail-open


def test_result_compression_failure_returns_the_raw_result(tmp_path, monkeypatch):
    px, _, _ = make(FakeBackend("git"), tmp_path=tmp_path)

    def boom(*a, **k):
        raise RuntimeError("secret content must not be logged")

    monkeypatch.setattr(levels, "compress_result", boom)
    (resp,) = call(px, "git_log", {"repo_path": "/r"})
    assert resp["result"] == fakeserver.canned_result("git_log", {"repo_path": "/r"})
    log = (tmp_path / "events.jsonl").read_text()
    assert '"err": "RuntimeError"' in log and "secret" not in log


def test_list_compression_failure_returns_the_raw_list(tmp_path, monkeypatch):
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=tmp_path)
    monkeypatch.setattr(levels.Surface, "tools_list", lambda self: 1 / 0)
    assert tools(px) == fakeserver.load_fixture("git")["tools"]


def test_call_path_failure_relays_the_raw_call(tmp_path, monkeypatch):
    px, _, _ = make(FakeBackend("git"), level="L2", results=False, tmp_path=tmp_path)
    tools(px)
    monkeypatch.setattr(levels.Surface, "resolve", lambda self, n: 1 / 0)
    (resp,) = call(px, "git_status", {"repo_path": "/r"})
    assert "git_status ok" in resp["result"]["content"][0]["text"]


def test_backend_errors_become_json_rpc_errors(tmp_path):
    b = FakeBackend("git")

    def dead(*a, **k):
        raise proxy.BackendError("git: server exited")

    b.request = dead
    px, _, _ = make(b, tmp_path=tmp_path)
    (resp,) = px.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
    assert resp["error"] == {"code": -32603, "message": "git: server exited"}


def test_proxy_requires_backends_and_a_known_level():
    with pytest.raises(ValueError):
        proxy.Proxy({}, level="L0")
    with pytest.raises(ValueError):
        proxy.Proxy({"g": FakeBackend("git")}, level="L7")


# ---------------------------------------------------------------- state, events, catalog


def test_events_are_content_free(tmp_path):
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=tmp_path)
    tools(px)
    call(px, "git_get_tool_schema", {"tool_name": "git_commit"})
    call(px, "git_commit", {"repo_path": "/very/secret/path", "message": "classified message"})
    raw = (tmp_path / "events.jsonl").read_text()
    assert "secret" not in raw and "classified" not in raw
    rows = [json.loads(ln) for ln in raw.splitlines()]
    allowed = events._STR_FIELDS | events._NUM_FIELDS
    assert all(set(r) <= allowed for r in rows)
    assert {r["ev"] for r in rows} >= {"list", "schema_fetch", "unlock", "call"}
    with pytest.raises(ValueError):
        events.EventLog("s", tmp_path / "e.jsonl").emit("call", "git", arguments="x")


def test_catalog_snapshot_feeds_the_diff_view(tmp_path):
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=tmp_path)
    tools(px)
    cat = events.read_catalogs(tmp_path)["git"]
    assert cat["level"] == "L2" and len(cat["tools"]) == 12
    row = next(t for t in cat["tools"] if t["name"] == "git_status")
    assert ["/inputSchema/title", "removed"] in [list(x) for x in row["dropped"]]
    assert row["tokens_lazy"] < row["tokens_before"]


def test_l3_pins_from_learned_usage_across_sessions(tmp_path):
    state = events.ServerState("git", tmp_path)
    for _ in range(4):
        state.update("old", used="git_status")
    px, _, _ = make(FakeBackend("git"), level="L3", tmp_path=tmp_path)
    assert [t["name"] for t in tools(px)][3:] == ["git_status"]


def test_state_is_bounded_and_survives_garbage(tmp_path):
    state = events.ServerState("git", tmp_path)
    state.path.parent.mkdir(parents=True, exist_ok=True)
    state.path.write_text("not json")
    assert state.load() == {"usage": {}, "sessions": {}}
    for i in range(events.MAX_SESSIONS + 5):
        state.update(f"s{i}", unlocked=["a"])
    assert len(state.load()["sessions"]) == events.MAX_SESSIONS


def test_event_log_rotates(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "MAX_LOG_BYTES", 200)
    log = events.EventLog("s", tmp_path / "events.jsonl")
    for _ in range(10):
        log.emit("list", "git", tokens_before=1, tokens_after=1)
    assert (tmp_path / "events.jsonl.1").exists()
    assert len(events.read_events(tmp_path / "events.jsonl")) >= 2


def test_null_log_writes_nothing(tmp_path):
    px, _, _ = make(FakeBackend("git"), tmp_path=tmp_path)
    px.log = events.NullLog()
    tools(px)
    call(px, "git_status", {"repo_path": "/r"})
    assert not (tmp_path / "events.jsonl").exists()


# ---------------------------------------------------------------- config


def test_config_parsing():
    specs, skipped = proxy.parse_servers(
        {
            "mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "pkg"], "env": {"A": "1"}, "cwd": "/tmp"},
                "remote": {"url": "https://x"},
                "off": {"command": "x", "disabled": True},
            }
        }
    )
    assert (
        [s.name for s in specs] == ["fs"] and specs[0].env == {"A": "1"} and specs[0].cwd == "/tmp"
    )
    assert len(skipped) == 2
    assert proxy.parse_servers({"servers": {"a": {"command": "x"}}})[0][0].args == []


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"mcpServers": {"a": "x"}},
        {"mcpServers": {"a": {"command": ""}}},
        {"mcpServers": {"a": {"command": "x", "args": [1]}}},
        {"mcpServers": {"a": {"command": "x", "env": {"A": 1}}}},
        {"mcpServers": {"a": {"command": "x", "cwd": 3}}},
        {"mcpServers": {"a": {"url": "https://x"}}},
    ],
)
def test_bad_configs_are_refused(data):
    with pytest.raises(proxy.ConfigError):
        proxy.parse_servers(data)


def test_load_config_reads_files(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": {"a": {"command": "x"}}}))
    assert proxy.load_config(p)[0][0].command == "x"
    p.write_text("{")
    with pytest.raises(proxy.ConfigError):
        proxy.load_config(p)


# ---------------------------------------------------------------- real stdio


def _spec(name="git"):
    return proxy.ServerSpec(name, sys.executable, ["-m", "distil.mcpproxy.fakeserver", name])


def test_stdio_end_to_end_through_a_real_subprocess(tmp_path):
    px = proxy.build([_spec()], level="L2", results=True, session="e2e")
    msgs = [
        INIT,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = io.BytesIO(b"".join(json.dumps(m).encode() + b"\n" for m in msgs) + b"not json\n[]\n")
    stdout = io.BytesIO()
    proxy.serve(px, stdin, stdout)
    out = [json.loads(ln) for ln in stdout.getvalue().splitlines()]
    by_id = {m.get("id"): m for m in out}
    assert by_id[1]["result"]["capabilities"]["tools"]["listChanged"] is True
    assert [t["name"] for t in by_id[2]["result"]["tools"]][0] == "git_get_tool_schema"
    assert px.backends["git"].proc.poll() is not None  # closed on EOF


def test_stdio_backend_request_and_exit(tmp_path):
    b = proxy.StdioBackend(_spec(), timeout=30)
    b.start()
    try:
        assert b.request("initialize", INIT["params"])["result"]["serverInfo"]["name"]
        b.notify("notifications/initialized")
        b.send({"jsonrpc": "2.0", "id": "x", "result": {}})
        assert b.request("tools/call", {"name": "git_status", "arguments": {}})["result"]
    finally:
        b.close()
    with pytest.raises(proxy.BackendError):
        b.request("ping")
    b.notify("x")  # a dead server swallows notifications
    b.send({})


def test_stdio_backend_times_out_and_detects_exit(tmp_path):
    silent = proxy.StdioBackend(
        proxy.ServerSpec("s", sys.executable, ["-c", "import time; time.sleep(30)"]), timeout=0.3
    )
    silent.start()
    try:
        with pytest.raises(proxy.BackendError, match="no answer"):
            silent.request("ping")
    finally:
        silent.close()
    dying = proxy.StdioBackend(
        proxy.ServerSpec("d", sys.executable, ["-c", "print('junk'); print('[1]')"]), timeout=10
    )
    dying.start()
    with pytest.raises(proxy.BackendError):
        dying.request("ping")
    dying.close()
    with pytest.raises(proxy.BackendError, match="not started"):
        proxy.StdioBackend(_spec())._write({})
    proxy.StdioBackend(_spec()).close()  # never started: nothing to do


def test_serve_survives_a_handler_crash():
    class Boom(proxy.Proxy):
        def handle(self, msg):
            raise RuntimeError("x")

        def close(self):
            pass

    px = Boom({"g": FakeBackend("git")})
    stdout = io.BytesIO()
    proxy.serve(
        px,
        io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n{"jsonrpc":"2.0","method":"n"}\n'),
        stdout,
    )
    assert json.loads(stdout.getvalue())["error"]["message"] == "distil mcp: RuntimeError"


def test_serve_ignores_a_closed_client():
    class Closed(io.BytesIO):
        def write(self, b):
            raise BrokenPipeError

    px = proxy.Proxy({"g": FakeBackend("git")})
    proxy.serve(px, io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n'), Closed())


def test_backend_reader_survives_a_raising_relay():
    b = proxy.StdioBackend(
        proxy.ServerSpec("n", sys.executable, ["-c", 'print(\'{"jsonrpc":"2.0","method":"x"}\')'])
    )
    got = threading.Event()

    def on_message(msg):
        got.set()
        raise RuntimeError

    b.on_message = on_message
    b.start()
    assert got.wait(10)
    b.close()


# ---------------------------------------------------------------- CLI


def _ns(**kw):
    base = {"mcp_cmd": "wrap", "name": None, "level": "L0", "no_results": False, "command": []}
    return argparse.Namespace(**{**base, **kw})


def test_cli_wrap_needs_a_command(capsys):
    assert cli.cmd_mcp(_ns(command=["--"])) == 2


def test_cli_wrap_fails_open_by_exec(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("proxy broke")

    execs = []
    monkeypatch.setattr(proxy, "build", boom)
    monkeypatch.setattr("os.execvp", lambda f, argv: execs.append(argv))
    cli.cmd_mcp(_ns(command=["--", "srv", "--flag"]))
    assert execs == [["srv", "--flag"]]
    assert "running the server directly" in capsys.readouterr().err


def test_cli_wrap_missing_server_is_127(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("srv")

    monkeypatch.setattr(proxy, "build", missing)
    assert cli.cmd_mcp(_ns(command=["srv"])) == 127


def test_cli_wrap_runs_and_warns_about_uncertified_levels(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(proxy, "build", lambda specs, **kw: seen.update(specs=specs, **kw) or "PX")
    monkeypatch.setattr(proxy, "serve", lambda px: seen.update(served=px))
    assert (
        cli.cmd_mcp(_ns(level="L2", command=["npx", "-y", "@modelcontextprotocol/server-git"])) == 0
    )
    assert seen["specs"][0].name == "git" and seen["level"] == "L2" and seen["served"] == "PX"
    assert "L2, R accuracy certificate is pending" in capsys.readouterr().err


def test_cli_serve(monkeypatch, tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"mcpServers": {"a": {"command": "x"}, "r": {"url": "https://x"}}}))
    monkeypatch.setattr(proxy, "build", lambda specs, **kw: "PX")
    monkeypatch.setattr(proxy, "serve", lambda px: None)
    ns = argparse.Namespace(mcp_cmd="serve", config=str(cfg), level="L0", no_results=True)
    assert cli.cmd_mcp(ns) == 0
    assert "skipped r" in capsys.readouterr().err
    cfg.write_text("{}")
    assert cli.cmd_mcp(ns) == 2


def test_cli_serve_does_not_fail_open(monkeypatch, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"mcpServers": {"a": {"command": "x"}}}))

    def boom(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(proxy, "build", boom)
    with pytest.raises(RuntimeError):
        cli.cmd_mcp(
            argparse.Namespace(mcp_cmd="serve", config=str(cfg), level="L0", no_results=False)
        )


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        (["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "filesystem"),
        (["uvx", "mcp-server-git"], "git"),
        (["my-mcp-tool"], "my-mcp-tool"),
        (["./server"], "server"),
    ],
)
def test_default_server_names(argv, name):
    assert cli._mcp_default_name(argv) == name


def test_parser_accepts_the_subcommands():
    p = cli.build_parser()
    ns = p.parse_args(["mcp", "wrap", "--name", "g", "--level", "L2", "--", "python", "-m", "x"])
    assert ns.mcp_cmd == "wrap" and ns.command[-3:] == ["python", "-m", "x"]
    assert p.parse_args(["mcp"]).mcp_cmd is None


def test_bare_mcp_still_runs_distils_own_server(monkeypatch):
    from distil import mcp_server

    ran = []
    monkeypatch.setattr(mcp_server, "serve", lambda: ran.append(1))
    assert cli.cmd_mcp(argparse.Namespace()) == 0 and ran == [1]


def test_state_root_defaults_under_distil_home(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    assert events.mcp_dir() == Path(tmp_path) / "mcp"
    monkeypatch.setenv("DISTIL_MCP_SESSION", "fixed")
    assert events.new_session_id() == "fixed"

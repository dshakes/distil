"""distil mcp watch + webdash /mcp: render from a fixture event log, never leak content."""

from __future__ import annotations

import argparse
import io
import json
import threading
import urllib.request

import pytest

from distil import cli, webdash
from distil.mcpproxy import events, watch

from test_mcp_proxy import FakeBackend, call, make, tools


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """A real session's log: L2 github proxy, two unlocks, calls, an expand."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    root = events.mcp_dir()
    px, _, _ = make(FakeBackend("git"), level="L2", tmp_path=root)
    tools(px)
    call(px, "git_get_tool_schema", {"tool_name": "git_log"})
    out = call(px, "git_log", {"repo_path": "/r"})
    handle = out[0]["result"]["content"][0]["text"].split("handle=")[1][:8]
    call(px, "git_expand", {"handle": handle})
    call(px, "git_invoke_tool", {"tool_name": "git_status", "arguments": {"repo_path": "/r"}})
    tools(px)
    px2, _, _ = make(FakeBackend("time"), level="L0", results=False, tmp_path=root)
    tools(px2)
    return root


def test_summary_counts_the_session(populated):
    s = watch.load_summary(populated)
    git = next(x for x in s["servers"] if x["server"] == "git")
    assert git["level"] == "L2" and git["list_changes"] == 2 and git["schema_fetches"] == 1
    assert git["calls"] == 2 and git["expands"] == 1
    assert git["defs_after"] < git["defs_before"]
    assert git["result_after"] < git["result_before"]
    states = {t["name"]: t["state"] for t in git["tools"]}
    assert (
        states["git_log"] == "unlocked"
        and states["git_status"] == "unlocked"
        and states["git_diff"] == "lazy"
    )
    log_row = next(t for t in git["tools"] if t["name"] == "git_log")
    assert log_row["calls"] == 1 and log_row["schema_fetches"] == 1 and log_row["result_before"] > 0
    time_ = next(x for x in s["servers"] if x["server"] == "time")
    assert {t["state"] for t in time_["tools"]} == {"full"}
    assert s["timeline"][0]["ev"] == "list"  # newest first
    assert set(s["certificate"]) == {"L0", "L1", "L2", "L3", "R"}


def test_render_text_screenshot(populated):
    text = watch.render_text(watch.load_summary(populated), width=140)
    assert "[git] level L2 (lazy)" in text and "cache 2 list change(s)" in text
    assert "[time] level L0 (lossless)" in text and "cache stable" in text
    assert "certificate  L0:pending" in text
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("git_log")]
    assert lines and "unlocked" in lines[0]
    assert all(len(ln) <= 140 for ln in text.splitlines())


def test_render_empty_and_truncated():
    assert "no MCP traffic yet" in watch.render_text({"servers": [], "certificate": {}})
    s = {
        "servers": [
            {
                "server": "x",
                "level": "L0",
                "requested": "L2",
                "results": True,
                "defs_before": 0,
                "defs_after": 0,
                "list_changes": 0,
                "schema_fetches": 0,
                "calls": 0,
                "result_before": 0,
                "result_after": 0,
                "expands": 0,
                "errors": 1,
                "tools": [
                    {
                        "name": f"t{i}",
                        "state": "full",
                        "def_before": 5,
                        "def_after": 5,
                        "schema_fetches": 0,
                        "calls": 0,
                        "result_before": 0,
                        "result_after": 0,
                        "dropped": 0,
                    }
                    for i in range(30)
                ],
            }
        ],
        "certificate": {},
    }
    text = watch.render_text(s)
    assert "asked L2; not smaller" in text and "… 5 more" in text and "errors 1" in text
    assert (
        watch._pct(0, 0) == "—" and watch._pct(100, 100) == "0%" and watch._pct(1000, 995) == "<1%"
    )


def test_run_watch_once_and_loop(populated, monkeypatch):
    buf = io.StringIO()
    assert watch.run_watch(once=True, stream=buf, root=populated) == 0
    assert "[git]" in buf.getvalue()

    def stop(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(watch.time, "sleep", stop)
    buf = io.StringIO()
    assert watch.run_watch(stream=buf, root=populated) == 0
    assert buf.getvalue().startswith("\x1b[H\x1b[2J")


def test_certificate_parsing(tmp_path):
    assert set(watch.certificate().values()) == {"pending"}  # the shipped file: nothing certified
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {"levels": {"L0": {"status": "certified"}, "L2": {"status": "bogus"}, "L3": "x"}}
        )
    )
    c = watch.certificate(p)
    assert c["L0"] == "certified" and c["L2"] == "pending" and c["L3"] == "pending"
    assert set(watch.certificate(tmp_path / "missing.json").values()) == {"pending"}


def test_tool_detail(populated):
    d = watch.tool_detail("git", "git_status", populated)
    assert (
        d["before"]["inputSchema"]["title"] == "GitStatus"
        and "title" not in d["after"]["inputSchema"]
    )
    assert ["/inputSchema/title", "removed"] in [list(x) for x in d["dropped"]]
    assert "git_get_tool_schema" in d["recoverable"]
    assert watch.tool_detail("git", "nope", populated) is None
    assert watch.tool_detail("nope", "x", populated) is None


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_webdash_mcp_routes(populated):
    server = webdash.build_server("127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, ctype, body = _get(port, "/mcp")
        page = body.decode()
        assert status == 200 and ctype.startswith("text/html")
        assert '<html lang="en">' in page and "<main>" in page and 'role="status"' in page
        assert "prefers-reduced-motion" in page and "focus-visible" in page
        assert "innerHTML" not in page  # server-controlled strings only via textContent
        data = json.loads(_get(port, "/mcp/data")[2])
        assert {s["server"] for s in data["servers"]} == {"git", "time"}
        detail = json.loads(_get(port, "/mcp/tool?server=git&tool=git_log")[2])
        assert detail["name"] == "git_log"
        with pytest.raises(urllib.error.HTTPError):
            _get(port, "/mcp/tool?server=git&tool=nope")
        with pytest.raises(urllib.error.HTTPError):
            _get(port, "/mcp/other")
        assert b"/mcp" in _get(port, "/")[2]  # the main dashboard links here
    finally:
        server.shutdown()
        server.server_close()


def test_cli_watch(populated, capsys, monkeypatch):
    ns = argparse.Namespace(mcp_cmd="watch", once=True, interval=1.0, web=False, port=0)
    assert cli.cmd_mcp(ns) == 0
    assert "[git]" in capsys.readouterr().out
    served = []
    monkeypatch.setattr(webdash, "serve_webdash", lambda port, open_browser: served.append(port))
    ns.web, ns.port = True, 9999
    assert cli.cmd_mcp(ns) == 0 and served == [9999]

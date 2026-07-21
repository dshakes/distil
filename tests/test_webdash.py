"""Local real-time dashboard: snapshot is content-free; page has no external calls."""

from __future__ import annotations

import json

from distil import webdash


def test_snapshot_is_content_free_and_correct(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    (tmp_path / "savings.jsonl").write_text(
        json.dumps(
            {
                "trajectory_id": "t",
                "model": "m",
                "turns": 1,
                "baseline_input_tokens": 1000,
                "distil_input_tokens": 400,
                "baseline_dollars": 1.0,
                "distil_dollars": 0.4,
                "tokenizer": "heuristic",
                "ts": 1.0,
                "acct": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    s = webdash._snapshot()
    assert set(s) == {
        "tokens_saved",
        "dollars_saved",
        "pct",
        "runs",
        "baseline_tokens",
        "distil_tokens",
        "equivalence",
        "subscription",
        "ts",
    }
    assert s["tokens_saved"] == 600 and s["pct"] == 60.0 and s["runs"] == 1
    assert set(s["equivalence"]) == {"pct", "shadowed"}
    # numbers / bools only — nothing that can carry prompt or path content
    for k, v in s.items():
        if k == "equivalence":
            continue
        assert isinstance(v, (int, float, bool)), k


def test_page_is_self_contained_local_only():
    # the served page must never call out to a remote host (local-only promise)
    assert "http://" not in webdash._PAGE and "https://" not in webdash._PAGE
    assert 'fetch("/data"' in webdash._PAGE  # reads the LOCAL endpoint only
    assert "nothing leaves this machine" in webdash._PAGE


def test_build_server_serves_endpoints(tmp_path, monkeypatch):
    import threading
    import urllib.request

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    srv = webdash.build_server("127.0.0.1", 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/data", timeout=3) as r:
            assert r.status == 200 and "tokens_saved" in json.loads(r.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as r:
            body = r.read()
            assert r.status == 200 and b"your tokens saved" in body
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=3)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_serve_webdash_runs_and_closes(tmp_path, monkeypatch):
    """serve_webdash prints, (optionally) opens a browser, serves, and closes."""
    import threading

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    holder = {}
    real_build = webdash.build_server

    def capture(host, port):
        srv = real_build(host, 0)
        holder["srv"] = srv
        return srv

    monkeypatch.setattr(webdash, "build_server", capture)
    t = threading.Thread(
        target=lambda: webdash.serve_webdash(port=0, open_browser=False), daemon=True
    )
    t.start()
    for _ in range(50):
        if "srv" in holder:
            break
        import time as _t

        _t.sleep(0.02)
    assert "srv" in holder
    holder["srv"].shutdown()
    t.join(timeout=3)


def test_serve_webdash_opens_browser(tmp_path, monkeypatch):
    """open_browser=True runs the browser-open thread (mocked)."""
    import threading
    import time as _t

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    opened = threading.Event()
    monkeypatch.setattr(webdash.webbrowser, "open", lambda url: opened.set())
    holder = {}
    real_build = webdash.build_server
    monkeypatch.setattr(
        webdash, "build_server", lambda h, p: holder.setdefault("srv", real_build(h, 0))
    )
    t = threading.Thread(
        target=lambda: webdash.serve_webdash(port=0, open_browser=True), daemon=True
    )
    t.start()
    assert opened.wait(3.0), "browser-open thread never ran"
    for _ in range(50):
        if "srv" in holder:
            break
        _t.sleep(0.02)
    holder["srv"].shutdown()
    t.join(timeout=3)

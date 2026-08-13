"""cmd_proxy --async must not silently drop --expand/--session-delta/--shadow —
it names them and points at the standard proxy (audit finding #1, #25-class)."""

import argparse
import sys
import types

from distil import cli


def _run(monkeypatch, capsys, **overrides):
    # Fake distil.aproxy so the import needs no aiohttp and starts no server.
    fake = types.ModuleType("distil.aproxy")
    fake.serve = lambda **kw: None
    monkeypatch.setitem(sys.modules, "distil.aproxy", fake)
    args = argparse.Namespace(
        use_async=True,
        host="127.0.0.1",
        port=0,
        upstream="https://x",
        lossless_only=False,
        verbatim=False,
        shape_output="off",
        no_record=True,
        pricing="claude-opus-4-8",
        expand=False,
        session_delta=False,
        shadow=0.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    assert cli.cmd_proxy(args) == 0
    return capsys.readouterr().err


def test_async_warns_on_expand(monkeypatch, capsys):
    err = _run(monkeypatch, capsys, expand=True, session_delta=True, shadow=0.5)
    assert "--expand" in err and "--session-delta" in err and "--shadow" in err
    assert "not supported on the async proxy" in err


def test_async_quiet_when_no_unsupported_flags(monkeypatch, capsys):
    err = _run(monkeypatch, capsys)  # plain async proxy, nothing unsupported
    assert "not supported on the async proxy" not in err


def test_proxy_and_wrap_agree_on_quality_loop_defaults():
    """`proxy` must default the quality loops the same as `wrap`.

    The managed daemon install (com.distil.proxy launch agent) runs `distil proxy`,
    not `distil wrap`, so a default that differs here silently disables the loop on
    every managed install: a real 767-request session sampled 0 for shadow and had no
    decision-equivalence evidence at all, while the same work under `wrap` samples ~2%.
    Retention is free (in-process, counts only), so there is no cost case for drift."""
    from distil.cli import build_parser

    p = build_parser()
    wrap = p.parse_args(["wrap", "echo"])
    proxy = p.parse_args(["proxy"])
    assert proxy.shadow == wrap.shadow > 0
    assert proxy.retention == wrap.retention > 0

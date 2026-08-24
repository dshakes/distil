"""CLI surfaces that fail silently when they regress.

Each test here covers a path whose failure mode is quiet: a wrap that routes
nothing while reporting success, or a store whose contents no command could show.
"""

from __future__ import annotations

import argparse


def test_wrapping_an_ide_extension_says_so_instead_of_routing_nothing(capsys):
    """An IDE extension has no argv to wrap and reads no base-URL variable.

    Wrapping one would set a variable nothing reads: the wrap reports success, the
    editor talks straight to the provider, and the user sees zero savings with no
    explanation. Say it before the session starts, and name the path that works.
    """
    from distil.cli import _warn_if_ide_not_wrappable

    _warn_if_ide_not_wrappable("cursor")
    err = capsys.readouterr().err
    assert "IDE extension" in err
    assert "distil proxy" in err, "must name the mechanism that actually works"
    assert "IDE-AGENTS.md" in err

    _warn_if_ide_not_wrappable("claude")  # a real CLI must stay silent
    assert capsys.readouterr().err == ""


# --- cross-agent recall store -------------------------------------------------
def test_memory_reports_the_shared_store(tmp_path, monkeypatch, capsys):
    """A handle minted by one agent is expandable by any other reading the same
    DISTIL_HOME. That already worked; nothing surfaced it, so nobody knew."""
    import importlib

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    import distil.mcp_server as ms

    importlib.reload(ms)
    ms.record_restore("abcd1234", "the original, stored by agent A")

    from distil.cli import cmd_memory

    assert cmd_memory(argparse.Namespace(clear=False)) == 0
    out = capsys.readouterr().out
    assert "stored originals : 1" in out
    assert "encrypted" in out
    # And a different process reading the same home recovers it.
    assert ms.load_restore("abcd1234") == "the original, stored by agent A"


def test_memory_reports_an_empty_store_without_error(tmp_path, monkeypatch, capsys):
    import importlib

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    import distil.mcp_server as ms

    importlib.reload(ms)
    from distil.cli import cmd_memory

    assert cmd_memory(argparse.Namespace(clear=False)) == 0
    assert "empty" in capsys.readouterr().out


def test_memory_clear_removes_the_originals(tmp_path, monkeypatch, capsys):
    import importlib

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    import distil.mcp_server as ms

    importlib.reload(ms)
    ms.record_restore("aaaa1111", "x" * 64)
    ms.record_restore("bbbb2222", "y" * 64)

    from distil.cli import cmd_memory

    assert cmd_memory(argparse.Namespace(clear=True)) == 0
    assert "cleared 2" in capsys.readouterr().out
    assert ms.load_restore("aaaa1111") is None

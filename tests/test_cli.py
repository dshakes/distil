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

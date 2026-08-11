"""Writing a measured finding into the agent's instruction file.

Two things make this safe to point at a real file: it never claims more than it
measured, and it never touches anything outside its own markers. Both are easy to
lose in a refactor and expensive to lose in the field — the first turns the file
into advice the agent learns to skim, the second turns distil into a tool that
rewrites files people asked it only to observe.
"""

from __future__ import annotations

from pathlib import Path

from distil import corrections


class _Dissection:
    """Stand-in with just the surface build_block reads."""

    def __init__(self, kinds, baseline):
        self._kinds = kinds
        self.baseline_tokens = baseline

    def blocks_by_kind(self):
        return self._kinds


# ---------------------------------------------------------------------------
# It only says what it measured
# ---------------------------------------------------------------------------


def test_a_dominant_pattern_is_reported_with_its_numbers() -> None:
    block = corrections.build_block(_Dissection([("log", 12, 60_000), ("code", 3, 40_000)], 0))
    assert block is not None
    assert "60,000" in block and "100,000" in block and "60%" in block
    assert "large log output" in block
    assert "12 distinct blocks" in block


def test_a_session_too_small_to_generalise_from_earns_nothing() -> None:
    """A pattern stated from one small session is a claim the data cannot carry."""
    assert corrections.build_block(_Dissection([("log", 2, 900)], 0)) is None


def test_no_dominant_pattern_earns_nothing() -> None:
    """A plurality is not a pattern. Three kinds split ~36/33/31 is not "your
    project is mostly logs", and saying so would be a claim the data cannot carry."""
    kinds = [("log", 5, 15_000), ("diff", 5, 14_000), ("code", 5, 13_000)]
    assert corrections.build_block(_Dissection(kinds, 0)) is None


def test_an_unrecognised_signature_earns_silence_not_invented_advice() -> None:
    assert corrections.build_block(_Dissection([("wat", 9, 90_000)], 0)) is None


def test_the_advice_differs_by_what_was_measured() -> None:
    """Generic advice is what gets skimmed; the useful move differs per content type."""
    log = corrections.build_block(_Dissection([("log", 9, 90_000)], 0))
    diff = corrections.build_block(_Dissection([("diff", 9, 90_000)], 0))
    assert log and diff and log != diff
    assert "tail" in log and "--stat" in diff


# ---------------------------------------------------------------------------
# It never touches anything outside its markers
# ---------------------------------------------------------------------------


def test_creating_preserves_existing_content(tmp_path: Path) -> None:
    p = tmp_path / "CLAUDE.local.md"
    p.write_text("# My rules\n\nAlways run the linter.\n", encoding="utf-8")
    assert corrections.apply_block(p, "BLOCKB") in ("created", "updated")
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# My rules\n\nAlways run the linter.\n")
    assert "BLOCKB" in text


def test_updating_replaces_only_the_managed_region(tmp_path: Path) -> None:
    p = tmp_path / "CLAUDE.local.md"
    first = f"{corrections.BEGIN}\nold finding\n{corrections.END}"
    p.write_text(f"# Mine\n\n{first}\n\n## After\nkeep me\n", encoding="utf-8")
    second = f"{corrections.BEGIN}\nnew finding\n{corrections.END}"
    assert corrections.apply_block(p, second) == "updated"
    text = p.read_text(encoding="utf-8")
    assert "new finding" in text and "old finding" not in text
    assert "# Mine" in text and "## After" in text and "keep me" in text


def test_rewriting_the_same_block_is_a_no_op(tmp_path: Path) -> None:
    """Re-running must not churn the file — a tool that rewrites on every run
    produces noise in `git status` and gets removed from people's workflows."""
    p = tmp_path / "CLAUDE.local.md"
    block = f"{corrections.BEGIN}\nsame\n{corrections.END}"
    corrections.apply_block(p, block)
    before = p.read_text(encoding="utf-8")
    assert corrections.apply_block(p, block) == "unchanged"
    assert p.read_text(encoding="utf-8") == before


def test_a_missing_file_is_created(tmp_path: Path) -> None:
    p = tmp_path / "CLAUDE.local.md"
    assert corrections.apply_block(p, "X") == "created"
    assert p.exists()


# ---------------------------------------------------------------------------
# Which files it may write unasked
# ---------------------------------------------------------------------------


def test_local_files_are_distils_to_write() -> None:
    for name in ("CLAUDE.local.md", "AGENTS.local.md", "GEMINI.local.md"):
        assert corrections.is_tracked_instruction_file(Path(name)) is False


def test_tracked_files_are_not() -> None:
    """CLAUDE.md is reviewed by the user's teammates. Editing it unasked is distil
    modifying a repo it was invited to observe."""
    for name in ("CLAUDE.md", "AGENTS.md", "GEMINI.md"):
        assert corrections.is_tracked_instruction_file(Path(name)) is True


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _args(**kw):
    import argparse

    d = dict(write=None, session=None, force=False, threshold=0.25, min_samples=5)
    d.update(kw)
    return argparse.Namespace(**d)


def test_the_command_refuses_a_tracked_file_and_offers_both_ways_out(
    tmp_path, monkeypatch, capsys
) -> None:
    from distil import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_learn_write", cli._learn_write)  # ensure we exercise the real one
    from distil import dissect as _d

    monkeypatch.setattr(_d, "list_sessions", lambda: [type("S", (), {"sid": "s1"})()])
    monkeypatch.setattr(_d, "dissect", lambda sid: _Dissection([("log", 9, 90_000)], 0))

    rc = cli.cmd_learn(_args(write="CLAUDE.md"))
    out = capsys.readouterr().out
    assert rc == 1
    assert not (tmp_path / "CLAUDE.md").exists(), "a tracked file was written unasked"
    assert "--force" in out and "CLAUDE.local.md" in out, "both ways out must be offered"


def test_the_command_writes_the_local_file(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli
    from distil import dissect as _d

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_d, "list_sessions", lambda: [type("S", (), {"sid": "s1"})()])
    monkeypatch.setattr(_d, "dissect", lambda sid: _Dissection([("log", 9, 90_000)], 0))

    rc = cli.cmd_learn(_args(write="CLAUDE.local.md"))
    assert rc == 0
    text = (tmp_path / "CLAUDE.local.md").read_text(encoding="utf-8")
    assert corrections.BEGIN in text and "large log output" in text
    assert "90,000" in text


def test_a_thin_session_writes_nothing_at_all(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli
    from distil import dissect as _d

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_d, "list_sessions", lambda: [type("S", (), {"sid": "s1"})()])
    monkeypatch.setattr(_d, "dissect", lambda sid: _Dissection([("log", 1, 500)], 0))

    rc = cli.cmd_learn(_args(write="CLAUDE.local.md"))
    assert rc == 0
    assert not (tmp_path / "CLAUDE.local.md").exists(), "wrote a claim it had not earned"
    assert "Nothing written" in capsys.readouterr().out


def test_a_file_that_is_not_utf8_is_read_and_preserved(tmp_path: Path) -> None:
    """A user's instruction file saved by a Windows editor is frequently cp1252.

    distil did not write that file and has no business demanding an encoding for
    it. A strict utf-8 read raises UnicodeDecodeError, turning "add a note to your
    file" into a traceback — on the platform least likely to be holding utf-8.
    This is the same class as the cp1252 crash in `distil stats`: a tool must not
    fail on the file it was pointed at.
    """
    p = tmp_path / "CLAUDE.local.md"
    # 0x97 is an em-dash in cp1252 and invalid as a lone utf-8 byte.
    p.write_bytes(b"# My rules \x97 written on Windows\n\nAlways run the linter.\n")

    block = f"{corrections.BEGIN}\nfinding\n{corrections.END}"
    assert corrections.apply_block(p, block) in ("created", "updated")

    raw = p.read_bytes()
    assert b"\x97" in raw, "the user's own bytes were mangled, not preserved"
    assert b"# My rules" in raw and b"Always run the linter." in raw
    assert b"distil:begin" in raw and b"finding" in raw


def test_the_markers_are_ascii(tmp_path: Path) -> None:
    """They are compared against bytes from a file distil did not write, and are
    echoed to consoles whose encoding it does not control. An em-dash here came
    back as a replacement character on a cp1252 console, so the marker stopped
    matching itself and the block was appended twice instead of updated."""
    corrections.BEGIN.encode("ascii")
    corrections.END.encode("ascii")


def test_line_endings_are_not_rewritten(tmp_path: Path) -> None:
    """A CRLF file must come back CRLF, and an LF file LF.

    Python's universal-newline translation silently rewrites every line ending on
    the way in and out. For a tool that claims to touch only its own block, that
    is a whole-file diff in `git status` — the churn that gets a tool removed from
    someone's workflow, and a direct contradiction of the byte-for-byte promise.
    """
    crlf = tmp_path / "crlf.local.md"
    crlf.write_bytes(b"# Windows file\r\n\r\nRule one.\r\n")
    corrections.apply_block(crlf, f"{corrections.BEGIN}\nfinding\n{corrections.END}")
    raw = crlf.read_bytes()
    assert b"# Windows file\r\n" in raw, "the user's CRLF endings were rewritten"
    assert b"\n" in raw and raw.count(b"\r\n") == raw.count(b"\n"), (
        "mixed line endings were left behind"
    )

    lf = tmp_path / "lf.local.md"
    lf.write_bytes(b"# Unix file\n\nRule one.\n")
    corrections.apply_block(lf, f"{corrections.BEGIN}\nfinding\n{corrections.END}")
    assert b"\r\n" not in lf.read_bytes(), "CRLF was introduced into an LF file"

"""Every `distil …` command printed in the docs must actually parse.

The cross-audit found two commands in the eval runbook that fail outright:

    distil bench --runner anthropic   -> unrecognized arguments: --runner anthropic
    distil wrap --shadow claude       -> invalid float value: 'claude'

Both were written by hand, read plausibly, and had never been run. That is the same
unverified-claim failure the probes in this PR exist to catch, pointed at the docs:
a runbook is a promise that these commands work, and the first thing a new user does
is paste one.

So the fix is not "correct those two lines" — it is this file, which extracts every
command from the docs and asserts the parser accepts it. Commands are PARSED, never
executed: this must stay free, offline and instant, or it becomes a gate people skip.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

from distil.cli import build_parser

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_FENCE = re.compile(r"```(?:bash|sh|console)\n(.*?)```", re.S)

# Placeholders in docs stand for a value the reader supplies. Substituting a
# well-typed sample keeps the parse honest without pretending the doc is executable.
_PLACEHOLDERS = {
    "<file>": "x.json",
    "<path>": "x.json",
    "<n>": "1",
    "YOUR_KEY": "k",
}


def _commands(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    for block in _FENCE.findall(text):
        for raw in block.splitlines():
            line = raw.split("#", 1)[0].split("|", 1)[0].strip()
            if not line:
                continue
            for old, new in _PLACEHOLDERS.items():
                line = line.replace(old, new)
            line = line.removeprefix("uv run ").removeprefix("$ ").strip()
            if not line.startswith("distil "):
                continue
            # `--` hands the rest to a wrapped program (`wrap --shadow 0.1 -- claude`);
            # argparse stops there and so do we.
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue  # unbalanced quotes: prose, not a command
            out.append(tokens[1:])
    return out


def _doc_files() -> list[Path]:
    """Every markdown doc, plus the README — which is where a first-time user starts,
    so a command that does not parse there costs the most."""
    return sorted(_DOCS.glob("*.md")) + [_DOCS.parent / "README.md"]


ALL: list[tuple[str, tuple[str, ...]]] = [
    (p.name, tuple(cmd)) for p in _doc_files() for cmd in _commands(p.read_text(encoding="utf-8"))
]


def test_the_extractor_found_commands() -> None:
    """A silent extraction failure would make every assertion below vacuous —
    the exact 'green against nothing' this PR keeps closing."""
    assert len(ALL) > 20, f"only found {len(ALL)} commands; the extractor is broken"
    assert any("fidelity" in c for _, c in ALL), "the eval runbook's commands were not seen"


@pytest.mark.parametrize("doc,cmd", ALL, ids=lambda v: v if isinstance(v, str) else " ".join(v))
def test_documented_command_parses(doc: str, cmd: tuple[str, ...]) -> None:
    parser = build_parser()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            parser.parse_args(list(cmd))
    except SystemExit as exc:
        # 0 is `--help`, which is a legitimate documented invocation. Anything else is
        # argparse rejecting the command a reader was told to run.
        if exc.code not in (0, None):
            pytest.fail(f"{doc}: `distil {' '.join(cmd)}` does not parse\n{buf.getvalue().strip()}")

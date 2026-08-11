"""Turn what a session measured into a note the agent reads next time.

`distil dissect` already knows which content dominated a session's tokens — large
logs, diffs, tracebacks, tabular output. That analysis is shown once, in a report
a human reads and closes, and the agent that produced the cost never sees it. Next
session it does the same thing again.

This writes the finding where the agent will actually read it: a managed block in
the project's agent instruction file. The guidance is keyed to what was measured on
THIS project, not generic advice — "your tool output here is mostly large logs" is
worth saying; "consider being efficient" is not.

What it will not do
-------------------
It writes only to a `*.local.md` file by default, because those are gitignored by
convention and a tracked `CLAUDE.md` is a file the user's teammates read and review.
Editing that without being asked is distil modifying a repo it was invited to
observe. `--force` is required to target a tracked file, and it says so.

It also declines to write anything it cannot support with a number. A correction
file that fills up with plausible-sounding advice teaches the agent to skim it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ASCII on purpose. These are machine-read delimiters, and they get compared against
# bytes from a file distil did not write. An em-dash here rendered as a replacement
# character on a cp1252 Windows console, so the marker no longer matched itself and
# the block was appended a second time instead of updated. Typography in a delimiter
# buys nothing and costs that.
BEGIN = "<!-- distil:begin - measured from your sessions; edit above or below, not inside -->"
END = "<!-- distil:end -->"

#: What each block signature means in a sentence the agent can act on. Keyed to the
#: dominant content type distil actually measured, because the useful advice differs:
#: filtering a log at the source is a different move from narrowing a JSON payload.
#:
#: The keys MUST be exactly the classes ``learn._content_class`` can return. The
#: first version of this table was written from imagination — it carried `diff`,
#: `traceback` and `columnar`, which that function never produces, and omitted
#: `json`, which it produces constantly. The dead keys were harmless; the missing
#: one meant `learn --write` silently declined on every JSON-dominated session,
#: which is most of them. `distil dissect` has a WIDER display vocabulary that
#: includes diff/traceback/columnar — it is not the producer of these signatures,
#: and copying from it is how the mistake happened. A test pins both directions.
_GUIDANCE = {
    "log": (
        "large log output",
        "Filter logs at the source rather than reading them whole — `tail -n`, "
        "`grep -m`, or a narrower time window. Most of a log's tokens are lines "
        "nobody reads.",
    ),
    "code": (
        "code listings",
        "Read the specific symbol or range you need rather than whole files — the "
        "surrounding file is rarely what the decision turns on.",
    ),
    "json": (
        "large JSON payloads",
        "Ask for the fields you need rather than whole documents — most of a JSON "
        "response is structure the decision never touches. Where a tool supports a "
        "query or projection, use it at the source.",
    ),
    "error": (
        "error output",
        "Capture the error and the command that produced it, not the full surrounding "
        "output stream.",
    ),
    "prose": (
        "long text output",
        "Prefer a targeted search over reading whole documents when you are looking for one fact.",
    ),
}


def build_block(dissection: Any, *, min_tokens: int = 20_000) -> str | None:
    """The managed block for one session, or None when there is nothing earned.

    Returns None below *min_tokens*: a note written off a trivial session states a
    pattern from too little evidence, and a file of weak claims is one the agent
    learns to skim.
    """
    kinds = dissection.blocks_by_kind()
    # Denominator from the SAME source as the numerator. Using baseline_tokens
    # here looked more impressive — a share of all input tokens — but it is
    # derived from ledger rows while the kinds come from per-request records, and
    # the ledger batches its writes. A session whose records are on disk but whose
    # ledger has not flushed yet then divides by zero and silently declines
    # forever. Found by running this against a live proxy rather than a fixture.
    total = sum(tokens for _sig, _n, tokens in kinds)
    if not kinds or total < min_tokens:
        return None
    top_sig, blocks, tokens = kinds[0]
    label, advice = _GUIDANCE.get(str(top_sig).split(":")[0], (None, None))
    if label is None:
        return None  # an unrecognised signature earns silence, not invented advice
    share = 100.0 * tokens / total if total else 0.0
    # A clear majority, not merely the largest slice. Against this denominator a
    # three-way split still shows ~36% for the biggest, and calling that "the
    # largest single source of context cost" would be a claim the data does not
    # support. Half or more is a pattern; a plurality is a coincidence.
    if share < 50.0:
        return None

    return (
        f"{BEGIN}\n"
        f"## Context cost in this project\n\n"
        f"Measured by distil over a recent session: **{label}** accounted for "
        f"{tokens:,} of {total:,} tokens in the bulky content it handled "
        f"({share:.0f}%), across {blocks} distinct blocks — the largest single "
        f"source of context cost here.\n\n"
        f"{advice}\n\n"
        f"Detail that is elided by distil stays recoverable: call `distil_expand` "
        f"with the handle in a `<< … handle=… >>` marker to get the exact original "
        f"back, rather than re-running the command that produced it.\n"
        f"{END}"
    )


def apply_block(path: Path, block: str) -> str:
    """Insert or replace the managed block in *path*. Returns 'created' | 'updated'
    | 'unchanged'.

    Everything outside the markers is preserved byte-for-byte — the user's own
    instructions are not distil's to rewrite, and a tool that reformats the file
    around its edit is one people stop pointing at real files.
    """
    # surrogateescape, not plain utf-8: this file belongs to the user and distil did
    # not write it. An instruction file saved by a Windows editor is frequently
    # cp1252, and a strict utf-8 read raises UnicodeDecodeError — turning "add a note
    # to your file" into a traceback, on the exact platform least likely to be
    # holding a utf-8 file. Round-tripping through surrogateescape preserves those
    # bytes exactly, which is also what the byte-for-byte promise below requires.
    # newline="" on BOTH sides, or Python's universal-newline translation quietly
    # rewrites every line ending in the file: read converts CRLF to LF, write
    # converts it back on Windows and to LF everywhere else. Pointing this at a
    # CRLF file from a LF machine would reformat the whole thing, which is a
    # whole-file diff in `git status` for a tool that claims to touch only its own
    # block. The promise below is byte-for-byte; this is what makes it true.
    existing = ""
    if path.exists():
        with open(path, encoding="utf-8", errors="surrogateescape", newline="") as fh:
            existing = fh.read()
    # Match the file's prevailing line ending so the inserted block does not leave
    # a mixed-ending file behind.
    nl = "\r\n" if "\r\n" in existing else "\n"
    block = block.replace("\r\n", "\n").replace("\n", nl)
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if pattern.search(existing):
        updated = pattern.sub(lambda _m: block, existing, count=1)
        if updated == existing:
            return "unchanged"
        _write(path, updated)
        return "updated"
    sep = (
        ""
        if not existing or existing.endswith(nl + nl)
        else (nl if existing.endswith(nl) else nl + nl)
    )
    _write(path, existing + sep + block + nl)
    return "created" if not existing else "updated"


def _write(path: Path, text: str) -> None:
    """Write without newline translation — see apply_block."""
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
        fh.write(text)


def is_tracked_instruction_file(path: Path) -> bool:
    """Whether *path* is a shared, committed instruction file rather than a local one.

    `CLAUDE.local.md` and friends are gitignored by convention and are the user's
    own scratch; `CLAUDE.md` / `AGENTS.md` are reviewed by their teammates. distil
    writing into the second without being asked is editing a repo it was invited to
    observe, so that requires --force.
    """
    return ".local." not in path.name

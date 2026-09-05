"""The exact-quote guarantee, where most reads actually happen: the shell.

1.49.0 keyed the exemption on the tool NAME. On 2,489 measured Claude Code sessions that
missed 33.6% of tool-result mass — whole-file reads issued as `cat`/`head`/`tail`/`nl`/
`sed -n`, which digested and left 39.3% of later Edit quotes with no byte-exact copy in
the forwarded view. These tests pin the classifier that closes it, and the guarantee it
serves: an Edit's `old_string` must still be findable in what distil forwards.
"""

from __future__ import annotations

import pytest

from distil.compress.provenance import (
    ToolCall,
    command_text,
    exact_quote_ids,
    observed_view,
    quote_hazard,
    read_span,
    required_quotes,
    whole_file_read_paths,
)

# --------------------------------------------------------------------------- classifier


@pytest.mark.parametrize(
    ("command", "paths"),
    [
        # --- the readers, plainly ---
        ("cat /app/main.py", ("/app/main.py",)),
        ("cat a.py b.py", ("a.py", "b.py")),
        ("cat -n /app/main.py", ("/app/main.py",)),  # numbered, still the file's bytes
        ("head -n 50 /app/main.py", ("/app/main.py",)),  # -n takes a VALUE, not a path
        ("head -50 /app/main.py", ("/app/main.py",)),
        ("tail -n 100 logs/app.log", ("logs/app.log",)),
        ("nl -ba /app/main.py", ("/app/main.py",)),
        ("bat /app/main.py", ("/app/main.py",)),
        ("/bin/cat /app/main.py", ("/app/main.py",)),  # absolute path to the binary
        ("cat '/app/my file.py'", ("/app/my file.py",)),  # quoted path
        ('cat "/app/my file.py"', ("/app/my file.py",)),
        ("cat -- -weird-name.py", ("-weird-name.py",)),
        # --- sed, only as a pure range printer ---
        ("sed -n '10,40p' /app/main.py", ("/app/main.py",)),
        ("sed -n '5p' /app/main.py", ("/app/main.py",)),
        ("sed -n '10,$p' /app/main.py", ("/app/main.py",)),
        ("sed -ne '1,20p' /app/main.py", ("/app/main.py",)),
        # --- sequences: every stage must be a read, and the paths union ---
        ("cat a.py && cat b.py", ("b.py",)),
        ("cat a.py; cat b.py", ("b.py",)),
        ("cat a.py || cat b.py", ("b.py",)),
        # The shape an agent actually uses to read a file. Requiring every stage to be
        # a reader would refuse this one and digest the quote.
        ("cd /repo && cat main.py", ("main.py",)),
        ("cd /repo; pytest -q; cat main.py", ("main.py",)),
    ],
)
def test_whole_file_reads_are_recognised(command: str, paths: tuple[str, ...]) -> None:
    assert whole_file_read_paths(command) == paths


@pytest.mark.parametrize(
    "command",
    [
        # A pipe means the agent saw the PIPELINE's output, not the file's. The bytes it
        # could quote back are grep's, and those are not what a later Edit matches on.
        "cat /app/main.py | grep def",
        "head -n 50 /app/main.py | tail -n 5",
        # Redirection sends the bytes to a file; the tool result is empty.
        "cat /app/main.py > /tmp/copy.py",
        "cat /app/main.py >> /tmp/copy.py",
        "cat /app/main.py | tee /tmp/copy.py",
        # sed as a transform, not a printer — the output is not the file's bytes.
        "sed 's/foo/bar/' /app/main.py",
        "sed -n 's/foo/bar/p' /app/main.py",
        "sed -i 's/foo/bar/' /app/main.py",
        # Not readers at all.
        "grep -rn TODO .",
        "pytest -q",
        "git show HEAD:/app/main.py",
        "awk '{print $1}' /app/main.py",
        # A reader with no file argument reads stdin — there is no path to key on.
        "cat",
        "head -n 5",
        # The last stage is what the agent read; a sequence ending in a non-reader is
        # not a file read however it started.
        "cat a.py && pytest -q",
        # A heredoc writes; those bytes are not what came back.
        "cat << EOF",
        "cat < /app/main.py",
        # Unparseable: we cannot say what it read, so we do not claim it.
        "cat 'unbalanced",
        "",
    ],
)
def test_non_reads_are_not_claimed(command: str) -> None:
    assert whole_file_read_paths(command) == ()


def test_command_text_reads_every_shape_a_shell_tool_uses() -> None:
    assert command_text({"command": "cat a.py"}) == "cat a.py"
    assert command_text({"cmd": "cat a.py"}) == "cat a.py"
    assert command_text({"command": ["cat", "a.py"]}) == "cat a.py"
    assert command_text({"file_path": "/a.py"}) == ""
    assert command_text(None) == ""
    assert command_text({"command": 7}) == ""


# --------------------------------------------------------------------------- exemption


def _shell(i: int, command: str, index: int) -> ToolCall:
    return ToolCall(id=f"t{i}", name="Bash", command=command, pos=index)


def test_named_read_tools_stay_exempt() -> None:
    """The 1.49.0 rule, unchanged: a Read is exempt however old."""
    calls = [ToolCall(id="t1", name="Read", command="", pos=1)]
    assert set(exact_quote_ids(calls)) == {"t1"}


def test_shell_read_is_now_exempt_too() -> None:
    assert set(exact_quote_ids([_shell(1, "cat /a.py", 1)])) == {"t1"}


def test_ordinary_shell_output_still_digests() -> None:
    """The exemption is provenance-scoped, not a blanket amnesty for the Bash tool."""
    assert set(exact_quote_ids([_shell(1, "pytest -q", 1)])) == set()


def test_only_the_latest_read_of_a_path_is_kept() -> None:
    """A superseded read is one the agent has a fresher byte-exact copy of."""
    calls = [_shell(1, "cat /a.py", 1), _shell(2, "cat /a.py", 5)]
    assert set(exact_quote_ids(calls)) == {"t2"}


def test_a_read_of_a_different_path_supersedes_nothing() -> None:
    calls = [_shell(1, "cat /a.py", 1), _shell(2, "cat /b.py", 5)]
    assert set(exact_quote_ids(calls)) == {"t1", "t2"}


def test_a_multi_file_read_survives_until_every_path_is_superseded() -> None:
    calls = [_shell(1, "cat /a.py /b.py", 1), _shell(2, "cat /a.py", 5)]
    assert set(exact_quote_ids(calls)) == {"t1", "t2"}, "/b.py has no fresher copy"
    calls.append(_shell(3, "cat /b.py", 9))
    assert set(exact_quote_ids(calls)) == {"t2", "t3"}


def test_disjoint_partial_reads_of_one_file_do_not_supersede_each_other() -> None:
    """Two windows onto the same file are not copies of each other.

    Keyed by path alone, `sed -n '200,280p' app.py` looked like a fresher copy of
    `sed -n '1,80p' app.py` and digested the first 80 lines — before any Edit existed, so
    the quote guard had nothing to catch and the loss was silent.
    """
    calls = [
        _shell(1, "sed -n '1,80p' app.py", 1),
        _shell(2, "sed -n '200,280p' app.py", 5),
    ]
    assert set(exact_quote_ids(calls)) == {"t1", "t2"}


def test_a_whole_file_read_supersedes_an_earlier_partial_one() -> None:
    """`cat` covers every slice, so the partial read really is redundant."""
    calls = [_shell(1, "sed -n '1,80p' app.py", 1), _shell(2, "cat app.py", 5)]
    assert set(exact_quote_ids(calls)) == {"t2"}


def test_a_narrower_reread_does_not_supersede_a_wider_one() -> None:
    """`head -20` after `head -50` shows strictly less; the 50 still has to survive."""
    calls = [_shell(1, "head -n 50 app.py", 1), _shell(2, "head -n 20 app.py", 5)]
    assert set(exact_quote_ids(calls)) == {"t1", "t2"}


def test_an_identical_reread_still_supersedes() -> None:
    """The case supersession exists for: the same slice, read again."""
    calls = [_shell(1, "head -n 50 app.py", 1), _shell(2, "head -n 50 app.py", 5)]
    assert set(exact_quote_ids(calls)) == {"t2"}


@pytest.mark.parametrize(
    ("command", "span"),
    [
        ("cat app.py", "all"),
        ("cat -n app.py", "all"),  # numbering is formatting, not extent
        ("nl -ba app.py", "all"),
        ("bat app.py", "all"),
        ("bat -r 10:20 app.py", "bat:-r 10:20"),  # ranged bat is partial
        ("head -n 50 app.py", "head:-n 50"),
        ("tail -n 5 app.py", "tail:-n 5"),
        ("sed -n '1,80p' app.py", "sed:-n 1,80p"),
        ("cd /repo && cat app.py", "all"),
    ],
)
def test_read_span_names_the_slice(command: str, span: str) -> None:
    assert read_span(command) == span


def test_a_superseded_read_inside_the_cached_prefix_is_kept_anyway() -> None:
    """Demoting a block the provider has already cached rewrites the whole prefix.

    That costs 1.25x on every byte from the flip onwards and buys one digest — the exact
    trade the recency carve-out was re-anchored to avoid. So supersession stops at the
    client's cache breakpoint.
    """
    calls = [_shell(1, "cat /a.py", 1), _shell(2, "cat /a.py", 9)]
    assert set(exact_quote_ids(calls, cached_through=4)) == {"t1", "t2"}
    assert set(exact_quote_ids(calls, cached_through=0)) == {"t2"}


def test_widen_turns_supersession_off_entirely() -> None:
    """The reaction to an observed quote miss: stop digesting the class."""
    calls = [_shell(1, "cat /a.py", 1), _shell(2, "cat /a.py", 5)]
    assert set(exact_quote_ids(calls, widen=True)) == {"t1", "t2"}


def test_the_two_rules_are_counted_separately() -> None:
    """A shell read and a Read are both exempt, but the census must say which rule froze
    each — otherwise the new rule's cost in production is only inferable, not measured."""
    calls = [
        ToolCall(id="t1", name="Read", command="", pos=1),
        _shell(2, "cat /a.py", 3),
    ]
    assert exact_quote_ids(calls) == {
        "t1": "tool_result_exact_quote",
        "t2": "tool_result_shell_read",
    }


# --------------------------------------------------------------------------- quotes


def test_required_quotes_collects_every_literal_match_shape() -> None:
    inputs = [
        ("Edit", {"old_string": "alpha"}),
        ("MultiEdit", {"edits": [{"old_string": "beta"}, {"old_string": "gamma"}]}),
        ("str_replace_editor", {"old_str": "delta"}),
        ("Bash", {"old_string": "not an edit"}),
        ("Edit", {"old_string": ""}),  # empty quote matches everything; not a hazard
        ("Edit", "malformed"),
    ]
    assert required_quotes(inputs) == ["alpha", "beta", "gamma", "delta"]


def test_quote_hazard_counts_survivors_in_json_escaped_form() -> None:
    view = observed_view([{"role": "user", "content": "def f():\n    return 1"}])
    survived, lost = quote_hazard(["def f():\n    return 1", "def g():"], view)
    assert (survived, lost) == (1, 1)


def test_observed_view_excludes_the_models_own_turns() -> None:
    """An Edit block carries its own old_string, so counting it would make the hazard
    check incapable of ever failing."""
    msgs = [
        {"role": "assistant", "content": [{"type": "tool_use", "input": {"old_string": "alpha"}}]},
        {"role": "user", "content": "beta"},
    ]
    view = observed_view(msgs)
    assert "alpha" not in view
    assert "beta" in view

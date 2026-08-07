"""Python ↔ TypeScript conformance for Tier-0.

distil's value is a decision-equivalence certificate measured against the PYTHON
compressor. A JS port that merely compresses "similarly" would hand users a
guarantee that does not describe the code they run.

So the JS port must produce **byte-identical** output, and this is the gate that
proves it. Both implementations run over one shared corpus; any divergence fails.
Where the JS side cannot match Python byte-for-byte (integral floats render as
"1.0" in Python and "1" in JS, and JS objects hoist integer-like keys), it is
required to DECLINE the transform rather than emit uncertified output — that is
also checked here, so "declines" cannot quietly become "does nothing at all".

Skipped when node is unavailable; CI has it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from distil.adapters.anthropic import _apply_tier0
from distil.compress.tier0 import collapse_runs, minify_json
from distil.tokenizer import DEFAULT as _tokenizer

TIER0_JS = Path(__file__).resolve().parent.parent / "packaging" / "npm" / "lib" / "tier0.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


# One corpus, exercised by every test below. Chosen to hit the real transforms
# and the declared divergences, not to flatter either implementation.
CORPUS: dict[str, str] = {
    "plain_run": "alpha\nalpha\nalpha\nbeta\n",
    "single_line": "just one line\n",
    "no_trailing_newline": "a\na\na",
    "long_log_run": "ERROR failed to connect\n" * 50 + "done\n",
    "mixed_runs": "x\nx\ny\nz\nz\nz\nz\n",
    "run_marker_lookalike": "<<x3>>\n<<x3>>\n<<x3>>\nreal\n",
    "empty": "",
    "whitespace_only": "   \n   \n   \n",
    "blank_lines": "\n\n\n\n\n",
    "unicode": "héllo wörld ✓\nhéllo wörld ✓\n",
    "json_simple": '{"b": 1, "a": 2}',
    "json_nested": '{"outer": {"inner": [1, 2, 3], "s": "x"}}',
    "json_array": "[1, 2, 3]",
    "json_strings": '{"msg": "line1\\nline2", "esc": "quote\\" here"}',
    "json_unicode": '{"k": "héllo ✓"}',
    "json_not_json": "{this is not json}",
    "json_partial": '{"a": 1',
    "json_with_run": '{"a": 1}\n{"a": 1}\n{"a": 1}\n',
    "tool_output": "\n".join(f"file_{i}.py: ok" for i in range(30)),
    "stack_trace": ('  File "x.py", line 1\n' * 12) + "ValueError: boom\n",
}

# Inputs where the JS side is REQUIRED to decline JSON minification because it
# cannot reproduce Python's bytes. Each names the reason.
MUST_DECLINE_JSON = {
    "integral_float": '{"a": 1.0}',
    "integral_float_nested": '{"a": [2.0, 3.5]}',
    "exponent": '{"a": 1e5}',
    "integer_like_key": '{"2": "b", "1": "a"}',
    "duplicate_key": '{"a": 1, "a": 2}',
}


def _node(fn: str, payload: str) -> str:
    """Call one exported function in tier0.js with a single string argument."""
    script = (
        f"const t=require({json.dumps(str(TIER0_JS))});"
        "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
        "const a=JSON.parse(d);"
        f"const r=t.{fn}(a);"
        "process.stdout.write(JSON.stringify(r===undefined?null:r));});"
    )
    out = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, f"node failed: {out.stderr}"
    return json.loads(out.stdout)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_collapse_runs_is_byte_identical(name):
    text = CORPUS[name]
    assert _node("collapseRuns", text) == collapse_runs(text), f"collapseRuns diverged on {name!r}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_apply_tier0_is_byte_identical(name):
    """The full production path, including reject-if-bigger by token count."""
    text = CORPUS[name]
    assert _node("applyTier0", text) == _apply_tier0(text), f"applyTier0 diverged on {name!r}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_token_counts_agree(name):
    """The port's reject-if-bigger decision depends on matching Python's count.

    Python's round() is round-half-to-even and JS's Math.round is round-half-up,
    so a naive port silently disagrees at specific lengths and then makes a
    different keep/discard call.
    """
    text = CORPUS[name]
    assert _node("countTokens", text) == _tokenizer.count(text), f"token count diverged on {name!r}"


# expandRuns is a textual inverse, and no textual inverse can separate a
# GENERATED "<<xN>>" marker from original content that looks like one. Inputs
# containing such lookalikes are excluded here and covered by the safety test
# below instead — which asserts the property that actually protects data:
# collapseRuns refuses to collapse them at all.
_MARKER_LOOKALIKE = {"run_marker_lookalike"}


@pytest.mark.parametrize("name", sorted(set(CORPUS) - _MARKER_LOOKALIKE))
def test_collapse_runs_round_trips_in_js(name):
    """Reversibility must hold in the port, not only in Python."""
    text = CORPUS[name]
    collapsed = _node("collapseRuns", text)
    assert _node("expandRuns", collapsed) == text, f"JS collapse/expand lost bytes on {name!r}"


def test_runs_of_marker_lookalikes_are_never_collapsed():
    """The real safety property, in both implementations.

    Collapsing a run of lines that already look like markers would make the
    generated marker indistinguishable from original content. Both sides must
    leave such input byte-exact rather than produce something unrecoverable.
    """
    text = CORPUS["run_marker_lookalike"]
    assert collapse_runs(text) == text, "python collapsed a marker lookalike"
    assert _node("collapseRuns", text) == text, "js collapsed a marker lookalike"


@pytest.mark.parametrize("name", sorted(MUST_DECLINE_JSON))
def test_js_declines_json_it_cannot_reproduce(name):
    """Declining is the contract; emitting different bytes is the bug."""
    text = MUST_DECLINE_JSON[name]
    assert _node("minifyJson", text) is None, (
        f"JS minified {name!r} instead of declining — output would be uncertified"
    )


@pytest.mark.parametrize("name", sorted(MUST_DECLINE_JSON))
def test_declining_still_returns_the_input_unharmed(name):
    """A decline must fall through to byte-exact text, not to empty output."""
    text = MUST_DECLINE_JSON[name]
    out = _node("applyTier0", text)
    assert out == text, f"declined input was altered: {out!r}"


def test_js_actually_minifies_the_safe_cases():
    """Guard against 'conformance' achieved by declining everything."""
    assert _node("minifyJson", '{"b": 1, "a": 2}') == '{"b":1,"a":2}'
    assert _node("minifyJson", "[1, 2, 3]") == "[1,2,3]"
    assert _node("minifyJson", '{"b": 1, "a": 2}') == minify_json('{"b": 1, "a": 2}')


def test_corpus_actually_exercises_compression():
    """A corpus where nothing compresses would make every assertion vacuous."""
    changed = [n for n, t in CORPUS.items() if _apply_tier0(t) != t]
    assert len(changed) >= 6, f"only {len(changed)} corpus entries compress at all"

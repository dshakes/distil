"""The public library API: `from distil import compress_messages, expand_handle`.

This is the surface third parties depend on, so it is tested as a *contract*:
lazy import stays lazy, input is never mutated, non-text passes through, the
model's own turns are never rewritten, and every handle round-trips byte-exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import distil
from distil import CompressionResult, compress_messages, expand_handle

# A tool result big and repetitive enough to be worth digesting.
BIG_LOG = (
    "\n".join(
        f"2026-08-07T12:{i:02d}:00Z INFO  worker.pool  heartbeat ok seq={i} latency_ms=12"
        for i in range(400)
    )
    + "\nFATAL  worker.pool  pool exhausted after 400 heartbeats\n"
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Give every test its own restore store.

    These tests write digest originals to disk and read them back. Sharing
    DISTIL_HOME with the rest of the suite makes them order-dependent: another
    test's TTL sweep or purge can delete a handle between compress and expand.
    Isolation is what makes the round-trip assertion mean what it says.
    """
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


def _msgs() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why did the worker pool die?"},
        {"role": "tool", "content": BIG_LOG},
    ]


def test_import_distil_does_not_pull_the_compression_stack() -> None:
    """`import distil` must stay cheap — the CLI does it on every --version."""
    code = "import sys, distil; print('distil.adapters.anthropic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_lazy_export_resolves_and_unknown_attr_raises() -> None:
    assert distil.compress_messages is compress_messages
    assert "compress_messages" in dir(distil)
    try:
        _ = distil.no_such_thing  # type: ignore[attr-defined]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError for unknown attribute")


def test_compress_returns_result_and_saves_tokens() -> None:
    r = compress_messages(_msgs())
    assert isinstance(r, CompressionResult)
    assert r.tokens_before > 0
    assert r.tokens_after < r.tokens_before, "a 400-line log should compress"
    assert r.tokens_saved == r.tokens_before - r.tokens_after
    assert 0 < r.saved_pct <= 100


def test_input_list_is_never_mutated() -> None:
    original = _msgs()
    snapshot = json.dumps(original)
    compress_messages(original)
    assert json.dumps(original) == snapshot


def test_assistant_turns_are_never_rewritten() -> None:
    msgs = [{"role": "assistant", "content": BIG_LOG}]
    out = compress_messages(msgs).messages
    assert out[0]["content"] == BIG_LOG
    assert out[0] is msgs[0], "unchanged messages should pass through by identity"


def test_non_string_content_passes_through() -> None:
    blocks = [{"type": "image", "source": {"data": "..."}}]
    msgs = [{"role": "user", "content": blocks}]
    out = compress_messages(msgs).messages
    assert out[0]["content"] is blocks


def test_every_handle_round_trips_byte_exactly() -> None:
    r = compress_messages(_msgs())
    assert r.handles, "a digested 400-line log should produce at least one handle"
    for h in r.handles:
        restored = expand_handle(h)
        assert restored is not None, f"handle {h} did not resolve"
        assert restored in BIG_LOG or BIG_LOG in restored


def test_expand_of_unknown_handle_returns_none_not_raises() -> None:
    assert expand_handle("deadbeef") is None
    assert expand_handle("not-a-handle") is None


def test_verbatim_mode_creates_no_handles() -> None:
    r = compress_messages(_msgs(), verbatim=True)
    assert r.handles == []


def test_duck_typed_objects_work_without_a_framework() -> None:
    class Msg:  # mimics a LangChain BaseMessage closely enough
        def __init__(self, role: str, content: str) -> None:
            self.role, self.content = role, content

        def model_copy(self, *, update: dict[str, object]) -> "Msg":
            return Msg(self.role, str(update.get("content", self.content)))

    out = compress_messages([Msg("tool", BIG_LOG)]).messages
    assert isinstance(out[0], Msg)
    assert len(out[0].content) < len(BIG_LOG)


def test_empty_input_is_safe() -> None:
    r = compress_messages([])
    assert r.messages == [] and r.tokens_before == 0 and r.saved_pct == 0.0


# --- T0.1: first-run honesty -------------------------------------------------
# `▼413 · 0% smaller` is a real number next to a zero. It reads as broken, and it
# is the most likely first-run uninstall trigger for traffic that legitimately
# compresses to very little.


def test_sub_one_percent_never_renders_as_zero() -> None:
    from distil.ledger import pct_smaller_label

    assert pct_smaller_label(0.004) == "<1% smaller"  # the reported ▼413 case
    assert pct_smaller_label(0.0001) == "<1% smaller"


def test_real_percentages_are_unchanged() -> None:
    from distil.ledger import pct_smaller_label

    assert pct_smaller_label(0.62) == "62% smaller"
    assert pct_smaller_label(0.015) == "2% smaller"
    assert pct_smaller_label(0.0) == "0% smaller"


def test_no_public_export_collides_with_any_submodule() -> None:
    """No name in `distil.__all__` may match a submodule of `distil`.

    Python binds a submodule onto its parent package the moment anything imports
    it, silently overwriting a same-named PEP-562 `__getattr__` export. Such a name
    resolves to the function in a fresh interpreter and to the MODULE in any program
    that happened to import the submodule — the same import yielding two different
    objects depending on unrelated import order.

    This bit twice: `compress` (distil/compress/) and `expand` (distil/expand.py),
    the second only under a full-suite run where an earlier test had imported the
    module. Enumerating submodules catches the next one at authoring time.
    """
    import pkgutil

    import distil as pkg

    submodules = {m.name for m in pkgutil.iter_modules(pkg.__path__)}
    clashes = sorted(set(pkg.__all__) & submodules)
    assert not clashes, (
        f"public export(s) {clashes} shadow submodule(s) of the same name; "
        f"rename the export (e.g. compress -> compress_messages)"
    )


def test_public_exports_survive_their_submodules_being_imported() -> None:
    """The exports must still be callable after the neighbouring modules load."""
    import importlib
    import types

    importlib.import_module("distil.compress.tier1")
    importlib.import_module("distil.expand")

    assert isinstance(distil.compress, types.ModuleType)
    assert isinstance(distil.expand, types.ModuleType)
    assert callable(distil.compress_messages)
    assert callable(distil.expand_handle)


def test_every_percentage_surface_uses_one_rule() -> None:
    """All four render sites must agree — this bug was fixed three times before
    the web dashboard was noticed.

    Sites: the Claude Code status line (cli), the terminal dashboard and the HTML
    session card (ledger), and the browser dashboard (webdash, via a preformatted
    `pct_label` so the JS cannot re-derive a rounded "0.0%").
    """
    import inspect

    from distil import cli, ledger, webdash

    for mod in (cli, ledger):
        src = inspect.getsource(mod)
        # Python surfaces must all route through the one helper.
        assert '% smaller"' not in src or "pct_smaller_label" in src, (
            f"{mod.__name__} formats a savings percentage without the shared helper"
        )

    # The browser dashboard cannot import Python, so it mirrors the rule in JS —
    # and must derive it from the raw counts, because the rounded `pct` collapses
    # a real-but-tiny saving to 0.0 before the browser ever sees it.
    js = inspect.getsource(webdash)
    assert "function pctLabel(" in js, "webdash must mirror the <1% rule"
    assert 'd.pct+"%"' not in js, "webdash must not render the rounded pct directly"


def test_webdash_payload_carries_a_non_broken_label(tmp_path, monkeypatch) -> None:
    from distil.ledger import pct_smaller_label

    # The label the browser receives for a real-but-tiny saving must not read as 0.
    assert pct_smaller_label(0.004) == "<1% smaller"

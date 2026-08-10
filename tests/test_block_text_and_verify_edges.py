"""Shapes that arrive from real providers, and a signature check that must fail closed.

`_block_text` decides what counts as billable text in a message. Content blocks
arrive in several shapes and providers add new ones; anything it fails to read
is silently priced at zero, which understates the very number distil exists to
report. `verify` guards a signed savings aggregate — every malformed input has
to come back False, never raise and never accidentally pass.
"""

from __future__ import annotations

from distil.simulate import _block_text
from distil.telemetry import SavingsAggregate, sign, verify


class TestBlockText:
    def test_a_bare_string_is_its_own_text(self) -> None:
        assert _block_text("hello") == "hello"

    def test_a_text_block(self) -> None:
        assert _block_text({"type": "text", "text": "hello"}) == "hello"

    def test_a_nested_content_list_is_flattened(self) -> None:
        block = {"content": [{"text": "a"}, {"text": "b"}]}
        assert _block_text(block) == "a\nb"

    def test_a_tool_use_block_is_priced_by_its_arguments(self) -> None:
        """These are frequently the largest blocks in an agent transcript.
        Reading them as empty skipped them entirely from the preview."""
        out = _block_text({"type": "tool_use", "name": "Read", "input": {"file_path": "/x"}})
        assert "/x" in out

    def test_unserialisable_arguments_fall_back_to_str(self) -> None:
        """A tool argument that json.dumps cannot handle must still be priced,
        not raise while rendering a report."""

        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        out = _block_text({"type": "tool_use", "name": "X", "input": {"o": Weird()}})
        assert out  # priced as something, not dropped and not an exception

    def test_a_tool_use_with_no_arguments_falls_back_to_its_name(self) -> None:
        assert _block_text({"type": "tool_use", "name": "Ping"}) == "Ping"

    def test_shapes_with_no_text_are_empty_not_errors(self) -> None:
        assert _block_text(None) == ""
        assert _block_text(42) == ""
        assert _block_text({"type": "image"}) == ""


class TestVerifyFailsClosed:
    def _agg(self) -> SavingsAggregate:
        return SavingsAggregate(
            instance_id="i1",
            tokens_saved=600,
            dollars_saved=1.5,
            runs=3,
            certified=True,
            ts=1786386000.0,
        )

    def test_a_signature_made_with_the_key_verifies(self) -> None:
        signed = sign(self._agg(), "k")
        assert verify(signed, "k") is True

    def test_a_different_key_does_not_verify(self) -> None:
        assert verify(sign(self._agg(), "k"), "other") is False

    def test_tampering_with_a_field_invalidates_it(self) -> None:
        signed = sign(self._agg(), "k")
        signed["tokens_saved"] = 999_999
        assert verify(signed, "k") is False

    def test_an_unknown_field_is_rejected_rather_than_raising(self) -> None:
        """A payload from a newer (or hostile) client must fail closed: the
        aggregate cannot be reconstructed, so it cannot be trusted."""
        signed = sign(self._agg(), "k")
        signed["surprise"] = "field"
        assert verify(signed, "k") is False

    def test_a_payload_with_no_signature_is_rejected(self) -> None:
        unsigned = {"instance_id": "i1", "runs": 1}
        assert verify(unsigned, "k") is False

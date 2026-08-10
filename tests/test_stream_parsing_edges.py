"""Malformed SSE must degrade, never crash the stream.

Every branch here sits between the provider and the user's agent. A provider
that emits a keep-alive comment, splits a frame across TCP reads, or ends the
body without a trailing blank line is not misbehaving — it is ordinary SSE. If
any of those raise, the user does not get a degraded response, they get a dead
stream mid-answer, which is indistinguishable from a provider outage.

These were the uncovered branches in streamexpand/claude_code: all of them error
paths, i.e. exactly the ones nobody exercises by hand.
"""

from __future__ import annotations

import json

from distil import streamexpand
from distil.transcripts.claude_code import ClaudeCodeAdapter


class TestParseEvent:
    def test_a_well_formed_event_parses(self) -> None:
        assert streamexpand._parse_event(b'data: {"type":"ping"}') == {"type": "ping"}

    def test_multi_line_data_is_concatenated(self) -> None:
        """SSE allows a payload split over several data: lines."""
        raw = b'data: {"type":\ndata: "ping"}'
        assert streamexpand._parse_event(raw) == {"type": "ping"}

    def test_an_event_with_no_data_line_is_skipped(self) -> None:
        """Comments and bare event: lines carry no payload — not an error."""
        assert streamexpand._parse_event(b": keep-alive") is None
        assert streamexpand._parse_event(b"event: message_stop") is None

    def test_the_done_sentinel_is_not_an_event(self) -> None:
        assert streamexpand._parse_event(b"data: [DONE]") is None

    def test_malformed_json_is_swallowed_not_raised(self) -> None:
        """A truncated payload must not take the stream down with it."""
        assert streamexpand._parse_event(b'data: {"type": "ping"') is None

    def test_a_non_object_payload_is_rejected(self) -> None:
        """Valid JSON that is not an object would break every downstream
        `evt.get(...)`, so it is filtered here rather than at each use."""
        assert streamexpand._parse_event(b"data: [1, 2, 3]") is None
        assert streamexpand._parse_event(b'data: "just a string"') is None


class TestIterEvents:
    @staticmethod
    def _reader(chunks: list[bytes]):
        it = iter(chunks)
        return lambda: next(it, b"")

    def test_events_split_across_reads_are_reassembled(self) -> None:
        """A frame arriving in two TCP reads is one event, not zero or two."""
        chunks = [b'data: {"n":', b'1}\n\ndata: {"n":2}\n\n']
        assert list(streamexpand._iter_events(self._reader(chunks))) == [{"n": 1}, {"n": 2}]

    def test_a_trailing_event_without_a_separator_is_still_yielded(self) -> None:
        """Bodies that end without a final blank line are common. Dropping the
        last event loses the end of the answer, silently."""
        chunks = [b'data: {"n":1}\n\ndata: {"last":true}']
        assert list(streamexpand._iter_events(self._reader(chunks)))[-1] == {"last": True}

    def test_trailing_whitespace_is_not_an_event(self) -> None:
        chunks = [b'data: {"n":1}\n\n\n\n']
        assert list(streamexpand._iter_events(self._reader(chunks))) == [{"n": 1}]

    def test_crlf_separators_are_honoured(self) -> None:
        chunks = [b'data: {"n":1}\r\n\r\ndata: {"n":2}\r\n\r\n']
        assert list(streamexpand._iter_events(self._reader(chunks))) == [{"n": 1}, {"n": 2}]


class TestFirstTimestamp:
    """`_first_ts` decides which transcript overlaps a session. Every failure
    here must read as "no timestamp" so the file is merely ranked last, never
    as an exception that aborts discovery for every other file too."""

    def test_the_first_timestamped_line_wins(self, tmp_path) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text(
            json.dumps({"timestamp": "2026-08-10T12:00:00Z"})
            + "\n"
            + json.dumps({"timestamp": "2026-08-10T13:00:00Z"})
            + "\n"
        )
        assert ClaudeCodeAdapter._first_ts(p) > 0

    def test_a_corrupt_leading_line_is_stepped_over(self, tmp_path) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text("{ not json\n" + json.dumps({"timestamp": "2026-08-10T12:00:00Z"}) + "\n")
        assert ClaudeCodeAdapter._first_ts(p) > 0

    def test_a_file_with_no_timestamps_is_zero_not_an_error(self, tmp_path) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text(json.dumps({"no": "timestamp"}) + "\n")
        assert ClaudeCodeAdapter._first_ts(p) == 0.0

    def test_an_empty_file_is_zero(self, tmp_path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert ClaudeCodeAdapter._first_ts(p) == 0.0

    def test_an_unreadable_file_is_zero(self, tmp_path) -> None:
        """A transcript belonging to another user, or deleted between the glob
        and the read, must not abort discovery."""
        assert ClaudeCodeAdapter._first_ts(tmp_path / "does-not-exist.jsonl") == 0.0

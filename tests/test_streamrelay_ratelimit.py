"""`capture_ratelimit` — the provider quota headers the proxy records per request."""

from __future__ import annotations

from email.message import Message

from distil.streamrelay import capture_ratelimit


def _headers(pairs: dict[str, str]) -> Message:
    msg = Message()
    for key, value in pairs.items():
        msg[key] = value
    return msg


def test_extracts_ratelimit_headers_case_insensitively_and_strips_the_vendor_prefix() -> None:
    got = capture_ratelimit(
        _headers(
            {
                "Anthropic-RateLimit-Requests-Limit": "50",
                "anthropic-ratelimit-input-tokens-remaining": "1900000",
                "Content-Type": "text/event-stream",
            }
        )
    )
    assert got == {"requests-limit": "50", "input-tokens-remaining": "1900000"}


def test_absent_headers_record_nothing_rather_than_an_empty_dict() -> None:
    """`{}` and "the provider sent none" would look identical on disk, and only one of
    them means the capture is working."""
    assert capture_ratelimit(_headers({"Content-Type": "application/json"})) is None


def test_a_hostile_headers_object_cannot_break_the_relay() -> None:
    """This runs on every response, including error paths. A diagnostic that can raise
    here would turn a recoverable upstream error into a failed request."""

    class Exploding:
        def items(self):
            raise RuntimeError("headers unavailable")

    assert capture_ratelimit(Exploding()) is None

"""Request-path guards shared by the proxy, async proxy and gateway.

The reason these live in one module is that they were not always shared: the
proxy refused a Transfer-Encoding body and the gateway read it as empty, which
is a request-smuggling difference between two servers that are documented as
handling requests identically. Tested here at the unit level; the end-to-end
desync is in tests/test_gateway_cov.py and tests/test_proxy_cov.py.
"""

from __future__ import annotations

import pytest

from distil.httpguard import framing_rejection


def test_a_plain_content_length_request_is_accepted() -> None:
    assert framing_rejection("42", None) is None
    assert framing_rejection(None, None) is None
    assert framing_rejection(None, "") is None


def test_transfer_encoding_alone_is_411() -> None:
    """Nothing here reads a TE body, so it would be read as empty and its bytes
    left queued on the socket — the next parse then sees a second request."""
    status, msg = framing_rejection(None, "chunked")
    assert status == 411
    assert "Content-Length" in msg


@pytest.mark.parametrize("te", ["chunked", "gzip, chunked", " Chunked ", "gzip", "identity"])
def test_any_transfer_encoding_is_refused_not_just_the_word_chunked(te: str) -> None:
    """The old guard matched the literal ``chunked``. The property that matters
    is not the codec name — it is that the body length does not come from
    Content-Length, which is the only framing these servers can read."""
    assert framing_rejection(None, te) is not None


def test_both_headers_together_is_400() -> None:
    """Two framings on one request: the TE.CL desync pair stated outright."""
    status, msg = framing_rejection("42", "chunked")
    assert status == 400
    assert "conflicting" in msg

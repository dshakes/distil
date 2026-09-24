"""RESEARCH-ONLY — not on the request path; see docs/research.html.

Content-addressed gist caching for static assets (tool schemas, system prompts).

This is the model-agnostic realization of "send the static prefix once, reference
it forever."  The key insight: large static payloads — tool schemas, system prompts,
retrieval preambles — are re-transmitted verbatim on every request even though they
never change.  A content-addressed cache lets a *proxy* or *session layer* detect
the repeat, swap the full text for a compact ref on the wire, and swap it back
before delivery.  The mechanism works against any API without provider cooperation.

Honest scope note:
    A *true* soft-prompt / gist-token approach additionally compresses the prompt
    into reusable KV-cache / soft tokens so the model never has to re-process them.
    That requires self-hosted or provider-side support (e.g., Anthropic's prompt
    caching, a local vLLM prefix cache).  The content-addressed version here is the
    portable fallback: it saves *transmission* tokens and, when combined with a
    caching-aware API client, also saves *processing* tokens — but it cannot give you
    KV-cache sharing on a black-box API unless that API independently implements
    prefix caching on the same canonical prefix position.
"""

from __future__ import annotations

import hashlib

from distil.tokenizer import DEFAULT, Tokenizer

_PREFIX = "gist:"
_HASH_LEN = 8


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:_HASH_LEN]


class GistCache:
    """In-process content-addressed store for large static text assets.

    Usage::

        cache = GistCache()
        ref = cache.register(big_schema_json)   # "gist:a3f9c12e"
        # … send ref over the wire or store it …
        original = cache.materialize(ref)        # round-trips perfectly
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}  # ref → original text
        self._seen: set[str] = set()  # refs already registered at least once
        self.registrations: int = 0  # total register() calls
        self.hits: int = 0  # register() calls for already-seen text

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register(self, text: str) -> str:
        """Store *text* and return its stable gist ref.

        Identical text always returns the same ref (automatic dedup).  A second
        call with the same text is counted as a *hit* — the caller would have
        paid full token cost on a repeat transmission but now pays only the ref.
        """
        self.registrations += 1
        ref = _PREFIX + _sha(text)
        if ref in self._seen:
            self.hits += 1
        else:
            self._store[ref] = text
            self._seen.add(ref)
        return ref

    def materialize(self, ref: str) -> str:
        """Return the original text for a previously registered ref.

        Raises:
            KeyError: if the ref was never registered in this cache instance.
        """
        if ref not in self._store:
            raise KeyError(ref)
        return self._store[ref]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def is_ref(text: str) -> bool:
        """Return True if *text* looks like a gist ref rather than content."""
        return (
            text.startswith(_PREFIX)
            and len(text) == len(_PREFIX) + _HASH_LEN
            and all(c in "0123456789abcdef" for c in text[len(_PREFIX) :])
        )

    def savings(self, text: str, tok: Tokenizer | None = None) -> int:
        """Tokens saved on a repeat send by transmitting the ref instead of *text*.

        Returns ``count(text) - count(ref)``.  A positive number means the ref
        is cheaper.  For a large tool schema the saving is typically hundreds of
        tokens per call.

        Args:
            text: the original static asset (not a ref).
            tok:  tokenizer to use; defaults to ``distil.tokenizer.DEFAULT``.
        """
        if tok is None:
            tok = DEFAULT
        ref = _PREFIX + _sha(text)
        return tok.count(text) - tok.count(ref)

    @property
    def dedup_rate(self) -> float:
        """Fraction of register() calls that were duplicate (hit) transmissions.

        Returns 0.0 when no registrations have been made.
        """
        if self.registrations == 0:
            return 0.0
        return self.hits / self.registrations

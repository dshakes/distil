"""RESEARCH-ONLY — not on the request path; see docs/research.html.

Phase 7 — BM25-filtered partial retrieval from a compression handle.

Provides a hand-rolled Okapi BM25 index (stdlib-only) and two helpers for
pulling back only the relevant slice of a digested original instead of the
whole thing.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lowercase word-boundary tokenization."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Okapi BM25 index over a list of string documents.

    Parameters
    ----------
    docs:
        The documents to index.
    k1:
        Term-frequency saturation parameter (default 1.5).
    b:
        Length normalisation parameter (default 0.75).
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = len(docs)

        # Per-document token frequencies and lengths
        self._tf: list[Counter[str]] = []
        self._dl: list[int] = []
        for doc in docs:
            tokens = _tokenize(doc)
            self._tf.append(Counter(tokens))
            self._dl.append(len(tokens))

        self._avgdl = sum(self._dl) / self.N if self.N else 1.0

        # Document frequency: number of docs containing each term
        self._df: Counter[str] = Counter()
        for tf in self._tf:
            for term in tf:
                self._df[term] += 1

    def _idf(self, term: str) -> float:
        """Okapi BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))."""
        df = self._df.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k (doc_index, score) pairs sorted by score descending.

        Uses the standard Okapi BM25 formula:

            score(d, q) = Σ  idf(t) * tf(t,d) * (k1+1)
                           t         ─────────────────────────────────────
                                     tf(t,d) + k1*(1-b + b*|d|/avgdl)
        """
        query_terms = _tokenize(query)
        scores: list[float] = [0.0] * self.N

        for term in query_terms:
            idf = self._idf(term)
            for i, tf in enumerate(self._tf):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                dl_norm = 1.0 - self.b + self.b * self._dl[i] / self._avgdl
                scores[i] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * dl_norm)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:k] if score > 0.0]


def expand_handle(original_text: str, query: str, k: int = 5) -> str:
    """Return up to *k* lines from *original_text* most relevant to *query*.

    Lines are BM25-ranked and the top-k are returned **in their original
    order** (partial retrieval — not reordered by score).
    """
    lines = original_text.splitlines()
    if not lines:
        return ""

    index = BM25Index(lines)
    hits = index.search(query, k=k)

    # Collect the matched indices and restore original document order
    matched_indices = sorted(idx for idx, _score in hits)
    return "\n".join(lines[i] for i in matched_indices)


def expand_handle_from_store(handle: str, store: object, query: str, k: int = 5) -> str:
    """Look up *handle* in *store* and call :func:`expand_handle`.

    *store* is duck-typed — it must expose an ``expand(handle)`` method that
    returns the original text for the given handle (compatible with
    ``RestoreStore`` / plain ``dict``-based restore maps from the compress
    pipeline).
    """
    original_text: str = store.expand(handle)  # type: ignore[attr-defined]
    return expand_handle(original_text, query, k=k)

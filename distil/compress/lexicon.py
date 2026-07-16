"""Semantic bridge — cheap, zero-runtime-dependency ways to match a query term to an answer
term that does not share its spelling. This is what lets query-aware salience anchor a line
like ``max_attempts = 5`` to the query "what's the retry limit?" even though they share no word.

Four composable mechanisms, unioned (`bridges`):

  ① Morphology — a suffix-stripping stemmer, so ``retries``/``retrying`` match ``retry``.
  ① Curated synonyms — a small hand-built technical alias map (retry↔attempt, limit↔max/cap…).
  ③ Char-trigram overlap — Jaccard over character trigrams, for typos + near-morphology.
  ④ Distributional vectors — an **optional** vector table (pure-Python cosine; no torch /
     numpy at runtime) loaded from ``semantic_vectors.json`` when present. None ships — a
     fresh install runs bridges ①–③ only, and a missing table scores 0.0 (never an error).
  ② Flywheel-learned associations live in :mod:`distil.query_assoc` (they need the hashed
     co-occurrence table) and are consulted by the caller, not here.

**Why this is safe to be imperfect.** Every bridge is used only to *widen* the query-relevant
keep set, which is unioned additively into the digest. A wrong bridge keeps a line that wasn't
needed — a little wasted compression — it can never drop a needed line or change an answer. So
the thresholds are tuned conservatively for precision, but a miss on either side is structurally
safe (reversibility + the decision-equivalence certificate are untouched).
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

# Split compound identifiers into their parts so `max_attempts` can bridge to `max`+`attempts`:
# snake_case, kebab-case, dotted paths, and camelCase boundaries.
_SUBTOKEN_SPLIT = re.compile(r"[_\-./]+|(?<=[a-z0-9])(?=[A-Z])")


def subtokens(term: str) -> set[str]:
    """The compound-identifier parts of *term* (lowercased, length >= 2), e.g.
    ``max_attempts`` -> {``max``, ``attempts``}. The whole term is added by the caller."""
    return {p.lower() for p in _SUBTOKEN_SPLIT.split(term) if len(p) >= 2}


# --------------------------------------------------------------------------- ① morphology
# Ordered longest-first so "ations" strips before "ation"/"tion"/"s". Conservative: only strip
# when a reasonable stem remains, so we never collapse short tokens into noise.
_SUFFIXES: tuple[str, ...] = (
    "ations",
    "ization",
    "isation",
    "ing",
    "ings",
    "ies",
    "ment",
    "ness",
    "tion",
    "sion",
    "ers",
    "er",
    "ed",
    "es",
    "ly",
    "s",
)
_MIN_STEM = 3


@lru_cache(maxsize=4096)
def stem(term: str) -> str:
    """A tiny, deterministic suffix stemmer (not Porter — deliberately minimal and safe).
    Strips at most one suffix, and only when >= _MIN_STEM characters remain."""
    t = term.lower()
    for suf in _SUFFIXES:
        if len(t) - len(suf) >= _MIN_STEM and t.endswith(suf):
            return t[: -len(suf)]
    return t


# --------------------------------------------------------------------------- ① synonyms
# Small, high-precision technical alias groups. Every term in a group is treated as a synonym
# of every other. Kept deliberately conservative — this is a floor, not an ontology.
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"retry", "retries", "attempt", "attempts", "reattempt", "tries"}),
    frozenset({"limit", "max", "maximum", "cap", "threshold", "ceiling", "bound", "upper"}),
    frozenset({"min", "minimum", "floor", "lower", "least"}),
    frozenset({"timeout", "deadline", "ttl", "expiry", "expiration", "expires"}),
    frozenset({"size", "count", "length", "len", "total", "num", "number", "quantity"}),
    frozenset({"error", "fail", "failure", "exception", "fault", "crash", "err"}),
    frozenset({"delay", "backoff", "interval", "wait", "sleep", "pause"}),
    frozenset({"host", "hostname", "server", "address", "addr", "endpoint", "url", "uri"}),
    frozenset({"port", "socket"}),
    frozenset({"user", "username", "account", "login", "userid", "uid"}),
    frozenset({"password", "passwd", "secret", "token", "credential", "apikey", "key"}),
    frozenset({"enable", "enabled", "active", "on", "true"}),
    frozenset({"disable", "disabled", "inactive", "off", "false"}),
    frozenset({"path", "dir", "directory", "folder", "location"}),
    frozenset({"version", "ver", "release", "revision", "rev"}),
    frozenset({"rate", "frequency", "freq", "throughput", "qps", "rps"}),
    frozenset({"memory", "mem", "ram", "heap"}),
    frozenset({"cpu", "processor", "cores", "threads"}),
    frozenset({"concurrency", "parallel", "workers", "pool", "poolsize"}),
    frozenset({"cache", "cached", "buffer"}),
)


def _build_synonyms() -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for group in _SYNONYM_GROUPS:
        for term in group:
            out.setdefault(term, set()).update(group - {term})
    return {k: frozenset(v) for k, v in out.items()}


_SYNONYMS: dict[str, frozenset[str]] = _build_synonyms()


def synonyms(term: str) -> frozenset[str]:
    return _SYNONYMS.get(term.lower(), frozenset())


# --------------------------------------------------------------------------- ③ char-trigram
def _trigrams(s: str) -> frozenset[str]:
    s = f"  {s.lower()} "
    return frozenset(s[i : i + 3] for i in range(len(s) - 2)) if len(s) >= 3 else frozenset()


def trigram_sim(a: str, b: str) -> float:
    """Jaccard similarity of character trigrams — cheap fuzzy match for typos / morphology."""
    if a == b:
        return 1.0
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / (len(ta) + len(tb) - inter)


# --------------------------------------------------------------------------- ④ vectors
_HERE = Path(__file__).resolve().parent
_VECTORS_PATH = _HERE / "semantic_vectors.json"
_VECTORS: dict[str, list[float]] | None = None
_VEC_NORMS: dict[str, float] = {}


def _load_vectors() -> dict[str, list[float]]:
    global _VECTORS
    if _VECTORS is None:
        try:
            data = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))
            _VECTORS = {k: [float(x) for x in v] for k, v in data.get("vectors", {}).items()}
        except (OSError, ValueError, json.JSONDecodeError):
            _VECTORS = {}
        for k, v in _VECTORS.items():
            _VEC_NORMS[k] = math.sqrt(sum(x * x for x in v)) or 1.0
    return _VECTORS


def embed_sim(a: str, b: str) -> float:
    """Cosine similarity of the bundled distributional vectors, or 0.0 when either term is
    out of vocabulary. Pure Python — no numpy/torch at runtime."""
    if a == b:
        return 1.0
    vecs = _load_vectors()
    va, vb = vecs.get(a.lower()), vecs.get(b.lower())
    if va is None or vb is None:
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    return dot / (_VEC_NORMS[a.lower()] * _VEC_NORMS[b.lower()])


def reset_vectors_cache() -> None:
    """Drop the cached vectors (tests, or after regenerating the table)."""
    global _VECTORS
    _VECTORS = None
    _VEC_NORMS.clear()


# --------------------------------------------------------------------------- unified bridge
# Conservative thresholds: additive-only means a false bridge only wastes a little compression,
# but we still keep precision high so we don't erode savings.
TRIGRAM_THRESHOLD = 0.5
EMBED_THRESHOLD = 0.60


def bridges(qterm: str, oterm: str) -> bool:
    """True if output-term *oterm* should count as matching query-term *qterm* via any of the
    static bridges (① morphology, ① synonyms, ③ trigram, ④ vectors)."""
    q, o = qterm.lower(), oterm.lower()
    if q == o:
        return True
    sq, so = stem(q), stem(o)
    if sq == so:  # ① morphology
        return True
    if o in _SYNONYMS.get(q, ()) or so in _SYNONYMS.get(sq, ()):  # ① synonyms (stem-aware)
        return True
    if trigram_sim(q, o) >= TRIGRAM_THRESHOLD:  # ③ fuzzy
        return True
    if embed_sim(q, o) >= EMBED_THRESHOLD:  # ④ distributional
        return True
    return False


def expanded_query(query_terms: frozenset[str]) -> frozenset[str]:
    """Query terms plus their stems and curated synonyms — a cheap static expansion used to
    widen exact-match anchors before the per-line bridge. (Trigram/vector bridges are matched
    per output-term in :func:`bridge_line_terms`, since they depend on the actual line tokens.)"""
    out: set[str] = set()
    for t in query_terms:
        tl = t.lower()
        out.add(tl)
        out.add(stem(tl))
        out |= {s.lower() for s in _SYNONYMS.get(tl, ())}
        out |= {stem(s) for s in _SYNONYMS.get(tl, ())}
    return frozenset(out)


def bridge_line_terms(line_terms: set[str], query_terms: frozenset[str]) -> bool:
    """True if any term in *line_terms* bridges to any *query_terms* — the per-line test used
    to widen the query-relevant set. Compound identifiers are split into sub-tokens first, so
    ``max_attempts`` bridges to ``limit``(→max) + ``retry``(→attempts). Cheap: exact/stem/synonym
    via the pre-expanded set, then the O(q·o) trigram/vector bridges only for residual terms."""
    if not line_terms or not query_terms:
        return False
    # Expand each line token with its compound sub-parts (max_attempts -> max, attempts).
    tokens: set[str] = set()
    for t in line_terms:
        tokens.add(t.lower())
        tokens |= subtokens(t)
    exp = expanded_query(query_terms)
    stems = {stem(t) for t in tokens}
    if tokens & exp or stems & exp:  # ① fast path (exact / stem / synonym)
        return True
    # ② learned associations — lazily loaded, a no-op (empty table) until the flywheel has
    # built one from real traffic. Kept out of the hot import path.
    from distil.query_assoc import learned_bridge

    for o in tokens:  # ②③④ residual — bounded: few query terms, few line tokens
        for q in query_terms:
            if (
                trigram_sim(q, o) >= TRIGRAM_THRESHOLD
                or embed_sim(q, o) >= EMBED_THRESHOLD
                or learned_bridge(q, o)
            ):
                return True
    return False


if __name__ == "__main__":  # self-check — each mechanism, no framework
    # ① morphology bridges inflections (attempts→attempt); a root like "retry" has no suffix,
    # so retry↔retries is carried by the synonym group instead.
    assert stem("attempts") == stem("attempt") == "attempt", (stem("attempts"), stem("attempt"))
    assert "attempt" in synonyms("retry") and "retry" in synonyms("attempt")
    assert "retries" in synonyms("retry")
    # ③ trigram catches close fuzzy pairs (config↔configs ≈ 0.67); farther inflections like
    # retry↔retries (≈ 0.4) fall to synonyms/morphology, not the fuzzy threshold.
    assert trigram_sim("config", "configs") > 0.5 and trigram_sim("retry", "banana") < 0.2
    # the money case: 'retry limit' bridges to the single token 'max_attempts' via sub-tokens
    # (max→limit, attempts→retry) — this is what anchors the semantic answer line.
    assert bridges("limit", "max") and bridges("retry", "attempts")
    assert subtokens("max_attempts") == {"max", "attempts"}
    assert bridge_line_terms({"max_attempts"}, frozenset({"retry", "limit"})) is True
    assert bridge_line_terms({"max_attempts"}, frozenset({"retry"})) is True  # via attempts
    # additive safety: a truly unrelated line does not bridge
    assert bridge_line_terms({"database_host"}, frozenset({"retry", "limit"})) is False
    # additive safety: unrelated terms don't bridge
    assert not bridges("retry", "database")
    print("lexicon self-check ok")

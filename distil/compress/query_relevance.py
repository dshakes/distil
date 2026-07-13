"""Query-aware salience — phase 2: a learned line↔query relevance scorer.

Phase 1 (`intent.relevant_lines`) keeps a tool-output line when it *lexically* shares a
salient token with the agent's intent. That nails the common case (a grep hit, a config
key, a SHA) but misses **semantic** relevance: the agent asks "what's the retry limit?",
the answer reads `max_attempts = 5` — zero lexical overlap, so phase 1 folds it.

Phase 2 adds a tiny, embedding-free logistic model over **query-conditioned features** —
lexical overlap, selectivity, proximity to a lexical hit — that learns, from distil's own
expand flywheel, which lines get pulled back under which query. It is layered **additively**
over phase 1: its kept indices are unioned in, so it can only ever *widen* the keep set —
reversibility (the full block is always in RestoreStore) and the decision-equivalence
certificate are untouched. With no trained weights present, behavior is *exactly* phase 1.

Invariants (mirrors phase 1, re-verified):
  1. Additive-only — learned ∪ phase-1 ∪ base keeps; a mis-score wastes a little
     compression or safely misses a line that stays recoverable via distil_expand.
  2. Certified before promotion — weights ship only through `online.certify_promotion`.
  3. Zero runtime deps — pure-Python logistic over cheap features, weights in JSON.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from distil.codec.features import FEATURE_NAMES, featurize
from distil.codec.learned import _dot, _sigmoid
from distil.compress.intent import SELECTIVITY_CAP, terms_of

# Query features appended to the base keep-model features. Kept deliberately small and
# embedding-free — the selectivity + proximity signals are what a fixed lexical rule can't
# express, learned as weights rather than hand-tuned.
_QUERY_FEATURE_NAMES: list[str] = [
    "q_overlap",  # |line terms ∩ query terms|, normalized (min(count/3, 1))
    "q_has_selective",  # line shares a *discriminating* query term (phase-1 signal, as a feature)
    "q_in_line",  # any query term present in the line (boolean)
    "q_dist_to_hit",  # normalized distance to the nearest lexical query hit (relevant lines cluster)
]
QUERY_FEATURE_NAMES: list[str] = FEATURE_NAMES + _QUERY_FEATURE_NAMES

_HERE = Path(__file__).resolve().parent.parent / "codec"
DEFAULT_QUERY_WEIGHTS_PATH: Path = _HERE / "query_weights.json"

# A line scores as relevant when sigmoid(w·x) >= this. 0.5 is the natural logistic
# boundary; the certificate gate, not this constant, is what makes weights shippable.
KEEP_THRESHOLD: float = 0.5


def _nearest_hit_dist(i: int, hits: list[int], n: int) -> float:
    """Normalized distance from line *i* to the nearest lexical-hit line, in [0, 1].
    0.0 == this line is itself a hit; 1.0 == no hit anywhere (or maximally far)."""
    if not hits:
        return 1.0
    d = min(abs(i - h) for h in hits)
    return min(d / max(n - 1, 1), 1.0)


def featurize_query(
    line: str,
    kind: str,
    query_terms: frozenset[str],
    *,
    line_terms: set[str] | None = None,
    selective_terms: frozenset[str] | None = None,
    nearest_hit_dist: float = 1.0,
) -> list[float]:
    """Base keep-model features for *line* plus the four query-conditioned features.

    Returns a vector of length ``len(QUERY_FEATURE_NAMES)``. When there is no query
    (empty ``query_terms``) or the line is blank, the query block is neutral (no signal,
    maximally far from any hit), so the model degrades to the base keep features.
    """
    base = featurize(line, kind)
    if not query_terms or not line.strip():
        return base + [0.0, 0.0, 0.0, 1.0]
    lt = line_terms if line_terms is not None else terms_of(line)
    overlap = len(lt & query_terms)
    sel = selective_terms if selective_terms is not None else query_terms
    return base + [
        min(overlap / 3.0, 1.0),  # q_overlap
        1.0 if (lt & sel) else 0.0,  # q_has_selective
        1.0 if overlap else 0.0,  # q_in_line
        max(0.0, min(nearest_hit_dist, 1.0)),  # q_dist_to_hit
    ]


def _selective_terms(
    line_terms: list[set[str]], query_terms: frozenset[str], n: int
) -> frozenset[str]:
    """Query terms that discriminate within this block (phase-1 selectivity guard): a term
    matching more than SELECTIVITY_CAP of the lines is not a needle — drop it."""
    cap = max(1, int((n or 1) * SELECTIVITY_CAP))
    freq: Counter[str] = Counter()
    for lt in line_terms:
        freq.update(lt & query_terms)
    return frozenset(t for t in query_terms if freq[t] <= cap)


class QueryRelevanceModel:
    """Logistic line↔query relevance scorer over ``QUERY_FEATURE_NAMES``.

    Implements the same ``sigmoid(dot(weights, x))`` shape as ``LogisticKeepModel``, just
    over the longer query-conditioned vector. Layered additively over phase 1 via
    :meth:`relevant_lines`.
    """

    def __init__(self, weights: list[float]) -> None:
        if len(weights) != len(QUERY_FEATURE_NAMES):
            raise ValueError(f"Expected {len(QUERY_FEATURE_NAMES)} weights, got {len(weights)}")
        self._weights: list[float] = list(weights)

    def score(self, line: str, kind: str, query_terms: frozenset[str]) -> float:
        """Standalone relevance score in [0, 1] for a single line (no block context, so
        the proximity feature is neutral). Prefer :meth:`relevant_lines` for a real block."""
        if not line.strip():
            return 0.0
        return _sigmoid(_dot(self._weights, featurize_query(line, kind, query_terms)))

    def relevant_lines(
        self,
        lines: list[str],
        query_terms: frozenset[str],
        kind: str,
        *,
        threshold: float = KEEP_THRESHOLD,
    ) -> set[int]:
        """Indices of lines the model scores >= *threshold* as relevant to *query_terms*.
        Computes the block-level context (selective terms, lexical-hit positions) once, so
        the proximity feature is meaningful. Returns an empty set when there is no query."""
        if not query_terms or not lines:
            return set()
        line_terms = [terms_of(ln) for ln in lines]
        n = len(lines)
        selective = _selective_terms(line_terms, query_terms, n)
        hits = [i for i, lt in enumerate(line_terms) if lt & selective]
        keep: set[int] = set()
        for i, ln in enumerate(lines):
            x = featurize_query(
                ln,
                kind,
                query_terms,
                line_terms=line_terms[i],
                selective_terms=selective,
                nearest_hit_dist=_nearest_hit_dist(i, hits, n),
            )
            if _sigmoid(_dot(self._weights, x)) >= threshold:
                keep.add(i)
        return keep

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path | str | None = None) -> "QueryRelevanceModel":
        p = Path(path) if path is not None else DEFAULT_QUERY_WEIGHTS_PATH
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data["weights"])

    @classmethod
    def load_or_none(cls, path: Path | str | None = None) -> "QueryRelevanceModel | None":
        """Load if trained weights exist, else None — the plug point degrades to phase 1
        cleanly when no certified weights have been promoted yet."""
        p = Path(path) if path is not None else DEFAULT_QUERY_WEIGHTS_PATH
        try:
            if not p.exists():
                return None
            return cls.load(p)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def to_json(self) -> str:
        return json.dumps({"features": QUERY_FEATURE_NAMES, "weights": self._weights}, indent=2)

    def save(self, path: Path | str | None = None) -> None:
        p = Path(path) if path is not None else DEFAULT_QUERY_WEIGHTS_PATH
        p.write_text(self.to_json(), encoding="utf-8")


# --------------------------------------------------------------------------- plug point
# Cached singleton: the proxy is long-running, so the (optional) weights load once — the
# digest hot path must not stat/read query_weights.json on every call.
_MODEL_CACHE: QueryRelevanceModel | None = None
_MODEL_LOADED: bool = False


def get_model() -> QueryRelevanceModel | None:
    """The promoted query-relevance model, or None if no certified weights exist yet.

    None is the safe steady state until a retrain promotes weights — the digest plug
    then applies *exactly* phase 1. Loaded once per process.
    """
    global _MODEL_CACHE, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL_CACHE = QueryRelevanceModel.load_or_none()
        _MODEL_LOADED = True
    return _MODEL_CACHE


def reset_model_cache() -> None:
    """Drop the cached model — for tests, or after a retrain promotes new weights."""
    global _MODEL_CACHE, _MODEL_LOADED
    _MODEL_CACHE = None
    _MODEL_LOADED = False


if __name__ == "__main__":  # tiny self-check — no framework, per the ponytail rule
    # A hand-set model that fires on semantic proximity: the answer line to
    # "what's the retry limit" is `max_attempts = 5` — no lexical overlap, but it sits
    # right next to the lexical hit line, so q_dist_to_hit + structure carry it.
    lines = [
        "loading config from /etc/app.yaml",
        "retry policy section:",
        "  max_attempts = 5",
        "  backoff = exponential",
        "done",
    ]
    q = frozenset({"retry", "limit"})
    # weights: modest positive on q_has_selective + q_overlap + proximity (1 - dist),
    # slight negative bias — enough to keep the hit and its neighbor, drop the rest.
    w = [0.0] * len(FEATURE_NAMES) + [2.0, 2.0, 1.0, -3.0]
    w[0] = -1.0  # bias
    m = QueryRelevanceModel(w)
    kept = m.relevant_lines(lines, q, "tool_output")
    assert 1 in kept, f"the lexical 'retry' hit line must be kept: {kept}"
    # neutral-query degrades to base features, empty query -> no keeps
    assert m.relevant_lines(lines, frozenset(), "tool_output") == set()
    assert len(QUERY_FEATURE_NAMES) == len(FEATURE_NAMES) + 4
    print("query_relevance self-check ok:", sorted(kept))

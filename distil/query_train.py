"""Query-aware salience — phase 2 training + promotion gate.

Turns the dark-collected flywheel labels (`query_flywheel.load_labels`) into a promoted
`QueryRelevanceModel`, reusing the keep-model's logistic trainer and held-out evaluation
(`codec.learned.train` / `_evaluate`) unchanged — only the label source and feature width
differ.

**Why the promotion gate is what it is.** Phase 2 is *additive*: its kept lines are unioned
over phase 1, so it can only ever widen the keep set. Widening never drops a needed line, so
it cannot cause a decision-equivalence regression — the property `certify_promotion` protects
is satisfied *structurally*, not statistically. The residual risk is the opposite: keeping too
much (wasted compression) or learning nothing. So the gate is a held-out quality bar —

  1. enough real labels (no promotion on noise),
  2. held-out recall **beats the phase-1 lexical baseline** (it recovers semantically-needed
     lines a fixed lexical rule misses — the whole point),
  3. precision stays above a floor (it isn't just keeping everything),

— and only then are the weights written to ``query_weights.json``. A candidate that fails the
bar is not promoted; the plug then keeps behaving as exactly phase 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from distil.codec.features import FEATURE_NAMES
from distil.codec.learned import _evaluate, train
from distil.compress.query_relevance import (
    QUERY_FEATURE_NAMES,
    DEFAULT_QUERY_WEIGHTS_PATH,
    QueryRelevanceModel,
)

# q_has_selective — the phase-1 lexical signal, as a feature. A pure phase-1 keep fires on
# exactly this, so it's the baseline the learned model must beat on recall.
_SELECTIVE_IDX = len(FEATURE_NAMES) + 1


_Sample = tuple[list[float], float]


def _split(samples: list[_Sample], test_fraction: float) -> tuple[list[_Sample], list[_Sample]]:
    """Deterministic train/test split, hashing the feature vector (reproducible without a
    seed, same idea as online.retrain hashing the raw line)."""
    train_set: list[_Sample] = []
    test_set: list[_Sample] = []
    for x, y in samples:
        digest = hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFF_FFFF
        (test_set if bucket < test_fraction else train_set).append((x, y))
    return train_set, test_set


def _baseline_recall(samples: list[tuple[list[float], float]]) -> float:
    """Phase-1 lexical recall on *samples*: of the positives, the fraction a pure lexical
    keep (q_has_selective == 1) would already recover."""
    pos = [x for x, y in samples if y == 1.0]
    if not pos:
        return 0.0
    hit = sum(1 for x in pos if x[_SELECTIVE_IDX] >= 0.5)
    return hit / len(pos)


def train_query_relevance(
    samples: list[tuple[list[float], float]],
    *,
    test_fraction: float = 0.2,
    epochs: int = 400,
) -> dict:
    """Train a query-relevance model on flywheel labels; return weights + held-out metrics
    (including the phase-1 baseline recall for comparison)."""
    if not samples:
        raise ValueError("Cannot train query-relevance on empty label set.")
    if any(len(x) != len(QUERY_FEATURE_NAMES) for x, _ in samples):
        raise ValueError(f"All feature vectors must have width {len(QUERY_FEATURE_NAMES)}.")

    train_set, test_set = _split(samples, test_fraction)
    if not train_set:
        raise RuntimeError("Empty train split — grow the label set or lower test_fraction.")
    held_out = bool(test_set)
    eval_set = test_set if held_out else train_set

    weights = train(train_set, epochs=epochs, lr=0.5, l2=1e-4, seed=0)
    metrics = _evaluate(eval_set, weights)
    return {
        "weights": weights,
        "held_out": held_out,
        "n_train": len(train_set),
        "n_test": len(eval_set),
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "baseline_recall": _baseline_recall(eval_set),
    }


def certify_and_promote(
    samples: list[tuple[list[float], float]] | None = None,
    *,
    weights_path: Path | str | None = None,
    min_samples: int = 200,
    min_precision: float = 0.5,
    min_recall_gain: float = 0.0,
) -> dict:
    """Train, gate, and (only if it passes) promote query-relevance weights.

    Returns a report dict with ``promoted: bool`` and the metrics + reason. Never promotes
    on too few labels, on recall that doesn't beat the phase-1 baseline, or on sub-floor
    precision. On promotion, writes ``query_weights.json`` and clears the model cache so the
    next digest picks it up.
    """
    if samples is None:
        from distil.query_flywheel import load_labels

        samples = load_labels()

    report: dict = {"promoted": False, "n_labels": len(samples)}
    if len(samples) < min_samples:
        report["reason"] = f"too few labels ({len(samples)} < {min_samples}) — keep collecting"
        return report

    m = train_query_relevance(samples)
    report.update(m)
    del report["weights"]  # keep the report light; the weights live in the file on promotion

    gain = m["recall"] - m["baseline_recall"]
    if gain < min_recall_gain:
        report["reason"] = (
            f"recall {m['recall']:.3f} does not beat phase-1 baseline "
            f"{m['baseline_recall']:.3f} by >= {min_recall_gain} — not promoted"
        )
        return report
    if m["precision"] < min_precision:
        report["reason"] = f"precision {m['precision']:.3f} < floor {min_precision} — not promoted"
        return report

    path = Path(weights_path) if weights_path is not None else DEFAULT_QUERY_WEIGHTS_PATH
    QueryRelevanceModel(m["weights"]).save(path)
    from distil.compress import query_relevance

    query_relevance.reset_model_cache()
    report["promoted"] = True
    report["reason"] = f"promoted: recall {m['recall']:.3f} vs baseline {m['baseline_recall']:.3f}"
    report["weights_path"] = str(path)
    return report

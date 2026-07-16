"""learned — logistic-regression token-keep classifier.

A from-scratch, stdlib-only logistic regression that implements the ``KeepModel``
Protocol.  Weights are trained by full-batch gradient descent on binary cross-entropy
with L2 regularisation and persisted to ``distil/codec/weights.json``.

Training labels are distilled from ``SalienceKeepModel``: a line is labelled 1.0
(keep) if the heuristic scores it >= 0.6, else 0.0 (drop).  This lets the learned
model generalise the deterministic rules to unseen text via the feature representation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

from distil.codec.features import FEATURE_NAMES, featurize
from distil.codec.keep_model import SalienceKeepModel
from distil.trajectory import Stability

if TYPE_CHECKING:
    from distil.corpus import CorpusEntry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH: Path = _HERE / "weights.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOLATILE_KINDS: frozenset[str] = frozenset({"tool_output", "retrieved"})


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LogisticKeepModel:
    """Logistic-regression keep classifier implementing the ``KeepModel`` Protocol.

    Parameters
    ----------
    weights:
        Coefficient vector of length ``len(FEATURE_NAMES)``.  The model computes
        ``sigmoid(dot(weights, featurize(line, kind)))`` and returns it as the
        salience score.
    """

    def __init__(self, weights: list[float]) -> None:
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} weights, got {len(weights)}")
        self._weights: list[float] = list(weights)

    # ------------------------------------------------------------------
    # KeepModel Protocol
    # ------------------------------------------------------------------

    def score(self, line: str, kind: str) -> float:
        """Return salience score in [0, 1] via sigmoid(w · featurize(line, kind)).

        Blank lines short-circuit to 0.0 — the bias weight alone would push
        the logit slightly positive (≈ w_bias), but blank lines carry zero
        information and must always score below any keep threshold.
        """
        if not line.strip():
            return 0.0
        x = featurize(line, kind)
        return _sigmoid(_dot(self._weights, x))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "LogisticKeepModel":
        """Load weights from *path* (default: ``distil/codec/weights.json``)."""
        p = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data["weights"])

    def to_json(self) -> str:
        """Serialise weights to a JSON string."""
        return json.dumps(
            {"features": FEATURE_NAMES, "weights": self._weights},
            indent=2,
        )

    def save(self, path: Path | str | None = None) -> None:
        """Persist weights to *path* (default: ``distil/codec/weights.json``)."""
        p = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
        p.write_text(self.to_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    samples: list[tuple[list[float], float]],
    *,
    epochs: int = 400,
    lr: float = 0.5,
    l2: float = 1e-4,
    seed: int = 0,
) -> list[float]:
    """Full-batch gradient descent on binary cross-entropy with L2 regularisation.

    Parameters
    ----------
    samples:
        List of ``(feature_vector, label)`` pairs where label is 0.0 or 1.0.
    epochs:
        Number of full passes over the dataset.
    lr:
        Learning rate (step size).
    l2:
        L2 regularisation coefficient (applied to all weights except bias at index 0).
    seed:
        Seed for the ``random.Random`` used to shuffle samples each epoch.

    Returns
    -------
    list[float]
        Trained weight vector of length ``len(FEATURE_NAMES)``.
    """
    if not samples:
        raise ValueError("Cannot train on empty sample set.")

    # Infer the feature width from the data, not the base FEATURE_NAMES — so the same
    # trainer serves both the 9-feature keep-model and the 13-feature query-relevance
    # model (phase 2). Base samples are 9-wide, so this is identical for the keep-model.
    n_feat = len(samples[0][0])
    # Initialise weights to zero (deterministic).
    w: list[float] = [0.0] * n_feat
    rng = random.Random(seed)

    for _ in range(epochs):
        # Shuffle order for each epoch (full-batch but order matters for
        # reproducibility across different epoch counts).
        indices = list(range(len(samples)))
        rng.shuffle(indices)

        # Accumulate gradient over all samples.
        grad = [0.0] * n_feat
        for idx in indices:
            x, y = samples[idx]
            y_hat = _sigmoid(_dot(w, x))
            # Residual: (predicted - true) for cross-entropy gradient.
            residual = y_hat - y
            for j in range(n_feat):
                grad[j] += residual * x[j]

        # Apply gradient + L2 regularisation (skip bias at index 0).
        m = len(samples)
        for j in range(n_feat):
            reg = l2 * w[j] if j > 0 else 0.0
            w[j] -= lr * (grad[j] / m + reg)

    return w


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(
    entries: list["CorpusEntry"] | None = None,
) -> tuple[list[tuple[list[float], float]], list[str]]:
    """Build a labelled dataset from the corpus.

    Scans every VOLATILE ``tool_output`` or ``retrieved`` block and assigns a
    binary label via ``SalienceKeepModel``: 1.0 if the heuristic score >= 0.6,
    else 0.0.

    Parameters
    ----------
    entries:
        Corpus entries to use.  If ``None``, ``load_corpus()`` is called.

    Returns
    -------
    tuple[list[tuple[list[float], float]], list[str]]
        ``(samples, raw_lines)`` where *samples* is a list of
        ``(feature_vector, label)`` pairs and *raw_lines* are the original line
        strings (parallel to *samples*).
    """
    if entries is None:
        from distil.corpus import load_corpus

        entries = load_corpus()

    heuristic = SalienceKeepModel()
    samples: list[tuple[list[float], float]] = []
    raw_lines: list[str] = []

    for entry in entries:
        for turn in entry.trajectory.turns:
            for block in turn.blocks:
                if block.stability is not Stability.VOLATILE:
                    continue
                if block.kind.value not in _VOLATILE_KINDS:
                    continue
                for line in block.text.splitlines():
                    kind = block.kind.value
                    score = heuristic.score(line, kind)
                    label = 1.0 if score >= 0.6 else 0.0
                    features = featurize(line, kind)
                    samples.append((features, label))
                    raw_lines.append(line)

    return samples, raw_lines


# ---------------------------------------------------------------------------
# Train from corpus (entry point)
# ---------------------------------------------------------------------------


def _bce_loss(samples: list[tuple[list[float], float]], w: list[float]) -> float:
    """Compute mean binary cross-entropy loss."""
    eps = 1e-12
    total = 0.0
    for x, y in samples:
        p = _sigmoid(_dot(w, x))
        p = max(eps, min(1.0 - eps, p))
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(samples) if samples else 0.0


def _split(
    samples: list[tuple[list[float], float]],
    raw_lines: list[str],
    test_fraction: float,
) -> tuple[
    list[tuple[list[float], float]],
    list[tuple[list[float], float]],
]:
    """Deterministic train/test split based on SHA-256 hash of the raw line."""
    train_set: list[tuple[list[float], float]] = []
    test_set: list[tuple[list[float], float]] = []
    for (x, y), raw in zip(samples, raw_lines):
        digest = hashlib.sha256(raw.encode()).hexdigest()
        # Use the first 8 hex digits as a 32-bit integer and compare to threshold.
        bucket = int(digest[:8], 16) / 0xFFFF_FFFF
        if bucket < test_fraction:
            test_set.append((x, y))
        else:
            train_set.append((x, y))
    return train_set, test_set


def _evaluate(samples: list[tuple[list[float], float]], w: list[float]) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 at threshold 0.5."""
    tp = fp = fn = tn = 0
    for x, y in samples:
        pred = 1 if _sigmoid(_dot(w, x)) >= 0.5 else 0
        label = int(y)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def train_from_corpus(*, test_fraction: float = 0.2) -> dict[str, object]:
    """Build dataset, train, evaluate, persist weights, and return metrics.

    Steps
    -----
    1. Build labelled dataset from ``load_corpus()``.
    2. Deterministic train/test split (hash-based, reproducible).
    3. Train logistic regression on the train set.
    4. Evaluate accuracy, precision, recall, F1 on the test set.
    5. Persist weights to ``distil/codec/weights.json``.
    6. Return a metrics dict including split sizes and loss values.

    Returns
    -------
    dict
        Keys: ``train_size``, ``test_size``, ``initial_bce``, ``final_train_bce``,
        ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    samples, raw_lines = build_dataset()
    train_set, test_set = _split(samples, raw_lines, test_fraction)

    if not train_set:
        raise RuntimeError("Training set is empty — check corpus path.")
    if not test_set:
        raise RuntimeError("Test set is empty — increase corpus or lower test_fraction.")

    # Initial loss (weights = 0 → all predictions = 0.5).
    w0 = [0.0] * len(FEATURE_NAMES)
    initial_bce = _bce_loss(train_set, w0)

    # Train.
    w = train(train_set, epochs=400, lr=0.5, l2=1e-4, seed=0)
    final_train_bce = _bce_loss(train_set, w)

    # Evaluate on held-out test set.
    metrics = _evaluate(test_set, w)

    # Persist weights + the metrics that back the README's headline number, so the
    # claim is always regenerable from `python -m distil.codec.learned` (seed-pinned).
    model = LogisticKeepModel(w)
    model.save(DEFAULT_WEIGHTS_PATH)

    result: dict[str, object] = {
        "train_size": len(train_set),
        "test_size": len(test_set),
        "initial_bce": initial_bce,
        "final_train_bce": final_train_bce,
        **metrics,
    }
    DEFAULT_WEIGHTS_PATH.with_name("metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point (python -m distil.codec.learned)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Training logistic keep model from corpus …", file=sys.stderr)
    result = train_from_corpus()
    print(json.dumps(result, indent=2))

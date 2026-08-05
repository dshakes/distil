"""The eval record — the envelope that makes a metric a result.

Before this, `--json` emitted metrics and nothing else. That is enough to read a
number and not enough to *use* one. Two runs reporting `overclaim_rate: 0.057`
cannot be compared without knowing which compressor produced them, against which
corpus, graded by what, on which version of the code — and a number nobody can
reproduce or compare is a number nobody should act on.

The envelope carries five things the metrics cannot:

  * **schema version** — so a consumer can tell when the shape changed rather than
    silently mis-parsing it;
  * **dataset fingerprint** — a content hash of exactly what was graded. Adding a
    trajectory changes the numbers; without the fingerprint that looks like a
    regression in the compressor;
  * **subject identity** — which compressor and tier. "94.3% hedge fidelity" is
    meaningless without it;
  * **grader provenance** — following the norm already set by
    :func:`distil.conformal.render_grader`: a synthetic oracle must never be
    mistaken for a model, because that conflation is what makes a result look
    stronger than it is;
  * **gates as records** — threshold, observed value and outcome, not just a pass.
    A gate whose threshold is invisible cannot be audited, and a gate that only
    ever passes is indistinguishable from no gate.

Content-free by construction: fingerprints are hashes, metrics are counts. No
prompt, path or tool output enters a record.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SCHEMA = "distil.eval/1"


@dataclass(frozen=True)
class Gate:
    """One threshold, what was observed against it, and which way it went."""

    name: str
    threshold: float | int | None
    observed: float | int
    passed: bool
    # Why this threshold and not zero. A bound with no rationale is a number someone
    # will "fix" later without knowing what it was protecting.
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRecord:
    schema: str = SCHEMA
    run: dict[str, Any] = field(default_factory=dict)
    subject: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    grader: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: list[Gate] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """False if any recorded gate failed. An empty gate list is not a pass."""
        return bool(self.gates) and all(g.passed for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run": self.run,
            "subject": self.subject,
            "dataset": self.dataset,
            "grader": self.grader,
            "metrics": self.metrics,
            "gates": [g.to_dict() for g in self.gates],
            "passed": self.passed,
        }


def fingerprint(entries: Iterable[Any]) -> str:
    """Content hash of the graded corpus, stable across load order.

    Sorted before hashing so a manifest reshuffle does not look like a different
    dataset. This is what lets you tell "the compressor regressed" apart from "the
    corpus changed" — the failure mode that made a hardcoded corpus-size assertion
    fail earlier in this repo's history for a reason it was never written to detect.
    """
    h = hashlib.sha256()
    parts: list[tuple[str, int, int, str, str]] = []
    for entry in entries:
        traj = getattr(entry, "trajectory", None)
        if traj is None:
            continue
        for t_i, turn in enumerate(traj.turns):
            for b_i, block in enumerate(turn.blocks):
                # Turn and block POSITION are part of the identity. The artifact and
                # continuation probes fold in order — a create-then-delete and a
                # delete-then-create are different final states — so hashing an
                # order-blind bag of blocks let two corpora with genuinely different
                # metrics share a fingerprint, which is the one thing the field must
                # never do. Sorting by the key (not the text) keeps the hash
                # independent of manifest ORDERING while staying sensitive to the
                # order that actually changes the answer.
                parts.append((str(traj.id), t_i, b_i, str(block.id), block.text))
    for traj_id, t_i, b_i, block_id, text in sorted(parts, key=lambda p: p[:4]):
        h.update(f"{traj_id}\x1f{t_i}\x1f{b_i}\x1f{block_id}\x1f{text}".encode("utf-8", "replace"))
        h.update(b"\x1e")
    return f"sha256:{h.hexdigest()[:16]}"


def describe_grader(kind: str) -> dict[str, str]:
    """Provenance for whatever produced the decision signal.

    Mirrors :func:`distil.conformal.render_grader`. The deterministic oracle reads a
    `DECISION:` marker out of fixture text; it is not a model and must never be
    reported as one.
    """
    if kind == "deterministic":
        return {
            "kind": "deterministic",
            "detail": "synthetic DECISION: oracle — NOT a model",
        }
    if kind in ("", "unspecified"):
        return {"kind": "unspecified", "detail": "provenance not recorded"}
    return {"kind": kind, "detail": "graded by a live model"}


def describe_env() -> dict[str, str]:
    from distil import __version__

    return {
        "distil": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "implementation": sys.implementation.name,
    }


def build(
    *,
    metrics: dict[str, Any],
    entries: Iterable[Any],
    compressor: Any,
    grader: str = "deterministic",
    gates: list[Gate] | None = None,
    started: float | None = None,
) -> EvalRecord:
    """Assemble a complete, reproducible record around a set of metrics."""
    entries = list(entries)
    domains = sorted({getattr(e, "domain", "") for e in entries} - {""})
    now = time.time()
    return EvalRecord(
        run={
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "duration_ms": int((now - started) * 1000) if started else None,
            "env": describe_env(),
        },
        subject={
            "compressor": type(compressor).__name__,
            "module": type(compressor).__module__,
        },
        dataset={
            "name": "corpus",
            "trajectories": len(entries),
            "domains": domains,
            "fingerprint": fingerprint(entries),
        },
        grader=describe_grader(grader),
        metrics=metrics,
        gates=list(gates or []),
    )

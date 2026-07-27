"""The trajectory data model — a captured agent run.

A trajectory is an ordered list of turns; each turn is the full context the
model saw at that step, decomposed into blocks. Blocks carry a *stability*
hint (what can live in a cacheable prefix) and an optional ground-truth
`decision_relevant` label used by ablation/certification corpora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Stability(str, Enum):
    STABLE = "stable"  # system prompt, tool schemas, settled facts -> cacheable prefix
    SETTLING = "settling"  # older history; identical across turns once written -> cacheable
    VOLATILE = "volatile"  # the current user turn / fresh tool output -> never cached


class Kind(str, Enum):
    SYSTEM = "system"
    TOOLS = "tools"
    HISTORY = "history"
    TOOL_OUTPUT = "tool_output"
    USER = "user"
    RETRIEVED = "retrieved"


@dataclass
class Block:
    id: str
    kind: Kind
    text: str
    stability: Stability = Stability.VOLATILE
    decision_relevant: bool = False
    # Provider-shaped non-text content carried alongside `text` — today, vision
    # blocks (``{"type": "image", "source": {...}}``).
    #
    # This exists because the certificate could not otherwise reach images at all.
    # `text` is a `str`, so a trajectory could not hold one, and every runner
    # graded a decision from text alone. The tempting shortcut was to serialize
    # the base64 into `text` and certify that — which runs, goes green, and
    # measures whether deleting a blob from a TEXT prompt changes a TEXT
    # decision. That is not the claim. The claim is that a model which actually
    # SAW an image decides the same way when a later identical copy becomes a
    # reference, and grading it requires real image content in front of a real
    # vision model. Hence a separate field the live runner renders as real
    # content blocks (ADR 0003 / ADR 0004 rank 1).
    #
    # Optional and defaulted, so every existing strategy, runner and corpus file
    # is unaffected: a block without media behaves exactly as before.
    media: list[dict] | None = None

    def copy_with(self, text: str) -> "Block":
        """Replace the text, carrying media through unchanged.

        Compressors rewrite `text`; none of them may silently drop the media a
        block carries, or a compressed turn would be graded against a prompt the
        uncompressed one never had.
        """
        return Block(self.id, self.kind, text, self.stability, self.decision_relevant, self.media)

    def copy_without_media(self) -> "Block":
        """Same block with its media removed — the vision strategy's elision."""
        return Block(self.id, self.kind, self.text, self.stability, self.decision_relevant, None)


@dataclass
class Turn:
    index: int
    blocks: list[Block]


@dataclass
class Trajectory:
    id: str
    model: str
    turns: list[Turn] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        turns = [
            Turn(
                index=t.get("index", i),
                blocks=[
                    Block(
                        id=b["id"],
                        kind=Kind(b["kind"]),
                        text=b["text"],
                        stability=Stability(b.get("stability", "volatile")),
                        decision_relevant=bool(b.get("decision_relevant", False)),
                        media=b.get("media") or None,
                    )
                    for b in t["blocks"]
                ],
            )
            for i, t in enumerate(d["turns"])
        ]
        return Trajectory(
            id=d.get("id", "trajectory"), model=d.get("model", "claude-opus-4"), turns=turns
        )

    @staticmethod
    def load(path: str | Path) -> "Trajectory":
        return Trajectory.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "turns": [
                {
                    "index": t.index,
                    "blocks": [
                        {
                            "id": b.id,
                            "kind": b.kind.value,
                            "text": b.text,
                            "stability": b.stability.value,
                            "decision_relevant": b.decision_relevant,
                            # Omitted entirely when absent, so every existing
                            # corpus file round-trips byte-identically.
                            **({"media": b.media} if b.media else {}),
                        }
                        for b in t.blocks
                    ],
                }
                for t in self.turns
            ],
        }

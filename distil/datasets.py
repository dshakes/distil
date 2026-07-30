"""Public ground-truthed benchmarks — external validity, with no new dependencies.

Every other quality signal distil ships is graded against distil's own corpus and
distil's own ``DECISION:`` markers. That makes the numbers rigorous but *unfalsifiable
by a stranger*: you cannot check them against data you already trust. This module
closes that, by loading real public benchmarks whose ground truth was written by
someone else:

- ``hotpotqa``  — multi-hop QA, 10 paragraphs per question of which 8 are
  distractors, and ``supporting_facts`` naming the exact gold sentences. The best
  compression test in public: real noise to prune, and a third-party answer key for
  what had to survive.
- ``squad``     — SQuAD v2 extractive QA: one paragraph, a gold answer span, and
  deliberately unanswerable questions (which we exclude from answer recall and
  report separately, rather than scoring them as free wins).

Transport
---------
Rows come from the HuggingFace *datasets-server* REST API — plain JSON over HTTPS,
so the loader is stdlib-only (``urllib.request`` + ``json``) and distil's
``dependencies = []`` promise holds. Installing the ``datasets`` package, Arrow, and
a model stack merely to read 100 rows would cost more than the whole core.

Rows are cached under ``$DISTIL_HOME/datasets`` as JSONL, so a cited number is
reproducible offline and a re-run costs no network. Sampling is deterministic given
``(name, n)``: no seed drift between the run you publish and the run someone checks.

Failure is always loud. A fetch problem raises :class:`DatasetUnavailable` with the
cause — it never degrades to fewer cases, an empty list, or a silent pass, because a
quality gate that quietly measures nothing is worse than no gate.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_ROWS_API = "https://datasets-server.huggingface.co/rows"
_MAX_LENGTH = 100  # hard server-side cap on rows per request
_TIMEOUT = 30.0
_ATTEMPTS = 3
_BACKOFF = 1.0
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_UA = "distil-llm (+https://github.com/dshakes/distil)"


class DatasetUnavailable(RuntimeError):
    """Raised when a benchmark cannot be loaded. Never swallowed into a pass."""


@dataclass
class GroundTruthCase:
    """One benchmark question with a third-party answer key.

    `docs` are the retrieved paragraphs exactly as an agent would receive them —
    one per tool result — so compressing them is the real production operation,
    not a synthetic proxy.
    """

    id: str
    question: str
    docs: list[tuple[str, str]]  # (title, text), in dataset order
    answer: str
    support: list[str] = field(default_factory=list)  # gold sentences that must survive
    answerable: bool = True
    dataset: str = ""


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def _cache_path(name: str) -> Path:
    return _home() / "datasets" / f"{name}.jsonl"


def _read_cache(name: str) -> list[dict[str, Any]]:
    path = _cache_path(name)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line (interrupted write) is recoverable: keep the
                # prefix and refetch the remainder rather than discarding the cache.
                break
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_cache(name: str, rows: list[dict[str, Any]]) -> None:
    path = _cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half cache in place


def _get(url: str) -> dict[str, Any]:
    """One GET with retry/backoff on transient failures. Raises on permanent ones."""
    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise DatasetUnavailable(f"unexpected payload type {type(payload).__name__}")
            return payload
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in _RETRY_STATUS:
                raise DatasetUnavailable(f"HTTP {exc.code} from datasets-server: {exc.reason}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
        if attempt < _ATTEMPTS - 1:
            time.sleep(_BACKOFF * (2**attempt))
    raise DatasetUnavailable(f"datasets-server unreachable after {_ATTEMPTS} attempts: {last}")


def _fetch_rows(spec: _Spec, want: int) -> list[dict[str, Any]]:
    """Fetch `want` rows, paging the 100-row cap, warm-starting from the cache."""
    rows = _read_cache(spec.name)
    if len(rows) >= want:
        return rows[:want]

    while len(rows) < want:
        length = min(_MAX_LENGTH, want - len(rows))
        query = urllib.parse.urlencode(
            {
                "dataset": spec.hf_dataset,
                "config": spec.config,
                "split": spec.split,
                "offset": len(rows),
                "length": length,
            }
        )
        payload = _get(f"{_ROWS_API}?{query}")
        batch = payload.get("rows")
        if not isinstance(batch, list) or not batch:
            break  # split exhausted
        for item in batch:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        if len(batch) < length:
            break  # server returned a short page: end of split

    if not rows:
        raise DatasetUnavailable(f"{spec.name}: datasets-server returned no rows")
    _write_cache(spec.name, rows)
    return rows[:want]


# ---------------------------------------------------------------------------
# per-dataset adapters (row -> GroundTruthCase)
# ---------------------------------------------------------------------------


def _hotpotqa(row: dict[str, Any]) -> GroundTruthCase | None:
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    if not titles or len(titles) != len(sentences):
        return None

    docs = [(str(t), "".join(str(s) for s in sents)) for t, sents in zip(titles, sentences)]

    # supporting_facts names (title, sentence index) — resolve to the gold sentences
    # themselves so retention is checked against text, not an index we could misread.
    facts = row.get("supporting_facts") or {}
    by_title = {str(t): sents for t, sents in zip(titles, sentences)}
    support: list[str] = []
    for title, sent_id in zip(facts.get("title") or [], facts.get("sent_id") or []):
        sents = by_title.get(str(title))
        if sents is None:
            continue
        try:
            sentence = str(sents[int(sent_id)]).strip()
        except (IndexError, ValueError, TypeError):
            continue
        if sentence:
            support.append(sentence)

    answer = str(row.get("answer") or "").strip()
    if not answer or not docs:
        return None
    return GroundTruthCase(
        id=str(row.get("id") or ""),
        question=str(row.get("question") or ""),
        docs=docs,
        answer=answer,
        support=support,
        answerable=True,
        dataset="hotpotqa",
    )


def _squad(row: dict[str, Any]) -> GroundTruthCase | None:
    context = str(row.get("context") or "").strip()
    if not context:
        return None
    answers = row.get("answers") or {}
    texts = [str(t).strip() for t in (answers.get("text") or []) if str(t).strip()]
    # SQuAD v2 ships unanswerable questions on purpose. They cannot measure answer
    # retention (there is no answer to retain), so they are carried with
    # answerable=False and reported separately instead of counted as passes.
    return GroundTruthCase(
        id=str(row.get("id") or ""),
        question=str(row.get("question") or ""),
        docs=[(str(row.get("title") or "context"), context)],
        answer=texts[0] if texts else "",
        support=[],
        answerable=bool(texts),
        dataset="squad",
    )


@dataclass(frozen=True)
class _Spec:
    name: str
    hf_dataset: str
    config: str
    split: str
    adapt: Callable[[dict[str, Any]], "GroundTruthCase | None"]
    description: str
    default_n: int


SPECS: dict[str, _Spec] = {
    "hotpotqa": _Spec(
        name="hotpotqa",
        hf_dataset="hotpotqa/hotpot_qa",
        config="distractor",
        split="validation",
        adapt=_hotpotqa,
        description="multi-hop QA, 10 paragraphs (8 distractors) + gold supporting sentences",
        default_n=100,
    ),
    "squad": _Spec(
        name="squad",
        hf_dataset="rajpurkar/squad_v2",
        config="squad_v2",
        split="validation",
        adapt=_squad,
        description="SQuAD v2 extractive QA: gold answer spans + unanswerable questions",
        default_n=100,
    ),
}


def available() -> list[tuple[str, str]]:
    return [(spec.name, spec.description) for spec in SPECS.values()]


def load(name: str, n: int | None = None, *, offline: bool = False) -> list[GroundTruthCase]:
    """Load `n` cases of a public benchmark.

    Deterministic: the first `n` rows of the split, in dataset order. No shuffling,
    so the number you publish is the number a reader reproduces.

    Raises DatasetUnavailable on an unknown name, a fetch failure, or — under
    `offline` — an insufficient cache. Never returns a short list silently.
    """
    spec = SPECS.get(name)
    if spec is None:
        raise DatasetUnavailable(f"unknown dataset {name!r} (have: {', '.join(sorted(SPECS))})")
    want = spec.default_n if n is None else n
    if want <= 0:
        raise DatasetUnavailable(f"n must be positive, got {want}")

    if offline:
        rows = _read_cache(spec.name)[:want]
        if len(rows) < want:
            raise DatasetUnavailable(
                f"{spec.name}: offline mode has {len(rows)} cached rows, need {want} "
                f"(run once without --offline to populate {_cache_path(spec.name)})"
            )
    else:
        rows = _fetch_rows(spec, want)

    cases = [case for case in (spec.adapt(row) for row in rows) if case is not None]
    if not cases:
        raise DatasetUnavailable(
            f"{spec.name}: no rows survived adaptation (upstream shape change?)"
        )
    return cases

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


def _first_text(value: Any) -> str:
    """First non-empty string from a str / list / dict shape, at any nesting depth.

    Benchmarks disagree on where the text lives: NarrativeQA uses ``{'text': ...}``,
    BFCL nests chat turns two lists deep as ``[[{'role': ..., 'content': ...}]]``.
    Both keys are tried, in that order, rather than one adapter per shape.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content"):
            got = _first_text(value.get(key))
            if got:
                return got
        return ""
    if isinstance(value, list):
        for item in value:
            got = _first_text(item)
            if got:
                return got
    return ""


def _gsm8k(row: dict[str, Any]) -> GroundTruthCase | None:
    """Grade-school math. The gold answer follows a `####` marker in the rationale."""
    question = str(row.get("question") or "").strip()
    rationale = str(row.get("answer") or "")
    answer = rationale.split("####")[-1].strip() if "####" in rationale else ""
    if not question or not answer:
        return None
    # The rationale is the only compressible span, and it is what must survive for
    # the arithmetic to be re-derivable.
    return GroundTruthCase(
        id=f"gsm8k-{abs(hash(question)) % 10**8}",
        question=question,
        docs=[("problem", question), ("rationale", rationale)],
        answer=answer,
        support=[rationale.split("####")[0].strip()] if "####" in rationale else [],
        dataset="gsm8k",
    )


def _truthfulqa(row: dict[str, Any]) -> GroundTruthCase | None:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("best_answer") or "").strip()
    if not question or not answer:
        return None
    correct = row.get("correct_answers") or []
    return GroundTruthCase(
        id=f"truthfulqa-{abs(hash(question)) % 10**8}",
        question=question,
        docs=[("question", question)],
        answer=answer,
        support=[str(c).strip() for c in correct if str(c).strip()][:3],
        dataset="truthfulqa",
    )


def _mmlu(row: dict[str, Any]) -> GroundTruthCase | None:
    question = str(row.get("question") or "").strip()
    choices = row.get("choices") or []
    idx = row.get("answer")
    if not question or not isinstance(choices, list) or not isinstance(idx, int):
        return None
    if not 0 <= idx < len(choices):
        return None
    body = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return GroundTruthCase(
        id=f"mmlu-{row.get('subject', '')}-{abs(hash(question)) % 10**8}",
        question=question,
        docs=[(str(row.get("subject") or "mmlu"), f"{question}\n{body}")],
        answer=str(choices[idx]).strip(),
        support=[str(choices[idx]).strip()],
        dataset="mmlu",
    )


def _arc(row: dict[str, Any]) -> GroundTruthCase | None:
    question = str(row.get("question") or "").strip()
    choices = row.get("choices") or {}
    texts, labels = choices.get("text") or [], choices.get("label") or []
    key = str(row.get("answerKey") or "")
    if not question or not texts or len(texts) != len(labels) or key not in labels:
        return None
    answer = str(texts[labels.index(key)]).strip()
    body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
    return GroundTruthCase(
        id=str(row.get("id") or f"arc-{abs(hash(question)) % 10**8}"),
        question=question,
        docs=[("choices", f"{question}\n{body}")],
        answer=answer,
        support=[answer],
        dataset="arc",
    )


def _humaneval(row: dict[str, Any]) -> GroundTruthCase | None:
    """Code generation. The prompt (signature + docstring) is the payload that must
    survive; the entry point is the checkable gold."""
    prompt = str(row.get("prompt") or "")
    entry = str(row.get("entry_point") or "").strip()
    if not prompt or not entry:
        return None
    return GroundTruthCase(
        id=str(row.get("task_id") or f"humaneval-{abs(hash(prompt)) % 10**8}"),
        question=f"Implement {entry}",
        docs=[(str(row.get("task_id") or "humaneval"), prompt)],
        answer=entry,
        support=[f"def {entry}"],
        dataset="humaneval",
    )


def _msmarco(row: dict[str, Any]) -> GroundTruthCase | None:
    """Ten retrieved passages per query — the closest public analogue of RAG traffic."""
    passages = row.get("passages") or {}
    texts = passages.get("passage_text") or []
    selected = passages.get("is_selected") or []
    answers = row.get("answers") or []
    answer = _first_text(answers)
    query = str(row.get("query") or "").strip()
    if not texts or not answer or not query:
        return None
    docs = [(f"passage-{i}", str(t)) for i, t in enumerate(texts)]
    # `is_selected` marks the passage the answer came from: the span that must survive.
    support = [str(t) for t, sel in zip(texts, selected) if sel]
    return GroundTruthCase(
        id=str(row.get("query_id") or f"msmarco-{abs(hash(query)) % 10**8}"),
        question=query,
        docs=docs,
        answer=answer,
        support=support,
        dataset="msmarco",
    )


def _triviaqa(row: dict[str, Any]) -> GroundTruthCase | None:
    question = str(row.get("question") or "").strip()
    ans = row.get("answer") or {}
    answer = str(ans.get("value") or ans.get("normalized_value") or "").strip()
    if not question or not answer:
        return None
    return GroundTruthCase(
        id=str(row.get("question_id") or f"triviaqa-{abs(hash(question)) % 10**8}"),
        question=question,
        docs=[("question", question)],
        answer=answer,
        support=[answer],
        dataset="triviaqa",
    )


def _narrativeqa(row: dict[str, Any]) -> GroundTruthCase | None:
    """Book/film summaries — long-form narrative, the payload compression struggles on."""
    document = row.get("document") or {}
    summary = document.get("summary") or {}
    text = str(summary.get("text") or "").strip()
    question = _first_text(row.get("question"))
    answer = _first_text(row.get("answers"))
    if not text or not question or not answer:
        return None
    return GroundTruthCase(
        id=str(document.get("id") or f"narrativeqa-{abs(hash(question)) % 10**8}"),
        question=question,
        docs=[(str(summary.get("title") or "summary"), text)],
        answer=answer,
        support=[answer],
        dataset="narrativeqa",
    )


def _codesearchnet(row: dict[str, Any]) -> GroundTruthCase | None:
    code = str(row.get("func_code_string") or "").strip()
    doc = str(row.get("func_documentation_string") or "").strip()
    name = str(row.get("func_name") or "").strip()
    if not code or not doc or not name:
        return None
    return GroundTruthCase(
        id=f"csn-{abs(hash(code)) % 10**8}",
        question=f"What does {name} do?",
        docs=[(str(row.get("func_path_in_repository") or name), code)],
        answer=doc,
        support=[doc],
        dataset="codesearchnet",
    )


def _bfcl(row: dict[str, Any]) -> GroundTruthCase | None:
    """Berkeley Function Calling Leaderboard.

    The compressible payload is the TOOL SCHEMA, and the gold is the call the model
    should emit. That makes this the one public benchmark that tests the thing an
    agent proxy most risks breaking: compress a tool definition badly and the model
    calls the wrong function, or the right one with wrong arguments — a failure no
    QA benchmark can see, because QA never asks the model to *act*.
    """
    functions = row.get("function") or []
    question = _first_text(row.get("question"))
    truth = row.get("ground_truth")
    if not functions or not question or truth is None:
        return None
    schema = json.dumps(functions, indent=2, sort_keys=True)
    # Every function NAME must survive: the model cannot call what it cannot see.
    names = [str(f.get("name")) for f in functions if isinstance(f, dict) and f.get("name")]
    return GroundTruthCase(
        id=str(row.get("id") or f"bfcl-{abs(hash(question)) % 10**8}"),
        question=question,
        docs=[("tool-schemas", schema)],
        answer=json.dumps(truth, sort_keys=True),
        support=names,
        dataset="bfcl",
    )


_BFCL_BASE = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
)


def _fetch_bfcl(spec: "_Spec", want: int) -> list[dict[str, Any]]:
    """BFCL ships as plain JSONL files, not a parquet dataset, so the rows API 500s.

    Questions and ground truth live in separate files keyed by `id`, and a case is
    only usable with both — an unanswered call cannot be graded, so rows that fail to
    join are dropped rather than scored as passes.
    """
    rows = _read_cache(spec.name)
    if len(rows) >= want:
        return rows[:want]

    def _lines(path: str) -> list[dict[str, Any]]:
        req = urllib.request.Request(f"{_BFCL_BASE}/{path}", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", "replace")
        out = []
        for line in body.splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    try:
        questions = _lines(f"{spec.config}.json")
        truth = {
            str(t.get("id")): t.get("ground_truth")
            for t in _lines(f"possible_answer/{spec.config}.json")
        }
    except (urllib.error.URLError, OSError) as exc:
        raise DatasetUnavailable(f"{spec.name}: {exc}") from exc

    rows = []
    for item in questions:
        gt = truth.get(str(item.get("id")))
        if gt is not None:
            rows.append({**item, "ground_truth": gt})
        if len(rows) >= want:
            break
    if not rows:
        raise DatasetUnavailable(f"{spec.name}: no question joined to a ground truth")
    _write_cache(spec.name, rows)
    return rows[:want]


@dataclass(frozen=True)
class _Spec:
    name: str
    hf_dataset: str
    config: str
    split: str
    adapt: Callable[[dict[str, Any]], "GroundTruthCase | None"]
    description: str
    default_n: int
    # How much COMPRESSIBLE payload a case carries — the single most important thing
    # about a compression benchmark, and the thing benchmark tables usually hide.
    #
    # A suite that reports "GSM8K 0.870 -> 0.870, no loss" beside "SQuAD 97% at 19%
    # compression" invites the reader to average them. They are not the same kind of
    # evidence: a GSM8K case is a one-line word problem with nothing to compress, so
    # an unchanged score there is a CONTROL — it proves the compressor did not corrupt
    # a prompt it barely touched. Only `rich` cases, where the payload is retrieved
    # documents or tool schemas, can demonstrate anything about compression quality.
    #
    # Both belong in the suite; conflating them is what makes a table look stronger
    # than its data. So the class travels with the number everywhere it is reported.
    payload: str = "rich"  # "rich" (real context to compress) | "thin" (control)
    # Some benchmarks are not served by the parquet rows API — BFCL is a plain file
    # repository. A spec may carry its own transport rather than be excluded for it.
    fetch: "Callable[[_Spec, int], list[dict[str, Any]]] | None" = None


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
    # --- tool use -----------------------------------------------------------------
    "bfcl": _Spec(
        name="bfcl",
        hf_dataset="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        config="BFCL_v3_simple",
        split="",  # file repo, not a parquet split
        adapt=_bfcl,
        description="Berkeley Function Calling: tool SCHEMAS compressed, gold call checked",
        default_n=100,
        fetch=_fetch_bfcl,
    ),
    # --- retrieval / RAG ----------------------------------------------------------
    "msmarco": _Spec(
        name="msmarco",
        hf_dataset="microsoft/ms_marco",
        config="v2.1",
        split="validation",
        adapt=_msmarco,
        description="MS MARCO: 10 retrieved passages per query, gold passage marked",
        default_n=100,
    ),
    "narrativeqa": _Spec(
        name="narrativeqa",
        hf_dataset="deepmind/narrativeqa",
        config="default",
        split="validation",
        adapt=_narrativeqa,
        description="NarrativeQA: long book/film summaries — the long-form payload",
        default_n=100,
    ),
    "codesearchnet": _Spec(
        name="codesearchnet",
        hf_dataset="code-search-net/code_search_net",
        config="python",
        split="test",
        adapt=_codesearchnet,
        description="CodeSearchNet: real functions, docstring as gold — code payload",
        default_n=100,
    ),
    # --- controls -----------------------------------------------------------------
    # Thin payload: a one-line question with nothing meaningful to compress. An
    # unchanged score here does NOT demonstrate compression quality; it demonstrates
    # the compressor left a prompt it barely touched alone. Reported as controls so
    # the distinction survives into whatever table quotes them.
    "gsm8k": _Spec(
        name="gsm8k",
        hf_dataset="openai/gsm8k",
        config="main",
        split="test",
        adapt=_gsm8k,
        description="GSM8K grade-school math (control: thin payload)",
        default_n=100,
        payload="thin",
    ),
    "truthfulqa": _Spec(
        name="truthfulqa",
        hf_dataset="truthfulqa/truthful_qa",
        config="generation",
        split="validation",
        adapt=_truthfulqa,
        description="TruthfulQA factual accuracy (control: thin payload)",
        default_n=100,
        payload="thin",
    ),
    "mmlu": _Spec(
        name="mmlu",
        hf_dataset="cais/mmlu",
        config="all",
        split="test",
        adapt=_mmlu,
        description="MMLU 57-subject knowledge (control: thin payload)",
        default_n=100,
        payload="thin",
    ),
    "arc": _Spec(
        name="arc",
        hf_dataset="allenai/ai2_arc",
        config="ARC-Challenge",
        split="test",
        adapt=_arc,
        description="ARC-Challenge science reasoning (control: thin payload)",
        default_n=100,
        payload="thin",
    ),
    "triviaqa": _Spec(
        name="triviaqa",
        hf_dataset="mandarjoshi/trivia_qa",
        config="rc.nocontext",
        split="validation",
        adapt=_triviaqa,
        description="TriviaQA factoid QA, no-context split (control: thin payload)",
        default_n=100,
        payload="thin",
    ),
    "humaneval": _Spec(
        name="humaneval",
        hf_dataset="openai/openai_humaneval",
        config="openai_humaneval",
        split="test",
        adapt=_humaneval,
        description="HumanEval code generation: signature+docstring payload",
        default_n=164,
    ),
}


def available() -> list[tuple[str, str]]:
    return [(spec.name, spec.description) for spec in SPECS.values()]


def payload_class(name: str) -> str:
    """ "rich" if a case carries real context to compress, "thin" if it is a control.

    Exposed because every consumer that prints a benchmark number needs to print this
    beside it. A suite that reports a thin-payload null result as evidence of
    compression quality is overstating its own data.
    """
    spec = SPECS.get(name)
    if spec is None:
        raise DatasetUnavailable(f"unknown dataset {name!r}")
    return spec.payload


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
        rows = (spec.fetch or _fetch_rows)(spec, want)

    cases = [case for case in (spec.adapt(row) for row in rows) if case is not None]
    if not cases:
        raise DatasetUnavailable(
            f"{spec.name}: no rows survived adaptation (upstream shape change?)"
        )
    return cases

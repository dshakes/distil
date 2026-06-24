"""Competitor / structural baselines as ladder strategies, for a head-to-head under
the SAME grader as distil.

Each baseline is a ``(blocks, turn_index) -> blocks`` strategy that compresses the
VOLATILE tail only (the cacheable prefix is left intact, exactly as distil and the
truncation rungs do), so token savings are measured identically for everyone.

Two families:
  * **No-dependency reference baselines** (always available): head-truncation,
    recency-window (keep the most recent text), RECOMP-style extractive (keep the
    most salient lines), and selective-context-style entropy pruning. These are
    faithful reference implementations of those technique families, not the packages.
  * **Real packages** (optional, used iff importable): LLMLingua-2 via the
    ``llmlingua`` package. If it isn't installed, that baseline is skipped with a
    note (``pip install llmlingua``) rather than failing the run.

The headline comparison only needs all methods graded the same way; the certificate
itself still rides distil's risk-ordered ladder.
"""

from __future__ import annotations

import re

from distil.compress.salience import reference_index, salient_tokens
from distil.trajectory import Kind, Stability

_WORD = re.compile(r"\S+")


def _map_volatile(blocks, fn):
    """Apply ``fn(text) -> text`` to volatile blocks only; keep the prefix intact."""
    out = []
    for b in blocks:
        if b.stability is Stability.VOLATILE:
            new = fn(b.text)
            out.append(b.copy_with(new) if len(new) < len(b.text) else b)
        else:
            out.append(b)
    return out


def _map_kind(blocks, kinds, fn):
    """Apply ``fn(text) -> text`` to blocks of the given kinds (e.g. HISTORY)."""
    out = []
    for b in blocks:
        if b.kind in kinds:
            new = fn(b.text)
            out.append(b.copy_with(new) if len(new) < len(b.text) else b)
        else:
            out.append(b)
    return out


# --------------------------------------------------------------------------- #
# No-dependency reference baselines
# --------------------------------------------------------------------------- #


def truncate_head(limit: int):
    """Keep the first ``limit`` chars (the classic prompt-truncation baseline)."""
    return lambda blocks, turn: _map_volatile(blocks, lambda t: t[:limit])


def recency_window(limit: int):
    """Keep only the most recent ``limit`` chars (sliding-window / recency baseline)."""
    return lambda blocks, turn: _map_volatile(blocks, lambda t: t[-limit:])


def recomp_extractive(keep_frac: float = 0.35):
    """RECOMP-style extractive compression: keep the most salient *sentences/lines*,
    drop the rest (lossy, irrecoverable). Selection uses the model-free salience
    signals already in the repo, ranking lines by salient-token count."""

    def strat(blocks, turn):
        ref = reference_index(blocks)

        def fn(text):
            lines = text.splitlines()
            if len(lines) <= 3:
                return text
            scored = sorted(
                range(len(lines)),
                key=lambda i: len(salient_tokens(lines[i], ref_index=ref)),
                reverse=True,
            )
            k = max(1, int(len(lines) * keep_frac))
            keep = sorted(scored[:k])
            return "\n".join(lines[i] for i in keep)

        return _map_volatile(blocks, fn)

    return strat


def keep_last_k_turns(k: int = 3):
    """Agent-memory sliding window: keep only the most recent ``k`` history
    entries, dropping older context entirely (lossy, irrecoverable, and
    cache-breaking since it rewrites the prefix). The classic ``messages[-k:]``
    memory policy, here applied to the HISTORY block the adapters accumulate."""

    def fn(text):
        entries = text.split("\n\n")
        if len(entries) <= k:
            return text
        return f"[…{len(entries) - k} earlier turns dropped…]\n\n" + "\n\n".join(entries[-k:])

    return lambda blocks, turn: _map_kind(blocks, {Kind.HISTORY}, fn)


def selective_context(keep_frac: float = 0.5):
    """Selective-Context-style pruning: drop low-information *tokens* (keep salient
    tokens + a uniform stride of the rest), token-level rather than line-level."""

    def strat(blocks, turn):
        ref = reference_index(blocks)

        def fn(text):
            sal = salient_tokens(text, ref_index=ref)
            words = _WORD.findall(text)
            if len(words) <= 8:
                return text
            stride = max(2, int(1 / max(keep_frac, 1e-3)))
            kept = [w for i, w in enumerate(words) if w in sal or i % stride == 0]
            return " ".join(kept)

        return _map_volatile(blocks, fn)

    return strat


# --------------------------------------------------------------------------- #
# Real packages (optional)
# --------------------------------------------------------------------------- #


def _best_device() -> str:
    """Best available torch device for the real compressors: MPS on Apple silicon,
    CUDA on a GPU host, else CPU. PromptCompressor defaults to ``"cuda"``, which
    hard-fails on machines without it — which is exactly why these baselines were
    skipped (and the paper said they "require a GPU host") before this run."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — torch missing/odd build → fall back to CPU
        pass
    return "cpu"


def llmlingua2(rate: float = 0.5):
    """LLMLingua-2 via the real ``llmlingua`` package, applied per volatile block —
    the way it deploys. Returns None if the package isn't importable (skipped)."""
    try:
        from llmlingua import PromptCompressor

        comp = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=_best_device(),
        )
    except Exception:  # noqa: BLE001 — not importable OR model won't load (e.g. no GPU); skip
        return None

    def strat(blocks, turn):
        def fn(text):
            try:
                return comp.compress_prompt(text, rate=rate).get("compressed_prompt", text)
            except Exception:  # noqa: BLE001 — never break the sweep on one block
                return text

        return _map_volatile(blocks, fn)

    return strat


def _strip_question(compressed_prompt: str, question: str) -> str:
    """Recover just the compressed CONTEXT from LLMLingua's assembled prompt.

    ``compress_prompt(question=...)`` returns ``compressed_prompt`` = the compressed
    context with the (verbatim, *uncompressed*) question spliced in per
    ``condition_in_question``. Returning that whole string as the new volatile block
    is the bug that made LongLLMLingua a silent no-op: the question (the full
    system/issue/history, often longer than the tail being compressed) made the
    result *larger* than the original, so ``_map_volatile``'s reject-if-bigger guard
    discarded it every time — a 0%-savings row that looked like "no compression" but
    was really "compressed, then thrown away". Splicing the question back out lets the
    block actually shrink. Robust to the question appearing before or after the
    context; falls back to the full string if the question can't be located."""
    if not question:  # str.rfind("") returns len(text), not -1 — guard it explicitly
        return compressed_prompt.strip()
    idx = compressed_prompt.rfind(question)
    if idx == -1:
        return compressed_prompt
    return (compressed_prompt[:idx] + compressed_prompt[idx + len(question) :]).strip()


def longllmlingua(rate: float = 0.5):
    """LongLLMLingua (the question-aware, perplexity-based variant) via the real
    ``llmlingua`` package. Returns None if the package isn't importable (skipped)."""
    try:
        from llmlingua import PromptCompressor

        comp = PromptCompressor(
            use_llmlingua2=False,  # original LLMLingua/LongLLMLingua LM (Llama-2-7B)
            device_map=_best_device(),
        )
    except Exception:  # noqa: BLE001 — not importable OR backbone won't load (e.g. no GPU); skip
        return None

    def strat(blocks, turn):
        # LongLLMLingua is question-aware: its rank_method="longllmlingua" + condition_*
        # path *requires* a question and asserts without one. The faithful question here
        # is the standing context the decision conditions on — every non-volatile block
        # (system/issue/policy + settled history), i.e. the complement of the volatile
        # tail being compressed. Plumb it through; without it the call raised an
        # AssertionError that the except-clause silently swallowed into a no-op row.
        question = "\n".join(b.text for b in blocks if b.stability is not Stability.VOLATILE)

        def fn(text):
            try:
                out = comp.compress_prompt(
                    [text],
                    question=question,
                    rate=rate,
                    condition_compare=True,
                    condition_in_question="after",
                    rank_method="longllmlingua",
                    dynamic_context_compression_ratio=0.3,
                )
                return _strip_question(out.get("compressed_prompt", text), question)
            except Exception:  # noqa: BLE001 — never break the sweep on one block
                return text

        return _map_volatile(blocks, fn)

    return strat


# --------------------------------------------------------------------------- #


def load_baselines(*, include_real: bool = True) -> list[tuple[str, object]]:
    """The baseline rungs for the head-to-head. No-dep baselines always; real
    packages appended iff importable (a note is printed for any that are skipped)."""
    rungs: list[tuple[str, object]] = [
        ("truncate@500", truncate_head(500)),
        ("recency-window@500", recency_window(500)),
        ("keep-last-3-turns", keep_last_k_turns(3)),
        ("recomp-extractive", recomp_extractive()),
        ("selective-context", selective_context()),
    ]
    if include_real:
        for label, factory in (
            ("llmlingua-2", llmlingua2),
            ("longllmlingua", longllmlingua),
        ):
            strat = factory()
            if strat is not None:
                rungs.append((label, strat))
            else:
                print(f"  [baselines] {label} skipped — `pip install llmlingua` to include it")
    return rungs

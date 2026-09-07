"""Benchmark adapter for LLMLingua-2 — run it head-to-head against Distil.

    pip install llmlingua
    PYTHONPATH=. distil benchmark --external benchmarks.llmlingua_adapter:compress:LLMLingua-2

The ``distil benchmark --external`` seam passes the per-turn block texts
(``list[str]``) and expects the compressed texts back 1:1 (``list[str]``).

LLMLingua-2 is naturally per-text (``PromptCompressor.compress_prompt``), so
this adapter is a thin, direct wrapper — see ``benchmarks/baselines.py``'s
``llmlingua2()`` for the same package used inside the in-repo baseline sweep.
The compressor is loaded once at import time and reused across calls, the way
a real deployment would amortize the model-load cost.
"""

from __future__ import annotations

_RATE = 0.5


def _best_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — torch missing/odd build → fall back to CPU
        pass
    return "cpu"


_comp = None


def _compressor():
    global _comp
    if _comp is None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as e:
            raise ImportError("llmlingua is not installed — run: pip install llmlingua") from e
        _comp = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=_best_device(),
        )
    return _comp


def compress(texts: list[str]) -> list[str]:
    comp = _compressor()
    out = []
    for t in texts:
        try:
            c = comp.compress_prompt(t, rate=_RATE).get("compressed_prompt", t)
        except Exception:  # noqa: BLE001 — never break the sweep on one block
            c = t
        # reject-if-bigger — the same invariant Distil applies to every block.
        out.append(c if len(c) < len(t) else t)
    return out

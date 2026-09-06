"""Benchmark adapter for headroom-ai — run it head-to-head against Distil.

    pip install headroom-ai
    PYTHONPATH=. distil benchmark --external benchmarks.headroom_adapter:compress:Headroom

The ``distil benchmark --external`` seam passes the per-turn block texts
(``list[str]``) and expects the compressed texts back 1:1 (``list[str]``).

Fairness notes (these matter — an unfair shape silently neuters Headroom):

* Headroom's router *protects plain user/system messages* and only compresses
  tool outputs. Presenting blocks as ``{"role": "user", "content": "<text>"}``
  therefore yields 0% — a wrong, unfair result. We instead present every block
  as a **tool_result** (Headroom's actual compression target), which unlocks its
  SmartCrusher / CodeCompressor / text pipeline.
* We pass a low ``model_limit`` so Headroom is maximally willing to compress —
  generous to it, not a strawman.
* Headroom is then judged by the SAME decision-equivalence + non-inferiority
  gate and the SAME cache-aware cost model as every other technique. If a
  Headroom transform changes a decision, the gate disqualifies it — exactly as
  it does for Distil's own aggressive modes. Give the competitor its best shot;
  let the gate be the judge.
"""

from __future__ import annotations


def _contents(result: object, n: int) -> list[str] | None:
    """Extract the n compressed block texts from headroom's CompressResult, 1:1."""
    msgs = getattr(result, "messages", result)
    if not (isinstance(msgs, list) and len(msgs) == n):
        return None
    out: list[str] = []
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):  # tool_result / block content → concat text fields
            out.append(
                "".join(
                    (b.get("content") or b.get("text") or "") if isinstance(b, dict) else str(b)
                    for b in c
                )
            )
        else:
            return None
    return out


def compress(texts: list[str]) -> list[str]:
    try:
        from headroom import compress as _hr
    except ImportError as e:
        raise ImportError("headroom-ai is not installed — run: pip install headroom-ai") from e

    msgs = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"b{i}", "content": t}]}
        for i, t in enumerate(texts)
    ]
    result = _hr(msgs, model="claude-sonnet-4-5", model_limit=1000)
    out = _contents(result, len(texts))
    if out is None:
        raise RuntimeError(
            "headroom.compress returned a shape this adapter can't map 1:1 "
            f"({type(result)!r}). Update _contents() to your installed headroom-ai version."
        )
    # reject-if-bigger — the same invariant Distil applies to every block.
    return [c if len(c) < len(t) else t for c, t in zip(out, texts)]

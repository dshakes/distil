"""Headroom offline worker for the policy simulator (JSONL over stdin/stdout).

Run under the hr-venv python. headroom-ai 0.40.0 (``importlib.metadata`` checked at start).

API used: ``headroom.compress(messages, model, frozen_message_count=N)`` (the library entry
point over the same transform pipeline the proxy runs) plus
``headroom.cache.prefix_tracker.extract_cache_stable_delta`` -- the proxy's default
"cache mode" step: when the new request append-only-extends the previous one, replay the
previously FORWARDED prefix byte-for-byte and compress only the appended delta; on a cold
start compress everything; on a non-append change forward unmodified (as the proxy does).

stdin  : {"conv": str, "model": str, "system": ..., "tools": [...], "messages": [...]}
stdout : {"messages": [...], "system": ..., "meta": {...}}   (one line per input line)
         or {"error": str} then exit 2 -- never a silent pass-through.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from importlib.metadata import version
from typing import Any

os.environ.setdefault("HEADROOM_TELEMETRY", "off")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# no tiktoken vocab download: headroom falls back to its char estimator (token counts are estimates)
os.environ.setdefault("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.1")

from headroom import compress
from headroom.cache.prefix_tracker import extract_cache_stable_delta

Msgs = list[dict[str, Any]]
# conv -> (previous original messages, previous forwarded messages)
_STATE: dict[str, tuple[Msgs, Msgs]] = {}


def _compress(msgs: Msgs, model: str, frozen: int) -> Any:
    r = compress(msgs, model=model, frozen_message_count=frozen)
    # compress() swallows exceptions and returns tokens_before == 0; refuse that.
    if msgs and r.tokens_before == 0:
        raise RuntimeError("headroom.compress failed internally (tokens_before == 0)")
    return r


def step(req: dict[str, Any]) -> dict[str, Any]:
    conv, model = req["conv"], req["model"]
    msgs: Msgs = req["messages"]
    prev = _STATE.get(conv)
    delta = extract_cache_stable_delta(msgs, prev[0], prev[1]) if prev else None
    if delta is not None:
        prefix, tail = delta
        r = _compress(prefix + tail, model, len(prefix))
        out, mode = prefix + r.messages[len(prefix) :], "cache_delta"
    elif prev:
        r, out, mode = None, msgs, "passthrough_prefix_mismatch"
    else:
        r = _compress(msgs, model, 0)
        out, mode = r.messages, "cold_start_full"
    _STATE[conv] = (copy.deepcopy(msgs), copy.deepcopy(out))
    meta: dict[str, Any] = {"mode": mode, "headroom": version("headroom-ai")}
    if r is not None:
        meta |= {
            "tokens_before": r.tokens_before,
            "tokens_after": r.tokens_after,
            "transforms": r.transforms_applied,
        }
    return {"messages": out, "system": req.get("system"), "meta": meta}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            resp = step(json.loads(line))
        except Exception as e:  # noqa: BLE001 - report then die loudly
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
            sys.exit(2)
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()

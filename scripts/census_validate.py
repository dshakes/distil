#!/usr/bin/env python3
"""Server-side census validation — the CI twin of the worker's validate.js.

Defense in depth: the ingest workflow re-validates every repository_dispatch
payload before it touches the metrics branch, so a compromised or bypassed
worker still cannot append garbage. Same frozen schema as distil/census.py,
same hard ceilings as lib/validate.js.

Usage: echo '<census json>' | python3 scripts/census_validate.py
Exits 0 and re-emits canonical JSON on stdout, exits 1 with the reason on
stderr otherwise.
"""

from __future__ import annotations

import json
import re
import sys

KEYS_V1 = {
    "schema",
    "install_id",
    "version",
    "os",
    "arch",
    "python",
    "runs",
    "tokens_saved",
    "dollars_saved",
    "ts",
}
KEYS_V2 = KEYS_V1 | {"billing", "by_model", "agents"}
KEYS_V3 = KEYS_V2 | {"surfaces", "shapes"}
KEYS_V4 = KEYS_V3 | {"equivalence", "modes"}
MODES = {"interactive", "headless", "sdk"}
NUM_CAPS = {"runs": 1e9, "tokens_saved": 1e13, "dollars_saved": 1e8, "ts": 4102444800}
AGENTS = {"claude", "codex", "gemini", "aider", "other"}
SURFACES = {"wrap", "proxy", "gateway"}
SHAPES = {"anthropic", "openai-chat", "openai-responses", "gemini"}
MAX_MODELS = 8


def validate(p: object) -> str | None:
    """Return None if valid, else the reason. Accepts schemas 1-3 (older
    clients stay in the wild; 2 added billing/by_model/agents, 3 added
    surfaces/shapes — integration attribution)."""
    if not isinstance(p, dict):
        return "not an object"
    schema = p.get("schema")
    if schema not in (1, 2, 3, 4):
        return "unknown schema version"
    if set(p) != {1: KEYS_V1, 2: KEYS_V2, 3: KEYS_V3, 4: KEYS_V4}[schema]:
        return "schema keys mismatch"
    if not re.fullmatch(r"[0-9a-f]{32}", str(p["install_id"])):
        return "install_id must be 32 hex chars"
    for k in ("version", "os", "arch", "python"):
        v = p[k]
        if not isinstance(v, str) or not 0 < len(v) <= 64:
            return f"bad {k}"
        if k != "version" and ("/" in v or "\\" in v):
            return f"bad {k}"
    for k, cap in NUM_CAPS.items():
        v = p[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= cap:
            return f"bad {k}"
    if schema >= 2:
        if p["billing"] not in ("subscription", "metered"):
            return "bad billing"
        bm = p["by_model"]
        if not isinstance(bm, dict) or len(bm) > MAX_MODELS:
            return "bad by_model"
        for m, v in bm.items():
            if not isinstance(m, str) or not 0 < len(m) <= 64 or "/" in m or "\\" in m:
                return "bad by_model key"
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1e13:
                return "bad by_model value"
        ag = p["agents"]
        if not isinstance(ag, list) or len(ag) > 6 or any(a not in AGENTS for a in ag):
            return "bad agents"
    if schema >= 3:
        for field, allow in (("surfaces", SURFACES), ("shapes", SHAPES)):
            d = p[field]
            if not isinstance(d, dict) or set(d) - allow:
                return f"bad {field}"
            for v in d.values():
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1e9:
                    return f"bad {field} value"
    if schema >= 4:
        eq = p["equivalence"]
        if not isinstance(eq, dict) or set(eq) != {"pct", "shadowed"}:
            return "bad equivalence"
        pct = eq["pct"]
        if pct is not None and (
            isinstance(pct, bool) or not isinstance(pct, (int, float)) or not 0 <= pct <= 100
        ):
            return "bad equivalence pct"
        sh = eq["shadowed"]
        if isinstance(sh, bool) or not isinstance(sh, int) or not 0 <= sh <= 1e9:
            return "bad equivalence shadowed"
        modes = p["modes"]
        if not isinstance(modes, dict) or set(modes) - MODES:
            return "bad modes"
        for v in modes.values():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1e9:
                return "bad modes value"
    return None


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"invalid json: {exc}", file=sys.stderr)
        return 1
    reason = validate(payload)
    if reason:
        print(f"rejected: {reason}", file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

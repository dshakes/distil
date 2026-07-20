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
NUM_CAPS = {"runs": 1e9, "tokens_saved": 1e13, "dollars_saved": 1e8, "ts": 4102444800}
AGENTS = {"claude", "codex", "gemini", "aider", "other"}
MAX_MODELS = 8


def validate(p: object) -> str | None:
    """Return None if valid, else the reason. Accepts schema 1 and 2 (schema 1
    clients are already in the wild; schema 2 added billing/by_model/agents)."""
    if not isinstance(p, dict):
        return "not an object"
    schema = p.get("schema")
    if schema not in (1, 2):
        return "unknown schema version"
    if set(p) != (KEYS_V1 if schema == 1 else KEYS_V2):
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
    if schema == 2:
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

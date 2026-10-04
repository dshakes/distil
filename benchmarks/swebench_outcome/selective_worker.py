"""Out-of-process Selective Context worker: `python selective_worker.py`.

`selective-context==0.1.4` hard-pins spacy==3.2.0 (wheels only up to CPython 3.10) and
click==8.0.4, so it cannot share an interpreter with the harness. The harness runs this file with
a Python that has it installed (`--selective-python`) and talks JSON lines over stdin/stdout.

  <- {"ready": true, "versions": {...}}      once, after the models load
  -> {"text": str, "ratio": float, "level": "sent"|"phrase"|"token"}
  <- {"context": str} | {"error": str}

Stdlib + selective_context only; must stay importable on Python 3.9.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from typing import Any


def main() -> int:
    out = sys.stdout
    sys.stdout = sys.stderr  # selective_context print()s while loading; keep the channel clean

    def emit(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    try:
        from selective_context import SelectiveContext

        sc = SelectiveContext(model_type="gpt2", lang="en")
        versions: dict[str, str | None] = {}
        for pkg in ("selective-context", "spacy", "torch", "transformers"):
            try:
                versions[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                versions[pkg] = None
    except Exception as e:  # noqa: BLE001 - reported to the harness verbatim
        emit({"error": f"{type(e).__name__}: {e}"})
        return 1
    emit({"ready": True, "versions": versions})
    for line in sys.stdin:
        req = json.loads(line)
        try:
            context, _ = sc(req["text"], reduce_ratio=req["ratio"], reduce_level=req["level"])
            emit({"context": context})
        except Exception as e:  # noqa: BLE001 - one bad chunk must not kill the worker
            emit({"error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

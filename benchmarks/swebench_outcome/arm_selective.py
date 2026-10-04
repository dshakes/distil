"""`selective` arm: Selective Context (Li et al., EMNLP 2023; `selective-context` on PyPI, MIT).

Upstream API: `SelectiveContext(model_type="gpt2", lang="en")`, then
`sc(text, reduce_ratio=0.35, reduce_level="phrase") -> (reduced_text, dropped_units)`: it drops the
lowest self-information (GPT-2 surprisal) lexical units. We apply it to tool-result text only,
once per distinct result (memoised, so the history prefix stays byte-stable across steps and the
prompt cache keeps working), with upstream's own defaults.

Adaptations, all forced by upstream and recorded in `arm_meta`:
- results shorter than MIN_CHARS are left alone (nothing to prune, and `ok`-style replies matter);
- text is cut at line boundaries into CHUNK_CHARS pieces before pruning: upstream scores whole
  "sentences" through GPT-2 (1024-token window), and logs/code have no sentence breaks;
- upstream collapses all whitespace inside a piece (`beautify_context`), so line structure
  survives only at piece boundaries. That is how the library behaves, not a harness choice;
- a result that comes back empty or not smaller is replaced by the original / a marker.

Needs a local LM: GPT-2 124M (`openai-community/gpt2`, MIT, `model.safetensors` 548 MB, fetched
from the Hugging Face hub on first use) plus spaCy `en_core_web_sm` and torch. Runs on CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from .arms import Arm, ArmUnavailable

SC_VERSION = "0.1.4"
SC_MODEL, SC_LANG = "gpt2", "en"
SC_RATIO, SC_LEVEL = 0.35, "phrase"  # upstream defaults
MIN_CHARS, CHUNK_CHARS = 200, 800
EMPTY = "[output removed by selective-context]"
WORKER = Path(__file__).with_name("selective_worker.py")
INSTALL_HINT = (
    f"pip install selective-context=={SC_VERSION} 'numpy<2' && python -m spacy download en_core_web_sm "
    "(needs CPython <= 3.10: it pins spacy==3.2.0); then pass that interpreter as "
    "--selective-python"
)


class SelectiveWorker:
    """JSON-lines client for selective_worker.py. `reduce(text)` is one GPT-2 pass per call."""

    def __init__(self, python: str = sys.executable, popen: Callable[..., Any] = subprocess.Popen):
        try:
            self.proc = popen(
                [python, str(WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise ArmUnavailable(f"selective: cannot start {python!r}: {e}. {INSTALL_HINT}") from e
        hello = self._read()
        if not hello.get("ready"):
            self.close()
            raise ArmUnavailable(f"selective: {hello.get('error', hello)}. {INSTALL_HINT}")
        self.versions: dict[str, str | None] = hello["versions"]
        if self.versions.get("selective-context") != SC_VERSION:
            self.close()
            raise ArmUnavailable(
                f"selective: pinned selective-context=={SC_VERSION}, found "
                f"{self.versions.get('selective-context')}. {INSTALL_HINT}"
            )

    def _read(self) -> dict[str, Any]:
        stdout: IO[str] = self.proc.stdout
        line = stdout.readline()
        if not line:
            raise ArmUnavailable(f"selective: worker exited (code {self.proc.poll()})")
        out: dict[str, Any] = json.loads(line)
        return out

    def reduce(self, text: str) -> str:
        stdin: IO[str] = self.proc.stdin
        stdin.write(json.dumps({"text": text, "ratio": SC_RATIO, "level": SC_LEVEL}) + "\n")
        stdin.flush()
        r = self._read()
        if "error" in r:
            raise RuntimeError(
                f"selective-context failed on a {len(text)}-char chunk: {r['error']}"
            )
        return str(r["context"])

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    out: list[str] = []
    cur = ""
    for line in text.splitlines(keepends=True):
        while len(line) > size:  # one huge line: hard split
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:size])
            line = line[size:]
        if len(cur) + len(line) > size and cur:
            out.append(cur)
            cur = ""
        cur += line
    if cur:
        out.append(cur)
    return out


def selective_arm(reduce: Callable[[str], str], meta: dict[str, Any] | None = None) -> Arm:
    """*reduce* maps one piece of text to its pruned form (the real worker, or a fake in tests)."""
    memo: dict[str, str] = {}

    def prune(text: str) -> str:
        if len(text) < MIN_CHARS:
            return text
        if text not in memo:
            parts = [reduce(c).strip() for c in chunks(text) if c.strip()]
            out = "\n".join(p for p in parts if p)
            memo[text] = text if len(out) >= len(text) else out or EMPTY
        return memo[text]

    def block(b: dict[str, Any]) -> dict[str, Any]:
        c = b.get("content")
        if isinstance(c, str):
            return {**b, "content": prune(c)}
        if isinstance(c, list):
            return {
                **b,
                "content": [
                    {**x, "text": prune(x["text"])} if x.get("type") == "text" else x for x in c
                ],
            }
        return b

    # What the latest request sent: tool-result chars before/after pruning. Each request
    # carries the whole history, so the last one is the task's footprint.
    last = {"tool_results": 0, "pruned_results": 0, "chars_before": 0, "chars_after": 0}

    def _text(b: dict[str, Any]) -> str:
        c = b.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(x.get("text", "") for x in c if x.get("type") == "text")
        return ""

    def transform(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        tally = dict.fromkeys(last, 0)
        for m in messages:
            c = m.get("content")
            if m.get("role") == "user" and isinstance(c, list):
                blocks = []
                for b in c:
                    if b.get("type") == "tool_result":
                        nb = block(b)
                        before, after = len(_text(b)), len(_text(nb))
                        tally["tool_results"] += 1
                        tally["pruned_results"] += after < before
                        tally["chars_before"] += before
                        tally["chars_after"] += after
                        b = nb
                    blocks.append(b)
                m = {**m, "content": blocks}
            out.append(m)
        last.update(tally)
        return out

    def on_response(_resp: Any, stats: dict[str, Any]) -> None:
        stats.update(last)

    return Arm(
        "selective",
        transform=transform,
        on_response=on_response,
        meta={
            "library": "selective-context",
            "version": SC_VERSION,
            "license": "MIT",
            "source": "https://github.com/liyucheng09/Selective_Context",
            "lm": SC_MODEL,
            "reduce_ratio": SC_RATIO,
            "reduce_level": SC_LEVEL,
            "min_chars": MIN_CHARS,
            "chunk_chars": CHUNK_CHARS,
            **(meta or {}),
        },
    )


def make_selective(python: str = sys.executable) -> tuple[Arm, SelectiveWorker]:
    w = SelectiveWorker(python)
    return selective_arm(w.reduce, {"worker_versions": w.versions}), w

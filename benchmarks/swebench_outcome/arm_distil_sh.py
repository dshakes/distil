"""`distil-sh` arm: distil's per-command shaping at the source (ADR 0026), modelled the way
the `rtk` arm models RTK's rewrite hook — at the bash tool boundary.

A command the Claude Code PreToolUse hook would rewrite (`distil.shell.plan`) runs in the
task container with `2>&1`, exactly as `distil sh` merges the streams; its output is then
shaped on the host by the same `distil.shell.shape` the CLI uses, at the metered-key tier
(the harness bills an API key, so the elide tier is on). The full output is kept in this
arm's own store and `distil expand <handle>` typed by the agent is answered from it, which
is what the real CLI does from the RestoreStore. A piped `| tail`/`| head` runs in the
container over the shaped output, so the agent sees what the real rewrite would print.

Every other command, and the editor tool, is untouched. The system prompt is unchanged.
Offline-testable with `FakeEnv`; nothing here spends money.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from distil.compress.tier1 import _handle
from distil.shell import FILTERS_VERSION, plan, shape

from .arms import Arm
from .env import Env

_EXIT = re.compile(r"\n\[exit (\d+)\]$")
_EXPAND = re.compile(r"^\s*distil expand ([0-9a-f]{8})(\s*\|.*)?\s*$", re.S)
SCRATCH = "/tmp/distil-sh"


class ShEnv:
    def __init__(self, inner: Env, stats: dict[str, Any]):
        self.inner, self.stats = inner, stats
        self.store: dict[str, str] = {}
        stats.update(
            commands=0,
            shaped=0,
            elided=0,
            chars_before=0,
            chars_after=0,
            expand_calls=0,
            expand_misses=0,
        )

    def _save(self, raw: str) -> str:
        h = _handle(raw)
        self.store[h] = raw
        return h

    def _through(self, text: str, name: str, pipe: str, timeout: int) -> str:
        path = f"{SCRATCH}/{name}"
        self.inner.write_file(path, text)
        return self.inner.exec(f"cat {shlex.quote(path)}{pipe}", timeout)

    def exec(self, cmd: str, timeout: int = 120) -> str:
        self.stats["commands"] += 1
        m = _EXPAND.match(cmd)
        if m:
            self.stats["expand_calls"] += 1
            h, pipe = m[1], m[2] or ""
            if h not in self.store:
                self.stats["expand_misses"] += 1
                return f"distil: no original found for handle {h!r}\n[exit 1]"
            if pipe:
                return self._through(self.store[h], f"{h}.log", pipe, timeout)
            return self.store[h]
        p = plan(cmd)
        if p is None:
            return self.inner.exec(cmd, timeout)
        prefix, body, suffix, kind = p
        out = self.inner.exec(f"{prefix}{body} 2>&1", timeout)
        m2 = _EXIT.search(out)
        raw, tail = (out[: m2.start()], out[m2.start() :]) if m2 else (out, "")
        shaped, tier = shape(raw, kind, lossy=True, save=self._save)
        self.stats["shaped"] += tier != "none"
        self.stats["elided"] += tier == "elide"
        self.stats["chars_before"] += len(raw)
        self.stats["chars_after"] += len(shaped)
        pipe = suffix.replace("2>&1", "").strip()
        if pipe:  # `| tail -30`: the shell's exit status is the pipe's, as in the plain arm
            return self._through(shaped, f"out-{_handle(shaped)}.txt", " " + pipe, timeout)
        return shaped + tail

    def exec_raw(self, argv: list[str], timeout: int = 60) -> tuple[int, str, str]:
        return self.inner.exec_raw(argv, timeout)

    def install_file(self, src: Path, dest: str) -> None:
        self.inner.install_file(src, dest)

    def read_file(self, path: str) -> str:
        return self.inner.read_file(path)

    def write_file(self, path: str, text: str) -> None:
        self.inner.write_file(path, text)

    def diff(self) -> str:
        return self.inner.diff()

    def close(self) -> None:
        self.inner.close()


def distil_sh_arm() -> Arm:
    from .arms import pkg_version

    return Arm(
        "distil-sh",
        wrap_env=lambda env, stats: ShEnv(env, stats),
        meta={
            "library": "distil",
            "version": pkg_version("distil-llm"),
            "filters": FILTERS_VERSION,
            "mode": "claude-code PreToolUse rewrite to `distil sh` (metered tier), bash tool only",
        },
    )

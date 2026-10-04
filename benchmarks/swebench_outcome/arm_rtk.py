"""`rtk` arm: RTK (rtk-ai/rtk, Apache-2.0) in its documented auto-rewrite-hook mode.

RTK is a command wrapper, not a text compressor: its Claude Code hook (`rtk-rewrite.sh`, v4)
sends each Bash command to `rtk rewrite "<cmd>"` and runs what comes back (`git status` ->
`rtk git status`, `cat f` -> `rtk read f`, `pytest ...` -> `rtk pytest ...`). The filtered output
of the rewritten command is what the agent reads. We model exactly that at the bash tool
boundary, inside the task container, using the real pinned binary:

  exit 0 -> rewrite (allow rule)    exit 3 -> rewrite (ask/no rule: default config)
  exit 1 -> no RTK equivalent       exit 2 -> deny rule: both run the command unchanged
  anything else (binary missing/crashed) -> EnvError, never a silent pass-through

Like the real hook, only bash commands are rewritten; the editor tool is not (Claude Code's
Read/Grep/Glob likewise bypass the hook). The system prompt is unchanged (RTK.md is silent by
default). Source for the protocol: hooks/claude/rtk-rewrite.sh at tag v0.51.0.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .arms import Arm, ArmUnavailable
from .env import Env, EnvError

RTK_VERSION = "0.51.0"
RTK_ASSET = "rtk-x86_64-unknown-linux-musl.tar.gz"  # SWE-bench images are linux/amd64
RTK_URL = f"https://github.com/rtk-ai/rtk/releases/download/v{RTK_VERSION}/{RTK_ASSET}"
# From the release's checksums.txt (v0.51.0).
RTK_SHA256 = "5028d3b19a8f0990d30fec9fbb07e32782bc5698e618fb1861aad8a9ccba4eb5"
CONTAINER_PATH = "/usr/local/bin/rtk"
REWRITE_EXIT = (0, 3)
PASSTHROUGH_EXIT = (1, 2)


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "distil-eval"


def fetch_rtk(
    cache: Path | None = None,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    sha256: str = RTK_SHA256,
) -> Path:
    """The pinned linux binary, downloaded once and verified against the pinned sha256."""
    cache = cache or cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / f"rtk-{RTK_VERSION}-linux-musl"
    tarball = cache / RTK_ASSET
    if not tarball.exists():
        try:
            with urlopen(RTK_URL, timeout=120) as r:
                tarball.write_bytes(r.read())
        except OSError as e:
            raise ArmUnavailable(
                f"rtk: cannot download {RTK_URL} ({e}); fetch it yourself and pass --rtk-bin"
            ) from e
    data = tarball.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        tarball.unlink()
        raise ArmUnavailable(f"rtk: {RTK_ASSET} sha256 {got} != pinned {sha256}; deleted")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            member = tf.extractfile("rtk")
            if member is None:
                raise KeyError("rtk")
            binary.write_bytes(member.read())
    except (KeyError, tarfile.TarError) as e:
        raise ArmUnavailable(f"rtk: no `rtk` binary in {RTK_ASSET}: {e!r}") from e
    binary.chmod(0o755)
    return binary


def rewrite(env: Env, cmd: str) -> str:
    """What the RTK hook would execute instead of *cmd*."""
    # env -u RTK_REWRITE_HOST: the real hook scrubs it (it relaxes RTK's approval gate).
    rc, out, err = env.exec_raw(["env", "-u", "RTK_REWRITE_HOST", "rtk", "rewrite", cmd], 30)
    if rc in REWRITE_EXIT:
        return out.rstrip("\n") or cmd
    if rc in PASSTHROUGH_EXIT:
        return cmd
    raise EnvError(f"rtk rewrite failed (exit {rc}): {(err or out).strip()[:300]}")


class RtkEnv:
    """Bash goes through `rtk rewrite`; everything else is the wrapped env unchanged."""

    def __init__(self, inner: Env, stats: dict[str, Any]):
        self.inner, self.stats = inner, stats
        stats.update(commands=0, rewritten=0)

    def exec(self, cmd: str, timeout: int = 120) -> str:
        new = rewrite(self.inner, cmd)
        self.stats["commands"] += 1
        self.stats["rewritten"] += new != cmd
        return self.inner.exec(new, timeout)

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


def rtk_arm(binary: Path) -> Arm:
    def wrap(env: Env, stats: dict[str, Any]) -> Env:
        env.install_file(binary, CONTAINER_PATH)
        rc, out, err = env.exec_raw([CONTAINER_PATH, "--version"])
        if rc or out.strip() != f"rtk {RTK_VERSION}":
            raise EnvError(
                f"container rtk is {out.strip() or err.strip()!r} (exit {rc}); pinned rtk {RTK_VERSION}"
            )
        return RtkEnv(env, stats)

    return Arm(
        "rtk",
        wrap_env=wrap,
        meta={
            "library": "rtk",
            "version": RTK_VERSION,
            "source": "https://github.com/rtk-ai/rtk",
            "license": "Apache-2.0",
            "mode": "claude-code auto-rewrite hook (`rtk rewrite`), bash tool only",
        },
    )

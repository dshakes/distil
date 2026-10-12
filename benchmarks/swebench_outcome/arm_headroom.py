"""`headroom` arm: the real Headroom proxy (`headroom-ai`, Apache-2.0) in front of the Anthropic API.

This is how Headroom is used with Claude: `headroom wrap claude` starts `python -m headroom.cli
proxy --port N --workers 1` and points the agent's ANTHROPIC_BASE_URL at it (cli/wrap.py
`_start_proxy`). We start the same command, with the same agent env (HEADROOM_AGENT_TYPE=claude,
HEADROOM_STACK=wrap_claude) and Headroom's own defaults (`cache` mode, CCR retrieval handled inside
the proxy), and send this harness's requests through it. Nothing about Headroom's transforms is
reimplemented: the arm only owns the process (start, /livez health check, stop) and a client whose
base_url is the proxy. The proxy forwards to api.anthropic.com with the request's own API key.

Harness-only deviations, recorded in `arm_meta`: `--no-subscription-tracking` (we bill an API key),
telemetry left at its default (off), and HOME / HEADROOM_WORKSPACE_DIR in a throwaway directory so
a run never reads or writes the maintainer's ~/.headroom or ~/.claude. HEADROOM_WRAP_OWNED is NOT
set: that flag makes the proxy exit when no `wrap` client is alive.

The proxy needs its own interpreter (CPython >= 3.10, `headroom-ai[proxy]`), passed as
--headroom-python, like the selective arm.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .arms import Arm, ArmUnavailable

HR_VERSION = "0.40.0"
HR_EXTRA = "proxy"
HR_MODE = "cache"  # the proxy's own default; recorded, not passed
STARTUP_TIMEOUT = 300.0  # the ML extras load models on first start
INSTALL_HINT = (
    f"uv venv --python 3.12 hr-venv && uv pip install --python hr-venv/bin/python "
    f"'headroom-ai[{HR_EXTRA}]=={HR_VERSION}'; then pass hr-venv/bin/python as --headroom-python"
)
_PROBE = (
    "import importlib.metadata as m, json; import fastapi, uvicorn, headroom; "
    "print(json.dumps({'headroom-ai': m.version('headroom-ai')}))"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def probe(python: str, run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Version of headroom-ai in *python*; ArmUnavailable if missing, proxy extra absent or wrong pin."""
    try:
        r = run([python, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ArmUnavailable(f"headroom: cannot run {python!r}: {e}. {INSTALL_HINT}") from e
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise ArmUnavailable(f"headroom: not importable in {python!r} ({tail[0]}). {INSTALL_HINT}")
    found: dict[str, Any] = json.loads(r.stdout.strip().splitlines()[-1])
    if found["headroom-ai"] != HR_VERSION:
        raise ArmUnavailable(
            f"headroom: pinned headroom-ai=={HR_VERSION}, found {found['headroom-ai']}. {INSTALL_HINT}"
        )
    return found


class HeadroomProxy:
    """A per-run Headroom proxy process. `url` is its base_url; `close()` stops it."""

    def __init__(
        self,
        python: str = sys.executable,
        port: int | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        timeout: float = STARTUP_TIMEOUT,
        extra_env: dict[str, str] | None = None,
    ):
        self.versions = probe(python)
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.dir = Path(tempfile.mkdtemp(prefix="headroom-arm-"))
        self.log_path = self.dir / "proxy.log"
        env = {**os.environ, **(extra_env or {})}
        env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        env.pop("HEADROOM_WRAP_OWNED", None)
        env.pop("HEADROOM_WORKERS", None)
        env.update(
            HOME=str(self.dir),
            HEADROOM_WORKSPACE_DIR=str(self.dir),
            HEADROOM_AGENT_TYPE="claude",
            HEADROOM_STACK="wrap_claude",
            PYTHONIOENCODING="utf-8",
            PYTHONSAFEPATH="1",
        )
        cmd = [
            python, "-m", "headroom.cli", "proxy",
            "--port", str(self.port), "--workers", "1", "--no-subscription-tracking",
        ]  # fmt: skip
        self._log = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        try:
            self.proc = popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT)
        except OSError as e:
            self._log.close()
            raise ArmUnavailable(f"headroom: cannot start {python!r}: {e}. {INSTALL_HINT}") from e
        self._wait_live(timeout)

    def _wait_live(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                self.close()
                raise ArmUnavailable(
                    f"headroom: proxy exited (code {self.proc.returncode}) before it was live; "
                    f"log tail: {self.log_tail()!r}"
                )
            try:
                with urllib.request.urlopen(self.url + "/livez", timeout=2) as r:  # noqa: S310
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        self.close()
        raise ArmUnavailable(
            f"headroom: proxy not live after {timeout:.0f}s; log: {self.log_tail()!r}"
        )

    def log_tail(self, n: int = 400) -> str:
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()


def headroom_arm(proxy_url: str, client_factory: Callable[[str], Any], meta: dict[str, Any]) -> Arm:
    """Requests go to the proxy; the arm itself has no transform, tools or handlers."""
    return Arm(
        "headroom",
        client=client_factory(proxy_url),
        meta={
            "library": "headroom-ai",
            "version": HR_VERSION,
            "license": "Apache-2.0",
            "source": "https://github.com/headroomlabs-ai/headroom",
            "integration": "proxy (headroom wrap claude's proxy command)",
            "mode": HR_MODE,
            "harness_flags": ["--no-subscription-tracking"],
            **meta,
        },
    )


def make_headroom(python: str = sys.executable) -> tuple[Arm, HeadroomProxy]:
    import anthropic

    proxy = HeadroomProxy(python)
    return (
        headroom_arm(
            proxy.url,
            lambda u: anthropic.Anthropic(base_url=u, max_retries=4),
            {"worker_versions": proxy.versions},
        ),
        proxy,
    )

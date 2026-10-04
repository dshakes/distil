"""Task environments. The agent only sees this interface; the patch is `Env.diff()`."""

from __future__ import annotations

import difflib
import shutil
import subprocess
import uuid
from collections.abc import Callable
from typing import Protocol


class EnvError(RuntimeError):
    """The sandbox itself failed (docker down, container died); never the agent's fault."""


class Env(Protocol):
    def exec(self, cmd: str, timeout: int = 120) -> str: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, text: str) -> None: ...
    def diff(self) -> str: ...
    def close(self) -> None: ...


class FakeEnv:
    """In-memory env for tests. `exec_fn(cmd) -> str` scripts bash output."""

    def __init__(
        self, files: dict[str, str] | None = None, exec_fn: Callable[[str], str] | None = None
    ):
        self.orig = dict(files or {})
        self.files = dict(self.orig)
        self.exec_fn = exec_fn
        self.cmds: list[str] = []
        self.closed = False

    def exec(self, cmd: str, timeout: int = 120) -> str:
        self.cmds.append(cmd)
        return self.exec_fn(cmd) if self.exec_fn else ""

    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_file(self, path: str, text: str) -> None:
        self.files[path] = text

    def diff(self) -> str:
        out: list[str] = []
        for p in sorted(set(self.orig) | set(self.files)):
            a, b = self.orig.get(p, ""), self.files.get(p, "")
            if a != b:
                out += difflib.unified_diff(
                    a.splitlines(True), b.splitlines(True), f"a/{p}", f"b/{p}"
                )
        return "".join(out)

    def close(self) -> None:
        self.closed = True


def instance_image(instance_id: str) -> str:
    """SWE-bench per-instance image name. Prefer the swebench package's own TestSpec."""
    try:  # API path ASSUMED (not importable in the authoring env); fall back to the known pattern.
        from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

        return make_test_spec({"instance_id": instance_id}, namespace="swebench").instance_image_key  # type: ignore[arg-type]
    except Exception:
        return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed && "


class DockerEnv:
    """Runs a prebuilt SWE-bench instance image via the docker CLI (no docker-py needed)."""

    def __init__(self, instance_id: str, image: str | None = None, workdir: str = "/testbed"):
        if not shutil.which("docker"):
            raise EnvError("docker CLI not found")
        self.workdir = workdir
        self.name = f"swo-{uuid.uuid4().hex[:10]}"
        self.image = image or instance_image(instance_id)
        # Model-driven commands run here: no network, no extra privileges, bounded resources.
        # SWE-bench instance images ship their dependencies, so tests run offline.
        r = self._run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.name,
                # SWE-bench instance images are x86_64-only; arm64 hosts run them under emulation.
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "512",
                "--memory",
                "4g",
                self.image,
                "sleep",
                "infinity",
            ]
        )
        if r.returncode:
            raise EnvError(f"docker run {self.image}: {r.stderr.strip()[:500]}")

    @staticmethod
    def _run(argv: list[str], timeout: int = 600, inp: str | None = None):
        try:
            return subprocess.run(
                argv, capture_output=True, text=True, errors="replace", timeout=timeout, input=inp
            )
        except subprocess.TimeoutExpired as e:
            raise EnvError(f"timeout: {argv[:3]}") from e

    def exec(self, cmd: str, timeout: int = 120) -> str:
        argv = ["docker", "exec", "-w", self.workdir, self.name, "bash", "-c", ACTIVATE + cmd]
        try:
            r = subprocess.run(
                argv, capture_output=True, text=True, errors="replace", timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return f"command timed out after {timeout}s"
        return (
            (r.stdout + r.stderr)
            if r.returncode == 0
            else f"{r.stdout}{r.stderr}\n[exit {r.returncode}]"
        )

    def read_file(self, path: str) -> str:
        r = self._run(["docker", "exec", "-w", self.workdir, self.name, "cat", path])
        if r.returncode:
            raise FileNotFoundError(path)
        return r.stdout

    def write_file(self, path: str, text: str) -> None:
        script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        r = self._run(
            [
                "docker",
                "exec",
                "-i",
                "-w",
                self.workdir,
                self.name,
                "bash",
                "-c",
                script,
                "_",
                path,
            ],
            inp=text,
        )
        if r.returncode:
            raise EnvError(f"write {path}: {r.stderr.strip()[:300]}")

    def diff(self) -> str:
        r = self._run(
            [
                "docker",
                "exec",
                "-w",
                self.workdir,
                self.name,
                "bash",
                "-c",
                "git add -A >/dev/null 2>&1; git diff --cached HEAD",
            ]
        )
        if r.returncode:
            raise EnvError(f"git diff: {r.stderr.strip()[:300]}")
        return r.stdout

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)

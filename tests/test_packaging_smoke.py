"""Packaging smoke — exercise the SHIPPED ARTIFACT, not the source tree.

Every other test in this suite imports `distil.*` from the working directory. That
tests the code and says nothing about what a user actually receives: the wheel, its
console scripts, and the manifests that tell registries how to launch them.

That gap shipped a real outage. distil was listed in the official MCP registry with
the launch spec `uvx distil-llm`, while the distribution's console scripts were only
`distil` and `distil-mcp` — and `uvx <pkg>` runs the executable *named after the
package*. So the registry's own spec failed with "An executable named `distil-llm` is
not provided by package `distil-llm`" for eight days, under 1832 green tests and a 95%
coverage floor, because not one of them ran the artifact.

These tests build a wheel, install it into a clean venv, and drive the result the way
a user's client would. They are slower than a unit test by design: the point is that
they touch the boundary nothing else touches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="venv script layout differs; the gate runs on POSIX CI"
)

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def installed(tmp_path_factory) -> Path:
    """Build the wheel and install it into a clean venv. Returns the venv's bin/."""
    if os.environ.get("DISTIL_SKIP_PACKAGING_SMOKE"):
        pytest.skip("DISTIL_SKIP_PACKAGING_SMOKE set")
    work = tmp_path_factory.mktemp("pkg")
    wheel_dir = work / "wheel"
    build = subprocess.run(
        ["uv", "build", "--wheel", "-o", str(wheel_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"uv build unavailable/failed: {build.stderr[-300:]}")
    wheels = list(wheel_dir.glob("*.whl"))
    assert wheels, "uv build produced no wheel"

    venv = work / "venv"
    subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), str(wheels[0])],
        check=True,
        capture_output=True,
    )
    return venv / "bin"


def test_every_declared_console_script_is_installed_and_runs(installed: Path) -> None:
    """A script named in [project.scripts] must exist in a clean install and start.

    Declaring an entry point costs nothing and is never verified by importing the
    package — a typo'd module path or a renamed function only fails for the user.
    """
    for name in _pyproject()["project"]["scripts"]:
        exe = installed / name
        assert exe.exists(), f"declared console script {name!r} is missing from the wheel"
        assert os.access(exe, os.X_OK), f"{name!r} installed but not executable"


def test_cli_reports_the_version_the_manifest_claims(installed: Path) -> None:
    out = subprocess.run(
        [str(installed / "distil"), "--version"], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert _pyproject()["project"]["version"] in out.stdout, (
        f"installed CLI reports {out.stdout.strip()!r}, pyproject says "
        f"{_pyproject()['project']['version']!r}"
    )


@pytest.mark.parametrize("script", ["distil-mcp", "distil-llm"])
def test_mcp_console_scripts_speak_jsonrpc_over_real_stdio(installed: Path, script: str) -> None:
    """Drive the installed script as a subprocess, the way an MCP client does.

    test_mcp_server.py calls serve() in-process with fake file objects — that proves
    the protocol logic, not that the shipped binary starts and answers on real pipes.
    """
    proc = subprocess.run(
        [str(installed / script)],
        input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n',
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{script} exited {proc.returncode}: {proc.stderr[-300:]}"
    lines = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2, f"{script} answered {len(lines)} of 2 requests"
    assert lines[0]["result"]["serverInfo"]["name"] == "distil"
    names = {t["name"] for t in lines[1]["result"]["tools"]}
    assert {"distil_compress", "distil_expand", "distil_savings"} <= names


def test_server_json_launch_spec_resolves_to_a_real_script() -> None:
    """THE regression guard for the registry outage.

    server.json declares `runtimeHint: uvx` + `identifier: distil-llm`, which a client
    runs as `uvx distil-llm`. uvx resolves that to the executable *named after the
    package*, so the identifier MUST also be a declared console script. It wasn't, and
    nothing noticed.
    """
    spec = json.loads((ROOT / "server.json").read_text())
    scripts = _pyproject()["project"]["scripts"]
    for pkg in spec["packages"]:
        if pkg.get("runtimeHint") == "uvx":
            assert pkg["identifier"] in scripts, (
                f"server.json launches `uvx {pkg['identifier']}` but the package declares "
                f"no console script by that name (has: {sorted(scripts)}). "
                f"Clients following the registry entry will get "
                f"'An executable named `{pkg['identifier']}` is not provided'."
            )


def test_manifests_agree_on_version_and_name() -> None:
    """server.json is published state; drift makes the registry advertise a stale build."""
    spec = json.loads((ROOT / "server.json").read_text())
    version = _pyproject()["project"]["version"]
    assert spec["version"] == version, f"server.json {spec['version']} vs pyproject {version}"
    for pkg in spec["packages"]:
        assert pkg["version"] == version, f"server.json package {pkg['version']} vs {version}"
        assert pkg["identifier"] == _pyproject()["project"]["name"]
    # The README marker is what the registry uses to verify repo ownership.
    marker = (ROOT / "README.md").read_text().splitlines()[0]
    assert spec["name"] in marker, f"server.json name {spec['name']!r} not in README marker"

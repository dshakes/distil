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
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="venv script layout differs; the gate runs on POSIX CI"
)

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    """The three pyproject fields these checks need, without tomllib.

    tomllib is stdlib only on 3.11+, and this package supports 3.9 — importing it at
    module scope broke *collection* on the 3.9 leg, taking the whole test run with it.
    release.sh reads the same fields with grep/sed for the same reason.
    """
    text = (ROOT / "pyproject.toml").read_text()
    project = {
        "version": re.search(r'^version = "([^"]+)"', text, re.M).group(1),
        "name": re.search(r'^name = "([^"]+)"', text, re.M).group(1),
        "scripts": {},
    }
    block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[)", text, re.M | re.S)
    if block:
        for line in block.group(1).splitlines():
            entry = re.match(r'^\s*([A-Za-z0-9._-]+)\s*=\s*"([^"]+)"', line)
            if entry:
                project["scripts"][entry.group(1)] = entry.group(2)
    return {"project": project}


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


def test_zipapp_reports_the_version_the_manifest_claims(tmp_path: Path) -> None:
    """The .pyz is a shipped release asset, and nothing else here runs it.

    It has no dist-info, so `importlib.metadata` misses; and the pyproject fallback
    in distil/__init__.py cannot read_text() inside a zip archive. Both paths fail
    silently to "0+source", which is exactly what v1.31.0, v1.32.0 and v1.33.0
    published — a customer-facing artifact that could not name its own version,
    through three releases and a green suite, because no test executed it.
    """
    build = subprocess.run(
        ["bash", str(ROOT / "scripts" / "build_pyz.sh")],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert build.returncode == 0, build.stderr[-500:]

    pyz = ROOT / "dist" / "distil.pyz"
    assert pyz.exists(), "build_pyz.sh reported success but produced no archive"

    # Run it where distil-llm is NOT installed. Under this suite's own interpreter
    # importlib.metadata finds the editable install and answers correctly, so the
    # zipapp path never executes and a broken archive still looks green — which is
    # how this defect reached three releases. A bare venv is the only honest probe.
    venv = tmp_path / "clean"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    python = venv / "bin" / "python"
    assert python.exists(), "bare venv produced no interpreter"

    out = subprocess.run(
        [str(python), str(pyz), "--version"], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-500:]
    expected = _pyproject()["project"]["version"]
    assert expected in out.stdout, (
        f"zipapp reports {out.stdout.strip()!r}, pyproject says {expected!r}"
    )
    assert "0+source" not in out.stdout, "the zipapp fell back instead of reading its stamp"


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


def test_npm_package_version_matches() -> None:
    """The npm wrapper is a published surface with a version badge in the README,
    and nothing was checking it. It drifted to 1.27.0 in-repo while the registry
    still served 1.25.0 and PyPI was on 1.33.1 — a badge advertising a build nine
    releases old. Same failure class as server.json: a manifest promises a version
    and nothing verifies the promise.
    """
    pkg = json.loads((ROOT / "packaging" / "npm" / "package.json").read_text())
    version = _pyproject()["project"]["version"]
    assert pkg["version"] == version, (
        f"packaging/npm/package.json is {pkg['version']}, package is {version} — "
        "bump it with the other manifests at release time"
    )
    assert pkg["name"] == _pyproject()["project"]["name"]


# ---------------------------------------------------------------------------
# Other declared surfaces: Docker, Homebrew, the Claude Code plugin.
# Same class as the registry outage — a manifest promises something and nothing
# ever checks that the promise resolves. These are static cross-manifest checks
# (fast, always run); the heavy "actually build the image" check is a CI job.
# ---------------------------------------------------------------------------


def test_dockerfile_entrypoint_is_a_real_console_script() -> None:
    """ENTRYPOINT ["distil"] is the image's whole interface; a rename breaks every pull."""
    text = (ROOT / "Dockerfile").read_text()
    entry = re.search(r'^ENTRYPOINT\s+\[\s*"([^"]+)"', text, re.M)
    assert entry, "Dockerfile declares no ENTRYPOINT"
    assert entry.group(1) in _pyproject()["project"]["scripts"], (
        f"Dockerfile ENTRYPOINT is {entry.group(1)!r} but that is not a declared console "
        f"script (has: {sorted(_pyproject()['project']['scripts'])})"
    )
    cmd = re.search(r'^CMD\s+\[\s*"([^"]+)"', text, re.M)
    if cmd:  # CMD is the default subcommand handed to that entrypoint
        help_text = subprocess.run(
            [sys.executable, "-m", "distil.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        ).stdout
        assert cmd.group(1) in help_text, (
            f"Dockerfile CMD is {cmd.group(1)!r} but the CLI exposes no such subcommand"
        )


def test_homebrew_formula_is_self_consistent() -> None:
    """url tag, version field and the sha comment must name the SAME tag.

    Not compared against pyproject on purpose: the formula pins the last *released*
    tag and legitimately lags an unreleased bump. What must never disagree is the
    formula with itself — that is how the sha comment drifted 18 releases.
    """
    text = (ROOT / "packaging" / "homebrew" / "distil.rb").read_text()
    url_tag = re.search(r"/tags/v([0-9][^\"/]*)\.tar\.gz", text)
    version = re.search(r'^\s*version\s+"([^"]+)"', text, re.M)
    comment = re.search(r"# sha256 is for the v([0-9]\S*?) source tarball", text)
    assert url_tag and version and comment, "formula is missing url/version/sha-comment"
    assert url_tag.group(1) == version.group(1) == comment.group(1), (
        f"formula disagrees with itself: url=v{url_tag.group(1)} "
        f"version={version.group(1)} sha-comment=v{comment.group(1)}"
    )
    assert re.search(r'^\s*sha256\s+"[a-f0-9]{64}"', text, re.M), "formula sha256 is malformed"


def test_claude_plugin_manifests_are_valid_and_current() -> None:
    """The plugin is a shipped surface too: stale metadata is what users see."""
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    version = _pyproject()["project"]["version"]
    for entry in market["plugins"]:
        src = (ROOT / entry["source"]).resolve()
        assert src.is_dir(), f"marketplace lists {entry['source']!r} but that path does not exist"
        manifest = src / ".claude-plugin" / "plugin.json"
        assert manifest.exists(), f"{entry['name']}: no .claude-plugin/plugin.json"
        plugin = json.loads(manifest.read_text())
        assert plugin["name"] == entry["name"], "marketplace/plugin name mismatch"
        assert plugin["version"] == version, (
            f"plugin.json is {plugin['version']}, package is {version} — the plugin "
            "manifest drifts silently because nothing bumps it at release time"
        )


def test_every_skill_and_command_has_usable_frontmatter() -> None:
    """A skill with no description is never selected; a command with none is unusable."""
    plugin_root = ROOT / "plugins" / "distil"
    skills = list((plugin_root / "skills").glob("*/SKILL.md"))
    assert skills, "no skills found — the glob or the layout changed"
    for skill in skills:
        head = skill.read_text()
        assert head.startswith("---"), f"{skill.name}: no YAML frontmatter"
        fm = head.split("---", 2)[1]
        name = re.search(r"^name:\s*(\S+)", fm, re.M)
        assert name, f"{skill}: frontmatter has no name"
        assert name.group(1) == skill.parent.name, (
            f"{skill}: frontmatter name {name.group(1)!r} != directory {skill.parent.name!r}"
        )
        assert re.search(r"^description:", fm, re.M), f"{skill}: frontmatter has no description"

    commands = list((plugin_root / "commands").glob("*.md"))
    assert commands, "no commands found"
    for cmd in commands:
        body = cmd.read_text()
        assert body.startswith("---"), f"{cmd.name}: no frontmatter"
        assert re.search(r"^description:", body.split("---", 2)[1], re.M), (
            f"{cmd.name}: no description — it renders blank in the slash-command list"
        )


def test_release_smoke_test_cannot_file_a_census_ping() -> None:
    """A throwaway venv must not borrow this machine's census identity.

    The release script installs the wheel into a temp venv and runs the CLI
    against it. Without an isolated DISTIL_HOME that venv inherits the real
    ~/.distil — same install id, same consent — so anything it reports lands on
    the public adoption board as an install that does not exist. That is not
    hypothetical: during the 1.36.0 verification the board showed an install on
    1.36.0 while the only real one was on 1.34.0.

    Pinned as text because the failure is a deleted line, not a wrong value.
    """
    script = (ROOT / "scripts" / "release.sh").read_text()
    smoke = script.split("clean-install smoke test", 1)
    assert len(smoke) > 1, "clean-install smoke block not found — did release.sh move?"
    block = smoke[1].split("push main", 1)[0]

    assert 'export DISTIL_HOME="$SMOKE/home"' in block, (
        "the release smoke test runs the CLI without isolating DISTIL_HOME — it can "
        "ping the census under this machine's install id"
    )
    # The isolation must be in force for every CLI invocation in the block, so it
    # has to come before the first one.
    isolate_at = block.index("export DISTIL_HOME=")
    first_cli = min(
        (block.index(c) for c in ('"$SMOKE/venv/bin/distil"',) if c in block),
        default=-1,
    )
    assert first_cli != -1, "smoke block no longer invokes the installed CLI"
    assert isolate_at < first_cli, "DISTIL_HOME is set after the CLI has already run"

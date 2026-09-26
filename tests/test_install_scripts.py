"""docs/install.sh and docs/install.ps1 are served from the Pages site and piped into a shell.

The risk worth a test is the one a reviewer skims past: a script that pipes a
download from some new host into `sh`/`iex`. Every URL the scripts name must be on
a short allowlist, and the only thing ever piped into a shell is Astral's uv
installer. The sh script's control flow is exercised offline against a stub `uv`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
SH = DOCS / "install.sh"
PS1 = DOCS / "install.ps1"

URL_HOSTS = {"astral.sh", "docs.astral.sh", "dshakes.github.io", "github.com"}
PIPE_TO_SHELL_HOSTS = {"astral.sh"}

_URL = re.compile(r"https?://[^\s\"'|)]+")
_PIPE_TO_SHELL = re.compile(
    r"(?:curl|wget|irm|iwr|Invoke-RestMethod|Invoke-WebRequest)\b[^|\n]*?(https?://[^\s\"'|)]+)"
    r"[^|\n]*\|\s*(?:sh|bash|zsh|iex|Invoke-Expression)\b",
    re.I,
)


@pytest.mark.parametrize("script", [SH, PS1], ids=lambda p: p.name)
def test_every_url_is_allowlisted(script: Path) -> None:
    hosts = {urlparse(u).hostname for u in _URL.findall(script.read_text(encoding="utf-8"))}
    assert hosts, f"{script.name}: found no URLs; the scanner is broken"
    assert hosts <= URL_HOSTS, f"{script.name}: unexpected hosts {hosts - URL_HOSTS}"


@pytest.mark.parametrize("script", [SH, PS1], ids=lambda p: p.name)
def test_only_astral_is_piped_into_a_shell(script: Path) -> None:
    # Comments show the user-facing one-liner (our own URL | sh); only code lines count.
    code = "\n".join(
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    piped = {urlparse(m.group(1)).hostname for m in _PIPE_TO_SHELL.finditer(code)}
    assert piped == PIPE_TO_SHELL_HOSTS, f"{script.name}: pipes {piped} into a shell"


needs_sh = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="POSIX sh")


@needs_sh
def test_sh_syntax() -> None:
    subprocess.run(["sh", "-n", str(SH)], check=True)


def _run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run install.sh with a stub `uv` that records its argv; no network, no real install."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    uv = bindir / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/uv.log"\n', encoding="utf-8")
    uv.chmod(0o755)
    full_env = {
        "PATH": f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": str(tmp_path),
        **env,
    }
    return subprocess.run(
        ["sh", str(SH), *args], env=full_env, capture_output=True, text=True, check=False
    )


@needs_sh
def test_installs_latest_and_names_the_next_step(tmp_path: Path) -> None:
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "uv.log").read_text().strip() == "tool install --upgrade distil-llm"
    assert "distil setup" in r.stdout


@needs_sh
def test_respects_the_version_pin(tmp_path: Path) -> None:
    r = _run(tmp_path, DISTIL_VERSION="1.54.0")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "uv.log").read_text().strip() == "tool install --upgrade distil-llm==1.54.0"


@needs_sh
def test_rejects_a_hostile_pin_before_touching_anything(tmp_path: Path) -> None:
    r = _run(tmp_path, DISTIL_VERSION="1.0; rm -rf ~")
    assert r.returncode != 0
    assert "DISTIL_VERSION" in r.stderr
    assert not (tmp_path / "uv.log").exists()


@needs_sh
def test_help_and_unknown_args(tmp_path: Path) -> None:
    assert "DISTIL_VERSION" in _run(tmp_path, "--help").stdout
    assert _run(tmp_path, "--bogus").returncode != 0
    assert not (tmp_path / "uv.log").exists()

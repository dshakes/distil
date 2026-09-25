# Install distil (https://github.com/dshakes/distil) on Windows as an isolated uv tool.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://dshakes.github.io/distil/install.ps1 | iex"
#   $env:DISTIL_VERSION = "1.54.0"; powershell -ExecutionPolicy ByPass -c "irm https://dshakes.github.io/distil/install.ps1 | iex"
#
# What it does, and nothing else:
#   1. if `uv` is missing, runs Astral's official uv installer (https://astral.sh/uv/install.ps1);
#      that installer is the only thing that may change your user PATH
#   2. `uv tool install --upgrade distil-llm` (or the pinned DISTIL_VERSION)
#   3. prints the next step: `distil setup`
# No admin rights. Safe to re-run: a second run upgrades in place.
param([switch]$Help)
$ErrorActionPreference = 'Stop'

if ($Help) {
    Write-Output @'
Install distil as an isolated uv tool.

Usage: install.ps1 [-Help]

Environment:
  DISTIL_VERSION   install exactly this version (e.g. 1.54.0); default: latest

Installs uv first if it is missing, using Astral's official installer.
Re-running upgrades in place. Next step after install: distil setup
'@
    return
}

# throw, not exit: under `irm | iex` in an open terminal, exit would close the window.
function Fail([string]$Message) { throw "distil install: error: $Message" }

$spec = 'distil-llm'
if ($env:DISTIL_VERSION) {
    if ($env:DISTIL_VERSION -notmatch '^[0-9A-Za-z.+-]+$') {
        Fail "DISTIL_VERSION must look like 1.54.0, got: $($env:DISTIL_VERSION)"
    }
    $spec = "distil-llm==$($env:DISTIL_VERSION)"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output "distil install: uv not found; installing it with Astral's official installer"
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) { Fail "the uv installer failed (see its output above)" }
    # The installer updates your user PATH for future shells; this one needs it now.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail "uv was installed but is not on PATH; open a new terminal and re-run"
    }
}

Write-Output "distil install: uv tool install --upgrade $spec"
uv tool install --upgrade $spec
if ($LASTEXITCODE -ne 0) { Fail "uv tool install $spec failed (see uv's output above)" }

Write-Output ""
if (Get-Command distil -ErrorAction SilentlyContinue) {
    Write-Output "distil is installed. Next step:"
} else {
    Write-Output "distil is installed, but uv's tool directory is not on your PATH yet."
    Write-Output "Run 'uv tool update-shell', open a new terminal, then:"
}
Write-Output "  distil setup"

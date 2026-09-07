"""Config-file wrap targets: tools whose ONLY routing knob is a config file,
not an environment variable, so they can't use ``onboard.AGENT_PRESETS``
(``distil wrap``'s env-var mechanism) at all.

Each preset is doc-cited (URL + verification date). Where a docs page turned
out to be client-rendered and returned nothing over a plain fetch, the field
names come from the tool's own schema/source on GitHub instead of a guess —
that's called out per preset below. A tool whose config shape could NOT be
pinned down this way (Crush's current bash-script ``crushrc``, Amp, Mistral
Vibe, and OpenClaw — a persistent multi-channel gateway, not a per-session
CLI) is deliberately not here; see docs/IDE-AGENTS.md for why.

Three strategies, cheapest/safest first:

  ``flag``    a documented ``--config <path>`` the tool only honors for THIS
              invocation (Continue): distil renders a self-contained temp
              config and never touches anything the tool would otherwise
              read. No backup needed — there is nothing to restore.
  ``overlay`` a documented local *override layer* the tool merges on top of
              its real config (Factory Droid's ``settings.local.json``):
              distil owns the whole file. If it didn't exist, cleanup is
              delete; if it did, byte-for-byte backup + restore.
  ``patch``   no flag and no override layer exist (Oh My Pi): distil backs
              up the real file byte-for-byte, splices in a marker-fenced
              provider block, and restores the exact original bytes on
              exit. ``restore_stale_backups`` repeats the restore at the
              top of the next ``distil wrap`` if a crash (SIGKILL, power
              loss) skipped it the first time.

Every preset also refuses to invent a credential: if the api-key env var it
would embed isn't set, it prints why and leaves the tool's config untouched
(same rule ``proxy.wrap_run``'s ``extra_env`` already follows).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Wraps a marker-fenced block spliced into a "patch"-strategy file, so a
#: human (or a future distil session) can tell our block apart from the rest
#: of the user's own config on sight.
_MARKER_BEGIN = "# --- distil wrap: injected, restored on exit ---"
_MARKER_END = "# --- end distil wrap ---"

ApplyFn = Callable[[str, str], "contextlib.AbstractContextManager[list[str]]"]


def _family(upstream: str) -> str:
    """ "anthropic" or "openai" — the same two wire shapes every entry in
    ``onboard.AGENT_PRESETS`` already keys off of (by env-var name), read
    here from ``--upstream`` instead since these tools have no env var."""
    return "anthropic" if "anthropic.com" in upstream else "openai"


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".distil-backup")


def _created_marker(path: Path) -> Path:
    """Sentinel for the "the file didn't exist before we wrote it" case —
    there's no `.distil-backup` to restore from, so restore means *delete*,
    and this marks that that's the right thing to do even after a crash."""
    return path.with_name(path.name + ".distil-created")


@dataclass(frozen=True)
class ConfigPreset:
    label: str
    strategy: str  # "flag" | "overlay" | "patch"
    doc_url: str
    verified: str  # "YYYY-MM-DD"
    apply: ApplyFn
    #: Stable paths this preset ever writes to directly (not temp files) —
    #: swept by restore_stale_backups() for crash recovery. Empty for "flag".
    paths: Callable[[], list[Path]]


# ---------------------------------------------------------------------------
# Continue (`cn`) — config.yaml models[].apiBase via the `--config` flag.
#
# Verified 2026-09-06: docs.continue.dev/cli/configuration confirms `--config
# <path>` is the highest-priority, session-only override — cn never reads or
# writes anything else when it's passed. The model-provider docs page is
# client-rendered (a plain fetch returns an empty shell), so the field names
# — top-level name/version/models, and each model's
# name/provider/model/apiBase/apiKey — come from the zod schema in
# continuedev/continue's packages/config-yaml/src/schemas/{models,index}.ts
# instead of a guess.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _continue_apply(upstream: str, base: str) -> Iterator[list[str]]:
    family = _family(upstream)
    provider, model, key_var = (
        ("anthropic", "claude-opus-4-8", "ANTHROPIC_API_KEY")
        if family == "anthropic"
        else ("openai", "gpt-5.2", "OPENAI_API_KEY")
    )
    api_key = os.environ.get(key_var, "")
    if not api_key:
        print(
            f"  ⚠ Continue: {key_var} is not set — skipping the config "
            "override (never inventing a credential)."
        )
        yield []
        return
    text = (
        "name: distil-wrap\n"
        'version: "1.0.0"\n'
        "models:\n"
        "  - name: distil\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n"
        f"    apiBase: {base}\n"
        f"    apiKey: {api_key}\n"
        "    roles: [chat, edit, apply, autocomplete]\n"
    )
    # 0600 by default (tempfile) in the OS temp dir, not the project tree —
    # the only place the resolved API key touches disk, gone in `finally`.
    fd, tmp_name = tempfile.mkstemp(prefix="distil-continue-", suffix=".yaml")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  → generated {tmp_path} → cn --config {tmp_path}")
        yield ["--config", str(tmp_path)]
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Factory Droid (`droid`) — customModels[] merged into settings.local.json.
#
# Verified 2026-09-06: docs.factory.ai/droid-cli/settings ("Local overrides" —
# settings.local.json merges on top of settings.json, user- or project-level)
# and docs.factory.ai/model-independence/byok (customModels schema: model,
# displayName, baseUrl, apiKey, provider). Only "generic-chat-completion-api"
# is confirmed as a provider-type string from the docs' own example; the
# "Understanding providers" section that would list an Anthropic-native type
# didn't render over a plain fetch, so this preset only fires for an
# OpenAI-shaped --upstream rather than guessing the Anthropic one.
# ---------------------------------------------------------------------------


def _factory_settings_path() -> Path:
    return Path.home() / ".factory" / "settings.local.json"


@contextlib.contextmanager
def _droid_apply(upstream: str, base: str) -> Iterator[list[str]]:
    if _family(upstream) != "openai":
        print(
            "  ⚠ Factory Droid: only the OpenAI-compatible provider type "
            '("generic-chat-completion-api") is verified for BYOK custom '
            "models — pass --upstream https://api.openai.com to route "
            "through distil (docs.factory.ai/model-independence/byok)."
        )
        yield []
        return
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "  ⚠ Factory Droid: OPENAI_API_KEY is not set — skipping the "
            "customModels overlay (never inventing a credential)."
        )
        yield []
        return

    path = _factory_settings_path()
    existed = path.exists()
    original = path.read_bytes() if existed else None
    try:
        doc = json.loads(original) if original else {}
        if not isinstance(doc, dict):
            raise ValueError("settings.local.json root is not an object")
    except (json.JSONDecodeError, ValueError):
        # ponytail: an unparseable settings.local.json is treated as empty
        # rather than aborting the wrap — the ORIGINAL bytes are still
        # backed up and restored untouched on exit either way.
        doc = {}
    models = [
        m
        for m in doc.get("customModels", [])
        if not (isinstance(m, dict) and m.get("model") == "distil")
    ]
    models.append(
        {
            "model": "distil",
            "displayName": "Distil (compressed)",
            "baseUrl": base,
            "apiKey": api_key,
            "provider": "generic-chat-completion-api",
        }
    )
    doc["customModels"] = models

    backup = _backup_path(path)
    sentinel = _created_marker(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        backup.write_bytes(original)  # type: ignore[arg-type]
    else:
        sentinel.touch()
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(
        f'  → merged a "distil" entry into {path}\'s customModels — '
        "select it in droid's model picker if it isn't already active"
    )
    try:
        yield []
    finally:
        if existed:
            path.write_bytes(original)  # type: ignore[arg-type]
        else:
            path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
        sentinel.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Oh My Pi (`omp`) — providers.<id>.baseUrl in ~/.omp/agent/models.yml.
#
# Verified 2026-09-06 against the omp project README (custom providers speak
# openai-completions, openai-responses, ..., or anthropic-messages; the
# providers.<id>.{baseUrl,api,apiKey,models[]} shape comes from the README's
# own example). No --config flag and no override layer are documented, so
# this is the one preset that edits the real file: byte-for-byte backup,
# a marker-fenced `distil` provider block spliced in right after the
# `providers:` line (2-space indent, matching the README's own convention —
# not detected from the file, see the ponytail note below), and an
# exact-bytes restore in `finally`, repeated by restore_stale_backups() if a
# crash skips it.
# ---------------------------------------------------------------------------

_PROVIDERS_LINE_RE = re.compile(r"^providers:[ \t]*\r?\n", re.MULTILINE)


def _omp_models_path() -> Path:
    return Path.home() / ".omp" / "agent" / "models.yml"


def _omp_fenced_block(family: str, base: str, api_key: str) -> str:
    api = "anthropic-messages" if family == "anthropic" else "openai-completions"
    model_id = "claude-opus-4-8" if family == "anthropic" else "gpt-5.2"
    return (
        f"{_MARKER_BEGIN}\n"
        "  distil:\n"
        f"    baseUrl: {base}\n"
        f"    api: {api}\n"
        f"    apiKey: {api_key}\n"
        "    models:\n"
        f"      - id: {model_id}\n"
        f"        name: Distil ({family})\n"
        f"{_MARKER_END}\n"
    )


def _omp_patch(text: str | None, fenced: str) -> str:
    if text is None:
        return f"providers:\n{fenced}"
    m = _PROVIDERS_LINE_RE.search(text)
    if m is None:
        # ponytail: appends a new top-level `providers:` section at EOF
        # rather than parsing the whole document to find "the right place" —
        # correct as long as `providers:` isn't already nested elsewhere
        # under a different key (checked above) and it's fine to be the
        # last section; good enough for a first cut, upgrade to a real YAML
        # parse if a user's models.yml ever violates that.
        sep = "" if not text or text.endswith("\n") else "\n"
        return f"{text}{sep}providers:\n{fenced}"
    return text[: m.end()] + fenced + text[m.end() :]


@contextlib.contextmanager
def _omp_apply(upstream: str, base: str) -> Iterator[list[str]]:
    family = _family(upstream)
    key_var = "ANTHROPIC_API_KEY" if family == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_var, "")
    if not api_key:
        print(
            f"  ⚠ Oh My Pi: {key_var} is not set — skipping the models.yml "
            "provider block (never inventing a credential)."
        )
        yield []
        return

    path = _omp_models_path()
    existed = path.exists()
    original = path.read_bytes() if existed else None
    fenced = _omp_fenced_block(family, base, api_key)
    new_text = _omp_patch(original.decode("utf-8") if original else None, fenced)

    backup = _backup_path(path)
    sentinel = _created_marker(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        backup.write_bytes(original)  # type: ignore[arg-type]
    else:
        sentinel.touch()
    path.write_text(new_text, encoding="utf-8")
    print(
        f'  → wrote provider "distil" into {path} — select it with '
        "`omp models set distil/<model>` if it isn't already active"
    )
    try:
        yield []
    finally:
        if existed:
            path.write_bytes(original)  # type: ignore[arg-type]
        else:
            path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
        sentinel.unlink(missing_ok=True)


CONFIG_PRESETS: dict[str, ConfigPreset] = {
    "cn": ConfigPreset(
        label="Continue",
        strategy="flag",
        doc_url="https://docs.continue.dev/cli/configuration",
        verified="2026-09-06",
        apply=_continue_apply,
        paths=lambda: [],  # temp file only — nothing stable to sweep
    ),
    "droid": ConfigPreset(
        label="Factory Droid",
        strategy="overlay",
        doc_url="https://docs.factory.ai/model-independence/byok",
        verified="2026-09-06",
        apply=_droid_apply,
        paths=lambda: [_factory_settings_path()],
    ),
    "omp": ConfigPreset(
        label="Oh My Pi",
        strategy="patch",
        doc_url="https://github.com/omnara-ai/omp",
        verified="2026-09-06",
        apply=_omp_apply,
        paths=lambda: [_omp_models_path()],
    ),
}


def restore_stale_backups() -> None:
    """Crash recovery: call once at the top of every `distil wrap`. If a
    prior wrap died before its `finally` ran (SIGKILL, power loss, `kill -9`),
    the backup or the created-sentinel is still sitting next to the real
    file — put it back (or delete what we created) so the tool sees the same
    state it started in. Fail-open and silent on success; this must never
    block a wrap that has nothing to do with a previous one."""
    for preset in CONFIG_PRESETS.values():
        for path in preset.paths():
            backup = _backup_path(path)
            sentinel = _created_marker(path)
            try:
                if backup.exists():
                    backup.replace(path)
                    print(f"distil wrap: restored {path} from a previous session's backup")
                elif sentinel.exists():
                    path.unlink(missing_ok=True)
                    sentinel.unlink()
                    print(f"distil wrap: removed {path} left behind by a previous session")
            except OSError:
                pass  # best-effort; never block a new wrap over old cleanup

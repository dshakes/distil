"""Config-file wrap targets: tools whose ONLY routing knob is a config file,
not an environment variable, so they can't use ``onboard.AGENT_PRESETS``
(``distil wrap``'s env-var mechanism) at all.

Each preset is doc-cited (URL + verification date). Where a docs page turned
out to be client-rendered and returned nothing over a plain fetch, the field
names come from the tool's own schema/source on GitHub instead of a guess —
that's called out per preset below. A tool whose config shape could NOT be
pinned down this way (Amp — its only verified base-URL setting, `amp.url`,
belongs to the VS Code extension, not the standalone CLI `wrap` launches,
whose own settings reference has no such key; Mistral Vibe; and OpenClaw — a
persistent multi-channel gateway, not a per-session CLI) is deliberately not
here; see docs/IDE-AGENTS.md for why.

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

Every write that can put a credential on disk — the real config file, and
its ``.distil-backup`` copy of what was there before — goes through
``_atomic_write_secure``: 0600-at-creation (POSIX) plus a same-directory
temp file swapped in with ``os.replace``, so a write that fails partway
(disk full, permission denied) leaves the original bytes untouched instead
of a half-written file, and the credential is never briefly world-readable
at the umask default. Process death (SIGKILL, power loss) mid-session is a
separate case, covered by the existing ``finally``/signal-handling in
``proxy.wrap_run`` for anything short of SIGKILL, and by
``restore_stale_backups()`` at the top of the next ``distil wrap`` for that.

Two `distil wrap` sessions can legitimately be patching the same config at
once (a second wrap of any tool calls ``restore_stale_backups()``
unconditionally). ``_claim_session``/``_release_session`` track that with a
per-target-path registry of pid-tagged entries, so a live sibling's
backup/sentinel is never mistaken for crash leftovers, and whichever session
turns out to be the LAST one running does the real restore — regardless of
exit order. Every claim-and-write and release-and-restore transition holds
``_session_lock(path)`` for its whole extent, so a release's "no live
siblings remain" check and a fresh sibling's claim can never straddle each
other — the failure mode a lock-free registry check alone still allows.

The registry says who is running; it does NOT say whether the shared backup
exists. ``_own_config`` decides that from the backup file itself, in the
same critical section as the claim and strictly before the patch, so no
crash can leave a config patched with nothing to restore it from. Patching
is idempotent for every strategy that touches a real file — droid and crush
key their entry (drop-and-re-append / dict assignment), omp strips its own
marker fence before splicing a new one — so re-patching an already-patched
config replaces the ``distil`` entry instead of stacking a second one; cn
never re-patches anything, since it renders a fresh self-contained temp
config per invocation and leaves the user's own files alone.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distil import _filelock

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


def _atomic_write_secure(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so a mid-write failure can never corrupt it,
    two concurrent writers can never collide, and a credential embedded in
    ``data`` is never briefly world-readable.

    A same-directory temp file (so ``os.replace`` is a same-filesystem atomic
    rename) with a name UNIQUE per call — ``tempfile.mkstemp`` rather than a
    static ``<name>.distil-tmp`` — is created 0600 *at* creation (the mode
    ``mkstemp`` always uses, POSIX or not) — not chmod-after, the same fix
    gateway_keys.py's ``_save_locked`` and atrest.py's ``_load_key`` already
    apply, both of which measured a real window at the process umask (0o644)
    between an ordinary write and a trailing chmod. A static temp name let
    two concurrent writers to the same target open/truncate the SAME temp
    file, so one's ``os.replace`` could publish the other's half-written
    bytes; a unique name per call makes that structurally impossible. The
    temp is fsync'd and swapped into place with one ``os.replace``. If
    anything raises before that replace (disk full, permission denied, an
    unparseable existing file upstream of this call), ``path`` is left
    completely untouched; the temp file never survives this function either
    way, success or failure.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".distil-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op once os.replace has moved it; else cleanup


# ---------------------------------------------------------------------------
# Per-session ownership. `_backup_path`/`_created_marker` name the ONE shared
# copy of a target's true pre-wrap bytes — there is only ever one "before any
# of us touched it" to restore to, no matter how many `distil wrap` sessions
# are concurrently patching the same file. What was missing was knowing
# whether that shared copy is still spoken for: two overlapping sessions on
# the same config used to collide on it directly (restore_stale_backups()
# couldn't tell a live sibling's bookkeeping from a dead one's), so this adds
# a per-session registry entry (pid + a monotonic token, so two sessions
# never share a registry filename) that both restore_stale_backups() and each
# session's own exit consult before touching the shared backup/sentinel.
# ---------------------------------------------------------------------------


def _session_id() -> str:
    # ponytail: pid + a monotonic timestamp, not a random uuid — stdlib, and
    # enough to keep two registrations from the SAME pid (a fast-recycled pid,
    # or a nested wrap) from landing on the same filename. Liveness itself is
    # checked on the pid alone.
    return f"{os.getpid()}.{time.monotonic_ns()}"


def _configure_kernel32(kernel32: Any) -> Any:
    """Pin explicit signatures on the three kernel32 calls below and return
    the same object.

    ctypes defaults every foreign function's ``restype`` to ``c_int``, which
    TRUNCATES a 64-bit ``HANDLE`` on Win64: ``OpenProcess`` hands back a
    mangled handle, ``GetExitCodeProcess``/``CloseHandle`` then fail with
    ERROR_INVALID_HANDLE, the fail-safe below fires, and EVERY pid reads
    alive — crash recovery silently dead on the one platform this whole
    registry exists for.

    Split out from ``_pid_is_alive`` so the configuration itself is
    assertable: the fake-kernel32 test can prove the branching logic but
    never this, because a Python stand-in has no ABI to get wrong."""
    import ctypes

    # Plain ctypes types rather than ctypes.wintypes — wintypes doesn't
    # import at all off Windows, and these are its DWORD/BOOL/HANDLE.
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32)
    kernel32.GetExitCodeProcess.restype = ctypes.c_int32
    kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel32.CloseHandle.restype = ctypes.c_int32
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    return kernel32


def _pid_is_alive(pid: int) -> bool:
    """Best-effort, stdlib-only, psutil-free liveness check. Fail-safe:
    anything short of a confirmed "no such process" counts as alive, since
    wrongly reclaiming a still-running session's config (restored or deleted
    out from under it) is far worse than leaving a truly-dead session's
    bookkeeping around a little longer.

    Called from ``restore_stale_backups()`` at the top of EVERY
    ``distil wrap``, so it must not raise for any reason: an exception here
    doesn't degrade crash recovery, it stops the CLI from starting at all."""
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True  # exists but not signalable by us — still alive
        return True
    # Windows: OpenProcess + GetExitCodeProcess via ctypes rather than a
    # pywin32/psutil dependency. A non-null handle alone does NOT mean
    # alive — Windows can still open a handle to a pid that already exited —
    # so the real test is GetExitCodeProcess() == STILL_ACTIVE.
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_INVALID_PARAMETER = 87
        kernel32 = _configure_kernel32(ctypes.windll.kernel32)  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # No such pid reads ERROR_INVALID_PARAMETER — confirmed dead.
            # Anything else (e.g. access denied on a pid we don't own) is
            # fail-safe: we couldn't disprove it's alive. Only pure Python
            # runs between OpenProcess and this read, so the thread's last
            # error is still the one OpenProcess set.
            return kernel32.GetLastError() != ERROR_INVALID_PARAMETER
        try:
            exit_code = ctypes.c_uint32()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True  # couldn't ask — fail-safe, assume alive
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001 — see the docstring: raising here breaks `distil wrap`
        return True  # couldn't ask — fail-safe, assume alive


def _registry_dir(path: Path) -> Path:
    return path.with_name(path.name + ".distil-sessions")


def _prune_dead_registrants(registry_dir: Path) -> None:
    try:
        entries = list(registry_dir.iterdir())
    except FileNotFoundError:
        return
    for entry in entries:
        pid_str = entry.name.split(".", 1)[0]
        if not pid_str.isdigit() or not _pid_is_alive(int(pid_str)):
            entry.unlink(missing_ok=True)


def _session_lock(path: Path) -> contextlib.AbstractContextManager[None]:
    """Cooperative cross-process lock over the WHOLE claim-and-write /
    release-and-restore transition for ``path`` — not just the registry
    dir's own mkdir, which is atomic on its own but not enough. Without
    this, a release deciding "no live siblings remain, I must restore" and a
    fresh claim ("register me, I'm about to write my own config") can
    interleave: release sees an empty registry, a new session claims and
    writes its fresh config in that gap, and release's restore then
    clobbers it. Callers hold this for their entire claim+write or
    release+restore block, so the two transitions can never straddle each
    other, regardless of which starts first.

    ``distil._filelock`` is the platform half (``fcntl.flock`` on POSIX,
    ``msvcrt.locking`` on Windows, fail-open if the platform call itself
    errors); the sidecar it locks is named after the session registry, so
    the wrapped config file is never touched by the locking itself.
    """
    return _filelock.locked(_registry_dir(path))


def _claim_session(path: Path) -> tuple[Path, str]:
    """Register this process as one of possibly several concurrent sessions
    patching ``path``, dropping any provably-dead registrant first so a
    crashed session can't keep a path claimed forever. Callers hold
    ``_session_lock(path)`` across this and everything they do with the
    result. Returns ``(registry_dir, my_id)``.

    There is deliberately no "am I the first claimant?" answer here any
    more: which session writes the shared backup is decided by whether the
    backup *file* exists — see ``_own_config`` for why registry emptiness
    was the wrong question."""
    registry_dir = _registry_dir(path)
    registry_dir.mkdir(parents=True, exist_ok=True)
    _prune_dead_registrants(registry_dir)
    my_id = _session_id()
    (registry_dir / my_id).touch()
    return registry_dir, my_id


def _release_session(registry_dir: Path, my_id: str) -> bool:
    """Unregister and report whether this was the last live session for the
    target path — the one that must perform the real content restore. A
    sibling whose death can't be proven is left registered, so the restore is
    skipped rather than pulled out from under a session that is still
    running (whoever the LAST surviving session turns out to be does the
    restore, regardless of exit order)."""
    (registry_dir / my_id).unlink(missing_ok=True)
    _prune_dead_registrants(registry_dir)
    try:
        last = not any(registry_dir.iterdir())
    except FileNotFoundError:
        last = True
    if last:
        with contextlib.suppress(OSError):
            registry_dir.rmdir()
    return last


@contextlib.contextmanager
def _own_config(path: Path, render: Callable[[bytes | None], bytes]) -> Iterator[None]:
    """Claim ``path`` for this session, make sure its true pre-wrap bytes stay
    recoverable, write ``render(current_bytes)``, and restore when the last
    live session on ``path`` exits. Every preset that touches a real config
    file goes through here, so the invariant below is stated once.

    The claim, the backup and the patch are ONE critical section under
    ``_session_lock(path)``, in that order, and "does the backup still need
    creating?" is answered by *whether the backup file exists* — never by
    "was the session registry empty?". Deciding it from registry emptiness
    left a window a SIGKILL could land in: a session that registered and
    died before writing its backup left a registry holding one dead entry,
    so the next ``distil wrap`` read ``is_first=False``, skipped the backup,
    and patched the real config anyway — the injected ``distil`` block was
    then permanent, with nothing left on disk that could undo it. Ordered
    this way there is no such window: nothing is patched until the backup
    (or the created-sentinel that stands in for one, when there was no file
    to back up) is on disk, so a crash at any point either leaves the config
    untouched or leaves ``restore_stale_backups()`` exactly what it needs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup_path(path)
    sentinel = _created_marker(path)
    with _session_lock(path):
        registry_dir, my_id = _claim_session(path)
        current = path.read_bytes() if path.exists() else None
        if not backup.exists() and not sentinel.exists():
            if current is None:
                sentinel.touch()
            else:
                _atomic_write_secure(backup, current)
        _atomic_write_secure(path, render(current))
    try:
        yield
    finally:
        with _session_lock(path):
            if _release_session(registry_dir, my_id):
                if backup.exists():
                    _atomic_write_secure(path, backup.read_bytes())
                    backup.unlink(missing_ok=True)
                elif sentinel.exists():
                    path.unlink(missing_ok=True)
                    sentinel.unlink(missing_ok=True)


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
    #: Upstream to use when the CLI got no explicit --upstream and no
    #: AGENT_PRESETS entry supplied one either (config-file presets are a
    #: separate registry from AGENT_PRESETS, so they'd otherwise silently
    #: fall through to cmd_wrap's hardcoded Anthropic default). None means
    #: "whatever family cmd_wrap already picked is fine" — true for
    #: omp/crush/cn, which read the family off --upstream instead of
    #: requiring one; only droid hard-requires an OpenAI-shaped upstream.
    default_upstream: str | None = None


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

    def _render(current: bytes | None) -> bytes:
        try:
            doc = json.loads(current) if current else {}
            if not isinstance(doc, dict):
                raise ValueError("settings.local.json root is not an object")
        except (json.JSONDecodeError, ValueError):
            # ponytail: an unparseable settings.local.json is treated as empty
            # rather than aborting the wrap — the ORIGINAL bytes are still
            # backed up and restored untouched on exit either way.
            doc = {}
        # Idempotent: any customModels entry we (or a sibling session) already
        # wrote is dropped before ours is appended, so patching an
        # already-patched overlay replaces the entry instead of stacking a
        # second one that would outlive the restore.
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
        return (json.dumps(doc, indent=2) + "\n").encode("utf-8")

    with _own_config(path, _render):
        print(
            f'  → merged a "distil" entry into {path}\'s customModels — '
            "select it in droid's model picker if it isn't already active"
        )
        yield []


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

#: A whole block distil spliced in previously, fence lines included.
_FENCED_BLOCK_RE = re.compile(
    re.escape(_MARKER_BEGIN) + r".*?" + re.escape(_MARKER_END) + r"[ \t]*\r?\n?",
    re.DOTALL,
)


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
    # Idempotent: strip a block we spliced in earlier (a sibling session's, or
    # one a crash left behind) before splicing the new one. Without this,
    # patching an already-patched models.yml appended a SECOND `distil`
    # provider — duplicate YAML keys, and one block too many for any single
    # restore to undo.
    text = _FENCED_BLOCK_RE.sub("", text)
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
    fenced = _omp_fenced_block(family, base, api_key)

    def _render(current: bytes | None) -> bytes:
        text = current.decode("utf-8") if current is not None else None
        return _omp_patch(text, fenced).encode("utf-8")

    with _own_config(path, _render):
        print(
            f'  → wrote provider "distil" into {path} — select it with '
            "`omp models set distil/<model>` if it isn't already active"
        )
        yield []


# ---------------------------------------------------------------------------
# Crush (`crush`) — providers.<id>.base_url in the legacy ~/.config/crush/
# crush.json.
#
# Verified 2026-09-07 against charmbracelet/crush's own schema.json
# ($defs.ProviderConfig: id/name/type/base_url/api_key/models, `type` enum
# includes "anthropic" and "openai") and docs/config/README.md, both on the
# `main` branch. Crush's *current* config format is a Bash script
# (`crushrc`, run with shell privileges); `crush.json` is explicitly the
# deprecated predecessor — "we plan to support it for the foreseeable
# future" — read from the same directory tier, lower priority than crushrc,
# so a `distil` provider entry there only takes effect where crushrc doesn't
# already define one with the same id. No `models` array is written: per
# ProviderConfig, `models` is optional and `discover_models` defaults to
# true, and inventing per-model cost/context-window figures Crush's schema
# would otherwise require is exactly the kind of guessed value this file
# exists to avoid.
# ---------------------------------------------------------------------------


def _crush_config_path() -> Path:
    return Path.home() / ".config" / "crush" / "crush.json"


@contextlib.contextmanager
def _crush_apply(upstream: str, base: str) -> Iterator[list[str]]:
    family = _family(upstream)
    key_var = "ANTHROPIC_API_KEY" if family == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_var, "")
    if not api_key:
        print(
            f"  ⚠ Crush: {key_var} is not set — skipping the crush.json "
            "provider entry (never inventing a credential)."
        )
        yield []
        return

    path = _crush_config_path()

    def _render(current: bytes | None) -> bytes:
        try:
            doc = json.loads(current) if current else {}
            if not isinstance(doc, dict):
                raise ValueError("crush.json root is not an object")
        except (json.JSONDecodeError, ValueError):
            # ponytail: an unparseable crush.json is treated as empty rather
            # than aborting the wrap — the ORIGINAL bytes are still backed up
            # and restored untouched on exit either way.
            doc = {}
        providers = doc.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        # Idempotent: keyed assignment, so re-patching replaces our provider
        # rather than adding a second one.
        providers["distil"] = {
            "id": "distil",
            "name": "Distil (compressed)",
            "type": family,
            "base_url": base,
            "api_key": api_key,
        }
        doc["providers"] = providers
        return (json.dumps(doc, indent=2) + "\n").encode("utf-8")

    with _own_config(path, _render):
        print(
            f'  → wrote provider "distil" into {path} — pick it from Crush\'s '
            "model picker (ctrl+l), or run `model add distil/<id>` in crushrc "
            "if it has no models listed yet"
        )
        yield []


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
        # _droid_apply hard-requires an OpenAI-shaped upstream (only
        # "generic-chat-completion-api" is verified) — without this, a bare
        # `distil wrap -- droid` inherited cmd_wrap's hardcoded Anthropic
        # default and the preset silently injected nothing.
        default_upstream="https://api.openai.com",
    ),
    "omp": ConfigPreset(
        label="Oh My Pi",
        strategy="patch",
        doc_url="https://github.com/omnara-ai/omp",
        verified="2026-09-06",
        apply=_omp_apply,
        paths=lambda: [_omp_models_path()],
    ),
    "crush": ConfigPreset(
        label="Crush",
        strategy="patch",
        doc_url="https://github.com/charmbracelet/crush/blob/main/docs/config/README.md",
        verified="2026-09-07",
        apply=_crush_apply,
        paths=lambda: [_crush_config_path()],
    ),
}


def restore_stale_backups() -> None:
    """Crash recovery: call once at the top of every `distil wrap`. If a
    prior wrap died before its `finally` ran (SIGKILL, power loss, `kill -9`),
    the backup or the created-sentinel is still sitting next to the real
    file — put it back (or delete what we created) so the tool sees the same
    state it started in. Fail-open and silent on success; this must never
    block a wrap that has nothing to do with a previous one.

    A backup/sentinel next to a target isn't proof of a crash by itself — a
    still-running sibling `distil wrap` session on the SAME config keeps its
    own backup/sentinel in place for its whole lifetime (see
    ``_claim_session``/``_release_session``). Before touching either file,
    prune dead entries from that path's session registry and check whether a
    live one remains; if it does, this is a live session, not a crash
    leftover, and must be left completely alone. Fail-safe: if liveness
    can't be disproven for every registrant, treat the path as still owned.

    A registry directory whose registrants are all dead is swept even when
    there is no backup and no sentinel beside it: that is the shape a crash
    between the claim and the backup write leaves (nothing was patched yet),
    and the shape a restore that already consumed its backup leaves. Either
    way the leftover directory is bookkeeping for a session that no longer
    exists, and the config file itself is left alone.

    Every path is swept inside its own catch-all: this is the first thing
    `distil wrap` does, so a registry left in any shape at all — by an older
    distil, by a half-finished manual cleanup, by a filesystem that answers
    an unexpected error — must not be able to stop the wrap from running.
    """
    for preset in CONFIG_PRESETS.values():
        for path in preset.paths():
            try:
                _restore_one(path)
            except Exception:  # noqa: BLE001 — old cleanup must never block a new wrap
                pass


def _restore_one(path: Path) -> None:
    """One target path's share of ``restore_stale_backups``; see there."""
    backup = _backup_path(path)
    sentinel = _created_marker(path)
    registry_dir = _registry_dir(path)
    if not backup.exists() and not sentinel.exists() and not registry_dir.exists():
        return
    with _session_lock(path):
        _prune_dead_registrants(registry_dir)
        try:
            still_live = any(registry_dir.iterdir())
        except FileNotFoundError:
            still_live = False
        if still_live:
            return  # a live sibling session owns this path — not ours to touch
        try:
            if backup.exists():
                backup.replace(path)
                print(f"distil wrap: restored {path} from a previous session's backup")
            elif sentinel.exists():
                path.unlink(missing_ok=True)
                sentinel.unlink()
                print(f"distil wrap: removed {path} left behind by a previous session")
            with contextlib.suppress(OSError):
                registry_dir.rmdir()
        except OSError:
            pass  # best-effort; never block a new wrap over old cleanup

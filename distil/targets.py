"""One catalogue of every agent distil can reach — and every one it cannot.

The same three facts (what routes it, which wire shape the proxy must speak,
and where that was verified from) were previously stated in five places: the
two preset registries, ``cli._IDE_NOT_WRAPPABLE``, and prose tables in
README.md, docs/IDE-AGENTS.md and docs/integrations.html. They drifted — the
docs still called Warp "no published base-URL override at all" a release after
Warp shipped one, and still called Cline "not a CLI" after Cline shipped one.

So the routing contracts stay where they are enforced (``onboard.AGENT_PRESETS``
for env-var targets, ``config_wrap.CONFIG_PRESETS`` for config-file ones) and
this module joins them with the doc metadata into ``catalog()``, which
``distil wrap --list`` prints and ``scripts/build_agent_tables.py`` renders the
docs tables from. ``tests/test_wrap_targets.py`` fails if a doc drifts from it.

``UNREACHABLE`` is the other half and is the part worth reading: a tool whose
routing knob could NOT be verified against its own primary documentation is
listed here with what was checked and when, never given a preset. A preset
built on a guessed variable reports success, starts a proxy, and routes zero
traffic — the one failure mode where distil lies to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Provider wire shapes the proxy speaks, as dispatched in ``proxy.py``:
#: ``/v1/messages``, ``/v1/chat/completions``, ``/v1/responses``, and Gemini's
#: ``:generateContent``. A target's shape decides which adapter compresses it.
ANTHROPIC = "Anthropic Messages"
OPENAI_CHAT = "OpenAI Chat Completions"
GEMINI = "Gemini generateContent"
EITHER = "Anthropic Messages or OpenAI Chat Completions"


@dataclass(frozen=True)
class Target:
    """One agent, and how (or whether) distil routes it."""

    key: str
    label: str
    #: "env" (a base-URL environment variable), "config" (a config file distil
    #: manages for the session), or "proxy" (not wrappable — point the tool's
    #: own setting at a running ``distil proxy`` / ``distil default``).
    mechanism: str
    shape: str
    #: The verified knob: an env var name, a config key, or an editor setting.
    knob: str
    doc_url: str
    verified: str  # YYYY-MM-DD the contract above was read from doc_url
    note: str = ""
    #: Other argv[0] names people type for this target (``wrap`` warns on them).
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def wrappable(self) -> bool:
        return self.mechanism != "proxy"


#: Targets `distil wrap` cannot reach, each with the primary source that was
#: read and the date. Three distinct reasons, all verified rather than assumed:
#:
#:   no knob        the tool publishes no base-URL override at all (Cursor CLI,
#:                  Amp, auggie, Antigravity, Tabnine). Checked in its own docs.
#:   not local      a base-URL override exists but structurally cannot point at
#:                  127.0.0.1 (Warp routes through its own backend and rejects
#:                  private addresses; Cody's override is a setting on the
#:                  Sourcegraph instance, not on the client).
#:   no session    the knob is real and verified, but nothing can scope a change
#:                  to one wrap: it is held in an editor's secret storage (Roo),
#:                  belongs to a daemon shared with other channels and users
#:                  (OpenClaw), has no process to launch at all (ZCode), or
#:                  needs the user to pick the new provider by hand afterwards
#:                  (Zed). Reached via `distil default` / `distil proxy`.
#:
#: "The settings file is global" is NOT on that list and never was a reason on
#: its own — `config_wrap` claims, backs up, patches and restores global files
#: for Crush, Oh My Pi, Factory Droid and Cline already. What IS disqualifying
#: is a global file some *other* file outranks for the directory the wrap runs
#: in: patching it reports success and routes nothing. Kilo Code is the worked
#: example — a project-local ./kilo.json shadows the global one, so it moved to
#: the higher-precedence KILO_CONFIG_CONTENT instead of being declined.
UNREACHABLE: tuple[Target, ...] = (
    Target(
        # Cursor's CLI installs its binary as plain `agent`
        # (cursor.com/docs/cli/overview, verified 2026-09-16), but distil
        # deliberately does NOT key or alias this entry on that name: `agent`
        # is what half the shell scripts and local wrappers in the world are
        # called, and a false "Cursor CLI routes nothing" on someone's own
        # `agent` is worse than staying quiet for the real one. The names
        # matched here are the unambiguous ones people actually type.
        key="cursor",
        label="Cursor CLI",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="— (HTTP_PROXY/HTTPS_PROXY only)",
        doc_url="https://cursor.com/docs/cli/reference/configuration",
        verified="2026-09-16",
        note="cli-config.json publishes no base-URL field; the only network knob "
        "is a whole-process HTTP proxy, not a per-request base URL. Its binary is "
        "`agent`, a name too generic for distil to claim",
        aliases=("cursor-agent",),
    ),
    Target(
        key="continue",
        label="Continue (VS Code extension)",
        mechanism="proxy",
        shape=EITHER,
        knob="~/.continue/config.yaml → models[].apiBase",
        doc_url="https://docs.continue.dev/reference",
        verified="2026-09-16",
        note="apiBase 'can be used to override the default API base', but the extension "
        "is started by the editor — no argv to wrap, and the file is editor-wide rather "
        "than per-session. The Continue CLI is a different tool and `distil wrap -- cn` "
        "does reach it",
    ),
    Target(
        key="roo",
        label="Roo Code",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob='API Provider → "OpenAI Compatible" → Base URL',
        doc_url="https://docs.roocode.com/features/api-configuration-profiles",
        verified="2026-09-16",
        note="a VS Code extension with no CLI, whose configuration profiles live in "
        "VS Code's own Secret Storage ('stored securely in VSCode's Secret Storage and "
        "never exposed in plain text') — there is no config file to patch",
    ),
    Target(
        key="windsurf",
        label="Windsurf",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="Settings → Cascade → custom endpoint",
        doc_url="https://docs.windsurf.com/windsurf/models",
        verified="2026-09-16",
        note="BYOK accepts a provider API KEY only (Claude 4 family), with no "
        "endpoint field in the documented flow",
    ),
    Target(
        key="zed",
        label="Zed agent",
        mechanism="proxy",
        shape=EITHER,
        knob="settings.json → language_models.anthropic_compatible.<name>.api_url "
        "(or openai_compatible)",
        doc_url="https://zed.dev/docs/ai/use-api-access",
        verified="2026-09-16",
        note="no single key redirects it: the BUILT-IN anthropic provider documents "
        "only available_models, never an api_url, so routing means ADDING an "
        "anthropic_compatible provider the user must then pick by hand in the model "
        "dropdown — and Zed's own Agent Settings page writes settings.json while it "
        "runs, so a session-scoped patch would be racing the editor for the file",
    ),
    Target(
        key="amp",
        label="Amp",
        mechanism="proxy",
        shape=ANTHROPIC,
        knob="— (HTTP_PROXY/HTTPS_PROXY only)",
        doc_url="https://ampcode.com/docs/markdown/cli/settings",
        verified="2026-09-16",
        note="re-checked: the CLI settings reference still has no base-URL key; "
        "amp.url belongs to the VS Code extension, not the CLI",
    ),
    Target(
        key="openclaw",
        label="OpenClaw",
        mechanism="proxy",
        shape=EITHER,
        knob="~/.openclaw/openclaw.json → models.providers.<id>.baseUrl",
        doc_url="https://docs.openclaw.ai/cli",
        verified="2026-09-16",
        note="the knob is verified, but OpenClaw's own README puts the model "
        "connection in a Gateway the CLI merely 'connects to' — one local control "
        "plane shared with Discord/WhatsApp/Slack channels and, on a team install, "
        "other people. A session-scoped patch would reconfigure a daemon serving "
        "them, and restore it mid-flight when one terminal exits",
    ),
    Target(
        key="warp",
        label="Warp",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="Settings → custom inference endpoint (public HTTPS URL only)",
        doc_url="https://docs.warp.dev/agents/inference/custom-inference-endpoint",
        verified="2026-09-16",
        note="Warp DOES publish an endpoint override now (the older 'no override at "
        "all' note was stale) — but the agent harness runs on Warp's servers and the "
        "docs reject localhost and private addresses, so a local distil proxy cannot "
        "be the target",
    ),
    Target(
        key="cody",
        label="Sourcegraph Cody",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="site config → modelConfiguration.providerOverrides (server-side)",
        doc_url="https://sourcegraph.com/docs/cody/enterprise/model-configuration",
        verified="2026-09-16",
        note="the override is an admin setting on the Sourcegraph instance, not on "
        "the client — a distil gateway in front of that instance is the fit, not wrap",
    ),
    Target(
        key="auggie",
        label="Augment (auggie)",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="—",
        doc_url="https://docs.augmentcode.com/cli/setup-auggie/authentication",
        verified="2026-09-16",
        note="AUGMENT_SESSION_AUTH carries the session token; no base-URL variable "
        "or config key is documented",
        aliases=("augment",),
    ),
    Target(
        key="antigravity",
        label="Google Antigravity",
        mechanism="proxy",
        shape=GEMINI,
        knob="—",
        doc_url="https://antigravity.google/docs/models",
        verified="2026-09-16",
        note="models are plan-selected from a fixed list; no BYOK and no endpoint "
        "override is documented",
    ),
    Target(
        key="trae",
        label="Trae",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="Settings → Models → custom model",
        doc_url="https://docs.trae.ai/ide/model",
        verified="2026-09-16",
        note="custom models exist, but every docs path returns the same "
        "client-rendered shell over a plain fetch — the config shape could not be "
        "verified against an authoritative source",
    ),
    Target(
        key="junie",
        label="JetBrains Junie",
        mechanism="proxy",
        shape=EITHER,
        knob="model profile → baseUrl",
        doc_url="https://junie.jetbrains.com/docs/",
        verified="2026-09-16",
        note="docs render client-side and return nothing over a plain fetch; the "
        "config shape could not be verified",
    ),
    Target(
        key="tabnine",
        label="Tabnine",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="—",
        doc_url="https://docs.tabnine.com/main/getting-started/tabnine-cli",
        verified="2026-09-16",
        note="clients point at a Tabnine server, not at an LLM endpoint; the CLI "
        "docs publish no model base-URL override",
    ),
    Target(
        key="zcode",
        label="ZCode (z.ai)",
        mechanism="proxy",
        shape=EITHER,
        knob="Settings → Providers → Base URL (Anthropic or OpenAI protocol)",
        doc_url="https://zcode.z.ai/en/docs/configuration",
        verified="2026-09-16",
        note="z.ai's own page calls it an Agentic Development Environment, a desktop "
        "app with no CLI — the Base URL field is verified and does take a local proxy, "
        "but there is no process for wrap to launch or scope a config change to",
    ),
    Target(
        key="code",
        label="VS Code Copilot (extension)",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="—",
        doc_url="https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models",
        verified="2026-09-16",
        note="the extension terminates at GitHub's own service and exposes no "
        "endpoint override; BYOK is a Copilot CLI feature, and that CLI IS wrapped",
    ),
    Target(
        key="cortex",
        label="Snowflake Cortex Code",
        mechanism="proxy",
        shape=OPENAI_CHAT,
        knob="—",
        doc_url="https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-code",
        verified="2026-09-16",
        note="no published base-URL override",
        aliases=("coco",),
    ),
)


def catalog() -> list[Target]:
    """Every target distil knows about, wrappable ones first, then the rest.

    Built by joining the two live preset registries with their doc metadata, so
    a preset added without metadata (or metadata without a preset) is a hard
    failure here rather than a doc that quietly goes stale.
    """
    from .config_wrap import CONFIG_PRESETS
    from .onboard import AGENT_META, AGENT_PRESETS

    out: list[Target] = []
    for cmd, (env_var, _upstream, label, _extra) in AGENT_PRESETS.items():
        meta = AGENT_META[cmd]
        out.append(
            Target(
                key=cmd,
                label=label,
                mechanism="env",
                shape=meta.shape,
                knob=env_var,
                doc_url=meta.doc_url,
                verified=meta.verified,
                note=meta.note,
            )
        )
    for cmd, preset in CONFIG_PRESETS.items():
        out.append(
            Target(
                key=cmd,
                label=preset.label,
                mechanism="config",
                shape=preset.shape,
                knob=preset.knob,
                doc_url=preset.doc_url,
                verified=preset.verified,
                note=f"{preset.strategy} strategy",
            )
        )
    out.sort(key=lambda t: (t.mechanism != "env", t.label.lower()))
    return out + sorted(UNREACHABLE, key=lambda t: t.label.lower())

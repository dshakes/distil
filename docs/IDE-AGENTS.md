# IDE agents — Cursor, Copilot, Cline, Continue, Windsurf

`distil wrap` cannot reach these. That is a structural limit, not a missing
feature, and this page explains the limit and the supported path.

## Why `wrap` does not work here

`distil wrap` works by launching a child process with `ANTHROPIC_BASE_URL` (or the
provider equivalent) set in its environment. That requires two things:

1. a command to launch, and
2. a published environment variable the agent honours.

IDE agents have neither. They run inside the editor's own process tree, started by
the editor rather than by your shell, and none of them documents an environment
variable for redirecting the API endpoint. distil therefore ships **no preset** for
them — see `distil/onboard.py`, where they are excluded by name with this reason.

A preset built on a guessed variable would be worse than nothing: `wrap` would
report success, start a proxy, and route zero traffic. You would see "distil is
on" and a savings counter frozen at zero, with no indication which of the two was
lying.

## The supported path: run a proxy, point the editor at it

Every one of these editors exposes a custom base URL or an OpenAI-compatible
endpoint setting. Point it at a distil proxy.

### 1. Start a persistent proxy

```bash
distil proxy --port 8788 --upstream https://api.anthropic.com
```

Keep it running (a terminal tab, `tmux`, or a login item). For an always-on
service managed for you:

```bash
distil default --always-on
```

That starts the service *and* proves it routes before writing any configuration —
if the probe fails, nothing is written.

> **Do not also use `distil wrap` once always-on is wired.** A base URL pinned in
> `~/.claude/settings.json` outranks the environment `wrap` sets, so the wrap's
> proxy would receive nothing. `distil wrap` now detects this and refuses rather
> than starting into a configuration that cannot work.

### 2. Point the editor at it

| Editor | Where |
|---|---|
| **Cursor** | Settings → Models → *Override OpenAI Base URL* → `http://127.0.0.1:8788/v1` |
| **Cline** | Extension settings → API Provider → *OpenAI Compatible* → Base URL `http://127.0.0.1:8788/v1` |
| **Continue** | `~/.continue/config.json` → the model entry's `apiBase` → `http://127.0.0.1:8788/v1` |
| **Windsurf** | Settings → Cascade → custom endpoint → `http://127.0.0.1:8788/v1` |
| **GitHub Copilot** | Not redirectable. Copilot terminates at GitHub's own service and exposes no endpoint override. distil cannot compress Copilot traffic. |

Paths move between releases; if a menu has been renamed, search the editor's
settings for "base URL" or "OpenAI compatible".

### 3. Verify it is actually routing

Do not trust the absence of an error — that is how the failure above hides.

```bash
distil doctor          # probes the configured base URL and reports what answered
distil dashboard       # watch the counter move as you use the editor
```

If `doctor` reports routing but savings stay at zero, that is usually genuine:
savings come from **large** tool output, and a short editor completion has little
to fold. See the `<1% smaller` note in the status-line docs.

## Copilot, specifically

There is no supported interception point. Copilot does not honour a base URL
override, and working around that would mean intercepting TLS to a service you do
not control — which distil will not ship and you should not run. If Copilot is
your only agent, distil has nothing to offer you today; that is an honest no
rather than a configuration you will fight for an afternoon.

## Editors that are also CLIs

Some tools ship both. Where a CLI exists, prefer it — `wrap` needs no
configuration and gives you the per-session proof ledger:

```bash
distil wrap -- claude      # Claude Code
distil wrap -- codex       # Codex CLI
distil wrap -- gemini      # Gemini CLI
distil wrap -- aider       # aider
distil wrap -- opencode    # OpenCode
distil wrap -- qwen        # Qwen Code
distil wrap -- goose       # goose
```

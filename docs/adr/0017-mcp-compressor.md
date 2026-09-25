# 0017 — `distil mcp`: an MCP proxy that compresses definitions and results, gated by a pre-registered accuracy test

- **Status:** accepted
- **Date:** 2026-09-25
- **Relates to:** `distil/mcpproxy/`, `distil/cli.py` (`cmd_mcp`), `distil/webdash.py`,
  `distil/certificates/mcp.json`, `docs/research/mcp-compressor-protocol.md`, ADR 0013
- **Adds a trust boundary:** distil now sits between an MCP client and third-party MCP
  servers, spawning them and relaying their traffic.

## Context

Every MCP server's tool list is resent on every request. A handful of servers is
thousands of tokens before the agent does anything; the eight reference servers alone
are about 20k (`benchmarks/results/mcp_toolbench/definitions.json`). Atlassian Labs'
`mcp-compressor` attacks this with a proxy that folds each server's tools into one
wrapper description (levels low/medium/high/max) and routes every call through
`<server>_get_tool_schema` + `<server>_invoke_tool`. Two costs come with that design:
the model never sees a real tool signature again (every call is a generic `invoke`),
and every tool costs an extra round trip on every use. It publishes token savings; it
does not publish whether the model still calls the right tool.

ADR 0013 settled Claude Code: its native `defer_loading` tool search defers unused
tools better than any proxy can, and distil stops switching it off. Everything else —
Codex, Cursor, Gemini CLI, opencode, Claude Desktop, Windsurf, custom agents — has no
such mechanism. Tool **results** are uncompressed for every client, Claude Code included.

## Decision

1. **A transparent stdio proxy** (`distil mcp wrap -- <cmd>`, `distil mcp serve --config`)
   that rewrites only `tools/list` and `tools/call` and relays everything else — resources,
   prompts, completion, logging, notifications, and server-to-client requests (`roots/list`,
   sampling) with ids remapped — so it is a drop-in. Stdlib only; threads, not asyncio
   (blocking line I/O on both sides, Windows pipes, and the existing server is sync).
2. **Explicit, named levels**, each strictly no larger than L0 (a surface that would grow
   is served at L0 and the watch view says so):
   - **L0 lossless** — removes only JSON-Schema annotation keywords with no validation
     meaning, and `$schema` only where the draft-07 → 2020-12 change is a no-op. Tested as
     a structural proof on all 78 vendored real schemas plus instance-level equivalence.
   - **L1 summary** — extractive (first sentence + ≤2 constraint sentences, verbatim, never
     generated); full text via `<server>_get_tool_schema`.
   - **L2 lazy — beyond Atlassian:** fetching a schema **unlocks the real tool** (original
     name, real `inputSchema`) for the rest of the session via
     `notifications/tools/list_changed`; `<server>_invoke_tool` is only the fallback for
     clients that never refresh. The unlocked set is append-only and persisted per session,
     and new tools go at the END of the list, so a cached prefix survives an unlock.
   - **L3 adaptive** — L2 plus pins learned from local tool-call counts (names and counts
     only), fixed for a session so pins never move the list mid-session.
   - **R results** (orthogonal) — distil's existing recoverable digest (columnar fold,
     tier-1) with the original in the shared restore store and `<server>_expand` to get it
     back; reject-if-not-smaller; never touches errors, non-text content, results with
     `structuredContent`, or file-read tools an agent must quote back byte-exact
     (`provenance.EXACT_QUOTE_TOOLS` plus the MCP servers' spellings).
3. **Accuracy is certified, not assumed.** `docs/research/mcp-compressor-protocol.md` is
   pre-registered before any live run: paired non-inferiority (TOST + bootstrap) on
   tool-selection and argument exact-match against the uncompressed reference, margin
   `CERT_MARGIN`, α = `BUDGET_DELTA` (the single risk budget), n from a power calculation,
   fixed-sequence L0 → L1 → L2 → L3, one look. `distil mcp bench` is its executable form:
   dry-run by default (mock model through the real proxy, no spend), live only behind
   `--live --budget-usd` with a hard cap.
4. **Default = L0 only** until a live run certifies more. R is opt-in (`--results`)
   like L1–L3 — it was on by default until the 2026-09-25 review — and every
   uncertified level prints its pending certificate status to stderr on every start. `distil/certificates/mcp.json` is edited
   only by a human, from a committed run.
5. **Fail open.** A compression error relays the backend's raw answer; `wrap` execs the
   server directly if the proxy cannot start.
6. **Observability is content-free and local.** `$DISTIL_HOME/mcp/events.jsonl` carries
   tool names, event kinds and token sizes — an allowlist enforced at write time — plus
   a per-server catalog of before/after definitions for the diff view. Tool names never
   reach the census or any telemetry. `distil mcp watch` and webdash `/mcp` render it.
7. **Installer with exact undo** (`distil mcp install <client>`): byte-exact backup,
   0600 atomic writes (`config_wrap._atomic_write_secure`), ownership recorded by SHA-256;
   undo restores byte-for-byte (and the original mode) when the file is untouched, and
   unwraps only distil's entries — keeping the user's later edits — when it is not.
   Codex's TOML is patched line-level and re-parsed; anything that does not round-trip is
   refused, and without `tomllib` (Python < 3.11) Codex is refused outright.

8. **Namespacing, not first-claimant.** One proxy per server (what `install` writes)
   keeps real tool names. When one proxy fronts several servers, every tool, meta tool
   and prompt is `<server>__<name>`; `safe_server` allows only `[A-Za-z0-9-]`, so the first
   `__` always ends the server part and no server can mint a sibling's name. Calls route
   through one name→server table built in config order; `list_changed` rebuilds a
   server's surface without moving anyone's routes. A name that still collides (a tool
   named like its server's own meta tool, a duplicate) is dropped and logged. Resources
   route by the server that listed the URI, prompts by namespace; unknown or ambiguous is
   an error. Server-to-client requests carry `_meta["io.distil/server"]` (elicitation
   messages are prefixed), and merged `instructions` are labelled per server.
9. **Hardening from the 2026-09-25 security review.** No backend I/O under the session
   lock (a server asking `roots/list` mid-`tools/list` stalled the proxy). Backends see
   internal request ids only; `notifications/cancelled` is translated and sent to the one
   server running the call. A `tools/call` that has reached a server is never re-sent by
   the fail-open path. `<server>_expand` returns only originals that server's results
   recorded this session. The id maps are bounded. R's exact-quote exemption also covers
   read/view/open/cat/show/contents-style names and read-only tools described as
   returning file or source content. `install` writes through symlinks, refreshes its
   backup to the pre-image of every write onto a file it did not last write, and restores
   byte-for-byte only from a clean pre-image of the current bytes — otherwise it unwraps
   distil's entries and keeps the user's. The webdash refuses non-loopback `Host`
   headers (DNS rebinding).

10. **Re-review (same day).** Resource routing counts every claim: an exact listing and
    every template whose literal prefix matches the URI are equal claims, so two servers
    claiming one URI in any combination is ambiguous (-32602) — an exact listing no longer
    outranks another server's template. Lists are merged server-side with every page
    followed (bounded); the client is never handed, and never forwards, a cursor. A
    template whose literal prefix is not at least `scheme://` is un-routable and dropped in
    multi-server mode. A meta-tool name that must be truncated carries a stable hash of the
    full server name, and configs whose meta names would still coincide are refused. A
    server may only cancel its own client-bound requests and report progress for its own
    calls; in-flight routes are kept per call, not per client id. `_handles` is bounded;
    `resources/list_changed` forgets owners; sampling requests carry the origin in their
    first visible message text. `install` records the backup's SHA-256 and restores only
    a backup that still matches it, and reads `installs.json` under its lock.
    Resolution always has BOTH exact listings and templates from every server in hand:
    populated-ness is tracked per list type (an empty listing counts as fetched), so the
    order in which a client listed things cannot change who owns a URI.

## Consequences

- distil now spawns and relays third-party MCP servers. It adds no capability they did
  not have: same command, same env, same cwd, stderr inherited. The webdash renders
  server-controlled strings with `textContent` only.
- Streamable-HTTP transport (front or back) is **not** built: stdio covers every installer
  target, and a correct HTTP frontend needs session ids, SSE resumption and auth. Remote
  (`url`) servers are skipped by `serve` and never touched by `install`.
- With prompt caching, the cost estimate suggested L2's extra round trip could cost more
  than the cached tool list it saves. **The live run confirmed it** (below): on the tool
  suite L2 billed $8.79 to raw's $2.17. L2's value is context-window headroom, not
  dollars, at least for short sessions on `claude-haiku-4-5`.

## Live result (2026-09-25, amended protocol, `claude-haiku-4-5`)

Pre-registered run, one look, $21.68 spent of an $85 ceiling that the harness's spend
meter enforced before every call (Amendment 1). Artifact:
`benchmarks/results/mcp_toolbench/live_claude-haiku-4-5.json`.

- **All of L0, L1, L2, L3 and R are certified** for `claude-haiku-4-5`: n=1000 tool
  tasks per level, n=630 result tasks, TOST plus a paired bootstrap at margin 0.02,
  α 0.05. Selection went from 0.944 (raw) to 0.952 / 0.954 / 0.998 / 0.988 (L0–L3).
  Argument match went from 0.912 to 0.921 / 0.954 / 0.963 / 0.954. R answer accuracy
  went from 0.9968 to 0.9952.
- Every compressed level beat raw, but that was **not** a pre-registered hypothesis and
  is not claimed as superiority.
- **Default unchanged: L0.** §10's rule needs a replication model (for R, a certificate
  on both models). None was run this round, so L1–L3 and R stay opt-in, now marked
  certified. Promoting L3 or R is a maintainer decision. It should wait for the
  replication model and for L2's cost to be measured in long sessions, because a fresh
  L3 install starts as L2.
- Scope of a certificate: one model, eight reference servers, synthetic single-call
  tasks (protocol §13).

# Distil — Claude Code plugin

A live token-savings **status line** and a **`/distil`** command for
[distil](https://github.com/dshakes/distil) context compression.

> **What this plugin does — honestly.** A Claude Code plugin *cannot* reroute a
> running session's API traffic (the base URL is read at launch) and *cannot* set
> the main status line from its own manifest. So this plugin ships the pieces that
> work cleanly: the `/distil` command, and a status-line script you wire in once.
> To actually compress traffic, run distil as a proxy or via `distil wrap`
> (see below) — then the savings show up here.

## Install

```
/plugin marketplace add dshakes/distil
/plugin install distil@distil
```

This gives you the **`/distil`** command (savings report + setup help) and the
**`distil-setup` skill** — ask Claude Code to "set up distil" or "route my agent
through distil" and it installs the CLI and wires your agent/SDK through compression.

## Enable the live savings status line

Add this to your `~/.claude/settings.json` (the plugin ships the script):

```json
{
  "statusLine": {
    "type": "command",
    "command": "${CLAUDE_PLUGIN_ROOT}/statusline.sh"
  }
}
```

It renders, e.g.:

During a live session (an agent currently routed through `distil wrap`/`proxy`):

```
distil · sess 120.0K→45.0K −62% · $0.31 · Σ 12.4M saved ($38.02) · ⚠de 97.5% (398)
```

**This session leads** (original → compressed tokens, percent trimmed, session
dollars); lifetime collapses to one `Σ` figure — the full breakdown lives in
`distil stats` and `/distil-dashboard`. Between sessions (>4 h idle) the line
falls back to the lifetime view:

```
distil · 1.2M→0.5M tok −58% · $2.01 saved · 128 runs · ⚠de 97.5% (398)
```

Decision-equivalence shows once shadow mode clears the reporting floor (50 A/B +
30 A/A); below it the segment counts up instead of printing a rate. `✓` at 99% and
above, `⚠` under 99%, `✗` under 95% (the glyph is an alarm, not decoration, and it
reads without color). The session/lifetime figures above are illustrative; the
equivalence reading is the maintainer's live sample, 97.5% [95.5, 99.5] over n=398
A/B on 2026-09-15 (`benchmarks/results/shadow-live-2026-09-15.json`).
With no savings yet it shows a hint instead.
Requires `distil` on `PATH` or `uvx` available.

**On a flat-rate subscription** (Claude Pro/Max) the per-token dollar figure is
notional — set `DISTIL_SUBSCRIPTION=1` in your environment to drop the cost and
show the token reduction only.

### Already have a status line?

Don't replace it — add distil as one more segment. In your existing status-line script:

```bash
distil_seg="$(distil statusline 2>/dev/null || true)"
[ -n "$distil_seg" ] && out="${out}  ·  ${distil_seg}"
```

`distil statusline` prints nothing when distil isn't installed, and `2>/dev/null || true`
keeps your line clean either way.

## Commands

| Command | What it does |
|---|---|
| `/distil-onboard` | Set up distil + a guided, tailored tour of how to use it |
| `/distil` | Your savings report + how to route more traffic through distil |
| `/distil-stats` | Full breakdown — orig→compressed tokens, cost, runs, per-trajectory bars |
| `/distil-shadow` | Decision-equivalence: did compression preserve your agent's next action? |
| `/distil-dashboard` | Open the HTML savings page — session, lifetime, decision-equivalence cards |
| `/distil-doctor` | Diagnose your setup — ledger, shadow validation, proxy round-trip, wiring |
| `/distil-certify` | Trajectory-level certificate: bound how many solvable tasks compression may cost |
| `/distil-badge` | Shareable shields.io badge of your measured savings |

Want a **live, refreshing view in your terminal**? Run the CLI directly (outside the
slash-command flow, e.g. in a split pane):

```bash
distil dashboard            # live TUI — token-trim + decision-equiv bars, Ctrl-C to exit
distil dashboard --once     # one static frame (pipe-friendly)
```

## Actually compress traffic

The status line reflects the local savings ledger. Populate it by routing an agent
through distil:

```
# Interactive + subscription/OAuth-safe (no tool injection, content un-digested):
distil wrap --lossless-only --verbatim -- claude

# Or a standalone proxy any SDK can point base_url at:
distil proxy                                   # Anthropic / OpenAI-compatible
distil proxy --upstream https://generativelanguage.googleapis.com   # Google Gemini
```

See the [distil docs](https://dshakes.github.io/distil) for the full story.

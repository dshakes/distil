# 0026 — Shell output shaped at the source (`distil sh`)

- **Status:** accepted (opt-in; offline-measured; outcome A/B not yet run)
- **Date:** 2026-10-05
- **Relates to:** `distil/shell.py`, `distil/hook.py` (`run_pre`, `install_hook(shell=True)`, `_shaped_at_source`), `benchmarks/shell_replay.py`, `benchmarks/results/2026-10-05/shell_replay*.json`, `benchmarks/swebench_outcome/arm_distil_sh.py`, `tests/test_shell.py`, `tests/fixtures/shell/`, ADR 0006, ADR 0015, ADR 0022, ADR 0025
- **Reopens:** ADR 0015 — on a different axis (outcome, not bill share); its measurement still stands

## Context

ADR 0015 declined per-command shell profiles because the bill share they could add on top
of the digest was ~1.4%. Since then the question moved from *bytes* to *outcome*. On all
300 SWE-bench Lite tasks RTK 0.51.0 (a command wrapper: its Claude Code PreToolUse hook
rewrites `git status` to `rtk git status`) was the only arm non-inferior to plain
(+0.3 pts) and ~21% cheaper per solved task, while distil was cost-neutral at −1.7 pts
(inconclusive); a Terminal-Bench pilot pointed the same way (n = 18). Two known causes on
distil's side: its digest elides a tool result *after* the agent has it in context, so
the agent re-runs commands to see what was folded (ADR 0025 helps only later re-reads);
and the paid-key digest mode is over its decision-change budget live, so it is held
(ADR 0022). RTK's output is compact before it is ever seen, so there is nothing to re-read.

What RTK does, read from its README and `hooks/claude/rtk-rewrite.sh` (v4, tag v0.51.0):
the hook pipes `tool_input.command` to `rtk rewrite`, whose exit code is the protocol —
0 rewrite + `permissionDecision: "allow"` (an allow rule matched), 1 no equivalent,
2 a deny rule matched (left to Claude Code), 3 rewrite but no decision (Claude Code
prompts). It covers git, test runners (failures only, passing tests collapsed to a count),
`ls`/`find`/`grep`, build and lint tools, cloud CLIs; `rtk gain` reports savings from a
local SQLite store, which also keeps the full output when a command fails or is truncated
(`[full output: rtk recall <id>]`). Missing `jq`/`rtk` or an old binary: the hook warns on
stderr and exits 0, i.e. the command runs unrewritten.

## Decision

Ship `distil sh -- <command>` and an opt-in Claude Code PreToolUse rewrite that inserts
it. The command runs; one deterministic, versioned filter keyed on the command shapes its
output; that is the only form that enters the context.

**(i) Lossless first; lossy only behind a handle.** Two tiers:

- *lossless* — ANSI escapes stripped and `\r` overwrites resolved (the visible text is
  unchanged), exact repeated lines collapsed to `<<xN>>` (the Tier-0 transform the proxy
  uses), and for `git status` the advice lines git prints about *how to use git*
  (`(use "git add <file>..." …)`) dropped — fixed strings that say nothing about the
  repository.
- *elide* — for test runners (pytest, unittest/Django `runtests.py`, `cargo test`,
  `go test`, npm/yarn/pnpm test, jest, vitest), the lines a runner prints for each
  *passing* test are dropped and one marker line names the count and the handle.

Recovery compared two ways. A raw-output **file path** is universal (`cat`, `grep`, `sed`
work on it) but is a new plaintext store of command output under `~/.distil`. A **handle**
in the existing RestoreStore is encrypted at rest, capped, TTL'd, already shared by every
distil surface, and `distil expand <h> | grep x` gives the same partial recovery. Chosen:
the handle — no new store, no weaker at-rest posture. The original is persisted and read
back byte-exact before a lossy line is emitted (`hook._persist`); if it cannot be, the
lossless form is printed.

**(ii) Subscriptions: lossless by default, elide only on explicit opt-in.** `distil sh`
changes what a tool returns, not the request distil forwards — the same position as the
PostToolUse hook, which on a subscription already runs Tier-0 by default and the digest
only with `--digest` (ADR 0006: the boundary is consent). The same rule, through the same
function (`hook.tier_decision` → `policy.may_digest`): metered key → elide on; Claude
Pro/Max login → lossless only unless the user installed with `--digest` or typed
`distil sh --digest`. Installing the rewrite at all is itself an explicit command; nothing
installs it implicitly.

**(iii) Fail-open, at every layer.** The rewrite only *prefixes* `distil sh --` onto one
simple command (optionally after `cd DIR &&`, optionally followed by `2>&1` and
`| tail`/`| head`); anything with `;`, `&&` chains, redirects, `$(…)`, backticks or
quoted metacharacters is left as written, so the shell parses exactly what the agent
wrote. The hook emits `{}` (no change) on any error, when `distil` is not on PATH, or when
`DISTIL_SH_OFF=1`. `distil sh` itself `exec`s the original command untouched when it
does not recognise it, when it cannot start it, or when `DISTIL_SH_OFF=1`; if shaping
raises, the raw output is printed. The exit code is always the command's own (signals as
128+n).

**Windows.** Claude Code runs its Bash tool through Git Bash there, so the shell that
receives the rewritten command is still bash: it parses quoting, globs and `cd DIR &&`
exactly as on POSIX and hands `distil sh` an argv. `distil sh` does not start a second
shell; it applies the leading `VAR=value` words itself (in every path, fail-open
included), resolves the program through `PATH`/`PATHEXT` (so `npm` finds `npm.cmd`,
which `CreateProcess` alone does not), classifies on the bare program name
(`C:\…\python.exe` is `python`), and treats the child's CRLF as LF. There is no `exec`,
so the fail-open path runs the command and returns its code; there are no POSIX signals,
so a terminated child's code is whatever `TerminateProcess` was given, not 128+n.

**Permissions.** Claude Code evaluates permission rules against the *rewritten* input, so
a rewrite could widen what a user allowed. Therefore: a command any `deny`/`ask` Bash rule
could touch is never rewritten (over-matching on purpose — a hit just means Claude Code's
rule decides on the original); `allow` is emitted only when one of the user's own allow
rules (exact, `prefix:*`, `prefix *`, or bare `Bash`) matches the original single command
— no `cd`, no pipe, which Claude Code would check as separate subcommands. Otherwise no
decision is returned and the normal flow (prompt, auto-mode classifier) judges the
rewritten command. Do not add a blanket `Bash(distil sh:*)` allow rule: it would allow any
command typed after it.

**(iv) Measured and attributable.** Every shaped run appends one content-free line to
`hook-receipts.jsonl` (`client: "sh"`, `tool: "sh:<filter>"`, chars before/after, `tier`,
`filters`), so `distil hook --stats` reports it next to the hook. The filter set carries
`FILTERS_VERSION` (`sh-v1`), stamped in receipts, in the marker line and in the outcome
arm's metadata; any change to what a filter keeps or drops bumps it, so a shadow or A/B
change can be attributed to it. Output of `distil sh` is exempt from the PostToolUse hook
(no second pass that would blur attribution).

**(v) Never hide a failure.** A line is dropped only when a filter's pass pattern matches
it *and* the veto does not: the shared keep policy's generic net (error, exception,
traceback, fail, warn, panic, fatal, URLs), any `FAIL`/`ERROR` token, or a count summary.
Property-tested on random interleavings and on every real fixture, in both tiers.

**Search and reads stay verbatim.** `grep`/`rg`/`find`/`ls`/`tree`, `git diff/log/show`
and whole-file reads are not rewritten. 1.56.3 measured that digesting search output (the
agent's file:line map) cost extra steps; nothing here is evidence that a search filter is
safe, so none ships.

**(vi) Opt-in, held like any unproven mode (ADR 0022).** `distil hook install --shell`
(or `distil setup --hooks --shell`). It flips to the default only on outcome evidence:
the `distil-sh` arm non-inferior to plain at the pre-registered 5-point margin on the
300-task SWE-bench Lite run **and** on a Terminal-Bench run of comparable power, with
$/solved at or below plain on both.

## What the offline replay says

`benchmarks/shell_replay.py`, heuristic tokens:

| corpus | what it is | result |
|---|---|---|
| 9 fixtures (`tests/fixtures/shell/`) | real tool output (git, pytest ×4, unittest, cargo, go, node:test) over synthetic projects; one real all-passing slice of this repo's suite | lossless 15.9%, elide 61.3% of those outputs |
| 300 SWE-bench Lite transcripts (`provider-cm` arm, h2h run; not committed) | every bash call the agent made, with the output it got | **27 of 1,390 calls match; 0.06% of bash-output tokens removed** |

The second row is the decision-relevant one, and it is close to zero. On this workload
bash output is grep (40%), `sed -n` slices (21%) and ad-hoc `python -c` (12%); test-runner
output is ~1.3% and `git status` never appears. RTK's per-call output on the same tasks was
only ~8% smaller than this arm's (different trajectories, so indicative only), so its
SWE-bench advantage is not explained by output compression on the commands `distil sh`
covers. **Running the paid `distil-sh` arm on SWE-bench Lite is therefore expected to
show no effect** — the harness arm exists so the comparison is reproducible and so a
test-heavy benchmark (Terminal-Bench) can measure it where the commands actually occur.

## Consequences

- New command `distil sh`, new hook entry (`PreToolUse`, matcher `Bash`, command
  `… -m distil.hook --pre --tier auto|digest`), installed and removed by the same
  ownership-recorded, foreign-hook-preserving code as the PostToolUse hook.
- Claude Code only. Cursor's `beforeShellExecution` can allow/deny but not rewrite;
  Gemini CLI's and Codex CLI's pre-tool hooks were not verified to rewrite input, so they
  are not claimed. Any agent can be told to call `distil sh --` directly.
- The proxy digest can still fold `distil sh` output later on a metered key. Not changed
  here; measure it in the outcome arm first.
- ADR 0015's figures are untouched: this does not claim a bill-share win.

## When to revisit

- The `distil-sh` arm clears the bar in (vi) → make the rewrite part of the default hook
  install.
- A filter for search output: only with an outcome run showing it does not add steps
  (the 1.56.3 regression is the prior).
- A new filter or pattern change: bump `FILTERS_VERSION`, add a captured fixture, keep the
  failure-line property test green.

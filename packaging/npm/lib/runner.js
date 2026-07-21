// Resolve how to run the (Python) distil CLI from Node, in preference order:
//   1. an already-installed `distil` on PATH (pipx/pip install)
//   2. `uvx --from distil-llm distil` (ephemeral, no install — the happy path)
//   3. `pipx run --spec distil-llm distil`
//   4. `python3 -m distil` (if the package is importable)
// Pure Node, zero dependencies. Exported for testing.

"use strict";

const { spawnSync } = require("node:child_process");

function has(cmd) {
  const probe = process.platform === "win32" ? "where" : "which";
  return spawnSync(probe, [cmd], { stdio: "ignore" }).status === 0;
}

/** Return {cmd, prefixArgs} for invoking distil, or null if no runner exists. */
function resolveRunner(env) {
  env = env || process.env;
  if (env.DISTIL_NPM_RUNNER) {
    // escape hatch for tests / custom setups: "cmd arg1 arg2"
    const parts = env.DISTIL_NPM_RUNNER.split(" ").filter(Boolean);
    return { cmd: parts[0], prefixArgs: parts.slice(1) };
  }
  if (has("distil")) return { cmd: "distil", prefixArgs: [] };
  if (has("uvx")) return { cmd: "uvx", prefixArgs: ["--from", "distil-llm", "distil"] };
  if (has("pipx")) return { cmd: "pipx", prefixArgs: ["run", "--spec", "distil-llm", "distil"] };
  return null;
}

const NO_RUNNER_HELP = [
  "distil-llm (npm) bridges to the distil CLI, which runs on Python.",
  "No runner found. Install one of:",
  "  • uv     → https://docs.astral.sh/uv/   (then `npx distil-llm …` just works via uvx)",
  "  • pipx   → pipx install distil-llm",
  "  • pip    → pip install distil-llm",
  "",
  "Then, e.g.:  npx distil-llm wrap -- claude",
].join("\n");

module.exports = { resolveRunner, has, NO_RUNNER_HELP };

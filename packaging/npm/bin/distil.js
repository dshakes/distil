#!/usr/bin/env node
// `npx distil-llm <args>` — thin bridge to the distil CLI (Python), so JS/TS
// developers get distil without touching pip. All args pass straight through.
//
//   npx distil-llm wrap -- claude          # route your agent through distil
//   npx distil-llm proxy                   # start the proxy for any SDK baseURL
//   npx distil-llm stats                   # your genuine savings

"use strict";

const { spawn } = require("node:child_process");
const { resolveRunner, NO_RUNNER_HELP } = require("../lib/runner.js");

const runner = resolveRunner();
if (!runner) {
  process.stderr.write(NO_RUNNER_HELP + "\n");
  process.exit(127);
}

const args = runner.prefixArgs.concat(process.argv.slice(2));
const child = spawn(runner.cmd, args, { stdio: "inherit" });
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code == null ? 1 : code);
});
child.on("error", (err) => {
  process.stderr.write(`distil: failed to launch ${runner.cmd}: ${err.message}\n`);
  process.exit(127);
});

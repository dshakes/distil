import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { resolveRunner } = require("../lib/runner.js");
const { distilBaseURL, distilOpenAIBaseURL } = require("../index.js");

test("DISTIL_NPM_RUNNER escape hatch parses cmd + args", () => {
  const r = resolveRunner({ DISTIL_NPM_RUNNER: "uvx --from distil-llm distil" });
  assert.deepEqual(r, { cmd: "uvx", prefixArgs: ["--from", "distil-llm", "distil"] });
});

test("resolveRunner returns a runner or null (never throws)", () => {
  const r = resolveRunner({}); // real PATH probe — shape check only
  assert.ok(r === null || (typeof r.cmd === "string" && Array.isArray(r.prefixArgs)));
});

test("distilBaseURL defaults and overrides", () => {
  const clean = { ...process.env };
  delete clean.DISTIL_BASE_URL;
  delete clean.DISTIL_PORT;
  const saved = process.env;
  process.env = clean;
  try {
    assert.equal(distilBaseURL(), "http://127.0.0.1:8788");
    assert.equal(distilBaseURL({ port: 9000 }), "http://127.0.0.1:9000");
    assert.equal(distilOpenAIBaseURL(), "http://127.0.0.1:8788/v1");
  } finally {
    process.env = saved;
  }
});

test("distilBaseURL honors DISTIL_BASE_URL", () => {
  const saved = process.env.DISTIL_BASE_URL;
  process.env.DISTIL_BASE_URL = "http://example.test:1234";
  try {
    assert.equal(distilBaseURL(), "http://example.test:1234");
  } finally {
    if (saved === undefined) delete process.env.DISTIL_BASE_URL;
    else process.env.DISTIL_BASE_URL = saved;
  }
});

// node --test packaging/census-worker/test/
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { validateCensus } = require("../lib/validate.js");

const good = () => ({
  schema: 1,
  install_id: "a".repeat(32),
  version: "1.21.0",
  os: "Darwin",
  arch: "arm64",
  python: "3.12.13",
  runs: 42,
  tokens_saved: 1000,
  dollars_saved: 1.23,
  ts: 1784500000,
});

test("accepts a well-formed census", () => {
  assert.equal(validateCensus(good()).ok, true);
});

test("rejects extra keys (frozen schema)", () => {
  const p = { ...good(), extra: "nope" };
  assert.equal(validateCensus(p).ok, false);
});

test("rejects missing keys", () => {
  const p = good();
  delete p.runs;
  assert.equal(validateCensus(p).ok, false);
});

test("rejects non-hex install ids", () => {
  assert.equal(validateCensus({ ...good(), install_id: "Z".repeat(32) }).ok, false);
  assert.equal(validateCensus({ ...good(), install_id: "abc" }).ok, false);
});

test("rejects absurd numbers (aggregate-skew guard)", () => {
  assert.equal(validateCensus({ ...good(), tokens_saved: 1e14 }).ok, false);
  assert.equal(validateCensus({ ...good(), dollars_saved: -1 }).ok, false);
  assert.equal(validateCensus({ ...good(), runs: Infinity }).ok, false);
  assert.equal(validateCensus({ ...good(), ts: 5e9 }).ok, false);
});

test("rejects path-ish platform strings", () => {
  assert.equal(validateCensus({ ...good(), os: "../../etc" }).ok, false);
});

test("rejects wrong schema version and non-objects", () => {
  assert.equal(validateCensus({ ...good(), schema: 2 }).ok, false);
  assert.equal(validateCensus(null).ok, false);
  assert.equal(validateCensus([1]).ok, false);
});

const goodV2 = () => ({
  ...good(),
  schema: 2,
  billing: "metered",
  by_model: { "claude-opus-4-8": 500 },
  agents: ["claude"],
});

test("accepts schema 2 and still accepts schema 1", () => {
  assert.equal(validateCensus(good()).ok, true);
  assert.equal(validateCensus(goodV2()).ok, true);
});

test("rejects bad v2 fields", () => {
  assert.equal(validateCensus({ ...goodV2(), billing: "free" }).ok, false);
  assert.equal(validateCensus({ ...goodV2(), agents: ["evil"] }).ok, false);
  assert.equal(validateCensus({ ...goodV2(), by_model: { "a/b": 1 } }).ok, false);
  assert.equal(validateCensus({ ...goodV2(), by_model: { m: 1e14 } }).ok, false);
  const nine = Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`m${i}`, 1]));
  assert.equal(validateCensus({ ...goodV2(), by_model: nine }).ok, false);
  const missing = goodV2();
  delete missing.agents;
  assert.equal(validateCensus(missing).ok, false);
});

const goodV3 = () => ({
  ...goodV2(),
  schema: 3,
  surfaces: { wrap: 100, proxy: 20 },
  shapes: { anthropic: 90, "openai-chat": 30 },
});

test("accepts schema 3 and rejects bad v3 fields", () => {
  assert.equal(validateCensus(goodV3()).ok, true);
  assert.equal(validateCensus({ ...goodV3(), surfaces: { botnet: 1 } }).ok, false);
  assert.equal(validateCensus({ ...goodV3(), shapes: { anthropic: -1 } }).ok, false);
  assert.equal(validateCensus({ ...goodV3(), shapes: { anthropic: 2e9 } }).ok, false);
  assert.equal(validateCensus({ ...goodV3(), surfaces: "wrap" }).ok, false);
  const missing = goodV3();
  delete missing.shapes;
  assert.equal(validateCensus(missing).ok, false);
  assert.equal(validateCensus({ ...goodV3(), agents: ["evil"] }).ok, false); // v2 rules still apply
});

const goodV4 = () => ({
  ...goodV3(),
  schema: 4,
  equivalence: { pct: 100, shadowed: 500 },
  modes: { interactive: 20, headless: 5, sdk: 1 },
});

test("accepts schema 4 and rejects bad v4 fields", () => {
  assert.equal(validateCensus(goodV4()).ok, true);
  assert.equal(validateCensus({ ...goodV4(), equivalence: { pct: null, shadowed: 3 } }).ok, true);
  assert.equal(validateCensus({ ...goodV4(), equivalence: { pct: 101, shadowed: 1 } }).ok, false);
  assert.equal(validateCensus({ ...goodV4(), equivalence: { pct: 50 } }).ok, false);
  assert.equal(validateCensus({ ...goodV4(), equivalence: { pct: 100, shadowed: -1 } }).ok, false);
  assert.equal(validateCensus({ ...goodV4(), modes: { evil: 1 } }).ok, false);
  assert.equal(validateCensus({ ...goodV4(), modes: { headless: -1 } }).ok, false);
  const missing = goodV4();
  delete missing.modes;
  assert.equal(validateCensus(missing).ok, false);
});

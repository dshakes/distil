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

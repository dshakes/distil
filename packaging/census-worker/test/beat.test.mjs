import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { validateBeat } = require("../lib/beat_validate.js");
const { pairs } = require("../lib/upstash.js");

const good = () => ({ v: 1, id: "a".repeat(32), tokens: 1000, rate: 5.5, ts: 1784600000 });

test("accepts a well-formed heartbeat", () => {
  assert.equal(validateBeat(good()).ok, true);
});

test("rejects bad heartbeats", () => {
  assert.equal(validateBeat({ ...good(), extra: 1 }).ok, false);
  assert.equal(validateBeat({ ...good(), v: 2 }).ok, false);
  assert.equal(validateBeat({ ...good(), id: "Z".repeat(32) }).ok, false);
  assert.equal(validateBeat({ ...good(), tokens: -1 }).ok, false);
  assert.equal(validateBeat({ ...good(), tokens: 1e14 }).ok, false);
  assert.equal(validateBeat({ ...good(), rate: "fast" }).ok, false);
  assert.equal(validateBeat(null).ok, false);
  const missing = good();
  delete missing.rate;
  assert.equal(validateBeat(missing).ok, false);
});

test("upstash pairs() flattens HGETALL output", () => {
  assert.deepEqual(pairs(["a", "10", "b", "20"]), { a: "10", b: "20" });
  assert.deepEqual(pairs(null), {});
});

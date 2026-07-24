import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { validateBeat } = require("../lib/beat_validate.js");
const { pairs } = require("../lib/upstash.js");
const { merge } = require("../api/beat.js");

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

test("merge applies the newest beat by ts (last-write-wins), drops stale replays", () => {
  const b = (tokens, ts) => ({ tokens, ts, rate: 1 });
  // newer ts, higher total → rises
  assert.deepEqual(merge({ ts: "100" }, b(1200, 200)), { tokens: 1200, ts: 200, rate: 1 });
  // newer ts but a LOWER total → propagates (honest re-baseline, not pinned to a ghost)
  assert.deepEqual(merge({ ts: "100" }, b(900, 200)), { tokens: 900, ts: 200, rate: 1 });
  // first beat for a new install (no stored ts) is applied
  assert.deepEqual(merge({ ts: undefined }, b(500, 50)), { tokens: 500, ts: 50, rate: 1 });
  // an OLDER (stale/replayed) beat is dropped so it can't rewind a fresher value
  assert.equal(merge({ ts: "300" }, b(1, 200)), null);
  // same ts → applied (newest wins, idempotent in practice)
  assert.deepEqual(merge({ ts: "200" }, b(1000, 200)), { tokens: 1000, ts: 200, rate: 1 });
});

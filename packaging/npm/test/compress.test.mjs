// Native JS behaviour tests. Cross-implementation byte-equality with Python is
// covered separately by tests/test_ts_conformance.py.
import assert from "node:assert/strict";
import test from "node:test";

import distil from "../index.js";

const { compress, collapseRuns, expandRuns, minifyJson, countTokens, applyTier0 } = distil;

const BIG_LOG = "ERROR failed to connect to worker pool\n".repeat(50);

test("compresses a large repetitive tool result", () => {
  const r = compress([{ role: "tool", content: BIG_LOG }]);
  assert.ok(r.tokensSaved > 0);
  assert.ok(r.savedPct > 50);
  assert.equal(r.tier, "lossless");
});

test("never rewrites the model's own turns", () => {
  const msgs = [{ role: "assistant", content: BIG_LOG }];
  const r = compress(msgs);
  assert.equal(r.messages[0].content, BIG_LOG);
  assert.equal(r.messages[0], msgs[0], "unchanged messages pass through by identity");
});

test("does not mutate the input array or its objects", () => {
  const msgs = [{ role: "tool", content: BIG_LOG }];
  const snapshot = JSON.stringify(msgs);
  compress(msgs);
  assert.equal(JSON.stringify(msgs), snapshot);
});

test("passes non-string content through untouched", () => {
  const blocks = [{ type: "image", source: { data: "..." } }];
  const r = compress([{ role: "user", content: blocks }]);
  assert.equal(r.messages[0].content, blocks);
});

test("empty input is safe", () => {
  const r = compress([]);
  assert.deepEqual(r.messages, []);
  assert.equal(r.savedPct, 0);
});

test("rejects a non-array argument loudly", () => {
  assert.throws(() => compress("not an array"), TypeError);
});

test("collapse round-trips ordinary text", () => {
  const text = "a\na\na\nb\nc\nc\n";
  assert.equal(expandRuns(collapseRuns(text)), text);
});

test("refuses to collapse runs that look like markers", () => {
  const text = "<<x3>>\n<<x3>>\n<<x3>>\nreal\n";
  assert.equal(collapseRuns(text), text);
});

test("declines JSON it cannot reproduce byte-identically", () => {
  assert.equal(minifyJson('{"a": 1.0}'), null, "integral float");
  assert.equal(minifyJson('{"a": 1e5}'), null, "exponent");
  assert.equal(minifyJson('{"2": "b", "1": "a"}'), null, "integer-like keys reorder in JS");
  assert.equal(minifyJson('{"a": 1, "a": 2}'), null, "duplicate key");
});

test("still minifies the safe cases", () => {
  assert.equal(minifyJson('{"b": 1, "a": 2}'), '{"b":1,"a":2}');
  assert.equal(minifyJson("[1, 2, 3]"), "[1,2,3]");
});

test("does not mistake text inside strings for numbers", () => {
  // "line1" looks like exponent notation to a naive source scan.
  assert.equal(minifyJson('{"msg": "line1"}'), '{"msg":"line1"}');
});

test("tier-0 never inflates", () => {
  const tiny = "a\na\na\n";
  assert.equal(applyTier0(tiny), tiny, "marker would cost more tokens than it saves");
});

test("token counting is defined for empty input", () => {
  assert.equal(countTokens(""), 0);
  assert.ok(countTokens("hello world") > 0);
});

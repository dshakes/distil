import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { aggregate, ACTIVE_WINDOW_S, RATE_WINDOW_S } = require("../api/live.js");

const NOW = 1_784_600_000;

test("total is Σ latest-per-install, drift-free", () => {
  const out = aggregate({ a: "100", b: "250" }, { a: NOW, b: NOW }, { a: "1", b: "2" }, NOW);
  assert.equal(out.total, 350);
  assert.equal(out.installs, 2);
});

test("a stale rate decays to 0 while the install still reads active", () => {
  // Beaten 700s ago with a non-zero rate: inside the 900s liveness window but
  // OUTSIDE the 600s rate window → counts as active, contributes 0 to rate.
  // This is the phantom-rate bug: a frozen total must not advertise "X / day".
  const age = 700;
  assert.ok(age <= ACTIVE_WINDOW_S && age > RATE_WINDOW_S);
  const out = aggregate({ a: "1000" }, { a: NOW - age }, { a: "17.38" }, NOW);
  assert.equal(out.active, 1, "still live");
  assert.equal(out.rate, 0, "stale rate must not leak into the aggregate");
});

test("a fresh rate is counted", () => {
  const out = aggregate({ a: "1000" }, { a: NOW - 100 }, { a: "5" }, NOW);
  assert.equal(out.active, 1);
  assert.equal(out.rate, 5);
});

test("a fully idle install drops out of both active and rate", () => {
  const out = aggregate({ a: "1000" }, { a: NOW - 1000 }, { a: "9" }, NOW);
  assert.equal(out.total, 1000, "its saved tokens still count toward the archive");
  assert.equal(out.active, 0);
  assert.equal(out.rate, 0);
});

// Census payload validation — the server-side twin of distil/census.py's
// frozen schema. Pure function, no I/O, unit-tested with node:test.
//
// Contract: EXACTLY the schema-1 keys, tight types, bounded sizes. Anything
// else is rejected — the ingest pipeline re-validates in CI (defense in
// depth), but nothing malformed should even reach the dispatch.

"use strict";

const MAX_BODY_BYTES = 1024; // a census is ~250 bytes; 1 KB is generous
const KEYS = [
  "schema",
  "install_id",
  "version",
  "os",
  "arch",
  "python",
  "runs",
  "tokens_saved",
  "dollars_saved",
  "ts",
];

// Hard ceilings — a hostile client must not be able to skew the community
// aggregates with absurd numbers (rollup enforces the same caps).
const MAX_TOKENS_SAVED = 1e13;
const MAX_DOLLARS_SAVED = 1e8;
const MAX_RUNS = 1e9;

function isShortString(v, max) {
  return typeof v === "string" && v.length > 0 && v.length <= max;
}

/** Validate a parsed census object. Returns {ok: true} or {ok: false, error}. */
function validateCensus(p) {
  if (typeof p !== "object" || p === null || Array.isArray(p)) {
    return { ok: false, error: "not an object" };
  }
  const keys = Object.keys(p).sort();
  if (keys.join(",") !== [...KEYS].sort().join(",")) {
    return { ok: false, error: "schema keys mismatch" };
  }
  if (p.schema !== 1) return { ok: false, error: "unknown schema version" };
  if (!/^[0-9a-f]{32}$/.test(String(p.install_id))) {
    return { ok: false, error: "install_id must be 32 hex chars" };
  }
  if (!isShortString(p.version, 64)) return { ok: false, error: "bad version" };
  for (const k of ["os", "arch", "python"]) {
    if (!isShortString(p[k], 64) || p[k].includes("/") || p[k].includes("\\")) {
      return { ok: false, error: `bad ${k}` };
    }
  }
  for (const [k, max] of [
    ["runs", MAX_RUNS],
    ["tokens_saved", MAX_TOKENS_SAVED],
    ["dollars_saved", MAX_DOLLARS_SAVED],
    ["ts", 4102444800], // 2100-01-01 — clock sanity
  ]) {
    const v = p[k];
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > max) {
      return { ok: false, error: `bad ${k}` };
    }
  }
  return { ok: true };
}

module.exports = { validateCensus, MAX_BODY_BYTES };

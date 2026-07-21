// Heartbeat validation — the near-real-time pulse is a tiny content-free
// payload, separate from the daily census schema. Numbers + a hex id only.
//
//   { v:1, id:<32 hex>, tokens:<int 0..1e13>, rate:<num 0..1e9>, ts:<int> }

"use strict";

const MAX_BEAT_BYTES = 512;
const KEYS = ["v", "id", "tokens", "rate", "ts"];

function validateBeat(p) {
  if (typeof p !== "object" || p === null || Array.isArray(p)) {
    return { ok: false, error: "not an object" };
  }
  if (Object.keys(p).sort().join(",") !== [...KEYS].sort().join(",")) {
    return { ok: false, error: "keys mismatch" };
  }
  if (p.v !== 1) return { ok: false, error: "unknown beat version" };
  if (!/^[0-9a-f]{32}$/.test(String(p.id))) {
    return { ok: false, error: "id must be 32 hex chars" };
  }
  for (const [k, max] of [
    ["tokens", 1e13],
    ["rate", 1e9],
    ["ts", 4102444800],
  ]) {
    const val = p[k];
    if (typeof val !== "number" || !Number.isFinite(val) || val < 0 || val > max) {
      return { ok: false, error: `bad ${k}` };
    }
  }
  return { ok: true };
}

module.exports = { validateBeat, MAX_BEAT_BYTES };

// Census payload validation — the server-side twin of distil/census.py's
// frozen schema. Pure function, no I/O, unit-tested with node:test.
//
// Contract: EXACTLY the schema-1 keys, tight types, bounded sizes. Anything
// else is rejected — the ingest pipeline re-validates in CI (defense in
// depth), but nothing malformed should even reach the dispatch.

"use strict";

const MAX_BODY_BYTES = 2048; // a schema-2 census is ~500 bytes; 2 KB is generous
const KEYS_V1 = [
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
const KEYS_V2 = [...KEYS_V1, "billing", "by_model", "agents"];
const KEYS_V3 = [...KEYS_V2, "surfaces", "shapes"];
const KEYS_V4 = [...KEYS_V3, "equivalence", "modes"];
const MODES = new Set(["interactive", "headless", "sdk"]);
const AGENTS = new Set(["claude", "codex", "gemini", "aider", "other"]);
const SURFACES = new Set(["wrap", "proxy", "gateway"]);
const SHAPES = new Set(["anthropic", "openai-chat", "openai-responses", "gemini"]);
const MAX_MODELS = 8;

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
  if (![1, 2, 3, 4].includes(p.schema)) {
    return { ok: false, error: "unknown schema version" };
  }
  const expect = { 1: KEYS_V1, 2: KEYS_V2, 3: KEYS_V3, 4: KEYS_V4 }[p.schema];
  const keys = Object.keys(p).sort();
  if (keys.join(",") !== [...expect].sort().join(",")) {
    return { ok: false, error: "schema keys mismatch" };
  }
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
  if (p.schema >= 2) {
    if (p.billing !== "subscription" && p.billing !== "metered") {
      return { ok: false, error: "bad billing" };
    }
    const bm = p.by_model;
    if (typeof bm !== "object" || bm === null || Array.isArray(bm)) {
      return { ok: false, error: "bad by_model" };
    }
    const entries = Object.entries(bm);
    if (entries.length > MAX_MODELS) return { ok: false, error: "bad by_model" };
    for (const [m, v] of entries) {
      if (!isShortString(m, 64) || m.includes("/") || m.includes("\\")) {
        return { ok: false, error: "bad by_model key" };
      }
      if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > MAX_TOKENS_SAVED) {
        return { ok: false, error: "bad by_model value" };
      }
    }
    if (!Array.isArray(p.agents) || p.agents.length > 6 || p.agents.some((a) => !AGENTS.has(a))) {
      return { ok: false, error: "bad agents" };
    }
  }
  if (p.schema >= 3) {
    for (const [field, allow] of [
      ["surfaces", SURFACES],
      ["shapes", SHAPES],
    ]) {
      const d = p[field];
      if (typeof d !== "object" || d === null || Array.isArray(d)) {
        return { ok: false, error: `bad ${field}` };
      }
      for (const [k, v] of Object.entries(d)) {
        if (!allow.has(k)) return { ok: false, error: `bad ${field}` };
        if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1e9) {
          return { ok: false, error: `bad ${field} value` };
        }
      }
    }
  }
  if (p.schema >= 4) {
    const eq = p.equivalence;
    if (typeof eq !== "object" || eq === null || Array.isArray(eq)) {
      return { ok: false, error: "bad equivalence" };
    }
    const keys = Object.keys(eq).sort().join(",");
    if (keys !== "pct,shadowed") return { ok: false, error: "bad equivalence" };
    if (eq.pct !== null) {
      if (typeof eq.pct !== "number" || !Number.isFinite(eq.pct) || eq.pct < 0 || eq.pct > 100) {
        return { ok: false, error: "bad equivalence pct" };
      }
    }
    if (typeof eq.shadowed !== "number" || !Number.isInteger(eq.shadowed) || eq.shadowed < 0 || eq.shadowed > 1e9) {
      return { ok: false, error: "bad equivalence shadowed" };
    }
    const modes = p.modes;
    if (typeof modes !== "object" || modes === null || Array.isArray(modes)) {
      return { ok: false, error: "bad modes" };
    }
    for (const [k, v] of Object.entries(modes)) {
      if (!MODES.has(k)) return { ok: false, error: "bad modes" };
      if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1e9) {
        return { ok: false, error: "bad modes value" };
      }
    }
  }
  return { ok: true };
}

module.exports = { validateCensus, MAX_BODY_BYTES };

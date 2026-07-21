// Minimal Upstash Redis REST client (fetch-based, zero deps). The live counter
// needs atomic per-install upserts + a fast aggregate read; Upstash's REST API
// is the right primitive for a Vercel function (no TCP pooling). Env:
//   UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
// Returns null when unconfigured so callers degrade gracefully (503 / fallback).

"use strict";

function creds() {
  // Accept both naming conventions: the classic Upstash integration
  // (UPSTASH_REDIS_REST_*) and Vercel's Marketplace Redis storage
  // (KV_REST_API_*) — whichever the connected store injected.
  const url = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
  return url && token ? { url, token } : null;
}

// Run a pipeline of commands (array of string arrays). Returns the results
// array, or throws on a transport/HTTP error.
async function pipeline(commands) {
  const c = creds();
  if (!c) return null;
  const resp = await fetch(`${c.url}/pipeline`, {
    method: "POST",
    headers: { Authorization: `Bearer ${c.token}`, "Content-Type": "application/json" },
    body: JSON.stringify(commands),
  });
  if (!resp.ok) throw new Error(`upstash ${resp.status}`);
  return resp.json();
}

// Upstash HGETALL returns a flat [f1, v1, f2, v2, …]; pair it into an object.
function pairs(flat) {
  const out = {};
  if (Array.isArray(flat)) {
    for (let i = 0; i + 1 < flat.length; i += 2) out[flat[i]] = flat[i + 1];
  }
  return out;
}

module.exports = { creds, pipeline, pairs };

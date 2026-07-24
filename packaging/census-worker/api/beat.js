// distil heartbeat ingest — the near-real-time community pulse.
//
// POST /v1/beat  { v:1, id, tokens, rate, ts }
//
// Validates (lib/beat_validate.js) then upserts the latest-per-install token
// total / last-seen / recent-rate into Upstash Redis. No git commit, no CI —
// this path is high-frequency by design and never touches the audit branch
// (the daily /v1/ping census remains the auditable archive). Stores no IPs.
//
// Returns 202 on accept, 503 when Upstash isn't configured (the client fails
// open — the daily census still carries the exact totals).

"use strict";

const { validateBeat, MAX_BEAT_BYTES } = require("../lib/beat_validate.js");
const { creds, pipeline } = require("../lib/upstash.js");

// A beat may only RAISE an install's stored total, never lower it. The client
// already sends a monotonic count-time-calibrated total; this holds the line for
// a pre-fix client, or one whose local accrual state (~/.distil/census-savings.json)
// was wiped and now re-accrues from a lower base — the community counter must
// never tick backward. Extracted pure so it has a runnable check without Upstash.
function monotonicTokens(stored, incoming) {
  return Math.max(Number(stored) || 0, Number(incoming) || 0);
}

// Best-effort per-instance rate limit (real limiting belongs in the firewall).
const bucket = { tokens: 120, last: Date.now() };
function rateLimited() {
  const now = Date.now();
  bucket.tokens = Math.min(120, bucket.tokens + ((now - bucket.last) / 1000) * 4);
  bucket.last = now;
  if (bucket.tokens < 1) return true;
  bucket.tokens -= 1;
  return false;
}

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.statusCode = 405;
    return res.end();
  }
  if (rateLimited()) {
    res.statusCode = 429;
    return res.end();
  }
  const body = req.body;
  if (!body || JSON.stringify(body).length > MAX_BEAT_BYTES) {
    res.statusCode = 413;
    return res.end();
  }
  const verdict = validateBeat(body);
  if (!verdict.ok) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify({ error: verdict.error }));
  }
  if (!creds()) {
    res.statusCode = 503; // Upstash not provisioned yet — client fails open
    return res.end();
  }
  try {
    // Read the current stored total first so the write can only raise it. Per
    // install, beats are ≤1/5min, so this read-then-write has no real race.
    const prev = await pipeline([["HGET", "hb:tok", body.id]]);
    const tokens = monotonicTokens(prev && prev[0] && prev[0].result, body.tokens);
    await pipeline([
      ["HSET", "hb:tok", body.id, String(tokens)],
      ["HSET", "hb:ts", body.id, String(body.ts)],
      ["HSET", "hb:rate", body.id, String(body.rate)],
    ]);
    res.statusCode = 202;
  } catch (e) {
    res.statusCode = 502;
  }
  res.end();
};

module.exports.monotonicTokens = monotonicTokens;

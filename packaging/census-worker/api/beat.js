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

// The client is authoritative for its own running total, so a beat is applied
// LAST-WRITE-WINS by timestamp: the newest beat sets tokens/ts/rate wholesale —
// even when the new total is LOWER. A legitimate re-baseline (local accrual state
// ~/.distil/census-savings.json wiped, or a downward calibration) MUST propagate;
// the old max() pinned the store to a stale high-water that no install could
// reproduce — exactly the ghost total this counter must never show. Beats are
// monotonic at source, so in normal flow the total only rises; it drops solely on
// a real reset, which is honest. An out-of-order OLDER beat (ts < stored) is
// dropped so it can't rewind a fresher value. Pure → runnable check without Upstash.
function merge(stored, incoming) {
  if (Number(incoming.ts) < (Number(stored && stored.ts) || 0)) return null; // stale/replayed
  return {
    tokens: Number(incoming.tokens) || 0,
    ts: Number(incoming.ts) || 0,
    rate: Number(incoming.rate) || 0,
  };
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
    // Read the current stored ts so an out-of-order/replayed OLDER beat can't
    // rewind a fresher value; otherwise the newest beat wins wholesale. Per
    // install, beats are ≤1/5min, so this read-then-write has no real race.
    const prev = await pipeline([["HGET", "hb:ts", body.id]]);
    const next = merge({ ts: prev && prev[0] && prev[0].result }, body);
    if (next) {
      await pipeline([
        ["HSET", "hb:tok", body.id, String(next.tokens)],
        ["HSET", "hb:ts", body.id, String(next.ts)],
        ["HSET", "hb:rate", body.id, String(next.rate)],
      ]);
    }
    res.statusCode = 202;
  } catch (e) {
    res.statusCode = 502;
  }
  res.end();
};

module.exports.merge = merge;

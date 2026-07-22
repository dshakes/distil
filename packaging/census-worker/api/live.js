// distil live aggregate — the near-real-time community counter's data source.
//
// GET /v1/live → { available, total, active, rate, installs, as_of }
//
//   total    exact community tokens saved = Σ latest-per-install (no drift —
//            summed on read, not an incremental counter)
//   active   installs seen within ACTIVE_WINDOW_S — "is this install running
//            distil right now" (liveness), NOT "saving more right now"
//   rate     Σ recent-rate of installs beaten within RATE_WINDOW_S only. A
//            reported rate is growth over ONE past beat interval; once an
//            install goes quiet it is a stale lagging measurement, so it only
//            counts while FRESH. Beyond RATE_WINDOW_S it decays to 0 even
//            though the install still reads "active". The page shows this as
//            context ("X / day"); it does NOT project the counter forward
//            (a stale rate would inflate the total while everyone's idle —
//            phantom tokens). The digits only move when `total` actually rises.
//
// CORS-open + no-store so the docs page (any origin) can poll it. Returns
// {available:false} when Upstash isn't configured, so the page cleanly falls
// back to the exact daily-census total.

"use strict";

const { creds, pipeline, pairs } = require("../lib/upstash.js");

const ACTIVE_WINDOW_S = 900; // 15 min — "recently active" (liveness)
// Rate is a lagging per-interval measurement; only trust it while fresh. 2× the
// client's 300s beat interval tolerates one dropped/delayed beat, then zeroes.
const RATE_WINDOW_S = 600;

// Pure aggregation over the heartbeat store — Σ-on-read, no drift. Extracted so
// the rate-freshness logic has a runnable check without Upstash. `now` in seconds.
function aggregate(tok, ts, rate, now) {
  let total = 0, active = 0, rateSum = 0, installs = 0;
  for (const id of Object.keys(tok)) {
    total += Number(tok[id]) || 0;
    installs += 1;
    const age = now - (Number(ts[id]) || 0);
    if (age <= ACTIVE_WINDOW_S) active += 1;
    if (age <= RATE_WINDOW_S) rateSum += Number(rate[id]) || 0;
  }
  return {
    available: true,
    total: Math.round(total),
    active,
    rate: Math.round(rateSum * 100) / 100,
    installs,
    as_of: Math.round(now),
  };
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Content-Type", "application/json");
  if (req.method !== "GET") {
    res.statusCode = 405;
    return res.end();
  }
  if (!creds()) {
    res.statusCode = 200;
    return res.end(JSON.stringify({ available: false }));
  }
  try {
    const out = await pipeline([
      ["HGETALL", "hb:tok"],
      ["HGETALL", "hb:ts"],
      ["HGETALL", "hb:rate"],
    ]);
    const tok = pairs(out[0] && out[0].result);
    const ts = pairs(out[1] && out[1].result);
    const rate = pairs(out[2] && out[2].result);
    res.statusCode = 200;
    res.end(JSON.stringify(aggregate(tok, ts, rate, Date.now() / 1000)));
  } catch (e) {
    res.statusCode = 200;
    res.end(JSON.stringify({ available: false }));
  }
};

module.exports.aggregate = aggregate;
module.exports.ACTIVE_WINDOW_S = ACTIVE_WINDOW_S;
module.exports.RATE_WINDOW_S = RATE_WINDOW_S;

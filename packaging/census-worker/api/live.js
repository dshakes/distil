// distil live aggregate — the near-real-time community counter's data source.
//
// GET /v1/live → { available, total, active, rate, installs, as_of }
//
//   total    exact community tokens saved = Σ latest-per-install (no drift —
//            summed on read, not an incremental counter)
//   active   installs seen within ACTIVE_WINDOW_S (drive the "is anyone
//            working right now" signal)
//   rate     Σ recent-rate of ACTIVE installs only → the page projects the
//            counter forward at this, so it ticks while people work and goes
//            STATIC the moment they idle (rate decays to 0). Honest.
//
// CORS-open + no-store so the docs page (any origin) can poll it. Returns
// {available:false} when Upstash isn't configured, so the page cleanly falls
// back to the exact daily-census total.

"use strict";

const { creds, pipeline, pairs } = require("../lib/upstash.js");

const ACTIVE_WINDOW_S = 900; // 15 min — "recently active"

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
    const now = Date.now() / 1000;
    let total = 0;
    let active = 0;
    let rateSum = 0;
    let installs = 0;
    for (const id of Object.keys(tok)) {
      const t = Number(tok[id]) || 0;
      total += t;
      installs += 1;
      if (now - (Number(ts[id]) || 0) <= ACTIVE_WINDOW_S) {
        active += 1;
        rateSum += Number(rate[id]) || 0;
      }
    }
    res.statusCode = 200;
    res.end(
      JSON.stringify({
        available: true,
        total: Math.round(total),
        active,
        rate: Math.round(rateSum * 100) / 100,
        installs,
        as_of: Math.round(now),
      })
    );
  } catch (e) {
    res.statusCode = 200;
    res.end(JSON.stringify({ available: false }));
  }
};

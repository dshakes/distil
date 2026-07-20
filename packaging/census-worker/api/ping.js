// distil census ingest — Vercel serverless function (zero dependencies).
//
// POST /v1/ping  { schema:1, install_id, version, os, arch, python,
//                  runs, tokens_saved, dollars_saved, ts }
//
// Validates strictly (lib/validate.js), then forwards the census to GitHub as
// a `repository_dispatch` event on DISPATCH_REPO; .github/workflows/
// census-ingest.yml re-validates and appends it to the metrics branch. The
// worker itself stores NOTHING: no database, no IP addresses, no logs beyond
// the platform's default request log.
//
// Security posture:
//   - strict schema allowlist, 1 KB body cap, POST-only, JSON-only
//   - client IP is never read, forwarded, or persisted by this code
//   - GITHUB_DISPATCH_TOKEN is a fine-grained PAT scoped to ONE repo with
//     Contents read/write only (see README.md) — the blast radius of a leak
//     is "can push commits to the public metrics repo"
//   - per-instance token-bucket rate limit (best effort on serverless; put
//     Vercel Firewall / WAF rate limiting in front for the real guarantee)

"use strict";

const { validateCensus, MAX_BODY_BYTES } = require("../lib/validate.js");

// ponytail: per-instance in-memory bucket — real rate limiting belongs in the
// platform firewall; this only blunts a single hot instance.
const bucket = { tokens: 30, last: Date.now() };
function rateLimited() {
  const now = Date.now();
  bucket.tokens = Math.min(30, bucket.tokens + ((now - bucket.last) / 1000) * 0.5);
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

  // Vercel parses JSON bodies into req.body; enforce the size cap regardless.
  const body = req.body;
  if (!body || JSON.stringify(body).length > MAX_BODY_BYTES) {
    res.statusCode = 413;
    return res.end();
  }
  const verdict = validateCensus(body);
  if (!verdict.ok) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify({ error: verdict.error }));
  }

  const repo = process.env.DISPATCH_REPO || "dshakes/distil";
  const token = process.env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    res.statusCode = 503;
    return res.end();
  }

  const resp = await fetch(`https://api.github.com/repos/${repo}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
      "User-Agent": "distil-census-worker",
    },
    body: JSON.stringify({ event_type: "census", client_payload: { census: body } }),
  });

  res.statusCode = resp.status === 204 ? 202 : 502;
  res.end();
};

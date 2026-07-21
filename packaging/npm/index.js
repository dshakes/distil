// distil-llm — TS/JS helpers for routing any SDK through the distil proxy.
//
// distil compresses at the network layer: start the proxy once
// (`npx distil-llm proxy`, default http://127.0.0.1:8788) and point your SDK's
// baseURL at it — no code changes to your calls, cache-aware reversible
// compression, per-request decision-equivalence proof.
//
//   import Anthropic from "@anthropic-ai/sdk";
//   import { distilBaseURL } from "distil-llm";
//   const client = new Anthropic({ baseURL: distilBaseURL() });
//
//   import { createAnthropic } from "@ai-sdk/anthropic";
//   const anthropic = createAnthropic({ baseURL: distilBaseURL() });
//
// Zero dependencies. The proxy itself is the integration; these are just the
// tiny glue so you don't hardcode the URL.

"use strict";

const DEFAULT_PORT = 8788;
const DEFAULT_HOST = "127.0.0.1";

/** Base URL of a local distil proxy. Reads DISTIL_BASE_URL if set, else builds
 *  http://<host>:<port> (defaults 127.0.0.1:8788). */
function distilBaseURL(opts) {
  opts = opts || {};
  if (process.env.DISTIL_BASE_URL) return process.env.DISTIL_BASE_URL;
  const host = opts.host || DEFAULT_HOST;
  const port = opts.port || Number(process.env.DISTIL_PORT) || DEFAULT_PORT;
  return `http://${host}:${port}`;
}

/** OpenAI-style clients expect the base URL to include the API version segment. */
function distilOpenAIBaseURL(opts) {
  return distilBaseURL(opts).replace(/\/+$/, "") + "/v1";
}

module.exports = { distilBaseURL, distilOpenAIBaseURL, DEFAULT_PORT, DEFAULT_HOST };

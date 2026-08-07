// Base-URL helpers, factored out of index.js so lib/ modules can use them
// without importing the package entry point (which imports them back).
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

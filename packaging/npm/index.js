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

const {
  distilBaseURL,
  distilOpenAIBaseURL,
  DEFAULT_PORT,
  DEFAULT_HOST,
} = require("./lib/baseurl");
const { compress } = require("./lib/compress");
const { distilMiddleware } = require("./lib/aisdk");
const { applyTier0, collapseRuns, expandRuns, minifyJson, countTokens } = require("./lib/tier0");

module.exports = {
  // in-process lossless compression (byte-identical to the Python engine)
  compress,
  // Vercel AI SDK middleware for wrapLanguageModel()
  distilMiddleware,
  // proxy routing helpers
  distilBaseURL,
  distilOpenAIBaseURL,
  DEFAULT_PORT,
  DEFAULT_HOST,
  // lower-level transforms, exposed for verification and testing
  applyTier0,
  collapseRuns,
  expandRuns,
  minifyJson,
  countTokens,
};

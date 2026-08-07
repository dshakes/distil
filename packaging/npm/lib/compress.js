// Public TypeScript/JavaScript compression API.
//
// Two tiers, two very different trust stories, and the difference is the whole
// design of this file:
//
//   Tier-0 (lossless)  — implemented natively in lib/tier0.js and proven
//                        BYTE-IDENTICAL to the Python engine by
//                        tests/test_ts_conformance.py. Safe to run in-process:
//                        no daemon, no network, no handles to keep alive.
//
//   Tier-1 (digest)    — NOT implemented here. It mints restore handles whose
//                        originals must live in ONE store shared with the proxy
//                        and the MCP server, or `expand` silently fails to
//                        resolve a handle the model can see. It is also the tier
//                        the decision-equivalence certificate actually measures.
//                        A second implementation would be a second thing to
//                        certify, and nobody would notice when they drifted.
//
// So `compress()` is lossless-only and always available in-process. The digest
// tier comes from routing your client through the proxy (see the note at the
// bottom of this file), which is the same engine, store, and certificate.

"use strict";

const { applyTier0, countTokens } = require("./tier0");

/** Roles whose content distil never rewrites. */
const NEVER_REWRITE = new Set(["assistant", "ai"]);

function roleOf(m) {
  if (!m || typeof m !== "object") return "";
  return String(m.role || m.type || "").toLowerCase();
}

/**
 * Compress a message list in-process, lossless tier only.
 *
 * Mirrors the Python `distil.compress_messages` contract: the input is never
 * mutated, unchanged messages pass through by identity, non-string content
 * (image blocks, tool_use structures) is untouched, and the model's own turns
 * are never rewritten.
 *
 * @param {Array<object>} messages OpenAI/Anthropic-style message objects
 * @returns {{messages: Array<object>, tokensBefore: number, tokensAfter: number,
 *            tokensSaved: number, savedPct: number, tier: "lossless"}}
 */
function compress(messages) {
  if (!Array.isArray(messages)) {
    throw new TypeError("compress(messages): expected an array of messages");
  }
  let before = 0;
  let after = 0;
  const out = messages.map((m) => {
    const content = m && typeof m === "object" ? m.content : undefined;
    if (typeof content !== "string") return m;
    before += countTokens(content);
    if (NEVER_REWRITE.has(roleOf(m))) {
      after += countTokens(content);
      return m;
    }
    const next = applyTier0(content);
    after += countTokens(next);
    return next === content ? m : Object.assign({}, m, { content: next });
  });
  return {
    messages: out,
    tokensBefore: before,
    tokensAfter: after,
    tokensSaved: before - after,
    // Can legitimately be ~0: savings come from LARGE tool output, and a request
    // that is mostly system prompt has nothing to fold.
    savedPct: before > 0 ? (100 * (before - after)) / before : 0,
    tier: "lossless",
  };
}

/**
 * How to get the reversible DIGEST tier from JS.
 *
 * There is deliberately no `compressViaProxy()` here. The distil proxy has no
 * "compress this payload and hand it back" endpoint, and inventing one would add
 * a new public API surface — accepting arbitrary content, needing its own
 * authentication story — to get something the existing design already provides.
 *
 * The proxy's contract is transparent interception: point your SDK's baseURL at
 * it and every request is compressed on the way through, by the same engine the
 * decision-equivalence certificate was measured against, minting handles into the
 * one restore store that `distil expand` and the MCP server read.
 *
 *   import Anthropic from "@anthropic-ai/sdk";
 *   import { distilBaseURL } from "distil-llm";
 *   const client = new Anthropic({ baseURL: distilBaseURL() });
 *
 * Use `compress()` above when you hold the message list and want lossless
 * savings with no daemon; route through the proxy when you want the digest.
 */

module.exports = { compress };

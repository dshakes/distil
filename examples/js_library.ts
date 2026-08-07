/**
 * Embed distil in a TypeScript agent — no proxy, no daemon.
 *
 *   npm i distil-llm
 *   npx tsx examples/js_library.ts
 *
 * This is the LOSSLESS tier, and it is byte-identical to the Python engine —
 * enforced by a conformance suite that runs both implementations over a shared
 * corpus and fails CI on any divergence.
 *
 * For the reversible DIGEST tier, route through a distil proxy (see below): the
 * digest mints restore handles that must share one store with the proxy and the
 * MCP server, and it is the tier the decision-equivalence certificate measures.
 */
import { compress, distilBaseURL } from "distil-llm";

const toolOutput = Array.from(
  { length: 400 },
  (_, i) => `2026-08-07T12:00:${String(i % 60).padStart(2, "0")}Z INFO worker heartbeat seq=${i}`,
).join("\n");

const messages = [
  { role: "system", content: "You are a site reliability engineer." },
  { role: "user", content: "Why did the worker pool die?" },
  { role: "tool", content: toolOutput },
];

const result = compress(messages);

console.log(`tokens: ${result.tokensBefore} -> ${result.tokensAfter}`);
console.log(`saved : ${result.tokensSaved} (${result.savedPct.toFixed(1)}% smaller)`);

// The input is never mutated; unchanged messages come back by identity.
console.assert(messages[2].content === toolOutput);
console.assert(result.messages[0] === messages[0]);

// Send result.messages exactly as you would have sent messages:
//
//   const response = await client.messages.create({ model, messages: result.messages });

// --- want the digest tier instead? -----------------------------------------
// Start a proxy (`npx distil-llm proxy`) and point your SDK at it. Same engine,
// same restore store, same certificate — and no code change at the call site.
//
//   import Anthropic from "@anthropic-ai/sdk";
//   const client = new Anthropic({ baseURL: distilBaseURL() });
console.log(`proxy would be at: ${distilBaseURL()}`);

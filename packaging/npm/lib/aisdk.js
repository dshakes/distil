// Vercel AI SDK middleware.
//
// The proxy route (`createAnthropic({ baseURL: distilBaseURL() })`) is still the
// way to get the reversible digest tier. This is for the other case: you want
// lossless savings in-process, with no daemon to keep alive — a serverless
// function, an edge worker, a CI job that runs once.
//
//   import { wrapLanguageModel } from "ai";
//   import { distilMiddleware } from "distil-llm";
//
//   const model = wrapLanguageModel({
//     model: gateway("anthropic/claude-sonnet-5"),
//     middleware: distilMiddleware(),
//   });
//
// Implements `transformParams` only. `wrapGenerate`/`wrapStream` are not
// implemented because compression happens on the way IN — wrapping the response
// would let us rewrite the model's own output, which distil does not do.

"use strict";

const { applyTier0, countTokens } = require("./tier0");

/**
 * A `LanguageModelV4Prompt` message carries `content` that is a plain STRING for
 * `system`, and an ARRAY OF PARTS for `user`, `assistant`, and `tool`. Both are
 * handled; anything unrecognised is passed through untouched, because a
 * compressor that mangles a shape it does not understand is worse than one that
 * saves nothing.
 */

/** Compress a text-bearing part, preserving every sibling key. */
function compressPart(part) {
  if (!part || typeof part !== "object") return part;

  // { type: "text", text: "..." } — user and tool text.
  if (part.type === "text" && typeof part.text === "string") {
    const next = applyTier0(part.text);
    return next === part.text ? part : { ...part, text: next };
  }

  // Tool results carry the payload under different keys across SDK versions:
  //   v5: { type: "tool-result", output: { type: "text", value: "..." } }
  //   v4: { type: "tool-result", result: "..." }
  // Both are handled rather than pinning one, so an SDK bump does not silently
  // stop compressing the single largest thing in an agent's context.
  if (part.type === "tool-result") {
    if (part.output && typeof part.output.value === "string") {
      const next = applyTier0(part.output.value);
      return next === part.output.value
        ? part
        : { ...part, output: { ...part.output, value: next } };
    }
    if (typeof part.result === "string") {
      const next = applyTier0(part.result);
      return next === part.result ? part : { ...part, result: next };
    }
  }
  return part; // file / image / tool-call parts are never rewritten
}

function compressMessage(message) {
  if (!message || typeof message !== "object") return message;

  // The model's own words are not distil's to edit.
  if (message.role === "assistant") return message;

  if (typeof message.content === "string") {
    const next = applyTier0(message.content);
    return next === message.content ? message : { ...message, content: next };
  }
  if (Array.isArray(message.content)) {
    const parts = message.content.map(compressPart);
    const changed = parts.some((p, i) => p !== message.content[i]);
    return changed ? { ...message, content: parts } : message;
  }
  return message;
}

function textOf(message) {
  if (typeof message?.content === "string") return message.content;
  if (Array.isArray(message?.content)) {
    return message.content
      .map((p) =>
        typeof p?.text === "string"
          ? p.text
          : typeof p?.output?.value === "string"
            ? p.output.value
            : typeof p?.result === "string"
              ? p.result
              : ""
      )
      .join("");
  }
  return "";
}

/**
 * Build AI SDK middleware that losslessly compresses the prompt before it is sent.
 *
 * @param {{onSavings?: (info: {tokensBefore: number, tokensAfter: number,
 *   tokensSaved: number, savedPct: number}) => void}} [opts]
 *   `onSavings` is called once per request with the measured delta. Provided
 *   because in-process compression has no proxy to report through, and savings
 *   you cannot see are savings you will not trust.
 * @returns {{transformParams: (arg: {params: object}) => Promise<object>}}
 */
function distilMiddleware(opts) {
  const onSavings = opts && typeof opts.onSavings === "function" ? opts.onSavings : null;
  return {
    transformParams: async ({ params }) => {
      if (!params || !Array.isArray(params.prompt)) return params;

      const before = onSavings ? params.prompt.reduce((n, m) => n + countTokens(textOf(m)), 0) : 0;
      const prompt = params.prompt.map(compressMessage);
      const changed = prompt.some((m, i) => m !== params.prompt[i]);

      if (onSavings) {
        const after = prompt.reduce((n, m) => n + countTokens(textOf(m)), 0);
        onSavings({
          tokensBefore: before,
          tokensAfter: after,
          tokensSaved: before - after,
          savedPct: before > 0 ? (100 * (before - after)) / before : 0,
        });
      }
      return changed ? { ...params, prompt } : params;
    },
  };
}

module.exports = { distilMiddleware };

/**
 * `compress` — the public in-process compression API.
 *
 * Tier-0 (lossless) only. The decision-equivalence certificate is measured
 * against the Python engine; a second digest implementation here would sit
 * outside that guarantee. For the reversible digest tier, route your SDK
 * through the distil proxy (`distilBaseURL()` in the distil-llm npm package).
 *
 * Invariants (shared with the Python path):
 *   - Input array and its objects are NEVER mutated.
 *   - Unchanged messages pass through by identity (===).
 *   - The model's own turns (assistant/ai role) are NEVER rewritten.
 *   - Non-string content (image blocks, tool_use structures) is untouched.
 *   - Tier-0 never inflates: compressed text is only kept when token count ≤ original.
 */

import { applyTier0, countTokens } from "./tier0.js";
import type {
  AnthropicContentBlock,
  AnthropicMessage,
  CompressOptions,
  CompressReceipt,
  CompressResult,
  MessageForProvider,
  OpenAIContentPart,
  OpenAIMessage,
  Provider,
} from "./types.js";

/** Roles whose content distil never rewrites. */
const NEVER_REWRITE = new Set(["assistant", "ai"]);

function isNeverRewriteRole(role: unknown): boolean {
  return typeof role === "string" && NEVER_REWRITE.has(role.toLowerCase());
}

// ── Anthropic ─────────────────────────────────────────────────────────────────
// TypeScript cannot narrow a discriminated union that contains a catch-all
// `{ type: string }` member, because `string` is a supertype of every literal.
// We use `Extract` to pull out the specific variant after the runtime check —
// the check IS the narrowing proof; the cast documents that invariant.

type AnthropicTextBlock = Extract<AnthropicContentBlock, { type: "text" }>;
type AnthropicToolResultBlock = Extract<AnthropicContentBlock, { type: "tool_result" }>;
type OpenAITextPart = Extract<OpenAIContentPart, { type: "text" }>;

function compressAnthropicBlock(block: AnthropicContentBlock): AnthropicContentBlock {
  if (block.type === "text") {
    const b = block as AnthropicTextBlock;
    const next = applyTier0(b.text);
    if (next === b.text) return block;
    return { ...b, text: next };
  }
  if (block.type === "tool_result") {
    const b = block as AnthropicToolResultBlock;
    if (typeof b.content !== "string") return block;
    const next = applyTier0(b.content);
    if (next === b.content) return block;
    return { ...b, content: next };
  }
  // image, tool_use, thinking, and all other provider-opaque blocks: pass through.
  return block;
}

function tokensInAnthropicContent(content: AnthropicMessage["content"]): number {
  if (typeof content === "string") return countTokens(content);
  let total = 0;
  for (const block of content) {
    if (block.type === "text") total += countTokens((block as AnthropicTextBlock).text);
    else if (block.type === "tool_result") {
      const c = (block as AnthropicToolResultBlock).content;
      if (typeof c === "string") total += countTokens(c);
    }
  }
  return total;
}

function compressAnthropicMessage(msg: AnthropicMessage): AnthropicMessage {
  if (isNeverRewriteRole(msg.role)) return msg;
  if (typeof msg.content === "string") {
    const next = applyTier0(msg.content);
    return next === msg.content ? msg : { ...msg, content: next };
  }
  let changed = false;
  const nextBlocks = msg.content.map((block) => {
    const nb = compressAnthropicBlock(block);
    if (nb !== block) changed = true;
    return nb;
  });
  return changed ? { ...msg, content: nextBlocks } : msg;
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

function compressOpenAIPart(part: OpenAIContentPart): OpenAIContentPart {
  if (part.type === "text") {
    const p = part as OpenAITextPart;
    const next = applyTier0(p.text);
    if (next === p.text) return part;
    return { ...p, text: next };
  }
  return part;
}

function tokensInOpenAIContent(content: OpenAIMessage["content"]): number {
  if (!content) return 0;
  if (typeof content === "string") return countTokens(content);
  let total = 0;
  for (const part of content) {
    if (part.type === "text") total += countTokens((part as OpenAITextPart).text);
  }
  return total;
}

function compressOpenAIMessage(msg: OpenAIMessage): OpenAIMessage {
  if (isNeverRewriteRole(msg.role)) return msg;
  if (!msg.content) return msg;
  if (typeof msg.content === "string") {
    const next = applyTier0(msg.content);
    return next === msg.content ? msg : { ...msg, content: next };
  }
  let changed = false;
  const nextParts = msg.content.map((part) => {
    const np = compressOpenAIPart(part);
    if (np !== part) changed = true;
    return np;
  });
  return changed ? { ...msg, content: nextParts } : msg;
}

// ── Token accounting helpers ──────────────────────────────────────────────────

function tokensInMessage(msg: AnthropicMessage | OpenAIMessage, provider: Provider): number {
  if (provider === "anthropic") {
    return tokensInAnthropicContent((msg as AnthropicMessage).content);
  }
  return tokensInOpenAIContent((msg as OpenAIMessage).content);
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Compress a message list in-process using the lossless tier.
 *
 * Both Anthropic and OpenAI message shapes are supported; pass `provider` to
 * select the correct content-block traversal. Unchanged messages are returned
 * by identity (`===`); the input array is never mutated.
 *
 * @example
 * ```ts
 * import { compress } from "distil-sdk";
 *
 * const { messages, receipt } = compress(messages, { provider: "anthropic" });
 * console.log(`Saved ${receipt.savedPct.toFixed(1)}% tokens`);
 * ```
 */
export function compress<P extends Provider>(
  messages: MessageForProvider<P>[],
  options: CompressOptions & { provider: P },
): CompressResult<MessageForProvider<P>>;

// Overload for callers using the base type without the generic:
export function compress(
  messages: (AnthropicMessage | OpenAIMessage)[],
  options: CompressOptions,
): CompressResult<AnthropicMessage | OpenAIMessage>;

export function compress(
  messages: (AnthropicMessage | OpenAIMessage)[],
  options: CompressOptions,
): CompressResult<AnthropicMessage | OpenAIMessage> {
  if (!Array.isArray(messages)) {
    throw new TypeError("compress(messages, options): messages must be an array");
  }

  const { provider } = options;
  let before = 0;
  let after = 0;

  const out = messages.map((msg) => {
    before += tokensInMessage(msg, provider);

    let next: AnthropicMessage | OpenAIMessage;
    if (provider === "anthropic") {
      next = compressAnthropicMessage(msg as AnthropicMessage);
    } else {
      next = compressOpenAIMessage(msg as OpenAIMessage);
    }

    after += tokensInMessage(next, provider);
    return next;
  });

  const receipt: CompressReceipt = {
    tokensBefore: before,
    tokensAfter: after,
    tokensSaved: before - after,
    savedPct: before > 0 ? (100 * (before - after)) / before : 0,
    tier: "lossless",
  };

  return { messages: out, receipt };
}

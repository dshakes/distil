/**
 * distil-sdk — in-process lossless compression for Anthropic and OpenAI message lists.
 *
 * Tier-0 lossless only. Byte-identical to the Python engine on the certified
 * transforms; the in-process path carries the same decision-equivalence guarantee
 * as the proxy for the lossless tier.
 *
 * For the reversible digest tier (larger savings, restore handles), route your
 * SDK through the distil proxy instead.
 */

export { compress } from "./compress.js";
export { applyTier0, collapseRuns, expandRuns, minifyJson, countTokens } from "./tier0.js";
export type {
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

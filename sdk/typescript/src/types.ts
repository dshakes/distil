/**
 * Message shape types for Anthropic and OpenAI APIs.
 *
 * These are structural (duck-typed) — no SDK imports needed at runtime.
 * Content types cover the shapes distil needs to inspect; fields distil
 * never reads are captured by index signatures so the compiler accepts
 * real SDK objects without casting.
 */

// ── Anthropic ─────────────────────────────────────────────────────────────────

/** A single block inside an Anthropic message's content array. */
export type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_result"; content?: string | AnthropicContentBlock[] }
  | { type: "tool_use" }
  | { type: "image" }
  // provider-opaque (thinking, redacted_thinking, server_tool_use, etc.)
  | { type: Exclude<string, "text" | "tool_result" | "tool_use" | "image"> };

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

/** A single part inside an OpenAI message's content array. */
export type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url" }
  | { type: "tool_result" }
  | { type: Exclude<string, "text" | "image_url" | "tool_result"> };

export interface OpenAIMessage {
  role: "user" | "assistant" | "system" | "tool" | "developer";
  content: string | OpenAIContentPart[] | null;
}

// ── Provider discriminated union ──────────────────────────────────────────────

export type Provider = "anthropic" | "openai";

/** Maps a provider name to its message type. */
export type MessageForProvider<P extends Provider> = P extends "anthropic"
  ? AnthropicMessage
  : OpenAIMessage;

// ── Receipt ───────────────────────────────────────────────────────────────────

/**
 * Content-free accounting receipt returned alongside compressed messages.
 * Contains only token counts — no content, no handles, no user data.
 */
export interface CompressReceipt {
  /** Estimated input tokens before compression. */
  readonly tokensBefore: number;
  /** Estimated input tokens after compression. */
  readonly tokensAfter: number;
  /** Tokens saved (tokensBefore − tokensAfter). */
  readonly tokensSaved: number;
  /**
   * Percent saved. Legitimately near 0 on small payloads — savings come from
   * large tool output; a request that is mostly system prompt has little to fold.
   */
  readonly savedPct: number;
  /** Always "lossless" — the in-process SDK only ships the lossless tier. */
  readonly tier: "lossless";
}

/** Result returned by `compress`. */
export interface CompressResult<M> {
  /** New array of messages; unchanged messages pass through by identity. */
  readonly messages: readonly M[];
  /** Content-free token accounting. */
  readonly receipt: CompressReceipt;
}

// ── Options ───────────────────────────────────────────────────────────────────

export interface CompressOptions {
  /**
   * Which provider's message shape the array uses.
   * Controls which roles are skipped and how content blocks are traversed.
   */
  provider: Provider;
}

// Type definitions for distil-llm.

export interface Message {
  role?: string;
  type?: string;
  content?: unknown;
  [k: string]: unknown;
}

export interface CompressionResult {
  /** New array. Unchanged messages are returned by identity. */
  messages: Message[];
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  /**
   * Percent smaller. Legitimately near 0 on small payloads — savings come from
   * LARGE tool output, and a request that is mostly system prompt has nothing
   * to fold.
   */
  savedPct: number;
  tier: "lossless";
}

export interface BaseURLOptions {
  host?: string;
  port?: number;
}

/**
 * Compress a message list in-process, lossless tier only.
 *
 * Byte-identical to the Python engine (enforced by a conformance suite). Tool
 * and user text get lossless transforms; the model's own turns and non-string
 * content are never rewritten; the input array is never mutated.
 *
 * For the reversible DIGEST tier, route your SDK through a distil proxy with
 * `distilBaseURL()` — that path mints restore handles into the shared store and
 * is the path the decision-equivalence certificate covers.
 */
export function compress(messages: Message[]): CompressionResult;

/** Base URL of a local distil proxy (honours DISTIL_BASE_URL / DISTIL_PORT). */
export function distilBaseURL(opts?: BaseURLOptions): string;
/** Same, with the `/v1` segment OpenAI-style clients expect. */
export function distilOpenAIBaseURL(opts?: BaseURLOptions): string;

export const DEFAULT_PORT: number;
export const DEFAULT_HOST: string;

/** Full Tier-0 pipeline: JSON minify, then run-collapse if it does not inflate. */
export function applyTier0(text: string): string;
/** Run-length-collapse consecutive identical lines to `L\n<<xN>>`. */
export function collapseRuns(text: string): string;
/**
 * Textual inverse of `collapseRuns`.
 * PRECONDITION: the original text contained no line matching /^<<x\d+>>$/ —
 * a generated marker and content that looks like one are indistinguishable.
 * distil's real recovery path is the restore store, not this function.
 */
export function expandRuns(text: string): string;
/** Minify JSON, or null when a byte-identical round-trip is not provable. */
export function minifyJson(text: string): string | null;
/** Heuristic token count, matching the Python HeuristicTokenizer. */
export function countTokens(text: string): number;

export interface SavingsInfo {
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  savedPct: number;
}

export interface DistilMiddlewareOptions {
  /** Called once per request with the measured delta. */
  onSavings?: (info: SavingsInfo) => void;
}

/**
 * Vercel AI SDK middleware that losslessly compresses the prompt in-process.
 *
 * Implements `transformParams` only — compression happens on the way IN, and
 * wrapping the response would mean rewriting the model's own output, which
 * distil does not do.
 *
 *   const model = wrapLanguageModel({ model: ..., middleware: distilMiddleware() });
 *
 * For the reversible digest tier, route through a distil proxy instead.
 */
export function distilMiddleware(
  opts?: DistilMiddlewareOptions
): { transformParams: (arg: { params: unknown }) => Promise<unknown> };

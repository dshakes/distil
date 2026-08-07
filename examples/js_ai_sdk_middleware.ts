/**
 * Vercel AI SDK — compress in-process with middleware.
 *
 *   npm i ai distil-llm
 *   npx tsx examples/js_ai_sdk_middleware.ts
 *
 * Use this when there is no daemon to keep alive: a serverless function, an edge
 * worker, a CI job. For the reversible digest tier, point the provider's baseURL
 * at a distil proxy instead (see examples/js_vercel_ai_sdk.ts).
 */
import { generateText, wrapLanguageModel } from "ai";
import { distilMiddleware } from "distil-llm";

const model = wrapLanguageModel({
  // any AI SDK model — gateway("anthropic/claude-sonnet-5"), openai("gpt-5"), …
  model: "anthropic/claude-sonnet-5" as never,
  middleware: distilMiddleware({
    // In-process compression has no proxy to report through, and savings you
    // cannot see are savings you will not trust.
    onSavings: (s) =>
      console.log(`distil: ${s.tokensBefore} -> ${s.tokensAfter} (${s.savedPct.toFixed(1)}% smaller)`),
  }),
});

const bigLog = Array.from({ length: 400 }, (_, i) => `GET /health 200 ${i}ms`).join("\n");

const { text } = await generateText({
  model,
  messages: [
    { role: "user", content: "Is the service healthy? Here are the last 400 probes." },
    { role: "user", content: bigLog },
  ],
});

console.log(text);

/**
 * What the middleware touches, and what it does not:
 *
 *   compressed  — system strings, user text parts, and tool results in BOTH the
 *                 v5 (`output.value`) and v4 (`result`) shapes
 *   untouched   — assistant messages (the model's own words are not distil's to
 *                 edit), file/image parts, and tool calls
 *
 * It implements `transformParams` only. `wrapGenerate`/`wrapStream` are
 * deliberately absent: compression happens on the way IN, and wrapping the
 * response would mean rewriting the model's output.
 */

/**
 * Anthropic TypeScript SDK + Distil proxy example.
 *
 * Run `distil proxy` first:
 *
 *     distil proxy --port 8788 --upstream https://api.anthropic.com
 *
 * Then the only change vs a normal client is `baseURL`:
 *
 *     npm install @anthropic-ai/sdk
 *     ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_anthropic.ts
 */

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({ baseURL: "http://127.0.0.1:8788" });

async function main() {
  const { data: message, response } = await client.messages
    .create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 64,
      messages: [{ role: "user", content: "Reply with exactly: OK" }],
    })
    .withResponse();

  console.log("x-distil-compressed:", response.headers.get("x-distil-compressed"));
  console.log(message.content[0].type === "text" ? message.content[0].text : message.content[0]);
}

main();

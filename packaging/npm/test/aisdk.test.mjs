// Vercel AI SDK middleware. Exercised against the real LanguageModelV4Prompt
// shapes — system content is a STRING, everything else is an array of parts —
// because a middleware tested only on a shape we invented is untested.
import assert from "node:assert/strict";
import test from "node:test";

import { distilMiddleware } from "../index.js";

const BIG = "ERROR failed to connect to worker pool\n".repeat(60);
const mw = distilMiddleware();

const run = (prompt) => mw.transformParams({ params: { prompt } });

test("compresses tool-result parts (AI SDK v5 output.value shape)", async () => {
  const out = await run([
    {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "c1",
          toolName: "bash",
          output: { type: "text", value: BIG },
        },
      ],
    },
  ]);
  const part = out.prompt[0].content[0];
  assert.ok(part.output.value.length < BIG.length);
  assert.equal(part.toolCallId, "c1", "sibling keys preserved");
  assert.equal(part.output.type, "text");
});

test("compresses tool-result parts (v4 result shape)", async () => {
  const out = await run([
    { role: "tool", content: [{ type: "tool-result", toolCallId: "c1", result: BIG }] },
  ]);
  assert.ok(out.prompt[0].content[0].result.length < BIG.length);
});

test("compresses user text parts", async () => {
  const out = await run([{ role: "user", content: [{ type: "text", text: BIG }] }]);
  assert.ok(out.prompt[0].content[0].text.length < BIG.length);
});

test("compresses a system message, whose content is a bare string", async () => {
  const out = await run([{ role: "system", content: BIG }]);
  assert.ok(out.prompt[0].content.length < BIG.length);
});

test("never rewrites assistant messages", async () => {
  const prompt = [{ role: "assistant", content: [{ type: "text", text: BIG }] }];
  const out = await run(prompt);
  assert.equal(out.prompt[0], prompt[0], "assistant message must pass through by identity");
});

test("leaves file and tool-call parts alone", async () => {
  const prompt = [
    {
      role: "user",
      content: [
        { type: "file", mediaType: "image/png", data: "..." },
        { type: "tool-call", toolCallId: "x", toolName: "bash", input: {} },
      ],
    },
  ];
  const out = await run(prompt);
  assert.equal(out.prompt[0], prompt[0]);
});

test("returns params untouched when there is nothing to compress", async () => {
  const params = { prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }] };
  const out = await mw.transformParams({ params });
  assert.equal(out, params, "unchanged params should be the same object");
});

test("tolerates a missing or malformed prompt", async () => {
  assert.deepEqual(await mw.transformParams({ params: {} }), {});
  const weird = { prompt: "not an array" };
  assert.equal(await mw.transformParams({ params: weird }), weird);
  const nulls = { prompt: [null, 42, { role: "user" }] };
  assert.deepEqual((await mw.transformParams({ params: nulls })).prompt, nulls.prompt);
});

test("preserves other params", async () => {
  const out = await mw.transformParams({
    params: { prompt: [{ role: "system", content: BIG }], temperature: 0.2, maxOutputTokens: 99 },
  });
  assert.equal(out.temperature, 0.2);
  assert.equal(out.maxOutputTokens, 99);
});

test("onSavings reports a real measured delta", async () => {
  const seen = [];
  const m = distilMiddleware({ onSavings: (i) => seen.push(i) });
  await m.transformParams({ params: { prompt: [{ role: "system", content: BIG }] } });
  assert.equal(seen.length, 1);
  assert.ok(seen[0].tokensSaved > 0);
  assert.ok(seen[0].savedPct > 50);
  assert.equal(seen[0].tokensBefore - seen[0].tokensAfter, seen[0].tokensSaved);
});

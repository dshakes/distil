# distil-llm

**Compress your AI agent's context — and prove its decisions don't change.**

[distil](https://github.com/dshakes/distil) is a cache-aware, **reversible**
context-compression layer for LLM agents. Unlike every other compressor, it
certifies — per request — that the agent's *next action* is unchanged
(decision-equivalence), with calibrated, honest savings numbers.

This npm package is the JS/TS bridge: a thin CLI shim + proxy helpers. The
compression runs in the distil proxy; your SDK just points its `baseURL` at it.

## Use it (no pip required)

```sh
# route your agent through distil — resolves uv/pipx/pip automatically
npx distil-llm wrap -- claude

# or start the proxy and point any SDK at it
npx distil-llm proxy            # http://127.0.0.1:8788
```

`npx distil-llm` bridges to the distil CLI via `uvx`/`pipx` (install
[uv](https://docs.astral.sh/uv/) once and it just works) — or `pip install
distil-llm` for a permanent install.

## Point your SDK at the proxy

```ts
import Anthropic from "@anthropic-ai/sdk";
import { distilBaseURL } from "distil-llm";

const client = new Anthropic({ baseURL: distilBaseURL() });
```

```ts
import { createAnthropic } from "@ai-sdk/anthropic";
import { distilBaseURL } from "distil-llm";

const anthropic = createAnthropic({ baseURL: distilBaseURL() });
```

```ts
import OpenAI from "openai";
import { distilOpenAIBaseURL } from "distil-llm";

const openai = new OpenAI({ baseURL: distilOpenAIBaseURL() }); // adds /v1
```

`distilBaseURL()` honors `DISTIL_BASE_URL` / `DISTIL_PORT` (defaults
`http://127.0.0.1:8788`).

## Why distil

- **Reversible** — Tier-0 lossless + Tier-1 digests the model can expand back on
  demand. Nothing is silently lost.
- **Proven, per request** — a live decision-equivalence shadow gate, not just
  offline benchmarks.
- **Honest** — savings are calibrated to your billed usage; the community board
  publishes real, auditable numbers.

Full docs: <https://dshakes.github.io/distil/> · License: Apache-2.0

## In-process compression (lossless tier)

```js
import { compress } from "distil-llm";

const r = compress(messages);           // never mutates the input
console.log(`${r.savedPct.toFixed(1)}% smaller`);
const response = await client.messages.create({ model, messages: r.messages });
```

Tool and user text get lossless transforms; **the model's own turns and
non-string content (image blocks, tool_use) are never rewritten**. Unchanged
messages are returned by identity.

This runs entirely in Node — no daemon, no network — and is **byte-identical to
the Python engine**, enforced by a conformance suite (`tests/test_ts_conformance.py`)
that runs both implementations over a shared corpus and fails on any divergence.
Where JS cannot reproduce Python's bytes (Python renders an integral float as
`1.0`, JS as `1`; JS objects hoist integer-like keys) the transform **declines**
rather than emitting output the certificate does not cover.

### Getting the reversible digest tier

`compress()` is lossless-only, on purpose. The digest tier mints restore handles
whose originals must live in one store shared with the proxy and the MCP server,
and it is the tier the decision-equivalence certificate measures — so there is
deliberately no second implementation of it here.

Route through the proxy instead, which is the same engine and the same store:

```js
import Anthropic from "@anthropic-ai/sdk";
import { distilBaseURL } from "distil-llm";

const client = new Anthropic({ baseURL: distilBaseURL() });
```

```bash
npx distil-llm proxy        # or: distil default --always-on
```

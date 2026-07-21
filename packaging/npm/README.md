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

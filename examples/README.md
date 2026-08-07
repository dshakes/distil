# Distil proxy — integration examples

Start the proxy once, then point any SDK at it — no other code changes needed.

```sh
distil proxy --port 8788 --upstream https://api.anthropic.com
```

## SDK → baseURL mapping

| Example file | SDK / framework | Key setting | Value |
|---|---|---|---|
| `python_anthropic.py` | Anthropic Python SDK | `base_url` | `http://127.0.0.1:8788` |
| `js_anthropic.ts` | Anthropic TypeScript SDK | `baseURL` in `new Anthropic({…})` | `http://127.0.0.1:8788` |
| `python_openai.py` | OpenAI Python SDK | `base_url` | `http://127.0.0.1:8788/v1` |
| `python_litellm.py` | LiteLLM | `api_base` | `http://127.0.0.1:8788` |
| `python_gemini.py` | Google Gemini (REST) | proxy `--upstream` | `https://generativelanguage.googleapis.com` |
| `js_vercel_ai_sdk.ts` | Vercel AI SDK (`@ai-sdk/anthropic`) | `baseURL` in `createAnthropic({…})` | `http://127.0.0.1:8788` |
| `js_langchain.ts` | LangChain.js (`@langchain/anthropic`) | `anthropicApiUrl` in `ChatAnthropic({…})` | `http://127.0.0.1:8788` |

## Headless agents (Agent SDK, `claude -p`, CI)

Headless agents honor `ANTHROPIC_BASE_URL`, so `distil wrap` covers them with
zero code changes — it starts the proxy, injects the env var, and tears down
on exit:

```sh
# Claude Code in print mode (scripts, cron, CI)
distil wrap -- claude -p "summarise this diff"

# Claude Agent SDK (Python or TS) — wraps the script that drives the agent
pip install claude-agent-sdk
distil wrap -- python examples/python_claude_agent_sdk.py
```

For a standing deployment (e.g. one proxy per CI runner), skip `wrap` and set
the env var against a long-running proxy instead:

```sh
distil proxy --port 8788 --upstream https://api.anthropic.com &
ANTHROPIC_BASE_URL=http://127.0.0.1:8788 claude -p "…"
```

| Example file | Client | Routing |
|---|---|---|
| `python_claude_agent_sdk.py` | Claude Agent SDK (drives the Claude Code CLI) | `distil wrap -- python …` or `ANTHROPIC_BASE_URL` |

### In-process (no proxy)

Some frameworks let you compress state directly, with no network hop:

| Example file | Framework | Seam |
|---|---|---|
| `python_langgraph.py` | LangGraph | `pre_model_hook=pre_model_hook()` (compresses graph state before the model node) |

```sh
pip install langgraph langchain-anthropic
ANTHROPIC_API_KEY=sk-ant-… python examples/python_langgraph.py
```

## Running the examples

### Python

```sh
# Install the SDK you want to use (Distil itself needs no extras)
pip install anthropic           # for python_anthropic.py
pip install openai              # for python_openai.py
pip install litellm             # for python_litellm.py
pip install claude-agent-sdk    # for python_claude_agent_sdk.py

# Start the proxy (in a separate terminal)
distil proxy --port 8788 --upstream https://api.anthropic.com

# Run the example
ANTHROPIC_API_KEY=sk-ant-… python examples/python_anthropic.py
```

### TypeScript / Node

```sh
# Install dependencies
npm install @anthropic-ai/sdk             # for js_anthropic.ts
npm install @ai-sdk/anthropic ai          # for js_vercel_ai_sdk.ts
npm install @langchain/anthropic          # for js_langchain.ts
npm install -D tsx                        # TypeScript runner

# Start the proxy (in a separate terminal)
distil proxy --port 8788 --upstream https://api.anthropic.com

# Run an example
ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_vercel_ai_sdk.ts
ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_langchain.ts
```

## How the proxy works

The proxy is a local HTTP server (`distil proxy`, default `http://127.0.0.1:8788`).
It intercepts `/v1/messages`, `/v1/chat/completions`, and `/v1/responses` requests,
compresses the `messages` array (lossless Tier-0 + reversible Tier-1 digests), then
forwards the smaller payload to the real upstream. All other paths pass through
unchanged. Your API key travels in the request headers exactly as normal — it is
never logged or stored by the proxy.

Two extra response headers are added for observability:

| Header | Meaning |
|---|---|
| `x-distil-compressed: 1` | Compression was applied this turn |
| `x-distil-tokens-saved: <n>` | Estimated input tokens saved |

With `distil proxy --session-delta` (cross-turn cache-delta coding), four more headers expose the cache picture:

| Header | Meaning |
|---|---|
| `x-distil-cache-prefix-msgs: <n>` | Leading messages left byte-identical vs the previous turn (the prompt-cache-read region) |
| `x-distil-cache-refs: <n>` | Blocks replaced by a back-reference (exact or delta) |
| `x-distil-cache-delta: <n>` | Of those, how many were cross-version diffs (re-read-after-edit) |
| `x-distil-cache-tokens-saved: <n>` | Volatile (fresh-billed) tokens removed by delta coding |

## Library API (no proxy, no daemon)

| Example | Shows |
|---|---|
| [`python_library.py`](python_library.py) | `compress_messages` / `expand_handle`, byte-exact recovery across processes |
| [`js_library.ts`](js_library.ts) | the same in TypeScript — byte-identical to the Python engine |
| [`js_ai_sdk_middleware.ts`](js_ai_sdk_middleware.ts) | Vercel AI SDK `wrapLanguageModel` middleware |
| [`python_agno.py`](python_agno.py) | Agno — message list or a wrapped model |
| [`python_strands.py`](python_strands.py) | Strands — content blocks and tool results |

The in-process libraries are **lossless-tier only**, deliberately: the reversible
digest mints restore handles that must share one store with the proxy and the MCP
server, and it is the tier the decision-equivalence certificate measures. Route
through a proxy when you want the digest.

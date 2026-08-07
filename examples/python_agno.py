"""Agno — compress an agent's messages in-process.

    python examples/python_agno.py

distil never imports agno: the integration is duck-typed, which is what keeps
distil a zero-dependency install and means an Agno release cannot break it. This
example therefore runs with agno NOT installed.
"""

from __future__ import annotations

from distil.integrations.agno import compress_messages, compressed_model

LOG = "\n".join(f"GET /health 200 {i}ms" for i in range(300))

# 1. Compress a message list you already hold.
msgs = [
    {"role": "user", "content": "is the service healthy?"},
    {"role": "tool", "content": LOG},
]
out = compress_messages(msgs)
print(f"tool result: {len(LOG):,} -> {len(out[1]['content']):,} chars")

# 2. Or wrap the model so every call is compressed on the way out. The wrapper is
#    a transparent proxy: every attribute except the invocation methods is
#    delegated untouched, so the object still behaves as Agno expects.
#
#     from agno.agent import Agent
#     from agno.models.openai import OpenAIChat
#     agent = Agent(model=compressed_model(OpenAIChat(id="gpt-5")))


class _DemoModel:  # stands in for an Agno model so the example runs anywhere
    id = "demo"

    def invoke(self, messages, **kw):
        return sum(len(m.get("content", "")) for m in messages)


wrapped = compressed_model(_DemoModel())
print(f"delegated attribute still works: model.id = {wrapped.id!r}")
print(f"chars actually sent to the model: {wrapped.invoke(msgs):,} (uncompressed: {len(LOG):,})")

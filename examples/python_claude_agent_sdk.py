"""Claude Agent SDK (headless agent) + Distil example.

The Agent SDK drives the bundled Claude Code CLI, which honors
``ANTHROPIC_BASE_URL`` — so the whole agent routes through Distil with zero
code changes. Easiest: let ``distil wrap`` start the proxy and inject the env
var around your script:

    pip install claude-agent-sdk
    distil wrap -- python examples/python_claude_agent_sdk.py

Or point at a standing proxy instead (e.g. CI):

    distil proxy --port 8788 --upstream https://api.anthropic.com &
    ANTHROPIC_BASE_URL=http://127.0.0.1:8788 python examples/python_claude_agent_sdk.py

Works with both API-key and Pro/Max subscription (OAuth) auth.
"""

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query


async def main() -> None:
    options = ClaudeAgentOptions(model="haiku", max_turns=1)
    async for message in query(prompt="Reply with exactly: OK", options=options):
        if type(message).__name__ == "ResultMessage":
            print("result:", message.result)


anyio.run(main)

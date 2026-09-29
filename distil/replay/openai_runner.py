"""Open-model AgentRunner over any OpenAI-compatible endpoint — vLLM, Ollama,
LM Studio, TGI, llama.cpp server, or OpenAI itself.

Lets you run the frontier/coverage harness at scale with **no per-call API cost**:
serve an open model locally (e.g. ``vllm serve meta-llama/Llama-3.1-8B-Instruct``)
and point ``--base-url`` at it. Zero dependencies — uses stdlib ``urllib`` and the
same decision prompt / fingerprint parser as every other runner.

``decide()`` returns the canonical ``{action,target}`` fingerprint; with
``samples>1`` it takes the majority vote, removing the model's run-to-run variance
that would otherwise masquerade as a compression-induced flip.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter

from ..trajectory import Block
from . import prompts


class OpenAIRunner:
    name = "openai"
    # decide() goes through a forced/strict decision tool; ExpandAwareRunner commits through it.
    structured_decision = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key_env: str = "OPENAI_API_KEY",
        samples: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        json_mode: bool = False,
        force_tool: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.samples = max(1, samples)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        # prefer a forced function call (structured target, no paraphrase noise);
        # transparently fall back to free-text+parse if the server rejects tools.
        self.force_tool = force_tool

    def decide(self, blocks: list[Block]) -> str:
        if self.samples == 1:
            return self._sample(blocks)
        votes = Counter(self._sample(blocks) for _ in range(self.samples))
        return votes.most_common(1)[0][0]

    def _sample(self, blocks: list[Block]) -> str:
        if self.force_tool:
            fp = self._tool_decide(blocks)
            if fp is not None:
                return fp  # structured path worked
        system, user = prompts.decision_prompt(blocks)
        return prompts.parse_fingerprint(self._raw(system, user))

    def _tool_decide(self, blocks: list[Block]) -> str | None:
        """Forced-function decision: structured {action,target}, no prose paraphrase.
        Returns None (→ caller falls back to text) if the server can't do tools."""
        system, user = prompts.render(blocks)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{user}\n\nRecord the single next action."},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": prompts.DECISION_TOOL_NAME,
                        "description": prompts.DECISION_TOOL_DESC,
                        "parameters": prompts.DECISION_PARAMS,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": prompts.DECISION_TOOL_NAME}},
        }
        try:
            body = self._post(payload)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None  # tools unsupported / server error → caller uses text path
        calls = (body.get("choices", [{}])[0].get("message", {}) or {}).get("tool_calls") or []
        if not calls:
            return None
        return prompts.fingerprint_from_args(calls[0].get("function", {}).get("arguments", ""))

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST"
        )
        # Retry on rate-limit / transient server errors with exponential backoff,
        # honoring Retry-After when the provider sends it. Rate limits are routine on
        # paid-but-low-tier keys; without this a 429 would crash the whole sweep.
        delay = 2.0
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (configured URL)
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else delay
                    time.sleep(min(wait, 60.0))
                    delay = min(delay * 2, 60.0)
                    continue
                raise
        raise RuntimeError("unreachable")  # loop either returns or re-raises

    def _raw(self, system: str, user: str) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:  # some servers reject this; opt-in
            payload["response_format"] = {"type": "json_object"}
        return self._post(payload).get("choices", [{}])[0].get("message", {}).get("content", "")

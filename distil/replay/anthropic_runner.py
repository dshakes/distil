"""Live AgentRunner backed by the Claude API — for billing-grade certification.

`decide()` renders a trajectory's blocks into a real Messages request and forces
a single structured decision via a strict tool, returning a canonical fingerprint
(action + target) of what the agent chose. Certification then compares that
fingerprint with and without compression — exactly as the offline
DeterministicRunner does, but against the real model.

Requires the `anthropic` SDK and credentials. Imported lazily so the core stays
dependency-free. NOTE: not exercised in this repo's offline test suite (no API
key); treat live results as UNVERIFIED until you run them against your account.
"""

from __future__ import annotations

from typing import Any

from ..trajectory import Block, Kind, Stability
from . import prompts

_DECISION_TOOL = {
    "name": prompts.DECISION_TOOL_NAME,
    "description": prompts.DECISION_TOOL_DESC,
    "strict": True,
    "input_schema": prompts.DECISION_PARAMS,
}


class AnthropicRunner:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        client: object | None = None,
        max_tokens: int = 4096,
        samples: int = 1,
        max_calls: int | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        # Hard ceiling on live API calls for unattended runs (nightly CI): the
        # run fails loudly when the budget is hit instead of spending silently.
        self.max_calls = max_calls
        self.calls_made = 0
        # Newer models deprecate `temperature`, so we can't pin sampling to 0.
        # Instead, take the MAJORITY decision over `samples` calls — the stable
        # "most-likely action" — which removes the model's own run-to-run variance
        # that would otherwise masquerade as a compression-induced divergence.
        self.samples = max(1, samples)

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ModuleNotFoundError:
                raise SystemExit(
                    "distil: the 'anthropic' package is needed for --runner anthropic "
                    "(live grading).\n"
                    "  install it:  pipx inject distil-llm anthropic   "
                    "(or: pip install anthropic)"
                ) from None
            try:
                self._client = Anthropic()
            except Exception as exc:  # noqa: BLE001 — missing/invalid key, etc.
                raise SystemExit(
                    f"distil: could not initialise the Anthropic client — {exc}\n"
                    "  set your key:  export ANTHROPIC_API_KEY=sk-ant-..."
                ) from None
        return self._client

    def _create(self, **kw: object) -> object:
        """Make a Messages API call, turning any failure (missing key, network,
        rate-limit) into a clean message instead of a raw traceback. SystemExit
        from _ensure_client (no package / no client) passes straight through."""
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise SystemExit(
                f"distil: live-call budget exhausted ({self.calls_made}/{self.max_calls} "
                "API calls) — raise --max-live-calls or shrink the trajectory set."
            )
        self.calls_made += 1
        try:
            return self._ensure_client().messages.create(**kw)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — auth / network / rate-limit
            raise SystemExit(
                f"distil: the Anthropic API call failed — {exc}\n"
                "  set your key:  export ANTHROPIC_API_KEY=sk-ant-..."
            ) from None

    def decide(self, blocks: list[Block]) -> str:
        if self.samples == 1:
            return self._sample(blocks)
        from collections import Counter

        votes = Counter(self._sample(blocks) for _ in range(self.samples))
        return votes.most_common(1)[0][0]

    def _sample(self, blocks: list[Block]) -> str:
        # Stable system/tool context -> system prompt; everything else -> the user turn.
        system_parts = [
            b.text
            for b in blocks
            if b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS)
        ]
        rest = [
            b
            for b in blocks
            if not (b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS))
        ]
        user = "\n\n".join(f"[{b.kind.value}] {b.text}" for b in rest)

        # Vision blocks are rendered as REAL provider content blocks — otherwise
        # the model never sees an image and any "vision certificate" would be
        # grading text ABOUT an image, the shortcut ADR 0004 rules out.
        #
        # INTERLEAVED with each block's own text, not batched ahead of it. The
        # first version hoisted every image to the front of the turn, which
        # severed each screenshot from the caption identifying it ("after the
        # rerun"). The live run caught it immediately: on the corpus's final
        # turn the baseline chose promote_release and the compressed arm chose
        # open_failing_build — a divergence produced by the RENDERING, not by
        # the compression under test. An A/B whose two arms differ in prompt
        # SHAPE is not measuring compression at all.
        has_media = any(b.media for b in rest)
        content: list[dict[str, Any]] = []
        if has_media:
            for b in rest:
                content.append({"type": "text", "text": f"[{b.kind.value}] {b.text}"})
                for item in b.media or ():
                    if isinstance(item, dict) and item.get("type") == "image":
                        content.append(item)
            content.append(
                {"type": "text", "text": "\nRecord the single next action you would take."}
            )

        # Constrain `action` to the tools the context actually declares — the
        # grader must pick from the same menu the agent would (kills free-typed
        # action paraphrases that register as false decision changes). Falls
        # back to the free-string schema when no declarations parse.
        decision_tool = _DECISION_TOOL
        actions = prompts.available_actions(blocks)
        if actions:
            import copy

            from typing import cast

            decision_tool = copy.deepcopy(_DECISION_TOOL)
            schema = cast("dict[str, Any]", decision_tool["input_schema"])
            schema["properties"]["action"]["enum"] = actions

        resp = self._create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="\n\n".join(system_parts) or "You are an autonomous agent.",
            tools=[decision_tool],
            tool_choice={"type": "tool", "name": "record_decision"},
            messages=[
                {
                    "role": "user",
                    # A plain string when there is no media, so the text-only
                    # path is byte-identical to what it sent before this change
                    # and every existing certificate stays comparable.
                    "content": (
                        content
                        if has_media
                        else user + "\n\nRecord the single next action you would take."
                    ),
                }
            ],
        )
        resp_any: Any = resp
        resp_blocks = resp_any.content
        for block in resp_blocks:
            if getattr(block, "type", None) == "tool_use":
                return prompts.fingerprint_from_args(block.input)
        return "<no-decision>"

    def _raw(self, system: str, user: str) -> str:
        """Free-form text completion (no forced tool) — used by the expand loop, which
        needs the model to choose between requesting an expansion and committing."""
        resp = self._create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        resp_any: Any = resp
        content = resp_any.content
        return "".join(
            getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text"
        )

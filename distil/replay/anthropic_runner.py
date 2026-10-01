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

from ..trajectory import Block
from . import prompts

_DECISION_TOOL = {
    "name": prompts.DECISION_TOOL_NAME,
    "description": prompts.DECISION_TOOL_DESC,
    "strict": True,
    "input_schema": prompts.DECISION_PARAMS,
}

# The live certifier. Chosen by benchmarks/model_migration_eval.py on real τ-bench
# traffic (60 held-out cases x 4 reps) against the incumbent claude-opus-4-8: it passed
# every pre-registered gate (expand action-equivalence +3.8 ± 8.3 pts, self-consistency
# 99.2% vs 92.5%) at 58% lower cost per certificate. The PAIR is what was measured —
# the same model at another effort is an unmeasured certifier.
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "low"
# CLI choices for --effort; "none" sends no output_config at all.
EFFORTS = ("low", "medium", "high", "xhigh", "max", "none")


def effort_for(model: str | None, effort: str | None) -> str | None:
    """Resolve a CLI ``--effort`` for *model*: an explicit value wins (``none`` -> send
    no output_config); unset means the measured pair's effort for the default model
    and NO output_config for any other. A blanket ``low`` would 400 every call to a
    model without effort support (claude-haiku-4-5), which is what the nightly gate
    ran on until this default moved."""
    if effort:
        return None if effort == "none" else effort
    return DEFAULT_EFFORT if model in (None, DEFAULT_MODEL) else None


# "effort not given" — distinct from an explicit None (= send no output_config), so an
# omitted effort can be resolved per model by effort_for.
_UNSET = "<unset>"


class AnthropicRunner:
    name = "anthropic"
    # decide() goes through a forced/strict decision tool; ExpandAwareRunner commits through it.
    structured_decision = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
        max_tokens: int = 4096,
        samples: int = 1,
        max_calls: int | None = None,
        effort: str | None = _UNSET,
    ) -> None:
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        # Hard ceiling on live API calls for unattended runs (nightly CI): the
        # run fails loudly when the budget is hit instead of spending silently.
        self.max_calls = max_calls
        self.calls_made = 0
        # Newer models (claude-opus-5-5, claude-sonnet-5-5, claude-fable-5-1)
        # reject forced tool_choice with a 400. Flipped on the first such 400;
        # the decision then goes through tool_choice=auto + the same strict tool.
        self.forced_tool_choice = True
        # Newer models deprecate `temperature`, so we can't pin sampling to 0.
        # Instead, take the MAJORITY decision over `samples` calls — the stable
        # "most-likely action" — which removes the model's own run-to-run variance
        # that would otherwise masquerade as a compression-induced divergence.
        self.samples = max(1, samples)
        # output_config.effort (low..max) - the main cost lever on current models.
        # None sends no output_config, i.e. the request is unchanged. Omitted, it is
        # resolved by effort_for: DEFAULT_EFFORT for the default model (the measured
        # pair), none for any other — claude-haiku-4-5 400s on an effort it lacks.
        self.effort = effort_for(model, None) if effort == _UNSET else effort

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
        if self.effort:
            kw.setdefault("output_config", {"effort": self.effort})
        try:
            return self._ensure_client().messages.create(**kw)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — auth / network / rate-limit
            tc = kw.get("tool_choice")
            if (
                isinstance(tc, dict)
                and tc.get("type") == "tool"
                and getattr(exc, "status_code", None) == 400
                and "tool_choice" in str(exc)
            ):
                self.forced_tool_choice = False
                return self._create(**{**kw, "tool_choice": {"type": "auto"}})
            if getattr(exc, "failure_class", None):
                # A classified harness error (e.g. the eval's served-model check) keeps
                # its class; turning it into SystemExit would file it as an API error.
                raise
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
        # The shared split, so DECISION: annotations never reach the model here either.
        system_parts, rest = prompts.split(blocks)
        user = prompts.render(blocks)[1]

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
                content.append(
                    {
                        "type": "text",
                        "text": f"[{b.kind.value}] {prompts.strip_annotations(b.text)}",
                    }
                )
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
            tool_choice=(
                {"type": "tool", "name": "record_decision"}
                if self.forced_tool_choice
                else {"type": "auto"}
            ),
            messages=[
                {
                    "role": "user",
                    # A plain string when there is no media — the text-only shape the
                    # media path was added beside. Not byte-identical to older runs on
                    # every trace: since 1.56.0 DECISION: annotation lines are stripped
                    # (prompts.strip_annotations), so certificates over traces carrying
                    # them are not comparable with ones made before.
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
        # Under tool_choice=auto the model may answer in text instead of calling
        # the tool; the same {action,target} parser the text runners use applies.
        return prompts.parse_fingerprint(
            "".join(
                getattr(b, "text", "") for b in resp_blocks if getattr(b, "type", None) == "text"
            )
        )

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

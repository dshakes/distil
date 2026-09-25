"""Harbor custom agent for the cost-truth benchmark (Python >= 3.12; imported only by Harbor).

``harbor trial start --agent benchmarks.cost_truth.harbor_agent:CostTruthAgent``

It is Harbor's stock ``claude-code`` agent (same pinned Claude Code install, same
bypassPermissions/stream-json flags, same background-task env) with three changes, all
applied identically to every arm:

1. after Claude Code is installed, the arm's tool is installed from the read-only,
   hash-verified host mount (``arms.install_script``) and configured per its docs
   (``arms.agent_setup_script``);
2. Claude Code is launched THROUGH the arm's documented integration, with the neutral
   meter as the last hop (``arms.launch_script``);
3. ``CLAUDE_CONFIG_DIR`` is NOT relocated (stock Harbor points it at the logs dir, which
   silently disables RTK's hook — it writes ``$HOME/.claude/settings.json``); instead the
   session transcript is copied to the logs dir afterwards (``arms.post_run_script``).

The arm, meter URL and mode arrive as agent kwargs (``--agent-kwarg ct_arm=rtk``), not as
agent env: Harbor exports agent env into every exec, which would put the arm's name in
the environment the model's Bash tool can read.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.claude_code import ClaudeCode, ClaudeCodeOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from benchmarks.cost_truth import arms


class CostTruthOptions(ClaudeCodeOptions):
    ct_arm: str = Field(default="control", description="cost-truth arm")
    ct_meter: str = Field(
        default="", description="neutral meter base URL as seen from the container"
    )
    ct_mode: str = Field(default="run", description="run | canary")
    ct_nonce: str = Field(default="", description="canary nonce")


class CostTruthAgent(ClaudeCode):
    options_model = CostTruthOptions
    options: CostTruthOptions

    def __init__(self, *args: Any, version: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, version=version or arms.CLAUDE_CODE_VERSION, **kwargs)
        if self.options.ct_arm not in arms.ARMS:
            raise ValueError(f"unknown cost-truth arm {self.options.ct_arm!r}")
        if not self.options.ct_meter:
            raise ValueError("ct_meter is required: every arm must be metered")

    @property
    def _logs(self) -> str:
        return self.environment_logs_dir.as_posix()

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)  # pinned Claude Code, Harbor's own installer
        await self.exec_as_root(environment, command=arms.install_script(self.options.ct_arm))
        await self.exec_as_agent(environment, command=arms.agent_setup_script(self.options.ct_arm))

    def _env(self) -> dict[str, str]:
        key = self._get_env("ANTHROPIC_API_KEY") or ""
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return {
            "ANTHROPIC_API_KEY": key,
            **arms.arm_env(self.options.ct_arm, self.options.ct_meter),
        }

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        arm, model = self.options.ct_arm, self._parsed_model_name or self.model_name or ""
        env = self._env()
        if self.options.ct_mode == "canary":
            await self.exec_as_agent(
                environment,
                command=arms.canary_script(arm, model, self._logs, self.options.ct_nonce),
                env=env,
            )
            return
        try:
            await self.exec_as_agent(
                environment,
                command=arms.launch_script(arm, model, self._logs),
                env={**env, "CT_INSTRUCTION": instruction},
            )
        finally:
            await self.exec_as_agent(
                environment, command=arms.post_run_script(arm, self._logs), env=env
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Harbor's transcript parser is a convenience here, never the meter; it must not
        # be able to fail a trial.
        try:
            super().populate_context_post_run(context)
        except Exception as exc:  # noqa: BLE001 - third-party parser, logged not swallowed
            self.logger.warning("cost-truth: transcript parse failed: %s", exc)

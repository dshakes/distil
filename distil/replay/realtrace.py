"""Adapters that load **real agent traces** into Distil's trajectory model.

This is the module that breaks the circularity flagged in ``docs/PAPER_PLAN.md``.
The bundled corpus plants ``DECISION:`` markers that the offline
``DeterministicRunner`` keys on, so "decision-equivalence" there is a tautology.
These adapters instead ingest **τ-bench** and **SWE-bench** trajectories, where:

  * nothing in the context tells the model what to do (no directive/marker), and
  * the decision is the agent's *actual next action* — a tool call (τ-bench) or
    an edit/command (SWE-bench) — which the model must INFER from context.

Graded with ``AnthropicRunner`` (a real model), decision-equivalence becomes a
genuine measurement, not a string-preservation check. The adapters return plain
``CorpusEntry`` objects, so the existing ``conformal.calibrate`` /
``certify`` machinery consumes them unchanged.

Each entry's trajectory carries, per decision point, the **gold action** recorded
in the trace — exposed via :func:`gold_actions` for downstream metrics (model↔gold
agreement, task success) without ever leaking the answer into the model's context.

Native formats accepted (both are common public shapes; parsers are defensive):

  τ-bench   : a JSON list of episodes, each ``{"messages"|"traj": [...], "reward": x}``
              where messages are ``{"role": system|user|assistant|tool, "content": str,
              "tool_calls": [{"function": {"name","arguments"}}]}``.
  SWE-bench : a SWE-agent ``.traj`` ``{"trajectory": [{"action","observation",
              "thought"?}], "info": {"exit_status"?, "resolved"?}}`` plus an optional
              top-level ``problem_statement`` / ``instance_id`` / ``repo``.

A normalized fixture of each (no planted answers) ships under
``benchmarks/fixtures/`` so the harness is exercisable offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..corpus import CorpusEntry
from ..trajectory import Block, Kind, Stability, Trajectory, Turn


@dataclass
class GoldDecision:
    """The action the agent actually took at a decision point, from the trace.

    Used for downstream metrics only — never injected into model context.
    ``fingerprint`` is the canonical ``{action,target}`` JSON the live runner also
    emits, so model↔gold agreement is a direct string compare.
    """

    trajectory_id: str
    turn_index: int
    action: str
    target: str

    @property
    def fingerprint(self) -> str:
        # same canonical form the grader's parse_fingerprint emits, so model↔gold
        # agreement compares like with like (paraphrase of the same tool ≠ mismatch)
        from .prompts import canonical

        return canonical(self.action, self.target)


# in-memory side table: (trajectory_id, turn_index) -> GoldDecision
_GOLD: dict[tuple[str, int], GoldDecision] = {}
# in-memory side table: trajectory_id -> task succeeded? (τ-bench reward / SWE resolved)
_SUCCESS: dict[str, bool] = {}


def success_label(entry: CorpusEntry) -> bool | None:
    """Did this trajectory's task succeed (τ-bench reward>0 / SWE-bench resolved)?
    ``None`` if the trace carried no outcome. Used for the downstream task-success
    metric — never injected into model context."""
    return _SUCCESS.get(entry.trajectory.id)


def gold_actions(entries: list[CorpusEntry]) -> dict[tuple[str, int], GoldDecision]:
    """Return the recorded gold decisions for the given entries (loaded by the
    adapters). Keyed by (trajectory_id, turn_index)."""
    keys = {(e.trajectory.id, t.index) for e in entries for t in e.trajectory.turns}
    return {k: v for k, v in _GOLD.items() if k in keys}


def _register_gold(traj_id: str, turn_index: int, action: str, target: str) -> None:
    _GOLD[(traj_id, turn_index)] = GoldDecision(traj_id, turn_index, action or "", target or "")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _norm_args(arguments) -> str:
    """A tool call's arguments → a single canonical 'target' string (first value,
    or the whole arg blob). Mirrors the {action,target} fingerprint the runner uses."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments.strip()
    if isinstance(arguments, dict) and arguments:
        # primary argument = the first value; stable across reorderings via sorted keys
        first_key = sorted(arguments)[0]
        return str(arguments[first_key])
    return json.dumps(arguments, sort_keys=True) if arguments else ""


def _structural_problems(traj: Trajectory) -> list[str]:
    """Light structural check for REAL traces (no DECISION-marker requirement —
    that requirement is exactly the circularity we are removing)."""
    problems: list[str] = []
    if len(traj.turns) < 1:
        problems.append(f"{traj.id}: no decision points")
    for turn in traj.turns:
        last_nonvol = -1
        first_vol = len(turn.blocks)
        for i, b in enumerate(turn.blocks):
            if b.stability is Stability.VOLATILE:
                first_vol = min(first_vol, i)
            else:
                last_nonvol = max(last_nonvol, i)
        if first_vol < last_nonvol:
            problems.append(f"{traj.id} turn {turn.index}: volatile block precedes a cacheable one")
        if not any(b.stability is Stability.VOLATILE for b in turn.blocks):
            problems.append(
                f"{traj.id} turn {turn.index}: no volatile block (nothing fresh to decide on)"
            )
    return problems


def validate_real(entries: list[CorpusEntry]) -> list[str]:
    out: list[str] = []
    for e in entries:
        out += _structural_problems(e.trajectory)
    return out


# --------------------------------------------------------------------------- #
# τ-bench
# --------------------------------------------------------------------------- #


def _tau_messages(episode: dict) -> list[dict]:
    return episode.get("messages") or episode.get("traj") or episode.get("trajectory") or []


def load_tau_bench(path: str | Path, *, model: str = "claude-opus-4-8") -> list[CorpusEntry]:
    """Load τ-bench episodes into trajectories.

    Each assistant message that issues a tool call is a decision point: the context
    is everything before it (system + tools as a STABLE prefix, prior exchange as
    SETTLING history, the most recent tool/user output as the VOLATILE tail), and
    the gold decision is that tool call's ``{name, primary-arg}``. No marker, no
    directive — the model must read the observation to choose the call.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes = raw if isinstance(raw, list) else raw.get("episodes") or raw.get("results") or [raw]
    entries: list[CorpusEntry] = []

    for ei, ep in enumerate(episodes):
        ep_id = str(ep.get("id") or ep.get("task_id") or ep.get("instance_id") or f"tau-{ei}")
        msgs = _tau_messages(ep)
        if not msgs:
            continue

        reward = ep.get("reward", ep.get("success"))
        if reward is not None:
            _SUCCESS[ep_id] = (
                float(reward) > 0 if isinstance(reward, (int, float)) else bool(reward)
            )

        system_text = ""
        tools_text = ""
        for m in msgs:
            if m.get("role") == "system" and not system_text:
                system_text = m.get("content") or ""
        if "tools" in ep:
            tools_text = json.dumps(ep["tools"], indent=2)

        turns = []
        history: list[str] = []  # settled exchange text, byte-stable once written
        pending_obs: list[str] = []  # fresh tool/user outputs since the last decision

        decision_no = 0
        for m in msgs:
            role = m.get("role")
            content = m.get("content") or ""
            calls = m.get("tool_calls") or []
            if role == "system":
                continue
            if role in ("user", "tool"):
                if content:
                    pending_obs.append(f"[{role}] {content}")
                continue
            if role == "assistant":
                if calls:
                    fn = calls[0].get("function", calls[0])
                    name = fn.get("name", "")
                    target = _norm_args(fn.get("arguments", {}))
                    blocks: list[Block] = []
                    if system_text:
                        blocks.append(
                            Block(f"{ep_id}:system", Kind.SYSTEM, system_text, Stability.STABLE)
                        )
                    if tools_text:
                        blocks.append(
                            Block(f"{ep_id}:tools", Kind.TOOLS, tools_text, Stability.STABLE)
                        )
                    if history:
                        blocks.append(
                            Block(
                                f"{ep_id}:hist@{decision_no}",
                                Kind.HISTORY,
                                "\n\n".join(history),
                                Stability.SETTLING,
                            )
                        )
                    obs = "\n\n".join(pending_obs) if pending_obs else content or "(continue)"
                    blocks.append(
                        Block(
                            f"{ep_id}:obs@{decision_no}",
                            Kind.TOOL_OUTPUT,
                            obs,
                            Stability.VOLATILE,
                            True,
                        )
                    )
                    turns.append(Turn(decision_no, blocks))
                    _register_gold(ep_id, decision_no, name, target)
                    # settle this exchange into history for subsequent turns
                    if pending_obs:
                        history.extend(pending_obs)
                        pending_obs = []
                    history.append(f"[assistant] called {name}({target})")
                    decision_no += 1
                elif content:
                    pending_obs.append(f"[assistant] {content}")

        if turns:
            traj = Trajectory(id=ep_id, model=model, turns=turns)
            entries.append(CorpusEntry(f"tau::{ep_id}", "tau-bench", ep.get("title", ep_id), traj))
    return entries


# --------------------------------------------------------------------------- #
# SWE-bench (SWE-agent .traj)
# --------------------------------------------------------------------------- #


def _swe_action_fingerprint(action: str) -> tuple[str, str]:
    """A shell/edit action string → (verb, primary-target). E.g.
    'edit 12:14 src/foo.py' → ('edit', 'src/foo.py'); 'python -m pytest' → ('python','-m')."""
    action = (action or "").strip()
    if not action:
        return ("", "")
    parts = action.split()
    verb = parts[0]
    target = next(
        (p for p in parts[1:] if "/" in p or "." in p), parts[1] if len(parts) > 1 else ""
    )
    return (verb, target)


# Fenced blocks of an SWE-agent response. The info string (```bash) is a language tag,
# never part of the command; the ACTION is the LAST block — a response may quote code
# (the issue's snippet, a diff) before it.
_FENCE = re.compile(r"```[ \t]*[\w+.-]*[ \t]*\n(.*?)```", re.S)


def swe_fenced_command(text: str) -> str:
    """First line of the LAST fenced block in *text* (an SWE-agent action), or ``""``."""
    blocks = _FENCE.findall(text or "")
    lines = [ln.strip() for ln in (blocks[-1] if blocks else "").split("\n") if ln.strip()]
    return lines[0] if lines else ""


def _swe_text(content) -> str:
    """A history message's text: a plain string, or the text parts of a content list."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _swe_call(m: dict) -> tuple[str, str, str] | None:
    """(tool, gold target, command line) from an assistant entry's own structured
    action — a function-calling ``tool_calls`` entry, or SWE-agent's ``action`` field —
    or None when it carries neither (then the fenced block in its text is the action)."""
    calls = m.get("tool_calls") or []
    if calls and isinstance(calls[0], dict):
        fn = calls[0].get("function", calls[0])
        name = str(fn.get("name") or "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"command": args}
        if name:
            cmd = args.get("command") if isinstance(args, dict) else None
            if name == "bash" and isinstance(cmd, str):
                line = cmd.strip().split("\n")[0]
                return "bash", line, line
            return name, _norm_args(args), f"{name} {json.dumps(args, sort_keys=True)}"
    action = m.get("action")
    if isinstance(action, str) and action.strip():
        line = action.strip().split("\n")[0]
        return "", "", line  # classified against the menu by the caller
    return None


def _swe_menu(system: str, extra: tuple[str, ...] = ()) -> tuple[str, list[str]]:
    """A TOOLS block of SWE-agent's own commands, from the ``signature:`` lines of its
    system prompt, plus any function-calling tool the run actually called (*extra*), as
    ``- name(args)`` lines (so ``prompts.available_actions`` parses them), plus ``bash``
    for every other shell command the agent may run."""
    names: list[str] = []
    lines = ["AVAILABLE TOOLS (SWE-agent commands, from the system prompt's signatures):"]
    for sig in re.findall(r"signature:\s*(.+)", system):
        parts = sig.split()
        if not parts or parts[0] in names:
            continue
        names.append(parts[0])
        lines.append(f"- {parts[0]}({' '.join(parts[1:])})")
    for name in extra:
        if name not in names and name != "bash":
            names.append(name)
            lines.append(f"- {name}(arguments): a tool this run called")
    lines.append("- bash(command): any other shell command (python, pytest, ls, cd, grep, rm, ...)")
    names.append("bash")
    return "\n".join(lines), names


def _swe_history_turns(raw: dict, inst: str) -> list[tuple[list[Block], str, str]]:
    """Decision points of a current SWE-agent ``.traj`` (``history`` key), as
    ``[(blocks, gold_action, gold_target)]``.

    Rebuilt from ``history`` — the exact messages the agent saw, in order — because the
    ``trajectory`` steps are the wrong unit: a step is action→observation, so the
    action PRODUCED its step's observation rather than answering it. Demo messages
    (``is_demo``) are the few-shot example, not this run, and are skipped.

    The decision at an assistant message sees everything before it: the system prompt,
    the command menu and the issue (STABLE), then every earlier (action, observation)
    in order (SETTLING), the newest observation last (VOLATILE). An observation is a
    ``user`` message after the issue, or a ``tool`` message (function-calling runs).
    Its gold is that assistant message's own command — from its ``tool_calls`` or
    ``action`` field when it has one, else the last fenced block of its text: an
    SWE-agent command by name, anything else as ``bash`` with the whole command line
    as target. A function-calling action is not in the message text, so it is appended
    to the agent's block as a fenced line: the agent saw its own call. The first action
    (issue only, no observation yet) is not a decision point.
    """
    h = [m for m in raw.get("history") or [] if isinstance(m, dict) and not m.get("is_demo")]
    system = next((_swe_text(m.get("content")) for m in h if m.get("role") == "system"), "")
    msgs = [m for m in h if m.get("role") in ("user", "assistant", "tool")]
    if not msgs or msgs[0]["role"] != "user":
        return []
    called = [c[0] for m in msgs if m["role"] == "assistant" and (c := _swe_call(m)) and c[0]]
    menu, names = _swe_menu(system, tuple(dict.fromkeys(called)))
    stable: list[Block] = []
    if system:
        stable.append(Block(f"{inst}:system", Kind.SYSTEM, system, Stability.STABLE, True))
    stable.append(Block(f"{inst}:tools", Kind.TOOLS, menu, Stability.STABLE, True))
    stable.append(
        Block(f"{inst}:task", Kind.USER, _swe_text(msgs[0].get("content")), Stability.STABLE, True)
    )
    out: list[tuple[list[Block], str, str]] = []
    prior: list[Block] = []
    for j, m in enumerate(msgs[1:], start=1):
        text = _swe_text(m.get("content"))
        if m["role"] == "assistant":
            call = _swe_call(m)
            if call is not None and call[0]:
                verb, target, line = call
            else:
                line = call[2] if call is not None else swe_fenced_command(text)
                v, _, rest = line.partition(" ")
                verb, target = (v, rest) if v in names else ("bash" if line else "", line)
            if prior and msgs[j - 1]["role"] in ("user", "tool"):
                fresh = prior[-1]
                blocks = [b.copy_with(b.text) for b in stable + prior[:-1]]
                blocks.append(
                    Block(fresh.id, Kind.TOOL_OUTPUT, fresh.text, Stability.VOLATILE, True)
                )
                out.append((blocks, verb, target))
            if line and swe_fenced_command(text) != line:
                text = f"{text}\n\n```\n{line}\n```" if text else f"```\n{line}\n```"
            prior.append(Block(f"{inst}:agent@{j}", Kind.HISTORY, text, Stability.SETTLING))
        else:
            prior.append(
                Block(
                    f"{inst}:obs@{j}", Kind.TOOL_OUTPUT, text or "(no output)", Stability.SETTLING
                )
            )
    return out


def load_swe_bench(path: str | Path, *, model: str = "claude-opus-4-8") -> list[CorpusEntry]:
    """Load SWE-agent ``.traj`` trajectories (single file or a directory of them).

    Two shapes. A current SWE-agent ``.traj`` (top-level ``history``) is rebuilt from
    its message history — see :func:`_swe_history_turns`. The older normalized shape
    (top-level ``problem_statement``/``system`` + ``trajectory`` steps, as the bundled
    fixture uses) keeps its original reading: each step is a decision point whose
    context is the problem statement + setup (STABLE), prior steps (SETTLING) and the
    step's observation (VOLATILE), gold the step's ``action``. Resolution status
    (``info.resolved`` / ``exit_status``) is carried for the downstream task-success
    metric.
    """
    p = Path(path)
    files = sorted(p.glob("*.traj")) + sorted(p.glob("*.json")) if p.is_dir() else [p]
    entries: list[CorpusEntry] = []

    # a single file may hold one trajectory (dict) or many (list) — normalize to a list
    raws: list[tuple[dict, str]] = []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(doc, list):
            raws += [(d, f.stem) for d in doc]
        else:
            raws.append((doc, f.stem))

    for raw, stem in raws:
        inst = str(raw.get("instance_id") or raw.get("id") or stem)
        problem = raw.get("problem_statement") or raw.get("issue") or ""
        setup = raw.get("system") or raw.get("setup") or ""
        steps = raw.get("trajectory") or raw.get("steps") or []
        info = raw.get("info") or {}
        resolved = bool(
            info.get("resolved", info.get("exit_status") == "submitted" and info.get("submission"))
        )

        stable: list[Block] = []
        if setup:
            stable.append(Block(f"{inst}:system", Kind.SYSTEM, setup, Stability.STABLE))
        if problem:
            stable.append(
                Block(f"{inst}:problem", Kind.SYSTEM, f"ISSUE:\n{problem}", Stability.STABLE)
            )

        turns = []
        if raw.get("history"):
            # Current SWE-agent shape: its top-level keys are environment/trajectory/
            # history/info, so the problem/system lookups above find nothing — and the
            # step loop below would pair each observation with the action that produced
            # it. The history holds the real conversation; read that instead.
            for di, (blocks, verb, target) in enumerate(_swe_history_turns(raw, inst)):
                turns.append(Turn(di, blocks))
                _register_gold(inst, di, verb, target)
            if turns:
                steps = []  # else fall through to the step reading, never a silent []
        history: list[str] = []
        for si, step in enumerate(steps):
            action = step.get("action") or ""
            obs = step.get("observation") or ""
            verb, target = _swe_action_fingerprint(action)
            blocks = [b.copy_with(b.text) for b in stable]
            if history:
                blocks.append(
                    Block(
                        f"{inst}:hist@{si}", Kind.HISTORY, "\n\n".join(history), Stability.SETTLING
                    )
                )
            blocks.append(
                Block(
                    f"{inst}:obs@{si}",
                    Kind.TOOL_OUTPUT,
                    obs or "(no observation)",
                    Stability.VOLATILE,
                    True,
                )
            )
            turns.append(Turn(si, blocks))
            _register_gold(inst, si, verb, target)
            if obs:
                history.append(f"[observation@{si}] {obs[:2000]}")
            history.append(f"[action@{si}] {action}")

        if turns:
            traj = Trajectory(id=inst, model=model, turns=turns)
            title = f"{inst} ({'resolved' if resolved else 'unresolved'})"
            entries.append(CorpusEntry(f"swe::{inst}", "swe-bench", title, traj))
            _GOLD[(inst, -1)] = GoldDecision(
                inst, -1, "RESOLVED" if resolved else "UNRESOLVED", inst
            )
            _SUCCESS[inst] = resolved
    return entries


def resolved_status(entry: CorpusEntry) -> bool | None:
    """SWE-bench only: did this trajectory resolve the issue (from the trace)? None
    if unknown / not a SWE entry."""
    g = _GOLD.get((entry.trajectory.id, -1))
    return None if g is None else (g.action == "RESOLVED")

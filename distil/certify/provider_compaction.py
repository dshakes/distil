"""Certify PROVIDER-native context manipulation — does it change the agent's decision?

distil's shadow/certificate machinery answers "does *distil's* compression change
the agent's next action?". This module points the same machinery at the thing no
vendor will ever measure about itself: **Anthropic context editing**
(``context-management-2025-06-27``), which silently clears old tool results from
the conversation before the model sees it. Every customer of the feature is
exposed to it; none of them get a decision-equivalence number for it.

The A/B is provider-manipulation-ON vs provider-manipulation-OFF on the *same*
multi-turn tool transcript, same model:

  * **baseline** — the transcript as-is (no ``context_management``).
  * **edited**   — identical request plus a ``clear_tool_uses_20250919`` edit with
    an aggressive trigger, so the server actually clears tool results.
  * **A/A**      — a second independent baseline. Without this arm the sampling
    noise floor is unknown and any decision-change rate is uninterpretable —
    exactly the mistake that made shadow v2 read 38% noise (see shadow.SIG_VERSION).

Ground truth on firing: the response's ``context_management.applied_edits`` field
reports what the server cleared. A case where editing never fired measures
nothing and is **excluded** from the A/B sample (counted separately) — without
that gate the sample is noise.

Non-negotiables, enforced in code, not intention:

  * hard live-call budget (``max_calls`` — SystemExit, same as AnthropicRunner);
  * majority-of-k voting per arm;
  * the protocol (n, α, δ, votes, trigger, model) is written to disk **before**
    the first API call — deciding n after seeing results is the failure mode
    this project criticises in others;
  * the negative result is a result: the report states the bound either way.

The same design runs against **OpenAI server-side compaction** (Responses API
``context_management`` with a ``compact_threshold``; firing ground-truthed by
the ``compaction`` output item) via :class:`OpenAIArms` — only the wire format
and the firing signal differ.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..conformal import certified_risk_bound, hb_pvalue
from ..shadow import SIG_VERSION, decision_signature

CONTEXT_MGMT_BETA = "context-management-2025-06-27"


# --------------------------------------------------------------------------- #
# Episodes -> native Anthropic multi-turn tool transcripts
# --------------------------------------------------------------------------- #
#
# Context editing clears tool_use/tool_result PAIRS from a real messages array —
# a transcript flattened into one user turn (what AnthropicRunner.decide sends)
# has nothing to clear, and the experiment is vacuous. So this harness converts
# tau-bench-shaped episodes into wire-format Anthropic messages instead.


@dataclass(frozen=True)
class Case:
    """One decision point: the transcript history, ending on a user turn, plus
    the tool menu. The model's next action (first tool_use, or text) is the
    decision compared across arms."""

    case_id: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """A tau-bench-shaped JSON list of episodes (see replay.realtrace for the
    accepted native format). Defensive: non-list or non-dict entries are dropped."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _tool_schema(name: str, arg_names: list[str]) -> dict[str, Any]:
    props = {a: {"type": "string"} for a in arg_names}
    return {
        "name": name,
        "description": f"Call the {name} tool.",
        "input_schema": {"type": "object", "properties": props},
    }


def _tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            obj = json.loads(arguments)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def to_case(episode: dict[str, Any], index: int) -> Case | None:
    """Convert one episode into a :class:`Case`, or None when it can't carry the
    experiment (no tool history, or no valid decision point).

    The history is cut *before the last assistant message* — that message is the
    trace's own next action, and the model must re-infer it from context. The
    remaining transcript must still contain at least one tool_use/tool_result
    pair, otherwise there is nothing for the provider to clear.
    """
    raw = episode.get("messages") or episode.get("traj")
    if not isinstance(raw, list) or not raw:
        return None

    # Cut at the decision point: drop the final assistant turn (and anything after).
    last_assistant = max(
        (i for i, m in enumerate(raw) if isinstance(m, dict) and m.get("role") == "assistant"),
        default=-1,
    )
    if last_assistant <= 0:
        return None
    history = raw[:last_assistant]

    system = ""
    messages: list[dict[str, Any]] = []
    tool_names: dict[str, list[str]] = {}
    call_no = 0
    pending_ids: list[str] = []  # tool_use ids awaiting a tool_result

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            system = "\n\n".join(p for p in (system, text) if p)
        elif role == "user":
            messages.append({"role": "user", "content": text or "(continue)"})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name")
                if not name:
                    continue
                call_no += 1
                cid = f"toolu_{call_no:03d}"
                pending_ids.append(cid)
                inp = _tool_input(fn.get("arguments"))
                tool_names.setdefault(name, sorted(inp.keys()))
                blocks.append({"type": "tool_use", "id": cid, "name": name, "input": inp})
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            if not pending_ids:
                continue  # orphan result — a tool_result must pair with a tool_use
            cid = pending_ids.pop(0)
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": cid, "content": text}],
                }
            )

    # A dangling tool_use without its result is an API error — trim it.
    while messages and pending_ids:
        tail = messages[-1]
        tail_blocks = tail.get("content")
        if (
            tail.get("role") == "assistant"
            and isinstance(tail_blocks, list)
            and any(b.get("type") == "tool_use" and b.get("id") in pending_ids for b in tail_blocks)
        ):
            messages.pop()
            pending_ids = []
        else:
            break
    if pending_ids:
        return None  # a dangling tool_use survived past a non-assistant tail — unusable

    has_pair = any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
        for m in messages
    )
    if not messages or not has_pair or messages[-1].get("role") != "user":
        return None

    # Tool menu: declared episode tools first, then any name only seen in calls.
    tools: list[dict[str, Any]] = []
    for t in episode.get("tools") or []:
        if isinstance(t, dict) and t.get("name"):
            tools.append(_tool_schema(t["name"], list(t.get("args") or [])))
            tool_names.pop(t["name"], None)
    for name, args in tool_names.items():
        tools.append(_tool_schema(name, args))

    return Case(
        case_id=str(episode.get("id", index)),
        system=system or "You are an autonomous agent. Take the single best next action.",
        messages=messages,
        tools=tools,
    )


# --------------------------------------------------------------------------- #
# The live arms
# --------------------------------------------------------------------------- #


def edit_config(*, trigger_tokens: int, keep_tool_uses: int) -> dict[str, Any]:
    """A ``clear_tool_uses_20250919`` edit aggressive enough to fire on
    modest transcripts (the default trigger is 100k input tokens)."""
    return {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": trigger_tokens},
                "keep": {"type": "tool_uses", "value": keep_tool_uses},
            }
        ]
    }


@dataclass
class ArmObservation:
    signature: str
    fired: bool
    cleared_input_tokens: int


class ProviderArms:
    """Issues the per-arm live calls against **Anthropic context editing**, with
    the same budget discipline as AnthropicRunner: a hard ``max_calls`` ceiling
    that fails loudly."""

    target = "anthropic-context-editing"
    label = "Anthropic context editing"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        client: object | None = None,
        max_tokens: int = 512,
        max_calls: int | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.calls_made = 0

    def edits(self, *, trigger_tokens: int, keep_tool_uses: int) -> Any:
        return edit_config(trigger_tokens=trigger_tokens, keep_tool_uses=keep_tool_uses)

    def _charge(self) -> None:
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise SystemExit(
                f"distil: live-call budget exhausted ({self.calls_made}/{self.max_calls}) "
                "— raise --max-live-calls or shrink the episode set."
            )
        self.calls_made += 1

    # A full run is hundreds of sequential calls over ~an hour; one transient
    # upstream 500 must not burn the whole spend (it did, live, on 2026-07-26 —
    # the SDK's own retries were exhausted). Bounded backoff on top; the final
    # failure is still a loud SystemExit, never a silent skip.
    _RETRIES = 3
    _BACKOFF_S = 8.0

    def _call(self, fn: Any, provider: str) -> Any:
        for attempt in range(1, self._RETRIES + 1):
            try:
                return fn()
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 — auth / network / rate-limit / 5xx
                if attempt == self._RETRIES:
                    raise SystemExit(
                        f"distil: the {provider} API call failed after {attempt} attempts — {exc}"
                    ) from None
                time.sleep(self._BACKOFF_S * attempt)

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise SystemExit(
                    "distil: the 'anthropic' package is needed for certify-provider "
                    "(live A/B).\n  install it:  pipx inject distil-llm anthropic"
                ) from None
            try:
                self._client = Anthropic()
            except Exception as exc:  # noqa: BLE001 — missing/invalid key, etc.
                raise SystemExit(
                    f"distil: could not initialise the Anthropic client — {exc}\n"
                    "  set your key:  export ANTHROPIC_API_KEY=sk-ant-..."
                ) from None
        return self._client

    def observe(self, case: Case, edits: Any | None) -> ArmObservation:
        self._charge()
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": case.system,
            "tools": case.tools,
            "messages": case.messages,
            "betas": [CONTEXT_MGMT_BETA],
        }
        if edits is not None:
            kw["context_management"] = edits
        client = self._ensure_client()
        resp = self._call(lambda: client.beta.messages.create(**kw), "Anthropic")

        body: dict[str, Any] = (
            resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)  # type: ignore[arg-type]
        )
        cm = body.get("context_management") or {}
        applied = cm.get("applied_edits") if isinstance(cm, dict) else None
        cleared = 0
        if isinstance(applied, list):
            cleared = sum(
                int(e.get("cleared_input_tokens", 0)) for e in applied if isinstance(e, dict)
            )
        return ArmObservation(
            signature=decision_signature(body),
            fired=bool(applied) and cleared > 0,
            cleared_input_tokens=cleared,
        )

    def majority(self, case: Case, edits: Any | None, votes: int) -> ArmObservation:
        obs = [self.observe(case, edits) for _ in range(max(1, votes))]
        sig = Counter(o.signature for o in obs).most_common(1)[0][0]
        return ArmObservation(
            signature=sig,
            fired=any(o.fired for o in obs),
            cleared_input_tokens=max(o.cleared_input_tokens for o in obs),
        )


def _responses_input(case: Case) -> list[dict[str, Any]]:
    """The Case's Anthropic wire messages, mapped to Responses API input items:
    tool_use -> function_call, tool_result -> function_call_output. One converter
    (episode -> Case) feeds both providers; this mapping is deterministic."""
    items: list[dict[str, Any]] = []
    for m in case.messages:
        content = m["content"]
        if isinstance(content, str):
            items.append({"role": m["role"], "content": content})
            continue
        for b in content:
            if b["type"] == "text":
                items.append({"role": m["role"], "content": b["text"]})
            elif b["type"] == "tool_use":
                items.append(
                    {
                        "type": "function_call",
                        "call_id": b["id"],
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    }
                )
            elif b["type"] == "tool_result":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": b["tool_use_id"],
                        "output": str(b.get("content", "")),
                    }
                )
    return items


def _responses_tools(case: Case) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object"}),
        }
        for t in case.tools
    ]


def _responses_signature(output: list[Any]) -> str:
    """Map Responses output items onto the shadow signature by rebuilding the
    Chat Completions shape — same normalization, same tool:/text/none space."""
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            chat = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": item.get("name"),
                                        "arguments": item.get("arguments"),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            return decision_signature(chat)
    if any(isinstance(i, dict) and i.get("type") == "message" for i in output):
        return "text"
    return "none"


class OpenAIArms(ProviderArms):
    """The same three-arm experiment against **OpenAI server-side compaction**
    (Responses API ``context_management`` with a ``compact_threshold``).

    Firing ground truth: a ``{"type": "compaction", "encrypted_content": ...}``
    item in the response output. The compaction payload is opaque, so unlike
    Anthropic there is no cleared-token count — ``cleared_input_tokens`` stays 0
    and firing is the item's presence alone.
    """

    target = "openai-compaction"
    label = "OpenAI server-side compaction"

    def __init__(
        self,
        model: str = "gpt-5.2",
        client: object | None = None,
        max_tokens: int = 512,
        max_calls: int | None = None,
    ) -> None:
        super().__init__(model=model, client=client, max_tokens=max_tokens, max_calls=max_calls)

    def edits(self, *, trigger_tokens: int, keep_tool_uses: int) -> Any:
        # keep_tool_uses has no OpenAI analog — compaction summarizes, not clears.
        return [{"type": "compaction", "compact_threshold": trigger_tokens}]

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise SystemExit(
                    "distil: the 'openai' package is needed for --provider openai "
                    "(live A/B).\n  install it:  pipx inject distil-llm openai"
                ) from None
            try:
                self._client = OpenAI()
            except Exception as exc:  # noqa: BLE001 — missing/invalid key, etc.
                raise SystemExit(
                    f"distil: could not initialise the OpenAI client — {exc}\n"
                    "  set your key:  export OPENAI_API_KEY=sk-..."
                ) from None
        return self._client

    def observe(self, case: Case, edits: Any | None) -> ArmObservation:
        self._charge()
        kw: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_tokens,
            "instructions": case.system,
            "tools": _responses_tools(case),
            "input": _responses_input(case),
        }
        if edits is not None:
            kw["context_management"] = edits
        client = self._ensure_client()
        resp = self._call(lambda: client.responses.create(**kw), "OpenAI")

        body: dict[str, Any] = (
            resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)  # type: ignore[arg-type]
        )
        output = body.get("output") or []
        fired = any(isinstance(i, dict) and i.get("type") == "compaction" for i in output)
        return ArmObservation(
            signature=_responses_signature(output),
            fired=fired,
            cleared_input_tokens=0,  # opaque compaction item — no cleared-token report
        )


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #


@dataclass
class CaseResult:
    case_id: str
    baseline_sig: str
    edited_sig: str
    aa_sig: str
    fired: bool
    cleared_input_tokens: int

    @property
    def ab_loss(self) -> float:
        return 1.0 if self.edited_sig != self.baseline_sig else 0.0

    @property
    def aa_loss(self) -> float:
        return 1.0 if self.aa_sig != self.baseline_sig else 0.0


@dataclass
class ProviderCompactionReport:
    protocol: dict[str, Any]
    protocol_sha256: str
    cases_total: int
    cases_fired: int
    ab_change_rate: float  # over FIRED cases only
    aa_change_rate: float  # over all cases — the sampling-noise floor
    adjusted_change_rate: float
    risk_bound: float  # (1-delta) upper bound on the raw A/B rate, fired cases
    certified: bool  # P(change rate <= alpha) >= 1-delta held on fired cases
    p_value: float
    underpowered: bool
    calls_made: int
    total_cleared_input_tokens: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def statement(self) -> str:
        label = self.protocol.get("label", "Provider context manipulation")
        if self.underpowered:
            return (
                f"UNDERPOWERED: {label} fired on only {self.cases_fired}/"
                f"{self.cases_total} cases — the A/B sample cannot support a verdict. "
                "Lower --trigger-tokens / --keep, or use longer transcripts."
            )
        a = self.protocol["alpha"]
        d = self.protocol["delta"]
        head = (
            f"{label} changed the agent's decision on "
            f"{self.ab_change_rate * 100:.1f}% of {self.cases_fired} fired cases "
            f"(A/A noise floor {self.aa_change_rate * 100:.1f}%, adjusted "
            f"{self.adjusted_change_rate * 100:.1f}%). "
            f"({1 - d:.0%}-confidence upper bound on the raw rate: "
            f"{self.risk_bound * 100:.1f}%.)"
        )
        if self.certified:
            return head + (
                f" CERTIFIED decision-safe at α={a}: the change rate is ≤ {a * 100:.1f}% "
                f"with {(1 - d) * 100:.0f}% confidence."
            )
        return head + f" NOT certified at α={a} — publish this number either way."


def preregister(
    out_dir: Path,
    *,
    target: str,
    label: str,
    model: str,
    n_cases: int,
    votes: int,
    alpha: float,
    delta: float,
    trigger_tokens: int,
    keep_tool_uses: int,
    max_calls: int | None,
) -> tuple[dict[str, Any], str]:
    """Write the protocol to disk BEFORE any live call, and return it with its
    hash. The report embeds both, so a post-hoc n/α/δ change is visible."""
    protocol = {
        "target": target,
        "label": label,
        "model": model,
        "n_cases": n_cases,
        "votes": votes,
        "alpha": alpha,
        "delta": delta,
        "trigger_tokens": trigger_tokens,
        "keep_tool_uses": keep_tool_uses,
        "max_calls": max_calls,
        "sig_version": SIG_VERSION,
        "created": time.time(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(protocol, sort_keys=True, indent=2)
    (out_dir / "protocol.json").write_text(blob + "\n", encoding="utf-8")
    return protocol, hashlib.sha256(blob.encode()).hexdigest()


def run_experiment(
    cases: list[Case],
    arms: ProviderArms,
    out_dir: Path,
    *,
    votes: int = 3,
    alpha: float = 0.1,
    delta: float = 0.05,
    trigger_tokens: int = 500,
    keep_tool_uses: int = 0,
    min_fired: int | None = None,
) -> ProviderCompactionReport:
    """Run the pre-registered three-arm experiment and write ``report.json``.

    Cost = ``3 * votes * len(cases)`` live calls, hard-capped by ``arms.max_calls``.
    """
    protocol, proto_sha = preregister(
        out_dir,
        target=arms.target,
        label=arms.label,
        model=arms.model,
        n_cases=len(cases),
        votes=votes,
        alpha=alpha,
        delta=delta,
        trigger_tokens=trigger_tokens,
        keep_tool_uses=keep_tool_uses,
        max_calls=arms.max_calls,
    )
    edits = arms.edits(trigger_tokens=trigger_tokens, keep_tool_uses=keep_tool_uses)

    results: list[CaseResult] = []
    for case in cases:
        baseline = arms.majority(case, None, votes)
        aa = arms.majority(case, None, votes)
        edited = arms.majority(case, edits, votes)
        results.append(
            CaseResult(
                case_id=case.case_id,
                baseline_sig=baseline.signature,
                edited_sig=edited.signature,
                aa_sig=aa.signature,
                fired=edited.fired,
                cleared_input_tokens=edited.cleared_input_tokens,
            )
        )

    fired = [r for r in results if r.fired]
    n_ab = len(fired)
    ab_rate = sum(r.ab_loss for r in fired) / n_ab if n_ab else 0.0
    aa_rate = sum(r.aa_loss for r in results) / len(results) if results else 0.0
    aa_agreement = 1.0 - aa_rate
    adjusted = (
        max(0.0, 1.0 - min(1.0, (1.0 - ab_rate) / aa_agreement)) if aa_agreement > 0 else ab_rate
    )
    need = min_fired if min_fired is not None else max(1, len(cases) // 2)
    underpowered = n_ab < need
    p = hb_pvalue(ab_rate, n_ab, alpha)
    report = ProviderCompactionReport(
        protocol=protocol,
        protocol_sha256=proto_sha,
        cases_total=len(results),
        cases_fired=n_ab,
        ab_change_rate=ab_rate,
        aa_change_rate=aa_rate,
        adjusted_change_rate=adjusted,
        risk_bound=certified_risk_bound(ab_rate, n_ab, delta),
        certified=(not underpowered) and p <= delta,
        p_value=p,
        underpowered=underpowered,
        calls_made=arms.calls_made,
        total_cleared_input_tokens=sum(r.cleared_input_tokens for r in results),
        results=results,
    )
    payload = asdict(report)
    payload["statement"] = report.statement
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report

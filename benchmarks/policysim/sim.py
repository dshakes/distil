"""Replay one trajectory through one policy and one provider cost model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from distil.adapters.anthropic import _apply_tier0

from .costmodel import TTL_S, Anthropic, Seg, TokenModel, Usage, segments
from .policies import Policy, protected
from .trajectory import Trajectory, block_text, blocks, thinking_key


def allocate_hidden(traj: Trajectory, tm: TokenModel) -> None:
    """Spread the billed output that is not visible text over the thinking blocks.

    Thinking text is omitted from the transcript (only its signature is kept) but is
    re-sent and billed as input on every later request. Billed output tokens = visible
    assistant tokens + thinking tokens, so the thinking total is known exactly per
    trajectory; it is split across blocks in proportion to signature length (the
    signature encrypts the thinking, so it grows with it).
    """
    visible = 0.0
    sigs: list[tuple[str, int]] = []
    for m in traj.messages:
        if m.get("role") != "assistant":
            continue
        for b in blocks(m):
            k = thinking_key(b)
            if k is None:
                visible += tm.count(block_text(b))
            else:
                sigs.append((k, max(1, len(k))))
    hidden = max(0.0, traj.output_tokens - visible)
    total = sum(w for _, w in sigs)
    traj.hidden = {k: round(hidden * w / total) for k, w in sigs} if total else {}


@dataclass
class Info:
    removed: float
    violations: int


@dataclass
class Result:
    usage: Usage
    usd: float
    requests: int
    #: Tokens removed by lossy transforms (beyond distil's lossless Tier-0), per unique
    #: tool result at its most-compressed rendering.
    removed: float
    #: Tool results whose rendering dropped a must-keep (error/failure/summary) line.
    violations: int
    last_request_usd: float
    per_step_output: float
    meta: dict[str, Any] = field(default_factory=dict)


_T0: dict[str, str] = {}


def _tier0(text: str) -> str:
    r = _T0.get(text)
    if r is None:
        if len(_T0) > 50_000:
            _T0.clear()
        r = _T0[text] = _apply_tier0(text)
    return r


def _results(messages: list[Any]) -> dict[str, str]:
    return {
        str(b.get("tool_use_id")): block_text(b)
        for m in messages
        if m.get("role") == "user"
        for b in blocks(m)
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }


def replay(
    traj: Trajectory, policy: Policy, tm: TokenModel, expand_tokens: float = 0.0
) -> tuple[list[list[Seg]], Info]:
    """What the policy sends at each request (as priced segments), plus its fidelity.

    A policy that leaves recovery handles also ships the ``distil_expand`` tool
    definition (``expand_tokens``) in every request, as the harness's distil arm did."""
    overhead = traj.overhead_tokens + (expand_tokens if policy.expand_tool else 0.0)
    policy.reset()
    removed: dict[str, float] = {}
    violated: set[str] = set()
    orig = _results(traj.messages)
    reqs: list[list[Seg]] = []
    for k, end in enumerate(traj.request_ends):
        sent = policy.step(traj.messages[:end], traj.times[k], TTL_S[traj.ttl])
        reqs.append(segments(sent, tm, traj.hidden, overhead))
        for tid, text in _results(sent).items():
            o = orig.get(tid)
            if o is None or text == o:
                continue
            base = _tier0(o)
            if text == base:
                continue
            removed[tid] = max(removed.get(tid, 0.0), tm.count(base) - tm.count(text))
            # A must-keep line counts as kept if its text survives anywhere (a lossless
            # run-collapse marker may follow it); dropped or reworded, it is a violation.
            if tid not in violated and any(
                ln.strip() and protected(ln) and ln.strip() not in text for ln in o.split("\n")
            ):
                violated.add(tid)
    return reqs, Info(sum(max(0.0, v) for v in removed.values()), len(violated))


def price(
    traj: Trajectory,
    reqs: list[list[Seg]],
    info: Info,
    prov: Any,
    tm: TokenModel,
    warm_first: bool = False,
) -> Result:
    prov.reset()
    total = Usage()
    if warm_first:  # another run already sent the identical first request (see calibrate)
        first = traj.messages[: traj.request_ends[0]]
        prov.request(segments(first, tm, traj.hidden, traj.overhead_tokens), -1.0, traj.ttl)
    last = Usage()
    for k, segs in enumerate(reqs):
        last = prov.request(segs, traj.times[k], traj.ttl)
        total += last
    if hasattr(prov, "close"):
        total.storage_usd += prov.close(traj.times[-1] + 60.0)
    n = len(reqs)
    total.output = float(traj.output_tokens)
    return Result(
        usage=total,
        usd=prov.price.usd(total),
        requests=n,
        removed=info.removed,
        violations=info.violations,
        last_request_usd=prov.price.usd(last),
        per_step_output=traj.output_tokens / n if n else 0.0,
    )


def simulate(
    traj: Trajectory,
    policy: Policy,
    prov: Any,
    tm: TokenModel,
    warm_first: bool = False,
    expand_tokens: float = 0.0,
) -> Result:
    reqs, info = replay(traj, policy, tm, expand_tokens)
    return price(traj, reqs, info, prov, tm, warm_first)


def anthropic_for(traj: Trajectory, framing: float) -> Anthropic:
    return Anthropic(model=traj.model.split("@")[0], framing=framing)

"""Fit the offline token model to billed usage, then measure the simulator's error on
held-out trajectories it was not fitted to.

The harness puts one breakpoint on the newest block and every request extends the last,
so under the documented Anthropic rules a trajectory of n requests with prefix sizes
P_1..P_n bills ``cache_write = P_n`` and ``cache_read = P_1 + ... + P_{n-1}``. Each P_k is
linear in three unknowns, given the billed output (which fixes the hidden thinking):

    P_k = O + s * A_k + b * B_k + out * f_k

O = tools+system tokens, s = tokens per regex piece, b = tokens per block, A_k / B_k the
visible pieces / blocks with the assistant's visible output netted out of the thinking
share, f_k the share of the trajectory's thinking (by signature length) already in the
prefix. Two equations per trajectory, weighted by 1/billed, solved by least squares on
the TRAIN half. The TEST half is then replayed through the full simulator (lookback,
minimums, TTL, the same code every policy is priced with) and compared component by
component with what Anthropic billed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any

from distil.expand import EXPAND_TOOL

from .costmodel import TokenModel, Usage, _pieces
from .policies import DistilServed, Policy
from .sim import allocate_hidden, anthropic_for, simulate
from .trajectory import Trajectory, block_text, blocks, thinking_key


def split(t: Trajectory) -> str:
    return "train" if int(hashlib.md5(t.id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def _features(t: Trajectory) -> tuple[list[tuple[float, float, float]], float, float]:
    """Per request (A_k, B_k, f_k); plus the trajectory's assistant pieces / blocks."""
    sig_total = 0
    va = na = 0.0
    for m in t.messages:
        if m.get("role") != "assistant":
            continue
        for b in blocks(m):
            k = thinking_key(b)
            if k is None:
                va += _pieces(block_text(b))
                na += 1
            else:
                sig_total += max(1, len(k))
    rows = []
    pieces = nblk = sig = 0.0
    j = 0
    for end in t.request_ends:
        for m in t.messages[j:end]:
            for b in blocks(m):
                k = thinking_key(b)
                if k is None:
                    pieces += _pieces(block_text(b))
                    nblk += 1
                else:
                    sig += max(1, len(k))
        j = end
        f = sig / sig_total if sig_total else 0.0
        rows.append((pieces - va * f, nblk - na * f, f))
    return rows, va, na


def _solve3(a: list[list[float]], y: list[float]) -> list[float]:
    m = [row[:] + [v] for row, v in zip(a, y)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(3):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * z for x, z in zip(m[r], m[c])]
    return [m[i][3] / m[i][i] for i in range(3)]


def fit(train: list[Trajectory]) -> tuple[TokenModel, float, float]:
    """(token model, overhead tokens, framing tokens per request).

    Fitted on the TOTAL input each trajectory was billed (uncached + cache write + cache
    read, summed over its requests): the docs define that sum as every input token the
    request carried, so it is independent of what happened to be cached — in particular
    of cache entries another arm left warm. The write/read split is then a test of the
    cache model alone.
    """
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    n_req = inp = 0.0
    for t in train:
        assert t.billed is not None
        n_req += len(t.request_ends)
        inp += t.billed["input"]
    framing = inp / max(1.0, n_req)
    for t in train:
        assert t.billed is not None
        rows, _, _ = _features(t)
        n = len(rows)
        billed = t.billed["input"] + t.billed["cache_write"] + t.billed["cache_read"]
        x = (float(n), sum(r[0] for r in rows), sum(r[1] for r in rows))
        y = billed - n * framing - t.output_tokens * sum(r[2] for r in rows)
        w = 1.0 / max(1.0, billed) ** 2
        for i in range(3):
            xty[i] += w * x[i] * y
            for j in range(3):
                xtx[i][j] += w * x[i] * x[j]
    o, s, b = _solve3(xtx, xty)
    return TokenModel(scale=s, per_block=b), o, framing


def _err(sim: float, billed: float) -> float:
    return (sim - billed) / billed if billed else 0.0


def evaluate(
    trajs: list[Trajectory],
    tm: TokenModel,
    framing: float,
    policy: Policy | None = None,
    warm_first: bool = False,
    expand_tokens: float = 0.0,
) -> dict[str, Any]:
    """Simulated vs billed, per component: aggregate error and per-trajectory spread."""
    pol = policy or Policy()
    comp = {"cache_write": 0.0, "cache_read": 0.0, "input": 0.0, "usd": 0.0}
    bill = dict.fromkeys(comp, 0.0)
    per: dict[str, list[float]] = {k: [] for k in comp}
    for t in trajs:
        assert t.billed is not None
        r = simulate(t, pol, anthropic_for(t, framing), tm, warm_first, expand_tokens)
        u = r.usage
        sim = {
            "cache_write": u.write_5m + u.write_1h,
            "cache_read": u.read,
            "input": u.input,
            "usd": r.usd,
        }
        prov = anthropic_for(t, framing)
        bu = Usage(
            input=t.billed["input"],
            write_5m=t.billed["cache_write"],
            read=t.billed["cache_read"],
            output=t.billed["output"],
        )
        b = {
            "cache_write": float(t.billed["cache_write"]),
            "cache_read": float(t.billed["cache_read"]),
            "input": float(t.billed["input"]),
            "usd": prov.price.usd(bu),
        }
        for k in comp:
            comp[k] += sim[k]
            bill[k] += b[k]
            per[k].append(abs(_err(sim[k], b[k])))
    out: dict[str, Any] = {"n": len(trajs)}
    for k in comp:
        xs = sorted(per[k])
        out[k] = {
            "simulated": round(comp[k], 4 if k == "usd" else 0),
            "billed": round(bill[k], 4 if k == "usd" else 0),
            "aggregate_error": round(_err(comp[k], bill[k]), 4),
            "per_traj_median_abs_error": round(statistics.median(xs), 4) if xs else None,
            "per_traj_p90_abs_error": round(xs[int(0.9 * (len(xs) - 1))], 4) if xs else None,
        }
    return out


def expand_tool_tokens(tm: TokenModel) -> float:
    return tm.count(json.dumps(EXPAND_TOOL))


def calibrate(
    lite: list[Trajectory], long: list[Trajectory], distil_arm: list[Trajectory]
) -> tuple[TokenModel, float, float, dict[str, Any]]:
    """Fit on half of the short (Lite) trajectories; test on the other half, on every
    long-horizon plain/rtk trajectory (never fitted: 50-100 steps, adaptive thinking) and
    on the distil arm replayed through the served adapter (a different policy)."""
    train = [t for t in lite if split(t) == "train"]
    test = [t for t in lite if split(t) == "test"]
    tm, overhead, framing = fit(train)
    for t in lite + long:
        t.overhead_tokens = round(overhead)
        allocate_hidden(t, tm)
    for t in distil_arm:
        t.overhead_tokens = round(overhead)
        allocate_hidden(t, tm)
    report: dict[str, Any] = {
        "token_model": {"scale_per_piece": tm.scale, "per_block": tm.per_block},
        "overhead_tokens": round(overhead, 1),
        "framing_tokens_per_request": round(framing, 3),
        "fit_on": f"{len(train)} SWE-bench Lite rtk-arm trajectories (hash split)",
        "lite_train": evaluate(train, tm, framing),
        "lite_test_cold": evaluate(test, tm, framing),
        "lite_test_warm_first": evaluate(test, tm, framing, warm_first=True),
        "long_plain_rtk_heldout": evaluate(long, tm, framing),
    }
    if distil_arm:
        report["long_distil_arm_replay"] = evaluate(
            distil_arm, tm, framing, DistilServed(False), expand_tokens=expand_tool_tokens(tm)
        )
    return tm, overhead, framing, report

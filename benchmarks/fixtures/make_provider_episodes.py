#!/usr/bin/env python3
"""Generate the provider-compaction calibration corpus (tau-bench-shaped).

`distil certify-provider` A/Bs provider context manipulation on multi-turn tool
transcripts. The bundled `tau_bench_sample.json` renders under 1000 tokens, and
OpenAI's server-side compaction enforces `compact_threshold >= 1000` — so it can
NEVER fire there. This corpus is the same design, sized for the experiment:

  * every decision-bearing fact lives in a TOOL RESULT (two rounds: order record
    + billing record), buried in plausible log noise, with NO directive telling
    the model what to do — the next action must be inferred from policy (system)
    + records (tool results). Clearing/compacting the tool results is therefore
    *plausibly* decision-relevant; whether it actually changes decisions is what
    the experiment measures.
  * each episode renders ~1.5-2.5k tokens, comfortably past both providers'
    triggers (Anthropic: configurable; OpenAI: >= 1000).

Deterministic (seeded) so the corpus is reproducible byte-for-byte. Synthetic,
and labeled as such — the controlled-corpus result is a first measurement, not a
production-traffic one.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

SYSTEM = (
    "You are a retail customer-support agent. Resolve each case with exactly one tool call.\n"
    "Policy: refunds only for items returned within 30 days of delivery AND unopened,\n"
    "and only when the payment method has no open chargeback; otherwise offer store\n"
    "credit. Cancel an order only if it has not yet shipped. Escalate fraud whenever the\n"
    "billing record shows a card mismatch or a chargeback velocity of 5 or more."
)

TOOLS = [
    {"name": "issue_refund", "args": ["order_id"]},
    {"name": "offer_store_credit", "args": ["order_id"]},
    {"name": "cancel_order", "args": ["order_id"]},
    {"name": "escalate_fraud", "args": ["order_id"]},
]

# (name, order record, billing record, gold next action) — gold is recorded for
# audit only; the harness never uses it (arms are compared to each other).
SCENARIOS = [
    (
        "refund-ok",
        "status=delivered days_since={d} condition=unopened amount={amt}",
        "method=visa-{last4} chargebacks_open=0 velocity=1 card_mismatch=false",
        "issue_refund",
        (2, 28),
    ),
    (
        "credit-opened",
        "status=delivered days_since={d} condition=opened amount={amt}",
        "method=visa-{last4} chargebacks_open=0 velocity=1 card_mismatch=false",
        "offer_store_credit",
        (2, 28),
    ),
    (
        "credit-late",
        "status=delivered days_since={d} condition=unopened amount={amt}",
        "method=mc-{last4} chargebacks_open=0 velocity=2 card_mismatch=false",
        "offer_store_credit",
        (35, 90),
    ),
    (
        "credit-chargeback",
        "status=delivered days_since={d} condition=unopened amount={amt}",
        "method=amex-{last4} chargebacks_open=1 velocity=2 card_mismatch=false",
        "offer_store_credit",
        (2, 28),
    ),
    (
        "cancel-ok",
        "status=processing shipped=false days_since={d} amount={amt}",
        "method=visa-{last4} chargebacks_open=0 velocity=1 card_mismatch=false",
        "cancel_order",
        (0, 2),
    ),
    (
        "cancel-shipped",
        "status=in_transit shipped=true days_since={d} amount={amt}",
        "method=mc-{last4} chargebacks_open=0 velocity=1 card_mismatch=false",
        "offer_store_credit",
        (1, 5),
    ),
    (
        "fraud-mismatch",
        "status=delivered days_since={d} condition=unopened amount={amt}",
        "method=visa-{last4} chargebacks_open=0 velocity=2 card_mismatch=true",
        "escalate_fraud",
        (2, 28),
    ),
    (
        "fraud-velocity",
        "status=delivered days_since={d} condition=opened amount={amt}",
        "method=mc-{last4} chargebacks_open=2 velocity={vel} card_mismatch=false",
        "escalate_fraud",
        (2, 28),
    ),
]

CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
NODES = ["a03", "b12", "c07", "d21", "e14"]


def _noise(rng: random.Random, stamp: str, lines: int) -> str:
    """Plausible per-request log chatter — foldable, ignorable, and bulky enough
    to carry the transcript past the compaction thresholds."""
    out = []
    for i in range(lines):
        kind = rng.choice(
            [
                f"INFO loaded customer profile; prior_tickets {rng.randint(0, 9)}; csat {rng.randint(30, 50) / 10}",
                f"INFO loyalty {rng.choice(['gold', 'silver', 'none'])}; newsletter {rng.choice(['on', 'off'])}",
                f"DEBUG inventory sync ok; carrier {rng.choice(CARRIERS)}; sla {rng.randint(1, 4)}d",
                f"DEBUG cache {rng.choice(['warm', 'cold'])}; shard {rng.randint(0, 15)}; node {rng.choice(NODES)}",
                f"INFO recommendation model v{rng.randint(2, 9)} scored {rng.randint(100, 999)} SKUs",
                f"DEBUG session token rotated; ttl {rng.randint(300, 3600)}s; region us-east-{rng.randint(1, 2)}",
                f"INFO a/b bucket {rng.choice(['control', 'variant'])}; banner {rng.choice(['spring', 'none'])}",
                f"DEBUG rate limiter ok; remaining {rng.randint(50, 500)}; window 60s",
            ]
        )
        out.append(f"log {stamp}:{i:02d}Z {kind}")
    return "\n".join(out)


# Routine follow-up rounds for the LONG (default-policy) corpus. Their results are
# pure operational noise — nothing decision-bearing. Under Anthropic's DEFAULT
# clearing policy (keep = 3 most recent tool uses) they are exactly what survives,
# while the decision-bearing get_order/get_billing rounds at the head get cleared.
ROUTINE_ROUNDS = [
    ("check_inventory", "in-stock counts by warehouse; restock ETAs"),
    ("get_shipping_events", "carrier scan history; no exceptions recorded"),
    ("get_communications", "email/SMS log; delivery notification sent"),
    ("get_promotions", "active promotions; none applied to this order"),
]


def episode(rng: random.Random, index: int, spec, *, routine_rounds: int = 0) -> dict:
    name, order_tpl, billing_tpl, gold, day_range = spec
    order_id = f"A{2000 + index}"
    d = rng.randint(*day_range)
    amt = f"{rng.randint(9, 400)}.{rng.randint(0, 99):02d}"
    last4 = rng.randint(1000, 9999)
    vel = rng.randint(5, 19)
    order_rec = order_tpl.format(d=d, amt=amt)
    billing_rec = billing_tpl.format(last4=last4, vel=vel)
    order_obs = (
        f"get_order({order_id}) ->\n"
        f"{_noise(rng, '2026-07-26T09:01', 28)}\n"
        f"RECORD {order_rec}\n"
        f"{_noise(rng, '2026-07-26T09:02', 6)}"
    )
    billing_obs = (
        f"get_billing({order_id}) ->\n"
        f"{_noise(rng, '2026-07-26T09:03', 28)}\n"
        f"BILLING {billing_rec}\n"
        f"{_noise(rng, '2026-07-26T09:04', 6)}"
    )
    ep = {
        "id": f"pc-{name}-{index}",
        "title": f"pc-{name}-{index}",
        "reward": 1,
        "gold_action": gold,  # audit only; never sent to the model, never used by the harness
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Please help with order {order_id}."},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "get_order", "arguments": {"order_id": order_id}}}
                ],
            },
            {"role": "tool", "content": order_obs},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "get_billing", "arguments": {"order_id": order_id}}}
                ],
            },
            {"role": "tool", "content": billing_obs},
        ],
    }
    for r in range(routine_rounds):
        tool_name, summary = ROUTINE_ROUNDS[r % len(ROUTINE_ROUNDS)]
        obs = (
            f"{tool_name}({order_id}) -> {summary}\n"
            f"{_noise(rng, f'2026-07-26T09:{10 + r:02d}', 30)}"
        )
        ep["messages"].append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": tool_name, "arguments": {"order_id": order_id}}}
                ],
            }
        )
        ep["messages"].append({"role": "tool", "content": obs})
    ep["messages"].append(
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": gold, "arguments": {"order_id": order_id}}}],
        }
    )
    return ep


def build(n: int = 40, seed: int = 20260726) -> int:
    rng = random.Random(seed)
    eps = [episode(rng, i, SCENARIOS[i % len(SCENARIOS)]) for i in range(n)]
    (HERE / "provider_compaction_episodes.json").write_text(json.dumps(eps, indent=2))
    return len(eps)


def build_long(n: int = 40, seed: int = 20260727) -> int:
    """The default-policy corpus: 2 decision-bearing rounds at the HEAD, then 4
    routine rounds — so Anthropic's default keep=3 clears exactly the records."""
    rng = random.Random(seed)
    eps = [episode(rng, i, SCENARIOS[i % len(SCENARIOS)], routine_rounds=4) for i in range(n)]
    (HERE / "provider_compaction_episodes_long.json").write_text(json.dumps(eps, indent=2))
    return len(eps)


if __name__ == "__main__":
    for builder, fname in (
        (build, "provider_compaction_episodes.json"),
        (build_long, "provider_compaction_episodes_long.json"),
    ):
        n = builder()
        blob = (HERE / fname).read_text()
        print(
            f"{n} episodes, {len(blob)} chars (~{len(blob) // (4 * n)} tokens/episode) -> {fname}"
        )

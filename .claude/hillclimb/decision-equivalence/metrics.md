# decision-equivalence — what each metric means

One case = one trajectory turn (36 curated turns from `corpus/`, 64 synthetic from
`benchmarks/corpus_xl`, one turn each, spread across history depths). Per (case, rep) the
runner asks distil's real `AnthropicRunner` for the agent's next `{action, target}` on
five views of the same context:

| arm | context |
|---|---|
| `full_a`, `full_b` | uncompressed, two independent samples |
| `distil` | `distil.compress.strategies.distil` (cache-aware) |
| `expand` | distil-compressed, graded through `ExpandAwareRunner` (the `distil_expand` recovery loop — how distil deploys) |
| `trunc` | volatile blocks truncated to 160 chars — **positive control** |

An arm that returns no decision never counts as a match.

| metric | = 1 when | reads as |
|---|---|---|
| `equiv_distil_act` (**headline**) | `full_a` and `distil` pick the same action | compression did not change *what the agent does* |
| `equiv_expand_act` | same, for the expand arm | the deployed mode is safe |
| `self_consist_act` | `full_a` and `full_b` pick the same action | noise floor — a divergence rate at or below 1 − this is model noise, not compression |
| `equiv_distil` / `equiv_expand` / `self_consist` | exact `{action,target}` fingerprint equal (distil's production equality) | stricter; also counts target paraphrase as a change — the gap to the `_act` column is paraphrase noise in distil's certifier |
| `decided` | `full_a` produced a parseable decision | certifier works at all on this model |
| `agree_ref_act` / `agree_ref` | `full_a` matches the frozen `claude-opus-4-8` rep-0 decision (on baseline: `full_b`) | how much the new model's choices differ from the incumbent — diagnostic, **not** a correctness score (the incumbent is not ground truth) |
| `trunc_detect` | the truncation control changed the action | the eval can see damage on this model; near 0 means a flip rate of 0 would be uninformative |

Perf: `latency_s` (sum of API time across the case's ~5 calls), `tool_calls` (API calls
per case), `savings` (distil token reduction for the turn — deterministic, model-independent).
`cost_usd` is derived at report time from each row's served `model` × `usage` at first-party
list prices (cache write 1.25×, read 0.1×), cache-cold: `--status` prints it.

## Migration decision rule

A candidate grader model is **safe to adopt** for certification when, on the 58 test cases:

1. `equiv_distil_act` paired Δ vs baseline has a 95% CI lower bound ≥ −5 pts;
2. `self_consist_act` and `decided` are not below baseline by more than their CI;
3. `trunc_detect` ≥ 80% of baseline's — it can still see damage.

The cost hillclimb adds: ≥ 15% cheaper $/case than the incumbent, with the saving explained
by fewer output tokens (effort) or a lower per-token price (tier).

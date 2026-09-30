# distil decision-equivalence on real traffic (τ-bench, direct API)

100 marker-free cases from public τ-bench trajectories (airline + retail; agents recorded by GPT-4o and Claude 3.5 Sonnet), drawn only from turns where distil saves ≥10% (median saving 13%, max 59%). The case list is in `benchmarks/model_migration_cases_real.json`; the raw files are fetched with `--fetch-tau-bench`. 2 reps. Every call went directly to api.anthropic.com. (An earlier run went through the local distil proxy and is quarantined in `_proxied/`.)

## Held-out (60 cases × 2 reps), paired vs claude-opus-4-8

| certifier | expand act-eq (deployed) | Δ vs base | self-consist (act) | Δ | distil act-eq (no recovery) | recorded act | $/case | out tok/case |
|---|---|---|---|---|---|---|---|---|
| claude-opus-4-8 | 91.7% | — | 95.0% | — | 61.7% | 51.7% | $0.146 | 612 |
| claude-opus-5-5 (default) | 90.8% | -0.8 ± 10.0 | 93.3% | -1.7 ± 6.6 | 73.3% | 44.2% | $0.134 | 1597 |
| claude-opus-5-5 @ low | 90.0% | -1.7 ± 9.0 | 94.2% | -0.8 ± 5.9 | 75.0% | 45.0% | $0.126 | 1197 |
| claude-sonnet-5-5 @ low | **94.2%** | **+2.5 ± 8.5** | **99.2%** | +4.2 ± 4.8 | 68.3% | 45.8% | **$0.062** | 1084 |
| claude-haiku-4-5 | 85.0% | -6.7 ± 10.0 | 96.7% | +1.7 ± 5.7 | 44.2% | 54.2% | $0.025 | 630 |

## What it says

- **distil with its recovery loop is close to the model's own noise floor on real traffic.** With `distil_expand` recovery, the next action matches the uncompressed decision 90–94% of the time, against 93–99% for two identical uncompressed calls. **Without the recovery loop it drops to 44–75%.** The digest is only safe when `distil_expand` is enabled, so serving should never run the reversible tier without it.
- **claude-opus-5-5 certifies as consistently as claude-opus-4-8** (93–94% vs 95% self-consistency). This retracts the earlier "noisier certifier" reading, which came from the leaky synthetic cases and proxied calls.
- **claude-sonnet-5-5 @ low is the cost winner**: best point estimate on every quality column, 58% cheaper per case, and the saving comes from its lower per-token price (it produces more output tokens than claude-opus-4-8). It does **not yet clear** the registered gate: its CI lower bound is -6.0 against a -5 bar. Two more reps on baseline and claude-sonnet-5-5 (~$41) would narrow the half-width to about ±6.
- **claude-haiku-4-5 is too weak to certify.** It loses about 7 pts on the deployed metric, and without the recovery loop it keeps only 44% of actions.
- "recorded act" (agreement with the recorded agent's action) is 44–54% for every model. τ-bench turns often admit several reasonable next actions (think, respond, transfer), so this column is a diagnostic and never a gate.

## Confirm (4 reps on baseline and v3, 2026-09-29)

claude-sonnet-5-5 @ low **passes every registered gate** on the 60 held-out cases × 4 reps:
- expand act-eq 94.6% vs 90.8%, Δ +3.8 ± 8.3 (lower bound -4.6 ≥ -5)
- self-consistency 99.2% vs 92.5%, Δ +6.7 ± 4.9, significantly better
- decided 100% vs 100%
- trunc caught 0.88× baseline (≥ 0.80)
- $0.062 vs $0.146 per case (-58%), from a lower per-token price

Recommended default certifier: claude-sonnet-5-5 @ effort=low.

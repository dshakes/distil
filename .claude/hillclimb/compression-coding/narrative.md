| round | change | served+rec act-eq train / test | served act-eq train / test | self-consist train / test | all-turns served savings | status |
|---|---|---|---|---|---|---|
| 0 | baseline (distil @ HEAD) | 70.7% / 78.0% | 61.0% / 59.3% | 96.3% / 94.1% | 52.6% | run, $35.95 |
| 1 | keep the touched numbered lines of `edit`/`goto` file views (analyzer proposal) | predicted ~74% train (80%: 70-79) | ~flat | — | **38.4%** on train trajs (-13 pts, measured offline) | **not run**: below the ±9-pt noise floor at a 13-pt savings cost |

On coding sessions (120 real SWE-agent GPT-4o trajectories on SWE-bench Lite, 4616 turns, 121.7M tokens), 85% of tokens are history.

- **Savings:** distil's certified per-turn strategy saves 2.6% of tokens; its serving adapter, which digests every earlier tool_result a caching client sends, saves 52.6%.
- **Risk:** on the 100-case decision set the served context changes the agent's next action in about 1 in 4 decisions even with distil_expand recovery (78.0% held-out agreement vs 94.1% between two identical calls). Most of this served path is untested by distil's certificate.

Round 1 targets that gap. Traces live in `baseline/traces/`; served text is regenerated offline with `serve()`.

## Round 1: categorized, not run

Of the 82 train rows, 24 are flips where both arms decided and full_a ≠ served_expand. Causes:
- 17: the newest `edit`/`goto` numbered file view was digested (edit and goto are not in EXACT_QUOTE_TOOLS). Only 7 of these are the consequential "full edits, served re-opens the file" kind.
- about 9: a different but defensible search or navigation choice (search_file vs search_dir, goto vs scroll_down).
- 3: plain noise (full_a ≠ full_b).

The best cache-safe fix recovers about 3 rows (+3.7 pts, inside noise) for -13 pts of all-turns served savings. Name-based exemptions cost far more (served savings falls to 7-44%). Not adopted. Proposal kept in `../proposals/coding-round1-view-window.py`.

**Interpretation:** at this noise floor (±9 pts with 59 test cases × 2 reps), action-level equivalence cannot separate consequential flips (a lost edit) from harmless ones (a different search command). The measurement that would decide whether served compression hurts coding agents is task outcome: run the agent end-to-end on SWE-bench with and without distil serving, and compare resolve rates. The decision-level eval should also grow (more cases and reps) before content rounds can pay.

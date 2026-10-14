# Haiku 5.5 on the long-horizon pilot tasks (SWE-bench Verified, 8 tasks)

Same 8 instances and settings as `swebench-verified-hard-max` (plain arm): effort max, max-tokens 32000,
100 steps, 3600 s timeout, seed 0. Haiku prompts above 100k tokens bill every token at 5x (`LONG_TIER` in
`benchmarks/swebench_outcome/agent.py`); 7 of 8 tasks crossed it, tripling the bill (x3.08 vs flat pricing).

| Model | Solved | Cost | $/solved |
|---|---|---|---|
| Haiku 5.5 | 7/8 | $3.37 | $0.48 |
| Sonnet 5.5 (cache reads repriced at 0.05x) | 7/8 | $18.93 | $2.70 |

Haiku misses sphinx-doc__sphinx-9229 (hit the 100-step limit); Sonnet misses pytest-dev__pytest-10356.
n=8 is a direction, not a verdict; the Sonnet run is from an earlier date.

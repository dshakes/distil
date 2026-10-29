# Haiku 5.5 on SWE-bench Lite (300), plain arm

`claude-haiku-5-5`, effort medium, 60 steps, seed 0. Graded locally with swebench 4.1.0.

| Model | Solved | Cost | $/solved |
|---|---|---|---|
| Haiku 5.5 | 207/300 | $1.18 | $0.0057 |
| Sonnet 5.5 (`swebench-outcome-300-h2h` plain, cache reads repriced at 0.05x) | 218/300 | $7.99 | $0.0366 |

Haiku is 0.16x Sonnet's $/solved at -11 solved. Haiku solves 15 tasks Sonnet misses; Sonnet solves 26 Haiku misses.
An oracle cascade (escalate to Sonnet when the hidden tests fail) would reach 233 at 0.48x, but no realistic
escalation signal gets there: empty patch, broken PASS_TO_PASS, or the agent's own last test run all land at
207-210 solved, 0.16x-0.37x. The saving comes from the cheaper model, not from routing.

Transcripts and run logs are not committed.

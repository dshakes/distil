# distil 1.59.0 (2,000-token lossy floor) vs Headroom 0.42.0: offline re-run on fresh samples

Same harness and arms as `../headroom-0-42-h2h/` (`benchmarks/headroom_h2h.py`), re-run
2026-10-31 because 1.59.0's lossy floor changes what distil removes. The original inputs file
was not kept, so the samples were re-collected from the same machine's Claude Code transcripts
with the same seeded rules: 347 tool results (60 each of logs, code, HTML, search and prose; 47
JSON, all the pool held) and 21 sessions of 12 requests. Headroom 0.42.0 was rebuilt in an
isolated venv with the proxy, code, ML, HTML and relevance extras. distil ran the floor and the
prose condenser from PR #249, which report version 1.58.0 because they were not yet released.
$0: no API calls. Only aggregates are committed (`results.json`); the samples are private.

Offline token / fact-recall / modelled-cache-bill only. Output tokens, extra steps and
`distil_expand` rounds are not priced, and agent behaviour is held fixed. The live outcome
result for the floor is `../swebench-lite-300-distil-floor/`.

| all 347 samples | tokens removed | visible recall | true recall | session $ vs plain (model) |
|---|---|---|---|---|
| distil default | 2.3% | 99.6% | 99.9% | −5.7% [−9.2, −2.8] |
| Headroom `coding` (proxy default) | 5.5% | 94.7% | 98.9% | −4.3% [−7.2, −1.9] |
| Headroom library default | 6.3% | 94.4% | 98.8% | −0.6% [−1.7, −0.0] |
| distil aggressive (+ cold-point eviction) | 30.3% | 86.0% | 99.9% | not run on sessions |
| Headroom `agent-90` | 28.5% | 73.0% | 98.7% | −15.7% [−21.2, −10.7] |
| distil lossless-only | 1.3% | 99.9% | 99.9% | −0.3% [−0.6, −0.1] |

Session $: Headroom in its proxy's delta engine. Paired, distil default vs Headroom `coding`:
−1.4 points [−5.8, +3.1], not significant; vs `agent-90`: +10.0 points [+5.0, +15.1], so
`agent-90` models cheaper. Most samples are under the 2,000-token floor, so distil's default now
removes little from them: that is the design (small output reaches the agent whole), and it is
why distil's default token figure fell from 13.7% on the 1.58.0 run.

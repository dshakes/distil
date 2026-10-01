> **SUPERSEDED:** seeded from the synthetic flow (leaky markers, proxied calls). Restart the compression hillclimb from `../decision-equivalence-real/`.

| round | change | status | train expand act-eq | train savings |
|---|---|---|---|---|
| 0 | baseline (distil @ HEAD, certifier claude-sonnet-5-5 @ low; seeded from decision-equivalence v3 reps 0-1) | run | 82.1% | 36.3% |
| 1 | keep a 1-before/2-after window around each pinned DECISION: line (tier1) | **not run** | predicted ~86.9% (80%: 83.3-90.5) | 33.9% (measured offline) |

Round 1 was not run. The analyzer found one behavior behind 12 of the 15 train expand-flips: tier1 pins the in-band `DECISION:` line verbatim and folds its surroundings, so the directive reads as settled and the model acts on it where the full context verifies first. The fix it proposed predicts about +5 pts, below the ±10-pt held-out noise floor, so it does not earn a paid round (patch kept in `../proposals/round1-decision-halo.patch`).

The same analysis exposed a flaw in the case set. `DECISION:` markers are ground-truth annotations for distil's offline DeterministicRunner, and they appear in model-visible content in all 100 cases (spliced mid-word into tool output in about 46). A live certifier is therefore reading the intended answer, and tier1's pin-DECISION rule makes it more prominent after compression. Real traffic carries no such markers, so the equivalence levels measured so far describe this artifact at least as much as distil. The fix is to strip the markers from everything the live model sees before running any arm, then re-baseline.

# 0020 — The default live certifier is claude-sonnet-5-5 at effort=low

- **Status:** accepted
- **Date:** 2026-09-30
- **Relates to:** `distil/replay/anthropic_runner.py` (`DEFAULT_MODEL`, `DEFAULT_EFFORT`, `effort_for`, `AnthropicRunner`), `distil/replay/expand_runner.py`, `distil/cli.py` (`--effort` on `certify`, `eval`, `benchmark`, `frontier`, `conformal`), `.github/workflows/live-cert.yml`, `benchmarks/model_migration_eval.py`, `benchmarks/model_migration_summary.py`, `benchmarks/results/model-migration/summary.json`, `docs/model-migration.html`

## Context

A live certificate is only as good as the model that grades it. Until 1.56.0 the grader was an accident of history: `AnthropicRunner` defaulted to `claude-opus-4-8`, and `distil certify` graded with whatever model label the trace carried. The nightly live-cert workflow used `claude-haiku-4-5`, chosen for price and never measured against anything.

Newer models exist, and two things made leaving the choice implicit untenable. First, the grader is a cost: an Opus-class grader on every live certificate is most of what a live run spends. Second, it is a dependency with a failure mode nobody had checked. Replacing a grader silently changes what "the next action is unchanged" means, so a cheaper grader could make compression look safer than it is, and a newer one could make it look worse for reasons that have nothing to do with distil.

So the question is a migration question, stated the way a model upgrade is: with the candidate grading the same compressed and uncompressed contexts, does it reach the same verdicts as the incumbent, and is it worth the change?

## Decision

1. **`DEFAULT_MODEL = "claude-sonnet-5-5"` and `DEFAULT_EFFORT = "low"`.** `AnthropicRunner` uses them unless told otherwise.
2. **`--effort` on every command that takes `--runner anthropic`** (`certify`, `eval`, `benchmark`, `frontier`, `conformal`), with `effort_for(model, effort)` resolving it: an explicit value wins, `none` sends no `output_config` at all, and unset means `low` for the default model and nothing for any other. A blanket `low` would have turned every call to a model without effort support (`claude-haiku-4-5` rejects an `output_config`) into a 400.
3. **Newer models that reject a forced `tool_choice` are handled.** `claude-opus-5-5`, `claude-sonnet-5-5` and `claude-fable-5-1` answer a forced `tool_choice` with a 400. `AnthropicRunner` falls back to `tool_choice=auto` with the same strict decision tool. Before this, `--runner anthropic` exited on every one of them.
4. **The nightly live-cert workflow moves from `claude-haiku-4-5` to `claude-sonnet-5-5` at low effort**, so the gate that runs unattended grades with the model that was measured.
5. **The choice is earned by a pre-registered migration eval, not declared.** `benchmarks/model_migration_eval.py` replays 100 real τ-bench decision points (60 held out, 4 repetitions each) under a fixed set of arms, and a candidate is accepted only if it passes every gate, written down before the run:
   - quality: the lower bound of the paired change in the deployed metric against `claude-opus-4-8` is at least −5 points;
   - self-consistency: two uncompressed samples agree at least as often as the incumbent's do;
   - the truncation control still diverges, at no less than 0.8 times the incumbent's rate;
   - the candidate is at least 15% cheaper per case;
   - any difference is explained by a mechanism found in the traces, not left as an unexplained delta.

   The deployed metric is `equiv_expand_act`: the next action is unchanged by compression when the model may recover a digest with `distil_expand`. It is the metric that matches how distil is served.

## Evidence

All figures are read from `benchmarks/results/model-migration/summary.json`, recomputed by `benchmarks/model_migration_summary.py` from the per-case rows. Held-out cases only, paired against `claude-opus-4-8`.

| candidate | deployed metric | paired change, lower bound (pts) | cost per case | gates |
|---|---|---|---|---|
| `claude-opus-4-8` (incumbent) | 90.8% | reference | $0.1458 | reference |
| `claude-opus-5-5` | 90.8% | +0.0, −10.0 | $0.1336 | fails the quality bound |
| `claude-opus-5-5` at low | 90.0% | −0.8, −9.8 | $0.1256 | fails the quality bound |
| **`claude-sonnet-5-5` at low** | **94.6%** | **+3.8, −4.5** | **$0.0619** | **passes all five** |
| `claude-haiku-4-5` | 85.0% | −5.8, −15.8 | $0.0248 | fails the quality bound |

`claude-sonnet-5-5` at low effort costs 57.5% less per case than the incumbent.

## Alternatives rejected

- **Keep `claude-opus-4-8`.** It is the reference, not a bad choice. It costs about 2.4 times as much per case and buys no measured quality. Staying would leave the certifier unmeasured against anything newer, which is the gap this decision closes.
- **`claude-opus-5-5`.** The point estimate matches the incumbent, at about the same cost (and a little less). The 60-case interval does not clear the −5 bound, so it buys neither a saving nor a proof. It remains a supported `--model` choice.
- **`claude-haiku-4-5`.** Cheapest by a wide margin, and too weak: the point estimate falls 5.8 points and the lower bound is −15.8. Its truncation control diverges more often than any other candidate's, so the control is not what fails; the decisions themselves are. A certificate graded by a weaker model than the agent it certifies is an optimistic one.
- **A newer-is-better rule with no gates.** The certifier is an instrument. An instrument is changed under a recorded acceptance test or not at all.

## Consequences

- **Certificates change their grader.** Every certificate stamps the model that graded it (`Certificate.grader`). The published live result, 83.2% token savings at 0% decision-change, was graded by `claude-opus-4-8` on 2026-07-05 and stays labelled that way. It is not re-graded by this change, and no page should imply it was.
- **The evidence is one task family.** The decision is backed by τ-bench traffic. The coding set was run for the served strategy (ADR 0021) on the default model only. A second replication model and a coding-traffic migration run are open, and the intervals are wide: 60 held-out cases cannot separate most candidates from the reference. The gate is built for that; it asks for a lower bound, not a point.
- **The eval is rerunnable.** A future model goes through the same harness (`docs/model-migration.html` shows how). The harness hash is gated, so a change to the runners or prompts has to be approved before its numbers can be mixed with the old ones.
- **Lessons recorded in the harness.** A distil-wrapped shell exports `ANTHROPIC_BASE_URL` to the local proxy, so an eval client must pin `api.anthropic.com` or it measures distil against itself. The bundled corpus plants `DECISION:` markers for the offline oracle, which leaks the answer to a live grader; live grading needs marker-free real traces, and live renders now strip the marker lines.

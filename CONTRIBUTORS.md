# Contributors

Distil is built by its maintainer and the people who send fixes and features.
Thank you.

Some contributions landed by cherry-pick rather than merge — the branch had drifted, or
review fixes were layered on top — so the commit history does not always carry the
contributor's name. That is a quirk of how it was integrated, not a measure of the work.
This file is the record.

## Maintainer

- **Chandra Shekhar Mudarapu** ([@dshakes](https://github.com/dshakes)) — author and maintainer.

## Contributors

- **pliablepixels** ([@pliablepixels](https://github.com/pliablepixels)) — per-content-type keep policy (`distil/compress/keep_policy.py`), so a digest keeps a test run's verdict instead of folding the answer and keeping the noise ([#23](https://github.com/dshakes/distil/pull/23), shipped 1.15.0); and `distil dissect` — the per-session deep-dive report, including `--serve` and agent-transcript correlation ([#27](https://github.com/dshakes/distil/pull/27), shipped 1.15.1).
- **Tolga Tuncoglu** ([@tolgatuncoglu](https://github.com/tolgatuncoglu)) — `distil default --always-on`: persist `ANTHROPIC_BASE_URL` into `~/.claude/settings.json` so Claude Code routes through distil on every launch, including sessions started from an IDE extension rather than a terminal ([#31](https://github.com/dshakes/distil/pull/31), shipped 1.26.0).

---

Sent a PR that landed? Add yourself here in the same PR — one line, newest last.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the bar a change has to clear.

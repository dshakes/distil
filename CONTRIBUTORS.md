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
- **PJ Doland** ([@pjdoland](https://github.com/pjdoland)) — a WCAG 2.2 pass over the docs site and every HTML surface distil generates. Keyboard access to navigation, session links, and copy controls, including two pages whose mobile nav button called a function that did not exist ([#34](https://github.com/dshakes/distil/pull/34)); status announcements and repaired ARIA wiring ([#35](https://github.com/dshakes/distil/pull/35)); AA contrast for muted text — the leaderboard's "not certified" marker had been rendering at 1.75:1 ([#36](https://github.com/dshakes/distil/pull/36)); described diagrams and data-table fallbacks for every chart ([#37](https://github.com/dshakes/distil/pull/37)); visible focus, skip links, and a dismissible dissect tooltip ([#38](https://github.com/dshakes/distil/pull/38)); pausable in-place polling replacing the meta refreshes on the gateway dashboard and dissect portal ([#39](https://github.com/dshakes/distil/pull/39)); real nav, table, and heading semantics ([#40](https://github.com/dshakes/distil/pull/40)); and new-tab warnings with first-use abbreviation expansions ([#41](https://github.com/dshakes/distil/pull/41)) — shipped 1.33.0.

---

Sent a PR that landed? Add yourself here in the same PR — one line, newest last.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the bar a change has to clear.

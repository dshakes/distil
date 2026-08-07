# Terminal recordings (VHS tapes)

Every GIF in `docs/assets/` is generated from a `.tape` script in this directory
by [charmbracelet/vhs](https://github.com/charmbracelet/vhs).

```bash
brew install vhs                       # or: go install github.com/charmbracelet/vhs@latest
vhs docs/tapes/library-api.tape        # → docs/assets/library-api.gif
vhs docs/tapes/outage-guard.tape       # → docs/assets/outage-guard.gif
```

**Why the scripts are checked in and the GIFs are regenerated.** A recording
nobody can re-make goes stale silently: the CLI changes, the output no longer
matches, and the README keeps showing a session that can no longer happen. A
tape is reviewable in a diff and re-renderable on demand, so a drifted GIF is a
one-command fix rather than an archaeology project.

VHS needs a real PTY and `ffmpeg`. It does not render reliably in a headless
agent shell, so regenerate these locally (or in a CI job with a TTY) after any
change to the commands they demonstrate.

| Tape | Shows |
|---|---|
| `library-api.tape` | `compress_messages` / `expand_handle` in-process, and byte-exact recovery |
| `outage-guard.tape` | `distil wrap` refusing a `settings.json` pin that would take every session down |

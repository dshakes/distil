# Adding distil to your framework

For maintainers of agent frameworks, SDKs, and coding tools. If you have ever
watched a user's context window fill with 400 lines of near-identical log output,
this is the ~30 lines that fixes it.

Everything below is a **reference implementation you can copy**, not a
description. Five integrations already ship this way
([`distil/integrations/`](../distil/integrations/)), and they are all the same
shape.

---

## The contract, in four rules

Break any of these and the integration is worse than nothing.

1. **Never import distil at your module scope.** Import it lazily inside the
   function, or guard it, so distil stays an optional dependency and your users
   who do not want it pay nothing.
2. **Never rewrite assistant messages.** The model's own words are not ours to
   edit. Rewriting them corrupts few-shot context and breaks caching.
3. **Never mutate the caller's list.** Return a new one. Pass unchanged messages
   through *by identity* so callers can cheaply detect a no-op.
4. **Fail open.** If compression raises, forward the original. Compression is an
   optimisation; losing it must not lose the turn.

---

## The 20-line version

```python
def compress_state(messages: list) -> list:
    """Compress a message list before the model call. Returns a NEW list."""
    try:
        from distil import compress_messages
    except ImportError:
        return messages          # distil not installed — do nothing, loudly nowhere
    try:
        return compress_messages(messages).messages
    except Exception:
        return messages          # fail open: never lose a turn to an optimisation
```

That is genuinely it for a framework whose messages are
`{"role": ..., "content": ...}` dicts or objects exposing `.role`/`.type` and
`.content`. distil's message handling is **duck-typed** — it never imports your
framework either, so your release cannot break us and ours cannot break you.

### TypeScript

```ts
import { compress } from "distil-llm";

export function compressMessages(messages) {
  try {
    return compress(messages).messages;
  } catch {
    return messages;
  }
}
```

---

## If your message shape is different

Two real examples from shipped integrations:

**Content is a list of typed blocks** (Strands, the AI SDK). Walk the blocks and
compress the text-bearing ones, preserving every sibling key — cache markers and
tool-call ids live there. See
[`integrations/strands.py`](../distil/integrations/strands.py).

**Tool results carry their payload under a version-dependent key.** The Vercel AI
SDK uses `output.value` in v5 and `result` in v4. Handle both rather than pinning
one, or an SDK bump silently stops compressing the largest thing in the context.
See [`packaging/npm/lib/aisdk.js`](../packaging/npm/lib/aisdk.js).

---

## Which tier you get, and why it matters

| How you integrate | Tier | Needs a running proxy? |
|---|---|---|
| In-process library (the code above) | **lossless only** | no |
| Point your client's `base_url` at `distil proxy` | reversible **digest** | yes |

The in-process libraries are lossless-tier **on purpose**. The digest tier mints
*restore handles* whose originals must live in one store shared with the proxy
and the MCP server — otherwise `expand` cannot resolve a handle the model can
see. It is also the tier the decision-equivalence certificate measures, so a
second implementation would be a second thing to certify.

**Say which tier your integration provides in your docs.** A user who wires the
library expecting digest-level savings will conclude distil under-delivers, and
they will be right to.

---

## What to test

Copy these; they are the four ways an integration goes wrong:

```python
def test_assistant_messages_are_never_rewritten(): ...
def test_input_is_not_mutated(): ...
def test_non_string_content_passes_through():      # images, tool_use structures
def test_it_works_with_distil_not_installed():     # optional dependency
```

Reference: [`tests/test_integrations_new.py`](../tests/test_integrations_new.py).

---

## Verifying it actually helps *your* users

Do not take our numbers. Ours are measured on our corpus.

```bash
distil wrap --shadow 0.1 -- <your framework's CLI>
distil shadow-stats
```

`--shadow` runs a fraction of requests twice — compressed and full — and compares
the agent's chosen next action. That measures decision-equivalence on **your**
workload, and a result there outranks anything in our repository, **including a
bad one**. If distil degrades your users' agents, we want that number more than
we want the integration.

---

## Sending it upstream

We would rather the integration live in **your** repo than ours: you can test it
against your own CI and your users find it where they already look.

- Keep distil an **optional** dependency. Never add it to your required install.
- Link back to [the tier table](#which-tier-you-get-and-why-it-matters) so your
  users know what they are getting.
- Open an issue on [dshakes/distil](https://github.com/dshakes/distil/issues) and
  we will review, test against your framework, and keep the integration working.

If you would rather it live here, that works too — see
[`CONTRIBUTING.md`](../CONTRIBUTING.md); a new integration is listed as a good
first change.

---

## What distil will not ask of you

- No account, no API key, no telemetry. The census is opt-in and numbers-only.
- No network call to anything we operate. distil forwards to *your user's*
  configured provider and nothing else.
- No required dependency. The core is stdlib-only, zero runtime deps.

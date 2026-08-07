# Code of Conduct

## The short version

Be decent. Argue about code, not people. Assume the person you are replying to is
competent and acting in good faith until they prove otherwise.

## Why this file is short

Distil's contribution bar is unusually mechanical: **a new compression strategy
must pass `make gate`** — non-inferior on every domain, byte-reversible — and no
green gate means no merge. That removes most of what community conflict is
usually about, because "is this good?" is answered by evidence rather than by
whoever argues longest.

What is left is ordinary human conduct, and the rules are the ones you would
expect.

## Expected behaviour

- Critique the work, and be specific. "This breaks reversibility on inputs with
  a `<<xN>>` lookalike" is useful. "This is bad" is not.
- Accept that a maintainer may decline a change that passes the gate. Scope is a
  judgment call, and "it works" is not the same as "it belongs".
- When you are wrong, say so plainly and move on. Nobody needs the ceremony.
- Report what actually happened. A benchmark you did not run, a test you did not
  see pass, a claim you have not checked — say which, and label it.

## Unacceptable behaviour

- Personal attacks, harassment, or discriminatory language of any kind.
- Publishing someone's private information.
- Sustained disruption of discussions or reviews.
- Sockpuppeting, or manufacturing consensus.
- Deliberately misrepresenting benchmark results or the certificate's scope. This
  project's value is that its numbers can be trusted; faking one is not a
  technical disagreement, it is a breach of the thing being built.

## Scope

Applies in the issue tracker, pull requests, discussions, and any space where you
are representing the project.

## Enforcement

Report to the maintainer through [private vulnerability
reporting](https://github.com/dshakes/distil/security/advisories/new) (it works
for conduct reports too and keeps the thread private) or by opening an issue if
the matter is not sensitive.

Reports are read by the maintainer. Responses range from a private word to a
permanent ban, proportional to the behaviour and its history. There is no formal
appeals process — this is a small project, not an institution, and pretending
otherwise would be theatre.

## Attribution

Adapted in spirit from the [Contributor Covenant](https://www.contributor-covenant.org),
rewritten to say what this project actually does rather than to fill a template.

"""Inline keep tags: ``<distil:keep>`` … ``</distil:keep>`` is never compressed.

A user (or a tool that knows its output) marks a span that must reach the model
byte-exact — a config block to be quoted back, a hash, a stack of numbers. Everything
outside the span compresses as usual; the span itself, **tags included**, passes
through untouched.

The tags are kept, not stripped. Stripping is itself a rewrite of the span distil
promised not to touch, the client that wrote the tags may rely on its own bytes
(a hash, a cached prefix, a later exact match), and the tags cost a handful of
tokens while telling the model why that span is verbatim. Nothing about stripping
could be proven safe for every client; keeping them is safe by construction.

Matching is literal and non-nesting: a span runs from an opening tag to the next
closing tag, and an unclosed opening tag protects everything after it (the safe
reading of a truncated span).
"""

from __future__ import annotations

from typing import Callable

OPEN = "<distil:keep>"
CLOSE = "</distil:keep>"


def apply(text: str, fn: Callable[[str], str]) -> str:
    """``fn`` over every stretch of *text* outside keep spans; spans pass through.

    Without a tag this is exactly ``fn(text)``. Each outside stretch is compressed on
    its own, so any handle a stretch gets recovers that stretch.
    """
    if OPEN not in text:
        return fn(text)
    out: list[str] = []
    i = 0
    while i < len(text):
        j = text.find(OPEN, i)
        if j < 0:
            out.append(fn(text[i:]))
            break
        if j > i:
            out.append(fn(text[i:j]))
        k = text.find(CLOSE, j + len(OPEN))
        end = len(text) if k < 0 else k + len(CLOSE)
        out.append(text[j:end])
        i = end
    return "".join(out)

"""Prefix stability — the observability half of cache alignment.

Provider prompt caching only pays when the stable prefix is **byte-identical**
between turns. One changed character at the front — a timestamp in a system prompt,
a re-ordered tool list, a session id — invalidates the whole cached span, and the
symptom is a bill, not an error.

distil already *prevents* that rather than warning about it: `strategies.distil`
compresses the volatile tail and leaves every stable block untouched, so the prefix
is byte-identical by construction, and `adapters.anthropic.place_cache_control`
marks the boundary so the provider can actually reuse it.

What was missing is the ability to *see* it. Prevention you cannot measure is a
claim, and when a prefix does drift the cause is almost always upstream of distil —
in how the caller assembles its own system prompt — which is exactly the case a
compressor cannot fix and must therefore report.

So this module answers one question, content-free: between two consecutive turns,
how much of the prefix survived byte-identical, and if it broke, where.

Counts and a hash escape; prompt text never does.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# Below this, a "stable prefix" is too short for any provider to cache and reporting
# it as a win would be noise. Anthropic needs ~1k tokens, OpenAI 1024, Google 32k;
# this is deliberately under all of them so the report never claims a hit that the
# provider would refuse, while still showing drift on small prompts.
_MIN_USEFUL_BYTES = 512

# The blocks a provider caches: everything ahead of the conversation. `messages` is
# excluded on purpose — it grows every turn by design, so including it would make the
# hash differ on every request and report drift where there is none.
STABLE_KEYS = ("system", "tools", "tool_choice")


@dataclass
class PrefixReport:
    """What survived byte-identical between two turns."""

    stable_bytes: int = 0
    stable_tokens_est: int = 0
    stable_hash: str = ""
    prefix_changed: bool = False
    # Where the first difference is, as a fraction of the shorter prompt. A break at
    # 0.02 is a volatile system prompt; at 0.98 it is the tail doing its job.
    break_ratio: float = 0.0
    total_bytes: int = 0
    cacheable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_bytes": self.stable_bytes,
            "stable_tokens_est": self.stable_tokens_est,
            "stable_hash": self.stable_hash,
            "prefix_changed": self.prefix_changed,
            "break_ratio": round(self.break_ratio, 4),
            "total_bytes": self.total_bytes,
            "cacheable": self.cacheable,
        }


def _flatten(messages: Any) -> str:
    """Render a payload to the byte sequence a provider would hash for its cache.

    Order matters and is preserved: the prefix is a byte prefix, so a reordered tool
    list is a changed prefix even though the set is identical.
    """
    out: list[str] = []
    # Provider order, not alphabetical. Sorting put `messages` ahead of `system`, so
    # appending a turn — the most ordinary thing an agent does — moved the volatile
    # part to the FRONT and every healthy turn reported drift. A cache prefix is a
    # byte prefix of what the provider actually receives, so the flattening has to
    # mirror that layout: stable blocks first, conversation last.
    _ORDER = {"system": 0, "tools": 1, "tool_choice": 2, "messages": 9, "input": 9}

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for key in sorted(node, key=lambda k: (_ORDER.get(k, 5), k)):
                if key == "cache_control":
                    continue  # the marker itself is not content; moving it is not drift
                out.append(str(key))
                walk(node[key])
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif node is not None:
            out.append(str(node))

    walk(messages)
    return "\x1f".join(out)


def analyse(previous: Any, current: Any, *, chars_per_token: float = 3.6) -> PrefixReport:
    """Compare two consecutive turns and report what stayed byte-identical.

    `previous` empty means this is the first turn: there is nothing to have drifted
    from, so the report is the prefix length with `prefix_changed=False`. Reporting a
    first turn as "changed" would flag every session start as a cache miss.
    """
    cur = _flatten(current)
    prev = _flatten(previous) if previous else ""
    total = len(cur)

    if not prev:
        digest = hashlib.sha256(cur.encode("utf-8", "replace")).hexdigest()[:16]
        return PrefixReport(
            stable_bytes=total,
            stable_tokens_est=int(total / chars_per_token),
            stable_hash=digest,
            prefix_changed=False,
            break_ratio=1.0,
            total_bytes=total,
            cacheable=total >= _MIN_USEFUL_BYTES,
        )

    limit = min(len(prev), len(cur))
    shared = 0
    while shared < limit and prev[shared] == cur[shared]:
        shared += 1

    stable = cur[:shared]
    return PrefixReport(
        stable_bytes=shared,
        stable_tokens_est=int(shared / chars_per_token),
        stable_hash=hashlib.sha256(stable.encode("utf-8", "replace")).hexdigest()[:16],
        # Drift means the shared prefix ended before the OLD payload did. A prefix
        # that merely grew — the normal case, a turn appended — is not drift, and
        # calling it drift would fire on every healthy turn.
        prefix_changed=shared < len(prev),
        break_ratio=(shared / limit) if limit else 0.0,
        total_bytes=total,
        cacheable=shared >= _MIN_USEFUL_BYTES,
    )


@dataclass
class CacheSummary:
    """What the provider actually billed, plus what our own prefix hashes explain.

    The two halves are deliberately different in kind. `read`/`create` are the
    provider's numbers — ground truth about money. `drifts` is our diagnosis of why,
    and is only ever an explanation for a measurement, never a substitute for one.
    """

    requests: int = 0
    read_tokens: int = 0
    create_tokens: int = 0
    uncached_tokens: int = 0
    drifts: int = 0  # consecutive pairs whose stable prefix hash changed
    pairs: int = 0  # consecutive pairs we could compare at all
    reported: bool = False  # did the provider report cache usage at all?
    # Rows written before 1.41 recorded reads and writes added together. That total
    # still proves caching was active, but it cannot yield a hit ratio — and printing
    # one anyway, from a sum, would be a confident number with nothing behind it.
    legacy_cache_tokens: int = 0
    legacy_rows: int = 0

    @property
    def hit_ratio(self) -> float:
        """Reads over cacheable tokens. 1.0 = every cacheable token was a hit."""
        total = self.read_tokens + self.create_tokens
        return (self.read_tokens / total) if total else 0.0

    @property
    def drift_ratio(self) -> float:
        return (self.drifts / self.pairs) if self.pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "read_tokens": self.read_tokens,
            "create_tokens": self.create_tokens,
            "uncached_tokens": self.uncached_tokens,
            "hit_ratio": round(self.hit_ratio, 4),
            "drifts": self.drifts,
            "pairs": self.pairs,
            "drift_ratio": round(self.drift_ratio, 4),
            "reported": self.reported,
            "legacy_cache_tokens": self.legacy_cache_tokens,
            "legacy_rows": self.legacy_rows,
        }


def summarise(records: list[dict[str, Any]]) -> CacheSummary:
    """Fold per-request ledger rows into one cache picture.

    Rows are compared in the order the proxy wrote them, which is the order the
    provider saw. Rows without a prefix hash (pre-1.41, or a body we could not read)
    are counted as requests but never as pairs — an unknown prefix is not a stable
    one, and treating it as stable would report a clean sheet for a session we could
    not actually check.
    """
    out = CacheSummary()
    last_hash = ""
    for rec in records:
        out.requests += 1
        read = int(rec.get("usage_cache_read") or 0)
        create = int(rec.get("usage_cache_create") or 0)
        out.read_tokens += read
        out.create_tokens += create
        out.uncached_tokens += int(rec.get("usage_input_tokens") or 0)
        if read or create:
            out.reported = True
        elif rec.get("usage_cache_tokens"):
            out.legacy_cache_tokens += int(rec.get("usage_cache_tokens") or 0)
            out.legacy_rows += 1
        cur = str(rec.get("prefix_hash") or "")
        if cur and last_hash:
            out.pairs += 1
            if cur != last_hash:
                out.drifts += 1
        if cur:
            last_hash = cur
    return out


def format_summary(summary: CacheSummary) -> str:
    s = summary
    lines = [f"requests        {s.requests:,}"]
    if s.reported:
        lines += [
            f"cache reads     {s.read_tokens:,} tokens  (billed at a discount)",
            f"cache writes    {s.create_tokens:,} tokens  (billed at a surcharge)",
            f"uncached        {s.uncached_tokens:,} tokens",
            f"hit ratio       {s.hit_ratio:.1%} of cacheable tokens were reads",
        ]
    elif s.legacy_rows:
        lines += [
            f"cache tokens    {s.legacy_cache_tokens:,} across {s.legacy_rows:,} requests — "
            "caching was active",
            "                but these rows predate the read/write split, so no hit ratio.",
            "                Requests proxied from 1.41 on will carry it.",
        ]
    else:
        lines.append(
            "cache usage     not reported by the provider on these requests — either "
            "prompt\n                caching is off for this client or the prefix never "
            "reached the minimum"
        )

    if not s.pairs:
        lines.append(
            "prefix drift    no comparable pairs (need two requests carrying a prefix "
            "hash)\n                — not the same as 'no drift'"
        )
    elif s.drifts:
        lines += [
            f"prefix drift    {s.drifts:,} of {s.pairs:,} turns changed the stable prefix "
            f"({s.drift_ratio:.0%})",
            "                Each one re-bills the whole prefix. distil never rewrites a",
            "                stable block, so the cause is upstream: a timestamp or session",
            "                id in the system prompt, or a tool list whose order varies.",
        ]
    else:
        lines.append(f"prefix drift    none across {s.pairs:,} turns — the prefix held")
    return "\n".join(lines)


def format_report(report: PrefixReport) -> str:
    lines = [
        f"stable prefix   {report.stable_bytes:,} B  (~{report.stable_tokens_est:,} tokens)"
        f"  hash {report.stable_hash}",
    ]
    if report.prefix_changed:
        lines += [
            f"  DRIFT           the prefix broke at {report.break_ratio:.1%} of the prompt",
            "  A provider cache is byte-exact, so everything after the break is re-billed",
            "  at full price. distil never rewrites a stable block, so a break here comes",
            "  from the caller's own assembly — a timestamp, a session id, a reordered",
            "  tool list. Move the volatile part after the stable prefix.",
        ]
    elif not report.cacheable:
        lines.append(
            f"  too short       {report.stable_bytes:,} B is below every provider's cache "
            "minimum — nothing to reuse yet"
        )
    else:
        lines.append("  stable          byte-identical to the previous turn: cacheable")
    return "\n".join(lines)

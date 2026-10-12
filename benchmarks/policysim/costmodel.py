"""Provider prompt-cache cost models: exact bytes in, billed token components out.

A request is a list of *segments* (one per content block, plus one for tools+system),
each with a content key and a token count. The provider model keeps the cache state a
real provider would (entries, expiry) and returns, per request, the uncached input, the
cache writes (5m / 1h), the cache reads and any storage charge.

Rules, from the providers' own documentation (fetched 2026-10-06):

Anthropic — platform.claude.com/docs/en/build-with-claude/prompt-caching
  * "Cache writes happen only at your breakpoint ... a hash of the prefix ending at that
    block." Reads "walk backward one block at a time" for an entry a prior request wrote;
    "The lookback window is 20 blocks ... a run of consecutive tool_use blocks counts as
    one position, and so does a run of consecutive tool_result blocks".
  * minimum cacheable prompt: 512 tokens (Sonnet 5.5, Opus 5.5/5, Fable), 1,024 (Opus
    4.8, Sonnet 5/4.6/4.5), 2,048 (Opus 4.7), 4,096 (Opus 4.6/4.5, Haiku 4.5); shorter
    prompts are "processed without caching".
  * write 1.25x (5m) / 2x (1h), read 0.1x (0.05x Opus 5.5, 0.025x Fable 5.1); a hit
    refreshes the TTL, measured "from the start of the request".
  * ``input_tokens`` = "tokens after the last cache breakpoint".
OpenAI — developers.openai.com/api/docs/guides/prompt-caching and /pricing
  * GPT-5.6+: minimum 1,024 tokens, "Exact eligible boundary", writes "1.25x the standard,
    uncached input-token rate", reads 0.1x, TTL "30 minutes after its most recent write or
    reuse". gpt-5.6-terra: $2.00 in / $0.20 cached / $12.00 out.
  * before GPT-5.6: cached length rounded "down to the nearest multiple of 128", "No
    additional cache-write charge", retention ~30 min (24h mode). gpt-5.4: $2.50 / $0.25 /
    $15.00.
Gemini — ai.google.dev/gemini-api/docs/caching, /generate-content/caching, /pricing
  * implicit caching on 2.5+, minimum 4,096 tokens on 3.x, "no cost saving guarantee";
    gemini-3.1-pro-preview (<=200k): $2.00 in, $0.20 cached, $12.00 out.
  * explicit caching: TTL "defaults to 1 hour", storage $4.50 / Mtok / hour (3.1 Pro).
  * NOT documented, so assumed here and flagged in every result: implicit-cache lifetime
    (300 s) and hit probability (1.0, an upper bound); explicit-cache creation billed at the
    standard input rate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from distil import pricing

from .trajectory import Message, block_text, blocks, thinking_key

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=200_000)
def _pieces(text: str) -> int:
    return len(_PIECE.findall(text))


@dataclass(frozen=True)
class TokenModel:
    """Offline token count: ``scale * pieces(text) + per_block``, fitted to billed usage.

    Claude's tokenizer is not public, so the scale is fitted per run (``calibrate``);
    the fit's held-out error is reported next to every dollar figure.
    """

    scale: float = 1.33
    per_block: float = 0.0

    def count(self, text: str) -> float:
        return self.scale * _pieces(text) + self.per_block


@dataclass
class Seg:
    key: str
    tokens: float
    group: str | None  # "tool_use"/"tool_result" runs are one lookback position
    text: str = ""


def _key(role: str, block: Any) -> str:
    if isinstance(block, dict) and "cache_control" in block:
        block = {k: v for k, v in block.items() if k != "cache_control"}
    raw = role + "\x00" + json.dumps(block, sort_keys=True, default=str)
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def segments(
    messages: list[Message], tm: TokenModel, hidden: dict[str, int], overhead: float
) -> list[Seg]:
    segs = [Seg("__tools+system__", overhead, None)]
    for m in messages:
        role = str(m.get("role", ""))
        for b in blocks(m):
            tk = thinking_key(b)
            if tk is not None:  # hidden thinking: the allocation, else its text if recorded
                n = hidden[tk] if tk in hidden else tm.count(str(b.get("thinking", "")))
                segs.append(Seg(_key(role, b), float(n), None))
                continue
            text = block_text(b)
            grp = b.get("type") if isinstance(b, dict) else None
            grp = grp if grp in ("tool_use", "tool_result") else None
            segs.append(Seg(_key(role, b), tm.count(text), grp, text))
    return segs


def _chain(segs: list[Seg]) -> list[str]:
    out, h = [], ""
    for s in segs:
        h = hashlib.blake2b((h + s.key).encode(), digest_size=16).hexdigest()
        out.append(h)
    return out


def _cum(segs: list[Seg]) -> list[float]:
    out, c = [], 0.0
    for s in segs:
        c += s.tokens
        out.append(c)
    return out


@dataclass
class Usage:
    input: float = 0.0
    write_5m: float = 0.0
    write_1h: float = 0.0
    read: float = 0.0
    output: float = 0.0
    storage_usd: float = 0.0

    def __iadd__(self, o: Usage) -> Usage:
        for f in ("input", "write_5m", "write_1h", "read", "output", "storage_usd"):
            setattr(self, f, getattr(self, f) + getattr(o, f))
        return self

    @property
    def prompt(self) -> float:
        return self.input + self.write_5m + self.write_1h + self.read

    def as_dict(self) -> dict[str, float]:
        return {
            "input": round(self.input),
            "cache_write_5m": round(self.write_5m),
            "cache_write_1h": round(self.write_1h),
            "cache_read": round(self.read),
            "output": round(self.output),
        }


@dataclass(frozen=True)
class Price:
    input: float  # $/Mtok
    output: float
    read_mult: float = 0.10
    write_5m_mult: float = 1.25
    write_1h_mult: float = 2.0

    def usd(self, u: Usage) -> float:
        tok = (
            u.input
            + u.write_5m * self.write_5m_mult
            + u.write_1h * self.write_1h_mult
            + u.read * self.read_mult
        )
        return (tok * self.input + u.output * self.output) / 1e6 + u.storage_usd


class Provider(Protocol):
    name: str
    price: Price

    def reset(self) -> None: ...

    def request(self, segs: list[Seg], t: float, ttl: str) -> Usage: ...


# ----------------------------------------------------------------------------- Anthropic

#: Minimum cacheable prompt by model family (prompt-caching doc, 2026-10-06).
ANTHROPIC_MIN = {
    "claude-sonnet-5-5": 512,
    "claude-opus-5-5": 512,
    "claude-opus-5": 512,
    "claude-fable-5-1": 512,
    "claude-fable-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-haiku-4-5": 4096,
}
TTL_S = {"5m": 300.0, "1h": 3600.0}


@dataclass
class Anthropic:
    model: str = "claude-sonnet-5-5"
    #: Tokens billed as uncached ``input_tokens`` per request (framing after the breakpoint).
    framing: float = 2.0
    lookback: int = 20
    name: str = "anthropic"
    price: Price = field(init=False)
    min_tokens: int = field(init=False)
    _store: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        p = pricing.resolve(self.model)
        if p is None:
            raise KeyError(f"no Anthropic price for {self.model}")
        self.price = Price(p.input_per_mtok, p.output_per_mtok, p.cache_read_mult)
        base = p.name
        self.min_tokens = ANTHROPIC_MIN.get(base, 1024)

    def reset(self) -> None:
        self._store = {}

    def _positions(self, segs: list[Seg], bp: int) -> list[int]:
        """Candidate entry positions, newest first: block ends, runs merged, <= lookback."""
        out: list[int] = []
        i = bp
        while i >= 0 and len(out) < self.lookback:
            out.append(i)
            g = segs[i].group
            i -= 1
            while g is not None and i >= 0 and segs[i].group == g:
                i -= 1
        return out

    def request(
        self, segs: list[Seg], t: float, ttl: str, breakpoints: list[int] | None = None
    ) -> Usage:
        h, c = _chain(segs), _cum(segs)
        bps = breakpoints if breakpoints is not None else [len(segs) - 1]
        last = max(bps)
        total = c[-1]
        if c[last] < self.min_tokens:
            return Usage(input=total + self.framing)
        hit = -1
        for bp in bps:
            for j in self._positions(segs, bp):
                e = self._store.get(h[j])
                if e is not None and e[0] > t:
                    hit = max(hit, j)
                    self._store[h[j]] = (t + e[1], e[1])  # a hit refreshes the TTL
                    break
        read = c[hit] if hit >= 0 else 0.0
        life = TTL_S[ttl]
        for bp in bps:
            self._store[h[bp]] = (t + life, life)
        write = c[last] - read
        u = Usage(input=total - c[last] + self.framing, read=read)
        if ttl == "1h":
            u.write_1h = write
        else:
            u.write_5m = write
        return u


# ----------------------------------------------------------------- automatic prefix caches


def _common_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class PrefixCache:
    """Automatic longest-prefix caching (OpenAI, Gemini implicit): no breakpoints.

    The cached length is the longest prefix shared with any live earlier request, at
    token granularity *inside* the first differing block (pro-rated by shared chars).
    """

    name: str
    price: Price
    min_tokens: int
    ttl_s: float
    granularity: int = 1
    write_charged: bool = False  # GPT-5.6+: the uncached remainder is written at 1.25x
    hit_prob: float = 1.0
    keep: int = 8  # ponytail: last 8 requests are the only realistic hits in an agent loop
    _hist: list[tuple[list[str], list[float], list[Seg], float]] = field(
        default_factory=list, init=False
    )

    def reset(self) -> None:
        self._hist = []

    def request(self, segs: list[Seg], t: float, ttl: str) -> Usage:
        h, c = _chain(segs), _cum(segs)
        total = c[-1]
        best, best_i = 0.0, -1
        for idx, (ph, pc, ps, used) in enumerate(self._hist):
            if t - used > self.ttl_s:
                continue
            n = min(len(h), len(ph))
            lo, hi = 0, n  # chained hashes: equality is monotone, binary search the break
            while lo < hi:
                mid = (lo + hi) // 2
                if h[mid] == ph[mid]:
                    lo = mid + 1
                else:
                    hi = mid
            m = lo
            got = c[m - 1] if m else 0.0
            if m < n and segs[m].text and ps[m].text:
                frac = _common_chars(segs[m].text, ps[m].text) / max(1, len(segs[m].text))
                got += segs[m].tokens * frac
            if got > best:
                best, best_i = got, idx
        cached = best * self.hit_prob
        if total < self.min_tokens:
            cached = 0.0
        if self.granularity > 1:
            cached = (cached // self.granularity) * self.granularity
        if best_i >= 0:
            ph, pc, ps, _ = self._hist[best_i]
            self._hist[best_i] = (ph, pc, ps, t)
        self._hist.append((h, c, segs, t))
        self._hist = self._hist[-self.keep :]
        rest = total - cached
        if self.write_charged and total >= self.min_tokens:
            return Usage(read=cached, write_5m=rest)
        return Usage(read=cached, input=rest)


@dataclass
class GeminiExplicit:
    """Explicit context caching: one cache object, re-created every ``every`` requests.

    Reads of the object are billed at the cached rate, creation at the standard input
    rate (assumption: the pricing page does not price creation separately), storage per
    token-hour while the object lives. The object holds the full request it was created
    from; later requests that extend it read it.
    """

    price: Price
    storage_per_mtok_h: float
    min_tokens: int = 4096
    every: int = 4
    name: str = "gemini-explicit"
    _obj: tuple[list[str], float, float] | None = field(default=None, init=False)
    _n: int = field(default=0, init=False)

    def reset(self) -> None:
        self._obj, self._n = None, 0

    def _storage(self, t: float) -> float:
        if self._obj is None:
            return 0.0
        _, tok, born = self._obj
        return tok * max(0.0, t - born) / 3600.0 * self.storage_per_mtok_h / 1e6

    def request(self, segs: list[Seg], t: float, ttl: str) -> Usage:
        h, c = _chain(segs), _cum(segs)
        total = c[-1]
        u = Usage()
        read = 0.0
        if self._obj is not None:
            oh, tok, _ = self._obj
            if len(oh) <= len(h) and h[len(oh) - 1] == oh[-1]:
                read = tok
        u.read = read
        u.input = total - read
        self._n += 1
        if total >= self.min_tokens and (self._obj is None or self._n % self.every == 0):
            u.storage_usd += self._storage(t)
            u.write_5m = total  # creation, priced at the standard input rate (mult 1.0)
            self._obj = (h, total, t)
        return u

    def close(self, t: float) -> float:
        s = self._storage(t)
        self._obj = None
        return s


def provider(name: str, model: str = "claude-sonnet-5-5", **kw: Any) -> Any:
    """The provider models the simulator ships, by name."""
    if name == "anthropic":
        return Anthropic(model=model, **kw)
    if name == "openai-5.6":  # gpt-5.6-terra
        return PrefixCache(name, Price(2.00, 12.00, 0.10, 1.25), 1024, 1800.0, 1, True)
    if name == "openai-5.4":  # gpt-5.4, pre-5.6 rules
        return PrefixCache(name, Price(2.50, 15.00, 0.10), 1024, 1800.0, 128, False)
    if name == "gemini-implicit":  # gemini-3.1-pro-preview, <=200k prompts
        return PrefixCache(name, Price(2.00, 12.00, 0.10), 4096, 300.0, 1, False, **kw)
    if name == "gemini-explicit":
        return GeminiExplicit(Price(2.00, 12.00, 0.10, 1.0), 4.50, **kw)
    raise KeyError(name)


PROVIDERS = ("anthropic", "openai-5.6", "openai-5.4", "gemini-implicit", "gemini-explicit")

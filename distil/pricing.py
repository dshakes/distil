"""Token pricing with the prompt-caching cost model.

Prices are USD per *million* tokens and are CONFIGURABLE — verify against the
current provider pricing page before trusting absolute dollars. The cache
multipliers follow Anthropic's documented model:
  * a 5-minute cache *write* costs 1.25x the base input price, and
  * a cache *read* (hit) costs 0.10x the base input price.
That 10x gap between fresh input and cached read is the entire reason
cache-aware compression beats naive compression.
"""

from __future__ import annotations

from dataclasses import dataclass

# The single source of model facts the code relies on (drift canary:
# tests/test_pricing_canary.py; offline maintainer check: scripts/check_pricing.py).

#: Cache economics as multipliers of the base input price (Anthropic's documented model).
CACHE_WRITE_5M_MULT = 1.25  # 5-minute TTL cache write
CACHE_WRITE_1H_MULT = 2.0  # 1-hour TTL cache write (what Claude Code asks for)
CACHE_READ_MULT = 0.10  # cache hit

#: Prompt-cache lifetime in seconds by ``cache_control.ttl`` (``None`` = the default).
CACHE_TTL_S: dict[str | None, float] = {None: 300.0, "5m": 300.0, "1h": 3600.0}

#: The model distil prices and configures by default, per provider family.
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-5.2"

#: Model ids distil routes but deliberately does NOT price: ``resolve()`` returns None
#: for them and callers count their tokens unweighted (never at Claude rates).
UNPRICED: frozenset[str] = frozenset({DEFAULT_OPENAI_MODEL})


@dataclass(frozen=True)
class Pricing:
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_mult: float = CACHE_WRITE_5M_MULT
    cache_read_mult: float = CACHE_READ_MULT
    cache_write_1h_mult: float = CACHE_WRITE_1H_MULT

    # per-token USD
    @property
    def input(self) -> float:
        return self.input_per_mtok / 1_000_000

    @property
    def output(self) -> float:
        return self.output_per_mtok / 1_000_000

    @property
    def cache_write(self) -> float:
        return self.input * self.cache_write_mult

    @property
    def cache_write_1h(self) -> float:
        return self.input * self.cache_write_1h_mult

    @property
    def cache_read(self) -> float:
        return self.input * self.cache_read_mult


# Public list prices (USD / Mtok), current Claude model IDs. VERIFY before billing use.
# Checked against https://platform.claude.com/docs/en/about-claude/pricing on 2026-10-04.
# Newer models whose cache hit is NOT 0.1x carry their own multiplier: Opus 5.5 reads
# at 0.05x and Fable 5.1 at 0.025x; without a row they would resolve, through the
# prefix fallback, to the older model's price.
CATALOG: dict[str, Pricing] = {
    "claude-fable-5-1": Pricing("claude-fable-5-1", 10.0, 50.0, cache_read_mult=0.025),
    "claude-fable-5": Pricing("claude-fable-5", 10.0, 50.0),
    "claude-opus-5-5": Pricing("claude-opus-5-5", 4.0, 20.0, cache_read_mult=0.05),
    "claude-opus-5": Pricing("claude-opus-5", 5.0, 25.0),
    "claude-opus-4-8": Pricing("claude-opus-4-8", 5.0, 25.0),
    "claude-opus-4-7": Pricing("claude-opus-4-7", 5.0, 25.0),
    "claude-opus-4-6": Pricing("claude-opus-4-6", 5.0, 25.0),
    "claude-opus-4-5": Pricing("claude-opus-4-5", 5.0, 25.0),
    # $2/$10 is Sonnet 5's standard price: the scheduled 2026-09-01 rise to $3/$15
    # was cancelled (the pricing page's footnote).
    "claude-sonnet-5-5": Pricing("claude-sonnet-5-5", 2.0, 10.0),
    "claude-sonnet-5": Pricing("claude-sonnet-5", 2.0, 10.0),
    "claude-sonnet-4-6": Pricing("claude-sonnet-4-6", 3.0, 15.0),
    "claude-sonnet-4-5": Pricing("claude-sonnet-4-5", 3.0, 15.0),
    "claude-haiku-4-5": Pricing("claude-haiku-4-5", 1.0, 5.0),
}


def get(name: str) -> Pricing:
    if name not in CATALOG:
        raise KeyError(f"unknown model {name!r}; known: {sorted(CATALOG)}")
    return CATALOG[name]


def resolve(model_id: str | None) -> Pricing | None:
    """Best-effort catalog lookup for a *wire* model id, or None when unknown.

    Handles the id shapes seen in real traffic: exact ids, dated snapshots
    (``claude-haiku-4-5-20251001``), Bedrock's ``anthropic.`` prefix, and
    Vertex's ``@`` version separator. Returning None (rather than guessing a
    price) is deliberate — an unknown model (e.g. a Gemini/OpenAI upstream)
    must never be silently billed at Claude rates.
    """
    if not model_id:
        return None
    mid = model_id.strip()
    if mid.startswith("anthropic."):
        mid = mid[len("anthropic.") :]
    mid = mid.split("@", 1)[0]
    if mid in CATALOG:
        return CATALOG[mid]
    # Dated snapshot / suffixed variant: longest catalog id that prefixes it.
    best = None
    for name, price in CATALOG.items():
        if mid.startswith(name + "-") and (best is None or len(name) > len(best.name)):
            best = price
    return best

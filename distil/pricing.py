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
#: for them and callers count their tokens unweighted (never at Claude rates). Empty since
#: the OpenAI and Gemini rows below: kept so a model can be routed before it is priced.
UNPRICED: frozenset[str] = frozenset()


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


def _auto(name: str, inp: float, cached: float, out: float, write: float = 1.0) -> Pricing:
    """A row for a provider that caches prefixes automatically (OpenAI, Gemini).

    The cache hit is the published cached-input price over the input price. There is no
    write surcharge (``write`` 1.0) unless the page lists one — OpenAI's GPT-5.6+ and 6.x
    rows bill a cache write at 1.25x — and no 1-hour write tier, so both write multipliers
    are the same. Writes only enter a cost when the provider REPORTS them.
    """
    return Pricing(name, inp, out, write, round(cached / inp, 6), write)


# OpenAI standard tier, short-context rows (<=272K input). Checked against
# https://developers.openai.com/api/docs/pricing on 2026-10-06. VERIFY before billing use.
# ponytail: one tier per row; long-context (>272K) prompts are underpriced ~2x. Add a
# threshold field if that traffic shows up.
CATALOG.update(
    {
        "gpt-6-astra": _auto("gpt-6-astra", 10.0, 1.0, 50.0, 1.25),
        "gpt-6.1-sol": _auto("gpt-6.1-sol", 2.0, 0.10, 10.0, 1.25),
        "gpt-6-sol": _auto("gpt-6-sol", 2.0, 0.20, 10.0, 1.25),
        "gpt-6-luna": _auto("gpt-6-luna", 0.10, 0.01, 0.50, 1.25),
        "gpt-5.6-sol": _auto("gpt-5.6-sol", 4.0, 0.40, 20.0, 1.25),
        "gpt-5.6-terra": _auto("gpt-5.6-terra", 2.0, 0.20, 12.0, 1.25),
        "gpt-5.6-luna": _auto("gpt-5.6-luna", 0.20, 0.02, 1.20, 1.25),
        "gpt-5.5": _auto("gpt-5.5", 5.0, 0.50, 30.0),
        "gpt-5.4": _auto("gpt-5.4", 2.5, 0.25, 15.0),
        "gpt-5.4-mini": _auto("gpt-5.4-mini", 0.75, 0.075, 4.5),
        "gpt-5.3-codex": _auto("gpt-5.3-codex", 1.75, 0.175, 14.0),
        "gpt-5.2": _auto("gpt-5.2", 1.75, 0.175, 14.0),
        "gpt-5.1": _auto("gpt-5.1", 1.25, 0.125, 10.0),
        "gpt-5": _auto("gpt-5", 1.25, 0.125, 10.0),
        "gpt-5-mini": _auto("gpt-5-mini", 0.25, 0.025, 2.0),
        "gpt-5-nano": _auto("gpt-5-nano", 0.05, 0.005, 0.40),
        "gpt-4.1": _auto("gpt-4.1", 2.0, 0.50, 8.0),
        "gpt-4.1-mini": _auto("gpt-4.1-mini", 0.40, 0.10, 1.60),
        "gpt-4o": _auto("gpt-4o", 2.5, 1.25, 10.0),
        "gpt-4o-mini": _auto("gpt-4o-mini", 0.15, 0.075, 0.60),
        "o3": _auto("o3", 2.0, 0.50, 8.0),
        "o3-mini": _auto("o3-mini", 1.10, 0.55, 4.40),
        # Pro rows publish no cached-input price: no discount. Listed so the prefix
        # fallback cannot price gpt-5-pro as gpt-5 (12x cheaper).
        "gpt-5.5-pro": _auto("gpt-5.5-pro", 30.0, 30.0, 180.0),
        "gpt-5.4-pro": _auto("gpt-5.4-pro", 30.0, 30.0, 180.0),
        "gpt-5.2-pro": _auto("gpt-5.2-pro", 21.0, 21.0, 168.0),
        "gpt-5-pro": _auto("gpt-5-pro", 15.0, 15.0, 120.0),
        "o4-mini": _auto("o4-mini", 1.10, 0.275, 4.40),
    }
)

# Gemini API paid tier, Standard, text input. Checked against
# https://ai.google.dev/gemini-api/docs/pricing on 2026-10-06; "context caching price" is
# the cache hit (implicit caching bills no write; explicit-cache storage per hour is out of
# scope). Output includes thinking tokens. VERIFY before billing use.
# ponytail: the <=200k-prompt tier only (Pro rows double above it); the 3.6-3.8 Flash
# price is the one published through 2026-12-31 and doubles on 2027-01-01.
CATALOG.update(
    {
        "gemini-3.1-pro-preview": _auto("gemini-3.1-pro-preview", 2.0, 0.20, 12.0),
        "gemini-3.8-flash": _auto("gemini-3.8-flash", 0.75, 0.075, 3.75),
        "gemini-3.7-flash": _auto("gemini-3.7-flash", 0.75, 0.075, 3.75),
        "gemini-3.6-flash": _auto("gemini-3.6-flash", 0.75, 0.075, 3.75),
        "gemini-3.5-flash": _auto("gemini-3.5-flash", 1.50, 0.15, 9.0),
        "gemini-3.1-flash-lite": _auto("gemini-3.1-flash-lite", 0.25, 0.025, 1.50),
        "gemini-3-flash-preview": _auto("gemini-3-flash-preview", 0.50, 0.05, 3.0),
        "gemini-2.5-pro": _auto("gemini-2.5-pro", 1.25, 0.125, 10.0),
        "gemini-2.5-flash": _auto("gemini-2.5-flash", 0.30, 0.03, 2.50),
        "gemini-2.5-flash-lite": _auto("gemini-2.5-flash-lite", 0.10, 0.01, 0.40),
    }
)

#: Routing prefixes a wire model id can carry in front of the catalog name: LiteLLM's
#: provider prefixes and Gemini's resource name (``models/gemini-2.5-pro``).
_ROUTE_PREFIXES = ("anthropic/", "openai/", "azure/", "gemini/", "google/", "vertex_ai/", "models/")


def get(name: str) -> Pricing:
    if name not in CATALOG:
        raise KeyError(f"unknown model {name!r}; known: {sorted(CATALOG)}")
    return CATALOG[name]


def resolve(model_id: str | None) -> Pricing | None:
    """Best-effort catalog lookup for a *wire* model id, or None when unknown.

    Handles the id shapes seen in real traffic: exact ids, dated snapshots
    (``claude-haiku-4-5-20251001``), Bedrock's ``anthropic.`` prefix, and
    Vertex's ``@`` version separator, LiteLLM's ``openai/``-style provider prefixes and
    Gemini's ``models/`` resource prefix. Returning None (rather than guessing a
    price) is deliberate — an unknown model must never be silently billed at
    another model's rates.
    """
    if not model_id:
        return None
    mid = model_id.strip()
    for pre in _ROUTE_PREFIXES:
        if mid.startswith(pre):
            mid = mid[len(pre) :]
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

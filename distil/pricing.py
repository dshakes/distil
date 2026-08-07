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


@dataclass(frozen=True)
class Pricing:
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_mult: float = 1.25  # 5-minute TTL cache write
    cache_read_mult: float = 0.10  # cache hit

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
    def cache_read(self) -> float:
        return self.input * self.cache_read_mult


# Public list prices (USD / Mtok), current Claude model IDs. VERIFY before billing use.
CATALOG: dict[str, Pricing] = {
    "claude-fable-5": Pricing("claude-fable-5", 10.0, 50.0),
    "claude-opus-5": Pricing("claude-opus-5", 5.0, 25.0),
    "claude-opus-4-8": Pricing("claude-opus-4-8", 5.0, 25.0),
    "claude-opus-4-7": Pricing("claude-opus-4-7", 5.0, 25.0),
    "claude-opus-4-6": Pricing("claude-opus-4-6", 5.0, 25.0),
    "claude-opus-4-5": Pricing("claude-opus-4-5", 5.0, 25.0),
    "claude-sonnet-5": Pricing("claude-sonnet-5", 3.0, 15.0),
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

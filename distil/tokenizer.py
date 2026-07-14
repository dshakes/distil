"""Token counting — pluggable.

The default is an offline heuristic so the whole system runs with zero
dependencies and no API key. Production should swap in the provider's real
tokenizer (Anthropic's `count_tokens` endpoint, or a local BPE) via the
`Tokenizer` protocol.

Honesty note: compression *ratios* are robust to the estimator (they cancel),
but absolute dollar figures depend on it. Treat the offline numbers as
directionally correct, not billing-grade, until a real tokenizer is wired in.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

# Word-ish + punctuation segmentation. BPE tends to split long/rare words into
# multiple sub-word units, so we inflate the raw piece count by a factor that
# lands within ~10-15% of cl100k/Claude tokenizers on mixed code+prose.
_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    def __init__(self, subword_factor: float = 1.33) -> None:
        self.subword_factor = subword_factor

    def count(self, text: str) -> int:
        if not text:
            return 0
        pieces = _PIECE.findall(text)
        return max(1, round(len(pieces) * self.subword_factor))


class AnthropicTokenizer:
    """Billing-grade Claude token counts via the Messages count_tokens endpoint.

    This is the *correct* tokenizer for Claude — tiktoken (OpenAI's BPE)
    undercounts Claude by ~15-20%, more on code. Requires the `anthropic` SDK
    and credentials (one network call per unique string). Counts are
    model-specific, so the model id must match what you price/run against.
    Results are memoized.
    """

    def __init__(self, model: str = "claude-opus-4-8", client: object | None = None) -> None:
        self.model = model
        self._client = client
        self._cache: dict[str, int] = {}

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from anthropic import Anthropic  # lazy: keep the core dependency-free
            except ModuleNotFoundError:
                raise SystemExit(
                    "distil: the 'anthropic' package is needed for --tokenizer anthropic.\n"
                    "  install it:  pipx inject distil-llm anthropic   "
                    "(or: pip install anthropic)"
                ) from None
            try:
                self._client = Anthropic()
            except Exception as exc:  # noqa: BLE001 — missing/invalid key, etc.
                raise SystemExit(
                    f"distil: could not initialise the Anthropic client — {exc}\n"
                    "  set your key:  export ANTHROPIC_API_KEY=sk-ant-..."
                ) from None
        return self._client

    def count(self, text: str) -> int:
        if not text:
            return 0
        if text in self._cache:
            return self._cache[text]
        client = self._ensure_client()
        try:
            resp = client.messages.count_tokens(  # type: ignore[attr-defined]
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
        except Exception as exc:  # noqa: BLE001 — missing key, network, rate-limit, etc.
            raise SystemExit(
                f"distil: the Anthropic token count call failed — {exc}\n"
                "  set your key:  export ANTHROPIC_API_KEY=sk-ant-...   "
                "(the offline default needs no key: drop --tokenizer anthropic)"
            ) from None
        self._cache[text] = resp.input_tokens
        return resp.input_tokens


DEFAULT: Tokenizer = HeuristicTokenizer()


class SubwordHeuristicTokenizer:
    """A closer offline BPE approximation than ``piece × 1.33`` — still zero-dependency.

    Byte-pair encoding splits *longer / rarer* words into more sub-tokens, so a single flat
    factor over word-pieces systematically mis-estimates: it over-counts short common words and
    under-counts long identifiers. This estimates tokens per alphanumeric piece from its length
    (~one BPE token per ~4 characters, the well-known rule of thumb), counts each punctuation
    symbol as one, and adds a small surcharge for non-ASCII (which BPE encodes as multiple byte
    tokens). Pair it with self-calibration (`distil.calibration`) for near-billing-grade counts
    with no network. Base for `--tokenizer subword`.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        toks = 0.0
        for piece in _PIECE.findall(text):
            if piece[:1].isalnum():
                toks += max(1.0, len(piece) / self.chars_per_token)
                # non-ASCII bytes cost extra BPE tokens (UTF-8 multi-byte → multiple byte tokens)
                non_ascii = sum(1 for ch in piece if ord(ch) > 127)
                toks += non_ascii * 0.5
            else:
                toks += 1.0  # a punctuation / symbol piece ≈ one token
        return max(1, round(toks))


def resolve(name: str = "heuristic", *, model: str = "claude-opus-4-8") -> Tokenizer:
    """Factory: 'heuristic' (flat, default), 'subword' (length-aware offline, closer to BPE),
    or 'anthropic' (billing-grade, needs the SDK + a network call per string)."""
    if name == "heuristic":
        return HeuristicTokenizer()
    if name == "subword":
        return SubwordHeuristicTokenizer()
    if name == "anthropic":
        return AnthropicTokenizer(model=model)
    raise ValueError(f"unknown tokenizer {name!r}; choose 'heuristic', 'subword', or 'anthropic'")

"""Distil — compression with a quality contract.

Thesis: in an agentic runtime you don't need byte-equivalence, you need
*decision-equivalence* — the agent takes the same actions and produces the same
final outputs whether or not the context was compressed. That is measurable,
certifiable, and compatible with aggressive compression.

This package demonstrates the two highest-leverage techniques end-to-end:
  1. Cache-aware compression (distil.compress.cache_aware) — the dominant cost
     in a multi-turn agent loop is cache *misses*, not context *size*. Keep the
     prefix byte-stable, compress only the volatile tail. Priced in real dollars.
  4. Causal / counterfactual pruning (distil.replay.ablation) — the eval engine
     is not a ruler, it is a discovery engine: it finds context that never
     changes a decision and is therefore free to drop. Certified by a TOST
     non-inferiority gate (distil.certify.stats).
"""

# Single-sourced from the installed distribution's metadata (pyproject `version`),
# so `distil --version` can never drift from the published package. The literal
# fallback is only used when running from source/zipapp with no installed metadata.
from importlib.metadata import PackageNotFoundError, version as _pkg_version  # noqa: E402

try:
    __version__ = _pkg_version("distil-llm")
except PackageNotFoundError:  # source checkout / zipapp without dist-info
    # A zipapp has no dist-info AND cannot read_text() its own pyproject, so the
    # branch below can only ever return "0+source" there. build_pyz.sh stamps
    # this module from pyproject at build time; zipimport can import it.
    try:
        from distil._version import __version__  # type: ignore[no-redef]
    except ImportError:
        # Read pyproject directly rather than hardcode a literal — a duplicated
        # version string drifts from the real one AND conflicts on every release
        # merge-back (a conflicted pyproject then bricks `uv run`). Single source.
        try:
            import pathlib
            import re

            _pp = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
                encoding="utf-8"
            )
            _m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', _pp)
            __version__ = _m.group(1) if _m else "0+source"
        except Exception:  # noqa: BLE001 — version must never break import
            __version__ = "0+source"


# ---------------------------------------------------------------------------
# Public library API
# ---------------------------------------------------------------------------
# Re-exported lazily via __getattr__ (PEP 562): `import distil` must stay cheap
# because the CLI imports this package on every `--version`/completion call, and
# the compression stack pulls in the whole adapter tree. Importing the name is
# what pays for it:  `from distil import compress_messages`.
#
# NOT named `compress`: `distil.compress` is an existing SUBPACKAGE. Python binds
# a submodule onto its parent package as soon as anything imports it, which would
# silently shadow a same-named __getattr__ export — so `from distil import compress`
# would hand back the module in any program that had touched the subpackage, and
# the function otherwise. A name that resolves to two different objects depending
# on unrelated import order is not an API. `compress_messages` also matches the
# vocabulary already used in adapters/anthropic.py and integrations/langchain.py.

__all__ = ["CompressionResult", "compress_messages", "expand_handle", "__version__"]

_LAZY = {"compress_messages": "api", "expand_handle": "api", "CompressionResult": "api"}


def __getattr__(name: str) -> object:  # pragma: no cover - trivial dispatch
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__() -> list[str]:  # pragma: no cover - REPL/tab-completion affordance
    return sorted(__all__)

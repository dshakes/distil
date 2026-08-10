"""Auth-mode gating (Roadmap Phase 4) — a safety boundary, not an optimization.

The hard-won lesson from this space: aggressive, lossy, tool-injecting
compression is fine on a metered pay-as-you-go key, but applying it to a
subscription / OAuth session can alter a metered conversation in ways that
violate provider terms (injected retrieval tools the user never authorized,
rewritten history). So the *mode* gates which strategies are even allowed:

  * PAYG          — full toolbox, including lossy strategies and tool injection.
  * SUBSCRIPTION  — lossless-only *by default*. No lossy output shaping, and no
                    Tier-1 digest stubs — because with no expand tool injected the
                    agent could never recover a stub, so it would be irreversibly
                    lossy in context. That is the reason for the verbatim force, and
                    it is conditional: an explicit `--expand` injects distil_expand,
                    which makes every stub recoverable — the exact hazard the force
                    guards against no longer exists. So an informed user opt-in
                    (`--expand`) re-enables the recoverable digest even here; the
                    default (no flag) stays lossless-only, and genuinely-lossy output
                    shaping stays PAYG-only regardless. See build_handler / issue #28.

This is a tightening *default*, not a cage: the software never silently applies
lossy or tool-injecting compression to a subscription, but it also does not override
a user who explicitly, knowingly asks for recoverable expand. A project config still
cannot loosen the default — only the end user, per-invocation, can.
"""

from __future__ import annotations

from enum import Enum


class AuthMode(str, Enum):
    PAYG = "payg"
    SUBSCRIPTION = "subscription"  # also covers OAuth / first-party app sessions


class PolicyError(RuntimeError):
    """Raised when a strategy is not permitted under the active auth mode."""


# Lossless strategies are always permitted; lossy ones only on PAYG.
_LOSSLESS = {"none", "distil"}
_LOSSY = {"naive", "aggressive"}


def allowed_strategies(mode: AuthMode) -> set[str]:
    if mode is AuthMode.PAYG:
        return _LOSSLESS | _LOSSY
    return set(_LOSSLESS)


def may_compress_lossy(mode: AuthMode) -> bool:
    return mode is AuthMode.PAYG


# There was a `may_inject_tools(mode)` here returning `mode is AuthMode.PAYG`,
# with tests asserting it. Nothing ever called it. The rule it stated — tool
# injection is PAYG-only — is not the rule distil actually implements: a
# subscription session that passes `--expand` DOES get distil_expand injected,
# deliberately and with a notice (see proxy.py), because an injected recovery
# tool is what makes a Tier-1 digest reversible.
#
# An unenforced policy with a passing test is worse than no policy: the test
# makes it look covered, so the drift between what distil says and what distil
# does is invisible in CI. The real rule is "injection on a flat-rate session
# requires an explicit opt-in", it lives at the one place that can enforce it,
# and ADR 0006 records why. Do not reintroduce this without a call site.


def guard(mode: AuthMode, strategy: str) -> None:
    """Raise PolicyError if `strategy` is not permitted under `mode`."""
    if strategy not in allowed_strategies(mode):
        raise PolicyError(
            f"strategy {strategy!r} is not permitted under auth mode {mode.value!r} "
            f"(lossless-only); allowed: {sorted(allowed_strategies(mode))}"
        )

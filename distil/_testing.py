"""The one door for test hooks: a hook is read only when ``DISTIL_TESTING=1``.

Production code must not grow behaviour switches that exist for the test suite
(a stray `DISTIL_HOTSWAP_TEST_FAIL_READY` in a user's shell would kill a worker).
Every such hook reads its variable through :func:`hook`, which returns the default unless the opt-in is set.
"""

from __future__ import annotations

import os

SWITCH = "DISTIL_TESTING"


def hook(name: str, default: str | None = None) -> str | None:
    """``os.environ.get(name, default)`` if ``DISTIL_TESTING=1``, else ``default``."""
    if os.environ.get(SWITCH) != "1":
        return default
    return os.environ.get(name, default)

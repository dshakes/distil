"""Guards docs/claims.json: the ledger of every numeric/comparative claim on the site.

Two things can go stale silently: a claim's snippet or anchor can drift off the
page it says it's on (the page changed, the ledger didn't), or a claim can be
quietly added or removed from the ledger without anyone noticing. Both are
caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CLAIMS_PATH = DOCS_DIR / "claims.json"
VALID_STATUSES = {"verified", "stale", "wrong", "unsourced"}

# Frozen total: bump this deliberately (in the same commit that edits
# claims.json) whenever a claim is genuinely added or removed. A change here
# without a matching, reviewed claims.json edit is exactly the drift this
# test exists to catch.
EXPECTED_ENTRY_COUNT = 21


def _load_claims() -> list[dict]:
    data = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    return data["entries"]


def _pages(entry: dict) -> list[str]:
    page = entry["page"]
    return [page] if isinstance(page, str) else list(page)


def test_claims_file_is_valid_json_with_expected_shape():
    entries = _load_claims()
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate claim id in docs/claims.json"
    for entry in entries:
        assert entry["status"] in VALID_STATUSES, entry["id"]
        assert entry["page"], entry["id"]
        assert entry["claim"], entry["id"]


def test_claim_count_is_frozen():
    """A silent add/remove is the failure mode this test exists for.

    If this fails because you deliberately added or resolved a claim, update
    EXPECTED_ENTRY_COUNT in this file in the same commit as the claims.json
    change.
    """
    entries = _load_claims()
    assert len(entries) == EXPECTED_ENTRY_COUNT, (
        f"docs/claims.json has {len(entries)} entries, expected "
        f"{EXPECTED_ENTRY_COUNT} — a claim was added or removed without "
        "updating this frozen count"
    )


@pytest.mark.parametrize("entry", _load_claims(), ids=lambda e: e["id"])
def test_claim_locator_still_matches_the_live_page(entry: dict):
    for page in _pages(entry):
        path = DOCS_DIR / page
        assert path.is_file(), f"{entry['id']}: {page} does not exist"
        text = path.read_text(encoding="utf-8")

        anchor = entry.get("anchor")
        if anchor:
            needle = f'id="{anchor}"'
            assert needle in text, f"{entry['id']}: anchor {needle!r} no longer found on {page}"

        snippet = entry.get("snippet")
        if snippet:
            assert snippet in text, (
                f"{entry['id']}: snippet {snippet!r} no longer found on {page} "
                "— the claim moved, changed, or was removed; update claims.json"
            )

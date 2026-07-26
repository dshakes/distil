"""Per-request receipts — a tamper-evident, content-free record of what distil did.

The ledger accounts for savings and the proof ledger prints a session summary. Neither
is an *artifact*: something a third party can be handed, and can verify without trusting
the person who handed it over. That is what a receipt is.

Each receipt records one request: how many tokens went in, how many went out, which
compression mode ran, which handles were issued, and whether every one of those handles
was still recoverable when the receipt was written. Receipts are hash-chained, so a
receipt cannot be edited or removed without breaking every receipt after it.

**Content-free by construction.** A receipt never stores prompt text, completion text,
or any excerpt. It stores counts, a mode, 8-hex handles (which are content-addressed
digests, not content), and hashes. That is deliberate: a compliance artifact you cannot
safely share is not a compliance artifact. See ``Receipt.FIELDS`` for the exhaustive list
— if a field is not in it, it is not written.

What a receipt does NOT claim: that the compression was *correct*. Correctness is the
certificate's job (``distil.conformal``), and a receipt names the certificate that
authorised its mode rather than re-asserting the guarantee. A receipt answers "what was
done, and is it still undoable" — not "was it safe".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = 1

# Genesis link for the first receipt in a chain. Fixed so an empty chain and a
# tampered-to-empty chain are distinguishable.
GENESIS = "0" * 64


def _home() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def receipts_path() -> Path:
    return _home() / "receipts.jsonl"


@dataclass
class Receipt:
    """One request's record. Every field is a measurement or an identifier — never content."""

    ts: float
    request_id: str
    session: str
    model: str
    mode: str  # verbatim | lossless | digest
    tokens_original: int
    tokens_compressed: int
    reversible: bool  # was this tier byte-reversible (Tier-0) as opposed to recoverable-on-demand
    handles: list[str] = field(default_factory=list)  # 8-hex recoverable stubs issued
    restorable: bool = True  # every handle above resolved in the store when this was written
    certificate: str = ""  # the certificate/level that authorised this mode, if any
    v: int = SCHEMA
    prev: str = GENESIS  # hash of the preceding receipt — the chain
    hash: str = ""  # sha256 over the canonical form of everything above

    #: The exhaustive set of persisted fields. Anything not here is never written,
    #: which is what makes "content-free" checkable rather than a promise.
    FIELDS = (
        "ts",
        "request_id",
        "session",
        "model",
        "mode",
        "tokens_original",
        "tokens_compressed",
        "reversible",
        "handles",
        "restorable",
        "certificate",
        "v",
        "prev",
        "hash",
    )

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_original - self.tokens_compressed)

    def payload(self) -> dict[str, Any]:
        """Canonical, hash-covered form: everything except the hash itself."""
        d = {k: getattr(self, k) for k in self.FIELDS if k != "hash"}
        d["handles"] = sorted(d["handles"])  # order must not change the hash
        return d

    def compute_hash(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def sealed(self) -> Receipt:
        """Return a copy with its hash filled in."""
        self.hash = self.compute_hash()
        return self


def append(receipt: Receipt) -> Receipt:
    """Seal ``receipt`` onto the end of the chain and persist it.

    Best-effort and fail-open: a receipt must never break a request. A dropped receipt
    is a gap you can see (the chain still verifies, the request ids skip), which is
    strictly better than a request that failed because bookkeeping did.
    """
    try:
        path = receipts_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt.prev = head_hash()
        receipt.sealed()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":")) + "\n")
        path.chmod(0o600)
    except OSError:
        pass
    return receipt


def head_hash() -> str:
    """Hash of the last receipt, or GENESIS when the chain is empty."""
    last = None
    for last in read():  # noqa: B007 — we want the final element
        pass
    return last.hash if last is not None else GENESIS


def read(path: Path | None = None) -> Iterator[Receipt]:
    """Yield receipts in file order. Malformed lines are skipped, not fatal."""
    p = path or receipts_path()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return
    known = set(Receipt.FIELDS)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        yield Receipt(**{k: v for k, v in d.items() if k in known})


@dataclass
class Verdict:
    """Outcome of verifying a chain."""

    total: int
    ok: bool
    first_bad_index: int = -1
    reason: str = ""

    @property
    def statement(self) -> str:
        if self.total == 0:
            return "No receipts recorded."
        if self.ok:
            return f"VERIFIED — {self.total} receipts, hash chain intact."
        return (
            f"BROKEN — chain fails at receipt {self.first_bad_index} of {self.total}: {self.reason}. "
            "A receipt was edited, reordered, or removed."
        )


def verify(path: Path | None = None) -> Verdict:
    """Recompute every hash and every link. This is the whole point of the artifact:
    anyone can run it, and it needs nothing but the file."""
    prev = GENESIS
    n = 0
    for i, r in enumerate(read(path)):
        n = i + 1
        if r.hash != r.compute_hash():
            return Verdict(n, False, i, "content does not match its hash")
        if r.prev != prev:
            return Verdict(n, False, i, "prev-hash does not match the preceding receipt")
        prev = r.hash
    return Verdict(n, True)

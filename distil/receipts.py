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

import contextlib
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import atrest

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

    **Reading the head and appending are one critical section.** A chain is a
    read-modify-write: two concurrent requests that both read head ``H`` both write a
    receipt claiming ``prev == H``, which forks the chain — and since the ledger now
    verifies it on every render, that surfaces as "chain BROKEN" on traffic that is
    perfectly healthy. The lock is the same cross-platform advisory lock every other
    store here uses, and is fail-open in the same way: unobtainable degrades to no lock
    rather than raising, because bookkeeping must not be the thing that fails a request.
    The owner-only opener below sits INSIDE that section: mode-at-creation and chain
    ordering are two properties of the same single write, not two writes.
    """
    from . import _filelock

    try:
        path = receipts_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _filelock.locked(path):
            receipt.prev = head_hash()
            receipt.sealed()
            # 0600 AT CREATION via the opener, not by the chmod below: a chmod after
            # the write leaves the file at the process umask for the whole write, and
            # a receipt names a session, a model and its handles. The chmod stays as
            # the upgrade path for a chain file created before this.
            with open(path, "a", encoding="utf-8", opener=atrest.owner_only) as fh:
                fh.write(json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":")) + "\n")
        path.chmod(0o600)
    except OSError:
        pass
    return receipt


_TAIL_BLOCK_SIZE = 64 * 1024  # generous for one JSON receipt line; a longer line just
# costs another block, never a wrong answer — see _reverse_lines.


def _reverse_lines(p: Path) -> Iterator[bytes]:
    """Yield raw lines from ``p``, last line first, reading backward in blocks.

    Cost is bounded by the trailing bytes actually inspected, not by the file: a
    multi-GB chain and a 200-byte one cost the same when the tail is healthy. A line
    longer than one block is held whole, but each block is searched once and the
    pieces are joined once — linear in the line, never quadratic. That matters for
    the unhealthy tail: a crash can leave megabytes of garbage with no newline, and
    this runs inside the append lock on every request.
    """
    try:
        with p.open("rb") as fh:
            pos = fh.seek(0, os.SEEK_END)
            parts: list[bytes] = []  # the line being assembled, last piece first
            while pos > 0:
                read_size = min(_TAIL_BLOCK_SIZE, pos)
                pos -= read_size
                fh.seek(pos)
                block = fh.read(read_size)
                end = len(block)
                nl = block.rfind(b"\n", 0, end)
                while nl != -1:
                    parts.append(block[nl + 1 : end])
                    yield b"".join(reversed(parts))
                    parts = []
                    end = nl
                    nl = block.rfind(b"\n", 0, end)
                parts.append(block[:end])
            head = b"".join(reversed(parts))
            if head:
                yield head
    except OSError:
        return


def head_hash() -> str:
    """Hash of the last receipt, or GENESIS when the chain is empty.

    Reads backward from the end of the file rather than the whole chain: this runs
    inside the append lock on every successful request (see ``append`` above), so an
    O(chain) scan here serializes every request behind however large the chain has
    grown — 0.53 s per request on a 94 MB, 102,767-row chain. A malformed or torn
    trailing line — a write cut short by a crash or a full disk — is skipped exactly
    as ``read()`` skips it going forward; this just walks backward to find the first
    line that parses instead of the last line forward that does.
    """
    for raw in _reverse_lines(receipts_path()):
        r = _parse(raw)
        if r is not None:
            return r.hash
    return GENESIS


def _parse(raw: bytes) -> Receipt | None:
    """One line into a receipt, or ``None`` if it is not one. Never raises."""
    line = raw.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    known = set(Receipt.FIELDS)
    try:
        return Receipt(**{k: v for k, v in d.items() if k in known})
    except TypeError:
        return None


def _lines(p: Path, start: int = 0) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(line_start, line_end, raw)`` from ``start``, streaming.

    Binary and one line at a time: the chain file is append-only and unbounded (83 MB
    on the maintainer's box), and ``read_text().splitlines()`` would hold the whole
    thing plus a list of every line. Offsets are accumulated rather than taken from
    ``tell()``, which text-mode iteration forbids anyway.
    """
    try:
        with p.open("rb") as fh:
            if start:
                fh.seek(start)
            off = start
            for raw in fh:
                begin, off = off, off + len(raw)
                yield begin, off, raw
    except OSError:
        return


def read(path: Path | None = None) -> Iterator[Receipt]:
    """Yield receipts in file order. Malformed lines are skipped, not fatal."""
    for _begin, _end, raw in _lines(path or receipts_path()):
        r = _parse(raw)
        if r is not None:
            yield r


@dataclass
class Verdict:
    """Outcome of verifying a chain."""

    total: int
    ok: bool
    first_bad_index: int = -1
    reason: str = ""
    #: Index of the first receipt this pass actually re-hashed. ``0`` means the whole
    #: chain was checked; a higher number means the prefix was taken from this machine's
    #: own checkpoint (see :func:`verify`).
    checked_from: int = 0

    @property
    def statement(self) -> str:
        if self.total == 0:
            return "No receipts recorded."
        if self.ok:
            scope = (
                f"VERIFIED — {self.total} receipts, hash chain intact."
                if not self.checked_from
                else (
                    f"VERIFIED — {self.total} receipts, hash chain intact "
                    f"({self.total - self.checked_from} re-checked since this machine's last "
                    f"pass; `distil receipts` re-hashes all {self.total})."
                )
            )
            return scope
        return (
            f"BROKEN — chain fails at receipt {self.first_bad_index} of {self.total}: {self.reason}. "
            "A receipt was edited, reordered, or removed."
        )


def _checkpoint_path() -> Path:
    return _home() / "receipts-verified.json"


def _load_checkpoint() -> tuple[int, str, int, int] | None:
    """``(count, head_hash, tail_start, tail_end)`` from the last good full pass."""
    try:
        raw = json.loads(_checkpoint_path().read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return (
            int(raw["count"]),
            str(raw["head"]),
            int(raw["tail_start"]),
            int(raw["tail_end"]),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _save_checkpoint(count: int, head: str, tail_start: int, tail_end: int) -> None:
    """Persist the resume point. Best-effort: losing it costs a full re-scan, nothing more."""
    from . import _filelock

    path = _checkpoint_path()
    payload = {"count": count, "head": head, "tail_start": tail_start, "tail_end": tail_end}
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _filelock.locked(path):
            with open(tmp, "w", encoding="utf-8", opener=atrest.owner_only) as fh:
                fh.write(json.dumps(payload))
                fh.flush()
            _filelock.replace_retrying(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _scan(p: Path, count: int, prev: str, tail_start: int, tail_end: int) -> Verdict:
    """Stream from byte ``tail_end``, chaining from ``prev``, with ``count`` behind us.

    ``tail_start``/``tail_end`` bracket the last receipt already verified, and are carried
    through untouched when the file has grown by nothing — otherwise a render that finds no
    new receipts would move the resume point onto empty space and force the next one to
    start over.

    Never materialises the chain. On a break it stops hashing but keeps counting, because
    "receipt 3 of 40" has to name the receipts that exist, not the ones checked before the
    break — the second number is what tells a reader how much of the artifact is in question.
    """
    first = count
    bad: tuple[int, str] | None = None
    for begin, end, raw in _lines(p, tail_end):
        r = _parse(raw)
        if r is None:
            continue
        idx, count = count, count + 1
        if bad is not None:
            continue
        if r.hash != r.compute_hash():
            bad = (idx, "content does not match its hash")
        elif r.prev != prev:
            bad = (idx, "prev-hash does not match the preceding receipt")
        else:
            prev, tail_start, tail_end = r.hash, begin, end
    if bad is not None:
        return Verdict(count, False, bad[0], bad[1], first)
    if count:
        _save_checkpoint(count, prev, tail_start, tail_end)
    return Verdict(count, True, checked_from=first)


def verify(path: Path | None = None, *, full: bool = True) -> Verdict:
    """Recompute every hash and every link. This is the whole point of the artifact:
    anyone can run it, and it needs nothing but the file.

    **A third party always does the full pass.** Hand someone this file and they run
    ``verify(path)`` — an explicit path never consults a checkpoint, so the guarantee the
    artifact exists to provide is unchanged. What is cached is *this* machine re-checking
    *its own* chain, which now happens on every wrap exit and every ``distil stats``: the
    resume point records how far a previous pass got, and only receipts appended since are
    re-hashed. Without that, a session's exit summary re-hashes an 83 MB chain.

    The checkpoint can only make the answer *cheaper*, never wronger about the part it
    checks: it is validated by re-reading AND re-hashing the last receipt it claims to
    have verified, and any failure from a resumed pass is discarded and re-run in full, so
    a stale checkpoint cannot print BROKEN over a healthy chain.

    **What a resumed pass does not do**, stated plainly because the artifact's whole value
    is that it does not overclaim: it does not re-hash receipts this machine already
    verified. An edit buried in that prefix, made by something with write access to
    ``~/.distil``, is found by the default full pass and by anyone you hand the file to — not by the
    resumed pass (``full=False``, the per-exit line and ``distil receipts --fast``). That is the same trust boundary the chain always had (whoever can
    rewrite the receipts can rewrite the checkpoint beside them); what is new is that the
    fast path says so rather than implying a full audit it did not perform.
    """
    p = path or receipts_path()
    if path is None and not full:
        ck = _load_checkpoint()
        if ck is not None:
            count, head, tail_start, tail_end = ck
            # The resume point is only usable if the receipt it names is still there,
            # byte for byte. This is what makes truncate-and-regrow visible.
            last = next((_parse(raw) for _b, _e, raw in _lines(p, tail_start)), None)
            # RE-HASHED, not just compared: editing a field of that receipt leaves its
            # stored `hash` untouched, so trusting the stored value would let the most
            # recent receipt — the one most worth editing — be changed unnoticed.
            if (
                last is not None
                and last.hash == head
                and last.compute_hash() == head
                and tail_end >= tail_start
            ):
                v = _scan(p, count, head, tail_start, tail_end)
                if v.ok:
                    return v
    return _scan(p, 0, GENESIS, 0, 0)

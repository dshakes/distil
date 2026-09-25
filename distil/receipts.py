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
import logging
import os
import time
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
    """The ACTIVE chain file — the only one ever appended to."""
    return _home() / "receipts.jsonl"


log = logging.getLogger(__name__)

#: Rotate the active chain into a sealed segment once it reaches this size. Checked
#: with one ``stat`` per append; the seal itself reads the segment once, so the cost
#: per receipt is O(1) amortized. ``0`` disables rotation.
SEGMENT_BYTES = 32 * 1024 * 1024


def segments_dir() -> Path:
    return _home() / "receipts-segments"


def segment_path(seg: int) -> Path:
    return segments_dir() / f"{seg:06d}.jsonl"


def segment_checkpoint_path(seg: int) -> Path:
    return segments_dir() / f"{seg:06d}.checkpoint.json"


def sealed_segments() -> list[int]:
    """Ids of the sealed segments on disk, oldest first.

    A checkpoint with no segment beside it is NOT a segment: it is what a crash between
    writing the checkpoint and moving the active file leaves behind (see ``_seal``), and
    the next seal overwrites it.

    ponytail: a directory listing, O(segments). It runs on a seal and on the one
    ``head_hash`` right after it, never per request; at 32 MiB a segment that is a
    handful of entries a year. Keep a manifest if it ever reaches thousands.
    """
    try:
        names = os.listdir(segments_dir())
    except OSError:
        return []
    return sorted({int(n[:-6]) for n in names if n.endswith(".jsonl") and n[:-6].isdigit()})


@dataclass
class Receipt:
    """One request's record. Every field is a measurement or an identifier — never content."""

    ts: float
    request_id: str
    session: str
    model: str
    mode: str  # verbatim | lossless | lossless-only | digest | drift-trip (an event row: no request, zero counts)
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
            _maybe_rotate(path)
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
    # An empty active file means a seal just happened (or nothing was ever written):
    # the head is the last receipt of the newest segment — also a tail read. This is
    # what links the first receipt of a new segment to the one sealed before it.
    ids = sealed_segments()
    if ids:
        for raw in _reverse_lines(segment_path(ids[-1])):
            r = _parse(raw)
            if r is not None:
                return r.hash
    return GENESIS


def _classify(raw: bytes) -> tuple[Receipt | None, str]:
    """One line into ``(receipt, kind)``. Never raises. ``kind`` is:

    * ``"blank"`` — nothing there.
    * ``"receipt"`` — a valid receipt.
    * ``"foreign"`` — not a JSON object at all: a torn write (a crash or full disk cuts a
      line short, and a truncated object never parses), or text that is not ours.
    * ``"invalid"`` — a JSON object that is not a valid receipt. No torn write produces
      this, and distil's writer always writes every field with its type, so it means the
      line was edited. Verification treats it as a break, not as noise to skip.
    """
    line = raw.strip()
    if not line:
        return None, "blank"
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None, "foreign"
    if not isinstance(d, dict):
        return None, "foreign"
    try:
        return _receipt_from_dict(d), "receipt"
    except ValueError:
        return None, "invalid"


def _parse(raw: bytes) -> Receipt | None:
    """One line into a receipt, or ``None`` if it is not one. Never raises."""
    return _classify(raw)[0]


_STR_FIELDS = ("request_id", "session", "model", "mode", "certificate", "prev", "hash")
_INT_FIELDS = ("tokens_original", "tokens_compressed", "v")
_BOOL_FIELDS = ("reversible", "restorable")


def _receipt_from_dict(d: Any) -> Receipt:
    """Build a receipt from untrusted JSON, checking every field's type.

    Constructing the dataclass checks nothing, and ``compute_hash`` then sorts ``handles``
    — so ``"handles": 5`` or ``[1, "a"]`` used to surface as a TypeError traceback from
    whatever hashed it next (a proof check, a verify pass). Raises ``ValueError`` instead,
    which every caller already treats as "not a receipt".
    """
    if not isinstance(d, dict):
        raise ValueError("receipt is not an object")
    known = set(Receipt.FIELDS)
    try:
        r = Receipt(**{k: v for k, v in d.items() if k in known})
    except TypeError as exc:
        raise ValueError(f"receipt fields: {exc}") from exc
    if isinstance(r.ts, bool) or not isinstance(r.ts, (int, float)):
        raise ValueError("receipt ts must be a number")
    for name in _STR_FIELDS:
        if not isinstance(getattr(r, name), str):
            raise ValueError(f"receipt {name} must be a string")
    for name in _INT_FIELDS:
        x = getattr(r, name)
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f"receipt {name} must be an integer")
    for name in _BOOL_FIELDS:
        if not isinstance(getattr(r, name), bool):
            raise ValueError(f"receipt {name} must be a boolean")
    if not isinstance(r.handles, list) or not all(isinstance(h, str) for h in r.handles):
        raise ValueError("receipt handles must be a list of strings")
    return r


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


def _chain_files() -> list[tuple[int | None, Path]]:
    """The whole history in order: every sealed segment, then the active file (``None``)."""
    return [(i, segment_path(i)) for i in sealed_segments()] + [(None, receipts_path())]


def read(path: Path | None = None) -> Iterator[Receipt]:
    """Yield receipts in chain order. Malformed lines are skipped, not fatal.

    With no ``path``, that is the whole history: every sealed segment, then the active file.
    """
    paths = [path] if path is not None else [p for _s, p in _chain_files()]
    for p in paths:
        for _begin, _end, raw in _lines(p):
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
    #: Non-blank lines that are not receipts at all (a torn write, foreign text). Skipped,
    #: never chained — and always named in :attr:`statement`, so a chain with holes in its
    #: file never reads as a clean VERIFIED.
    skipped: int = 0

    @property
    def statement(self) -> str:
        if self.total == 0 and not self.skipped:
            return "No receipts recorded."
        if self.ok and self.skipped:
            lines = "line is" if self.skipped == 1 else "lines are"
            return (
                f"VERIFIED WITH GAPS — {self.total} receipts, hash chain intact, but "
                f"{self.skipped} {lines} not a receipt and were skipped (a torn write, or "
                "text that is not ours). A torn trailing write is expected after a crash; "
                "anything else is worth a look."
            )
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


def _load_checkpoint() -> tuple[int, str, int, int, int, int] | None:
    """``(count, head_hash, file_index, tail_start, tail_end, skipped)`` from the last good
    pass. ``skipped`` counts the non-receipt lines before ``tail_end``.

    ``file_index`` is the position in :func:`_chain_files` of the file holding that last
    receipt. A seal renames the active file into the next segment without touching a byte,
    so the index and both offsets stay valid across rotation. A checkpoint written before
    segments existed has no index, and ``0`` is exactly right for it: its file is either
    still the active one, or segment 0 after the lazy migration.
    """
    try:
        raw = json.loads(_checkpoint_path().read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return (
            int(raw["count"]),
            str(raw["head"]),
            int(raw.get("file", 0)),
            int(raw["tail_start"]),
            int(raw["tail_end"]),
            int(raw.get("skipped", 0)),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _save_checkpoint(
    count: int, head: str, file_index: int, tail_start: int, tail_end: int, skipped: int = 0
) -> None:
    """Persist the resume point. Best-effort: losing it costs a full re-scan, nothing more."""
    from . import _filelock

    path = _checkpoint_path()
    payload = {
        "count": count,
        "head": head,
        "file": file_index,
        "tail_start": tail_start,
        "tail_end": tail_end,
        "skipped": skipped,
    }
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


# ---------------------------------------------------------------------------
# Merkle tree over a segment's receipt hashes (RFC 6962 / RFC 9162 shape)
# ---------------------------------------------------------------------------
#
# Domain-separated: a leaf is H(0x00 || receipt-hash) and an interior node is
# H(0x01 || left || right), so no interior node can be passed off as a leaf. The leaf
# input is the receipt's own ``hash`` field as ASCII hex — which the verifier recomputes
# from the receipt's content, so an inclusion proof proves the content, not a label.


def _leaf(receipt_hash: str) -> bytes:
    return hashlib.sha256(b"\x00" + str(receipt_hash).encode()).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly below ``n`` (``n >= 2``)."""
    return 1 << ((n - 1).bit_length() - 1)


def _mth(leaves: list[bytes], lo: int, hi: int) -> bytes:
    if hi - lo == 1:
        return leaves[lo]
    k = _split(hi - lo)
    return _node(_mth(leaves, lo, lo + k), _mth(leaves, lo + k, hi))


def merkle_root(leaves: list[bytes]) -> bytes:
    """RFC 6962 Merkle Tree Hash over already-hashed leaves."""
    return _mth(leaves, 0, len(leaves)) if leaves else hashlib.sha256(b"").digest()


def _audit_path(leaves: list[bytes], m: int, lo: int, hi: int) -> list[bytes]:
    if hi - lo == 1:
        return []
    k = _split(hi - lo)
    if m < k:
        return _audit_path(leaves, m, lo, lo + k) + [_mth(leaves, lo + k, hi)]
    return _audit_path(leaves, m - k, lo + k, hi) + [_mth(leaves, lo, lo + k)]


def _path_len(index: int, size: int) -> int:
    """Length of the RFC 6962 audit path for leaf ``index`` in a tree of ``size`` leaves."""
    n, m, length = size, index, 0
    while n > 1:
        k = _split(n)
        if m < k:
            n = k
        else:
            m, n = m - k, n - k
        length += 1
    return length


def verify_inclusion(index: int, size: int, leaf: bytes, path: list[bytes], root: bytes) -> bool:
    """RFC 9162 §2.1.3.2 inclusion check. Needs the leaf, the path and the root — nothing else.

    What this proves depends on where ``size`` came from. A leaf that hashes up to ``root``
    is in that tree whatever ``index``/``size`` claim (the 0x00/0x01 prefixes stop an
    interior node passing as a leaf). The POSITION is only as good as ``size``: RFC 9162
    paths for different ``(index, size)`` pairs can coincide — index 1 of 2 also verifies
    as index 2 of 3 — so a position is only a claim when ``size`` is itself authenticated
    (see :func:`verify_proof`). The path-length check below rejects every pair whose tree
    shape cannot produce a path of this length; it cannot reject the ones that can.
    """
    if index < 0 or index >= size or len(path) != _path_len(index, size):
        return False
    fn, sn, r = index, size - 1, leaf
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = _node(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = _node(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


# ---------------------------------------------------------------------------
# Sealed segments
# ---------------------------------------------------------------------------


CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class Checkpoint:
    """What a seal commits to. Small enough to publish anywhere; enough to verify one
    segment, or one receipt inside it, without any other part of the history."""

    segment: int
    rows: int
    first: str  # hash of the segment's first receipt
    last: str  # hash of its last receipt — the link the next segment's first receipt carries
    root: str  # hex Merkle root over every receipt hash in the segment, in order
    v: int = 1

    def canonical(self) -> str:
        """The exact line ``distil receipts --checkpoints`` prints and the file holds."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """sha256 of :meth:`canonical` — the value to pin. Pinning this pins ``rows``,
        ``segment``, ``first`` and ``last`` along with the root; pinning the root alone
        pins only the set of leaves."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: Any) -> Checkpoint:
        """Validate untrusted input (a checkpoint file, or one embedded in a proof)."""
        if not isinstance(d, dict):
            raise ValueError("checkpoint is not an object")
        try:
            ck = cls(
                segment=d["segment"],
                rows=d["rows"],
                first=d["first"],
                last=d["last"],
                root=d["root"],
                v=d.get("v", 1),
            )
        except KeyError as exc:
            raise ValueError(f"checkpoint is missing {exc}") from exc
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (ck.segment, ck.rows)):
            raise ValueError("checkpoint segment/rows must be integers")
        if not all(isinstance(x, str) for x in (ck.first, ck.last, ck.root)):
            raise ValueError("checkpoint first/last/root must be strings")
        if isinstance(ck.v, bool) or not isinstance(ck.v, int) or ck.v != CHECKPOINT_SCHEMA:
            raise ValueError(f"checkpoint v must be the integer {CHECKPOINT_SCHEMA}")
        return ck


def load_segment_checkpoint(seg: int) -> Checkpoint | None:
    try:
        return Checkpoint.from_dict(
            json.loads(segment_checkpoint_path(seg).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError):
        return None


def _segment_mismatch(
    ck: Checkpoint | None, seg: int, leaves: list[bytes], first: str, last: str
) -> str:
    """Why a segment's contents disagree with its checkpoint, or ``""``."""
    if ck is None:
        return f"sealed segment {seg} has no readable checkpoint"
    if isinstance(ck.v, bool) or ck.v != CHECKPOINT_SCHEMA:
        return f"segment {seg}'s checkpoint has schema v={ck.v!r}, not {CHECKPOINT_SCHEMA}"
    if ck.segment != seg:
        return f"segment {seg}'s checkpoint names segment {ck.segment}"
    if ck.rows != len(leaves):
        return f"segment {seg} holds {len(leaves)} receipts, its checkpoint sealed {ck.rows}"
    if (ck.first, ck.last) != (first, last):
        return f"segment {seg}'s first/last receipt does not match its checkpoint"
    if merkle_root(leaves).hex() != ck.root:
        return f"segment {seg}'s Merkle root does not match its checkpoint"
    return ""


def _write_atomic(path: Path, text: str) -> None:
    from . import _filelock

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", opener=atrest.owner_only) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    _filelock.replace_retrying(tmp, path)


def _seal(active: Path) -> Checkpoint | None:
    """Move the active chain into the next sealed segment. Caller holds the append lock.

    **Order is the crash story.** The checkpoint is written (atomically) FIRST, then the
    active file is renamed into place. A crash between the two leaves a checkpoint with no
    segment — ignored by every reader, overwritten by the next seal — and the active file
    exactly as it was. A crash after the rename leaves a sealed segment with its checkpoint
    and no active file, which ``head_hash`` reads as "link to the newest segment". There is
    no state in which a receipt's bytes were rewritten: the segment IS the old active file,
    renamed, which is also what lets a legacy single-file chain migrate by becoming
    segment 0 untouched.
    """
    from . import _filelock

    # The cheap steps that can fail go first, so a seal that cannot succeed fails before
    # it has parsed the whole active file. The chmod is on EVERY seal: a directory that
    # already existed (made by hand, or by an older build) keeps whatever mode it had.
    segments_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    segments_dir().chmod(0o700)
    leaves: list[bytes] = []
    first = last = ""
    for r in read(active):
        leaves.append(_leaf(r.hash))
        first = first or str(r.hash)
        last = str(r.hash)
    if not leaves:
        return None
    ids = sealed_segments()
    seg = ids[-1] + 1 if ids else 0
    ck = Checkpoint(seg, len(leaves), first, last, merkle_root(leaves).hex())
    _write_atomic(segment_checkpoint_path(seg), ck.canonical() + "\n")
    _filelock.replace_retrying(active, segment_path(seg))
    segment_path(seg).chmod(0o600)  # a legacy chain may predate the owner-only opener
    return ck


#: After a failed seal, wait this long before this process tries again. Without it a seal
#: that keeps failing (an unwritable segments dir; on Windows, a reader holding the active
#: file open so the rename is refused) re-parses the whole active file on every append,
#: inside the lock every request waits on.
SEAL_RETRY_SECONDS = 60.0
_seal_retry_at = 0.0  # time.monotonic() before which no seal is attempted


def _maybe_rotate(active: Path) -> None:
    """Seal the active file if it has outgrown ``SEGMENT_BYTES``. Never loses a receipt:
    a failed seal leaves the active file where it was and the append goes ahead.

    ponytail: per-process backoff, fixed interval. Each proxy process retries at most once
    a minute; the active file just grows past the threshold meanwhile, which costs nothing
    but a larger segment. Make it exponential if a seal ever fails for days at a time.
    Also module-level, not per ``DISTIL_HOME``: a failure under one home delays seals
    under another in the same process by up to a minute, which only tests do; key it by
    home if a process ever serves several.
    """
    global _seal_retry_at
    if SEGMENT_BYTES <= 0 or time.monotonic() < _seal_retry_at:
        return
    try:
        size = active.stat().st_size
    except OSError:
        return  # no active file (fresh, or just sealed): nothing to seal, nothing failed
    if size < SEGMENT_BYTES:
        return
    try:
        sealed = _seal(active)
        # A full-size active file with no receipt in it (all foreign/garbage) can't be
        # sealed; back off like a failure instead of re-reading it on every append.
        _seal_retry_at = 0.0 if sealed is not None else time.monotonic() + SEAL_RETRY_SECONDS
    except OSError:
        # Rotation is housekeeping; the receipt being written is the record. A seal that
        # failed half-way is retried later (see _seal's crash ordering).
        _seal_retry_at = time.monotonic() + SEAL_RETRY_SECONDS
        log.debug("receipt segment seal failed; appending to the active chain", exc_info=True)


def _scan(
    files: list[tuple[int | None, Path]],
    file_index: int,
    count: int,
    prev: str,
    tail_start: int,
    tail_end: int,
    *,
    save: bool,
    skipped: int = 0,
) -> Verdict:
    """Stream the chain from byte ``tail_end`` of ``files[file_index]`` onward, chaining
    from ``prev``, with ``count`` receipts behind us.

    ``tail_start``/``tail_end`` bracket the last receipt already verified, and are carried
    through untouched when the file has grown by nothing — otherwise a render that finds no
    new receipts would move the resume point onto empty space and force the next one to
    start over.

    Every sealed segment read from its first byte is also checked against its checkpoint
    (row count, first/last hash, Merkle root). A segment the resume point sits inside is
    only chained, not re-rooted: its prefix is the part a resumed pass trusts.

    Never materialises the chain. On a break it stops hashing but keeps counting, because
    "receipt 3 of 40" has to name the receipts that exist, not the ones checked before the
    break — the second number is what tells a reader how much of the artifact is in question.
    """
    first_checked = count
    bad: tuple[int, str] | None = None
    last_file = file_index
    # Non-receipt lines up to the resume point. Lines after it are re-read (and so
    # re-counted) by every pass until a receipt moves the resume point past them.
    skipped_at_tail = skipped
    for fi in range(file_index, len(files)):
        seg, p = files[fi]
        start = tail_end if fi == file_index else 0
        leaves: list[bytes] | None = [] if seg is not None and start == 0 else None
        seg_first_idx, seg_first, seg_last = count, "", ""
        for begin, end, raw in _lines(p, start):
            r, kind = _classify(raw)
            if kind == "foreign":
                skipped += 1
            if kind == "invalid":
                # Receipt-shaped but not a receipt: an edit. It takes a receipt's index
                # so "fails at receipt i of N" points at it.
                idx, count = count, count + 1
                if bad is None:
                    bad = (idx, "a receipt-shaped line has fields that are not a valid receipt")
                continue
            if r is None:
                continue
            idx, count = count, count + 1
            if bad is not None:
                continue
            if leaves is not None:
                leaves.append(_leaf(r.hash))
                seg_first = seg_first or str(r.hash)
                seg_last = str(r.hash)
            if r.hash != r.compute_hash():
                bad = (idx, "content does not match its hash")
            elif r.prev != prev:
                bad = (idx, "prev-hash does not match the preceding receipt")
            else:
                prev, tail_start, tail_end, last_file = r.hash, begin, end, fi
                skipped_at_tail = skipped
        if bad is None and leaves is not None and seg is not None:
            why = _segment_mismatch(load_segment_checkpoint(seg), seg, leaves, seg_first, seg_last)
            if why:
                bad = (seg_first_idx, why)
    if bad is not None:
        return Verdict(count, False, bad[0], bad[1], first_checked, skipped)
    if count and save:
        _save_checkpoint(count, prev, last_file, tail_start, tail_end, skipped_at_tail)
    return Verdict(count, True, checked_from=first_checked, skipped=skipped)


def verify(path: Path | None = None, *, full: bool = True) -> Verdict:
    """Recompute every hash and every link. This is the whole point of the artifact:
    anyone can run it, and it needs nothing but the files.

    With no ``path`` this is the WHOLE history: every sealed segment in order (each also
    checked against its checkpoint's Merkle root), then the active file, with the chain
    carried across each boundary — so a deleted, reordered or re-sealed segment breaks it.
    An explicit ``path`` is one self-contained chain from genesis (a legacy file, or a
    whole history someone concatenated for you). One sealed segment on its own is
    :func:`verify_segment`; one receipt on its own is :func:`verify_proof`.

    **A third party always does the full pass.** An explicit path never consults a
    checkpoint, so the guarantee the artifact exists to provide is unchanged. What is
    cached is *this* machine re-checking *its own* chain, which now happens on every wrap
    exit and every ``distil stats``: the resume point records how far a previous pass got,
    and only receipts appended since are re-hashed. Without that, a session's exit summary
    re-hashes an 83 MB chain.

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

    **A seal can land mid-pass.** Verify runs without the append lock (it must not stall
    requests for a whole-history re-hash), so the file list it took can go stale: the
    active file it opens may already be the fresh one, whose first receipt links to a
    segment the list does not have. That reads as a broken link on a healthy chain, so a
    failure is re-run once against a fresh listing when the listing changed. A real break
    survives the retry; a seal does not.
    """
    if path is not None:
        return _scan([(None, path)], 0, 0, GENESIS, 0, 0, save=False)
    files = _chain_files()
    v = _verify_history(files, full)
    if not v.ok:
        now = _chain_files()
        if now != files:
            v = _verify_history(now, full)
    return v


def _verify_history(files: list[tuple[int | None, Path]], full: bool) -> Verdict:
    if not full:
        ck = _load_checkpoint()
        if ck is not None:
            count, head, fi, tail_start, tail_end, skipped = ck
            if 0 <= fi < len(files):
                p = files[fi][1]
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
                    v = _scan(
                        files, fi, count, head, tail_start, tail_end, save=True, skipped=skipped
                    )
                    if v.ok:
                        return v
    return _scan(files, 0, 0, GENESIS, 0, 0, save=True)


def verify_segment(seg_path: Path, checkpoint: Checkpoint | None) -> Verdict:
    """Verify ONE sealed segment against its checkpoint, reading nothing else.

    Every receipt is re-hashed and every link inside the segment re-checked; the segment's
    link to its predecessor is pinned by ``checkpoint.first`` (whose hash covers that
    receipt's ``prev``), and the whole contents by the Merkle root. Continuity with the
    neighbouring segments is the full-history pass's job.
    """
    count = 0
    prev: str | None = None
    bad: tuple[int, str] | None = None
    leaves: list[bytes] = []
    first = last = ""
    skipped = 0
    for _b, _e, raw in _lines(seg_path):
        r, kind = _classify(raw)
        skipped += kind == "foreign"
        if kind == "invalid":
            idx, count = count, count + 1
            if bad is None:
                bad = (idx, "a receipt-shaped line has fields that are not a valid receipt")
            continue
        if r is None:
            continue
        idx, count = count, count + 1
        if bad is not None:
            continue
        leaves.append(_leaf(r.hash))
        first = first or str(r.hash)
        last = str(r.hash)
        if r.hash != r.compute_hash():
            bad = (idx, "content does not match its hash")
        elif prev is not None and r.prev != prev:
            bad = (idx, "prev-hash does not match the preceding receipt")
        prev = r.hash
    if bad is None:
        if checkpoint is not None:
            seg = checkpoint.segment
        else:  # name the segment the caller asked about, not a sentinel
            seg = int(seg_path.stem) if seg_path.stem.isdigit() else -1
        why = _segment_mismatch(checkpoint, seg, leaves, first, last)
        if why:
            bad = (0, why)
    if bad is not None:
        return Verdict(count, False, bad[0], bad[1], skipped=skipped)
    return Verdict(count, True, skipped=skipped)


def prove(request_id: str) -> dict[str, Any] | None:
    """An inclusion proof for the receipt with ``request_id``, or ``None``.

    Only a SEALED receipt has one — the active file has no root yet. The proof bundles the
    receipt, its segment's checkpoint, its index and the audit path: everything
    :func:`verify_proof` needs, and nothing from any other receipt but sibling hashes.
    Newest segment first, since that is where a receipt someone is asking about usually is.
    """
    for seg in reversed(sealed_segments()):
        ck = load_segment_checkpoint(seg)
        if ck is None:
            continue
        receipts = list(read(segment_path(seg)))
        for i, r in enumerate(receipts):
            if r.request_id == request_id:
                leaves = [_leaf(x.hash) for x in receipts]
                return {
                    "v": 1,
                    "receipt": asdict(r),
                    "checkpoint": asdict(ck),
                    "index": i,
                    "path": [h.hex() for h in _audit_path(leaves, i, 0, len(leaves))],
                }
    return None


def verify_proof(
    bundle: Any, root: str | None = None, checkpoint_hash: str | None = None
) -> tuple[bool, str]:
    """Check an inclusion proof. Reads no file: the bundle and what you pinned are the
    whole input. What a success means depends entirely on what was pinned:

    * ``checkpoint_hash`` — the sha256 of a checkpoint record you already trust (a line of
      ``distil receipts --checkpoints``, see :meth:`Checkpoint.digest`). That pins
      ``segment``, ``rows``, ``first``, ``last`` and ``root`` together, so the receipt's
      segment and position are proven, and the message names them.
    * ``root`` alone — proves the receipt is a leaf of the tree with that root, and nothing
      about WHERE: ``rows`` and ``index`` still come from the bundle, unauthenticated, and
      paths for different ``(index, rows)`` pairs can coincide. The message says only
      "included under root R".
    * neither — the bundle is checked against its own checkpoint, which is circular: it
      shows the bundle is self-consistent and proves nothing about your log. The message
      says that too.
    """
    try:
        if not isinstance(bundle, dict):
            raise ValueError("proof is not an object")
        v = bundle.get("v", 1)
        if type(v) is not int or v != 1:  # a future/foreign proof shape is not a v1 proof
            raise ValueError(f"unsupported proof version {v!r}")
        ck = Checkpoint.from_dict(bundle["checkpoint"])
        r = _receipt_from_dict(bundle["receipt"])
        index = bundle["index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("index must be an integer")
        raw_path = bundle["path"]
        if not isinstance(raw_path, list) or not all(isinstance(h, str) for h in raw_path):
            raise ValueError("path must be a list of hex strings")
        path = [bytes.fromhex(h) for h in raw_path]
        want = bytes.fromhex(root if root is not None else ck.root)
        if not all(len(h) == 32 for h in (*path, want)):
            raise ValueError("path entries and roots must be 32-byte hashes")
        if checkpoint_hash is not None:
            bytes.fromhex(checkpoint_hash)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"malformed proof: {exc}"
    if checkpoint_hash is not None and checkpoint_hash.lower() != ck.digest():
        return False, "the proof's checkpoint is not the checkpoint you pinned"
    if root is not None and root.lower() != ck.root.lower():
        return False, "the proof's checkpoint root is not the root you pinned"
    if r.hash != r.compute_hash():
        return False, "receipt content does not match its hash"
    if not verify_inclusion(index, ck.rows, _leaf(r.hash), path, want):
        return False, f"receipt does not hash up to root {want.hex()[:16]}… by this path"
    short = want.hex()[:16]
    if checkpoint_hash is not None:
        return True, (
            f"INCLUDED — receipt {r.request_id} is #{index} of {ck.rows} in sealed segment "
            f"{ck.segment} (pinned checkpoint {checkpoint_hash.lower()[:16]}…, root {short}…)"
        )
    if root is not None:
        return True, (
            f"INCLUDED — receipt {r.request_id} is in the tree under pinned root {short}… "
            "(its position and segment are not proven; pin the checkpoint hash for those)"
        )
    return True, (
        f"SELF-CONSISTENT ONLY — receipt {r.request_id} hashes up to the root carried in its "
        "own bundle; nothing was pinned, so this says nothing about your log"
    )

"""Preserved-thinking guard: never forward a thinking block whose prefix distil changed.

ADR 0028, second addendum. On the preserved-thinking models (``anthropic.preserved_thinking``)
a thinking block is valid only while ``system``, ``tools`` and every message content block
before it carry the same values they had when the block was produced (JSON formatting and
key order aside; thinking blocks themselves are not in that prefix), and the thinking
blocks kept form an unbroken run of the original sequence. Anything else is a 400 on
current accounts.

distil cannot always forward the same prefix: a restart loses prefix replay, a history
over its byte cap is not replayed, a fail-open turn goes out uncompressed, and the expand
loop produces thinking over a hidden round the client never sees. So this module, run
LAST on every Anthropic Messages body for these models, checks rather than hopes:

* it records, per request, the forwarded and the client-side prefix hash at the end of the
  history (``R``). When a new signature shows up next turn at message *k*, the request whose
  client prefix equals messages[:k] produced it, so its production prefix is that request's
  forwarded hash, extended by the block's own message as the client returned it;
* every thinking block is checked against its production hash. From the first one that
  fails (distil changed something before it), every thinking block is removed — "remove the
  newest", which the API allows — and the removal is persisted so a removed block is never
  re-added. A block whose CLIENT-side prefix changed is the client's own edit and is passed
  through untouched, exactly as without distil;
* a signature distil has no record of is kept only when distil changed nothing before it.

State is content-free: truncated sha256 of signatures and of canonical prefixes, appended
to ``$DISTIL_HOME/thinkguard.jsonl`` so a restart or hot-swap keeps verifying.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

log = logging.getLogger("distil.thinkguard")

_THINK = ("thinking", "redacted_thinking")
# Stand-in for an assistant message the drop would leave empty (the API rejects one).
# Deterministic, so every later turn renders it identically.
PLACEHOLDER = "[thinking omitted]"
# ponytail: in-memory LRU caps and a compaction threshold; past them the oldest records
# fall out and their blocks are judged "unknown" (kept only if distil changed nothing
# before them). A session needs ~1 record per assistant message; a request record is
# read only by the next turn, so the newest few thousand are plenty.
_MAX_BLOCKS = 50_000
_MAX_REQS = 5_000
_MAX_DROPPED = 50_000
_SLACK = 10_000  # lines of superseded records tolerated before the file is rewritten

_LOCK = threading.Lock()
_BLOCKS: "OrderedDict[str, tuple[str, str, str]]" = OrderedDict()  # bid -> (F, O, pred)
_REQS: "OrderedDict[str, tuple[str, str]]" = OrderedDict()  # scope+O_end -> (F_end, pred)
_DROPPED: "OrderedDict[str, None]" = OrderedDict()
_FILE: dict[str, Any] = {"path": None, "off": 0, "lines": 0}


def reset() -> None:
    with _LOCK:
        _BLOCKS.clear()
        _REQS.clear()
        _DROPPED.clear()
        _FILE.update(path=None, off=0, lines=0)


def _path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "thinkguard.jsonl"


def _put(d: "OrderedDict[str, Any]", k: str, v: Any, cap: int) -> None:
    d[k] = v
    d.move_to_end(k)
    while len(d) > cap:
        d.popitem(last=False)


def _apply_line(rec: dict[str, Any]) -> None:
    if "e" in rec:
        _put(_BLOCKS, rec["e"], (rec["f"], rec["o"], rec.get("p") or ""), _MAX_BLOCKS)
    elif "r" in rec:
        _put(_REQS, rec["r"], (rec["f"], rec.get("p") or ""), _MAX_REQS)
    elif "d" in rec:
        _put(_DROPPED, rec["d"], None, _MAX_DROPPED)


def _sync() -> None:
    """Fold in lines another process (a hot-swap peer) appended. Caller holds _LOCK."""
    path = _path()
    if _FILE["path"] != str(path):
        _BLOCKS.clear()
        _REQS.clear()
        _DROPPED.clear()
        _FILE.update(path=str(path), off=0, lines=0)
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= _FILE["off"]:
        return
    try:
        with path.open("rb") as f:
            f.seek(_FILE["off"])
            chunk = f.read(size - _FILE["off"])
    except OSError:
        return
    end = chunk.rfind(b"\n") + 1  # a half-written last line waits for the next sync
    for line in chunk[:end].splitlines():
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                _apply_line(rec)
        except (ValueError, KeyError, TypeError):
            continue
        _FILE["lines"] += 1
    _FILE["off"] += end


def _append(recs: list[dict[str, Any]]) -> None:
    """Persist *recs* (already applied in memory). Caller holds _LOCK. Fail-open."""
    if not recs:
        return
    for r in recs:
        _apply_line(r)
    try:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs).encode()
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        _FILE["off"] += len(data)
        _FILE["lines"] += len(recs)
        if _FILE["lines"] > 2 * (len(_BLOCKS) + len(_REQS) + len(_DROPPED)) + _SLACK:
            _compact(path)
    except Exception:  # noqa: BLE001 — persistence must never break a request
        log.debug("thinkguard state not persisted", exc_info=True)


def _compact(path: Path) -> None:
    from .config_wrap import _atomic_write_secure

    lines = [{"d": k} for k in _DROPPED]
    lines += [{"r": k, "f": f, "p": p} for k, (f, p) in _REQS.items()]
    lines += [{"e": k, "f": f, "o": o, "p": p} for k, (f, o, p) in _BLOCKS.items()]
    data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in lines).encode()
    _atomic_write_secure(path, data)
    _FILE.update(off=len(data), lines=len(lines))


# --------------------------------------------------------------------------- hashing


def _bid(block: dict[str, Any]) -> Optional[str]:
    from .adapters.anthropic import _thinking_key

    return _thinking_key(block)


def _norm(block: Any) -> Any:
    """One text block in one spelling (a bare string and ``[{"type":"text"}]`` render the
    same), anything else as is. Comparison only."""
    if isinstance(block, str):
        return {"text": block}
    if (
        isinstance(block, dict)
        and block.get("type") == "text"
        and set(block) <= {"type", "text", "cache_control", "index"}
    ):
        return {"text": block.get("text")}
    return block


def _h(prev: str, item: Any) -> str:
    from .prefixreplay import canonical

    return hashlib.sha256((prev + "\0" + canonical(item)).encode("utf-8", "replace")).hexdigest()[
        :32
    ]


def _blocks(msg: Any) -> list[Any]:
    c = msg.get("content") if isinstance(msg, dict) else None
    return c if isinstance(c, list) else [c]


def _is_think(b: Any) -> bool:
    return isinstance(b, dict) and b.get("type") in _THINK


def _fold(h: str, msg: Any) -> Iterator[tuple[Optional[str], str]]:
    """Walk one message: yields ``(bid, prefix hash)`` for each thinking block, then
    ``(None, hash at the end of the message)``."""
    h = _h(h, {"role": msg.get("role") if isinstance(msg, dict) else None})
    for b in _blocks(msg):
        if _is_think(b):
            yield _bid(b) or "", h
        else:
            h = _h(h, _norm(b))
    yield None, h


def _walk(body: dict[str, Any]) -> tuple[list[str], dict[str, tuple[int, str]]]:
    """``(bases, blocks)``: bases[i] is the prefix hash before message i (len n+1);
    blocks maps each thinking block's id to ``(message index, prefix hash)``."""
    h = _h("", {"system": {"content": body.get("system")}, "tools": body.get("tools")})
    bases = [h]
    blocks: dict[str, tuple[int, str]] = {}
    for i, m in enumerate(body.get("messages") or []):
        for bid, hh in _fold(h, m):
            if bid is None:
                h = hh
            elif bid:
                blocks.setdefault(bid, (i, hh))
        bases.append(h)
    return bases, blocks


def has_thinking(messages: Iterable[Any]) -> bool:
    return any(_is_think(b) for m in messages for b in _blocks(m))


def _in_scope(body: dict[str, Any]) -> bool:
    from .adapters.anthropic import preserved_thinking

    return preserved_thinking(str(body.get("model") or "")) and isinstance(
        body.get("messages"), list
    )


# --------------------------------------------------------------------------- public


def drop(blocks: Iterable[Any]) -> None:
    """Never forward these thinking blocks again (the expand loop's continuation
    thinking: produced over a hidden round, valid on no history the client can send)."""
    recs = [{"d": b} for b in (_bid(x) for x in blocks if _is_think(x)) if b]
    if recs:
        with _LOCK:
            _sync()
            _append(recs)


def guard(
    client: dict[str, Any],
    forwarded: dict[str, Any],
    *,
    scope: str = "",
    extras: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Return *forwarded* with every thinking block that would fail its prefix check
    removed (and all later ones), and record this request for the next turn's blocks.

    *client* is the body as the client sent it. Untouched (same object) unless the model
    is a preserved-thinking one and a block has to go. Fail-open: an error returns
    *forwarded* as is.
    """
    try:
        if not _in_scope(forwarded):
            return forwarded
        return _guard(client, forwarded, scope, extras)
    except Exception:  # noqa: BLE001 — never break a request
        log.debug("thinkguard failed; forwarding as is", exc_info=True)
        return forwarded


def _guard(
    client: dict[str, Any], fwd: dict[str, Any], scope: str, extras: Optional[dict[str, str]]
) -> dict[str, Any]:
    raw = client.get("messages")
    cmsgs: list[Any] = raw if isinstance(raw, list) else []
    obases, oblocks = _walk(client)
    fbases, fblocks = _walk(fwd)
    recs: list[dict[str, Any]] = []
    remove: set[str] = set()
    with _LOCK:
        _sync()
        prev = ""  # last kept thinking block, for the chain rule
        bad = False
        for bid, (k, f) in fblocks.items():  # insertion order is document order
            if bid in _DROPPED:
                remove.add(bid)
                continue
            if bad:
                remove.add(bid)
                continue
            e = _BLOCKS.get(bid)
            o = oblocks.get(bid, (k, ""))[1]
            if e is None and 0 <= k < len(cmsgs):
                e = _infer(scope, obases[k], cmsgs[k], bid, oblocks)
                if e is not None:
                    recs.append({"e": bid, "f": e[0], "o": e[1], "p": e[2]})
            if e is None:
                ok = o == f  # no record: keep only if distil changed nothing before it
            elif f == e[0]:
                ok = not prev or prev == e[2]
            else:
                ok = o != e[1]  # the client edited its own history: its call, not ours
            if not ok:
                bad = True
                remove.add(bid)
                continue
            prev = bid
        out = fwd if not remove else _strip(fwd, remove)
        recs += [{"d": b} for b in sorted(remove) if b not in _DROPPED]
        n = len(cmsgs)
        if n and len(fbases) == n + 1:
            # This request produces the message at index n: remember its prefix.
            recs.append({"r": scope + obases[n], "f": fbases[n], "p": prev})
        _append(recs)
    if remove and extras is not None:
        extras["x-distil-thinking-dropped"] = str(len(remove))
        log.info("dropped %d thinking block(s) whose prefix changed", len(remove))
    return out


def _infer(
    scope: str, obase: str, cmsg: Any, bid: str, oblocks: dict[str, tuple[int, str]]
) -> Optional[tuple[str, str, str]]:
    """Production record of a block first seen now: the request whose client prefix was
    messages[:k] produced message k. Caller holds _LOCK."""
    r = _REQS.get(scope + obase)
    if r is None:
        return None
    f, pred = r
    for b2, hh in _fold(f, cmsg):
        if b2 == bid:
            return hh, oblocks[bid][1] if bid in oblocks else "", pred
        if b2:
            pred = b2
    return None


def _strip(body: dict[str, Any], remove: set[str]) -> dict[str, Any]:
    msgs = []
    for m in body["messages"]:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(_is_think(b) and _bid(b) in remove for b in c):
            kept = [b for b in c if not (_is_think(b) and _bid(b) in remove)]
            m = {**m, "content": kept or [{"type": "text", "text": PLACEHOLDER}]}
        msgs.append(m)
    return {**body, "messages": msgs}


def lineage(body: dict[str, Any]) -> bool:
    """A preserved-thinking conversation that carries (or is about to produce) thinking.
    distil keeps ``system`` fixed for these: output shaping, which edits ``system`` and
    toggles with the drift-guard hold, stays off for the whole lineage."""
    if not _in_scope(body):
        return False
    t = body.get("thinking")
    on = isinstance(t, dict) and t.get("type") not in (None, "disabled")
    return on or has_thinking(body["messages"])

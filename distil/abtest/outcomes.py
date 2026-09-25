"""Per-session outcomes for the A/B, kept where the TTL sweep cannot reach them.

``sessions/<sid>.*`` is swept after 7 days by ``distil wrap``; a 5% holdout needs
months of sessions. So each randomised session is folded, content-free, into one row
of ``ab.jsonl`` — at wrap start (every session the sweep is about to delete) and at
wrap exit (this one). ``distil ab`` folds again before it reads. The store is
append-only; the last row per session wins, so a session re-folded after it ended
simply supersedes its "open" row.

Dollars are the provider's own usage fields priced at list (cache read 0.1x, cache
write 1.25x — :mod:`distil.pricing`). Those counts are the provider's, so no token
calibration applies. Unknown models are not priced (no Claude rate is ever guessed
onto a Gemini/OpenAI upstream) and such sessions are counted, not estimated.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from .. import pricing
from . import DISTIL, HOLDOUT, keyed_hash

log = logging.getLogger(__name__)

SCHEMA = 1
#: A session with no exit breadcrumb and no activity for this long was abandoned
#: (terminal closed, machine slept through it, wrap killed). Before that it is "open"
#: and excluded from estimates — symmetric in both arms.
ABANDON_S = 6 * 3600.0


def store_path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "ab.jsonl"


def _sessions_dir() -> Path:
    return store_path().parent / "sessions"


@dataclass
class SessionOutcome:
    sid: str
    arm: str
    rate: float
    start: float
    end: float
    model: str  # the model that carried most of the session's cost (or requests)
    client: str  # "claude-cli/2.1" — name + major.minor
    ws: str  # keyed hash of the workspace (cwd); the CUPED grouping key
    billing: str  # "metered" | "subscription" | "unknown"
    cost: float | None  # list-price $ incl. cache; None if nothing could be priced
    turns: int  # assistant requests answered 2xx
    tasks: int  # user-initiated segments (>= 1); 1 when the session predates the marker
    in_tokens: int  # uncached + cache read + cache write
    out_tokens: int
    errors: int  # non-2xx upstream responses
    expanded: int  # 2xx requests that ran a distil_expand re-query (see report caveats)
    wall_s: float
    ended: str  # "ok" | "error" | "abandoned" | "open"
    unpriced: int  # 2xx requests whose usage or model could not be priced
    src: float  # max mtime of the inputs this row was folded from
    v: int = SCHEMA

    @property
    def complete(self) -> bool:
        return self.ended != "open"

    @property
    def cost_per_task(self) -> float | None:
        return None if self.cost is None else self.cost / max(1, self.tasks)


_CLIENT = re.compile(r"^([A-Za-z][\w.\-]{0,40})/v?(\d{1,4})(?:\.(\d{1,4}))?")


def client_tag(user_agent: str | None) -> str:
    """``claude-cli/2.1.3 (external, cli)`` → ``claude-cli/2.1``. Name and version
    only; anything that does not look like a product token is dropped entirely."""
    m = _CLIENT.match((user_agent or "").strip())
    if not m:
        return ""
    return f"{m.group(1)}/{m.group(2)}" + (f".{m.group(3)}" if m.group(3) else "")


def is_user_turn(body: dict[str, Any] | None) -> bool | None:
    """Does this request open a new user turn (vs. continue a tool loop)?

    Content-free: looks only at roles and block TYPES of the last message. None when
    the shape is not one we know, so the caller records nothing rather than a guess.
    """
    if not isinstance(body, dict):
        return None
    for key in ("messages", "input", "contents"):
        items = body.get(key)
        if isinstance(items, list) and items:
            last = items[-1]
            break
    else:
        return None
    if not isinstance(last, dict):
        return None
    role = last.get("role")
    if key == "input" and last.get("type") not in (None, "message"):
        return False  # function_call_output etc.
    if role != "user":
        return False  # chat "tool" role, Gemini "function", or an assistant prefill
    parts = last.get("content") if key != "contents" else last.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if isinstance(p, dict) and (p.get("type") == "tool_result" or "functionResponse" in p):
                return False
    return True


def _price(row: dict[str, Any]) -> float | None:
    p = pricing.resolve(str(row.get("model") or ""))
    if p is None or row.get("usage_input_tokens") is None:
        return None
    return (
        int(row.get("usage_input_tokens") or 0) * p.input
        + int(row.get("usage_cache_read") or 0) * p.cache_read
        + int(row.get("usage_cache_create") or 0) * p.cache_write
        + int(row.get("usage_output_tokens") or 0) * p.output
    )


def _ended(exit_text: str | None, last_activity: float, now: float) -> str:
    if exit_text:
        m = re.search(r"child exit code (-?\d+)", exit_text)
        return "ok" if m and m.group(1) == "0" else "error"
    return "abandoned" if now - last_activity > ABANDON_S else "open"


def fold_session(
    manifest: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    *,
    exit_text: str | None = None,
    src: float = 0.0,
    now: float | None = None,
) -> SessionOutcome | None:
    """One session's outcome, or None if it was never randomised."""
    ab = manifest.get("ab")
    if not isinstance(ab, dict) or ab.get("arm") not in (DISTIL, HOLDOUT):
        return None
    now = time.time() if now is None else now
    rows = sorted((r for r in rows if isinstance(r, dict)), key=lambda r: r.get("ts") or 0)
    ok = [r for r in rows if isinstance(r.get("status"), int) and 200 <= r["status"] < 300]
    by_model: dict[str, list[float]] = {}  # model -> [cost, requests]
    clients: dict[str, int] = {}
    cost, unpriced, priced = 0.0, 0, 0
    for r in ok:
        c = _price(r)
        agg = by_model.setdefault(str(r.get("model") or "unknown"), [0.0, 0])
        agg[1] += 1
        if c is None:
            unpriced += 1
        else:
            priced += 1
            cost += c
            agg[0] += c
        tag = str(r.get("client") or "")
        if tag:
            clients[tag] = clients.get(tag, 0) + 1
    model = max(by_model, key=lambda m: (by_model[m][0], by_model[m][1])) if by_model else ""
    marked = [r for r in ok if "user_turn" in r]
    tasks = (
        sum(1 for r in marked if r.get("user_turn") and str(r.get("model") or "unknown") == model)
        if marked
        else 1
    )
    client = max(clients, key=lambda k: clients[k]) if clients else str(manifest.get("tool") or "")
    ts = [float(r.get("ts") or 0.0) for r in rows if r.get("ts")]
    start = float(manifest.get("started_ts") or (min(ts) if ts else 0.0))
    end = max(ts) if ts else start
    return SessionOutcome(
        sid=str(manifest.get("sid") or ""),
        arm=str(ab["arm"]),
        rate=float(ab.get("rate") or 0.0),
        start=start,
        end=end,
        model=model,
        client=client,
        ws=keyed_hash(str(manifest.get("cwd") or "")) if manifest.get("cwd") else "",
        billing=str(manifest.get("billing") or "unknown"),
        cost=cost if priced else None,
        turns=len(ok),
        tasks=max(1, tasks),
        in_tokens=sum(
            int(r.get("usage_input_tokens") or 0)
            + int(r.get("usage_cache_read") or 0)
            + int(r.get("usage_cache_create") or 0)
            for r in ok
        ),
        out_tokens=sum(int(r.get("usage_output_tokens") or 0) for r in ok),
        errors=len(rows) - len(ok),
        expanded=sum(1 for r in ok if r.get("expanded_handles")),
        wall_s=max(0.0, end - start),
        ended=_ended(exit_text, max([end, src]), now),
        unpriced=unpriced,
        src=src,
    )


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


def load(path: Path | None = None) -> dict[str, SessionOutcome]:
    """Latest row per session. Unreadable or foreign-schema lines are skipped."""
    p = path or store_path()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        log.warning("distil ab: cannot read %s: %s", p, exc)
        return {}
    names = {f.name for f in fields(SessionOutcome)}
    out: dict[str, SessionOutcome] = {}
    for line in text.splitlines():
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("v") != SCHEMA:
                continue
            o = SessionOutcome(**{k: v for k, v in raw.items() if k in names})
        except (ValueError, TypeError):
            continue
        if o.cost is not None and not math.isfinite(o.cost):
            continue
        out[o.sid] = o
    return out


def append(rows: list[SessionOutcome], path: Path | None = None) -> None:
    if not rows:
        return
    from .. import _filelock, atrest

    p = path or store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _filelock.locked(p), open(p, "a", encoding="utf-8", opener=atrest.owner_only) as fh:
        for o in rows:
            fh.write(json.dumps(asdict(o), sort_keys=True) + "\n")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def harvest(only: str | None = None, *, now: float | None = None) -> int:
    """Fold every randomised session on disk (or just *only*) into the store.

    Re-folds a session only when its inputs changed or it was still "open" last time.
    Returns the number of rows written. Never raises: bookkeeping must not stop a wrap.
    """
    try:
        known = load()
        todo: list[SessionOutcome] = []
        d = _sessions_dir()
        manifests = [d / f"{only}.json"] if only else sorted(d.glob("*.json"))
        for mp in manifests:
            try:
                man = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(man, dict) or not isinstance(man.get("ab"), dict):
                continue
            sid = str(man.get("sid") or mp.stem)
            man["sid"] = sid
            req, ex = mp.with_name(sid + ".requests.jsonl"), mp.with_name(sid + ".exit")
            src = max(_mtime(req), _mtime(ex))
            prev = known.get(sid)
            if prev is not None and prev.src == src and prev.complete:
                continue
            exit_text = None
            if ex.exists():
                try:
                    exit_text = ex.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    exit_text = None
            o = fold_session(man, _read_rows(req), exit_text=exit_text, src=src, now=now)
            if o is not None and (prev is None or asdict(prev) != asdict(o)):
                todo.append(o)
        append(todo)
        return len(todo)
    except Exception:  # noqa: BLE001 — the A/B record must never break a wrap
        log.debug("distil ab: harvest failed", exc_info=True)
        return 0

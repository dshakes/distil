"""Offline stand-ins for the provider and the agent, so the whole pipeline runs at $0.

``MockUpstream`` answers ``POST /v1/messages`` (JSON or SSE) with a ``usage`` object
computed from the request the way a prefix cache would bill it: the longest prefix it has
seen within the TTL is a cache read, the rest is a cache write. Tokens are ``chars // 4``.

``scripted_agent`` is a fake ``claude -p``: a multi-turn loop whose tool outputs an arm
shrinks (the claimed saving), sometimes at the price of extra turns or a rewritten earlier
turn (a cache bust). The effects are SYNTHETIC and deliberately do not favour distil —
they exist to exercise every branch of the meter and the analysis, not to predict results.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

CACHE_TTL_S = 300.0


def _tok(obj: Any) -> int:
    return max(1, len(json.dumps(obj, sort_keys=True)) // 4)


class _Cache:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.seen: dict[str, float] = {}
        self.lock = threading.Lock()

    def bill(self, req: dict[str, Any]) -> tuple[int, int]:
        """(cache_read, cache_write) tokens for this request; then cache every prefix."""
        units: list[Any] = [[req.get("tools"), req.get("system")], *req.get("messages", [])]
        now, h, hashes = self.clock(), "", []
        for u in units:
            h = hashlib.sha256((h + json.dumps(u, sort_keys=True)).encode()).hexdigest()
            hashes.append(h)
        sizes = [_tok(u) for u in units]
        with self.lock:
            hit = 0
            for i, hh in enumerate(hashes):
                if self.seen.get(hh, -1.0) >= now:
                    hit = i + 1
            for hh in hashes:
                self.seen[hh] = now + CACHE_TTL_S
        return sum(sizes[:hit]), sum(sizes[hit:])


class _Handler(BaseHTTPRequestHandler):
    server: MockUpstream
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
        self.server.requests += 1
        if self.server.fail_next > 0:
            self.server.fail_next -= 1
            return self._send(529, "application/json", b'{"type":"error"}')
        read, write = self.server.cache.bill(req)
        last = json.dumps(req.get("messages", [])[-1:], sort_keys=True).encode()
        out = min(
            int(req.get("max_tokens") or 1), 150 + int(hashlib.sha256(last).hexdigest(), 16) % 300
        )
        usage = {
            "input_tokens": 3,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write,
            "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": 0},
            "output_tokens": out,
        }
        model = req.get("model")
        if req.get("stream"):
            start = {**usage, "output_tokens": 1}
            events: list[dict[str, Any]] = [
                {"type": "message_start", "message": {"model": model, "usage": start}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
                {"type": "message_delta", "usage": {"output_tokens": out}},
                {"type": "message_stop"},
            ]
            body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
            return self._send(200, "text/event-stream", body.encode())
        doc = {"model": model, "content": [{"type": "text", "text": "ok"}], "usage": usage}
        self._send(200, "application/json", json.dumps(doc).encode())

    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockUpstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.cache = _Cache(clock)
        self.requests = 0
        self.fail_next = 0
        self._t = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def __enter__(self) -> MockUpstream:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
        self.server_close()


@dataclass(frozen=True)
class Effect:
    """SYNTHETIC arm behaviour for the dry run only."""

    shrink: float = 1.0  # fraction of each tool output that survives
    extra_turn_p: float = 0.0  # chance a turn needs a follow-up turn
    bust_p: float = 0.0  # chance a turn rewrites an earlier tool result (cache bust)
    success_delta: float = 0.0


#: Deliberately mixed so the dry run exercises cheaper / neutral / dearer-by-cache-bust
#: branches; distil is assigned the cache-busting (worse) profile on purpose.
SYNTHETIC_EFFECTS: dict[str, Effect] = {
    "control": Effect(),
    "rtk": Effect(shrink=0.6, extra_turn_p=0.05),
    "headroom": Effect(shrink=0.8),
    "distil": Effect(shrink=0.5, bust_p=0.15, success_delta=-0.03),
}

_TOOLS = [{"name": f"tool_{i}", "description": "d" * 400} for i in range(12)]
_SYSTEM = "You are a coding agent. " * 200


@dataclass
class AgentResult:
    solved: bool
    turns: int
    claimed_tokens_saved: int


def _post(base_url: str, body: dict[str, Any]) -> None:
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url + "/v1/messages",
        data=raw,
        headers={"content-type": "application/json", "x-api-key": "dry-run"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def scripted_agent(base_url: str, task: str, seed: int, arm: str, model: str) -> AgentResult:
    """A deterministic fake ``claude -p`` run for (task, seed, arm)."""
    eff = SYNTHETIC_EFFECTS.get(arm, Effect())
    task_rng = random.Random(f"{task}")  # task difficulty: same for every arm and seed
    base_turns, out_chars, p_solve = (
        task_rng.randint(4, 10),
        task_rng.randint(800, 6000),
        task_rng.uniform(0.5, 0.95),
    )
    rng = random.Random(f"{task}|{seed}|{arm}")
    msgs: list[dict[str, Any]] = [{"role": "user", "content": f"Task {task}: make the tests pass."}]
    turns, claimed, t = 0, 0, 0
    budget = base_turns
    while t < budget:
        stream = t % 2 == 1
        _post(
            base_url,
            {
                "model": model,
                "max_tokens": 4096,
                "stream": stream,
                "tools": _TOOLS,
                "system": _SYSTEM,
                "messages": msgs,
            },
        )
        turns += 1
        kept = int(out_chars * eff.shrink)
        claimed += (out_chars - kept) // 4
        msgs.append({"role": "assistant", "content": f"run step {t}"})
        msgs.append({"role": "user", "content": f"[{task}:{seed}:{t}] " + "x" * kept})
        if rng.random() < eff.bust_p and len(msgs) > 3:
            msgs[2] = {"role": "user", "content": msgs[2]["content"] + " (re-digested)"}
        if rng.random() < eff.extra_turn_p:
            budget += 1
        t += 1
    solved = rng.random() < p_solve + eff.success_delta
    return AgentResult(solved=solved, turns=turns, claimed_tokens_saved=claimed)

"""distil vs Headroom 0.42.0 — offline, $0 head-to-head on real tool output.

Pre-registered definitions: benchmarks/results/headroom-0-42-h2h/README.md. Three steps,
all under ONE python that has both packages installed (Headroom in an isolated venv with
distil installed editable next to it):

    PY=<venv>/bin/python
    $PY -m benchmarks.headroom_h2h collect --out <scratch>/inputs.json   # real content: keep out of git
    $PY -m benchmarks.headroom_h2h arm --arm <name> --inputs ... --out <scratch>/arms/<name>.json
    $PY -m benchmarks.headroom_h2h report --arms <scratch>/arms --out benchmarks/results/headroom-0-42-h2h/results.json

`all` runs collect + every arm (one subprocess each: Headroom's pipeline is a process-wide
singleton configured from env) + report. Arm files hold counts, hashes and timings only.
No LLM API is called; Headroom's Kompress weights come from Hugging Face.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.policysim.costmodel import Anthropic, segments
from benchmarks.policysim.policies import _cc_last, _strip_cc
from benchmarks.policysim.trajectory import _request_ends, block_text

Msg = dict[str, Any]
CTYPES = ("logs", "json", "code", "html", "search", "prose")
MIN_CHARS, MAX_CHARS, PER_TYPE = 1_500, 40_000, 60
SESSIONS, TURNS = 30, 12
SAMPLE_ID = "h2h_sample"

# Already compressed at capture time (distil hook / Headroom) -> would bias the comparison.
_MARKED = re.compile(r"handle=[0-9a-f]{8}|<<distil|hash=[0-9a-f]{8}|<<ccr:|lines omitted")
_CODE_EXT = re.compile(
    r"\.(py|pyi|ts|tsx|js|jsx|mjs|go|rs|java|kt|c|cc|cpp|h|hpp|rb|php|swift|sh|zsh|bash|"
    r"sql|toml|ya?ml|css|scss|lua|scala|cs|proto|tf)\b"
)
_PROSE_EXT = re.compile(r"\.(md|mdx|txt|rst|adoc)\b")
_HTML_TAG = re.compile(r"<(?:html|body|div|head|!doctype|table|p)\b", re.I)
_SEARCH_CMD = re.compile(r"^\s*(?:grep|rg|ag|find|fd|ls|git\s+grep)\b")
_VIEW_CMD = re.compile(r"^\s*(?:cat|sed|head|tail|nl|git\s+(?:diff|show))\b")

ARMS: dict[str, dict[str, Any]] = {
    "distil-lossless": {"tool": "distil", "kw": {"verbatim": True}},
    "distil-default": {"tool": "distil", "kw": {}},
    "distil-aggressive": {"tool": "distil", "kw": {"evict": True}},
    "headroom-densify": {"tool": "headroom", "kw": {"mode": "agent"}},
    "headroom-default": {"tool": "headroom", "kw": {}},
    # Profiles run through the real proxy pipeline: compress(savings_profile=...) does not
    # deliver the profile's router knobs (force_kompress, max_items, ...); measured, the two
    # paths give different outputs on the same input.
    "headroom-coding": {"tool": "headroom", "kw": {}, "proxy_profile": "coding"},
    "headroom-agent90": {"tool": "headroom", "kw": {}, "proxy_profile": "agent-90"},
}
# Same-tier pairs for the paired bootstrap: (distil arm, Headroom arm).
PAIRS = (
    ("distil-lossless", "headroom-densify"),
    ("distil-default", "headroom-default"),
    ("distil-default", "headroom-coding"),
    ("distil-aggressive", "headroom-agent90"),
)


# ----------------------------------------------------------------------------- collect


def classify(tool: str, inp: dict[str, Any], text: str) -> str | None:
    """Content type by the pre-registered rule (README), first match wins."""
    s = text.strip()
    if s[:1] in "[{":
        try:
            json.loads(s)
            return "json"
        except ValueError:
            pass
    if len(re.findall(r"<[a-zA-Z!/][^>]{0,200}>", text)) >= 20 and _HTML_TAG.search(text):
        return "html"
    cmd = str(inp.get("command", "")) if isinstance(inp, dict) else ""
    path = str(inp.get("file_path") or inp.get("path") or "") if isinstance(inp, dict) else ""
    if tool in ("Grep", "Glob", "WebSearch") or (tool == "Bash" and _SEARCH_CMD.match(cmd)):
        return "search"
    if (tool == "Read" and _CODE_EXT.search(path)) or (
        tool == "Bash" and _VIEW_CMD.match(cmd) and (_CODE_EXT.search(cmd) or "git" in cmd)
    ):
        return "code"
    if (tool == "Read" and _PROSE_EXT.search(path)) or tool == "WebFetch" or "page_text" in tool:
        return "prose"
    if tool in ("Bash", "Monitor", "BashOutput"):
        return "logs"
    return None


def _sessions(root: Path) -> list[tuple[str, list[Msg]]]:
    from distil.whatif import discover, parse_transcript

    out = []
    for path in discover(root):
        try:
            sess = parse_transcript(path.read_text(errors="replace").splitlines())
        except OSError:
            continue
        if sess.shape != "anthropic":
            continue
        msgs = [_no_thinking(m) for m in sess.messages]
        out.append((hashlib.sha256(str(path).encode()).hexdigest()[:12], msgs))
    return out


def _no_thinking(m: Msg) -> Msg:
    c = m.get("content")
    if isinstance(c, list):
        c = [b for b in c if not (isinstance(b, dict) and "thinking" in str(b.get("type")))]
        c = c or [{"type": "text", "text": "."}]
    return {"role": m.get("role"), "content": c}


def _query(msgs: list[Msg], upto: int) -> str:
    for m in reversed(msgs[:upto]):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        texts = (
            [c]
            if isinstance(c, str)
            else [
                b.get("text", "")
                for b in c or []
                if isinstance(b, dict) and b.get("type") == "text"
            ]
        )
        t = " ".join(x for x in texts if x and not x.startswith("<")).strip()
        if t:
            return t[:400]
    return "Continue the task."


def collect(root: Path, corpus: Path) -> dict[str, Any]:
    pools: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in CTYPES}
    sessions: list[dict[str, Any]] = []
    for sid, msgs in _sessions(root):
        uses = {
            b.get("id"): b
            for m in msgs
            if m.get("role") == "assistant" and isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        for i, m in enumerate(msgs):
            if m.get("role") != "user" or not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                    continue
                text = block_text(b)
                if not MIN_CHARS <= len(text) <= MAX_CHARS or _MARKED.search(text):
                    continue
                use = uses.get(b.get("tool_use_id")) or {}
                tool, inp = str(use.get("name", "")), use.get("input") or {}
                ctype = classify(tool, inp, text)
                h = hashlib.sha256(text.encode()).hexdigest()
                if ctype and h not in pools[ctype]:
                    pools[ctype][h] = {
                        "id": h[:16],
                        "ctype": ctype,
                        "source": "claude-code",
                        "tool": tool,
                        "input": inp,
                        "query": _query(msgs, i),
                        "text": text,
                    }
        ends = _request_ends(msgs)
        if len(ends) >= TURNS:
            head = msgs[: ends[TURNS - 1] + 1]  # + the reply, so request 12 still ends
            results = [
                block_text(b)
                for m in head
                if isinstance(m.get("content"), list)
                for b in m["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if results and not any(_MARKED.search(t) or len(t) > MAX_CHARS for t in results):
                sessions.append({"id": sid, "messages": head})
    for f in sorted(corpus.glob("*.json")):
        data = json.loads(f.read_text())
        turns = data.get("turns") if isinstance(data, dict) else None
        if not turns:
            continue
        users = [b["text"] for b in turns[0]["blocks"] if b.get("kind") == "user"]
        query = (users[0] if users else "Continue the task.")[:400]
        for turn in turns:
            for b in turn.get("blocks", []):
                text = str(b.get("text", ""))
                if b.get("stability") != "volatile" or not MIN_CHARS <= len(text) <= MAX_CHARS:
                    continue
                ctype = classify("Bash", {"command": "cat"}, text) or "logs"
                h = hashlib.sha256(text.encode()).hexdigest()
                pools[ctype].setdefault(
                    h,
                    {
                        "id": h[:16],
                        "ctype": ctype,
                        "source": "corpus",
                        "tool": "Bash",
                        "input": {"command": f"cat {b.get('id', 'out')}"},
                        "query": query,
                        "text": text,
                    },
                )
    rng = random.Random(0)
    samples = []
    for c in CTYPES:
        pool = sorted(pools[c].values(), key=lambda s: s["id"])
        samples += rng.sample(pool, min(PER_TYPE, len(pool)))
    sessions.sort(key=lambda s: s["id"])
    sessions = rng.sample(sessions, min(SESSIONS, len(sessions)))
    return {
        "samples": samples,
        "sessions": sessions,
        "pool_sizes": {c: len(pools[c]) for c in CTYPES},
    }


# --------------------------------------------------------------------------- the tools


class Tok:
    """One tokenizer for every arm (policysim's TokenModel interface)."""

    def __init__(self) -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> float:
        return float(len(self.enc.encode(text, disallowed_special=())))


class Tool:
    """One arm: ``run(messages) -> (messages, recover(text) -> blobs, failed)``."""

    def __init__(self, arm: str) -> None:
        spec = ARMS[arm]
        self.arm, self.tool, self.kw = arm, spec["tool"], dict(spec["kw"])
        if self.tool == "headroom":
            os.environ.setdefault("HEADROOM_TELEMETRY", "off")
            from headroom.transforms.kompress_compressor import warm_kompress_model

            self.kompress_ready = warm_kompress_model()
            from headroom import compress
            from headroom.cache.compression_store import get_compression_store

            self._hr, self._store, self._proxy = compress, get_compression_store, None
            prof = spec.get("proxy_profile")
            if prof:  # exactly what `headroom proxy` builds at startup for this profile
                os.environ["HEADROOM_SAVINGS_PROFILE"] = prof
                from headroom.agent_savings import proxy_pipeline_kwargs, seed_proxy_env_defaults
                from headroom.proxy.server import HeadroomProxy, _proxy_config_from_env

                seed_proxy_env_defaults()
                cfg = _proxy_config_from_env()
                self._proxy = HeadroomProxy(cfg).anthropic_pipeline
                self.kw = proxy_pipeline_kwargs(cfg)
                self.proxy_mode = cfg.mode
        else:
            from distil.adapters.anthropic import compress_messages

            self._d = compress_messages

    def run(self, msgs: list[Msg], frozen: int = 0) -> tuple[list[Msg], Any, bool]:
        if self.tool == "distil":
            kw = dict(self.kw)
            if kw.pop("evict", False):
                kw["evict"] = frozenset({SAMPLE_ID})
            out, store = self._d(msgs, persist=False, **kw)

            def recover(text: str) -> list[str]:
                from distil.retention import _HANDLE_RE

                return [store.expand(h) for h in _HANDLE_RE.findall(text) if h in store.handles]

            return out, recover, False
        model = "claude-sonnet-4-5-20250929"
        if self._proxy is not None:  # the anthropic handler's call, as in proxy/handlers
            from headroom.utils import extract_user_query

            r = self._proxy.apply(
                messages=msgs,
                model=model,
                model_limit=200_000,
                context=extract_user_query(msgs),
                frozen_message_count=frozen,
                prefix_replay_guaranteed=True,
                **self.kw,
            )
            failed = False  # the pipeline raises; caught by the caller
        else:
            r = self._hr(msgs, model=model, frozen_message_count=frozen, **self.kw)
            failed = bool(msgs) and r.tokens_before == 0  # compress() swallowed an exception

        def recover(text: str) -> list[str]:
            blobs = []
            for h in re.findall(r"(?:hash=|<<ccr:)([0-9A-Za-z]{6,64})", text):
                e = self._store().retrieve(h)
                if e is not None:
                    blobs.append(str(e.original_content))
            return blobs

        return r.messages, recover, failed


# ------------------------------------------------------------------------- block samples


def layout(s: dict[str, Any]) -> list[Msg]:
    def turn(tid: str, name: str, inp: Any, content: str) -> list[Msg]:
        return [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}],
            },
        ]

    msgs: list[Msg] = [{"role": "user", "content": s["query"]}]
    msgs += turn(SAMPLE_ID, s["tool"] or "Bash", s["input"], s["text"])
    for i in range(3):
        msgs += turn(f"f{i}", "Bash", {"command": "true"}, "ok")
    return msgs


def _find_result(msgs: list[Msg]) -> str:
    for m in msgs:
        c = m.get("content")
        for b in c if isinstance(c, list) else []:
            if isinstance(b, dict) and b.get("tool_use_id") == SAMPLE_ID:
                return block_text(b)
    return ""


def score(original: str, compressed: str, blobs: list[str]) -> dict[str, Any]:
    """distil's retention-harness classification, with recovery = this tool's own blobs."""
    from distil.retention import _in_recovery, _normalize, _survives, extract_targets

    norm, recovery = _normalize(compressed), "\n".join(blobs)
    out: dict[str, Any] = {"total": 0, "retained": 0, "recoverable": 0}
    for dim, values in extract_targets(original).items():
        for v in values:
            out["total"] += 1
            if _survives(dim, v, compressed, norm):
                out["retained"] += 1
            elif _in_recovery(v, recovery):
                out["recoverable"] += 1
    return out


def reversibility(tool: str, original: str, compressed: str, blobs: list[str], facts: dict) -> str:
    if compressed == original:
        return "identical"
    if original in blobs:
        return "exact"
    if tool == "headroom":
        from headroom.compress import _string_pair_is_lossless

        if _string_pair_is_lossless(original, compressed):
            return "decoder"
    intact = facts["retained"] + facts["recoverable"] == facts["total"]
    return "facts-intact" if intact else "lossy"


def run_samples(t: Tool, samples: list[dict[str, Any]], tok: Tok) -> list[dict[str, Any]]:
    rows = []
    for warm in samples[:3]:  # first calls pay one-off imports/JIT; not timed
        t.run(layout(warm))
    for s in samples:
        msgs, error = layout(s), ""
        t0 = time.perf_counter()
        try:
            out, recover, failed = t.run(msgs)
        except Exception as e:  # noqa: BLE001 - counted, never hidden
            out, recover, failed, error = msgs, (lambda _t: []), True, type(e).__name__
        ms = (time.perf_counter() - t0) * 1000
        comp = _find_result(out)
        blobs = recover(comp)
        facts = score(s["text"], comp, blobs)
        before, after = tok.count(s["text"]), tok.count(comp)
        rows.append(
            {
                "id": s["id"],
                "ctype": s["ctype"],
                "source": s["source"],
                "ms": round(ms, 3),
                "failed": failed,
                "error": error,
                "tok_before": before,
                "tok_after": after,
                "reduction": 1 - after / before if before else 0.0,
                **facts,
                "reversibility": reversibility(t.tool, s["text"], comp, blobs, facts),
            }
        )
    return rows


# ------------------------------------------------------------------------------ sessions


def _canon(msgs: list[Msg]) -> list[str]:
    return [json.dumps(m, sort_keys=True, default=str) for m in _strip_cc(msgs)]


def price_session(requests: list[list[Msg]], tok: Tok) -> dict[str, Any]:
    """Price forwarded requests with the policysim Anthropic cache model; count prefix holds."""
    prov = Anthropic(model="claude-sonnet-5-5")
    usd = sent = read = 0.0
    held = []
    prev: list[str] | None = None
    for k, req in enumerate(requests):
        u = prov.request(segments(_strip_cc(req), tok, {}, 0.0), t=10.0 * k, ttl="5m")
        usd += prov.price.usd(u)
        sent += u.prompt
        read += u.read
        cur = _canon(req)
        if prev is not None:
            held.append(cur[: len(prev)] == prev)
        prev = cur
    return {"usd": usd, "tokens": sent, "read": read, "prefix_held": held}


def run_sessions(
    t: Tool | None, sessions: list[dict[str, Any]], tok: Tok, mode: str
) -> list[dict[str, Any]]:
    """``mode``: plain | served (distil: adapter + prefix replay) | delta | stateless."""
    from distil import prefixreplay

    rows = []
    for sess in sessions:
        msgs = sess["messages"]
        ends = _request_ends(msgs)[:TURNS]
        forwarded: list[list[Msg]] = []
        times, fails = [], 0
        prev_orig: list[Msg] | None = None
        prev_fwd: list[Msg] | None = None
        prefixreplay.reset()
        for k, end in enumerate(ends):
            req = _cc_last(msgs[:end])
            t0 = time.perf_counter()
            try:
                if t is None:
                    out = req
                elif mode == "served":
                    out, _, failed = t.run(req)
                    out = prefixreplay.replay(sess["id"], req, out)[0]
                    fails += failed
                elif mode == "delta":
                    out, failed = _headroom_delta(t, req, prev_orig, prev_fwd)
                    fails += failed
                else:
                    out, _, failed = t.run(req)
                    fails += failed
            except Exception:  # noqa: BLE001 - a crashed request forwards the original
                out, fails = req, fails + 1
            times.append((time.perf_counter() - t0) * 1000)
            prev_orig, prev_fwd = copy.deepcopy(req), copy.deepcopy(out)
            forwarded.append(out)
        rows.append(
            {
                "id": sess["id"],
                "requests": len(ends),
                "fails": fails,
                "ms": times,
                **price_session(forwarded, tok),
            }
        )
    return rows


def _headroom_delta(
    t: Tool, req: list[Msg], prev_orig: list[Msg] | None, prev_fwd: list[Msg] | None
) -> tuple[list[Msg], bool]:
    """Headroom proxy cache mode, as benchmarks/policysim/headroom_worker.py reproduces it."""
    from headroom.cache.prefix_tracker import extract_cache_stable_delta

    delta = extract_cache_stable_delta(req, prev_orig, prev_fwd) if prev_orig else None
    if delta is not None:
        prefix, tail = delta
        out, _, failed = t.run(prefix + tail, frozen=len(prefix))
        return prefix + out[len(prefix) :], failed
    if prev_orig:
        return req, False  # non-append change: the proxy forwards unmodified
    out, _, failed = t.run(req)
    return out, failed


# -------------------------------------------------------------------------------- report


def boot(xs: list[float], n: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Mean and percentile-bootstrap 95% CI of ``xs`` (paired differences, usually)."""
    if not xs:
        return {"n": 0}
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return {
        "n": k,
        "mean": statistics.fmean(xs),
        "lo": means[int(0.025 * n)],
        "hi": means[int(0.975 * n) - 1],
    }


def _recall(r: dict[str, Any], visible: bool) -> float | None:
    if not r["total"]:
        return None
    return (r["retained"] + (0 if visible else r["recoverable"])) / r["total"]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def report(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"arms": {}, "paired": {}, "sessions": {}}
    for name, a in arms.items():
        rows = a.get("samples", [])
        per: dict[str, Any] = {}
        for ct in (*CTYPES, "all"):
            rs = [r for r in rows if ct in ("all", r["ctype"])]
            if not rs:
                continue
            vis = [x for r in rs if (x := _recall(r, True)) is not None]
            tru = [x for r in rs if (x := _recall(r, False)) is not None]
            rev = {
                k: sum(r["reversibility"] == k for r in rs)
                for k in ("identical", "exact", "decoder", "facts-intact", "lossy")
            }
            per[ct] = {
                "n": len(rs),
                "n_with_facts": len(tru),
                "reduction": statistics.fmean(r["reduction"] for r in rs),
                "visible_recall": statistics.fmean(vis) if vis else None,
                "true_recall": statistics.fmean(tru) if tru else None,
                "facts_lost": sum(r["total"] - r["retained"] - r["recoverable"] for r in rs),
                "facts_total": sum(r["total"] for r in rs),
                "exactly_recoverable": (rev["identical"] + rev["exact"] + rev["decoder"]) / len(rs),
                "reversibility": rev,
            }
        ms = [r["ms"] for r in rows]
        out["arms"][name] = {
            **{k: v for k, v in a.items() if k not in ("samples", "sessions", "sessions_by_mode")},
            "by_ctype": per,
            "sample_latency_ms": {"p50": _pct(ms, 0.5), "p95": _pct(ms, 0.95)},
            "sample_failures": sum(r["failed"] for r in rows),
        }
    for d, h in PAIRS:
        if d not in arms or h not in arms:
            continue
        hd = {r["id"]: r for r in arms[h]["samples"]}
        res: dict[str, Any] = {}
        for ct in (*CTYPES, "all"):
            pairs = [
                (r, hd[r["id"]])
                for r in arms[d]["samples"]
                if r["id"] in hd and ct in ("all", r["ctype"])
            ]
            res[ct] = {
                "reduction_d_minus_h": boot([a["reduction"] - b["reduction"] for a, b in pairs]),
                "true_recall_d_minus_h": boot(
                    [_recall(a, False) - _recall(b, False) for a, b in pairs if a["total"]]
                ),
                "visible_recall_d_minus_h": boot(
                    [_recall(a, True) - _recall(b, True) for a, b in pairs if a["total"]]
                ),
            }
        out["paired"][f"{d} vs {h}"] = res
    plain = {r["id"]: r for r in arms.get("plain", {}).get("sessions", [])}
    for name, a in arms.items():
        for mode, rows in a.get("sessions_by_mode", {}).items():
            held = [x for r in rows for x in r["prefix_held"]]
            req_ms = [x for r in rows for x in r["ms"]]
            rel = [r["usd"] / plain[r["id"]]["usd"] - 1 for r in rows if r["id"] in plain]
            out["sessions"][f"{name}/{mode}"] = {
                "n_sessions": len(rows),
                "n_requests": len(req_ms),
                "usd_per_session": statistics.fmean(r["usd"] for r in rows),
                "tokens_per_session": statistics.fmean(r["tokens"] for r in rows),
                "cache_read_share": sum(r["read"] for r in rows) / sum(r["tokens"] for r in rows),
                "prefix_held": sum(held) / len(held) if held else None,
                "usd_vs_plain": boot(rel),
                "request_latency_ms": {"p50": _pct(req_ms, 0.5), "p95": _pct(req_ms, 0.95)},
                "request_failures": sum(r["fails"] for r in rows),
                "_usd": {r["id"]: r["usd"] for r in rows},
            }
    # paired session $ differences, distil served vs every Headroom session mode
    for d in ("distil-default/served", "distil-lossless/served"):
        dd = out["sessions"].get(d, {}).get("_usd", {})
        for k, v in out["sessions"].items():
            if k.startswith("headroom") and dd:
                hh = v["_usd"]
                out["paired"][f"{d} vs {k} (usd/session, vs plain)"] = boot(
                    [(dd[i] - hh[i]) / plain[i]["usd"] for i in dd if i in hh and i in plain]
                )
    for v in out["sessions"].values():
        v.pop("_usd", None)
    return out


# ---------------------------------------------------------------------------------- main


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    out = {"python": sys.version.split()[0]}
    for pkg in ("distil-llm", "headroom-ai", "tiktoken", "onnxruntime", "transformers"):
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = "absent"
    return out


def cmd_arm(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.inputs).read_text())
    tok = Tok()
    res: dict[str, Any] = {"arm": args.arm, "versions": _versions()}
    if args.arm == "plain":
        res["sessions_by_mode"] = {"plain": run_sessions(None, data["sessions"], tok, "plain")}
        res["sessions"] = res["sessions_by_mode"]["plain"]
    else:
        t0 = time.perf_counter()
        t = Tool(args.arm)
        res["init_s"] = round(time.perf_counter() - t0, 2)
        res["kompress_ready"] = getattr(t, "kompress_ready", None)
        res["headroom_proxy_mode"] = getattr(t, "proxy_mode", None)
        res["tool_kwargs"] = {
            k: v for k, v in t.kw.items() if isinstance(v, (str, int, float, bool))
        }
        res["samples"] = run_samples(t, data["samples"], tok)
        modes = ["served"] if t.tool == "distil" else ["delta", "stateless"]
        if args.arm != "distil-aggressive":  # cold-point needs an expired cache; none offline
            res["sessions_by_mode"] = {m: run_sessions(t, data["sessions"], tok, m) for m in modes}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    c.add_argument("--corpus", default="corpus")
    c.add_argument("--out", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True, choices=["plain", *ARMS])
    a.add_argument("--inputs", required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--arms", required=True)
    r.add_argument("--out", required=True)
    al = sub.add_parser("all")
    al.add_argument("--scratch", required=True)
    al.add_argument("--out", default="benchmarks/results/headroom-0-42-h2h/results.json")
    args = ap.parse_args(argv)

    if args.cmd == "collect":
        data = collect(Path(args.root), Path(args.corpus))
        Path(args.out).write_text(json.dumps(data))
        print(
            json.dumps(
                {
                    "samples": len(data["samples"]),
                    "sessions": len(data["sessions"]),
                    "pool_sizes": data["pool_sizes"],
                }
            )
        )
    elif args.cmd == "arm":
        cmd_arm(args)
    elif args.cmd in ("report", "all"):
        if args.cmd == "all":
            scratch = Path(args.scratch)
            inputs = scratch / "inputs.json"
            main(["collect", "--out", str(inputs)])
            for arm in ["plain", *ARMS]:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.headroom_h2h",
                        "arm",
                        "--arm",
                        arm,
                        "--inputs",
                        str(inputs),
                        "--out",
                        str(scratch / "arms" / f"{arm}.json"),
                    ],
                    check=True,
                )
            args.arms = str(scratch / "arms")
        arms = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.arms).glob("*.json"))}
        res = report(arms)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

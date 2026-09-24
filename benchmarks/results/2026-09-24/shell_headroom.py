"""How much would per-command shell-output profiles (RTK-style) add over what distil does today?

Measured on REAL data only, never synthetic:

* ``~/.claude/projects/**/*.jsonl`` — Claude Code's own transcripts: the original (pre-distil)
  tool_result text, the Bash command that produced it, and the provider's billed ``usage`` for
  every request. This is the only local source that has shell-output TEXT.
* ``~/.distil/sessions/*.requests.jsonl`` — distil's content-free per-request census, used to
  cross-check the composition of what the proxy actually forwarded.

Reads both, writes nothing but the JSON next to this file. Emits aggregates only — no command
text or output leaves this script except the command *family* (``git status``, ``pytest``).

Bill model (input-token equivalents, Anthropic list ratios): input 1, cache write 1.25, cache
read 0.1, output 5. A tool_result first appears in the next request's cache WRITE, then is a
cache READ on every later request of that session. So a token removed deterministically at
first sight is worth ``1.25 + 0.1 * (later_requests - 1)``. Compaction (which drops old results)
is handled by ending a result's life at the next ``compact_boundary``; 5-minute cache expiry
(re-writes) is not modelled — on this traffic cache reads outnumber writes ~29:1, so it is
small.

Run: uv run --python 3.12 python benchmarks/results/2026-09-24/shell_headroom.py
"""

from __future__ import annotations

import json
import os
import shlex
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path.home()
# ~/.distil is read-only here: distil's restore store persists every digest handle to disk,
# so point every distil path at a throwaway dir BEFORE importing it, and stub the write.
_SCRATCH = tempfile.mkdtemp(prefix="shell-headroom-")
os.environ["HOME"] = os.environ["DISTIL_HOME"] = _SCRATCH

import distil.adapters.anthropic as _anthropic  # noqa: E402
from distil.adapters.anthropic import RestoreStore, _compress_tool_result_text  # noqa: E402
from distil.compress.provenance import whole_file_read_paths  # noqa: E402
from distil.tokenizer import DEFAULT as TOK  # noqa: E402

_anthropic._record_restore = lambda handle, original: True  # type: ignore[assignment]
OUT = Path(__file__).with_name("shell_headroom.json")
W_IN, W_WRITE, W_READ, W_OUT = 1.0, 1.25, 0.1, 5.0
RTK_CEILING = 0.90  # top of RTK's own 60-90% claim, applied to EVERY eligible shell output
DECISION_THRESHOLD = 0.02
# The command families RTK ships profiles for (its README), as `family()` spells them.
RTK_FAMILIES = ("git ", "cargo ", "npm ", "pnpm ", "yarn ", "go ", "docker ", "kubectl ")
RTK_EXACT = {
    "pytest",
    "ls",
    "find",
    "grep",
    "rg",
    "tree",
    "jest",
    "vitest",
    "tsc",
    "eslint",
    "git",
    "npx",
}
_SUBCOMMANDS = {"git", "cargo", "npm", "pnpm", "yarn", "go", "gh", "docker", "kubectl", "uv"}
_PREAMBLE = {"cd", "export", "source", "set", "sleep", "env"}


def family(command: str) -> str:
    """Coarse command family: the first real stage, with its subcommand for multi-tools."""
    norm = command.replace("||", ";").replace("&&", ";").replace("\n", ";").replace("|", ";")
    for stage in (s.strip() for s in norm.split(";")):
        try:
            toks = shlex.split(stage)
        except ValueError:
            toks = stage.split()
        toks = [t for t in toks if "=" not in t or t.startswith("-")]  # drop VAR=x prefixes
        if not toks or toks[0] in _PREAMBLE:
            continue
        if any(t.endswith("pytest") for t in toks):
            return "pytest"
        head = toks[0].rsplit("/", 1)[-1]
        if head in _SUBCOMMANDS and len(toks) > 1 and not toks[1].startswith("-"):
            return f"{head} {toks[1]}"
        return head
    return "(empty)"


def result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def tokens_after(text: str, *, verbatim: bool, recent: bool) -> int:
    return TOK.count(_compress_tool_result_text(text, RestoreStore(), verbatim, recent))


def scan_transcripts() -> dict:
    bill = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    requests_total = 0
    sessions = 0
    # (tokens, later_requests, family, is_bash, is_read, removed_first_sight, removed_digest,
    #  removed_lossless)
    rows: list[tuple[int, int, str, bool, bool, int, int, int]] = []
    for path in sorted((HOME / ".claude" / "projects").rglob("*.jsonl")):
        seen_msgs: set[str] = set()
        bash_cmd: dict[str, str] = {}
        pending: list[tuple[int, str, str | None]] = []  # (request idx at arrival, text, cmd)
        compactions: list[int] = []  # request index at each /compact: older results leave
        n_req = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("subtype") == "compact_boundary":
                compactions.append(n_req)
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if rec.get("type") == "assistant":
                mid = msg.get("id")
                usage = msg.get("usage")
                if mid and mid not in seen_msgs and isinstance(usage, dict):
                    seen_msgs.add(mid)
                    n_req += 1
                    bill["input"] += usage.get("input_tokens") or 0
                    bill["cache_write"] += usage.get("cache_creation_input_tokens") or 0
                    bill["cache_read"] += usage.get("cache_read_input_tokens") or 0
                    bill["output"] += usage.get("output_tokens") or 0
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        inp = b.get("input") or {}
                        if b.get("name") == "Bash" and isinstance(inp.get("command"), str):
                            bash_cmd[str(b.get("id"))] = inp["command"]
            elif rec.get("type") == "user" and isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        text = result_text(b.get("content"))
                        if text:
                            pending.append((n_req, text, bash_cmd.get(str(b.get("tool_use_id")))))
        if not n_req:
            continue
        sessions += 1
        requests_total += n_req
        for idx, text, cmd in pending:
            # Lives in context until the next compaction after it arrived, or session end.
            later = min([c for c in compactions if c > idx] or [n_req]) - idx
            if later <= 0:
                continue  # never sent to the provider
            tok = TOK.count(text)
            is_bash = cmd is not None
            is_read = bool(is_bash and whole_file_read_paths(cmd or ""))
            if is_bash:
                first = tok - tokens_after(text, verbatim=True, recent=True)
                dig = tok - tokens_after(text, verbatim=False, recent=False)
                lossless = tok - tokens_after(text, verbatim=True, recent=False)
            else:
                first = dig = lossless = 0
            rows.append(
                (tok, later, family(cmd) if cmd else "", is_bash, is_read, first, dig, lossless)
            )
    return {"bill": bill, "requests": requests_total, "sessions": sessions, "rows": rows}


def pooled_calibration() -> float:
    """Billed/heuristic token ratio pooled over every model distil has learned (read-only).

    Tool-result sizes are counted with distil's offline heuristic; the bill is the provider's
    own count. Without this the numerator is on a different scale from the denominator.
    """
    try:
        models = json.loads((HOME / ".distil" / "calibration.json").read_text())["models"]
        billed = sum(m["billed_sum"] for m in models.values())
        est = sum(m["est_sum"] for m in models.values())
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"no calibration ({exc}); using 1.0", file=sys.stderr)
        return 1.0
    return billed / est if est else 1.0


def scan_census() -> dict:
    buckets: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    n = 0
    for path in sorted((HOME / ".distil" / "sessions").glob("*.requests.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            census = rec.get("census")
            if not isinstance(census, dict):
                continue
            n += 1
            modes[str(rec.get("mode"))] += 1
            for k, v in census.items():
                if isinstance(v, int):
                    buckets[k] += v
    total = sum(buckets.values()) or 1
    return {
        "requests": n,
        "modes": dict(modes.most_common()),
        "bucket_share": {k: round(v / total, 4) for k, v in buckets.most_common()},
        "note": "payload-token share per request, summed (NOT bill-weighted; content-free, "
        "has no tool-name split, so shell output is inside tool_result_* buckets)",
    }


def main() -> None:
    t = scan_transcripts()
    b = t["bill"]
    bill = (
        W_IN * b["input"]
        + W_WRITE * b["cache_write"]
        + W_READ * b["cache_read"]
        + W_OUT * b["output"]
    )
    rows = t["rows"]
    calib = pooled_calibration()
    bill /= calib  # every share below is heuristic-token numerator over calibrated denominator

    def cached_w(later: int) -> float:
        return W_WRITE + W_READ * (later - 1)

    all_tr = sum(r[0] * cached_w(r[1]) for r in rows)
    bash = [r for r in rows if r[3]]
    eligible = [r for r in bash if not r[4]]  # whole-file reads stay byte-exact by ADR
    bash_w = sum(r[0] * cached_w(r[1]) for r in bash)
    elig_w = sum(r[0] * cached_w(r[1]) for r in eligible)
    first_w = sum(r[5] * cached_w(r[1]) for r in eligible)
    dig_w = sum(r[6] * cached_w(r[1]) for r in eligible)
    loss_w = sum(r[7] * cached_w(r[1]) for r in eligible)

    fam_tok: dict[str, float] = defaultdict(float)
    fam_n: Counter[str] = Counter()
    fam_raw: Counter[str] = Counter()
    fam_dig: Counter[str] = Counter()
    for r in eligible:
        fam_tok[r[2]] += r[0] * cached_w(r[1])
        fam_n[r[2]] += 1
        fam_raw[r[2]] += r[0]
        fam_dig[r[2]] += r[6]
    top = sorted(fam_tok.items(), key=lambda kv: -kv[1])[:15]

    # What a profile could ADD. On the traffic measured (census: 100% digest mode, Claude Code
    # marks its last message cache_control) distil already digests every non-read shell output
    # at FIRST sight — recency is anchored to the client's cache breakpoint, so nothing is
    # exempt — which makes `digest` the cache-stable baseline a profile has to beat. Per row a
    # profile at RTK's BEST claimed ratio adds max(0, 0.9*raw - already_removed). Lossless-only
    # mode is out of scope: a content-dropping profile needs distil_expand for recovery, which
    # that mode never injects.
    def incremental(rs: list[tuple[int, int, str, bool, bool, int, int, int]]) -> float:
        return sum(max(0.0, RTK_CEILING * r[0] - r[6]) * cached_w(r[1]) for r in rs) / bill

    ceiling = incremental(eligible)
    rtk_rows = [r for r in eligible if r[2].startswith(RTK_FAMILIES) or r[2] in RTK_EXACT]
    realistic = incremental(rtk_rows)
    # Same families on a lossless-only session (subscription default): nothing digests there,
    # so a profile would face the raw bytes — but it drops content, so it needs the
    # distil_expand tool that mode never injects. Reported for the reopen condition only.
    rtk_lossless = (
        sum(max(0.0, RTK_CEILING * r[0] - r[7]) * cached_w(r[1]) for r in rtk_rows) / bill
    )
    residual_w = sum((r[0] - r[6]) * cached_w(r[1]) for r in eligible)

    sizes = [r[0] for r in eligible]
    result = {
        "date": "2026-09-24",
        "source": "~/.claude/projects/**/*.jsonl (real Claude Code transcripts, provider usage)",
        "sessions": t["sessions"],
        "requests": t["requests"],
        "bill_input_equiv_tokens": round(bill * calib),
        "calibration_billed_per_heuristic_token": round(calib, 4),
        "bill_components": b,
        "bill_weights": {
            "input": W_IN,
            "cache_write": W_WRITE,
            "cache_read": W_READ,
            "output": W_OUT,
        },
        "tool_results": len(rows),
        "bash_results": len(bash),
        "bash_eligible_results": len(eligible),
        "bash_eligible_tokens_p50_p90_max": [
            int(statistics.median(sizes)) if sizes else 0,
            int(statistics.quantiles(sizes, n=10)[-1]) if len(sizes) >= 10 else 0,
            max(sizes, default=0),
        ],
        "share_of_bill": {
            "all_tool_results": round(all_tr / bill, 4),
            "bash_outputs": round(bash_w / bill, 4),
            "bash_outputs_excl_file_reads": round(elig_w / bill, 4),
            "bash_eligible_residual_after_digest": round(residual_w / bill, 4),
        },
        "distil_already_removes_share_of_bill": {
            "digest": round(dig_w / bill, 4),
            "lossless_only": round(loss_w / bill, 4),
            "recency_exempt_tier0": round(first_w / bill, 4),
            "note": "digest is the cache-stable first-sight rendering on clients that mark "
            "cache_control (recency is anchored to the breakpoint, so nothing is exempt); "
            "recency_exempt_tier0 applies only to clients that mark nothing",
        },
        "distil_already_removes_pct_of_bash_tokens": {
            "recency_exempt_tier0": round(sum(r[5] for r in eligible) / max(1, sum(sizes)), 4),
            "digest": round(sum(r[6] for r in eligible) / max(1, sum(sizes)), 4),
        },
        "top_families_by_bill_weight": [
            {
                "family": f,
                "n": fam_n[f],
                "share_of_bill": round(w / bill, 4),
                "raw_tokens": fam_raw[f],
                "digest_removes_pct": round(fam_dig[f] / max(1, fam_raw[f]), 3),
            }
            for f, w in top
        ],
        "incremental_headroom_share_of_bill": {
            "rtk_families_90pct_net_of_digest": round(realistic, 4),
            "ceiling_90pct_every_shell_output_net_of_digest": round(ceiling, 4),
            "rtk_families_90pct_if_lossless_only_session": round(rtk_lossless, 4),
            "rtk_family_results": len(rtk_rows),
        },
        "threshold": DECISION_THRESHOLD,
        "decision": "BUILD" if realistic >= DECISION_THRESHOLD else "STOP",
        "census_cross_check": scan_census(),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

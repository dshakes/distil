"""``distil mcp bench`` — the pre-registered tool-use accuracy harness for every level.

Protocol: ``docs/research/mcp-compressor-protocol.md`` (hypotheses, metrics, sample
size, stopping rule, what counts as a failure). This module is its executable form.

Two modes, one code path:

* **dry run** (default) — a scripted mock model (``oracle``, ``invoke``, ``noisy``,
  ``degraded``, ``no-expand``) drives the REAL proxy (``proxy.Proxy`` over in-process
  fixture backends), so tool surfaces, lazy unlocks, ``list_changed``, the invoke
  fallback, result digests and ``<server>_expand`` are exactly what ships. No network,
  no spend. It validates the plumbing and the statistics — including that a degraded
  arm FAILS certification — and measures the exact token surface of every arm.
* **live** — the same loop against the Anthropic Messages API. Refuses to start
  without ``--live``, ``--budget-usd``, ``ANTHROPIC_API_KEY``, and an up-front cost
  estimate under the budget; stops the moment actual spend reaches the cap.

Arms: ``raw`` (the backend's own tool list, no proxy compression — the reference),
``L0``–``L3`` (definitions), and a separate result-comprehension suite comparing raw
tool results against ``R``.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import conformal as _conformal
from .. import pricing
from ..certify.stats import mcnemar_noninferiority, tost
from ..drift import BUDGET_DELTA
from . import events, fakeserver, levels, proxy

#: The single risk budget's equivalence margin: ``conformal.CERT_MARGIN`` where the
#: risk-budget work has landed, otherwise the identical value ``certify.stats.tost``
#: defaults to. One number, never a second copy of it.
CERT_MARGIN: float = float(
    getattr(_conformal, "CERT_MARGIN", inspect.signature(tost).parameters["margin"].default)
)
#: One-sided test size: the single risk budget's ``BUDGET_DELTA`` (``distil.drift``).
ALPHA: float = BUDGET_DELTA
#: Pre-registered sample sizes (see ``required_n`` and the protocol's power section).
N_TOOL_TASKS = 1000
N_RESULT_TASKS = 630
#: Assumed worst-case discordant-pair rate the sample size is powered for.
P_DISCORDANT = 0.06
P_DISCORDANT_R = 0.04
POWER = 0.80
MAX_TURNS = 4
#: Anthropic's documented tool-use system-prompt overhead for current Claude models with
#: ``tool_choice: auto`` (tokens per request). Verify on the pricing page before a run.
TOOL_SYSTEM_TOKENS = 346
#: Mean output tokens per model turn assumed by the cost estimate (a tool_use block).
EST_OUTPUT_TOKENS = 120
#: The heuristic tokenizer can under-count Claude's by this much; budgets carry it.
ESTIMATE_SAFETY = 1.25
SEQUENCE = ("L0", "L1", "L2", "L3")  # fixed-sequence testing order (protocol §5)
LIVE_MODELS = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
SYSTEM = (
    "You are an agent with access to tools from several MCP servers. Complete the user's "
    "request by calling exactly one tool with the right arguments. Do not ask questions."
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

_POOLS: dict[str, list[Any]] = {
    "path": [
        f"/workspace/{p}/{f}"
        for p in ("app/src", "api/handlers", "docs", "infra/terraform", "web/components")
        for f in ("main.py", "config.yaml", "README.md", "server.ts", "utils.go")
    ],
    "dir": [
        f"/workspace/{d}"
        for d in (
            "app",
            "api",
            "docs",
            "infra",
            "web",
            "scripts",
            "tests",
            "data",
            "build",
            "assets",
            "lib",
            "cmd",
            "pkg",
            "deploy",
            "notes",
            "tmp",
            "vendor",
            "examples",
            "bench",
            "tools",
        )
    ],
    "repo": [
        f"/home/dev/{r}"
        for r in (
            "billing",
            "atlas",
            "orbit",
            "ledger",
            "relay",
            "ember",
            "quartz",
            "harbor",
            "nimbus",
            "pylon",
            "sable",
            "tundra",
            "vertex",
            "willow",
            "zephyr",
            "cobalt",
            "delta",
            "fable",
            "garnet",
            "helix",
        )
    ],
    "branch": [
        "feature/login",
        "fix/null-deref",
        "release/2.4",
        "dev",
        "staging",
        "feat/search",
        "chore/deps",
        "hotfix/crash",
        "exp/cache",
        "docs/api",
        "refactor/db",
        "feat/export",
        "fix/timeout",
        "perf/index",
        "feat/sso",
        "fix/typo",
        "ci/matrix",
        "feat/billing",
        "fix/race",
        "feat/i18n",
    ],
    "owner": [
        "octo-org",
        "acme",
        "torvalds",
        "rust-lang",
        "vercel",
        "pallets",
        "psf",
        "golang",
        "nodejs",
        "denoland",
    ],
    "ghrepo": ["widgets", "api", "cli", "docs", "sdk", "web", "infra", "tools", "core", "site"],
    "n": list(range(3, 400, 17)),
    "k": [3, 5, 7, 10, 12, 15, 20, 25, 30, 40],
    "msg": [
        "fix the flaky retry test",
        "bump version to 2.4.1",
        "add input validation",
        "remove dead code",
        "update the changelog",
        "handle empty payloads",
        "document the CLI flags",
        "speed up the index build",
        "tighten the error messages",
        "switch to structured logging",
    ],
    "title": [
        "Crash on empty config",
        "Add dark mode",
        "Docs: fix install steps",
        "Timeout talking to the API",
        "Support Python 3.13",
        "Flaky CI on Windows",
        "Memory leak in worker",
        "Wrong default port",
        "Improve error for bad token",
        "Export to CSV",
    ],
    "topic": [
        "vector databases",
        "rust async runtimes",
        "terraform modules",
        "graphql servers",
        "static site generators",
        "kubernetes operators",
        "llm evaluation",
        "web accessibility",
        "time series storage",
        "feature flags",
    ],
    "glob": [
        "*.py",
        "*.md",
        "*.test.ts",
        "Dockerfile",
        "*.yaml",
        "*_test.go",
        "*.sql",
        "*.json",
        "*.rs",
        "*.proto",
    ],
    "word": [
        "archive",
        "exports",
        "fixtures",
        "migrations",
        "snapshots",
        "reports",
        "backups",
        "drafts",
        "cache",
        "logs",
    ],
    "sha": [
        "3f9a1c07",
        "a1b2c3d4",
        "deadbeef",
        "9e8d7c6b",
        "0badc0de",
        "c0ffee12",
        "fe1dab1e",
        "abad1dea",
        "1337beef",
        "facade00",
    ],
    "entity": [
        "Alice",
        "Project Orion",
        "Berlin office",
        "Q3 roadmap",
        "Bob",
        "Payments team",
        "Kafka cluster",
        "Carol",
        "Design review",
        "Onboarding doc",
    ],
    "url": [
        f"https://{h}/{p}"
        for h in (
            "example.com",
            "docs.python.org",
            "developer.mozilla.org",
            "go.dev",
            "rust-lang.org",
        )
        for p in ("guide", "reference/api", "blog/latest", "faq")
    ],
    "tz": [
        "Europe/Berlin",
        "America/New_York",
        "Asia/Tokyo",
        "Australia/Sydney",
        "America/Los_Angeles",
        "Asia/Kolkata",
        "Europe/London",
        "America/Sao_Paulo",
        "Africa/Nairobi",
        "Pacific/Auckland",
    ],
    "hhmm": [
        "09:30",
        "14:00",
        "17:45",
        "08:15",
        "22:10",
        "06:00",
        "12:30",
        "19:05",
        "11:11",
        "23:59",
    ],
    "person": [
        "Linus",
        "Ada Lovelace",
        "grace-hopper",
        "Guido",
        "Margaret Hamilton",
        "Ken Thompson",
        "Barbara Liskov",
        "Dennis",
        "Katherine Johnson",
        "Alan Kay",
    ],
    "ident": [
        "parse_config",
        "RetryPolicy",
        "useDebounce",
        "ErrNotFound",
        "fetch_with_timeout",
        "TokenBucket",
        "normalize_path",
        "HttpClient",
        "compute_hash",
        "LruCache",
    ],
    "a": list(range(2, 90, 7)),
    "b": list(range(5, 300, 23)),
}

# (server, acceptable tool names, prompt template, gold-argument template)
TEMPLATES: list[tuple[str, tuple[str, ...], str, dict[str, Any]]] = [
    (
        "filesystem",
        ("read_text_file", "read_file"),
        "Show me the contents of {path}.",
        {"path": "{path}"},
    ),
    (
        "filesystem",
        ("read_text_file", "read_file"),
        "Print only the first {k} lines of {path}.",
        {"path": "{path}", "head": "{k}"},
    ),
    (
        "filesystem",
        ("write_file",),
        "Create the file {path} containing exactly: {msg}",
        {"path": "{path}", "content": "{msg}"},
    ),
    (
        "filesystem",
        ("list_directory",),
        "What files and folders are directly inside {dir}?",
        {"path": "{dir}"},
    ),
    (
        "filesystem",
        ("list_directory_with_sizes",),
        "List {dir} with file sizes, sorted by size.",
        {"path": "{dir}", "sortBy": "size"},
    ),
    (
        "filesystem",
        ("create_directory",),
        "Make a new folder at {dir}/{word}.",
        {"path": "{dir}/{word}"},
    ),
    (
        "filesystem",
        ("move_file",),
        "Move {path} to {dir}/{word}/moved.txt.",
        {"source": "{path}", "destination": "{dir}/{word}/moved.txt"},
    ),
    (
        "filesystem",
        ("search_files",),
        "Find every file matching '{glob}' anywhere under {dir}.",
        {"path": "{dir}", "pattern": "{glob}"},
    ),
    (
        "filesystem",
        ("get_file_info",),
        "When was {path} last modified, and how big is it?",
        {"path": "{path}"},
    ),
    (
        "filesystem",
        ("directory_tree",),
        "Give me a recursive JSON tree of {dir}.",
        {"path": "{dir}"},
    ),
    (
        "filesystem",
        ("list_allowed_directories",),
        "Which directories is the filesystem server allowed to access?",
        {},
    ),
    (
        "git",
        ("git_status",),
        "What's the git status of the repository at {repo}?",
        {"repo_path": "{repo}"},
    ),
    ("git", ("git_diff_unstaged",), "Show my unstaged changes in {repo}.", {"repo_path": "{repo}"}),
    (
        "git",
        ("git_diff_staged",),
        "Show what is currently staged for commit in {repo}.",
        {"repo_path": "{repo}"},
    ),
    (
        "git",
        ("git_diff",),
        "Diff the working tree of {repo} against the {branch} branch.",
        {"repo_path": "{repo}", "target": "{branch}"},
    ),
    (
        "git",
        ("git_commit",),
        'Commit the staged changes in {repo} with the message "{msg}".',
        {"repo_path": "{repo}", "message": "{msg}"},
    ),
    (
        "git",
        ("git_add",),
        "Stage the file src/app.py in {repo}.",
        {"repo_path": "{repo}", "files": ["src/app.py"]},
    ),
    (
        "git",
        ("git_reset",),
        "Unstage everything that is staged in {repo}.",
        {"repo_path": "{repo}"},
    ),
    (
        "git",
        ("git_log",),
        "Show the last {k} commits in {repo}.",
        {"repo_path": "{repo}", "max_count": "{k}"},
    ),
    (
        "git",
        ("git_create_branch",),
        "Create a new local branch called {branch} in {repo}.",
        {"repo_path": "{repo}", "branch_name": "{branch}"},
    ),
    (
        "git",
        ("git_checkout",),
        "Switch {repo} to the existing {branch} branch.",
        {"repo_path": "{repo}", "branch_name": "{branch}"},
    ),
    (
        "git",
        ("git_show",),
        "Show the contents of commit {sha} in {repo}.",
        {"repo_path": "{repo}", "revision": "{sha}"},
    ),
    (
        "github",
        ("search_repositories",),
        "Search GitHub for repositories about {topic}.",
        {"query": "{topic}"},
    ),
    (
        "github",
        ("get_file_contents",),
        "Get README.md from the {owner}/{ghrepo} repository on GitHub.",
        {"owner": "{owner}", "repo": "{ghrepo}", "path": "README.md"},
    ),
    (
        "github",
        ("create_issue",),
        'Open a GitHub issue in {owner}/{ghrepo} titled "{title}".',
        {"owner": "{owner}", "repo": "{ghrepo}", "title": "{title}"},
    ),
    (
        "github",
        ("list_issues",),
        "List the open issues in {owner}/{ghrepo} on GitHub.",
        {"owner": "{owner}", "repo": "{ghrepo}", "state": "open"},
    ),
    (
        "github",
        ("get_issue",),
        "Show GitHub issue #{n} in {owner}/{ghrepo}.",
        {"owner": "{owner}", "repo": "{ghrepo}", "issue_number": "{n}"},
    ),
    (
        "github",
        ("add_issue_comment",),
        'Comment "{msg}" on GitHub issue #{n} in {owner}/{ghrepo}.',
        {"owner": "{owner}", "repo": "{ghrepo}", "issue_number": "{n}", "body": "{msg}"},
    ),
    (
        "github",
        ("create_pull_request",),
        'Open a pull request in {owner}/{ghrepo} from {branch} into main titled "{title}".',
        {
            "owner": "{owner}",
            "repo": "{ghrepo}",
            "title": "{title}",
            "head": "{branch}",
            "base": "main",
        },
    ),
    (
        "github",
        ("get_pull_request",),
        "Show pull request #{n} in {owner}/{ghrepo}.",
        {"owner": "{owner}", "repo": "{ghrepo}", "pull_number": "{n}"},
    ),
    (
        "github",
        ("merge_pull_request",),
        "Squash-merge pull request #{n} in {owner}/{ghrepo}.",
        {"owner": "{owner}", "repo": "{ghrepo}", "pull_number": "{n}", "merge_method": "squash"},
    ),
    (
        "github",
        ("list_commits",),
        "List the recent commits of {owner}/{ghrepo} on GitHub.",
        {"owner": "{owner}", "repo": "{ghrepo}"},
    ),
    (
        "github",
        ("fork_repository",),
        "Fork {owner}/{ghrepo} into my own GitHub account.",
        {"owner": "{owner}", "repo": "{ghrepo}"},
    ),
    ("github", ("search_code",), 'Search all GitHub code for "{ident}".', {"q": "{ident}"}),
    (
        "github",
        ("get_pull_request_files",),
        "Which files changed in pull request #{n} of {owner}/{ghrepo}?",
        {"owner": "{owner}", "repo": "{ghrepo}", "pull_number": "{n}"},
    ),
    (
        "github",
        ("create_branch",),
        "Create a branch {branch} in the {owner}/{ghrepo} GitHub repository.",
        {"owner": "{owner}", "repo": "{ghrepo}", "branch": "{branch}"},
    ),
    ("github", ("search_users",), "Find GitHub users matching {person}.", {"q": "{person}"}),
    (
        "memory",
        ("search_nodes",),
        "Search my knowledge graph for anything about {topic}.",
        {"query": "{topic}"},
    ),
    (
        "memory",
        ("open_nodes",),
        "Open the knowledge-graph entity {entity}.",
        {"names": ["{entity}"]},
    ),
    ("memory", ("read_graph",), "Dump my entire knowledge graph.", {}),
    (
        "memory",
        ("delete_entities",),
        "Delete the entity {entity} from my knowledge graph.",
        {"entityNames": ["{entity}"]},
    ),
    ("fetch", ("fetch",), "Fetch {url} and show me the page.", {"url": "{url}"}),
    (
        "fetch",
        ("fetch",),
        "Fetch the raw, unsimplified HTML of {url}.",
        {"url": "{url}", "raw": True},
    ),
    ("time", ("get_current_time",), "What time is it right now in {tz}?", {"timezone": "{tz}"}),
    (
        "time",
        ("convert_time",),
        "Convert {hhmm} in {tz} to Asia/Dubai time.",
        {"source_timezone": "{tz}", "time": "{hhmm}", "target_timezone": "Asia/Dubai"},
    ),
    (
        "everything",
        ("get-sum",),
        "Use the test server to add {a} and {b}.",
        {"a": "{a}", "b": "{b}"},
    ),
    ("everything", ("echo",), 'Echo back the message "{msg}".', {"message": "{msg}"}),
]

_SLOT_RE = re.compile(r"\{(\w+)\}")


@dataclass
class Task:
    id: str
    server: str
    tools: tuple[str, ...]
    prompt: str
    gold: dict[str, Any]


def _fill(tmpl: Any, slots: dict[str, Any]) -> Any:
    if isinstance(tmpl, str):
        whole = _SLOT_RE.fullmatch(tmpl)
        if whole:  # a lone slot keeps its type (an int stays an int)
            return slots[whole.group(1)]
        return _SLOT_RE.sub(lambda m: str(slots[m.group(1)]), tmpl)
    if isinstance(tmpl, list):
        return [_fill(x, slots) for x in tmpl]
    if isinstance(tmpl, dict):
        return {k: _fill(v, slots) for k, v in tmpl.items()}
    return tmpl


def generate_tasks(n: int = N_TOOL_TASKS, seed: int = 0) -> list[Task]:
    """Deterministic: the same (n, seed) always gives the same tasks, in order."""
    rng = random.Random(seed)
    out: list[Task] = []
    seen: set[str] = set()
    i = 0
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        server, tools, prompt, gold = TEMPLATES[i % len(TEMPLATES)]
        i += 1
        slots = {k: rng.choice(v) for k, v in _POOLS.items()}
        text = _fill(prompt, slots)
        if text in seen and len(seen) < len(TEMPLATES) * 200:
            continue
        seen.add(text)
        out.append(Task(f"t{len(out):04d}", server, tools, text, _fill(gold, slots)))
    return out


# -- result-comprehension tasks (R) ----------------------------------------


@dataclass
class ResultTask:
    id: str
    tool: str
    prompt: str
    result: str
    answer: str


def generate_result_tasks(n: int = N_RESULT_TASKS, seed: int = 0) -> list[ResultTask]:
    rng = random.Random(seed + 1)
    out: list[ResultTask] = []
    for i in range(n):
        kind = i % 3
        if kind == 0:  # directory listing with sizes: which file is largest?
            rows = [
                (f"file_{rng.randrange(10**6):06d}.dat", rng.randrange(1, 90_000))
                for _ in range(rng.randrange(150, 400))
            ]
            big = max(rows, key=lambda r: r[1])[0]
            pos = rng.randrange(20, len(rows) - 20)
            name = f"archive_{rng.randrange(10**6):06d}.bin"
            rows.insert(pos, (name, 250_000 + rng.randrange(10_000)))
            big = name
            text = "\n".join(f"[FILE] {n_:<20} {s:>9,} bytes" for n_, s in rows)
            out.append(
                ResultTask(
                    f"r{i:04d}",
                    "list_directory_with_sizes",
                    "Which file in this listing is the largest? Answer with the file name only.",
                    text,
                    big,
                )
            )
        elif kind == 1:  # a build log: what error code did it fail with?
            code = f"E{rng.randrange(1000, 9999)}"
            lines = [
                f"[{t:05d}] step {t % 17} ok: compiled module_{rng.randrange(999)}"
                for t in range(rng.randrange(200, 500))
            ]
            pos = rng.randrange(40, len(lines) - 40)
            lines.insert(
                pos, f"[{pos:05d}] linker failed with code {code} in module_{rng.randrange(999)}"
            )
            text = "\n".join(lines) + "\nBUILD FAILED"
            out.append(
                ResultTask(
                    f"r{i:04d}",
                    "git_log",
                    "What error code did the build fail with? Answer with the code only.",
                    text,
                    code,
                )
            )
        else:  # JSON records: what is the status of order X?
            ids = rng.sample(range(10_000, 99_999), rng.randrange(60, 160))
            statuses = ("shipped", "pending", "cancelled", "returned", "processing")
            records = [
                {
                    "order_id": o,
                    "status": rng.choice(statuses),
                    "total": round(rng.random() * 500, 2),
                    "region": rng.choice(("eu", "us", "apac")),
                }
                for o in ids
            ]
            target = rng.choice(records)
            text = json.dumps(records, indent=1)
            out.append(
                ResultTask(
                    f"r{i:04d}",
                    "search_issues",
                    f"What is the status of order {target['order_id']}? Answer with the status word only.",
                    text,
                    str(target["status"]),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return (
            a is b
            or (isinstance(a, str) and a.lower() == str(b).lower())
            or (isinstance(b, str) and b.lower() == str(a).lower())
        )
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b))
    if (
        isinstance(a, (int, float))
        and isinstance(b, str)
        or isinstance(b, (int, float))
        and isinstance(a, str)
    ):
        try:
            return math.isclose(float(a), float(b))
        except ValueError:
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(any(_eq(x, y) for y in b) for x in a)
    return bool(a == b)


def score(task: Task, called: str | None, args: Any) -> dict[str, int]:
    """Pre-registered metrics for one task (protocol §3). 1 = correct."""
    tool_ok = int(called in task.tools)
    args = args if isinstance(args, dict) else {}
    args_ok = int(tool_ok == 1 and all(k in args and _eq(v, args[k]) for k, v in task.gold.items()))
    return {"selection": tool_ok, "args": args_ok}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def required_n(
    p_discordant: float, margin: float = CERT_MARGIN, alpha: float = ALPHA, power: float = POWER
) -> int:
    """Paired binary non-inferiority sample size at a true difference of zero.

    n = (z_{1-α} + z_{power})² · p_d / margin²  (Connor 1987 / Nam 1997, the McNemar
    variance of correlated proportions at δ=0).
    """
    from statistics import NormalDist

    z = NormalDist().inv_cdf
    return math.ceil((z(1 - alpha) + z(power)) ** 2 * p_discordant / margin**2)


def compare(
    base: list[int],
    arm: list[int],
    *,
    margin: float = CERT_MARGIN,
    alpha: float = ALPHA,
    boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired non-inferiority of *arm* vs *base* (lists of 0/1, same tasks, same order)."""
    if len(base) != len(arm) or not base:
        raise ValueError("paired comparison needs two equal-length, non-empty lists")
    n = len(base)
    diffs = [float(a - b) for a, b in zip(arm, base)]
    t = tost(diffs, margin=margin, alpha=alpha)
    losses = sum(1 for a, b in zip(arm, base) if b == 1 and a == 0)
    gains = sum(1 for a, b in zip(arm, base) if b == 0 and a == 1)
    mc = mcnemar_noninferiority(losses, gains, n, margin=margin)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    lo = means[int(alpha * boot)]  # one-sided (1-α) lower bound
    ci = (means[int(0.025 * boot)], means[min(boot - 1, int(0.975 * boot))])
    return {
        "n": n,
        "base_rate": sum(base) / n,
        "arm_rate": sum(arm) / n,
        "mean_diff": t.mean_diff,
        "losses": losses,
        "gains": gains,
        "discordant_rate": (losses + gains) / n,
        "tost_p": t.p_non_inferior,
        "boot_lower": lo,
        "boot_ci95": list(ci),
        "mcnemar_ci95": [mc.ci95_low, mc.ci95_high],
        "margin": margin,
        "alpha": alpha,
        "non_inferior": bool(t.non_inferior and lo > -margin),
    }


def verdicts(
    arms: dict[str, dict[str, list[int]]], p_design: float = P_DISCORDANT
) -> dict[str, dict[str, Any]]:
    """Fixed-sequence certification (protocol §5): L0 → L1 → L2 → L3, stop at first failure.

    A level is ``certified`` iff BOTH primary metrics are non-inferior to ``raw``.
    ``inconclusive`` when it passed on point estimate but the observed discordance
    exceeded the design assumption (underpowered) — never promoted.
    """
    out: dict[str, dict[str, Any]] = {}
    stopped = False
    for lv in SEQUENCE:
        if lv not in arms:
            continue
        if stopped:
            out[lv] = {
                "status": "not-tested",
                "why": "an earlier level in the fixed sequence failed",
            }
            continue
        comps = {m: compare(arms["raw"][m], arms[lv][m]) for m in ("selection", "args")}
        ok = all(c["non_inferior"] for c in comps.values())
        underpowered = any(c["discordant_rate"] > p_design for c in comps.values())
        status = "certified" if ok else "failed"
        if ok and underpowered:
            status = "inconclusive"
        if status != "certified":
            stopped = True
        out[lv] = {"status": status, **comps}
    return out


# ---------------------------------------------------------------------------
# The model side: a scripted mock for dry runs, the Anthropic API for live runs
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One model decision: a tool call (``name``, ``args``) or a final text answer."""

    name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)


Model = Callable[[list[dict[str, Any]], list[dict[str, Any]], Any], Turn]


def _gold_name(tools: list[dict[str, Any]], task: Task) -> str | None:
    names = {t["name"] for t in tools}
    for n in task.tools:
        if n in names:
            return n
    return None


def scripted(
    behavior: str = "oracle",
    seed: int = 0,
    error_rate: float = 0.02,
    degrade: dict[str, float] | None = None,
) -> Callable[..., Turn]:
    """A deterministic mock model. ``arm`` and ``task`` are passed so errors can be paired.

    * ``oracle``    — always right; fetches a schema first when the tool is not listed.
    * ``invoke``    — oracle, but uses ``<server>_invoke_tool`` after fetching (a
      client that never refreshes its tool list).
    * ``noisy``     — oracle with an independent ``error_rate`` chance per (task, arm)
      of calling a plausible wrong tool: realistic discordance, no true difference.
    * ``degraded``  — oracle, except ``degrade[arm]`` error on the named arms: a true
      loss the statistics must catch.
    * ``no-expand`` — result tasks: answers from visible text only, never expands.
    """
    degrade = degrade or {}

    def model(tools: list[dict[str, Any]], history: list[dict[str, Any]], ctx: Any) -> Turn:
        arm, task = ctx
        if isinstance(task, ResultTask):
            visible = "\n".join(str(h.get("text", "")) for h in history if h.get("role") == "tool")
            if task.answer in visible:
                return Turn(text=task.answer)
            handle = re.search(r"handle=([0-9a-f]{8})", visible)
            expand = next((t["name"] for t in tools if t["name"].endswith("_expand")), None)
            already = any(
                h.get("name", "").endswith("_expand") for h in history if h.get("role") == "call"
            )
            if behavior != "no-expand" and handle and expand and not already:
                return Turn(name=expand, args={"handle": handle.group(1)})
            return Turn(text="unknown")
        rng = random.Random(f"{seed}:{task.id}:{arm}")
        rate = (error_rate if behavior == "noisy" else 0.0) + (
            degrade.get(arm, 0.0) if behavior == "degraded" else 0.0
        )
        wrong = rng.random() < rate
        target = task.tools[0]
        if wrong:
            same = [
                n for t2 in TEMPLATES if t2[0] == task.server for n in t2[1] if n not in task.tools
            ]
            target = rng.choice(same) if same else target
        fetched = [
            h for h in history if h.get("role") == "call" and h["name"].endswith("_get_tool_schema")
        ]
        invoke = levels.meta_name(task.server, "invoke_tool")
        if fetched and behavior == "invoke" and any(t["name"] == invoke for t in tools):
            return Turn(name=invoke, args={"tool_name": target, "arguments": dict(task.gold)})
        if any(t["name"] == target for t in tools):
            return Turn(name=target, args=dict(task.gold))
        if not wrong and _gold_name(tools, task) is not None:
            return Turn(name=_gold_name(tools, task), args=dict(task.gold))
        schema_tool = levels.meta_name(task.server, "get_tool_schema")
        if not fetched and any(t["name"] == schema_tool for t in tools):
            return Turn(name=schema_tool, args={"tool_name": target})
        if fetched and any(t["name"] == invoke for t in tools):
            return Turn(name=invoke, args={"tool_name": target, "arguments": dict(task.gold)})
        return Turn(text="I cannot find a suitable tool.")

    return model


class AnthropicModel:
    """The live model: Anthropic Messages API over stdlib HTTP, with a hard spend cap."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        model: str,
        budget_usd: float,
        api_key: str | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.price = pricing.get(model)
        self.budget = budget_usd
        self.spent = 0.0
        self.api_key: str = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — a live run needs it")
        self._open = opener or urllib.request.urlopen

    def __call__(
        self, tools: list[dict[str, Any]], history: list[dict[str, Any]], ctx: Any
    ) -> Turn:
        if self.spent >= self.budget:
            raise BudgetExceeded(f"spent ${self.spent:.2f} of ${self.budget:.2f}")
        api_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema") or {"type": "object"},
            }
            for t in tools
        ]
        if api_tools:
            api_tools[-1] = {**api_tools[-1], "cache_control": {"type": "ephemeral"}}
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": SYSTEM,
            "tools": api_tools,
            "messages": _to_messages(history),
        }
        data = self._post(body)
        u = data.get("usage") or {}
        p = self.price
        self.spent += (
            u.get("input_tokens", 0) * p.input
            + u.get("cache_creation_input_tokens", 0) * p.cache_write
            + u.get("cache_read_input_tokens", 0) * p.cache_read
            + u.get("output_tokens", 0) * p.output
        )
        for block in data.get("content") or []:
            if block.get("type") == "tool_use":
                return Turn(name=block.get("name"), args=block.get("input") or {}, usage=u)
        text = "".join(
            b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text"
        )
        return Turn(text=text, usage=u)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode(),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        for attempt in range(4):
            try:
                with self._open(req, timeout=120) as resp:
                    parsed: dict[str, Any] = json.loads(resp.read())
                    return parsed
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 529) or attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")  # pragma: no cover


class BudgetExceeded(RuntimeError):
    pass


def _to_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Harness history → Anthropic messages (tool_use / tool_result pairs)."""
    msgs: list[dict[str, Any]] = []
    for i, h in enumerate(history):
        if h["role"] == "user":
            msgs.append({"role": "user", "content": h["text"]})
        elif h["role"] == "call":
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"tu{i}", "name": h["name"], "input": h["args"]}
                    ],
                }
            )
        elif h["role"] == "tool":
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"tu{i - 1}", "content": h["text"]}
                    ],
                }
            )
    return msgs


# ---------------------------------------------------------------------------
# Running an arm
# ---------------------------------------------------------------------------


class _InProc:
    """A fixture MCP server living in this process (fakeserver.handle)."""

    def __init__(
        self, name: str, fixture: dict[str, Any], results: dict[str, str] | None = None
    ) -> None:
        self.name = name
        self.fixture = fixture
        self.results = results or {}
        self.on_message: Callable[[dict[str, Any]], None] = lambda msg: None

    def request(self, method: str, params: Any = None, *, id: Any = None) -> dict[str, Any]:
        if (
            method == "tools/call"
            and isinstance(params, dict)
            and params.get("name") in self.results
        ):
            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {"content": [{"type": "text", "text": self.results[params["name"]]}]},
            }
        out = fakeserver.handle(
            self.fixture,
            {
                "jsonrpc": "2.0",
                "id": id if id is not None else 1,
                "method": method,
                "params": params,
            },
        )
        return out or {"jsonrpc": "2.0", "id": id, "result": {}}

    def notify(self, method: str, params: Any = None) -> None:
        return None

    def send(self, msg: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


def fixtures() -> dict[str, dict[str, Any]]:
    return {n: fakeserver.load_fixture(n) for n in fakeserver.fixture_names()}


def usage_profile(tasks: list[Task]) -> dict[str, dict[str, int]]:
    """L3's learned usage, pre-registered: gold-tool counts of the tasks' first half.

    Only the first half informs the pins, and every task is scored, so the pins are
    fixed before the run and cannot be tuned on outcomes.
    """
    prof: dict[str, dict[str, int]] = {}
    for t in tasks[: len(tasks) // 2]:
        s = prof.setdefault(t.server, {})
        s[t.tools[0]] = s.get(t.tools[0], 0) + 1
    return prof


def _session(
    arm: str,
    fx: dict[str, dict[str, Any]],
    state_root: Path,
    results: bool = False,
    raw_results: dict[str, str] | None = None,
) -> proxy.Proxy | None:
    if arm == "raw":
        return None
    backends: dict[str, proxy.Backend] = {n: _InProc(n, f, raw_results) for n, f in fx.items()}
    return proxy.Proxy(
        backends,
        level=arm,
        results=results,
        log=events.NullLog(),
        state_root=state_root,
        persist=False,
        record=lambda h, t: True,
    )


def _list(px: proxy.Proxy | None, fx: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if px is None:
        return [t for f in fx.values() for t in f["tools"]]
    return list(
        px.handle({"jsonrpc": "2.0", "id": "l", "method": "tools/list"})[0]["result"]["tools"]
    )


def _tool_tokens(tools: list[dict[str, Any]]) -> int:
    return sum(levels.definition_tokens(t) for t in tools)


def run_tool_arm(
    arm: str, tasks: list[Task], model: Model, fx: dict[str, dict[str, Any]], state_root: Path
) -> dict[str, Any]:
    """Run every task through one arm. Returns per-task metrics and token counts."""
    per: dict[str, list[int]] = {"selection": [], "args": [], "success": [], "round_trips": []}
    in_tok: list[int] = []
    for task in tasks:
        px = _session(arm, fx, state_root)
        tools = _list(px, fx)
        history: list[dict[str, Any]] = [{"role": "user", "text": task.prompt}]
        called, args, trips, tok = None, {}, 0, 0
        for _ in range(MAX_TURNS):
            tok += (
                TOOL_SYSTEM_TOKENS
                + _tool_tokens(tools)
                + sum(levels.tokens(json.dumps(h)) for h in history)
            )
            turn = model(tools, history, (arm, task))
            if turn.name is None:
                break
            trips += 1
            if px is None:
                called, args = turn.name, turn.args
                break
            kind, backend_tool = "unknown", None
            for surf in px._surfaces().values():
                kind, backend_tool = surf.resolve(turn.name)
                if kind != "unknown":
                    break
            # Every call goes through the real proxy, final ones included.
            out = px.handle(
                {
                    "jsonrpc": "2.0",
                    "id": trips,
                    "method": "tools/call",
                    "params": {"name": turn.name, "arguments": turn.args},
                }
            )
            if kind == "schema":
                history += [
                    {"role": "call", "name": turn.name, "args": turn.args},
                    {"role": "tool", "text": out[0]["result"]["content"][0]["text"]},
                ]
                if any(m.get("method") == "notifications/tools/list_changed" for m in out[1:]):
                    tools = _list(px, fx)
                continue
            if kind == "tool":
                called, args = backend_tool, turn.args
            elif kind == "invoke":
                called, args = turn.args.get("tool_name"), turn.args.get("arguments") or {}
            else:
                called, args = turn.name, turn.args  # unknown: scored as a wrong tool
            break
        s = score(task, called, args)
        valid = int(s["args"] == 1 and _schema_ok(fx[task.server], called, args))
        per["selection"].append(s["selection"])
        per["args"].append(s["args"])
        per["success"].append(valid)
        per["round_trips"].append(trips)
        in_tok.append(tok)
    return {**per, "input_tokens": in_tok}


def _schema_ok(fixture: dict[str, Any], tool: str | None, args: Any) -> bool:
    """Secondary 'task success': required keys present, no key the schema does not declare."""
    t = next((x for x in fixture["tools"] if x["name"] == tool), None)
    if t is None or not isinstance(args, dict):
        return False
    schema = t.get("inputSchema") or {}
    props = schema.get("properties") or {}
    extra_ok = schema.get("additionalProperties", True) is not False
    return all(k in args for k in schema.get("required") or []) and (
        extra_ok or all(k in props for k in args)
    )


def run_result_arm(arm: str, tasks: list[ResultTask], model: Model) -> dict[str, Any]:
    """``raw`` vs ``R``: the model sees a tool result and must answer from it."""
    correct: list[int] = []
    expands: list[int] = []
    in_tok: list[int] = []
    for task in tasks:
        text = task.result
        tools: list[dict[str, Any]] = []
        store: dict[str, str] = {}
        if arm == "R":
            fake = {"content": [{"type": "text", "text": text}]}
            new, _info = levels.compress_result(
                task.tool, fake, record=lambda h, t: store.setdefault(h, t) == t
            )
            text = new["content"][0]["text"]
            tools = [t for t in levels.Surface("bench", [], "L0", True).meta_tools()]
        history: list[dict[str, Any]] = [
            {"role": "user", "text": task.prompt},
            {"role": "call", "name": task.tool, "args": {}},
            {"role": "tool", "text": text},
        ]
        answer, n_exp, tok = "", 0, 0
        for _ in range(MAX_TURNS):
            tok += (
                TOOL_SYSTEM_TOKENS
                + _tool_tokens(tools)
                + sum(levels.tokens(json.dumps(h)) for h in history)
            )
            turn = model(tools, history, (arm, task))
            if turn.name is None:
                answer = turn.text
                break
            n_exp += 1
            original = store.get(str(turn.args.get("handle")))
            history += [
                {"role": "call", "name": turn.name, "args": turn.args},
                {"role": "tool", "text": original or "error: no original found"},
            ]
        correct.append(int(task.answer.lower() in answer.lower()))
        expands.append(n_exp)
        in_tok.append(tok)
    return {"correct": correct, "expands": expands, "input_tokens": in_tok}


# ---------------------------------------------------------------------------
# Cost estimate + the whole run
# ---------------------------------------------------------------------------


def definition_table() -> dict[str, Any]:
    """Model-facing definition tokens per fixture server, per level (deterministic).

    Lazy levels are measured at session start (nothing unlocked), which is the list
    the first request carries; ``requested`` != ``level`` where reject-if-bigger fell
    back to L0.
    """
    rows: dict[str, Any] = {}
    for name, fx in fixtures().items():
        tools = fx["tools"]
        row: dict[str, Any] = {"tools": len(tools), "raw": _tool_tokens(tools)}
        for lv in SEQUENCE:
            surf = levels.Surface(name, tools, lv, results=False)
            row[lv] = {"tokens": _tool_tokens(surf.tools_list()), "level": surf.level}
        rows[name] = row
    total = {"raw": sum(r["raw"] for r in rows.values())}
    for lv in SEQUENCE:
        total[lv] = sum(r[lv]["tokens"] for r in rows.values())
    pct = {lv: round(100.0 * (1 - total[lv] / total["raw"]), 1) for lv in SEQUENCE}
    return {
        "tokenizer": "distil heuristic (calibrate against a provider count)",
        "note": "lazy levels at session start; L3 equals L2 until usage has been learned",
        "servers": rows,
        "total": total,
        "pct_smaller_than_raw": pct,
    }


def estimate_cost(
    tool_runs: dict[str, dict[str, Any]],
    result_runs: dict[str, dict[str, Any]],
    models: tuple[str, ...] = LIVE_MODELS,
) -> dict[str, Any]:
    """Dollars for the live run, per model, from the oracle dry run's exact turn structure.

    ``no_cache`` bills every input token at the base rate (the upper bound);
    ``cached`` bills each arm's tool list as one cache write, then cache reads (the
    list is identical across tasks within an arm, and the live client marks it).
    Both carry ``ESTIMATE_SAFETY`` for tokenizer error.
    """
    out: dict[str, Any] = {
        "assumptions": {
            "output_tokens_per_turn": EST_OUTPUT_TOKENS,
            "tool_system_tokens": TOOL_SYSTEM_TOKENS,
            "safety": ESTIMATE_SAFETY,
        },
        "models": {},
    }
    for m in models:
        p = pricing.get(m)
        rows: dict[str, Any] = {}
        total_nc = total_c = 0.0
        for arm, run in {
            **tool_runs,
            **{f"results:{k}": v for k, v in result_runs.items()},
        }.items():
            inp = sum(run["input_tokens"])
            turns = sum(
                max(1, t) for t in run.get("round_trips", [1] * len(run["input_tokens"]))
            ) + (sum(run.get("expands", [])) if "expands" in run else 0)
            outp = turns * EST_OUTPUT_TOKENS
            nc = (inp * p.input + outp * p.output) * ESTIMATE_SAFETY
            tools_tok = run.get("tools_tokens", 0)
            cached_part = tools_tok * max(0, turns - 1)
            c = (
                (inp - cached_part) * p.input
                + tools_tok * p.cache_write
                + cached_part * p.cache_read
                + outp * p.output
            ) * ESTIMATE_SAFETY
            rows[arm] = {
                "input_tokens": inp,
                "output_tokens": outp,
                "usd_no_cache": round(nc, 2),
                "usd_cached": round(c, 2),
            }
            total_nc += nc
            total_c += c
        out["models"][m] = {
            "arms": rows,
            "usd_no_cache": round(total_nc, 2),
            "usd_cached": round(total_c, 2),
            "price_per_mtok": {"input": p.input_per_mtok, "output": p.output_per_mtok},
        }
    return out


def run(
    *,
    model: Model,
    n_tools: int = N_TOOL_TASKS,
    n_results: int = N_RESULT_TASKS,
    arms: tuple[str, ...] = ("raw", *SEQUENCE),
    seed: int = 0,
) -> dict[str, Any]:
    tasks = generate_tasks(n_tools, seed)
    rtasks = generate_result_tasks(n_results, seed)
    fx = fixtures()
    tool_runs: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="distil-mcp-bench-") as tmp:
        root = Path(tmp)
        for server, counts in usage_profile(tasks).items():
            _seed_usage(root, server, counts)
        for arm in arms:
            px = _session(arm, fx, root)
            listed = _list(px, fx)
            run_ = run_tool_arm(arm, tasks, model, fx, root)
            run_["tools_tokens"] = _tool_tokens(listed)
            tool_runs[arm] = run_
    result_runs = {arm: run_result_arm(arm, rtasks, model) for arm in ("raw", "R")}
    summary: dict[str, Any] = {"arms": {}, "results": {}}
    for arm, r in tool_runs.items():
        n = len(r["selection"])
        summary["arms"][arm] = {
            "n": n,
            "selection": sum(r["selection"]) / n,
            "args": sum(r["args"]) / n,
            "success": sum(r["success"]) / n,
            "mean_round_trips": sum(r["round_trips"]) / n,
            "extra_round_trips": sum(max(0, t - 1) for t in r["round_trips"]) / n,
            "tools_tokens": r["tools_tokens"],
            "mean_input_tokens": sum(r["input_tokens"]) / n,
        }
    for arm, r in result_runs.items():
        n = len(r["correct"])
        summary["results"][arm] = {
            "n": n,
            "correct": sum(r["correct"]) / n,
            "mean_expands": sum(r["expands"]) / n,
            "mean_input_tokens": sum(r["input_tokens"]) / n,
        }
    cert = (
        verdicts({a: {m: tool_runs[a][m] for m in ("selection", "args")} for a in tool_runs})
        if "raw" in tool_runs
        else {}
    )
    r_comp = compare(result_runs["raw"]["correct"], result_runs["R"]["correct"])
    r_ok = r_comp["non_inferior"] and r_comp["discordant_rate"] <= P_DISCORDANT_R
    cert["R"] = {
        "status": "certified" if r_ok else ("inconclusive" if r_comp["non_inferior"] else "failed"),
        "correct": r_comp,
    }
    return {
        "protocol": "docs/research/mcp-compressor-protocol.md",
        "protocol_version": 1,
        "seed": seed,
        "n_tool_tasks": len(tasks),
        "n_result_tasks": len(rtasks),
        "margin": CERT_MARGIN,
        "alpha": ALPHA,
        "summary": summary,
        # Verdicts, not a certificate: distil/certificates/mcp.json is only ever edited by a
        # human, from a committed live run. A dry run's verdicts describe the mock model.
        "verdicts": cert,
        "_runs": {"tools": tool_runs, "results": result_runs},
    }


def _seed_usage(root: Path, server: str, counts: dict[str, int]) -> None:
    events._write_json(
        events.ServerState(server, root).path, {"usage": dict(counts), "sessions": {}}
    )

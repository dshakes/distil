"""Capture REAL command output for the `distil sh` filters (ADR 0026), then replay it.

    uv run --python 3.12 python benchmarks/shell_replay.py capture   # rewrites tests/fixtures/shell/
    uv run --python 3.12 python benchmarks/shell_replay.py replay    # writes the replay JSON
    uv run --python 3.12 python benchmarks/shell_replay.py swebench DIR  # real agent outputs

`capture` builds throwaway projects in a temp dir (a git repo with staged, modified and
untracked files; a Python package with 60 tests, two failing and one skipped; the same
shape as a Rust crate, a Go module and a node:test file) and runs the real tools on
them, plus one real all-passing `pytest -v` over a slice of this repo's own suite.
Paths and timings are normalised so a re-capture diffs cleanly. Every fixture is real
tool output; the projects themselves are synthetic, so the replay is a per-command
ratio, not a traffic-weighted estimate.

`replay` runs every fixture through `distil.shell.shape` in both tiers and records
heuristic-token reductions per fixture and in aggregate.

`swebench DIR` replays real agent traffic: every bash tool call in the SWE-bench outcome
transcripts under DIR (one JSON message list per task, as `swebench_outcome.run` writes
them), with the output the agent actually got. Each command goes through the same
`distil.shell.plan` the PreToolUse hook uses; matched ones are shaped as `distil sh` would
shape them. It reports the share of all bash-output tokens removed, which is the number
that bounds what the rewrite can do on that workload.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "shell"
OUT = ROOT / "benchmarks" / "results" / "2026-10-05" / "shell_replay.json"

#: fixture file -> the command line it was captured from (the filter is chosen from it).
COMMANDS = {
    "git_status.txt": "git status",
    "pytest_default.txt": "python -m pytest -p no:cacheprovider",
    "pytest_verbose.txt": "python -m pytest -v -p no:cacheprovider",
    "pytest_verbose_color.txt": "python -m pytest -v --color=yes -p no:cacheprovider",
    "pytest_distil_suite_verbose.txt": "python -m pytest -v -p no:cacheprovider tests/test_keep_policy.py tests/test_keeptags.py",
    "unittest_verbose.txt": "python -m unittest -v",
    "cargo_test.txt": "cargo test",
    "go_test_verbose.txt": "go test -v ./...",
    "node_test.txt": "npm test",
}

_PY_TESTS = "\n".join(
    [
        "import pytest, unittest",
        *(f"def test_case_{i:02d}():\n    assert {i} + 1 == {i + 1}\n" for i in range(57)),
        "def test_parses_header():\n    assert 'a,b'.split(',') == ['a', 'b', 'c']\n",
        "def test_raises_on_empty():\n    raise ValueError('empty input: expected at least one row')\n",
        "@pytest.mark.skip(reason='needs network')\ndef test_fetch():\n    pass\n",
    ]
)
_UNITTEST = "\n".join(
    [
        "import unittest",
        "class Case(unittest.TestCase):",
        *(
            f"    def test_case_{i:02d}(self):\n        self.assertEqual({i} + 1, {i + 1})"
            for i in range(57)
        ),
        "    def test_parses_header(self):\n        self.assertEqual('a,b'.split(','), ['a', 'b', 'c'])",
        "    def test_raises_on_empty(self):\n        raise ValueError('empty input: expected at least one row')",
    ]
)
_RUST = "\n".join(
    [
        "#[cfg(test)]\nmod tests {",
        *(
            f"    #[test]\n    fn case_{i:02d}() {{ assert_eq!({i} + 1, {i + 1}); }}"
            for i in range(57)
        ),
        "    #[test]\n    fn parses_header() { assert_eq!(\"a,b\".split(',').count(), 3); }",
        '    #[test]\n    fn raises_on_empty() { panic!("empty input: expected at least one row"); }',
        "}",
    ]
)
_GO = "\n".join(
    [
        'package demo\n\nimport "testing"\n',
        *(
            f'func TestCase{i:02d}(t *testing.T) {{ if {i}+1 != {i + 1} {{ t.Fatal("bad") }} }}'
            for i in range(57)
        ),
        'func TestParsesHeader(t *testing.T) { t.Fatalf("got %d fields, want 3", 2) }',
        'func TestRaisesOnEmpty(t *testing.T) { t.Error("empty input: expected at least one row") }',
    ]
)
_NODE = "\n".join(
    [
        "const test = require('node:test');\nconst assert = require('node:assert');",
        *(f"test('case {i:02d}', () => assert.strictEqual({i} + 1, {i + 1}));" for i in range(57)),
        "test('parses header', () => assert.deepStrictEqual('a,b'.split(','), ['a', 'b', 'c']));",
        "test('raises on empty', () => { throw new Error('empty input: expected at least one row'); });",
    ]
)


def _run(cmd: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    r = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, errors="replace", env=env
    )
    return r.stdout + r.stderr


def _norm(text: str, tmp: Path) -> str:
    text = text.replace(str(tmp), "/tmp/proj").replace(str(ROOT), "/repo")
    text = re.sub(r"\b\d+\.\d+s\b", "0.01s", text)  # timings
    text = re.sub(r"\(\d+(?:\.\d+)?m?s\)", "(0.01s)", text)
    text = re.sub(r"duration_ms [\d.]+", "duration_ms 1", text)
    text = re.sub(
        r"platform \S+ -- Python [\d.]+, pytest-[\d.]+, pluggy-[\d.]+",
        "platform x -- Python 3.x, pytest-x, pluggy-x",
        text,
    )
    text = re.sub(r"(pluggy-x) -- \S+", r"\1 -- python", text)
    text = re.sub(r"rootdir: .*", "rootdir: /tmp/proj", text)
    text = re.sub(r"plugins: .*\n", "", text)
    return text


def capture() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d).resolve()
        git = tmp / "repo"
        git.mkdir()
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(tmp),
            "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        }
        _run(
            "git init -q -b main && touch a.py b.py c.py && git add . && git commit -qm init",
            git,
            env,
        )
        (git / "a.py").write_text("x = 1\n")
        (git / "b.py").write_text("y = 2\n")
        _run("git add b.py && git rm -q --cached c.py", git, env)
        for n in ("new_feature.py", "notes.md", "scratch/"):
            p = git / n
            p.mkdir() if n.endswith("/") else p.write_text("z\n")
        (git / "scratch" / "x.txt").write_text("t\n")
        (FIX / "git_status.txt").write_text(_norm(_run("git status", git, env), tmp))

        py = tmp / "pyproj"
        py.mkdir()
        (py / "test_demo.py").write_text(_PY_TESTS)
        for name in ("pytest_default.txt", "pytest_verbose.txt", "pytest_verbose_color.txt"):
            cmd = COMMANDS[name].replace("python", sys.executable, 1)
            (FIX / name).write_text(_norm(_run(cmd, py), tmp))
        cmd = COMMANDS["pytest_distil_suite_verbose.txt"].replace("python", sys.executable, 1)
        (FIX / "pytest_distil_suite_verbose.txt").write_text(_norm(_run(cmd, ROOT), tmp))
        ut = tmp / "utproj"
        ut.mkdir()
        (ut / "test_demo.py").write_text(_UNITTEST)
        (FIX / "unittest_verbose.txt").write_text(
            _norm(_run(f"{sys.executable} -m unittest -v", ut), tmp)
        )

        if shutil.which("cargo"):
            rs = tmp / "rsproj"
            _run("cargo new -q --lib rsproj", tmp)
            (rs / "src" / "lib.rs").write_text(_RUST)
            _run("cargo build -q --tests", rs)
            (FIX / "cargo_test.txt").write_text(_norm(_run("cargo test", rs), tmp))
        if shutil.which("go"):
            go = tmp / "goproj"
            go.mkdir()
            (go / "go.mod").write_text("module demo\n\ngo 1.21\n")
            (go / "demo_test.go").write_text(_GO)
            (FIX / "go_test_verbose.txt").write_text(_norm(_run("go test -v ./...", go), tmp))
        if shutil.which("node"):
            nd = tmp / "nodeproj"
            nd.mkdir()
            (nd / "demo.test.js").write_text(_NODE)
            out = _run("node --test --test-reporter=spec demo.test.js", nd)
            (FIX / "node_test.txt").write_text(_norm(out, tmp))
    print(f"captured {len(list(FIX.glob('*.txt')))} fixtures in {FIX}")


def replay() -> None:
    sys.path.insert(0, str(ROOT))
    from distil.shell import FILTERS_VERSION, classify, shape
    from distil.tokenizer import resolve

    tok = resolve("heuristic")
    rows, tot = [], {"raw": 0, "lossless": 0, "elide": 0}
    for name, cmd in COMMANDS.items():
        p = FIX / name
        if not p.exists():
            continue
        raw = p.read_text()
        import shlex

        kind = classify(shlex.split(cmd))
        assert kind is not None, cmd
        ll, _ = shape(raw, kind, lossy=False, save=lambda t: "0" * 8)
        el, tier = shape(raw, kind, lossy=True, save=lambda t: "0" * 8)
        r = {
            "fixture": name,
            "command": cmd,
            "filter": kind,
            "tier_applied": tier,
            "raw": tok.count(raw),
            "lossless": tok.count(ll),
            "elide": tok.count(el),
        }
        rows.append(r)
        for k in tot:
            tot[k] += r[k]
    out = {
        "filters": FILTERS_VERSION,
        "tokenizer": "heuristic",
        "method": "each fixture (real tool output over a synthetic project, see "
        "benchmarks/shell_replay.py) through distil.shell.shape; 'lossless' is the "
        "subscription default, 'elide' the metered-key default (handle stubbed)",
        "corpus_limits": "9 fixtures from synthetic projects (60-test suites with 2 failures) "
        "plus one real all-passing slice of this repo's suite; ratios are per command, NOT "
        "weighted by any real agent's command mix",
        "fixtures": rows,
        "aggregate": {
            "raw_tokens": tot["raw"],
            "lossless_reduction_pct": round(100 * (1 - tot["lossless"] / tot["raw"]), 1),
            "elide_reduction_pct": round(100 * (1 - tot["elide"] / tot["raw"]), 1),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["aggregate"]))


_EXIT = re.compile(r"\n\[exit \d+\]$")


def swebench(src: str) -> None:
    sys.path.insert(0, str(ROOT))
    from collections import Counter

    from distil.shell import FILTERS_VERSION, plan, shape
    from distil.tokenizer import resolve

    tok = resolve("heuristic")
    files = sorted(Path(src).glob("*.json"))
    calls = matched = 0
    tot = {"raw": 0, "matched_raw": 0, "lossless": 0, "elide": 0}
    kinds: Counter[str] = Counter()
    family: Counter[str] = Counter()
    for f in files:
        msgs = json.loads(f.read_text())
        cmds: dict[str, str] = {}
        for m in msgs:
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if b.get("type") == "tool_use" and b.get("name") == "bash":
                    cmds[b["id"]] = str((b.get("input") or {}).get("command", ""))
                if b.get("type") == "tool_result" and b.get("tool_use_id") in cmds:
                    out = b.get("content")
                    if isinstance(out, list):
                        out = "".join(x.get("text", "") for x in out if isinstance(x, dict))
                    out = str(out or "")
                    calls += 1
                    n = tok.count(out)
                    tot["raw"] += n
                    words = re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", cmds[b["tool_use_id"]]).split()
                    family[words[0] if words else "?"] += n
                    p = plan(cmds[b["tool_use_id"]])
                    if p is None:
                        continue
                    matched += 1
                    kinds[p[3]] += 1
                    m2 = _EXIT.search(out)
                    body, tail = (out[: m2.start()], out[m2.start() :]) if m2 else (out, "")
                    ll, _ = shape(body, p[3], lossy=False, save=lambda t: "0" * 8)
                    el, _ = shape(body, p[3], lossy=True, save=lambda t: "0" * 8)
                    tot["matched_raw"] += n
                    tot["lossless"] += tok.count(ll + tail)
                    tot["elide"] += tok.count(el + tail)
    saved_ll = tot["matched_raw"] - tot["lossless"]
    saved_el = tot["matched_raw"] - tot["elide"]
    out_path = OUT.with_name("shell_replay_swebench.json")
    res = {
        "filters": FILTERS_VERSION,
        "tokenizer": "heuristic",
        "source": f"{len(files)} SWE-bench Lite transcripts, `provider-cm` arm of the 300-task "
        "head-to-head run (tool outputs are what the container returned; context editing acts "
        "server-side). Transcripts are not committed; this file is the aggregate.",
        "bash_calls": calls,
        "matched_calls": matched,
        "matched_by_filter": dict(kinds.most_common()),
        "bash_output_tokens": tot["raw"],
        "matched_output_tokens": tot["matched_raw"],
        "lossless_saved_pct_of_all_bash_output": round(100 * saved_ll / max(tot["raw"], 1), 2),
        "elide_saved_pct_of_all_bash_output": round(100 * saved_el / max(tot["raw"], 1), 2),
        "elide_saved_pct_of_matched_output": round(100 * saved_el / max(tot["matched_raw"], 1), 1),
        "output_token_share_by_first_word": {
            w: round(100 * n / max(tot["raw"], 1), 1) for w, n in family.most_common(8)
        },
    }
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "swebench":
        swebench(sys.argv[2])
    else:
        {"capture": capture, "replay": replay}[sys.argv[1] if len(sys.argv) > 1 else "replay"]()

"""benchmarks/session_replay.py: parsing variants, bucketing, privacy, determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks import session_replay as sr

# Strings that stand in for private transcript content. None may reach any output.
SECRETS = (
    "SECRET-PROJECT-NAME",
    "/Users/zed/private/path.py",
    "super-secret-command-xyz",
    "SESSION-ID-7f3a",
    "confidential prose sentinel",
    "mcp__private_server__tool",
)


def _line(role: str, content: Any, **extra: Any) -> str:
    rec = {"type": role, "sessionId": SECRETS[3], "message": {"role": role, "content": content}}
    return json.dumps({**rec, **extra})


def _assistant_tool(tid: str, name: str, command: str, mid: str) -> str:
    block = {"type": "tool_use", "id": tid, "name": name, "input": {"command": command}}
    return _line(
        "assistant", [block], **{"message": {"role": "assistant", "id": mid, "content": [block]}}
    )


def _result(tid: str, text: str) -> str:
    return _line("user", [{"type": "tool_result", "tool_use_id": tid, "content": text}])


def _big_log(n: int = 400) -> str:
    return "\n".join(f"2024-01-01 INFO worker-{i % 7} processed batch {i} ok" for i in range(n))


def make_session(n_calls: int, *, command: str = "pytest -q") -> list[str]:
    lines = [_line("user", f"please help {SECRETS[4]} in {SECRETS[1]}")]
    for i in range(n_calls):
        lines.append(_assistant_tool(f"t{i}", "Bash", command, f"m{i}"))
        lines.append(_result(f"t{i}", _big_log() + f"\nrun {i}"))
    lines.append(_line("assistant", [{"type": "text", "text": "done"}]))
    return lines


# ----------------------------------------------------------------- parsing


def test_parse_merges_streamed_assistant_blocks_and_parallel_results() -> None:
    a1 = {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}
    a2 = {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "pwd"}}
    lines = [
        _line("user", "hi"),
        json.dumps(
            {"type": "assistant", "message": {"role": "assistant", "id": "m1", "content": [a1]}}
        ),
        json.dumps(
            {"type": "assistant", "message": {"role": "assistant", "id": "m1", "content": [a2]}}
        ),
        _result("a", "x"),
        _result("b", "y"),
    ]
    sess = sr.parse_transcript(lines)
    assert [m["role"] for m in sess.messages] == ["user", "assistant", "user"]
    assert len(sess.messages[1]["content"]) == 2
    assert len(sess.messages[2]["content"]) == 2
    assert sr.request_indices(sess.messages) == [0, 2]


def test_parse_counts_bad_lines_and_ignores_other_records() -> None:
    lines = [
        "{not json",
        "[]",
        json.dumps({"type": "summary", "summary": "x"}),
        json.dumps({"type": "user", "message": {"role": "robot", "content": "x"}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": 5}}),
        json.dumps(
            {"type": "user", "isSidechain": True, "message": {"role": "user", "content": "s"}}
        ),
        json.dumps(
            {
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {"role": "assistant", "content": "e"},
            }
        ),
        "",
        _line("user", "ok"),
    ]
    sess = sr.parse_transcript(lines)
    assert sess.bad_lines == 4
    assert len(sess.messages) == 1


def test_cache_control_recorded_vs_placed() -> None:
    marked = {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}
    assert sr.parse_transcript([_line("user", [marked])]).cache_recorded
    plain = sr.parse_transcript([_line("user", "a"), _line("assistant", "b"), _line("user", "c")])
    assert not plain.cache_recorded
    req = sr.with_breakpoint(plain.messages[:3], False)
    assert req[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in plain.messages[2]["content"][-1]  # input not mutated
    assert sr.with_breakpoint(plain.messages[:3], True)[-1] == plain.messages[2]


def test_discover_skips_subagents_and_is_sorted(tmp_path: Path) -> None:
    for rel in ("b/x.jsonl", "a/y.jsonl", "a/subagents/z.jsonl", "a/notes.txt"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
    assert [p.name for p in sr.discover(tmp_path)] == ["y.jsonl", "x.jsonl"]


# ----------------------------------------------------------------- classification / planning


@pytest.mark.parametrize(
    ("cmd", "cls"),
    [
        ("grep -rn foo src | head", "search"),
        ("cat src/a.py", "whole_file_read"),
        ("git status", "vcs"),
        ("cd x && pytest -q", "build_test"),
        ("ls -la", "listing"),
        ("curl https://x", "network_infra"),
        (f"./{SECRETS[2]} --go", "other"),
        ("", "other"),
    ],
)
def test_shell_class_is_fixed_vocabulary(cmd: str, cls: str) -> None:
    assert sr.shell_class(cmd) == cls


def test_exemption_switch_restores_state() -> None:
    assert sr.provenance.is_shell_search("grep -rn x .")
    with sr.shell_search_exemption(False):
        assert not sr.provenance.is_shell_search("grep -rn x .")
    assert sr.provenance.is_shell_search("grep -rn x .")


@pytest.mark.parametrize(
    ("n", "label"),
    [(1, "1-5"), (5, "1-5"), (6, "6-20"), (50, "21-50"), (51, "51-100"), (101, "100+")],
)
def test_length_bucket_edges(n: int, label: str) -> None:
    assert sr.LENGTH_BUCKETS[sr._bucket(n, sr.LENGTH_BUCKETS)][0] == label


def test_plan_requests_all_sampled_budget_and_over_budget() -> None:
    ends = list(range(0, 400, 2))  # 200 requests
    sizes = [10] * 400
    assert sr.plan_requests(ends[:5], sizes, 100, 10**9) == [(e, True) for e in ends[:5]]
    plan = sr.plan_requests(ends, sizes, 20, 10**9)
    assert plan is not None
    rec = [e for e, r in plan if r]
    assert len(rec) == 20 and rec[-1] == ends[-1]
    # every recorded request (but the first) is directly preceded by its true predecessor
    pos = {e: i for i, e in enumerate(ends)}
    for i, (e, r) in enumerate(plan):
        if r and pos[e] > 0:
            assert plan[i - 1][0] == ends[pos[e] - 1]
    trimmed = sr.plan_requests(ends, sizes, 100, 40_000)
    assert trimmed is not None and sum(r for _, r in trimmed) < 100
    assert sr.plan_requests(ends, sizes, 100, 1) is None


def test_percentiles() -> None:
    assert sr._pct([], 0.9) == 0.0
    assert sr._pct([1.0], 0.9) == 1.0
    assert sr._pct([0.0, 1.0], 0.5) == 0.5
    assert sr._dist([]) == {"median": 0.0, "mean": 0.0, "p90": 0.0}


# ----------------------------------------------------------------- end to end


def _write(root: Path, name: str, lines: list[str]) -> None:
    d = root / SECRETS[0]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    _write(tmp_path, SECRETS[3], make_session(8))  # 6-20 requests
    _write(tmp_path, "short", make_session(1))
    _write(tmp_path, "search", make_session(7, command=f"grep -rn {SECRETS[2]} ."))
    _write(tmp_path, "empty", ["{bad", json.dumps({"type": "summary"})])
    _write(
        tmp_path,
        "mcp",
        [_line("user", "q"), _assistant_tool("m", SECRETS[5], "x", "i"), _result("m", _big_log())],
    )
    return tmp_path


def test_run_aggregates_and_buckets(corpus: Path) -> None:
    out = sr.run(corpus, None, seed=0, jobs=1)
    assert out["sample"]["transcripts_found"] == 5
    assert out["sample"]["sessions_replayed"] == 4
    assert out["sample"]["sessions_skipped"] == {"empty": 1}
    assert out["sample"]["unparseable_lines"] == 0  # the bad line sits in a skipped session
    served = out["served"]
    lengths = served["by_session_length"]
    assert lengths["6-20"]["sessions"] == 2
    assert lengths["1-5"]["sessions"] == 2
    assert served["overall"]["weighted_token_saving"] > 0
    assert served["tool_results"]["bytes_digested_share"] > 0
    # shell-search stays verbatim when served, digests in the counterfactual
    assert served["by_bash_command_class"]["search"]["digest_rate"] == 0.0
    cf = out["without_shell_search_exemption"]["tool_results"]["bytes_digested_share"]
    assert cf > served["tool_results"]["bytes_digested_share"]
    assert served["by_tool"]["mcp"]["results"] == 1  # private MCP name collapsed to a bucket
    assert served["cache"]["bust_rate"] <= 1.0
    assert out["method"]["cache_control"].startswith("not recorded")


def test_privacy_nothing_from_input_reaches_output(corpus: Path, tmp_path_factory: Any) -> None:
    out_dir = tmp_path_factory.mktemp("out")
    assert sr.main(["--root", str(corpus), "--jobs", "1", "--out", str(out_dir)]) == 0
    blob = "".join(p.read_text() for p in out_dir.iterdir())
    assert blob
    for secret in (*SECRETS, str(corpus), corpus.name, "worker-3", "processed batch"):
        assert secret not in blob


def test_deterministic_across_runs_and_workers(corpus: Path) -> None:
    a = sr.run(corpus, 3, seed=7, jobs=1)
    b = sr.run(corpus, 3, seed=7, jobs=2)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert sr.render_report(a) == sr.render_report(b)


def test_sampling_depends_on_seed_and_caps(corpus: Path) -> None:
    assert sr.run(corpus, 2, seed=0, jobs=1)["sample"]["transcripts_found"] == 5
    seen = {sr.run(corpus, 2, seed=s, jobs=1)["sample"]["requests_total"] for s in range(6)}
    assert len(seen) > 1  # different seeds pick different transcripts


def test_error_paths_are_counted_not_raised(tmp_path: Path) -> None:
    assert sr.process_file(tmp_path / "missing.jsonl") == "error:FileNotFoundError"
    big = tmp_path / "big.jsonl"
    big.write_text("\n".join(make_session(12)))
    assert sr.process_file(big, budget_bytes=1) == "over_work_budget"
    capped = sr.process_file(big, max_requests=10)
    assert not isinstance(capped, str)
    assert capped.n_requests == 13
    assert len(capped.served.requests) == 10


def test_recorded_cache_control_is_reported(tmp_path: Path) -> None:
    marked = {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    _write(tmp_path, "r", [_line("user", [marked])])
    out = sr.run(tmp_path, None, seed=0, jobs=1)
    assert out["method"]["cache_control"] == "recorded by transcript"
    assert out["sample"]["sessions_with_recorded_cache_control"] == 1


def test_quote_hazard_is_measured(tmp_path: Path) -> None:
    src = "\n".join(f"line {i} of the module" for i in range(60))
    lines = [
        _line("user", "edit it"),
        _assistant_tool("r", "Bash", "cat a.py", "m1"),
        _result("r", src),
        _line(
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": "e",
                    "name": "Edit",
                    "input": {
                        "file_path": "a.py",
                        "old_string": "line 3 of the module",
                        "new_string": "x",
                    },
                }
            ],
            **{
                "message": {
                    "role": "assistant",
                    "id": "m2",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "e",
                            "name": "Edit",
                            "input": {
                                "file_path": "a.py",
                                "old_string": "line 3 of the module",
                                "new_string": "x",
                            },
                        }
                    ],
                }
            },
        ),
        _result("e", "ok"),
    ]
    _write(tmp_path, "h", lines)
    h = sr.run(tmp_path, None, seed=0, jobs=1)["served"]["quote_hazard"]
    assert h["requests_with_literal_edits"] == 1
    assert h["quotes_survived"] == 1 and h["quotes_lost"] == 0

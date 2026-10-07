"""`distil sh` (ADR 0026): per-command shaping at the source, and its PreToolUse rewrite.

Fixtures in tests/fixtures/shell/ are real tool output captured by
benchmarks/shell_replay.py. The property that matters most is negative: no failure,
error, warning or summary line is ever dropped, and the exit code is always the
command's own.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from distil import hook, shell

FIX = Path(__file__).parent / "fixtures" / "shell"
RUNNERS = {
    "pytest_verbose.txt": "pytest",
    "pytest_verbose_color.txt": "pytest",
    "pytest_default.txt": "pytest",
    "pytest_distil_suite_verbose.txt": "pytest",
    "unittest_verbose.txt": "unittest",
    "cargo_test.txt": "cargo-test",
    "go_test_verbose.txt": "go-test",
    "node_test.txt": "js-test",
}


@pytest.fixture(autouse=True)
def _metered(monkeypatch):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")


def _keep(text: str) -> str:
    return "deadbeef"


# ------------------------------------------------------------------ what gets rewritten


@pytest.mark.parametrize(
    "cmd,new",
    [
        ("git status", "distil sh -- git status"),
        ("pytest -q tests/x.py", "distil sh -- pytest -q tests/x.py"),
        ("py.test x", "distil sh -- py.test x"),
        ("python3 -m pytest -k 'a or b'", "distil sh -- python3 -m pytest -k 'a or b'"),
        ("python -m unittest -v", "distil sh -- python -m unittest -v"),
        ("./tests/runtests.py --verbosity 2 x", "distil sh -- ./tests/runtests.py --verbosity 2 x"),
        ("python tests/runtests.py x", "distil sh -- python tests/runtests.py x"),
        ("PYTHONPATH=. pytest a", "distil sh -- PYTHONPATH=. pytest a"),
        ("cargo test", "distil sh -- cargo test"),
        ("go test -v ./...", "distil sh -- go test -v ./..."),
        ("npm run test", "distil sh -- npm run test"),
        ("npx jest", "distil sh -- npx jest"),
        ("vitest run", "distil sh -- vitest run"),
        (
            "cd /testbed && python -m pytest t.py -x 2>&1 | tail -30",
            "cd /testbed && distil sh -- python -m pytest t.py -x 2>&1 | tail -30",
        ),
    ],
)
def test_rewrite_only_prefixes(cmd, new):
    assert shell.rewrite(cmd) == new
    assert shell.rewrite(cmd, digest=True) == new.replace("distil sh --", "distil sh --digest --")


@pytest.mark.parametrize(
    "cmd",
    [
        "grep -rn foo .",  # search stays verbatim (1.56.3)
        "rg foo",
        "find . -name '*.py'",
        "ls -la",
        "git diff",
        "git log",
        "cat app.py",
        "npm install test",
        "pytest a; echo done",
        "pytest a && echo done",
        "pytest a > out.txt",
        "echo $(pytest)",
        "pytest `x`",
        "pytest a | grep FAIL",
        "distil sh -- pytest a",
        "pytest 'unclosed",
        "",
        "FOO=1",
    ],
)
def test_everything_else_is_left_alone(cmd):
    assert shell.rewrite(cmd) is None


# ------------------------------------------------------------------ filters


def test_visible_strips_ansi_and_resolves_carriage_returns():
    raw = "\x1b[32mok\x1b[0m\r\nprogress 10%\rprogress 100%\n\x1b]0;title\x07done"
    assert shell.visible(raw) == "ok\nprogress 100%\ndone"


def test_git_status_drops_only_advice():
    raw = (FIX / "git_status.txt").read_text(encoding="utf-8")
    out, tier = shell.shape(raw, "git-status", lossy=False, save=_keep)
    assert tier == "lossless"
    kept = [ln for ln in raw.splitlines() if '(use "git' not in ln]
    assert out.splitlines() == kept
    assert "new_feature.py" in out and "deleted:    c.py" in out


@pytest.mark.parametrize("name,kind", sorted(RUNNERS.items()))
@pytest.mark.parametrize("lossy", [False, True])
def test_no_failure_error_or_summary_line_is_ever_dropped(name, kind, lossy):
    raw = (FIX / name).read_text(encoding="utf-8")
    out, _ = shell.shape(raw, kind, lossy=lossy, save=_keep)
    seen = shell.visible(out).splitlines()
    for ln in shell.visible(raw).splitlines():
        if shell._protected(ln):
            assert ln in seen, ln


@pytest.mark.parametrize("name,kind", sorted(RUNNERS.items()))
def test_lossless_tier_keeps_every_distinct_visible_line(name, kind):
    raw = (FIX / name).read_text(encoding="utf-8")
    out, _ = shell.shape(raw, kind, lossy=False, save=_keep)
    assert set(shell.visible(raw).splitlines()) <= set(out.splitlines())
    assert "distil expand" not in out


@pytest.mark.parametrize(
    "name,kind",
    [(n, k) for n, k in sorted(RUNNERS.items()) if n != "pytest_default.txt"],
)
def test_elide_tier_shrinks_and_points_at_the_full_output(name, kind):
    raw = (FIX / name).read_text(encoding="utf-8")
    out, tier = shell.shape(raw, kind, lossy=True, save=_keep)
    assert tier == "elide" and len(out) < len(raw) * 0.75
    assert out.rstrip().endswith("Full output: `distil expand deadbeef`]")
    assert shell.FILTERS_VERSION in out


@pytest.mark.parametrize(
    "name,kind",
    [(n, k) for n, k in sorted(RUNNERS.items()) if n != "pytest_default.txt"],
)
def test_crlf_output_shapes_the_same(name, kind):
    """A Windows child writes CRLF; the filters must still see whole lines."""
    raw = (FIX / name).read_text(encoding="utf-8")
    out, tier = shell.shape(raw.replace("\n", "\r\n"), kind, lossy=True, save=_keep)
    assert tier == "elide" and out == shell.shape(raw, kind, lossy=True, save=_keep)[0]


@pytest.mark.parametrize(
    "argv,kind",
    [
        (["C:\\v\\Scripts\\python.exe", "-m", "pytest"], "pytest"),
        (["D:\\py\\python3.12.EXE", "C:\\t\\runtests.py"], "unittest"),
        ([".venv/bin/python3", "-m", "unittest"], "unittest"),
        (["pytest.exe", "-q"], "pytest"),
        (["npm.cmd", "test"], "js-test"),
        (["git.exe", "status"], "git-status"),
        (["C:\\bin\\cargo.exe", "test"], "cargo-test"),
        (["pythonista", "-m", "pytest"], None),
    ],
)
def test_classify_windows_and_relative_program_paths(argv, kind):
    assert shell.classify(argv) == kind


def test_property_random_interleavings_never_lose_a_failure():
    passing = [
        "tests/a.py::test_x PASSED  [ 10%]",
        "test_y (m.C.test_y) ... ok",
        "test t::z ... ok",
        "=== RUN   TestZ",
        "--- PASS: TestZ (0.00s)",
        "  ✓ renders (3 ms)",
        "tests/b.py ....   [ 50%]",
        "....",
    ]
    bad = [
        "tests/a.py::test_err PASSED but logged an ERROR",
        "test_error_path (m.C) ... ok",
        "--- FAIL: TestQ (0.01s)",
        "E   AssertionError: boom",
        "Traceback (most recent call last):",
        "panic: runtime error",
        "warning: unused variable",
        "=== 3 failed, 9 passed in 1.0s ===",
        "test result: FAILED. 1 passed; 1 failed",
        "  ✓ fails over to backup (3 ms)",
    ]
    rng = random.Random(26)
    for _ in range(300):
        lines = [rng.choice(passing) for _ in range(rng.randint(0, 30))]
        for b in rng.sample(bad, rng.randint(1, len(bad))):
            lines.insert(rng.randint(0, len(lines)), b)
        for kind in shell._PASS:
            out, _ = shell.shape("\n".join(lines), kind, lossy=True, save=_keep)
            for b in bad:
                if b in lines:
                    assert b in out, (kind, b)


def test_nothing_lossy_without_a_stored_original():
    raw = (FIX / "pytest_verbose.txt").read_text(encoding="utf-8")
    out, tier = shell.shape(raw, "pytest", lossy=True, save=lambda t: None)
    assert tier == "none" and out == raw


def test_too_few_passing_lines_is_not_worth_a_marker():
    raw = "a.py::t1 PASSED\na.py::t2 PASSED\n=== 2 passed in 0.1s ==="
    assert shell.shape(raw, "pytest", lossy=True, save=_keep) == (raw, "none")


def test_unknown_kind_elides_nothing():
    assert shell.elide("x\ny", "nope") == ("x\ny", 0)


# ------------------------------------------------------------------ `distil sh` itself


def _script(tmp_path: Path, body: str) -> list[str]:
    p = tmp_path / "runtests.py"  # classified as a unittest-style runner
    p.write_text(body)
    return [sys.executable, str(p)]


_RUNNER = (
    "import sys\n"
    "for i in range(40): print(f'test_{i} (m.C.test_{i}) ... ok')\n"
    "print('test_bad (m.C.test_bad) ... FAIL')\n"
    "print('AssertionError: nope', file=sys.stderr)\n"
    "print('Ran 41 tests'); print('FAILED (failures=1)')\n"
    "sys.exit(3)\n"
)


def test_main_shapes_keeps_failures_and_returns_the_exit_code(tmp_path, capsys):
    from distil.mcp_server import load_restore

    rc = shell.main(_script(tmp_path, _RUNNER))
    out = capsys.readouterr().out
    assert rc == 3
    assert "test_bad (m.C.test_bad) ... FAIL" in out and "AssertionError: nope" in out
    assert "FAILED (failures=1)" in out and "test_3 (m.C.test_3) ... ok" not in out
    h = out.rsplit("distil expand ", 1)[1][:8]
    assert "test_3 (m.C.test_3) ... ok" in (load_restore(h) or "")
    rows = [json.loads(x) for x in hook._receipt_path().read_text().splitlines()]
    assert rows[-1]["client"] == "sh" and rows[-1]["tool"] == "sh:unittest"
    assert rows[-1]["filters"] == shell.FILTERS_VERSION and rows[-1]["tier"] == "elide"


def test_subscription_gets_the_lossless_tier_only(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert shell.main(_script(tmp_path, _RUNNER)) == 3
    assert "distil expand" not in capsys.readouterr().out
    assert shell.main(_script(tmp_path, _RUNNER), digest=True) == 3  # explicit opt-in
    assert "distil expand" in capsys.readouterr().out


def test_shaping_failure_prints_the_raw_output(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("filter bug")

    monkeypatch.setattr(shell, "shape", boom)
    assert shell.main(_script(tmp_path, _RUNNER)) == 3
    assert "test_3 (m.C.test_3) ... ok" in capsys.readouterr().out


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX signals: on Windows os.kill(SIGTERM) is TerminateProcess(15), a plain exit code",
)
def test_signal_exit_maps_to_the_shell_convention(tmp_path):
    body = "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n"
    assert shell.main(_script(tmp_path, body)) == 143


def test_env_assignments_reach_the_child(tmp_path, capsys):
    p = tmp_path / "runtests.py"
    p.write_text("import os; print(os.environ['SH_PROBE'])\n")
    assert shell.main(["SH_PROBE=hello", sys.executable, str(p)]) == 0
    assert capsys.readouterr().out.strip() == "hello"
    assert shell.main(["ONLY=assignments"]) == 0  # nothing to run, as in a shell


def test_usage_without_a_command(capsys):
    assert shell.main([]) == 2


def _run_cli(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-m", "distil.cli", "sh", *args], capture_output=True, text=True, env=env
    )


def test_unclassified_and_kill_switch_exec_the_command_untouched(tmp_path):
    r = _run_cli(["--", sys.executable, "-c", "print('a\\x1b[0m'); raise SystemExit(4)"], {})
    assert r.returncode == 4 and r.stdout == "a\x1b[0m\n"
    script = _script(tmp_path, _RUNNER)
    r = _run_cli(["--", *script], {"DISTIL_SH_OFF": "1"})
    assert r.returncode == 3 and "test_3 (m.C.test_3) ... ok" in r.stdout
    probe = [sys.executable, "-c", "import os; print(os.environ['SH_PROBE'])"]
    r = _run_cli(["--", "SH_PROBE=hi", *probe], {})  # assignments honoured when fail-open too
    assert r.returncode == 0 and r.stdout.strip() == "hi"


def test_missing_command_is_127(tmp_path, capsys):
    assert shell.main(["/nonexistent/bin/pytest"]) == 127  # classified, cannot start
    assert shell._exec_untouched(["definitely-not-a-command-xyz"]) == 127


def test_cli_entry(tmp_path):
    r = _run_cli(["--", *_script(tmp_path, _RUNNER)], {"DISTIL_SUBSCRIPTION": "0"})
    assert r.returncode == 3 and "passing-test lines elided" in r.stdout


# ------------------------------------------------------------------ PreToolUse rewrite


def _settings(tmp_path: Path, perms: dict) -> Path:
    from distil.hook import config_path

    p = config_path("claude")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"permissions": perms}))
    return p


@pytest.fixture
def on_path(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/distil" if n == "distil" else None)


def _ev(cmd: str, **kw) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": cmd, "description": "d"}, **kw}


def test_pre_rewrites_and_keeps_other_input_fields(tmp_path, on_path):
    out = shell.pre_tool_use(_ev("pytest -q"))
    assert out is not None
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["updatedInput"] == {"command": "distil sh -- pytest -q", "description": "d"}
    assert "permissionDecision" not in hso  # no allow rule: Claude Code's flow decides


def test_pre_mirrors_an_allow_rule_for_the_original_command(tmp_path, on_path):
    _settings(tmp_path, {"allow": ["Bash(pytest:*)", "Read"]})
    hso = shell.pre_tool_use(_ev("pytest -q"))["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    # a cd or a pipe is a second subcommand: no decision is mirrored
    for cmd in ("cd x && pytest -q", "pytest -q | tail -5"):
        assert "permissionDecision" not in shell.pre_tool_use(_ev(cmd))["hookSpecificOutput"]
    # an allow rule for something else does not count
    _settings(tmp_path, {"allow": ["Bash(git status)", "Bash(np*:*)"]})
    assert "permissionDecision" not in shell.pre_tool_use(_ev("pytest"))["hookSpecificOutput"]
    _settings(tmp_path, {"allow": ["Bash"]})
    assert shell.pre_tool_use(_ev("pytest"))["hookSpecificOutput"]["permissionDecision"] == "allow"
    _settings(tmp_path, {"allow": ["Bash(pytest -q)"]})
    assert (
        shell.pre_tool_use(_ev("pytest -q"))["hookSpecificOutput"]["permissionDecision"] == "allow"
    )


@pytest.mark.parametrize(
    "rules,blocked",
    [
        ({"deny": ["Bash(pytest:*)"]}, {"pytest -q", "cd x && pytest -q"}),
        ({"ask": ["Bash(git status)"]}, {"git status"}),
        ({"deny": ["Bash"]}, {"pytest -q", "git status", "cd x && pytest -q"}),
        ({"ask": ["Bash(cd:*)"]}, {"cd x && pytest -q"}),
    ],
)
def test_pre_never_rewrites_what_a_deny_or_ask_rule_could_touch(tmp_path, on_path, rules, blocked):
    _settings(tmp_path, rules)
    for cmd in ("pytest -q", "git status", "cd x && pytest -q"):
        assert (shell.pre_tool_use(_ev(cmd)) is None) == (cmd in blocked), cmd


def test_pre_unrelated_deny_rule_does_not_block(tmp_path, on_path):
    _settings(tmp_path, {"deny": ["Bash(git push:*)", "Bash(rm:*)"]})
    assert shell.pre_tool_use(_ev("git status")) is not None


def test_pre_declines_when_it_should(tmp_path, monkeypatch, on_path):
    assert shell.pre_tool_use({"tool_name": "Read", "tool_input": {"command": "pytest"}}) is None
    assert shell.pre_tool_use({"tool_name": "Bash", "tool_input": "pytest"}) is None
    assert shell.pre_tool_use(_ev("grep -rn x .")) is None
    monkeypatch.setenv("DISTIL_SH_OFF", "1")
    assert shell.pre_tool_use(_ev("pytest")) is None


def test_pre_declines_when_distil_is_not_on_path(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda n: None)
    assert shell.pre_tool_use(_ev("pytest")) is None


def test_pre_survives_odd_settings(tmp_path, on_path):
    from distil.hook import config_path

    p = config_path("claude")
    p.parent.mkdir(parents=True, exist_ok=True)
    for text in ("not json", '{"permissions": ["x"]}', "[]"):
        p.write_text(text)
        assert shell.pre_tool_use(_ev("pytest", cwd=str(tmp_path))) is not None


def test_run_pre_never_raises(on_path):
    assert hook.run_pre("not json") == "{}"
    assert hook.run_pre("[]") == "{}"
    assert hook.run_pre(json.dumps(_ev("ls"))) == "{}"
    out = json.loads(hook.run_pre(json.dumps(_ev("pytest")), "digest"))
    assert out["hookSpecificOutput"]["updatedInput"]["command"] == "distil sh --digest -- pytest"


def test_hook_main_pre_flag(monkeypatch, capsys, on_path):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_ev("cargo test"))))
    assert hook.main(["--pre", "--tier", "auto"]) == 0
    assert "distil sh -- cargo test" in capsys.readouterr().out


def test_post_tool_hook_leaves_distil_sh_output_alone():
    big = "line of output that is fairly long and distinct %d\n"
    payload = {"stdout": "".join(big % i for i in range(300)), "stderr": "", "interrupted": False}
    assert hook.compress_tool_output("Bash", payload, {"command": "distil sh -- pytest"}) is None
    assert hook.compress_tool_output("Bash", payload, {"command": "python x.py"}) is not None


# ------------------------------------------------------------------ install / uninstall


def test_install_uninstall_shell_round_trip_keeps_foreign_hooks(capsys):
    from distil.hook import config_path, hook_status, install_hook, uninstall_hook

    p = config_path("claude")
    p.parent.mkdir(parents=True, exist_ok=True)
    foreign = {"matcher": "Edit", "hooks": [{"type": "command", "command": "mine"}]}
    p.write_text(json.dumps({"hooks": {"PreToolUse": [foreign]}}))
    assert install_hook("claude", shell=True) == 0
    assert install_hook("claude", shell=True, digest=True) == 0  # idempotent
    pre = json.loads(p.read_text())["hooks"]["PreToolUse"]
    assert pre[0] == foreign and len(pre) == 2 and "--pre --tier digest" in json.dumps(pre[1])
    assert hook_status("claude", shell=True)[0] and not hook_status("claude")[0]
    assert hook.print_status() == 0 and "shell rewrite" in capsys.readouterr().out
    assert install_hook("claude") == 0  # the post-tool hook coexists
    assert uninstall_hook("claude", shell=True) == 0
    data = json.loads(p.read_text())
    assert data["hooks"]["PreToolUse"] == [foreign] and data["hooks"]["PostToolUse"]
    assert uninstall_hook("claude") == 0
    assert install_hook("cursor", shell=True) == 0 and uninstall_hook("cursor", shell=True) == 0


def test_install_shell_on_fresh_home_is_fully_undone():
    from distil.hook import config_path, install_hook, uninstall_hook

    p = config_path("claude")
    assert not p.exists()
    assert install_hook("claude", shell=True) == 0 and p.exists()
    assert uninstall_hook("claude", shell=True) == 0 and not p.exists()


def test_cli_hook_shell_and_setup(capsys):
    import argparse

    from distil.cli import cmd_hook, cmd_setup_front
    from distil.hook import hook_status

    assert cmd_hook(argparse.Namespace(action="install", shell=True, digest=False)) == 0
    assert hook_status("claude", shell=True)[0]
    assert cmd_hook(argparse.Namespace(action="uninstall", shell=True)) == 0
    assert not hook_status("claude", shell=True)[0]
    ns = argparse.Namespace(
        settings=False, statusline_only=False, hooks=True, digest=False, shell=True
    )
    assert cmd_setup_front(ns) == 0
    assert hook_status("claude", shell=True)[0] and hook_status("claude")[0]


def test_windows_fail_open_keeps_the_commands_exit_code(monkeypatch):
    """Windows has no exec; os.execvp would spawn the command and exit 0, hiding a failure."""
    calls = []

    class Done:
        returncode = 3

    monkeypatch.setattr(shell.sys, "platform", "win32")
    monkeypatch.setattr(shell.subprocess, "run", lambda argv, env: calls.append(argv) or Done())
    monkeypatch.setattr(shell.os, "execvpe", lambda *a: (_ for _ in ()).throw(AssertionError))
    assert shell._exec_untouched(["make", "lint"]) == 3
    assert calls == [["make", "lint"]]


def test_windows_resolves_the_program_through_pathext(monkeypatch):
    """CreateProcess does not apply PATHEXT: `npm` must become `npm.cmd` before spawning."""
    seen = {}

    def which(cmd, path=None):
        seen["path"] = path
        return "C:\\node\\npm.cmd"

    monkeypatch.setattr(shell.sys, "platform", "win32")
    monkeypatch.setattr("shutil.which", which)
    argv, env = shell._split_env(["PATH=C:\\node", "CI=1", "npm", "test"])
    assert argv == ["C:\\node\\npm.cmd", "test"] and env["CI"] == "1"
    assert seen["path"] == "C:\\node"  # resolved on the child's PATH, as a shell would

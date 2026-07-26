"""The result/verdict line of command output must survive digestion.

Regression for a content-agnostic miss a power-user comparison surfaced
(moonlit-piroshki-fbaa04.netlify.app, tested on ZoneMinder/zmNinjaNg): a
*passing* test run's summary ("1955 passed", which carries no error word) was
folded into a retrieval handle while the ERROR/WARN stdout the run emitted on
purpose was kept — distil "compressed away the answer and kept the noise".
See distil/compress/tier1.py:_SUMMARY_RE.
"""

from distil.compress.tier1 import digest, _must_keep


def _log(summary: str) -> str:
    """A passing-run shape: head + neutral filler + on-purpose noise + verdict near tail."""
    lines = ["RUN v1.6.0 /repo", "", "stdout | crypto.test.ts"]
    lines += [
        f"rendering sample component {i} to virtual dom" for i in range(30)
    ]  # neutral -> folds
    lines += [f"ERROR [Crypto] Decryption failed attempt {i}" for i in range(20)]  # kept (noise)
    lines += [f"WARN  [Auth] token near expiry {i}" for i in range(20)]  # kept (noise)
    lines += ["", summary, " Duration  3.21s"]  # verdict NOT in tail
    return "\n".join(lines)


def test_passing_verdict_survives_digest():
    for summary in [
        "  Tests  1955 passed (1955)",  # vitest, all green
        "Tests:       1955 passed, 1955 total",  # jest
        "===== 1955 passed in 12.34s =====",  # pytest
        "test result: ok. 42 passed; 0 failed; 0 ignored",  # cargo
        "  1955 passing",  # mocha
    ]:
        d, changed = digest(_log(summary))
        assert changed, "log should compress"
        assert "handle=" in d, "neutral middle must still fold — real compression preserved"
        assert summary.strip() in d, f"verdict folded away:\n{summary!r}\nGOT:\n{d}"


def test_go_and_build_verdicts_kept():
    for line in [
        "ok  \tgithub.com/x/y\t0.02s",
        "PASS",
        "--- FAIL: TestFoo (0.00s)",
        "BUILD SUCCESSFUL in 3s",
        "process exited with exit code 1",
    ]:
        assert _must_keep(line), f"result line dropped: {line!r}"


def test_no_false_keep_on_prose():
    # ordinary code/log lines must NOT trip the result net, or compression regresses
    for line in [
        "const passed = true;",
        "// this test is important",
        "logger.info('user logged in ok')",
        "return ok(result)",
        "the request passed through the gateway",
    ]:
        assert not _must_keep(line), f"false keep hurts compression: {line!r}"


def test_salience_layer_pins_verdicts():
    # same bug class on the salience path: a green verdict has no error word and
    # 0-1 digit groups, so it scored below the keep bar while ERROR noise scored 0.95
    from distil.codec.keep_model import SalienceKeepModel

    m = SalienceKeepModel()
    for line in [
        "ok  \tgithub.com/x/y\t0.02s",
        "PASS",
        "BUILD SUCCESSFUL in 3s",
        "  Tests  1955 passed (1955)",
    ]:
        assert m.score(line, "tool_result") == 1.0, f"salience dropped verdict: {line!r}"
    for line in ["const passed = true;", "return ok(result)"]:
        assert m.score(line, "tool_result") < 1.0, f"false verdict pin: {line!r}"


def test_changed_is_false_when_nothing_was_elided():
    """`changed` must mean the output actually differs from the input.

    Regression for a real defect: digest() returned True unconditionally. On content
    where every line is must-keep — a test log whose verdict policy pins each PASS
    line — nothing is dropped, no marker is emitted, and the output is byte-identical.
    Reporting True there made callers persist an original for a block they never
    digested: RestoreStore._record wrote plaintext to ~/.distil/restore for content
    that can never need recovery, burning the FIFO cap and widening the at-rest
    surface for nothing. The MCP server likewise handed back a handle for text it had
    not compressed.
    """
    from distil.compress.tier1 import digest

    # every line is a verdict line -> all must-keep -> nothing to drop
    all_keep = "run_tests()\n" + "\n".join(
        f"PASS tests/test_mod_{i}.py::case_{i} in 0.0{i}s" for i in range(400)
    )
    out, changed = digest(all_keep)
    assert out == all_keep, "nothing should have been elided"
    assert changed is False, "changed=True on byte-identical output is the bug"
    assert "handle=" not in out

    # ordinary noise still digests and still reports changed
    noisy = "get_logs()\n" + "\n".join(
        f"info: handler processed item {i} normally" for i in range(400)
    )
    out2, changed2 = digest(noisy)
    assert changed2 is True
    assert len(out2) < len(noisy)
    assert "handle=" in out2


def test_unchanged_block_records_no_restore_entry(tmp_path, monkeypatch):
    """The consequence that matters: no plaintext stored for content left verbatim."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil.adapters.anthropic import compress_messages

    all_keep = "run_tests()\n" + "\n".join(
        f"PASS tests/test_mod_{i}.py::case_{i} in 0.0{i}s" for i in range(400)
    )
    msgs = []
    for k in range(4):
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{k}", "content": all_keep}],
            }
        )
        msgs.append({"role": "assistant", "content": f"step {k}"})
    msgs.append({"role": "user", "content": "summarise"})

    compressed, store = compress_messages(msgs, verbatim=False, keep=None)
    assert not (getattr(store, "handles", None) or []), (
        "stored an original for an un-digested block"
    )
    assert compressed == msgs, "nothing should have changed"


if __name__ == "__main__":
    test_passing_verdict_survives_digest()
    test_go_and_build_verdicts_kept()
    test_no_false_keep_on_prose()
    test_salience_layer_pins_verdicts()
    print("ok — verdict-preservation holds across vitest/jest/pytest/cargo/mocha/go")

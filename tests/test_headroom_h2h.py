"""The pre-registered rules behind benchmarks/headroom_h2h.py, without either compressor."""

from benchmarks.headroom_h2h import boot, classify, price_session, score


class _Tok:
    def count(self, text: str) -> float:
        return float(len(text.split()))


def _req(*results: str) -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "go " * 600}]
    for i, r in enumerate(results):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": r}],
            }
        )
    return msgs


def test_classify_first_match_wins() -> None:
    assert classify("Bash", {"command": "grep -rn x ."}, '{"a": 1}') == "json"
    assert classify("Bash", {"command": "curl x"}, "<html><body>" + "<p>x</p>" * 20) == "html"
    assert classify("Bash", {"command": "rg foo"}, "a.py:1: foo") == "search"
    assert classify("Read", {"file_path": "/x/y.py"}, "def f(): pass") == "code"
    assert classify("Read", {"file_path": "/x/README.md"}, "# Title") == "prose"
    assert classify("Bash", {"command": "pytest -q"}, "3 passed") == "logs"
    assert classify("TodoWrite", {}, "ok") is None


def test_a_rewritten_prefix_is_a_bust_and_costs_more() -> None:
    stable = price_session([_req("a " * 300), _req("a " * 300, "b " * 300)], _Tok())
    bust = price_session([_req("a " * 300), _req("A " * 300, "b " * 300)], _Tok())
    assert stable["prefix_held"] == [True] and bust["prefix_held"] == [False]
    assert stable["read"] > 0 and bust["read"] == 0
    assert bust["usd"] > stable["usd"]


def test_score_uses_only_the_tools_own_recovery_blobs() -> None:
    original = "retry_limit: 3\nport=8787\nError: disk full on /var/lib/app/data"
    assert score(original, original, [])["retained"] == score(original, original, [])["total"]
    gone = score(original, "<<elided handle=deadbeef>>", [])
    assert gone["retained"] == gone["recoverable"] == 0 and gone["total"] >= 3
    back = score(original, "<<elided handle=deadbeef>>", [original])
    assert back["recoverable"] == back["total"]


def test_boot_ci_brackets_the_mean() -> None:
    assert boot([0.5] * 20)["lo"] == boot([0.5] * 20)["hi"] == 0.5
    b = boot([0.0, 1.0] * 50)
    assert b["n"] == 100 and b["lo"] < 0.5 < b["hi"]

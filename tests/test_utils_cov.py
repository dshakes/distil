"""Coverage gap tests for utility/adapter modules.

Covers the branches not exercised by the existing suite. All tests are
offline (no network, no real LLM, no real ONNX/torch). Tests that would
require unavailable deps (distil_core Rust, onnxruntime, transformers,
anthropic SDK) are either skipped or use injection seams.
"""

from __future__ import annotations

import importlib
import json
import sys
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# distil/__init__.py — PackageNotFoundError fallback
# ===========================================================================


def test_version_fallback_reads_pyproject(monkeypatch):
    """When importlib.metadata.version raises PackageNotFoundError, __version__
    falls back to reading pyproject.toml (which exists in this repo checkout)."""
    import distil as _mod
    from importlib.metadata import PackageNotFoundError

    # Make version() always raise so the fallback code path runs on reload.
    monkeypatch.setattr(
        "importlib.metadata.version", lambda pkg: (_ for _ in ()).throw(PackageNotFoundError(pkg))
    )

    try:
        importlib.reload(_mod)
        # The fallback reads pyproject.toml from the repo root; should find a real version.
        assert _mod.__version__
        assert _mod.__version__ != "0+source", (
            "fallback returned 0+source — pyproject.toml not found or version key missing"
        )
    finally:
        # Always restore normal state so other tests see the real version.
        monkeypatch.undo()
        importlib.reload(_mod)


# ===========================================================================
# distil/tokenizer.py — AnthropicTokenizer paths
# ===========================================================================


def test_anthropic_tokenizer_sdk_not_installed(monkeypatch):
    """_ensure_client raises SystemExit when anthropic package is absent."""
    from distil.tokenizer import AnthropicTokenizer

    tok = AnthropicTokenizer()
    # Pretend the sdk is absent by making the import fail.
    original = sys.modules.get("anthropic")
    sys.modules["anthropic"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(SystemExit, match="anthropic"):
            tok._ensure_client()
    finally:
        if original is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = original


def test_anthropic_tokenizer_client_init_failure(monkeypatch):
    """_ensure_client raises SystemExit when Anthropic() constructor fails."""
    from distil.tokenizer import AnthropicTokenizer

    tok = AnthropicTokenizer()

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.side_effect = Exception("bad key")

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            tok._ensure_client()


def test_anthropic_tokenizer_count_uses_cache():
    """count() on a pre-injected client uses the cache on repeat calls."""
    from distil.tokenizer import AnthropicTokenizer

    mock_client = MagicMock()
    resp = MagicMock()
    resp.input_tokens = 42
    mock_client.messages.count_tokens.return_value = resp

    tok = AnthropicTokenizer(client=mock_client)
    assert tok.count("hello world") == 42
    assert tok.count("hello world") == 42  # cache hit — no second call
    assert mock_client.messages.count_tokens.call_count == 1


def test_anthropic_tokenizer_count_empty_string():
    """count('') returns 0 immediately without calling the client."""
    from distil.tokenizer import AnthropicTokenizer

    mock_client = MagicMock()
    tok = AnthropicTokenizer(client=mock_client)
    assert tok.count("") == 0
    mock_client.messages.count_tokens.assert_not_called()


def test_anthropic_tokenizer_api_failure_raises_systemexit():
    """count() raises SystemExit when the API call fails."""
    from distil.tokenizer import AnthropicTokenizer

    mock_client = MagicMock()
    mock_client.messages.count_tokens.side_effect = Exception("network error")

    tok = AnthropicTokenizer(client=mock_client)
    with pytest.raises(SystemExit, match="token count call failed"):
        tok.count("some text that is long enough to trigger the call")


def test_resolve_anthropic_returns_anthropic_tokenizer():
    """resolve('anthropic') returns an AnthropicTokenizer (with no real key)."""
    from distil.tokenizer import AnthropicTokenizer, resolve

    tok = resolve("anthropic")
    assert isinstance(tok, AnthropicTokenizer)


def test_resolve_unknown_raises():
    """resolve() with an unknown name raises ValueError."""
    from distil.tokenizer import resolve

    with pytest.raises(ValueError, match="unknown tokenizer"):
        resolve("gpt2")


# ===========================================================================
# distil/cachedelta.py — uncovered branches
# ===========================================================================


def test_msg_hash_fallback_on_unhashable():
    """_msg_hash falls back to str() when json.dumps raises (circular ref)."""
    from distil.cachedelta import _msg_hash

    # Create an object that json.dumps cannot serialise even with default=str
    # but str() works fine.  Use an object whose __str__ doesn't error.
    class _Unserializable:
        def __repr__(self) -> str:
            return "unserializable"

        def __str__(self) -> str:
            return "unserializable"

    # Inject it as a key in a dict so json.dumps fails (non-string key).
    obj = {_Unserializable(): "val"}
    # json.dumps with default=str still fails on non-string dict keys.
    h = _msg_hash(obj)
    assert len(h) == 64  # SHA-256 hex


def test_get_session_evicts_oldest_when_full():
    """get_session evicts the oldest session when _MAX_SESSIONS is exceeded."""
    from distil.cachedelta import (
        _MAX_SESSIONS,
        _SESSIONS,
        _SESSIONS_LOCK,
        get_session,
        reset_sessions,
    )

    reset_sessions()
    # Fill to the max.
    keys = [f"key_{i:05d}" for i in range(_MAX_SESSIONS + 2)]
    for k in keys:
        get_session(k)
    with _SESSIONS_LOCK:
        assert len(_SESSIONS) == _MAX_SESSIONS
    # The first two keys should have been evicted.
    assert keys[0] not in _SESSIONS
    assert keys[1] not in _SESSIONS
    reset_sessions()


def test_best_base_skips_empty_btext():
    """_best_base skips bases whose text is empty."""
    from distil.cachedelta import _best_base

    text = "x" * 500
    # All bases have empty text — should return None.
    result = _best_base(text, [("h1", ""), ("h2", "")])
    assert result is None


def test_best_base_skips_length_ratio_too_small():
    """_best_base skips bases whose length ratio is below _NEAR_DUP_RATIO."""
    from distil.cachedelta import _best_base

    long_text = "a" * 1000
    short_text = "a" * 10  # ratio = 10/1000 = 0.01 < 0.5
    result = _best_base(long_text, [("h", short_text)])
    assert result is None


def test_best_base_skips_low_quick_ratio():
    """_best_base skips bases where quick_ratio < _NEAR_DUP_RATIO."""
    from distil.cachedelta import _best_base

    # Two strings of similar length but totally different content.
    text = "abc " * 125  # 500 chars
    unrelated = "xyz " * 125  # 500 chars, same length, different content
    result = _best_base(text, [("h", unrelated)])
    # quick_ratio on fully disjoint strings is ~0; should skip.
    assert result is None


def test_rewrite_tool_texts_string_content_unchanged():
    """_rewrite_tool_texts returns the same object when string content doesn't change."""
    from distil.cachedelta import _rewrite_tool_texts

    msg = {"role": "user", "content": "short text"}
    result = _rewrite_tool_texts(msg, lambda t: t)  # identity — no change
    assert result is msg


def test_rewrite_tool_texts_list_sub_block_changed():
    """_rewrite_tool_texts handles tool_result with a list of text sub-blocks."""
    from distil.cachedelta import _MIN_CHARS, _rewrite_tool_texts

    long_text = "a" * (_MIN_CHARS + 100)
    sub_block = {"type": "text", "text": long_text}
    msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t",
                "content": [sub_block],
            }
        ],
    }

    # Transform that always appends "X" — so sub_block text will change.
    result = _rewrite_tool_texts(msg, lambda t: t + "X")
    assert result is not msg
    new_sub = result["content"][0]["content"][0]
    assert new_sub["text"] == long_text + "X"


def test_delta_encode_non_list_passthrough():
    """delta_encode returns the input unchanged when messages is not a list."""
    from distil.cachedelta import delta_encode

    result, store, stats = delta_encode("not a list")  # type: ignore[arg-type]
    assert result == "not a list"
    assert stats.prefix_msgs == 0


# ===========================================================================
# distil/codec/learned.py — uncovered branches
# ===========================================================================


def test_logistic_keep_model_wrong_weight_count():
    """LogisticKeepModel raises ValueError on wrong weight count."""
    from distil.codec.learned import LogisticKeepModel

    with pytest.raises(ValueError, match="weights"):
        LogisticKeepModel([0.0])  # too few


def test_train_empty_samples_raises():
    """train() raises ValueError on empty sample set."""
    from distil.codec.learned import train

    with pytest.raises(ValueError, match="empty"):
        train([])


def test_evaluate_false_negative_path():
    """_evaluate covers the fn (false negative: pred=0, label=1) branch."""
    from distil.codec.learned import FEATURE_NAMES, _evaluate

    # Weight vector that predicts EVERYTHING as 0 (negative logit for all).
    w = [-10.0] + [0.0] * (len(FEATURE_NAMES) - 1)
    # Sample with label=1: any feature vector — the model will predict 0 (fn).
    samples = [([1.0] + [0.0] * (len(FEATURE_NAMES) - 1), 1.0)]
    metrics = _evaluate(samples, w)
    assert metrics["fn"] == 1.0
    assert metrics["recall"] == 0.0


# ===========================================================================
# distil/codec/transformer.py — injection-seam paths (no ONNX needed)
# ===========================================================================


def _make_fake_session(logits_3d: list) -> Any:
    """Build a minimal fake ORT session that returns the given logits."""

    class _FakeSession:
        def run(self, output_names, input_feed):
            return [logits_3d]

    return _FakeSession()


def test_transformer_score_with_token_type_ids():
    """TransformerKeepModel.score passes token_type_ids when the encoder emits them."""
    from distil.codec.transformer import TransformerKeepModel

    # logits: shape [1, 2, 2]  —  2 tokens, 2 labels each
    logits = [[[0.0, 1.0], [0.0, 1.0]]]  # both tokens predict keep
    session = _make_fake_session(logits)

    def encode(line: str) -> dict:
        return {"input_ids": [1, 2], "attention_mask": [1, 1], "token_type_ids": [0, 0]}

    model = TransformerKeepModel(session, encode)
    score = model.score("hello world", "tool_output")
    assert 0.0 <= score <= 1.0


def test_transformer_score_with_input_names_filter():
    """TransformerKeepModel.score restricts feeds to declared input_names."""
    from distil.codec.transformer import TransformerKeepModel

    logits = [[[0.0, 2.0], [0.0, 2.0]]]
    session = _make_fake_session(logits)

    def encode(line: str) -> dict:
        return {"input_ids": [1, 2], "attention_mask": [1, 1], "token_type_ids": [0, 0]}

    # Only allow input_ids and attention_mask — token_type_ids filtered out.
    model = TransformerKeepModel(session, encode, input_names={"input_ids", "attention_mask"})
    score = model.score("test text here", "tool_output")
    assert score > 0.5  # high logit for keep


def test_transformer_score_index_error_in_logits():
    """TransformerKeepModel.score handles IndexError when logits array is misshapen."""
    from distil.codec.transformer import TransformerKeepModel

    # logits_3d is empty — logits_3d[0][tok_idx] will raise IndexError.
    logits = [[]]  # no per-token data
    session = _make_fake_session(logits)

    def encode(line: str) -> dict:
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    model = TransformerKeepModel(session, encode)
    # Should not raise; returns 0.0 (no probs collected).
    score = model.score("hello", "tool_output")
    assert score == 0.0


def test_transformer_score_padding_skipped():
    """TransformerKeepModel.score skips padded tokens (attention_mask == 0)."""
    from distil.codec.transformer import TransformerKeepModel

    # 3 tokens but token[1] is padding.
    logits = [[[0.0, 1.0], [99.0, -99.0], [0.0, 1.0]]]
    session = _make_fake_session(logits)

    def encode(line: str) -> dict:
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 0, 1]}

    model = TransformerKeepModel(session, encode)
    score = model.score("token skip test", "tool_output")
    assert 0.0 <= score <= 1.0


def test_transformer_from_pretrained_no_onnxruntime():
    """from_pretrained raises ImportError when onnxruntime is absent."""
    from distil.codec.transformer import TransformerKeepModel

    with patch.dict(sys.modules, {"onnxruntime": None}):
        with pytest.raises(ImportError, match="onnxruntime"):
            TransformerKeepModel.from_pretrained("/fake/model.onnx", "/fake/tok")


def test_transformer_from_pretrained_no_transformers():
    """from_pretrained raises ImportError when transformers is absent."""
    from distil.codec.transformer import TransformerKeepModel

    fake_ort = MagicMock()
    with patch.dict(sys.modules, {"onnxruntime": fake_ort, "transformers": None}):
        with pytest.raises(ImportError, match="transformers"):
            TransformerKeepModel.from_pretrained("/fake/model.onnx", "/fake/tok")


# ===========================================================================
# distil/ingest.py — uncovered branches
# ===========================================================================


def test_ingest_anthropic_assistant_string_content():
    """ingest_anthropic_request handles assistant message with string content."""
    from distil.ingest import ingest_anthropic_request
    from distil.trajectory import Kind, Stability

    body = {
        "messages": [
            {"role": "assistant", "content": "I will help you."},
        ]
    }
    blocks = ingest_anthropic_request(body)
    assert len(blocks) == 1
    assert blocks[0].kind == Kind.HISTORY
    assert blocks[0].stability == Stability.SETTLING
    assert blocks[0].text == "I will help you."


def test_ingest_anthropic_tool_result_list_content():
    """ingest_anthropic_request handles tool_result with list-of-text content."""
    from distil.ingest import ingest_anthropic_request
    from distil.trajectory import Kind

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "part one"},
                            {"type": "text", "text": "part two"},
                        ],
                    }
                ],
            }
        ]
    }
    blocks = ingest_anthropic_request(body)
    assert len(blocks) == 1
    assert blocks[0].kind == Kind.TOOL_OUTPUT
    assert "part one" in blocks[0].text
    assert "part two" in blocks[0].text


def test_ingest_openai_content_to_text_list():
    """_openai_content_to_text flattens list content."""
    from distil.ingest import _openai_content_to_text

    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "x"}},  # skipped
        {"type": "text", "text": "world"},
    ]
    result = _openai_content_to_text(content)
    assert result == "hello\nworld"


def test_ingest_file_jsonl(tmp_path):
    """ingest_file parses a .jsonl file with multiple requests."""
    from distil.ingest import ingest_file

    req1 = {"messages": [{"role": "user", "content": "hello"}]}
    req2 = {"messages": [{"role": "user", "content": "world"}]}
    jl = tmp_path / "session.jsonl"
    jl.write_text(json.dumps(req1) + "\n\n" + json.dumps(req2) + "\n")
    traj = ingest_file(str(jl), provider="anthropic")
    assert traj.id == "session"
    assert len(traj.turns) == 2


def test_ingest_file_jsonl_skips_malformed_lines(tmp_path):
    """ingest_file skips unparseable jsonl lines gracefully."""
    from distil.ingest import ingest_file

    req = {"messages": [{"role": "user", "content": "ok"}]}
    jl = tmp_path / "mixed.jsonl"
    jl.write_text("not json\n" + json.dumps(req) + "\n")
    traj = ingest_file(str(jl), provider="anthropic")
    assert len(traj.turns) == 1


# ===========================================================================
# distil/mcp_server.py — uncovered branches
# ===========================================================================


@pytest.fixture()
def _mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


def test_save_store_fifo_bound(_mcp_store):
    """_save_store evicts oldest entries when store exceeds _MAX_STORE_ENTRIES."""
    from distil import mcp_server as mcp

    # Build a store with one more than the max.
    store = {f"h{i:04d}": f"val{i}" for i in range(mcp._MAX_STORE_ENTRIES + 1)}
    first_key = next(iter(store))
    mcp._save_store(store)
    # The oldest key should have been popped.
    assert first_key not in store
    assert len(store) == mcp._MAX_STORE_ENTRIES


def test_save_store_oserror_silenced(tmp_path, monkeypatch):
    """_save_store silences OSError (best-effort)."""
    import distil.mcp_server as mcp

    # Point to an impossible path so write fails.
    monkeypatch.setenv("DISTIL_HOME", "/dev/null/impossible")
    mcp._save_store({"k": "v"})  # must not raise


def test_compress_non_string_text(_mcp_store):
    """distil_compress returns an error when text is not a string."""
    from distil import mcp_server as mcp

    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "distil_compress", "arguments": {"text": 42}},
        }
    )
    assert resp["result"]["isError"] is True
    assert "string" in resp["result"]["content"][0]["text"]


def test_compress_unchanged_small_text(_mcp_store):
    """distil_compress returns handle=null when text is already compact."""
    from distil import mcp_server as mcp

    # A very short text won't be changed by digest.
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "distil_compress", "arguments": {"text": "hi"}},
        }
    )
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["handle"] is None
    assert payload["tokens_saved"] == 0


def test_expand_non_string_handle(_mcp_store):
    """distil_expand returns an error when handle is not a string."""
    from distil import mcp_server as mcp

    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "distil_expand", "arguments": {"handle": 99}},
        }
    )
    assert resp["result"]["isError"] is True
    assert "string" in resp["result"]["content"][0]["text"]


def test_savings_tool(_mcp_store):
    """distil_savings returns a JSON dict with runs/tokens_saved/dollars_saved."""
    from distil import mcp_server as mcp

    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "distil_savings", "arguments": {}},
        }
    )
    assert resp["result"]["isError"] is False
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert "runs" in payload
    assert "tokens_saved" in payload
    assert "dollars_saved" in payload


def test_tool_exception_surfaces_as_error(_mcp_store):
    """When a tool function raises, handle_message returns isError=True."""
    from distil import mcp_server as mcp

    def _bad_fn(args):
        raise RuntimeError("boom")

    with patch.dict(mcp._DISPATCH, {"distil_compress": _bad_fn}):
        resp = mcp.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "distil_compress", "arguments": {"text": "x"}},
            }
        )
    assert resp["result"]["isError"] is True
    assert "boom" in resp["result"]["content"][0]["text"]


def test_ping_method(_mcp_store):
    """handle_message responds to 'ping' with an empty result."""
    from distil import mcp_server as mcp

    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 7, "method": "ping"})
    assert resp["result"] == {}


def test_serve_skips_blank_lines_and_bad_json(_mcp_store):
    """serve() skips blank lines and malformed JSON without crashing."""
    import io
    from distil import mcp_server as mcp

    stdin = io.StringIO("\n\nbad json\n")
    stdout = io.StringIO()
    mcp.serve(stdin=stdin, stdout=stdout)
    assert stdout.getvalue() == ""  # nothing written


# ===========================================================================
# distil/onboard.py — uncovered branches
# ===========================================================================


def test_install_method_pipx(monkeypatch):
    """install_method returns 'pipx' when /pipx/ is in the path blob."""
    import distil as _d
    from distil import onboard

    # install_method does `from . import __file__` which reads distil.__file__.
    monkeypatch.setattr(_d, "__file__", "/home/user/.local/pipx/venvs/distil/distil/__init__.py")
    monkeypatch.setattr(onboard.shutil, "which", lambda _: None)
    result = onboard.install_method()
    assert result == "pipx"


def test_install_method_uv(monkeypatch):
    """install_method returns 'uv' when /uv/tools/ is in the path blob."""
    from distil import onboard
    import distil as _d

    monkeypatch.setattr(
        _d, "__file__", "/home/user/.local/share/uv/tools/distil/distil/__init__.py"
    )
    monkeypatch.setattr(onboard.shutil, "which", lambda _: None)
    result = onboard.install_method()
    assert result == "uv"


def test_install_method_uvx(monkeypatch):
    """install_method returns 'uvx' when /uv/ (but not /uv/tools/) is in blob."""
    from distil import onboard
    import distil as _d

    monkeypatch.setattr(_d, "__file__", "/home/user/.cache/uv/temporary/distil/__init__.py")
    monkeypatch.setattr(onboard.shutil, "which", lambda _: None)
    result = onboard.install_method()
    assert result == "uvx"


def test_latest_pypi_version_offline(monkeypatch):
    """latest_pypi_version returns None when the network is unreachable."""
    from distil import onboard
    import urllib.request

    def _raise(*args, **kw):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    assert onboard.latest_pypi_version() is None


def test_is_outdated_same_base_prerelease():
    """is_outdated returns True when installed is a pre-release of the same base."""
    from distil.onboard import is_outdated

    assert is_outdated("1.7.0.dev0", "1.7.0") is True


def test_is_outdated_older_base():
    """is_outdated returns True when installed is a strictly older base version."""
    from distil.onboard import is_outdated

    assert is_outdated("1.6.0", "1.7.0") is True


def test_is_outdated_same_or_newer_is_false():
    """is_outdated returns False when up to date or newer."""
    from distil.onboard import is_outdated

    assert is_outdated("1.7.0", "1.7.0") is False
    assert is_outdated("1.8.0", "1.7.0") is False
    assert is_outdated("1.7.0", None) is False


def test_best_install_command_brew(monkeypatch):
    """best_install_command returns a brew path when brew is available."""
    from distil.onboard import best_install_command

    assert "brew" in best_install_command(["brew"])


def test_best_install_command_scoop():
    """best_install_command returns a scoop path on Windows-like env."""
    from distil.onboard import best_install_command

    cmd = best_install_command(["scoop"])
    assert "scoop" in cmd


def test_report_structure():
    """report() returns a dict with all required keys."""
    from distil.onboard import Env, report

    env = Env(
        os_name="Darwin",
        managers=["pipx"],
        agents=[("claude", "Claude Code")],
        has_anthropic=True,
        has_api_key=True,
        subscription=False,
        installed_version="1.7.0",
        method="pipx",
    )
    r = report(env, "1.8.0")
    assert r["upgrade_available"] is True
    assert r["upgrade_command"] is not None
    assert r["best_install_command"] == "pipx install distil-llm"
    assert r["billing"] == "metered"


# ===========================================================================
# distil/online.py — uncovered branches
# ===========================================================================


def test_retrain_empty_labels_raises():
    """retrain() raises ValueError on empty label set."""
    from distil.online import retrain

    with pytest.raises(ValueError, match="empty"):
        retrain([])


def test_retrain_empty_train_set_raises():
    """retrain() raises RuntimeError when all samples land in the test split."""
    from distil.online import retrain

    # Use test_fraction=1.0 to force everything into the test set.
    labels = [("line one", 1), ("line two", 0)]
    with pytest.raises(RuntimeError, match="Training set is empty"):
        retrain(labels, test_fraction=1.0)


def test_retrain_empty_test_split_warns():
    """retrain() warns when the test split is empty (tiny corpus)."""
    from distil.online import retrain

    # With test_fraction=0.0, no sample lands in the test set → warns.
    labels = [("line one", 1), ("line two", 0), ("line three", 1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = retrain(labels, test_fraction=0.0)
    assert any("overfit" in str(w.message) or "empty test split" in str(w.message) for w in caught)
    assert "weights" in result


def test_certify_promotion_returns_false_on_fail(monkeypatch):
    """certify_promotion returns False when certify says FAIL."""
    from distil.online import certify_promotion
    from distil.corpus import CorpusEntry
    from distil.trajectory import Block, Kind, Stability, Trajectory, Turn

    # Build a minimal 3-turn corpus entry.
    stable_block = Block(
        id="sys",
        kind=Kind.SYSTEM,
        text="system DECISION: go",
        stability=Stability.STABLE,
        decision_relevant=False,
    )
    vol_dr = Block(
        id="out_dr",
        kind=Kind.TOOL_OUTPUT,
        text="\n".join(f"line {i} DECISION: action" for i in range(20)),
        stability=Stability.VOLATILE,
        decision_relevant=True,
    )
    vol_noise = Block(
        id="out_noise",
        kind=Kind.TOOL_OUTPUT,
        text="\n".join(f"noise line {i}" for i in range(20)),
        stability=Stability.VOLATILE,
        decision_relevant=False,
    )
    turns = [Turn(i, [stable_block, vol_dr, vol_noise]) for i in range(3)]
    traj = Trajectory(id="t", model="claude-opus-4-8", turns=turns)
    entry = CorpusEntry(file="t.json", domain="test", title="t", trajectory=traj)

    # Patch certify in distil.online's namespace (that's where certify_promotion calls it).
    import distil.online as _online_mod

    fake_report = MagicMock()
    fake_report.verdict = "FAIL"
    monkeypatch.setattr(_online_mod, "certify", lambda *a, **kw: fake_report)

    from distil.codec.learned import FEATURE_NAMES

    weights = [0.0] * len(FEATURE_NAMES)
    result = certify_promotion(weights, [entry])
    assert result is False


def test_online_round_promote_to_writes_file(tmp_path, monkeypatch):
    """online_round writes weights when certified and promote_to is given."""
    from distil.online import online_round

    # Patch collect_causal_labels, retrain, certify_promotion.
    from distil.codec.learned import FEATURE_NAMES

    fake_weights = [0.0] * len(FEATURE_NAMES)

    monkeypatch.setattr(
        "distil.online.collect_causal_labels",
        lambda entries: [("line x", 1), ("line y", 0)],
    )
    monkeypatch.setattr(
        "distil.online.retrain",
        lambda labels, **kw: {"weights": fake_weights, "accuracy": 1.0, "f1": 1.0},
    )
    monkeypatch.setattr(
        "distil.online.certify_promotion",
        lambda w, entries, **kw: True,
    )
    monkeypatch.setattr(
        "distil.online.load_corpus",
        lambda: [],
    )

    out_path = tmp_path / "weights.json"
    result = online_round(entries=[], promote_to=str(out_path))
    assert result["promoted"] is True
    assert result["certified"] is True
    data = json.loads(out_path.read_text())
    assert "weights" in data


# ===========================================================================
# distil/corpus.py — uncovered branches
# ===========================================================================


def test_default_corpus_dir_env_override(monkeypatch):
    """_default_corpus_dir returns the $DISTIL_CORPUS path when set."""
    monkeypatch.setenv("DISTIL_CORPUS", "/fake/corpus")
    from distil import corpus as _corpus_mod

    result = _corpus_mod._default_corpus_dir()
    from pathlib import Path

    assert result == Path("/fake/corpus")


def test_default_corpus_dir_fallback_to_cwd(monkeypatch, tmp_path):
    """_default_corpus_dir falls back to cwd/corpus when no manifest exists."""
    from pathlib import Path

    monkeypatch.delenv("DISTIL_CORPUS", raising=False)

    from distil import corpus as _corpus_mod

    # Point corpus.__file__ to a location with no _corpus/corpus siblings.
    # The function does `here = Path(__file__).resolve().parent`, so using a
    # tmp_path sub-directory ensures neither candidate has a manifest.json.
    monkeypatch.setattr(_corpus_mod, "__file__", str(tmp_path / "distil" / "corpus.py"))

    result = _corpus_mod._default_corpus_dir()
    assert result == Path.cwd() / "corpus"


def _make_traj(
    *,
    n_turns: int = 3,
    model: str = "claude-opus-4-8",
    traj_id: str = "t",
    stable_text: str = "system DECISION: go",
    stable_changed: bool = False,
    volatile_before_stable: bool = False,
) -> Any:
    """Build a minimal Trajectory for validate() tests."""
    from distil.trajectory import Block, Kind, Stability, Trajectory, Turn

    Block(
        id="sys",
        kind=Kind.SYSTEM,
        text=stable_text,
        stability=Stability.STABLE,
        decision_relevant=False,
    )
    vol_dr = Block(
        id="out_dr",
        kind=Kind.TOOL_OUTPUT,
        text="\n".join(f"line {i} DECISION: act" for i in range(5)),
        stability=Stability.VOLATILE,
        decision_relevant=True,
    )
    vol_noise = Block(
        id="out_noise",
        kind=Kind.TOOL_OUTPUT,
        text="noise here",
        stability=Stability.VOLATILE,
        decision_relevant=False,
    )

    turns = []
    for i in range(n_turns):
        s = Block(
            id="sys",
            kind=Kind.SYSTEM,
            text=(stable_text + f" {i}") if stable_changed and i > 0 else stable_text,
            stability=Stability.STABLE,
            decision_relevant=False,
        )
        if volatile_before_stable:
            blocks = [vol_noise, s]
        else:
            blocks = [s, vol_dr, vol_noise]
        turns.append(Turn(i, blocks))

    return Trajectory(id=traj_id, model=model, turns=turns)


def test_validate_too_few_turns():
    """validate reports an error when trajectory has fewer than 3 turns."""
    from distil.corpus import validate

    traj = _make_traj(n_turns=2)
    problems = validate(traj)
    assert any("3 turns" in p for p in problems)


def test_validate_unknown_model():
    """validate reports an error for an unknown model name."""
    from distil.corpus import validate

    traj = _make_traj(model="gpt-99-ultra")
    problems = validate(traj)
    assert any("unknown model" in p for p in problems)


def test_validate_stable_block_changed_across_turns():
    """validate reports when a STABLE block has different text in two turns."""
    from distil.corpus import validate

    traj = _make_traj(stable_changed=True)
    problems = validate(traj)
    assert any("changed" in p for p in problems)


def test_validate_volatile_before_stable():
    """validate reports when a volatile block precedes a stable block."""
    from distil.corpus import validate

    traj = _make_traj(volatile_before_stable=True)
    problems = validate(traj)
    assert any("precedes" in p for p in problems)


def test_validate_no_stable_decision():
    """validate reports when no STABLE block contains DECISION:."""
    from distil.corpus import validate

    traj = _make_traj(stable_text="system prompt without the marker")
    problems = validate(traj)
    assert any("STABLE block" in p for p in problems)


def test_validate_no_volatile_decision():
    """validate reports when no decision-relevant volatile block has DECISION:."""
    from distil.trajectory import Block, Kind, Stability, Trajectory, Turn

    from distil.corpus import validate

    stable = Block(
        id="sys",
        kind=Kind.SYSTEM,
        text="sys DECISION: go",
        stability=Stability.STABLE,
        decision_relevant=False,
    )
    # volatile, decision_relevant=True but text has NO DECISION marker.
    vol_dr = Block(
        id="dr",
        kind=Kind.TOOL_OUTPUT,
        text="no marker here",
        stability=Stability.VOLATILE,
        decision_relevant=True,
    )
    vol_noise = Block(
        id="noise",
        kind=Kind.TOOL_OUTPUT,
        text="noise",
        stability=Stability.VOLATILE,
        decision_relevant=False,
    )
    turns = [Turn(i, [stable, vol_dr, vol_noise]) for i in range(3)]
    traj = Trajectory(id="t", model="claude-opus-4-8", turns=turns)
    problems = validate(traj)
    assert any("volatile" in p.lower() for p in problems)


def test_validate_no_prunable_noise():
    """validate reports when every volatile block is decision-relevant or has a marker."""
    from distil.trajectory import Block, Kind, Stability, Trajectory, Turn

    from distil.corpus import validate

    stable = Block(
        id="sys",
        kind=Kind.SYSTEM,
        text="sys DECISION: go",
        stability=Stability.STABLE,
        decision_relevant=False,
    )
    vol_dr = Block(
        id="dr",
        kind=Kind.TOOL_OUTPUT,
        text="DECISION: act",
        stability=Stability.VOLATILE,
        decision_relevant=True,
    )
    # This block is NOT decision-relevant but has DECISION: in text — disqualifies as prunable noise.
    vol_nd = Block(
        id="nd",
        kind=Kind.TOOL_OUTPUT,
        text="DECISION: extra marker",
        stability=Stability.VOLATILE,
        decision_relevant=False,
    )
    turns = [Turn(i, [stable, vol_dr, vol_nd]) for i in range(3)]
    traj = Trajectory(id="t", model="claude-opus-4-8", turns=turns)
    problems = validate(traj)
    assert any("prunable" in p.lower() or "noise block" in p.lower() for p in problems)


# ===========================================================================
# distil/adapters/gemini.py — list branch in _compress_json_value and
#                             no-change shortcut in _compress_part
# ===========================================================================


def test_compress_json_value_list_branch():
    """_compress_json_value compresses items in a list (functionResponse.response)."""
    from distil.adapters.gemini import _compress_json_value
    from distil.adapters.anthropic import RestoreStore

    store = RestoreStore()
    # A list with one large string value that will be transformed.
    big = "a" * 600
    result = _compress_json_value([big, "short"], store, verbatim=False)
    # The big entry should have been processed (possibly unchanged for tier0 text).
    assert isinstance(result, list)
    assert len(result) == 2


def test_compress_json_value_list_no_change():
    """_compress_json_value returns the same list object when nothing changed."""
    from distil.adapters.gemini import _compress_json_value
    from distil.adapters.anthropic import RestoreStore

    store = RestoreStore()
    orig = ["short", "also short"]
    result = _compress_json_value(orig, store, verbatim=False)
    assert result is orig  # identity: nothing changed


def test_compress_part_function_response_unchanged():
    """_compress_part returns same part object when functionResponse.response is unchanged."""
    from distil.adapters.gemini import _compress_part
    from distil.adapters.anthropic import RestoreStore

    store = RestoreStore()
    resp_data = {"result": "already compact"}
    part = {"functionResponse": {"name": "fn", "response": resp_data}}
    result = _compress_part(part, store, role="user", verbatim=False)
    assert result is part


# ===========================================================================
# distil/integrations/langgraph.py — attribute-style state with copy()
# ===========================================================================


def test_compress_state_attribute_style_with_copy():
    """compress_state handles attribute-style state (Pydantic-like) via .copy()."""
    from distil.integrations.langgraph import compress_state

    class _FakeState:
        def __init__(self, messages):
            self.messages = messages

        def copy(self, *, update):
            new = _FakeState(self.messages)
            for k, v in update.items():
                setattr(new, k, v)
            return new

    msgs = [{"role": "user", "content": "hello"}]
    state = _FakeState(msgs)
    result = compress_state(state)
    assert hasattr(result, "messages")


def test_compress_state_attribute_style_no_copy():
    """compress_state returns original state when copy() raises AttributeError."""
    from distil.integrations.langgraph import compress_state

    class _BadState:
        messages = [{"role": "user", "content": "hi"}]

        def copy(self, **kw):
            raise AttributeError("no copy")

    state = _BadState()
    result = compress_state(state)
    assert result is state


def test_compress_state_attribute_style_missing_key():
    """compress_state returns state untouched when attribute doesn't exist."""
    from distil.integrations.langgraph import compress_state

    class _NoMessages:
        pass

    state = _NoMessages()
    result = compress_state(state)
    assert result is state


# ===========================================================================
# distil/speculative.py — uncovered branches
# ===========================================================================


def test_speculative_controller_infeasible_always_full():
    """SpeculativeController.decide returns 'full' when feasible=False."""
    from distil.speculative import SpeculativeController

    ctrl = SpeculativeController(
        threshold=0.5,
        certified_miss_rate=0.5,
        escalation_rate=1.0,
        alpha=0.05,
        n=10,
        feasible=False,
    )
    assert ctrl.decide(0.0) == "full"
    assert ctrl.decide(1.0) == "full"


def test_calibrate_speculative_empty_scores():
    """calibrate_speculative on empty input returns infeasible controller."""
    from distil.speculative import calibrate_speculative

    ctrl = calibrate_speculative([], [])
    assert ctrl.feasible is False
    assert ctrl.n == 0


def test_calibrate_speculative_all_diverged():
    """calibrate_speculative handles all-diverged scenario (miss_rate high)."""
    from distil.speculative import calibrate_speculative

    # All turns diverged; escalation-all is the only certified option.
    scores = [0.9, 0.8, 0.7, 0.6]
    diverged = [1, 1, 1, 1]
    ctrl = calibrate_speculative(scores, diverged, alpha=0.05)
    # Should be feasible (escalate everything gives 0 misses).
    assert ctrl.feasible is True


# ===========================================================================
# distil/fidelity.py — uncovered branches
# ===========================================================================


def test_sha_is_used_internally():
    """_sha returns the SHA-256 hex of the input text."""
    from distil.fidelity import _sha
    import hashlib

    text = "hello world"
    assert _sha(text) == hashlib.sha256(text.encode()).hexdigest()


def test_numeric_precision_invalid_json():
    """numeric_precision_preserved returns False for invalid JSON input."""
    from distil.fidelity import numeric_precision_preserved

    assert numeric_precision_preserved("not json", '{"a": 1}') is False


def test_sha_manifest_returns_per_block_hashes():
    """sha_manifest returns a dict mapping block ids to SHA-256 hashes."""
    from distil.fidelity import sha_manifest
    from distil.trajectory import Block, Kind, Stability
    import hashlib

    blocks = [
        Block(id="a", kind=Kind.SYSTEM, text="hello", stability=Stability.STABLE),
        Block(id="b", kind=Kind.USER, text="world", stability=Stability.VOLATILE),
    ]
    manifest = sha_manifest(blocks)
    assert manifest["a"] == hashlib.sha256(b"hello").hexdigest()
    assert manifest["b"] == hashlib.sha256(b"world").hexdigest()


# ===========================================================================
# distil/native.py — Rust path is unavailable; cover the BACKEND value
# ===========================================================================


def test_native_backend_is_python_when_rust_absent():
    """BACKEND is 'python' when distil_core is not installed (our CI)."""
    from distil import native

    # In this CI environment, distil_core is not built.
    # The Python fallback is loaded — verify BACKEND reflects that.
    if native.BACKEND == "python":
        assert callable(native.minify_json)
        assert callable(native.collapse_runs)
        assert callable(native.count_tokens)
    else:
        # Rust is available; at least the BACKEND value is valid.
        assert native.BACKEND == "rust"


# ===========================================================================
# distil/cachedelta.py — remaining uncovered branches
# ===========================================================================


def test_rewrite_tool_texts_string_content_changed():
    """_rewrite_tool_texts returns an updated msg when string content is transformed (line 253)."""
    from distil.cachedelta import _rewrite_tool_texts

    msg = {"role": "user", "content": "hello"}
    result = _rewrite_tool_texts(msg, str.upper)
    assert result is not msg
    assert result["content"] == "HELLO"


def test_rewrite_tool_texts_sub_list_with_unchanged_item():
    """_rewrite_tool_texts appends non-text sub-items unchanged via line 282."""
    from distil.cachedelta import _rewrite_tool_texts

    # tool_result with list content: one text sub-item (transforms) + one non-text (line 282)
    text_sub = {"type": "text", "text": "hello"}
    image_sub = {"type": "image", "data": "base64abc"}  # not 'text' type → line 282
    msg = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": [text_sub, image_sub]}],
    }
    result = _rewrite_tool_texts(msg, str.upper)
    new_content = result["content"][0]["content"]
    assert new_content[0]["text"] == "HELLO"
    assert new_content[1] is image_sub  # unchanged sub-item passed through at line 282


# cachedelta line 201: dead code.  _exact_marker is ~163 chars; text must be >= 400 chars
# (_MIN_CHARS) to reach the marker check, so the marker is always shorter than text.
# The `return text` branch after `if len(exact) < len(text):` is structurally unreachable.


# ===========================================================================
# distil/adapters/gemini.py — remaining uncovered branches (lines 92, 94)
# ===========================================================================


def test_compress_json_value_list_item_changed(monkeypatch):
    """Line 92: changed=True when a list item is compressed to a different string."""
    from distil.adapters import gemini as _gemini
    from distil.adapters.anthropic import RestoreStore

    compressed = "<<digest-abc>>"
    # Force the string compressor to report a change for large inputs.
    monkeypatch.setattr(
        _gemini,
        "_compress_tool_result_text",
        lambda t, s, v, r=False: compressed if len(t) > 100 else t,
    )

    store = RestoreStore()
    big = "x" * 600
    result = _gemini._compress_json_value([big], store, verbatim=False)
    assert result == [compressed]  # out_list returned because changed=True (line 92)


def test_compress_json_value_primitive_passthrough():
    """Line 94: non-str/dict/list values are returned unchanged."""
    from distil.adapters.gemini import _compress_json_value
    from distil.adapters.anthropic import RestoreStore

    store = RestoreStore()
    assert _compress_json_value(42, store, verbatim=False) == 42
    assert _compress_json_value(True, store, verbatim=False) is True
    assert _compress_json_value(None, store, verbatim=False) is None


# ===========================================================================
# distil/codec/learned.py — train_from_corpus (lines 322-345)
# ===========================================================================


def test_train_from_corpus_mocked(tmp_path, monkeypatch):
    """train_from_corpus runs its full body when build_dataset and _split are mocked."""
    import distil.codec.learned as _learned
    from distil.codec.features import FEATURE_NAMES

    n = len(FEATURE_NAMES)
    fake_samples = [([float(i % 2)] + [0.0] * (n - 1), float(i % 2)) for i in range(10)]
    fake_lines = [f"line_{i}" for i in range(10)]
    train_s, test_s = fake_samples[:8], fake_samples[8:]

    monkeypatch.setattr(_learned, "build_dataset", lambda: (fake_samples, fake_lines))
    monkeypatch.setattr(_learned, "_split", lambda s, r, tf: (train_s, test_s))
    # Use fast training (epochs not passed through, but mock ignores kw).
    monkeypatch.setattr(_learned, "train", lambda s, **kw: [0.0] * n)
    monkeypatch.setattr(_learned, "DEFAULT_WEIGHTS_PATH", tmp_path / "weights.json")

    result = _learned.train_from_corpus()
    assert result["train_size"] == 8
    assert result["test_size"] == 2
    assert "accuracy" in result
    assert (tmp_path / "weights.json").exists()


def test_train_from_corpus_empty_train_raises(monkeypatch):
    """train_from_corpus raises RuntimeError when _split returns an empty train set (line 326)."""
    import distil.codec.learned as _learned
    from distil.codec.features import FEATURE_NAMES

    n = len(FEATURE_NAMES)
    fake_samples = [([0.0] * n, 0.0)]
    monkeypatch.setattr(_learned, "build_dataset", lambda: (fake_samples, ["line"]))
    monkeypatch.setattr(_learned, "_split", lambda s, r, tf: ([], s))  # empty train

    with pytest.raises(RuntimeError, match="Training set is empty"):
        _learned.train_from_corpus()


def test_train_from_corpus_empty_test_raises(monkeypatch):
    """train_from_corpus raises RuntimeError when _split returns an empty test set (line 328)."""
    import distil.codec.learned as _learned
    from distil.codec.features import FEATURE_NAMES

    n = len(FEATURE_NAMES)
    fake_samples = [([0.0] * n, 0.0)]
    monkeypatch.setattr(_learned, "build_dataset", lambda: (fake_samples, ["line"]))
    monkeypatch.setattr(_learned, "_split", lambda s, r, tf: (s, []))  # empty test

    with pytest.raises(RuntimeError, match="Test set is empty"):
        _learned.train_from_corpus()


# ===========================================================================
# distil/codec/transformer.py — from_pretrained success path (lines 246-260)
# ===========================================================================


def test_transformer_from_pretrained_success():
    """from_pretrained creates a TransformerKeepModel and the encode closure is callable (lines 246-260)."""
    import types
    from distil.codec.transformer import TransformerKeepModel

    # Fake onnxruntime module
    fake_ort = types.ModuleType("onnxruntime")
    fake_session = MagicMock()
    fake_input = MagicMock()
    fake_input.name = "input_ids"
    fake_session.get_inputs.return_value = [fake_input]
    # Configure run() to return valid logits: shape [1, 2-tokens, 2-labels]
    # run() returns [logits_3d] where logits_3d has shape [1, seq, num_labels]
    fake_session.run.return_value = [[[[0.2, 0.8], [0.3, 0.7]]]]
    fake_ort.InferenceSession = MagicMock(return_value=fake_session)

    # Fake transformers module
    fake_transformers = types.ModuleType("transformers")
    fake_tok_instance = MagicMock()
    fake_tok_instance.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=fake_tok_instance)

    with patch.dict(sys.modules, {"onnxruntime": fake_ort, "transformers": fake_transformers}):
        model = TransformerKeepModel.from_pretrained("/fake/model.onnx", "/fake/tok")

    assert isinstance(model, TransformerKeepModel)
    fake_ort.InferenceSession.assert_called_once_with("/fake/model.onnx")
    fake_transformers.AutoTokenizer.from_pretrained.assert_called_once_with("/fake/tok")

    # Call score to exercise the encode closure body (lines 251-258):
    # _encode("hello world") → tokenizer("hello world", ...) → {"input_ids": ..., ...}
    score = model.score("hello world", "tool_output")
    assert 0.0 <= score <= 1.0


# ===========================================================================
# distil/ingest.py — remaining uncovered branches (lines 124, 164, 290)
# ===========================================================================


def test_ingest_anthropic_list_content_non_dict_skipped():
    """Non-dict items in list content are skipped via `continue` at line 124."""
    from distil.ingest import ingest_anthropic_request

    body = {
        "messages": [
            {"role": "user", "content": ["plain string", {"type": "text", "text": "hello"}]}
        ]
    }
    blocks = ingest_anthropic_request(body)
    # "plain string" is not a dict → line 124 continue; only the text block is kept
    assert any(b.text == "hello" for b in blocks)
    assert all(b.text != "plain string" for b in blocks)


def test_ingest_anthropic_tool_result_non_str_non_list_content():
    """tool_result with non-str/non-list content falls back to str() at line 164."""
    from distil.ingest import ingest_anthropic_request
    from distil.trajectory import Kind

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": 42}],
            }
        ]
    }
    blocks = ingest_anthropic_request(body)
    assert len(blocks) == 1
    assert blocks[0].text == "42"
    assert blocks[0].kind == Kind.TOOL_OUTPUT


def test_openai_content_to_text_non_str_non_list():
    """_openai_content_to_text falls back to str() for non-str/non-list content (line 290)."""
    from distil.ingest import _openai_content_to_text

    assert _openai_content_to_text(42) == "42"
    assert _openai_content_to_text(None) == ""  # falsy → ""
    assert _openai_content_to_text({"k": "v"}) == str({"k": "v"})


# ===========================================================================
# distil/onboard.py — remaining uncovered branches (lines 73, 106)
# ===========================================================================


def test_install_method_pip_fallback(monkeypatch):
    """install_method returns 'pip' when no pipx/uv/homebrew marker is in the path (line 73)."""
    import distil as _d
    from distil import onboard

    monkeypatch.setattr(
        _d,
        "__file__",
        "/home/user/.venv/lib/python3.12/site-packages/distil/__init__.py",
    )
    monkeypatch.setattr(onboard.shutil, "which", lambda _: None)
    result = onboard.install_method()
    assert result == "pip"


def test_latest_pypi_version_success(monkeypatch):
    """latest_pypi_version returns the version string on a successful PyPI response (line 106)."""
    import io
    import urllib.request
    from distil import onboard

    data = json.dumps({"info": {"version": "9.9.9"}})

    class _FakeCtx:
        def __enter__(self):
            return io.StringIO(data)

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeCtx())
    assert onboard.latest_pypi_version() == "9.9.9"


# ===========================================================================
# distil/online.py — load_corpus() default-entries branches (lines 83, 240, 313)
# ===========================================================================


def test_collect_causal_labels_default_entries(monkeypatch):
    """collect_causal_labels() calls load_corpus() when entries is not given (line 83)."""
    import distil.online as _online

    monkeypatch.setattr(_online, "load_corpus", lambda: [])
    labels = _online.collect_causal_labels()  # no entries arg → load_corpus() at line 83
    assert labels == []


def test_certify_promotion_default_entries(monkeypatch):
    """certify_promotion() calls load_corpus() when entries is not given (line 240)."""
    import distil.online as _online
    from distil.codec.learned import FEATURE_NAMES

    monkeypatch.setattr(_online, "load_corpus", lambda: [])
    # Empty corpus → nothing to certify → returns True (all entries pass vacuously)
    result = _online.certify_promotion([0.0] * len(FEATURE_NAMES))
    assert result is True


def test_online_round_default_entries(monkeypatch):
    """online_round() calls load_corpus() when entries is not given (line 313)."""
    import distil.online as _online
    from distil.codec.learned import FEATURE_NAMES

    monkeypatch.setattr(_online, "load_corpus", lambda: [])
    monkeypatch.setattr(
        _online,
        "collect_causal_labels",
        lambda entries: [("line x", 1), ("line y", 0)],
    )
    monkeypatch.setattr(
        _online,
        "retrain",
        lambda labels, **kw: {"weights": [0.0] * len(FEATURE_NAMES), "accuracy": 1.0, "f1": 1.0},
    )
    monkeypatch.setattr(_online, "certify_promotion", lambda w, entries, **kw: True)

    result = _online.online_round()  # no entries → load_corpus() at line 313
    assert "certified" in result


# ===========================================================================
# distil/replay/prompts.py — remaining uncovered branches (lines 80-82, 134-135, 147)
# ===========================================================================


def test_parse_expand_json_error_falls_back_to_handle_scrape():
    """parse_expand falls back to handle-scraping when the matched JSON is malformed (lines 80-82)."""
    from distil.replay.prompts import parse_expand

    # Matches _EXPAND pattern ({..."expand"...}) but is not valid JSON.
    # _HANDLE will find the 8-hex token 'deadbeef' inside the match.
    text = 'prior text {"expand": [deadbeef]} trailing'
    result = parse_expand(text)
    assert result == ["deadbeef"]


def test_fingerprint_from_args_invalid_json_string():
    """fingerprint_from_args falls back to parse_fingerprint on invalid JSON (lines 134-135)."""
    from distil.replay.prompts import fingerprint_from_args

    result = fingerprint_from_args("not valid json {{{")
    assert result == "<no-decision>"


def test_parse_fingerprint_empty_string():
    """parse_fingerprint returns '<no-decision>' for empty input (line 147)."""
    from distil.replay.prompts import parse_fingerprint

    assert parse_fingerprint("") == "<no-decision>"


# native.py line 40 (collapse_runs Rust impl) and speculative.py line 89
# (the post-loop fallback) are both documented as unreachable without distil_core
# and by the mathematical guarantee that escalate-all always certifies, respectively.

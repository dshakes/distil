"""VS Code Copilot Chat: reachable through BYOK Custom Endpoint (re-checked 2026-09-25)."""

from __future__ import annotations

import json

from distil.cli import main, vscode_entry
from distil.targets import UNREACHABLE


def test_entry_matches_the_documented_custom_endpoint_shape():
    (provider,) = vscode_entry(9000)
    assert provider["vendor"] == "customendpoint" and provider["apiType"] == "messages"
    assert provider["apiKey"].startswith("${input:"), "never a literal key"
    (model,) = provider["models"]
    assert model["url"] == "http://127.0.0.1:9000/v1/messages"
    assert {"id", "name", "toolCalling", "maxInputTokens", "maxOutputTokens"} <= set(model)


def test_setup_vscode_prints_parseable_json(capsys):
    assert main(["setup", "--vscode", "--port", "8123"]) == 0
    out = capsys.readouterr().out
    body = out[out.index("[") : out.rindex("]") + 1]
    assert json.loads(body) == vscode_entry(8123)
    assert "Custom Endpoint" in out


def test_catalogue_no_longer_says_no_override():
    (code,) = [t for t in UNREACHABLE if t.key == "code"]
    assert code.verified == "2026-09-25"
    assert "customendpoint" in code.knob
    assert "language-models" in code.doc_url

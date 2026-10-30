"""distil.integrations.crewai — duck-typed tool wrapper (crewai is not installed)."""

from __future__ import annotations

import json

import pytest

from distil.integrations.crewai import compress_tool_output, wrap_tools

BIG = json.dumps([{"id": i, "status": "ok", "msg": "x" * 40} for i in range(400)])


class FakeTool:
    name = "search"

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def _run(self, *args, **kwargs):
        self.calls += 1
        return self.result

    def run(self, *args, **kwargs):  # CrewAI's BaseTool.run delegates to _run
        return self._run(*args, **kwargs)


def test_big_string_result_is_compressed_and_original_untouched():
    tool = FakeTool(BIG)
    (wrapped,) = wrap_tools([tool])
    out = wrapped.run(q="a")
    assert isinstance(out, str) and len(out) < len(BIG)
    assert wrapped is not tool and wrapped.name == "search"
    assert tool.run() == BIG  # the caller's tool still returns raw output


def test_small_and_non_string_results_pass_through():
    (small,) = wrap_tools([FakeTool("ok")])
    assert small.run() == "ok"
    (obj,) = wrap_tools([FakeTool({"a": 1})])
    assert obj.run() == {"a": 1}


def test_pydantic_style_tool_uses_model_copy():
    class PydLike(FakeTool):
        def model_copy(self):
            c = PydLike(self.result)
            return c

    (w,) = wrap_tools([PydLike(BIG)])
    assert len(w.run()) < len(BIG)


def test_non_tool_is_rejected():
    with pytest.raises(TypeError, match="not a CrewAI tool"):
        wrap_tools([object()])


def test_compress_tool_output_never_grows():
    assert len(compress_tool_output(BIG)) <= len(BIG)
    assert compress_tool_output("") == ""

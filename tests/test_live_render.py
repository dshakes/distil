"""Live renders never show the corpus's DECISION: annotations.

Those lines are the offline DeterministicRunner's answer key. A live model shown them
is reading the answer, so every live runner's prompt drops them — while the offline
oracle, which reads blocks directly, still sees every one.
"""

from __future__ import annotations

from types import SimpleNamespace

from distil.compress.strategies import distil as distil_strategy
from distil.corpus import load_corpus
from distil.replay import prompts
from distil.replay.anthropic_runner import AnthropicRunner
from distil.replay.runner import DeterministicRunner


def _turns():
    return [t for e in load_corpus() for t in e.trajectory.turns]


def _has_annotation(text: str) -> bool:
    return any(ln.lstrip().startswith("DECISION:") for ln in text.split("\n"))


def test_corpus_carries_annotations_for_the_oracle() -> None:
    # the premise: without planted lines this whole test would be vacuous
    assert any(_has_annotation(b.text) for t in _turns() for b in t.blocks)


def test_live_renders_have_no_decision_lines() -> None:
    for t in _turns():
        for fn in (prompts.render, prompts.decision_prompt, prompts.expand_prompt):
            system, user = fn(t.blocks)
            assert not _has_annotation(system) and not _has_annotation(user)


def test_deterministic_runner_still_reads_them() -> None:
    decided = [DeterministicRunner().decide(t.blocks) for t in _turns()]
    assert all(d != "<no-op>" for d in decided)


def test_full_and_compressed_arms_are_stripped_identically() -> None:
    """Stripping must not be a difference between arms: the annotations removed from
    the compressed render are exactly the ones removed from the full render."""
    for t in _turns():
        comp = distil_strategy(t.blocks, t.index)

        def dropped(blocks):
            return sorted(
                ln.strip()
                for b in blocks
                for ln in b.text.split("\n")
                if ln.lstrip().startswith("DECISION:")
            )

        assert dropped(t.blocks) == dropped(comp)
        assert not _has_annotation(prompts.render(comp)[1])


def test_anthropic_runner_request_has_no_decision_lines() -> None:
    seen: list[dict] = []

    class Client:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kw):
            seen.append(kw)
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input={"action": "a", "target": "b"})]
            )

    turn = next(t for t in _turns() if any(_has_annotation(b.text) for b in t.blocks))
    AnthropicRunner(client=Client()).decide(turn.blocks)
    req = seen[0]
    assert not _has_annotation(req["system"])
    assert not _has_annotation(str(req["messages"]))


def test_prose_mentioning_the_word_is_kept() -> None:
    text = "the DECISION: markers are an offline convention\nDECISION: roll back"
    assert prompts.strip_annotations(text) == "the DECISION: markers are an offline convention"


def test_runner_effort_resolves_per_model() -> None:
    from distil.replay.anthropic_runner import DEFAULT_EFFORT, AnthropicRunner

    # omitted: the measured pair's effort for the default model, none for haiku (400s on it)
    assert AnthropicRunner().effort == DEFAULT_EFFORT
    assert AnthropicRunner(model="claude-haiku-4-5").effort is None
    # explicit values win, including an explicit None
    assert AnthropicRunner(effort=None).effort is None
    assert AnthropicRunner(model="claude-haiku-4-5", effort="high").effort == "high"


def test_grader_provenance_records_model_and_effort() -> None:
    from types import SimpleNamespace

    from distil.evalrecord import describe_grader

    g = describe_grader("anthropic", SimpleNamespace(model="claude-sonnet-5-5", effort="low"))
    assert g["model"] == "claude-sonnet-5-5" and g["effort"] == "low"
    assert "model" not in describe_grader("deterministic")


def test_render_version_is_exported() -> None:
    from distil.replay import prompts

    assert prompts.RENDER_VERSION

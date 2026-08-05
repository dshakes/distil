"""Every documented alternative must actually fire.

Two Blocking findings in this PR were the same bug in different places: a `\\b`
positioned where it can never match, so one alternative of an alternation silently
never fired. `[x]` checkboxes were invisible to the continuation probe; `Objective:`
never produced an obligation, leaving `dropped_work` at zero for a dropped task.

Both were found by an auditor trying a marker I had not tried. Fixing the reported
instance and leaving its neighbours is how that class survives, so this file
enumerates EVERY alternative of EVERY alternation in the probe modules and asserts
each one matches something.

A probe whose pattern silently never fires reports a perfect score on text it never
parsed — which is the failure mode the probes themselves exist to catch.
"""

from __future__ import annotations

import pytest

from distil.artifacts import Op, extract_ops
from distil.continuation import Status, extract_obligations
from distil.overclaim import _HEDGE_CLASSES, extract_claims


class TestContinuationMarkers:
    @pytest.mark.parametrize(
        "text",
        [
            "- [ ] add the serialiser now",
            "- TODO: add the serialiser now",
            "- FIXME: add the serialiser now",
            "- NEXT: add the serialiser now",
            "- REMAINING: add the serialiser now",
            "- PENDING: add the serialiser now",
            "STILL need to add the serialiser",
            "STILL to add the serialiser now",
        ],
    )
    def test_every_pending_marker_fires(self, text: str) -> None:
        obs = extract_obligations(text)
        assert obs, f"no obligation extracted from {text!r}"
        assert any(o.status is Status.PENDING for o in obs)

    @pytest.mark.parametrize(
        "text",
        [
            "- [x] added the parser today",
            "- [X] added the parser today",
            "- DONE: added the parser today",
            "- COMPLETE: added the parser today",
            "- COMPLETED: added the parser today",
            "- FIXED: added the parser today",
            "- ✓ added the parser today",
            "- ✔ added the parser today",
        ],
    )
    def test_every_done_marker_fires(self, text: str) -> None:
        obs = extract_obligations(text)
        assert obs, f"no obligation extracted from {text!r}"
        assert any(o.status is Status.DONE for o in obs)

    @pytest.mark.parametrize(
        "text",
        [
            "Your task is to add the retry wrapper",
            "Your task is add the retry wrapper",
            "The goal is to migrate the billing module",
            "The goal is migrate the billing module",
            "Objective: ship the parser rewrite",
            "You must remove the legacy client",
        ],
    )
    def test_every_goal_marker_fires(self, text: str) -> None:
        assert extract_obligations(text), f"no obligation extracted from {text!r}"


class TestArtifactVerbs:
    @pytest.mark.parametrize(
        "text,op",
        [
            ('Write(file_path="src/a.py")', Op.CREATE),
            ('Edit(file_path="src/a.py")', Op.MODIFY),
            ('Read(file_path="src/a.py")', Op.READ),
            ("rm src/a.py", Op.DELETE),
            ("rm -rf src/a.py", Op.DELETE),
            ("git rm src/a.py", Op.DELETE),
            ("cat src/a.py", Op.READ),
            ("head src/a.py", Op.READ),
            ("tail src/a.py", Op.READ),
            ("less src/a.py", Op.READ),
            ("touch src/a.py", Op.CREATE),
            ("created src/a.py", Op.CREATE),
            ("wrote src/a.py", Op.CREATE),
            ("added src/a.py", Op.CREATE),
            ("generated src/a.py", Op.CREATE),
            ("modified src/a.py", Op.MODIFY),
            ("edited src/a.py", Op.MODIFY),
            ("updated src/a.py", Op.MODIFY),
            ("patched src/a.py", Op.MODIFY),
            ("changed src/a.py", Op.MODIFY),
            ("deleted src/a.py", Op.DELETE),
            ("removed src/a.py", Op.DELETE),
            ("dropped src/a.py", Op.DELETE),
            ("unlinked src/a.py", Op.DELETE),
            ("read src/a.py", Op.READ),
            ("opened src/a.py", Op.READ),
            ("examined src/a.py", Op.READ),
            ("inspected src/a.py", Op.READ),
            ("viewed src/a.py", Op.READ),
            ("+++ b/src/a.py", Op.MODIFY),
            # apply_patch envelopes — the edit format used by Codex and several other
            # agent harnesses. Missing these made an entire agent family's file
            # operations invisible, so their trajectories scored as an untouched
            # workspace.
            ("*** Add File: src/a.py", Op.CREATE),
            ("*** Delete File: src/a.py", Op.DELETE),
            ("*** Update File: src/a.py", Op.MODIFY),
            ("sed -i 's/x/y/' src/a.py", Op.MODIFY),
            ("echo hi > src/a.py", Op.CREATE),
            ("echo hi >> src/a.py", Op.MODIFY),
        ],
    )
    def test_every_verb_fires(self, text: str, op: Op) -> None:
        assert ("src/a.py", op) in extract_ops(text), f"{text!r} did not yield {op}"

    @pytest.mark.parametrize(
        "text,src_op,dst_op",
        [
            ("mv src/a.py src/b.py", Op.DELETE, Op.CREATE),
            ("git mv src/a.py src/b.py", Op.DELETE, Op.CREATE),
            ("cp src/a.py src/b.py", Op.READ, Op.CREATE),
        ],
    )
    def test_two_path_verbs_assign_both_ends(self, text: str, src_op: Op, dst_op: Op) -> None:
        """`mv old new` is a delete AND a create. Scoring either end alone leaves the
        ledger describing a workspace that never existed."""
        ops = extract_ops(text)
        assert ("src/a.py", src_op) in ops, f"{text!r} lost the source op"
        assert ("src/b.py", dst_op) in ops, f"{text!r} lost the destination op"

    @pytest.mark.parametrize(
        "text",
        [
            "def f() -> Dict:",  # a type hint is not a redirect into its return type
            "> Some note in a markdown quote",  # a blockquote has no command before it
            "  > Indented quote about Dockerfile",
            ">> Nested quote text",
            "curl https://x 2>&1",  # stderr redirect names no path
            "if Rate > Burst:",  # the comparison operator, not a redirect
            "Accept > 100 requests",
            "<style>.row>.c-0{gap:0}</style>",  # CSS child combinator
            "<title>Rate limits</title>",  # HTML tag close
        ],
    )
    def test_redirect_lookalikes_create_no_artifact(self, text: str) -> None:
        """`>` is a redirect, a comparison, a quote marker, a tag close and a CSS
        combinator, and agent transcripts contain all five.

        Every case here was measured producing a phantom artifact on the real corpus:
        adding the redirect verbs took a 7-artifact corpus to 26, inventing files
        named "Accept", "Error", "The" and ".c-0" — each then scored as perfectly
        preserved. A probe that reports fidelity on files no agent ever touched is
        exactly the silent-and-green failure this module exists to detect.
        """
        assert extract_ops(text) == [], f"{text!r} invented an artifact"

    def test_the_corpus_yields_only_its_real_artifacts(self) -> None:
        """The measurement that caught the phantoms, kept as the guard.

        Per-pattern unit tests all passed while the corpus ledger was wrong, because
        the phantoms came from text no unit test contained. This asserts the whole
        pipeline against the whole corpus.
        """
        from distil.artifacts import build_ledger
        from distil.corpus import load_corpus

        texts = [b.text for e in load_corpus() for t in e.trajectory.turns for b in t.blocks]
        paths = set(build_ledger(texts).state)
        assert paths == {
            "myapp/pagination.py",
            "net/client.py",
            "net/legacy_retry.py",
            "net/retry.py",
            "net/scratch_bench.py",
            "pagination.py",
            "tests/test_retry.py",
        }, f"ledger drifted from the corpus's real artifacts: {sorted(paths)}"

    @pytest.mark.parametrize(
        "path",
        [
            "src/a.py",
            "a/b/c.tar.gz",
            "x.test.tsx",
            ".env",
            ".gitignore",
            "Dockerfile",
            "Makefile",
            "LICENSE",
        ],
    )
    def test_every_documented_path_shape_fires(self, path: str) -> None:
        assert any(p == path for p, _ in extract_ops(f"created {path}")), f"{path!r} invisible"


class TestHedgeClasses:
    def test_every_hedge_term_fires(self) -> None:
        """Each term in each class must bind to an anchor. A term that never
        matches makes its whole class partially blind, and the resulting
        overclaim rate is quietly wrong rather than obviously broken."""
        dead: list[tuple[str, str]] = []
        for cls, terms in _HEDGE_CLASSES.items():
            for term in terms:
                claims = extract_claims(f"the timeout is {term} 4000 ms")
                if not any(c.hedge_class == cls for c in claims):
                    dead.append((cls, term))
        assert not dead, f"hedge terms that never fire: {dead}"

    def test_opposed_classes_are_real_and_symmetric(self) -> None:
        """Every opposed pair must name classes that exist and point at each other.

        A typo here fails open: the lookup misses, the inversion falls through to the
        "still hedged" branch, and a reversed bound is scored as faithful — silently,
        which is the failure mode these probes exist to catch.
        """
        from distil.overclaim import _OPPOSED

        for cls, opposite in _OPPOSED.items():
            assert cls in _HEDGE_CLASSES, f"{cls!r} is not a hedge class"
            assert opposite in _HEDGE_CLASSES, f"{opposite!r} is not a hedge class"
            assert _OPPOSED[opposite] == cls, f"{cls!r}/{opposite!r} is not symmetric"

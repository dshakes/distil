"""Overclaim — hedge preservation under compression.

The load-bearing case is `test_the_stripped_hedge`: byte-identical value, missing
qualifier. Every recall metric scores that perfect; this one must not.
"""

from __future__ import annotations

from distil.overclaim import (
    Claim,
    OverclaimProbe,
    extract_claims,
    format_probe,
    score,
)


class TestExtraction:
    def test_approx_hedge_binds_to_number(self) -> None:
        claims = extract_claims("the timeout is approximately 4000 ms")
        assert Claim("4000", "approx") in claims

    def test_modal_hedge(self) -> None:
        assert any(c.hedge_class == "modal" for c in extract_claims('the "retry path" may fail'))

    def test_unhedged_number_is_not_a_claim(self) -> None:
        # Plain facts belong to retention, not here. Counting them would double-gate.
        assert extract_claims("the timeout is 4000 ms") == set()

    def test_distant_hedge_does_not_bind(self) -> None:
        text = "approximately one thing. " + "x" * 80 + " then 4000 ms exactly"
        assert not any(c.anchor == "4000" for c in extract_claims(text))

    def test_longest_hedge_wins(self) -> None:
        # "at least" must not be shredded into the "bound" term "least" plus noise,
        # nor "not sure" into "sure".
        assert any(c.hedge_class == "bound" for c in extract_claims("at least 5 retries"))

    def test_tilde_is_a_hedge(self) -> None:
        assert any(c.hedge_class == "approx" for c in extract_claims("~4000 ms"))

    def test_empty_text(self) -> None:
        assert extract_claims("") == set()


class TestScoring:
    def test_hedge_preserved(self) -> None:
        t = "the timeout is approximately 4000 ms"
        p = score(t, t)
        assert p.total == 1 and p.preserved == 1 and p.overclaimed == 0
        assert p.fidelity == 1.0

    def test_the_stripped_hedge(self) -> None:
        """The whole point: value identical, hedge gone."""
        p = score("the timeout is approximately 4000 ms", "the timeout is 4000 ms")
        assert p.overclaimed == 1 and p.preserved == 0
        assert p.overclaim_rate == 1.0

    def test_recall_metrics_would_score_that_perfect(self) -> None:
        original, compressed = "timeout approximately 4000 ms", "timeout 4000 ms"
        assert "4000" in compressed, "token recall = 100%"
        assert score(original, compressed).fidelity == 0.0, "hedge fidelity = 0%"

    def test_synonym_swap_is_not_an_overclaim(self) -> None:
        # "approximately" -> "about": still hedged, same class. Legitimate reshaping.
        p = score("approximately 4000 ms", "about 4000 ms")
        assert p.overclaimed == 0 and p.preserved == 1

    def test_class_change_still_counts_as_hedged(self) -> None:
        # "approximately 4000" -> "up to 4000": different class, still uncertain.
        p = score("approximately 4000 ms", "up to 4000 ms")
        assert p.overclaimed == 0

    def test_lost_value_is_not_an_overclaim(self) -> None:
        # If the anchor vanished entirely that is plain loss — retention's problem.
        # Counting it here too would double-penalise the same failure.
        p = score("approximately 4000 ms", "nothing survived")
        assert p.overclaimed == 0 and p.total == 0

    def test_underclaim_counted_separately(self) -> None:
        p = score("the timeout is 4000 ms", "the timeout is approximately 4000 ms")
        assert p.underclaimed == 1
        assert p.overclaimed == 0, "adding caution is not overclaiming"

    def test_modal_stripped(self) -> None:
        p = score('the "retry path" may fail', 'the "retry path" fail')
        assert p.overclaimed == 1

    def test_empty_inputs_are_vacuous(self) -> None:
        p = score("", "")
        assert p.total == 0 and p.fidelity == 1.0 and p.overclaim_rate == 0.0


class TestReporting:
    def test_add_accumulates(self) -> None:
        a = OverclaimProbe(total=2, preserved=1, overclaimed=1)
        a.add(OverclaimProbe(total=1, preserved=1, underclaimed=3))
        assert (a.total, a.preserved, a.overclaimed, a.underclaimed) == (3, 2, 1, 3)

    def test_format_names_the_failure(self) -> None:
        out = format_probe(OverclaimProbe(total=2, preserved=1, overclaimed=1))
        assert "uncertainty dropped" in out

    def test_format_nothing_to_grade(self) -> None:
        assert "nothing to grade" in format_probe(OverclaimProbe())

    def test_to_dict_is_content_free(self) -> None:
        d = score("approximately 4000 secretname", "4000 secretname").to_dict()
        assert "secretname" not in str(d)
        assert set(d) == {
            "total",
            "preserved",
            "overclaimed",
            "underclaimed",
            "fidelity",
            "overclaim_rate",
        }

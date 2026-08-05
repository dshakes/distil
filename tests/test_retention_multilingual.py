"""Multilingual fact retention.

Before these, the harness had two silent failures on non-Latin text:

  1. CJK-labelled facts (`超时=4000`) were never extracted, so recall was reported
     against a fact set that excluded them — a perfect score on text nobody looked at.
  2. A CJK fact that WAS present scored as lost, because Python's `\\w` is
     Unicode-aware: every boundary lookaround is unsatisfiable inside CJK, which is
     written without spaces.

Both directions are covered here, plus the ASCII behaviour they must not disturb.
"""

from __future__ import annotations

from distil.retention import _in_recovery, _min_len, extract_targets


class TestExtraction:
    def test_cjk_label_is_extracted(self) -> None:
        found = extract_targets("超时=4000")["numerics"]
        assert any("4000" in f for f in found), "a CJK-labelled fact must be probe-able"

    def test_cjk_unit_stays_attached(self) -> None:
        found = extract_targets("タイムアウト=4000毫秒")["numerics"]
        assert any("4000毫秒" in f for f in found), "the unit is part of the fact"

    def test_ascii_extraction_is_unchanged(self) -> None:
        found = extract_targets("invoices=88ms rows=1200")["numerics"]
        assert "invoices=88ms" in found and "rows=1200" in found

    def test_mid_token_guard_still_holds(self) -> None:
        # The earlier audit fix: `invoices=88` must not be clipped out of `88ms`.
        assert "invoices=88" not in extract_targets("invoices=88ms")["numerics"]


class TestMatching:
    def test_present_cjk_fact_is_not_reported_lost(self) -> None:
        # The bug: boundary lookarounds cannot match between two CJK word characters.
        assert _in_recovery("4000毫秒", "超时4000毫秒です") is True

    def test_absent_cjk_fact_is_still_lost(self) -> None:
        assert _in_recovery("4000毫秒", "まったく別のテキスト") is False

    def test_single_char_cjk_is_below_the_floor(self) -> None:
        # One han character is too weak a signal; crediting it would inflate recall.
        assert _in_recovery("毫", "x毫x") is False

    def test_two_char_cjk_is_above_the_floor(self) -> None:
        assert _in_recovery("毫秒", "超时4000毫秒") is True

    def test_ascii_boundary_discipline_is_unchanged(self) -> None:
        # Regression guard for the audit finding: "12" must not match inside a path.
        assert _in_recovery("12", "file-12.csv") is False
        assert _in_recovery("12", "value is 12 here") is True

    def test_mixed_script_uses_the_cjk_path(self) -> None:
        assert _in_recovery("timeout超时", "設定timeout超时です") is True


class TestLengthFloor:
    def test_floor_is_script_dependent(self) -> None:
        assert _min_len("毫秒") == 2
        assert _min_len("rows") == 4

    def test_mixed_script_takes_the_cjk_floor(self) -> None:
        assert _min_len("a毫") == 2

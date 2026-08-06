"""Prefix stability diagnostics.

The load-bearing test is `test_appending_a_turn_is_not_drift`. Providers cache a
BYTE prefix, so the diagnostic has to flatten a payload the way a provider lays it
out — stable blocks first, conversation last. Sorting the keys alphabetically put
`messages` ahead of `system`, and every healthy turn then reported drift: a warning
that fires on normal behaviour is one people turn off, which is worse than no
warning at all.
"""

from __future__ import annotations

from distil import prefix

_STABLE_SYSTEM = [{"type": "text", "text": "You are an agent. " + "rules " * 200}]


def _turn(system, *contents: str) -> dict:
    return {
        "system": system,
        "messages": [{"role": "user", "content": c} for c in contents],
    }


class TestDriftDetection:
    def test_appending_a_turn_is_not_drift(self) -> None:
        """The prefix GREW. That is what a conversation does."""
        before = _turn(_STABLE_SYSTEM, "first")
        after = _turn(_STABLE_SYSTEM, "first", "second")
        report = prefix.analyse(before, after)
        assert not report.prefix_changed
        assert report.cacheable and report.stable_bytes > 1000

    def test_a_volatile_system_prompt_is_drift(self) -> None:
        """A timestamp at the front invalidates everything after it."""
        after = _turn([{"type": "text", "text": "You are an agent at 12:05. " + "rules " * 200}])
        report = prefix.analyse(_turn(_STABLE_SYSTEM, "first"), after)
        assert report.prefix_changed
        assert report.break_ratio < 0.1, "the break is near the front, where it costs most"

    def test_a_reordered_tool_list_is_drift(self) -> None:
        """Same set, different order: a byte prefix does not care that it is a set."""
        a = {"system": _STABLE_SYSTEM, "tools": [{"name": "alpha"}, {"name": "beta"}]}
        b = {"system": _STABLE_SYSTEM, "tools": [{"name": "beta"}, {"name": "alpha"}]}
        assert prefix.analyse(a, b).prefix_changed

    def test_a_first_turn_is_not_drift(self) -> None:
        """Nothing to have drifted from. Flagging it marks every session start a miss."""
        report = prefix.analyse(None, _turn(_STABLE_SYSTEM, "first"))
        assert not report.prefix_changed and report.stable_bytes > 0

    def test_moving_the_cache_marker_is_not_drift(self) -> None:
        """`cache_control` is a boundary marker, not content. Re-placing it is exactly
        what a cache-aware client does between turns."""
        a = {"system": [{"type": "text", "text": "x" * 800}]}
        b = {
            "system": [{"type": "text", "text": "x" * 800, "cache_control": {"type": "ephemeral"}}]
        }
        assert not prefix.analyse(a, b).prefix_changed


class TestReporting:
    def test_a_short_prefix_is_not_claimed_as_a_win(self) -> None:
        """Below every provider's minimum there is nothing to reuse, and saying
        'cacheable' would promise a hit the provider would refuse."""
        report = prefix.analyse(None, {"system": [{"type": "text", "text": "tiny"}]})
        assert not report.cacheable
        assert "too short" in prefix.format_report(report)

    def test_drift_names_the_cause_it_can_and_cannot_fix(self) -> None:
        after = _turn([{"type": "text", "text": "at 12:05 " + "rules " * 200}])
        text = prefix.format_report(prefix.analyse(_turn(_STABLE_SYSTEM, "a"), after))
        assert "DRIFT" in text
        assert "caller's own assembly" in text, (
            "distil never rewrites a stable block, so this is upstream — say so"
        )

    def test_the_report_is_content_free(self) -> None:
        secret = "SECRET-TOKEN-abc123"
        report = prefix.analyse(None, {"system": [{"type": "text", "text": secret * 60}]})
        blob = str(report.to_dict())
        assert secret not in blob, "counts and a hash escape; prompt text never does"

    def test_to_dict_is_stable_shape(self) -> None:
        keys = set(prefix.analyse(None, {"system": "x"}).to_dict())
        assert keys == {
            "stable_bytes",
            "stable_tokens_est",
            "stable_hash",
            "prefix_changed",
            "break_ratio",
            "total_bytes",
            "cacheable",
        }

    def test_stable_keys_exclude_messages(self) -> None:
        """`messages` grows every turn by design. Hashing it would make every request
        differ and report 100% drift on a perfectly healthy session."""
        assert "messages" not in prefix.STABLE_KEYS


class TestSessionSummary:
    """`summarise` folds ledger rows. Reads/writes are the provider's numbers;
    drift is our explanation of them. Neither may stand in for the other."""

    def test_a_held_prefix_reads_as_no_drift(self) -> None:
        rows = [
            {"prefix_hash": "aaa", "usage_cache_read": 900, "usage_cache_create": 100},
            {"prefix_hash": "aaa", "usage_cache_read": 900},
            {"prefix_hash": "aaa", "usage_cache_read": 900},
        ]
        s = prefix.summarise(rows)
        assert s.drifts == 0 and s.pairs == 2
        assert s.reported and s.hit_ratio == 2700 / 2800

    def test_a_thrashing_prefix_is_visible_as_drift(self) -> None:
        """Writing every turn and never reading is the pathology drift causes."""
        rows = [{"prefix_hash": h, "usage_cache_create": 900} for h in ("a", "b", "c")]
        s = prefix.summarise(rows)
        assert s.drifts == 2 and s.drift_ratio == 1.0
        assert s.hit_ratio == 0.0
        assert "re-bills" in prefix.format_summary(s)

    def test_rows_without_a_hash_are_not_counted_as_stable(self) -> None:
        """An unknown prefix is not a held one. Counting it as a clean pair reports a
        green sheet for a session that was never checked."""
        s = prefix.summarise([{"ts": 1}, {"ts": 2}, {"ts": 3}])
        assert s.requests == 3 and s.pairs == 0 and s.drifts == 0
        assert "not the same as 'no drift'" in prefix.format_summary(s)

    def test_legacy_rows_do_not_produce_a_hit_ratio(self) -> None:
        """Pre-1.41 rows summed reads and writes. A ratio derived from that sum would
        be a confident number with nothing behind it."""
        s = prefix.summarise([{"usage_cache_tokens": 5000}, {"usage_cache_tokens": 4000}])
        assert not s.reported and s.legacy_cache_tokens == 9000
        text = prefix.format_summary(s)
        assert "no hit ratio" in text
        assert "caching was active" in text, "the rows still prove caching happened"

    def test_no_cache_usage_at_all_says_so(self) -> None:
        s = prefix.summarise([{"usage_input_tokens": 100}])
        assert not s.reported and not s.legacy_rows
        assert "not reported by the provider" in prefix.format_summary(s)

    def test_the_summary_is_content_free(self) -> None:
        s = prefix.summarise([{"prefix_hash": "abc", "usage_cache_read": 5}])
        assert set(s.to_dict()) >= {"hit_ratio", "drift_ratio", "requests"}
        assert all(isinstance(v, (int, float, bool)) for v in s.to_dict().values())

    def test_the_hash_is_stable_across_processes(self) -> None:
        """A drift report is compared BETWEEN runs, so a salted hash would make every
        comparison meaningless."""
        import subprocess
        import sys

        code = (
            "from distil import prefix; print(prefix.analyse(None, {'system': 'abc'}).stable_hash)"
        )
        seen = {
            subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
            for _ in range(3)
        }
        assert len(seen) == 1, f"hash drifted across processes: {seen}"

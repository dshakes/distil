"""One session id, or the per-session state silently writes nothing.

The savings recorder mints a fallback session id when `distil wrap` did not set
one. Ledger rows carried that id; every session-scoped PATH resolves the id from
`os.environ`, which still had nothing. So on an always-on install — launchd and
systemd do not inherit a shell environment — the request records, the manifest
and the liveness marker were all no-ops, while the ledger looked healthy.

The visible symptom was `distil dissect` reporting `blocks=0` for sessions worth
more than a million tokens. The analysis was fine; its input was never recorded.

This is the third surface of one root cause. The other two (a status line calling
routed sessions "bypassing", ledger rows attributed to the proxy rather than the
wrap) were fixed where they showed. This tests the cause.
"""

from __future__ import annotations

import json
import os

from distil import ledger
from distil.runtime import RuntimeSavings as Savings


class TestTheIdIsExportedNotJustHeld:
    def test_a_minted_id_reaches_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        s = Savings(ledger_path=tmp_path / "savings.jsonl")
        assert s.session_id, "no id was minted"
        assert os.environ.get("DISTIL_SESSION") == s.session_id, (
            "the id went onto ledger rows but not into the environment, so every "
            "session_*_path() helper still resolves to None"
        )

    def test_the_path_helpers_resolve_to_that_same_id(self, monkeypatch, tmp_path):
        """The actual invariant: one id, and every consumer sees it."""
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        s = Savings(ledger_path=tmp_path / "savings.jsonl")

        for resolve in (
            ledger.session_marker_path,
            ledger.session_requests_path,
            ledger.session_manifest_path,
        ):
            p = resolve()
            assert p is not None, f"{resolve.__name__} still returns None — writes are dropped"
            assert s.session_id in p.name, (
                f"{resolve.__name__} resolved to a DIFFERENT session than the ledger rows"
            )

    def test_a_wrap_session_is_never_renamed(self, monkeypatch, tmp_path):
        """setdefault, not assignment. Overwriting would rename the session the
        agent and its status line are already using, splitting one session in two."""
        monkeypatch.setenv("DISTIL_SESSION", "s-from-wrap")
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        s = Savings(ledger_path=tmp_path / "savings.jsonl")
        assert s.session_id == "s-from-wrap"
        assert os.environ["DISTIL_SESSION"] == "s-from-wrap"

    def test_an_explicit_session_id_still_wins(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        s = Savings(session_id="explicit", ledger_path=tmp_path / "savings.jsonl")
        assert s.session_id == "explicit"


class TestPerRequestStateIsActuallyWritten:
    """The end the user sees: records on disk, and dissect able to read them."""

    def test_a_request_record_lands_on_disk(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        s = Savings(ledger_path=tmp_path / "savings.jsonl")

        ledger.append_session_request({"ts": 1.0, "blocks": [{"sig": "log", "tokens": 900}]})
        path = ledger.session_requests_path()
        assert path is not None and path.exists(), (
            "append_session_request silently no-opped — this is the bug, restated"
        )
        rec = json.loads(path.read_text().splitlines()[0])
        assert rec["blocks"][0]["sig"] == "log"
        assert s.session_id in path.name

    def test_dissect_sees_the_blocks_it_previously_reported_as_zero(self, monkeypatch, tmp_path):
        """The symptom that exposed all of this: blocks=0 on a large session."""
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        s = Savings(ledger_path=tmp_path / "savings.jsonl")

        ledger.append_session_request(
            {
                "ts": 1.0,
                "blocks": [
                    {"h": "aaaa1111", "sig": "log", "tokens": 5_000},
                    {"h": "bbbb2222", "sig": "diff", "tokens": 1_000},
                ],
            }
        )

        from distil.dissect import dissect

        d = dissect(s.session_id)
        assert d.blocks, "dissect still sees no blocks — the record did not reach it"
        kinds = dict((k, tok) for k, _n, tok in d.blocks_by_kind())
        assert kinds.get("log") == 5_000 and kinds.get("diff") == 1_000

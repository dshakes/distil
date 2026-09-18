"""The exit summary must not get slower the longer you use distil.

Every verdict in the proof ledger reads an append-only artifact that grows one row per
request and never shrinks — the maintainer's own ``receipts.jsonl`` is 83 MB. A render
that re-parses those files end to end is O(lifetime): fine on the fixtures, and a
multi-second stall plus a memory spike on a real install. These are the tests that fail
if that property is lost again.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc

import pytest

from distil import receipts as R
from distil.shadow import SIG_VERSION

RECEIPTS = 200_000
SHADOW = 50_000
#: Warm renders must be this many times cheaper than the cold render measured on the
#: SAME machine in the SAME test — an absolute budget is a statement about the runner.
WARM_RATIO = 3.0


def _traced() -> bool:
    """Is a line tracer or coverage monitor installed?"""
    if sys.gettrace() is not None:
        return True
    mon = getattr(sys, "monitoring", None)
    return bool(mon is not None and mon.get_tool(mon.COVERAGE_ID) is not None)


#: A wall-clock budget means nothing under coverage, which traces every line of the
#: hashing loop these tests exist to keep out of the render. The memory assertions below
#: still run everywhere; the timings run in CI's `gate` legs, which execute the suite
#: without `--cov`, so they are gated here rather than deleted.
needs_untraced = pytest.mark.skipif(_traced(), reason="wall-clock budget under coverage")


def _write_receipts(home, n: int) -> None:
    """A genuine chain: each receipt's ``prev`` is the real hash of the one before."""
    prev = R.GENESIS
    out = []
    for i in range(n):
        rec = R.Receipt(
            ts=1000.0 + i,
            request_id=f"req{i}",
            session="s1",
            model="claude-opus-4-8",
            mode="digest",
            tokens_original=1000,
            tokens_compressed=400,
            reversible=False,
            prev=prev,
        )
        prev = rec.sealed().hash
        out.append(json.dumps(rec.__dict__, sort_keys=True, separators=(",", ":")))
    (home / "receipts.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")


def _write_shadow(home, n: int) -> None:
    row = {
        "equivalent": True,
        "aa_equal": True,
        "ts": 1000.0,
        "kind": "paired",
        "sig": SIG_VERSION,
        "mode": "digest",
        "in_a": 1000,
        "in_b": 500,
        "out_a": 200,
        "out_b": 100,
    }
    line = json.dumps(row)
    (home / "shadow.jsonl").write_text((line + "\n") * n, encoding="utf-8")


SESSION = "s12345-9999"


def _write_savings(home) -> None:
    """One booked request, so the exit summary has a session to print a block about."""
    row = {
        "trajectory_id": "live-proxy",
        "model": "claude-opus-4-8",
        "turns": 1,
        "baseline_dollars": 0.04,
        "distil_dollars": 0.02,
        "baseline_input_tokens": 2000,
        "distil_input_tokens": 1000,
        "tokenizer": "heuristic",
        "ts": time.time(),
        "session": SESSION,
        "mode": "digest",
        "acct": 2,
    }
    (home / "savings.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.fixture()
def big(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_receipts(tmp_path, RECEIPTS)
    _write_shadow(tmp_path, SHADOW)
    _write_savings(tmp_path)
    return tmp_path


@needs_untraced
def test_the_whole_exit_render_is_fast_once_the_chain_is_checkpointed(big):
    """First render pays for the chain it has never seen; every render after resumes.

    Timed on ``build_ledger_text`` — the entire block a wrap exit prints, not just the
    verdicts — because that is what a user actually waits for, and because the session
    line and the verdicts read the same unbounded ledger. Timing only the verdicts hid a
    second full parse of ``shadow.jsonl`` sitting right next to them.
    """
    from distil.proof_ledger import build_ledger_text

    t0 = time.perf_counter()
    cold_text = build_ledger_text(SESSION, 0.0)  # cold: full chain verify + checkpoint written
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    text = build_ledger_text(SESSION, 0.0)
    warm = time.perf_counter() - t0

    assert cold_text is not None and text is not None
    assert f"{RECEIPTS} receipts, chain verified" in cold_text, cold_text
    assert f"{RECEIPTS} receipts, chain intact" in text, text
    assert warm * WARM_RATIO < cold, (
        f"warm exit render {warm * 1000:.0f} ms vs cold {cold * 1000:.0f} ms — the checkpoint is not being used"
    )


def test_the_exit_render_parses_the_shadow_ledger_once(big):
    """The structural twin of the timing tests, and the one that holds on a slow box.

    Three verdicts and the session line quote this file. Each of them loading it for
    itself is how the render went O(lifetime) in the first place, and a wall-clock budget
    only catches that on hardware fast enough for the rest.
    """
    from distil import shadow as S
    from distil.proof_ledger import build_ledger_text

    passes = 0
    real = S._rows

    def counted(path=None):
        nonlocal passes
        passes += 1
        yield from real(path)

    S._rows = counted
    try:
        build_ledger_text(SESSION, 0.0)  # cold
        passes = 0
        assert build_ledger_text(SESSION, 0.0) is not None
    finally:
        S._rows = real
    assert passes == 1, f"{passes} passes over shadow.jsonl in one exit render"


@needs_untraced
def test_appending_one_receipt_does_not_re_hash_the_chain(big):
    """The steady state is 'one more receipt since last time', and that is what it costs."""
    from distil.proof_ledger import proof_lines

    t0 = time.perf_counter()
    proof_lines()  # cold: the whole chain, once
    cold = time.perf_counter() - t0
    R.append(R.Receipt(1.0, "extra", "s1", "claude-opus-4-8", "digest", 10, 5, False))
    t0 = time.perf_counter()
    lines = dict(proof_lines())
    warm = time.perf_counter() - t0
    assert f"{RECEIPTS + 1} receipts, chain intact — 1 re-checked" in lines["receipts"]
    assert warm * WARM_RATIO < cold, (
        f"{warm * 1000:.0f} ms to fold one appended receipt vs {cold * 1000:.0f} ms cold"
    )


def test_verification_memory_does_not_scale_with_the_file(tmp_path, monkeypatch):
    """Peak allocation must stay far below the artifact being verified.

    Measured on a smaller file so the assertion is about the SHAPE (bounded, not
    proportional): ``read_text().splitlines()`` on this fixture allocates the file plus a
    list of every line, which is multiples of the ceiling below.
    """
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_receipts(tmp_path, 20_000)
    size = (tmp_path / "receipts.jsonl").stat().st_size
    assert size > 4_000_000, "fixture too small to distinguish streaming from slurping"

    tracemalloc.start()
    try:
        v = R.verify(full=True)  # the expensive direction: every receipt re-hashed
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert v.ok and v.total == 20_000
    assert peak < size // 4, f"peak {peak:,}B verifying a {size:,}B chain — not streaming"


def test_shadow_load_does_not_hold_the_whole_file(tmp_path, monkeypatch):
    """Same property for the other unbounded artifact the verdicts read."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    _write_shadow(tmp_path, 20_000)
    from distil.shadow import ShadowLedger

    size = (tmp_path / "shadow.jsonl").stat().st_size
    tracemalloc.start()
    try:
        led = ShadowLedger.load(current_only=True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert led.samples == 20_000
    # The tallies it keeps (paired diffs, per-replay costs) are the product, not the file.
    assert peak < size * 2, f"peak {peak:,}B reading a {size:,}B ledger"

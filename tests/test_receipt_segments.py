"""Rotated receipt segments + per-segment Merkle checkpoints.

The chain only grows (94 MB / 102k rows on the maintainer's box), so the active file is
sealed into immutable segments. What must survive that: one unbroken chain across every
boundary, a legacy single-file chain that keeps verifying and migrates without a byte
rewritten, a crash at any step of a seal, and head_hash staying a tail read. What it adds:
one segment verifiable alone, and one receipt verifiable from a proof and a root alone.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from distil import receipts as R


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    # The seal backoff is per-process state: one test's failed seal must not suppress
    # the next test's seals.
    monkeypatch.setattr(R, "_seal_retry_at", 0.0)
    return tmp_path


def _mk(i: int) -> R.Receipt:
    return R.Receipt(
        ts=1000.0 + i,
        request_id=f"req{i}",
        session="s1",
        model="claude-opus-4-8",
        mode="digest",
        tokens_original=1000,
        tokens_compressed=400,
        reversible=False,
        handles=[f"aaaaaaa{i % 10}"],
        certificate="ltt@digest",
    )


def _fill(n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        R.append(_mk(i))


@pytest.fixture()
def small_segments(monkeypatch):
    """Seal roughly every 5 receipts (one receipt line is ~300 bytes)."""
    monkeypatch.setattr(R, "SEGMENT_BYTES", 1500)


# ---------------------------------------------------------------------------
# Merkle primitives — checked against the definition, not against themselves
# ---------------------------------------------------------------------------


def test_every_leaf_of_every_small_tree_proves_and_nothing_else_does():
    for size in range(1, 34):
        leaves = [R._leaf(f"{i:064x}") for i in range(size)]
        root = R.merkle_root(leaves)
        for i in range(size):
            path = R._audit_path(leaves, i, 0, size)
            assert R.verify_inclusion(i, size, leaves[i], path, root), (size, i)
            assert not R.verify_inclusion((i + 1) % size, size, leaves[i], path, root) or size == 1
            assert not R.verify_inclusion(i, size, R._leaf("f" * 64), path, root)
            assert not R.verify_inclusion(i, size, leaves[i], path, b"\x00" * 32)
        assert not R.verify_inclusion(size, size, leaves[0], [], root)


def test_merkle_root_matches_rfc6962_shape_for_three_leaves():
    a, b, c = (R._leaf(x * 64) for x in "abc")
    assert R.merkle_root([a, b, c]) == R._node(R._node(a, b), c)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_preserves_one_chain_across_segments(home, small_segments):
    _fill(23)
    segs = R.sealed_segments()
    assert len(segs) >= 3, segs
    v = R.verify()
    assert v.ok and v.total == 23, v.statement
    chain = list(R.read())
    assert [r.request_id for r in chain] == [f"req{i}" for i in range(23)]
    # Each segment's first receipt links to the previous segment's checkpointed head.
    for prev_seg, seg in zip(segs, segs[1:]):
        first = next(R.read(R.segment_path(seg)))
        assert first.prev == R.load_segment_checkpoint(prev_seg).last
    assert next(R.read(R.segment_path(segs[0]))).prev == R.GENESIS


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes only")
def test_segments_and_checkpoints_are_owner_only(home, small_segments):
    old = os.umask(0o022)
    try:
        _fill(12)
    finally:
        os.umask(old)
    for seg in R.sealed_segments():
        assert stat.S_IMODE(R.segment_path(seg).stat().st_mode) == 0o600
        assert stat.S_IMODE(R.segment_checkpoint_path(seg).stat().st_mode) == 0o600
    assert stat.S_IMODE(R.segments_dir().stat().st_mode) == 0o700


def test_checkpoint_records_what_the_task_names(home, small_segments):
    _fill(8)
    ck = R.load_segment_checkpoint(0)
    rows = list(R.read(R.segment_path(0)))
    assert ck is not None
    assert (ck.segment, ck.rows, ck.first, ck.last) == (0, len(rows), rows[0].hash, rows[-1].hash)
    assert ck.root == R.merkle_root([R._leaf(r.hash) for r in rows]).hex()


# ---------------------------------------------------------------------------
# verify: full history, one segment, one receipt
# ---------------------------------------------------------------------------


def test_verify_single_segment_reads_no_other_segment(home, small_segments, monkeypatch):
    _fill(23)
    target = R.segment_path(1)
    ck = R.load_segment_checkpoint(1)
    real_open = Path.open
    opened: list[Path] = []

    def spy(self: Path, *a: object, **k: object):
        opened.append(self)
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", spy)
    v = R.verify_segment(target, ck)
    monkeypatch.setattr(Path, "open", real_open)
    assert v.ok and v.total == ck.rows, v.statement
    assert set(opened) == {target}


def test_inclusion_proof_verifies_from_the_bundle_alone(home, small_segments, monkeypatch):
    _fill(23)
    proof = R.prove("req7")
    assert proof is not None
    ck = R.load_segment_checkpoint(proof["checkpoint"]["segment"])
    root, ck_hash = ck.root, ck.digest()
    bundle = json.loads(json.dumps(proof))  # what a third party receives: plain JSON

    # Nothing on disk is consulted: point DISTIL_HOME at nothing and forbid opens.
    monkeypatch.setenv("DISTIL_HOME", str(home / "nowhere"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("verify_proof opened a file"))
    ok, why = R.verify_proof(bundle, checkpoint_hash=ck_hash)
    assert ok, why
    assert why.startswith(f"INCLUDED — receipt req7 is #{proof['index']} of ")
    ok, why = R.verify_proof(bundle, root=root)
    assert ok and "under pinned root" in why and "#" not in why, why
    ok, why = R.verify_proof(bundle)
    assert ok and why.startswith("SELF-CONSISTENT ONLY"), why


def test_inclusion_proof_rejects_edits_wrong_roots_and_junk(home, small_segments):
    _fill(23)
    proof = R.prove("req7")
    assert proof is not None

    edited = json.loads(json.dumps(proof))
    edited["receipt"]["tokens_compressed"] = 1
    assert R.verify_proof(edited) == (False, "receipt content does not match its hash")

    # Edited AND re-hashed: the content is self-consistent, the Merkle path is not.
    rehashed = json.loads(json.dumps(proof))
    rehashed["receipt"]["tokens_compressed"] = 1
    r = R.Receipt(**rehashed["receipt"])
    rehashed["receipt"]["hash"] = r.compute_hash()
    ok, _ = R.verify_proof(rehashed)
    assert not ok

    moved = json.loads(json.dumps(proof))
    moved["index"] = proof["index"] + 1
    assert not R.verify_proof(moved)[0]

    ok, why = R.verify_proof(proof, root="00" * 32)
    assert not ok and "pinned" in why

    hostile_receipts = [
        {**proof["receipt"], "handles": 5},
        {**proof["receipt"], "handles": [1, "a"]},
        {**proof["receipt"], "tokens_original": "1000"},
        {**proof["receipt"], "reversible": 0},
        {**proof["receipt"], "ts": None},
    ]
    for junk in (
        None,
        [],
        {"receipt": {}},
        {**proof, "path": ["zz"]},
        {**proof, "path": "ab"},
        {**proof, "path": ["ab"]},  # hex, but not a 32-byte hash
        {**proof, "index": True},
        *({**proof, "receipt": rec} for rec in hostile_receipts),
    ):
        ok, why = R.verify_proof(junk)
        assert not ok and why.startswith("malformed proof"), (junk, why)
    assert R.verify_proof(proof, checkpoint_hash="xyz")[1].startswith("malformed proof")


def _two_leaf_bundle() -> tuple[dict, R.Checkpoint]:
    a, b = _mk(0).sealed(), _mk(1)
    b.prev = a.hash
    b.sealed()
    leaves = [R._leaf(a.hash), R._leaf(b.hash)]
    ck = R.Checkpoint(0, 2, a.hash, b.hash, R.merkle_root(leaves).hex())
    from dataclasses import asdict

    bundle = {
        "v": 1,
        "receipt": asdict(b),
        "checkpoint": asdict(ck),
        "index": 1,
        "path": [h.hex() for h in R._audit_path(leaves, 1, 0, 2)],
    }
    return bundle, ck


def test_a_forged_position_passes_a_bare_root_but_is_never_printed():
    """The reviewer's case: index 1 of 2 also verifies as index 2 of 3 under the same
    root. With only the root pinned that forgery is unavoidable, so the output must not
    claim a position; with the checkpoint pinned it must fail."""
    bundle, ck = _two_leaf_bundle()
    forged = json.loads(json.dumps(bundle))
    forged["index"] = 2
    forged["checkpoint"].update(rows=3, segment=7)
    leaf = R._leaf(bundle["receipt"]["hash"])
    path = [bytes.fromhex(h) for h in bundle["path"]]
    assert R.verify_inclusion(2, 3, leaf, path, bytes.fromhex(ck.root))  # the ambiguity is real

    ok, why = R.verify_proof(forged, root=ck.root)
    assert ok, why
    assert "#2" not in why and "segment 7" not in why and "of 3" not in why, why
    assert "not proven" in why

    ok, why = R.verify_proof(forged, checkpoint_hash=ck.digest())
    assert (ok, why) == (False, "the proof's checkpoint is not the checkpoint you pinned")
    ok, why = R.verify_proof(bundle, checkpoint_hash=ck.digest())
    assert ok and "#1 of 2 in sealed segment 0" in why, why


def test_every_forgeable_position_is_caught_by_a_pinned_checkpoint():
    """Exhaustive over small trees: any (index, rows) that differs from the truth either
    fails the path check, or — where the path shape coincides — fails the pin."""
    for size in range(1, 20):
        receipts, prev = [], R.GENESIS
        for i in range(size):
            r = _mk(i)
            r.prev = prev
            receipts.append(r.sealed())
            prev = r.hash
        leaves = [R._leaf(r.hash) for r in receipts]
        root = R.merkle_root(leaves)
        ck = R.Checkpoint(0, size, receipts[0].hash, receipts[-1].hash, root.hex())
        for i, r in enumerate(receipts):
            path = R._audit_path(leaves, i, 0, size)
            assert len(path) == R._path_len(i, size)
            from dataclasses import asdict

            for n2 in range(1, 24):
                for i2 in range(n2):
                    if (i2, n2) == (i, size):
                        continue
                    forged = {
                        "receipt": asdict(r),
                        "checkpoint": {**asdict(ck), "rows": n2},
                        "index": i2,
                        "path": [h.hex() for h in path],
                    }
                    ok, why = R.verify_proof(forged, checkpoint_hash=ck.digest())
                    assert not ok, (size, i, n2, i2, why)


def test_checkpoints_output_line_hashes_to_the_digest(home, small_segments, capsys):
    import hashlib

    from distil.cli import main

    _fill(12)
    assert main(["receipts", "--checkpoints"]) == 0
    out, err = capsys.readouterr()
    for line, seg in zip(out.splitlines(), R.sealed_segments()):
        digest = hashlib.sha256(line.encode()).hexdigest()
        assert digest == R.load_segment_checkpoint(seg).digest()
        assert f"segment {seg} checkpoint sha256 {digest}" in err


def test_active_receipts_have_no_proof_yet(home, small_segments):
    _fill(7)
    last = list(R.read())[-1]
    assert not any(r.request_id == last.request_id for r in R.read(R.segment_path(0)))
    assert R.prove(last.request_id) is None


# ---------------------------------------------------------------------------
# Tampering a sealed segment
# ---------------------------------------------------------------------------


def _rewrite(path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    edit(rows)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def test_edited_sealed_receipt_is_detected(home, small_segments):
    _fill(23)
    _rewrite(R.segment_path(1), lambda rows: rows[1].update(tokens_compressed=1))
    v = R.verify()
    assert not v.ok and v.reason == "content does not match its hash"
    assert not R.verify_segment(R.segment_path(1), R.load_segment_checkpoint(1)).ok


def test_re_chained_sealed_segment_is_caught_by_its_merkle_root(home, small_segments):
    """The strong edit: change a receipt, re-hash it AND every later link inside the
    segment so the chain is internally consistent. Only the checkpoint catches it."""
    _fill(23)

    def rechain(rows: list[dict]) -> None:
        rows[1]["tokens_compressed"] = 1
        prev = rows[0]["hash"]
        for row in rows[1:]:
            row["prev"] = prev
            r = R.Receipt(**row)
            row["hash"] = r.compute_hash()
            prev = row["hash"]

    ck0 = R.load_segment_checkpoint(0)
    _rewrite(R.segment_path(0), rechain)
    v = R.verify_segment(R.segment_path(0), ck0)
    assert not v.ok and ("Merkle root" in v.reason or "first/last" in v.reason), v.reason
    assert not R.verify().ok


def test_deleted_sealed_segment_breaks_the_history(home, small_segments):
    _fill(23)
    R.segment_path(1).unlink()
    v = R.verify()
    assert not v.ok and v.reason == "prev-hash does not match the preceding receipt"


def test_missing_checkpoint_for_a_sealed_segment_is_reported(home, small_segments):
    _fill(12)
    R.segment_checkpoint_path(0).unlink()
    v = R.verify()
    assert not v.ok and "no readable checkpoint" in v.reason


# ---------------------------------------------------------------------------
# Legacy single-file chain: verifies unchanged, migrates lazily as segment 0
# ---------------------------------------------------------------------------


def test_legacy_chain_verifies_then_migrates_byte_for_byte(home, monkeypatch):
    monkeypatch.setattr(R, "SEGMENT_BYTES", 0)  # the pre-segment world: one file
    _fill(9)
    legacy = R.receipts_path().read_bytes()
    assert R.verify().ok and R.verify(R.receipts_path()).ok
    assert R.verify(full=False).ok  # leaves an old-shape resume point behind
    ck = json.loads((home / "receipts-verified.json").read_text())
    ck.pop("file")
    (home / "receipts-verified.json").write_text(json.dumps(ck))

    monkeypatch.setattr(R, "SEGMENT_BYTES", 1)  # first append past the threshold seals it
    R.append(_mk(9))
    assert R.sealed_segments() == [0]
    assert R.segment_path(0).read_bytes() == legacy  # never rewritten, only renamed
    v = R.verify(full=False)  # the legacy resume point still resumes (file 0 == segment 0)
    assert v.ok and v.total == 10 and v.checked_from == 9, v.statement
    v = R.verify()
    assert v.ok and v.total == 10, v.statement


# ---------------------------------------------------------------------------
# Crash mid-rotation
# ---------------------------------------------------------------------------


def test_crash_between_checkpoint_and_rename_loses_nothing(home, small_segments, monkeypatch):
    _fill(4)
    from distil import _filelock

    real = _filelock.replace_retrying
    boom = {"on": True}

    def crash_on_segment_move(src: Path, dst: Path) -> None:
        if boom["on"] and dst.suffix == ".jsonl":
            raise OSError("simulated crash mid-seal")
        real(src, dst)

    monkeypatch.setattr(_filelock, "replace_retrying", crash_on_segment_move)
    _fill(4, start=4)  # crosses the threshold; every seal attempt "crashes"
    assert R.sealed_segments() == []
    assert R.segment_checkpoint_path(0).exists()  # the orphan the crash leaves behind
    v = R.verify()
    assert v.ok and v.total == 8, v.statement  # the orphan is ignored; nothing was lost

    boom["on"] = False
    monkeypatch.setattr(R, "_seal_retry_at", 0.0)  # the backoff has "expired"
    _fill(1, start=8)  # the next append seals, overwriting the orphan
    assert R.sealed_segments() == [0]
    assert R.load_segment_checkpoint(0).rows == 8
    v = R.verify()
    assert v.ok and v.total == 9, v.statement


def test_crash_after_rename_before_next_append_links_correctly(home, small_segments):
    _fill(4)
    R._seal(R.receipts_path())  # the seal completed; the append it preceded never ran
    assert not R.receipts_path().exists()
    assert R.head_hash() == R.load_segment_checkpoint(0).last
    assert R.verify().ok
    _fill(1, start=4)
    v = R.verify()
    assert v.ok and v.total == 5, v.statement


def test_torn_tail_in_the_sealed_file_is_skipped_consistently(home, small_segments):
    _fill(3)
    with R.receipts_path().open("a") as fh:
        fh.write('{"ts": 1, "torn')  # a write cut short, no newline
    R._seal(R.receipts_path())
    assert R.load_segment_checkpoint(0).rows == 3
    assert R.verify_segment(R.segment_path(0), R.load_segment_checkpoint(0)).ok
    _fill(1, start=3)
    assert R.verify().ok


# ---------------------------------------------------------------------------
# head_hash stays O(1) across rotation; the fast pass survives a seal
# ---------------------------------------------------------------------------


def _bytes_read_by_head_hash(monkeypatch) -> int:  # type: ignore[no-untyped-def]
    real_open = Path.open
    total = 0

    def counting_open(self: Path, *a: object, **k: object):
        nonlocal total
        fh = real_open(self, *a, **k)
        orig = fh.read

        def counted(n: int = -1) -> bytes:
            nonlocal total
            data = orig(n)
            total += len(data)
            return data

        fh.read = counted  # type: ignore[method-assign]
        return fh

    monkeypatch.setattr(Path, "open", counting_open)
    try:
        R.head_hash()
    finally:
        monkeypatch.setattr(Path, "open", real_open)
    return total


def test_head_hash_reads_a_tail_not_the_history(home, monkeypatch):
    monkeypatch.setattr(R, "SEGMENT_BYTES", 400 * 1024)
    _fill(3_000)
    history = sum(R.segment_path(s).stat().st_size for s in R.sealed_segments())
    assert len(R.sealed_segments()) >= 2 and history > R._TAIL_BLOCK_SIZE * 8
    # Mid-segment: the active file's tail.
    assert _bytes_read_by_head_hash(monkeypatch) <= R._TAIL_BLOCK_SIZE
    # Right after a seal (empty active): the newest segment's tail, not the history.
    R._seal(R.receipts_path())
    assert _bytes_read_by_head_hash(monkeypatch) <= R._TAIL_BLOCK_SIZE
    assert R.head_hash() == list(R.read())[-1].hash


def test_fast_pass_resumes_across_a_seal(home, small_segments):
    _fill(12)
    assert R.verify(full=False).ok
    _fill(6, start=12)  # at least one seal happens in here
    v = R.verify(full=False)
    assert v.ok and v.total == 18 and v.checked_from == 12, v.statement


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_segment_checkpoints_prove_and_check(home, small_segments, capsys, tmp_path):
    from distil.cli import main

    _fill(23)
    assert main(["receipts"]) == 0
    assert "segments" in capsys.readouterr().out

    assert main(["receipts", "--segment", "1"]) == 0
    assert "segment 1: VERIFIED" in capsys.readouterr().out
    assert main(["receipts", "--segment", "99"]) == 1
    capsys.readouterr()

    assert main(["receipts", "--checkpoints"]) == 0
    cks = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    ck0_hash = R.load_segment_checkpoint(0).digest()
    assert [c["segment"] for c in cks] == R.sealed_segments()

    assert main(["receipts", "--prove", "req3"]) == 0
    proof_file = tmp_path / "proof.json"
    proof_file.write_text(capsys.readouterr().out)
    root = next(c["root"] for c in cks if c["segment"] == 0)
    assert main(["receipts", "--check-proof", str(proof_file), "--root", root]) == 0
    assert "INCLUDED" in capsys.readouterr().out
    assert main(["receipts", "--check-proof", str(proof_file), "--root", "11" * 32]) == 1
    assert "NOT INCLUDED" in capsys.readouterr().out
    assert main(["receipts", "--check-proof", str(proof_file), "--checkpoint-hash", ck0_hash]) == 0
    assert "#" in capsys.readouterr().out
    assert main(["receipts", "--check-proof", str(proof_file)]) == 0
    out, err = capsys.readouterr()
    assert "nothing pinned" in err and "SELF-CONSISTENT ONLY" in out
    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({**json.loads(proof_file.read_text()), "receipt": {"handles": 5}}))
    assert main(["receipts", "--check-proof", str(junk)]) == 1
    assert "malformed proof" in capsys.readouterr().out

    assert main(["receipts", "--prove", "nope"]) == 1

    _rewrite(R.segment_path(1), lambda rows: rows[0].update(model="x"))
    assert main(["receipts", "--segment", "1"]) == 1
    assert main(["receipts"]) == 1


# ---------------------------------------------------------------------------
# A seal that keeps failing costs one attempt per backoff, not one per append
# ---------------------------------------------------------------------------


def test_persistently_failing_seal_is_attempted_a_bounded_number_of_times(
    home, small_segments, monkeypatch
):
    real_mkdir, real_read = Path.mkdir, R.read
    parses = 0

    def no_segments_dir(self: Path, *a: object, **k: object) -> None:
        if self.name == "receipts-segments":
            raise PermissionError("simulated: segments dir not creatable")
        real_mkdir(self, *a, **k)  # type: ignore[arg-type]

    def counting_read(path: Path | None = None):  # type: ignore[no-untyped-def]
        nonlocal parses
        if path == R.receipts_path():
            parses += 1
        return real_read(path)

    monkeypatch.setattr(Path, "mkdir", no_segments_dir)
    monkeypatch.setattr(R, "read", counting_read)
    attempts = 0
    real_seal = R._seal

    def counting_seal(active: Path):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        return real_seal(active)

    monkeypatch.setattr(R, "_seal", counting_seal)
    _fill(300)
    assert attempts == 1, f"{attempts} seal attempts in 300 appends"
    assert parses == 0, "a seal that could not create its directory parsed the chain first"
    assert R.verify().ok and R.verify().total == 300  # nothing lost; all still active

    # The backoff expires: exactly one more attempt, and it succeeds once mkdir works.
    monkeypatch.setattr(Path, "mkdir", real_mkdir)
    monkeypatch.setattr(R.time, "monotonic", lambda: 1e12)
    _fill(1, start=300)
    assert attempts == 2 and R.sealed_segments() == [0]
    assert R.verify().ok


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes only")
def test_an_existing_segments_dir_is_tightened_to_owner_only(home, small_segments):
    R.segments_dir().mkdir(mode=0o755)
    R.segments_dir().chmod(0o755)
    _fill(8)
    assert R.sealed_segments()
    assert stat.S_IMODE(R.segments_dir().stat().st_mode) == 0o700


def test_a_seal_landing_mid_verify_is_not_reported_as_broken(home, small_segments, monkeypatch):
    """verify() lists the segments, then a seal renames the active file, then verify opens
    the new active file: its first receipt links to a segment the stale list lacks."""
    _fill(8)
    stale = R._chain_files()
    R._seal(R.receipts_path())
    _fill(2, start=8)
    calls = {"n": 0}
    real = R._chain_files

    def first_call_stale():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return stale if calls["n"] == 1 else real()

    monkeypatch.setattr(R, "_chain_files", first_call_stale)
    v = R.verify()
    assert v.ok and v.total == 10, v.statement
    assert calls["n"] >= 2


def test_a_real_break_survives_the_retry(home, small_segments):
    _fill(12)
    R.segment_path(0).unlink()
    assert not R.verify().ok


# ---------------------------------------------------------------------------
# A line that is not a receipt is never silent
# ---------------------------------------------------------------------------


def _append_line(text: str) -> None:
    with R.receipts_path().open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _type_invalid(i: int) -> str:
    """A well-formed receipt whose `handles` has the wrong type: only an edit makes this."""
    r = _mk(i).sealed()
    from dataclasses import asdict

    return json.dumps({**asdict(r), "handles": 5}, sort_keys=True)


def test_type_invalid_last_line_is_broken_and_stays_broken(home):
    _fill(4)
    _append_line(_type_invalid(4))
    v = R.verify()
    assert not v.ok and v.first_bad_index == 4 and v.total == 5, v.statement
    assert "not a valid receipt" in v.reason
    _fill(1, start=5)  # the next append chains past it — the break must still be reported
    v = R.verify()
    assert not v.ok and v.first_bad_index == 4, v.statement
    assert not R.verify(full=False).ok


def test_type_invalid_mid_chain_line_is_broken(home):
    _fill(2)
    _append_line(_type_invalid(2))
    _fill(2, start=3)
    v = R.verify()
    assert not v.ok and v.first_bad_index == 2 and v.total == 5, v.statement


def test_torn_or_foreign_lines_are_counted_never_clean(home):
    _fill(3)
    _append_line('{"ts": 1, "torn')
    v = R.verify()
    assert v.ok and v.skipped == 1, v.statement
    assert v.statement.startswith("VERIFIED WITH GAPS — 3 receipts")
    assert "1 line is not a receipt" in v.statement
    _append_line("42")  # mid-file once the next receipt lands
    _fill(1, start=3)
    v = R.verify()
    assert v.ok and v.skipped == 2 and "VERIFIED WITH GAPS" in v.statement, v.statement
    # The resumed pass reports the same count: the ones before its resume point are
    # carried in the checkpoint, not forgotten.
    _fill(1, start=4)
    v = R.verify(full=False)
    assert v.ok and v.checked_from > 0 and v.skipped == 2, v.statement
    from distil.proof_ledger import _receipts_line

    assert "2 line(s) in the file are not receipts" in _receipts_line()


def test_a_type_invalid_line_in_a_sealed_segment_breaks_that_segment(home, small_segments):
    _fill(12)
    with R.segment_path(0).open("a", encoding="utf-8") as fh:
        fh.write(_type_invalid(99) + "\n")
    v = R.verify_segment(R.segment_path(0), R.load_segment_checkpoint(0))
    assert not v.ok and "not a valid receipt" in v.reason, v.statement
    assert not R.verify().ok


# ---------------------------------------------------------------------------
# Checkpoint schema version is validated, not trusted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_v", [2, 0, True, 1.0, "1", None])
def test_checkpoint_v_must_be_the_integer_one(bad_v):
    d = {"segment": 0, "rows": 1, "first": "a", "last": "a", "root": "00" * 32, "v": bad_v}
    with pytest.raises(ValueError, match="checkpoint v"):
        R.Checkpoint.from_dict(d)
    assert R.Checkpoint.from_dict({**d, "v": 1}).v == 1
    assert R.Checkpoint.from_dict({k: x for k, x in d.items() if k != "v"}).v == 1


def test_a_checkpoint_with_another_schema_is_a_segment_mismatch(home, small_segments):
    _fill(8)
    ck = R.load_segment_checkpoint(0)
    from dataclasses import replace

    v = R.verify_segment(R.segment_path(0), replace(ck, v=2))
    assert not v.ok and "schema" in v.reason, v.statement
    path = R.segment_checkpoint_path(0)
    path.write_text(json.dumps({**json.loads(path.read_text()), "v": 2}))
    v = R.verify()
    assert not v.ok and "no readable checkpoint" in v.reason, v.statement


# ---------------------------------------------------------------------------
# Review follow-ups (#191): incomplete pins, proof version, naming, dead seals
# ---------------------------------------------------------------------------


def test_checkpoints_exits_nonzero_when_a_segment_has_no_checkpoint(home, small_segments, capsys):
    from distil.cli import main

    _fill(12)
    segs = R.sealed_segments()
    assert len(segs) >= 2
    R.segment_path(segs[0]).with_suffix(".checkpoint.json").unlink()
    assert main(["receipts", "--checkpoints"]) == 1  # an incomplete pin set is not success
    _, err = capsys.readouterr()
    assert f"segment {segs[0]}: checkpoint missing" in err


def test_a_proof_of_another_version_is_malformed(home, small_segments):
    _fill(12)
    proof = json.loads(json.dumps(R.prove("req3")))
    ck = R.load_segment_checkpoint(proof["checkpoint"]["segment"])
    assert R.verify_proof(proof, checkpoint_hash=ck.digest())[0]
    for v in (2, 0, 1.0, "1", True, None):
        ok, why = R.verify_proof(dict(proof, v=v), checkpoint_hash=ck.digest())
        assert not ok and "version" in why, (v, why)


def test_a_missing_checkpoint_names_the_segment_not_a_sentinel(home, small_segments):
    _fill(12)
    seg = R.sealed_segments()[0]
    v = R.verify_segment(R.segment_path(seg), None)
    assert not v.ok and f"segment {seg}" in v.statement and "-1" not in v.statement


def test_duplicate_segment_names_list_once(home, small_segments):
    _fill(12)
    seg = R.sealed_segments()[0]
    src = R.segment_path(seg)
    (src.parent / f"{seg}.jsonl").write_bytes(src.read_bytes())  # unpadded twin
    assert R.sealed_segments().count(seg) == 1


def test_a_full_active_file_with_no_receipt_backs_off(home, monkeypatch):
    monkeypatch.setattr(R, "SEGMENT_BYTES", 1024)
    R.receipts_path().parent.mkdir(parents=True, exist_ok=True)
    R.receipts_path().write_bytes(b"x" * 2048 + b"\n")  # full-size, nothing sealable
    calls = []
    real = R._seal
    monkeypatch.setattr(R, "_seal", lambda p: calls.append(p) or real(p))
    for i in range(20):
        R.append(_mk(i))
    assert len(calls) == 1  # no re-read of the active file on every append

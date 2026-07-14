"""Query-aware salience — ② flywheel-learned associations (the moat).

The curated synonym map (`compress.lexicon`) is a floor. This learns the bridges *specific to
your traffic*: which output terms the agent actually pulls back under which query terms. On a
codebase that calls its retry cap ``max_tries`` or its tenant ``org``, distil learns that link
from real expands — no curation, no embeddings.

**Content-free by construction.** We never store a term. At digest time we record, per digested
block, the **hashed** query terms and the **hashed** output sub-tokens (a 12-hex SHA-1 slice);
the expand outcome comes from the existing content-free expand log, joined by handle. Training
computes PMI over the hashed pairs — a pair is a learned bridge when the two hashed tokens
co-occur in *expanded* blocks far more than chance. At lookup we hash the live terms the same
way and check the pair. Nothing reconstructable to a prompt, a line, or a term is ever written.

Learned associations are unioned into the semantic bridge additively, so — like every other
bridge — a spurious association only wastes a little compression, never drops a needed line.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_HASH_LEN = 12  # 48 bits — collision-negligible at realistic vocab sizes, and one-way


def _h(term: str) -> str:
    return hashlib.sha1(term.lower().encode("utf-8")).hexdigest()[:_HASH_LEN]


def _cooccur_path() -> Path:
    import os

    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "query_cooccur.jsonl"


def _assoc_path() -> Path:
    return Path(__file__).resolve().parent / "compress" / "query_assoc.json"


def record_cooccurrence(
    handle: str,
    query_terms: frozenset[str],
    output_subtokens: set[str],
    *,
    path: Path | None = None,
    cap: int = 64,
) -> None:
    """Append the hashed query-terms × output-subtokens co-occurrence for one digested block.
    Content-free (hashes only). Fail-open; capped so a huge block can't bloat the log."""
    if not query_terms or not output_subtokens:
        return
    try:
        qs = sorted({_h(t) for t in query_terms})[:cap]
        os_ = sorted({_h(t) for t in output_subtokens})[:cap]
        if not qs or not os_:
            return
        p = path or _cooccur_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"h": handle, "q": qs, "o": os_, "ts": time.time()}) + "\n")
    except Exception:  # noqa: BLE001 — the moat signal must never break the request path
        pass


def build_associations(
    *,
    cooccur_path: Path | None = None,
    signal_path: Path | None = None,
    min_count: int = 3,
    min_ratio: float = 0.6,
    save_to: Path | None = None,
) -> dict[str, list[str]]:
    """Learn (query, output) associations by the **expand-conditioned** signal: a hashed pair
    is a bridge when it co-occurs in *expanded* blocks at least *min_count* times AND at least
    *min_ratio* of all its co-occurrences were expanded (so the pair predicts the agent needing
    the block, not just that both are common). This avoids PMI's degeneracy when a pair
    co-occurs everywhere. Returns {hashed_q: [hashed_o, ...]}; persisted if *save_to*."""
    cp = cooccur_path or _cooccur_path()
    expanded = _expanded_handles(signal_path)

    all_pair: dict[tuple[str, str], int] = {}
    exp_pair: dict[tuple[str, str], int] = {}
    try:
        if not cp.exists():
            return {}
        for line in cp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qs, os_ = rec.get("q") or [], rec.get("o") or []
            if not isinstance(qs, list) or not isinstance(os_, list):
                continue
            was_expanded = rec.get("h") in expanded
            for q in set(qs):
                for o in set(os_):
                    key = (q, o)
                    all_pair[key] = all_pair.get(key, 0) + 1
                    if was_expanded:
                        exp_pair[key] = exp_pair.get(key, 0) + 1
    except OSError:
        return {}

    table: dict[str, list[str]] = {}
    for key, c_all in all_pair.items():
        c_exp = exp_pair.get(key, 0)
        if c_exp >= min_count and (c_exp / c_all) >= min_ratio:
            q, o = key
            table.setdefault(q, []).append(o)

    if save_to is not None or (save_to is None and table):
        dest = save_to or _assoc_path()
        try:
            dest.write_text(json.dumps({"assoc": table}, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
    return table


def _expanded_handles(signal_path: Path | None) -> set[str]:
    out: set[str] = set()
    try:
        from distil.expand import _default_signal_path

        sp = signal_path or _default_signal_path()
        if sp.exists():
            for line in sp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    h = json.loads(line).get("handle")
                    if isinstance(h, str):
                        out.add(h)
                except json.JSONDecodeError:
                    continue
    except (OSError, ImportError):
        pass
    return out


# --------------------------------------------------------------------------- lookup
_ASSOC: dict[str, frozenset[str]] | None = None


def _load_assoc() -> dict[str, frozenset[str]]:
    global _ASSOC
    if _ASSOC is None:
        try:
            data = json.loads(_assoc_path().read_text(encoding="utf-8"))
            _ASSOC = {k: frozenset(v) for k, v in data.get("assoc", {}).items()}
        except (OSError, ValueError, json.JSONDecodeError):
            _ASSOC = {}
    return _ASSOC


def reset_cache() -> None:
    global _ASSOC
    _ASSOC = None


def learned_bridge(qterm: str, oterm: str) -> bool:
    """True if the (query, output) pair is a learned association. Hashes both live terms and
    checks the table — symmetric (either direction counts)."""
    assoc = _load_assoc()
    if not assoc:
        return False
    hq, ho = _h(qterm), _h(oterm)
    return ho in assoc.get(hq, frozenset()) or hq in assoc.get(ho, frozenset())


if __name__ == "__main__":  # self-check — content-free + PMI join, no framework
    import tempfile

    d = Path(tempfile.mkdtemp())
    cp, sp, ap = d / "co.jsonl", d / "sig.jsonl", d / "assoc.json"
    # 'retry' (query) co-occurs with 'maxtries' (output) in 4 expanded blocks; noise elsewhere.
    for i in range(4):
        record_cooccurrence(f"exp{i}", frozenset({"retry"}), {"maxtries", "noise"}, path=cp)
    record_cooccurrence("never", frozenset({"retry"}), {"unrelated"}, path=cp)
    blob = cp.read_text()
    assert "retry" not in blob and "maxtries" not in blob, "co-occurrence log must be content-free"
    sp.write_text(
        "".join(json.dumps({"handle": f"exp{i}", "recovered_chars": 9}) + "\n" for i in range(4))
    )
    table = build_associations(
        cooccur_path=cp, signal_path=sp, min_count=3, min_ratio=0.6, save_to=ap
    )
    assert _h("maxtries") in table.get(_h("retry"), []), "should learn retry↔maxtries from expands"
    # 'unrelated' co-occurred only in the NON-expanded block → filtered by the ratio gate
    assert _h("unrelated") not in table.get(_h("retry"), []), "non-expanded pair must be filtered"
    print(
        "query_assoc self-check ok: learned",
        sum(len(v) for v in table.values()),
        "association(s), content-free",
    )

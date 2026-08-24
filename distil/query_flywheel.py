"""Query-aware salience — phase 2 dark collection (the training label source).

The moat traffic labels this for free. Every time the agent **expands** a digested block,
that is a label: "given the query live at digest time, this folded content was actually
needed." Phase 2 pairs that with the query — so a retrain can learn which *semantically*
relevant lines to pin inline next time.

**Content-free by construction.** We never persist a prompt, a response, a tool result, or
even a raw query term. At digest time we record, per dropped line, only its *numeric*
query-conditioned feature vector (lexical-overlap count, selectivity, proximity — all
derived numbers), keyed by the block's already-content-free handle. The expand side reuses
`expand.record_signal` (which logs the handle, never content). Training joins the two by
handle: a dropped-line feature row under an expanded handle is a positive, under a
never-expanded handle a weak negative.

Off by default — only the live proxy calls `enable()`. Sampled and fail-open, so the
flywheel can never slow or break a request.
"""

from __future__ import annotations

import json
import threading as _threading
import time
from pathlib import Path

from distil.compress.query_relevance import (
    QUERY_FEATURE_NAMES,
    _nearest_hit_dist,
    _selective_terms,
    featurize_query,
)
from distil.compress.intent import terms_of

_enabled: bool = False
_sample_rate: float = 0.25  # bound hot-path I/O; a block is recorded whole or not at all


def enable(sample_rate: float = 0.25) -> None:
    """Turn on dark collection (the proxy calls this at startup). Sampled to bound I/O."""
    global _enabled, _sample_rate
    _enabled = True
    _sample_rate = max(0.0, min(sample_rate, 1.0))


def disable() -> None:
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


# The digest of the request currently being compressed, so a flywheel row can be joined
# to its shadow verdict later. Thread-local because ThreadingHTTPServer handles each
# request on its own thread; set by the proxy, never required.
_req_tls = _threading.local()


def set_request_digest(digest: str | None) -> None:
    """Tag subsequent rows on this thread with the request they came from."""
    _req_tls.digest = digest


def _current_request_digest() -> str | None:
    return getattr(_req_tls, "digest", None)


def _default_path() -> Path:
    import os

    # Same DISTIL_HOME convention as the ledger / expand signal log — resolved at call time.
    return (
        Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "query_flywheel.jsonl"
    )


def _sampled(handle: str, rate: float) -> bool:
    """Deterministic per-handle sampling — a block's record is atomic (all rows or none),
    and the same block sampled the same way across a retry."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    try:
        return (int(handle[:8], 16) / 0xFFFF_FFFF) < rate
    except ValueError:
        return False


def maybe_record(
    handle: str,
    intent: frozenset[str],
    lines: list[str],
    kind: str,
    dropped_indices: set[int],
    *,
    path: Path | None = None,
) -> None:
    """Record the numeric query-features of the *dropped* lines of a digested block.

    No-op unless collection is enabled (offline/tests are untouched), there is a query, and
    some lines were actually folded. Fail-open: any error is swallowed — the flywheel must
    never slow or break the request path.
    """
    if not _enabled or not intent or not dropped_indices:
        return
    if not _sampled(handle, _sample_rate):
        return
    try:
        line_terms = [terms_of(ln) for ln in lines]
        n = len(lines)
        selective = _selective_terms(line_terms, intent, n)
        hits = [i for i, lt in enumerate(line_terms) if lt & selective]
        rows: list[list[float]] = []
        for i in sorted(dropped_indices):
            if 0 <= i < n:
                x = featurize_query(
                    lines[i],
                    kind,
                    intent,
                    line_terms=line_terms[i],
                    selective_terms=selective,
                    nearest_hit_dist=_nearest_hit_dist(i, hits, n),
                )
                rows.append([round(v, 4) for v in x])  # numeric only — no content
        if not rows:
            return
        p = path or _default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec: dict[str, object] = {"h": handle, "rows": rows, "ts": time.time()}
        req = _current_request_digest()
        if req:
            rec["req"] = req  # join key for the shadow decision-change label
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        # ② also record the hashed query×output-subtoken co-occurrence (content-free) so the
        # flywheel can learn domain-specific associations. See distil.query_assoc.
        from distil import query_assoc
        from distil.compress.lexicon import subtokens

        out_subs: set[str] = set()
        for i in sorted(dropped_indices):
            if 0 <= i < n:
                for t in line_terms[i]:
                    out_subs.add(t)
                    out_subs |= subtokens(t)
        query_assoc.record_cooccurrence(handle, intent, out_subs)
    except Exception:  # noqa: BLE001 — the moat signal must never break the request path
        pass


def _decision_changed_digests(path: Path | None = None) -> set[str]:
    """Request digests whose decision CHANGED under compression, per shadow mode.

    This is the label the flywheel actually wants. The expand signal answers "did the
    model ask for this block back", which only exists under ``--expand``; a shadow A/B
    verdict answers "did compressing this request change what the agent did next",
    which is the property the certificate is about — and shadow runs by default on
    every configuration, subscription included.

    A/A rows are excluded: they re-run the SAME compressed request twice, so a
    disagreement there is provider nondeterminism, not a compression effect. Training
    on them would teach the model to keep content because the sampler was noisy.
    """
    out: set[str] = set()
    try:
        from distil.shadow import _state_dir

        sp = path or (_state_dir() / "shadow.jsonl")
        if not sp.exists():
            return out
        for line in sp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "aa" or rec.get("equivalent") is not False:
                continue
            # ShadowLedger._append flattens `evidence` into the row, so `digest` is
            # top-level rather than nested.
            digest = rec.get("digest")
            if isinstance(digest, str):
                out.add(digest)
    except (OSError, ImportError, AttributeError):
        pass  # the flywheel must never depend on shadow being present
    return out


def load_labels(
    *,
    flywheel_path: Path | None = None,
    signal_path: Path | None = None,
) -> list[tuple[list[float], float]]:
    """Join the digest-time feature rows with the expand log into ``(features, label)`` pairs.

    A dropped-line row whose block handle was later expanded is a positive (the agent needed
    that region under its query); a row whose handle was never expanded is a weak negative.
    Returns the training set for ``QueryRelevanceModel`` — the same shape ``codec.learned.train``
    consumes, so the existing training + certification pipeline is reused unchanged.
    """
    fp = flywheel_path or _default_path()
    expanded: set[str] = set()
    try:
        from distil.expand import _default_signal_path

        sp = signal_path or _default_signal_path()
        if sp.exists():
            for line in sp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # A MISS is written under "miss", never "handle": the model asked for
                # that block and distil could not return it. Counting it as a positive
                # would teach the keep-model that the block it failed to produce was
                # safe to drop — precisely backwards.
                h = rec.get("handle")
                if isinstance(h, str):
                    expanded.add(h)
    except (OSError, ImportError):
        pass

    changed = _decision_changed_digests()
    samples: list[tuple[list[float], float]] = []
    try:
        if not fp.exists():
            return samples
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = rec.get("h")
            rows = rec.get("rows")
            if not isinstance(h, str) or not isinstance(rows, list):
                continue
            # Two independent positives, either of which means "the agent needed this":
            #   * an EXPAND — the model asked for the block back (only ever available
            #     under --expand, and never on a subscription); and
            #   * a shadow DECISION CHANGE — compressing this request provably altered
            #     the agent's next action. Strictly stronger evidence, on by default at
            #     2%, and it works on every configuration including subscription mode.
            # Without the second source the flywheel could only learn from the one
            # configuration users are steered away from, which is why the shipped model
            # stayed inert.
            label = 1.0 if (h in expanded or rec.get("req") in changed) else 0.0
            for row in rows:
                if isinstance(row, list) and len(row) == len(QUERY_FEATURE_NAMES):
                    samples.append(([float(v) for v in row], label))
    except OSError:
        pass
    return samples


if __name__ == "__main__":  # self-check — content-free + join correctness, no framework
    import tempfile

    d = Path(tempfile.mkdtemp())
    fw, sig = d / "fw.jsonl", d / "sig.jsonl"
    enable(1.0)
    lines = ["retry section:", "  max_attempts = 5", "  timeout = 30", "unrelated noise line"]
    maybe_record("aaaa1111", frozenset({"retry"}), lines, "tool_output", {1, 2, 3}, path=fw)
    maybe_record("bbbb2222", frozenset({"retry"}), lines, "tool_output", {1, 2, 3}, path=fw)
    # only aaaa1111 was expanded
    sig.write_text(json.dumps({"handle": "aaaa1111", "recovered_chars": 40, "ts": 1.0}) + "\n")
    # content-free: the raw feature rows must not contain any source text
    blob = fw.read_text()
    assert "max_attempts" not in blob and "retry" not in blob, "flywheel must be content-free"
    labels = load_labels(flywheel_path=fw, signal_path=sig)
    pos = sum(1 for _x, y in labels if y == 1.0)
    neg = sum(1 for _x, y in labels if y == 0.0)
    assert pos == 3 and neg == 3, f"expected 3 pos / 3 neg, got {pos}/{neg}"
    disable()
    print(f"query_flywheel self-check ok: {pos} pos, {neg} neg, content-free")

"""Semantic bridge (query-aware salience): ① morphology + synonyms, ③ trigram, ④ vectors,
② flywheel-learned associations, and the additive integration into phase-1 relevant_lines.

The overarching safety invariant, asserted throughout: every bridge is ADDITIVE — it can only
add a kept line, never drop one — so a wrong bridge wastes a little compression, never an answer.
"""

import json
import tempfile
from pathlib import Path

from distil.compress import lexicon
from distil.compress.intent import relevant_lines, terms_of


# --- ① morphology + curated synonyms ------------------------------------------------


def test_stem_inflections():
    assert lexicon.stem("attempts") == lexicon.stem("attempt") == "attempt"
    assert lexicon.stem("timeouts") == lexicon.stem("timeout")
    assert lexicon.stem("ok") == "ok"  # too short to strip — safe


def test_curated_synonyms_bidirectional():
    assert "attempt" in lexicon.synonyms("retry")
    assert "retry" in lexicon.synonyms("attempt")
    assert "max" in lexicon.synonyms("limit") and "cap" in lexicon.synonyms("limit")


def test_subtokens_split_compound_identifiers():
    assert lexicon.subtokens("max_attempts") == {"max", "attempts"}
    assert lexicon.subtokens("maxAttempts") == {"max", "attempts"}
    assert lexicon.subtokens("connection.pool-size") == {"connection", "pool", "size"}


# --- ③ trigram ----------------------------------------------------------------------


def test_trigram_similarity():
    assert lexicon.trigram_sim("config", "config") == 1.0
    assert lexicon.trigram_sim("config", "configs") > 0.5  # near-morphology
    assert lexicon.trigram_sim("retry", "banana") < 0.2  # unrelated


# --- ④ vectors (inert by default — no bundled table ships) --------------------------


def test_embed_sim_inert_without_table():
    lexicon.reset_vectors_cache()
    assert lexicon.embed_sim("retry", "attempt") == 0.0  # no table → no-op
    assert lexicon.embed_sim("x", "x") == 1.0  # identity still holds


# --- the bridge + additive safety ---------------------------------------------------


def test_bridge_hits_semantic_answer_via_subtokens():
    # the money case: 'retry limit' bridges to the token 'max_attempts' (max→limit, attempts→retry)
    assert lexicon.bridge_line_terms({"max_attempts"}, frozenset({"retry", "limit"})) is True
    assert lexicon.bridge_line_terms({"max_attempts"}, frozenset({"retry"})) is True


def test_bridge_additive_safety_no_false_positive():
    # a genuinely unrelated line must NOT bridge (keeps compression; additive-safe either way)
    assert (
        lexicon.bridge_line_terms({"database_host", "localhost"}, frozenset({"retry", "limit"}))
        is False
    )
    assert lexicon.bridge_line_terms(set(), frozenset({"retry"})) is False
    assert lexicon.bridge_line_terms({"retry"}, frozenset()) is False


# --- integration: phase-1 relevant_lines widens with bridge=True --------------------


def test_relevant_lines_bridge_pins_semantic_answer():
    lines = [
        "database:",
        "  host: localhost",
        "retry:",
        "  max_attempts: 5",
        "  backoff: exp",
        "logging:",
    ]
    q = terms_of("what is the retry limit")
    lexical = relevant_lines(lines, q)  # exact only
    bridged = relevant_lines(lines, q, bridge=True)  # + semantic bridge
    assert 3 not in lexical, "lexical alone misses max_attempts"
    assert 3 in bridged, "bridge pins max_attempts (the answer)"
    assert lexical <= bridged, "bridge is ADDITIVE — never removes a lexical keep"


# --- ② flywheel-learned associations (content-free) ---------------------------------


def _fresh_home(monkeypatch):
    d = Path(tempfile.mkdtemp())
    monkeypatch.setenv("DISTIL_HOME", str(d))
    return d


def test_cooccurrence_is_content_free(monkeypatch):
    from distil import query_assoc

    d = _fresh_home(monkeypatch)
    cp = d / "co.jsonl"
    query_assoc.record_cooccurrence("h1", frozenset({"tenant"}), {"org", "secret_value"}, path=cp)
    blob = cp.read_text()
    assert "tenant" not in blob and "org" not in blob and "secret_value" not in blob


def test_learned_association_expand_conditioned(monkeypatch, tmp_path):
    from distil import query_assoc

    cp, sp, ap = tmp_path / "co.jsonl", tmp_path / "sig.jsonl", tmp_path / "assoc.json"
    for i in range(4):  # tenant↔org in expanded blocks
        query_assoc.record_cooccurrence(f"e{i}", frozenset({"tenant"}), {"org"}, path=cp)
    for i in range(3):  # a distractor only in non-expanded blocks
        query_assoc.record_cooccurrence(f"n{i}", frozenset({"tenant"}), {"distractor"}, path=cp)
    sp.write_text("".join(json.dumps({"handle": f"e{i}"}) + "\n" for i in range(4)))
    query_assoc.build_associations(
        cooccur_path=cp, signal_path=sp, min_count=3, min_ratio=0.6, save_to=ap
    )
    monkeypatch.setattr(query_assoc, "_assoc_path", lambda: ap)
    query_assoc.reset_cache()
    assert query_assoc.learned_bridge("tenant", "org") is True
    assert query_assoc.learned_bridge("tenant", "distractor") is False  # ratio-filtered
    query_assoc.reset_cache()


def test_learned_bridge_inert_without_table(monkeypatch, tmp_path):
    from distil import query_assoc

    monkeypatch.setattr(query_assoc, "_assoc_path", lambda: tmp_path / "missing.json")
    query_assoc.reset_cache()
    assert query_assoc.learned_bridge("tenant", "org") is False  # no table → no-op
    query_assoc.reset_cache()


# --- coverage: all four bridge branches + ④ active + the flywheel co-occurrence hook ------


def test_bridges_all_four_branches(monkeypatch, tmp_path):
    assert lexicon.bridges("attempts", "attempt")  # ① morphology
    assert lexicon.bridges("retry", "attempt")  # ① synonym
    assert lexicon.bridges("config", "configs")  # ③ trigram
    assert not lexicon.bridges("retry", "database")  # no bridge
    # ④ vectors — active with a temp table (near-parallel vectors)
    vpath = tmp_path / "vec.json"
    vpath.write_text(json.dumps({"vectors": {"alpha": [1.0, 0.0], "beta": [0.99, 0.14]}}))
    monkeypatch.setattr(lexicon, "_VECTORS_PATH", vpath)
    lexicon.reset_vectors_cache()
    assert lexicon.embed_sim("alpha", "beta") > 0.9
    assert lexicon.embed_sim("alpha", "missing") == 0.0  # OOV
    assert lexicon.bridges("alpha", "beta")  # ④ branch
    lexicon.reset_vectors_cache()


def test_flywheel_records_cooccurrence_content_free(monkeypatch, tmp_path):
    from distil import query_assoc
    from distil import query_flywheel as fw

    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    monkeypatch.setattr(fw, "_enabled", True)
    monkeypatch.setattr(fw, "_sample_rate", 1.0)
    lines = ["retry:", "  max_attempts: 5", "  backoff: exp"]
    fw.maybe_record(
        "hb123456", frozenset({"retry"}), lines, "tool_output", {1, 2}, path=tmp_path / "fw.jsonl"
    )
    co = query_assoc._cooccur_path()
    assert co.exists(), "② co-occurrence recorded alongside the feature rows"
    blob = co.read_text()
    assert "max_attempts" not in blob and "retry" not in blob  # content-free (hashes only)
    fw.disable()

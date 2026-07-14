"""B — the length-aware subword tokenizer: a closer offline BPE approximation than the flat
heuristic, zero-dependency. Exact billing-grade validation lives in the economical claude -p /
anthropic check; here we assert the structural properties that make it a better base."""

from distil.tokenizer import HeuristicTokenizer, SubwordHeuristicTokenizer, resolve


def test_resolve_subword():
    assert isinstance(resolve("subword"), SubwordHeuristicTokenizer)
    assert isinstance(resolve("heuristic"), HeuristicTokenizer)


def test_empty_and_nonnegative():
    t = SubwordHeuristicTokenizer()
    assert t.count("") == 0
    assert t.count("x") >= 1


def test_length_aware_long_identifiers_cost_more():
    t = SubwordHeuristicTokenizer()
    # BPE splits long identifiers into more tokens; the flat heuristic treats every piece the
    # same. So subword must charge a long compound identifier more than a short word.
    assert t.count("configure_connection_pool_max_attempts") > t.count("cat")
    # and more than the flat heuristic would for the same single piece
    long_id = "configuration"
    assert t.count(long_id) > HeuristicTokenizer().count(long_id)


def test_non_ascii_surcharge():
    t = SubwordHeuristicTokenizer()
    assert t.count("café résumé naïve") > t.count("cafe resume naive")  # multi-byte costs extra


def test_monotonic_in_length():
    t = SubwordHeuristicTokenizer()
    short, long = "retry policy", "retry policy configuration with exponential backoff settings"
    assert t.count(long) > t.count(short)

"""
core/coverage.py — pair selection and saturation.

Small module, but it decides two things a run depends on: which pair each
iteration fuses, and when `run()` is allowed to stop. It was rewritten
twice recently (the producer-consumer heuristic was removed, then pair
de-duplication was removed and put back), so the contract is worth pinning:

  * a pair not yet fused is preferred, and recorded once handed out
  * once every pair is covered, selection keeps working (re-fusing a pair
    still yields a different child, since each technique re-randomises)
    rather than looping forever or returning None
  * is_saturated is what lets run() stop on a small corpus
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.coverage import PairwiseCoverageMatrix  # noqa: E402
from core.fusion import Seed  # noqa: E402


def _corpus(n):
    return [Seed(content=f"x{i}", metadata={}, id=f"id{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def test_pairs_are_unordered():
    cov = PairwiseCoverageMatrix()
    cov.record("a", "b")
    assert cov.has_been_fused("a", "b")
    assert cov.has_been_fused("b", "a"), "(a,b) and (b,a) are the same pair"


def test_recording_the_same_pair_twice_counts_once():
    cov = PairwiseCoverageMatrix()
    cov.record("a", "b")
    cov.record("b", "a")
    assert cov.covered_count() == 1


def test_unknown_pair_is_not_fused():
    cov = PairwiseCoverageMatrix()
    assert not cov.has_been_fused("a", "b")
    assert cov.covered_count() == 0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_selection_records_the_pair_it_returns():
    cov = PairwiseCoverageMatrix()
    a, b = cov.select_parents(_corpus(5))
    assert cov.has_been_fused(a.id, b.id)
    assert cov.covered_count() == 1


def test_selection_never_returns_the_same_seed_twice():
    cov = PairwiseCoverageMatrix()
    corpus = _corpus(4)
    for _ in range(50):
        a, b = cov.select_parents(corpus)
        assert a is not b and a.id != b.id


def test_every_pair_is_eventually_covered():
    """4 seeds = 6 pairs; the retry budget should find them all."""
    cov = PairwiseCoverageMatrix()
    corpus = _corpus(4)
    for _ in range(6):
        cov.select_parents(corpus)
    assert cov.covered_count() == 6
    assert cov.is_saturated(corpus)


def test_selection_keeps_working_after_saturation():
    """Past saturation the matrix hands out a repeat rather than stalling —
    re-fusing a pair still produces a different child."""
    cov = PairwiseCoverageMatrix()
    corpus = _corpus(3)
    for _ in range(3):
        cov.select_parents(corpus)
    assert cov.is_saturated(corpus)
    for _ in range(20):
        a, b = cov.select_parents(corpus)
        assert a is not None and b is not None


# ---------------------------------------------------------------------------
# Saturation / edges
# ---------------------------------------------------------------------------

def test_saturation_needs_every_pair():
    cov = PairwiseCoverageMatrix()
    corpus = _corpus(3)          # 3 pairs
    cov.record("id0", "id1")
    cov.record("id0", "id2")
    assert not cov.is_saturated(corpus)
    cov.record("id1", "id2")
    assert cov.is_saturated(corpus)


def test_empty_corpus_is_not_saturated_and_yields_nothing():
    cov = PairwiseCoverageMatrix()
    assert not cov.is_saturated([])
    assert cov.select_parents([]) == (None, None)


def test_single_seed_corpus_self_pairs():
    """One seed has no pair partner; self-fusion is the only option and
    must not crash or return None."""
    cov = PairwiseCoverageMatrix()
    corpus = _corpus(1)
    assert not cov.is_saturated(corpus)
    a, b = cov.select_parents(corpus)
    assert a is corpus[0] and b is corpus[0]


def test_coverage_ratio_is_capped_at_one():
    cov = PairwiseCoverageMatrix()
    corpus = _corpus(3)
    assert cov.coverage_ratio(corpus) == 0.0
    for _ in range(3):
        cov.select_parents(corpus)
    assert cov.coverage_ratio(corpus) == 1.0
    # ids no longer in the corpus must not push the ratio above 1
    cov.record("ghost1", "ghost2")
    assert cov.coverage_ratio(corpus) <= 1.0
    assert cov.coverage_ratio([]) == 1.0


def test_no_producer_consumer_scoring_remains():
    """The heuristic was removed; selection must not depend on seed
    content or metadata in any way."""
    cov = PairwiseCoverageMatrix()
    assert not hasattr(cov, "guided")
    assert not hasattr(cov, "_profile_cache")
    bare = [Seed(content="", metadata={}, id=f"n{i}") for i in range(4)]
    a, b = cov.select_parents(bare)      # no metadata at all
    assert a is not None and b is not None

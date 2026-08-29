import random
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .fusion import Seed

logger = logging.getLogger("FFL.Coverage")


class PairwiseCoverageMatrix:
    """
    Tracks which (seed_a, seed_b) pairs have been fused together, and picks
    the next pair uniformly at random from those not yet fused.

    Internally a set of frozensets, so (a, b) and (b, a) are the same pair.
    De-duplication is the only bias applied: among unfused pairs every
    combination is equally likely. There is no compatibility scoring — the
    producer-consumer heuristic that used to gate this selection was
    removed; picking pairs on a static "does A produce what B consumes"
    signal cost throughput without a measured payoff.

    Once every pair has been fused the matrix is saturated: selection falls
    back to re-fusing a random pair (which still yields a different child,
    since every technique re-randomises its own choices) and callers can
    use is_saturated() to stop instead.
    """

    def __init__(self):
        self._fused: set = set()

    def record(self, id_a: str, id_b: str) -> None:
        """Mark the pair (id_a, id_b) as fused."""
        self._fused.add(frozenset((id_a, id_b)))

    def has_been_fused(self, id_a: str, id_b: str) -> bool:
        return frozenset((id_a, id_b)) in self._fused

    def covered_count(self) -> int:
        return len(self._fused)

    def is_saturated(self, corpus) -> bool:
        """Return True when every possible pair in corpus has been fused."""
        n = len(corpus)
        total = n * (n - 1) // 2
        return total > 0 and len(self._fused) >= total

    def coverage_ratio(self, corpus) -> float:
        """Fraction of all possible pairs that have been fused."""
        n = len(corpus)
        total = n * (n - 1) // 2
        if total == 0:
            return 1.0
        # _fused may contain IDs no longer in corpus; cap at total
        return min(len(self._fused), total) / total

    def select_parents(self, corpus) -> Tuple[Optional["Seed"], Optional["Seed"]]:
        """
        Two seeds drawn uniformly at random, preferring a pair that has not
        been fused yet.

        Retry budget scales with corpus size so large corpora still find an
        uncovered pair without excessive looping; once the budget is spent
        (i.e. coverage is close to saturated) a random pair is used and
        recorded anyway, so fuzzing never stalls.
        """
        n = len(corpus)
        if n < 2:
            return (corpus[0], corpus[0]) if corpus else (None, None)

        max_tries = max(n, 50)
        for _ in range(max_tries):
            a, b = random.sample(corpus, 2)
            if not self.has_been_fused(a.id, b.id):
                self.record(a.id, b.id)
                return a, b

        logger.debug("Pairwise coverage saturated; falling back to random selection.")
        a, b = random.sample(corpus, 2)
        self.record(a.id, b.id)
        return a, b

import random
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .fusion import Seed

logger = logging.getLogger("FFL.Coverage")


class PairwiseCoverageMatrix:
    """
    Tracks which (seed_a, seed_b) pairs have been fused together.

    Internally stored as a set of frozensets so order doesn't matter.
    Guides parent selection toward unexplored combinations; falls back
    to random sampling once all pairs are saturated.
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
        Return two seeds that haven't been fused before, recording the pair.
        Falls back to random when all pairs are exhausted.
        """
        n = len(corpus)
        if n < 2:
            return (corpus[0], corpus[0]) if corpus else (None, None)

        # Random retry loop: fast when coverage is sparse, degrades gracefully
        # as saturation approaches.  Retry budget scales with corpus size so
        # large corpora still find uncovered pairs without excessive looping.
        max_tries = max(n, 50)
        for _ in range(max_tries):
            a, b = random.sample(corpus, 2)
            if not self.has_been_fused(a.id, b.id):
                self.record(a.id, b.id)
                return a, b

        # All reachable pairs covered — allow re-fusion so fuzzing continues
        logger.debug("Pairwise coverage saturated; falling back to random selection.")
        a, b = random.sample(corpus, 2)
        self.record(a.id, b.id)
        return a, b

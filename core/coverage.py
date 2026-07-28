import random
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

from .resource_matching import extract_resource_profile, compatibility_score

if TYPE_CHECKING:
    from .fusion import Seed

logger = logging.getLogger("FFL.Coverage")


class PairwiseCoverageMatrix:
    """
    Tracks which (seed_a, seed_b) pairs have been fused together.

    Internally stored as a set of frozensets so order doesn't matter.
    Guides parent selection toward unexplored, producer/consumer-compatible
    combinations (see core/resource_matching.py) when guided_fusion_enabled
    (--guided-fusion) is on; falls back to plain unfused sampling, then
    fully random sampling, once those are exhausted.

    --guided-fusion only takes effect when pre_analysis_enabled (--pre-
    analysis) is also on: extract_resource_profile's produce/consume
    signal is built entirely from dry-run-collected metadata (var_types,
    struct_names, functions, imports, ...) — with that metadata absent, it
    would fall back to raw-token overlap, a much weaker, closer-to-noise
    signal not worth the cost of scoring every candidate pair. So
    compatibility scoring only runs when *both* flags are on; otherwise
    select_parents just prefers an unfused pair uniformly at random.
    """

    def __init__(self, pre_analysis_enabled: bool = True, guided_fusion_enabled: bool = False):
        self._fused: set = set()
        self._profile_cache: dict = {}
        self.guided = guided_fusion_enabled and pre_analysis_enabled
        if guided_fusion_enabled and not pre_analysis_enabled:
            logger.warning(
                "--guided-fusion has no effect without --pre-analysis (its "
                "compatibility signal is built entirely from dry-run-"
                "collected metadata) — falling back to unfused-pair-only "
                "random parent selection."
            )

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

    def _profile(self, seed: "Seed"):
        prof = self._profile_cache.get(seed.id)
        if prof is None:
            prof = extract_resource_profile(seed)
            self._profile_cache[seed.id] = prof
        return prof

    def select_parents(self, corpus) -> Tuple[Optional["Seed"], Optional["Seed"]]:
        """
        Producer-consumer guided selection: among unfused pairs, prefer one
        where a resource one seed produces matches a consumption site the
        other exposes (core/resource_matching.py). Falls back to the best
        unfused-but-incompatible pair seen during the search, then to plain
        random sampling once all pairs are saturated — so sparse metadata
        never stalls fuzzing, it just loses the compatibility bias.

        Without --guided-fusion (or without --pre-analysis alongside it —
        see class docstring), skips compatibility scoring entirely and
        just prefers an unfused pair uniformly at random.
        """
        n = len(corpus)
        if n < 2:
            return (corpus[0], corpus[0]) if corpus else (None, None)

        # Retry loop: fast when coverage is sparse, degrades gracefully as
        # saturation approaches. Retry budget scales with corpus size so
        # large corpora still find uncovered pairs without excessive looping.
        max_tries = max(n, 50)

        if not self.guided:
            for _ in range(max_tries):
                a, b = random.sample(corpus, 2)
                if not self.has_been_fused(a.id, b.id):
                    self.record(a.id, b.id)
                    return a, b
            logger.debug("Pairwise coverage saturated; falling back to random selection.")
            a, b = random.sample(corpus, 2)
            self.record(a.id, b.id)
            return a, b

        fallback = None
        for _ in range(max_tries):
            a, b = random.sample(corpus, 2)
            if self.has_been_fused(a.id, b.id):
                continue
            if fallback is None:
                fallback = (a, b)
            if compatibility_score(self._profile(a), self._profile(b)) > 0:
                self.record(a.id, b.id)
                return a, b

        if fallback is not None:
            logger.debug("No producer/consumer-compatible unfused pair found in budget; using best unfused fallback.")
            self.record(fallback[0].id, fallback[1].id)
            return fallback

        # All reachable pairs covered — allow re-fusion so fuzzing continues
        logger.debug("Pairwise coverage saturated; falling back to random selection.")
        a, b = random.sample(corpus, 2)
        self.record(a.id, b.id)
        return a, b

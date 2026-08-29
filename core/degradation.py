"""
core/degradation.py — counters for the paths that quietly do nothing.

Most of this fuzzer's failure modes are not crashes. They are silent
degradations: a strategy raises and the iteration is skipped, a fusion
produces the host unchanged, a cached field is missing so an expensive
analysis runs again, a crash bundle can't resolve a parent. Every one of
those keeps the run looking healthy — throughput stays up, the status bar
scrolls — while a technique contributes nothing.

Real examples this module exists because of:
  * a NameError in one import line disabled Swift *and* Clang state fusion
    completely; the only trace was a WARNING line among thousands
  * MLIR state fusion emitted the host verbatim on 30/30 pairs, because the
    donor continuation was always cut to empty. No signal at all
  * `has_declaration` was never written to the corpus, so every candidate
    pair re-scanned both seeds forever
  * two thirds of saved crash bundles were missing a parent file

So: count them, and print the counts at the end of a run. A number that
should be zero and isn't is far easier to notice than a log line that
should have appeared and didn't.

Thread-safe: fusion runs on a worker pool, and Counter's += is a
read-modify-write that loses increments under concurrency.
"""

import logging
import threading
from collections import Counter
from typing import Dict, Optional

logger = logging.getLogger("FFL.Degradation")

# Cap on distinct detail strings kept per category, so an unbounded source
# (e.g. exception messages carrying seed ids) can't grow without limit.
_MAX_DETAILS_PER_CATEGORY = 8


class DegradationLog:
    """Counts of degraded outcomes, by category and optional detail."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: Counter = Counter()
        self._details: Dict[str, Counter] = {}

    def record(self, category: str, detail: Optional[str] = None) -> None:
        with self._lock:
            self._counts[category] += 1
            if detail is not None:
                bucket = self._details.setdefault(category, Counter())
                if detail in bucket or len(bucket) < _MAX_DETAILS_PER_CATEGORY:
                    bucket[detail] += 1
                else:
                    bucket["(other)"] += 1

    def count(self, category: str) -> int:
        with self._lock:
            return self._counts[category]

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._details.clear()

    def report(self, total_iterations: int = 0, log=logger) -> None:
        """Print what degraded, loudest first. Silent when nothing did."""
        with self._lock:
            counts = dict(self._counts)
            details = {k: dict(v) for k, v in self._details.items()}
        if not counts:
            return

        log.info("Degradations (paths that produced no fused program, or lost information):")
        for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            share = f" ({100.0 * n / total_iterations:.1f}% of iterations)" if total_iterations else ""
            log.info(f"    {n:8d}  {category}{share}")
            for detail, dn in sorted(details.get(category, {}).items(), key=lambda kv: -kv[1]):
                log.info(f"             {dn:8d}  {detail}")


# One shared instance: the alternative is threading a log object through
# fusion, orchestration and analysis, which is a lot of plumbing for a
# diagnostic counter.
degradations = DegradationLog()


def record(category: str, detail: Optional[str] = None) -> None:
    degradations.record(category, detail)

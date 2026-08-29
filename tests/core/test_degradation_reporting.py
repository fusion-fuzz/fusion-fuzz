"""
Degraded outcomes must be counted, not swallowed.

The fuzzer's characteristic failure is not a crash, it is a technique that
quietly stops contributing while throughput stays healthy. `_fuse_once`
catches Exception and skips the iteration; a strategy that finds nothing to
apply returns the host unchanged; a crash bundle silently omits a parent it
cannot resolve. Each of those looks exactly like a normal run.

These tests pin that every one of those paths increments a counter, and
that the counter names the *strategy*, not just the exception — a NameError
inside one language's strategy is otherwise indistinguishable from a
malformed seed, which is precisely how one went unnoticed for a whole
session of work.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.degradation import DegradationLog, degradations  # noqa: E402
from core.fusion import Seed  # noqa: E402
from core.orchestrator import FusionFuzzLoop  # noqa: E402


# ---------------------------------------------------------------------------
# The counter itself
# ---------------------------------------------------------------------------

def test_counts_by_category_and_detail():
    log = DegradationLog()
    log.record("fusion raised", "ClangStateFusionStrategy: NameError")
    log.record("fusion raised", "ClangStateFusionStrategy: NameError")
    log.record("fusion raised", "MLIRStateFusionStrategy: IndexError")
    log.record("no viable strategy for the selected pair")

    assert log.count("fusion raised") == 3
    assert log.count("no viable strategy for the selected pair") == 1
    assert log.snapshot() == {"fusion raised": 3,
                              "no viable strategy for the selected pair": 1}


def test_detail_cardinality_is_bounded():
    """An unbounded detail source (exception text carrying seed ids) must
    not grow the log without limit."""
    log = DegradationLog()
    for i in range(200):
        log.record("fusion raised", f"unique detail {i}")
    assert log.count("fusion raised") == 200
    # details are capped; the overflow lands in a single "(other)" bucket
    assert len(log._details["fusion raised"]) <= 9


def test_report_is_silent_when_nothing_degraded(caplog):
    log = DegradationLog()
    with caplog.at_level("INFO"):
        log.report()
    assert caplog.text == ""


def test_report_names_the_categories(caplog):
    log = DegradationLog()
    log.record("fusion raised", "SwiftStateFusionStrategy: NameError")
    with caplog.at_level("INFO"):
        log.report(total_iterations=100)
    assert "fusion raised" in caplog.text
    assert "SwiftStateFusionStrategy: NameError" in caplog.text
    assert "1.0% of iterations" in caplog.text


def test_reset_clears_everything():
    log = DegradationLog()
    log.record("a", "d")
    log.reset()
    assert log.snapshot() == {}


# ---------------------------------------------------------------------------
# The loop's instrumentation
# ---------------------------------------------------------------------------

class _RaisingStrategy:
    """Stands in for the real bug: a strategy that raises NameError on every
    call, which the loop catches and logs at WARNING."""

    def is_viable_pair(self, a, b):
        return True

    def fuse(self, a, b):
        raise NameError("name 'truncate_to_balanced' is not defined")


class _NoOpStrategy:
    """A strategy that reports, via its mode, that it applied nothing."""

    def is_viable_pair(self, a, b):
        return True

    def fuse(self, a, b):
        return Seed(content=a.content,
                    metadata={"parents": [a.id, b.id], "mode": "state_nocontinuation_ab"})


class _UnviableStrategy:
    def is_viable_pair(self, a, b):
        return False

    def fuse(self, a, b):  # pragma: no cover - never reached
        raise AssertionError("should not be called")


def _loop(strategies):
    corpus = [Seed(content=f"x = {i}\n", metadata={}) for i in range(4)]
    config = {"project_name": "cpython", "execution": {"concurrency": 1}}
    return FusionFuzzLoop(config=config, strategies=strategies, initial_corpus=corpus)


def test_raising_strategy_is_counted_and_names_the_strategy():
    degradations.reset()
    loop = _loop([_RaisingStrategy()])
    for _ in range(5):
        loop._fuse_once()

    snap = degradations.snapshot()
    assert snap.get("fusion raised") == 5, snap
    detail = degradations._details["fusion raised"]
    assert any("_RaisingStrategy" in d and "NameError" in d for d in detail), detail


def test_strategy_that_applies_nothing_is_counted():
    """Counted per *child*, not per iteration: one pair yields
    children_per_pair children (default 2), and each one that came out as
    the host alone is a wasted execution in its own right."""
    degradations.reset()
    loop = _loop([_NoOpStrategy()])
    for _ in range(3):
        loop._fuse_once()
    snap = degradations.snapshot()
    assert snap.get("fusion applied nothing (child ~= host)") == 3 * loop.children_per_pair, snap


def test_unviable_pair_is_counted():
    degradations.reset()
    loop = _loop([_UnviableStrategy()])
    for _ in range(4):
        loop._fuse_once()
    assert degradations.snapshot().get("no viable strategy for the selected pair") == 4


def test_unresolvable_crash_parent_is_counted():
    degradations.reset()
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        loop = _loop([_NoOpStrategy()])
        loop.original_cwd = tmp
        child = Seed(content="x", metadata={"root_parents": ["ghost_a", "ghost_b"]})

        class _Result:
            return_code, stdout, stderr, execution_time, crashed = 1, "", "boom", 0.1, True
            signature = command = None

        loop._save_crash_bundle(child, _Result(), "sig_ghosts")
    finally:
        os.chdir(cwd)
    assert degradations.snapshot().get("crash bundle: parent not resolvable") == 1


def test_healthy_fusion_records_no_degradation():
    """The counter must stay at zero on a normal run, or it is noise."""
    class _GoodStrategy:
        def is_viable_pair(self, a, b):
            return True

        def fuse(self, a, b):
            return Seed(content=a.content + b.content,
                        metadata={"parents": [a.id, b.id], "mode": "state4_live2_ab"})

    degradations.reset()
    loop = _loop([_GoodStrategy()])
    for _ in range(10):
        children, _, _ = loop._fuse_once()
        assert children
    assert degradations.snapshot() == {}

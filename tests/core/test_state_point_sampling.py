"""
State-fusion splice-point selection: proportional-to-live-variables sampling.

The previous behaviour kept only the points whose live-variable count
equalled the file's maximum and then chose uniformly among those, so a
point with 3 live variables next to one with 4 could never be selected at
all. Selection is now weighted: every safe point is eligible and a point's
chance of being drawn is proportional to its live-variable count.

These tests pin that, plus the fallbacks it has to keep:
  * more live variables => selected more often
  * non-maximum points stay selectable
  * zero / missing live counts fall back to distance-to-centre weighting
    (5 points -> 1,2,3,2,1), never to a crash
  * every line is eligible — the old safety filter is gone by design
  * a seeded `random` still gives a reproducible sequence
"""

import collections
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.state_analysis import (  # noqa: E402
    StatePoint,
    _center_weights,
    find_state_points,
    find_most_complex_state,
    pick_state_point,
    weighted_order,
)

# Live-variable counts grow line by line, so the last lines carry the
# highest weight and the early ones a lower — but nonzero — weight.
GRADED = """a = 1
b = 2
c = 3
d = 4
e = 5
"""

# Nothing is ever declared: every safe line has weight 0.
NO_VARS = """print(1)
print(2)
print(3)
print(4)
"""


def _counts_by_line(content, language, draws=4000, seed=0):
    random.seed(seed)
    points = find_state_points(content, language)
    tally = collections.Counter()
    for _ in range(draws):
        tally[pick_state_point(content, language,
                               cached=[p.to_dict() for p in points]).line_idx] += 1
    return points, tally


def test_all_safe_points_are_eligible_not_just_the_maximum():
    points = find_state_points(GRADED, "cpython")
    counts = {p.line_idx: p.live_count for p in points}
    assert len(points) > 1
    # The old implementation returned only the maximum-count lines.
    assert len(set(counts.values())) > 1, f"only one distinct weight: {counts}"
    assert min(counts.values()) < max(counts.values())


def test_higher_live_count_is_selected_more_often():
    points, tally = _counts_by_line(GRADED, "cpython")
    weight = {p.line_idx: p.live_count for p in points}
    heavy = max(weight, key=lambda i: weight[i])
    light = min((i for i in weight if weight[i] > 0), key=lambda i: weight[i])
    assert weight[heavy] > weight[light]
    assert tally[heavy] > tally[light], (
        f"line {heavy} (weight {weight[heavy]}) drawn {tally[heavy]}x but "
        f"line {light} (weight {weight[light]}) drawn {tally[light]}x"
    )


def test_selection_frequency_tracks_the_weights():
    """Not just ordering: the draw rate should be near weight/sum(weights)."""
    points, tally = _counts_by_line(GRADED, "cpython", draws=8000, seed=7)
    total_weight = sum(p.live_count for p in points)
    draws = sum(tally.values())
    for p in points:
        if p.live_count == 0:
            continue
        expected = p.live_count / total_weight
        observed = tally[p.line_idx] / draws
        assert abs(observed - expected) < 0.05, (
            f"line {p.line_idx}: expected ~{expected:.3f}, observed {observed:.3f}"
        )


def test_non_maximum_points_are_actually_selected():
    points, tally = _counts_by_line(GRADED, "cpython")
    best = max(p.live_count for p in points)
    below_max = {p.line_idx for p in points if 0 < p.live_count < best}
    assert below_max, "fixture has no below-maximum point to test with"
    assert below_max & set(tally), "no below-maximum point was ever selected"


@pytest.mark.parametrize("language", ["cpython", "php", "clang"])
def test_points_carry_their_weight_through_the_metadata_cache(language):
    src = {"cpython": GRADED,
           "php": "<?php\n$a = 1;\n$b = 2;\n$c = 3;\n",
           "clang": "int main(void) {\n  int a = 1;\n  int b = 2;\n  int c = 3;\n}\n"}[language]
    points = find_state_points(src, language)
    assert points
    round_tripped = [StatePoint.from_dict(p.to_dict()) for p in points]
    assert [p.live_count for p in round_tripped] == [p.live_count for p in points]


def test_zero_live_variable_content_falls_back_safely():
    points = find_state_points(NO_VARS, "cpython")
    assert points, "a file with no declarations must still yield a splice point"
    assert all(p.live_count == 0 for p in points)
    assert all(p.category == "nolive" for p in points)
    random.seed(3)
    chosen = pick_state_point(NO_VARS, "cpython")
    assert chosen is not None
    assert 0 <= chosen.line_idx < len(NO_VARS.splitlines())


def test_center_weights_are_triangular():
    assert _center_weights(5) == [1, 2, 3, 2, 1]
    assert _center_weights(4) == [1, 2, 2, 1]
    assert _center_weights(1) == [1]
    assert _center_weights(0) == []


def test_zero_weight_fallback_prefers_the_middle():
    """With no live variables to weight by, selection is weighted by
    distance to the centre (1,2,3,2,1 for five points), not uniform."""
    five = "x\ny\nz\nw\nv\n"          # 5 safe lines, no declarations
    points = find_state_points(five, "cpython")
    assert len(points) == 5 and all(p.live_count == 0 for p in points)

    cached = [p.to_dict() for p in points]
    random.seed(0)
    draws = 9000
    tally = collections.Counter(
        pick_state_point(five, "cpython", cached=cached).line_idx
        for _ in range(draws))

    expected = [w / sum(_center_weights(5)) for w in _center_weights(5)]
    for idx, want in enumerate(expected):
        got = tally[idx] / draws
        assert abs(got - want) < 0.04, f"line {idx}: want ~{want:.3f}, got {got:.3f}"
    # The ordering that matters: middle > shoulders > ends.
    assert tally[2] > tally[1] > tally[0]
    assert tally[2] > tally[3] > tally[4]


def test_all_zero_weight_cache_does_not_collapse_to_one_point():
    """A stale cache (or a weightless file) must not crash on an all-zero
    weight vector, and must still vary its choice."""
    cached = [StatePoint(i, "nolive", "", "", 0).to_dict() for i in range(4)]
    random.seed(0)
    seen = {pick_state_point(NO_VARS, "cpython", cached=cached).line_idx
            for _ in range(200)}
    assert len(seen) > 1, "zero-weight fallback never varied its choice"


def test_weighted_order_zero_weights_prefer_the_middle():
    points = [StatePoint(i, "nolive", "", "", 0) for i in range(5)]
    random.seed(1)
    firsts = collections.Counter(weighted_order(points)[0].line_idx
                                 for _ in range(4000))
    assert firsts[2] > firsts[1] > firsts[0]
    assert firsts[2] > firsts[3] > firsts[4]


def test_cache_predating_live_count_recovers_the_weight_from_the_category():
    """Entries written before live_count existed still carry the count in
    their category ("live7"); it must be recovered, not silently zeroed."""
    legacy = {"line_idx": 4, "category": "live7", "matched_text": "", "indent": ""}
    assert StatePoint.from_dict(legacy).live_count == 7
    unknown = {"line_idx": 1, "category": "nolive", "matched_text": "", "indent": ""}
    assert StatePoint.from_dict(unknown).live_count == 0


def test_every_line_is_an_eligible_anchor():
    """The old "is this line safe to anchor at" filter (real code rather
    than string/comment interior, non-negative delimiter depth) was removed
    deliberately: structural integrity is enforced where the text is cut
    (segment_boundaries requires depth exactly zero), and an anchor landing
    somewhere odd yields an ill-formed child, which is a legitimate input
    to a compiler rather than something to filter out up front."""
    src = (
        'a = 1\n'
        'blob = """\n'
        'x = 111\n'
        'y = 222\n'
        '"""\n'
        'b = 2\n'
    )
    offered = {p.line_idx for p in find_state_points(src, "cpython")}
    assert offered == set(range(len(src.splitlines()))), (
        f"some lines were filtered out: {set(range(6)) - offered}")


def test_selection_is_deterministic_for_a_seeded_rng():
    def run():
        random.seed(1234)
        return [pick_state_point(GRADED, "cpython").line_idx for _ in range(50)]
    assert run() == run()

    # An explicit Random instance is independent of the global one.
    a = [pick_state_point(GRADED, "cpython", rng=random.Random(9)).line_idx
         for _ in range(50)]
    b = [pick_state_point(GRADED, "cpython", rng=random.Random(9)).line_idx
         for _ in range(50)]
    assert a == b


def test_weighted_order_is_a_permutation_and_prefers_heavy_points():
    points = [StatePoint(0, "live1", "", "", 1), StatePoint(1, "live9", "", "", 9)]
    random.seed(5)
    firsts = collections.Counter(weighted_order(points)[0].line_idx for _ in range(2000))
    # StatePoint is an unfrozen dataclass (unhashable), so compare by index.
    assert sorted(p.line_idx for p in weighted_order(points)) == [0, 1]
    assert firsts[1] > firsts[0], f"weighted_order ignored the weights: {firsts}"


def test_weighted_order_handles_empty_and_all_zero_input():
    assert weighted_order([]) == []
    zeros = [StatePoint(i, "nolive", "", "", 0) for i in range(3)]
    random.seed(2)
    assert sorted(p.line_idx for p in weighted_order(zeros)) == [0, 1, 2]


def test_legacy_alias_still_resolves():
    assert find_most_complex_state is find_state_points

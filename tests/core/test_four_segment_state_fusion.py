"""
State fusion must interleave four segments, not graft a suffix.

Given programs A and B with fusion points p and q, the result has to be

    A_prefix + B_prefix + A_suffix + B_suffix

with A_prefix/A_suffix partitioning A around p, B_prefix/B_suffix
partitioning B around q, every non-empty segment represented, and each
input's own statement order intact.

The scheme this replaces produced A_prefix + B_suffix + A_suffix: B's
prefix was dropped entirely, so B's continuation ran without the code that
established the state it expects, and B's own setup never met A at all.
Several tests below assert specifically that that shape is gone.

Segment inclusion rule under test: the fusion-point line belongs to the
*prefix* (a StatePoint denotes the state *after* that line executes).
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.fusion import Seed, get_strategies  # noqa: E402
from core.state_analysis import (  # noqa: E402
    StatePoint,
    interleave_segments,
    segment_boundaries,
    split_at_point,
)

A_SRC = "a0 = 0\na1 = 1\na2 = 2\na3 = 3\n"
B_SRC = "b0 = 0\nb1 = 1\nb2 = 2\nb3 = 3\n"


def _point(idx):
    return StatePoint(idx, f"live{idx}", "", "", idx)


def _lines(text):
    return [ln.split("#")[0].strip() for ln in text.splitlines() if ln.strip()]


def _positions(text, names):
    body = _lines(text)
    return [body.index(n) for n in names]


# ---------------------------------------------------------------------------
# 1. All four segments present, in the required order
# ---------------------------------------------------------------------------

def test_all_four_segments_appear_in_order():
    text, seg = interleave_segments(A_SRC, B_SRC, _point(1), _point(1), "cpython")
    body = _lines(text)
    assert body == ["a0 = 0", "a1 = 1", "b0 = 0", "b1 = 1",
                    "a2 = 2", "a3 = 3", "b2 = 2", "b3 = 3"]
    assert seg.a_split == 2 and seg.b_split == 2
    assert seg.a_prefix and seg.b_prefix and seg.a_suffix and seg.b_suffix


def test_fusion_point_line_belongs_to_the_prefix():
    """The point denotes the state *after* its line, so that line is the
    last thing in the prefix."""
    _, seg = interleave_segments(A_SRC, B_SRC, _point(2), _point(0), "cpython")
    assert seg.a_prefix[-1] == "a2 = 2"
    assert seg.a_suffix[0] == "a3 = 3"
    assert seg.b_prefix == ["b0 = 0"]
    assert seg.b_suffix[0] == "b1 = 1"


def test_old_suffix_graft_shape_is_rejected():
    """The previous scheme emitted A_prefix + B_suffix + A_suffix, i.e. B's
    prefix missing and B's suffix in the middle. Both must be false now."""
    text, _ = interleave_segments(A_SRC, B_SRC, _point(1), _point(1), "cpython")
    body = _lines(text)

    assert "b0 = 0" in body, "B_prefix was dropped — this is the old graft shape"
    a2, b2 = body.index("a2 = 2"), body.index("b2 = 2")
    assert a2 < b2, "B_suffix must follow A_suffix, not be spliced before it"

    # And the exact old sequence must not be producible.
    old_shape = ["a0 = 0", "a1 = 1", "b2 = 2", "b3 = 3", "a2 = 2", "a3 = 3"]
    assert body != old_shape


# ---------------------------------------------------------------------------
# 2. Order preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,q", [(0, 0), (0, 2), (1, 1), (2, 0), (3, 3)])
def test_relative_order_preserved_within_each_input(p, q):
    text, _ = interleave_segments(A_SRC, B_SRC, _point(p), _point(q), "cpython")
    a_pos = _positions(text, ["a0 = 0", "a1 = 1", "a2 = 2", "a3 = 3"])
    b_pos = _positions(text, ["b0 = 0", "b1 = 1", "b2 = 2", "b3 = 3"])
    assert a_pos == sorted(a_pos), f"A reordered: {a_pos}"
    assert b_pos == sorted(b_pos), f"B reordered: {b_pos}"


def test_every_statement_survives_exactly_once():
    text, _ = interleave_segments(A_SRC, B_SRC, _point(2), _point(1), "cpython")
    body = _lines(text)
    for name in ["a0 = 0", "a1 = 1", "a2 = 2", "a3 = 3",
                 "b0 = 0", "b1 = 1", "b2 = 2", "b3 = 3"]:
        assert body.count(name) == 1, f"{name} appears {body.count(name)}x"


# ---------------------------------------------------------------------------
# 3. Edge cases: empty segments, boundaries
# ---------------------------------------------------------------------------

def test_point_on_last_line_leaves_empty_suffixes():
    """Both suffixes empty is legal — the result degenerates to A + B."""
    text, seg = interleave_segments(A_SRC, B_SRC, _point(3), _point(3), "cpython")
    assert seg.a_suffix == [] and seg.b_suffix == []
    assert _lines(text) == ["a0 = 0", "a1 = 1", "a2 = 2", "a3 = 3",
                            "b0 = 0", "b1 = 1", "b2 = 2", "b3 = 3"]


def test_single_line_inputs():
    text, seg = interleave_segments("a = 1\n", "b = 2\n", _point(0), _point(0), "cpython")
    assert _lines(text) == ["a = 1", "b = 2"]
    assert seg.a_suffix == [] and seg.b_suffix == []


def test_empty_input_yields_no_fusion():
    assert interleave_segments("", B_SRC, _point(0), _point(0), "cpython") is None
    assert interleave_segments(A_SRC, "", _point(0), _point(0), "cpython") is None


def test_split_snaps_to_a_legal_boundary_in_brace_languages():
    """A point inside a function body is not a legal cut: the prefix would
    leave an unclosed brace. The split moves to the next depth-0 line."""
    src = "int f(void){\n  int x = 1;\n  int y = 2;\n}\nint g = 3;\n"
    assert segment_boundaries(src, "clang") == [3, 4]
    prefix, suffix, split = split_at_point(src, _point(1), "clang")
    assert split == 4, "cut must land after the closing brace"
    assert prefix[-1] == "}"
    assert suffix == ["int g = 3;"]


def test_python_split_never_separates_a_header_from_its_body():
    src = "x = 1\ndef f():\n    return 2\ny = 3\n"
    # line 1 is `def f():` — cutting after it would orphan the body
    assert 1 not in segment_boundaries(src, "cpython")
    prefix, suffix, split = split_at_point(src, _point(1), "cpython")
    assert split == 3 and prefix[-1] == "    return 2"


def test_brace_balance_is_preserved_in_the_fused_output():
    a = "int fa(void){\n  int x = 1;\n  return x;\n}\nint ga = 5;\n"
    b = "int fb(void){\n  int y = 2;\n  return y;\n}\nint gb = 7;\n"
    text, _ = interleave_segments(a, b, _point(1), _point(1), "clang")
    assert text.count("{") == text.count("}")


# ---------------------------------------------------------------------------
# 4. Real strategies use the new path
# ---------------------------------------------------------------------------

FOUR_SEGMENT_LANGUAGES = {
    "cpython": "a%d = %d\nb%d = a%d + 1\nprint(b%d)\nc%d = 9\n",
    "clang": "int f%d(void){\n  int x = %d;\n  return x;\n}\nint g%d = %d;\n",
    "php": "--TEST--\nt%d\n--FILE--\n<?php\n$a%d = %d;\n$b%d = $a%d + 1;\nvar_dump($b%d);\n?>\n",
    "swift": "let n%d: Int = %d\nlet m%d: Int = n%d + 1\nprint(m%d)\n",
    "flang": "program p%d\n  integer :: x%d\n  x%d = %d\n  print *, x%d\nend program\n",
    "haskell": "f%d :: Int -> Int\nf%d a = a + %d\n",
    "naga": "fn f%d() -> i32 {\n  return %d;\n}\nvar<private> g%d: i32 = %d;\n",
}


@pytest.mark.parametrize("project,template", sorted(FOUR_SEGMENT_LANGUAGES.items()))
def test_language_strategy_uses_four_segment_path(project, template):
    """`state4_*` in the mode is how a child records that it came from the
    four-segment path rather than the graft fallback."""
    random.seed(5)
    slots = template.count("%d")
    seeds = [Seed(content=template % tuple([i] * slots), metadata={}) for i in (0, 1)]
    strategy = [s for s in get_strategies(project, state_fusion=True, pre_analysis_enabled=True)
                if "State" in type(s).__name__][0]

    modes = set()
    for i in range(20):
        random.seed(i)
        modes.add(strategy.fuse(seeds[0], seeds[1]).metadata.get("mode", "?"))
    assert any(m.startswith("state4_") for m in modes), f"never took the four-segment path: {modes}"


def test_strategy_records_segment_ranges_in_metadata():
    random.seed(3)
    a = Seed(content="a0 = 0\na1 = 1\na2 = 2\n", metadata={})
    b = Seed(content="b0 = 0\nb1 = 1\nb2 = 2\n", metadata={})
    strategy = [s for s in get_strategies("cpython", state_fusion=True, pre_analysis_enabled=True)
                if "State" in type(s).__name__][0]
    child = strategy.fuse(a, b)
    assert child.metadata["mode"].startswith("state4_")
    for key in ("a_point", "b_point", "a_split", "b_split", "segment_lines"):
        assert key in child.metadata, f"missing {key}"
    counts = child.metadata["segment_lines"]
    assert set(counts) == {"a_prefix", "b_prefix", "a_suffix", "b_suffix"}


def test_real_strategy_output_contains_material_from_both_halves_of_b():
    """End-to-end version of the anti-regression check: B's prefix *and*
    B's suffix both have to reach the child."""
    a = Seed(content="a0 = 0\na1 = 1\na2 = 2\na3 = 3\n", metadata={})
    b = Seed(content="b0 = 0\nb1 = 1\nb2 = 2\nb3 = 3\n", metadata={})
    strategy = [s for s in get_strategies("cpython", state_fusion=True, pre_analysis_enabled=True)
                if "State" in type(s).__name__][0]

    saw_both = False
    for i in range(40):
        random.seed(i)
        child = strategy.fuse(a, b)
        meta = child.metadata
        if meta.get("segment_lines", {}).get("b_prefix") and meta["segment_lines"].get("b_suffix"):
            body = _lines(child.content)
            assert "b0 = 0" in body and "b3 = 3" in body
            saw_both = True
            break
    assert saw_both, "no draw produced both a non-empty B_prefix and B_suffix"

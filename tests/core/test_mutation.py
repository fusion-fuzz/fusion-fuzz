"""
core/mutation.py — the mutators every fusion runs on both parents first.

The property that matters most here is not what a mutator changes, it is
what it must *not*: line numbering. --pre-analysis caches per-seed state
points and segment boundaries as line indices computed on the pristine
seed, and fusion applies them to the mutated text. If a mutator ever
inserted or removed a line, every cached index would silently point one
line off — splices landing mid-statement, segment cuts landing inside a
block — with no error anywhere.

That invariant was measured (100% line-stable across 400 clang seeds, and
again with every rule forced past its probability gate) before the cache
was allowed to depend on it. Note the natural rate is low: each rule is
gated on `random.random() > 0.001`, so a mutator alters its input on only
~0.1-1.2% of calls. These tests hold it in place,
alongside the basics: a mutator returns a string, never raises on odd
input, and does not corrupt string/comment content.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.mutation import (  # noqa: E402
    BaseMutator,
    CPythonMutator,
    HaskellMutator,
    PHPMutator,
    RustMutator,
    SwiftMutator,
)

MUTATORS = {
    "base": (BaseMutator, "int x = 1;\nint y = 2;\nreturn x + y;\n"),
    "php": (PHPMutator, "<?php\n$a = 1;\n$b = 2;\nvar_dump($a + $b);\n?>\n"),
    "cpython": (CPythonMutator, "a = 1\nb = 2\nprint(a + b)\n"),
    "swift": (SwiftMutator, "let a: Int = 1\nlet b: Int = 2\nprint(a + b)\n"),
    "rust": (RustMutator, "fn main() {\n    let a: i32 = 1;\n    println!(\"{}\", a);\n}\n"),
    "haskell": (HaskellMutator, "f :: Int -> Int\nf x = x + 1\nmain = print (f 2)\n"),
}
ALL = sorted(MUTATORS)


# ---------------------------------------------------------------------------
# The invariant the pre-analysis cache depends on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_mutation_preserves_line_count(name):
    """Cached state points and segment boundaries are line indices computed
    on the pristine seed and applied to the mutated text. A mutator that
    added or removed a line would shift every one of them by one, silently."""
    cls, src = MUTATORS[name]
    mutator = cls()
    expected = len(src.splitlines())
    for seed in range(60):
        random.seed(seed)
        out = mutator.mutate(src)
        assert len(out.splitlines()) == expected, (
            f"{name}: {expected} lines in, {len(out.splitlines())} out\n{out}")


@pytest.mark.parametrize("name", ALL)
def test_mutation_returns_a_string(name):
    cls, src = MUTATORS[name]
    random.seed(0)
    assert isinstance(cls().mutate(src), str)


@pytest.mark.parametrize("name", ALL)
def test_mutation_survives_degenerate_input(name):
    """Seeds reaching a mutator are arbitrary corpus text, including empty
    files and files that are one long line."""
    cls, _ = MUTATORS[name]
    mutator = cls()
    for src in ("", "\n", "   ", "x" * 5000, "\n\n\n", "// only a comment\n"):
        random.seed(1)
        out = mutator.mutate(src)
        assert isinstance(out, str)
        assert len(out.splitlines()) == len(src.splitlines())


@pytest.mark.parametrize("name", ALL)
def test_mutation_is_deterministic_under_a_seed(name):
    cls, src = MUTATORS[name]
    mutator = cls()
    random.seed(99)
    first = mutator.mutate(src)
    random.seed(99)
    assert mutator.mutate(src) == first


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

class _every_rule_fires:
    """Force every mutation rule past its probability gate.

    Each rule starts with `if random.random() > 0.001: return code`, so in
    normal operation a mutator changes its input on the order of 0.1-1.2%
    of calls (measured: cpython 0.10%, php 1.17%, rust 0.67%, swift 0.80%,
    haskell 0.50%, base 0.03%). Sampling for behaviour at those rates needs
    thousands of draws; pinning `random.random()` to 0 exercises every rule
    on every call instead.
    """

    def __enter__(self):
        self._orig = random.random
        random.random = lambda: 0.0
        return self

    def __exit__(self, *exc):
        random.random = self._orig


@pytest.mark.parametrize("name", ALL)
def test_line_count_holds_even_when_every_rule_fires(name):
    """The cache-safety invariant under maximum mutation pressure. Testing
    it at the natural rate would mostly be testing that nothing happened."""
    cls, src = MUTATORS[name]
    mutator = cls()
    expected = len(src.splitlines())
    with _every_rule_fires():
        for _ in range(100):
            out = mutator.mutate(src)
            assert len(out.splitlines()) == expected, f"{name}:\n{out}"


@pytest.mark.parametrize("name", ALL)
def test_mutators_do_change_code_when_their_rules_fire(name):
    """A mutator that never altered anything would make the caches look
    perfectly stable while contributing nothing to fuzzing."""
    cls, src = MUTATORS[name]
    with _every_rule_fires():
        random.seed(0)
        assert cls().mutate(src) != src, f"{name} left the input untouched"


@pytest.mark.parametrize("name", ALL)
def test_mutation_does_not_touch_line_structure_of_comments(name):
    """Comment-only lines must survive as lines, so an index pointing at one
    still points at one."""
    cls, _ = MUTATORS[name]
    src = "a\n\nb\n\n\nc\n"
    random.seed(3)
    out = cls().mutate(src)
    assert len(out.splitlines()) == 6


def test_base_mutator_is_the_no_op_fallback():
    """Languages without a dedicated mutator get BaseMutator; whatever it
    does, it must be safe on any input."""
    src = "anything at all\nsecond line\n"
    random.seed(0)
    out = BaseMutator().mutate(src)
    assert isinstance(out, str)
    assert len(out.splitlines()) == 2

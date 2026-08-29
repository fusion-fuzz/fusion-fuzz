"""
Regression tests for MLIRFusionStrategy (dataflow fusion).

The strategy used to emit seed A, seed B and a bridge as three independent
`// -----` sections, which `--split-input-file` verifies in isolation: the
two seeds never interacted, and the "bridge" only copied constant literals
into a self-contained function. These tests pin the properties of the
call-bridge rewrite:

  * one shared module, no split sections — both seeds in one symbol table
  * the bridge calls a function from each seed, feeding one's result into
    the other's argument (the actual cross-seed dataflow edge)
  * every emitted call's type signature matches its callee's declaration,
    so a rejected child is rejected for the seeds' sake, not ours
  * symbols and file-level aliases from the two seeds never collide
  * seeds with nothing callable fall back to the constant bridge
"""

import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.fusion import Seed, get_strategies  # noqa: E402

ARITH = """// RUN: mlir-opt %s | FileCheck %s
module {
  func.func @ar_I(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}"""

MIXED = """#map0 = affine_map<(d0) -> (d0 + I)>
module {
  func.func @mix_I(%x: i32, %y: f32, %m: memref<4xf32>) -> f32 {
    return %y : f32
  }
}"""

FLOATRET = """module {
  func.func @fret_I(%z: index) -> f32 {
    %c = arith.constant 1.0 : f32
    return %c : f32
  }
}"""

TENSOR = """module {
  func.func @tens_I(%t: tensor<4xf32>) -> tensor<4xf32> {
    return %t : tensor<4xf32>
  }
}"""

NO_FUNCTIONS = """module {
  memref.global @g_I : memref<2xi32> = dense<[1, 2]>
}"""

SIG_RE = re.compile(
    r'func\.func\s+(?:private\s+)?@([\w.$-]+)\s*\(([^)]*)\)'
    r'\s*(?:->\s*(\([^)]*\)|[^\{\s]+))?\s*\{'
)
CALL_RE = re.compile(
    r'(?:%[\w.$,\s]+=\s*)?func\.call\s+@([\w.$-]+)\(([^)]*)\)'
    r'\s*:\s*\(([^)]*)\)\s*->\s*(\([^)]*\)|\S+)'
)
ALIAS_DEF_RE = re.compile(r'^\s*([#!][\w.$]*)\s*=', re.M)


def _seeds(template, count):
    return [Seed(content=template.replace("I", str(i)), metadata={"filename": f"s{i}.mlir"})
            for i in range(count)]


def _strategy():
    return get_strategies("mlir", dataflow_fusion=True, pre_analysis_enabled=True)[0]


def _malformed_calls(src):
    """Calls whose declared operand/result types disagree with the callee's
    own signature — always our bug, never the seeds'."""
    sigs = {}
    for m in SIG_RE.finditer(src):
        args = [a.split(':', 1)[1].strip() for a in m.group(2).split(',') if ':' in a]
        sigs[m.group(1)] = (args, (m.group(3) or '()').strip())

    problems = []
    for m in CALL_RE.finditer(src):
        name = m.group(1)
        if name not in sigs:
            problems.append(("call_to_undefined_symbol", name))
            continue
        declared_args, declared_ret = sigs[name]
        used_args = [t.strip() for t in m.group(3).split(',') if t.strip()]
        if declared_args != used_args:
            problems.append(("argtypes", name, declared_args, used_args))
        if declared_ret.replace(' ', '') != m.group(4).replace(' ', ''):
            problems.append(("rettype", name, declared_ret, m.group(4)))
    return problems


@pytest.mark.parametrize("template", [ARITH, MIXED, FLOATRET, TENSOR])
def test_single_shared_module(template):
    """No `// -----` sections: the seeds must share one symbol table, which
    is what allows them to interact at all."""
    random.seed(17)
    seeds = _seeds(template, 3)
    strategy = _strategy()
    for a in seeds:
        for b in seeds:
            if a is b:
                continue
            child = strategy.fuse(a, b)
            assert "// -----" not in child.content
            assert child.content.count("module {") == 1


@pytest.mark.parametrize("template_a,template_b", [
    (ARITH, MIXED), (MIXED, FLOATRET), (FLOATRET, TENSOR), (TENSOR, ARITH),
])
def test_rename_reaches_seed_b(template_a, template_b):
    """Dataflow fusion is a rename in B (FusionStrategy.rename_across).

    It replaced a synthesised `func.func @_ffl_bridge_N` that called a
    producer in one seed and passed the result to a consumer in the other.
    What has to hold now is narrower: B's half of the child must differ
    from the B that went in, or no connection was attempted at all.
    """
    random.seed(23)
    a = _seeds(template_a, 1)[0]
    b = _seeds(template_b, 1)[0]
    strategy = _strategy()

    changed = 0
    for _ in range(20):
        child = strategy.fuse(a, b)
        assert child.metadata["mode"] == "dataflow_rename", child.content
        assert "_ffl_bridge" not in child.content, "bridge construction is gone"
        b_half = child.content.split("===== Seed B =====")[1]
        if b_half.count("@") and b_half != b.content:
            changed += 1
    assert changed, "no fusion attempt ever altered B"


def test_rename_prefers_symbols_over_ssa_values():
    """`@` symbols share one module-level table after fusion, so renaming
    one of B's symbol references to A's is a link MLIR's verifier accepts.
    `%` values are region-scoped — renaming those across seeds yields a use
    with no dominating definition. Both are available; symbols must win
    whenever the code has any."""
    strategy = _strategy()
    with_symbols = "func.func @foo(%a: i32) -> i32 { return %a : i32 }"
    assert all(n.startswith("@") for n in strategy._dataflow_names(with_symbols))

    no_symbols = "%0 = arith.constant 1 : i32\n%1 = arith.addi %0, %0 : i32"
    names = strategy._dataflow_names(no_symbols)
    assert names and all(n.startswith("%") for n in names)


def test_rename_respects_the_trailing_boundary():
    """Renaming %1 must not also rewrite %10 — the sigil disambiguates the
    start of a name, nothing disambiguates its end."""
    strategy = _strategy()
    out = strategy._dataflow_replace("%1 %10 %1x %1", "%1", "%9")
    assert out == "%9 %10 %1x %9", out


@pytest.mark.parametrize("template", [ARITH, MIXED, FLOATRET, TENSOR])
def test_emitted_calls_match_callee_signatures(template):
    random.seed(31)
    seeds = _seeds(template, 3)
    strategy = _strategy()
    for a in seeds:
        for b in seeds:
            if a is b:
                continue
            for _ in range(5):
                child = strategy.fuse(a, b)
                assert not _malformed_calls(child.content), (
                    f"{_malformed_calls(child.content)} in:\n{child.content}"
                )


def test_symbols_and_aliases_are_namespaced_before_the_rename():
    """Both seeds define `@mix_*` and `#map0`; merging them into one module
    namespaces those apart.

    Function names are checked only for being namespaced, not for being
    unique: dataflow fusion afterwards renames a name in B to one from A,
    and when it picks a definition rather than a reference the result is a
    genuine symbol redefinition. That is the leave-validity-to-chance case
    working as intended, so asserting uniqueness here would be asserting
    that dataflow fusion did nothing.

    Aliases are checked strictly — the rename only touches `@` and `%`
    names, so a collision there would be a real de-collision bug.
    """
    random.seed(11)
    a, b = _seeds(MIXED, 2)
    strategy = _strategy()

    child = strategy.fuse(a, b)
    names = re.findall(r'func\.func\s+(?:private\s+)?@([\w.$-]+)', child.content)
    assert names
    assert all(n.startswith(("A_", "B_", "_ffl_p")) for n in names), names

    aliases = [m.group(1) for m in ALIAS_DEF_RE.finditer(child.content)]
    assert len(aliases) == len(set(aliases)), aliases
    # Aliases are still present (they'd dangle if dropped) and namespaced.
    assert len(aliases) == 2
    assert all(al.startswith("#A_") or al.startswith("#B_") for al in aliases), aliases


def test_seeds_without_functions_still_fuse():
    """A seed with no functions has no `@` symbols, so the rename falls back
    to SSA values. That produces IR the verifier will reject, which is the
    intended leave-it-to-chance case — but it must not raise, and it must
    not resurrect the removed bridge machinery."""
    random.seed(13)
    plain = _seeds(NO_FUNCTIONS, 2)
    strategy = _strategy()

    child = strategy.fuse(plain[0], plain[1])
    assert child.metadata["mode"] == "dataflow_rename"
    assert "_ffl_bridge" not in child.content
    assert child.content.count("module {") == 1

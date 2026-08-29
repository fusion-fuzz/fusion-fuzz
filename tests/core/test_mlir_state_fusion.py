"""
Regression tests for MLIRStateFusionStrategy.

The strategy used to be a no-op in its intended mode: MLIR's live-variable
count is monotonic (SSA values never leave a region), so the state points
core/state_analysis.py returns sit at the end of a block, and cutting the
donor continuation before its terminator then left nothing to graft — with
--pre-analysis on, every child came out as the host verbatim.

These tests pin the properties the fix has to keep:
  * a donor continuation is actually grafted, with and without pre-analysis
  * grafted ops land before the host's terminator, not after it
  * grafted SSA definitions don't collide with the host's names
  * dangling donor operands are rebound to host values in scope
  * region-opening results (%r of an scf.for) aren't referenced inside
    their own region
  * block labels are never grafted (they'd need a terminator before them)
"""

import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.fusion import Seed, get_strategies  # noqa: E402
from core.state_analysis import find_most_complex_state  # noqa: E402

ARITH = """module {
  func.func @ar_I(%a: i32, %b: i32) -> i32 {
    %cI = arith.constant I : i32
    %0 = arith.addi %a, %b : i32
    %1 = arith.muli %0, %cI : i32
    return %1 : i32
  }
}"""

MEMREF = """// RUN: mlir-opt %s | FileCheck %s
module {
  func.func @mem_I(%m: memref<4xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %k = arith.constant 1.5e+00 : f32
    %v = memref.load %m[%c0] : memref<4xf32>
    %s = arith.addf %v, %k : f32
    memref.dealloc %m : memref<4xf32>
    return %s : f32
  }
}"""

BLOCKS = """module {
  func.func @br_I(%arg0: i32) -> i32 {
    %cI = arith.constant I : i32
    %p = arith.cmpi slt, %arg0, %cI : i32
    cf.cond_br %p, ^bb1(%arg0 : i32), ^bb2(%cI : i32)
  ^bb1(%x: i32):
    %r1 = arith.addi %x, %cI : i32
    cf.br ^bb3(%r1 : i32)
  ^bb2(%y: i32):
    %r2 = arith.muli %y, %cI : i32
    cf.br ^bb3(%r2 : i32)
  ^bb3(%z: i32):
    return %z : i32
  }
}"""

LOOP = """module {
  func.func @loop_I(%m: memref<8xf32>, %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0.0 : f32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f32) {
      %v = memref.load %m[%i] : memref<8xf32>
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    }
    %f = arith.mulf %r, %init : f32
    return %f : f32
  }
}"""

SSA_RE = re.compile(r"%[A-Za-z0-9_$.]+")
DEF_RE = re.compile(r"^\s*(%[A-Za-z0-9_$.]+)\s*=")
TERM_RE = re.compile(r"^\s*(?:func\.return|return|cf\.br|cf\.cond_br|scf\.yield)\b")
LABEL_RE = re.compile(r"^\s*\^")


def _seeds(template, count, with_cache):
    seeds = []
    for i in range(count):
        body = template.replace("I", str(i))
        meta = {"filename": f"s{i}.mlir"}
        if with_cache:
            # Mirror what --pre-analysis persists onto each seed.
            meta["most_complex_states"] = [
                p.to_dict() for p in find_most_complex_state(body, "mlir", "projects/mlir")
            ]
        seeds.append(Seed(content=body, metadata=meta))
    return seeds


def _strategy(pre_analysis):
    return [
        s for s in get_strategies("mlir", state_fusion=True, pre_analysis_enabled=pre_analysis)
        if type(s).__name__ == "MLIRStateFusionStrategy"
    ][0]


def _structural_problems(src):
    """Structural errors that would make mlir-opt reject the child for
    reasons unrelated to the two seeds' semantics."""
    problems = set()
    defined = set()
    after_terminator = False
    for line in src.splitlines():
        code = line.split("//")[0]
        stripped = code.strip()
        if not stripped:
            continue
        if "func.func" in stripped:
            defined = set(re.findall(r"(%[\w$.]+)\s*:", code))
            after_terminator = False
            continue
        if stripped.startswith("}"):
            after_terminator = False
            continue
        if LABEL_RE.match(code):
            if not after_terminator:
                problems.add("block_label_without_terminator")
            for name in re.findall(r"(%[\w$.]+)\s*:", code):
                if name in defined:
                    problems.add("blockarg_redefinition")
                defined.add(name)
            after_terminator = False
            continue
        if after_terminator:
            problems.add("ops_after_terminator")
        m = DEF_RE.match(code)
        if m and m.group(1) in defined:
            problems.add("ssa_redefinition")
        if stripped.endswith("{"):
            # Region-opening op (`%r = scf.for %i = ... iter_args(%acc = ...) {`):
            # every name on it either is the result or binds a region block
            # argument, all of which are in scope from here on.
            defined.update(SSA_RE.findall(code))
            continue
        uses = SSA_RE.findall(code[code.index("=") + 1:] if m else code)
        for use in uses:
            if use not in defined:
                problems.add("undefined_use")
        if m:
            defined.add(m.group(1))
        if TERM_RE.match(code):
            after_terminator = True
    return problems


@pytest.mark.parametrize("pre_analysis", [True, False])
@pytest.mark.parametrize("template", [ARITH, MEMREF, BLOCKS, LOOP])
def test_donor_is_actually_grafted(template, pre_analysis):
    """The bug: with --pre-analysis the child was the host verbatim."""
    random.seed(1234)
    seeds = _seeds(template, 4, with_cache=pre_analysis)
    strategy = _strategy(pre_analysis)

    grafted = 0
    total = 0
    for a in seeds:
        for b in seeds:
            if a is b:
                continue
            total += 1
            if "state fusion" in strategy.fuse(a, b).content:
                grafted += 1
    assert grafted == total, f"only {grafted}/{total} children got a donor continuation"


@pytest.mark.parametrize("template", [ARITH, MEMREF, BLOCKS, LOOP])
def test_children_are_structurally_wellformed(template):
    random.seed(99)
    seeds = _seeds(template, 4, with_cache=True)
    strategy = _strategy(True)

    for a in seeds:
        for b in seeds:
            if a is b:
                continue
            for _ in range(10):
                child = strategy.fuse(a, b)
                assert not _structural_problems(child.content), (
                    f"{_structural_problems(child.content)} in:\n{child.content}"
                )


def test_grafted_op_consumes_host_values():
    """Dangling donor operands rebind to host state — that edge is the
    whole point of state fusion, so assert it happens rather than the
    donor arriving self-contained."""
    random.seed(7)
    host = _seeds(ARITH, 1, with_cache=True)[0]
    donor = _seeds(MEMREF, 1, with_cache=True)[0]
    strategy = _strategy(True)

    host_names = {"%a", "%b", "%c0", "%0", "%1"}
    saw_host_value = False
    for _ in range(20):
        child = strategy.fuse(host, donor)
        for line in child.content.splitlines():
            if "state fusion" in line:
                code = line.split("//")[0]
                # Operands are everything on the right of `=`, or the whole
                # line for result-less ops like `memref.dealloc %x`.
                operands = SSA_RE.findall(code.split("=", 1)[1] if "=" in code else code)
                if any(op in host_names for op in operands):
                    saw_host_value = True
    assert saw_host_value, "grafted ops never referenced a host value"


def test_loop_result_not_used_inside_its_own_region():
    """%r of an `scf.for` must not be referenced from inside the loop body."""
    random.seed(3)
    hosts = _seeds(LOOP, 2, with_cache=True)
    donors = _seeds(ARITH, 2, with_cache=True)
    strategy = _strategy(True)

    for host in hosts:
        for donor in donors:
            for _ in range(25):
                child = strategy.fuse(host, donor)
                inside = False
                for line in child.content.splitlines():
                    stripped = line.split("//")[0].strip()
                    if "scf.for" in stripped:
                        inside = True
                        continue
                    if inside and stripped.startswith("}"):
                        inside = False
                        continue
                    if inside and "=" in stripped:
                        rhs = stripped.split("=", 1)[1]
                        assert "%r" not in SSA_RE.findall(rhs), (
                            f"loop result used inside its own region:\n{child.content}"
                        )


def test_degenerate_donor_reports_no_continuation():
    """A donor whose block is only a terminator can't contribute — the
    child must say so rather than silently claim a fusion."""
    strategy = _strategy(True)
    host = _seeds(ARITH, 1, with_cache=True)[0]
    donor = Seed(content="module {\n  func.func @z() {\n    return\n  }\n}", metadata={})

    child = strategy.fuse(host, donor)
    assert child.metadata["mode"].startswith("state_nocontinuation")
    assert "state fusion" not in child.content

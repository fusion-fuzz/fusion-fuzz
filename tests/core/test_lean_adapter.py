"""
Tests for the Lean 4 adapter: core/lean_text.py (how a file is cut), the
three fusion strategies, and the crash oracle (projects/lean/analyzer.py).

The oracle strings are the ones the from-source toolchain prints:
`PANIC at <decl> <module>:<line>:<col>: <msg>` (Init/Util.lean,
mkPanicMessageWithDecl), `LEAN ASSERTION VIOLATION` (src/runtime/debug.cpp)
and the kernel's `(kernel) ...` messages (Lean/Message.lean).
"""

import importlib.util
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core import lean_text as L  # noqa: E402
from core.fusion import Seed, get_strategies  # noqa: E402
from core.state_analysis import segment_boundaries  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("ffl_lean_analyzer_test", "projects/lean/analyzer.py")


HOST = """import Lean
open Lean

/-- A doc comment belongs to the def below it. -/
@[simp] def double (n : Nat) : Nat := n + n

namespace Foo
def bump : Nat → Nat
  | 0 => 1
  | n+1 => n + 2

theorem bump_pos (n : Nat) : 0 < bump n := by
  cases n
  · decide
  · simp [bump]
end Foo

structure Point where
  x : Nat
  y : Nat
  deriving Repr

def main : IO Unit := do
  IO.println "host"
  IO.println (double 2)
"""

DONOR = """/-- info: 4 -/
#guard_msgs in
#eval 2 + 2

def double (k : Nat) : Nat := 2 * k

inductive Color where
  | red | green
  deriving Repr

theorem double_eq (k : Nat) : double k = 2 * k := by
  unfold double
  rfl

def main : IO Unit := do
  let c := Color.red
  IO.println (repr c)
  IO.println (double 3)
"""


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------

def test_commands_merge_prefixes_and_track_namespaces():
    body = L.split_header(HOST)[1]
    cmds = L.split_commands(body)
    kws = [c.keyword for c in cmds]
    assert kws[:3] == ["open", "def", "namespace"]
    double = cmds[1]
    assert double.text.startswith("/--") and double.name == "double"
    bump = next(c for c in cmds if c.name == "bump")
    assert bump.fqname == "Foo.bump" and bump.depth == 1
    # the match alternatives stay inside `bump`
    assert "| n+1" in bump.text


def test_guard_msgs_prefix_belongs_to_command():
    cmds = L.split_commands(DONOR)
    assert cmds[0].keyword == "#eval" and "#guard_msgs" in cmds[0].text


def test_nested_block_comments_are_masked():
    src = "/- outer /- inner -/ still comment -/\ndef x := 1\n"
    assert "still" not in L.mask(src)
    assert [c.keyword for c in L.split_commands(src)] == ["def"]


def test_merge_headers_dedups_and_keeps_module():
    assert L.merge_headers(["import Lean"], ["import Lean", "import Std"]) == \
        ["import Lean", "import Std"]
    merged = L.merge_headers(["module", "public import Lean"], ["import Std"])
    assert merged[0] == "module" and "public import Lean" in merged


def test_rename_collisions_renames_donor_uses_too():
    out, mapping = L.rename_collisions(L.split_header(DONOR)[1], L.split_header(HOST)[1])
    assert mapping["double"] == "double_ffl" and mapping["main"] == "main_ffl"
    assert "theorem double_eq (k : Nat) : double_ffl k" in out
    assert "IO.println (double_ffl 3)" in out


def test_segment_boundaries_are_between_commands():
    body = L.split_header(HOST)[1]
    lines = body.split("\n")
    for b in segment_boundaries(body, "lean"):
        nxt = next((ln for ln in lines[b + 1:] if ln.strip()), None)
        assert nxt is None or not nxt[:1].isspace()
        assert not lines[b].lstrip().startswith("@[")


def test_scope_close_and_reopen():
    body = L.split_header(HOST)[1]
    cmds = L.split_commands(body)
    k = next(i for i, c in enumerate(cmds) if c.name == "bump_pos")
    stack = L.scope_stack_before(cmds, k, body.split("\n"))
    assert [s["name"] for s in stack] == ["Foo"]
    assert L.close_scopes(stack) == ["end Foo"]
    assert L.reopen_scopes(stack)[0] == "namespace Foo"


def test_blocks_stop_at_termination_by():
    src = ("def f (n : Nat) : Nat := by\n  exact n\n  termination_by n\n")
    (b,) = L.blocks(src)
    assert b.end == 2


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@pytest.fixture
def pair():
    return Seed(content=HOST, metadata={"type": "lean"}), Seed(content=DONOR, metadata={"type": "lean"})


def test_all_three_strategies_registered():
    names = sorted(type(s).__name__ for s in get_strategies("lean"))
    assert names == ["LeanDeclarationFusionStrategy", "LeanFusionStrategy", "LeanStateFusionStrategy"]


@pytest.mark.parametrize("seed", range(40))
def test_children_are_one_file_with_one_header(pair, seed):
    random.seed(seed)
    a, b = pair
    for strat in get_strategies("lean"):
        strat.mutation = False
        kids = strat.fuse_bidirectional(a, b) if hasattr(strat, "fuse_bidirectional") else [strat.fuse(a, b)]
        for kid in kids:
            assert kid.metadata["extension"] == ".lean"
            text = kid.content
            assert text.count("import Lean") <= 1
            # header first: nothing but imports before the first command
            first_cmd = next(ln for ln in text.split("\n") if ln.strip() and not ln.startswith("import"))
            assert not first_cmd.startswith("import")
            # the donor's colliding names were renamed, never duplicated
            assert text.count("def double (") <= 1
            assert text.count("def main ") <= 1


def test_chained_fusion_keeps_names_unique(pair):
    """The orchestrator feeds a fused child back in as the host with the
    same donor; the donor's names must not be declared twice."""
    random.seed(1)
    a, b = pair
    df, decl, state = get_strategies("lean")
    child = df.fuse(a, b)
    child = state.fuse(child, b)
    decls = [d.fqname for d in L.declarations(L.split_header(child.content)[1])]
    assert len(decls) == len(set(decls))


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

PANIC_LEAN = ("PANIC at Lean.Expr.bindingBody! Lean.Expr:1012:23: binding expected\n"
              "backtrace:\n/lib/libleanshared.so(lean_panic_fn+0x10) [0x1]\n"
              "/lib/libleanshared.so(l_Lean_Meta_whnfCore+0x20) [0x2]\n")
PANIC_USER_INIT = ("PANIC at Array.get!Internal Init.GetElem:200:4: index out of bounds\n"
                   "backtrace:\n/lib/libleanshared.so(lean::ir_interpreter::call+0x1) [0x1]\n")
PANIC_USER_OWN = "PANIC at f t:3:14: oops\n"
ASSERT = "LEAN ASSERTION VIOLATION\nFile: /src/kernel/type_checker.cpp\nLine: 120\nis_sort(t)\n"
KERNEL = "t.lean:3:0: error: (kernel) declaration has metavariables 'foo'\n"
KERNEL_POSITIVITY = "t.lean:3:0: error: (kernel) arg #1 of 'T.mk' has a non positive occurrence\n"
IR_CHECK = ("t.lean:2:0: error: failed to compile definition, compiler IR check failed at `f`. "
            "Error: unknown variable 'x_1'\n")
IR_USER = ("t.lean:2:0: error: failed to compile definition, compiler IR check failed at `f`. "
           "Error: depends on declaration 'g', which has no executable code; consider marking "
           "definition as 'noncomputable'\n")
STACK = "Stack overflow detected. Aborting.\n"
ASAN = "SUMMARY: AddressSanitizer: heap-use-after-free /src/runtime/object.cpp:88 in lean_dec_ref\n"


@pytest.mark.parametrize("output,rc,kind", [
    (PANIC_LEAN, 0, "panic"),
    (PANIC_USER_INIT, 0, "ok"),
    (PANIC_USER_OWN, 0, "ok"),
    (ASSERT, 134, "assert"),
    (KERNEL, 1, "kernel"),
    (KERNEL_POSITIVITY, 1, "rejected"),
    (IR_CHECK, 1, "internal_error"),
    (IR_USER, 1, "rejected"),
    (STACK, 134, "resource"),
    (ASAN, 1, "sanitizer"),
    ("", 139, "signal"),
    ("t.lean:1:0: error: unknown identifier 'x'\n", 1, "rejected"),
    ("", 124, "timeout"),
    ("INTERNAL PANIC: executed 'sorry'\n", 1, "rejected"),
])
def test_classify(output, rc, kind):
    assert analyzer.classify(output, rc)["kind"] == kind


def test_meta_programs_are_labelled():
    v = analyzer.classify(PANIC_LEAN, 0, program="run_meta pure ()", uses_meta=True)
    assert v["is_bug"] and v["signature"].endswith("[meta]")


def test_expected_message_in_program_is_not_a_bug():
    prog = "/-- error: (kernel) declaration has metavariables 'foo' -/\n#guard_msgs in\n"
    assert not analyzer.classify(KERNEL, 1, program=prog)["is_bug"]


def test_signatures_are_stable_across_ids():
    a = analyzer.classify("error: unknown free variable '_uniq.1234'\n", 1)["signature"]
    b = analyzer.classify("error: unknown free variable '_uniq.98'\n", 1)["signature"]
    assert a == b


def test_error_recovery_hits_are_labelled():
    out = ("t.lean:1:44: error(lean.synthInstanceFailed): failed to synthesize instance\n"
           "t.lean:1:52: error: (kernel) application type mismatch\n")
    v = analyzer.classify(out, 1)
    assert v["is_bug"] and v["signature"].endswith("[after error]")
    assert not analyzer.classify(KERNEL, 1)["signature"].endswith("[after error]")


def test_json_output_gives_the_same_signature():
    js = ('{"severity":"error","data":"`grind` internal error, trying to assert equality\\n  x = y"}\n')
    plain = "t.lean:1:2: error: `grind` internal error, trying to assert equality\n  x = y\n"
    assert analyzer.classify(js, 1)["signature"] == analyzer.classify(plain, 1)["signature"]


def test_user_extension_panic_inside_lean_frames_is_not_a_bug():
    out = ("PANIC at Option.get! Init.Data.Option.BasicAux:22:14: value is none\nbacktrace:\n"
           "/lib/libleanshared.so(lean_panic_fn+0x2a) [0x1]\n/lib/libleanshared.so(+0x9b456e7) [0x2]\n"
           "/lib/libleanshared.so(Lean.PrettyPrinter.Delaborator.delabFor+0x10) [0x3]\n")
    assert not analyzer.classify(out, 0, program="@[delab app.Foo] def d : Delab := ...", uses_meta=True)["is_bug"]
    assert analyzer.classify(out, 0, uses_meta=False)["is_bug"]


def test_unsafe_runtime_assert_is_not_a_bug():
    out = "LEAN ASSERTION VIOLATION\nFile: /src/include/lean/lean.h\nLine: 766\nlean_is_closure(o)\n"
    assert not analyzer.classify(out, 134, program="unsafe def t := evalConst", program_unsafe=True)["is_bug"]
    assert analyzer.classify(out, 134)["is_bug"]


def test_check_types_lcnf_errors_are_ignored():
    out = "t.lean:2:0: error: type mismatch at LCNF application\n"
    assert not analyzer.classify(out, 1, program="set_option compiler.checkTypes true\n")["is_bug"]
    assert analyzer.classify(out, 1)["is_bug"]
